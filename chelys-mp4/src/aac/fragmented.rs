// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The fragmented MP4 file, written as the stream is coded.

use std::fmt;
use std::num::NonZeroU32;

use chelys_aac::{Config, ConfigError, Finished, FrameSink, NonFiniteSample, Rate, Sample};

use super::{Stream, raw_encoder};
use crate::bitrate::window_frames;
use crate::movie::fragmented::{FragmentSink, Fragmenter, init_segment};

/// What a [`FragmentedEncoder`] is to write, beyond the stream.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct FragmentOptions {
    /// The samples of each channel that the input will have, if known, as
    /// [`Finished::samples`] counts them; `None` by default.
    pub length: Option<u64>,
    /// The duration that each fragment is to have, in samples of each
    /// channel; by default the sample rate's, one second.
    pub fragment_samples: Option<NonZeroU32>,
}

impl FragmentOptions {
    /// Declares the samples of each channel that the input will have, so
    /// that the file presents them exactly, as an [`Encoder`]'s does: the
    /// encoder takes no more, and its [`finish`](FragmentedEncoder::finish)
    /// fails with fewer.
    ///
    /// [`Encoder`]: super::Encoder
    pub fn length(mut self, samples: u64) -> Self {
        self.length = Some(samples);
        self
    }

    /// Has each fragment last about `samples` samples of each channel: as
    /// many whole frames as end nearest each multiple of `samples`, and a
    /// frame at least. A duration in samples is exact at every sample rate,
    /// as the file's times are in samples. It is at most 2^32 - 1 samples,
    /// 12 hours at 96 kHz, which keeps every fragment's tables within their
    /// fields. A fragment's frames are held in memory until it ends: about
    /// the bitrate times its duration.
    pub fn fragment_samples(mut self, samples: NonZeroU32) -> Self {
        self.fragment_samples = Some(samples);
        self
    }
}

/// Why a [`FragmentedEncoder`] refused a call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodeError {
    /// A sample is NaN or infinite.
    NonFiniteSample(NonFiniteSample),
    /// The call would take the input past the length declared.
    Length(LengthError),
}

impl From<NonFiniteSample> for EncodeError {
    fn from(e: NonFiniteSample) -> Self {
        Self::NonFiniteSample(e)
    }
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteSample(e) => e.fmt(f),
            Self::Length(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for EncodeError {}

/// An input whose length is not the one declared in
/// [`FragmentOptions::length`], which the init segment gives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct LengthError {
    /// The samples of each channel declared.
    pub length: u64,
    /// The samples of each channel the input would have had.
    pub samples: u64,
}

impl fmt::Display for LengthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "an input of {} samples of each channel, where {} were declared",
            self.samples, self.length
        )
    }
}

impl std::error::Error for LengthError {}

/// Encodes PCM as AAC-LC, as [`chelys_aac::Encoder`] does, into a
/// fragmented MP4 file (ISO/IEC 14496-12, clause 8.8), written as the
/// frames come: its [`init_segment`](Self::init_segment), then fragments
/// of about a second each, which a [`FragmentSink`] takes, so that the
/// file can be sent or played as it is encoded, and little of it is held
/// in memory.
///
/// Where the input's length is declared in advance
/// ([`FragmentOptions::length`]), the file gives a player its exact length
/// and place, as an [`Encoder`]'s does. Otherwise its edit list leaves out
/// the [`DELAY`](chelys_aac::DELAY) samples that a decoder's output starts
/// with, and its media ends with the input, but it does not give where the
/// presentation ends, so that a player may present the
/// [`Finished::padding`] that the last frame decodes to.
///
/// The frames are the same as an [`Encoder`]'s of the same input: the
/// fragments' `MediaDataBox`es hold, one after another, what the file
/// written whole holds in one. Its `ES_Descriptor` is written before the
/// first frame, so that it gives a bound on the maximum bitrate, and at a
/// constant rate the rate asked for as an estimate of the average, where an
/// [`Encoder`]'s gives both as measured on the frames.
///
/// ```
/// use chelys_aac::{Config, Rate};
/// use chelys_mp4::aac::{FragmentOptions, FragmentedEncoder};
///
/// let config = Config::new(44_100, 2, Rate::Constant { bits_per_second: 128_000 });
/// let mut encoder = FragmentedEncoder::new(config, FragmentOptions::default())?;
/// let mut mp4 = encoder.init_segment().to_vec();
/// let mut fragments = 0;
/// let mut out = |fragment: &[u8], _: chelys_mp4::FragmentInfo| {
///     mp4.extend_from_slice(fragment);
///     fragments += 1;
/// };
/// encoder.encode_interleaved(&[0.0f32; 2 * 3 * 44_100], &mut out)?;
/// let finished = encoder.finish(&mut out)?;
/// assert_eq!(finished.samples, 3 * 44_100);
/// assert_eq!(fragments, 4);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// [`Encoder`]: super::Encoder
pub struct FragmentedEncoder {
    encoder: chelys_aac::Encoder,
    fragmenter: Fragmenter,
    init_segment: Vec<u8>,
    length: Option<u64>,
    /// The samples of each channel taken.
    samples: u64,
}

/// Gives each frame to the fragmenter, and the fragments it ends to the
/// sink, as the encoder codes them.
struct Fragments<'a, O: ?Sized> {
    fragmenter: &'a mut Fragmenter,
    out: &'a mut O,
}

impl<O: FragmentSink + ?Sized> FrameSink for Fragments<'_, O> {
    fn frame(&mut self, frame: &[u8]) {
        if let Some(fragment) = self.fragmenter.push_frame(frame) {
            self.out.fragment(&fragment.bytes, fragment.info);
        }
    }
}

impl FragmentedEncoder {
    /// An encoder of the stream `config` asks for, as [`Encoder::new`], into
    /// the file `options` asks for.
    ///
    /// # Errors
    ///
    /// As [`Encoder::new`].
    ///
    /// [`Encoder::new`]: super::Encoder::new
    pub fn new(config: Config, options: FragmentOptions) -> Result<Self, ConfigError> {
        let encoder = raw_encoder(config)?;
        let stream = Stream::of(&encoder);
        let (max_bitrate, avg_bitrate) = bitrates_before(config.rate(), &stream);
        let track = stream.track(max_bitrate, avg_bitrate);
        let one_second = NonZeroU32::new(stream.sample_rate).expect("a sample rate");
        let fragment_samples = options.fragment_samples.unwrap_or(one_second);
        Ok(Self {
            fragmenter: Fragmenter::new(stream.frame_samples, fragment_samples),
            init_segment: init_segment(&track, options.length),
            encoder,
            length: options.length,
            samples: 0,
        })
    }

    /// The init segment: the file's `FileTypeBox` and `MovieBox`, which
    /// come before every fragment. It is fixed when the encoder is made, and
    /// can be given again, as to each reader that starts at a later fragment.
    pub fn init_segment(&self) -> &[u8] {
        &self.init_segment
    }

    /// The number of channels the input has.
    pub fn channels(&self) -> usize {
        self.encoder.channels()
    }

    /// The samples of each channel taken so far.
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Refuses a call of `samples` more samples of each channel, if they
    /// would take the input past its declared length.
    fn check_length(&self, samples: usize) -> Result<(), LengthError> {
        let samples = self.samples.saturating_add(samples as u64);
        match self.length {
            Some(length) if samples > length => Err(LengthError { length, samples }),
            _ => Ok(()),
        }
    }

    /// Encodes interleaved samples, as
    /// [`chelys_aac::Encoder::encode_interleaved`], and gives `out` the
    /// fragments they complete.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Length`] if the call would take the input past the
    /// length declared, and [`EncodeError::NonFiniteSample`] if a sample is
    /// NaN or infinite. Either way none of the call is taken: `out` is given
    /// nothing, and the encoder is as it was before the call.
    ///
    /// # Panics
    ///
    /// As [`chelys_aac::Encoder::encode_interleaved`].
    pub fn encode_interleaved<S: Sample, O: FragmentSink + ?Sized>(
        &mut self,
        samples: &[S],
        out: &mut O,
    ) -> Result<(), EncodeError> {
        let len = samples.len() / self.channels();
        self.check_length(len).map_err(EncodeError::Length)?;
        let mut fragments = Fragments {
            fragmenter: &mut self.fragmenter,
            out,
        };
        self.encoder.encode_interleaved(samples, &mut fragments)?;
        self.samples += len as u64;
        Ok(())
    }

    /// Encodes one slice of samples per channel, as
    /// [`chelys_aac::Encoder::encode_planar`], and gives `out` the
    /// fragments they complete.
    ///
    /// # Errors
    ///
    /// As [`encode_interleaved`](Self::encode_interleaved): none of a call
    /// refused is taken.
    ///
    /// # Panics
    ///
    /// As [`chelys_aac::Encoder::encode_planar`].
    pub fn encode_planar<S: Sample, O: FragmentSink + ?Sized>(
        &mut self,
        channels: &[&[S]],
        out: &mut O,
    ) -> Result<(), EncodeError> {
        let len = channels.first().map_or(0, |channel| channel.len());
        self.check_length(len).map_err(EncodeError::Length)?;
        let mut fragments = Fragments {
            fragmenter: &mut self.fragmenter,
            out,
        };
        self.encoder.encode_planar(channels, &mut fragments)?;
        self.samples += len as u64;
        Ok(())
    }

    /// Queues `bytes` for the data streams of the frames coded from now on,
    /// as [`chelys_aac::Encoder::push_data_stream`].
    pub fn push_data_stream(&mut self, bytes: &[u8]) {
        self.encoder.push_data_stream(bytes);
    }

    /// The bytes of data streams queued that no frame has carried yet.
    pub fn data_stream_pending(&self) -> usize {
        self.encoder.data_stream_pending()
    }

    /// Ends the fragment being gathered now, and gives it to `out`, if it
    /// has a frame: a fragment always has one, so that this does nothing
    /// where no frame has been coded since the last. The next fragment ends
    /// where it would have, at the next multiple of the fragments' duration
    /// past this one's end.
    ///
    /// Only frames coded are written, and a frame is coded only once the
    /// encoder has the input it looks ahead to (see [`chelys_aac::Encoder`]):
    /// the latest input, a frame or two of it, is not yet in the fragment.
    pub fn end_fragment<O: FragmentSink + ?Sized>(&mut self, out: &mut O) {
        if let Some(fragment) = self.fragmenter.flush() {
            out.fragment(&fragment.bytes, fragment.info);
        }
    }

    /// Ends the stream, as [`chelys_aac::Encoder::finish`], and gives `out`
    /// the fragments left, the last ending with the last frame.
    ///
    /// # Errors
    ///
    /// [`LengthError`] if the input is shorter than the length declared,
    /// which the init segment already gives: the file would not present
    /// what it says, and is to be discarded. `out` is then given nothing.
    pub fn finish<O: FragmentSink + ?Sized>(self, out: &mut O) -> Result<Finished, LengthError> {
        let Self {
            encoder,
            mut fragmenter,
            length,
            samples,
            ..
        } = self;
        if let Some(length) = length.filter(|&length| length != samples) {
            return Err(LengthError { length, samples });
        }
        let frame_samples = encoder.samples_per_frame() as u32;
        let mut ending = Ending {
            fragments: Fragments {
                fragmenter: &mut fragmenter,
                out,
            },
            last: None,
        };
        let finished = encoder.finish(&mut ending);
        // The encoder codes the frame that holds the input's last sample,
        // or with no input the delay's, only as it finishes, once it has
        // padded the input past it: so the last frame is here.
        let last = ending.last.take();
        debug_assert!(
            last.is_some(),
            "the last frame comes as the encoder finishes"
        );
        if let Some(last) = last {
            // Finished::padding is less than a frame.
            let last_duration = frame_samples - finished.padding as u32;
            let fragment = fragmenter.finish(&last, last_duration);
            out.fragment(&fragment.bytes, fragment.info);
        }
        Ok(finished)
    }
}

/// Gives each frame but the last to the fragments, which it holds back, so
/// that the last fragment can end with it, as the encoder finishes.
struct Ending<'a, O: ?Sized> {
    fragments: Fragments<'a, O>,
    last: Option<Vec<u8>>,
}

impl<O: FragmentSink + ?Sized> FrameSink for Ending<'_, O> {
    fn frame(&mut self, frame: &[u8]) {
        if let Some(before) = self.last.replace(frame.to_vec()) {
            self.fragments.frame(&before);
        }
    }
}

/// The `maxBitrate` and `avgBitrate` of a stream at `rate`, which a
/// [`FragmentedEncoder`] gives before its first frame: a bound on the
/// maximum that every stream so configured keeps to, and at a constant rate
/// an estimate of the average.
///
/// Spec reading: `avgBitrate` is "the average bitrate in bits per second of
/// this elementary stream" (ISO/IEC 14496-1, clause 6.2.6.6.2), which is
/// not known until the stream has ended. At a constant rate the rate asked
/// for, which counts the raw frames alone, is written in its place as an
/// estimate of it: the target that the frames keep to, but for what the
/// bit reservoir lends and keeps. It is not the stream's average, which an
/// [`Encoder`] gives as measured: that falls short of the target by what
/// the reservoir holds at the end, as a stream of one frame, a whole number
/// of bytes long, shows. At other rates it is 0, as for an [`Encoder`] (see
/// [`Encoder::finish`]).
///
/// Spec reading: `maxBitrate` is "the maximum bitrate in bits per second of
/// this elementary stream in any time window of one second duration"
/// (ISO/IEC 14496-1, clause 6.2.6.6.2), which has no value for "not known":
/// a bound is given, of the bits of any window, counted as an [`Encoder`]
/// counts them (see `bitrate::max_bitrate`), `w` frames at most. At a
/// constant rate the bit reservoir (ISO/IEC 13818-7, clause 8.2.3.2) holds
/// from 0 to `max_bit_reservoir` bits, at most the decoder's buffer less
/// the mean frame, so that `w` frames in a row take at most `w` mean frames
/// and `max_bit_reservoir` bits: less than the rate and the buffer, as `w -
/// 1` frames last less than a second. At other rates no frame is longer
/// than the decoder's buffer (clause 8.2.3.1): `w` buffers.
///
/// [`Encoder::finish`]: super::Encoder::finish
/// [`Encoder`]: super::Encoder
fn bitrates_before(rate: Rate, stream: &Stream) -> (u32, u32) {
    let buffer_bits = 8 * stream.buffer_size;
    match rate {
        Rate::Constant { bits_per_second } => {
            (bits_per_second.saturating_add(buffer_bits), bits_per_second)
        }
        _ => {
            let window = window_frames(stream.frame_samples, stream.sample_rate);
            (window.saturating_mul(buffer_bits), 0)
        }
    }
}

#[cfg(test)]
mod tests {
    use chelys_aac::DELAY;

    use super::*;
    use crate::aac::Encoder;
    use crate::boxes::read::{FourCcs, boxes};

    fn config(sample_rate: u32, channels: usize) -> Config {
        Config::new(
            sample_rate,
            channels,
            Rate::Constant {
                bits_per_second: 64_000 * channels as u32,
            },
        )
    }

    fn position(haystack: &[u8], needle: &[u8]) -> usize {
        (haystack.windows(needle.len()).position(|w| w == needle))
            .unwrap_or_else(|| panic!("{needle:?} not found"))
    }

    /// A tone of `len` samples in stereo, from sample `phase` on.
    fn tone(len: usize, phase: usize) -> Vec<f32> {
        (0..2 * len)
            .map(|i| ((phase + i / 2) as f32 * 0.03 + (i % 2) as f32).sin() * 0.3)
            .collect()
    }

    /// The fragmented file of `pcm`, in calls of `chunk` samples, and what
    /// each fragment holds.
    fn fragmented(
        config: Config,
        options: FragmentOptions,
        pcm: &[f32],
        chunk: usize,
    ) -> (Vec<u8>, Vec<crate::FragmentInfo>) {
        let mut encoder = FragmentedEncoder::new(config, options).unwrap();
        let mut file = encoder.init_segment().to_vec();
        let mut infos = Vec::new();
        let mut out = |fragment: &[u8], info: crate::FragmentInfo| {
            file.extend_from_slice(fragment);
            infos.push(info);
        };
        for part in pcm.chunks(chunk * encoder.channels()) {
            encoder.encode_interleaved(part, &mut out).unwrap();
        }
        encoder.finish(&mut out).unwrap();
        (file, infos)
    }

    /// The payloads of the MediaDataBoxes of `file`, one after another.
    fn media_data(file: &[u8]) -> Vec<u8> {
        (boxes(file, 0).iter())
            .filter(|b| &b.kind == b"mdat")
            .flat_map(|b| b.payload.to_vec())
            .collect()
    }

    #[test]
    fn the_fragments_hold_the_frames_of_the_file_written_whole() {
        let pcm = tone(5 * 16_000 + 77, 0);
        for rate in [
            Rate::Constant {
                bits_per_second: 48_000,
            },
            Rate::Quality { level: 50 },
        ] {
            let config = Config::new(16_000, 2, rate);
            let mut encoder = Encoder::new(config).unwrap();
            encoder.encode_interleaved(&pcm).unwrap();
            let mut whole = Vec::new();
            let finished = encoder.finish(&mut whole).unwrap();
            for chunk in [1000, 16_000] {
                let (file, infos) = fragmented(config, FragmentOptions::default(), &pcm, chunk);
                assert_eq!(media_data(&file), media_data(&whole), "{rate:?}");
                let top = boxes(&file, 0).kinds();
                assert_eq!(top[..2], ["ftyp", "moov"]);
                assert!(top[2..].chunks(2).all(|pair| pair == ["moof", "mdat"]));
                assert_eq!(infos.len(), top.len() / 2 - 1);
                let mut decode_time = 0;
                for (i, info) in infos.iter().enumerate() {
                    assert_eq!(info.sequence_number as usize, i + 1);
                    assert_eq!(info.decode_time, decode_time);
                    // The last frame lasts to the end of the input.
                    let padding = if i + 1 == infos.len() {
                        finished.padding
                    } else {
                        0
                    };
                    assert_eq!(info.duration, 1024 * u64::from(info.frames) - padding);
                    decode_time += info.duration;
                }
                assert_eq!(decode_time, DELAY as u64 + finished.samples);
                // A second at 16 kHz is 15.625 frames: the seconds end at
                // frames 16, 31, 47, 63 (62.5 rounded up) and 78.
                let frames: Vec<u32> = infos.iter().map(|info| info.frames).collect();
                assert_eq!(frames[..5], [16, 15, 16, 16, 15]);
            }
        }
    }

    /// Whatever the padding, from none to a frame's less one, with or
    /// without the length declared, and with a fragment ended just before
    /// the stream: the last fragment holds the last frame, which lasts to
    /// the end of the input, and the fragments' durations add up to the
    /// delay and the input.
    #[test]
    fn the_last_frame_lasts_to_the_end_of_the_input() {
        for len in [0, 1, 2, 1023, 1024, 1025, 2047, 2048, 3 * 1024 + 500] {
            for length in [None, Some(len as u64)] {
                for end_first in [false, true] {
                    let context = format!("{len} samples, {length:?}, {end_first}");
                    let mut options = FragmentOptions::default();
                    if let Some(length) = length {
                        options = options.length(length);
                    }
                    let mut encoder = FragmentedEncoder::new(config(44_100, 2), options).unwrap();
                    let mut infos = Vec::new();
                    let mut out = |_: &[u8], info: crate::FragmentInfo| infos.push(info);
                    encoder.encode_interleaved(&tone(len, 0), &mut out).unwrap();
                    if end_first {
                        encoder.end_fragment(&mut out);
                    }
                    let mut last_infos = Vec::new();
                    let mut last_out = |_: &[u8], info: crate::FragmentInfo| last_infos.push(info);
                    let finished = encoder.finish(&mut last_out).unwrap();
                    let held = infos.len();
                    infos.extend(last_infos);
                    assert!(finished.padding < 1024, "{context}");
                    assert_eq!(infos.len(), held + 1, "{context}");
                    let last = infos.last().unwrap();
                    let frames: u64 = infos.iter().map(|info| u64::from(info.frames)).sum();
                    assert_eq!(frames, finished.frames, "{context}");
                    assert_eq!(
                        last.duration,
                        1024 * u64::from(last.frames) - finished.padding,
                        "{context}"
                    );
                    let duration: u64 = infos.iter().map(|info| info.duration).sum();
                    assert_eq!(duration, DELAY as u64 + len as u64, "{context}");
                }
            }
        }
    }

    #[test]
    fn a_fragment_ends_when_asked_if_it_has_a_frame() {
        let config = config(44_100, 2);
        let mut encoder = FragmentedEncoder::new(config, FragmentOptions::default()).unwrap();
        let mut infos = Vec::new();
        let mut out = |_: &[u8], info: crate::FragmentInfo| infos.push(info);
        encoder.end_fragment(&mut out);
        // The encoder looks a frame ahead, so that a frame's input codes no
        // frame yet.
        encoder
            .encode_interleaved(&tone(1024, 0), &mut out)
            .unwrap();
        encoder.end_fragment(&mut out);
        encoder
            .encode_interleaved(&tone(10 * 1024, 1024), &mut out)
            .unwrap();
        encoder.end_fragment(&mut out);
        encoder.end_fragment(&mut out);
        encoder
            .encode_interleaved(&tone(100 * 1024, 11 * 1024), &mut out)
            .unwrap();
        let finished = encoder.finish(&mut out).unwrap();
        let frames: Vec<u32> = infos.iter().map(|info| info.frames).collect();
        // The first second ends at its 43rd frame, and the second at its 86th.
        assert_eq!(frames, [10, 33, 43, finished.frames as u32 - 86]);
        assert_eq!(infos.last().unwrap().sequence_number, 4);
    }

    #[test]
    fn the_input_is_held_to_its_declared_length() {
        let config = config(44_100, 2);
        let options = FragmentOptions::default().length(3000);
        let mut encoder = FragmentedEncoder::new(config, options).unwrap();
        let mut file = Vec::new();
        encoder
            .encode_interleaved(&tone(2000, 0), &mut file)
            .unwrap();
        let before = file.clone();
        let error = encoder.encode_interleaved(&tone(1001, 0), &mut file);
        let expected = LengthError {
            length: 3000,
            samples: 3001,
        };
        assert_eq!(error, Err(EncodeError::Length(expected)));
        let error = encoder.encode_planar(&[&[0.0f32; 1001][..], &[0.0; 1001]], &mut file);
        assert_eq!(error, Err(EncodeError::Length(expected)));
        assert_eq!((file == before, encoder.samples()), (true, 2000));
        let mut short = FragmentedEncoder::new(config, options).unwrap();
        short.encode_interleaved(&tone(2999, 0), &mut file).unwrap();
        let error = short.finish(&mut file).unwrap_err();
        assert_eq!((error.length, error.samples), (3000, 2999));
        encoder
            .encode_interleaved(&tone(1000, 2000), &mut file)
            .unwrap();
        assert_eq!(encoder.finish(&mut file).unwrap().samples, 3000);
    }

    /// A call refused, for a sample that is not finite or for the input's
    /// length, though it holds whole frames before the sample, leaves the
    /// fragments as if it had not been made.
    #[test]
    fn a_refused_call_leaves_the_fragments_unchanged() {
        let (a, b) = (tone(30_000, 0), tone(60_000, 30_000));
        let mut refused = tone(50_000, 7000);
        refused[2 * 45_000 + 1] = f32::NAN;
        let options = FragmentOptions::default().length(90_000);
        let file = |refuse: &dyn Fn(&mut FragmentedEncoder, &mut Vec<u8>)| {
            let mut encoder = FragmentedEncoder::new(config(48_000, 2), options).unwrap();
            let mut file = encoder.init_segment().to_vec();
            encoder.encode_interleaved(&a, &mut file).unwrap();
            refuse(&mut encoder, &mut file);
            encoder.encode_interleaved(&b, &mut file).unwrap();
            encoder.finish(&mut file).unwrap();
            file
        };
        let whole = file(&|_, _| {});
        let non_finite = file(&|encoder, file| {
            let error = encoder.encode_interleaved(&refused, file).unwrap_err();
            assert!(matches!(error, EncodeError::NonFiniteSample(_)), "{error}");
        });
        let too_long = file(&|encoder, file| {
            let error = encoder
                .encode_interleaved(&tone(60_001, 0), file)
                .unwrap_err();
            assert!(matches!(error, EncodeError::Length(_)), "{error}");
        });
        assert!(non_finite == whole);
        assert!(too_long == whole);
    }

    /// The `maxBitrate` and `avgBitrate` of the `ES_Descriptor` in `file`.
    fn bitrates(file: &[u8]) -> (u32, u32) {
        let at = position(file, &[0x04, 17, 0x40, 0x15]) + 4;
        let field = |at: usize| u32::from_be_bytes(file[at..at + 4].try_into().unwrap());
        (field(at + 3), field(at + 7))
    }

    /// The maximum given before the first frame bounds the maximum measured
    /// once the stream has ended, and at a constant rate the average given,
    /// the rate asked for, is an estimate that the average measured does not
    /// exceed; over streams of tones and noise, silence and data streams, at
    /// every kind of rate.
    #[test]
    fn the_bitrates_given_in_advance_hold_for_those_measured() {
        let mut seed = 1u32;
        let mut noise = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 24) as f32 - 0.5
        };
        let mixed: Vec<f32> = (0..2 * 44_100)
            .map(|i| match i / 20_000 {
                0 => 0.0,
                1 | 3 => noise(),
                _ => (i as f32 * 0.05).sin() * 0.3,
            })
            .collect();
        for (sample_rate, len) in [
            (8_000, 3 * 8_000),
            (44_100, 2 * 44_100),
            (96_000, 3 * 48_000 / 2),
        ] {
            let pcm = &mixed[..len];
            for rate in [
                Rate::Constant {
                    bits_per_second: sample_rate,
                },
                Rate::Constant {
                    bits_per_second: sample_rate / 4,
                },
                Rate::Mean {
                    bits_per_second: sample_rate / 2,
                },
                Rate::Variable {
                    noise_offset_db: -6.0,
                },
                Rate::Quality { level: 100 },
            ] {
                for data_stream in [0, 16] {
                    let context = format!("{sample_rate} Hz, {rate:?}, {data_stream}");
                    let config =
                        Config::new(sample_rate, 1, rate).data_stream_bytes_per_frame(data_stream);
                    let mut encoder = Encoder::new(config).unwrap();
                    encoder.push_data_stream(&vec![7; 1 << 20]);
                    encoder.encode_interleaved(pcm).unwrap();
                    let mut whole = Vec::new();
                    encoder.finish(&mut whole).unwrap();
                    let fragmented = FragmentedEncoder::new(config, FragmentOptions::default());
                    let init = fragmented.unwrap().init_segment().to_vec();
                    let (measured_max, measured_avg) = bitrates(&whole);
                    let (max, avg) = bitrates(&init);
                    assert!(measured_max <= max, "{context}: {measured_max} > {max}");
                    match rate {
                        Rate::Constant { bits_per_second } => {
                            assert_eq!(avg, bits_per_second, "{context}");
                            assert!(measured_avg <= avg, "{context}");
                        }
                        _ => assert_eq!(avg, 0, "{context}"),
                    }
                }
            }
        }
    }
}
