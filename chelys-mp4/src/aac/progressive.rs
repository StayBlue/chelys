// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The MP4 file written whole once the stream has ended.

use std::io::{self, Write};

use chelys_aac::{Config, ConfigError, Finished, FrameSink, NonFiniteSample, Sample};

use super::{Stream, raw_encoder};
use crate::bitrate::{avg_bitrate, max_bitrate};
use crate::movie::progressive::write_file;

/// Encodes PCM as AAC-LC, as [`chelys_aac::Encoder`] does, into an MP4
/// file, which [`finish`](Self::finish) writes whole.
///
/// The file gives a player the input's exact length and place: its edit
/// list leaves out the [`DELAY`](chelys_aac::DELAY) samples that a
/// decoder's output starts with, and the [`Finished::padding`] it ends
/// with, as an ADTS stream cannot.
///
/// The frames are kept in memory until [`finish`](Self::finish) writes the
/// file, its sample tables before them, as they describe each frame: about
/// the bitrate times the duration, 4.8 MB for five minutes at 128 kbit/s.
/// A [`FragmentedEncoder`](super::FragmentedEncoder) writes a file as the
/// frames come instead.
pub struct Encoder {
    encoder: chelys_aac::Encoder,
    /// Whether the stream is coded at a constant rate, which sets its
    /// `avgBitrate`.
    constant_rate: bool,
    frames: Frames,
}

/// The frames of the stream, one after another, and their sizes.
#[derive(Default)]
struct Frames {
    data: Vec<u8>,
    sizes: Vec<u32>,
}

impl FrameSink for Frames {
    fn frame(&mut self, frame: &[u8]) {
        self.data.extend_from_slice(frame);
        self.sizes.push(frame.len() as u32);
    }
}

impl Encoder {
    /// An encoder of the stream `config` asks for, each frame as its
    /// `raw_data_block()` (see [`Framing::Raw`](chelys_aac::Framing::Raw)), as ISO/IEC 14496-3 maps
    /// each to an access unit, and so an MP4 sample (clause 1.6.2.2.2.1),
    /// whatever framing `config` asks for.
    ///
    /// # Errors
    ///
    /// As [`chelys_aac::Encoder::new`] with raw frames: among others,
    /// [`ConfigError::AdtsOnly`] if `config` asks for the CRC or a header
    /// flag of ADTS.
    pub fn new(config: Config) -> Result<Self, ConfigError> {
        Ok(Self {
            encoder: raw_encoder(config)?,
            constant_rate: config.rate().is_constant(),
            frames: Frames::default(),
        })
    }

    /// The number of channels the input has.
    pub fn channels(&self) -> usize {
        self.encoder.channels()
    }

    /// Encodes interleaved samples, as [`chelys_aac::Encoder::encode_interleaved`].
    ///
    /// # Errors
    ///
    /// [`NonFiniteSample`], as [`chelys_aac::Encoder::encode_interleaved`], if a
    /// sample is NaN or infinite: none of the call is taken. No frame is
    /// added to the file, and the encoder is as it was before the call.
    ///
    /// # Panics
    ///
    /// As [`chelys_aac::Encoder::encode_interleaved`].
    pub fn encode_interleaved<S: Sample>(&mut self, samples: &[S]) -> Result<(), NonFiniteSample> {
        self.encoder.encode_interleaved(samples, &mut self.frames)
    }

    /// Encodes one slice of samples per channel, as
    /// [`chelys_aac::Encoder::encode_planar`].
    ///
    /// # Errors
    ///
    /// [`NonFiniteSample`], as [`chelys_aac::Encoder::encode_planar`], if a
    /// sample is NaN or infinite: none of the call is taken. No frame is
    /// added to the file, and the encoder is as it was before the call.
    ///
    /// # Panics
    ///
    /// As [`chelys_aac::Encoder::encode_planar`].
    pub fn encode_planar<S: Sample>(&mut self, channels: &[&[S]]) -> Result<(), NonFiniteSample> {
        self.encoder.encode_planar(channels, &mut self.frames)
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

    /// Ends the stream, as [`chelys_aac::Encoder::finish`], and writes the
    /// file to `out`, which need not be seekable, as the file is written
    /// from its start to its end.
    ///
    /// # Errors
    ///
    /// If `out` fails. The frames are then lost with the encoder, and `out`
    /// may already hold the start of the file, which is not a file that
    /// plays: it is to be discarded.
    pub fn finish<W: Write + ?Sized>(self, out: &mut W) -> io::Result<Finished> {
        let Self {
            encoder,
            constant_rate,
            mut frames,
        } = self;
        let stream = Stream::of(&encoder);
        let finished = encoder.finish(&mut frames);
        // Spec reading: avgBitrate "shall be set to zero" "For streams with
        // variable bitrate". A stream coded at a constant rate, the bit
        // reservoir (ISO/IEC 13818-7, clause 8.2.3.2) letting its frames
        // vary about the mean, is not one: its average is given, measured
        // over the stream. Rate::is_constant decides which a stream is, so
        // that a rate of another kind is decided there.
        let (frame_samples, sample_rate) = (stream.frame_samples, stream.sample_rate);
        let avg_bitrate = avg_bitrate(&frames.sizes, frame_samples, sample_rate);
        let track = stream.track(
            max_bitrate(&frames.sizes, frame_samples, sample_rate),
            if constant_rate { avg_bitrate } else { 0 },
        );
        let durations = vec![frame_samples; frames.sizes.len()];
        write_file(
            &track,
            &frames.sizes,
            &durations,
            finished.samples,
            &frames.data,
            out,
        )?;
        Ok(finished)
    }
}

#[cfg(test)]
mod tests {
    use chelys_aac::{Framing, Rate};

    use super::*;

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

    #[test]
    fn the_file_holds_the_raw_frames_and_their_config() {
        let mut encoder = Encoder::new(config(48_000, 2)).unwrap();
        let mut raw = chelys_aac::Encoder::new(config(48_000, 2).framing(Framing::Raw)).unwrap();
        let pcm: Vec<f32> = (0..2 * 5000)
            .map(|i| ((i as f32) * 0.01).sin() * 0.3)
            .collect();
        encoder.encode_interleaved(&pcm).unwrap();
        let mut frames = Vec::new();
        raw.encode_interleaved(&pcm, &mut frames).unwrap();
        let asc = raw.audio_specific_config();
        raw.finish(&mut frames);
        let mut file = Vec::new();
        let finished = encoder.finish(&mut file).unwrap();
        assert_eq!(finished.samples, 5000);
        assert_eq!(finished.frames, 6);
        assert!(file.ends_with(&frames));
        // The DecSpecificInfoTag, its size and the AudioSpecificConfig().
        position(&file, &[&[0x05, 2][..], &asc].concat());
        let elst = position(&file, b"elst");
        assert_eq!(
            &file[elst + 12..elst + 20],
            [&5000u32.to_be_bytes()[..], &1024u32.to_be_bytes()].concat()
        );
    }

    #[test]
    fn adts_options_are_refused() {
        assert_eq!(
            Encoder::new(config(44_100, 1).crc(true)).err(),
            Some(ConfigError::AdtsOnly)
        );
        assert!(Encoder::new(config(44_100, 1).framing(Framing::Adts)).is_ok());
    }

    #[test]
    fn the_decoder_config_gives_the_buffer_and_bitrates() {
        let mut encoder = Encoder::new(config(44_100, 6)).unwrap();
        encoder
            .encode_interleaved(&vec![0.0f32; 6 * 44_100])
            .unwrap();
        let mut file = Vec::new();
        let frames = encoder.finish(&mut file).unwrap().frames;
        let at = position(&file, &[0x04, 17, 0x40, 0x15]) + 4;
        // 5.1: 6144 bits for each of 5 channels.
        assert_eq!(file[at..at + 3], [0, 0x0f, 0x00]);
        let max = u32::from_be_bytes(file[at + 3..at + 7].try_into().unwrap());
        let avg = u32::from_be_bytes(file[at + 7..at + 11].try_into().unwrap());
        assert!(avg <= max, "{avg} {max}");
        // The rate, less what the reservoir holds at the end, at most the
        // buffer, over the stream's duration.
        let duration = frames as f64 * 1024.0 / 44_100.0;
        let held = (6144.0 * 5.0 / duration) as u32;
        assert!((384_000 - held..=384_000).contains(&avg), "{avg}");
    }

    /// The bufferSizeDB, maxBitrate and avgBitrate of the file of `len`
    /// samples of a tone in stereo at 44.1 kHz.
    fn decoder_config_of(len: usize) -> (u32, u32, u32) {
        let mut encoder = Encoder::new(config(44_100, 2)).unwrap();
        let pcm: Vec<f32> = (0..2 * len)
            .map(|i| ((i / 2) as f32 * 0.05).sin() * 0.3)
            .collect();
        encoder.encode_interleaved(&pcm).unwrap();
        let mut file = Vec::new();
        encoder.finish(&mut file).unwrap();
        let at = position(&file, &[0x04, 17, 0x40, 0x15]) + 4;
        let field = |at: usize| u32::from_be_bytes(file[at..at + 4].try_into().unwrap());
        (field(at - 1) & 0xff_ffff, field(at + 3), field(at + 7))
    }

    #[test]
    fn short_and_empty_streams_have_a_maximum_no_lower_than_their_average() {
        for len in [0, 1, 1000, 22_050, 44_100, 3 * 44_100] {
            let (buffer, max, avg) = decoder_config_of(len);
            assert_eq!(buffer, 1536, "{len}");
            assert!(avg > 0 && avg <= max, "{len} samples: {avg} > {max}");
        }
    }

    /// A stream at a variable or mean rate has an avgBitrate of 0, as
    /// ISO/IEC 14496-1 asks "For streams with variable bitrate", and its
    /// maxBitrate as measured.
    #[test]
    fn a_variable_rate_has_no_average() {
        for rate in [
            Rate::Variable {
                noise_offset_db: 0.0,
            },
            Rate::Mean {
                bits_per_second: 96_000,
            },
        ] {
            let mut encoder = Encoder::new(Config::new(44_100, 2, rate)).unwrap();
            let pcm: Vec<f32> = (0..2 * 44_100)
                .map(|i| ((i / 2) as f32 * 0.05).sin() * 0.3)
                .collect();
            encoder.encode_interleaved(&pcm).unwrap();
            let mut file = Vec::new();
            encoder.finish(&mut file).unwrap();
            let at = position(&file, &[0x04, 17, 0x40, 0x15]) + 4;
            let field = |at: usize| u32::from_be_bytes(file[at..at + 4].try_into().unwrap());
            assert!(field(at + 3) > 0, "{rate:?}");
            assert_eq!(field(at + 7), 0, "{rate:?}");
        }
    }

    /// A call refused for a sample that is not finite, though it holds
    /// whole frames before the sample, leaves the file as if it had not
    /// been made, whether more samples follow it or the stream ends there.
    #[test]
    fn a_refused_call_leaves_the_file_unchanged() {
        let tone = |len: usize, phase: usize| -> Vec<f32> {
            (0..2 * len)
                .map(|i| ((phase + i / 2) as f32 * 0.03 + (i % 2) as f32).sin() * 0.3)
                .collect()
        };
        let (a, b) = (tone(1500, 0), tone(2000, 1500));
        let mut refused = tone(3000, 7000);
        refused[2 * 2500 + 1] = f32::NAN;
        let refused_planar: Vec<Vec<f32>> = (0..2)
            .map(|c| refused.iter().copied().skip(c).step_by(2).collect())
            .collect();

        // The data stream's bytes are queued before the refused call, and
        // the frames after it carry them, a few to each.
        let file = |data: &[u8],
                    refuse: &dyn Fn(&mut Encoder) -> Option<NonFiniteSample>,
                    rest: Option<&[f32]>| {
            let config = config(48_000, 2).data_stream_bytes_per_frame(4);
            let mut encoder = Encoder::new(config).unwrap();
            encoder.encode_interleaved(&a).unwrap();
            encoder.push_data_stream(data);
            let pending = encoder.data_stream_pending();
            if let Some(error) = refuse(&mut encoder) {
                assert_eq!((error.channel, error.sample), (1, 2500));
            }
            assert_eq!(encoder.data_stream_pending(), pending);
            if let Some(rest) = rest {
                encoder.encode_interleaved(rest).unwrap();
            }
            let mut file = Vec::new();
            let finished = encoder.finish(&mut file).unwrap();
            assert_eq!(finished.data_stream_dropped, 0);
            file
        };
        for rest in [Some(&b[..]), None] {
            let whole = file(b"chelys", &|_| None, rest);
            // The comparisons below would see a change in the queued bytes.
            assert!(
                file(b"chelyz", &|_| None, rest) != whole,
                "{}",
                rest.is_some()
            );
            let interleaved = file(
                b"chelys",
                &|encoder| Some(encoder.encode_interleaved(&refused).unwrap_err()),
                rest,
            );
            let planar = file(
                b"chelys",
                &|encoder| {
                    Some(
                        encoder
                            .encode_planar(&[&refused_planar[0], &refused_planar[1]])
                            .unwrap_err(),
                    )
                },
                rest,
            );
            assert!(interleaved == whole, "interleaved, {}", rest.is_some());
            assert!(planar == whole, "planar, {}", rest.is_some());
        }
    }
}
