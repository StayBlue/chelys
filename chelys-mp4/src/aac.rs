// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! AAC-LC from chelys's encoder in an MP4 file: its raw frames as the
//! samples of an `'mp4a'` track, which its `AudioSpecificConfig()`
//! describes.

use std::io::{self, Write};

use chelys_aac::{Config, ConfigError, Finished, FrameSink, Framing, Sample};

use super::bitrate::{avg_bitrate, max_bitrate};
use super::descriptor::{AUDIO_ISO_IEC_14496_3, AUDIO_STREAM, DecoderConfig, es_descriptor};
use super::movie::{AudioTrack, write_file};

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
    /// `raw_data_block()` (see [`Framing::Raw`]), as ISO/IEC 14496-3 maps
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
            encoder: chelys_aac::Encoder::new(config.framing(Framing::Raw))?,
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
    /// # Panics
    ///
    /// As [`chelys_aac::Encoder::encode_interleaved`].
    pub fn encode_interleaved<S: Sample>(&mut self, samples: &[S]) {
        self.encoder.encode_interleaved(samples, &mut self.frames);
    }

    /// Encodes one slice of samples per channel, as
    /// [`chelys_aac::Encoder::encode_planar`].
    ///
    /// # Panics
    ///
    /// As [`chelys_aac::Encoder::encode_planar`].
    pub fn encode_planar<S: Sample>(&mut self, channels: &[&[S]]) {
        self.encoder.encode_planar(channels, &mut self.frames);
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
        let sample_rate = encoder.sample_rate();
        let channels = encoder.channels();
        let frame_samples = encoder.samples_per_frame() as u32;
        let specific_info = encoder.audio_specific_config();
        let buffer_size = encoder.decoder_buffer_bytes();
        let finished = encoder.finish(&mut frames);
        // Spec reading: the objectTypeIndication is "Audio ISO/IEC
        // 14496-3", 0x40, whose DecoderSpecificInfo is the
        // AudioSpecificConfig() (ISO/IEC 14496-3, clause 1.6.2.1), which
        // names the stream AAC LC: with perceptual noise substitution it is
        // MPEG-4 AAC, and without it AAC LC of either standard. That of
        // ISO/IEC 13818-7's Low Complexity profile, 0x67, would take an
        // adif_header() as its DecoderSpecificInfo instead (ISO/IEC
        // 14496-1, clause 6.2.6.7.2).
        //
        // Spec reading: bufferSizeDB, "the size of the decoding buffer",
        // the buffer at the decoder's input (ISO/IEC 14496-1, clause
        // 6.1.2.4), is that of the decoder's input buffer of ISO/IEC
        // 13818-7 (clause 8.2.3.1), within which the bit reservoir keeps
        // the stream.
        //
        // Spec reading: avgBitrate "shall be set to zero" "For streams with
        // variable bitrate". A stream coded at a constant rate, the bit
        // reservoir (ISO/IEC 13818-7, clause 8.2.3.2) letting its frames
        // vary about the mean, is not one: its average is given, measured
        // over the stream. Rate::is_constant decides which a stream is, so
        // that a rate of another kind is decided there.
        let avg_bitrate = avg_bitrate(&frames.sizes, frame_samples, sample_rate);
        let es_descriptor = es_descriptor(&DecoderConfig {
            object_type_indication: AUDIO_ISO_IEC_14496_3,
            stream_type: AUDIO_STREAM,
            buffer_size,
            max_bitrate: max_bitrate(&frames.sizes, frame_samples, sample_rate),
            avg_bitrate: if constant_rate { avg_bitrate } else { 0 },
            specific_info: &specific_info,
        });
        let track = AudioTrack {
            coding_name: *b"mp4a",
            es_descriptor: &es_descriptor,
            sample_rate,
            channels: channels as u16,
            sizes: &frames.sizes,
            frame_samples,
            delay: chelys_aac::DELAY as u32,
            length: finished.samples,
            roll_distance: -1,
        };
        write_file(&track, &frames.data, out)?;
        Ok(finished)
    }
}

#[cfg(test)]
mod tests {
    use chelys_aac::Rate;

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
        encoder.encode_interleaved(&pcm);
        let mut frames = Vec::new();
        raw.encode_interleaved(&pcm, &mut frames);
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
        encoder.encode_interleaved(&vec![0.0f32; 6 * 44_100]);
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
        encoder.encode_interleaved(&pcm);
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
            encoder.encode_interleaved(&pcm);
            let mut file = Vec::new();
            encoder.finish(&mut file).unwrap();
            let at = position(&file, &[0x04, 17, 0x40, 0x15]) + 4;
            let field = |at: usize| u32::from_be_bytes(file[at..at + 4].try_into().unwrap());
            assert!(field(at + 3) > 0, "{rate:?}");
            assert_eq!(field(at + 7), 0, "{rate:?}");
        }
    }
}
