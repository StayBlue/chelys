// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! What an [`Encoder`](super::Encoder) is asked for, and its checks: a
//! [`Config`] is lowered to the stream's configuration only if every frame
//! the encoder could write is one the standard allows.

use std::fmt;

use super::adts::Flags;
use super::channel_configuration::ChannelConfiguration;
use super::sampling_frequency::SamplingFrequency;
use super::stream::StreamConfig;

/// The sample rates the encoder takes, in Hz: those that have a
/// `sampling_frequency_index` (Table 35), highest first.
pub const SAMPLE_RATES: [u32; 12] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
];

/// How the frames' lengths are chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Rate {
    /// A constant rate: frames as long as a channel of `bits_per_second`
    /// carries, give or take what the bit reservoir lends and keeps
    /// (clause 8.2.3.2), and padded with fill where it is full. The rate
    /// counts every byte written, ADTS headers and their CRC included.
    Constant { bits_per_second: u32 },
}

/// How each frame is given to the [`FrameSink`](super::FrameSink).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Framing {
    /// Each frame in an `adts_frame()`, whose header tells a decoder all it
    /// needs (clause 8.1.3): a stream that plays as it is, as an `.aac`
    /// file does.
    #[default]
    Adts,
    /// Each frame as its `raw_data_block()` alone, for a container that
    /// keeps the frames apart and carries the
    /// [`Encoder::audio_specific_config`](super::Encoder::audio_specific_config),
    /// such as an MPEG-4 file.
    Raw,
}

/// What to encode, and how: the sample rate, channels and rate, and options
/// set by the methods that take and return it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    sample_rate: u32,
    channels: usize,
    rate: Rate,
    framing: Framing,
    flags: Flags,
    crc: bool,
    data_stream_bytes_per_frame: usize,
}

impl Config {
    /// A configuration of `channels` channels, 1 (mono) or 2 (stereo), in
    /// ADTS. [`Encoder::new`] checks it.
    ///
    /// [`Encoder::new`]: super::Encoder::new
    pub fn new(sample_rate: u32, channels: usize, rate: Rate) -> Self {
        Self {
            sample_rate,
            channels,
            rate,
            framing: Framing::default(),
            flags: Flags::default(),
            crc: false,
            data_stream_bytes_per_frame: 0,
        }
    }

    /// How each frame is given: in ADTS, the default, or raw.
    pub fn framing(mut self, framing: Framing) -> Self {
        self.framing = framing;
        self
    }

    /// Protects each ADTS frame's header and the start of its audio with a
    /// CRC-16 check word, which decoders may check to detect errors: the
    /// first 192 bits of each channel element and the first 128 of the
    /// second channel of a pair, and all of any data stream element (clause
    /// 8.1.3.2.1). The word's 16 bits count towards the rate. ADTS only.
    pub fn crc(mut self, on: bool) -> Self {
        self.crc = on;
        self
    }

    /// Sets each ADTS header's `private_bit`, which ISO/IEC 11172-3 defines
    /// as a "bit for private use" (clause 8.1.3.2.2 refers to it). ADTS
    /// only.
    pub fn private_bit(mut self, on: bool) -> Self {
        self.flags.private_bit = on;
        self
    }

    /// Sets each ADTS header's `original_copy`. ADTS only.
    ///
    /// The standard defines it by reference to ISO/IEC 11172-3's
    /// `copyright`, which is set where the stream is copyright protected,
    /// though its name is that of 11172-3's `original/copy` (clauses
    /// 8.1.3.2.2 and 8.1.2.2.2): the bit is set as asked, and what it
    /// marks is left to the reader.
    pub fn original_copy(mut self, on: bool) -> Self {
        self.flags.original_copy = on;
        self
    }

    /// Sets each ADTS header's `home`. ADTS only.
    ///
    /// The standard defines it by reference to ISO/IEC 11172-3's
    /// `original/copy`, which is set where the stream is an original rather
    /// than a copy (clauses 8.1.3.2.2 and 8.1.2.2.2).
    pub fn home(mut self, on: bool) -> Self {
        self.flags.home = on;
        self
    }

    /// Gives each frame up to `bytes` bytes of data, from the bytes queued
    /// with [`Encoder::push_data_stream`], in data stream elements (clause
    /// 8.6) after its audio, which then has as many bits fewer, and some
    /// more for the elements' fields: 16 bits an element, 24 from 255
    /// bytes, and an element for each 510. A frame with less data queued
    /// keeps the rest for audio. At most as many as the rate brings each
    /// frame beyond the shortest.
    ///
    /// The elements carry one data stream, with the `element_instance_tag`
    /// 0, its bytes unaligned (`data_byte_align_flag` 0). A decoder ignores
    /// them, or gives their bytes, "any additional data, e.g. auxiliary
    /// information, that is not part of the audio information itself"
    /// (clause 8.6.2), in order; as each frame says how many it carries,
    /// they need no framing of their own to be told from the audio.
    ///
    /// [`Encoder::push_data_stream`]: super::Encoder::push_data_stream
    pub fn data_stream_bytes_per_frame(mut self, bytes: usize) -> Self {
        self.data_stream_bytes_per_frame = bytes;
        self
    }

    /// Whether [`Encoder::new`] takes this configuration.
    ///
    /// # Errors
    ///
    /// As [`Encoder::new`].
    ///
    /// [`Encoder::new`]: super::Encoder::new
    pub fn check(&self) -> Result<(), ConfigError> {
        self.lower().map(|_| ())
    }

    pub(super) fn data_stream_bytes(&self) -> usize {
        self.data_stream_bytes_per_frame
    }

    /// The stream's configuration, if it is one the stream can have.
    pub(super) fn lower(&self) -> Result<StreamConfig, ConfigError> {
        let sample_rate = self.sample_rate;
        let sampling_frequency = SamplingFrequency::from_hz(sample_rate)
            .ok_or(ConfigError::UnsupportedSampleRate { sample_rate })?;
        let channel_configuration = ChannelConfiguration::from_channels(self.channels).ok_or(
            ConfigError::UnsupportedChannels {
                channels: self.channels,
            },
        )?;
        if self.framing == Framing::Raw && (self.crc || self.flags != Flags::default()) {
            return Err(ConfigError::AdtsOnly);
        }
        let Rate::Constant { bits_per_second } = self.rate;
        let config = StreamConfig {
            sampling_frequency,
            channel_configuration,
            bits_per_second,
            framing: self.framing,
            flags: self.flags,
            crc_protected: self.crc,
        };
        if !config.is_valid() {
            return Err(ConfigError::UnsupportedBitrate {
                bits_per_second,
                min: config.min_bits_per_second(),
                max: config.max_bits_per_second(),
            });
        }
        let max = config.max_data_stream_bytes();
        if self.data_stream_bytes_per_frame > max {
            return Err(ConfigError::DataStreamTooLong {
                bytes_per_frame: self.data_stream_bytes_per_frame,
                max,
            });
        }
        Ok(config)
    }
}

/// Why a [`Config`] cannot be encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConfigError {
    /// The sample rate is not one of [`SAMPLE_RATES`].
    UnsupportedSampleRate { sample_rate: u32 },
    /// The channels are neither 1 nor 2.
    UnsupportedChannels { channels: usize },
    /// The bitrate is outside the range, from `min` to `max`, of the sample
    /// rate, channels and framing: below the bits of the shortest frame, or
    /// above those of the decoder's input buffer, 6144 a channel (clause
    /// 8.2.3.3), less a byte for the bit reservoir.
    UnsupportedBitrate {
        bits_per_second: u32,
        min: u32,
        max: u32,
    },
    /// The CRC or a header flag was asked for in [`Framing::Raw`], whose
    /// frames have no ADTS header to carry them.
    AdtsOnly,
    /// More bytes of data streams a frame were asked for than the rate
    /// brings each frame beyond the shortest, `max`.
    DataStreamTooLong { bytes_per_frame: usize, max: usize },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::UnsupportedSampleRate { sample_rate } => {
                write!(f, "unsupported sample rate: {sample_rate}")
            }
            Self::UnsupportedChannels { channels } => {
                write!(f, "unsupported number of channels: {channels}")
            }
            Self::UnsupportedBitrate {
                bits_per_second,
                min,
                max,
            } => write!(
                f,
                "unsupported bitrate: {bits_per_second} bit/s, not {min} to {max} bit/s"
            ),
            Self::AdtsOnly => write!(
                f,
                "raw frames have no ADTS header to carry its CRC or flags"
            ),
            Self::DataStreamTooLong {
                bytes_per_frame,
                max,
            } => write!(
                f,
                "{bytes_per_frame} bytes of data streams a frame exceed the {max} the bitrate \
                 brings each frame beyond the shortest"
            ),
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Encoder;

    fn config(framing: Framing) -> Config {
        Config::new(
            48_000,
            2,
            Rate::Constant {
                bits_per_second: 96_000,
            },
        )
        .framing(framing)
    }

    #[test]
    fn every_adts_header_carries_the_flags() {
        let config = config(Framing::Adts)
            .private_bit(true)
            .original_copy(false)
            .home(true);
        let mut encoder = Encoder::new(config).unwrap();
        let mut frames = Vec::new();
        let mut out = |frame: &[u8]| frames.push(frame.to_vec());
        encoder.encode_interleaved(&[0.0f32; 2 * 4096], &mut out);
        encoder.finish(&mut out);
        assert_eq!(frames.len(), 5);
        for frame in frames {
            // private_bit is bit 22, original_copy 26 and home 27.
            assert_eq!(frame[2] & 0b10, 0b10);
            assert_eq!((frame[3] >> 4) & 0b11, 0b01);
        }
    }

    #[test]
    fn a_frame_carries_at_most_what_the_rate_brings_beyond_the_shortest() {
        let max = config(Framing::Adts)
            .lower()
            .unwrap()
            .max_data_stream_bytes();
        // 256 bytes a frame, of which the header takes 7, and a pair of
        // silent channels with its id_syn_ele, ID_END and the data stream
        // element's fields 71 bits, within 9 bytes with the data.
        assert_eq!(max, 256 - 7 - 9);
        for framing in [Framing::Adts, Framing::Raw] {
            let ok = config(framing).data_stream_bytes_per_frame(max);
            assert_eq!(ok.check(), Ok(()));
        }
        let over = config(Framing::Adts).data_stream_bytes_per_frame(max + 1);
        assert_eq!(
            over.check(),
            Err(ConfigError::DataStreamTooLong {
                bytes_per_frame: max + 1,
                max
            })
        );
    }

    #[test]
    fn raw_frames_have_no_header_for_the_crc_or_flags() {
        assert_eq!(config(Framing::Raw).check(), Ok(()));
        assert_eq!(config(Framing::Raw).home(false).check(), Ok(()));
        for config in [
            config(Framing::Raw).crc(true),
            config(Framing::Raw).private_bit(true),
            config(Framing::Raw).original_copy(true),
            config(Framing::Raw).home(true),
        ] {
            assert_eq!(config.check(), Err(ConfigError::AdtsOnly));
        }
    }
}
