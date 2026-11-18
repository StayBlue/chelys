// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! What an [`Encoder`](super::Encoder) is asked for, and its checks: a
//! [`Config`] is lowered to the stream's configuration only if every frame
//! the encoder could write is one the standard allows.

use std::fmt;

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
    /// counts every byte written, ADTS headers included.
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
        }
    }

    /// How each frame is given: in ADTS, the default, or raw.
    pub fn framing(mut self, framing: Framing) -> Self {
        self.framing = framing;
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
        let Rate::Constant { bits_per_second } = self.rate;
        let config = StreamConfig {
            sampling_frequency,
            channel_configuration,
            bits_per_second,
            framing: self.framing,
        };
        if !config.is_valid() {
            return Err(ConfigError::UnsupportedBitrate {
                bits_per_second,
                min: config.min_bits_per_second(),
                max: config.max_bits_per_second(),
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
        }
    }
}

impl std::error::Error for ConfigError {}
