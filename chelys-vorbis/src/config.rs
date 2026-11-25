// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! What an [`Encoder`](super::Encoder) is asked for, and its checks.

use std::fmt;

use super::calibration;

/// The most channels the encoder takes: those of the largest layout that
/// clause 4.3.9 orders, 7.1.
pub const MAX_CHANNELS: usize = 8;

/// How much the stream may take.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Rate {
    /// A variable rate: each line of a block is allowed the noise that the
    /// psychoacoustic model allows it, scaled by `10^(noise_offset_db /
    /// 10)`, and each packet is as long as that takes. Above 0 it allows
    /// more noise and gives shorter packets, below 0 less. No line is
    /// allowed less noise than the rounding of 16-bit PCM makes, whatever
    /// the offset. At 44.1 kHz in stereo, music takes about 181 kbit/s at
    /// 0 dB.
    Variable { noise_offset_db: f64 },
    /// A variable rate at `level`: a level of the [`quality`](crate::quality)
    /// scale that chelys's encoders share, from 0 to
    /// [`MAX_LEVEL`](crate::quality::MAX_LEVEL), higher for better quality.
    ///
    /// It is [`Rate::Variable`] with the noise offset the level maps to,
    /// calibrated to the scale's grades at 44.1 kHz in two channels: music
    /// in stereo averages about 144 kbit/s at level 50, 263 at 80 and 303
    /// at 100. Every other configuration takes the same offsets, so that
    /// there a level gives the scale's grades only roughly.
    Quality { level: u8 },
}

/// What to encode: the sample rate, the channels and the rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    sample_rate: u32,
    channels: usize,
    rate: Rate,
}

impl Config {
    /// A configuration of `channels` channels, from 1 to [`MAX_CHANNELS`],
    /// at `sample_rate` Hz, which is any but 0, coded at `rate`.
    /// [`Encoder::new`] checks it.
    ///
    /// The input gives the channels in the order the stream sends them,
    /// which clause 4.3.9 maps to speakers:
    ///
    /// | `channels` | Order |
    /// |---|---|
    /// | 1 | mono |
    /// | 2 | left, right |
    /// | 3 | left, center, right |
    /// | 4 | front left, front right, rear left, rear right |
    /// | 5 | front left, center, front right, rear left, rear right |
    /// | 6 | front left, center, front right, rear left, rear right, LFE |
    /// | 7 | front left, center, front right, side left, side right, rear center, LFE |
    /// | 8 | front left, center, front right, side left, side right, rear left, rear right, LFE |
    ///
    /// Spec reading: above eight channels "channel use and order is defined
    /// by the application"; chelys defines none, and refuses them.
    ///
    /// [`Encoder::new`]: super::Encoder::new
    pub fn new(sample_rate: u32, channels: usize, rate: Rate) -> Self {
        Self {
            sample_rate,
            channels,
            rate,
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn rate(&self) -> Rate {
        self.rate
    }

    /// Whether [`Encoder::new`] takes this configuration.
    ///
    /// # Errors
    ///
    /// As [`Encoder::new`].
    ///
    /// [`Encoder::new`]: super::Encoder::new
    pub fn check(&self) -> Result<(), ConfigError> {
        self.noise_offset_db().map(|_| ())
    }

    /// The noise offset in dB that the rate asks for (see
    /// [`Rate::Variable`]), once the configuration is checked.
    pub(crate) fn noise_offset_db(&self) -> Result<f64, ConfigError> {
        if self.sample_rate == 0 {
            return Err(ConfigError::UnsupportedSampleRate {
                sample_rate: self.sample_rate,
            });
        }
        if !(1..=MAX_CHANNELS).contains(&self.channels) {
            return Err(ConfigError::UnsupportedChannels {
                channels: self.channels,
            });
        }
        match self.rate {
            Rate::Variable { noise_offset_db } if noise_offset_db.is_finite() => {
                Ok(noise_offset_db)
            }
            Rate::Variable { noise_offset_db } => Err(ConfigError::NoiseOffset { noise_offset_db }),
            Rate::Quality { level } if level <= crate::quality::MAX_LEVEL => {
                Ok(calibration::noise_offset_db(level))
            }
            Rate::Quality { level } => Err(ConfigError::QualityLevel { level }),
        }
    }
}

/// Why a [`Config`] cannot be encoded.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ConfigError {
    /// The sample rate is 0, which clause 4.2.2 does not allow.
    UnsupportedSampleRate { sample_rate: u32 },
    /// The channels are none, or more than [`MAX_CHANNELS`].
    UnsupportedChannels { channels: usize },
    /// The noise offset of [`Rate::Variable`] is not a finite number.
    NoiseOffset { noise_offset_db: f64 },
    /// The level of [`Rate::Quality`] is above
    /// [`MAX_LEVEL`](crate::quality::MAX_LEVEL).
    QualityLevel { level: u8 },
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
            Self::NoiseOffset { noise_offset_db } => {
                write!(
                    f,
                    "the noise offset is not a finite number: {noise_offset_db}"
                )
            }
            Self::QualityLevel { level } => write!(
                f,
                "the level of quality is above {}: {level}",
                crate::quality::MAX_LEVEL
            ),
        }
    }
}

impl std::error::Error for ConfigError {}
