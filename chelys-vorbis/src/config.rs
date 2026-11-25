// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! What an [`Encoder`](super::Encoder) is asked for, and its checks.

use std::fmt;

/// The most channels the encoder takes: those of the largest layout that
/// clause 4.3.9 orders, 7.1.
pub const MAX_CHANNELS: usize = 8;

/// What to encode: the sample rate and channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    sample_rate: u32,
    channels: usize,
}

impl Config {
    /// A configuration of `channels` channels, from 1 to [`MAX_CHANNELS`],
    /// at `sample_rate` Hz, which is any but 0. [`Encoder::new`] checks it.
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
    pub fn new(sample_rate: u32, channels: usize) -> Self {
        Self {
            sample_rate,
            channels,
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Whether [`Encoder::new`] takes this configuration.
    ///
    /// # Errors
    ///
    /// As [`Encoder::new`].
    ///
    /// [`Encoder::new`]: super::Encoder::new
    pub fn check(&self) -> Result<(), ConfigError> {
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
        Ok(())
    }
}

/// Why a [`Config`] cannot be encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConfigError {
    /// The sample rate is 0, which clause 4.2.2 does not allow.
    UnsupportedSampleRate { sample_rate: u32 },
    /// The channels are none, or more than [`MAX_CHANNELS`].
    UnsupportedChannels { channels: usize },
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
        }
    }
}

impl std::error::Error for ConfigError {}
