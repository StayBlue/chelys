// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The channel configurations coded, with the elements that carry their
//! channels (Table 42; Table 1.23 of ISO/IEC 14496-3 gives the same for
//! these).

/// A `channel_configuration` of Table 42.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelConfiguration {
    /// One `single_channel_element()`, the center front speaker.
    Mono,
    /// One `channel_pair_element()`, the left and right front speakers.
    Stereo,
}

impl ChannelConfiguration {
    pub fn from_channels(channels: usize) -> Option<Self> {
        match channels {
            1 => Some(Self::Mono),
            2 => Some(Self::Stereo),
            _ => None,
        }
    }

    /// The `channel_configuration` of ADTS (Table 8) and the
    /// `channelConfiguration` of the `AudioSpecificConfig()`.
    pub fn value(self) -> u32 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
        }
    }

    /// The channels, which are also the NCC, the number of channels the
    /// decoder's input buffer is sized by (clause 8.2.3.1): an SCE counts
    /// one, a CPE two.
    pub fn channels(self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
        }
    }
}
