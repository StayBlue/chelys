// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The channel configurations coded, with the elements that carry their
//! channels (Table 42; Table 1.23 of ISO/IEC 14496-3 gives the same for
//! these).

/// An audio channel element, as a channel configuration lists it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementKind {
    /// A `single_channel_element()`.
    Single,
    /// A `channel_pair_element()`.
    Pair,
}

impl ElementKind {
    /// Its `id_syn_ele` (Table 36).
    pub fn id_syn_ele(self) -> u32 {
        match self {
            Self::Single => 0x0,
            Self::Pair => 0x1,
        }
    }

    /// The channels it carries.
    pub fn channels(self) -> usize {
        match self {
            Self::Single => 1,
            Self::Pair => 2,
        }
    }
}

/// A `channel_configuration` of Table 42.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelConfiguration {
    /// 1: the center front speaker.
    Mono,
    /// 2: the left and right front speakers.
    Stereo,
}

impl ChannelConfiguration {
    /// The configuration of `channels` channels, 1 or 2.
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

    /// The audio channel elements of each `raw_data_block()`, "listed in
    /// order received" (Table 42).
    pub fn elements(self) -> &'static [ElementKind] {
        use ElementKind::{Pair, Single};
        match self {
            Self::Mono => &[Single],
            Self::Stereo => &[Pair],
        }
    }

    /// The channels of all its elements.
    pub fn channels(self) -> usize {
        self.elements().iter().map(|e| e.channels()).sum()
    }

    /// The NCC, the number of channels the decoder's input buffer is sized
    /// by (clause 8.2.3.1): "once the number of SCEs ... plus twice the
    /// number of CPEs" (clause 3.58).
    pub fn ncc(self) -> usize {
        self.channels()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_configuration_has_its_channels() {
        for (channels, ncc) in [(1, 1), (2, 2)] {
            let configuration = ChannelConfiguration::from_channels(channels).unwrap();
            assert_eq!(configuration.channels(), channels);
            assert_eq!(configuration.ncc(), ncc);
        }
        for channels in [0, 3] {
            assert_eq!(ChannelConfiguration::from_channels(channels), None);
        }
    }
}
