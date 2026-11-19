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
    /// An `lfe_channel_element()`, the low frequency enhancement channel
    /// (clause 8.4).
    Lfe,
}

impl ElementKind {
    /// Its `id_syn_ele` (Table 36).
    pub fn id_syn_ele(self) -> u32 {
        match self {
            Self::Single => 0x0,
            Self::Pair => 0x1,
            Self::Lfe => 0x3,
        }
    }

    /// The channels it carries.
    pub fn channels(self) -> usize {
        match self {
            Self::Single | Self::Lfe => 1,
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
    /// 3: the center front speaker, then the left and right front
    /// speakers.
    Three,
    /// 4: the center front speaker, the left and right front speakers, and
    /// the rear surround speaker.
    Four,
    /// 5: the center front speaker, the left and right front speakers, and
    /// the left and right surround rear speakers.
    Five,
    /// 6, 5+1: those of [`Self::Five`], then the front low frequency
    /// effects speaker.
    FiveOne,
    /// 7, 7+1: the center front speaker, the left and right center front
    /// speakers, the left and right outside front speakers, the left and
    /// right surround rear speakers, and the front low frequency effects
    /// speaker.
    SevenOne,
}

impl ChannelConfiguration {
    /// The configuration of `channels` channels, if one has as many: each
    /// count from 1 to 6 has one, and 8 has 7+1. 7 has none, as Table 42
    /// lists no configuration of 7 channels.
    pub fn from_channels(channels: usize) -> Option<Self> {
        match channels {
            1 => Some(Self::Mono),
            2 => Some(Self::Stereo),
            3 => Some(Self::Three),
            4 => Some(Self::Four),
            5 => Some(Self::Five),
            6 => Some(Self::FiveOne),
            8 => Some(Self::SevenOne),
            _ => None,
        }
    }

    /// The `channel_configuration` of ADTS (Table 8) and the
    /// `channelConfiguration` of the `AudioSpecificConfig()`.
    pub fn value(self) -> u32 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Three => 3,
            Self::Four => 4,
            Self::Five => 5,
            Self::FiveOne => 6,
            Self::SevenOne => 7,
        }
    }

    /// The audio channel elements of each `raw_data_block()`, "listed in
    /// order received" (Table 42).
    pub fn elements(self) -> &'static [ElementKind] {
        use ElementKind::{Lfe, Pair, Single};
        match self {
            Self::Mono => &[Single],
            Self::Stereo => &[Pair],
            Self::Three => &[Single, Pair],
            Self::Four => &[Single, Pair, Single],
            Self::Five => &[Single, Pair, Pair],
            Self::FiveOne => &[Single, Pair, Pair, Lfe],
            Self::SevenOne => &[Single, Pair, Pair, Pair, Lfe],
        }
    }

    /// The channels of all its elements, the low frequency enhancement
    /// channel among them.
    pub fn channels(self) -> usize {
        self.elements().iter().map(|e| e.channels()).sum()
    }

    /// The NCC, the number of channels the decoder's input buffer is sized
    /// by (clause 8.2.3.1): "once the number of SCEs ... plus twice the
    /// number of CPEs" (clause 3.58). A low frequency enhancement channel
    /// is not counted: "All bits for LFE's ... must be supplied from the
    /// total buffer requirements based on the independent CCE's, SCE's,
    /// and CPE's" (clause 8.2.3.1).
    pub fn ncc(self) -> usize {
        (self.elements().iter())
            .filter(|&&e| e != ElementKind::Lfe)
            .map(|e| e.channels())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_configuration_has_its_channels() {
        for (channels, ncc) in [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 5), (8, 7)] {
            let configuration = ChannelConfiguration::from_channels(channels).unwrap();
            assert_eq!(configuration.channels(), channels);
            assert_eq!(configuration.ncc(), ncc);
        }
        for channels in [0, 7, 9] {
            assert_eq!(ChannelConfiguration::from_channels(channels), None);
        }
    }
}
