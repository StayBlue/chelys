// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Scalefactor bands (clause 8.3.4): the first spectral line of each band
//! of a window, `swb_offset_long_window` and `swb_offset_short_window`,
//! by sampling frequency (Tables 45 to 57).
//!
//! Each table is transcribed as printed, with the offset that closes its
//! last band, the window's line count, at the end; its length is thus
//! `num_swb + 1`.

use crate::sampling_frequency::SamplingFrequency;

/// The two transform lengths (clause 8.3.3): the 1024 lines of a
/// LONG_WINDOW, LONG_START_WINDOW or LONG_STOP_WINDOW, and the 128 of each
/// SHORT_WINDOW of an EIGHT_SHORT_SEQUENCE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transform {
    Long,
    Short,
}

impl Transform {
    /// The spectral lines of one window.
    pub fn lines(self) -> usize {
        match self {
            Self::Long => 1024,
            Self::Short => 128,
        }
    }
}

/// The most scalefactor bands of any window: the 51 of a long window at
/// 32 kHz (Table 47).
pub const MAX_BANDS: usize = 51;

/// `swb_offset_long_window` or `swb_offset_short_window` at
/// `sampling_frequency`: the first line of each scalefactor band of a
/// window, then the window's line count.
pub fn swb_offsets(sampling_frequency: SamplingFrequency, transform: Transform) -> &'static [u16] {
    use SamplingFrequency::*;
    match (transform, sampling_frequency) {
        (Transform::Long, Hz96000 | Hz88200) => &LONG_88200_96000,
        (Transform::Long, Hz64000) => &LONG_64000,
        (Transform::Long, Hz48000 | Hz44100) => &LONG_44100_48000,
        (Transform::Long, Hz32000) => &LONG_32000,
        (Transform::Long, Hz24000 | Hz22050) => &LONG_22050_24000,
        (Transform::Long, Hz16000 | Hz12000 | Hz11025) => &LONG_11025_TO_16000,
        (Transform::Long, Hz8000) => &LONG_8000,
        (Transform::Short, Hz96000 | Hz88200) => &SHORT_88200_96000,
        (Transform::Short, Hz64000) => &SHORT_64000,
        (Transform::Short, Hz48000 | Hz44100 | Hz32000) => &SHORT_32000_TO_48000,
        (Transform::Short, Hz24000 | Hz22050) => &SHORT_22050_24000,
        (Transform::Short, Hz16000 | Hz12000 | Hz11025) => &SHORT_11025_TO_16000,
        (Transform::Short, Hz8000) => &SHORT_8000,
    }
}

/// `num_swb_long_window` or `num_swb_short_window`: the scalefactor bands
/// of a window, and so the most that `max_sfb` can count.
///
/// Spec reading: the standard bounds `max_sfb` only by the width of its
/// field, 6 bits for a long window and 4 for short ones (Table 15), which
/// would let it pass `num_swb`. But a decoder reads `swb_offset[max_sfb]`
/// (clause 8.3.4), which the tables give only up to `num_swb`, so
/// `max_sfb` is taken to be at most `num_swb`.
pub fn num_swb(sampling_frequency: SamplingFrequency, transform: Transform) -> usize {
    swb_offsets(sampling_frequency, transform).len() - 1
}

/// The scalefactor bands of a window group (clause 8.3.4): of a long
/// window, or of a group of 1 to 8 short windows of an
/// EIGHT_SHORT_SEQUENCE, each of whose bands holds that band of every
/// window of the group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupBands {
    transform: Transform,
    swb_offsets: &'static [u16],
    windows: u16,
}

impl GroupBands {
    /// The bands of a long window at `sampling_frequency`.
    pub fn long(sampling_frequency: SamplingFrequency) -> Self {
        Self {
            transform: Transform::Long,
            swb_offsets: swb_offsets(sampling_frequency, Transform::Long),
            windows: 1,
        }
    }

    /// The bands of a group of `windows` short windows at
    /// `sampling_frequency`.
    ///
    /// # Panics
    ///
    /// If `windows` is not 1 to 8: a caller's error.
    pub fn short(sampling_frequency: SamplingFrequency, windows: usize) -> Self {
        assert!(
            (1..=8).contains(&windows),
            "a group has 1 to 8 windows, not {windows}"
        );
        Self {
            transform: Transform::Short,
            swb_offsets: swb_offsets(sampling_frequency, Transform::Short),
            windows: windows as u16,
        }
    }

    pub fn transform(self) -> Transform {
        self.transform
    }

    /// The windows of the group: 1 for a long window.
    pub fn windows(self) -> usize {
        usize::from(self.windows)
    }

    /// `num_swb`: the bands of each window, and the most that `max_sfb`
    /// can count (see [`num_swb`]).
    pub fn num_swb(self) -> usize {
        self.swb_offsets.len() - 1
    }

    /// `sect_sfb_offset[g][sfb]`: the first of the group's lines, in the
    /// order they are sent (clause 8.3.5), in band `sfb`, or for `num_swb`
    /// the end of the last.
    pub fn offset(self, sfb: usize) -> usize {
        usize::from(self.swb_offsets[sfb]) * usize::from(self.windows)
    }

    /// The group's lines: 1024, or 128 for each short window.
    pub fn lines(self) -> usize {
        self.offset(self.num_swb())
    }
}

/// Table 45: num_swb_long_window 49, at 44.1 and 48 kHz.
const LONG_44100_48000: [u16; 50] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 1024,
];

/// Table 46: num_swb_short_window 14, at 32, 44.1 and 48 kHz.
const SHORT_32000_TO_48000: [u16; 15] = [0, 4, 8, 12, 16, 20, 28, 36, 44, 56, 68, 80, 96, 112, 128];

/// Table 47: num_swb_long_window 51, at 32 kHz.
const LONG_32000: [u16; 52] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 960, 992, 1024,
];

/// Table 48: num_swb_long_window 40, at 8 kHz.
const LONG_8000: [u16; 41] = [
    0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 172, 188, 204, 220, 236, 252, 268,
    288, 308, 328, 348, 372, 396, 420, 448, 476, 508, 544, 580, 620, 664, 712, 764, 820, 880, 944,
    1024,
];

/// Table 49: num_swb_short_window 15, at 8 kHz.
const SHORT_8000: [u16; 16] = [
    0, 4, 8, 12, 16, 20, 24, 28, 36, 44, 52, 60, 72, 88, 108, 128,
];

/// Table 50: num_swb_long_window 43, at 11.025, 12 and 16 kHz.
const LONG_11025_TO_16000: [u16; 44] = [
    0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 100, 112, 124, 136, 148, 160, 172, 184, 196, 212,
    228, 244, 260, 280, 300, 320, 344, 368, 396, 424, 456, 492, 532, 572, 616, 664, 716, 772, 832,
    896, 960, 1024,
];

/// Table 51: num_swb_short_window 15, at 11.025, 12 and 16 kHz.
const SHORT_11025_TO_16000: [u16; 16] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 60, 72, 88, 108, 128,
];

/// Table 52: num_swb_long_window 47, at 22.05 and 24 kHz.
const LONG_22050_24000: [u16; 48] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124, 136,
    148, 160, 172, 188, 204, 220, 240, 260, 284, 308, 336, 364, 396, 432, 468, 508, 552, 600, 652,
    704, 768, 832, 896, 960, 1024,
];

/// Table 53: num_swb_short_window 15, at 22.05 and 24 kHz.
const SHORT_22050_24000: [u16; 16] = [
    0, 4, 8, 12, 16, 20, 24, 28, 36, 44, 52, 64, 76, 92, 108, 128,
];

/// Table 54: num_swb_long_window 47, at 64 kHz.
const LONG_64000: [u16; 48] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 100, 112, 124, 140,
    156, 172, 192, 216, 240, 268, 304, 344, 384, 424, 464, 504, 544, 584, 624, 664, 704, 744, 784,
    824, 864, 904, 944, 984, 1024,
];

/// Table 55: num_swb_short_window 12, at 64 kHz.
const SHORT_64000: [u16; 13] = [0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 92, 128];

/// Table 56: num_swb_long_window 41, at 88.2 and 96 kHz.
const LONG_88200_96000: [u16; 42] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 96, 108, 120, 132,
    144, 156, 172, 188, 212, 240, 276, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024,
];

/// Table 57: num_swb_short_window 12, at 88.2 and 96 kHz.
const SHORT_88200_96000: [u16; 13] = [0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 92, 128];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bands_cover_each_window_in_steps_of_four_lines() {
        for fs in SamplingFrequency::ALL {
            for transform in [Transform::Long, Transform::Short] {
                let offsets = swb_offsets(fs, transform);
                assert_eq!(offsets[0], 0);
                assert_eq!(usize::from(offsets[offsets.len() - 1]), transform.lines());
                // Clause 9.1: every band holds a multiple of 4 lines.
                assert!(offsets.windows(2).all(|w| w[0] < w[1] && w[0] % 4 == 0));
                assert!(num_swb(fs, transform) <= MAX_BANDS);
            }
        }
    }

    #[test]
    fn band_counts_follow_the_tables() {
        use SamplingFrequency::*;
        let counts = [
            (Hz96000, 41, 12),
            (Hz88200, 41, 12),
            (Hz64000, 47, 12),
            (Hz48000, 49, 14),
            (Hz44100, 49, 14),
            (Hz32000, 51, 14),
            (Hz24000, 47, 15),
            (Hz22050, 47, 15),
            (Hz16000, 43, 15),
            (Hz12000, 43, 15),
            (Hz11025, 43, 15),
            (Hz8000, 40, 15),
        ];
        for (fs, long, short) in counts {
            assert_eq!(num_swb(fs, Transform::Long), long, "{fs:?}");
            assert_eq!(num_swb(fs, Transform::Short), short, "{fs:?}");
        }
        // Table 43 gives 49 and 14 at 48 kHz too.
        assert_eq!(MAX_BANDS, num_swb(Hz32000, Transform::Long));
    }

    #[test]
    fn spot_entries_read_as_printed() {
        use SamplingFrequency::*;
        // Table 45, swb 24; Table 48, swb 39; Table 54, swb 29; Table 57,
        // swb 11.
        assert_eq!(swb_offsets(Hz44100, Transform::Long)[24], 196);
        assert_eq!(swb_offsets(Hz8000, Transform::Long)[39], 944);
        assert_eq!(swb_offsets(Hz64000, Transform::Long)[29], 304);
        assert_eq!(swb_offsets(Hz96000, Transform::Short)[11], 92);
    }

    #[test]
    fn groups_of_short_windows_hold_each_band_of_every_window() {
        use SamplingFrequency::*;
        let long = GroupBands::long(Hz96000);
        assert_eq!(
            (long.num_swb(), long.lines(), long.offset(40)),
            (41, 1024, 960)
        );
        // Table 46: band 13 starts at line 112.
        let group = GroupBands::short(Hz44100, 3);
        assert_eq!((group.num_swb(), group.windows()), (14, 3));
        assert_eq!(
            (long.transform(), group.transform()),
            (Transform::Long, Transform::Short)
        );
        assert_eq!((group.offset(13), group.lines()), (3 * 112, 3 * 128));
    }

    #[test]
    #[should_panic(expected = "1 to 8 windows")]
    fn a_group_holds_at_most_eight_windows() {
        GroupBands::short(SamplingFrequency::Hz48000, 9);
    }
}
