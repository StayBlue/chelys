// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! A channel's windows in a frame, as its `ics_info()` (Table 15) sends
//! them: the `window_sequence` (Table 44), the `window_shape`, and, for an
//! EIGHT_SHORT_SEQUENCE, the grouping of its short windows; and what they
//! set: the transform and the window groups whose scalefactor bands the
//! frame's lines are sent in.

use chelys_core::bitstream::BitWriter;

use crate::sampling_frequency::SamplingFrequency;
use crate::scalefactor_bands::{GroupBands, Transform};

/// The SHORT_WINDOWs of an EIGHT_SHORT_SEQUENCE (Table 44), and so the most
/// window groups of a channel, each window a group of its own.
pub const SHORT_WINDOWS: usize = 8;

/// `scale_factor_grouping` (clause 8.3.4): which of the eight short windows
/// of an EIGHT_SHORT_SEQUENCE share a group, and so one set of
/// scalefactors, with the window before them.
///
/// Its seven bits are read from the most significant: bit `6 - i` is set
/// where window `i + 1` belongs to the group of window `i`, and clear where
/// it starts a group of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grouping(u8);

impl Grouping {
    /// Each short window a group of its own.
    pub const NONE: Self = Self(0);

    /// The grouping whose groups start at the windows `starts` gives: the
    /// first window always starts one, whatever `starts[0]` says.
    pub fn from_starts(starts: [bool; SHORT_WINDOWS]) -> Self {
        let bits = (1..SHORT_WINDOWS)
            .filter(|&w| !starts[w])
            .fold(0, |bits, w| bits | (1 << (SHORT_WINDOWS - 1 - w)));
        Self(bits)
    }

    /// The value sent as `scale_factor_grouping`, 7 bits.
    pub fn bits(self) -> u8 {
        self.0
    }

    /// Whether window `w` starts a group.
    pub fn starts_group(self, w: usize) -> bool {
        w == 0 || self.0 & (1 << (SHORT_WINDOWS - 1 - w)) == 0
    }

    /// `num_window_groups` (clause 8.3.4).
    pub fn num_groups(self) -> usize {
        (0..SHORT_WINDOWS).filter(|&w| self.starts_group(w)).count()
    }

    /// `window_group_length[g]` of each group in turn (clause 8.3.4).
    pub fn lengths(self) -> impl Iterator<Item = usize> {
        let mut w = 0;
        std::iter::from_fn(move || {
            if w == SHORT_WINDOWS {
                return None;
            }
            let start = w;
            w += 1;
            while w < SHORT_WINDOWS && !self.starts_group(w) {
                w += 1;
            }
            Some(w - start)
        })
    }
}

/// `window_sequence` (Table 44).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowSequence {
    /// One LONG_WINDOW.
    OnlyLong,
    /// One LONG_START_WINDOW, from long windows to short ones.
    LongStart,
    /// Eight SHORT_WINDOWs, grouped as `grouping` says.
    EightShort { grouping: Grouping },
    /// One LONG_STOP_WINDOW, from short windows to long ones.
    LongStop,
}

impl WindowSequence {
    /// An EIGHT_SHORT_SEQUENCE with each window a group of its own.
    pub const EIGHT_SHORT: Self = Self::EightShort {
        grouping: Grouping::NONE,
    };

    /// The value sent as `window_sequence`.
    pub fn number(self) -> u32 {
        match self {
            Self::OnlyLong => 0,
            Self::LongStart => 1,
            Self::EightShort { .. } => 2,
            Self::LongStop => 3,
        }
    }

    /// The transform of each of the sequence's windows (clause 15.3.1).
    pub fn transform(self) -> Transform {
        match self {
            Self::EightShort { .. } => Transform::Short,
            _ => Transform::Long,
        }
    }

    /// Whether the sequence may follow `previous` as one of "the meaningful
    /// block transitions" of clause 15.3.1, which the encoder keeps to: a
    /// window that starts long only after one that ends long, and one that
    /// starts short only after one that ends short. The grouping does not
    /// matter.
    ///
    /// Clause 15.3.1 permits every other transition too, as one that "will
    /// still result in a reasonably smooth transition", but the halves
    /// that overlap there differ, and their aliasing does not cancel.
    pub fn follows(self, previous: Self) -> bool {
        let starts_short = matches!(self, Self::EightShort { .. } | Self::LongStop);
        let ended_short = matches!(previous, Self::LongStart | Self::EightShort { .. });
        starts_short == ended_short
    }
}

/// `window_shape` (clause 15.3.2): the shape of the right half of a
/// frame's window, and of every half of its short windows but the first's
/// left, which take that of the frame before.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowShape {
    /// The sine window, `window_shape` 0.
    Sine,
    /// The Kaiser-Bessel derived window, `window_shape` 1.
    Kbd,
}

impl WindowShape {
    /// The value sent as `window_shape`.
    pub fn number(self) -> u32 {
        match self {
            Self::Sine => 0,
            Self::Kbd => 1,
        }
    }
}

/// A channel's windows in a frame: the fields of its `ics_info()` that do
/// not depend on its lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Windows {
    pub sequence: WindowSequence,
    pub shape: WindowShape,
}

impl Windows {
    /// An ONLY_LONG_SEQUENCE with the sine window, as a silent frame and
    /// the tests take.
    pub const LONG_SINE: Self = Self {
        sequence: WindowSequence::OnlyLong,
        shape: WindowShape::Sine,
    };

    /// The scalefactor bands of each window group, in the order the groups
    /// are sent: for a long window, one group of its one window, and for
    /// eight short ones, the groups of their grouping.
    pub fn groups(
        self,
        sampling_frequency: SamplingFrequency,
    ) -> std::iter::Take<std::array::IntoIter<GroupBands, SHORT_WINDOWS>> {
        let mut groups = [GroupBands::long(sampling_frequency); SHORT_WINDOWS];
        let count = match self.sequence {
            WindowSequence::EightShort { grouping } => {
                for (group, len) in groups.iter_mut().zip(grouping.lengths()) {
                    *group = GroupBands::short(sampling_frequency, len);
                }
                grouping.num_groups()
            }
            _ => 1,
        };
        groups.into_iter().take(count)
    }

    /// [`groups`](Self::groups), each with its first line among the
    /// frame's lines in the order they are sent.
    pub fn groups_with_start(
        self,
        sampling_frequency: SamplingFrequency,
    ) -> impl Iterator<Item = (usize, GroupBands)> + Clone {
        self.groups(sampling_frequency).scan(0, |start, bands| {
            let first = *start;
            *start += bands.lines();
            Some((first, bands))
        })
    }

    /// The bits [`write_ics_info`](Self::write_ics_info) writes.
    pub fn ics_info_bits(self) -> u32 {
        match self.sequence {
            // ics_reserved_bit, window_sequence, window_shape, max_sfb,
            // scale_factor_grouping.
            WindowSequence::EightShort { .. } => 1 + 2 + 1 + 4 + 7,
            // ics_reserved_bit, window_sequence, window_shape, max_sfb,
            // predictor_data_present.
            _ => 1 + 2 + 1 + 6 + 1,
        }
    }

    /// Writes the `ics_info()` (Table 15) of these windows, with `max_sfb`.
    ///
    /// # Panics
    ///
    /// If `max_sfb` does not fit its field: a caller's error.
    pub fn write_ics_info(self, max_sfb: usize, w: &mut BitWriter) {
        w.write(0, 1); // ics_reserved_bit: "Shall be '0'" (clause 8.3.1.1)
        w.write(self.sequence.number(), 2);
        w.write(self.shape.number(), 1);
        match self.sequence {
            WindowSequence::EightShort { grouping } => {
                assert!(max_sfb < 1 << 4, "max_sfb {max_sfb} exceeds its field");
                w.write(max_sfb as u32, 4);
                w.write(u32::from(grouping.bits()), 7);
            }
            _ => {
                assert!(max_sfb < 1 << 6, "max_sfb {max_sfb} exceeds its field");
                w.write(max_sfb as u32, 6);
                // predictor_data_present: prediction is not permitted in
                // the Low Complexity profile (clause 7.1.2).
                w.write(0, 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use chelys_core::bitstream::binary_digits;

    use super::*;

    fn ics_info(windows: Windows, max_sfb: usize) -> String {
        let mut w = BitWriter::new();
        windows.write_ics_info(max_sfb, &mut w);
        let len = w.bit_len();
        assert_eq!(len as u32, windows.ics_info_bits());
        binary_digits(&w.into_bytes())[..len].to_owned()
    }

    #[test]
    fn ics_info_of_long_windows() {
        // ics_reserved_bit, window_sequence, window_shape, max_sfb,
        // predictor_data_present.
        assert_eq!(
            ics_info(Windows::LONG_SINE, 49),
            ["0", "00", "0", "110001", "0"].concat()
        );
        let windows = Windows {
            sequence: WindowSequence::LongStop,
            shape: WindowShape::Kbd,
        };
        assert_eq!(
            ics_info(windows, 3),
            ["0", "11", "1", "000011", "0"].concat()
        );
    }

    #[test]
    fn ics_info_of_eight_short_windows() {
        // Figure 4: groups of 3, 1, 2 and 2 windows.
        let grouping = Grouping::from_starts([true, false, false, true, true, false, true, false]);
        assert_eq!(grouping.bits(), 0b1100101);
        let windows = Windows {
            sequence: WindowSequence::EightShort { grouping },
            shape: WindowShape::Sine,
        };
        // ics_reserved_bit, window_sequence, window_shape, max_sfb,
        // scale_factor_grouping.
        assert_eq!(
            ics_info(windows, 14),
            ["0", "10", "0", "1110", "1100101"].concat()
        );
    }

    #[test]
    fn groupings_give_their_groups_as_clause_8_3_4_counts_them() {
        for bits in 0..128u8 {
            let grouping = Grouping(bits);
            // The pseudo code of clause 8.3.4.
            let mut lengths = vec![1];
            for i in 0..7 {
                if bits & (1 << (6 - i)) == 0 {
                    lengths.push(1);
                } else {
                    *lengths.last_mut().unwrap() += 1;
                }
            }
            assert_eq!(
                grouping.lengths().collect::<Vec<_>>(),
                lengths,
                "{bits:07b}"
            );
            assert_eq!(grouping.num_groups(), lengths.len());
            let starts = std::array::from_fn(|w| grouping.starts_group(w));
            assert_eq!(Grouping::from_starts(starts), grouping);
        }
        assert_eq!(Grouping::NONE.lengths().collect::<Vec<_>>(), [1; 8]);
    }

    #[test]
    fn a_long_window_is_one_group_of_the_long_bands() {
        let fs = SamplingFrequency::Hz44100;
        let groups: Vec<_> = Windows::LONG_SINE.groups(fs).collect();
        assert_eq!(groups, [GroupBands::long(fs)]);
    }

    #[test]
    fn eight_short_windows_are_their_groups_of_short_bands() {
        let fs = SamplingFrequency::Hz22050;
        let grouping = Grouping::from_starts([true, false, false, false, true, true, false, false]);
        let windows = Windows {
            sequence: WindowSequence::EightShort { grouping },
            shape: WindowShape::Kbd,
        };
        let groups: Vec<_> = windows.groups(fs).collect();
        assert_eq!(
            groups,
            [4, 1, 3].map(|windows| GroupBands::short(fs, windows))
        );
        assert_eq!(groups.iter().map(|g| g.lines()).sum::<usize>(), 1024);
    }

    #[test]
    fn the_meaningful_transitions_are_those_of_clause_15_3_1() {
        use WindowSequence::*;
        let all = [OnlyLong, LongStart, WindowSequence::EIGHT_SHORT, LongStop];
        for from in all {
            let to: Vec<_> = all.into_iter().filter(|to| to.follows(from)).collect();
            let expected = match from {
                OnlyLong | LongStop => vec![OnlyLong, LongStart],
                LongStart | EightShort { .. } => vec![WindowSequence::EIGHT_SHORT, LongStop],
            };
            assert_eq!(to, expected, "from {from:?}");
        }
    }
}
