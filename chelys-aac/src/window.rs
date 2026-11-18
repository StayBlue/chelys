// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! A channel's windows in a frame, as its `ics_info()` (Table 15) sends
//! them: the `window_sequence` (Table 44), the `window_shape`, and, for an
//! EIGHT_SHORT_SEQUENCE, the grouping of its short windows; and what they
//! set: the transform and the window groups whose scalefactor bands the
//! frame's lines are sent in.
//!
//! The encoder codes every frame as an ONLY_LONG_SEQUENCE with the sine
//! window for now. Block switching adds the other sequences, the short ones
//! with their grouping, and the Kaiser-Bessel derived window adds its shape.

use chelys_core::bitstream::BitWriter;

use crate::sampling_frequency::SamplingFrequency;
use crate::scalefactor_bands::GroupBands;

/// The SHORT_WINDOWs of an EIGHT_SHORT_SEQUENCE (Table 44), and so the most
/// window groups of a channel, each window a group of its own.
pub const SHORT_WINDOWS: usize = 8;

/// `window_sequence` (Table 44).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowSequence {
    /// One LONG_WINDOW.
    OnlyLong,
}

impl WindowSequence {
    /// The value sent as `window_sequence`.
    pub fn number(self) -> u32 {
        match self {
            Self::OnlyLong => 0,
        }
    }
}

/// `window_shape` (clause 15.3.2): the shape of the right half of a
/// frame's window, the left half taking that of the frame before.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowShape {
    /// The sine window, `window_shape` 0.
    Sine,
}

impl WindowShape {
    /// The value sent as `window_shape`.
    pub fn number(self) -> u32 {
        match self {
            Self::Sine => 0,
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
    /// An ONLY_LONG_SEQUENCE with the sine window.
    ///
    /// Spec gap: the encoder chooses the window shape frame by frame
    /// (clause C.3.1.1), and Annex C gives no rule for the choice. The
    /// sine window is taken for every frame, so that the left half of each
    /// window, which takes the shape of the frame before (clause 15.3.2),
    /// is the sine window too.
    pub const LONG_SINE: Self = Self {
        sequence: WindowSequence::OnlyLong,
        shape: WindowShape::Sine,
    };

    /// The scalefactor bands of each window group, in the order the groups
    /// are sent: for a long window, one group of its one window.
    pub fn groups(self, sampling_frequency: SamplingFrequency) -> impl Iterator<Item = GroupBands> {
        match self.sequence {
            WindowSequence::OnlyLong => std::iter::once(GroupBands::long(sampling_frequency)),
        }
    }

    /// The bits [`write_ics_info`](Self::write_ics_info) writes.
    pub fn ics_info_bits(self) -> u32 {
        match self.sequence {
            // ics_reserved_bit, window_sequence, window_shape, max_sfb,
            // predictor_data_present.
            WindowSequence::OnlyLong => 1 + 2 + 1 + 6 + 1,
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
            WindowSequence::OnlyLong => {
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

    #[test]
    fn ics_info_of_a_long_window() {
        let mut w = BitWriter::new();
        Windows::LONG_SINE.write_ics_info(49, &mut w);
        let len = w.bit_len();
        assert_eq!(len as u32, Windows::LONG_SINE.ics_info_bits());
        // ics_reserved_bit, window_sequence, window_shape, max_sfb,
        // predictor_data_present.
        assert_eq!(
            &binary_digits(&w.into_bytes())[..len],
            ["0", "00", "0", "110001", "0"].concat()
        );
    }

    #[test]
    fn a_long_window_is_one_group_of_the_long_bands() {
        let fs = SamplingFrequency::Hz44100;
        let groups: Vec<_> = Windows::LONG_SINE.groups(fs).collect();
        assert_eq!(groups, [GroupBands::long(fs)]);
    }
}
