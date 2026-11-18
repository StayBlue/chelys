// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! M/S stereo (clause 12.1): the bands of a pair that share their window
//! coded as middle and side rather than left and right.

use chelys_core::bitstream::BitWriter;

use crate::window::SHORT_WINDOWS;

/// `ms_used[g][sfb]` (clause 12.1.2.1) of a pair's window groups: a bit for
/// each band of each group.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MsMask {
    used: [u64; SHORT_WINDOWS],
}

impl MsMask {
    /// No band coded as middle and side.
    pub const NONE: Self = Self {
        used: [0; SHORT_WINDOWS],
    };

    #[cfg(test)]
    pub fn set(&mut self, g: usize, sfb: usize, used: bool) {
        if used {
            self.used[g] |= 1 << sfb;
        } else {
            self.used[g] &= !(1 << sfb);
        }
    }

    pub fn is_used(&self, g: usize, sfb: usize) -> bool {
        (self.used[g] >> sfb) & 1 == 1
    }

    /// `ms_mask_present` for the bands `0..max_sfb` of `groups` window
    /// groups: 0 where no band is coded as middle and side, 2 where every
    /// band is, and 1, with a flag for each band, where some are.
    pub fn present(&self, groups: usize, max_sfb: usize) -> u32 {
        let bands = if max_sfb == 0 {
            0
        } else {
            u64::MAX >> (64 - max_sfb)
        };
        let used = self.used[..groups].iter().map(|&used| used & bands);
        if used.clone().all(|used| used == 0) {
            0
        } else if used.clone().all(|used| used == bands) {
            2
        } else {
            1
        }
    }

    /// The bits of `ms_mask_present` and the `ms_used` flags that follow
    /// it, as [`write`](Self::write) writes them.
    #[cfg(test)]
    pub fn bits(&self, groups: usize, max_sfb: usize) -> u32 {
        2 + self.flag_bits(groups, max_sfb)
    }

    /// The bits of the `ms_used` flags alone.
    pub fn flag_bits(&self, groups: usize, max_sfb: usize) -> u32 {
        if self.present(groups, max_sfb) == 1 {
            (groups * max_sfb) as u32
        } else {
            0
        }
    }

    /// Writes `ms_mask_present` and, where it is 1, `ms_used[g][sfb]` for
    /// the bands `0..max_sfb` of each of `groups` window groups (Table 14).
    pub fn write(&self, groups: usize, max_sfb: usize, w: &mut BitWriter) {
        let present = self.present(groups, max_sfb);
        w.write(present, 2);
        if present == 1 {
            for g in 0..groups {
                for sfb in 0..max_sfb {
                    w.write_bool(self.is_used(g, sfb));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use chelys_core::bitstream::binary_digits;

    use super::*;

    fn written(mask: &MsMask, groups: usize, max_sfb: usize) -> String {
        let mut w = BitWriter::new();
        mask.write(groups, max_sfb, &mut w);
        let len = w.bit_len();
        assert_eq!(len as u32, mask.bits(groups, max_sfb));
        binary_digits(&w.into_bytes())[..len].to_owned()
    }

    #[test]
    fn the_mask_is_sent_only_where_some_bands_are_middle_and_side() {
        let mut mask = MsMask::NONE;
        assert_eq!(written(&mask, 1, 49), "00");
        for sfb in 0..49 {
            mask.set(0, sfb, true);
        }
        assert_eq!(written(&mask, 1, 49), "10");
        // A band above max_sfb is not sent.
        mask.set(0, 49, false);
        assert_eq!(written(&mask, 1, 49), "10");
        mask.set(0, 3, false);
        let flags: String = (0..49)
            .map(|sfb| if sfb == 3 { '0' } else { '1' })
            .collect();
        assert_eq!(written(&mask, 1, 49), ["01", &flags].concat());
        // Group by group.
        let mut mask = MsMask::NONE;
        mask.set(1, 2, true);
        assert_eq!(written(&mask, 2, 3), ["01", "000", "001"].concat());
        assert_eq!(written(&mask, 1, 3), "00");
        assert_eq!(written(&mask, 2, 0), "00");
    }
}
