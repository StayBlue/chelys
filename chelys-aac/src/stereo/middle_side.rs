// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! M/S stereo (clause 12.1): the bands of a pair that share their window
//! coded as middle and side rather than left and right.

use std::cmp::Ordering;

use chelys_core::bitstream::BitWriter;

use crate::channel::FRAME_LINES;
use crate::loops::inner;
use crate::sampling_frequency::SamplingFrequency;
use crate::scalefactor_bands::MAX_BANDS;
use crate::stereo::BandCoding;
use crate::stereo::intensity::Intensity;
use crate::window::{SHORT_WINDOWS, Windows};

/// The middle and side channels of a pair's lines, `M = (L + R) / 2` and
/// `S = (L - R) / 2`, which clause 12.1.3 decodes as `L = M + S` and `R =
/// M - S`, into `mid` and `side`.
pub fn matrix(
    left: &[f64; FRAME_LINES],
    right: &[f64; FRAME_LINES],
    mid: &mut [f64; FRAME_LINES],
    side: &mut [f64; FRAME_LINES],
) {
    for (((m, s), &l), &r) in mid.iter_mut().zip(side.iter_mut()).zip(left).zip(right) {
        *m = (l + r) / 2.0;
        *s = (l - r) / 2.0;
    }
}

/// Chooses the bands of a pair that shares `windows` to code as middle and
/// side (clause C.6.1, steps 3 to 5): from the lines `[left, right, mid,
/// side]` and what each band of each group of each may take as noise,
/// `allowed`, the bits each band takes coded as left and right and as
/// middle and side are estimated, each channel's band quantized with the
/// coarsest step within its allowance (see [`inner::band_bits`]), and a
/// band is coded as middle and side where that takes fewer.
///
/// Spec reading: steps 3 and 4 quantize each channel's lines "with the
/// appropriate calculated and quantized threshold", and count "the number
/// of bits actually required" to code them. They are estimated band by
/// band, before the loops quantize the pair: a band's bits are taken as
/// those of its lines in its best codebook alone. What else a band costs
/// is left out: its share of the sections, which may differ where the two
/// codings take different codebooks; its scalefactor, which a band
/// quantized to zero in a section of `ZERO_HCB` does not send; and the
/// `ms_used` flags. The estimate chooses well enough that a better one
/// has little to gain: graded with PEAQ (GstPEAQ 0.6.1, black box), on 8
/// tracks of pop music at 44.1 kHz in stereo, the bands so chosen gave
/// means of -3.583, -1.087 and -0.282 at 64, 128 and 192 kbit/s, every
/// band that may be coded as middle and side so coded -3.595, -1.115 and
/// -0.290, and none -3.598, -1.395 and -0.318; on 3 tracks of rock music
/// and their 3 windows of 3 s with the most frames coded short, -3.579,
/// -0.803 and -0.142, -3.591, -0.850 and -0.162, and -3.577, -1.128 and
/// -0.153.
///
/// Spec gap: step 5 does not say which coding a band takes where both
/// take as many bits, as where all four channels' bands quantize to zero.
/// Such bands take the coding of all the others where they agree, so that
/// no flag need be sent for each band (see [`MsMask::present`]), and else
/// left and right.
///
/// Spec gap: Annex C does not say how M/S stereo and temporal noise
/// shaping combine, whose filters the decoder applies to each channel after
/// it decodes the middle and side (clause 4, figure 2). A band is coded as
/// middle and side only where the two channels' lines are weighed alike
/// (see [`Lines::weighted`](inner::Lines::weighted)), as where no filter
/// spans it or the same filter spans it in both: where only one channel's
/// lines are shaped, the middle and side mix shaped and unshaped lines,
/// whose noise one channel's filter raises and the other's does not. So
/// coded, a sweep of 30 Hz to 18 kHz beside another, at 44.1 kHz in
/// stereo at 96 kbit/s, decoded at an SNR of 53.4 dB, against 56.0
/// unshaped; coded left and right there, at 55.3 dB.
///
/// The bands `intensity` codes in intensity stereo are coded neither way
/// (clause 12.1.3), and their flags are 0, which leaves each band's phase
/// as its codebook gives it (clause 12.2.3).
///
/// Spec reading: with `ms_mask_present` 2, `invert_intensity` is +1 for
/// every band (clause 12.2.3), and the bands in intensity stereo are not
/// decoded as middle and side (clause 12.1.3), so that a frame whose other
/// bands are all coded as middle and side could send 2 and no flags. Where a frame has
/// bands in intensity stereo, it sends 1 and their flags of 0 instead,
/// which no reading of `ms_used` can take as inverting them. Settled by
/// decoding with ffmpeg and FAAD2 (black box), which disagree on 2: sent
/// with `ms_mask_present` 2, ffmpeg decoded the right channel's tones in
/// intensity stereo with their phase inverted, against clause 12.2.3, and
/// FAAD2 decoded them as the clause says, to the very samples of the same
/// frames sent with 1 and flags of 0. Sent with 1 and flags of 0, both
/// decode them in the phase their codebooks give, at every sampling
/// frequency from 22.05 kHz (FAAD2's phase checked above 24 kHz, below
/// which it gives samples at twice the sampling frequency). So 1 is sent,
/// for ffmpeg.
/// The conformance tests are `intensity_stereo_keeps_each_tones_panning`
/// and `intensity_bands_under_ms_mask_present_2_keep_their_phase_in_faad_only`.
pub fn choose(
    windows: Windows,
    sampling_frequency: SamplingFrequency,
    lines: [&[f64; FRAME_LINES]; 4],
    weights: [&[f64; FRAME_LINES]; 4],
    allowed: [&[[f64; MAX_BANDS]]; 4],
    intensity: &Intensity,
) -> MsMask {
    let mut mask = MsMask::NONE;
    let mut ties = MsMask::NONE;
    let (mut any_left_right, mut any_mid_side) = (false, false);
    let mut scratch = inner::BandScratch::default();
    for band in super::bands(windows, sampling_frequency, MsMask::NONE, intensity) {
        if band.coding == BandCoding::Intensity {
            continue;
        }
        let (g, sfb, range) = (band.group, band.sfb, band.lines);
        if weights[0][range.clone()] != weights[1][range.clone()] {
            any_left_right = true;
            continue;
        }
        let [l, r, m, s] = std::array::from_fn(|c| {
            let (lines, weights) = (&lines[c][range.clone()], &weights[c][range.clone()]);
            inner::band_bits(lines, weights, allowed[c][g][sfb], &mut scratch)
        });
        let (left_right, mid_side) = (l + r, m + s);
        match mid_side.cmp(&left_right) {
            Ordering::Equal => ties.set(g, sfb, true),
            Ordering::Less => {
                mask.set(g, sfb, true);
                any_mid_side = true;
            }
            Ordering::Greater => any_left_right = true,
        }
    }
    if any_mid_side && !any_left_right {
        for g in 0..SHORT_WINDOWS {
            mask.used[g] |= ties.used[g];
        }
    }
    mask
}

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
