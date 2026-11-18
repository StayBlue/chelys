// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The calculation of the scalefactor selection information, `scfsi`
//! (clause C.1.5.4.2.3): which scfsi bands of a frame's second granule take
//! over the first granule's scalefactors.

use crate::granule::{BlockType, GRANULE_LINES};
use crate::header::SamplingFrequency;
use crate::scalefactor_bands::{self, LONG_BANDS};
use crate::scalefactors::SCFSI_STARTS;
use crate::side_info::SCFSI_BANDS;

/// The criteria's suggested constants (clause C.1.5.4.2.3).
const EN_TOT_KRIT: i32 = 10;
const EN_DIF_KRIT: i32 = 100;
const EN_SCFSI_BAND_KRIT: i32 = 10;
const XM_SCFSI_BAND_KRIT: i32 = 10;

/// What clause C.1.5.4.2.3 stores of a granule to decide `scfsi`.
///
/// Spec gap: the clause takes `log2` of energies and allowed distortions that
/// may be zero. A zero is kept as `None`: two zeros are taken as equal, and a
/// zero against anything else as too far apart for `scfsi`.
///
/// Spec reading: `int` is taken as the integer part, rounding towards zero.
#[derive(Clone, Debug, PartialEq)]
pub struct GranuleSummary {
    block_type: BlockType,
    /// `en_tot`, `None` if every spectral value is zero.
    en_tot: Option<i32>,
    /// `en(sb)` and `xm(sb)` of each long scalefactor band; not used for
    /// short blocks.
    en: [Option<i32>; LONG_BANDS],
    xm: [Option<i32>; LONG_BANDS],
}

fn int_log2(x: f64) -> Option<i32> {
    (x > 0.0).then(|| x.log2().trunc() as i32)
}

/// `|a - b|`, with zeros (`None`) as [`GranuleSummary`] describes.
fn difference(a: Option<i32>, b: Option<i32>) -> i32 {
    match (a, b) {
        (Some(a), Some(b)) => i32::try_from(a.abs_diff(b)).unwrap_or(i32::MAX),
        (None, None) => 0,
        _ => i32::MAX,
    }
}

impl GranuleSummary {
    /// The summary of a granule of `block_type` with lines `xr` and allowed
    /// distortion `xmin`, both in the order they are sent.
    pub fn new(
        xr: &[f64; GRANULE_LINES],
        xmin: &[f64],
        block_type: BlockType,
        sampling_frequency: SamplingFrequency,
    ) -> Self {
        let energy = |lines: &[f64]| lines.iter().map(|x| x * x).sum::<f64>();
        let mut en = [None; LONG_BANDS];
        let mut xm = [None; LONG_BANDS];
        if !block_type.short_windows() {
            let bands = scalefactor_bands::long_bands(sampling_frequency);
            for (sb, band) in bands.iter().enumerate() {
                en[sb] = int_log2(energy(&xr[band.start..band.end()]));
                xm[sb] = int_log2(xmin[sb]);
            }
        }
        Self {
            block_type,
            en_tot: int_log2(energy(xr)),
            en,
            xm,
        }
    }
}

/// The `scfsi` of each scfsi band for the second granule of a frame, given
/// the summaries of the `first` and `second` granules of one channel.
///
/// Spec reading: the general criterion "the spectral values are not all
/// zero" is applied to both granules.
pub fn select(first: &GranuleSummary, second: &GranuleSummary) -> [bool; SCFSI_BANDS] {
    let general = first.en_tot.is_some()
        && second.en_tot.is_some()
        && !first.block_type.short_windows()
        && !second.block_type.short_windows()
        && difference(first.en_tot, second.en_tot) < EN_TOT_KRIT
        && sum_differences(&first.en, &second.en, 0..LONG_BANDS) < EN_DIF_KRIT;
    if !general {
        return [false; SCFSI_BANDS];
    }
    std::array::from_fn(|band| {
        let bands = SCFSI_STARTS[band]..SCFSI_STARTS[band + 1];
        sum_differences(&first.en, &second.en, bands.clone()) < EN_SCFSI_BAND_KRIT
            && sum_differences(&first.xm, &second.xm, bands) < XM_SCFSI_BAND_KRIT
    })
}

fn sum_differences(
    a: &[Option<i32>; LONG_BANDS],
    b: &[Option<i32>; LONG_BANDS],
    bands: std::ops::Range<usize>,
) -> i32 {
    bands.fold(0, |sum, sb| sum.saturating_add(difference(a[sb], b[sb])))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: SamplingFrequency = SamplingFrequency::Hz44100;

    fn summary(level: f64, block_type: BlockType) -> GranuleSummary {
        let xr = std::array::from_fn(|i| level * if i % 3 == 0 { 1.0 } else { -0.5 });
        GranuleSummary::new(&xr, &[level * level * 1e-3; LONG_BANDS], block_type, FS)
    }

    #[test]
    fn similar_granules_share_every_scfsi_band() {
        let a = summary(0.1, BlockType::Normal);
        assert_eq!(select(&a, &a), [true; SCFSI_BANDS]);
        // 2^0.5 louder moves en by at most one step per band and xm by one.
        let b = summary(0.1 * 2f64.sqrt(), BlockType::Normal);
        assert_eq!(select(&a, &b), [true; SCFSI_BANDS]);
    }

    #[test]
    fn short_blocks_and_silence_disable_scfsi() {
        let long = summary(0.1, BlockType::Normal);
        assert_eq!(select(&long, &summary(0.1, BlockType::Short)), [false; 4]);
        assert_eq!(select(&summary(0.1, BlockType::Short), &long), [false; 4]);
        assert_eq!(select(&long, &summary(0.0, BlockType::Normal)), [false; 4]);
        // Start and stop blocks have long scalefactors.
        assert_eq!(select(&long, &summary(0.1, BlockType::Stop)), [true; 4]);
    }

    /// A granule 2^5 louder differs by 10 in `en_tot`, which is too much.
    #[test]
    fn a_change_in_total_energy_disables_scfsi() {
        let quiet = summary(0.01, BlockType::Normal);
        let loud = summary(0.01 * 32.0, BlockType::Normal);
        assert_eq!(select(&quiet, &loud), [false; 4]);
    }

    /// A change in one scfsi band's energy or allowed distortion keeps
    /// `scfsi` from that band only.
    #[test]
    fn a_change_in_one_band_disables_its_scfsi_band() {
        let xr: [f64; GRANULE_LINES] = std::array::from_fn(|i| 0.1 * ((i % 7) as f64 - 3.0));
        let xmin = [1e-4; LONG_BANDS];
        let first = GranuleSummary::new(&xr, &xmin, BlockType::Normal, FS);
        let band = scalefactor_bands::long_bands(FS)[12];
        let mut louder = xr;
        for x in &mut louder[band.start..band.end()] {
            *x *= 64.0;
        }
        let second = GranuleSummary::new(&louder, &xmin, BlockType::Normal, FS);
        assert_eq!(select(&first, &second), [true, true, false, true]);

        let mut stricter = xmin;
        stricter[18] = 1e-8;
        let second = GranuleSummary::new(&xr, &stricter, BlockType::Normal, FS);
        assert_eq!(select(&first, &second), [true, true, true, false]);
    }

    #[test]
    fn int_rounds_towards_zero() {
        assert_eq!(int_log2(3.0), Some(1));
        assert_eq!(int_log2(0.3), Some(-1));
        assert_eq!(int_log2(0.0), None);
    }
}
