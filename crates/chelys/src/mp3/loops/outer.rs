//! The outer iteration loop, or distortion control loop (clause C.1.5.4.3,
//! figure C.9.b), for long blocks without `scfsi`.

use super::inner::{inner_loop, step_size};
use crate::mp3::header::SamplingFrequency;
use crate::mp3::scalefactor_bands::{self, Band, LONG_BANDS};
use crate::mp3::scalefactors::{PRETAB, Scalefactors, part2_length};
use crate::mp3::{GRANULE_LINES, QuantizedGranule};

/// The upper scalefactor bands whose distortion decides preemphasis.
const PREEMPHASIS_BANDS: std::ops::Range<usize> = LONG_BANDS - 4..LONG_BANDS;

/// Quantises the nonzero lines `xr` in at most `available_bits`, amplifying
/// the scalefactor bands whose quantisation noise exceeds the allowed
/// distortion `xmin`.
///
/// Each iteration amplifies the lines by the current scalefactors, runs the
/// inner loop with the bits left after the scalefactors, and measures the
/// distortion `xfsf` of every band (clause C.1.5.4.3.3). It saves that
/// result, then amplifies each band whose distortion exceeds its `xmin`, and
/// raises its `xmin` with it (clause C.1.5.4.3.5). It stops when no band
/// exceeds its `xmin`, when every band has been amplified, or when a
/// scalefactor would pass its limit (clause C.1.5.4.3.6), and returns the
/// saved result: the last one whose distortion was measured.
///
/// Spec reading: figure C.9.b asks whether any band has "more than the allowed
/// distortion" after the amplification. That is read as a question about the
/// distortion measured in the iteration against the allowance it was measured
/// with. Comparing it with the allowances just raised would end the loop as
/// soon as raising them sufficed, and restoring the saved scalefactors would
/// then undo the amplification it was meant to keep.
///
/// Spec gap: clause C.1.5.4.3.4 leaves the choice of preemphasis to the
/// implementation and suggests the rule used here: `preflag` is set if, after
/// the first inner loop, all of the upper four bands exceed their `xmin`.
///
/// `scalefac_scale` starts at 0. Clause C.1.5.4.3 switches it to 1 when the
/// scalefactors would otherwise pass their limits (figure C.9.b leaves this
/// out). Spec gap: the clause says only that the scalefactors are then
/// "corrected accordingly"; here they are converted to the coarser step
/// rounding up, so that no band is amplified less than before. Only when they
/// would pass the limits again does the loop stop.
///
/// The amplified lines are recomputed from `xr` each iteration rather than
/// multiplied incrementally as clause C.1.5.4.3 describes; the result is
/// the same.
///
/// # Panics
///
/// If `xr` cannot be quantised even without scalefactors, which takes lines
/// far beyond the decoder's output range.
pub fn outer_loop(
    xr: &[f64; GRANULE_LINES],
    xmin: &[f64; LONG_BANDS],
    available_bits: usize,
    sampling_frequency: SamplingFrequency,
) -> QuantizedGranule {
    let bands = scalefactor_bands::long_bands(sampling_frequency);
    let mut scalefactors = Scalefactors::default();
    let mut saved: Option<QuantizedGranule> = None;
    let mut start_gain = None;

    loop {
        let amplified = amplify(xr, &scalefactors, bands);
        let compress = scalefactors
            .compress()
            .expect("scalefactors are kept within their limits");
        let granule = available_bits
            .checked_sub(part2_length(compress))
            .and_then(|bits| inner_loop(&amplified, bits, sampling_frequency, start_gain));
        let Some(mut granule) = granule else {
            return saved.expect("lines without scalefactors always quantise");
        };
        granule.scalefactors = scalefactors.clone();
        start_gain = Some(granule.global_gain);

        let distortion = distortion(&amplified, &granule, bands);
        let allowed_for = |scalefactors: &Scalefactors| -> [f64; LONG_BANDS] {
            std::array::from_fn(|sb| xmin[sb] * scalefactors.amplification(sb).powi(2))
        };
        let measured_against = allowed_for(&scalefactors);
        let violated = (0..LONG_BANDS).any(|sb| distortion[sb] > measured_against[sb]);
        let first = saved.is_none();
        saved = Some(granule);

        // C.1.5.4.3.4: preemphasis, which raises the allowed distortion of the
        // preemphasised bands along with their lines.
        if first
            && PREEMPHASIS_BANDS
                .clone()
                .all(|sb| distortion[sb] > measured_against[sb])
        {
            scalefactors.preflag = true;
        }

        // C.1.5.4.3.5: amplify the bands with more than the allowed distortion.
        let allowed = allowed_for(&scalefactors);
        for sb in 0..LONG_BANDS {
            if allowed[sb] - distortion[sb] < 0.0 {
                scalefactors.scalefac_l[sb] += 1;
            }
        }

        // C.1.5.4.3.6: the conditions for stopping, in the order of figure C.9.b.
        if scalefactors.scalefac_l.iter().all(|&s| s > 0) {
            break;
        }
        if exceeds_limits(&scalefactors) {
            if scalefactors.scalefac_scale {
                break;
            }
            scalefactors = coarser(&scalefactors);
        }
        if !violated {
            break;
        }
    }
    saved.expect("the loop runs at least once")
}

/// `xr` with each scalefactor band multiplied by its amplification. Lines
/// above the last band have no scalefactor and stay as they are.
fn amplify(
    xr: &[f64; GRANULE_LINES],
    scalefactors: &Scalefactors,
    bands: &[Band; LONG_BANDS],
) -> [f64; GRANULE_LINES] {
    let mut amplified = *xr;
    for (sb, band) in bands.iter().enumerate() {
        let factor = scalefactors.amplification(sb);
        for x in &mut amplified[band.start..band.end()] {
            *x *= factor;
        }
    }
    amplified
}

/// `xfsf(sb)` of clause C.1.5.4.3.3: the mean squared difference between
/// the amplified lines and their requantised values in each band.
fn distortion(
    amplified: &[f64; GRANULE_LINES],
    granule: &QuantizedGranule,
    bands: &[Band; LONG_BANDS],
) -> [f64; LONG_BANDS] {
    let step = step_size(granule.global_gain);
    bands.map(|band| {
        let sum: f64 = (band.start..band.end())
            .map(|i| {
                let requantised =
                    f64::from(granule.values[i].unsigned_abs()).powf(4.0 / 3.0) * step;
                (amplified[i].abs() - requantised).powi(2)
            })
            .sum();
        sum / band.width as f64
    })
}

fn exceeds_limits(scalefactors: &Scalefactors) -> bool {
    scalefactors
        .scalefac_l
        .iter()
        .enumerate()
        .any(|(sb, &s)| s > Scalefactors::limit(sb))
}

/// The scalefactors with `scalefac_scale` set: each band gets the smallest
/// scalefactor that amplifies it at least as much as before. A band
/// amplified by `2^((s + p) / 2)` with `scalefac_scale` 0, where `p` is its
/// preemphasis, needs `ceil((s - p) / 2)` steps of 2, and none if
/// preemphasis alone already amplifies it enough.
fn coarser(scalefactors: &Scalefactors) -> Scalefactors {
    let mut coarser = scalefactors.clone();
    coarser.scalefac_scale = true;
    for (sb, scalefac) in coarser.scalefac_l.iter_mut().enumerate() {
        let pretab = if scalefactors.preflag { PRETAB[sb] } else { 0 };
        *scalefac = scalefac.saturating_sub(pretab).div_ceil(2);
    }
    coarser
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: SamplingFrequency = SamplingFrequency::Hz44100;

    /// A spectrum falling from low to high frequencies, with every line
    /// nonzero.
    fn spectrum() -> [f64; GRANULE_LINES] {
        std::array::from_fn(|i| {
            let level = 0.05 * (-(i as f64) / 150.0).exp();
            if i % 2 == 0 { level } else { -0.7 * level }
        })
    }

    fn total_bits(granule: &QuantizedGranule) -> usize {
        let compress = granule.scalefactors.compress().unwrap();
        let bands = scalefactor_bands::long_bands(FS);
        part2_length(compress) + granule.coding.bit_count(&granule.values, bands)
    }

    /// With a generous allowance nothing needs amplifying, and the result is
    /// the inner loop's alone.
    #[test]
    fn stops_at_once_when_every_band_is_within_its_allowance() {
        let xr = spectrum();
        let granule = outer_loop(&xr, &[f64::INFINITY; LONG_BANDS], 1500, FS);
        assert_eq!(granule.scalefactors, Scalefactors::default());
        let inner = inner_loop(&xr, 1500, FS, None).unwrap();
        assert_eq!(granule.global_gain, inner.global_gain);
    }

    /// Bands asked for less noise than an even spread gives are amplified,
    /// and the result still fits.
    #[test]
    fn amplifies_the_bands_that_need_less_noise() {
        let xr = spectrum();
        let bands = scalefactor_bands::long_bands(FS);
        let plain = inner_loop(&xr, 1500, FS, None).unwrap();
        let noise = distortion(&xr, &plain, bands);
        // Bands 4 to 7 must be 10 dB quieter; the rest may be as they are.
        let xmin: [f64; LONG_BANDS] = std::array::from_fn(|sb| {
            if (4..8).contains(&sb) {
                noise[sb] / 10.0
            } else {
                f64::INFINITY
            }
        });
        let granule = outer_loop(&xr, &xmin, 1500, FS);
        assert!(total_bits(&granule) <= 1500);
        for sb in 4..8 {
            assert!(granule.scalefactors.scalefac_l[sb] > 0, "band {sb}");
        }
        for sb in (0..4).chain(8..LONG_BANDS) {
            assert_eq!(granule.scalefactors.scalefac_l[sb], 0, "band {sb}");
        }
        assert!(!granule.scalefactors.preflag);
    }

    /// When no allowance can be met, the loop still ends within the limits of
    /// the scalefactors and the bits.
    #[test]
    fn ends_within_the_limits_when_nothing_can_be_met() {
        let xr = spectrum();
        for bits in [60, 300, 1500, 4000] {
            let granule = outer_loop(&xr, &[0.0; LONG_BANDS], bits, FS);
            assert!(total_bits(&granule) <= bits, "{bits}");
            assert!(granule.scalefactors.compress().is_some(), "{bits}");
        }
    }

    /// Preemphasis is switched on when all of the upper four bands are too
    /// noisy after the first inner loop.
    #[test]
    fn preemphasis_follows_the_upper_bands() {
        let xr = spectrum();
        let bands = scalefactor_bands::long_bands(FS);
        let plain = inner_loop(&xr, 1500, FS, None).unwrap();
        let noise = distortion(&xr, &plain, bands);
        let xmin: [f64; LONG_BANDS] = std::array::from_fn(|sb| {
            if PREEMPHASIS_BANDS.contains(&sb) {
                noise[sb] / 4.0
            } else {
                f64::INFINITY
            }
        });
        assert!(outer_loop(&xr, &xmin, 1500, FS).scalefactors.preflag);
        let mut xmin = xmin;
        xmin[LONG_BANDS - 1] = f64::INFINITY;
        assert!(!outer_loop(&xr, &xmin, 1500, FS).scalefactors.preflag);
    }

    /// The conversion to `scalefac_scale` never amplifies a band less.
    #[test]
    fn coarser_scalefactors_amplify_at_least_as_much() {
        for preflag in [false, true] {
            let mut scalefactors = Scalefactors {
                preflag,
                ..Scalefactors::default()
            };
            for sb in 0..LONG_BANDS {
                scalefactors.scalefac_l[sb] = (sb % 17) as u8;
            }
            let coarser = coarser(&scalefactors);
            for sb in 0..LONG_BANDS {
                let (before, after) = (scalefactors.amplification(sb), coarser.amplification(sb));
                assert!(after >= before, "band {sb}");
                // Rounding up costs at most one step of the finer scale.
                let finer_step = 2f64.sqrt() * (1.0 + 1e-12);
                assert!(
                    after <= before * finer_step || coarser.scalefac_l[sb] == 0,
                    "band {sb}"
                );
            }
        }
    }

    #[test]
    fn distortion_is_the_mean_squared_requantisation_error() {
        let bands = scalefactor_bands::long_bands(FS);
        let mut xr = [0.0; GRANULE_LINES];
        xr[0] = 1.5;
        let mut values = [0; GRANULE_LINES];
        values[0] = 1;
        let granule = QuantizedGranule::new(210, values, FS);
        let noise = distortion(&xr, &granule, bands);
        assert!((noise[0] - 0.25 / bands[0].width as f64).abs() < 1e-12);
    }
}
