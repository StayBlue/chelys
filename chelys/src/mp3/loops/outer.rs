// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The outer iteration loop, or distortion control loop (clause C.1.5.4.3,
//! figure C.9.b).

use super::inner::{inner_loop, step_size};
use crate::mp3::header::SamplingFrequency;
use crate::mp3::scalefactor_bands::{self, Band, LONG_BANDS};
use crate::mp3::scalefactors::{PRETAB, Scalefactors, ZeroPart};
use crate::mp3::side_info::SCFSI_BANDS;
use crate::mp3::stereo::IntensityPositions;
use crate::mp3::{BlockType, GRANULE_LINES, QuantizedGranule};

/// The scalefactors of a frame's first granule, and the scfsi bands in which
/// the second granule takes them over (clause C.1.5.4.2.3).
#[derive(Clone, Copy, Debug)]
pub struct Reuse<'a> {
    pub first: &'a Scalefactors,
    pub scfsi: [bool; SCFSI_BANDS],
}

/// A granule quantised by the iteration loops.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    pub granule: QuantizedGranule,
    /// Whether the distortion of every scalefactor band sent is within its
    /// allowance: whether the granule had the bits it needed.
    pub within_allowance: bool,
}

/// The upper scalefactor bands whose distortion decides preemphasis.
const PREEMPHASIS_BANDS: std::ops::Range<usize> = LONG_BANDS - 4..LONG_BANDS;

/// Quantises the nonzero lines `xr` of a granule of `block_type` in at most
/// `available_bits`, amplifying the scalefactor bands whose quantisation
/// noise exceeds the allowed distortion `xmin`. The lines and `xmin` are in
/// the order they are sent; for short blocks each scalefactor band counts
/// once per window (see
/// [`coding_bands`](crate::mp3::scalefactor_bands::coding_bands)).
///
/// Each iteration amplifies the lines by the current scalefactors, runs the
/// inner loop with the bits left after the scalefactors, and measures the
/// distortion `xfsf` of every band (clause C.1.5.4.3.3). It saves that
/// result, then amplifies each band whose distortion exceeds its `xmin`, and
/// raises its `xmin` with it (clause C.1.5.4.3.5). It stops when no band
/// exceeds its `xmin`, when every band has been amplified, or when a
/// scalefactor would pass its limit (clause C.1.5.4.3.6), and returns, of
/// the results it measured, the one whose noise exceeds the allowances least.
///
/// Spec reading: when the loop stops with bands still over their allowance,
/// clause C.1.5.4.3.6 gives as its output the scalefactors saved last, those
/// of the last iteration (clause C.1.5.4.3.7). This deliberately departs from
/// that, and keeps the iteration whose excess is least: the sum, over the
/// bands whose distortion keeps the loop going, of the logarithm of how far
/// each band's distortion exceeds its allowance. Short of bits, amplifying
/// the bands over their allowance takes bits for scalefactors and a coarser
/// step for every band, and the last iteration can be the worst. Every
/// iteration fits the bits, so any of them is a valid granule. Graded with
/// PEAQ (GstPEAQ 0.6.1, black box), 22 tracks of pop music at 44.1 kHz in
/// stereo rose from a mean of -1.59 to -1.21 at 128 kbit/s (from -1.99 to
/// -1.61 for the worst track), from -2.84 to -2.60 at 96 kbit/s, and from
/// -0.25 to -0.23 at 192 kbit/s; `--vbr`, whose frames end within their
/// allowance, did not change. The largest excess, in place of the sum, and
/// the sum weighted by the bands' widths both graded lower.
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
/// Short blocks never use it (clause 2.4.2.7), nor does the right channel of
/// a frame coded in intensity stereo at the lower sampling frequencies
/// (clause 2.4.3.2 of ISO/IEC 13818-3).
///
/// With `reuse` in at least one scfsi band, the second granule of a frame
/// takes over the first granule's `preflag` and `scalefac_scale`
/// (`ifqstep`), and its scalefactors in those bands, which are then never
/// amplified (clauses C.1.5.4.3.4 and C.1.5.4.3.5). Spec reading: as those
/// bands cannot be amplified, their distortion does not keep the loop going;
/// and as `ifqstep` must stay that of the first granule, the loop stops,
/// rather than switching `scalefac_scale`, when a scalefactor would pass its
/// limit. Should the lines not quantise at all with the scalefactors taken
/// over, the granule is coded without `scfsi`.
///
/// With `intensity`, the granule is the right channel of a frame coded in
/// intensity stereo, whose zero part sends intensity stereo positions in
/// place of scalefactors (see [`IntensityPositions::mark`]). Those bands'
/// lines quantise to zero whatever their scalefactors, so the loop works
/// with scalefactors as usual, and each result is marked before its bits are
/// counted. If the marks take more bits than the scalefactors they replace,
/// the inner loop runs again with that many fewer bits.
///
/// `scalefac_scale` starts at 0. Clause C.1.5.4.3 switches it to 1 when the
/// scalefactors would otherwise pass their limits (figure C.9.b leaves this
/// out). Spec gap: the clause says only that the scalefactors are then
/// "corrected accordingly"; here they are converted to the coarser step
/// rounding up, so that no band is amplified less than before. Only when they
/// would pass the limits again does the loop stop.
///
/// At MPEG-2.5's 8 kHz, the scalefactors of the highest region stay zero
/// (see [`fixed_at_zero`](scalefactor_bands::fixed_at_zero)), and, like
/// bands taken over with scfsi, their distortion does not keep the loop
/// going. Spec reading: preemphasis is not used there, as `pretab` would
/// amplify those bands, where US 6,185,539 B1 codes them with "a
/// multiplication factor of 1" (claim 1).
///
/// The amplified lines are recomputed from `xr` each iteration rather than
/// multiplied incrementally as clause C.1.5.4.3 describes; the result is
/// the same.
///
/// # Panics
///
/// If `xr` cannot be quantised even without scalefactors, which takes lines
/// far beyond the decoder's output range, or the marks of `intensity` take
/// more than `available_bits`, which takes fewer than its
/// [`silent_bits`](IntensityPositions::silent_bits).
pub fn outer_loop(
    xr: &[f64; GRANULE_LINES],
    xmin: &[f64],
    block_type: BlockType,
    available_bits: usize,
    sampling_frequency: SamplingFrequency,
    reuse: Option<Reuse>,
    intensity: Option<&IntensityPositions>,
) -> Outcome {
    let bands = scalefactor_bands::coding_bands(block_type, sampling_frequency);
    assert_eq!(xmin.len(), bands.len(), "one xmin per scalefactor");
    let fixed = scalefactor_bands::fixed_at_zero(block_type, sampling_frequency);
    let reuse = reuse.filter(|reuse| reuse.scfsi.contains(&true));
    assert!(
        reuse.is_none() || intensity.is_none(),
        "intensity stereo positions are not taken over with scfsi"
    );
    let mut scalefactors = Scalefactors::new(block_type, sampling_frequency.version());
    scalefactors.intensity = intensity.map(|_| ZeroPart::UNMARKED);
    if let Some(reuse) = reuse {
        assert!(!block_type.short_windows(), "short blocks have no scfsi");
        scalefactors.preflag = reuse.first.preflag;
        scalefactors.scalefac_scale = reuse.first.scalefac_scale;
        scalefactors.scfsi = reuse.scfsi;
        for (sb, scalefac) in scalefactors.values_mut().iter_mut().enumerate() {
            if reuse.scfsi[crate::mp3::scalefactors::scfsi_band(sb)] {
                *scalefac = reuse.first.values()[sb];
            }
        }
    }
    // The least excess measured, and the result that had it.
    let mut best: Option<(f64, Outcome)> = None;
    let mut start_gain = None;

    loop {
        let amplified = amplify(xr, &scalefactors, &bands);
        let granule = fit(
            &amplified,
            &scalefactors,
            block_type,
            available_bits,
            sampling_frequency,
            start_gain,
            intensity,
        );
        let Some(granule) = granule else {
            if best.is_none() && reuse.is_some() {
                return outer_loop(
                    xr,
                    xmin,
                    block_type,
                    available_bits,
                    sampling_frequency,
                    None,
                    intensity,
                );
            }
            return best.expect("lines without scalefactors always quantise").1;
        };
        start_gain = Some(granule.global_gain);

        let distortion = distortion(&amplified, &granule, &bands);
        let allowed_for = |scalefactors: &Scalefactors| -> Vec<f64> {
            (xmin.iter().enumerate())
                .map(|(sb, xmin)| xmin * scalefactors.amplification(sb).powi(2))
                .collect()
        };
        let measured_against = allowed_for(&scalefactors);
        let excess: f64 = (0..bands.len())
            .filter(|&sb| scalefactors.sent(sb) && !fixed.contains(&sb))
            .map(|sb| (distortion[sb] / measured_against[sb]).max(1.0).ln())
            .sum();
        let violated = excess > 0.0;
        let first = best.is_none();
        if best.as_ref().is_none_or(|(least, _)| excess < *least) {
            let outcome = Outcome {
                granule,
                within_allowance: !violated,
            };
            best = Some((excess, outcome));
        }

        // C.1.5.4.3.4: preemphasis, which raises the allowed distortion of the
        // preemphasised bands along with their lines.
        if first
            && scalefactors.can_preemphasise()
            && reuse.is_none()
            && fixed.is_empty()
            && PREEMPHASIS_BANDS
                .clone()
                .all(|sb| distortion[sb] > measured_against[sb])
        {
            scalefactors.preflag = true;
        }

        // C.1.5.4.3.5: amplify the bands with more than the allowed distortion.
        let allowed = allowed_for(&scalefactors);
        let sent: Vec<bool> = (0..bands.len()).map(|sb| scalefactors.sent(sb)).collect();
        for (sb, scalefac) in scalefactors.values_mut().iter_mut().enumerate() {
            if sent[sb] && !fixed.contains(&sb) && allowed[sb] - distortion[sb] < 0.0 {
                *scalefac += 1;
            }
        }

        // C.1.5.4.3.6: the conditions for stopping, in the order of figure C.9.b.
        let mut free = (scalefactors.values().iter().enumerate())
            .filter(|(sb, _)| !fixed.contains(sb))
            .map(|(_, &s)| s);
        if free.all(|s| s > 0) {
            break;
        }
        if exceeds_limits(&scalefactors) {
            if scalefactors.scalefac_scale || reuse.is_some() {
                break;
            }
            scalefactors = coarser(&scalefactors);
        }
        if !violated {
            break;
        }
    }
    best.expect("the loop runs at least once").1
}

/// The inner loop for the lines `amplified` by `scalefactors`, in at most
/// `available_bits` with the scalefactors, marked for intensity stereo if
/// `intensity` is given. `None` if they do not fit.
fn fit(
    amplified: &[f64; GRANULE_LINES],
    scalefactors: &Scalefactors,
    block_type: BlockType,
    available_bits: usize,
    sampling_frequency: SamplingFrequency,
    start_gain: Option<u8>,
    intensity: Option<&IntensityPositions>,
) -> Option<QuantizedGranule> {
    let compress = scalefactors
        .compress()
        .expect("scalefactors are kept within their limits");
    let mut huffman_bits = available_bits.checked_sub(scalefactors.part2_length(compress))?;
    loop {
        let mut granule = inner_loop(
            amplified,
            huffman_bits,
            block_type,
            sampling_frequency,
            start_gain,
        )?;
        granule.scalefactors = scalefactors.clone();
        let Some(intensity) = intensity else {
            return Some(granule);
        };
        intensity.mark(&mut granule.scalefactors, &granule.values);
        let bits = granule.bit_len(sampling_frequency);
        if bits <= available_bits {
            return Some(granule);
        }
        huffman_bits = huffman_bits.checked_sub(bits - available_bits)?;
    }
}

/// How far a quantised granule's noise goes beyond its allowance: the
/// largest ratio, over its scalefactor bands, of the distortion `xfsf` to
/// the allowed distortion `xmin`, both as the outer loop measures them, of
/// the lines `xr` the granule was quantised from. At most 1 if every band is
/// within its allowance.
///
/// Unlike the outer loop's own test, every band counts: those whose
/// scalefactors are taken over with scfsi, or fixed at zero, are noise as
/// much as the rest. A band whose lines are all zero allows no distortion
/// and has none. Spec gap: the lines above the last scalefactor band have
/// no allowed distortion (clause C.1.5.4.1 gives `xmin` for the bands
/// only), and are not measured.
pub fn excess(
    xr: &[f64; GRANULE_LINES],
    xmin: &[f64],
    granule: &QuantizedGranule,
    sampling_frequency: SamplingFrequency,
) -> f64 {
    let bands = scalefactor_bands::coding_bands(granule.block_type, sampling_frequency);
    assert_eq!(xmin.len(), bands.len(), "one xmin per scalefactor");
    let amplified = amplify(xr, &granule.scalefactors, &bands);
    let distortion = distortion(&amplified, granule, &bands);
    (distortion.iter().zip(xmin).enumerate())
        .map(|(sb, (&distortion, &xmin))| {
            let allowed = xmin * granule.scalefactors.amplification(sb).powi(2);
            match (distortion, allowed) {
                (0.0, _) => 0.0,
                (_, 0.0) => f64::INFINITY,
                _ => distortion / allowed,
            }
        })
        .fold(0.0, f64::max)
}

/// `xr` with each scalefactor band multiplied by its amplification. Lines
/// above the last band have no scalefactor and stay as they are.
fn amplify(
    xr: &[f64; GRANULE_LINES],
    scalefactors: &Scalefactors,
    bands: &[Band],
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
    bands: &[Band],
) -> Vec<f64> {
    let step = step_size(granule.global_gain);
    bands
        .iter()
        .map(|band| {
            let sum: f64 = (band.start..band.end())
                .map(|i| {
                    let requantised =
                        f64::from(granule.values[i].unsigned_abs()).powf(4.0 / 3.0) * step;
                    (amplified[i].abs() - requantised).powi(2)
                })
                .sum();
            sum / band.width as f64
        })
        .collect()
}

fn exceeds_limits(scalefactors: &Scalefactors) -> bool {
    (scalefactors.values().iter().enumerate())
        .any(|(sb, &s)| scalefactors.sent(sb) && s > scalefactors.limit(sb))
}

/// The scalefactors with `scalefac_scale` set: each band gets the smallest
/// scalefactor that amplifies it at least as much as before. A band
/// amplified by `2^((s + p) / 2)` with `scalefac_scale` 0, where `p` is its
/// preemphasis, needs `ceil((s - p) / 2)` steps of 2, and none if
/// preemphasis alone already amplifies it enough.
fn coarser(scalefactors: &Scalefactors) -> Scalefactors {
    let mut coarser = scalefactors.clone();
    coarser.scalefac_scale = true;
    for (sb, scalefac) in coarser.values_mut().iter_mut().enumerate() {
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
        granule.bit_len(FS)
    }

    /// With a generous allowance nothing needs amplifying, and the result is
    /// the inner loop's alone.
    #[test]
    fn stops_at_once_when_every_band_is_within_its_allowance() {
        let xr = spectrum();
        let granule = outer_loop(
            &xr,
            &[f64::INFINITY; LONG_BANDS],
            BlockType::Normal,
            1500,
            FS,
            None,
            None,
        )
        .granule;
        assert_eq!(granule.scalefactors, Scalefactors::default());
        let inner = inner_loop(&xr, 1500, BlockType::Normal, FS, None).unwrap();
        assert_eq!(granule.global_gain, inner.global_gain);
    }

    /// Bands asked for less noise than an even spread gives are amplified,
    /// and the result still fits.
    #[test]
    fn amplifies_the_bands_that_need_less_noise() {
        let xr = spectrum();
        let bands = scalefactor_bands::long_bands(FS);
        let plain = inner_loop(&xr, 1500, BlockType::Normal, FS, None).unwrap();
        let noise = distortion(&xr, &plain, bands);
        // Bands 4 to 7 must be 10 dB quieter; the rest may be as they are.
        let xmin: [f64; LONG_BANDS] = std::array::from_fn(|sb| {
            if (4..8).contains(&sb) {
                noise[sb] / 10.0
            } else {
                f64::INFINITY
            }
        });
        let granule = outer_loop(&xr, &xmin, BlockType::Normal, 1500, FS, None, None).granule;
        assert!(total_bits(&granule) <= 1500);
        for sb in 4..8 {
            assert!(granule.scalefactors.values()[sb] > 0, "band {sb}");
        }
        for sb in (0..4).chain(8..LONG_BANDS) {
            assert_eq!(granule.scalefactors.values()[sb], 0, "band {sb}");
        }
        assert!(!granule.scalefactors.preflag);
    }

    /// When no allowance can be met, the loop still ends within the limits of
    /// the scalefactors and the bits.
    #[test]
    fn ends_within_the_limits_when_nothing_can_be_met() {
        let xr = spectrum();
        for bits in [60, 300, 1500, 4000] {
            let granule = outer_loop(
                &xr,
                &[0.0; LONG_BANDS],
                BlockType::Normal,
                bits,
                FS,
                None,
                None,
            )
            .granule;
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
        let plain = inner_loop(&xr, 1500, BlockType::Normal, FS, None).unwrap();
        let noise = distortion(&xr, &plain, bands);
        let xmin: [f64; LONG_BANDS] = std::array::from_fn(|sb| {
            if PREEMPHASIS_BANDS.contains(&sb) {
                noise[sb] / 4.0
            } else {
                f64::INFINITY
            }
        });
        assert!(
            outer_loop(&xr, &xmin, BlockType::Normal, 1500, FS, None, None)
                .granule
                .scalefactors
                .preflag
        );
        let mut xmin = xmin;
        xmin[LONG_BANDS - 1] = f64::INFINITY;
        assert!(
            !outer_loop(&xr, &xmin, BlockType::Normal, 1500, FS, None, None)
                .granule
                .scalefactors
                .preflag
        );
    }

    /// The second granule takes over the first granule's scalefactors in
    /// the scfsi bands and its `preflag` and `scalefac_scale`, and never
    /// amplifies those bands, however far their distortion exceeds `xmin`.
    /// The loop still ends, within its bits.
    #[test]
    fn scfsi_bands_keep_the_first_granules_scalefactors() {
        let xr = spectrum();
        let mut first = Scalefactors {
            preflag: true,
            ..Scalefactors::default()
        };
        for (sb, scalefac) in first.values_mut().iter_mut().enumerate() {
            *scalefac = (sb % 4) as u8;
        }
        let scfsi = [true, false, true, false];
        for bits in [300, 1500] {
            let reuse = Reuse {
                first: &first,
                scfsi,
            };
            let granule = outer_loop(
                &xr,
                &[0.0; LONG_BANDS],
                BlockType::Normal,
                bits,
                FS,
                Some(reuse),
                None,
            )
            .granule;
            let scalefactors = &granule.scalefactors;
            assert_eq!(scalefactors.scfsi, scfsi);
            assert!(scalefactors.preflag && !scalefactors.scalefac_scale);
            for sb in (0..6).chain(11..16) {
                assert_eq!(scalefactors.values()[sb], first.values()[sb], "band {sb}");
            }
            assert!((6..11).any(|sb| scalefactors.values()[sb] != first.values()[sb]));
            assert!(total_bits(&granule) <= bits, "{bits}");
        }
    }

    /// Without any scfsi band, reuse changes nothing.
    #[test]
    fn reuse_without_scfsi_is_ignored() {
        let xr = spectrum();
        let first = Scalefactors {
            preflag: true,
            scalefac_scale: true,
            ..Scalefactors::default()
        };
        let reuse = Reuse {
            first: &first,
            scfsi: [false; SCFSI_BANDS],
        };
        let xmin = [1e-9; LONG_BANDS];
        assert_eq!(
            outer_loop(&xr, &xmin, BlockType::Normal, 1500, FS, Some(reuse), None),
            outer_loop(&xr, &xmin, BlockType::Normal, 1500, FS, None, None)
        );
    }

    /// The right channel of a frame in intensity stereo sends its marks in
    /// place of the scalefactors of its zero part, within its bits, even
    /// when the marks need wider fields than the scalefactors did.
    #[test]
    fn intensity_marks_fit_in_the_bits() {
        let bands = scalefactor_bands::long_bands(FS);
        let mut xr = spectrum();
        xr[bands[12].start..].fill(0.0);
        let positions = IntensityPositions::new(&spectrum(), &xr, BlockType::Normal, 12, FS);
        for bits in [80, 150, 300, 1500] {
            let outcome = outer_loop(
                &xr,
                &[0.0; LONG_BANDS],
                BlockType::Normal,
                bits,
                FS,
                None,
                Some(&positions),
            );
            let granule = &outcome.granule;
            assert!(total_bits(granule) <= bits, "{bits}");
            assert!(!outcome.within_allowance, "{bits}");
            let mut marked = granule.scalefactors.clone();
            positions.mark(&mut marked, &granule.values);
            assert_eq!(marked, granule.scalefactors, "{bits}");
            assert_ne!(granule.scalefactors.values()[20], 0, "{bits}");
        }
        let outcome = outer_loop(
            &xr,
            &[f64::INFINITY; LONG_BANDS],
            BlockType::Normal,
            1500,
            FS,
            None,
            Some(&positions),
        );
        assert!(outcome.within_allowance);
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
                scalefactors.values_mut()[sb] = (sb % 17) as u8;
            }
            let coarser = coarser(&scalefactors);
            for sb in 0..LONG_BANDS {
                let (before, after) = (scalefactors.amplification(sb), coarser.amplification(sb));
                assert!(after >= before, "band {sb}");
                // Rounding up costs at most one step of the finer scale.
                let finer_step = 2f64.sqrt() * (1.0 + 1e-12);
                assert!(
                    after <= before * finer_step || coarser.values()[sb] == 0,
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
        let granule = QuantizedGranule::new(210, values, BlockType::Normal, FS);
        let noise = distortion(&xr, &granule, bands);
        assert!((noise[0] - 0.25 / bands[0].width as f64).abs() < 1e-12);
    }
}
