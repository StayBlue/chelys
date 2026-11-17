// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Quantisation of frequency lines (clause C.1.5.4.4.1) and the inner
//! iteration loop that sets the step size (clause C.1.5.4.4, figure C.9.c).

use crate::mp3::header::SamplingFrequency;
use crate::mp3::huffman::MAX_MAGNITUDE;
use crate::mp3::{BlockType, GRANULE_LINES, QuantizedGranule, scalefactor_bands};

/// Spec gap: clause C.1.5.4.1 sets `global_gain = qquant + system_constant`
/// and leaves the constant to the implementation. The decoder requantises with
/// `2^((global_gain - 210) / 4)` (clause 2.4.3.4.7.1), so with `xr` on the
/// decoder's scale, taking `qquant + quantanf = global_gain - 210` in the
/// quantizer of clause C.1.5.4.4.1 makes it the inverse of the decoder's
/// requantizer.
const GLOBAL_GAIN_OFFSET: i32 = 210;

/// The quantizer step size of `global_gain`, `2^((global_gain - 210) / 4)`.
pub fn step_size(global_gain: u8) -> f64 {
    (f64::from(i32::from(global_gain) - GLOBAL_GAIN_OFFSET) / 4.0).exp2()
}

/// Quantises `xr` with step size `global_gain`:
/// `ix(i) = nint((|xr(i)| / 2^((global_gain - 210) / 4))^0.75 - 0.0946)`,
/// carrying the sign of `xr(i)`. Returns `None` if a magnitude would exceed
/// the largest the Huffman tables can code (clause C.1.5.4.4.2).
pub fn quantize(xr: &[f64; GRANULE_LINES], global_gain: u8) -> Option<[i32; GRANULE_LINES]> {
    Lines::new(xr).quantize(global_gain)
}

/// A granule's lines with their magnitudes raised to the power 0.75 once, so
/// that each step size tried costs one multiplication per line:
/// `(|xr| / step)^0.75 = |xr|^0.75 * step^-0.75`.
struct Lines<'a> {
    xr: &'a [f64; GRANULE_LINES],
    powered: [f64; GRANULE_LINES],
}

impl<'a> Lines<'a> {
    fn new(xr: &'a [f64; GRANULE_LINES]) -> Self {
        Self {
            xr,
            powered: xr.map(|x| x.abs().powf(0.75)),
        }
    }

    fn quantize(&self, global_gain: u8) -> Option<[i32; GRANULE_LINES]> {
        let scale = step_size(global_gain).powf(-0.75);
        let mut ix = [0; GRANULE_LINES];
        let mut largest = 0;
        for ((ix, &powered), &xr) in ix.iter_mut().zip(&self.powered).zip(self.xr) {
            let magnitude = round_to_magnitude(powered * scale - 0.0946);
            largest = largest.max(magnitude);
            *ix = if xr.is_sign_negative() {
                -magnitude
            } else {
                magnitude
            };
        }
        (largest <= MAX_MAGNITUDE as i32).then_some(ix)
    }
}

/// `nint(x)`, rounding halves away from zero as [`f64::round`] does, but
/// zero for negative `x`, and at most [`BEYOND_MAGNITUDE`], beyond
/// [`MAX_MAGNITUDE`]. Without SSE4.1, `f64::round` is a call to the C
/// library; this takes a conversion to an integer, which truncates, and a
/// comparison, and gives the same values.
fn round_to_magnitude(x: f64) -> i32 {
    let x = x.clamp(0.0, f64::from(BEYOND_MAGNITUDE));
    let truncated = x as i32;
    // Exact: `x` and its truncation lie within 1 of each other.
    let fraction = x - f64::from(truncated);
    truncated + i32::from(fraction >= 0.5)
}

/// A magnitude that the Huffman tables cannot code.
const BEYOND_MAGNITUDE: i32 = 2 * MAX_MAGNITUDE as i32;

/// The inner iteration loop (clause C.1.5.4.4, figure C.9.c): quantises
/// `xr`, the lines of a granule of `block_type` in the order they are sent,
/// with the smallest step size whose Huffman coding fits in
/// `available_bits`, or returns `None` if not even the largest step does.
///
/// The loop raises the step size one `global_gain` at a time until the
/// quantised values are within the range of the tables (clause C.1.5.4.4.2)
/// and their bits fit. Spec reading: the decision in figure C.9.c has its "y"
/// and "n" swapped relative to that text, which is followed.
///
/// Spec gap: clause C.1.5.4.2.1 leaves the starting step, `quantanf`, to the
/// implementation, asking only that the first iteration need more bits than
/// are available and suggesting it be chosen to save iterations. Here the
/// search starts from `start_gain` if given, as the outer loop does with the
/// step of its last iteration, and otherwise from a bisection of the whole
/// range. Either way it ends on a step that fits with the next finer step not
/// fitting, as the loop of figure C.9.c would started from that finer step.
///
/// A granule whose lines are all zero skips the loop (figure C.9.a).
pub fn inner_loop(
    xr: &[f64; GRANULE_LINES],
    available_bits: usize,
    block_type: BlockType,
    sampling_frequency: SamplingFrequency,
    start_gain: Option<u8>,
) -> Option<QuantizedGranule> {
    if xr.iter().all(|&x| x == 0.0) {
        return Some(QuantizedGranule::new(
            0,
            [0; GRANULE_LINES],
            block_type,
            sampling_frequency,
        ));
    }
    let bands = scalefactor_bands::coding_bands(block_type, sampling_frequency);
    let lines = Lines::new(xr);
    let attempt = |global_gain: u8| {
        let values = lines.quantize(global_gain)?;
        let granule = QuantizedGranule::new(global_gain, values, block_type, sampling_frequency);
        let fits = granule.coding.bit_count(&granule.values, &bands) <= available_bits;
        fits.then_some(granule)
    };

    let start = start_gain.unwrap_or_else(|| bisect(&attempt));
    if let Some(mut granule) = attempt(start) {
        // Finer steps while they fit.
        while let Some(finer) = granule.global_gain.checked_sub(1).and_then(&attempt) {
            granule = finer;
        }
        return Some(granule);
    }
    // Coarser steps until one fits: doubling the distance, then bisecting.
    let (mut over, mut distance) = (start, 1u8);
    let fits = loop {
        let next = over.saturating_add(distance);
        if attempt(next).is_some() {
            break next;
        }
        if next == u8::MAX {
            return None;
        }
        over = next;
        distance = distance.saturating_mul(2);
    };
    let fits = bisect_between(&attempt, over, fits);
    attempt(fits)
}

/// A step size near the finest that fits, by bisection over the whole range.
/// Spec gap: this assumes that larger steps never need more bits, which the
/// Huffman tables do not quite guarantee; the search in [`inner_loop`] then
/// ends on a step whose next finer step does not fit.
fn bisect(attempt: &impl Fn(u8) -> Option<QuantizedGranule>) -> u8 {
    if attempt(0).is_some() {
        return 0;
    }
    if attempt(u8::MAX).is_none() {
        return u8::MAX;
    }
    bisect_between(attempt, 0, u8::MAX)
}

/// Narrows `over`, which does not fit, and `fits`, which does, to adjacent
/// steps and returns the one that fits.
fn bisect_between(
    attempt: &impl Fn(u8) -> Option<QuantizedGranule>,
    mut over: u8,
    mut fits: u8,
) -> u8 {
    while fits - over > 1 {
        let middle = over + (fits - over) / 2;
        if attempt(middle).is_some() {
            fits = middle;
        } else {
            over = middle;
        }
    }
    fits
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decoder restores `sign(ix) * |ix|^(4/3) * 2^((global_gain - 210) / 4)`
    /// (clause 2.4.3.4.7.1), so values on the quantizer's grid come back
    /// unchanged.
    #[test]
    fn values_on_the_grid_survive_requantisation() {
        let global_gain = 190;
        let step = 2f64.powf((190.0 - 210.0) / 4.0);
        let mut xr = [0.0; GRANULE_LINES];
        for (i, xr) in xr.iter_mut().enumerate().take(20) {
            let ix = i as f64 - 10.0;
            *xr = ix.signum() * ix.abs().powf(4.0 / 3.0) * step;
        }
        let ix = quantize(&xr, global_gain).unwrap();
        assert_eq!(
            &ix[..20],
            &std::array::from_fn::<i32, 20, _>(|i| i as i32 - 10)
        );
    }

    /// The -0.0946 offset rounds magnitudes just over halfway down.
    #[test]
    fn rounds_with_the_offset() {
        let mut xr = [0.0; GRANULE_LINES];
        // 0.55^(4/3) quantises to nint(0.55 - 0.0946) = 0; 0.6^(4/3) to 1.
        xr[0] = 0.55f64.powf(4.0 / 3.0);
        xr[1] = -(0.6f64.powf(4.0 / 3.0));
        let ix = quantize(&xr, 210).unwrap();
        assert_eq!(ix[..2], [0, -1]);
    }

    #[test]
    fn rounds_as_f64_round_does() {
        let mut x = 0.0f64;
        let mut cases = vec![
            -1.5,
            -0.5,
            -0.0,
            0.0,
            0.5,
            1.5,
            2.5,
            8191.5,
            1e300,
            f64::MAX,
        ];
        for _ in 0..200_000 {
            x = (x * 1.618_033_988_7 + 0.123_456_789) % 9000.0;
            cases.extend([
                x,
                x.floor() + 0.5,
                (x.floor() + 0.5).next_down(),
                x - 4500.0,
            ]);
        }
        for x in cases {
            let expected = x.round().clamp(0.0, f64::from(BEYOND_MAGNITUDE)) as i32;
            assert_eq!(round_to_magnitude(x), expected, "{x}");
        }
    }

    #[test]
    fn rejects_magnitudes_the_tables_cannot_code() {
        let mut xr = [0.0; GRANULE_LINES];
        xr[0] = 10_000f64.powf(4.0 / 3.0);
        assert!(quantize(&xr, 210).is_none());
        assert!(quantize(&xr, 230).is_some());
    }

    fn tone_granule() -> [f64; GRANULE_LINES] {
        std::array::from_fn(|i| {
            let decay = (-(i as f64) / 60.0).exp();
            0.3 * decay * if i % 3 == 0 { 1.0 } else { -0.6 }
        })
    }

    #[test]
    fn inner_loop_uses_the_finest_step_that_fits() {
        let fs = SamplingFrequency::Hz44100;
        let bands = scalefactor_bands::long_bands(fs);
        let xr = tone_granule();
        for available in [100, 400, 1500, 4000] {
            let granule = inner_loop(&xr, available, BlockType::Normal, fs, None).unwrap();
            let bits = granule.coding.bit_count(&granule.values, bands);
            assert!(bits <= available, "{bits} bits for {available}");
            // One step finer either leaves the tables' range or needs too many bits.
            let finer = quantize(&xr, granule.global_gain - 1).map(|values| {
                let finer =
                    QuantizedGranule::new(granule.global_gain - 1, values, BlockType::Normal, fs);
                finer.coding.bit_count(&finer.values, bands)
            });
            assert!(finer.is_none_or(|bits| bits > available), "{available}");
        }
    }

    /// Wherever the search starts, it ends on the same step.
    #[test]
    fn inner_loop_ends_on_the_same_step_from_any_start() {
        let fs = SamplingFrequency::Hz44100;
        let xr = tone_granule();
        let expected = inner_loop(&xr, 800, BlockType::Normal, fs, None)
            .unwrap()
            .global_gain;
        for start in [0, 100, expected - 1, expected, expected + 1, 200, 255] {
            let granule = inner_loop(&xr, 800, BlockType::Normal, fs, Some(start)).unwrap();
            assert_eq!(granule.global_gain, expected, "from {start}");
        }
    }

    #[test]
    fn inner_loop_gives_up_when_nothing_fits() {
        // Lines so loud that even the largest step leaves 1s in every pair.
        let xr = [1e6; GRANULE_LINES];
        assert!(inner_loop(&xr, 10, BlockType::Normal, SamplingFrequency::Hz44100, None).is_none());
    }

    #[test]
    fn inner_loop_skips_silence() {
        let granule = inner_loop(
            &[0.0; GRANULE_LINES],
            0,
            BlockType::Short,
            SamplingFrequency::Hz32000,
            None,
        )
        .unwrap();
        assert!(granule.values.iter().all(|&v| v == 0));
        assert_eq!(granule.coding.big_values, 0);
    }

    #[test]
    fn inner_loop_fits_even_with_no_bits() {
        let fs = SamplingFrequency::Hz48000;
        let granule = inner_loop(&tone_granule(), 0, BlockType::Normal, fs, None).unwrap();
        let bands = scalefactor_bands::long_bands(fs);
        assert_eq!(granule.coding.bit_count(&granule.values, bands), 0);
    }
}
