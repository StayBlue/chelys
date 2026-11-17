// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Quantisation of frequency lines (clause C.1.5.4.4.1) and the inner
//! iteration loop that sets the step size (clause C.1.5.4.4, figure C.9.c).

use crate::mp3::header::SamplingFrequency;
use crate::mp3::huffman::MAX_MAGNITUDE;
use crate::mp3::huffman::{Counted, SpectrumCoding};
use crate::mp3::scalefactor_bands::Band;
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

/// `step_size(global_gain)^-0.75`, by which [`Lines::quantize`] multiplies
/// the lines' powers, from a table of every `global_gain`.
fn scale(global_gain: u8) -> f64 {
    static TABLE: std::sync::OnceLock<[f64; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        std::array::from_fn(|global_gain| step_size(global_gain as u8).powf(-0.75))
    });
    table[usize::from(global_gain)]
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
#[derive(Clone)]
pub struct Lines {
    xr: [f64; GRANULE_LINES],
    powered: [f64; GRANULE_LINES],
    /// One past the last line that is not zero: those above quantise to
    /// zero at every step.
    end: usize,
}

impl Lines {
    pub fn new(xr: &[f64; GRANULE_LINES]) -> Self {
        Self::with_powers(xr, xr.map(power_three_quarters))
    }

    fn with_powers(xr: &[f64; GRANULE_LINES], powered: [f64; GRANULE_LINES]) -> Self {
        Self {
            xr: *xr,
            powered,
            end: end(xr),
        }
    }

    /// Makes these lines `xr` with each of the `bands` multiplied by its
    /// factor in `factors`, given that those of `applied` were last, and
    /// sets `applied` to them: only the bands whose factor changes are
    /// multiplied again and take their powers again. Lines above the last
    /// band stay as `xr` has them.
    pub fn amplify(
        &mut self,
        xr: &[f64; GRANULE_LINES],
        bands: &[Band],
        factors: &[f64],
        applied: &mut [f64],
    ) {
        for ((band, &factor), applied) in bands.iter().zip(factors).zip(applied) {
            if factor == *applied {
                continue;
            }
            *applied = factor;
            let lines = band.start..band.end();
            let lines = (self.xr[lines.clone()].iter_mut())
                .zip(&mut self.powered[lines.clone()])
                .zip(&xr[lines]);
            for ((line, powered), &x) in lines {
                *line = x * factor;
                *powered = power_three_quarters(*line);
            }
        }
        self.end = end(&self.xr);
    }

    pub fn xr(&self) -> &[f64; GRANULE_LINES] {
        &self.xr
    }

    /// [`quantize`] with these lines.
    pub fn quantize(&self, global_gain: u8) -> Option<[i32; GRANULE_LINES]> {
        self.quantize_below(global_gain, GRANULE_LINES)
    }

    /// [`quantize`](Self::quantize), given that the lines from `nonzero`
    /// up quantise to zero, as they do at any step at which they did at a
    /// finer one (see [`nonzero_end`]): the step's scale, and so each line's
    /// value, only falls as the step grows.
    pub fn quantize_below(&self, global_gain: u8, nonzero: usize) -> Option<[i32; GRANULE_LINES]> {
        let scale = scale(global_gain);
        let mut ix = [0; GRANULE_LINES];
        let mut largest = 0;
        let end = self.end.min(nonzero);
        for ((ix, &powered), &xr) in ix[..end].iter_mut().zip(&self.powered).zip(&self.xr) {
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

/// One past the last of the quantised `values` that is not zero.
pub fn nonzero_end(values: &[i32; GRANULE_LINES]) -> usize {
    values
        .iter()
        .rposition(|&v| v != 0)
        .map_or(0, |last| last + 1)
}

/// One past the last of the lines `xr` that is not zero.
fn end(xr: &[f64; GRANULE_LINES]) -> usize {
    xr.iter()
        .rposition(|&x| x != 0.0)
        .map_or(0, |last| last + 1)
}

/// `|x|^0.75`, as `(|x| * |x|^0.5)^0.5`: two square roots, which the
/// processor computes, rather than a call to the C library's `pow`. It
/// differs from `pow` by about the last bit.
fn power_three_quarters(x: f64) -> f64 {
    let x = x.abs();
    (x * x.sqrt()).sqrt()
}

/// `nint(x)`, rounding halves away from zero as [`f64::round`] does, but
/// zero for negative `x`, and at most [`BEYOND_MAGNITUDE`], beyond
/// [`MAX_MAGNITUDE`]. Without SSE4.1, `f64::round` is a call to the C
/// library, and a conversion to an integer is taken one line at a time;
/// this takes only additions and comparisons, which the compiler can
/// vectorise, and gives the same values.
fn round_to_magnitude(x: f64) -> i32 {
    let x = x.clamp(0.0, f64::from(BEYOND_MAGNITUDE));
    // Exact: from 2^52 on, the integers are the doubles, so adding it
    // rounds `x` to the nearest one, halves to the even one, and
    // subtracting it again gives that.
    let shifted = x + TWO_POW_52;
    let nearest = shifted - TWO_POW_52;
    // Exact, as `x` and the nearest integer lie within 1/2 of each other: a
    // half rounded down to the even integer is rounded up instead.
    let half_down = i32::from(x - nearest == 0.5);
    // 2^52 plus an integer holds it in the low bits of its mantissa.
    shifted.to_bits() as i32 + half_down
}

/// `2^52`, from which on the doubles are spaced 1 apart.
const TWO_POW_52: f64 = 4_503_599_627_370_496.0;

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
/// Spec reading: figure C.9.c counts the bits of every step it tries. Here,
/// whether they fit is often settled by bounds on them instead (see
/// [`Counted::bits_between`]), without searching every split of the
/// `big_values` regions, for speed. The answer is always the one the count
/// gives, so the loop tries the same steps and keeps the same one.
///
/// Spec gap: clause C.1.5.4.2.1 leaves the starting step, `quantanf`, to the
/// implementation, asking that the first iteration need more bits than are
/// available and suggesting it be adapted to the bitrate and the signal to
/// save iterations. The formula it gives, `system_const * log_c(sfm)`,
/// leaves `system_const` and `c` open, and as the spectral flatness `sfm`
/// does not change with the level of the lines, it cannot give a step that
/// depends on it, as the step that fits does. It is not used: the search
/// starts from `start_gain` if given, as the outer loop does with the step
/// of its last iteration, and otherwise from [`estimated_gain`].
/// Spec reading: that step may already fit, and the search then goes to
/// finer steps, which the loop of figure C.9.c never does. Either way it
/// ends on a step that fits with the next finer step not fitting, as that
/// loop would started from that finer step.
///
/// A granule whose lines are all zero skips the loop (figure C.9.a).
pub fn inner_loop(
    xr: &[f64; GRANULE_LINES],
    available_bits: usize,
    block_type: BlockType,
    sampling_frequency: SamplingFrequency,
    start_gain: Option<u8>,
) -> Option<QuantizedGranule> {
    inner_loop_lines(
        &Lines::new(xr),
        available_bits,
        block_type,
        sampling_frequency,
        start_gain,
    )
}

/// [`inner_loop`] for the [`Lines`] `lines`.
pub fn inner_loop_lines(
    lines: &Lines,
    available_bits: usize,
    block_type: BlockType,
    sampling_frequency: SamplingFrequency,
    start_gain: Option<u8>,
) -> Option<QuantizedGranule> {
    inner_loop_step(
        lines,
        available_bits,
        block_type,
        sampling_frequency,
        start_gain,
        None,
    )
    .map(|step| step.granule(block_type, sampling_frequency))
}

/// [`inner_loop_lines`], giving the step it ends on, whose granule, with
/// its Huffman coding, is made only if it is kept. Without `start_gain`,
/// the search starts from the step `estimate` gives, if it is that of
/// `lines`, or from that of [`estimated_gain`].
pub fn inner_loop_step(
    lines: &Lines,
    available_bits: usize,
    block_type: BlockType,
    sampling_frequency: SamplingFrequency,
    start_gain: Option<u8>,
    estimate: Option<&StepEstimate>,
) -> Option<Step> {
    let bands = scalefactor_bands::coding_bands(block_type, sampling_frequency);
    if lines.xr.iter().all(|&x| x == 0.0) {
        let values = [0; GRANULE_LINES];
        return Some(Step {
            global_gain: 0,
            values,
            counted: Box::new(SpectrumCoding::count(
                &values,
                block_type,
                &bands,
                sampling_frequency,
            )),
        });
    }
    // A step that fits, with its values; the granule is made only for the
    // step the search ends on.
    let attempt = |global_gain: u8| {
        let values = lines.quantize(global_gain)?;
        let counted = SpectrumCoding::count(&values, block_type, &bands, sampling_frequency);
        counted.bits_between(0, available_bits).then(|| Step {
            global_gain,
            values,
            counted: Box::new(counted),
        })
    };
    // Each search keeps the step it ends on, so that no step is quantised
    // twice.
    let start = start_gain.unwrap_or_else(|| match estimate {
        Some(estimate) => estimate.gain(available_bits),
        None => estimated_gain(lines, available_bits),
    });
    if let Some(mut step) = attempt(start) {
        // Finer steps while they fit.
        while let Some(finer) = step.global_gain.checked_sub(1).and_then(&attempt) {
            step = finer;
        }
        return Some(step);
    }
    // Coarser steps until one fits: doubling the distance, then bisecting.
    let (mut over, mut distance) = (start, 1u8);
    let fits = loop {
        let next = over.saturating_add(distance);
        if let Some(step) = attempt(next) {
            break step;
        }
        if next == u8::MAX {
            return None;
        }
        over = next;
        distance = distance.saturating_mul(2);
    };
    Some(bisect_between(&attempt, over, fits))
}

/// A step size, the values quantised with it, and their counts.
pub struct Step {
    pub global_gain: u8,
    pub values: [i32; GRANULE_LINES],
    /// The values counted for their coding, which the step's granule is
    /// made with.
    counted: Box<Counted>,
}

impl Step {
    /// The granule of the step, with all scalefactors zero.
    pub fn granule(
        self,
        block_type: BlockType,
        sampling_frequency: SamplingFrequency,
    ) -> QuantizedGranule {
        let (coding, _) = self.counted.choose();
        QuantizedGranule::with_coding(
            self.global_gain,
            self.values,
            block_type,
            sampling_frequency,
            coding,
        )
    }
}

/// The bits that the Huffman coding of a granule is estimated to need for
/// each value that is not zero, by [`estimated_gain`]. This and
/// [`ESTIMATED_BITS_PER_MAGNITUDE_BIT`] were fitted by least squares to the
/// bits counted at the steps around those the inner loop ended on, for the
/// granules of a 4.7-minute track of pop music at 128 kbit/s.
const ESTIMATED_BITS_PER_VALUE: f64 = 2.7;
/// The bits that the Huffman coding of a granule is estimated to need for
/// each bit of the magnitude of a value above that of 1, by
/// [`estimated_gain`].
const ESTIMATED_BITS_PER_MAGNITUDE_BIT: f64 = 1.12;

/// The step from which the inner loop searches when it has none from the
/// outer loop: the finest at which the Huffman coding of `lines` is
/// estimated to need at most `available_bits`.
///
/// The bits are estimated from the values quantised at each step, as
/// [`ESTIMATED_BITS_PER_VALUE`] for each value that is not zero and
/// [`ESTIMATED_BITS_PER_MAGNITUDE_BIT`] for each bit of its magnitude.
/// Each line is quantised to zero only from a step on, and each step
/// finer adds 3/16 of a bit to the magnitude of every value that is not.
/// So the estimate takes only the step at which each line's value stops
/// being zero, from the logarithm of its power `|xr|^0.75`.
///
/// On the granules that the constants were fitted to, the estimate is
/// within a step of the step the loop ends on for 95% of them; at other
/// bitrates and sampling frequencies, within two for almost all. The loop
/// then takes two or three steps instead of the ten or so of a bisection
/// of the whole range of `global_gain`.
fn estimated_gain(lines: &Lines, available_bits: usize) -> u8 {
    StepEstimate::new(lines).gain(available_bits)
}

/// A granule's lines as they are, without scalefactors, and the estimate
/// of the inner loop's first step for them, worked out once for lines that
/// are quantised again with other bits, as at each bitrate a frame is tried
/// at.
pub struct Unscaled {
    pub lines: Lines,
    pub estimate: StepEstimate,
}

impl Unscaled {
    pub fn new(xr: &[f64; GRANULE_LINES]) -> Self {
        let lines = Lines::new(xr);
        let estimate = StepEstimate::new(&lines);
        Self { lines, estimate }
    }
}

/// What [`estimated_gain`] takes of a granule's lines.
pub struct StepEstimate {
    /// The lines whose values stop being zero at each step.
    steps: [u16; 257],
}

impl StepEstimate {
    pub fn new(lines: &Lines) -> Self {
        // The values of a line are not zero up to the step `z` where
        // `powered * 2^(-3/16 (z - 210)) - 0.0946 >= 0.5`. Lines that are zero
        // at every step are left out, and those that are zero at none are
        // counted as if they were from the step after the coarsest.
        let mut steps = [0u16; 257];
        let lowest_nonzero = (0.5f64 + 0.0946).log2();
        for &powered in &lines.powered[..lines.end] {
            if powered == 0.0 {
                continue;
            }
            let z = f64::from(GLOBAL_GAIN_OFFSET)
                + (approximate_log2(powered) - lowest_nonzero) * 16.0 / 3.0;
            if z >= 0.0 {
                steps[z.min(256.0) as usize] += 1;
            }
        }
        Self { steps }
    }

    /// [`estimated_gain`] for the lines of this estimate.
    pub fn gain(&self, available_bits: usize) -> u8 {
        let steps = &self.steps;
        // From the coarsest step down: the values not zero, and the bits of
        // their magnitudes in steps of 3/16 of a bit.
        let (mut values, mut magnitude_steps) = (u32::from(steps[256]), 0u32);
        for global_gain in (0..=u8::MAX).rev() {
            magnitude_steps += values;
            values += u32::from(steps[usize::from(global_gain)]);
            let bits = ESTIMATED_BITS_PER_VALUE * f64::from(values)
                + ESTIMATED_BITS_PER_MAGNITUDE_BIT * 3.0 / 16.0 * f64::from(magnitude_steps);
            if bits > available_bits as f64 {
                return global_gain.saturating_add(1);
            }
        }
        0
    }
}

/// `log2(x)` for positive normal `x`, within 0.09: the bits of `x` taken as
/// a fixed-point number, where the exponent gives the integer part and the
/// mantissa `m` the fraction, approximating `log2(1 + m)` with `m`. It
/// takes no call to the C library, and its error is within half of the
/// 3/16 of a bit between steps.
fn approximate_log2(x: f64) -> f64 {
    const MANTISSA_BITS: u32 = 52;
    const EXPONENT_BIAS: f64 = 1023.0;
    // The bits of a positive `x` are below 2^63, so they convert as an `i64`
    // to the same value as they would as a `u64`, which takes more
    // instructions.
    (x.to_bits() as i64) as f64 / (1u64 << MANTISSA_BITS) as f64 - EXPONENT_BIAS
}

/// Narrows `over`, which does not fit, and the step of `fits`, which does,
/// to adjacent steps and returns the one that fits. Spec gap: this assumes
/// that larger steps never need more bits, which the Huffman tables do not
/// quite guarantee; the search in [`inner_loop`] then ends on a step whose
/// next finer step does not fit.
fn bisect_between(attempt: &impl Fn(u8) -> Option<Step>, mut over: u8, mut fits: Step) -> Step {
    while fits.global_gain - over > 1 {
        let middle = over + (fits.global_gain - over) / 2;
        match attempt(middle) {
            Some(step) => fits = step,
            None => over = middle,
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
    fn approximate_log2_is_within_half_a_step() {
        let mut x = f64::MIN_POSITIVE;
        while x < 1e300 {
            let error = (approximate_log2(x) - x.log2()).abs();
            assert!(error <= 0.09, "{x}: {error}");
            x *= 1.0137;
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
