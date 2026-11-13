//! Quantisation of frequency lines (clause C.1.5.4.4.1) and the inner
//! iteration loop that sets the step size (clause C.1.5.4.4, figure C.9.c).

use super::header::SamplingFrequency;
use super::huffman::MAX_MAGNITUDE;
use super::{GRANULE_LINES, QuantizedGranule, scalefactor_bands};

/// Clause C.1.5.4.1 sets `global_gain = qquant + system_constant` and leaves
/// the constant to the implementation. The decoder requantises with
/// `2^((global_gain - 210) / 4)` (clause 2.4.3.4.7.1), so with `xr` on the
/// decoder's scale, taking `qquant + quantanf = global_gain - 210` in the
/// quantizer of clause C.1.5.4.4.1 makes it the inverse of the decoder's
/// requantizer.
const GLOBAL_GAIN_OFFSET: i32 = 210;

/// Quantises `xr` with step size `global_gain`:
/// `ix(i) = nint((|xr(i)| / 2^((global_gain - 210) / 4))^0.75 - 0.0946)`,
/// carrying the sign of `xr(i)`. Returns `None` if a magnitude would exceed
/// the largest the Huffman tables can code (clause C.1.5.4.4.2).
pub fn quantize(xr: &[f64; GRANULE_LINES], global_gain: u8) -> Option<[i32; GRANULE_LINES]> {
    let step = 2f64.powf(f64::from(i32::from(global_gain) - GLOBAL_GAIN_OFFSET) / 4.0);
    let mut ix = [0; GRANULE_LINES];
    for (ix, &xr) in ix.iter_mut().zip(xr) {
        let magnitude = ((xr.abs() / step).powf(0.75) - 0.0946).round().max(0.0);
        if magnitude > f64::from(MAX_MAGNITUDE) {
            return None;
        }
        *ix = magnitude.copysign(xr) as i32;
    }
    Some(ix)
}

/// The inner iteration loop (clause C.1.5.4.4, figure C.9.c): quantises
/// `xr` with the smallest step size whose Huffman coding fits in
/// `available_bits`.
///
/// The loop raises the step size one `global_gain` at a time until the
/// quantised values are within the range of the tables (clause C.1.5.4.4.2;
/// the decision in figure C.9.c has its "y" and "n" swapped relative to that
/// text) and their bits fit. Clause C.1.5.4.2.1 leaves the starting step,
/// `quantanf`, to the implementation, asking only that the first iteration
/// need more bits than are available and suggesting it be chosen to save
/// iterations; here it is found by bisection, one step below the first fit.
///
/// A granule whose lines are all zero skips the loop (figure C.9.a).
pub fn inner_loop(
    xr: &[f64; GRANULE_LINES],
    available_bits: usize,
    sampling_frequency: SamplingFrequency,
) -> QuantizedGranule {
    if xr.iter().all(|&x| x == 0.0) {
        return QuantizedGranule::new(0, [0; GRANULE_LINES], sampling_frequency);
    }
    let bands = scalefactor_bands::long_bands(sampling_frequency);
    let attempt = |global_gain: u8| {
        let values = quantize(xr, global_gain)?;
        let granule = QuantizedGranule::new(global_gain, values, sampling_frequency);
        let fits = granule.coding.bit_count(&granule.values, bands) <= available_bits;
        fits.then_some(granule)
    };

    // The smallest gain that fits, assuming larger steps never need more bits.
    // At the largest gain every line quantises to zero, which always fits.
    let (mut over, mut fits) = (0u8, u8::MAX);
    if attempt(over).is_some() {
        fits = over;
    }
    while fits - over > 1 {
        let middle = over + (fits - over) / 2;
        if attempt(middle).is_some() {
            fits = middle;
        } else {
            over = middle;
        }
    }

    let mut global_gain = fits.saturating_sub(1);
    loop {
        if let Some(granule) = attempt(global_gain) {
            return granule;
        }
        global_gain += 1;
    }
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
            let granule = inner_loop(&xr, available, fs);
            let bits = granule.coding.bit_count(&granule.values, bands);
            assert!(bits <= available, "{bits} bits for {available}");
            // One step finer either leaves the tables' range or needs too many bits.
            let finer = quantize(&xr, granule.global_gain - 1).map(|values| {
                let finer = QuantizedGranule::new(granule.global_gain - 1, values, fs);
                finer.coding.bit_count(&finer.values, bands)
            });
            assert!(finer.is_none_or(|bits| bits > available), "{available}");
        }
    }

    #[test]
    fn inner_loop_skips_silence() {
        let granule = inner_loop(&[0.0; GRANULE_LINES], 0, SamplingFrequency::Hz32000);
        assert!(granule.values.iter().all(|&v| v == 0));
        assert_eq!(granule.coding.big_values, 0);
    }

    #[test]
    fn inner_loop_fits_even_with_no_bits() {
        let fs = SamplingFrequency::Hz48000;
        let granule = inner_loop(&tone_granule(), 0, fs);
        let bands = scalefactor_bands::long_bands(fs);
        assert_eq!(granule.coding.bit_count(&granule.values, bands), 0);
    }
}
