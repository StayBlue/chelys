//! Quantisation of frequency lines (clause C.1.5.4.4.1).

use super::GRANULE_LINES;
use super::huffman::MAX_MAGNITUDE;

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
}
