//! The Layer III iteration loops (clause C.1.5.4): the outer loop, which
//! shapes the quantisation noise with scalefactors, around the inner loop,
//! which fits the granule to the available bits. Long blocks only.

mod inner;
mod outer;

pub use inner::{inner_loop, quantize, step_size};
pub use outer::outer_loop;

use super::header::SamplingFrequency;
use super::scalefactor_bands::LONG_BANDS;
use super::{GRANULE_LINES, QuantizedGranule};

/// The iteration loop of one granule of one channel (figure C.9.a): a granule
/// whose lines are all zero is coded as such, any other goes through the
/// outer loop with the allowed distortion `xmin` of each scalefactor band.
pub fn quantize_granule(
    xr: &[f64; GRANULE_LINES],
    xmin: &[f64; LONG_BANDS],
    available_bits: usize,
    sampling_frequency: SamplingFrequency,
) -> QuantizedGranule {
    if xr.iter().all(|&x| x == 0.0) {
        return QuantizedGranule::new(0, [0; GRANULE_LINES], sampling_frequency);
    }
    outer_loop(xr, xmin, available_bits, sampling_frequency)
}
