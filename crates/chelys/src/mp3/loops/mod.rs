//! The Layer III iteration loops (clause C.1.5.4): the outer loop, which
//! shapes the quantisation noise with scalefactors, around the inner loop,
//! which fits the granule to the available bits.

mod inner;
mod outer;
pub mod scfsi;

pub use inner::{inner_loop, quantize, step_size};
pub use outer::{Outcome, Reuse, outer_loop};

use super::header::SamplingFrequency;
use super::stereo::IntensityPositions;
use super::{BlockType, GRANULE_LINES, QuantizedGranule};

/// The iteration loop of one granule of one channel (figure C.9.a): a granule
/// whose lines are all zero is coded as such, any other goes through the
/// outer loop with the allowed distortion `xmin` of each scalefactor. The
/// lines and `xmin` are in the order they are sent (see
/// [`coding_bands`](super::scalefactor_bands::coding_bands)). In a frame's
/// second granule, `reuse` gives the scfsi bands that take over the first
/// granule's scalefactors. The right channel of a frame coded in intensity
/// stereo gives its positions in `intensity`.
///
/// # Panics
///
/// If the scalefactors that `intensity` marks do not fit in
/// `available_bits`, which takes fewer than [`MAX_POSITION_BITS`](super::stereo::MAX_POSITION_BITS).
pub fn quantize_granule(
    xr: &[f64; GRANULE_LINES],
    xmin: &[f64],
    block_type: BlockType,
    available_bits: usize,
    sampling_frequency: SamplingFrequency,
    reuse: Option<Reuse>,
    intensity: Option<&IntensityPositions>,
) -> Outcome {
    if xr.iter().all(|&x| x == 0.0) {
        let mut granule =
            QuantizedGranule::new(0, [0; GRANULE_LINES], block_type, sampling_frequency);
        if let Some(intensity) = intensity {
            intensity.mark(
                &mut granule.scalefactors,
                &granule.values,
                sampling_frequency,
            );
            assert!(
                granule.bit_len(sampling_frequency) <= available_bits,
                "the intensity stereo positions do not fit in {available_bits} bits"
            );
        }
        return Outcome {
            granule,
            within_allowance: true,
        };
    }
    outer_loop(
        xr,
        xmin,
        block_type,
        available_bits,
        sampling_frequency,
        reuse,
        intensity,
    )
}
