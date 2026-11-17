// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The Layer III iteration loops (clause C.1.5.4): the outer loop, which
//! shapes the quantisation noise with scalefactors, around the inner loop,
//! which fits the granule to the available bits.

mod inner;
mod outer;
pub mod scfsi;

pub use inner::{Unscaled, inner_loop, quantize, step_size};
pub use outer::{
    Bits, Outcome, Reuse, SAVING_NOISE_DB, Saving, excess, outer_loop, outer_loop_with,
};

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
/// stereo gives its positions in `intensity`. `unscaled` keeps these lines
/// as they are, with the estimate of the inner loop's first step for them,
/// for coding them again with other bits (see [`outer_loop_with`]).
///
/// # Panics
///
/// If the scalefactors that `intensity` marks do not fit in the
/// [`available`](Bits::available) bits, which takes fewer than its
/// [`silent_bits`](IntensityPositions::silent_bits).
#[allow(clippy::too_many_arguments)]
pub fn quantize_granule(
    xr: &[f64; GRANULE_LINES],
    xmin: &[f64],
    block_type: BlockType,
    bits: Bits,
    sampling_frequency: SamplingFrequency,
    reuse: Option<Reuse>,
    intensity: Option<&IntensityPositions>,
    unscaled: Option<&std::cell::OnceCell<Unscaled>>,
) -> Outcome {
    if xr.iter().all(|&x| x == 0.0) {
        let mut granule =
            QuantizedGranule::new(0, [0; GRANULE_LINES], block_type, sampling_frequency);
        if let Some(intensity) = intensity {
            intensity.mark(&mut granule.scalefactors, &granule.values);
            assert!(
                granule.bit_len(sampling_frequency) <= bits.available,
                "the intensity stereo positions do not fit in {} bits",
                bits.available
            );
        }
        return Outcome {
            granule,
            within_allowance: true,
        };
    }
    outer_loop_with(
        xr,
        xmin,
        block_type,
        bits,
        sampling_frequency,
        reuse,
        intensity,
        unscaled,
    )
}
