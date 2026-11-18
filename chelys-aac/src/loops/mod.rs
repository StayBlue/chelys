// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The AAC iteration loops (clause C.7): the inner loop, which fits a
//! channel's lines to the bits available, and the outer loop around it,
//! which shapes the quantization noise with scalefactors.

pub mod inner;
pub mod outer;

use inner::{Counting, Lines};

use crate::channel::{CodedChannel, FRAME_LINES};
use crate::sampling_frequency::SamplingFrequency;
use crate::scalefactor_bands::MAX_BANDS;
use crate::window::{SHORT_WINDOWS, Windows};

/// The offset in dB that every allowed distortion `xmin` is raised by
/// before the loops (clause C.7.4.4).
///
/// Spec gap: clause C.7.4.4 says the outer loop's procedure "is only valid
/// in the case the number of available bits is equal to the number of
/// required bits corresponding to the perceptual entropy", and that
/// otherwise a constant noise to mask ratio over the bands "can be
/// realised by applying an offset to the target allowed distortion
/// xmin(sb), that is the same for all scale factor bands, prior to
/// starting the loops module", without saying how large. The perceptual
/// entropy cannot size it here: as printed, it is negative in most frames
/// of music, whose partitions above about 16 kHz lie below the thresholds
/// in quiet (see [`Model::analyze`](crate::psychoacoustic::Model::analyze)).
/// Without an offset, a quarter of the frames of pop music at 128 kbit/s
/// in stereo ended with every band amplified after some 30 iterations,
/// with a step some 40 dB coarser than the first and most bands over their
/// allowance. A fixed offset is taken, the smaller of the two that grade
/// best with PEAQ, with the outer loop keeping its iteration of least
/// excess (see [`outer::outer_loop`] for the grades): 3 dB grades 0.024
/// below 6 dB at 128 kbit/s and 0.018 above it at 192 kbit/s, and as
/// 6 dB at 64 kbit/s.
pub const XMIN_OFFSET_DB: f64 = 3.0;

/// The iteration loop of one channel (figure C.6): a channel whose lines
/// are all zero is coded as such; any other is quantized by the outer loop
/// to fit in `available_bits`, the bits of its
/// `individual_channel_stream()`, with the noise of each scalefactor band
/// of each window group shaped to its allowed distortion `xmin`, raised by
/// [`XMIN_OFFSET_DB`] (see [`outer::outer_loop`]).
///
/// The bits the channel leaves unused (figure C.6) are not counted here:
/// the frame's length, spent from the bit reservoir, leaves them there.
///
/// # Panics
///
/// If not even a channel without lines fits in `available_bits`: a
/// caller's error.
pub fn quantize_channel(
    spectrum: &[f64; FRAME_LINES],
    windows: Windows,
    sampling_frequency: SamplingFrequency,
    available_bits: u32,
    xmin: &[[f64; MAX_BANDS]],
) -> CodedChannel {
    if spectrum.iter().all(|&x| x == 0.0) {
        return CodedChannel::silent(windows, sampling_frequency);
    }
    let counting = Counting::new(
        Lines::new(spectrum),
        windows,
        sampling_frequency,
        available_bits,
    );
    let offset = 10f64.powf(XMIN_OFFSET_DB / 10.0);
    let mut target = [[0.0; MAX_BANDS]; SHORT_WINDOWS];
    for (target, xmin) in target.iter_mut().zip(xmin) {
        *target = xmin.map(|xmin| xmin * offset);
    }
    outer::outer_loop(counting, &target[..xmin.len()])
}
