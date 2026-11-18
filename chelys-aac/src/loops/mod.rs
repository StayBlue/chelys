// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The AAC iteration loops (clause C.7): the inner loop, which fits a
//! channel's lines to the bits available, and, to come, the outer loop
//! around it, which shapes the quantization noise with scalefactors.

mod inner;

use inner::inner_loop;

use crate::channel::{CodedChannel, FRAME_LINES};
use crate::sampling_frequency::SamplingFrequency;
use crate::window::Windows;

/// The iteration loop of one channel (figure C.6): a channel whose lines
/// are all zero is coded as such; any other is quantized to fit in
/// `available_bits`, the bits of its `individual_channel_stream()`.
///
/// The outer loop (clause C.7.4.1) is not run yet: it takes the allowed
/// distortion of each band, `xmin`, from the psychoacoustic model (clause
/// C.1), which the encoder does not compute yet. The inner loop runs alone,
/// with every scalefactor 0, as clause C.7.2.1 resets them.
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
) -> CodedChannel {
    if spectrum.iter().all(|&x| x == 0.0) {
        return CodedChannel::silent(windows, sampling_frequency);
    }
    inner_loop(spectrum, windows, sampling_frequency, available_bits)
}
