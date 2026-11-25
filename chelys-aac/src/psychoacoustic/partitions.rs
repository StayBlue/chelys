// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The threshold calculation partitions of the psychoacoustic model
//! (clause C.1.4, step 5), by sampling frequency and FFT: Tables C.1 to
//! C.24, which `chelys_core::psychoacoustic::partitions` transcribes.

pub use chelys_core::psychoacoustic::partitions::Row;
use chelys_core::psychoacoustic::partitions::{self, Fft};

use crate::sampling_frequency::SamplingFrequency;
use crate::scalefactor_bands::Transform;

/// The partitions of the long FFT, whose lines are those of a long window,
/// or of the short FFT, those of a short window, at `sampling_frequency`.
pub fn table(sampling_frequency: SamplingFrequency, transform: Transform) -> &'static [Row] {
    let fft = match transform {
        Transform::Long => Fft::Long,
        Transform::Short => Fft::Short,
    };
    partitions::table(sampling_frequency.hz(), fft).expect("a table for every sampling frequency")
}
