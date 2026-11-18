// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! What chelys's encoders share: the bit writer their bitstreams are
//! written with, the PCM samples they take, and the FFT of their
//! psychoacoustic models and filterbanks. Each codec's crate re-exports
//! what its public API needs.

pub mod bitstream;
pub mod fft;
pub mod sample;

pub use sample::Sample;
