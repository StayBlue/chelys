// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! What chelys's encoders share: the bit writer their bitstreams are
//! written with, the CRC-16 that protects them, the PCM samples they take,
//! where they give the streams they code, the FFT of their psychoacoustic
//! models and filterbanks, and the scale of quality of their variable
//! bitrates. Each codec's crate re-exports what
//! its public API needs.

pub mod bitstream;
pub mod crc;
pub mod fft;
pub mod quality;
pub mod sample;
pub mod sink;

pub use sample::{NonFiniteSample, Sample};
pub use sink::FrameSink;
