// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! What chelys's encoders share: the bit writer their bitstreams are
//! written with, the CRC-16 that protects them, the PCM samples they take,
//! where they give the streams they code, the FFT of their psychoacoustic
//! models and filterbanks, the MDCT of their filterbanks, the partitions and
//! spreading function of their psychoacoustic models, the scale of
//! quality of their variable bitrates, and the steering of their mean
//! rates. Each codec's crate re-exports what its public API needs.

pub mod bitstream;
pub mod crc;
pub mod fft;
pub mod mdct;
pub mod psychoacoustic;
pub mod quality;
pub mod sample;
pub mod sink;
pub mod steering;

pub use sample::{NonFiniteSample, Sample};
pub use sink::FrameSink;
