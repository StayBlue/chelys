// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! What chelys's encoders share: the bit writer their bitstreams are
//! written with, and the PCM samples they take. Each codec's crate
//! re-exports what its public API needs.

pub mod bitstream;
pub mod sample;

pub use sample::Sample;
