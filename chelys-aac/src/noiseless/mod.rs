// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Noiseless coding (clause 9) of an individual channel stream's quantized
//! lines: their division into sections of scalefactor bands, each coded
//! with one codebook (`section_data()`, Table 17), the scalefactors of the
//! bands (`scale_factor_data()`, Table 18) and the lines themselves
//! (`spectral_data()`, Table 20), with the Huffman codebooks of Annex A,
//! and the lines that the pulse escape restores (`pulse_data()`, Table 21);
//! and the choice of sections and codebooks as Annex C describes it
//! (clause C.8).
//!
//! Each part works on one window group, whose fields an
//! `individual_channel_stream()` sends group by group: a long window, or a
//! group of short windows of an EIGHT_SHORT_SEQUENCE. The choice counts
//! the bits of each coding without writing it, for the rate control loops
//! to count many.
//!
//! Spectrum clipping (clause C.8.2), which Annex C says "may be applied"
//! to a long window's lines first, sends some of them as pulses where that
//! takes fewer bits.

pub mod clipping;
pub mod codebooks;
pub mod pulses;
pub mod scalefactors;
pub mod sections;
pub mod spectrum;
mod tables;

/// For chelys's tests, and through `internals` for its conformance tests.
#[cfg(any(test, feature = "unstable"))]
pub mod decoder;
