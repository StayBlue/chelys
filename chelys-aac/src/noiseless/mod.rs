// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Noiseless coding (clause 9) of an individual channel stream's quantized
//! lines: their division into sections of scalefactor bands, each coded
//! with one codebook (`section_data()`, Table 17), the scalefactors of the
//! bands (`scale_factor_data()`, Table 18) and the lines themselves
//! (`spectral_data()`, Table 20), with the Huffman codebooks of Annex A;
//! and the choice of sections and codebooks as Annex C describes it
//! (clause C.8).
//!
//! Each part works on one window group, whose fields an
//! `individual_channel_stream()` sends group by group: a long window, or a
//! group of short windows of an EIGHT_SHORT_SEQUENCE. The choice counts
//! the bits of each coding without writing it, for the rate control loops
//! to count many.
//!
//! Spectrum clipping (clause C.8.2), the pulse escape that Annex C says
//! "may be applied", is not.

pub mod codebooks;
mod tables;
