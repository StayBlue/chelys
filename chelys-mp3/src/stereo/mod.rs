// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Joint stereo for Layer III (Annex G.2): MS stereo, and intensity stereo.
//! The decoder's inverse is clause 2.4.3.4.9, with the intensity stereo of
//! the lower sampling frequencies in clause 2.4.3.2 of ISO/IEC 13818-3.

pub mod intensity;
pub mod middle_side;
