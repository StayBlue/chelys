// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Joint coding of a channel pair (clause 12): M/S stereo, sent as the
//! pair's `ms_mask_present` and `ms_used` flags, and intensity stereo, sent
//! in the right channel's sections and scalefactors.

pub mod intensity;
pub mod middle_side;
