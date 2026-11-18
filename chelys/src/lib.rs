// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Research audio encoders implemented from their specifications, one
//! crate each, offered here as a module each behind a feature of the same
//! name.
//!
//! ```
//! # #[cfg(feature = "mp3")] {
//! use chelys::mp3::{ChannelMode, Config, Encoder, Rate};
//!
//! let config = Config::new(44_100, ChannelMode::Mono, Rate::Constant { kbps: 64 });
//! let mut encoder = Encoder::new(config).unwrap();
//! let mut mp3 = Vec::new();
//! encoder.encode_interleaved(&[0.0f32; 44_100], &mut mp3);
//! encoder.finish(&mut mp3);
//! # }
//! ```

#[cfg(feature = "aac")]
#[doc(inline)]
pub use chelys_aac as aac;
#[cfg(feature = "mp3")]
#[doc(inline)]
pub use chelys_mp3 as mp3;
