// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Ogg streams (RFC 3533, with its verified errata) of chelys's Vorbis I
//! ([`vorbis`]), laid out as Appendix A.2 of the Vorbis I specification
//! asks. The `chelys` crate offers it as `chelys::ogg`.
//!
//! Clause references refer to RFC 3533, or where they name it, to the
//! Vorbis I specification of the Xiph.Org Foundation, dated July 4, 2020.
//!
//! A stream is one logical stream of Vorbis, unmultiplexed, as Appendix
//! A.1.1 asks of an Ogg Vorbis stream. It is written as the packets come,
//! from its start to its end, so that it can be written where nothing can
//! be sought, as a pipe, and played as it is written. Streams of different
//! serial numbers can be chained, one written after another (clause 4).
//!
//! ```
//! use chelys_ogg::vorbis::Encoder;
//! use chelys_vorbis::{Config, Rate};
//!
//! let config = Config::new(48_000, 1, Rate::Quality { level: 80 });
//! let mut encoder = Encoder::new(config, 1)?;
//! let mut ogg = encoder.header_pages().to_vec();
//! let mut pages = 0;
//! let mut out = |page: &[u8]| {
//!     ogg.extend_from_slice(page);
//!     pages += 1;
//! };
//! encoder.encode_interleaved(&[0.0f32; 48_000], &mut out)?;
//! encoder.finish(&mut out);
//! assert!(pages > 0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod page;
pub mod vorbis;

/// Where an encoder gives the pages it writes, one call each, in their
/// order, after its header pages.
pub trait PageSink {
    /// Takes a page: its header and its segments.
    fn page(&mut self, page: &[u8]);
}

/// Appends each page, as to the stream after its header pages.
impl PageSink for Vec<u8> {
    fn page(&mut self, page: &[u8]) {
        self.extend_from_slice(page);
    }
}

/// Calls the closure with each page, as to send each apart.
impl<F: FnMut(&[u8])> PageSink for F {
    fn page(&mut self, page: &[u8]) {
        self(page);
    }
}

/// The page writer, for chelys's own conformance tests. It follows the
/// code, not a stable API, and may change in any release.
#[cfg(feature = "unstable")]
pub mod internals {
    pub use crate::page::{Limits, MAX_SEGMENTS, NO_PACKET_ENDS, Pager, crc};

    /// A [`vorbis::Encoder`](crate::vorbis::Encoder) whose pages of audio
    /// are ended at `limits`.
    ///
    /// # Errors
    ///
    /// As [`vorbis::Encoder::new`](crate::vorbis::Encoder::new).
    pub fn vorbis_encoder(
        config: chelys_vorbis::Config,
        serial_number: u32,
        limits: Limits,
    ) -> Result<crate::vorbis::Encoder, chelys_vorbis::ConfigError> {
        crate::vorbis::Encoder::with_limits(config, serial_number, limits)
    }
}
