// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Vorbis I, implemented from the "Vorbis I specification" of the Xiph.Org
//! Foundation, dated July 4, 2020. The `chelys` crate offers it as
//! `chelys::vorbis`.
//!
//! Clause references refer to that specification. It defines only the
//! bitstream and its decoding: the encoder, and the codebooks, floors,
//! residues, mappings and modes of the setup header it writes, are
//! chelys's own.
//!
//! The encoder is being built: for now each packet codes silence, whatever
//! the input, in a stream that is valid as it will be.
//!
//! The encoder gives Vorbis packets, not a container: its three
//! [`Headers`], then each audio packet with its granule position, from
//! which a container such as Ogg (Appendix A) times it and trims the end of
//! the stream.
//!
//! ```
//! use chelys_vorbis::{Config, Encoder, Packet};
//!
//! let mut encoder = Encoder::new(Config::new(44_100, 2))?;
//! let headers = encoder.headers();
//! let pcm = vec![0.0f32; 2 * 44_100]; // a second of silence, interleaved
//! let mut packets = Vec::new();
//! let mut out = |packet: Packet<'_>| packets.push((packet.data.to_vec(), packet.granule_position));
//! encoder.encode_interleaved(&pcm, &mut out)?;
//! let finished = encoder.finish(&mut out);
//! assert_eq!(packets.last().unwrap().1, 44_100);
//! assert_eq!(finished.samples, 44_100);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod audio;
mod bitpack;
mod codebook;
mod config;
mod encoder;
mod header;
mod packet;
mod setup;

pub use chelys_core::{NonFiniteSample, Sample};
pub use config::{Config, ConfigError, MAX_CHANNELS};
pub use encoder::{Encoder, Finished, Headers};
pub use header::VENDOR;
pub use packet::{Packet, PacketSink};

/// The encoder's building blocks, for chelys's own conformance tests: the
/// bitstream's fields. They follow the code, not a stable API, and may
/// change in any release.
#[cfg(feature = "unstable")]
pub mod internals {
    /// Wraps each module in one that re-exports its public items, as a
    /// private module cannot itself be re-exported.
    macro_rules! reexport {
        ($($module:ident),* $(,)?) => {
            $(pub mod $module {
                pub use crate::$module::*;
            })*
        };
    }
    reexport!(audio, bitpack, codebook, header, setup);
}
