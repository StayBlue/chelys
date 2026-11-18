// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! MPEG-1 Audio Layer III, its extension to lower sampling frequencies in
//! ISO/IEC 13818-3, and MPEG-2.5, implemented from their specifications.
//! The `chelys` crate offers it as `chelys::mp3`.
//!
//! Clause, table and figure references refer to ISO/IEC 11172-3:1993 (first
//! edition), or, where they name it, to ISO/IEC 13818-3:1998 (second
//! edition).
//!
//! ```
//! use chelys_mp3::{ChannelMode, Config, Encoder, Rate};
//!
//! let config = Config::new(44_100, ChannelMode::Stereo, Rate::Constant { kbps: 128 });
//! let mut encoder = Encoder::new(config)?;
//! let pcm = vec![0.0f32; 2 * 44_100]; // a second of silence, interleaved
//! let mut mp3 = Vec::new();
//! encoder.encode_interleaved(&pcm, &mut mp3);
//! encoder.finish(&mut mp3);
//! # Ok::<(), chelys_mp3::ConfigError>(())
//! ```

mod ancillary;
mod bitstream;
mod config;
mod crc;
mod encoder;
mod filterbank;
mod granule;
mod header;
mod huffman;
mod info_tag;
mod loops;
mod psychoacoustic;
mod rate;
mod reservoir;
mod scalefactor_bands;
mod scalefactors;
mod side_info;
mod stereo;
mod stream;

pub use config::{ChannelMode, Config, ConfigError, Rate, SAMPLE_RATES, bitrates};
pub use encoder::{Encoder, Finished, Sample};

/// The encoder's building blocks, for chelys's own conformance tests: the
/// stages of Annex C, the bitstream's fields and tables, and
/// ways to encode a stream that the public API does not offer, such as
/// from quantised granules. They follow the code, not a stable API, and
/// may change in any release.
#[cfg(feature = "unstable")]
pub mod internals {
    pub use super::granule::*;
    pub use super::stream::*;

    /// Wraps each module in one that re-exports its public items, as a
    /// private module cannot itself be re-exported.
    macro_rules! reexport {
        ($($module:ident),* $(,)?) => {
            $(pub mod $module {
                pub use crate::$module::*;
            })*
        };
    }
    reexport!(
        filterbank,
        header,
        huffman,
        info_tag,
        loops,
        psychoacoustic,
        rate,
        reservoir,
        scalefactor_bands,
        scalefactors,
        side_info,
        stereo,
    );

    pub mod bitstream {
        pub use crate::bitstream::*;
    }
}

/// The samples by which a decoder's output lags the encoder's input: the
/// delays of the analysis and synthesis polyphase filterbanks and of the
/// overlapping blocks of the hybrid filterbank, together. An [`Encoder`]
/// ends the stream at least this far past the input, so that the decoded
/// output holds all of it.
///
/// A decoder that does not read the Xing/Info tag gives the input
/// [`Finished::samples`] long after this many samples, and after the
/// samples of the tag's frame if the stream has one. The tag itself gives
/// a decoder the delay less what the decoder adds.
///
/// Spec gap: the standard does not give the delay. Settled by decoding
/// with mpg123 and LAME (black box): given an encoder delay of this less
/// their own (see `info_tag`), both give the input sample for sample; the
/// conformance test is `decoders_read_the_tag_at_every_sampling_frequency`.
/// ffmpeg's output, untrimmed, lags the input by as much.
pub const DELAY: usize = 1057;
