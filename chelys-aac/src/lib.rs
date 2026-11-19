// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! MPEG-2 AAC in the Low Complexity profile, with the configuration that
//! MPEG-4 gives it, implemented from their specifications. The `chelys`
//! crate offers it as `chelys::aac`.
//!
//! Clause and table references refer to ISO/IEC 13818-7:2006, or, where
//! they name it, to ISO/IEC 14496-3:2019.
//!
//! The encoder is being built: for now it switches to eight short windows
//! where the psychoacoustic model of Annex C finds an attack, shapes the
//! quantization noise within each window in time with temporal noise
//! shaping, codes a pair's channels as middle and side or in intensity
//! stereo where Annex C finds it pays, and fits each channel's lines to
//! its share of the frame's bits with the iteration loops of Annex C, which
//! shape the quantization noise of each scalefactor band to what the model
//! allows.
//!
//! ```
//! use chelys_aac::{Config, Encoder, Rate};
//!
//! let config = Config::new(44_100, 2, Rate::Constant { bits_per_second: 128_000 });
//! let mut encoder = Encoder::new(config)?;
//! // A second of a 440 Hz tone in both channels, interleaved.
//! let pcm: Vec<f32> = (0..44_100)
//!     .map(|t| 0.5 * (std::f32::consts::TAU * 440.0 * t as f32 / 44_100.0).sin())
//!     .flat_map(|x| [x, x])
//!     .collect();
//! let mut aac = Vec::new(); // in ADTS
//! encoder.encode_interleaved(&pcm, &mut aac);
//! encoder.finish(&mut aac);
//! # Ok::<(), chelys_aac::ConfigError>(())
//! ```

mod adts;
mod audio_specific_config;
mod channel;
mod channel_configuration;
mod config;
mod encoder;
mod filterbank;
mod loops;
mod noiseless;
mod psychoacoustic;
mod raw_data_block;
mod reservoir;
mod sampling_frequency;
mod scalefactor_bands;
mod stereo;
mod stream;
mod tns;
mod window;

pub use chelys_core::Sample;
pub use config::{Config, ConfigError, Framing, Rate, SAMPLE_RATES};
pub use encoder::{Encoder, Finished};
pub use stream::FrameSink;

/// The encoder's building blocks, for chelys's own conformance tests: the
/// bitstream's fields, and ways to encode a stream that the public API
/// does not offer. They follow the code, not a stable API, and may change
/// in any release.
#[cfg(feature = "unstable")]
pub mod internals {
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
        adts,
        audio_specific_config,
        channel,
        channel_configuration,
        filterbank,
        loops,
        noiseless,
        psychoacoustic,
        raw_data_block,
        reservoir,
        sampling_frequency,
        scalefactor_bands,
        stereo,
        tns,
        window,
    );
}

/// The samples by which a decoder's output lags the encoder's input: one
/// frame's. A frame's window spans the 1024 samples it codes and the 1024
/// before them (clause C.3.1.1), and a decoder gives out its first half
/// overlapped and added with the second half of the frame before
/// (clause 15.3.3): the samples a frame decodes to are those of the block
/// before its own. An [`Encoder`] ends the stream at least this far past
/// the input, so that the decoded output holds all of it.
///
/// An [`Encoder`] gives each frame only once the block after it has
/// arrived, from which the frame's windows are chosen; that delays the
/// frames it gives, not where the input lands in the decoded output.
///
/// Spec gap: the standard does not say what comes before the first block
/// of input. It is taken as silence, so that the first frame decodes that
/// silence and the input follows it whole. These samples are to be
/// discarded: they decode to near silence only, as the quantization of
/// the input spreads into them. Settled by decoding with ffmpeg
/// (black box): the input lands exactly this many samples into its
/// output, at every sampling frequency, in mono and stereo; the
/// conformance test is
/// `ffmpeg_decodes_the_input_a_frame_late_at_every_sample_rate`.
///
/// Decoding with ffmpeg (black box) gives each frame's 1024 samples, from
/// the first frame on, and trims none: this many samples, the input, then
/// [`Finished::padding`]; the conformance test is
/// `ffmpeg_decodes_silence_at_every_sample_rate`.
pub const DELAY: usize = 1024;
