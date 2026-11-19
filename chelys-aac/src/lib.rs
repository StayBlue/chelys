// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! MPEG-2 AAC in the Low Complexity profile, with the configuration that
//! MPEG-4 gives it, implemented from their specifications. The `chelys`
//! crate offers it as `chelys::aac`.
//!
//! Clause and table references refer to ISO/IEC 13818-7:2006, or, where
//! they name it, to ISO/IEC 14496-3:2019.
//!
//! The encoder takes PCM at the twelve sampling frequencies from 8 to
//! 96 kHz, and codes:
//!
//! - each frame's channels in a long window, or in eight short windows
//!   where the psychoacoustic model of Annex C finds an attack, with start
//!   and stop windows between them: the Kaiser-Bessel derived window shape
//!   for long and stop windows, and the sine window for start and short
//!   ones; consecutive short windows of like energy grouped, to share their
//!   scalefactors;
//! - the quantization noise of each scalefactor band shaped to what the
//!   model allows, and each channel's lines fitted to its share of the
//!   frame's bits, by the iteration loops of Annex C;
//! - the quantization noise within each window shaped in time with
//!   temporal noise shaping (clause 14), from the analysis of Annex C;
//! - a channel pair's bands as middle and side (M/S stereo), or in
//!   intensity stereo, where Annex C finds it pays;
//! - the channel configurations 1 to 7 (Table 42): mono, stereo, 3, 4 and
//!   5 channels, and 5.1 and 7.1, whose low frequency enhancement channel
//!   is coded in its lowest lines;
//! - a constant rate, with the bit reservoir (clause 8.2.3.2) lending
//!   frames bits and keeping those they leave, and fill where it is full;
//! - each frame in ADTS, with an optional CRC and the header's
//!   `private_bit`, `original_copy` and `home` as asked, or as its
//!   `raw_data_block()` alone, for a container that carries the
//!   `AudioSpecificConfig()` of ISO/IEC 14496-3 that describes the stream;
//! - bytes of data beside the audio, in data stream elements (clause 8.6).
//!
//! Spectrum clipping (clause C.8.2), which sends part of up to four lines
//! of a long window as pulse data, is implemented but not applied: Annex
//! C says it "may be applied", and on music it saved about a tenth of a
//! bit for each long window, changed grades of PEAQ by 0.001 at most, and
//! made the encoder 47% slower.
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
/// the input spreads into them. Settled by decoding with ffmpeg and FAAD2
/// (black box): the input lands exactly this many samples into their
/// output, in mono and stereo, in ffmpeg's at every sampling frequency and
/// in FAAD2's above 24 kHz; the conformance tests are
/// `ffmpeg_decodes_the_input_a_frame_late_at_every_sample_rate` and
/// `faad_decodes_the_input_a_frame_late_at_every_sample_rate`. At 24 kHz
/// and below, FAAD2 differs: it gives twice the samples, at twice the
/// sampling frequency, every other one of which gives the input 481
/// samples later still.
///
/// Decoding with ffmpeg, and with FAAD2 above 24 kHz (black box), gives
/// each frame's 1024 samples, from the first frame on, and trims none:
/// this many samples, the input, then [`Finished::padding`]; the
/// conformance tests are `ffmpeg_decodes_silence_at_every_sample_rate` and
/// `faad_decodes_silence_at_every_sample_rate`.
pub const DELAY: usize = 1024;
