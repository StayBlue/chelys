// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! How a [`FrameEncoder`](super::frame_encoder::FrameEncoder) analyses
//! and quantises granules: by the encoding process of Annex C, or, for
//! chelys's own tests, with one `global_gain` for every granule.

#[cfg(feature = "unstable")]
mod fixed_gain;
mod model;

#[cfg(feature = "unstable")]
pub use fixed_gain::{encode_pcm_with_global_gain, encode_with_block_types};
pub(super) use model::ModelCoding;

use crate::granule::{BlockType, GRANULE_LINES, QuantizedGranule};
use crate::reservoir::Reservoir;

/// One granule of one channel, analysed and filtered, before quantisation.
pub(super) struct Filtered<A> {
    pub(super) block_type: BlockType,
    pub(super) analysis: A,
    /// The lines in the order they are sent.
    pub(super) xr: [f64; GRANULE_LINES],
}

/// A frame's granules, quantised, with one entry per channel each, and the
/// joint stereo coding they use.
pub(super) struct CodedFrame {
    pub(super) granules: Vec<Vec<QuantizedGranule>>,
    pub(super) ms_stereo: bool,
    pub(super) intensity_stereo: bool,
    /// The largest [`loops::excess`](crate::loops::excess) of its granules: at most 1 if all are
    /// within their allowed distortion.
    pub(super) excess: f64,
}

/// How [`FrameEncoder`](super::frame_encoder::FrameEncoder) analyses and quantises granules.
pub(super) trait Coding {
    /// What the analysis of a granule gives besides its block type.
    type Analysis;

    /// Given each channel's PCM granule by granule, starting with the first,
    /// returns the block type and analysis of the granule before the one it
    /// is given: `None` only for the first. It is given each granule two
    /// granules ahead of the filterbank, which needs the block types of the
    /// granule it filters and of the one after.
    fn look_ahead(
        &mut self,
        channel: usize,
        pcm: &[f64; GRANULE_LINES],
    ) -> Option<(BlockType, Self::Analysis)>;

    /// Readies a frame's granules, indexed `[gr][ch]`, for quantisation,
    /// once however many times the frame is quantised.
    fn prepare(&mut self, _frame: &mut [Vec<Filtered<Self::Analysis>>]) {}

    /// Quantises a frame's granules, indexed `[gr][ch]`, given the share of
    /// the frame's main data of each granule of each channel, accounting in
    /// the reservoir for the bits each uses. Intensity stereo is tried only
    /// if `intensity` is set and the mode allows it. If `only_within` is
    /// set, the frame is wanted only if its noise is within its allowance
    /// ([`CodedFrame::excess`] at most 1), and `None` may be returned as
    /// soon as it is known not to be.
    fn quantize(
        &mut self,
        frame: &[Vec<Filtered<Self::Analysis>>],
        mean_bits: usize,
        reservoir: &mut Reservoir,
        intensity: bool,
        only_within: bool,
    ) -> Option<CodedFrame>;
}
