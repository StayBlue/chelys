// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The AAC iteration loops (clause C.7): the inner loop, which fits a
//! channel's lines to the bits available, and the outer loop around it,
//! which shapes the quantization noise with scalefactors.

pub mod inner;
pub mod outer;

use inner::{Counting, Lines};

use crate::channel::{CodedChannel, FRAME_LINES};
use crate::sampling_frequency::SamplingFrequency;
use crate::scalefactor_bands::MAX_BANDS;
use crate::stereo::intensity::Intensity;
use crate::tns::Tns;
use crate::window::Windows;

/// The offset in dB that every allowed distortion `xmin` is raised by
/// before the loops (clause C.7.4.4).
///
/// Spec gap: clause C.7.4.4 says the outer loop's procedure "is only valid
/// in the case the number of available bits is equal to the number of
/// required bits corresponding to the perceptual entropy", and that
/// otherwise a constant noise to mask ratio over the bands "can be
/// realised by applying an offset to the target allowed distortion
/// xmin(sb), that is the same for all scale factor bands, prior to
/// starting the loops module", without saying how large. The perceptual
/// entropy cannot size it here: as printed, it is negative in most frames
/// of music, whose partitions above about 16 kHz lie below the thresholds
/// in quiet (see [`Model::analyze`](crate::psychoacoustic::Model::analyze)).
/// Without an offset, a quarter of the frames of pop music at 128 kbit/s
/// in stereo ended with every band amplified after some 30 iterations,
/// with a step some 40 dB coarser than the first and most bands over their
/// allowance. A fixed offset is taken, the smaller of the two that grade
/// best with PEAQ, with the outer loop keeping its iteration of least
/// excess (see [`outer::outer_loop`] for the grades): 3 dB grades 0.024
/// below 6 dB at 128 kbit/s and 0.018 above it at 192 kbit/s, and as
/// 6 dB at 64 kbit/s.
pub const XMIN_OFFSET_DB: f64 = 3.0;

/// Raises each allowed distortion `xmin` by [`XMIN_OFFSET_DB`], as the
/// loops take it.
pub fn raise_xmin(xmin: &mut [[f64; MAX_BANDS]]) {
    let offset = 10f64.powf(XMIN_OFFSET_DB / 10.0);
    for xmin in xmin.iter_mut().flatten() {
        *xmin *= offset;
    }
}

/// How the loops are to code one channel: everything they take but its
/// lines and their allowed distortion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelCoding<'a> {
    pub windows: Windows,
    pub sampling_frequency: SamplingFrequency,
    /// The bits the channel's `individual_channel_stream()` may take.
    pub available_bits: u32,
    /// Whether the channel shares the `ics_info()` of its pair, which its
    /// `individual_channel_stream()` then does not send.
    pub common_window: bool,
    /// The fewest bands the channel's groups span: where it shares its
    /// pair's window, the `max_sfb` of the channel of the pair quantized
    /// before it, as both channels send one `max_sfb`.
    pub least_max_sfb: usize,
    /// For each `max_sfb` the channel may send, the bits its coding adds to
    /// its element besides its own `individual_channel_stream()`: where it
    /// shares its pair's window, those that extend the channel quantized
    /// before it to that `max_sfb`, and the pair's `ms_used` flags. They
    /// are counted with the channel's bits.
    pub element_bits: [u32; MAX_BANDS + 1],
    /// For the right channel of a pair that shares its window, the bands it
    /// codes in intensity stereo, whose lines are to be zero.
    pub intensity: Option<&'a Intensity>,
    /// The filters that shaped the channel's lines, which it sends.
    pub tns: &'a Tns,
}

impl ChannelCoding<'_> {
    /// The coding of a channel of a `single_channel_element()`, or of a
    /// pair that does not share its window, in `available_bits`, without
    /// temporal noise shaping.
    #[cfg(test)]
    pub fn single(
        windows: Windows,
        sampling_frequency: SamplingFrequency,
        available_bits: u32,
    ) -> Self {
        Self {
            windows,
            sampling_frequency,
            available_bits,
            common_window: false,
            least_max_sfb: 0,
            element_bits: [0; MAX_BANDS + 1],
            intensity: None,
            tns: &Tns::NONE,
        }
    }
}

/// The iteration loop of one channel (figure C.6): quantizes the lines
/// `spectrum` by the outer loop to fit as `coding` asks, with the noise of
/// each scalefactor band of each window group, each line's error weighed
/// by `weights` (see [`Lines::weighted`]), shaped to its allowed
/// distortion `xmin`, raised by [`XMIN_OFFSET_DB`] (see [`raise_xmin`] and
/// [`outer::outer_loop`]). Gives the channel and how far its noise exceeds
/// that allowance (see [`outer::excess`]).
///
/// The bits the channel leaves unused (figure C.6) are not counted here:
/// the frame's length, spent from the bit reservoir, leaves them there.
///
/// # Panics
///
/// If not even a channel without lines fits in the available bits: a
/// caller's error.
pub fn quantize(
    spectrum: &[f64; FRAME_LINES],
    weights: &[f64; FRAME_LINES],
    xmin: &[[f64; MAX_BANDS]],
    coding: ChannelCoding<'_>,
) -> (CodedChannel, f64) {
    let counting = Counting::new(Lines::weighted(spectrum, weights), coding);
    outer::outer_loop(counting, xmin)
}

/// [`quantize`] for a `single_channel_element()`'s channel, without its
/// excess.
pub fn quantize_channel(
    spectrum: &[f64; FRAME_LINES],
    weights: &[f64; FRAME_LINES],
    xmin: &[[f64; MAX_BANDS]],
    coding: ChannelCoding<'_>,
) -> CodedChannel {
    quantize(spectrum, weights, xmin, coding).0
}
