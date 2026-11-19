// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Intensity stereo (clause 12.2): the bands of a pair that shares its
//! window coded as one signal in its left channel, which the right channel
//! takes scaled by its intensity stereo position.

use crate::channel::FRAME_LINES;
use crate::noiseless::codebooks::Codebook;
use crate::noiseless::sections::IntensityBands;
use crate::sampling_frequency::SamplingFrequency;
use crate::scalefactor_bands::{MAX_BANDS, Transform, swb_offsets};
use crate::window::{SHORT_WINDOWS, Windows};

/// The bands of a pair's window groups coded in intensity stereo, each
/// with its codebook and its intensity stereo position, `is_position`
/// (clause 12.2.3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Intensity {
    pub bands: [IntensityBands; SHORT_WINDOWS],
    pub positions: [[i8; MAX_BANDS]; SHORT_WINDOWS],
}

impl Intensity {
    /// No band coded in intensity stereo.
    pub const NONE: Self = Self {
        bands: [IntensityBands::NONE; SHORT_WINDOWS],
        positions: [[0; MAX_BANDS]; SHORT_WINDOWS],
    };

    /// The bands of each group below `max_sfb`, with their positions,
    /// which, all within [`MAX_POSITION`] of 0, need no change to stay
    /// within reach of each other.
    pub fn below(&self, max_sfb: usize) -> Self {
        Self {
            bands: self.bands.map(|bands| bands.below(max_sfb)),
            positions: self.positions,
        }
    }
}

/// The largest intensity stereo position sent, either way: one of 30 has
/// the right channel 45 dB below the left (see [`code`]).
pub const MAX_POSITION: i8 = 30;

/// The lower border frequency `f0` above which Annex C codes a pair in
/// intensity stereo: "An average value of f0 = 6 kHz is appropriate for
/// most types of signals" (clause C.6.2).
pub const BORDER_HZ: f64 = 6000.0;

/// The first scalefactor band of a window of `transform` at
/// `sampling_frequency` that lies above `border` Hz: the first whose lowest
/// line's frequency is `border` or more.
pub fn first_band_above(
    sampling_frequency: SamplingFrequency,
    transform: Transform,
    border: f64,
) -> usize {
    let offsets = swb_offsets(sampling_frequency, transform);
    let line_hz = f64::from(sampling_frequency.hz()) / (2 * transform.lines()) as f64;
    (offsets.iter())
        .position(|&offset| f64::from(offset) * line_hz >= border)
        .unwrap_or(offsets.len())
        .min(offsets.len() - 1)
}

/// Codes the bands of each window group of `windows` from `first` up in
/// intensity stereo as clause C.6.2 describes, in the pair's lines `left`
/// and `right`, and the allowed distortion of their bands, `xmin`: puts the
/// intensity signal in place of the left channel's lines and zeros in place
/// of the right channel's, and gives the bands with their codebooks and
/// intensity stereo positions.
///
/// For each band of a group, the energies of the left and right channels'
/// lines, `El` and `Er`, and of their sum, `Es`, give the band's position
/// `is_position = NINT(2 log2(El / Er))` and its intensity signal,
/// `speci(i) = (specl(i) + specr(i)) sqrt(El / Es)`, which carries the left
/// channel's energy; the decoder scales it by `0.5^(0.25 is_position)`,
/// about `sqrt(Er / El)`, for the right channel (clause 12.2.3).
///
/// Spec reading: this deliberately departs from clause C.6.2, which sets
/// `INTENSITY_HCB`, in phase, for every band, where the band's channels are
/// opposed, the sum of their lines' products negative: there their sum
/// cancels, and its energy `Es` can be all but 0, which would have the
/// intensity signal scale noise up to the left channel's energy and the
/// right channel take it in phase with the left. Such a band is coded with
/// `INTENSITY_HCB2`, out of phase, which the decoder takes with the
/// opposite sign (clause 12.2.3), and its intensity signal is the
/// difference of the channels' lines, `(specl(i) - specr(i)) sqrt(El /
/// Ed)`, with `Ed` the difference's energy.
///
/// Spec reading: this deliberately departs from clause C.6.2 too where
/// either channel is silent in a band, where `log2(El / Er)` is not
/// defined. Where the left channel is, the intensity signal, which carries
/// its energy, would be silent, and so would the right channel's band;
/// where the right channel is, its band is better sent as no band than
/// with a position. Such a band is not coded in intensity stereo: each
/// channel codes its own lines there, if it has any.
///
/// Spec gap: the positions are sent as differences that the scalefactor
/// codebook codes from -60 to 60 (Table A.1), from 0 for the first (clause
/// 12.2.3), which do not reach every ratio of the channels' energies. A
/// band whose position would lie beyond [`MAX_POSITION`] either way, a
/// ratio of 45 dB, is not coded in intensity stereo: each channel codes its
/// own lines there. Any two positions are then within 60 of each other,
/// whichever bands are sent. A position limited to the range instead would
/// scale the right channel from the left channel's band by up to the
/// excess: a tone in the right channel alone, beside the leakage of the
/// left channel's window into its band, decoded 28 dB too quiet.
///
/// Spec gap: clause C.6.2 codes the intensity signal through "the
/// standard process for quantization and encoding", without saying what
/// noise its bands may take. Its noise reaches the right channel scaled by
/// `Er / El`, so a band is allowed the less of the left channel's
/// allowance and the right channel's scaled up by `El / Er`.
pub fn code(
    windows: Windows,
    sampling_frequency: SamplingFrequency,
    first: usize,
    [left, right]: [&mut [f64; FRAME_LINES]; 2],
    [left_xmin, right_xmin]: [&mut [[f64; MAX_BANDS]]; 2],
) -> Intensity {
    let mut intensity = Intensity::NONE;
    for (g, (start, bands)) in windows.groups_with_start(sampling_frequency).enumerate() {
        let left = &mut left[start..start + bands.lines()];
        let right = &mut right[start..start + bands.lines()];
        for sfb in first.min(bands.num_swb())..bands.num_swb() {
            let (l, r) = (&mut left[bands.band(sfb)], &mut right[bands.band(sfb)]);
            let energy = |lines: &[f64]| lines.iter().map(|x| x * x).sum::<f64>();
            let (el, er) = (energy(l), energy(r));
            if el == 0.0 || er == 0.0 {
                continue;
            }
            let position = (2.0 * (el / er).log2()).round();
            if position.abs() > f64::from(MAX_POSITION) {
                continue;
            }
            let position = position as i8;
            let product: f64 = l.iter().zip(&*r).map(|(l, r)| l * r).sum();
            let (codebook, sign) = if product < 0.0 {
                (Codebook::INTENSITY_OUT_OF_PHASE, -1.0)
            } else {
                (Codebook::INTENSITY, 1.0)
            };
            let es: f64 = l.iter().zip(&*r).map(|(l, r)| (l + sign * r).powi(2)).sum();
            let gain = if es == 0.0 { 0.0 } else { (el / es).sqrt() };
            for (l, r) in l.iter_mut().zip(r.iter_mut()) {
                *l = (*l + sign * *r) * gain;
                *r = 0.0;
            }
            intensity.bands[g].set(sfb, Some(codebook));
            intensity.positions[g][sfb] = position;
            if er > 0.0 {
                left_xmin[g][sfb] = left_xmin[g][sfb].min(right_xmin[g][sfb] * el / er);
            }
        }
    }
    intensity
}
