// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Joint stereo for Layer III (Annex G.2): MS stereo, with when to switch it
//! on, the MS matrix and the limitation of the side channel's bandwidth, and
//! intensity stereo, with the bands it codes and their positions. The
//! decoder's inverse is clause 2.4.3.4.9, with the intensity stereo of the
//! lower sampling frequencies in clause 2.4.3.2 of ISO/IEC 13818-3.

use std::f64::consts::{PI, SQRT_2};
use std::ops::Range;

use super::header::{SamplingFrequency, Version};
use super::scalefactor_bands::{self, LONG_BANDS};
use super::scalefactors::{Scalefactors, ZeroPart};
use super::{BlockType, GRANULE_LINES};

/// The FFT lines the switching criterion sums over, `i = 0` to `511`
/// (Annex G.2 a)).
pub const CRITERION_LINES: usize = 512;

/// MS stereo is switched on when the left and right spectra differ by less
/// than this share of their sum (Annex G.2 a)).
const SWITCH_SHARE: f64 = 0.8;

/// Whether to code a frame in MS stereo (Annex G.2 a)), given the line
/// energies of the left and right channels' FFT spectra, lines 0 to 511, for
/// each of the frame's granules.
///
/// Spec reading: the criterion is printed as
/// `sum(rl_i^2 - rr_i^2) < 0.8 * sum(rl_i^2 + rr_i^2)`. As a signed sum this
/// only asks that the left channel have less than 9 times the energy of the
/// right, so a signal in the right channel alone would switch MS on and one
/// in the left alone would not. It is read with the difference of each line
/// taken as its magnitude, `sum |rl_i^2 - rr_i^2|`, which asks that the two
/// spectra be alike, as MS coding needs, and treats the channels alike.
/// `rl_i` and `rr_i` are taken as the magnitudes `r(w)` of the model's long
/// FFT (clause D.2.4 b)), so that their squares are the line energies the
/// text names; were they the energies themselves, as its wording could also
/// mean, the squares would make the criterion stricter, asking for energies
/// within a factor of 3 rather than 9.
///
/// Spec gap: `mode_extension`, which switches MS stereo, is sent once per
/// frame, but the model's spectra are per granule. The sums run over both
/// granules of the frame.
pub fn ms_stereo(granules: &[(&[f64], &[f64])]) -> bool {
    let (difference, sum) = granules
        .iter()
        .flat_map(|(left, right)| left.iter().zip(right.iter()))
        .fold((0.0, 0.0), |(d, s), (l, r)| (d + (l - r).abs(), s + l + r));
    sum > 0.0 && difference < SWITCH_SHARE * sum
}

/// Turns one granule's left and right lines into the middle and side lines
/// of MS stereo, `M_i = (L_i + R_i) / sqrt(2)` and
/// `S_i = (L_i - R_i) / sqrt(2)` (Annex G.2 b)), which the decoder turns back
/// (clause 2.4.3.4.9.2).
pub fn middle_side(
    left: &[f64; GRANULE_LINES],
    right: &[f64; GRANULE_LINES],
) -> ([f64; GRANULE_LINES], [f64; GRANULE_LINES]) {
    (
        std::array::from_fn(|i| (left[i] + right[i]) / SQRT_2),
        std::array::from_fn(|i| (left[i] - right[i]) / SQRT_2),
    )
}

/// The limitation of the side channel's bandwidth (Annex G.2 b)): all side
/// lines above the highest scalefactor band of `block_type` are set to zero.
/// The lines are in the order they are sent.
///
/// Spec reading: this deliberately leaves out the sparsing of the side
/// channel that Annex G.2 b) goes on to describe, which zeroes each pair of
/// side lines in a long scalefactor band whose energy is below `s_sb` times
/// that of the pair's left and right lines, with `s_sb` 0.10 from band 5,
/// rising to 0.90 by band 18, and 1.00 and 1.50 in bands 19 and 20. A pair's
/// side energy never exceeds its left and right energy, so bands 19 and 20
/// would always be mono, and from band 11 (about 2 kHz) a channel 6 dB
/// quieter than the other would be decoded, like the other, at their
/// average: level differences between the channels would be lost.
///
/// Settled by decoding with ffmpeg (black box): at 44.1 kHz, tones with the
/// right channel at half the left decode at 13.5 and 7.7 dB SNR with
/// sparsing, at both 64 and 128 kbit/s, and at 35.2 and 29.6 dB (64 kbit/s)
/// and 59.8 and 54.4 dB (128 kbit/s) without it, against 34 dB (64 kbit/s)
/// and 58 dB (128 kbit/s) in left/right stereo. The same tones in both
/// channels gain the same either way. The conformance test
/// `alike_channels_use_ms_stereo` fails with sparsing.
pub fn limit_side(
    side: &mut [f64; GRANULE_LINES],
    block_type: BlockType,
    sampling_frequency: SamplingFrequency,
) {
    let bands = scalefactor_bands::coding_bands(block_type, sampling_frequency);
    let top = bands.iter().map(|band| band.end()).max().expect("bands");
    side[top..].fill(0.0);
}

/// The allowed distortion of the middle and side channels, from that of the
/// left and right channels.
///
/// Spec gap: Annex G does not say how to find it. The decoder's matrix gives
/// the left and right channels each the mean of the middle and side
/// channels' noise energies, if those are uncorrelated, so allowing each of
/// them the smaller of the left and right allowances in every scalefactor
/// band keeps both within theirs. A band with no energy in one channel
/// allows that channel no noise, which middle and side lines that are not
/// zero could never meet; such a band takes the other channel's allowance,
/// as its noise in the silent channel lies under the other's signal.
pub fn middle_side_distortion(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter()
        .zip(right)
        .map(|(&l, &r)| match (l, r) {
            (0.0, other) | (other, 0.0) => other,
            _ => l.min(r),
        })
        .collect()
}

/// `intensity_scale`, sent in the right channel's `scalefac_compress` at the
/// lower sampling frequencies, which sets the step between intensity stereo
/// positions: `i0 = 1/sqrt(2)` for 1, `1/sqrt(sqrt(2))` for 0 (clause
/// 2.4.3.2 of ISO/IEC 13818-3).
///
/// Spec gap: neither standard says how to choose it. It is always 1: each
/// step of position then halves the energy of the quieter channel, 3 dB,
/// which keeps every level within 1.5 dB of its ratio, closer than the
/// positions of ISO/IEC 11172-3 at their closest (0 and 4.8 dB at positions
/// 3 and 4), and in the 3 bits those positions take reaches 9 dB either way,
/// where steps of 1.5 dB would reach 4.5 dB.
pub const INTENSITY_SCALE: u16 = 1;

/// `i0` for [`INTENSITY_SCALE`].
const I0: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// The most steps of `i0` that a position at the lower sampling frequencies
/// gives the quieter channel: positions 13 and 14, 21 dB with
/// [`INTENSITY_SCALE`] 1.
///
/// Spec reading: clause 2.4.3.2 of ISO/IEC 13818-3 allows positions up to
/// 30, in a field of 5 bits, but not every decoder reads them so. Positions
/// are kept below 16.
const MAX_STEPS: i32 = 7;

/// The steps of `i0` that a position at the lower sampling frequencies never
/// gives: position 7, the left channel four steps quieter than the right.
///
/// Spec reading: position 7 is a position in a field of 4 or 5 bits (clause
/// 2.4.3.2 of ISO/IEC 13818-3), but not every decoder reads it so. A band
/// whose nearest position is 7 takes the nearer of 5 and 9.
const SKIPPED_STEPS: i32 = -4;

/// The `bound` of Layers I and II in intensity stereo: the subband from
/// which they code it (Annex G.1, clause 2.4.2.3), from the fewest subbands
/// to the most.
const LAYER_I_II_BOUNDS: [usize; 4] = [16, 12, 8, 4];

/// The frequency lines of a granule in each polyphase subband (clause
/// 2.4.3.4.10).
const SUBBAND_LINES: usize = 18;

/// The long scalefactor bands from which a frame may be coded in intensity
/// stereo, from the fewest bands to the most.
///
/// Spec gap: Annex G.2 c) does not say which bands to code in intensity
/// stereo. Annex G.1 does for Layers I and II: from subband 16, 12, 8 or 4,
/// depending on the bits needed. The same four steps are taken here, each
/// from the first scalefactor band that starts at or above the subband, so
/// that intensity stereo never reaches below where those layers would let
/// it. The lowest, from subband 4, starts at 2.3 to 3.0 kHz, above the
/// "about 2 kHz" where G.1 says the temporal envelope, which intensity
/// stereo keeps, takes over localisation from the fine structure, which it
/// loses. At the lower sampling frequencies, whose subbands are half as wide,
/// it starts at 1.1 to 1.7 kHz, below that, as it does for Layers I and II
/// there, whose `bound` is that of ISO/IEC 11172-3 (clause 2.4.2.3 of
/// ISO/IEC 13818-3).
pub fn intensity_bounds(sampling_frequency: SamplingFrequency) -> [usize; 4] {
    let bands = scalefactor_bands::long_bands(sampling_frequency);
    LAYER_I_II_BOUNDS.map(|subband| {
        bands
            .iter()
            .position(|band| band.start >= SUBBAND_LINES * subband)
            .expect("every bound lies below the highest band")
    })
}

/// The intensity stereo position of a scalefactor band,
/// `is_pos = nint(12 / pi * arctan(sqrt(L_energy / R_energy)))` (Annex
/// G.2 c)), from the energies of its left and right lines: 0 for the right
/// channel alone, 6 for the left alone.
///
/// Spec reading: a band with no energy in the right channel divides by
/// zero; it takes the limit, position 6, the left channel alone.
///
/// Spec gap: a band with no energy in either channel has no position. It
/// takes the middle one, 3; its lines are zero in both channels anyway.
pub fn intensity_position(left_energy: f64, right_energy: f64) -> u8 {
    if left_energy == 0.0 && right_energy == 0.0 {
        return 3;
    }
    // A silent right channel makes the ratio infinite, and the position 6.
    let angle = (left_energy / right_energy).sqrt().atan();
    (12.0 / PI * angle).round() as u8
}

/// The intensity stereo position of a scalefactor band at the lower
/// sampling frequencies, from the energies of its left and right lines.
///
/// Spec reading: Annex C.1.3 of ISO/IEC 13818-3 has the encoder take every
/// basic step of ISO/IEC 11172-3, but the position of Annex G.2 c) is the
/// one by which that standard's decoder splits the sum, and clause 2.4.3.2
/// of ISO/IEC 13818-3 decodes positions otherwise: it multiplies the lines
/// sent by `k_l` and `k_r`, both 1 at position 0, and at odd positions
/// `k_l = i0^((is_pos + 1) / 2)`, at even ones `k_r = i0^(is_pos / 2)`, the
/// other 1. The position taken is the one whose `k_l / k_r` lies nearest the
/// ratio `sqrt(L_energy / R_energy)` that Annex G.2 c) takes the arctangent
/// of, nearest in steps of `i0`, of which there are up to `MAX_STEPS`,
/// other than `SKIPPED_STEPS`.
///
/// Spec gap: a band with no energy in either channel has no position. It
/// takes position 0, the same level in both; its lines are zero in both
/// channels anyway.
pub fn lsf_intensity_position(left_energy: f64, right_energy: f64) -> u8 {
    if left_energy == 0.0 && right_energy == 0.0 {
        return 0;
    }
    // Steps of i0 in amplitude; a silent channel makes them infinite.
    let exact = (left_energy / right_energy).sqrt().ln() / -I0.ln();
    let steps = exact
        .round()
        .clamp(f64::from(-MAX_STEPS), f64::from(MAX_STEPS)) as i32;
    let steps = match steps {
        SKIPPED_STEPS if exact >= f64::from(SKIPPED_STEPS) => SKIPPED_STEPS + 1,
        SKIPPED_STEPS => SKIPPED_STEPS - 1,
        steps => steps,
    };
    match steps {
        0 => 0,
        n if n > 0 => (2 * n) as u8,
        n => (-2 * n - 1) as u8,
    }
}

/// The factors by which the decoder multiplies the lines sent in the left
/// channel for a band at intensity stereo `position`, to give the left and
/// the right channel: `is_ratio / (1 + is_ratio)` and `1 / (1 + is_ratio)`
/// with `is_ratio = tan(is_pos * pi / 12)` (clause 2.4.3.4.9.3), or at the
/// lower sampling frequencies `k_l` and `k_r` (clause 2.4.3.2 of ISO/IEC
/// 13818-3).
///
/// Spec reading: step 5 of clause 2.4.3.4.9.3 computes `R_i` from `L_i`
/// after step 4 has set `L_i` to the left channel's share. It is read as the
/// lines sent, as before step 4: so the two shares add up to them, which
/// Annex G.2 c) sends as `L_i + R_i`. Clause 2.4.3.2 of ISO/IEC 13818-3
/// swaps the two steps so that this is so.
fn shares(version: Version, position: u8) -> (f64, f64) {
    match version {
        Version::Mpeg1 => {
            // sin / (sin + cos) and cos / (sin + cos), exact at the ends,
            // where tan is 0 or infinite.
            let (sin, cos) = match position {
                0 => (0.0, 1.0),
                6 => (1.0, 0.0),
                _ => (f64::from(position) * PI / 12.0).sin_cos(),
            };
            (sin / (sin + cos), cos / (sin + cos))
        }
        Version::Mpeg2 => match position {
            0 => (1.0, 1.0),
            p if p % 2 == 1 => (I0.powi(i32::from(p.div_ceil(2))), 1.0),
            p => (1.0, I0.powi(i32::from(p / 2))),
        },
    }
}

/// The positions of the bands of the right channel that a frame codes in
/// intensity stereo, for the scalefactors of its zero part (clause
/// 2.4.3.4.9.3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntensityPositions {
    /// The first long scalefactor band coded in intensity stereo.
    bound: usize,
    /// `is_pos` of each band from `bound`.
    positions: [u8; LONG_BANDS],
    sampling_frequency: SamplingFrequency,
}

impl IntensityPositions {
    /// The positions of one granule coded from long scalefactor band
    /// `bound` up in intensity stereo, each found from the energies of the
    /// band's `left` and `right` lines (Annex G.2 c), and
    /// [`lsf_intensity_position`] at the lower sampling frequencies).
    ///
    /// Spec gap: the lines above the highest scalefactor band have no
    /// scalefactor, and so no position, and the standard does not say how a
    /// decoder treats them in intensity stereo. They lie in the right
    /// channel's zero part, which the figure of clause 2.4.3.4.9.3 shows
    /// coded in intensity stereo up to the end of the spectrum, and are
    /// coded like the bands (see [`code`](Self::code)). They are taken to
    /// share the position of the highest band, which is found from the
    /// energies of that band's lines and theirs together.
    ///
    /// Settled by decoding with ffmpeg (black box): at 64 kbit/s and
    /// 44.1 kHz, an 18 kHz tone with the right channel at half the left
    /// decodes at 0.0948 and 0.0547 for the 0.0951 and 0.0549 its position
    /// gives. With the highest band's position found from that band's lines
    /// alone, the left channel decodes at 0.0800: ffmpeg splits the lines
    /// above the highest band by its position. The conformance test
    /// `intensity_stereo_keeps_the_level_of_each_channel` fails then.
    ///
    /// # Panics
    ///
    /// If `bound` is not a long scalefactor band. Short blocks are not coded
    /// in intensity stereo.
    pub fn new(
        left: &[f64; GRANULE_LINES],
        right: &[f64; GRANULE_LINES],
        bound: usize,
        sampling_frequency: SamplingFrequency,
    ) -> Self {
        let position = match sampling_frequency.version() {
            Version::Mpeg1 => intensity_position,
            Version::Mpeg2 => lsf_intensity_position,
        };
        let mut positions = [0; LONG_BANDS];
        for (sb, lines) in coded_bands(bound, sampling_frequency) {
            positions[sb] = position(energy(&left[lines.clone()]), energy(&right[lines]));
        }
        Self {
            bound,
            positions,
            sampling_frequency,
        }
    }

    /// The first long scalefactor band coded in intensity stereo.
    pub fn bound(&self) -> usize {
        self.bound
    }

    /// Codes a granule's lines from the bound up in intensity stereo (Annex
    /// G.2 c)): sends the sum of its `left` and `right` lines in the left
    /// channel, `L_i = L_i + R_i`, and zero in the right. `coded` holds the
    /// lines sent in the left and right channels, which below the bound are
    /// the left and right lines or, in MS stereo, the middle and side lines.
    /// The decoder splits the sum by the positions (clause 2.4.3.4.9.3).
    ///
    /// Spec reading: the decoder of ISO/IEC 11172-3 splits the sum into
    /// shares that add up to it, but at the lower sampling frequencies it
    /// multiplies the lines sent by `k_l` and `k_r`, the larger of which is 1
    /// (clause 2.4.3.2 of ISO/IEC 13818-3), so that the sum itself would
    /// decode too loud: twice the level of each channel where the two are
    /// alike. There the sum is scaled in each band so that the two channels
    /// decode with the energy of its left and right lines together. For a
    /// signal panned between the channels, as intensity stereo takes it to
    /// be, at the level difference its position gives, that sends the
    /// louder channel's lines.
    pub fn code(
        &self,
        left: &[f64; GRANULE_LINES],
        right: &[f64; GRANULE_LINES],
        coded: [&mut [f64; GRANULE_LINES]; 2],
    ) {
        let fs = self.sampling_frequency;
        let [coded_left, coded_right] = coded;
        for (sb, lines) in coded_bands(self.bound, fs) {
            let sum: Vec<f64> = lines.clone().map(|i| left[i] + right[i]).collect();
            let scale = match fs.version() {
                Version::Mpeg1 => 1.0,
                Version::Mpeg2 => {
                    let (k_l, k_r) = shares(fs.version(), self.positions[sb]);
                    let wanted = energy(&left[lines.clone()]) + energy(&right[lines.clone()]);
                    let decoded = (k_l * k_l + k_r * k_r) * energy(&sum);
                    if decoded > 0.0 {
                        (wanted / decoded).sqrt()
                    } else {
                        0.0
                    }
                }
            };
            for (i, x) in lines.zip(sum) {
                coded_left[i] = scale * x;
                coded_right[i] = 0.0;
            }
        }
    }

    /// The allowed distortion of the sum of the left and right channels
    /// sent in long scalefactor band `sb`, from or above the bound, given
    /// the allowed distortion of the left and right channels.
    ///
    /// Spec gap: Annex G does not say how to find it. The decoder gives each
    /// channel the lines sent times its `shares`, and so those factors,
    /// squared, of their noise energy. The sum is allowed the most noise
    /// that keeps each channel within its allowance. A channel whose share
    /// is zero sets no limit, and nor does one that allows no noise because
    /// it has no energy in the band, as in [`middle_side_distortion`].
    pub fn sum_distortion(&self, sb: usize, left: f64, right: f64) -> f64 {
        assert!(sb >= self.bound, "band {sb} is below the bound");
        let (left_share, right_share) =
            shares(self.sampling_frequency.version(), self.positions[sb]);
        [(left, left_share), (right, right_share)]
            .into_iter()
            .filter(|&(xmin, share)| xmin > 0.0 && share > 0.0)
            .map(|(xmin, share)| xmin / (share * share))
            .reduce(f64::min)
            .unwrap_or(0.0)
    }

    /// Sets the scalefactors of the right channel's zero part, given its
    /// quantised `values`: the position of each band from the bound, and a
    /// mark in each band below it that lies above the last nonzero value, as
    /// Annex G.2 c) asks so that the decoder does not decode those bands in
    /// intensity stereo. The mark is the "illegal" position, sent as
    /// [`Scalefactors::write`] gives it. At the lower sampling frequencies a
    /// granule with marks is not kept (see `quantize_frame`), but they still
    /// count towards the bits of its scalefactors while it is quantised.
    ///
    /// Spec reading: clause 2.4.3.4.9.1 has the zero part begin above the
    /// band of the last nonzero value, and also at line
    /// `big_values * 2 + count1 * 4`. Here the `count1` partition ends with
    /// the pair holding the last nonzero value (clause C.1.5.4.4.4), and
    /// scalefactor bands start on even lines, so the two agree.
    ///
    /// Spec reading: Annex G.2 c) asks for the mark in every band of zeros
    /// outside the intensity coded part. Only the bands of the zero part
    /// could be decoded in intensity stereo, so the bands of zeros below the
    /// last nonzero value keep their scalefactors.
    ///
    /// Spec reading: in MS stereo the right channel is the side channel,
    /// which Annex G.2 c) names, as the "right/difference channel", among
    /// those whose marked bands are kept from intensity stereo decoding.
    /// Clause 2.4.3.4.9.1 has MS stereo end at the zero part, but a marked
    /// band is taken to be decoded as though outside it, in MS stereo, as
    /// its middle channel would otherwise be decoded as the left channel
    /// alone.
    ///
    /// Settled by decoding with ffmpeg (black box): in MS and intensity
    /// stereo, with the side channel zero below the bound and so every band
    /// there marked, a 440 Hz tone the same in both channels at 0.2 decodes
    /// at 0.1997 in each, where decoding the marked bands in left/right
    /// stereo would give 0.283 in the left and nothing in the right. The
    /// conformance test is `intensity_stereo_keeps_the_level_of_each_channel`.
    ///
    /// # Panics
    ///
    /// If a value from the bound is not zero.
    pub fn mark(&self, scalefactors: &mut Scalefactors, values: &[i32; GRANULE_LINES]) {
        let bands = scalefactor_bands::long_bands(self.sampling_frequency);
        assert!(
            values[bands[self.bound].start..].iter().all(|&v| v == 0),
            "the right channel is zero from the intensity stereo bound"
        );
        let zero_from = values.iter().rposition(|&v| v != 0).map_or(0, |last| {
            bands.iter().position(|b| last < b.end()).unwrap() + 1
        });
        for sb in zero_from..LONG_BANDS {
            // The marked bands send no value of their own.
            scalefactors.values_mut()[sb] = if sb < self.bound {
                0
            } else {
                self.positions[sb]
            };
        }
        scalefactors.intensity = Some(ZeroPart {
            start: zero_from,
            bound: self.bound,
        });
    }

    /// The bits the right channel takes with all its lines quantised to
    /// zero, every band below the bound marked and every band from it
    /// sending its position: the bits the iteration loops can always fit it
    /// in, by quantising more coarsely, and so the fewest it must be given.
    ///
    /// In ISO/IEC 11172-3 these are 63, every band in 3 bits; at the lower
    /// sampling frequencies they depend on the positions, and a granule
    /// quantised so is not kept, as its marks are not sent there (see
    /// `quantize_frame`).
    pub fn silent_bits(&self) -> usize {
        let mut scalefactors =
            Scalefactors::new(BlockType::Normal, self.sampling_frequency.version());
        self.mark(&mut scalefactors, &[0; GRANULE_LINES]);
        let compress = scalefactors.compress().expect("marks that fit a field");
        scalefactors.part2_length(compress)
    }
}

/// The long scalefactor bands from `bound` up, each with the lines it codes
/// in intensity stereo: the highest also those above it (see
/// [`IntensityPositions::new`]).
fn coded_bands(
    bound: usize,
    sampling_frequency: SamplingFrequency,
) -> impl Iterator<Item = (usize, Range<usize>)> {
    let bands = scalefactor_bands::long_bands(sampling_frequency);
    bands.iter().enumerate().skip(bound).map(|(sb, band)| {
        let end = if sb == LONG_BANDS - 1 {
            GRANULE_LINES
        } else {
            band.end()
        };
        (sb, band.start..end)
    })
}

fn energy(lines: &[f64]) -> f64 {
    lines.iter().map(|x| x * x).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitWriter;

    #[test]
    fn alike_channels_switch_ms_on_and_either_alone_does_not() {
        let tone = [1.0, 4.0, 0.5, 0.0];
        let quieter = tone.map(|e| e * 0.5);
        let silent = [0.0; 4];
        assert!(ms_stereo(&[(&tone, &tone)]));
        assert!(ms_stereo(&[(&tone, &quieter)]));
        assert!(!ms_stereo(&[(&tone, &silent)]));
        assert!(!ms_stereo(&[(&silent, &tone)]));
        assert!(!ms_stereo(&[(&silent, &silent)]));
        // Over a frame: alike in one granule, apart in the other.
        let other = [0.0, 0.0, 0.0, 100.0];
        assert!(!ms_stereo(&[(&tone, &tone), (&tone, &other)]));
    }

    #[test]
    fn middle_side_inverts_like_the_decoder() {
        let left: [f64; GRANULE_LINES] = std::array::from_fn(|i| (i % 7) as f64 - 3.0);
        let right: [f64; GRANULE_LINES] = std::array::from_fn(|i| (i % 5) as f64 * 0.5);
        let (m, s) = middle_side(&left, &right);
        for i in 0..GRANULE_LINES {
            assert!(((m[i] + s[i]) / SQRT_2 - left[i]).abs() < 1e-12);
            assert!(((m[i] - s[i]) / SQRT_2 - right[i]).abs() < 1e-12);
        }
    }

    /// Lines above the highest band are cleared, and only those.
    #[test]
    fn side_is_limited_to_the_scalefactor_bands() {
        let fs = SamplingFrequency::Hz44100;
        let left: [f64; GRANULE_LINES] = std::array::from_fn(|i| 1.0 + (i % 3) as f64);
        let right = [0.5; GRANULE_LINES];
        for (block_type, top) in [
            (
                BlockType::Normal,
                scalefactor_bands::long_bands(fs)[20].end(),
            ),
            (
                BlockType::Short,
                3 * scalefactor_bands::short_bands(fs)[11].end(),
            ),
        ] {
            let (_, mut side) = middle_side(&left, &right);
            limit_side(&mut side, block_type, fs);
            assert!(side[top..].iter().all(|&s| s == 0.0), "{block_type:?}");
            assert!(side[..top].iter().all(|&s| s != 0.0), "{block_type:?}");
        }
    }

    /// The Layer I and II bounds, subbands 16, 12, 8 and 4, start at lines
    /// 288, 216, 144 and 72; each step is the first band starting there or
    /// above (Table B.8, and Table B.2 of ISO/IEC 13818-3).
    #[test]
    fn intensity_bounds_follow_the_layer_i_and_ii_subbands() {
        assert_eq!(
            intensity_bounds(SamplingFrequency::Hz16000),
            [18, 16, 14, 10]
        );
        assert_eq!(
            intensity_bounds(SamplingFrequency::Hz22050),
            [18, 16, 14, 10]
        );
        assert_eq!(
            intensity_bounds(SamplingFrequency::Hz24000),
            [18, 16, 14, 10]
        );
        assert_eq!(
            intensity_bounds(SamplingFrequency::Hz32000),
            [18, 17, 15, 12]
        );
        assert_eq!(
            intensity_bounds(SamplingFrequency::Hz44100),
            [19, 18, 16, 12]
        );
        assert_eq!(
            intensity_bounds(SamplingFrequency::Hz48000),
            [20, 18, 16, 12]
        );
    }

    /// `nint(12 / pi * arctan(sqrt(L / R)))`: amplitudes 2 to 1 lie at
    /// 63.4 degrees, nearest position 4 (60 degrees), and 1 to 2 at 26.6
    /// degrees, nearest 2 (30 degrees).
    #[test]
    fn positions_follow_the_energy_ratio() {
        assert_eq!(intensity_position(1.0, 1.0), 3);
        assert_eq!(intensity_position(4.0, 1.0), 4);
        assert_eq!(intensity_position(1.0, 4.0), 2);
        assert_eq!(intensity_position(1.0, 0.0), 6);
        assert_eq!(intensity_position(0.0, 1.0), 0);
        assert_eq!(intensity_position(0.0, 0.0), 3);
    }

    /// From the bound up, the left channel sends the sum and the right
    /// zero; below it the lines sent are left as they are. The highest
    /// band's position counts the lines above it.
    #[test]
    fn intensity_sends_the_sum_from_the_bound() {
        let fs = SamplingFrequency::Hz44100;
        let bands = scalefactor_bands::long_bands(fs);
        let left: [f64; GRANULE_LINES] = std::array::from_fn(|i| 1.0 + (i % 3) as f64);
        let mut right = [0.5; GRANULE_LINES];
        // The highest band is silent; the lines above it are in the left
        // channel alone.
        right[bands[20].start..].fill(0.0);
        let mut left_top = left;
        left_top[bands[20].start..bands[20].end()].fill(0.0);
        let positions = IntensityPositions::new(&left_top, &right, 12, fs);
        assert_eq!(positions.bound(), 12);
        assert_eq!(positions.positions[20], 6);
        let (mut coded_left, mut coded_right) = middle_side(&left_top, &right);
        let (middle, side) = (coded_left, coded_right);
        positions.code(&left_top, &right, [&mut coded_left, &mut coded_right]);
        let start = bands[12].start;
        assert_eq!(coded_left[..start], middle[..start]);
        assert_eq!(coded_right[..start], side[..start]);
        for i in start..GRANULE_LINES {
            assert_eq!(coded_left[i], left_top[i] + right[i]);
            assert_eq!(coded_right[i], 0.0);
        }
    }

    /// The zero part of the right channel sends the positions from the
    /// bound and 7 below it; the bands up to the last nonzero value keep
    /// their scalefactors.
    #[test]
    fn the_zero_part_is_marked() {
        let fs = SamplingFrequency::Hz44100;
        let bands = scalefactor_bands::long_bands(fs);
        let left: [f64; GRANULE_LINES] = std::array::from_fn(|i| 1.0 + (i % 3) as f64);
        let positions = IntensityPositions::new(&left, &[0.5; GRANULE_LINES], 12, fs);
        let mut values = [0; GRANULE_LINES];
        values[bands[5].start + 1] = -2;
        let mut scalefactors = Scalefactors::default();
        scalefactors.values_mut()[2] = 9;
        positions.mark(&mut scalefactors, &values);
        let marked = scalefactors.values();
        assert_eq!(marked[..6], [0, 0, 9, 0, 0, 0]);
        assert_eq!(marked[12..], positions.positions[12..]);
        assert_eq!(
            scalefactors.intensity,
            Some(ZeroPart {
                start: 6,
                bound: 12
            })
        );
        // Bands 0 to 10 in 4 bits, 11 to 20 in 3: 9, then the marks.
        let compress = scalefactors.compress().unwrap();
        let mut w = BitWriter::new();
        scalefactors.write(compress, &mut w);
        let bits: String = w.into_bytes().iter().map(|b| format!("{b:08b}")).collect();
        assert_eq!(&bits[..4 * 6], "000000001001000000000000");
        assert_eq!(&bits[4 * 6..4 * 11], "01110111011101110111");
        assert_eq!(&bits[4 * 11..4 * 11 + 3], "111");

        let mut scalefactors = Scalefactors::default();
        positions.mark(&mut scalefactors, &[0; GRANULE_LINES]);
        assert_eq!(scalefactors.intensity.unwrap().start, 0);
        // Marks in 3 bits for all 21 bands.
        assert_eq!(positions.silent_bits(), 3 * LONG_BANDS);
    }

    /// At the lower sampling frequencies each step of position halves the
    /// energy of the quieter channel: the louder channel at twice the
    /// amplitude of the other is 2 steps, positions 4 (left) and 3 (right).
    #[test]
    fn lsf_positions_step_by_i0() {
        assert_eq!(lsf_intensity_position(1.0, 1.0), 0);
        assert_eq!(lsf_intensity_position(4.0, 1.0), 4);
        assert_eq!(lsf_intensity_position(1.0, 4.0), 3);
        assert_eq!(lsf_intensity_position(2.0, 1.0), 2);
        assert_eq!(lsf_intensity_position(1.0, 2.0), 1);
        // Nearest in steps: 2^1.4 is 1 step, 2^1.6 is 2.
        assert_eq!(lsf_intensity_position(1.4f64.exp2(), 1.0), 2);
        assert_eq!(lsf_intensity_position(1.6f64.exp2(), 1.0), 4);
        assert_eq!(lsf_intensity_position(1.0, 0.0), 14);
        assert_eq!(lsf_intensity_position(0.0, 1.0), 13);
        assert_eq!(lsf_intensity_position(40f64.exp2(), 1.0), 14);
        // Never position 7, the left channel 4 steps quieter: 3 or 5.
        assert_eq!(lsf_intensity_position(1.0, 3.6f64.exp2()), 5);
        assert_eq!(lsf_intensity_position(1.0, 3.9f64.exp2()), 5);
        assert_eq!(lsf_intensity_position(1.0, 4.1f64.exp2()), 9);
        assert_eq!(lsf_intensity_position(1.0, 4.4f64.exp2()), 9);
        assert_eq!(lsf_intensity_position(4f64.exp2(), 1.0), 8);
        assert_eq!(lsf_intensity_position(0.0, 0.0), 0);
        for (position, expected) in [(0, (1.0, 1.0)), (3, (0.5, 1.0)), (4, (1.0, 0.5))] {
            let (k_l, k_r) = shares(Version::Mpeg2, position);
            assert!((k_l - expected.0).abs() < 1e-12, "{position}");
            assert!((k_r - expected.1).abs() < 1e-12, "{position}");
        }
    }

    /// At the lower sampling frequencies the sum is scaled so that the
    /// channels decode with their energy: alike channels send either, and a
    /// tone with the right channel at half the left sends the left.
    #[test]
    fn lsf_intensity_sends_the_louder_channel() {
        let fs = SamplingFrequency::Hz22050;
        let start = scalefactor_bands::long_bands(fs)[10].start;
        let tone: [f64; GRANULE_LINES] = std::array::from_fn(|i| ((i * 7) % 5) as f64 - 2.0);
        let half = tone.map(|x| 0.5 * x);
        for (right, position) in [(tone, 0), (half, 4)] {
            let positions = IntensityPositions::new(&tone, &right, 10, fs);
            assert!(positions.positions[10..].iter().all(|&p| p == position));
            let (mut coded_left, mut coded_right) = (tone, right);
            positions.code(&tone, &right, [&mut coded_left, &mut coded_right]);
            for i in start..GRANULE_LINES {
                assert!((coded_left[i] - tone[i]).abs() < 1e-12, "{position}, {i}");
                assert_eq!(coded_right[i], 0.0);
            }
            assert_eq!(coded_left[..start], tone[..start]);
        }
    }

    /// At the lower sampling frequencies a mark is the largest value of its
    /// field, so a right channel all zero needs a bit for each band below
    /// the bound, and its positions a field that sends them below the mark.
    #[test]
    fn lsf_marks_take_a_field_of_at_least_one_bit() {
        let fs = SamplingFrequency::Hz22050;
        let tone: [f64; GRANULE_LINES] = std::array::from_fn(|i| ((i * 7) % 5) as f64 - 2.0);
        let alike = IntensityPositions::new(&tone, &tone, 14, fs);
        // Bands 0 to 13 marked in 1 bit; positions 0 in bands 14 to 20, the
        // third partition of int_scalefac_compress below 180, in no bits.
        assert_eq!(alike.silent_bits(), 14);
        let half = tone.map(|x| 0.5 * x);
        let panned = IntensityPositions::new(&tone, &half, 14, fs);
        // Position 4 needs a field of 3 bits, below its mark 7.
        assert_eq!(panned.silent_bits(), 7 + 7 + 3 * 7);
    }

    /// In the middle position each channel gets half the sum, and a quarter
    /// of its noise energy; at the ends only one channel limits the noise.
    #[test]
    fn the_sum_is_allowed_what_keeps_both_channels_within_theirs() {
        let positions = IntensityPositions {
            bound: 10,
            positions: std::array::from_fn(|sb| match sb {
                10 => 3,
                11 => 6,
                12 => 0,
                _ => 2,
            }),
            sampling_frequency: SamplingFrequency::Hz44100,
        };
        assert!((positions.sum_distortion(10, 2.0, 1.0) - 4.0).abs() < 1e-12);
        assert_eq!(positions.sum_distortion(11, 2.0, 1.0), 2.0);
        assert_eq!(positions.sum_distortion(12, 2.0, 1.0), 1.0);
        // A silent channel sets no limit.
        assert!((positions.sum_distortion(10, 0.0, 1.0) - 4.0).abs() < 1e-12);
        assert_eq!(positions.sum_distortion(10, 0.0, 0.0), 0.0);
        // Position 2, 30 degrees: the right channel gets cos / (sin + cos)
        // of the sum, the larger share, and so limits it.
        let (sin, cos) = (PI / 6.0).sin_cos();
        let expected = 1.0 * ((sin + cos) / cos).powi(2);
        assert!((positions.sum_distortion(13, 1.0, 1.0) - expected).abs() < 1e-12);
        // At the lower sampling frequencies position 4 gives the right
        // channel half the lines sent, and a quarter of their noise.
        let lsf = IntensityPositions {
            sampling_frequency: SamplingFrequency::Hz22050,
            positions: [4; LONG_BANDS],
            ..positions
        };
        assert!((lsf.sum_distortion(10, 2.0, 1.0) - 2.0).abs() < 1e-12);
        assert!((lsf.sum_distortion(10, 2.0, 0.25) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn middle_and_side_get_the_smaller_allowance() {
        assert_eq!(
            middle_side_distortion(&[1.0, 0.2, 0.0, 0.0], &[0.5, 0.3, 4.0, 0.0]),
            [0.5, 0.2, 4.0, 0.0]
        );
    }
}
