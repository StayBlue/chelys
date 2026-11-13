//! Joint stereo for Layer III (Annex G.2): MS stereo, with when to switch it
//! on, the MS matrix and the limitation of the side channel's bandwidth, and
//! intensity stereo, with the bands it codes and their positions. The
//! decoder's inverse is clause 2.4.3.4.9.

use std::f64::consts::{PI, SQRT_2};

use super::header::SamplingFrequency;
use super::scalefactor_bands::{self, LONG_BANDS};
use super::scalefactors::Scalefactors;
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

/// The intensity stereo position that marks a band of the right channel as
/// not coded in intensity stereo, the "illegal" `is_pos` (clause
/// 2.4.3.4.9.3, Annex G.2 c)).
pub const NOT_INTENSITY: u8 = 7;

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
/// loses.
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

/// The positions of the bands of the right channel that a frame codes in
/// intensity stereo, for the scalefactors of its zero part (clause
/// 2.4.3.4.9.3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntensityPositions {
    /// The first long scalefactor band coded in intensity stereo.
    bound: usize,
    /// `is_pos` of each band from `bound`.
    positions: [u8; LONG_BANDS],
}

impl IntensityPositions {
    /// The positions of one granule coded from long scalefactor band
    /// `bound` up in intensity stereo, each found from the energies of the
    /// band's `left` and `right` lines (Annex G.2 c)).
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
        let bands = scalefactor_bands::long_bands(sampling_frequency);
        let energy = |lines: &[f64]| lines.iter().map(|x| x * x).sum::<f64>();
        let mut positions = [0; LONG_BANDS];
        for (sb, band) in bands.iter().enumerate().skip(bound) {
            let end = if sb == LONG_BANDS - 1 {
                GRANULE_LINES
            } else {
                band.end()
            };
            let lines = band.start..end;
            positions[sb] = intensity_position(energy(&left[lines.clone()]), energy(&right[lines]));
        }
        Self { bound, positions }
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
    pub fn code(
        &self,
        left: &[f64; GRANULE_LINES],
        right: &[f64; GRANULE_LINES],
        coded: [&mut [f64; GRANULE_LINES]; 2],
        sampling_frequency: SamplingFrequency,
    ) {
        let start = scalefactor_bands::long_bands(sampling_frequency)[self.bound].start;
        let [coded_left, coded_right] = coded;
        for i in start..GRANULE_LINES {
            coded_left[i] = left[i] + right[i];
            coded_right[i] = 0.0;
        }
    }

    /// The allowed distortion of the sum of the left and right channels
    /// sent in long scalefactor band `sb`, from or above the bound, given
    /// the allowed distortion of the left and right channels.
    ///
    /// Spec gap: Annex G does not say how to find it. With
    /// `is_ratio = tan(is_pos * pi / 12)`, the decoder gives the left
    /// channel the sum times `is_ratio / (1 + is_ratio)` and the right the
    /// sum times `1 / (1 + is_ratio)` (clause 2.4.3.4.9.3), and so those
    /// shares, squared, of its noise energy.
    ///
    /// Spec reading: step 5 of clause 2.4.3.4.9.3 computes `R_i` from
    /// `L_i` after step 4 has set `L_i` to the left channel's share. It is
    /// read as the sum, as before step 4: so the two shares add up to the
    /// sum, which Annex G.2 c) sends as `L_i + R_i`. The sum is allowed the most
    /// noise that keeps each channel within its allowance. A channel whose
    /// share is zero sets no limit, and nor does one that allows no noise
    /// because it has no energy in the band, as in [`middle_side_distortion`].
    pub fn sum_distortion(&self, sb: usize, left: f64, right: f64) -> f64 {
        assert!(sb >= self.bound, "band {sb} is below the bound");
        let angle = f64::from(self.positions[sb]) * PI / 12.0;
        // is_ratio / (1 + is_ratio) and 1 / (1 + is_ratio), exact at the
        // ends, where tan is 0 or infinite.
        let (sin, cos) = match self.positions[sb] {
            0 => (0.0, 1.0),
            6 => (1.0, 0.0),
            _ => angle.sin_cos(),
        };
        [(left, sin), (right, cos)]
            .into_iter()
            .filter(|&(xmin, share)| xmin > 0.0 && share > 0.0)
            .map(|(xmin, share)| xmin * ((sin + cos) / share).powi(2))
            .reduce(f64::min)
            .unwrap_or(0.0)
    }

    /// Sets the scalefactors of the right channel's zero part, given its
    /// quantised `values`: the position of each band from the bound, and
    /// [`NOT_INTENSITY`] in each band below it that lies above the last
    /// nonzero value, as Annex G.2 c) asks so that the decoder does not
    /// decode those bands in intensity stereo.
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
    pub fn mark(
        &self,
        scalefactors: &mut Scalefactors,
        values: &[i32; GRANULE_LINES],
        sampling_frequency: SamplingFrequency,
    ) {
        let bands = scalefactor_bands::long_bands(sampling_frequency);
        assert!(
            values[bands[self.bound].start..].iter().all(|&v| v == 0),
            "the right channel is zero from the intensity stereo bound"
        );
        let zero_from = values.iter().rposition(|&v| v != 0).map_or(0, |last| {
            bands.iter().position(|b| last < b.end()).unwrap() + 1
        });
        for sb in zero_from..LONG_BANDS {
            scalefactors.values_mut()[sb] = if sb < self.bound {
                NOT_INTENSITY
            } else {
                self.positions[sb]
            };
        }
    }
}

/// The most bits the scalefactors of the right channel take in intensity
/// stereo when all its lines quantise to zero: every long scalefactor in 3
/// bits, as positions up to [`NOT_INTENSITY`] need.
pub const MAX_POSITION_BITS: usize = 3 * LONG_BANDS;

#[cfg(test)]
mod tests {
    use super::*;

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
    /// above (Table B.8).
    #[test]
    fn intensity_bounds_follow_the_layer_i_and_ii_subbands() {
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
        positions.code(&left_top, &right, [&mut coded_left, &mut coded_right], fs);
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
        positions.mark(&mut scalefactors, &values, fs);
        let marked = scalefactors.values();
        assert_eq!(marked[..6], [0, 0, 9, 0, 0, 0]);
        assert!(marked[6..12].iter().all(|&s| s == NOT_INTENSITY));
        assert_eq!(marked[12..], positions.positions[12..]);

        let mut scalefactors = Scalefactors::default();
        positions.mark(&mut scalefactors, &[0; GRANULE_LINES], fs);
        assert!(
            scalefactors.values()[..12]
                .iter()
                .all(|&s| s == NOT_INTENSITY)
        );
        // Marks in 3 bits for all 21 bands.
        let compress = scalefactors.compress().unwrap();
        assert_eq!(scalefactors.part2_length(compress), MAX_POSITION_BITS);
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
    }

    #[test]
    fn middle_and_side_get_the_smaller_allowance() {
        assert_eq!(
            middle_side_distortion(&[1.0, 0.2, 0.0, 0.0], &[0.5, 0.3, 4.0, 0.0]),
            [0.5, 0.2, 4.0, 0.0]
        );
    }
}
