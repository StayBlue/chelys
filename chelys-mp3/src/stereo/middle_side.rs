// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! MS stereo (Annex G.2 a) and b)): when to switch it on, the MS matrix,
//! the limitation of the side channel's bandwidth, and the allowed
//! distortion of the middle and side channels. The decoder's inverse is
//! clause 2.4.3.4.9.2.

use std::f64::consts::SQRT_2;

use crate::granule::{BlockType, GRANULE_LINES};
use crate::header::SamplingFrequency;
use crate::scalefactor_bands;

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
pub fn switches_on(granules: &[(&[f64], &[f64])]) -> bool {
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
pub fn matrix(
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
pub fn distortion(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter()
        .zip(right)
        .map(|(&l, &r)| match (l, r) {
            (0.0, other) | (other, 0.0) => other,
            _ => l.min(r),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alike_channels_switch_ms_on_and_either_alone_does_not() {
        let tone = [1.0, 4.0, 0.5, 0.0];
        let quieter = tone.map(|e| e * 0.5);
        let silent = [0.0; 4];
        assert!(switches_on(&[(&tone, &tone)]));
        assert!(switches_on(&[(&tone, &quieter)]));
        assert!(!switches_on(&[(&tone, &silent)]));
        assert!(!switches_on(&[(&silent, &tone)]));
        assert!(!switches_on(&[(&silent, &silent)]));
        // Over a frame: alike in one granule, apart in the other.
        let other = [0.0, 0.0, 0.0, 100.0];
        assert!(!switches_on(&[(&tone, &tone), (&tone, &other)]));
    }

    #[test]
    fn middle_side_inverts_like_the_decoder() {
        let left: [f64; GRANULE_LINES] = std::array::from_fn(|i| (i % 7) as f64 - 3.0);
        let right: [f64; GRANULE_LINES] = std::array::from_fn(|i| (i % 5) as f64 * 0.5);
        let (m, s) = matrix(&left, &right);
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
            let (_, mut side) = matrix(&left, &right);
            limit_side(&mut side, block_type, fs);
            assert!(side[top..].iter().all(|&s| s == 0.0), "{block_type:?}");
            assert!(side[..top].iter().all(|&s| s != 0.0), "{block_type:?}");
        }
    }

    #[test]
    fn middle_and_side_get_the_smaller_allowance() {
        assert_eq!(
            distortion(&[1.0, 0.2, 0.0, 0.0], &[0.5, 0.3, 4.0, 0.0]),
            [0.5, 0.2, 4.0, 0.0]
        );
    }
}
