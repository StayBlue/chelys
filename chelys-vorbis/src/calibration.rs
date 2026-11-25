// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The noise offset of each level of the [`quality`] scale.
//!
//! The specification describes no encoder, and has no scale of quality.
//! The offsets are calibrated with PEAQ (GstPEAQ 0.6.1, black box) as
//! [`quality`] defines the scale: those of [`CALIBRATION_LEVELS`] are the
//! offsets whose mean grade over its 51 tracks of nine kinds of music meets
//! each level's target, interpolated between offsets of -16.5 to 7.5 dB
//! swept in steps of 1.5 dB, and rounded to 0.1 dB.
//!
//! The offsets follow the encoder: they are to be calibrated again as it
//! changes.

use chelys_core::quality::{self, CALIBRATION_LEVELS};

/// Offsets in dB, for each of [`CALIBRATION_LEVELS`].
type Offsets = [f64; CALIBRATION_LEVELS.len()];

/// Two channels, coupled, calibrated at 44.1 kHz: from 62 kbit/s on
/// average at level 0 to 105 at 50, 191 at 80 and 223 at 100, where the
/// mean grades were -3.012, -0.746, -0.122 and 0.003.
///
/// Swept, the mean grade was 0.133 at -16.5 dB, 0.111 at -15, 0.059 at
/// -13.5, -0.021 at -12, -0.122 at -10.5, -0.237 at -9, -0.384 at -7.5,
/// -0.545 at -6, -0.783 at -4.5, -1.134 at -3, -1.642 at -1.5, -2.269 at 0,
/// -2.854 at 1.5, -3.257 at 3, -3.479 at 4.5, -3.595 at 6 and -3.661 at
/// 7.5.
const STEREO: Offsets = [
    2.0, 0.4, -0.8, -2.0, -3.2, -4.7, -6.6, -8.7, -10.5, -11.9, -12.4,
];

/// Channels coded each on its own, calibrated in two channels at 44.1 kHz
/// coded so: from 72 kbit/s on average at level 0 to 119 at 50, 188 at 80
/// and 221 at 100, where the mean grades were -3.006, -0.741, -0.120 and
/// 0.000.
///
/// Swept, the mean grade was 0.109 at -16.5 dB, 0.128 at -15, 0.159 at
/// -13.5, 0.135 at -12, 0.074 at -10.5, -0.032 at -9, -0.113 at -7.5,
/// -0.252 at -6, -0.447 at -4.5, -0.682 at -3, -1.052 at -1.5, -1.637 at 0,
/// -2.388 at 1.5, -3.042 at 3, -3.407 at 4.5, -3.574 at 6 and -3.656 at
/// 7.5.
const UNCOUPLED: Offsets = [
    2.9, 1.6, 0.6, -0.4, -1.4, -2.7, -4.3, -5.9, -7.4, -9.0, -9.4,
];

/// The noise offset, in dB, of `level` (see
/// [`Rate::Quality`](crate::Rate::Quality)) in a stream of `channels`
/// channels.
///
/// Spec gap: every configuration takes the offsets calibrated at 44.1 kHz
/// in two channels: a stereo stream, whose channels are coupled, those of
/// [`STEREO`]; any other, whose channels are coded each on its own, those
/// of [`UNCOUPLED`]. An offset is a measure against the psychoacoustic
/// model's allowance, whatever the sampling frequency and the channels, as
/// each channel is coded within its own allowance, or a stereo pair's
/// within a multiple of it (see
/// [`angle_allowance`](crate::coupling::angle_allowance)); how far the
/// grades of other configurations stray from the scale's has not been
/// measured.
///
/// # Panics
///
/// If `level` is above [`quality::MAX_LEVEL`].
pub(crate) fn noise_offset_db(level: u8, channels: usize) -> f64 {
    let offsets = if channels == 2 { &STEREO } else { &UNCOUPLED };
    quality::interpolate(level, offsets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, ConfigError, Rate};

    /// A higher level never allows more noise.
    #[test]
    fn the_offset_falls_as_the_level_rises() {
        for offsets in [STEREO, UNCOUPLED] {
            for pair in offsets.windows(2) {
                assert!(pair[1] < pair[0], "{offsets:?}");
            }
        }
    }

    /// A level is the variable rate of its offset, in every configuration.
    #[test]
    fn a_level_is_the_variable_rate_of_its_offset() {
        let configurations = [
            (44_100, 2),
            (44_100, 1),
            (48_000, 1),
            (22_050, 2),
            (96_000, 8),
        ];
        for (sample_rate, channels) in configurations {
            let offsets = if channels == 2 { STEREO } else { UNCOUPLED };
            for (index, level) in CALIBRATION_LEVELS.into_iter().chain([55]).enumerate() {
                let expected = offsets
                    .get(index)
                    .copied()
                    .unwrap_or_else(|| (offsets[5] + offsets[6]) / 2.0);
                let config = Config::new(sample_rate, channels, Rate::Quality { level });
                let offset = config.noise_offset_db().unwrap();
                assert!(
                    (offset - expected).abs() < 1e-12,
                    "{sample_rate} Hz, {channels} channels, level {level}"
                );
            }
        }
    }

    #[test]
    fn a_level_above_the_scale_is_refused() {
        let config = Config::new(44_100, 2, Rate::Quality { level: 101 });
        assert_eq!(
            config.check(),
            Err(ConfigError::QualityLevel { level: 101 })
        );
    }
}
