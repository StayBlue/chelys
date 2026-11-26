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
/// average at level 0 to 104 at 50, 179 at 80 and 205 at 100, where the
/// mean grades, interpolated between the offsets swept, were -2.989,
/// -0.760, -0.126 and 0.003.
///
/// Swept, the mean grade was 0.118 at -16.5 dB, 0.143 at -15, 0.128 at
/// -13.5, 0.082 at -12, 0.003 at -10.5, -0.110 at -9, -0.343 at -7.5,
/// -0.574 at -6, -0.884 at -4.5, -1.304 at -3, -1.870 at -1.5, -2.501 at 0,
/// -3.024 at 1.5, -3.357 at 3, -3.535 at 4.5, -3.629 at 6 and -3.679 at
/// 7.5.
const STEREO: Offsets = [
    1.4, -0.2, -1.4, -2.6, -3.8, -5.1, -6.6, -8.0, -8.9, -10.1, -10.5,
];

/// Channels coded each on its own, calibrated in two channels at 44.1 kHz
/// coded so: from 76 kbit/s on average at level 0 to 117 at 50, 181 at 80
/// and 207 at 100, where the mean grades, interpolated between the offsets
/// swept, were -3.009, -0.760, -0.117 and -0.002.
///
/// Swept, the mean grade was 0.102 at -16.5 dB, 0.134 at -15, 0.141 at
/// -13.5, 0.111 at -12, 0.058 at -10.5, -0.024 at -9, -0.131 at -7.5,
/// -0.275 at -6, -0.466 at -4.5, -0.732 at -3, -1.149 at -1.5, -1.776 at 0,
/// -2.527 at 1.5, -3.130 at 3, -3.455 at 4.5, -3.605 at 6 and -3.677 at
/// 7.5.
const UNCOUPLED: Offsets = [
    2.7, 1.3, 0.3, -0.7, -1.7, -2.9, -4.4, -6.1, -7.7, -8.9, -9.4,
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
