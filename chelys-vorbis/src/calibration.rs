// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The noise offset of each level of the [`quality`] scale.
//!
//! The specification describes no encoder, and has no scale of quality.
//! The offsets are calibrated with PEAQ (GstPEAQ 0.6.1, black box) as
//! [`quality`] defines the scale: those of [`CALIBRATION_LEVELS`] are the
//! offsets whose mean grade over its 51 tracks of nine kinds of music meets
//! each level's target, interpolated between offsets of -9 to 10.5 dB swept
//! in steps of 1.5 dB, and rounded to 0.1 dB.
//!
//! The offsets follow the encoder: they are to be calibrated again as it
//! changes, such as with channel coupling.

use chelys_core::quality::{self, CALIBRATION_LEVELS};

/// Offsets in dB, for each of [`CALIBRATION_LEVELS`].
type Offsets = [f64; CALIBRATION_LEVELS.len()];

/// Two channels, calibrated at 44.1 kHz: from 92 kbit/s on average at
/// level 0 to 144 at 50, 263 at 80 and 303 at 100, where the mean grades
/// were -3.090, -0.728, -0.119 and -0.001.
///
/// Swept, the mean grade was 0.080 at -9 dB, 0.005 at -7.5, -0.077 at -6,
/// -0.154 at -4.5, -0.228 at -3, -0.302 at -1.5, -0.398 at 0, -0.545 at
/// 1.5, -0.821 at 3, -1.415 at 4.5, -2.502 at 6, -3.230 at 7.5, -3.115 at 9
/// and -2.423 at 10.5, where PEAQ had stopped following the coding (see
/// [`quality`]).
const STEREO: Offsets = [7.0, 5.9, 5.2, 4.6, 3.7, 2.6, 0.8, -2.1, -5.2, -6.9, -7.4];

/// The noise offset, in dB, of `level` (see
/// [`Rate::Quality`](crate::Rate::Quality)).
///
/// Spec gap: every configuration takes the offsets calibrated at 44.1 kHz
/// in two channels, one channel and more than two included. An offset is a
/// measure against the psychoacoustic model's allowance, whatever the
/// sampling frequency and the channels, as each channel is coded within its
/// own allowance; how far the grades of other configurations stray from
/// the scale's has not been measured.
///
/// # Panics
///
/// If `level` is above [`quality::MAX_LEVEL`].
pub(crate) fn noise_offset_db(level: u8) -> f64 {
    quality::interpolate(level, &STEREO)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, ConfigError, Rate};

    /// A higher level never allows more noise.
    #[test]
    fn the_offset_falls_as_the_level_rises() {
        for pair in STEREO.windows(2) {
            assert!(pair[1] < pair[0], "{STEREO:?}");
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
            for (index, level) in CALIBRATION_LEVELS.into_iter().chain([55]).enumerate() {
                let expected = STEREO
                    .get(index)
                    .copied()
                    .unwrap_or_else(|| (STEREO[5] + STEREO[6]) / 2.0);
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
