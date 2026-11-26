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

/// Two channels, coupled, calibrated at 44.1 kHz: from 59 kbit/s on
/// average at level 0 to 102 at 50, 166 at 80 and 189 at 100, where the
/// mean grades, interpolated between the offsets swept, were -2.989,
/// -0.758, -0.122 and -0.001.
///
/// Swept, the mean grade was 0.098 at -16.5 dB, 0.126 at -15, 0.123 at
/// -13.5, 0.082 at -12, 0.007 at -10.5, -0.108 at -9, -0.333 at -7.5,
/// -0.553 at -6, -0.809 at -4.5, -1.172 at -3, -1.692 at -1.5, -2.319 at 0,
/// -2.886 at 1.5, -3.275 at 3, -3.492 at 4.5, -3.607 at 6 and -3.669 at
/// 7.5.
const STEREO: Offsets = [
    1.9, 0.3, -1.0, -2.1, -3.4, -4.8, -6.5, -7.9, -8.9, -10.0, -10.4,
];

/// Channels coded each on its own, calibrated in two channels at 44.1 kHz
/// coded so: from 71 kbit/s on average at level 0 to 117 at 50, 172 at 80
/// and 197 at 100, where the mean grades, interpolated between the offsets
/// swept, were -3.011, -0.740, -0.122 and -0.001.
///
/// Swept, the mean grade was 0.096 at -16.5 dB, 0.130 at -15, 0.134 at
/// -13.5, 0.100 at -12, 0.041 at -10.5, -0.050 at -9, -0.170 at -7.5,
/// -0.345 at -6, -0.596 at -4.5, -0.956 at -3, -1.499 at -1.5, -2.193 at 0,
/// -2.836 at 1.5, -3.273 at 3, -3.511 at 4.5, -3.629 at 6 and -3.690 at
/// 7.5.
const UNCOUPLED: Offsets = [
    2.1, 0.6, -0.6, -1.6, -2.7, -3.9, -5.2, -6.6, -8.1, -9.3, -9.8,
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
    use crate::config::Control;
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
                let Ok(Control::Variable {
                    noise_offset_db: offset,
                }) = config.control()
                else {
                    panic!("{sample_rate} Hz, {channels} channels, level {level}");
                };
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
