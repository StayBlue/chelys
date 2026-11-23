// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The noise offset of each level of the [`quality`] scale, in one channel
//! and in more.
//!
//! Spec gap: neither standard has a scale of quality, nor says how much
//! noise a variable rate should allow. The offsets are calibrated with
//! PEAQ (GstPEAQ 0.6.1, black box) as [`quality`] defines the scale: those
//! of [`CALIBRATION_LEVELS`] are the offsets whose mean grade over its 51
//! tracks of nine kinds of music meets each level's target, interpolated
//! between offsets of -15 to 12 dB swept in steps of 1.5 to 3 dB, and
//! rounded to 0.1 dB.

use chelys_core::quality::{self, CALIBRATION_LEVELS};

/// Offsets in dB, for each of [`CALIBRATION_LEVELS`].
type Offsets = [f64; CALIBRATION_LEVELS.len()];

/// Two channels, calibrated at 44.1 kHz: from 65 kbit/s on average at
/// level 0 to 139 at 50, 190 at 80 and 228 at 100, where the mean grades
/// were -0.727, -0.118 and 0.002.
///
/// Swept, the mean grade was 0.060 at -15 dB, 0.047 at -12, 0.015 at -9,
/// -0.013 at -7.5, -0.057 at -6, -0.124 at -4.5, -0.231 at -3, -0.403 at
/// -1.5, -0.655 at 0, -1.066 at 1.5, -1.607 at 3, -2.178 at 4.5, -2.786 at
/// 6, -3.075 at 7.5, -3.256 at 9 and -3.549 at 12.
const STEREO: Offsets = [7.1, 5.1, 3.8, 2.6, 1.5, 0.3, -1.0, -2.7, -4.6, -6.9, -8.2];

/// One channel, calibrated at 44.1 kHz on the tracks mixed down to the
/// mean of their channels: from 41 kbit/s on average at level 0 to 80 at
/// 50, 106 at 80 and 123 at 100, where the mean grades were -0.741,
/// -0.111 and 0.002.
///
/// Swept, the mean grade was 0.068 at -15 dB, 0.059 at -12, 0.034 at -9,
/// 0.011 at -7.5, -0.025 at -6, -0.081 at -4.5, -0.171 at -3, -0.321 at
/// -1.5, -0.563 at 0, -0.982 at 1.5, -1.561 at 3, -2.131 at 4.5, -2.770 at
/// 6, -3.047 at 7.5, -3.213 at 9 and -3.495 at 12.
const MONO: Offsets = [7.2, 5.2, 3.9, 2.8, 1.8, 0.7, -0.5, -2.0, -3.9, -5.9, -7.1];

/// The noise offset, in dB, of `level` (see
/// [`Rate::Variable`](crate::Rate::Variable)) in `channels` channels: one
/// channel's table in one, and two channels' in more.
///
/// Spec gap: the other sampling frequencies take the offsets calibrated at
/// 44.1 kHz, as chelys's MP3 encoder takes its own (PEAQ grades streams at
/// 22.05 kHz and below far lower at any bitrate, and cannot judge those at
/// 11.025 kHz and below); and the configurations of three channels and
/// more those of two. An offset is a measure against the psychoacoustic
/// model's allowance, whatever the sampling frequency and the channels:
/// each channel is coded within its own allowance, a pair's channels as
/// those of two, and the grade falls steadily as the offset rises.
///
/// # Panics
///
/// If `level` is above [`quality::MAX_LEVEL`].
pub(crate) fn noise_offset_db(level: u8, channels: usize) -> f64 {
    let offsets = match channels {
        1 => &MONO,
        _ => &STEREO,
    };
    quality::interpolate(level, offsets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rate::Control;
    use crate::{Config, ConfigError, Rate};

    const TABLES: [&Offsets; 2] = [&STEREO, &MONO];

    /// A higher level never allows more noise.
    #[test]
    fn the_offset_falls_as_the_level_rises() {
        for (table, offsets) in TABLES.iter().enumerate() {
            for pair in offsets.windows(2) {
                assert!(pair[1] < pair[0], "table {table}: {offsets:?}");
            }
        }
    }

    /// A level is the variable rate of its offset, for the configuration's
    /// own table.
    #[test]
    fn a_level_lowers_to_the_variable_rate_of_its_offset() {
        let cases = [
            (44_100, 2, &STEREO),
            (44_100, 1, &MONO),
            (48_000, 1, &MONO),
            (32_000, 2, &STEREO),
            (22_050, 2, &STEREO),
            (8_000, 1, &MONO),
            (48_000, 6, &STEREO),
            (96_000, 8, &STEREO),
        ];
        for (sample_rate, channels, offsets) in cases {
            for (index, level) in CALIBRATION_LEVELS.into_iter().chain([55]).enumerate() {
                let noise_offset_db = offsets
                    .get(index)
                    .copied()
                    .unwrap_or_else(|| (offsets[5] + offsets[6]) / 2.0);
                let lower = |rate| Config::new(sample_rate, channels, rate).lower().unwrap();
                let mut quality = lower(Rate::Quality { level });
                let variable = lower(Rate::Variable { noise_offset_db });
                let context = format!("{sample_rate} Hz, {channels} channels, level {level}");
                let Control::Variable {
                    noise_offset_db: offset,
                } = quality.control
                else {
                    panic!("{context}: {quality:?}");
                };
                assert!((offset - noise_offset_db).abs() < 1e-12, "{context}");
                quality.control = variable.control;
                assert_eq!(quality, variable, "{context}");
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
