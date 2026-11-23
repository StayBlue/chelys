// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The noise offset of each level of the [`quality`] scale, in one channel
//! and in two.
//!
//! Spec gap: neither standard has a scale of quality, nor says how much
//! noise a variable bitrate should allow. The offsets are calibrated with
//! PEAQ (GstPEAQ 0.6.1, black box) as [`quality`] defines the scale: those
//! of [`CALIBRATION_LEVELS`] are the offsets whose mean grade over its 51
//! tracks of nine kinds of music meets each level's target, interpolated
//! between offsets of -6 to 12 dB swept in steps of 1.5 to 3 dB, and
//! rounded to 0.1 dB.

use chelys_core::quality::{self, CALIBRATION_LEVELS};

/// Offsets in dB, for each of [`CALIBRATION_LEVELS`].
type Offsets = [f64; CALIBRATION_LEVELS.len()];

/// Two channels, calibrated at 44.1 kHz in joint stereo: from 72 kbit/s
/// on average at level 0 to 143 at 50, 211 at 80 and 260 at 100.
///
/// No offset reaches level 0's grade of -3: the mean grade falls to -2.98
/// at 12 dB, and below that PEAQ stops judging (see [`quality`]), so level
/// 0 is 12 dB. Swept, the mean grade was 0.041 at -6 dB, 0.026 at -3,
/// -0.024 at 0, -0.081 at 1.5, -0.160 at 3, -0.286 at 4.5, -0.467 at 6,
/// -0.894 at 7.5, -1.678 at 9, -2.752 at 10.5 and -2.978 at 12.
const STEREO: Offsets = [12.0, 10.0, 9.3, 8.6, 7.8, 7.0, 6.0, 4.3, 2.2, 0.1, -1.5];

/// One channel, calibrated at 44.1 kHz on the tracks mixed down to the
/// mean of their channels: from 42 kbit/s on average at level 0 to 85 at
/// 50, 112 at 80 and 128 at 100.
///
/// As in two channels, no offset reaches -3; the mean grade falls to
/// -2.91 at 9 dB, which is level 0. Swept, the mean grade was 0.089 at
/// -3 dB, -0.016 at 0, -0.113 at 1.5, -0.260 at 3, -0.524 at 4.5, -1.291
/// at 6, -2.793 at 7.5, -2.912 at 9 and -2.384 at 12.
const MONO: Offsets = [9.0, 7.1, 6.6, 6.2, 5.6, 4.9, 4.3, 3.1, 1.6, 0.2, -0.5];

/// The noise offset, in dB, of `level` (see
/// [`VariableRate`](crate::rate::VariableRate)) in `channels` channels.
///
/// Spec gap: the lower sampling frequencies, and MPEG-2.5, take the offsets
/// calibrated at 44.1 kHz. PEAQ grades every stream at 22.05 kHz far below
/// the same stream's grade at 44.1 kHz would be, and no offset there meets
/// the grades of levels 50 to 100: over the tracks converted to 22.05 kHz
/// in two channels, the mean grade was -1.031 at -3 dB (142 kbit/s), -1.491
/// at 0, -2.145 at 3, -2.846 at 6, -3.439 at 9 and -3.770 at 12 (32 kbit/s).
/// Calibrated to the targets, half the scale would be one offset. An offset
/// is a measure against the psychoacoustic model's allowance, whatever the
/// sampling frequency, and the grade still falls steadily as it rises.
/// MPEG-2.5 is not graded at all, as PEAQ cannot judge it.
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
    use crate::{ChannelMode, Config, Rate};

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
            (44_100, ChannelMode::Stereo, &STEREO),
            (32_000, ChannelMode::DualChannel, &STEREO),
            (48_000, ChannelMode::Mono, &MONO),
            (22_050, ChannelMode::Stereo, &STEREO),
            (16_000, ChannelMode::Mono, &MONO),
            (11_025, ChannelMode::Stereo, &STEREO),
        ];
        for (sample_rate, mode, offsets) in cases {
            for (index, level) in CALIBRATION_LEVELS.into_iter().chain([55]).enumerate() {
                let quality = Rate::Quality {
                    level,
                    min_kbps: None,
                    max_kbps: Some(128),
                };
                let noise_offset_db = offsets
                    .get(index)
                    .copied()
                    .unwrap_or_else(|| (offsets[5] + offsets[6]) / 2.0);
                let variable = Rate::Variable {
                    min_kbps: None,
                    max_kbps: Some(128),
                    noise_offset_db,
                };
                let lower = |rate| Config::new(sample_rate, mode, rate).lower().unwrap();
                let (quality, variable) = (lower(quality), lower(variable));
                let crate::rate::StreamRate::Variable(rate) = quality.rate else {
                    panic!("{quality:?}");
                };
                let crate::rate::StreamRate::Variable(expected) = variable.rate else {
                    panic!("{variable:?}");
                };
                assert!(
                    (rate.noise_offset_db - expected.noise_offset_db).abs() < 1e-12
                        && rate.lowest == expected.lowest
                        && rate.highest == expected.highest,
                    "{sample_rate} Hz {mode:?}, level {level}"
                );
            }
        }
    }
}
