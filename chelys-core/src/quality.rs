// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The scale of quality that chelys's encoders share at a variable
//! bitrate: a level from 0 to [`MAX_LEVEL`], higher for better quality.
//!
//! A level stands for a grade, [`target_odg`]: the mean Objective
//! Difference Grade (ODG) of PEAQ, the basic version of ITU-R BS.1387, as
//! GstPEAQ 0.6.1 computes it, with both signals given to it at 48 kHz as
//! 16-bit PCM, over the whole of each of 51 tracks of nine kinds of music,
//! the tuning set of chelys's corpus, at 44.1 kHz in stereo. The levels
//! run from -3 ("annoying") at 0 to 0 ("imperceptible") at 100, closer
//! together in grade toward the top:
//!
//! `target_odg(level) = -3 (1 - level / 100)²`
//!
//! Each encoder maps a level onto a setting of its own, such as the noise
//! its bands are allowed beyond what its psychoacoustic model allows. The
//! setting of each of [`CALIBRATION_LEVELS`] is the one whose mean grade on
//! the tuning set is nearest the level's target, and [`interpolate`] gives
//! those between them. An encoder may calibrate other configurations, such
//! as one channel, to the same targets on the tuning set converted to them.
//! Where PEAQ grades a configuration apart, as it grades lower sampling
//! frequencies far lower at any bitrate, an encoder may take over the
//! settings of one it could calibrate.
//!
//! What is promised: within a configuration, a higher level gives a
//! stream of no lower quality, and mostly a larger one. Across encoders, a
//! level gives about the same quality at 44.1 kHz in stereo, where each is
//! calibrated, as PEAQ judges it on music like the tuning set's, and only
//! roughly so in other configurations, such as 48 kHz, where an encoder
//! takes over settings it has not measured there. A level does not give a
//! bitrate, which follows the music.
//!
//! The floor of -3: PEAQ's grade stops following the coding below about
//! -3. On 18 of the tuning set's tracks, chelys's MP3 encoder at 44.1 kHz
//! in stereo graded -2.96 at an average of 73 kbit/s, but -2.06 at 45
//! kbit/s and -1.36 at 32 kbit/s, the grade rising again as the coding
//! grew worse. An encoder that cannot reach -3 maps level 0 onto the
//! setting of its lowest mean grade.
//!
//! The square: ODG flattens as it nears 0, so that equal steps of it take
//! ever more bits. Linear in the grade, the levels from 75 to 100 would
//! hold all of -1 to 0, where most streams are coded; squared, the levels
//! from 50 to 100 do. Chelys's MP3 encoder at 44.1 kHz in stereo averages
//! 72 kbit/s at level 0, 143 at 50, 211 at 80 and 260 at 100.

/// The highest level, of the best quality.
pub const MAX_LEVEL: u8 = 100;

/// The levels an encoder calibrates a configuration at, from 0 to
/// [`MAX_LEVEL`] in steps of 10.
pub const CALIBRATION_LEVELS: [u8; 11] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

/// The mean Objective Difference Grade that `level` stands for.
///
/// # Panics
///
/// If `level` is above [`MAX_LEVEL`].
pub fn target_odg(level: u8) -> f64 {
    assert!(level <= MAX_LEVEL, "a level of 0 to {MAX_LEVEL}: {level}");
    let below = 1.0 - f64::from(level) / f64::from(MAX_LEVEL);
    -3.0 * below * below
}

/// An encoder's setting for `level`, given its setting at each of
/// [`CALIBRATION_LEVELS`]: linear between the two levels around it.
///
/// # Panics
///
/// If `level` is above [`MAX_LEVEL`].
pub fn interpolate(level: u8, settings: &[f64; CALIBRATION_LEVELS.len()]) -> f64 {
    assert!(level <= MAX_LEVEL, "a level of 0 to {MAX_LEVEL}: {level}");
    let below = CALIBRATION_LEVELS
        .iter()
        .rposition(|&calibrated| calibrated <= level)
        .expect("the lowest calibration level is 0");
    let Some(&above_level) = CALIBRATION_LEVELS.get(below + 1) else {
        return settings[below];
    };
    let below_level = CALIBRATION_LEVELS[below];
    let t = f64::from(level - below_level) / f64::from(above_level - below_level);
    settings[below] + t * (settings[below + 1] - settings[below])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_targets_run_from_annoying_to_imperceptible() {
        assert_eq!(target_odg(0), -3.0);
        assert_eq!(target_odg(50), -0.75);
        assert_eq!(target_odg(MAX_LEVEL), 0.0);
        for level in 0..MAX_LEVEL {
            assert!(target_odg(level) < target_odg(level + 1), "{level}");
        }
    }

    #[test]
    fn settings_are_interpolated_between_calibration_levels() {
        let settings = CALIBRATION_LEVELS.map(|level| 2.0 * f64::from(level) - 1.0);
        for level in 0..=MAX_LEVEL {
            let setting = interpolate(level, &settings);
            assert!(
                (setting - (2.0 * f64::from(level) - 1.0)).abs() < 1e-12,
                "{level}"
            );
        }
        let mut settings = [0.0; CALIBRATION_LEVELS.len()];
        settings[3] = 10.0;
        assert_eq!(interpolate(25, &settings), 5.0);
        assert_eq!(interpolate(30, &settings), 10.0);
        assert!((interpolate(37, &settings) - 3.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "a level of 0 to 100")]
    fn a_level_above_the_scale_panics() {
        interpolate(101, &[0.0; CALIBRATION_LEVELS.len()]);
    }
}
