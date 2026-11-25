// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The encoder's first psychoacoustic choices: how much noise each line of
//! a block's spectrum may take, from which the floor is set.
//!
//! The specification describes no encoder (clause 1.1.5); this model is
//! chelys's own, and deliberately simple. The noise allowed at each post of
//! the floor is the power of the spectrum around it, averaged over a sixth
//! of an octave on each side, [`MASKING_OFFSET_DB`] below it, and no less
//! than [`ABSOLUTE_NOISE`]. The floor, which is the quantizer's step, is
//! set so that rounding to it makes that noise.

use super::floor::{Floor, INVERSE_DB_TABLE, MULTIPLIER, RANGE};

/// How far below the power of the spectrum around a post the noise there
/// may be, in dB.
pub const MASKING_OFFSET_DB: f64 = 20.0;

/// The least noise power a line is allowed, at the decoder's scale of the
/// spectrum: that of the rounding of 16-bit PCM, white noise of power `1 /
/// (12 * 32768^2)`, some 101 dB below full scale, whose power in each line
/// of a long block is `4 / N` times its own, `N` the block's 2048 samples.
pub const ABSOLUTE_NOISE: f64 = 4.0 / 2048.0 / (12.0 * 32768.0 * 32768.0);

/// The half-width of the average around each post, in octaves.
const SPREAD_OCTAVES: f64 = 1.0 / 6.0;

/// The Y value of each post, in the floor's list order, of the floor for
/// the spectrum `lines`, at the decoder's scale.
pub fn floor_targets(lines: &[f64], floor: &Floor) -> Vec<i32> {
    let mut sums = Vec::with_capacity(lines.len() + 1);
    sums.push(0.0);
    for &x in lines {
        sums.push(sums.last().unwrap() + x * x);
    }
    let ratio = 2f64.powf(SPREAD_OCTAVES);
    let offset = 10f64.powf(-MASKING_OFFSET_DB / 10.0);
    floor
        .xs()
        .iter()
        .map(|&x| {
            // At least the lines beside the post's own.
            let center = (x as usize).min(lines.len() - 1);
            let low = ((center as f64 / ratio).floor() as usize).min(center.saturating_sub(1));
            let high = ((center as f64 * ratio).ceil() as usize)
                .max(center + 1)
                .min(lines.len() - 1);
            let power = (sums[high + 1] - sums[low]) / (high + 1 - low) as f64;
            let noise = (power * offset).max(ABSOLUTE_NOISE);
            // Rounding to a step `s` makes noise of power `s^2 / 12`.
            y_value((12.0 * noise).sqrt())
        })
        .collect()
}

/// The Y value whose floor is nearest `amplitude`, in dB.
fn y_value(amplitude: f64) -> i32 {
    let lowest = f64::from(INVERSE_DB_TABLE[0]);
    let step = (f64::from(INVERSE_DB_TABLE[255]) / lowest).ln() / 255.0;
    let index = (amplitude / lowest).ln() / step / f64::from(MULTIPLIER);
    (index.round() as i32).clamp(0, RANGE - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn y_values_follow_the_table() {
        for y in 0..RANGE {
            let amplitude = f64::from(INVERSE_DB_TABLE[(y * MULTIPLIER as i32) as usize]);
            assert_eq!(y_value(amplitude), y);
        }
        assert_eq!(y_value(1e-12), 0);
        assert_eq!(y_value(1e6), RANGE - 1);
    }

    #[test]
    fn silence_gets_the_absolute_floor() {
        let floor = Floor::new(0);
        let targets = floor_targets(&[0.0; 1024], &floor);
        let expected = y_value((12.0 * ABSOLUTE_NOISE).sqrt());
        assert!(targets.iter().all(|&y| y == expected));
    }
}
