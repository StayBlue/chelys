// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The window of a block (clause 4.3.1), with which a decoder laps its
//! inverse transform with the blocks beside it, and with which the encoder
//! weights the samples it transforms.

use std::f64::consts::FRAC_PI_2;

/// The window of a block of `n` samples, in a stream whose short blocks
/// are of `blocksize_0`, as clause 4.3.1 generates it: for a long block,
/// `previous_long` and `next_long` are its `previous_window_flag` and
/// `next_window_flag`; a short block takes neither.
///
/// The slopes are those of the Vorbis window, `sin(pi/2 sin^2((x + 0.5) / n
/// pi))`, whose square and that of the slope it laps with add up to 1, as
/// the transform's aliasing needs to cancel.
pub fn window(
    n: usize,
    blocksize_0: usize,
    long: bool,
    previous_long: bool,
    next_long: bool,
) -> Vec<f64> {
    let center = n / 2;
    let (left_start, left_end, left_n) = if long && !previous_long {
        (
            n / 4 - blocksize_0 / 4,
            n / 4 + blocksize_0 / 4,
            blocksize_0 / 2,
        )
    } else {
        (0, center, n / 2)
    };
    let (right_start, right_end, right_n) = if long && !next_long {
        (
            n * 3 / 4 - blocksize_0 / 4,
            n * 3 / 4 + blocksize_0 / 4,
            blocksize_0 / 2,
        )
    } else {
        (center, n, n / 2)
    };
    let slope = |i: usize, start: usize, len: usize, phase: f64| {
        let x = ((i - start) as f64 + 0.5) / len as f64 * FRAC_PI_2 + phase;
        (FRAC_PI_2 * x.sin().powi(2)).sin()
    };
    (0..n)
        .map(|i| {
            if i < left_start {
                0.0
            } else if i < left_end {
                slope(i, left_start, left_n, 0.0)
            } else if i < right_start {
                1.0
            } else if i < right_end {
                slope(i, right_start, right_n, FRAC_PI_2)
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lapping_slopes_are_power_complementary() {
        let long = window(2048, 256, true, true, true);
        for i in 0..1024 {
            let sum = long[i].powi(2) + long[i + 1024].powi(2);
            assert!((sum - 1.0).abs() < 1e-12, "{i}");
        }
        // A long block's slope toward a short block laps with the short
        // block's.
        let short = window(256, 256, false, true, true);
        let hybrid = window(2048, 256, true, true, false);
        for i in 0..128 {
            let sum = hybrid[1536 - 64 + i].powi(2) + short[i].powi(2);
            assert!((sum - 1.0).abs() < 1e-12, "{i}");
        }
        assert!(hybrid[1536 + 64..].iter().all(|&w| w == 0.0));
        assert!(hybrid[1024..1536 - 64].iter().all(|&w| w == 1.0));
    }

    #[test]
    fn the_long_window_is_the_slope_function_over_the_block() {
        let long = window(2048, 256, true, true, true);
        for (i, &w) in long.iter().enumerate() {
            let x = (i as f64 + 0.5) / 2048.0 * std::f64::consts::PI;
            let expected = (FRAC_PI_2 * x.sin().powi(2)).sin();
            assert!((w - expected).abs() < 1e-12, "{i}");
        }
    }
}
