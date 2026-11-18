// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The MDCT of clause C.3.1.2:
//!
//! `X_k = 2 * sum_{n=0}^{N-1} z_n cos(2 pi / N (n + n_0)(k + 1/2))` for
//! `0 <= k < N/2`, with `n_0 = (N/2 + 1) / 2`,
//!
//! computed through a DCT-IV of `N/2` points and a complex DFT of `N/4`.
//!
//! With `M = N/2`, the cosine's argument is `pi / M (m + 1/2)(k + 1/2)`
//! for `m = n + M/2`, whose cosine is even about `m = -1/2` and odd about
//! `m = M - 1/2`. So the `N` windowed samples fold into `M`, with which the
//! sum is a DCT-IV, `Y_k = sum_{m=0}^{M-1} u_m cos(pi / M (m + 1/2)(k +
//! 1/2))`:
//!
//! `u_m = -z_{3M/2 - 1 - m} - z_{3M/2 + m}` for `0 <= m < M/2`, and
//! `u_m = z_{m - M/2} - z_{3M/2 - 1 - m}` for `M/2 <= m < M`.
//!
//! The DCT-IV takes the `M/2` complex values `v_p = (u_{2p} + i u_{M - 1 -
//! 2p}) e^(-i pi (4p + 1) / 4M)`, their DFT `V_q`, and `W_q = V_q e^(-i pi
//! q / M)`; then `Y_{2q} = Re W_q` and `Y_{M - 1 - 2q} = -Im W_q`.

use std::f64::consts::PI;

use super::fft::{Complex, Fft};

/// The MDCT of one window length, with its twiddle factors precomputed.
#[derive(Debug)]
pub struct Mdct {
    /// `e^(-i pi (4p + 1) / 4M)` for `p < M/2`.
    pre: Vec<Complex>,
    /// `2 e^(-i pi q / M)` for `q < M/2`: the twiddle after the DFT, with
    /// the MDCT's factor of 2.
    post: Vec<Complex>,
    fft: Fft,
}

impl Mdct {
    /// The MDCT of windows of `window_len` samples, `N`.
    ///
    /// # Panics
    ///
    /// If `window_len` is not a power of two of 4 at least: a caller's
    /// error.
    pub fn new(window_len: usize) -> Self {
        assert!(
            window_len.is_power_of_two() && window_len >= 4,
            "an MDCT of {window_len} samples"
        );
        let lines = window_len / 2;
        let m = lines as f64;
        Self {
            pre: (0..lines / 2)
                .map(|p| Complex::unit(-PI * (4 * p + 1) as f64 / (4.0 * m)))
                .collect(),
            post: (0..lines / 2)
                .map(|q| {
                    let w = Complex::unit(-PI * q as f64 / m);
                    Complex::new(2.0 * w.re, 2.0 * w.im)
                })
                .collect(),
            fft: Fft::new(lines / 2),
        }
    }

    /// The lines of each window, `N/2`.
    pub fn lines(&self) -> usize {
        2 * self.pre.len()
    }

    /// The lines `X_k` of the windowed samples `z`, into `lines`.
    ///
    /// # Panics
    ///
    /// If `z` is not a window's samples, or `lines` not its lines: a
    /// caller's error.
    pub fn transform(&self, z: &[f64], lines: &mut [f64]) {
        let m = self.lines();
        assert_eq!(z.len(), 2 * m, "a window's samples");
        assert_eq!(lines.len(), m, "a window's lines");
        let u = |m_: usize| {
            if m_ < m / 2 {
                -z[3 * m / 2 - 1 - m_] - z[3 * m / 2 + m_]
            } else {
                z[m_ - m / 2] - z[3 * m / 2 - 1 - m_]
            }
        };
        let mut v: Vec<Complex> = (self.pre.iter().enumerate())
            .map(|(p, &pre)| Complex::new(u(2 * p), u(m - 1 - 2 * p)).mul(pre))
            .collect();
        self.fft.transform(&mut v);
        for (q, (v, &post)) in v.iter().zip(&self.post).enumerate() {
            let w = v.mul(post);
            lines[2 * q] = w.re;
            lines[m - 1 - 2 * q] = -w.im;
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// The MDCT as clause C.3.1.2 writes it, summed directly.
    pub fn direct(z: &[f64]) -> Vec<f64> {
        let n = z.len() as f64;
        let n0 = (n / 2.0 + 1.0) / 2.0;
        (0..z.len() / 2)
            .map(|k| {
                let sum: f64 = (z.iter().enumerate())
                    .map(|(i, &z)| z * (2.0 * PI / n * (i as f64 + n0) * (k as f64 + 0.5)).cos())
                    .sum();
                2.0 * sum
            })
            .collect()
    }

    #[test]
    fn matches_the_formula_of_the_standard() {
        for window_len in [4, 16, 256, 2048] {
            let mdct = Mdct::new(window_len);
            let z: Vec<f64> = (0..window_len)
                .map(|n| (n as f64 * 0.731).sin() + 0.25 * (n as f64 * 0.05).cos())
                .collect();
            let mut lines = vec![0.0; window_len / 2];
            mdct.transform(&z, &mut lines);
            let direct = direct(&z);
            let largest = direct.iter().fold(0.0f64, |a, x| a.max(x.abs()));
            for (k, (fast, direct)) in lines.iter().zip(&direct).enumerate() {
                assert!(
                    (fast - direct).abs() <= 1e-12 * largest.max(1.0) * window_len as f64,
                    "N = {window_len}, k = {k}: {fast} against {direct}"
                );
            }
        }
    }
}
