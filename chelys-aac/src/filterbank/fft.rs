// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! A complex discrete Fourier transform of a power-of-two length, by the
//! radix-2 decimation in time, for the MDCT to take its lines from.

use std::f64::consts::TAU;

/// A complex number, as its real and imaginary parts.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// `e^(i * angle)`.
    pub fn unit(angle: f64) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self::new(cos, sin)
    }

    pub fn mul(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }

    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }

    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }
}

/// The DFT of one length, `X_q = sum_p x_p e^(-2 pi i p q / L)`, with its
/// twiddle factors and bit reversal precomputed.
#[derive(Debug)]
pub struct Fft {
    /// `e^(-2 pi i k / L)` for `k < L / 2`.
    twiddles: Vec<Complex>,
    /// Each index with its bits reversed.
    reversed: Vec<u32>,
}

impl Fft {
    /// # Panics
    ///
    /// If `len` is not a power of two: a caller's error.
    pub fn new(len: usize) -> Self {
        assert!(len.is_power_of_two(), "{len} is not a power of two");
        let bits = len.trailing_zeros();
        Self {
            twiddles: (0..len / 2)
                .map(|k| Complex::unit(-TAU * k as f64 / len as f64))
                .collect(),
            reversed: (0..len as u32)
                .map(|i| i.reverse_bits().checked_shr(32 - bits).unwrap_or(0))
                .collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.reversed.len()
    }

    /// Transforms `x` in place.
    ///
    /// # Panics
    ///
    /// If `x` is not of the transform's length: a caller's error.
    pub fn transform(&self, x: &mut [Complex]) {
        let len = self.len();
        assert_eq!(x.len(), len, "the transform's length");
        for (i, &j) in self.reversed.iter().enumerate() {
            let j = j as usize;
            if i < j {
                x.swap(i, j);
            }
        }
        let mut half = 1;
        while half < len {
            // The twiddles of a butterfly of 2 * half points are every
            // (len / (2 * half))th of those of the whole length.
            let stride = len / (2 * half);
            for start in (0..len).step_by(2 * half) {
                for k in 0..half {
                    let twiddle = self.twiddles[k * stride];
                    let (a, b) = (x[start + k], x[start + k + half].mul(twiddle));
                    x[start + k] = a.add(b);
                    x[start + k + half] = a.sub(b);
                }
            }
            half *= 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_definition() {
        for len in [1, 2, 4, 64, 512] {
            let x: Vec<Complex> = (0..len)
                .map(|i| Complex::new((i as f64 * 0.37).sin(), (i as f64 * 1.3).cos()))
                .collect();
            let mut fast = x.clone();
            Fft::new(len).transform(&mut fast);
            for (q, fast) in fast.iter().enumerate() {
                let direct = (x.iter().enumerate()).fold(Complex::default(), |sum, (p, &x)| {
                    sum.add(x.mul(Complex::unit(-TAU * (p * q) as f64 / len as f64)))
                });
                assert!(
                    (fast.re - direct.re).abs() < 1e-9 && (fast.im - direct.im).abs() < 1e-9,
                    "{len}: {q}"
                );
            }
        }
    }
}
