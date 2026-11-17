// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The "standard forward FFT" of clause D.2.4 b): an iterative radix-2
//! decimation-in-time transform, `X(k) = sum x(n) e^(-2 pi i k n / N)`, without
//! normalisation, and its form for real input, [`RealFft`].

use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

pub struct Fft {
    /// The pairs of positions that the bit reversal swaps.
    swaps: Vec<(u32, u32)>,
    /// For each stage, from the butterflies of span 2 up, the twiddles
    /// `e^(-2 pi i k / (2 h))` of its `h` butterflies, one after the other.
    twiddles: Vec<Complex>,
}

impl Fft {
    /// # Panics
    ///
    /// If `size` is not a power of two of at least 2.
    pub fn new(size: usize) -> Self {
        assert!(
            size >= 2 && size.is_power_of_two(),
            "FFT size {size} is not a power of two of at least 2"
        );
        let twiddle = |k: usize| {
            let angle = -2.0 * PI * k as f64 / size as f64;
            Complex {
                re: angle.cos(),
                im: angle.sin(),
            }
        };
        let bits = size.trailing_zeros();
        let swaps = (0..size)
            .map(|i| (i, i.reverse_bits() >> (usize::BITS - bits)))
            .filter(|&(i, j)| i < j)
            .map(|(i, j)| (i as u32, j as u32))
            .collect();
        let mut twiddles = Vec::with_capacity(size);
        let mut half = 1;
        while half < size {
            let stride = size / (2 * half);
            twiddles.extend((0..half).map(|k| twiddle(k * stride)));
            half *= 2;
        }
        Self { swaps, twiddles }
    }

    /// Transforms `x` in place.
    ///
    /// # Panics
    ///
    /// If `x` is not the size the transform was made for.
    pub fn transform(&self, x: &mut [Complex]) {
        let n = self.twiddles.len() + 1;
        assert_eq!(x.len(), n, "FFT input size");
        for &(i, j) in &self.swaps {
            x.swap(i as usize, j as usize);
        }
        // The butterflies of span 2 have the twiddle 1, by which multiplying
        // changes nothing but at most the sign of a zero, and is left out.
        for pair in x.chunks_exact_mut(2) {
            let (even, odd) = (pair[0], pair[1]);
            pair[0] = Complex {
                re: even.re + odd.re,
                im: even.im + odd.im,
            };
            pair[1] = Complex {
                re: even.re - odd.re,
                im: even.im - odd.im,
            };
        }
        let mut half = 2;
        while half < n {
            let twiddles = &self.twiddles[half - 1..][..half];
            for block in x.chunks_exact_mut(2 * half) {
                let (evens, odds) = block.split_at_mut(half);
                for ((even, odd), twiddle) in evens.iter_mut().zip(odds).zip(twiddles) {
                    let product = odd.mul(*twiddle);
                    let sum = Complex {
                        re: even.re + product.re,
                        im: even.im + product.im,
                    };
                    *odd = Complex {
                        re: even.re - product.re,
                        im: even.im - product.im,
                    };
                    *even = sum;
                }
            }
            half *= 2;
        }
    }
}

/// The FFT of real input of `N` samples, through the FFT of `N / 2`
/// complex ones: the even samples as their real parts and the odd ones as
/// their imaginary parts, giving `Z(k) = E(k) + i O(k)`, where `E` and `O`
/// are the transforms of the even and odd samples. Since those samples are
/// real, `E(k) = (Z(k) + conj(Z(N/2 - k))) / 2` and
/// `O(k) = (Z(k) - conj(Z(N/2 - k))) / 2i`, and
/// `X(k) = E(k) + e^(-2 pi i k / N) O(k)`, with `Z(N/2) = Z(0)`.
pub struct RealFft {
    half: Fft,
    /// `e^(-2 pi i k / N)` for `k <= N / 2`.
    twiddles: Vec<Complex>,
}

impl RealFft {
    /// # Panics
    ///
    /// If `size` is not a power of two of at least 4.
    pub fn new(size: usize) -> Self {
        assert!(
            size >= 4 && size.is_power_of_two(),
            "FFT size {size} is not a power of two of at least 4"
        );
        let twiddles = (0..=size / 2)
            .map(|k| {
                let angle = -2.0 * PI * k as f64 / size as f64;
                Complex {
                    re: angle.cos(),
                    im: angle.sin(),
                }
            })
            .collect();
        Self {
            half: Fft::new(size / 2),
            twiddles,
        }
    }

    /// The lines `X(0)` to `X(N / 2)` of the transform of `x`; the rest are
    /// their conjugates.
    ///
    /// # Panics
    ///
    /// If `x` is not the size the transform was made for.
    pub fn transform(&self, x: &[f64]) -> Vec<Complex> {
        let half = self.twiddles.len() - 1;
        assert_eq!(x.len(), 2 * half, "FFT input size");
        let mut z: Vec<Complex> = x
            .chunks_exact(2)
            .map(|pair| Complex {
                re: pair[0],
                im: pair[1],
            })
            .collect();
        self.half.transform(&mut z);
        let line = |k: usize, zk: Complex, mirrored: Complex| {
            let mirrored = mirrored.conj();
            let even = Complex {
                re: (zk.re + mirrored.re) / 2.0,
                im: (zk.im + mirrored.im) / 2.0,
            };
            // (zk - mirrored) / 2i
            let odd = Complex {
                re: (zk.im - mirrored.im) / 2.0,
                im: -(zk.re - mirrored.re) / 2.0,
            };
            let odd = odd.mul(self.twiddles[k]);
            Complex {
                re: even.re + odd.re,
                im: even.im + odd.im,
            }
        };
        // Z(N/2) = Z(0), so the first and last lines take Z(0) twice.
        let mut x = Vec::with_capacity(half + 1);
        x.push(line(0, z[0], z[0]));
        x.extend((1..half).map(|k| line(k, z[k], z[half - k])));
        x.push(line(half, z[0], z[0]));
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_direct_dft() {
        for n in [2, 8, 256] {
            let input: Vec<Complex> = (0..n)
                .map(|t| Complex {
                    re: ((t * 7919) % 31) as f64 - 15.0,
                    im: ((t * 104_729) % 17) as f64 - 8.0,
                })
                .collect();
            let mut fast = input.clone();
            Fft::new(n).transform(&mut fast);
            for (k, fast) in fast.iter().enumerate() {
                let mut direct = Complex::default();
                for (t, x) in input.iter().enumerate() {
                    let angle = -2.0 * PI * (k * t) as f64 / n as f64;
                    let term = x.mul(Complex {
                        re: angle.cos(),
                        im: angle.sin(),
                    });
                    direct.re += term.re;
                    direct.im += term.im;
                }
                assert!((fast.re - direct.re).abs() < 1e-9, "n {n}, k {k}");
                assert!((fast.im - direct.im).abs() < 1e-9, "n {n}, k {k}");
            }
        }
    }

    #[test]
    fn real_input_matches_the_complex_transform() {
        for n in [4, 8, 256, 1024] {
            let input: Vec<f64> = (0..n).map(|t| ((t * 7919) % 31) as f64 - 15.0).collect();
            let mut complex: Vec<Complex> =
                (input.iter()).map(|&re| Complex { re, im: 0.0 }).collect();
            Fft::new(n).transform(&mut complex);
            let real = RealFft::new(n).transform(&input);
            assert_eq!(real.len(), n / 2 + 1);
            for (k, (real, complex)) in real.iter().zip(&complex).enumerate() {
                assert!((real.re - complex.re).abs() < 1e-9, "n {n}, k {k}");
                assert!((real.im - complex.im).abs() < 1e-9, "n {n}, k {k}");
            }
        }
    }
}
