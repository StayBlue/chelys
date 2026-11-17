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
    /// For each butterfly of span 2, the position whose value the bit
    /// reversal brings to its first input, `rev(2m)`; that of its second is
    /// `rev(2m + 1) = rev(2m) + N/2`.
    firsts: Vec<u32>,
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
        let firsts = (0..size / 2)
            .map(|m| ((2 * m).reverse_bits() >> (usize::BITS - bits)) as u32)
            .collect();
        let mut twiddles = Vec::with_capacity(size);
        let mut half = 1;
        while half < size {
            let stride = size / (2 * half);
            twiddles.extend((0..half).map(|k| twiddle(k * stride)));
            half *= 2;
        }
        Self { firsts, twiddles }
    }

    /// Transforms `x` in place.
    ///
    /// # Panics
    ///
    /// If `x` is not the size the transform was made for.
    #[cfg(test)]
    pub fn transform(&self, x: &mut [Complex]) {
        let input = x.to_vec();
        self.transform_from(|i| input[i], x);
    }

    /// The transform of the values `input(i)` into `x`.
    ///
    /// # Panics
    ///
    /// If `x` is not the size the transform was made for.
    pub fn transform_from(&self, input: impl Fn(usize) -> Complex, x: &mut [Complex]) {
        let n = self.twiddles.len() + 1;
        assert_eq!(x.len(), n, "FFT input size");
        // The bit reversal, with the butterflies of span 2 taken as their
        // inputs are gathered. Their twiddle is 1, by which multiplying
        // changes nothing but at most the sign of a zero, and is left out.
        for (pair, &first) in x.chunks_exact_mut(2).zip(&self.firsts) {
            let first = first as usize;
            let (even, odd) = (input(first), input(first + n / 2));
            pair[0] = Complex {
                re: even.re + odd.re,
                im: even.im + odd.im,
            };
            pair[1] = Complex {
                re: even.re - odd.re,
                im: even.im - odd.im,
            };
        }
        // Three stages at a time, of spans `2 h`, `4 h` and `8 h`, then two,
        // each value loaded and stored once for them: the same butterflies as
        // one stage after the other.
        let butterfly = |even: Complex, odd: Complex, twiddle: Complex| {
            let product = odd.mul(twiddle);
            let sum = Complex {
                re: even.re + product.re,
                im: even.im + product.im,
            };
            let difference = Complex {
                re: even.re - product.re,
                im: even.im - product.im,
            };
            (sum, difference)
        };
        let mut half = 2;
        while 8 * half <= n {
            let first = &self.twiddles[half - 1..][..half];
            let second = &self.twiddles[2 * half - 1..][..2 * half];
            let third = &self.twiddles[4 * half - 1..][..4 * half];
            for block in x.chunks_exact_mut(8 * half) {
                for k in 0..half {
                    let e: [Complex; 8] = std::array::from_fn(|j| block[k + j * half]);
                    let (e0, e1) = butterfly(e[0], e[1], first[k]);
                    let (e2, e3) = butterfly(e[2], e[3], first[k]);
                    let (e4, e5) = butterfly(e[4], e[5], first[k]);
                    let (e6, e7) = butterfly(e[6], e[7], first[k]);
                    let (e0, e2) = butterfly(e0, e2, second[k]);
                    let (e1, e3) = butterfly(e1, e3, second[half + k]);
                    let (e4, e6) = butterfly(e4, e6, second[k]);
                    let (e5, e7) = butterfly(e5, e7, second[half + k]);
                    let (e0, e4) = butterfly(e0, e4, third[k]);
                    let (e1, e5) = butterfly(e1, e5, third[half + k]);
                    let (e2, e6) = butterfly(e2, e6, third[2 * half + k]);
                    let (e3, e7) = butterfly(e3, e7, third[3 * half + k]);
                    for (j, value) in [e0, e1, e2, e3, e4, e5, e6, e7].into_iter().enumerate() {
                        block[k + j * half] = value;
                    }
                }
            }
            half *= 8;
        }
        while 4 * half <= n {
            let first = &self.twiddles[half - 1..][..half];
            let second = &self.twiddles[2 * half - 1..][..2 * half];
            for block in x.chunks_exact_mut(4 * half) {
                let (low, high) = block.split_at_mut(2 * half);
                let (a, b) = low.split_at_mut(half);
                let (c, d) = high.split_at_mut(half);
                for k in 0..half {
                    let (a1, b1) = butterfly(a[k], b[k], first[k]);
                    let (c1, d1) = butterfly(c[k], d[k], first[k]);
                    (a[k], c[k]) = butterfly(a1, c1, second[k]);
                    (b[k], d[k]) = butterfly(b1, d1, second[half + k]);
                }
            }
            half *= 4;
        }
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
    /// The complex values of the half-size transform, and the lines given,
    /// kept from one transform to the next for their memory.
    z: Vec<Complex>,
    lines: Vec<Complex>,
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
            z: Vec::with_capacity(size / 2),
            lines: Vec::with_capacity(size / 2 + 1),
        }
    }

    /// The lines `X(0)` to `X(N / 2)` of the transform of `x`; the rest are
    /// their conjugates.
    ///
    /// # Panics
    ///
    /// If `x` is not the size the transform was made for.
    pub fn transform(&mut self, x: &[f64]) -> &[Complex] {
        let half = self.twiddles.len() - 1;
        assert_eq!(x.len(), 2 * half, "FFT input size");
        let z = &mut self.z;
        z.resize(half, Complex::default());
        self.half.transform_from(
            |i| Complex {
                re: x[2 * i],
                im: x[2 * i + 1],
            },
            z,
        );
        let twiddles = &self.twiddles;
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
            let odd = odd.mul(twiddles[k]);
            Complex {
                re: even.re + odd.re,
                im: even.im + odd.im,
            }
        };
        // Z(N/2) = Z(0), so the first and last lines take Z(0) twice.
        let x = &mut self.lines;
        x.clear();
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
            let mut fft = RealFft::new(n);
            let real = fft.transform(&input);
            assert_eq!(real.len(), n / 2 + 1);
            for (k, (real, complex)) in real.iter().zip(&complex).enumerate() {
                assert!((real.re - complex.re).abs() < 1e-9, "n {n}, k {k}");
                assert!((real.im - complex.im).abs() < 1e-9, "n {n}, k {k}");
            }
        }
    }
}
