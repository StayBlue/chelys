//! The "standard forward FFT" of clause D.2.4 b): an iterative radix-2
//! decimation-in-time transform, `X(k) = sum x(n) e^(-2 pi i k n / N)`, without
//! normalisation.

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
}

pub struct Fft {
    /// `e^(-2 pi i k / N)` for `k < N / 2`.
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
        let twiddles = (0..size / 2)
            .map(|k| {
                let angle = -2.0 * PI * k as f64 / size as f64;
                Complex {
                    re: angle.cos(),
                    im: angle.sin(),
                }
            })
            .collect();
        Self { twiddles }
    }

    /// Transforms `x` in place.
    ///
    /// # Panics
    ///
    /// If `x` is not the size the transform was made for.
    pub fn transform(&self, x: &mut [Complex]) {
        let n = 2 * self.twiddles.len();
        assert_eq!(x.len(), n, "FFT input size");
        let bits = n.trailing_zeros();
        for i in 0..n {
            let j = i.reverse_bits() >> (usize::BITS - bits);
            if i < j {
                x.swap(i, j);
            }
        }
        let mut half = 1;
        while half < n {
            let stride = n / (2 * half);
            for start in (0..n).step_by(2 * half) {
                for k in 0..half {
                    let odd = x[start + half + k].mul(self.twiddles[k * stride]);
                    let even = x[start + k];
                    x[start + k] = Complex {
                        re: even.re + odd.re,
                        im: even.im + odd.im,
                    };
                    x[start + half + k] = Complex {
                        re: even.re - odd.re,
                        im: even.im - odd.im,
                    };
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
}
