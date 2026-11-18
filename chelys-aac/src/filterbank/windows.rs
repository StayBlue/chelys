// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The window shapes of clause 15.3.2: the sine window and the
//! Kaiser-Bessel derived (KBD) window, of a long window's 2048 samples or a
//! short window's 256.

use std::f64::consts::PI;
use std::sync::OnceLock;

use crate::scalefactor_bands::Transform;
use crate::window::WindowShape;

/// The window of `shape` of the length `N` of `transform`'s windows, both
/// halves: `W_LEFT,N(n)` for `0 <= n < N/2` and `W_RIGHT,N(n)` above.
pub fn window(shape: WindowShape, transform: Transform) -> &'static [f64] {
    static WINDOWS: [OnceLock<Vec<f64>>; 4] = [const { OnceLock::new() }; 4];
    let len = 2 * transform.lines();
    let index = 2 * shape.number() as usize + usize::from(transform == Transform::Short);
    WINDOWS[index].get_or_init(|| match shape {
        WindowShape::Sine => sine(len),
        WindowShape::Kbd => kbd(len),
    })
}

/// The sine window of `N` samples: `W_SIN_LEFT,N(n) = W_SIN_RIGHT,N(n) =
/// sin(pi / N (n + 1/2))`.
fn sine(len: usize) -> Vec<f64> {
    (0..len)
        .map(|n| (PI / len as f64 * (n as f64 + 0.5)).sin())
        .collect()
}

/// The kernel window alpha factor `alpha` of a KBD window of `N` samples:
/// 4 for `N = 2048` and 6 for `N = 256`.
fn alpha(len: usize) -> f64 {
    match len {
        2048 => 4.0,
        _ => 6.0,
    }
}

/// `I_0[x] = sum_{k=0}^{inf} [(x/2)^k / k!]^2`, the zeroth order modified
/// Bessel function of the first kind, summed until its terms no longer
/// change the sum.
fn bessel_i0(x: f64) -> f64 {
    let mut term = 1.0f64;
    let mut sum = 1.0;
    for k in 1.. {
        term *= x / 2.0 / f64::from(k);
        let squared = term * term;
        if sum + squared == sum {
            break;
        }
        sum += squared;
    }
    sum
}

/// The KBD window of `N` samples:
///
/// `W_KBD_LEFT,N(n) = sqrt(sum_{p=0}^{n} W'(p, alpha) / sum_{p=0}^{N/2}
/// W'(p, alpha))` for `0 <= n < N/2`,
///
/// `W_KBD_RIGHT,N(n) = sqrt(sum_{p=0}^{N-n-1} W'(p, alpha) /
/// sum_{p=0}^{N/2} W'(p, alpha))` for `N/2 <= n < N`,
///
/// with the Kaiser-Bessel kernel `W'(n, alpha) = I_0[pi alpha sqrt(1.0 -
/// ((n - N/4) / (N/4))^2)] / I_0[pi alpha]` for `0 <= n <= N/2`.
fn kbd(len: usize) -> Vec<f64> {
    let quarter = len as f64 / 4.0;
    let alpha = alpha(len);
    let denominator = bessel_i0(PI * alpha);
    // sum_{p=0}^{n} W'(p, alpha) for 0 <= n <= N/2.
    let mut cumulative = Vec::with_capacity(len / 2 + 1);
    let mut sum = 0.0;
    for n in 0..=len / 2 {
        let r = (n as f64 - quarter) / quarter;
        sum += bessel_i0(PI * alpha * (1.0 - r * r).sqrt()) / denominator;
        cumulative.push(sum);
    }
    let total = cumulative[len / 2];
    (0..len)
        .map(|n| {
            let top = if n < len / 2 { n } else { len - n - 1 };
            (cumulative[top] / total).sqrt()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHAPES: [WindowShape; 2] = [WindowShape::Sine, WindowShape::Kbd];
    const TRANSFORMS: [Transform; 2] = [Transform::Long, Transform::Short];

    /// The halves that overlap cancel each other's aliasing: `W(n)^2 +
    /// W(n + N/2)^2 = 1`, the window of each shape symmetric, so that a
    /// left half of either shape meets a right half of the same shape.
    #[test]
    fn windows_meet_the_princen_bradley_condition() {
        for shape in SHAPES {
            for transform in TRANSFORMS {
                let w = window(shape, transform);
                let half = w.len() / 2;
                assert_eq!(w.len(), 2 * transform.lines());
                for n in 0..half {
                    let sum = w[n] * w[n] + w[n + half] * w[n + half];
                    assert!((sum - 1.0).abs() < 1e-12, "{shape:?} {transform:?} {n}");
                    assert!((w[n] - w[w.len() - 1 - n]).abs() < 1e-15);
                }
                assert!(w[..half].windows(2).all(|p| p[0] < p[1]));
            }
        }
    }

    #[test]
    fn the_bessel_function_sums_its_series() {
        assert_eq!(bessel_i0(0.0), 1.0);
        // I_0(1) = 1 + 1/4 + 1/64 + 1/2304 + ...
        assert!((bessel_i0(1.0) - 1.266_065_877_752_008_4).abs() < 1e-15);
        // I_0(x) for large x approaches e^x / sqrt(2 pi x) from above, by
        // a factor of about 1 + 1/(8x).
        let x = PI * 6.0;
        let asymptote = x.exp() / (2.0 * PI * x).sqrt() * (1.0 + 1.0 / (8.0 * x));
        assert!((bessel_i0(x) / asymptote - 1.0).abs() < 1e-3);
    }

    /// The KBD window rises from its ends more slowly than the sine
    /// window, and nears 1 sooner: nearer 0 in the first eighth of the
    /// window, and nearer 1 in the eighth before its middle.
    #[test]
    fn the_kbd_window_is_more_compact_than_the_sine() {
        for transform in TRANSFORMS {
            let (sine, kbd) = (
                window(WindowShape::Sine, transform),
                window(WindowShape::Kbd, transform),
            );
            let n = sine.len();
            for i in 0..=n / 8 {
                assert!(kbd[i] < sine[i], "{transform:?} {i}");
            }
            for i in 3 * n / 8..n / 2 {
                assert!(kbd[i] > sine[i], "{transform:?} {i}");
            }
        }
    }
}
