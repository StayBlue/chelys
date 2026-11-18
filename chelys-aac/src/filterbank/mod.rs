// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The encoder's filterbank (clause C.3): each frame's window spans the
//! 1024 samples of the block before and the 1024 of its own (clause
//! C.3.1.1), is shaped by its window sequence and shape (clause 15.3.2),
//! and is transformed by the MDCT (clause C.3.1.2), the inverse of the
//! decoder's IMDCT (clause 15.3.1) with its overlap and add (clause
//! 15.3.3).

mod fft;
mod mdct;

use std::sync::OnceLock;

use crate::channel::FRAME_LINES;
use crate::stream::FRAME_SAMPLES;
use crate::window::{WindowSequence, WindowShape, Windows};
use mdct::Mdct;

/// The samples of a long window, `N` (clause 15.3.1): two blocks.
const LONG_WINDOW: usize = 2 * FRAME_SAMPLES;

/// The scale of the encoder's input at the decoder's output: a sample at
/// full scale, 1.0, is 32768 there.
///
/// Spec reading: the decoder's gains are "scaled such that the integer
/// part of the output of the IMDCT can be used directly as a 16-bit PCM
/// audio output" (clause 8.3.6), whose full scale is 32768, where the
/// encoder's input is at full scale 1.0 (an `i16` sample `x` is `x /
/// 32768`). The output meant is taken to be the samples the IMDCT gives
/// once windowed and overlapped and added (clause 15.3.3), the decoder's
/// output. So the MDCT takes its input at that scale, and the quantizer
/// (clause C.7.4), the inverse of the decoder's, gives lines that decode
/// to it. Settled by decoding with ffmpeg (black box): it decodes a tone
/// at its level, within 0.1 dB, at full scale 1.0; the conformance test is
/// `a_tone_keeps_its_frequency_and_level_at_a_low_bitrate`.
const PCM_SCALE: f64 = 32_768.0;

/// The sine window of clause 15.3.2 of `N` samples:
/// `W_SIN(n) = sin(pi / N (n + 1/2))`, its left half for `0 <= n < N/2`
/// and its right half above.
fn sine_window(len: usize) -> Vec<f64> {
    (0..len)
        .map(|n| (std::f64::consts::PI / len as f64 * (n as f64 + 0.5)).sin())
        .collect()
}

/// The long window of `shape`, both halves.
fn long_window(shape: WindowShape) -> &'static [f64] {
    static SINE: OnceLock<Vec<f64>> = OnceLock::new();
    match shape {
        WindowShape::Sine => SINE.get_or_init(|| sine_window(LONG_WINDOW)),
    }
}

fn long_mdct() -> &'static Mdct {
    static MDCT: OnceLock<Mdct> = OnceLock::new();
    MDCT.get_or_init(|| Mdct::new(LONG_WINDOW))
}

/// One channel's filterbank: the block before the next frame's, and the
/// shape of the window before.
#[derive(Clone, Debug)]
pub struct Filterbank {
    /// The samples of the block before, at [`PCM_SCALE`].
    previous: Box<[f64; FRAME_SAMPLES]>,
    /// The shape of the frame before, `None` before the first frame.
    previous_shape: Option<WindowShape>,
}

impl Default for Filterbank {
    fn default() -> Self {
        Self::new()
    }
}

impl Filterbank {
    /// A filterbank before the first frame, whose block before is silent.
    ///
    /// Spec gap: the first frame's window spans "the last 1024 values of
    /// the previous window_sequence" (clause C.3.1.1), of which there are
    /// none. They are taken as silence, so that the first frame decodes
    /// that silence and the input follows it whole (see [`DELAY`](crate::DELAY)).
    pub fn new() -> Self {
        Self {
            previous: Box::new([0.0; FRAME_SAMPLES]),
            previous_shape: None,
        }
    }

    /// The spectrum of the frame whose block is `block`, the next 1024
    /// samples of the channel at full scale 1.0, with `windows`: its lines
    /// in the order they are sent (clause 8.3.5).
    ///
    /// The window's left half takes the shape of the frame before, or for
    /// the first frame its own.
    ///
    /// Spec reading: Annex C says that the first half "is constrained to
    /// use the appropriate window shape from the preceding frame" (clause
    /// C.3.1.1), and gives no rule for the first frame, which has none. The
    /// decoder's rule is taken, so that the encoder's window is the
    /// decoder's: "For the first block of the bitstream to be decoded the
    /// window_shape of the left and right half of the window are
    /// identical" (clause 15.3.2).
    pub fn spectrum(
        &mut self,
        block: &[f64; FRAME_SAMPLES],
        windows: Windows,
    ) -> [f64; FRAME_LINES] {
        let left = long_window(self.previous_shape.unwrap_or(windows.shape));
        let right = long_window(windows.shape);
        let mut z = [0.0; LONG_WINDOW];
        match windows.sequence {
            WindowSequence::OnlyLong => {
                // x'_{i,n}: the block before, then this one (clause
                // C.3.1.1), windowed (clause 15.3.2, a).
                let (before, after) = z.split_at_mut(FRAME_SAMPLES);
                for ((z, &x), &w) in before.iter_mut().zip(&*self.previous).zip(left) {
                    *z = w * x;
                }
                let right = &right[FRAME_SAMPLES..];
                for ((z, &x), &w) in after.iter_mut().zip(block).zip(right) {
                    *z = w * x * PCM_SCALE;
                }
            }
        }
        let mut lines = [0.0; FRAME_LINES];
        long_mdct().transform(&z, &mut lines);
        for (previous, &x) in self.previous.iter_mut().zip(block) {
            *previous = x * PCM_SCALE;
        }
        self.previous_shape = Some(windows.shape);
        lines
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{PI, TAU};

    use super::*;

    /// The decoder's IMDCT (clause 15.3.1), `x_{i,n} = 2/N sum_k spec[k]
    /// cos(2 pi / N (n + n_0)(k + 1/2))`, its windowing (clause 15.3.2, a)
    /// and its overlap and add (clause 15.3.3), written from the decoding
    /// clauses apart from the encoder, for an ONLY_LONG_SEQUENCE with the
    /// sine window.
    struct Decoder {
        /// The right half of the windowed IMDCT of the frame before.
        overlap: Vec<f64>,
    }

    impl Decoder {
        fn new() -> Self {
            Self {
                overlap: vec![0.0; 1024],
            }
        }

        /// The frame's 1024 output samples.
        fn decode(&mut self, spec: &[f64; 1024]) -> Vec<f64> {
            let n = 2048.0;
            let n0 = (n / 2.0 + 1.0) / 2.0;
            let window = |i: usize| (PI / n * (i as f64 + 0.5)).sin();
            let z: Vec<f64> = (0..2048)
                .map(|i| {
                    let x: f64 = (spec.iter().enumerate())
                        .map(|(k, &s)| {
                            s * (2.0 * PI / n * (i as f64 + n0) * (k as f64 + 0.5)).cos()
                        })
                        .sum();
                    window(i) * 2.0 / n * x
                })
                .collect();
            let out = (z[..1024].iter().zip(&self.overlap))
                .map(|(z, overlap)| z + overlap)
                .collect();
            self.overlap = z[1024..].to_vec();
            out
        }
    }

    fn signal(len: usize) -> Vec<f64> {
        (0..len)
            .map(|t| {
                let t = t as f64;
                0.5 * (TAU * 0.013 * t).sin() + 0.3 * (TAU * 0.31 * t + 1.0).sin()
                    - 0.1 * (t * 0.0007).cos()
            })
            .collect()
    }

    /// Time domain aliasing cancellation: the decoder's output, a frame
    /// late, is the encoder's input at the decoder's scale.
    #[test]
    fn the_decoder_reconstructs_the_input_a_frame_late() {
        let input = signal(5 * 1024);
        let mut filterbank = Filterbank::new();
        let mut decoder = Decoder::new();
        let mut output = Vec::new();
        for block in input.chunks_exact(1024) {
            let spectrum = filterbank.spectrum(block.try_into().unwrap(), Windows::LONG_SINE);
            output.extend(decoder.decode(&spectrum));
        }
        // The first frame decodes the silent block before the input.
        assert!(output[..1024].iter().all(|x| x.abs() < 1e-6));
        for (t, (&x, &y)) in input.iter().zip(&output[1024..]).enumerate() {
            assert!(
                (x * PCM_SCALE - y).abs() < 1e-6,
                "sample {t}: {x} against {y}"
            );
        }
    }

    /// The spectrum is the MDCT of clause C.3.1.2, summed directly, of the
    /// windowed blocks at the decoder's scale.
    #[test]
    fn the_spectrum_is_the_mdct_of_the_windowed_blocks() {
        let input = signal(2048);
        let mut filterbank = Filterbank::new();
        filterbank.spectrum(input[..1024].try_into().unwrap(), Windows::LONG_SINE);
        let spectrum = filterbank.spectrum(input[1024..].try_into().unwrap(), Windows::LONG_SINE);
        let window = sine_window(2048);
        let z: Vec<f64> = (input.iter().zip(&window))
            .map(|(x, w)| x * w * PCM_SCALE)
            .collect();
        let direct = mdct::tests::direct(&z);
        for (k, (x, y)) in spectrum.iter().zip(&direct).enumerate() {
            assert!((x - y).abs() < 1e-6 * y.abs().max(1.0), "line {k}");
        }
    }

    /// A tone at the centre of line `k` lands in it and its neighbours.
    #[test]
    fn a_tone_lands_in_its_line() {
        let k = 100;
        let input: Vec<f64> = (0..3 * 1024)
            .map(|t| (PI / 1024.0 * (k as f64 + 0.5) * t as f64).cos())
            .collect();
        let mut filterbank = Filterbank::new();
        let mut spectrum = [0.0; 1024];
        for block in input.chunks_exact(1024) {
            spectrum = filterbank.spectrum(block.try_into().unwrap(), Windows::LONG_SINE);
        }
        let energy =
            |range: std::ops::Range<usize>| -> f64 { spectrum[range].iter().map(|x| x * x).sum() };
        let share = energy(k - 1..k + 2) / energy(0..1024);
        assert!(share > 0.98, "{share}");
    }
}
