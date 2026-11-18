// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The encoder's filterbank (clause C.3): each frame's window spans the
//! 1024 samples of the block before and the 1024 of its own (clause
//! C.3.1.1), is shaped by its window sequence and shape (clause 15.3.2),
//! and is transformed by the MDCT (clause C.3.1.2), the inverse of the
//! decoder's IMDCT (clause 15.3.1) with its overlap and add (clause
//! 15.3.3): once over the whole window, or once over each of the eight
//! short windows of an EIGHT_SHORT_SEQUENCE.

mod mdct;
pub mod windows;

use crate::channel::FRAME_LINES;
use crate::sampling_frequency::SamplingFrequency;
use crate::scalefactor_bands::{GroupBands, Transform};
use crate::stream::FRAME_SAMPLES;
use crate::window::{SHORT_WINDOWS, WindowSequence, WindowShape, Windows};
use mdct::Mdct;
use windows::window;

/// The samples of a long window, `N` (clause 15.3.1): two blocks.
const LONG_WINDOW: usize = 2 * FRAME_SAMPLES;
/// The samples of a short window, `N`.
const SHORT_WINDOW: usize = 256;
/// The lines of a short window.
const SHORT_LINES: usize = SHORT_WINDOW / 2;
/// The first sample of the first short window in the window of an
/// EIGHT_SHORT_SEQUENCE, each of the others 128 samples after the one
/// before (clause 15.3.2, c), where its zeros before it end.
pub const FIRST_SHORT: usize = 448;
/// Where the short slope of a LONG_START_WINDOW starts, and that of the
/// short windows after it ends (clause 15.3.2, b).
const START_SLOPE: usize = FIRST_SHORT + LONG_WINDOW / 2;

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
pub const PCM_SCALE: f64 = 32_768.0;

/// One channel's filterbank: its transforms, the block before the next
/// frame's, and the windows of the frame before.
#[derive(Debug)]
pub struct Filterbank {
    long: Mdct,
    short: Mdct,
    /// The bands of one short window, by which the lines of a group of
    /// short windows are interleaved.
    short_bands: GroupBands,
    /// The samples of the block before, at [`PCM_SCALE`].
    previous: Box<[f64; FRAME_SAMPLES]>,
    /// The windows of the frame before, `None` before the first frame.
    previous_windows: Option<Windows>,
    // Buffers kept from one frame to the next for their memory.
    /// `x'_{i,n}` (clause C.3.1.1): the block before, then this one.
    samples: Box<[f64; LONG_WINDOW]>,
    /// The windowed samples of one transform.
    windowed: Box<[f64; LONG_WINDOW]>,
    /// The lines of the eight short windows, one window after another.
    by_window: Box<[f64; FRAME_LINES]>,
}

impl Filterbank {
    /// A filterbank at `sampling_frequency` before the first frame, whose
    /// block before is silent.
    ///
    /// Spec gap: the first frame's window spans "the last 1024 values of
    /// the previous window_sequence" (clause C.3.1.1), of which there are
    /// none. They are taken as silence, so that the first frame decodes
    /// that silence and the input follows it whole (see [`DELAY`](crate::DELAY)).
    pub fn new(sampling_frequency: SamplingFrequency) -> Self {
        Self {
            long: Mdct::new(LONG_WINDOW),
            short: Mdct::new(SHORT_WINDOW),
            short_bands: GroupBands::short(sampling_frequency, 1),
            previous: Box::new([0.0; FRAME_SAMPLES]),
            previous_windows: None,
            samples: Box::new([0.0; LONG_WINDOW]),
            windowed: Box::new([0.0; LONG_WINDOW]),
            by_window: Box::new([0.0; FRAME_LINES]),
        }
    }

    /// The spectrum of the frame whose block is `block`, the next 1024
    /// samples of the channel at full scale 1.0, with `windows`, into
    /// `lines`, in the order they are sent (clause 8.3.5): for a long
    /// window in ascending order, and for eight short ones window group by
    /// window group, and within each group scalefactor band by band, the
    /// band's lines of each window of the group in turn (clause C.8.4).
    ///
    /// The window's left half takes the shape of the frame before, or for
    /// the first frame its own; its form, long or short, is that of the
    /// sequence's own left half. The encoder keeps to the meaningful
    /// transitions (see [`WindowSequence::follows`]), so that it is the
    /// form of the right half of the frame before.
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
        lines: &mut [f64; FRAME_LINES],
    ) {
        debug_assert!(
            (self.previous_windows).is_none_or(|p| windows.sequence.follows(p.sequence)),
            "{windows:?} after {:?}",
            self.previous_windows
        );
        let left_shape = self.previous_windows.map_or(windows.shape, |p| p.shape);
        // x'_{i,n}: the block before, then this one (clause C.3.1.1).
        let (before, after) = self.samples.split_at_mut(FRAME_SAMPLES);
        before.copy_from_slice(&*self.previous);
        for (x, &s) in after.iter_mut().zip(block) {
            *x = s * PCM_SCALE;
        }
        self.previous.copy_from_slice(after);
        self.previous_windows = Some(windows);

        match windows.sequence {
            WindowSequence::EightShort { grouping } => {
                self.eight_short(left_shape, windows.shape);
                // Interleave the windows' lines, group by group (clause
                // C.8.4).
                let bands = self.short_bands;
                let mut out = lines.iter_mut();
                let mut first = 0;
                for len in grouping.lengths() {
                    for sfb in 0..bands.num_swb() {
                        let band = bands.offset(sfb)..bands.offset(sfb + 1);
                        for w in first..first + len {
                            let window = &self.by_window[w * SHORT_LINES..][..SHORT_LINES];
                            // The band's lines first, so that the zip
                            // takes no line past them.
                            for (&x, out) in window[band.clone()].iter().zip(&mut out) {
                                *out = x;
                            }
                        }
                    }
                    first += len;
                }
            }
            sequence => {
                self.long_window(sequence, left_shape, windows.shape);
                self.long.transform(&*self.windowed, lines);
            }
        }
    }

    /// Windows the samples with the long window of `sequence` (clause
    /// 15.3.2, a, b and d), its left half of the shape `left` and its right
    /// half of `right`.
    fn long_window(&mut self, sequence: WindowSequence, left: WindowShape, right: WindowShape) {
        let half = FRAME_SAMPLES;
        let long_left = &window(left, Transform::Long)[..half];
        let long_right = &window(right, Transform::Long)[half..];
        let short_left = &window(left, Transform::Short)[..SHORT_LINES];
        let short_right = &window(right, Transform::Short)[SHORT_LINES..];
        let (z, x) = (&mut *self.windowed, &*self.samples);
        let (z_left, z_right) = z.split_at_mut(half);
        let (x_left, x_right) = x.split_at(half);
        // The left half: W_LEFT,2048, or zeros, W_LEFT,256 and ones.
        match sequence {
            WindowSequence::OnlyLong | WindowSequence::LongStart => {
                multiply(z_left, x_left, long_left);
            }
            _ => {
                let (zeros, rest) = z_left.split_at_mut(FIRST_SHORT);
                zeros.fill(0.0);
                let (slope, ones) = rest.split_at_mut(SHORT_LINES);
                multiply(slope, &x_left[FIRST_SHORT..], short_left);
                ones.copy_from_slice(&x_left[FIRST_SHORT + SHORT_LINES..]);
            }
        }
        // The right half: W_RIGHT,2048, or ones, W_RIGHT,256 and zeros.
        match sequence {
            WindowSequence::OnlyLong | WindowSequence::LongStop => {
                multiply(z_right, x_right, long_right);
            }
            _ => {
                let (ones, rest) = z_right.split_at_mut(START_SLOPE - half);
                ones.copy_from_slice(&x_right[..START_SLOPE - half]);
                let (slope, zeros) = rest.split_at_mut(SHORT_LINES);
                multiply(slope, &x_right[START_SLOPE - half..], short_right);
                zeros.fill(0.0);
            }
        }
    }

    /// Transforms the samples with the eight short windows of an
    /// EIGHT_SHORT_SEQUENCE (clause 15.3.2, c), into
    /// [`Self::by_window`]: window `j` spans the samples from `448 + 128 j`,
    /// with the window `W_0` of the left half of the shape `left` and the
    /// right half of `right`, and `W_1` to `W_7` of `right` alone.
    fn eight_short(&mut self, left: WindowShape, right: WindowShape) {
        let first = window(left, Transform::Short);
        let rest = window(right, Transform::Short);
        for j in 0..SHORT_WINDOWS {
            let start = FIRST_SHORT + j * SHORT_LINES;
            let x = &self.samples[start..start + SHORT_WINDOW];
            let z = &mut self.windowed[..SHORT_WINDOW];
            let (z_left, z_right) = z.split_at_mut(SHORT_LINES);
            let left = if j == 0 { first } else { rest };
            multiply(z_left, &x[..SHORT_LINES], &left[..SHORT_LINES]);
            multiply(z_right, &x[SHORT_LINES..], &rest[SHORT_LINES..]);
            let lines = &mut self.by_window[j * SHORT_LINES..][..SHORT_LINES];
            self.short.transform(z, lines);
        }
    }
}

/// `z(n) = x(n) * w(n)`.
fn multiply(z: &mut [f64], x: &[f64], w: &[f64]) {
    for ((z, &x), &w) in z.iter_mut().zip(x).zip(w) {
        *z = x * w;
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{PI, TAU};

    use super::*;
    use crate::window::Grouping;

    /// The decoder of clause 15, written from the decoding clauses apart
    /// from the encoder: the lines of each window taken from their order in
    /// `spectral_data()` (clause 8.3.5), the IMDCT (clause 15.3.1), `x_{i,n}
    /// = 2/N sum_k spec[i][k] cos(2 pi / N (n + n_0)(k + 1/2))`, the
    /// windows and the overlap and add within an EIGHT_SHORT_SEQUENCE
    /// (clause 15.3.2), and the overlap and add with the frame before
    /// (clause 15.3.3).
    struct Decoder {
        fs: SamplingFrequency,
        /// The right half of the windowed samples of the frame before.
        overlap: Vec<f64>,
        previous_shape: Option<WindowShape>,
    }

    /// `I_0(x) = sum_k ((x/2)^k / k!)^2`, to 100 terms.
    fn bessel(x: f64) -> f64 {
        let mut term = 1.0f64;
        let mut sum = 1.0;
        for k in 1..100 {
            term *= x / 2.0 / k as f64;
            sum += term * term;
        }
        sum
    }

    /// `W_LEFT,N(n)` or `W_RIGHT,N(n)` of `shape` (clause 15.3.2), as the
    /// decoder computes them.
    fn decoder_window(shape: WindowShape, n_len: usize, n: usize) -> f64 {
        static WINDOWS: [std::sync::OnceLock<Vec<f64>>; 4] =
            [const { std::sync::OnceLock::new() }; 4];
        let index = 2 * usize::from(shape == WindowShape::Kbd) + usize::from(n_len == 256);
        let window = WINDOWS[index].get_or_init(|| {
            let big_n = n_len as f64;
            match shape {
                WindowShape::Sine => (0..n_len)
                    .map(|n| (PI / big_n * (n as f64 + 0.5)).sin())
                    .collect(),
                WindowShape::Kbd => {
                    let alpha = if n_len == 2048 { 4.0 } else { 6.0 };
                    let kernel = |p: usize| {
                        let r = (p as f64 - big_n / 4.0) / (big_n / 4.0);
                        bessel(PI * alpha * (1.0 - r * r).sqrt()) / bessel(PI * alpha)
                    };
                    let sum = |to: usize| (0..=to).map(kernel).sum::<f64>();
                    let total = sum(n_len / 2);
                    (0..n_len)
                        .map(|n| {
                            let top = if n < n_len / 2 { n } else { n_len - n - 1 };
                            (sum(top) / total).sqrt()
                        })
                        .collect()
                }
            }
        });
        window[n]
    }

    fn imdct(spec: &[f64]) -> Vec<f64> {
        let lines = spec.len();
        // cos(2 pi / N (n + n_0)(k + 1/2)) at n * N/2 + k, for each N.
        static COSINES: [std::sync::OnceLock<Vec<f64>>; 2] =
            [const { std::sync::OnceLock::new() }; 2];
        let cosines = COSINES[usize::from(lines == 128)].get_or_init(|| {
            let n = 2.0 * lines as f64;
            let n0 = (n / 2.0 + 1.0) / 2.0;
            (0..2 * lines * lines)
                .map(|i| {
                    let (i, k) = ((i / lines) as f64, (i % lines) as f64);
                    (2.0 * PI / n * (i + n0) * (k + 0.5)).cos()
                })
                .collect()
        });
        (cosines.chunks_exact(lines))
            .map(|cosines| {
                let sum: f64 = spec.iter().zip(cosines).map(|(s, c)| s * c).sum();
                2.0 / (2 * lines) as f64 * sum
            })
            .collect()
    }

    impl Decoder {
        fn new(fs: SamplingFrequency) -> Self {
            Self {
                fs,
                overlap: vec![0.0; 1024],
                previous_shape: None,
            }
        }

        /// The frame's 1024 output samples.
        fn decode(&mut self, spec: &[f64; 1024], windows: Windows) -> Vec<f64> {
            let shape = windows.shape;
            let previous = self.previous_shape.unwrap_or(shape);
            self.previous_shape = Some(shape);
            let left = |n_len: usize, n: usize| decoder_window(previous, n_len, n);
            let right = |n_len: usize, n: usize| decoder_window(shape, n_len, n);
            let z: Vec<f64> = match windows.sequence {
                WindowSequence::EightShort { grouping } => {
                    // De-interleave (clause 8.3.5): groups in order, within
                    // each a band of every window of the group in turn.
                    let offsets = crate::scalefactor_bands::swb_offsets(self.fs, Transform::Short);
                    let mut by_window = [[0.0; 128]; 8];
                    let mut i = 0;
                    let mut first = 0;
                    for len in grouping.lengths() {
                        for band in offsets.windows(2) {
                            for window in &mut by_window[first..first + len] {
                                let band = usize::from(band[0])..usize::from(band[1]);
                                for line in &mut window[band] {
                                    *line = spec[i];
                                    i += 1;
                                }
                            }
                        }
                        first += len;
                    }
                    let mut z = vec![0.0; 2048];
                    for (j, spec) in by_window.iter().enumerate() {
                        let x = imdct(spec);
                        for (n, x) in x.iter().enumerate() {
                            let w = match (j, n < 128) {
                                (0, true) => left(256, n),
                                (_, true) => right(256, n),
                                (_, false) => right(256, n),
                            };
                            z[448 + 128 * j + n] += x * w;
                        }
                    }
                    z
                }
                sequence => {
                    let x = imdct(spec);
                    (x.iter().enumerate())
                        .map(|(n, x)| {
                            let w = match (sequence, n) {
                                (WindowSequence::LongStop, 0..448) => 0.0,
                                (WindowSequence::LongStop, 448..576) => left(256, n - 448),
                                (WindowSequence::LongStop, 576..1024) => 1.0,
                                (_, 0..1024) => left(2048, n),
                                (WindowSequence::LongStart, 1024..1472) => 1.0,
                                (WindowSequence::LongStart, 1472..1600) => {
                                    right(256, n + 128 - 1472)
                                }
                                (WindowSequence::LongStart, _) => 0.0,
                                _ => right(2048, n),
                            };
                            x * w
                        })
                        .collect()
                }
            };
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

    /// Codes `input` with `windows` frame by frame and decodes it: the
    /// decoder's output, a frame late, is the input at the decoder's scale.
    fn reconstructs(fs: SamplingFrequency, input: &[f64], windows: &[Windows]) {
        let mut filterbank = Filterbank::new(fs);
        let mut decoder = Decoder::new(fs);
        let mut output = Vec::new();
        let mut spectrum = [0.0; 1024];
        for (block, &windows) in input.chunks_exact(1024).zip(windows) {
            filterbank.spectrum(block.try_into().unwrap(), windows, &mut spectrum);
            output.extend(decoder.decode(&spectrum, windows));
        }
        // The first frame decodes the silent block before the input.
        assert!(output[..1024].iter().all(|x| x.abs() < 1e-6));
        for (t, (&x, &y)) in input.iter().zip(&output[1024..]).enumerate() {
            assert!(
                (x * PCM_SCALE - y).abs() < 1e-6,
                "sample {t} of {windows:?}: {x} against {y}"
            );
        }
    }

    /// Time domain aliasing cancellation across every meaningful
    /// transition of clause 15.3.1, with either shape and both, in every
    /// grouping of the short windows and at sampling frequencies of every
    /// short band table.
    #[test]
    fn the_decoder_reconstructs_the_input_across_every_transition() {
        use WindowSequence::*;
        let short = |bits: u8| EightShort {
            grouping: Grouping::from_starts(std::array::from_fn(|w| bits & (1 << w) != 0)),
        };
        // Every transition: OL-OL, OL-LS, LS-ES, ES-ES, ES-LStop,
        // LStop-LS, LS-LStop, LStop-OL, starting with eight short windows.
        let sequences = [
            short(0),
            short(0xff),
            LongStop,
            OnlyLong,
            OnlyLong,
            LongStart,
            short(0b0101_0101),
            short(0b1000_1001),
            LongStop,
            LongStart,
            LongStop,
            OnlyLong,
        ];
        let input = signal(sequences.len() * 1024);
        let shapes: [&dyn Fn(usize) -> WindowShape; 3] =
            [&|_| WindowShape::Sine, &|_| WindowShape::Kbd, &|i| {
                if i % 3 == 1 {
                    WindowShape::Kbd
                } else {
                    WindowShape::Sine
                }
            }];
        for fs in [
            SamplingFrequency::Hz44100,
            SamplingFrequency::Hz8000,
            SamplingFrequency::Hz96000,
        ] {
            for shape in shapes {
                let windows: Vec<Windows> = (sequences.iter().enumerate())
                    .map(|(i, &sequence)| Windows {
                        sequence,
                        shape: shape(i),
                    })
                    .collect();
                reconstructs(fs, &input, &windows);
            }
        }
    }

    /// The spectrum of a long window is the MDCT of clause C.3.1.2, summed
    /// directly, of the windowed blocks at the decoder's scale.
    #[test]
    fn the_spectrum_is_the_mdct_of_the_windowed_blocks() {
        let input = signal(2048);
        let mut filterbank = Filterbank::new(SamplingFrequency::Hz44100);
        let mut spectrum = [0.0; 1024];
        filterbank.spectrum(
            input[..1024].try_into().unwrap(),
            Windows::LONG_SINE,
            &mut spectrum,
        );
        filterbank.spectrum(
            input[1024..].try_into().unwrap(),
            Windows::LONG_SINE,
            &mut spectrum,
        );
        let z: Vec<f64> = (input.iter().enumerate())
            .map(|(n, x)| x * decoder_window(WindowShape::Sine, 2048, n) * PCM_SCALE)
            .collect();
        let direct = mdct::tests::direct(&z);
        for (k, (x, y)) in spectrum.iter().zip(&direct).enumerate() {
            assert!((x - y).abs() < 1e-6 * y.abs().max(1.0), "line {k}");
        }
    }

    /// A tone at the centre of a line lands in it and its neighbours, in a
    /// long window and in each short one.
    #[test]
    fn a_tone_lands_in_its_line() {
        let fs = SamplingFrequency::Hz48000;
        let energy = |lines: &[f64]| -> f64 { lines.iter().map(|x| x * x).sum() };
        let tone = |lines: f64, k: usize| -> Vec<f64> {
            (0..3 * 1024)
                .map(|t| (PI / lines * (k as f64 + 0.5) * t as f64).cos())
                .collect()
        };
        let input = tone(1024.0, 100);
        let mut filterbank = Filterbank::new(fs);
        let mut spectrum = [0.0; 1024];
        for block in input.chunks_exact(1024) {
            filterbank.spectrum(block.try_into().unwrap(), Windows::LONG_SINE, &mut spectrum);
        }
        let share = energy(&spectrum[99..102]) / energy(&spectrum);
        assert!(share > 0.98, "{share}");

        // Line 50 of each short window, with no grouping.
        let input = tone(128.0, 50);
        let mut filterbank = Filterbank::new(fs);
        let windows = [
            Windows::LONG_SINE,
            Windows {
                sequence: WindowSequence::LongStart,
                shape: WindowShape::Kbd,
            },
            Windows {
                sequence: WindowSequence::EIGHT_SHORT,
                shape: WindowShape::Kbd,
            },
        ];
        for (block, windows) in input.chunks_exact(1024).zip(windows) {
            filterbank.spectrum(block.try_into().unwrap(), windows, &mut spectrum);
        }
        for w in 0..8 {
            let window = &spectrum[w * 128..][..128];
            let share = energy(&window[49..52]) / energy(window);
            assert!(share > 0.9, "window {w}: {share}");
        }
    }

    /// The lines of a group of short windows are interleaved band by band.
    #[test]
    fn short_windows_are_interleaved_within_their_groups() {
        let fs = SamplingFrequency::Hz44100;
        let input = signal(2048);
        let mut ungrouped = [0.0; 1024];
        let mut grouped = [0.0; 1024];
        let grouping = Grouping::from_starts([true, false, false, true, true, true, true, false]);
        for (windows, lines) in [(Grouping::NONE, &mut ungrouped), (grouping, &mut grouped)] {
            let mut filterbank = Filterbank::new(fs);
            let windows = Windows {
                sequence: WindowSequence::EightShort { grouping: windows },
                shape: WindowShape::Sine,
            };
            filterbank.spectrum(input[..1024].try_into().unwrap(), windows, lines);
            filterbank.spectrum(input[1024..].try_into().unwrap(), windows, lines);
        }
        // Table 46: band 1 holds lines 4 to 7. The first group, of windows
        // 0 to 2, sends band 0 of each window, then band 1 of each.
        assert_eq!(grouped[..4], ungrouped[..4]);
        assert_eq!(grouped[4..8], ungrouped[128..132]);
        assert_eq!(grouped[8..12], ungrouped[256..260]);
        assert_eq!(grouped[12..16], ungrouped[4..8]);
        // The groups of one window, 3 to 5, are in ascending order.
        assert_eq!(grouped[3 * 128..6 * 128], ungrouped[3 * 128..6 * 128]);
        // The last group, of windows 6 and 7, ends with band 13 (lines 112
        // to 127) of window 7.
        assert_eq!(grouped[1024 - 16..], ungrouped[1024 - 16..]);
    }
}
