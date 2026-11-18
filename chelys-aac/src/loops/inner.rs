// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The quantization of a channel's lines (clause C.7.4) and the inner
//! iteration loop (clause C.7.4.5, figure C.8), which sets
//! `common_scalefac` so that the lines fit the bits available.

use crate::channel::{self, CodedChannel, CodedGroup, FRAME_LINES};
use crate::noiseless::codebooks::MAX_MAGNITUDE;
use crate::noiseless::scalefactors;
use crate::noiseless::sections::{self, Sections};
use crate::raw_data_block::element::COMMON_WINDOW;
use crate::sampling_frequency::SamplingFrequency;
use crate::scalefactor_bands::{GroupBands, MAX_BANDS};
use crate::window::Windows;

/// `SF_OFFSET` (clauses 11.3.3 and C.7.4): `sf_decoder = common_scalefac
/// - scalefactor + SF_OFFSET`.
const SF_OFFSET: i32 = 100;

/// `MAGIC_NUMBER` (clause C.7.4), which the quantizer adds before
/// truncating.
const MAGIC_NUMBER: f64 = 0.4054;

/// `MAX_QUANT` (clause C.7.2.1): "the maximum quantized value which can be
/// encoded in the bitstream, defined as 8191".
const MAX_QUANT: f64 = MAX_MAGNITUDE as f64;

/// `quantizer_change` of the first run of the inner loop (clause
/// C.7.4.5).
const FIRST_CHANGE: i32 = 32;

/// A magnitude that no codebook codes, at which the quantizer stops.
const BEYOND_MAGNITUDE: f64 = 2.0 * MAX_QUANT;

/// The scale of `x_quant = int(mdct_scaled * 2^(-3/16 * common_scalefac)
/// + MAGIC_NUMBER)` (clause C.7.4.2) at the `global_gain` it is sent with,
/// `common_scalefac + SF_OFFSET` (see [`quantize`]), from a table of every
/// `global_gain`.
fn step_scale(global_gain: u8) -> f64 {
    static TABLE: std::sync::OnceLock<[f64; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        std::array::from_fn(|gain| (-3.0 / 16.0 * f64::from(gain as i32 - SF_OFFSET)).exp2())
    });
    table[usize::from(global_gain)]
}

/// `|x|^(3/4)`, as `(|x| * |x|^(1/2))^(1/2)`: two square roots, which the
/// processor computes, rather than a call to the C library's `pow`. It
/// differs from `pow` by about the last bit.
fn power_three_quarters(x: f64) -> f64 {
    let x = x.abs();
    (x * x.sqrt()).sqrt()
}

/// A channel's lines with the powers the quantizer takes of them, taken
/// once for every step tried.
pub struct Lines<'a> {
    spectrum: &'a [f64; FRAME_LINES],
    /// `abs(mdct_line)^(3/4)`: `mdct_scaled` of clause C.7.4.1 with every
    /// scalefactor 0.
    scaled: [f64; FRAME_LINES],
    /// One past the last line that is not zero: those above quantize to
    /// zero at every step.
    end: usize,
}

impl<'a> Lines<'a> {
    pub fn new(spectrum: &'a [f64; FRAME_LINES]) -> Self {
        Self {
            spectrum,
            scaled: spectrum.map(power_three_quarters),
            end: spectrum
                .iter()
                .rposition(|&x| x != 0.0)
                .map_or(0, |i| i + 1),
        }
    }

    /// `start_common_scalefac = ceiling(16/3 * log2(max_mdct_line^(3/4) /
    /// MAX_QUANT))` (clause C.7.2.1), as the `global_gain` it is sent with,
    /// within its range: the smallest step at which every line can be
    /// coded.
    ///
    /// Spec gap: for lines whose largest is below about 0.005 (at the
    /// decoder's scale), the formula gives a step below the range of
    /// `global_gain`, `common_scalefac` -100, which clause C.7.2.1 does not
    /// provide for. The step is raised to the range's, the finest the
    /// bitstream can send: the lines are coded with fewer levels than
    /// `MAX_QUANT`, as many as that step gives them.
    fn start_gain(&self) -> u8 {
        let largest = self.scaled.iter().fold(0.0f64, |a, &x| a.max(x));
        let start = (16.0 / 3.0 * (largest / MAX_QUANT).log2()).ceil();
        (start + f64::from(SF_OFFSET)).clamp(0.0, 255.0) as u8
    }

    /// Quantizes the lines with the step of `global_gain` (clause
    /// C.7.4.5, `quantize_spectrum`) into `x_quant`, carrying the sign of
    /// each line.
    ///
    /// `x_quant(i) = int(mdct_scaled(i) * 2^(-3/16 * common_scalefac) +
    /// MAGIC_NUMBER)`, where `int` truncates. Magnitudes beyond
    /// [`MAX_MAGNITUDE`] are left beyond it, for counting to refuse.
    ///
    /// Spec reading: clause C.7.4 gives the quantizer first as `int((abs(
    /// mdct_line) * 2^(-1/4 * (sf_decoder - SF_OFFSET)))^(3/4) +
    /// MAGIC_NUMBER)`, the inverse of the decoder's, then, with
    /// `sf_decoder = common_scalefac - scalefactor + SF_OFFSET`, as
    /// `2^(-1/4 * (scalefactor - common_scalefac))`, whose sign is the
    /// opposite of what the substitution gives, and of clauses C.7.4.1,
    /// C.7.4.2 and C.7.4.5, and of the text after it ("a positive change
    /// increases the magnitude of x_quant"). The second form is taken as
    /// misprinted, and the first and clause C.7.4.2 are followed. (The
    /// distortion of clause C.7.4.3 has the same sign as the second.)
    fn quantize(&self, global_gain: u8, x_quant: &mut [i32; FRAME_LINES]) {
        let scale = step_scale(global_gain);
        let end = self.end;
        let lines = (x_quant[..end].iter_mut())
            .zip(&self.scaled)
            .zip(self.spectrum);
        for ((q, &scaled), &x) in lines {
            let magnitude = (scaled * scale + MAGIC_NUMBER).min(BEYOND_MAGNITUDE) as i32;
            *q = if x < 0.0 { -magnitude } else { magnitude };
        }
        x_quant[end..].fill(0);
    }
}

/// The coding of a channel's lines that the loop last counted.
struct Counting<'a> {
    lines: Lines<'a>,
    windows: Windows,
    sampling_frequency: SamplingFrequency,
    available_bits: u32,
    /// The lines quantized at the step counted last.
    x_quant: [i32; FRAME_LINES],
    /// Each group's lines counted at the step counted last, kept to reuse
    /// their memory.
    counted: Vec<sections::Counted>,
    /// The sections chosen for each group at the step counted last, if
    /// it fits.
    sections: Vec<Sections>,
}

impl Counting<'_> {
    /// Quantizes the lines with the step of `global_gain`, sending the
    /// scalefactor bands below `band_limit` at most, and counts their bits
    /// as the bitstream codes them (`bit_count()`, clause C.7.4.5): whether
    /// they fit the bits available. A line that no codebook codes does not.
    ///
    /// Every scalefactor is `global_gain`: `sf_decoder = common_scalefac -
    /// scalefactor + SF_OFFSET` with each `scalefactor` 0 (clause C.7.2.1),
    /// where `global_gain` is "the value of the first scalefactor" (clause
    /// 11.2.2). So the bits counted are those the channel is written in.
    fn fits(&mut self, global_gain: u8, band_limit: usize) -> bool {
        self.lines.quantize(global_gain, &mut self.x_quant);
        if band_limit < MAX_BANDS {
            let mut start = 0;
            for bands in self.windows.groups(self.sampling_frequency) {
                let unsent = bands.offset(band_limit.min(bands.num_swb()));
                self.x_quant[start + unsent..start + bands.lines()].fill(0);
                start += bands.lines();
            }
        }
        let max_sfb = self
            .group_lines()
            .map(|(bands, lines)| sections::max_sfb(lines, bands))
            .max()
            .unwrap_or(0);
        let mut least_bits = channel::bits(self.windows, COMMON_WINDOW, 0);
        self.sections.clear();
        self.counted.clear();
        let mut start = 0;
        for bands in self.windows.groups(self.sampling_frequency) {
            let lines = &self.x_quant[start..start + bands.lines()];
            start += bands.lines();
            let Some(group) = Sections::count(lines, bands, max_sfb) else {
                return false;
            };
            least_bits += group.least_bits();
            if least_bits > self.available_bits {
                return false;
            }
            self.counted.push(group);
        }
        (self.sections).extend(self.counted.drain(..).map(sections::Counted::choose));
        let flat = [global_gain; MAX_BANDS];
        let scalefactor_bits = scalefactors::bits(
            global_gain,
            (self.sections.iter()).map(|sections| (sections, &flat[..max_sfb])),
        )
        .expect("equal scalefactors differ by 0");
        let noiseless: u32 = (self.sections.iter())
            .map(|s| s.section_data_bits() + s.spectral_data_bits())
            .sum();
        channel::bits(self.windows, COMMON_WINDOW, noiseless + scalefactor_bits)
            <= self.available_bits
    }

    /// Each group's bands with its quantized lines.
    fn group_lines(&self) -> impl Iterator<Item = (GroupBands, &[i32])> {
        let mut start = 0;
        self.windows
            .groups(self.sampling_frequency)
            .map(move |bands| {
                let lines = &self.x_quant[start..start + bands.lines()];
                start += bands.lines();
                (bands, lines)
            })
    }

    /// The channel of the lines and sections counted last, at
    /// `global_gain`, which fitted.
    fn into_channel(self, global_gain: u8) -> CodedChannel {
        let max_sfb = self.sections.first().map_or(0, Sections::max_sfb);
        let groups = (self.sections.into_iter())
            .map(|sections| CodedGroup {
                sections,
                scalefactors: vec![global_gain; max_sfb],
            })
            .collect();
        CodedChannel::new(
            self.windows,
            self.sampling_frequency,
            global_gain,
            self.x_quant,
            groups,
        )
        .expect("equal scalefactors differ by 0")
    }
}

/// The inner iteration loop (clause C.7.4.5): quantizes `spectrum`, a
/// channel's lines in the order they are sent, with every scalefactor 0,
/// and searches `common_scalefac` by successive approximation from
/// `start_common_scalefac` (clause C.7.2.1) for the smallest step whose
/// coding fits in `available_bits`, the bits of the channel's
/// `individual_channel_stream()`.
///
/// Spec reading: the pseudo code of clause C.7.4.5 changes
/// `common_scalefac` after each count, also after the last, which fits,
/// and so ends with it a step below the quantization it counted last, at
/// a step it has not counted, which with bits that fall as the step grows
/// is mostly one that does not fit. Figure C.8 adds `quantizer_change` to
/// `common_scalefac` only where it is not 0, and so ends at the step
/// counted last. The figure is followed: the quantization counted last is
/// the one coded, with the `common_scalefac` it was made with, so that the
/// bits coded are those counted. The pseudo code is followed for the rest,
/// as the figure does not give the values of `quantizer_change` or the
/// rule that keeps it 1 while the bits do not fit. With the steps of 32,
/// 16, 8, 4 and 2 that come before, and bits that fall as the step grows,
/// the step coded is the smallest that fits or the one above it.
///
/// Spec reading: successive approximation assumes that the bits fall as
/// the step grows, and they need not: the noiseless coding of the lines
/// can cost more at a coarser step. Where they do not fall, the search can
/// stop at a step above a finer one that fits, and it is followed as
/// written. For example, at 48 kHz in mono at 6375 bit/s, the third frame
/// of `0.8 * cos(pi * 73 * n / 1024 + 0.9)` costs 70 bits at `global_gain`
/// 167 and 168, 71 at 169 to 172 and 69 at 173; with about 70 bits
/// available it is coded at 173, though 167 and 168 fit.
///
/// Spec reading: `common_scalefac` "must not become less than
/// start_common_scalefac" (clause C.7.2.1), so a change that would take it
/// lower takes it there instead. Where the first count, at the start,
/// fits, every later step is the start again, so the loop ends there.
/// Likewise a change that would take `global_gain` past 255, the largest
/// scalefactor (clause 11.3.2), takes it there instead.
///
/// Spec gap: Annex C does not say what to do with lines whose coding does
/// not fit even at the largest step that `global_gain` allows, as from
/// samples far beyond full scale at the lowest bitrates. At that step,
/// the highest scalefactor bands are left unsent, one at a time, until the
/// rest fit; the channel without any fits, as the lowest bitrate is that
/// of frames of channels without lines (see
/// [`StreamConfig::min_bits_per_second`](crate::stream::StreamConfig::min_bits_per_second)).
///
/// # Panics
///
/// If not even a channel without lines fits in `available_bits`: a
/// caller's error.
pub fn inner_loop(
    spectrum: &[f64; FRAME_LINES],
    windows: Windows,
    sampling_frequency: SamplingFrequency,
    available_bits: u32,
) -> CodedChannel {
    let lines = Lines::new(spectrum);
    let start = lines.start_gain();
    let mut counting = Counting {
        lines,
        windows,
        sampling_frequency,
        available_bits,
        x_quant: [0; FRAME_LINES],
        counted: Vec::new(),
        sections: Vec::new(),
    };
    let all_bands = MAX_BANDS;
    let mut gain = i32::from(start);
    let mut change = FIRST_CHANGE;
    loop {
        let counted = gain as u8;
        let fits = counting.fits(counted, all_bands);
        if fits && counted == start {
            return counting.into_channel(counted);
        }
        if !fits && counted == u8::MAX {
            break;
        }
        gain = if fits { gain - change } else { gain + change };
        gain = gain.clamp(i32::from(start), i32::from(u8::MAX));
        change /= 2;
        if change == 0 && !fits {
            change = 1;
        }
        if change == 0 {
            return counting.into_channel(counted);
        }
    }
    for band_limit in (0..all_bands).rev() {
        if counting.fits(u8::MAX, band_limit) {
            return counting.into_channel(u8::MAX);
        }
    }
    panic!("not even a channel without lines fits in {available_bits} bits");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spectrum(f: impl FnMut(usize) -> f64) -> [f64; FRAME_LINES] {
        std::array::from_fn(f)
    }

    #[test]
    fn the_quantizer_inverts_the_decoders_requantization() {
        // At sf_decoder = global_gain, a line decodes to x_quant^(4/3) *
        // 2^(0.25 * (global_gain - 100)) (clauses 10.3 and 11.3.3).
        let lines = spectrum(|i| (i as f64 * 37.5).powf(1.3) * if i % 2 == 0 { 1.0 } else { -1.0 });
        let quantizer = Lines::new(&lines);
        let mut x_quant = [0; FRAME_LINES];
        // From the start step, at which the largest line is codable, up.
        assert_eq!(quantizer.start_gain(), 110);
        for gain in [110, 140, 200] {
            quantizer.quantize(gain, &mut x_quant);
            let step = (0.25 * f64::from(i32::from(gain) - 100)).exp2();
            for (&x, &q) in lines.iter().zip(&x_quant) {
                let decoded = f64::from(q).abs().powf(4.0 / 3.0) * step * f64::from(q.signum());
                // Within the quantizer's step at that magnitude.
                let next = (f64::from(q.abs()) + 1.0).powf(4.0 / 3.0) * step;
                assert!(
                    (x - decoded).abs() <= next - decoded.abs(),
                    "{gain}: {x} as {q}"
                );
                assert!(q == 0 || q.signum() == x.signum() as i32);
            }
        }
    }

    #[test]
    fn the_quantizer_rounds_up_from_0_5946() {
        // int(x + 0.4054): 0.5946 is the least fraction that rounds up.
        let lines = spectrum(|i| match i {
            0 => 0.5945f64.powf(4.0 / 3.0),
            1 => 0.5947f64.powf(4.0 / 3.0),
            2 => 1.5947f64.powf(4.0 / 3.0),
            _ => 0.0,
        });
        let mut x_quant = [0; FRAME_LINES];
        // global_gain 100: common_scalefac 0, a step of 1.
        Lines::new(&lines).quantize(100, &mut x_quant);
        assert_eq!(x_quant[..4], [0, 1, 2, 0]);
    }

    #[test]
    fn the_start_step_codes_the_largest_line() {
        for largest in [1e-3, 1.0, 1234.5, 3.2e7, 6.0e12] {
            let lines = spectrum(|i| if i == 7 { -largest } else { largest / 3.0 });
            let quantizer = Lines::new(&lines);
            let start = quantizer.start_gain();
            let mut x_quant = [0; FRAME_LINES];
            quantizer.quantize(start, &mut x_quant);
            assert!(x_quant[7].unsigned_abs() <= MAX_MAGNITUDE, "{largest}");
            if start > 0 {
                // One step finer, the largest line is beyond MAX_QUANT
                // before the quantizer rounds it.
                quantizer.quantize(start - 1, &mut x_quant);
                assert!(x_quant[7].unsigned_abs() >= MAX_MAGNITUDE, "{largest}");
            }
        }
    }

    /// At the clamp of the encoder's input, 65536 times full scale, no line
    /// is above `2 * 32768 * 65536 * sum_n w(n)` (clause C.3.1.2), with
    /// the sum of the sine window `2048 * 2 / pi`; the first step for such
    /// a line leaves the loop most of the range of `global_gain`.
    #[test]
    fn the_clamp_leaves_the_start_step_within_global_gain() {
        let bound = 2.0 * 32_768.0 * 65_536.0 * 2048.0 * 2.0 / std::f64::consts::PI;
        let lines = spectrum(|i| if i == 0 { bound } else { 0.0 });
        assert_eq!(Lines::new(&lines).start_gain(), 201);
    }

    fn noise(seed: u64) -> [f64; FRAME_LINES] {
        let mut state = seed;
        spectrum(|i| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let uniform = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            uniform * 1e6 / (1.0 + i as f64 / 50.0)
        })
    }

    #[test]
    fn the_loop_finds_the_smallest_step_that_fits_or_the_next() {
        let fs = SamplingFrequency::Hz44100;
        for (seed, available) in [(1, 100), (2, 400), (3, 1500), (4, 4000), (5, 12_000)] {
            let lines = noise(seed);
            let channel = inner_loop(&lines, Windows::LONG_SINE, fs, available);
            assert!(channel.bits(COMMON_WINDOW) <= available, "{available}");
            let mut counting = Counting {
                lines: Lines::new(&lines),
                windows: Windows::LONG_SINE,
                sampling_frequency: fs,
                available_bits: available,
                x_quant: [0; FRAME_LINES],
                counted: Vec::new(),
                sections: Vec::new(),
            };
            let start = counting.lines.start_gain();
            let smallest = (start..=u8::MAX)
                .find(|&gain| counting.fits(gain, MAX_BANDS))
                .unwrap();
            let gain = channel.global_gain();
            assert!(
                (smallest..=smallest + 1).contains(&gain),
                "{available}: {gain}, {smallest}"
            );
            assert!(gain > start + 1, "{available}: the loop ran");
        }
    }

    #[test]
    fn plenty_of_bits_leave_the_start_step() {
        let fs = SamplingFrequency::Hz48000;
        let lines = noise(2);
        let channel = inner_loop(&lines, Windows::LONG_SINE, fs, 1 << 20);
        assert_eq!(channel.global_gain(), Lines::new(&lines).start_gain());
    }

    #[test]
    fn lines_too_loud_for_any_step_lose_their_highest_bands() {
        let fs = SamplingFrequency::Hz8000;
        let lines = spectrum(|i| if i % 2 == 0 { 1e14 } else { -1e14 });
        let silent = CodedChannel::silent(Windows::LONG_SINE, fs).bits(COMMON_WINDOW);
        for available in [silent, silent + 30, 300] {
            let channel = inner_loop(&lines, Windows::LONG_SINE, fs, available);
            assert!(channel.bits(COMMON_WINDOW) <= available);
            assert_eq!(channel.global_gain(), u8::MAX);
            assert!(channel.max_sfb() < 40);
        }
    }
}
