// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The encoder's temporal noise shaping (clause C.5): for each window, the
//! linear prediction of its lines over frequency, the decision to filter
//! them, the quantization of the filter's reflection coefficients, and the
//! filtering of the lines with the inverse of the decoder's filter.

use std::f64::consts::FRAC_PI_2;
use std::ops::Range;

use super::{Filter, MAX_ORDER, NoiseGains, Tns, WindowFilters, max_bands, max_order};
use crate::channel::FRAME_LINES;
use crate::sampling_frequency::SamplingFrequency;
use crate::scalefactor_bands::{
    Transform, deinterleave, first_band_above, interleave, swb_offsets,
};
use crate::window::{SHORT_WINDOWS, Windows};

/// The lowest frequency of the target range of the filter: "A suitable
/// choice is to cover a frequency range from 1.5 kHz to the uppermost
/// possible scalefactor band with one filter" (clause C.5).
const TARGET_FROM_HZ: f64 = 1500.0;

/// The lowest frequency of the lines the linear prediction is computed
/// from: "For better stability, coefficients corresponding to frequencies
/// below 2.5 kHz may be excluded from this process" (clause C.5), which
/// they are.
const PREDICTION_FROM_HZ: f64 = 2500.0;

/// The threshold `t` that the prediction gain must exceed for the lines to
/// be filtered: "A suitable threshold value is t = 1.4" (clause C.5).
const GAIN_THRESHOLD: f64 = 1.4;

/// The threshold `p` below which reflection coefficients are removed from
/// the tail: "A suitable threshold for truncation is p = 0.1" (clause C.5).
const TRUNCATION_THRESHOLD: f64 = 0.1;

/// `coef_res`, the bits of resolution of the reflection coefficients: "An
/// appropriate choice for coef_res is 4 bits" (clause C.5).
const COEF_RES: u8 = 4;

/// The memory the encoder's temporal noise shaping works in.
#[derive(Debug)]
pub struct Shaping {
    /// The lines of eight short windows, window by window.
    by_window: Box<[f64; FRAME_LINES]>,
}

impl Default for Shaping {
    fn default() -> Self {
        Self {
            by_window: Box::new([0.0; FRAME_LINES]),
        }
    }
}

impl Shaping {
    /// Shapes the quantization noise of a channel's lines `spectrum` of
    /// `windows`, in the order they are sent, as clause C.5 does each
    /// window's: filters the lines of each window whose prediction gain
    /// exceeds the threshold, over the target bands (see [`target_bands`]),
    /// of the windows that `active` lets it shape; gives the filters as the
    /// channel's `tns_data()` sends them, and in `gains` how far the
    /// decoder's filters raise the noise of the lines they span.
    pub fn shape(
        &mut self,
        windows: Windows,
        sampling_frequency: SamplingFrequency,
        top: Option<usize>,
        spectrum: &mut [f64; FRAME_LINES],
        active: [bool; SHORT_WINDOWS],
        gains: &mut NoiseGains,
    ) -> Tns {
        let transform = windows.sequence.transform();
        let range = TargetRange::new(sampling_frequency, transform, top);
        let mut tns = Tns::new(transform);
        *gains = NoiseGains::NONE;
        match transform {
            Transform::Long => {
                if active[0] {
                    let (filters, gain) = range.shape(spectrum);
                    if filters.n_filt > 0 {
                        gains.set(0, gain, range.bands.clone());
                    }
                    *tns.window_mut(0) = filters;
                }
            }
            Transform::Short => {
                let groups = windows.groups(sampling_frequency);
                deinterleave(groups.clone(), &spectrum[..], &mut self.by_window[..]);
                let lines = transform.lines();
                for (w, window) in self.by_window.chunks_exact_mut(lines).enumerate() {
                    if active[w] {
                        let (filters, gain) = range.shape(window);
                        if filters.n_filt > 0 {
                            gains.set(w, gain, range.bands.clone());
                        }
                        *tns.window_mut(w) = filters;
                    }
                }
                if tns.is_present() {
                    interleave(groups, &self.by_window[..], &mut spectrum[..]);
                }
            }
        }
        tns
    }
}

/// The scalefactor bands of each window of `transform` at
/// `sampling_frequency` that its filter spans, its target range (clause
/// C.5): from the first at 1.5 kHz or above to `TNS_MAX_BANDS`, or to the
/// band `top` where that is lower.
///
/// Spec gap: clause C.5 gives the range's frequencies, not its lines. A
/// band is taken to lie at its lowest line's frequency, `k fs / (2 N)`:
/// the range starts at the first band whose lowest line is at 1.5 kHz or
/// above, and the linear prediction takes the range's lines from the first
/// at 2.5 kHz or above.
///
/// Spec reading: the range ends below `TNS_MAX_BANDS` (clause 7.1.6), "the
/// uppermost possible scalefactor band", as the decoder filters no band
/// above it (clause 14.3). Nor does it filter a band at or above `max_sfb`,
/// which the encoder does not know when it shapes the lines; the filter
/// runs upward, so the lines below `max_sfb` decode alike, and those above
/// are not sent. `top` cuts the range of long windows only: a short
/// window's `n_filt`, 1 bit, cannot send the filter of order 0 above it
/// (see `TargetRange::shape`).
pub fn target_bands(
    sampling_frequency: SamplingFrequency,
    transform: Transform,
    top: Option<usize>,
) -> Range<usize> {
    TargetRange::new(sampling_frequency, transform, top).bands
}

/// The scalefactor bands of a window that its filter spans, and the lines
/// the linear prediction is computed from.
struct TargetRange {
    transform: Transform,
    offsets: &'static [u16],
    /// The first band and the band after the last.
    bands: Range<usize>,
    /// The first line of the linear prediction.
    prediction_from: usize,
    /// `TNS_MAX_BANDS`.
    max_bands: usize,
}

impl TargetRange {
    /// The target range of each window of `transform` at
    /// `sampling_frequency` (see [`target_bands`]).
    fn new(
        sampling_frequency: SamplingFrequency,
        transform: Transform,
        top: Option<usize>,
    ) -> Self {
        let offsets = swb_offsets(sampling_frequency, transform);
        let max_bands = max_bands(sampling_frequency, transform);
        let top = top.filter(|_| transform == Transform::Long);
        let top = top.map_or(max_bands, |top| top.min(max_bands));
        let bottom = first_band_above(sampling_frequency, transform, TARGET_FROM_HZ).min(top);
        let line_hz = f64::from(sampling_frequency.hz()) / (2 * transform.lines()) as f64;
        let prediction_from = ((PREDICTION_FROM_HZ / line_hz).ceil() as usize)
            .clamp(usize::from(offsets[bottom]), usize::from(offsets[top]));
        Self {
            transform,
            offsets,
            bands: bottom..top,
            prediction_from,
            max_bands,
        }
    }

    /// `num_swb` of the window.
    fn num_swb(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Shapes the noise of one window's lines, `lines`, in ascending
    /// frequency (clause C.5): computes the linear prediction of the target
    /// range's lines, and where its gain exceeds the threshold, filters
    /// them with its quantized prediction error filter, and gives the
    /// filters to send.
    ///
    /// Spec reading: clause C.5 sends one filter, of `length` "Number of
    /// processed scalefactor bands" (Table C.25). The decoder counts the
    /// first filter's bands down from `num_swb`, not from `TNS_MAX_BANDS`,
    /// and filters none above `TNS_MAX_BANDS` (clause 14.3); so the filter
    /// is sent with the bands from the range's first to `num_swb`, and
    /// filters those of the range. Settled by decoding with ffmpeg (black
    /// box): clicks, pulse trains and struck bars so shaped, in long and
    /// short windows, decode in place at their level at every sampling
    /// frequency from 8 to 96 kHz, and so do pairs whose range is cut at
    /// `top` with a first filter of order 0; the conformance test is
    /// `temporal_noise_shaping_decodes_in_place`.
    ///
    /// This deliberately departs from Table C.25, which sends `n_filt` 1,
    /// where the range ends below `TNS_MAX_BANDS`, at `top`, as in a pair
    /// coded in intensity stereo (see `ElementEncoder`'s temporal noise
    /// shaping): a first filter of order 0, which filters nothing, spans
    /// the bands from `top` up, and the filter follows it, `n_filt` 2.
    fn shape(&self, lines: &mut [f64]) -> (WindowFilters, f64) {
        let order = max_order(self.transform);
        let range = self.line(self.bands.start)..self.line(self.bands.end);
        let mut r = [0.0; MAX_ORDER + 1];
        autocorrelation(&lines[self.prediction_from..range.end], &mut r[..=order]);
        let mut parcor = [0.0; MAX_ORDER];
        let Some(gain) = levinson_durbin(&r[..=order], &mut parcor[..order]) else {
            return (WindowFilters::NONE, 1.0);
        };
        if gain <= GAIN_THRESHOLD {
            return (WindowFilters::NONE, 1.0);
        }
        let mut filter = Filter::none(0);
        let mut quantized = [0.0; MAX_ORDER];
        for ((coef, rq), &r) in (filter.coef.iter_mut())
            .zip(&mut quantized)
            .zip(&parcor[..order])
        {
            *coef = quantize(r, COEF_RES);
            *rq = dequantize(*coef, COEF_RES);
        }
        // Remove the coefficients below the threshold from the tail.
        //
        // Spec reading: clause C.5 removes "reflection coefficients with an
        // absolute value smaller than a threshold p" after quantizing them,
        // and then "The remaining reflection coefficients rq[] are
        // converted": the quantized coefficients are compared with the
        // threshold. With 4 bits, the least magnitude that is not 0, sin(1 /
        // iqfac_m) of index -1, is 0.184, so the coefficients removed are
        // those quantized to 0.
        let order = (quantized[..order].iter())
            .rposition(|rq| rq.abs() >= TRUNCATION_THRESHOLD)
            .map_or(0, |last| last + 1);
        // Spec gap: clause C.5 does not say what is sent where every
        // coefficient is removed, which leaves a filter of order 0 that
        // filters nothing. The window is sent without filters, which
        // decodes alike in fewer bits.
        if order == 0 {
            return (WindowFilters::NONE, 1.0);
        }
        filter.order = order as u8;
        // Clause C.5: "if the two most significant bits of each quantized
        // reflection coefficient are either '00' or '11', coeff_compress may
        // be set to a value of one", which it is.
        let half = 1i8 << (COEF_RES - 2);
        filter.compress = (filter.coef[..order].iter()).all(|coef| (-half..half).contains(coef));
        let mut lpc = [0.0; MAX_ORDER + 1];
        step_up(&quantized[..order], &mut lpc[..=order]);
        let gain = noise_gain(&lpc[..=order], range.len());
        analysis_filter(&mut lines[range], &lpc[..=order]);

        let num_swb = self.num_swb();
        let mut window = WindowFilters {
            coef_res: COEF_RES,
            ..WindowFilters::NONE
        };
        let top = self.bands.end;
        if top < self.max_bands {
            window.filters[0] = Filter::none((num_swb - top) as u8);
            filter.length = (top - self.bands.start) as u8;
            window.filters[1] = filter;
            window.n_filt = 2;
        } else {
            filter.length = (num_swb - self.bands.start) as u8;
            window.filters[0] = filter;
            window.n_filt = 1;
        }
        (window, gain)
    }

    /// The first line of band `sfb`, or the window's lines for `num_swb`.
    fn line(&self, sfb: usize) -> usize {
        usize::from(self.offsets[sfb])
    }
}

/// Adds to `r` the autocorrelation of `lines` at each lag from 0 to
/// `r.len() - 1`.
fn autocorrelation(lines: &[f64], r: &mut [f64]) {
    for (lag, r) in r.iter_mut().enumerate() {
        *r += (lines.iter().zip(lines.get(lag..).unwrap_or_default()))
            .map(|(x, y)| x * y)
            .sum::<f64>();
    }
}

/// The linear prediction (clause C.5) of lines whose autocorrelation is
/// `r`, by the Levinson-Durbin recursion, of the order of `parcor`: gives
/// its reflection coefficients in `parcor`, and its prediction gain, the
/// ratio of `r[0]`, the energy of the lines, to that of their prediction
/// error; or `None` where the lines are all zero.
///
/// The reflection coefficients are those of the step-up procedure of
/// clause 14.3, whose prediction error filter is `1 + a[1] z^-1 + ... +
/// a[order] z^-order`: `a[m] = parcor[m - 1]`, and `a[i] += parcor[m - 1]
/// a[m - i]` below it.
///
/// Spec gap: clause C.5 leaves the procedure to "Standard LPC procedures as
/// known from speech processing", naming the Levinson-Durbin algorithm; the
/// autocorrelation that it starts from is taken over the lines alone,
/// unwindowed. A reflection coefficient is
/// kept just within 1, where rounding could take it there, for lines all
/// but perfectly predictable; the gain is then large, and is taken as
/// infinite only where the error's energy rounds to 0.
fn levinson_durbin(r: &[f64], parcor: &mut [f64]) -> Option<f64> {
    let order = parcor.len();
    if !(r[0] > 0.0 && r[0].is_finite()) {
        return None;
    }
    parcor.fill(0.0);
    let mut a = [0.0; MAX_ORDER + 1];
    a[0] = 1.0;
    let mut error = r[0];
    for m in 1..=order {
        let acc: f64 = (0..m).map(|i| a[i] * r[m - i]).sum();
        let k = (-acc / error).clamp(-1.0 + f64::EPSILON, 1.0 - f64::EPSILON);
        parcor[m - 1] = k;
        let previous = a;
        for i in 1..m {
            a[i] = previous[i] + k * previous[m - i];
        }
        a[m] = k;
        error *= 1.0 - k * k;
        if error <= 0.0 {
            return Some(f64::INFINITY);
        }
    }
    Some(r[0] / error)
}

/// The index of the reflection coefficient `r` quantized with `coef_res`
/// bits (clause C.5): `NINT(arcsin(r) * iqfac)` for `r` of 0 and above,
/// with `iqfac = (2^(coef_res - 1) - 0.5) / (pi / 2)`, and below 0,
/// `NINT(arcsin(r) * iqfac_m)`, with `iqfac_m = (2^(coef_res - 1) + 0.5) /
/// (pi / 2)`.
///
/// Spec gap: a coefficient of exactly 1 or -1 would give an index one
/// beyond the `coef_res` bits it is sent in; it is limited to their range.
fn quantize(r: f64, coef_res: u8) -> i8 {
    let (iqfac, iqfac_m) = iqfac(coef_res);
    let r = r.clamp(-1.0, 1.0);
    let scale = if r >= 0.0 { iqfac } else { iqfac_m };
    let half = f64::from(1u8 << (coef_res - 1));
    // f64::round rounds half-integers away from zero, as NINT does
    // (clause 2).
    (r.asin() * scale).round().clamp(-half, half - 1.0) as i8
}

/// The reflection coefficient of `index` (clause C.5, and as the decoder's
/// `tns_decode_coef()` gives it, clause 14.3): `sin(index / iqfac)`, or
/// `sin(index / iqfac_m)` below 0.
fn dequantize(index: i8, coef_res: u8) -> f64 {
    let (iqfac, iqfac_m) = iqfac(coef_res);
    let scale = if index >= 0 { iqfac } else { iqfac_m };
    (f64::from(index) / scale).sin()
}

/// `iqfac` and `iqfac_m` of `coef_res` bits (clause C.5).
fn iqfac(coef_res: u8) -> (f64, f64) {
    let half = f64::from(1u8 << (coef_res - 1));
    ((half - 0.5) / FRAC_PI_2, (half + 0.5) / FRAC_PI_2)
}

/// How far the decoder's all-pole filter of the prediction error filter
/// `a` raises the power of white noise over `len` lines, on average: its
/// state is 0 at the first line, so that the noise of line `n` is raised
/// by `h[0]^2 + ... + h[n]^2`, with `h` the filter's impulse response, and
/// the mean is `1/len sum_j h[j]^2 (len - j)`.
fn noise_gain(a: &[f64], len: usize) -> f64 {
    let order = a.len() - 1;
    // The last `order` values of the impulse response, cyclically.
    let mut h = [0.0; MAX_ORDER + 1];
    let mut sum = 0.0;
    for j in 0..len {
        let mut y = if j == 0 { 1.0 } else { 0.0 };
        for i in 1..=order.min(j) {
            y -= a[i] * h[(j - i) % (MAX_ORDER + 1)];
        }
        h[j % (MAX_ORDER + 1)] = y;
        sum += y * y * (len - j) as f64;
    }
    sum / len as f64
}

/// The step-up procedure (clause 14.3, "Conversion to LPC coefficients"):
/// the `order + 1` coefficients `a` of the prediction error filter of the
/// reflection coefficients `rq`, `a[0] = 1`.
fn step_up(rq: &[f64], a: &mut [f64]) {
    a.fill(0.0);
    a[0] = 1.0;
    for m in 1..=rq.len() {
        let previous: [f64; MAX_ORDER + 1] =
            std::array::from_fn(|i| a.get(i).copied().unwrap_or(0.0));
        for i in 1..m {
            a[i] = previous[i] + rq[m - 1] * previous[m - i];
        }
        a[m] = rq[m - 1];
    }
}

/// The encoder's filter (clause C.5), run upward over `lines` in place, its
/// state 0 before the first: `y[n] = x[n] + a[1] x[n-1] + ... + a[order]
/// x[n-order]`, the inverse of the decoder's all-pole filter.
fn analysis_filter(lines: &mut [f64], a: &[f64]) {
    // From the top down, so that each line's inputs below it are not yet
    // overwritten.
    for n in (0..lines.len()).rev() {
        let taps = (a.len() - 1).min(n);
        let mut y = lines[n];
        for i in 1..=taps {
            y += a[i] * lines[n - i];
        }
        lines[n] = y;
    }
}

#[cfg(test)]
mod tests {
    use chelys_core::bitstream::BitWriter;

    use super::*;
    use crate::noiseless::decoder::{self, BitReader, IcsInfo, Noise};

    /// Shapes `lines` in every window that clause C.5 would.
    fn shape(
        windows: Windows,
        fs: SamplingFrequency,
        top: Option<usize>,
        lines: &mut [f64; FRAME_LINES],
    ) -> Tns {
        let (all, mut gains) = ([true; SHORT_WINDOWS], NoiseGains::NONE);
        Shaping::default().shape(windows, fs, top, lines, all, &mut gains)
    }
    use crate::scalefactor_bands::num_swb;
    use crate::window::{Grouping, WindowSequence, WindowShape};

    #[test]
    fn indices_quantize_as_clause_c_5_has_them() {
        // iqfac = 7.5 / (pi / 2) and iqfac_m = 8.5 / (pi / 2) for 4 bits.
        assert_eq!(quantize(0.9, 4), 5);
        assert_eq!(quantize(-0.9, 4), -6);
        assert_eq!(quantize(0.05, 4), 0);
        assert_eq!(quantize(-0.3, 3), -1);
        // At the ends, limited to the 4 bits sent.
        assert_eq!(quantize(1.0, 4), 7);
        assert_eq!(quantize(-1.0, 4), -8);
        assert_eq!(quantize(1.0, 3), 3);
        assert_eq!(quantize(-1.0, 3), -4);
    }

    /// Every index decodes to the coefficient the decoder's
    /// `tns_decode_coef()` gives it, sent in full or compressed, and
    /// quantizes back to itself.
    #[test]
    fn indices_decode_as_the_decoder_decodes_them() {
        for coef_res in [3u8, 4] {
            let bits = u32::from(coef_res);
            let half = 1i8 << (coef_res - 1);
            for index in -half..half {
                let rq = dequantize(index, coef_res);
                assert_eq!(quantize(rq, coef_res), index);
                let sent = index as u32 & ((1 << bits) - 1);
                let a = decoder::tns_decode_coef(1, bits, 0, &[sent]);
                assert_eq!(a[1], rq, "{coef_res} bits, {index}");
                if (-half / 2..half / 2).contains(&index) {
                    let sent = index as u32 & ((1 << (bits - 1)) - 1);
                    let a = decoder::tns_decode_coef(1, bits, 1, &[sent]);
                    assert_eq!(a[1], rq, "{coef_res} bits, {index}, compressed");
                }
            }
        }
    }

    #[test]
    fn the_step_up_procedure_is_the_decoders() {
        let mut noise = Noise::new(3);
        for order in 1..=MAX_ORDER {
            let index: Vec<i8> = (0..order).map(|_| noise.below(16) as i8 - 8).collect();
            let rq: Vec<f64> = index.iter().map(|&i| dequantize(i, 4)).collect();
            let mut a = [0.0; MAX_ORDER + 1];
            step_up(&rq, &mut a[..=order]);
            let sent: Vec<u32> = index.iter().map(|&i| i as u32 & 0xf).collect();
            let expected = decoder::tns_decode_coef(order, 4, 0, &sent);
            for (a, b) in a[..=order].iter().zip(&expected) {
                assert!((a - b).abs() < 1e-12, "order {order}: {a} against {b}");
            }
        }
    }

    /// The lines of a click, whose MDCT is a cosine over frequency.
    fn click(lines: usize, at: f64, noise: &mut Noise) -> Vec<f64> {
        (0..lines)
            .map(|k| {
                let n = lines as f64;
                let jitter = f64::from(noise.below(1000)) / 1000.0 - 0.5;
                1000.0 * (std::f64::consts::PI / n * (at + n / 2.0 + 0.5) * (k as f64 + 0.5)).cos()
                    + 10.0 * jitter
            })
            .collect()
    }

    /// Writes `tns` and reads it back as the decoder does, for the windows
    /// of `info`.
    fn sent(tns: &Tns, info: &IcsInfo) -> Option<Vec<decoder::TnsWindow>> {
        let mut w = BitWriter::new();
        tns.write(&mut w);
        let len = w.bit_len();
        assert_eq!(len as u32, tns.bits());
        w.write(0, 7);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        let present = r.read(1) == 1;
        let windows = present.then(|| decoder::tns_data(&mut r, info));
        assert_eq!(r.position(), len);
        windows
    }

    /// The decoder's filter restores the lines that the encoder's filtered,
    /// in a long window at every sampling frequency, from the target range's
    /// first band to `TNS_MAX_BANDS`, leaving the lines outside it as they
    /// were.
    #[test]
    fn the_decoder_restores_a_long_window() {
        let mut noise = Noise::new(5);
        for fs in SamplingFrequency::ALL {
            let original = click(1024, 300.0, &mut noise);
            let mut lines: [f64; FRAME_LINES] = original.clone().try_into().unwrap();
            let tns = shape(Windows::LONG_SINE, fs, None, &mut lines);
            assert!(tns.is_present(), "{fs:?}");
            let window = tns.windows()[0];
            assert_eq!(window.n_filt, 1);
            let range = TargetRange::new(fs, Transform::Long, None);
            let filter = window.filters[0];
            assert_eq!(
                usize::from(filter.length),
                num_swb(fs, Transform::Long) - range.bands.start
            );
            assert!((1..=12).contains(&filter.order) && !filter.downward);
            let changed = (0..1024).filter(|&k| lines[k] != original[k]);
            let (first, last) = (range.line(range.bands.start), range.line(range.bands.end));
            assert!(
                changed.clone().all(|k| (first..last).contains(&k)),
                "{fs:?}"
            );
            assert!(changed.count() > (last - first) / 2, "{fs:?}");
            let info = IcsInfo {
                window_sequence: 0,
                window_shape: 0,
                max_sfb: num_swb(fs, Transform::Long),
                group_lengths: vec![1],
            };
            let windows = sent(&tns, &info).expect("tns_data_present");
            decoder::tns_decode_frame(&mut lines, &info, &windows, fs);
            for (k, (x, y)) in original.iter().zip(&lines).enumerate() {
                assert!(
                    (x - y).abs() < 1e-9 * 1000.0,
                    "{fs:?}, line {k}: {x} against {y}"
                );
            }
        }
    }

    /// Each short window is shaped apart, in the order the lines are sent,
    /// and restored by the decoder; windows without a click are not.
    #[test]
    fn the_decoder_restores_short_windows() {
        let mut noise = Noise::new(8);
        for fs in [
            SamplingFrequency::Hz8000,
            SamplingFrequency::Hz44100,
            SamplingFrequency::Hz96000,
        ] {
            let grouping =
                Grouping::from_starts([true, false, true, true, false, false, true, false]);
            let windows = Windows {
                sequence: WindowSequence::EightShort { grouping },
                shape: WindowShape::Kbd,
            };
            // Clicks in windows 1, 2 and 6; noise in the others.
            let mut by_window = vec![0.0; 1024];
            for (w, window) in by_window.chunks_exact_mut(128).enumerate() {
                let lines = if [1, 2, 6].contains(&w) {
                    click(128, 10.0 + 9.0 * w as f64, &mut noise)
                } else {
                    (0..128)
                        .map(|_| f64::from(noise.below(2001)) - 1000.0)
                        .collect()
                };
                window.copy_from_slice(&lines);
            }
            let mut lines = [0.0; FRAME_LINES];
            interleave(windows.groups(fs), &by_window, &mut lines);
            let tns = shape(windows, fs, None, &mut lines);
            let shaped: Vec<usize> = (tns.windows().iter().enumerate())
                .filter(|(_, window)| window.n_filt > 0)
                .map(|(w, _)| w)
                .collect();
            assert_eq!(shaped, [1, 2, 6], "{fs:?}");
            let info = IcsInfo {
                window_sequence: 2,
                window_shape: 1,
                max_sfb: num_swb(fs, Transform::Short),
                group_lengths: grouping.lengths().collect(),
            };
            let tns_windows = sent(&tns, &info).expect("tns_data_present");
            let mut spec = vec![0.0; 1024];
            deinterleave(windows.groups(fs), &lines, &mut spec);
            decoder::tns_decode_frame(&mut spec, &info, &tns_windows, fs);
            for (k, (x, y)) in by_window.iter().zip(&spec).enumerate() {
                assert!(
                    (x - y).abs() < 1e-9 * 1000.0,
                    "{fs:?}, line {k}: {x} against {y}"
                );
            }
        }
    }

    #[test]
    fn noise_and_silence_are_not_shaped() {
        let mut noise = Noise::new(13);
        let mut lines: [f64; FRAME_LINES] =
            std::array::from_fn(|_| f64::from(noise.below(2001)) - 1000.0);
        let original = lines;
        let fs = SamplingFrequency::Hz48000;
        assert_eq!(
            shape(Windows::LONG_SINE, fs, None, &mut lines),
            Tns::new(Transform::Long)
        );
        assert_eq!(lines, original);
        let mut silence = [0.0; FRAME_LINES];
        let tns = shape(Windows::LONG_SINE, fs, None, &mut silence);
        assert!(!tns.is_present());
    }

    /// A range cut below `TNS_MAX_BANDS` is sent as a filter of order 0
    /// over the bands above it, then the filter, and leaves those bands'
    /// lines as they were.
    #[test]
    fn a_range_cut_short_sends_the_bands_above_unfiltered() {
        let fs = SamplingFrequency::Hz44100;
        let mut noise = Noise::new(21);
        let original = click(1024, 100.0, &mut noise);
        let mut lines: [f64; FRAME_LINES] = original.clone().try_into().unwrap();
        let tns = shape(Windows::LONG_SINE, fs, Some(28), &mut lines);
        let window = tns.windows()[0];
        assert_eq!(window.n_filt, 2);
        // Table 45: 49 bands; band 14 is the first at 1.5 kHz (line 72, at
        // 1550 Hz), band 28 starts at line 292.
        assert_eq!(
            (window.filters[0].length, window.filters[0].order),
            (49 - 28, 0)
        );
        assert_eq!(window.filters[1].length, 28 - 14);
        assert!(window.filters[1].order > 0);
        assert_eq!(lines[292..], original[292..]);
        assert_eq!(lines[..72], original[..72]);
        let info = IcsInfo {
            window_sequence: 0,
            window_shape: 0,
            max_sfb: 49,
            group_lengths: vec![1],
        };
        let windows = sent(&tns, &info).expect("tns_data_present");
        decoder::tns_decode_frame(&mut lines, &info, &windows, fs);
        for (x, y) in original.iter().zip(&lines) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    /// The noise gain of a filter of order 0 is 1, and of `1 / (1 - 0.5
    /// z^-1)`, whose impulse response is `0.5^j`, the mean over `len` lines
    /// of `(1 - 0.25^(n + 1)) / 0.75`.
    #[test]
    fn the_noise_gain_is_the_mean_power_gain_of_the_decoders_filter() {
        assert_eq!(noise_gain(&[1.0], 100), 1.0);
        for len in [1, 2, 10, 400] {
            let expected = (0..len)
                .map(|n| (1.0 - 0.25f64.powi(n + 1)) / 0.75)
                .sum::<f64>()
                / f64::from(len);
            let gain = noise_gain(&[1.0, -0.5], len as usize);
            assert!(
                (gain - expected).abs() < 1e-12,
                "{len}: {gain} against {expected}"
            );
        }
    }

    /// The prediction gain is the ratio of the energies of the lines and
    /// of their prediction error.
    #[test]
    fn the_prediction_gain_is_that_of_the_filter_it_gives() {
        let mut noise = Noise::new(34);
        let lines = click(700, 40.0, &mut noise);
        let mut parcor = [0.0; MAX_ORDER];
        let mut r = [0.0; MAX_ORDER + 1];
        autocorrelation(&lines, &mut r);
        let gain = levinson_durbin(&r, &mut parcor).unwrap();
        let mut a = [0.0; MAX_ORDER + 1];
        step_up(&parcor, &mut a);
        let mut error = lines.clone();
        analysis_filter(&mut error, &a);
        let energy = |x: &[f64]| x.iter().map(|x| x * x).sum::<f64>();
        // The autocorrelation method counts the filter's run-out past the
        // last line too, which the filtered lines leave out: the gain it
        // gives is the lower.
        let ratio = energy(&lines) / energy(&error);
        assert!(gain > 100.0 && ratio >= gain, "{gain} against {ratio}");
    }
}
