// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The psychoacoustic model of clause C.1: for each channel's frame, the
//! energy that each scalefactor band of a long window may take as
//! quantization noise, `xmin(n)`, and the perceptual entropy, from which
//! the bits the frame asks for follow (step 15).
//!
//! The model runs the long FFT on the frame's window and the eight short
//! FFTs on its short windows, whose unpredictability the long blocks' takes
//! (step 4). The thresholds of short blocks, and the choice between long
//! and short blocks (step 13), are to come.

pub mod partitions;

use std::f64::consts::PI;
use std::ops::Range;
use std::sync::OnceLock;

use chelys_core::fft::{Complex, RealFft};

use crate::channel::FRAME_LINES;
use crate::filterbank::PCM_SCALE;
use crate::sampling_frequency::SamplingFrequency;
use crate::scalefactor_bands::{GroupBands, MAX_BANDS, Transform};
use crate::stream::FRAME_SAMPLES;
use crate::window::SHORT_WINDOWS;
use partitions::Row;

/// The shift length `iblen` of the long FFT (clause C.1.1): a frame's
/// samples.
const LONG_IBLEN: usize = FRAME_SAMPLES;
/// The shift length `iblen` of the short FFT.
const SHORT_IBLEN: usize = 128;
/// The first sample of the first short window in a frame's window of
/// `2 * LONG_IBLEN` samples, each of the others `SHORT_IBLEN` after the one
/// before (clause 15.3.2, EIGHT_SHORT_SEQUENCE).
const FIRST_SHORT: usize = 448;

/// The lines of the long FFT whose unpredictability the long FFT gives;
/// the short FFTs give it above (clause C.1.4, step 4).
const LONG_CW_LINES: usize = 6;

/// `NMT(b)` and `TMN(b)` in dB, for all `b` (step 8).
const NMT: f64 = 6.0;
const TMN: f64 = 18.0;

/// `rpelev` of long blocks (step 11).
const RPELEV_LONG: f64 = 2.0;

/// `pew1` and `pew2` of long blocks (step 15).
const PEW1_LONG: f64 = 0.3;
const PEW2_LONG: f64 = 6.0;

/// The limit of `bit_allocation` (step 15).
const MAX_BIT_ALLOCATION: f64 = 3000.0;

/// The spreading function `sprdngf(i, j)` of clause C.1.3, from `i`, the
/// Bark value of the signal being spread, to `j`, that of the band it is
/// spread into, as printed.
///
/// Spec reading: the function is taken as printed, `tmpx = 3.0 (j - i)`
/// towards higher frequencies and `1.5 (j - i)` towards lower ones. Model
/// 2 of ISO/IEC 11172-3 prints the same rule, which chelys's MP3 encoder
/// reads with the factors swapped, as the `norm` column of that standard's
/// tables bears out; Annex C prints no such column, and computes the
/// normalization from the function (step 6), so nothing here bears out
/// either order.
pub fn spreading_function(i: f64, j: f64) -> f64 {
    let tmpx = if j >= i { 3.0 } else { 1.5 } * (j - i);
    let tmpz = 8.0 * f64::min((tmpx - 0.5).powi(2) - 2.0 * (tmpx - 0.5), 0.0);
    let tmpy = 15.811389 + 7.5 * (tmpx + 0.474) - 17.5 * (1.0 + (tmpx + 0.474).powi(2)).sqrt();
    if tmpy < -100.0 {
        0.0
    } else {
        10f64.powf((tmpz + tmpy) / 10.0)
    }
}

/// The Hann window of step 2 for a shift length of `iblen`:
/// `0.5 - 0.5 cos(pi (i + 0.5) / iblen)` for `0 <= i < 2 iblen`.
fn hann(iblen: usize) -> Vec<f64> {
    (0..2 * iblen)
        .map(|i| 0.5 - 0.5 * (PI * (i as f64 + 0.5) / iblen as f64).cos())
        .collect()
}

/// The energy, in the FFT of shift length `iblen`, that the thresholds in
/// quiet of the tables put at 0 dB: that of the line of a sine wave of
/// +-1/2 LSB (step 11).
///
/// Spec gap: step 11 puts `qsthr` "relative to the level that a sine wave
/// of + or - 1/2 lsb has in the FFT", to be converted "after considering
/// the FFT normalization actually used", and does not say which level of a
/// sine, whose energy the window spreads over several lines. The level of
/// the line the sine is centred on is taken. A sine of amplitude `a` at
/// the centre of a line gives that line `a / 2` times the sum of the Hann
/// window, which is `iblen`, in the unnormalized FFT of step 2; with
/// samples at the decoder's scale, that of 16-bit PCM, where an LSB is 1
/// (see [`PCM_SCALE`]), that is `iblen / 4`.
///
/// Spec reading: step 11 compares `qsthr(b)` with `nb(b)`, the threshold
/// of the whole partition, and step 14 divides that among its lines. The
/// tables' `qsthr` rises with a partition's width by its share of the
/// lines, as Table C.1 gives 33.82 dB to the 9 lines of partition 10 and
/// 34.28 dB, 0.46 dB more, to the 10 of partition 11, beside it: each is
/// a partition's threshold in quiet, the line's level times its lines, and
/// is compared as printed.
fn quiet_reference(iblen: usize) -> f64 {
    (iblen as f64 / 4.0).powi(2)
}

/// What the model computes once for each sampling frequency for long
/// blocks.
#[derive(Debug)]
struct LongTables {
    partitions: &'static [Row],
    /// `sprdngf(bval(bb), bval(b))` at `bb * partitions + b`.
    spreading: Vec<f64>,
    /// `rnorm(b)` (step 6).
    rnorm: Vec<f64>,
    /// `qsthr(b)` as an energy (step 11).
    quiet: Vec<f64>,
    bands: GroupBands,
    /// The partitions that each scalefactor band's lines fall in.
    band_partitions: Vec<Range<usize>>,
}

impl LongTables {
    fn new(sampling_frequency: SamplingFrequency) -> Self {
        let partitions = partitions::table(sampling_frequency, Transform::Long);
        let n = partitions.len();
        let mut spreading = vec![0.0; n * n];
        for (bb, from) in partitions.iter().enumerate() {
            for (b, to) in partitions.iter().enumerate() {
                spreading[bb * n + b] = spreading_function(from.bval, to.bval);
            }
        }
        let rnorm = (0..n)
            .map(|b| 1.0 / (0..n).map(|bb| spreading[bb * n + b]).sum::<f64>())
            .collect();
        let reference = quiet_reference(LONG_IBLEN);
        let quiet = (partitions.iter())
            .map(|p| reference * 10f64.powf(p.qsthr / 10.0))
            .collect();
        let bands = GroupBands::long(sampling_frequency);
        let partition_of = |w: usize| {
            (partitions.iter())
                .position(|p| p.lines().contains(&w))
                .expect("the partitions cover the lines")
        };
        let band_partitions = (0..bands.num_swb())
            .map(|n| partition_of(bands.offset(n))..partition_of(bands.offset(n + 1) - 1) + 1)
            .collect();
        Self {
            partitions,
            spreading,
            rnorm,
            quiet,
            bands,
            band_partitions,
        }
    }

    fn get(sampling_frequency: SamplingFrequency) -> &'static Self {
        static TABLES: [OnceLock<LongTables>; 12] = [const { OnceLock::new() }; 12];
        TABLES[sampling_frequency.index() as usize].get_or_init(|| Self::new(sampling_frequency))
    }
}

/// The magnitude `r(w)` and phase `f(w)` of each line of an FFT's last two
/// blocks, from which step 3 predicts the next, the phase as `e^(i f(w))`.
#[derive(Debug)]
struct Predictor {
    last: Vec<Polar>,
    before_last: Vec<Polar>,
}

#[derive(Clone, Copy, Debug)]
struct Polar {
    r: f64,
    /// `e^(i f)`.
    phase: Complex,
}

impl Default for Polar {
    /// A line of the zeroed arrays that the model starts from (clause
    /// C.1.1): no magnitude, and the phase 0.
    fn default() -> Self {
        Self {
            r: 0.0,
            phase: Complex { re: 1.0, im: 0.0 },
        }
    }
}

impl Predictor {
    fn new(lines: usize) -> Self {
        Self {
            last: vec![Polar::default(); lines],
            before_last: vec![Polar::default(); lines],
        }
    }

    /// The unpredictability `c(w)` of the predicted lines of the block
    /// whose transform is `x` (steps 3 and 4), into `c`, and the block kept
    /// as the last.
    ///
    /// `c(w) = |r(w) e^(i f(w)) - r_pred(w) e^(i f_pred(w))| / (r(w) +
    /// |r_pred(w)|)`, with `r_pred(w) = 2 r(t-1) - r(t-2)` and `f_pred(w) =
    /// 2 f(t-1) - f(t-2)`: the distance of the two points whose cosines and
    /// sines step 4 takes. `e^(i f_pred)` is had as `e^(i f(t-1))^2
    /// e^(-i f(t-2))`, the same but for rounding, rather than through the
    /// angles.
    ///
    /// Spec gap: a line with no magnitude has no phase. It is taken as 0,
    /// as the arrays start (clause C.1.1). A line that has no magnitude
    /// and none predicted divides zero by zero, and says nothing of how
    /// predictable the signal is. It is taken as 1, wholly unpredictable,
    /// the most `c(w)` can be (the distance of the two points is at most
    /// the sum of their magnitudes): the long FFT weights its own lines by
    /// their energy, which is then 0, but takes the unpredictability of its
    /// lines from the sixth on as the least over the short FFTs (step 4),
    /// where a short window of silence, as before an onset, would otherwise
    /// make the whole frame tonal.
    fn unpredictability(&mut self, x: &[Complex], c: &mut [f64]) {
        for (w, (x, c)) in x.iter().zip(c).enumerate() {
            let (last, before) = (self.last[w], self.before_last[w]);
            let r_pred = 2.0 * last.r - before.r;
            let doubled = last.phase * last.phase;
            let conj = Complex {
                re: before.phase.re,
                im: -before.phase.im,
            };
            let predicted = doubled * conj;
            let r = (x.re * x.re + x.im * x.im).sqrt();
            let dre = x.re - r_pred * predicted.re;
            let dim = x.im - r_pred * predicted.im;
            let scale = r + r_pred.abs();
            *c = if scale == 0.0 {
                1.0
            } else {
                (dre * dre + dim * dim).sqrt() / scale
            };
            let phase = if r == 0.0 {
                Polar::default().phase
            } else {
                Complex {
                    re: x.re / r,
                    im: x.im / r,
                }
            };
            self.before_last[w] = last;
            self.last[w] = Polar { r, phase };
        }
    }
}

/// The threshold of a long block in each scalefactor band, on the scale
/// of the FFT, and the perceptual entropy (steps 12 and 14).
#[derive(Clone, Debug, PartialEq)]
pub struct LongThreshold {
    /// `epart(n)`: the FFT's energy in each band.
    pub epart: [f64; MAX_BANDS],
    /// `npart(n)`: the noise level each band allows.
    pub npart: [f64; MAX_BANDS],
    /// `PE`.
    pub perceptual_entropy: f64,
    bands: GroupBands,
}

impl LongThreshold {
    /// `xmin(n) = npart(n) * codec_e(n) / epart(n)` (step 14): the energy
    /// of quantization error that each band of a long window's `spectrum`
    /// may take, where `codec_e(n)` is the energy of its MDCT lines.
    ///
    /// Spec gap: a band whose FFT has no energy makes the ratio undefined.
    /// Its signal is below any threshold, as `SMR(n) = epart(n) / npart(n)`
    /// is 0, so it is allowed its whole MDCT energy, `codec_e(n)`.
    pub fn xmin(&self, spectrum: &[f64; FRAME_LINES]) -> [f64; MAX_BANDS] {
        let mut xmin = [0.0; MAX_BANDS];
        for (n, xmin) in xmin[..self.bands.num_swb()].iter_mut().enumerate() {
            let lines = &spectrum[self.bands.offset(n)..self.bands.offset(n + 1)];
            let codec_e: f64 = lines.iter().map(|x| x * x).sum();
            *xmin = if self.epart[n] == 0.0 {
                codec_e
            } else {
                self.npart[n] * codec_e / self.epart[n]
            };
        }
        xmin
    }

    /// `bit_allocation = pew1 * PE + pew2 * sqrt(PE)`, limited to 0 to
    /// 3000 (step 15).
    ///
    /// Spec gap: the entropy is negative where the thresholds exceed the
    /// energy, as in silence, -9535 at 44.1 kHz, and in most frames of
    /// music (see [`Model::analyze`]), where `sqrt(PE)` is undefined; the
    /// allocation is then 0, the lower limit.
    pub fn bit_allocation(&self) -> f64 {
        let pe = self.perceptual_entropy.max(0.0);
        (PEW1_LONG * pe + PEW2_LONG * pe.sqrt()).min(MAX_BIT_ALLOCATION)
    }
}

/// The model of one channel, with what it keeps from one frame to the
/// next.
#[derive(Debug)]
pub struct Model {
    long: &'static LongTables,
    /// The newest `iblen` samples of the frame before (step 1), at
    /// [`PCM_SCALE`].
    previous: Box<[f64; LONG_IBLEN]>,
    long_fft: RealFft,
    short_fft: RealFft,
    long_hann: Vec<f64>,
    short_hann: Vec<f64>,
    long_predictor: Predictor,
    short_predictor: Predictor,
    /// `nb(b)` of the last block, before pre-echo control (step 11).
    nb_last: Vec<f64>,
    // Buffers kept from one frame to the next for their memory.
    windowed: Vec<f64>,
    energy: Vec<f64>,
    c: Vec<f64>,
    short_c: Vec<f64>,
    short_c_min: Vec<f64>,
}

impl Model {
    /// The model of a channel at `sampling_frequency`, before its first
    /// frame: the block before it and the blocks its predictions start from
    /// silent, as the filterbank takes them (see
    /// [`Filterbank::new`](crate::filterbank::Filterbank::new)).
    pub fn new(sampling_frequency: SamplingFrequency) -> Self {
        let long = LongTables::get(sampling_frequency);
        Self {
            long,
            previous: Box::new([0.0; LONG_IBLEN]),
            long_fft: RealFft::new(2 * LONG_IBLEN),
            short_fft: RealFft::new(2 * SHORT_IBLEN),
            long_hann: hann(LONG_IBLEN),
            short_hann: hann(SHORT_IBLEN),
            long_predictor: Predictor::new(LONG_CW_LINES),
            short_predictor: Predictor::new(SHORT_IBLEN + 1),
            nb_last: vec![0.0; long.partitions.len()],
            windowed: vec![0.0; 2 * LONG_IBLEN],
            energy: vec![0.0; LONG_IBLEN],
            c: vec![0.0; LONG_IBLEN],
            short_c: vec![0.0; SHORT_IBLEN + 1],
            short_c_min: vec![0.0; SHORT_IBLEN + 1],
        }
    }

    /// Analyzes the frame whose block is `block`, the channel's next 1024
    /// samples at full scale 1.0: its window of `2 * iblen` samples, the
    /// block before and this one (step 1), is the window of the frame's
    /// MDCT (clause C.3.1.1), in which the psychoacoustic model's window is
    /// thus centred (clause C.1.1).
    ///
    /// Spec reading: step 4 takes the unpredictability of a long block's
    /// lines from the sixth on as "the minimum of the unpredictability of
    /// all short FFT's", whose lines are eight times as wide. Each line of
    /// the long FFT takes that of the short FFTs' line nearest it in
    /// frequency, `(w + 4) / 8`. The short FFTs are those of the frame's
    /// eight short windows (clause 15.3.2), each centred, as the long FFT
    /// is, in the window of its transform; each is predicted from the two
    /// before it, across frames.
    ///
    /// Spec reading: the perceptual entropy (step 12) weighs each partition
    /// by `w_high(b) - w_low(b)`, one less than its lines, where step 14
    /// counts a partition's lines as `w_high(b) + 1 - w_low(b)` and the
    /// tables give that count as its width, so that a partition of one line
    /// adds nothing whatever its energy. Nothing in the text shows the
    /// weight to be misprinted, and it is taken as printed. It matters most
    /// where the energy lies in the narrowest partitions: at 44.1 kHz, the
    /// frame in which half of full scale at 0 Hz sets in after silence has
    /// an entropy of -159 so, and of -16 with each partition's lines; white
    /// noise setting in after silence, a median of 306 against 350.
    ///
    /// Spec reading: the entropy is taken as printed, with no floor: a
    /// partition whose threshold exceeds its energy plus one subtracts from
    /// it, so silence and quiet passages, whose partitions sit below the
    /// thresholds in quiet, have a negative entropy. At 44.1 kHz, silence
    /// has -9535, a steady tone of 1 kHz at half of full scale about
    /// -8980, and pop music a median of about -1100, where the partitions
    /// above about 16 kHz lie below their thresholds in quiet; white noise
    /// at a quarter of full scale has about +300. A floor at 0 for each
    /// partition would instead give silence an entropy of its own, from
    /// partitions whose threshold is below 1, as chelys's MP3 encoder found
    /// of model 2.
    pub fn analyze(&mut self, block: &[f64; FRAME_SAMPLES]) -> LongThreshold {
        let tables = self.long;
        // Step 1: the window of the block before and this one.
        let (before, after) = self.windowed.split_at_mut(LONG_IBLEN);
        before.copy_from_slice(&*self.previous);
        for (s, &x) in after.iter_mut().zip(block) {
            *s = x * PCM_SCALE;
        }
        self.previous.copy_from_slice(after);

        // Steps 3 and 4 for the short FFTs: the least unpredictability of
        // each of their lines over the eight.
        self.short_c_min.fill(f64::INFINITY);
        let mut short_windowed = [0.0; 2 * SHORT_IBLEN];
        for k in 0..SHORT_WINDOWS {
            let start = FIRST_SHORT + k * SHORT_IBLEN;
            let samples = &self.windowed[start..start + 2 * SHORT_IBLEN];
            for ((sw, &s), &w) in short_windowed.iter_mut().zip(samples).zip(&self.short_hann) {
                *sw = s * w;
            }
            let x = self.short_fft.transform(&short_windowed);
            self.short_predictor.unpredictability(x, &mut self.short_c);
            for (min, &c) in self.short_c_min.iter_mut().zip(&self.short_c) {
                *min = min.min(c);
            }
        }

        // Step 2: the long FFT of the windowed samples.
        for (s, &w) in self.windowed.iter_mut().zip(&self.long_hann) {
            *s *= w;
        }
        let x = self.long_fft.transform(&self.windowed);
        for (energy, x) in self.energy.iter_mut().zip(x) {
            *energy = x.re * x.re + x.im * x.im;
        }
        // Steps 3 and 4.
        self.long_predictor
            .unpredictability(&x[..LONG_CW_LINES], &mut self.c[..LONG_CW_LINES]);
        for (w, c) in self.c.iter_mut().enumerate().skip(LONG_CW_LINES) {
            *c = self.short_c_min[(w + 4) / 8];
        }

        // Step 5: the energy and weighted unpredictability of each
        // partition.
        let n = tables.partitions.len();
        let mut e = [0.0; MAX_PARTITIONS];
        let mut c = [0.0; MAX_PARTITIONS];
        for (b, p) in tables.partitions.iter().enumerate() {
            for w in p.lines() {
                e[b] += self.energy[w];
                c[b] += self.energy[w] * self.c[w];
            }
        }

        let mut perceptual_entropy = 0.0;
        let mut nb = [0.0; MAX_PARTITIONS];
        for (b, nb) in nb[..n].iter_mut().enumerate() {
            // Step 6: spread, and normalized.
            let (mut ecb, mut ct) = (0.0, 0.0);
            for bb in 0..n {
                let spreading = tables.spreading[bb * n + b];
                ecb += e[bb] * spreading;
                ct += c[bb] * spreading;
            }
            // Spec gap: without energy `cb(b) = ct(b) / ecb(b)` is
            // undefined; its threshold is 0 whatever the tonality, and the
            // threshold in quiet sets it (step 11). It is taken as 0.
            let cb = if ecb == 0.0 { 0.0 } else { ct / ecb };
            let en = ecb * tables.rnorm[b];
            // Step 7: the tonality, within 0 and 1.
            let tb = (-0.299 - 0.43 * cb.ln()).clamp(0.0, 1.0);
            // Steps 8 to 10.
            let snr = tb * TMN + (1.0 - tb) * NMT;
            let bc = 10f64.powf(-snr / 10.0);
            *nb = en * bc;
        }
        // Step 11, against the last block's nb(b).
        //
        // Spec reading: step 11 compares the threshold with `nb_l(b)`, "the
        // threshold of partition b for the last block". Both step 10 and
        // this step call their result `nb(b)`; the more literal reading is
        // the threshold this step gives, which is left in `nb(b)`. This
        // deliberately departs from it, and takes the threshold of step 10:
        // the energy threshold the last block's signal set, so
        // that a block after a quieter one, as at an attack, is held near
        // the quieter one's threshold, while a steady signal keeps its own.
        // Taking the threshold this step gives would let a threshold rise
        // by `rpelev`, 3 dB, a frame at most, so that after silence a loud
        // signal would take some twenty frames to reach its own. The block
        // before the first is silent, with no threshold.
        for b in 0..n {
            let limited = nb[b].min(self.nb_last[b] * RPELEV_LONG);
            self.nb_last[b] = nb[b];
            nb[b] = tables.quiet[b].max(limited);
            // Step 12.
            let p = &tables.partitions[b];
            let weight = f64::from(p.w_high - p.w_low);
            perceptual_entropy -= weight * (nb[b] / (e[b] + 1.0)).log10();
        }

        // Step 14.
        let bands = tables.bands;
        let mut epart = [0.0; MAX_BANDS];
        let mut npart = [0.0; MAX_BANDS];
        for (band, range) in tables.band_partitions.iter().enumerate() {
            let lines = bands.offset(band)..bands.offset(band + 1);
            epart[band] = self.energy[lines.clone()].iter().sum();
            let thr = (range.clone())
                .map(|b| nb[b] / tables.partitions[b].lines().len() as f64)
                .fold(f64::INFINITY, f64::min);
            npart[band] = thr * lines.len() as f64;
        }
        LongThreshold {
            epart,
            npart,
            perceptual_entropy,
            bands,
        }
    }
}

/// The most partitions of any table: the 72 of the long FFT at 88.2 kHz
/// (Table C.21).
const MAX_PARTITIONS: usize = 72;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_table_has_more_partitions_than_the_most() {
        let most = (SamplingFrequency::ALL.iter())
            .flat_map(|&fs| {
                [Transform::Long, Transform::Short].map(|t| partitions::table(fs, t).len())
            })
            .max();
        assert_eq!(most, Some(MAX_PARTITIONS));
    }

    #[test]
    fn the_spreading_function_peaks_where_it_spreads_into_itself() {
        // tmpx 0: tmpz 0 and tmpy 15.811389 + 7.5 * 0.474 - 17.5 *
        // sqrt(1 + 0.474^2), about 0 dB.
        assert!((spreading_function(5.0, 5.0) - 1.0).abs() < 1e-3);
        for d in [0.5, 1.0, 2.0, 4.0] {
            assert!(spreading_function(5.0, 5.0 + d) < spreading_function(5.0, 5.0 + d / 2.0));
            assert!(spreading_function(5.0, 5.0 - d) < spreading_function(5.0, 5.0 - d / 2.0));
        }
        // Far away it is cut to zero.
        assert_eq!(spreading_function(0.0, 20.0), 0.0);
        assert_eq!(spreading_function(20.0, 0.0), 0.0);
    }

    fn tone(frequency: f64, amplitude: f64, len: usize, fs: f64) -> Vec<f64> {
        (0..len)
            .map(|t| amplitude * (2.0 * PI * frequency * t as f64 / fs).sin())
            .collect()
    }

    fn run(model: &mut Model, input: &[f64]) -> Vec<LongThreshold> {
        input
            .chunks_exact(FRAME_SAMPLES)
            .map(|block| model.analyze(block.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn silence_has_the_threshold_in_quiet_and_no_entropy_to_speak_of() {
        let fs = SamplingFrequency::Hz44100;
        let mut model = Model::new(fs);
        let thresholds = run(&mut model, &vec![0.0; 3 * FRAME_SAMPLES]);
        let tables = LongTables::get(fs);
        for threshold in &thresholds {
            assert_eq!(threshold.epart, [0.0; MAX_BANDS]);
            assert!(threshold.perceptual_entropy < 0.0);
            assert_eq!(threshold.bit_allocation(), 0.0);
            for n in 0..tables.bands.num_swb() {
                assert!(threshold.npart[n] > 0.0);
            }
        }
    }

    #[test]
    fn a_tone_is_masked_less_than_noise() {
        let fs = SamplingFrequency::Hz48000;
        let len = 8 * FRAME_SAMPLES;
        let sine = tone(1000.0, 0.25, len, 48_000.0);
        let mut state = 7u64;
        let noise: Vec<f64> = (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                0.25 * ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
            })
            .collect();
        let smr = |input: &[f64]| {
            let threshold = run(&mut Model::new(fs), input).pop().unwrap();
            // Band 10, lines 40 to 47 at 48 kHz, 938 to 1125 Hz, holds
            // the tone.
            (
                threshold.epart[10] / threshold.npart[10],
                threshold.perceptual_entropy,
            )
        };
        let (tone_smr, tone_pe) = smr(&sine);
        let (noise_smr, noise_pe) = smr(&noise);
        // A tone asks for about TMN = 18 dB and more over its own band,
        // which holds all of its energy; noise for about NMT = 6 dB.
        assert!(tone_smr > 10f64.powf(1.8), "{tone_smr}");
        assert!(noise_smr < 10f64.powf(1.2), "{noise_smr}");
        // Noise across the band carries far more entropy than one tone.
        assert!(noise_pe > 4.0 * tone_pe, "{noise_pe} against {tone_pe}");
    }

    #[test]
    fn an_attack_after_silence_is_held_near_the_quiet_threshold() {
        let fs = SamplingFrequency::Hz44100;
        let mut input = vec![0.0; 4 * FRAME_SAMPLES];
        input.extend(tone(3000.0, 0.5, 4 * FRAME_SAMPLES, 44_100.0));
        let thresholds = run(&mut Model::new(fs), &input);
        // The frame whose window first holds the tone is held at twice the
        // last block's threshold, of silence, and so at the threshold in
        // quiet; the frames after it take their own.
        let band = 20; // lines 132 to 143, 2842 to 3101 Hz
        let tables = LongTables::get(fs);
        let quiet = (tables.band_partitions[band].clone())
            .map(|b| tables.quiet[b] / tables.partitions[b].lines().len() as f64)
            .fold(f64::INFINITY, f64::min)
            * 12.0;
        let attack = thresholds[4].npart[band];
        assert!(
            (attack - quiet).abs() <= 1e-9 * quiet,
            "{attack} against {quiet}"
        );
        let steady = thresholds[6].npart[band];
        assert!(steady > 50.0 * attack, "{steady} against {attack}");
    }

    #[test]
    fn xmin_scales_npart_to_the_mdct() {
        let fs = SamplingFrequency::Hz44100;
        let input = tone(2500.0, 0.3, 3 * FRAME_SAMPLES, 44_100.0);
        let threshold = run(&mut Model::new(fs), &input).pop().unwrap();
        let mut spectrum = [0.0; FRAME_LINES];
        spectrum[100] = 2.0;
        spectrum[101] = -1.0;
        let xmin = threshold.xmin(&spectrum);
        let bands = GroupBands::long(fs);
        let band = (0..bands.num_swb())
            .find(|&n| bands.offset(n + 1) > 100)
            .unwrap();
        let expected = threshold.npart[band] * 5.0 / threshold.epart[band];
        assert!((xmin[band] - expected).abs() <= 1e-12 * expected);
        // A band without MDCT energy allows none.
        assert_eq!(xmin[band + 5], 0.0);
    }
}
