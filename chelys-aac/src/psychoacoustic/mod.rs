// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The psychoacoustic model of clause C.1: for each channel's frame, the
//! energy that each scalefactor band of its long window, or of each of its
//! short windows, may take as quantization noise, `xmin(n)`, and the
//! perceptual entropy, from which the bits the frame asks for follow (step
//! 15); and the choice between long and short blocks (step 13), which
//! [`switching`] makes a frame ahead.

pub mod partitions;
pub mod switching;

use std::f64::consts::PI;
use std::ops::Range;
use std::sync::OnceLock;

use chelys_core::fft::{Complex, RealFft};

use crate::channel::FRAME_LINES;
use crate::filterbank::{FIRST_SHORT, PCM_SCALE};
use crate::sampling_frequency::SamplingFrequency;
use crate::scalefactor_bands::{GroupBands, MAX_BANDS, Transform};
use crate::stream::FRAME_SAMPLES;
use crate::window::{SHORT_WINDOWS, WindowSequence, Windows};
use partitions::Row;

/// The shift length `iblen` of the long FFT (clause C.1.1): a frame's
/// samples.
const LONG_IBLEN: usize = FRAME_SAMPLES;
/// The shift length `iblen` of the short FFT.
const SHORT_IBLEN: usize = 128;

/// The lines of the long FFT whose unpredictability the long FFT gives;
/// the short FFTs give it above (clause C.1.4, step 4).
const LONG_CW_LINES: usize = 6;

/// `NMT(b)` and `TMN(b)` in dB, for all `b` (step 8).
const NMT: f64 = 6.0;
const TMN: f64 = 18.0;

/// `rpelev` of long and of short blocks (step 11).
const RPELEV_LONG: f64 = 2.0;
const RPELEV_SHORT: f64 = 1.0;

/// `pew1` and `pew2` of long blocks (step 15).
const PEW1_LONG: f64 = 0.3;
const PEW2_LONG: f64 = 6.0;
/// `pew1` and `pew2` of short blocks (step 15).
const PEW1_SHORT: f64 = 0.6;
const PEW2_SHORT: f64 = 24.0;

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

/// What the model computes once for each sampling frequency and FFT.
#[derive(Debug)]
struct Tables {
    partitions: &'static [Row],
    /// `sprdngf(bval(bb), bval(b))` at `bb * partitions + b`.
    spreading: Vec<f64>,
    /// The partitions `bb` that spread into each partition `b`: those
    /// whose `sprdngf(bval(bb), bval(b))` is not 0, which lie together
    /// about `b`, as the function falls with the distance either way.
    spread_from: Vec<Range<usize>>,
    /// `rnorm(b)` (step 6).
    rnorm: Vec<f64>,
    /// `qsthr(b)` as an energy (step 11).
    quiet: Vec<f64>,
    /// The scalefactor bands of one window of the FFT's transform.
    bands: GroupBands,
    /// The partitions that each scalefactor band's lines fall in.
    band_partitions: Vec<Range<usize>>,
}

impl Tables {
    fn new(sampling_frequency: SamplingFrequency, transform: Transform) -> Self {
        let partitions = partitions::table(sampling_frequency, transform);
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
        let spread_from = (0..n)
            .map(|b| {
                let spreads = |bb: &usize| spreading[bb * n + b] != 0.0;
                let first = (0..n)
                    .find(spreads)
                    .expect("a partition spreads into itself");
                let last = (0..n)
                    .rev()
                    .find(spreads)
                    .expect("a partition spreads into itself");
                debug_assert!((first..=last).all(|bb| spreads(&bb)));
                first..last + 1
            })
            .collect();
        // The FFT's shift length is its window's lines.
        let reference = quiet_reference(transform.lines());
        let quiet = (partitions.iter())
            .map(|p| reference * 10f64.powf(p.qsthr / 10.0))
            .collect();
        let bands = match transform {
            Transform::Long => GroupBands::long(sampling_frequency),
            Transform::Short => GroupBands::short(sampling_frequency, 1),
        };
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
            spread_from,
            rnorm,
            quiet,
            bands,
            band_partitions,
        }
    }

    fn get(sampling_frequency: SamplingFrequency, transform: Transform) -> &'static Self {
        static TABLES: [[OnceLock<Tables>; 2]; 12] = [const { [const { OnceLock::new() }; 2] }; 12];
        let index = usize::from(transform == Transform::Short);
        TABLES[sampling_frequency.index() as usize][index]
            .get_or_init(|| Self::new(sampling_frequency, transform))
    }

    /// Steps 5 to 12 and the FFT's part of step 14 for one block, from the
    /// energy `r(w)^2` and unpredictability `c(w)` of each line of its FFT,
    /// with pre-echo control against `nb_last`, the threshold of the last
    /// block of this FFT, which it updates, raised by `rpelev`.
    ///
    /// Spec gap: step 11 compares the threshold with `nb_l(b)`, "the
    /// threshold of partition b for the last block". Both step 10 and this
    /// step call their result `nb(b)`, so the text does not say which is
    /// meant; the more literal reading is the
    /// threshold this step gives, which is left in `nb(b)`. This
    /// deliberately departs from it, and takes the threshold of step 10:
    /// the energy threshold the last block's signal set, so that a block
    /// after a quieter one, as at an attack, is held near the quieter one's
    /// threshold, while a steady signal keeps its own. Taking the threshold
    /// this step gives would let a long block's threshold rise by `rpelev`,
    /// 3 dB, a frame at most, so that after silence a loud signal would
    /// take some twenty frames to reach its own; and a short block's, whose
    /// `rpelev` is 1, never rise at all but through the threshold in quiet.
    /// The block before the first is silent, with no threshold.
    fn threshold(
        &self,
        energy: &[f64],
        unpredictability: &[f64],
        nb_last: &mut [f64],
        rpelev: f64,
        out: &mut Threshold,
    ) {
        // Step 5: the energy and weighted unpredictability of each
        // partition.
        let n = self.partitions.len();
        let mut e = [0.0; MAX_PARTITIONS];
        let mut c = [0.0; MAX_PARTITIONS];
        for (b, p) in self.partitions.iter().enumerate() {
            for w in p.lines() {
                e[b] += energy[w];
                c[b] += energy[w] * unpredictability[w];
            }
        }

        let mut perceptual_entropy = 0.0;
        let mut nb = [0.0; MAX_PARTITIONS];
        for (b, nb) in nb[..n].iter_mut().enumerate() {
            // Step 6: spread, and normalized; the other partitions spread
            // nothing into this one.
            let (mut ecb, mut ct) = (0.0, 0.0);
            for bb in self.spread_from[b].clone() {
                let spreading = self.spreading[bb * n + b];
                ecb += e[bb] * spreading;
                ct += c[bb] * spreading;
            }
            // Spec gap: without energy `cb(b) = ct(b) / ecb(b)` is
            // undefined; its threshold is 0 whatever the tonality, and the
            // threshold in quiet sets it (step 11). It is taken as 0.
            let cb = if ecb == 0.0 { 0.0 } else { ct / ecb };
            let en = ecb * self.rnorm[b];
            // Step 7: the tonality, within 0 and 1.
            let tb = (-0.299 - 0.43 * cb.ln()).clamp(0.0, 1.0);
            // Steps 8 to 10.
            let snr = tb * TMN + (1.0 - tb) * NMT;
            let bc = 10f64.powf(-snr / 10.0);
            *nb = en * bc;
        }
        // Step 11, against the last block's nb(b) of step 10.
        for b in 0..n {
            let limited = nb[b].min(nb_last[b] * rpelev);
            nb_last[b] = nb[b];
            nb[b] = self.quiet[b].max(limited);
            // Step 12.
            let p = &self.partitions[b];
            let weight = f64::from(p.w_high - p.w_low);
            perceptual_entropy -= weight * (nb[b] / (e[b] + 1.0)).log10();
        }

        // Step 14.
        let bands = self.bands;
        out.bands = bands;
        out.perceptual_entropy = perceptual_entropy;
        for (band, range) in self.band_partitions.iter().enumerate() {
            let lines = bands.offset(band)..bands.offset(band + 1);
            out.epart[band] = energy[lines.clone()].iter().sum();
            let thr = (range.clone())
                .map(|b| nb[b] / self.partitions[b].lines().len() as f64)
                .fold(f64::INFINITY, f64::min);
            out.npart[band] = thr * lines.len() as f64;
        }
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

/// The threshold of one block, long or short, in each scalefactor band of
/// its window, on the scale of the FFT, and its perceptual entropy (steps
/// 12 and 14).
#[derive(Clone, Debug, PartialEq)]
pub struct Threshold {
    /// `epart(n)`: the FFT's energy in each band.
    pub epart: [f64; MAX_BANDS],
    /// `npart(n)`: the noise level each band allows.
    pub npart: [f64; MAX_BANDS],
    /// `PE`.
    pub perceptual_entropy: f64,
    bands: GroupBands,
}

impl Threshold {
    fn new(bands: GroupBands) -> Self {
        Self {
            epart: [0.0; MAX_BANDS],
            npart: [0.0; MAX_BANDS],
            perceptual_entropy: 0.0,
            bands,
        }
    }

    /// `xmin(n) = npart(n) * codec_e(n) / epart(n)` (step 14): the energy
    /// of quantization error that band `n` of the block's window may take,
    /// where `codec_e(n)` is the energy of its MDCT lines.
    ///
    /// Spec gap: a band whose FFT has no energy makes the ratio undefined.
    /// Its signal is below any threshold, as `SMR(n) = epart(n) / npart(n)`
    /// is 0, so it is allowed its whole MDCT energy, `codec_e(n)`.
    pub fn xmin(&self, n: usize, codec_e: f64) -> f64 {
        if self.epart[n] == 0.0 {
            codec_e
        } else {
            self.npart[n] * codec_e / self.epart[n]
        }
    }
}

/// What the model gives for a frame of one channel: the threshold of its
/// long block and those of its eight short blocks.
#[derive(Clone, Debug, PartialEq)]
pub struct Analysis {
    pub long: Threshold,
    pub short: [Threshold; SHORT_WINDOWS],
    /// The largest rise of the perceptual entropy from one short block to
    /// the next over the frame's eight, from the last of the frame before,
    /// by which [`switching`] finds an attack.
    pub entropy_rise: f64,
}

impl Analysis {
    /// The analysis of a frame not yet analyzed, at `sampling_frequency`.
    pub fn new(sampling_frequency: SamplingFrequency) -> Self {
        Self {
            long: Threshold::new(GroupBands::long(sampling_frequency)),
            short: std::array::from_fn(|_| {
                Threshold::new(GroupBands::short(sampling_frequency, 1))
            }),
            entropy_rise: 0.0,
        }
    }

    /// The perceptual entropy of the frame coded with `sequence`: that of
    /// its long block, or for eight short windows, those of the eight
    /// short blocks summed (step 15).
    pub fn perceptual_entropy(&self, sequence: WindowSequence) -> f64 {
        match sequence.transform() {
            Transform::Long => self.long.perceptual_entropy,
            Transform::Short => self.short.iter().map(|t| t.perceptual_entropy).sum(),
        }
    }

    /// `bit_allocation = pew1 * PE + pew2 * sqrt(PE)`, limited to 0 to
    /// 3000 (step 15), with the constants and entropy of `sequence`'s
    /// blocks: `pew1 = 0.3` and `pew2 = 6.0` for long blocks, and `pew1 =
    /// 0.6` and `pew2 = 24` for short ones.
    ///
    /// Spec gap: the entropy is negative where the thresholds exceed the
    /// energy, as in silence, -9535 at 44.1 kHz, and in most frames of
    /// music (see [`Model::analyze`]), where `sqrt(PE)` is undefined; the
    /// allocation is then 0, the lower limit.
    pub fn bit_allocation(&self, sequence: WindowSequence) -> f64 {
        let (pew1, pew2) = match sequence.transform() {
            Transform::Long => (PEW1_LONG, PEW2_LONG),
            Transform::Short => (PEW1_SHORT, PEW2_SHORT),
        };
        let pe = self.perceptual_entropy(sequence).max(0.0);
        (pew1 * pe + pew2 * pe.sqrt()).min(MAX_BIT_ALLOCATION)
    }

    /// The allowed distortion `xmin` of each band of each window group of
    /// a frame coded with `windows`, whose lines in the order they are sent
    /// are `spectrum`, into `xmin`, a group at a time: for a long window,
    /// its bands' `xmin(n)` of the long block (step 14), and for eight short
    /// windows, those of each window's short block, summed over the windows
    /// of each group.
    ///
    /// Spec gap: Annex C gives `xmin(n)` for each short block, and the loops
    /// shape the noise of a group's scalefactor band, which holds that band
    /// of each of its windows (clause C.7.1, item 6), with one scalefactor;
    /// it does not say what such a band is allowed. It is allowed the sum
    /// of what each of its windows' bands is allowed, as a band of a long
    /// window is allowed the noise of each of its lines summed (step 14):
    /// the loops measure the distortion of the group's band as the error of
    /// all of its lines (clause C.7.4.3). The windows grouped are those of
    /// like energy (see [`grouping`](switching::grouping)), whose bands
    /// are allowed alike.
    pub fn xmin(
        &self,
        windows: Windows,
        spectrum: &[f64; FRAME_LINES],
        xmin: &mut Vec<[f64; MAX_BANDS]>,
    ) {
        let energy = |lines: &[f64]| -> f64 { lines.iter().map(|x| x * x).sum() };
        xmin.clear();
        match windows.sequence {
            WindowSequence::EightShort { grouping } => {
                let mut start = 0;
                let mut first = 0;
                for len in grouping.lengths() {
                    let mut group = [0.0; MAX_BANDS];
                    let bands = self.short[0].bands;
                    for (n, xmin) in group[..bands.num_swb()].iter_mut().enumerate() {
                        let width = bands.offset(n + 1) - bands.offset(n);
                        let band = start + bands.offset(n) * len;
                        for (i, threshold) in self.short[first..first + len].iter().enumerate() {
                            let lines = &spectrum[band + i * width..][..width];
                            *xmin += threshold.xmin(n, energy(lines));
                        }
                    }
                    xmin.push(group);
                    start += bands.lines() * len;
                    first += len;
                }
            }
            _ => {
                let bands = self.long.bands;
                let mut group = [0.0; MAX_BANDS];
                for (n, xmin) in group[..bands.num_swb()].iter_mut().enumerate() {
                    let lines = &spectrum[bands.offset(n)..bands.offset(n + 1)];
                    *xmin = self.long.xmin(n, energy(lines));
                }
                xmin.push(group);
            }
        }
    }
}

/// The model of one channel, with what it keeps from one frame to the
/// next.
#[derive(Debug)]
pub struct Model {
    long: &'static Tables,
    short: &'static Tables,
    /// The newest `iblen` samples of the frame before (step 1), at
    /// [`PCM_SCALE`].
    previous: Box<[f64; LONG_IBLEN]>,
    long_fft: RealFft,
    short_fft: RealFft,
    long_hann: Vec<f64>,
    short_hann: Vec<f64>,
    long_predictor: Predictor,
    short_predictor: Predictor,
    /// `nb(b)` of the last long block and of the last short block, before
    /// pre-echo control (step 11).
    long_nb_last: Vec<f64>,
    short_nb_last: Vec<f64>,
    // Buffers kept from one frame to the next for their memory.
    windowed: Vec<f64>,
    energy: Vec<f64>,
    c: Vec<f64>,
    short_energy: Vec<f64>,
    short_c: Vec<f64>,
    short_c_min: Vec<f64>,
    /// The perceptual entropy of the last short block, `None` before the
    /// first.
    last_short_entropy: Option<f64>,
}

impl Model {
    /// The model of a channel at `sampling_frequency`, before its first
    /// frame: the block before it and the blocks its predictions start from
    /// silent, as the filterbank takes them (see
    /// [`Filterbank::new`](crate::filterbank::Filterbank::new)).
    pub fn new(sampling_frequency: SamplingFrequency) -> Self {
        let long = Tables::get(sampling_frequency, Transform::Long);
        let short = Tables::get(sampling_frequency, Transform::Short);
        Self {
            long,
            short,
            previous: Box::new([0.0; LONG_IBLEN]),
            long_fft: RealFft::new(2 * LONG_IBLEN),
            short_fft: RealFft::new(2 * SHORT_IBLEN),
            long_hann: hann(LONG_IBLEN),
            short_hann: hann(SHORT_IBLEN),
            long_predictor: Predictor::new(LONG_CW_LINES),
            short_predictor: Predictor::new(SHORT_IBLEN + 1),
            long_nb_last: vec![0.0; long.partitions.len()],
            short_nb_last: vec![0.0; short.partitions.len()],
            windowed: vec![0.0; 2 * LONG_IBLEN],
            energy: vec![0.0; LONG_IBLEN],
            c: vec![0.0; LONG_IBLEN],
            short_energy: vec![0.0; SHORT_IBLEN + 1],
            short_c: vec![0.0; SHORT_IBLEN + 1],
            short_c_min: vec![0.0; SHORT_IBLEN + 1],
            last_short_entropy: None,
        }
    }

    /// Analyzes the frame whose block is `block`, the channel's next 1024
    /// samples at full scale 1.0, into `out`: its window of `2 * iblen`
    /// samples, the block before and this one (step 1), is the window of
    /// the frame's MDCT (clause C.3.1.1), in which the psychoacoustic
    /// model's window is thus centred (clause C.1.1). The short FFTs are
    /// those of the frame's eight short windows (clause 15.3.2), each
    /// centred, as the long FFT is, in the window of its transform.
    ///
    /// The thresholds of the long and of the short blocks are computed for
    /// every frame, as the two processes of clause C.1.1, each with its
    /// shift length, run over the whole signal: each short block's
    /// threshold is limited by the last short block's (step 11), across
    /// frames, whatever the windows the frames are coded with.
    ///
    /// Spec gap: step 4 takes the unpredictability of a long block's lines
    /// from the sixth on as "the minimum of the unpredictability of all
    /// short FFT's", whose lines are eight times as wide, and does not say
    /// which of their lines a long line takes. Each line of the long FFT
    /// takes that of the short FFTs' line nearest it in frequency, `(w +
    /// 4) / 8`.
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
    ///
    /// Step 4 says the formula of `c(w)` "is used for each of the short
    /// blocks with the short FFT": each short block's own unpredictability
    /// is that of its lines, predicted from the two short blocks before it,
    /// across frames.
    pub fn analyze(&mut self, block: &[f64; FRAME_SAMPLES], out: &mut Analysis) {
        // Step 1: the window of the block before and this one.
        let (before, after) = self.windowed.split_at_mut(LONG_IBLEN);
        before.copy_from_slice(&*self.previous);
        for (s, &x) in after.iter_mut().zip(block) {
            *s = x * PCM_SCALE;
        }
        self.previous.copy_from_slice(after);

        // The short blocks, steps 2 to 14 for each, and for the long block
        // the least unpredictability of each line over the eight (step 4).
        self.short_c_min.fill(f64::INFINITY);
        let mut short_windowed = [0.0; 2 * SHORT_IBLEN];
        for (k, threshold) in out.short.iter_mut().enumerate() {
            let start = FIRST_SHORT + k * SHORT_IBLEN;
            let samples = &self.windowed[start..start + 2 * SHORT_IBLEN];
            for ((sw, &s), &w) in short_windowed.iter_mut().zip(samples).zip(&self.short_hann) {
                *sw = s * w;
            }
            let x = self.short_fft.transform(&short_windowed);
            for (energy, x) in self.short_energy.iter_mut().zip(x) {
                *energy = x.re * x.re + x.im * x.im;
            }
            self.short_predictor.unpredictability(x, &mut self.short_c);
            for (min, &c) in self.short_c_min.iter_mut().zip(&self.short_c) {
                *min = min.min(c);
            }
            self.short.threshold(
                &self.short_energy,
                &self.short_c,
                &mut self.short_nb_last,
                RPELEV_SHORT,
                threshold,
            );
        }
        // The first short block of the stream rises from itself.
        let mut last = (self.last_short_entropy).unwrap_or(out.short[0].perceptual_entropy);
        out.entropy_rise = f64::NEG_INFINITY;
        for threshold in &out.short {
            out.entropy_rise = out.entropy_rise.max(threshold.perceptual_entropy - last);
            last = threshold.perceptual_entropy;
        }
        self.last_short_entropy = Some(last);

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
        // Steps 5 to 14.
        self.long.threshold(
            &self.energy,
            &self.c,
            &mut self.long_nb_last,
            RPELEV_LONG,
            &mut out.long,
        );
    }
}

/// The most partitions of any table: the 72 of the long FFT at 88.2 kHz
/// (Table C.21).
const MAX_PARTITIONS: usize = 72;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::window::Grouping;

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

    fn run(fs: SamplingFrequency, input: &[f64]) -> Vec<Analysis> {
        let mut model = Model::new(fs);
        input
            .chunks_exact(FRAME_SAMPLES)
            .map(|block| {
                let mut analysis = Analysis::new(fs);
                model.analyze(block.try_into().unwrap(), &mut analysis);
                analysis
            })
            .collect()
    }

    fn long(fs: SamplingFrequency, input: &[f64]) -> Vec<Threshold> {
        run(fs, input).into_iter().map(|a| a.long).collect()
    }

    #[test]
    fn silence_has_the_threshold_in_quiet_and_no_entropy_to_speak_of() {
        let fs = SamplingFrequency::Hz44100;
        for analysis in run(fs, &vec![0.0; 3 * FRAME_SAMPLES]) {
            for threshold in std::iter::once(&analysis.long).chain(&analysis.short) {
                assert_eq!(threshold.epart, [0.0; MAX_BANDS]);
                assert!(threshold.perceptual_entropy < 0.0);
                for n in 0..threshold.bands.num_swb() {
                    assert!(threshold.npart[n] > 0.0);
                }
            }
            for sequence in [WindowSequence::OnlyLong, WindowSequence::EIGHT_SHORT] {
                assert_eq!(analysis.bit_allocation(sequence), 0.0);
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
            let threshold = long(fs, input).pop().unwrap();
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
        let thresholds = long(fs, &input);
        // The frame whose window first holds the tone is held at twice the
        // last block's threshold, of silence, and so at the threshold in
        // quiet; the frames after it take their own.
        let band = 20; // lines 132 to 143, 2842 to 3101 Hz
        let tables = Tables::get(fs, Transform::Long);
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
        let analysis = run(fs, &input).pop().unwrap();
        let mut spectrum = [0.0; FRAME_LINES];
        spectrum[100] = 2.0;
        spectrum[101] = -1.0;
        let mut xmin = Vec::new();
        analysis.xmin(Windows::LONG_SINE, &spectrum, &mut xmin);
        assert_eq!(xmin.len(), 1);
        let bands = GroupBands::long(fs);
        let band = (0..bands.num_swb())
            .find(|&n| bands.offset(n + 1) > 100)
            .unwrap();
        let threshold = &analysis.long;
        let expected = threshold.npart[band] * 5.0 / threshold.epart[band];
        assert!((xmin[0][band] - expected).abs() <= 1e-12 * expected);
        // A band without MDCT energy allows none.
        assert_eq!(xmin[0][band + 5], 0.0);
    }

    /// A group's band is allowed what each of its windows' bands is
    /// allowed, summed, each from its own lines, which the group sends
    /// interleaved.
    #[test]
    fn a_groups_xmin_sums_its_windows() {
        let fs = SamplingFrequency::Hz44100;
        let input = tone(2500.0, 0.3, 3 * FRAME_SAMPLES, 44_100.0);
        let analysis = run(fs, &input).pop().unwrap();
        let grouping = Grouping::from_starts([true, true, false, false, true, true, true, true]);
        let windows = Windows {
            sequence: WindowSequence::EightShort { grouping },
            shape: crate::window::WindowShape::Sine,
        };
        let spectrum: [f64; FRAME_LINES] = std::array::from_fn(|i| (i % 7) as f64 - 3.0);
        let mut xmin = Vec::new();
        analysis.xmin(windows, &spectrum, &mut xmin);
        assert_eq!(xmin.len(), 6);
        // Band 9 (lines 56 to 67 of each window, Table 46) of the second
        // group, of windows 1 to 3, which starts at line 128.
        let band = 128 + 3 * 56;
        let expected: f64 = (0..3)
            .map(|i| {
                let lines = &spectrum[band + 12 * i..][..12];
                let codec_e: f64 = lines.iter().map(|x| x * x).sum();
                analysis.short[1 + i].xmin(9, codec_e)
            })
            .sum();
        assert!((xmin[1][9] - expected).abs() <= 1e-12 * expected);
        assert!(expected > 0.0);
    }

    /// Each short block sees what lies in its own window: an attack late in
    /// the frame's block raises the entropy of the short blocks over it
    /// alone, and that of the long block over all of them.
    #[test]
    fn short_blocks_place_an_attack_in_time() {
        let fs = SamplingFrequency::Hz44100;
        let mut input = tone(1000.0, 0.01, 4 * FRAME_SAMPLES, 44_100.0);
        let mut state = 9u64;
        // Noise from sample 500 of the fourth block, which of the short
        // windows of its frame only window 7 spans: window j spans samples
        // 128 j - 576 to 128 j - 321 of the frame's block.
        for x in &mut input[3 * FRAME_SAMPLES + 500..] {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            *x += 0.3 * ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5);
        }
        let analyses = run(fs, &input);
        let pe = |a: &Analysis, w: usize| a.short[w].perceptual_entropy;
        let attack = &analyses[3];
        let short: Vec<f64> = (0..8).map(|w| pe(attack, w)).collect();
        assert!(
            short[..7].iter().all(|&pe| pe < short[7] - 100.0),
            "{short:?}"
        );
        let (before, after) = (
            analyses[2].long.perceptual_entropy,
            attack.long.perceptual_entropy,
        );
        assert!(after > before + 1000.0, "{after} against {before}");
    }
}
