// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The encoder's psychoacoustic model: how much noise each line of a long
//! block may take, from which the floor is set.
//!
//! The specification describes no encoder (clause 1.1.5), and this model is
//! chelys's own. It follows the psychoacoustic model of ISO/IEC 13818-7
//! (clause C.1), which chelys's AAC encoder implements, with these
//! differences:
//!
//! - Its partitions are those of [`partitions`], at any sample rate.
//! - Its long FFT is that of a long block, the samples of the block's
//!   MDCT, and its eight short FFTs those of the eight short blocks that
//!   would take its place, centered on 128 samples apart from the 576th
//!   of the block: each of 256 samples from sample `448 + 128 k`, as in
//!   AAC. Blocks are 1024 samples apart, as AAC's frames are.
//! - The threshold in quiet of each partition is the level of one of its
//!   lines times its width in Bark, as chelys's AAC encoder reads it at
//!   higher bitrates (`QuietThreshold::Noise` there): half the distance
//!   between the `bval(b)` of its neighbours, or the distance to its one
//!   neighbour at either end. The partitions at the highest `bval(b)` of
//!   the tables, 24, whose width the tables do not give, take
//!   [`TOP_WIDTH_BARK`], and the partition below them its width from its
//!   lower neighbour alone. All are lowered by [`QUIET_LOWERED_DB`].
//! - Nothing past step 11 is taken: the threshold of each partition is
//!   shared evenly among its lines, and becomes the noise each line of the
//!   block's spectrum may take, as noise at the scale of the spectrum
//!   that the decoder takes (see [`Model::analyze`]).
//!
//! The floor is the quantizer's step, set at each post so that rounding to
//! it makes no more noise than the model allows the lines about the post
//! (see [`floor_targets`]).

pub mod partitions;

use std::f64::consts::PI;

use chelys_core::fft::{Complex, RealFft};
use chelys_core::psychoacoustic::spreading_function;

use super::floor::{Floor, INVERSE_DB_TABLE, MULTIPLIER, RANGE};
use partitions::{LINES, Partition};

/// The samples of a long block, and of the long FFT.
const BLOCK: usize = 2 * LINES;

/// The samples of a short block, and of a short FFT.
const SHORT_BLOCK: usize = 256;

/// The short blocks that take the place of a long one.
const SHORT_BLOCKS: usize = 8;

/// The first sample of the first short block in the long block whose place
/// the short blocks take: a short block after a long one is centered
/// `2048 / 4 + 256 / 4` samples after the long one's center (clause 4.3.8),
/// 576 samples into the long block it replaces, which begins 1024 samples
/// before the long one's center.
const FIRST_SHORT: usize = BLOCK / 4 + SHORT_BLOCK / 4 - SHORT_BLOCK / 2;

/// The lines of the long FFT whose unpredictability the long FFT gives;
/// the short FFTs give it above (clause C.1.4, step 4).
const LONG_CW_LINES: usize = 6;

/// `NMT(b)` and `TMN(b)` in dB, for all `b` (step 8).
const NMT: f64 = 6.0;
const TMN: f64 = 18.0;

/// `rpelev` of long blocks (step 11).
const RPELEV: f64 = 2.0;

/// The highest `bval(b)` of the tables, at which those of the highest
/// partitions stop rising.
const TOP_BARK: f64 = 24.0;

/// The width in Bark of each partition at [`TOP_BARK`], from about 15 kHz
/// up, by which the level of one of its lines is multiplied for its
/// threshold in quiet: one critical band. Those partitions span 1 to 3 kHz
/// at 44.1 and 48 kHz, no wider than the critical band below them, of
/// about 3.5 kHz.
///
/// chelys's AAC encoder takes their threshold as printed, the level of a
/// line times their lines, 17 to 19 dB more over their 50 to 80 lines.
/// That takes everything above about 16 kHz out of most music: graded
/// with PEAQ (GstPEAQ 0.6.1, black box), on 30 s of each of 8 tracks of
/// the tuning set, one of each kind of music but live recordings, with
/// the threshold not lowered, the streams' mean grade was -1.496 as
/// printed and -1.387 at one Bark, for 1% more bits.
const TOP_WIDTH_BARK: f64 = 1.0;

/// How far, in dB, the threshold in quiet is lowered below the tables'.
///
/// The tables put it relative to an LSB of 16-bit PCM (clause C.1.4, step
/// 11), whatever the level the signal is played at. The floor follows the
/// threshold wherever the music lies below it, as in quiet passages, which
/// are then coded with noise just below it, and with no allowance for a
/// louder playback. Lowered by 20 dB, as chelys's AAC encoder lowers it at
/// variable rates for the same reason, the 8 clips of [`TOP_WIDTH_BARK`]
/// graded -0.629, against -1.387 unlowered, for 20% more bits; and
/// against -0.850 with the encoder's earlier rule, noise a fixed 9.5 dB
/// below the power about each post of the floor, at the same mean
/// bitrate. On the whole of the tuning set's 51 tracks, at 44.1 kHz in
/// stereo, the model graded -0.578 at 189 kbit/s, and the earlier rule
/// -0.712 at the same size, 9.65 dB below; the model graded higher on 30
/// tracks, by 0.43 to 0.59 on average on electronic music, hip-hop and
/// pop, and lower by 0.03 to 0.15 on the other kinds but rock.
const QUIET_LOWERED_DB: f64 = 20.0;

/// The energy, in the long FFT of samples at full scale 1.0, of the line
/// that a sine wave of +-1/2 LSB of 16-bit PCM is centered on, to which
/// the tables' thresholds in quiet are relative (step 11): `a / 2` times
/// the sum of the Hann window, `BLOCK / 2`, for `a = 1 / 65536`, squared.
const QUIET_REFERENCE: f64 = (BLOCK as f64 / 4.0 / 65536.0) * (BLOCK as f64 / 4.0 / 65536.0);

/// The power of noise in a line of the spectrum that the decoder takes,
/// for each unit of the energy of the same noise in a line of the long
/// FFT: white noise of power `P` has the energy `P` times the sum of the
/// squared Hann window, `3 N / 8`, in each line of the FFT of `N` samples,
/// and the power `4 P / N` in each line of the spectrum (see
/// [`ABSOLUTE_NOISE`]'s scale).
const NOISE_SCALE: f64 = 4.0 / BLOCK as f64 / (3.0 * BLOCK as f64 / 8.0);

/// The least noise power a line is allowed, at the decoder's scale of the
/// spectrum: that of the rounding of 16-bit PCM, white noise of power `1 /
/// (12 * 32768^2)`, some 101 dB below full scale, whose power in each line
/// of a long block is `4 / N` times its own, `N` the block's 2048 samples.
const ABSOLUTE_NOISE: f64 = 4.0 / BLOCK as f64 / (12.0 * 32768.0 * 32768.0);

/// What the model computes once for each sample rate.
#[derive(Debug)]
pub struct Tables {
    partitions: Vec<Partition>,
    /// `sprdngf(bval(bb), bval(b))` at `bb * n + b`, for `n` partitions.
    spreading: Vec<f64>,
    /// `rnorm(b)` (step 6).
    rnorm: Vec<f64>,
    /// The threshold in quiet of each partition as an energy of the long
    /// FFT.
    quiet: Vec<f64>,
}

impl Tables {
    pub fn new(sample_rate: u32) -> Self {
        let partitions = partitions::partitions(sample_rate);
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
        Self {
            quiet: quiet(&partitions),
            partitions,
            spreading,
            rnorm,
        }
    }

    /// Each partition's values convolved with the spreading function (step
    /// 6).
    fn spread(&self, values: &[f64], out: &mut [f64]) {
        let n = self.partitions.len();
        out.fill(0.0);
        for (bb, &value) in values.iter().enumerate() {
            if value != 0.0 {
                let row = &self.spreading[bb * n..(bb + 1) * n];
                for (out, &sprdngf) in out.iter_mut().zip(row) {
                    *out += value * sprdngf;
                }
            }
        }
    }
}

/// The threshold in quiet of each of `partitions` as an energy of the long
/// FFT, as the [module](self) describes.
fn quiet(partitions: &[Partition]) -> Vec<f64> {
    let bval = |b: usize| partitions[b].bval;
    let last = partitions
        .iter()
        .position(|p| p.bval >= TOP_BARK)
        .unwrap_or(partitions.len())
        .saturating_sub(1);
    (partitions.iter().enumerate())
        .map(|(b, p)| {
            let line = QUIET_REFERENCE * 10f64.powf((p.quiet_db - QUIET_LOWERED_DB) / 10.0);
            let width = if b > last || last == 0 {
                TOP_WIDTH_BARK
            } else {
                match b {
                    0 => bval(1) - bval(0),
                    _ if b == last => bval(last) - bval(last - 1),
                    _ => (bval(b + 1) - bval(b - 1)) / 2.0,
                }
            };
            line * width
        })
        .collect()
}

/// The Hann window of 2 `iblen` samples (step 2): `0.5 - 0.5 cos(pi (i +
/// 0.5) / iblen)`.
fn hann(samples: usize) -> Vec<f64> {
    let iblen = (samples / 2) as f64;
    (0..samples)
        .map(|i| 0.5 - 0.5 * (PI * (i as f64 + 0.5) / iblen).cos())
        .collect()
}

/// Step 7: the tonality `tb(b)` of a partition of normalized
/// unpredictability `cb(b)`, within 0 and 1.
fn tonality(cb: f64) -> f64 {
    (-0.299 - 0.43 * cb.ln()).clamp(0.0, 1.0)
}

/// Steps 8 and 9: the power ratio `bc(b)` of a partition of tonality
/// `tb(b)`.
fn power_ratio(tb: f64) -> f64 {
    let snr = tb * TMN + (1.0 - tb) * NMT;
    10f64.powf(-snr / 10.0)
}

/// The magnitude and phase of each line of an FFT's last two blocks, from
/// which steps 3 and 4 predict the next.
#[derive(Debug)]
struct Predictor {
    /// Of each line, the magnitude and `e^(i f)` of the last block and of
    /// the block before it.
    last: Vec<(f64, Complex)>,
    before_last: Vec<(f64, Complex)>,
}

/// A line of the silence the model starts from: no magnitude, and the
/// phase 0 (clause C.1.1).
const SILENT_LINE: (f64, Complex) = (0.0, Complex { re: 1.0, im: 0.0 });

impl Predictor {
    fn new(lines: usize) -> Self {
        Self {
            last: vec![SILENT_LINE; lines],
            before_last: vec![SILENT_LINE; lines],
        }
    }

    /// The unpredictability `c(w)` of the lines `x` of the next block, into
    /// `c`, and the block kept as the last.
    ///
    /// `c(w) = |r e^(i f) - r_pred e^(i f_pred)| / (r + |r_pred|)`, with
    /// `r_pred = 2 r(t-1) - r(t-2)` and `f_pred = 2 f(t-1) - f(t-2)`. A
    /// line without magnitude takes the phase 0, and one without magnitude
    /// and none predicted is wholly unpredictable, 1, as in chelys's AAC
    /// encoder.
    fn unpredictability(&mut self, x: &[Complex], c: &mut [f64]) {
        for (w, (x, c)) in x.iter().zip(c).enumerate() {
            let ((r1, f1), (r2, f2)) = (self.last[w], self.before_last[w]);
            let r_pred = 2.0 * r1 - r2;
            let conj = Complex {
                re: f2.re,
                im: -f2.im,
            };
            let predicted = f1 * f1 * conj;
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
                SILENT_LINE.1
            } else {
                Complex {
                    re: x.re / r,
                    im: x.im / r,
                }
            };
            self.before_last[w] = self.last[w];
            self.last[w] = (r, phase);
        }
    }
}

/// The model of one channel, with what it keeps from one block to the
/// next.
#[derive(Debug)]
pub struct Model {
    /// The factor by which each partition's threshold is scaled, from the
    /// rate's noise offset.
    allowance: f64,
    long_fft: RealFft,
    short_fft: RealFft,
    long_hann: Vec<f64>,
    short_hann: Vec<f64>,
    long_predictor: Predictor,
    short_predictor: Predictor,
    /// `nb(b)` of the last block, before pre-echo control (step 11).
    nb_last: Vec<f64>,
    // Buffers kept from one block to the next for their memory.
    windowed: Vec<f64>,
    short_windowed: Vec<f64>,
    energy: Vec<f64>,
    c: Vec<f64>,
    short_c: Vec<f64>,
    short_c_min: Vec<f64>,
    e: Vec<f64>,
    weighted: Vec<f64>,
    ecb: Vec<f64>,
    ct: Vec<f64>,
}

impl Model {
    /// The model of a channel before its first block, which follows
    /// silence, which allows each partition its threshold scaled by
    /// `10^(noise_offset_db / 10)`.
    pub fn new(tables: &Tables, noise_offset_db: f64) -> Self {
        let n = tables.partitions.len();
        Self {
            allowance: 10f64.powf(noise_offset_db / 10.0),
            long_fft: RealFft::new(BLOCK),
            short_fft: RealFft::new(SHORT_BLOCK),
            long_hann: hann(BLOCK),
            short_hann: hann(SHORT_BLOCK),
            long_predictor: Predictor::new(LONG_CW_LINES),
            short_predictor: Predictor::new(SHORT_BLOCK / 2 + 1),
            nb_last: vec![0.0; n],
            windowed: vec![0.0; BLOCK],
            short_windowed: vec![0.0; SHORT_BLOCK],
            energy: vec![0.0; LINES],
            c: vec![0.0; LINES],
            short_c: vec![0.0; SHORT_BLOCK / 2 + 1],
            short_c_min: vec![0.0; SHORT_BLOCK / 2 + 1],
            e: vec![0.0; n],
            weighted: vec![0.0; n],
            ecb: vec![0.0; n],
            ct: vec![0.0; n],
        }
    }

    /// The noise power that each line of the spectrum of the long block of
    /// `samples`, at full scale 1.0, may take, into `noise`, at the scale
    /// of the spectrum that the decoder takes; each call takes the block
    /// 1024 samples after the last.
    ///
    /// Each partition's threshold, an energy of the long FFT, is scaled by
    /// the model's allowance, shared evenly among its lines, and taken as
    /// noise: each line's share times [`NOISE_SCALE`]. Line `k` of the spectrum, centered at `k + 1/2`
    /// lines of the FFT, takes the share of line `k` of the FFT. No line
    /// takes less than [`ABSOLUTE_NOISE`].
    pub fn analyze(&mut self, tables: &Tables, samples: &[f64], noise: &mut [f64]) {
        assert_eq!(samples.len(), BLOCK);
        assert_eq!(noise.len(), LINES);

        // The short blocks, and for the long block the least
        // unpredictability of each of their lines over the eight (step 4).
        self.short_c_min.fill(f64::INFINITY);
        for k in 0..SHORT_BLOCKS {
            let start = FIRST_SHORT + k * SHORT_BLOCK / 2;
            let short = &samples[start..start + SHORT_BLOCK];
            for ((s, &x), &w) in self
                .short_windowed
                .iter_mut()
                .zip(short)
                .zip(&self.short_hann)
            {
                *s = x * w;
            }
            let x = self.short_fft.transform(&self.short_windowed);
            self.short_predictor.unpredictability(x, &mut self.short_c);
            for (min, &c) in self.short_c_min.iter_mut().zip(&self.short_c) {
                *min = min.min(c);
            }
        }

        // Steps 2 to 4: the long FFT, the energy of its lines, and their
        // unpredictability, each line from the sixth on taking that of the
        // short FFTs' line nearest it in frequency, as in chelys's AAC
        // encoder.
        for ((s, &x), &w) in self.windowed.iter_mut().zip(samples).zip(&self.long_hann) {
            *s = x * w;
        }
        let x = self.long_fft.transform(&self.windowed);
        for (energy, x) in self.energy.iter_mut().zip(x) {
            *energy = x.re * x.re + x.im * x.im;
        }
        self.long_predictor
            .unpredictability(&x[..LONG_CW_LINES], &mut self.c[..LONG_CW_LINES]);
        for (w, c) in self.c.iter_mut().enumerate().skip(LONG_CW_LINES) {
            *c = self.short_c_min[((w + 4) / 8).min(SHORT_BLOCK / 2)];
        }

        // Step 5: each partition's energy and weighted unpredictability.
        for (b, p) in tables.partitions.iter().enumerate() {
            let lines = p.lines.clone();
            self.e[b] = self.energy[lines.clone()].iter().sum();
            self.weighted[b] = (self.energy[lines.clone()].iter())
                .zip(&self.c[lines])
                .map(|(&e, &c)| e * c)
                .sum();
        }

        // Step 6: spread.
        tables.spread(&self.e, &mut self.ecb);
        tables.spread(&self.weighted, &mut self.ct);

        // Steps 7 to 11, each partition's threshold, shared among its lines.
        for (b, p) in tables.partitions.iter().enumerate() {
            let ecb = self.ecb[b];
            // Without energy, `cb(b)` is undefined, and the threshold 0
            // whatever the tonality.
            let cb = if ecb == 0.0 { 0.0 } else { self.ct[b] / ecb };
            let nb = ecb * tables.rnorm[b] * power_ratio(tonality(cb));
            let limited = nb.min(self.nb_last[b] * RPELEV);
            self.nb_last[b] = nb;
            let thr = tables.quiet[b].max(limited) * self.allowance;
            let line = (thr / p.lines.len() as f64 * NOISE_SCALE).max(ABSOLUTE_NOISE);
            noise[p.lines.clone()].fill(line);
        }
    }
}

/// The Y value of each post of `floor`, in its list order, whose floor
/// rounds each line about the post to no more than the noise `noise` gives
/// the line: rounding to a step `s` makes noise of power `s^2 / 12`. The
/// lines about a post are those nearer it than any other post, and it is
/// set for the least noise any of them may take, as the floor between two
/// posts follows a straight line in dB from one to the other.
pub fn floor_targets(noise: &[f64], floor: &Floor) -> Vec<i32> {
    let xs = floor.xs();
    let mut sorted: Vec<u32> = xs.to_vec();
    sorted.sort_unstable();
    let lines = noise.len();
    xs.iter()
        .map(|&x| {
            let i = sorted.binary_search(&x).expect("a post of the floor");
            let low = if i == 0 {
                0
            } else {
                (sorted[i - 1] + x).div_ceil(2)
            };
            let high = sorted.get(i + 1).map_or(x + 1, |&next| (x + next) / 2 + 1);
            let (low, high) = ((low as usize).min(lines - 1), (high as usize).min(lines));
            let least = noise[low..high.max(low + 1)]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            y_value((12.0 * least).sqrt())
        })
        .collect()
}

/// The Y value whose floor is nearest `amplitude`, in dB.
fn y_value(amplitude: f64) -> i32 {
    let lowest = f64::from(INVERSE_DB_TABLE[0]);
    let step = (f64::from(INVERSE_DB_TABLE[255]) / lowest).ln() / 255.0;
    let index = (amplitude / lowest).ln() / step / f64::from(MULTIPLIER);
    (index.round() as i32).clamp(0, RANGE - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn y_values_follow_the_table() {
        for y in 0..RANGE {
            let amplitude = f64::from(INVERSE_DB_TABLE[(y * MULTIPLIER as i32) as usize]);
            assert_eq!(y_value(amplitude), y);
        }
        assert_eq!(y_value(1e-12), 0);
        assert_eq!(y_value(1e6), RANGE - 1);
    }

    #[test]
    fn the_short_blocks_lie_where_they_would_take_a_long_blocks_place() {
        // Centered from 576 to 1472: 576 after the center of the long block
        // before, at 0, and 576 before that of the long block after, at
        // 2048.
        assert_eq!(FIRST_SHORT + SHORT_BLOCK / 2, 576);
        let last = FIRST_SHORT + (SHORT_BLOCKS - 1) * SHORT_BLOCK / 2;
        assert_eq!(last + SHORT_BLOCK / 2 + 576, BLOCK);
    }

    fn analyze(sample_rate: u32, signal: impl Fn(usize) -> f64, blocks: usize) -> Vec<f64> {
        let tables = Tables::new(sample_rate);
        let mut model = Model::new(&tables, 0.0);
        let mut noise = vec![0.0; LINES];
        for k in 0..blocks {
            let samples: Vec<f64> = (0..BLOCK).map(|t| signal(k * LINES + t)).collect();
            model.analyze(&tables, &samples, &mut noise);
        }
        noise
    }

    #[test]
    fn silence_gets_the_threshold_in_quiet() {
        let noise = analyze(44_100, |_| 0.0, 3);
        let tables = Tables::new(44_100);
        for (b, p) in tables.partitions.iter().enumerate() {
            let expected =
                (tables.quiet[b] / p.lines.len() as f64 * NOISE_SCALE).max(ABSOLUTE_NOISE);
            assert!(noise[p.lines.clone()].iter().all(|&n| n == expected));
        }
    }

    /// The threshold of the partition holding `line`, against its energy
    /// in the long FFT, in dB, in the last of `blocks` blocks of `signal`.
    fn margin_db(signal: impl Fn(usize) -> f64, blocks: usize, line: usize) -> f64 {
        let tables = Tables::new(44_100);
        let mut model = Model::new(&tables, 0.0);
        let mut noise = vec![0.0; LINES];
        for k in 0..blocks {
            let samples: Vec<f64> = (0..BLOCK).map(|t| signal(k * LINES + t)).collect();
            model.analyze(&tables, &samples, &mut noise);
        }
        let b = (tables.partitions.iter())
            .position(|p| p.lines.contains(&line))
            .unwrap();
        let thr = noise[line] * tables.partitions[b].lines.len() as f64 / NOISE_SCALE;
        10.0 * (thr / model.e[b]).log10()
    }

    #[test]
    fn a_steady_tone_is_masked_less_than_noise() {
        let mut state = 1u32;
        let white: Vec<f64> = (0..8 * BLOCK)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                f64::from(state >> 8) / f64::from(1 << 24) - 0.5
            })
            .collect();
        let line = 93;
        let noise_db = margin_db(|t| white[t], 6, line);
        assert!((-9.0..-3.0).contains(&noise_db), "{noise_db:.1} dB");
        let f = (line as f64 + 0.25) / BLOCK as f64;
        let tone_db = margin_db(|t| 0.5 * (2.0 * PI * f * t as f64).sin(), 6, line);
        assert!(
            tone_db < noise_db - 8.0,
            "{tone_db:.1} against {noise_db:.1} dB"
        );
    }

    #[test]
    fn a_loud_block_after_silence_is_held_near_the_silence() {
        let onset = 3 * LINES + LINES / 2;
        let loud = |t: usize| {
            if t >= onset {
                0.5 * (t as f64 * 0.3).sin()
            } else {
                0.0
            }
        };
        let held = analyze(44_100, loud, 3);
        let steady = analyze(44_100, |t| 0.5 * (t as f64 * 0.3).sin(), 6);
        let line = (0.3 / (2.0 * PI) * BLOCK as f64) as usize;
        assert!(held[line] < steady[line] / 100.0);
    }

    #[test]
    fn each_post_is_set_for_the_least_noise_about_it() {
        let floor = Floor::new(0);
        let noise: Vec<f64> = (0..LINES).map(|k| 1e-6 * (1.0 + k as f64)).collect();
        let targets = floor_targets(&noise, &floor);
        let mut sorted: Vec<u32> = floor.xs().to_vec();
        sorted.sort_unstable();
        for (&x, &y) in floor.xs().iter().zip(&targets) {
            let i = sorted.binary_search(&x).unwrap();
            let low = if i == 0 {
                0
            } else {
                (sorted[i - 1] + x).div_ceil(2) as usize
            };
            assert_eq!(
                y,
                y_value((12.0 * noise[low.min(LINES - 1)]).sqrt()),
                "post {x}"
            );
        }
    }
}
