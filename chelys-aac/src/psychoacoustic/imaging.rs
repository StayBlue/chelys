// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The thresholds of a channel pair's middle and side channels, and the
//! "imaging control process" that adjusts them and those of its left and
//! right channels (clause C.6.1, steps 1 and 2).

use std::f64::consts::PI;

use chelys_core::fft::Complex;

use super::{
    Analysis, Block, BlockThreshold, MAX_PARTITIONS, Model, Quiet, RPELEV_LONG, RPELEV_SHORT,
    Tables, quiet_energies,
};
use crate::sampling_frequency::SamplingFrequency;
use crate::scalefactor_bands::Transform;
use crate::window::SHORT_WINDOWS;

/// `bmax(b) = 10^(-3 (0.5 + 0.5 cos(pi min(bval(b), 15.5) / 15.5)))`, the
/// BMLD protection ratio of a partition of Bark value `bval` (clause
/// C.6.1): -30 dB at 0 Bark, rising to 0 dB at 15.5 Bark and above.
fn bmax(bval: f64) -> f64 {
    10f64.powf(-3.0 * (0.5 + 0.5 * (PI * bval.min(15.5) / 15.5).cos()))
}

/// The model of a pair's middle and side channels, `M = (L + R) / 2` and
/// `S = (L - R) / 2`, run beside its two channels' models, with its output
/// delayed by a frame as theirs is (see
/// [`BlockSwitching`](super::switching::BlockSwitching)).
#[derive(Debug)]
pub struct MidSideModel {
    long: &'static Tables,
    short: &'static Tables,
    /// `bmax(b)` of each partition of the long and the short FFT.
    long_bmax: Vec<f64>,
    short_bmax: Vec<f64>,
    /// `nb(b)` of the last long block and of the last short block of the
    /// middle and of the side channel, before pre-echo control (step 11).
    long_nb_last: [Vec<f64>; 2],
    short_nb_last: [Vec<f64>; 2],
    /// The middle and side channels of the frame whose windows are chosen
    /// next, and of the frame analyzed last.
    current: Box<Frame>,
    ahead: Box<Frame>,
    primed: bool,
    /// The threshold in quiet of each partition of the long and the short
    /// FFT, as the model reads it.
    long_quiet: Vec<f64>,
    short_quiet: Vec<f64>,
}

/// The middle and side channels of a frame, through step 10, with the
/// thresholds of the blocks before its own, and their analyses as far as
/// they are finished: only the threshold of the frame's transform is used,
/// that of its long block or of its short ones, and only that is finished
/// from step 11 (see [`MidSideModel::current`]).
#[derive(Debug)]
struct Frame {
    long_blocks: [Block; 2],
    short_blocks: [Box<[Block; SHORT_WINDOWS]>; 2],
    /// `nb(b)` of the long block and of the short block before the frame's,
    /// before pre-echo control.
    long_nb_before: [Vec<f64>; 2],
    short_nb_before: [Vec<f64>; 2],
    analyses: [Box<Analysis>; 2],
    /// The transform whose thresholds the analyses hold, if either's.
    finished: Option<Transform>,
}

impl MidSideModel {
    pub fn new(sampling_frequency: SamplingFrequency, quiet: impl Into<Quiet>) -> Self {
        let quiet = quiet.into();
        let long = Tables::get(sampling_frequency, Transform::Long, quiet.reading);
        let short = Tables::get(sampling_frequency, Transform::Short, quiet.reading);
        let bmaxes = |tables: &Tables| tables.partitions.iter().map(|p| bmax(p.bval)).collect();
        let frame = || {
            let analysis = || Box::new(Analysis::new(sampling_frequency));
            Box::new(Frame {
                long_blocks: std::array::from_fn(|_| Block::new(super::LONG_IBLEN)),
                short_blocks: std::array::from_fn(|_| {
                    Box::new(std::array::from_fn(|_| Block::new(super::SHORT_IBLEN)))
                }),
                long_nb_before: std::array::from_fn(|_| vec![0.0; long.partitions.len()]),
                short_nb_before: std::array::from_fn(|_| vec![0.0; short.partitions.len()]),
                analyses: [analysis(), analysis()],
                finished: None,
            })
        };
        Self {
            long,
            short,
            long_bmax: bmaxes(long),
            short_bmax: bmaxes(short),
            long_nb_last: std::array::from_fn(|_| vec![0.0; long.partitions.len()]),
            short_nb_last: std::array::from_fn(|_| vec![0.0; short.partitions.len()]),
            current: frame(),
            ahead: frame(),
            primed: false,
            long_quiet: quiet_energies(sampling_frequency, Transform::Long, quiet),
            short_quiet: quiet_energies(sampling_frequency, Transform::Short, quiet),
        }
    }

    /// The analyses of the middle and of the side channel of the frame
    /// whose windows are chosen next, for a frame coded with `transform`:
    /// their thresholds of its long block or of its short blocks, finished
    /// from step 11 as [`look_ahead`](Self::look_ahead) would have, and
    /// only those. The others are not to be used.
    pub fn current(&mut self, transform: Transform) -> [&Analysis; 2] {
        let frame = &mut *self.current;
        if frame.finished != Some(transform) {
            frame.finished = Some(transform);
            for m in 0..2 {
                let analysis = &mut frame.analyses[m];
                let mut nb_last = [0.0; MAX_PARTITIONS];
                match transform {
                    Transform::Long => {
                        let nb_last = &mut nb_last[..self.long.partitions.len()];
                        nb_last.copy_from_slice(&frame.long_nb_before[m]);
                        let block = &frame.long_blocks[m];
                        (self.long).finish(
                            &block.energy,
                            &block.threshold,
                            nb_last,
                            RPELEV_LONG,
                            &self.long_quiet,
                            &mut analysis.long,
                        );
                    }
                    Transform::Short => {
                        let nb_last = &mut nb_last[..self.short.partitions.len()];
                        nb_last.copy_from_slice(&frame.short_nb_before[m]);
                        for (block, out) in frame.short_blocks[m].iter().zip(&mut analysis.short) {
                            (self.short).finish(
                                &block.energy,
                                &block.threshold,
                                nb_last,
                                RPELEV_SHORT,
                                &self.short_quiet,
                                out,
                            );
                        }
                    }
                }
            }
        }
        let [mid, side] = &frame.analyses;
        [mid, side]
    }

    /// Analyzes the next frame of the pair whose channels' models `left`
    /// and `right` have taken it through steps 1 to 10 (see
    /// `Model::blocks`), before they finish it: computes the thresholds of
    /// its middle and side channels (step 1) and adjusts all four by the
    /// imaging control (step 2); the middle and side channels' analyses are
    /// finished with them from step 11 when asked for (see
    /// [`current`](Self::current)).
    ///
    /// Spec reading: step 1 computes the "raw thresholds" of the middle and
    /// side channels "rather than using the tonality for the M or S
    /// threshold, one uses the more tonal value from the L or R calculation
    /// in each threshold calculation band, and proceed with the
    /// psychoacoustic model for M and S from the M and S energies and the
    /// minimum of the L or R values for C(w) in each threshold calculation
    /// band". The more tonal of the two is that of the lesser normalized
    /// unpredictability `cb(b)` (step 7), so each partition of the middle
    /// and side channels takes the lesser of the left and right channels'
    /// `cb(b)`, which is the minimum of each partition's unpredictability
    /// and the more tonal value both. The minimum of `c(w)` of each line,
    /// weighted anew by the middle and side channels' energies, would read
    /// "C(w)" more literally, but not "in each threshold calculation band".
    /// A partition without energy in one channel, whose `cb(b)` is not
    /// defined (see `Tables::raw_threshold`), takes the other's: its
    /// placeholder of 0 would otherwise make it the most tonal. Their
    /// energies `r(w)^2` are those of
    /// the lines `(X_L(w) + X_R(w)) / 2` and `(X_L(w) - X_R(w)) / 2` of the
    /// left and right channels' FFTs, which the FFT, being linear, gives
    /// for the middle and side channels' samples.
    ///
    /// Spec reading: step 2 inserts the adjusted thresholds "as the values
    /// for cb(b) into step 11", where `cb(b)` is the normalized
    /// unpredictability of step 6 and step 11 takes the threshold `nb(b)`;
    /// the definitions after the steps say the final thresholds are
    /// "returned as nb(b) in step 11". They are taken as the `nb(b)` that
    /// step 11 starts from, and as each channel's threshold of the last
    /// block there (see `Tables::finish`).
    pub fn look_ahead(&mut self, left: &mut Model, right: &mut Model) {
        if self.primed {
            std::mem::swap(&mut self.current, &mut self.ahead);
        }
        self.primed = true;
        let frame = &mut *self.ahead;
        frame.finished = None;
        let [mid_long, side_long] = &mut frame.long_blocks;
        pair_block(
            self.long,
            &self.long_bmax,
            [&mut left.long_block, &mut right.long_block],
            [mid_long, side_long],
        );
        // Each block's nb(b) is the next block's nb_l(b) of step 11 (see
        // `Tables::finish`), whatever is finished of either.
        for (m, block) in frame.long_blocks.iter().enumerate() {
            let nb_last = &mut self.long_nb_last[m];
            frame.long_nb_before[m].copy_from_slice(nb_last);
            let n = nb_last.len();
            nb_last.copy_from_slice(&block.threshold.nb[..n]);
        }
        let [mid_short, side_short] = &mut frame.short_blocks;
        for k in 0..SHORT_WINDOWS {
            pair_block(
                self.short,
                &self.short_bmax,
                [&mut left.short_blocks[k], &mut right.short_blocks[k]],
                [&mut mid_short[k], &mut side_short[k]],
            );
        }
        for (m, blocks) in frame.short_blocks.iter().enumerate() {
            let nb_last = &mut self.short_nb_last[m];
            frame.short_nb_before[m].copy_from_slice(nb_last);
            let last = &blocks[SHORT_WINDOWS - 1].threshold.nb;
            let n = nb_last.len();
            nb_last.copy_from_slice(&last[..n]);
        }
        // The middle and side channels' windows are those of the pair,
        // whose attacks are found in its channels.
        for analysis in &mut frame.analyses {
            analysis.entropy_rise = 0.0;
            analysis.entropy_step = 0.0;
        }
    }
}

/// Steps 1 and 2 of clause C.6.1 for one block of the pair: the middle and
/// side channels' block `[mid, side]` from the left and right channels'
/// `[left, right]`, through step 10, then the imaging control of each
/// partition, which leaves the adjusted thresholds of all four in their
/// `nb(b)`.
fn pair_block(
    tables: &Tables,
    bmax: &[f64],
    [left, right]: [&mut Block; 2],
    [mid, side]: [&mut Block; 2],
) {
    for (((m, s), l), r) in (mid.x.iter_mut())
        .zip(side.x.iter_mut())
        .zip(&left.x)
        .zip(&right.x)
    {
        *m = Complex {
            re: (l.re + r.re) / 2.0,
            im: (l.im + r.im) / 2.0,
        };
        *s = Complex {
            re: (l.re - r.re) / 2.0,
            im: (l.im - r.im) / 2.0,
        };
    }
    for block in [&mut *mid, &mut *side] {
        for (energy, x) in block.energy.iter_mut().zip(&block.x) {
            *energy = x.re * x.re + x.im * x.im;
        }
    }
    tables.energies(
        [&mid.energy, &side.energy],
        [&mut mid.threshold, &mut side.threshold],
    );
    for block in [&mut *mid, &mut *side] {
        let threshold = &mut block.threshold;
        // The lesser cb(b) has the greater tonality (step 7), and so the
        // lesser power ratio bc(b) (steps 8 and 9): the same as steps 7 to
        // 10 give from it.
        for b in 0..tables.partitions.len() {
            let (left, right) = (&left.threshold, &right.threshold);
            let more_tonal = match (left.en[b] == 0.0, right.en[b] == 0.0) {
                (true, false) => right,
                (false, true) => left,
                _ if left.cb[b] <= right.cb[b] => left,
                _ => right,
            };
            threshold.cb[b] = more_tonal.cb[b];
            threshold.bc[b] = more_tonal.bc[b];
            threshold.nb[b] = threshold.en[b] * threshold.bc[b];
        }
    }
    imaging_control(
        bmax,
        [
            &mut left.threshold,
            &mut right.threshold,
            &mut mid.threshold,
            &mut side.threshold,
        ],
    );
}

/// The imaging control process of clause C.6.1 for each partition `b` of
/// a block, from the raw thresholds `Xthr`, `nb(b)` of step 10, and the
/// spread energies `Xengy`, `en(b)` of step 6, of the left, right, middle
/// and side channels, into their final thresholds `Xfthr`, left in their
/// `nb(b)`:
///
/// ```text
/// t = Mthr / Sthr; if (t > 1) t = 1 / t
/// Rfthr = max(Rthr * t, min(Rthr, bmax * Rengy))
/// Lfthr = max(Lthr * t, min(Lthr, bmax * Lengy))
/// t = min(Lthr, Rthr)
/// Mfthr = min(t, max(Mthr, min(Sengy * bmax, Sthr)))
/// Sfthr = min(t, max(Sthr, min(Mengy * bmax, Mthr)))
/// ```
///
/// Spec reading: the definitions call `Xengy` the "spread energy (en(b)
/// from step 6)", as step 1 does, while step 5 says that `e(b)` "is equal
/// to Xengy". The spread energy of clause C.6.1 is taken, which is on the
/// scale of the thresholds compared with it: `nb(b)` is `en(b)` times the
/// power ratio of step 9.
///
/// Spec reading: `Lfthr` is printed `max(Lthr*t, min (Lthr, bmax)
/// *Lengy)`, which multiplies a threshold by an energy and compares a
/// threshold with the ratio `bmax`, where `Rfthr`, beside it, is `max(Rthr
/// *t, min (Rthr, bmax*Rengy)`; it is taken as misprinted, and read as
/// `Rfthr` is. `Rfthr`, `Mfthr` and `Sfthr` are printed with a closing
/// parenthesis short, and are read as above. The definitions list `Mengy,
/// Sengy, Rengy, Sengy`, taken as `Lengy` for the second `Sengy`.
///
/// Spec gap: `t = Mthr / Sthr` is undefined where both are 0, as in a
/// silent partition. Their ratio is then taken as 1: neither channel's
/// threshold stands out from the other's. Where one alone is 0, as the
/// side channel's where both channels are the same, `t` is 0.
fn imaging_control(bmax: &[f64], [left, right, mid, side]: [&mut BlockThreshold; 4]) {
    for (b, &bmax) in bmax.iter().enumerate().take(MAX_PARTITIONS) {
        let (lthr, rthr, mthr, sthr) = (left.nb[b], right.nb[b], mid.nb[b], side.nb[b]);
        let (most, least) = (mthr.max(sthr), mthr.min(sthr));
        let t = if most == 0.0 { 1.0 } else { least / most };
        left.nb[b] = (lthr * t).max(lthr.min(bmax * left.en[b]));
        right.nb[b] = (rthr * t).max(rthr.min(bmax * right.en[b]));
        let t = lthr.min(rthr);
        mid.nb[b] = t.min(mthr.max((side.en[b] * bmax).min(sthr)));
        side.nb[b] = t.min(sthr.max((mid.en[b] * bmax).min(mthr)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::psychoacoustic::QuietThreshold;
    use crate::stream::FRAME_SAMPLES;

    #[test]
    fn bmax_rises_from_minus_30_db_to_0_db_at_15_5_bark() {
        assert!((bmax(0.0) - 1e-3).abs() < 1e-12);
        assert!((bmax(7.75) - 10f64.powf(-1.5)).abs() < 1e-12);
        assert!((bmax(15.5) - 1.0).abs() < 1e-12);
        assert!((bmax(24.0) - 1.0).abs() < 1e-12);
    }

    fn threshold(nb: f64, en: f64) -> BlockThreshold {
        BlockThreshold {
            nb: [nb; MAX_PARTITIONS],
            en: [en; MAX_PARTITIONS],
            ..BlockThreshold::ZERO
        }
    }

    /// The imaging control as printed, for one partition.
    #[test]
    fn imaging_control_follows_clause_c_6_1() {
        let (bmax_value, b) = (0.01, 0);
        let bmax = [bmax_value];
        let mut left = threshold(4.0, 400.0);
        let mut right = threshold(2.0, 300.0);
        let mut mid = threshold(1.0, 500.0);
        let mut side = threshold(8.0, 20.0);
        imaging_control(&bmax, [&mut left, &mut right, &mut mid, &mut side]);
        // t = 1 / 8.
        assert_eq!(left.nb[b], f64::max(4.0 / 8.0, f64::min(4.0, 0.01 * 400.0)));
        assert_eq!(
            right.nb[b],
            f64::max(2.0 / 8.0, f64::min(2.0, 0.01 * 300.0))
        );
        // t = min(Lthr, Rthr) = 2.
        assert_eq!(
            mid.nb[b],
            f64::min(2.0, f64::max(1.0, f64::min(20.0 * 0.01, 8.0)))
        );
        assert_eq!(
            side.nb[b],
            f64::min(2.0, f64::max(8.0, f64::min(500.0 * 0.01, 1.0)))
        );
    }

    /// Where both channels are the same, the side channel is silent, and
    /// the left and right thresholds fall to the BMLD protection.
    #[test]
    fn a_mono_pair_protects_left_and_right() {
        let mut left = threshold(4.0, 400.0);
        let mut right = threshold(4.0, 400.0);
        let mut mid = threshold(4.0, 400.0);
        let mut side = threshold(0.0, 0.0);
        imaging_control(&[0.001], [&mut left, &mut right, &mut mid, &mut side]);
        assert_eq!(left.nb[0], 0.4);
        assert_eq!(right.nb[0], 0.4);
        assert_eq!(mid.nb[0], 4.0);
        assert_eq!(side.nb[0], 0.4);
        // Silent channels keep their thresholds.
        let [mut left, mut right, mut mid, mut side] = [0; 4].map(|_| threshold(0.0, 0.0));
        imaging_control(&[0.001], [&mut left, &mut right, &mut mid, &mut side]);
        assert_eq!([left.nb[0], right.nb[0], mid.nb[0], side.nb[0]], [0.0; 4]);
    }

    /// The middle channel of a pair of one signal has the threshold that
    /// signal has alone, before the imaging control; the side channel of a
    /// pair of one signal and its negation likewise.
    #[test]
    fn middle_and_side_of_one_signal_are_that_signal() {
        let fs = SamplingFrequency::Hz44100;
        let samples: [f64; FRAME_SAMPLES] =
            std::array::from_fn(|t| 0.3 * (t as f64 * 0.07).sin() + 0.1 * (t as f64 * 1.3).sin());
        let negated = samples.map(|x| -x);
        let mut alone = Model::new(fs, QuietThreshold::Noise);
        alone.blocks(&samples);
        for (right_samples, which) in [(samples, 0), (negated, 1)] {
            let mut left = Model::new(fs, QuietThreshold::Noise);
            let mut right = Model::new(fs, QuietThreshold::Noise);
            left.blocks(&samples);
            right.blocks(&right_samples);
            let mut pair = MidSideModel::new(fs, QuietThreshold::Noise);
            let tables = pair.long;
            let mut blocks = [
                Block::new(super::super::LONG_IBLEN),
                Block::new(super::super::LONG_IBLEN),
            ];
            let [mid, side] = &mut blocks;
            pair_block(
                tables,
                &vec![1.0; tables.partitions.len()],
                [&mut left.long_block, &mut right.long_block],
                [mid, side],
            );
            let one = &blocks[which].threshold;
            for b in 0..tables.partitions.len() {
                let expected = alone.long_block.threshold.en[b];
                assert!((one.en[b] - expected).abs() <= 1e-9 * expected, "{b}");
            }
            pair.look_ahead(&mut left, &mut right);
        }
    }
}
