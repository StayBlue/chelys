// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The analysis half of the Layer III hybrid filterbank: the polyphase
//! subband filter (clause C.1.3, figure C.4) followed by the MDCT, with the
//! window and transform of the normal, start, short or stop block type, and
//! the encoder's aliasing butterflies (clause C.1.5.3.3), and of mixed
//! blocks, whose two lowest subbands have the normal window (clause 2.4.2.7).

mod analysis_window;

use std::f64::consts::PI;

use crate::mp3::{BlockType, GRANULE_LINES};
use analysis_window::ANALYSIS_WINDOW;

const SUBBANDS: usize = 32;
/// The rows of the polyphase filter's matrixing that remain once those
/// equal to another, or to its negative, are folded together (see
/// [`Filterbank::subband_samples`]).
const FOLDED_ROWS: usize = 33;
/// The buffer that holds the filter's 512 input samples: room for 16
/// passes of 32 samples before they must be moved.
const HISTORY: usize = 1024;
/// Subband samples per granule, and MDCT lines per subband.
const SUBBAND_SAMPLES: usize = GRANULE_LINES / SUBBANDS;
/// Samples in a long MDCT block: the subband samples of two granules.
const LONG_BLOCK: usize = 2 * SUBBAND_SAMPLES;
/// The subbands a mixed block transforms with the normal window (clause
/// 2.4.2.7).
const MIXED_LONG_SUBBANDS: usize = 2;
/// Samples in each of the three short MDCT blocks of a short block, and the
/// lines each gives.
const SHORT_BLOCK: usize = 12;
const SHORT_LINES: usize = SHORT_BLOCK / 2;
/// Where the short blocks start in the block of 36 subband samples (clause
/// C.1.5.3.3: `y(k)_i = x'_(i + 6 + 6k)`).
const SHORT_BLOCK_STARTS: [usize; 3] = [6, 12, 18];

/// The decoder's long-block IMDCT (clause 2.4.3.4.10.2) with windowing and
/// overlap-add returns the encoder's MDCT input (clause C.1.5.3.3) scaled by
/// `n / 4 = 9`, since neither transform is normalised, so the MDCT output is
/// divided by 9. Clause C.1.5.4.1 leaves this scaling to the implementation as
/// part of its "system constant".
///
/// Spec gap: the standard does not state the gain of the polyphase analysis
/// and synthesis filters. Taking both as unity, the result is on the scale of
/// the decoder's requantised values `xr` (clause 2.4.3.4.7.1).
///
/// Settled by decoding with ffmpeg (black box): the conformance test
/// `pcm_round_trips_at_every_sampling_frequency` decodes the input back at
/// over 50 dB SNR with this scaling. Doubling the constant brings that down
/// to 6 dB, as halving the output should.
const LONG_BLOCK_GAIN: f64 = LONG_BLOCK as f64 / 4.0;

/// The same scaling for short blocks: the decoder's short IMDCT (clause
/// 2.4.3.4.10.2) returns each windowed short block scaled by `n / 4 = 3`.
/// Start and stop blocks are long blocks, scaled by 9, so each side of
/// their overlap with a short block comes back at unity. The conformance
/// test `short_blocks_round_trip_at_every_sampling_frequency` decodes short
/// blocks, and every change of block type, at over 45 dB SNR with this
/// scaling.
const SHORT_BLOCK_GAIN: f64 = SHORT_BLOCK as f64 / 4.0;

/// Table B.9: the coefficients `c_i` of the aliasing butterflies.
const ALIAS_COEFFICIENTS: [f64; 8] = [
    -0.6, -0.535, -0.33, -0.185, -0.095, -0.041, -0.0142, -0.0037,
];

/// Filters one channel of PCM into granules of 576 frequency lines.
pub struct Filterbank {
    /// The input sample vector `X` of figure C.4, most recent sample first,
    /// at `history[start..][..512]`: each pass moves `start` down rather
    /// than shifting `X`, which is moved back to the end of the buffer only
    /// once `start` reaches its beginning.
    history: Box<[f64; HISTORY]>,
    start: usize,
    /// The rows `k` of `M_ik` of clause C.1.3 that
    /// [`subband_samples`](Self::subband_samples) multiplies, 0 to 16 and 33
    /// to 48, indexed `[row][i]`, where `row` is `k` up to 16 and `k - 16`
    /// above. This and the kernels below are indexed by input sample first,
    /// as [`transform`] takes them.
    matrix: Box<[[f64; SUBBANDS]; FOLDED_ROWS]>,
    /// The MDCT kernel of clause C.1.5.3.3 for `n = 36`, indexed `[k][i]`,
    /// with the window of the normal, start and stop block types folded in,
    /// in that order.
    long_mdct: Box<[[[f64; SUBBAND_SAMPLES]; LONG_BLOCK]; 3]>,
    /// The MDCT kernel for `n = 12`, indexed `[k][i]`, with the short window
    /// folded in.
    short_mdct: [[f64; SHORT_LINES]; SHORT_BLOCK],
    /// The MDCT kernel for `n = 36` with no window, indexed `[k][i]`, for
    /// the windows of [`low_window`](Self::low_window).
    plain_mdct: Box<[[f64; SUBBAND_SAMPLES]; LONG_BLOCK]>,
    /// The block type of the previous granule.
    last: BlockType,
    /// The previous granule's subband samples, indexed `[subband][time]`.
    previous: [[f64; SUBBAND_SAMPLES]; SUBBANDS],
    /// `cs_i` and `ca_i` of Table B.9.
    butterflies: [(f64, f64); 8],
}

impl Default for Filterbank {
    fn default() -> Self {
        Self::new()
    }
}

impl Filterbank {
    pub fn new() -> Self {
        let mut matrix = Box::new([[0.0; SUBBANDS]; FOLDED_ROWS]);
        for (row, column) in matrix.iter_mut().enumerate() {
            let k = if row <= 16 { row } else { row + 16 };
            for (i, m) in column.iter_mut().enumerate() {
                *m = ((2 * i + 1) as f64 * (k as f64 - 16.0) * PI / 64.0).cos();
            }
        }
        let long_mdct = Box::new(
            [BlockType::Normal, BlockType::Start, BlockType::Stop]
                .map(|block_type| mdct_kernel(|k| long_window(block_type, k))),
        );
        let short_mdct = mdct_kernel(short_window);
        let plain_mdct = Box::new(mdct_kernel(|_| 1.0));
        let butterflies = ALIAS_COEFFICIENTS.map(|c| {
            let norm = (1.0 + c * c).sqrt();
            (1.0 / norm, c / norm)
        });
        Self {
            history: Box::new([0.0; HISTORY]),
            start: HISTORY - 512,
            matrix,
            long_mdct,
            short_mdct,
            plain_mdct,
            last: BlockType::Normal,
            previous: [[0.0; SUBBAND_SAMPLES]; SUBBANDS],
            butterflies,
        }
    }

    /// Filters the next 576 PCM samples, nominally in -1.0 to 1.0, into the
    /// granule's frequency lines `xr`, transformed as `block_type` asks,
    /// given the block type of the granule after it, `next`. The lines
    /// describe the block formed with the previous granule, so they lag the
    /// input by one granule.
    ///
    /// Line `18 * subband + k` holds MDCT output `k` of `subband`. For short
    /// blocks, line `18 * subband + 6 * window + k` holds output `k` of short
    /// window `window`: the subband order the decoder reorders short blocks
    /// into before its IMDCT (clause 2.4.3.4.8), not the order of the
    /// bitstream. Mixed blocks hold their two lowest subbands as long blocks
    /// do and the rest as short blocks do.
    pub fn granule(
        &mut self,
        pcm: &[f64; GRANULE_LINES],
        block_type: BlockType,
        next: BlockType,
    ) -> [f64; GRANULE_LINES] {
        let mut current = [[0.0; SUBBAND_SAMPLES]; SUBBANDS];
        for (time, samples) in pcm.chunks_exact(SUBBANDS).enumerate() {
            let subbands = self.subband_samples(samples.try_into().unwrap());
            for (subband, &sample) in subbands.iter().enumerate() {
                // Spec gap: the decoder negates every odd time sample of every
                // odd subband to compensate for the frequency inversion of the
                // polyphase filterbank (clause 2.4.3.4.10.5). Annex C does not
                // mention it for the encoder, which is taken to apply the same
                // inversion so that the decoder restores the input.
                //
                // Settled by decoding with ffmpeg (black box): without the
                // inversion, the conformance test
                // `pcm_round_trips_at_every_sampling_frequency` decodes at
                // about 1 dB SNR instead of over 50 dB.
                let invert = subband % 2 == 1 && time % 2 == 1;
                current[subband][time] = if invert { -sample } else { sample };
            }
        }

        let low_window = self.low_window(block_type, next);
        let mut xr = [0.0; GRANULE_LINES];
        for (subband, lines) in xr.chunks_exact_mut(SUBBAND_SAMPLES).enumerate() {
            let mut block = [0.0; LONG_BLOCK];
            block[..SUBBAND_SAMPLES].copy_from_slice(&self.previous[subband]);
            block[SUBBAND_SAMPLES..].copy_from_slice(&current[subband]);
            let mdct = match (&low_window, block_type) {
                (Some(window), _) if subband < MIXED_LONG_SUBBANDS => {
                    self.windowed_mdct(&block, window)
                }
                (_, BlockType::Mixed) => self.mdct(&block, BlockType::Short),
                _ => self.mdct(&block, block_type),
            };
            lines.copy_from_slice(&mdct);
        }
        self.previous = current;
        self.last = block_type;
        // The encoder's butterflies are those of the decoder (clause
        // C.1.5.3.3), which leaves them out for short blocks (clause
        // 2.4.3.4.10.1).
        //
        // Spec reading: clause 2.4.3.4.10.1 leaves the butterflies out for
        // every granule of block type 2, which mixed blocks are too. A mixed
        // block's two lowest subbands are long blocks, though, whose aliasing
        // the butterfly between them reduces as in any long block, and it is
        // applied there alone. Settled by decoding with ffmpeg, mpg123 and
        // LAME (black box): all three decode mixed blocks at over 45 dB SNR
        // with it, and at 12 to 17 dB without it. The conformance test is
        // `mixed_blocks_round_trip`.
        match block_type {
            BlockType::Short => {}
            BlockType::Mixed => self.reduce_aliasing_below(&mut xr, MIXED_LONG_SUBBANDS),
            _ => self.reduce_aliasing(&mut xr),
        }
        xr
    }

    /// The analysis window of the two lowest subbands of a granule of
    /// `block_type`, between the previous granule and `next`, if a mixed
    /// block is among the three; otherwise those subbands have the window of
    /// their block type, as all others do.
    ///
    /// The decoder windows the two lowest subbands of a mixed block with the
    /// normal window (clause 2.4.2.7), and those of the start block before it
    /// and the stop block after it with theirs. Those windows do not overlap
    /// as the windows of figure C.7's sequences do: the start window's second
    /// half and the normal window's first half, squared, do not add up to 1,
    /// nor does the time aliasing of one cancel that of the other.
    ///
    /// Spec gap: Annex C gives the windows of the four block types but not
    /// those of a mixed block. The windows here are chosen so that the
    /// decoder's output is the input wherever two blocks overlap: given the
    /// decoder's windows `p` of the second half of the earlier block and `q`
    /// of the first half of the later one, `n` running over the 18 samples
    /// of their overlap and `m = 17 - n`, the earlier block's analysis
    /// window is `q(m) / d(n)` and the later block's `p(m) / d(n)`, where
    /// `d(n) = p(n) q(m) + q(n) p(m)`. Where the windows do overlap as figure
    /// C.7 has them, `d(n)` is 1 and these are the decoder's windows; at a
    /// mixed block's transitions `d(n)` lies between 0.88 and 1. Next to a
    /// short block, whose windows these are not, a half keeps the window of
    /// its own block type.
    ///
    /// Settled by decoding with ffmpeg, mpg123 and LAME (black box): all
    /// three window a mixed block's two lowest subbands with the normal
    /// window, and decode every change of block type to and from mixed
    /// blocks at over 45 dB SNR with these windows, and at 16 dB with the
    /// decoder's own. The conformance test is `mixed_blocks_round_trip`.
    fn low_window(&self, block_type: BlockType, next: BlockType) -> Option<[f64; LONG_BLOCK]> {
        let mixed = [self.last, block_type, next].contains(&BlockType::Mixed);
        if !mixed || block_type == BlockType::Short {
            return None;
        }
        let own = low_synthesis(block_type).expect("a long window");
        let (first, second) = own.split_at(SUBBAND_SAMPLES);
        let mut window = own;
        window[..SUBBAND_SAMPLES].copy_from_slice(&match low_synthesis(self.last) {
            Some(last) => matched_halves(&last[SUBBAND_SAMPLES..], first).1,
            None => first.try_into().unwrap(),
        });
        window[SUBBAND_SAMPLES..].copy_from_slice(&match low_synthesis(next) {
            Some(next) => matched_halves(second, &next[..SUBBAND_SAMPLES]).0,
            None => second.try_into().unwrap(),
        });
        Some(window)
    }

    /// The long MDCT of one subband's block of 36 samples with `window`.
    fn windowed_mdct(
        &self,
        block: &[f64; LONG_BLOCK],
        window: &[f64; LONG_BLOCK],
    ) -> [f64; SUBBAND_SAMPLES] {
        transform(&self.plain_mdct, |m, k| m * window[k] * block[k])
            .map(|sum| sum / LONG_BLOCK_GAIN)
    }

    /// The windowed MDCT of one subband's block of 36 samples, with the
    /// lines of the three short windows one after the other for a short
    /// block.
    ///
    /// # Panics
    ///
    /// For a mixed block, whose subbands are transformed as long or short
    /// ones.
    fn mdct(&self, block: &[f64; LONG_BLOCK], block_type: BlockType) -> [f64; SUBBAND_SAMPLES] {
        let long = match block_type {
            BlockType::Normal => &self.long_mdct[0],
            BlockType::Start => &self.long_mdct[1],
            BlockType::Stop => &self.long_mdct[2],
            BlockType::Mixed => unreachable!("mixed blocks have long and short subbands"),
            BlockType::Short => {
                let mut lines = [0.0; SUBBAND_SAMPLES];
                for (window, start) in SHORT_BLOCK_STARTS.into_iter().enumerate() {
                    let samples = &block[start..start + SHORT_BLOCK];
                    let window_lines = transform(&self.short_mdct, |m, k| m * samples[k]);
                    for (line, sum) in lines[SHORT_LINES * window..].iter_mut().zip(window_lines) {
                        *line = sum / SHORT_BLOCK_GAIN;
                    }
                }
                return lines;
            }
        };
        transform(long, |m, k| m * block[k]).map(|sum| sum / LONG_BLOCK_GAIN)
    }

    /// One pass of figure C.4: shifts in 32 samples, oldest first, and returns
    /// the 32 subband samples.
    fn subband_samples(&mut self, samples: &[f64; SUBBANDS]) -> [f64; SUBBANDS] {
        if self.start < SUBBANDS {
            self.history.copy_within(
                self.start..self.start + 512 - SUBBANDS,
                HISTORY - 512 + SUBBANDS,
            );
            self.start = HISTORY - 512 + SUBBANDS;
        }
        self.start -= SUBBANDS;
        let history: &mut [f64; 512] = (&mut self.history[self.start..][..512]).try_into().unwrap();
        for (i, &sample) in samples.iter().enumerate() {
            // "for i=31 down to 0 do X[i]=next_input_audio_sample"
            history[SUBBANDS - 1 - i] = sample;
        }
        let mut y = [0.0; 64];
        for (i, y) in y.iter_mut().enumerate() {
            *y = (0..8)
                .map(|j| ANALYSIS_WINDOW[i + 64 * j] * history[i + 64 * j])
                .sum();
        }
        // Spec reading: figure C.4 sums `M_ik * Y_k` over all 64 `k`. As
        // `M_ik = cos((2i + 1)(k - 16) pi / 64)` (clause C.1.3), row `32 - k`
        // of the matrix equals row `k`, and row `96 - k` is its negative. So
        // `Y_k` and `Y_(32 - k)` are added, for `k` from 0 to 15, and
        // `Y_(96 - k)` subtracted from `Y_k`, for `k` from 33 to 47, before
        // multiplying by the row, and rows 16 and 48 multiply `Y_k` alone,
        // which halves the multiplications, for speed. The sums are the
        // same but for rounding.
        let folded: [f64; FOLDED_ROWS] = std::array::from_fn(|row| match row {
            0..16 => y[row] + y[32 - row],
            16 => y[16],
            17..32 => y[row + 16] - y[80 - row],
            _ => y[48],
        });
        transform(&self.matrix, |m, k| m * folded[k])
    }

    /// The encoder's aliasing butterflies (figure C.8) between each pair of
    /// adjacent subbands, applied to the eight lines on either side of the
    /// boundary as in figure A.5.
    fn reduce_aliasing(&self, xr: &mut [f64; GRANULE_LINES]) {
        self.reduce_aliasing_below(xr, SUBBANDS);
    }

    /// The butterflies between each pair of adjacent subbands of the lowest
    /// `subbands`.
    fn reduce_aliasing_below(&self, xr: &mut [f64; GRANULE_LINES], subbands: usize) {
        for subband in 1..subbands {
            let boundary = SUBBAND_SAMPLES * subband;
            for (i, &(cs, ca)) in self.butterflies.iter().enumerate() {
                let (lower, upper) = (boundary - 1 - i, boundary + i);
                let (a, b) = (xr[lower], xr[upper]);
                xr[lower] = a * cs + b * ca;
                xr[upper] = b * cs - a * ca;
            }
        }
    }
}

/// The window of a long block of `block_type` at sample `k` of 36 (clause
/// C.1.5.3.3).
fn long_window(block_type: BlockType, k: usize) -> f64 {
    let normal = (PI / 36.0 * (k as f64 + 0.5)).sin();
    match (block_type, k) {
        (BlockType::Normal, _) => normal,
        (BlockType::Start, 0..18) | (BlockType::Stop, 18..) => normal,
        (BlockType::Start, 18..24) | (BlockType::Stop, 12..18) => 1.0,
        (BlockType::Start, 24..30) => short_window(k - 18),
        (BlockType::Stop, 6..12) => short_window(k - 6),
        (BlockType::Start | BlockType::Stop, _) => 0.0,
        (BlockType::Short | BlockType::Mixed, _) => {
            unreachable!("short blocks have three short windows")
        }
    }
}

/// The decoder's window of the two lowest subbands of a granule of
/// `block_type`: the normal window for a mixed block (clause 2.4.2.7), the
/// window of its block type for the other long blocks, and `None` for a
/// short block, which has three short windows.
fn low_synthesis(block_type: BlockType) -> Option<[f64; LONG_BLOCK]> {
    let long = match block_type {
        BlockType::Short => return None,
        BlockType::Mixed => BlockType::Normal,
        long => long,
    };
    Some(std::array::from_fn(|k| long_window(long, k)))
}

/// The analysis windows of two blocks where they overlap (see
/// [`Filterbank::low_window`]), given the decoder's window `p` of the
/// second half of the earlier block and `q` of the first half of the later
/// one: the second half of the earlier block's analysis window, and the
/// first half of the later one's.
fn matched_halves(p: &[f64], q: &[f64]) -> ([f64; SUBBAND_SAMPLES], [f64; SUBBAND_SAMPLES]) {
    let m = |n: usize| SUBBAND_SAMPLES - 1 - n;
    let d = |n: usize| p[n] * q[m(n)] + q[n] * p[m(n)];
    (
        std::array::from_fn(|n| q[m(n)] / d(n)),
        std::array::from_fn(|n| p[m(n)] / d(n)),
    )
}

/// The window of a short block at sample `k` of 12 (clause C.1.5.3.3).
fn short_window(k: usize) -> f64 {
    (PI / 12.0 * (k as f64 + 0.5)).sin()
}

/// For each output `i` of a matrix indexed `[k][i]`, the sum over `k` of
/// `term(m, k)`, where `m` is its element `[k][i]`. Each sum is taken in
/// order from -0.0, as [`Iterator::sum`] does, so the sums are the same to
/// the bit; but all of them at once, one input `k` at a time, so that their
/// additions, which do not depend on each other, overlap, and the compiler
/// can vectorise them.
fn transform<const K: usize, const I: usize>(
    matrix: &[[f64; I]; K],
    term: impl Fn(f64, usize) -> f64,
) -> [f64; I] {
    let mut sums = [-0.0; I];
    for (k, column) in matrix.iter().enumerate() {
        for (sum, &m) in sums.iter_mut().zip(column) {
            *sum += term(m, k);
        }
    }
    sums
}

/// The MDCT of clause C.1.5.3.3 for `n` windowed samples, as a matrix
/// indexed `[k][i]` with `window(k)` folded in.
fn mdct_kernel<const N: usize, const LINES: usize>(
    window: impl Fn(usize) -> f64,
) -> [[f64; LINES]; N] {
    let n = N as f64;
    std::array::from_fn(|k| {
        std::array::from_fn(|i| {
            let kernel =
                (PI / (2.0 * n) * (2.0 * k as f64 + 1.0 + n / 2.0) * (2.0 * i as f64 + 1.0)).cos();
            window(k) * kernel
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The printed coefficients are all multiples of 2^-21 rounded to nine
    /// decimals. A misread digit almost always breaks this.
    #[test]
    fn analysis_window_lies_on_a_2_pow_21_grid() {
        for (i, c) in ANALYSIS_WINDOW.iter().enumerate() {
            let scaled = c * f64::from(1 << 21);
            assert!((scaled - scaled.round()).abs() < 0.0011, "C[{i}] = {c}");
        }
    }

    /// The two printed halves mirror each other: `C[512 - i] = -C[i]`, except
    /// at multiples of 64, where the two are equal. Checking one half against
    /// the other catches misread digits and signs.
    #[test]
    fn analysis_window_halves_mirror_each_other() {
        for i in 1..256 {
            let (a, b) = (ANALYSIS_WINDOW[i], ANALYSIS_WINDOW[512 - i]);
            let expected = if i % 64 == 0 { a } else { -a };
            assert_eq!(b, expected, "C[{i}] and C[{}]", 512 - i);
        }
    }

    /// The decoder's IMDCT, windowing and overlap-add (clauses 2.4.3.4.10.2
    /// to 2.4.3.4.10.4) for one subband, written out from the formulas to
    /// check the encoder's MDCT against.
    fn decoder_imdct(
        lines: &[f64; SUBBAND_SAMPLES],
        block_type: BlockType,
        overlap: &mut [f64; SUBBAND_SAMPLES],
    ) -> [f64; SUBBAND_SAMPLES] {
        let imdct = |x: &[f64], n: usize, i: usize| -> f64 {
            let n = n as f64;
            x.iter()
                .enumerate()
                .map(|(k, &x)| {
                    x * (PI / (2.0 * n) * (2.0 * i as f64 + 1.0 + n / 2.0) * (2.0 * k as f64 + 1.0))
                        .cos()
                })
                .sum()
        };
        let sine = |n: f64, i: f64| (PI / n * (i + 0.5)).sin();
        let mut z = [0.0; LONG_BLOCK];
        if block_type == BlockType::Short {
            // Each short block j is windowed and placed at 6 + 6 j.
            //
            // Spec reading: clause 2.4.3.4.10.3 windows the short blocks
            // y(j) for j = 0 to 2, but its concatenation names them y(1) to
            // y(3). They are read as the same three blocks in order: the
            // offsets of the concatenation, i - 6, i - 12 and i - 18, are
            // those of the encoder's short blocks 0 to 2 (clause C.1.5.3.3).
            for j in 0..3 {
                for i in 0..12 {
                    let y = imdct(&lines[6 * j..6 * j + 6], 12, i) * sine(12.0, i as f64);
                    z[6 + 6 * j + i] += y;
                }
            }
        } else {
            for (i, z) in z.iter_mut().enumerate() {
                let x = imdct(lines, 36, i);
                let fi = i as f64;
                *z = x * match (block_type, i) {
                    (BlockType::Start, 18..=23) | (BlockType::Stop, 12..=17) => 1.0,
                    (BlockType::Start, 24..=29) => sine(12.0, fi - 18.0),
                    (BlockType::Start, 30..=35) | (BlockType::Stop, 0..=5) => 0.0,
                    (BlockType::Stop, 6..=11) => sine(12.0, fi - 6.0),
                    _ => sine(36.0, fi),
                };
            }
        }
        let mut out = [0.0; SUBBAND_SAMPLES];
        for i in 0..SUBBAND_SAMPLES {
            out[i] = z[i] + overlap[i];
            overlap[i] = z[i + SUBBAND_SAMPLES];
        }
        out
    }

    /// The encoder's MDCT for every block type inverts through the decoder's
    /// IMDCT in every sequence of block types that figure C.7 allows.
    #[test]
    fn mdct_inverts_through_the_decoder_imdct() {
        use BlockType::*;
        let types = [
            Normal, Normal, Start, Short, Short, Stop, Normal, Start, Short, Stop, Start, Short,
            Stop, Normal,
        ];
        let filterbank = Filterbank::new();
        let signal: Vec<f64> = (0..SUBBAND_SAMPLES * types.len())
            .map(|t| ((t * 7919) % 101) as f64 / 50.0 - 1.0)
            .collect();
        let mut overlap = [0.0; SUBBAND_SAMPLES];
        let mut previous = [0.0; SUBBAND_SAMPLES];
        for (g, (current, &block_type)) in
            signal.chunks_exact(SUBBAND_SAMPLES).zip(&types).enumerate()
        {
            let mut block = [0.0; LONG_BLOCK];
            block[..SUBBAND_SAMPLES].copy_from_slice(&previous);
            block[SUBBAND_SAMPLES..].copy_from_slice(current);
            let lines = filterbank.mdct(&block, block_type);
            let out = decoder_imdct(&lines, block_type, &mut overlap);
            // Output g restores input g - 1, the first half of block g.
            if g > 0 {
                let expected = &signal[SUBBAND_SAMPLES * (g - 1)..][..SUBBAND_SAMPLES];
                for (o, e) in out.iter().zip(expected) {
                    assert!(
                        (o - e).abs() < 1e-12,
                        "granule {g} ({block_type:?}): {o} != {e}"
                    );
                }
            }
            previous.copy_from_slice(current);
        }
    }

    /// In the two lowest subbands, the decoder windows a mixed block with the
    /// normal window and the start and stop blocks around it with theirs
    /// (clause 2.4.2.7). With the analysis windows of
    /// [`Filterbank::low_window`] its output is still the input, in every
    /// sequence of block types that figure C.7 allows with mixed blocks in
    /// place of short ones, and at the start of a stream that begins with a
    /// mixed block. With the decoder's windows instead, it is not.
    #[test]
    fn mixed_blocks_invert_through_the_decoder_imdct() {
        use BlockType::*;
        let sequences: [&[BlockType]; 2] = [
            &[
                Normal, Normal, Start, Mixed, Mixed, Stop, Normal, Start, Mixed, Stop, Start,
                Mixed, Stop, Normal,
            ],
            &[Mixed, Mixed, Stop, Start, Mixed, Mixed],
        ];
        for types in sequences {
            let signal: Vec<f64> = (0..SUBBAND_SAMPLES * (types.len() + 1))
                .map(|t| ((t * 7919) % 101) as f64 / 50.0 - 1.0)
                .collect();
            let mut worst_literal: f64 = 0.0;
            for literal in [false, true] {
                let mut filterbank = Filterbank::new();
                let mut overlap = [0.0; SUBBAND_SAMPLES];
                let mut previous = [0.0; SUBBAND_SAMPLES];
                for (g, current) in signal.chunks_exact(SUBBAND_SAMPLES).enumerate() {
                    let Some(&block_type) = types.get(g) else {
                        break;
                    };
                    let next = types.get(g + 1).copied().unwrap_or(Normal);
                    let mut block = [0.0; LONG_BLOCK];
                    block[..SUBBAND_SAMPLES].copy_from_slice(&previous);
                    block[SUBBAND_SAMPLES..].copy_from_slice(current);
                    let window = match (literal, filterbank.low_window(block_type, next)) {
                        (false, Some(window)) => window,
                        _ => low_synthesis(block_type).unwrap(),
                    };
                    let lines = filterbank.windowed_mdct(&block, &window);
                    let decoded_as = if block_type == Mixed {
                        Normal
                    } else {
                        block_type
                    };
                    let out = decoder_imdct(&lines, decoded_as, &mut overlap);
                    if g > 0 {
                        let expected = &signal[SUBBAND_SAMPLES * (g - 1)..][..SUBBAND_SAMPLES];
                        for (o, e) in out.iter().zip(expected) {
                            if literal {
                                worst_literal = worst_literal.max((o - e).abs());
                            } else {
                                assert!(
                                    (o - e).abs() < 1e-12,
                                    "{types:?}, granule {g} ({block_type:?}): {o} != {e}"
                                );
                            }
                        }
                    }
                    filterbank.last = block_type;
                    previous.copy_from_slice(current);
                }
            }
            assert!(worst_literal > 0.1, "{types:?}: {worst_literal}");
        }
    }

    /// The three short windows follow each other in time, and their lines
    /// lie one window after the other: a click at subband sample 27 of the
    /// block lies only in the last short window (samples 18 to 29).
    #[test]
    fn short_windows_lie_in_time_order() {
        let filterbank = Filterbank::new();
        let mut block = [0.0; LONG_BLOCK];
        block[27] = 1.0;
        let lines = filterbank.mdct(&block, BlockType::Short);
        assert!(
            lines[..2 * SHORT_LINES].iter().all(|&x| x == 0.0),
            "{lines:?}"
        );
        assert!(
            lines[2 * SHORT_LINES..].iter().any(|&x| x != 0.0),
            "{lines:?}"
        );
    }

    /// A granule's lines are the MDCT of each subband's block, with the
    /// aliasing butterflies applied to every block type but short.
    #[test]
    fn short_blocks_skip_the_butterflies() {
        let pcm = |g: usize| -> [f64; GRANULE_LINES] {
            std::array::from_fn(|t| (((g * GRANULE_LINES + t) * 7919) % 101) as f64 / 101.0 - 0.5)
        };
        for block_type in [
            BlockType::Normal,
            BlockType::Start,
            BlockType::Short,
            BlockType::Stop,
        ] {
            let mut filterbank = Filterbank::new();
            filterbank.granule(&pcm(0), BlockType::Normal, BlockType::Normal);
            let previous = filterbank.previous;
            let xr = filterbank.granule(&pcm(1), block_type, BlockType::Normal);
            let mut expected = [0.0; GRANULE_LINES];
            for (subband, lines) in expected.chunks_exact_mut(SUBBAND_SAMPLES).enumerate() {
                let mut block = [0.0; LONG_BLOCK];
                block[..SUBBAND_SAMPLES].copy_from_slice(&previous[subband]);
                block[SUBBAND_SAMPLES..].copy_from_slice(&filterbank.previous[subband]);
                lines.copy_from_slice(&filterbank.mdct(&block, block_type));
            }
            if block_type != BlockType::Short {
                filterbank.reduce_aliasing(&mut expected);
            }
            assert_eq!(xr, expected, "{block_type:?}");
        }
    }

    /// The decoder's butterflies (clause 2.4.3.4.10.1) undo the encoder's.
    #[test]
    fn decoder_alias_reduction_undoes_the_encoder_butterflies() {
        let filterbank = Filterbank::new();
        let original: [f64; GRANULE_LINES] = std::array::from_fn(|i| ((i * 31) % 17) as f64 - 8.0);
        let mut xr = original;
        filterbank.reduce_aliasing(&mut xr);
        assert_ne!(xr, original);
        for sb in 1..SUBBANDS {
            for (i, &(cs, ca)) in filterbank.butterflies.iter().enumerate() {
                let (lower, upper) = (18 * sb - 1 - i, 18 * sb + i);
                let (a, b) = (xr[lower], xr[upper]);
                xr[lower] = a * cs - b * ca;
                xr[upper] = b * cs + a * ca;
            }
        }
        for (a, b) in xr.iter().zip(&original) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    /// A sinusoid at the centre of subband `sb`, `(2 sb + 1) * fs / 128`,
    /// lands in that subband's lines.
    #[test]
    fn a_tone_lands_in_its_subband() {
        for subband in [0, 5, 6, 31] {
            let mut filterbank = Filterbank::new();
            let frequency = (2 * subband + 1) as f64 / 128.0;
            let mut energy = [0.0; SUBBANDS];
            for g in 0..6 {
                let pcm = std::array::from_fn(|t| {
                    (2.0 * PI * frequency * (GRANULE_LINES * g + t) as f64).sin() * 0.5
                });
                let xr = filterbank.granule(&pcm, BlockType::Normal, BlockType::Normal);
                if g >= 3 {
                    for (sb, lines) in xr.chunks_exact(SUBBAND_SAMPLES).enumerate() {
                        energy[sb] += lines.iter().map(|x| x * x).sum::<f64>();
                    }
                }
            }
            let total: f64 = energy.iter().sum();
            assert!(
                energy[subband] > 0.99 * total,
                "subband {subband}: {energy:?}"
            );
        }
    }
}
