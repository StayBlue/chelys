// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Huffman coding of one granule's quantised spectrum, `Huffmancodebits()`
//! (clauses 2.4.1.7 and 2.4.2.7), partitioned and with tables chosen as in the
//! inner iteration loop (clauses C.1.5.4.4.3 to C.1.5.4.4.8).
//!
//! The values are given in the order they are sent, and the scalefactor
//! bands as [`coding_bands`](crate::mp3::scalefactor_bands::coding_bands)
//! gives them for the granule's block type.

use super::{PairTable, QuadTable, pair_table};
use crate::bitstream::BitWriter;
use crate::mp3::header::SamplingFrequency;
use crate::mp3::scalefactor_bands::{Band, LONG_BANDS, MIXED_LONG_LINES};
use crate::mp3::{BlockType, GRANULE_LINES};

/// The largest magnitude allowed in the `big_values` partition (clause
/// 2.4.2.7).
pub const MAX_MAGNITUDE: u32 = 8191;

/// How a granule's 576 quantised values are split into partitions and which
/// tables code them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpectrumCoding {
    /// Pairs in the `big_values` partition, starting at line 0.
    pub big_values: usize,
    /// Quadruples in the `count1` partition that follows. Not transmitted: the
    /// decoder infers it from `part2_3_length`.
    pub count1: usize,
    /// For granules with `window_switching_flag` set only the first two are
    /// sent, as region 2 is empty.
    pub table_select: [u8; 3],
    /// Sent only without `window_switching_flag`; otherwise they hold the
    /// defaults the decoder sets (clause 2.4.2.7).
    pub region0_count: u8,
    pub region1_count: u8,
    /// Where region 1 begins in a mixed block, which `region0_count` does
    /// not say at every sampling frequency; `None` for other block types.
    pub mixed_region1: Option<usize>,
    pub count1_table: QuadTable,
}

impl SpectrumCoding {
    /// Partitions `values` of a granule of `block_type` at
    /// `sampling_frequency` and chooses tables following clause C.1.5.4.4.
    ///
    /// Spec reading: a mixed block's `region0_count` of 7 puts 8 bands in
    /// region 0 (clause 2.4.2.7). In ISO/IEC 11172-3 a mixed block has
    /// "8+9*3=35" scalefactor bands, so these are its 8 long bands, which
    /// end at line 36, where its short bands begin. ISO/IEC 13818-3 does not
    /// say which they are at its sampling frequencies, where a mixed block
    /// sends 6 long bands and 8 long bands of the table would end at line
    /// 54. Region 1 is taken to begin at line 36 there too, below the short
    /// bands, as for the short blocks of every sampling frequency.
    ///
    /// Settled by decoding with ffmpeg, mpg123 and LAME (black box): at 16,
    /// 22.05 and 24 kHz all three begin region 1 of a mixed block at line
    /// 36, and decode mixed blocks at over 45 dB SNR only with it there; at
    /// line 54 at 2 to 8 dB. The conformance test is
    /// `mixed_blocks_round_trip`.
    ///
    /// Spec gap: US 6,185,539 B1 does not say where region 1 of a mixed
    /// block begins at MPEG-2.5's 11.025 and 12 kHz either. There a mixed
    /// block codes both regions with the same table, so that it decodes
    /// alike wherever region 1 begins.
    ///
    /// Settled by decoding with ffmpeg, mpg123 and LAME (black box): at
    /// 11.025 and 12 kHz, ffmpeg begins region 1 of a mixed block at line
    /// 36, and mpg123 and LAME at line 54, the end of 8 long bands. The
    /// conformance test is `mixed_blocks_round_trip` again.
    ///
    /// # Panics
    ///
    /// If a magnitude exceeds [`MAX_MAGNITUDE`].
    pub fn choose(
        values: &[i32; GRANULE_LINES],
        block_type: BlockType,
        bands: &[Band],
        sampling_frequency: SamplingFrequency,
    ) -> Self {
        Self::choose_counted(values, block_type, bands, sampling_frequency).0
    }

    /// [`choose`](Self::choose), with the [`bit_count`](Self::bit_count) of
    /// the coding chosen, which the choice works out anyway.
    fn choose_counted(
        values: &[i32; GRANULE_LINES],
        block_type: BlockType,
        bands: &[Band],
        sampling_frequency: SamplingFrequency,
    ) -> (Self, usize) {
        let counted = Self::choose_and_count(values, block_type, bands, sampling_frequency);
        debug_assert_eq!(counted.1, counted.0.bit_count(values, bands));
        counted
    }

    fn choose_and_count(
        values: &[i32; GRANULE_LINES],
        block_type: BlockType,
        bands: &[Band],
        sampling_frequency: SamplingFrequency,
    ) -> (Self, usize) {
        assert!(
            values.iter().all(|v| v.unsigned_abs() <= MAX_MAGNITUDE),
            "quantised magnitudes are limited to {MAX_MAGNITUDE}"
        );
        let (start, end, zeros_end) = Self::partitions(values);
        let run = Self::partitioned(values, start, end, block_type, bands, sampling_frequency);
        let Some(zeros_end) = zeros_end else {
            return run;
        };
        let shorter = Self::partitioned(
            values,
            zeros_end,
            end,
            block_type,
            bands,
            sampling_frequency,
        );
        if shorter.1 < run.1 { shorter } else { run }
    }

    /// `values` counted for the choice of their coding, so that whether its
    /// bits fit ([`Counted::bits_between`]) and the coding itself
    /// ([`Counted::choose`]) can both be had from one count.
    pub fn count(
        values: &[i32; GRANULE_LINES],
        block_type: BlockType,
        bands: &[Band],
        sampling_frequency: SamplingFrequency,
    ) -> Counted {
        match block_type {
            BlockType::Normal => {
                let (start, end, zeros_end) = Self::partitions(values);
                let run = PartitionCounts::new(values, start, end, bands, None);
                // The shorter run's pairs start with the run's.
                let shorter = zeros_end
                    .map(|start| PartitionCounts::new(values, start, end, bands, Some(&run.pairs)));
                Counted::Normal { run, shorter }
            }
            _ => {
                let (coding, bits) =
                    Self::choose_counted(values, block_type, bands, sampling_frequency);
                Counted::Chosen { coding, bits }
            }
        }
    }

    /// The partitions to try: `count1` from the first line returned, or
    /// from the third if given, to the second.
    fn partitions(values: &[i32; GRANULE_LINES]) -> (usize, usize, Option<usize>) {
        // C.1.5.4.4.3: pairs of zeros at the top of the spectrum ("rzero").
        let mut end = GRANULE_LINES;
        while end >= 2 && values[end - 2..end] == [0, 0] {
            end -= 2;
        }
        // C.1.5.4.4.4: quadruples of magnitudes up to 1 below them ("count1").
        let mut start = end;
        while start >= 4
            && values[start - 4..start]
                .iter()
                .all(|v| v.unsigned_abs() <= 1)
        {
            start -= 4;
        }
        // Spec reading: clause C.1.5.4.4.4 puts the whole run of such
        // quadruples in "count1", where each costs a bit at least, even one
        // of zeros. In `big_values` zeros can cost nothing, in a region coded
        // with table 0. So `count1` is also tried from above the run's last
        // quadruple of zeros, and the partition needing fewer bits is kept.
        // Where little is left in a sparse spectrum, as in the right
        // channel of a frame in intensity stereo, that saves most of the
        // bits. It is tried only where the run holds at least 8 quadruples
        // of zeros, as it seldom saves bits with fewer and takes a second
        // count.
        let mut zero_quads = (start..end)
            .step_by(4)
            .filter(|&quad| values[quad..quad + 4] == [0; 4]);
        if zero_quads.clone().count() < 8 {
            return (start, end, None);
        }
        let zeros_end = zero_quads.next_back().expect("quadruples of zeros") + 4;
        (start, end, Some(zeros_end))
    }

    /// The coding of `values` with the `big_values` partition below line
    /// `start` and the `count1` partition from it to line `end`, and its bits.
    fn partitioned(
        values: &[i32; GRANULE_LINES],
        start: usize,
        end: usize,
        block_type: BlockType,
        bands: &[Band],
        sampling_frequency: SamplingFrequency,
    ) -> (Self, usize) {
        let (big_values, count1) = (start / 2, (end - start) / 4);
        let end = start;

        // C.1.5.4.4.5: whichever quadruple table needs fewer bits.
        let quads = &values[end..end + 4 * count1];
        let (bits_a, bits_b) = (
            quad_bits(QuadTable::A, quads),
            quad_bits(QuadTable::B, quads),
        );
        let (count1_table, quad_bits) = if bits_b < bits_a {
            (QuadTable::B, bits_b)
        } else {
            (QuadTable::A, bits_a)
        };

        // With window_switching_flag set, region0_count and region1_count are
        // not sent but set by default: region 0 holds 8 long bands, or 9
        // short bands counted once per window, and region 1 the rest (clause
        // 2.4.2.7). A mixed block has the count of long blocks.
        let (region0_count, region1_count) = match block_type {
            BlockType::Normal => {
                return PartitionCounts::new(values, end, end + 4 * count1, bands, None).coding();
            }
            BlockType::Start | BlockType::Stop | BlockType::Mixed => (7, 36),
            BlockType::Short => (8, 36),
        };
        let mut coding = Self {
            big_values,
            count1,
            table_select: [0; 3],
            region0_count,
            region1_count,
            mixed_region1: (block_type == BlockType::Mixed).then_some(MIXED_LONG_LINES),
            count1_table,
        };
        if block_type == BlockType::Mixed && sampling_frequency.mpeg25() {
            let (table, pair_bits) = cheapest_table(&values[..2 * big_values]);
            coding.table_select = [table, table, 0];
            return (coding, pair_bits + quad_bits);
        }
        let mut start = 0;
        let mut pair_bits = 0;
        for (region, end) in coding.region_ends(bands).into_iter().enumerate() {
            let (table, bits) = cheapest_table(&values[start..end]);
            coding.table_select[region] = table;
            pair_bits += bits;
            start = end;
        }
        (coding, pair_bits + quad_bits)
    }

    /// The line just past each of the three `big_values` regions. Region 0
    /// holds `region0_count + 1` scalefactor bands, or in a mixed block ends
    /// at [`mixed_region1`](Self::mixed_region1), and region 1 the next
    /// `region1_count + 1` (clause 2.4.2.7); all three end at `big_values`
    /// pairs at the latest.
    pub fn region_ends(&self, bands: &[Band]) -> [usize; 3] {
        let lines = 2 * self.big_values;
        let region0_bands = usize::from(self.region0_count) + 1;
        let region1_bands = usize::from(self.region1_count) + 1;
        let boundary = |band: usize| bands.get(band).map_or(lines, |b| b.start.min(lines));
        [
            self.mixed_region1
                .map_or(boundary(region0_bands), |start| start.min(lines)),
            boundary(region0_bands + region1_bands),
            lines,
        ]
    }

    /// The bits [`write`](Self::write) produces for `values`, counted as in
    /// clauses C.1.5.4.4.5 and C.1.5.4.4.8.
    pub fn bit_count(&self, values: &[i32; GRANULE_LINES], bands: &[Band]) -> usize {
        self.bit_count_below(values, bands, GRANULE_LINES)
    }

    /// The bits of [`bit_count`](Self::bit_count) that code the pairs and
    /// quadruples starting below line `end`.
    ///
    /// # Panics
    ///
    /// If `end` is odd.
    pub fn bit_count_below(
        &self,
        values: &[i32; GRANULE_LINES],
        bands: &[Band],
        end: usize,
    ) -> usize {
        assert!(end.is_multiple_of(2), "pairs start on even lines");
        let mut bits = 0;
        let mut start = 0;
        for (region, region_end) in self.region_ends(bands).into_iter().enumerate() {
            let table_select = self.table_select[region];
            let table = pair_table(table_select)
                .unwrap_or_else(|| panic!("table {table_select} is not used"));
            bits += pair_bits(table, &values[start..region_end.min(end).max(start)]);
            start = region_end;
        }
        let quads = end.saturating_sub(start).div_ceil(4).min(self.count1);
        bits + quad_bits(self.count1_table, &values[start..start + 4 * quads])
    }

    /// Writes the `Huffmancodebits()` of `values` (clause 2.4.1.7).
    ///
    /// # Panics
    ///
    /// If a value cannot be coded with the table selected for it, or lies
    /// above the `count1` partition without being zero.
    pub fn write(&self, values: &[i32; GRANULE_LINES], bands: &[Band], w: &mut BitWriter) {
        let mut start = 0;
        for (region, end) in self.region_ends(bands).into_iter().enumerate() {
            let table_select = self.table_select[region];
            let table = pair_table(table_select)
                .unwrap_or_else(|| panic!("table {table_select} is not used"));
            for pair in values[start..end].chunks_exact(2) {
                write_pair(table, pair[0], pair[1], w);
            }
            start = end;
        }
        let count1_end = start + 4 * self.count1;
        for quad in values[start..count1_end].chunks_exact(4) {
            write_quad(self.count1_table, quad, w);
        }
        assert!(
            values[count1_end..].iter().all(|&v| v == 0),
            "nonzero values above the count1 partition"
        );
    }
}

/// Values counted for the choice of their coding (see
/// [`SpectrumCoding::count`]).
pub enum Counted {
    /// A normal block's: each partition to try, with what its coding
    /// needs counted, but its split not yet searched.
    Normal {
        run: PartitionCounts,
        shorter: Option<PartitionCounts>,
    },
    /// Any other block's, whose coding is chosen at once.
    Chosen { coding: SpectrumCoding, bits: usize },
}

impl Counted {
    /// Whether the coding chosen for the values needs between `least` and
    /// `most` bits. For a normal block, the bits lie between the fewest any
    /// partition can need and the fewest that a split of each needs, and
    /// the splits are searched only where those bounds do not settle it.
    pub fn bits_between(&self, least: usize, most: usize) -> bool {
        let between = self.bounded_bits_between(least, most);
        debug_assert_eq!(between, (least..=most).contains(&self.bits()));
        between
    }

    /// The bits of the coding chosen, searching every split.
    fn bits(&self) -> usize {
        match self {
            Self::Normal { run, shorter } => std::iter::once(run)
                .chain(shorter)
                .map(|partition| partition.coding().1)
                .min()
                .expect("a partition"),
            Self::Chosen { bits, .. } => *bits,
        }
    }

    fn bounded_bits_between(&self, least: usize, most: usize) -> bool {
        let (run, shorter) = match self {
            Self::Normal { run, shorter } => (run, shorter),
            Self::Chosen { bits, .. } => return (least..=most).contains(bits),
        };
        let partitions = std::iter::once(run).chain(shorter);
        let lower = (partitions.clone())
            .map(|partition| partition.quad_bits().1 + partition.pairs.least())
            .min()
            .expect("a partition");
        if lower > most {
            return false;
        }
        let upper = (partitions.clone())
            .map(|partition| partition.quad_bits().1 + partition.pairs.upper())
            .min()
            .expect("a partition");
        if upper < least {
            return false;
        }
        if least <= lower && upper <= most {
            return true;
        }
        // The fewest bits are at most `most` if some partition has a split
        // within them, and at least `least` if none has one with fewer.
        let within = |bits: usize| {
            (partitions.clone()).any(|partition| {
                (bits.checked_sub(partition.quad_bits().1))
                    .is_some_and(|pair_bits| partition.pairs.split_within(pair_bits))
            })
        };
        (upper <= most || within(most)) && (least <= lower || !within(least - 1))
    }

    /// The coding [`SpectrumCoding::choose`] gives the values, and its bits.
    pub fn choose(self) -> (SpectrumCoding, usize) {
        match self {
            Self::Normal { run, shorter } => {
                let run = run.coding();
                match shorter.map(|shorter| shorter.coding()) {
                    Some(shorter) if shorter.1 < run.1 => shorter,
                    _ => run,
                }
            }
            Self::Chosen { coding, bits } => (coding, bits),
        }
    }
}

/// One partition of a normal block's values, counted: the `big_values`
/// pairs below line `2 * big_values` and the `count1` quadruples above.
pub struct PartitionCounts {
    big_values: usize,
    count1: usize,
    /// The quadruples' bits with tables A and B.
    quad_bits: [usize; 2],
    pairs: BandBits,
}

impl PartitionCounts {
    /// The partition with `count1` from line `start` to line `end`, taking
    /// over the bands that `prefix` counted whole, if given, from pairs that
    /// these start with.
    fn new(
        values: &[i32; GRANULE_LINES],
        start: usize,
        end: usize,
        bands: &[Band],
        prefix: Option<&BandBits>,
    ) -> Self {
        let quads = &values[start..end];
        Self {
            big_values: start / 2,
            count1: (end - start) / 4,
            quad_bits: quad_bits_with_both(quads),
            pairs: BandBits::new(&values[..start], bands, prefix),
        }
    }

    /// C.1.5.4.4.5: whichever quadruple table needs fewer bits, and those
    /// bits.
    fn quad_bits(&self) -> (QuadTable, usize) {
        let [bits_a, bits_b] = self.quad_bits;
        if bits_b < bits_a {
            (QuadTable::B, bits_b)
        } else {
            (QuadTable::A, bits_a)
        }
    }

    /// The coding with the split of the pairs that needs the fewest bits,
    /// and its bits.
    fn coding(&self) -> (SpectrumCoding, usize) {
        let (count1_table, quad_bits) = self.quad_bits();
        let ((region0_count, region1_count), table_select, pair_bits) = self.pairs.split();
        let coding = SpectrumCoding {
            big_values: self.big_values,
            count1: self.count1,
            table_select,
            region0_count,
            region1_count,
            mixed_region1: None,
            count1_table,
        };
        (coding, pair_bits + quad_bits)
    }
}

/// The bits of the `big_values` pairs of a normal block, band by band, with
/// every set of code words, which [`split`](Self::split) splits.
struct BandBits {
    /// The bands that hold pairs. A split that gives a region only bands
    /// above them needs as many bits as one that ends the region where
    /// they begin, which comes first and so is kept.
    count: usize,
    /// The bands that hold pairs up to their end, which pairs that start
    /// with these but go on further hold the same.
    whole: usize,
    /// For each band, each set of code words' bits for the bands below it:
    /// for a set of one table, its bits, a band it cannot code counting
    /// nothing, as no region with it is given to it; for a set of tables
    /// that differ only in `linbits`, the bits but for the escape fields,
    /// whose number is counted apart, in `escapes_below`.
    below: [[u16; CODE_SETS]; LONG_BANDS + 1],
    escapes_below: [u16; LONG_BANDS + 1],
    /// The fewest bits that the bands below each band need, each band with
    /// the table that codes it alone in the fewest: no split needs fewer.
    least_below: [usize; LONG_BANDS + 1],
    /// The largest magnitude of each band.
    max: [u32; LONG_BANDS],
    /// The split, once searched: both whether a step's bits fit and the
    /// coding of the step kept can need it.
    split: std::cell::OnceCell<Split>,
}

/// `(region0_count, region1_count)`, the table of each region and the bits
/// of the pairs.
type Split = ((u8, u8), [u8; 3], usize);

/// A split found by [`BandBits::splits_below`]: its bits,
/// `(region0_count, region1_count)` and the bands at which its first two
/// regions end.
type SplitFound = (usize, (u8, u8), (usize, usize));

impl BandBits {
    /// The bits of the pairs `values`, taking over the bands that `prefix`
    /// counted whole, if given, from pairs that `values` start with.
    fn new(values: &[i32], bands: &[Band], prefix: Option<&Self>) -> Self {
        let lines = values.len();
        assert!(
            bands.len() <= LONG_BANDS,
            "a normal block has {LONG_BANDS} bands"
        );
        let boundary = |band: usize| bands.get(band).map_or(lines, |b| b.start.min(lines));
        let count = (0..bands.len())
            .take_while(|&band| boundary(band) < lines)
            .count();
        let tables = PairTables::get();
        let mut below = [[0u16; CODE_SETS]; LONG_BANDS + 1];
        let mut escapes_below = [0u16; LONG_BANDS + 1];
        let mut least_below = [0usize; LONG_BANDS + 1];
        let mut max = [0u32; LONG_BANDS];
        let first = match prefix {
            Some(prefix) => {
                let whole = prefix.whole.min(count);
                below[..=whole].copy_from_slice(&prefix.below[..=whole]);
                escapes_below[..=whole].copy_from_slice(&prefix.escapes_below[..=whole]);
                least_below[..=whole].copy_from_slice(&prefix.least_below[..=whole]);
                max[..whole].copy_from_slice(&prefix.max[..whole]);
                whole
            }
            None => 0,
        };
        for band in first..count {
            let PairCounts {
                codes,
                escapes,
                largest,
            } = PairCounts::new(&values[boundary(band)..boundary(band + 1)]);
            max[band] = largest;
            let [previous, next] = below
                .get_disjoint_mut([band, band + 1])
                .expect("distinct bands");
            for set in 0..CODE_SETS {
                let bits = if largest <= tables.largest[set] {
                    codes[set]
                } else {
                    0
                };
                next[set] = previous[set] + bits;
            }
            escapes_below[band + 1] = escapes_below[band] + escapes;
            let least = tables.fewest_bits(&codes, escapes, largest);
            least_below[band + 1] = least_below[band] + least;
        }
        // A granule's pairs need too few bits for the sums to overflow.
        assert!(below[count].iter().all(|&bits| bits <= i16::MAX as u16 / 2));
        let whole = (0..count)
            .take_while(|&band| bands.get(band + 1).is_some_and(|next| next.start <= lines))
            .count();
        Self {
            count,
            whole,
            below,
            escapes_below,
            least_below,
            max,
            split: std::cell::OnceCell::new(),
        }
    }

    /// No split needs fewer bits than this.
    fn least(&self) -> usize {
        self.least_below[self.count]
    }

    /// The bits of one split, which the best needs no more than.
    fn upper(&self) -> usize {
        let count = self.count;
        let end0 = (count / 3).clamp(1.min(count), 16);
        let end1 = (count - count / 4).clamp(end0, end0 + 8).min(count);
        [0..end0, end0..end1, end1..count]
            .map(|range| {
                let largest = self.max[range.clone()].iter().copied().max().unwrap_or(0);
                self.region_bits(range, largest)
            })
            .iter()
            .sum()
    }

    /// The table that codes the bands `range` in the fewest bits, and those
    /// bits.
    fn region(&self, range: std::ops::Range<usize>) -> (u8, usize) {
        let largest = self.max[range.clone()].iter().copied().max().unwrap_or(0);
        let (bits, escapes) = self.region_counts(range);
        PairTables::get().cheapest(&bits, escapes, largest)
    }

    /// [`region`](Self::region), given the largest magnitude of the bands.
    fn region_bits(&self, range: std::ops::Range<usize>, largest: u32) -> usize {
        let (bits, escapes) = self.region_counts(range);
        PairTables::get().fewest_bits(&bits, escapes, largest)
    }

    /// The bits of the bands `range` with each set of code words, but for
    /// escape fields, and their number of escapes.
    fn region_counts(&self, range: std::ops::Range<usize>) -> ([u16; CODE_SETS], u16) {
        let (end, start) = (&self.below[range.end], &self.below[range.start]);
        let escapes = self.escapes_below[range.end] - self.escapes_below[range.start];
        let bits = std::array::from_fn(|set| end[set].wrapping_sub(start[set]));
        (bits, escapes)
    }

    /// SUBDIVIDE (clause C.1.5.4.4.6) and the choice of tables (clause
    /// C.1.5.4.4.7) for the pairs: the split needing the fewest bits,
    /// `(region0_count, region1_count)`, the table of each region and the
    /// bits of the pairs.
    ///
    /// Spec gap: "The split strategy is up to the implementation" (clause
    /// C.1.5.4.4.6), which gives assigning a third of the bands to the first
    /// region and a quarter to the last only as "a very simple one for
    /// instance". Here every split the fields allow is tried, each region with
    /// the table that codes it in the fewest bits (see [`cheapest_table`]), and
    /// the split needing the fewest bits overall is kept. The bits it saves let
    /// the inner loop take a finer step. Graded with PEAQ (GstPEAQ 0.6.1, black
    /// box), 22 tracks of pop music at 44.1 kHz in stereo rose from a mean of
    /// -1.88 to -1.80 at 96 kbit/s, from -0.69 to -0.66 at 128 kbit/s and from
    /// -0.18 to -0.17 at 192 kbit/s, and 14 of rock from -0.86 to -0.82 at
    /// 128 kbit/s (from -1.57 to -1.47 for the worst track) and from -0.22 to
    /// -0.21 at 192 kbit/s; at `--vbr` the streams came out 1 to 2% smaller at
    /// the same grade. Encoding takes about 1.4 times as long.
    fn split(&self) -> Split {
        *self.split.get_or_init(|| self.search_split())
    }

    /// [`split`](Self::split), searched.
    fn search_split(&self) -> Split {
        // The split of `upper` needs no fewer bits than the best, so those
        // bits plus one bound the search from the start.
        let (bits, counts, (end0, end1)) = self
            .splits_below(self.upper() + 1, true)
            .expect("at least one split");
        let tables = [0..end0, end0..end1, end1..self.count].map(|range| self.region(range).0);
        (counts, tables, bits)
    }

    /// Whether some split needs at most `most` bits, searching only until
    /// one is found where the split is not yet known.
    fn split_within(&self, most: usize) -> bool {
        match self.split.get() {
            Some(&(_, _, bits)) => bits <= most,
            None => self.splits_below(most + 1, false).is_some(),
        }
    }

    /// The first split, in the order [`split`](Self::split) tries them,
    /// that needs fewer than `limit` bits, or with `fewest`, the first of
    /// those needing the fewest; with its bits, `(region0_count,
    /// region1_count)` and the bands at which its first two regions end. The
    /// regions' tables are not chosen.
    fn splits_below(&self, mut limit: usize, fewest: bool) -> Option<SplitFound> {
        let (count, least_below) = (self.count, &self.least_below);
        // The largest magnitudes of the regions are carried from one
        // region to the next that it grows into.
        let region = |range, largest| self.region_bits(range, largest);
        // The last region's bits, for each band it may start at.
        let mut last = [0usize; LONG_BANDS + 1];
        let mut largest2 = 0;
        for start in (0..=count).rev() {
            if start < count {
                largest2 = largest2.max(self.max[start]);
            }
            last[start] = region(start..count, largest2);
        }
        let mut largest0 = 0;
        let mut best = None;
        // region0_count and region1_count are 4- and 3-bit fields.
        // Splits that cannot need fewer bits than the limit, which is that
        // of the best so far, are skipped: only one needing fewer would be
        // kept.
        for region0 in 0..16u8 {
            let end0 = (usize::from(region0) + 1).min(count);
            if end0 > 0 {
                largest0 = largest0.max(self.max[end0 - 1]);
            }
            let bits0 = region(0..end0, largest0);
            let (mut end, mut largest1) = (end0, 0);
            for region1 in 0..8u8 {
                let end1 = (end0 + usize::from(region1) + 1).min(count);
                for band in end..end1 {
                    largest1 = largest1.max(self.max[band]);
                }
                end = end1;
                let bits2 = last[end1];
                let at_least = bits0 + (least_below[end1] - least_below[end0]) + bits2;
                if at_least >= limit {
                    if end1 == count {
                        break;
                    }
                    continue;
                }
                let bits1 = region(end0..end1, largest1);
                let bits = bits0 + bits1 + bits2;
                if bits < limit {
                    best = Some((bits, (region0, region1), (end0, end1)));
                    if !fewest {
                        return best;
                    }
                    limit = bits;
                }
                if end1 == count {
                    break;
                }
            }
            if end0 == count {
                break;
            }
        }
        best
    }
}

/// What the bits of some pairs with each set of code words come to.
struct PairCounts {
    /// The bits of their code words in each set, with a sign bit for each
    /// nonzero value, and nothing for a pair that a set has no code word
    /// for.
    codes: [u16; CODE_SETS],
    /// Their values of 15 or more, which each have an escape field.
    escapes: u16,
    /// Their largest magnitude.
    largest: u32,
}

impl PairCounts {
    fn new(values: &[i32]) -> Self {
        let tables = PairTables::get();
        let mut counts = Self {
            codes: [0; CODE_SETS],
            escapes: 0,
            largest: 0,
        };
        for pair in values.chunks_exact(2) {
            let [x, y] = [pair[0], pair[1]].map(i32::unsigned_abs);
            counts.largest = counts.largest.max(x).max(y);
            counts.escapes += u16::from(x >= 15) + u16::from(y >= 15);
            let lens = &tables.lens[(16 * x.min(15) + y.min(15)) as usize];
            for (codes, &len) in counts.codes.iter_mut().zip(lens) {
                *codes += u16::from(len);
            }
        }
        counts
    }
}

/// The sets of code words that the pair tables use, in the order of their
/// first table: each table but 16 to 23, which share one, and 24 to 31,
/// which share another.
const CODE_SETS: usize = 16;

/// The sets of code words of more than one table: those of tables 16 to 23
/// and 24 to 31, which differ only in `linbits`.
const ESCAPE_SETS: [usize; 2] = [CODE_SETS - 2, CODE_SETS - 1];

/// What [`BandBits`] needs of each table that is used, worked out once.
struct PairTables {
    /// The largest magnitude each set of code words can code, in its own
    /// array to be vectorised.
    largest: [u32; CODE_SETS],
    /// What [`cheapest`](Self::cheapest) takes from each set of code words
    /// for each class of largest magnitude, indexed by [`magnitude_class`].
    choices: [Choice; MAGNITUDE_CLASSES],
    /// For each pair of magnitudes, capped at 15, as `16 * x + y`, the length
    /// of its code word in each set of code words with its sign bits; zero
    /// where a set has none.
    lens: [[u8; CODE_SETS]; 256],
}

/// For a largest magnitude, each set of code words' first table that can
/// code it, which of the set needs the fewest bits.
#[derive(Clone, Copy, Default)]
struct Choice {
    /// Whether the set has such a table.
    codes: [bool; CODE_SETS],
    /// Its `linbits`, the bits of each escape field.
    linbits: [i16; CODE_SETS],
    table_select: [u8; CODE_SETS],
}

/// The classes of largest magnitude for which [`Choice`] differs: each up to
/// 15, and above it each width of the escape field that it needs.
const MAGNITUDE_CLASSES: usize = 16 + MAX_MAGNITUDE.ilog2() as usize + 1;

/// The class of `largest` among the [`MAGNITUDE_CLASSES`]: the magnitude up
/// to 15, and above it 15 plus the bits of `largest - 15`, which are the
/// fewest `linbits` that code it, as an escape field codes `largest - 15`.
fn magnitude_class(largest: u32) -> usize {
    match largest.checked_sub(15) {
        None | Some(0) => largest as usize,
        Some(escaped) => 15 + escaped.ilog2() as usize + 1,
    }
}

#[derive(Clone, Copy)]
struct TableEntry {
    table_select: u8,
    /// The largest magnitude it can code, or for a set of code words, the
    /// largest that any of its tables can.
    largest: u32,
    linbits: u32,
}

impl PairTables {
    fn get() -> &'static Self {
        static TABLES: std::sync::OnceLock<PairTables> = std::sync::OnceLock::new();
        TABLES.get_or_init(|| {
            let mut sets: Vec<(&'static PairTable, Vec<TableEntry>)> = Vec::new();
            for (table_select, table) in
                (0..32u8).filter_map(|t| pair_table(t).map(|table| (t, table)))
            {
                let entry = TableEntry {
                    table_select,
                    largest: if table.linbits == 0 {
                        table.size as u32 - 1
                    } else {
                        15 + (1 << table.linbits) - 1
                    },
                    linbits: table.linbits,
                };
                match sets
                    .iter_mut()
                    .find(|(set, _)| std::ptr::eq(set.codes, table.codes))
                {
                    Some((_, entries)) => entries.push(entry),
                    None => sets.push((table, vec![entry])),
                }
            }
            assert_eq!(sets.len(), CODE_SETS, "sets of code words");
            let mut lens = [[0; CODE_SETS]; 256];
            for (set, (table, entries)) in sets.iter().enumerate() {
                assert_eq!(
                    entries.len() > 1,
                    ESCAPE_SETS.contains(&set),
                    "set {set} of code words"
                );
                for x in 0..table.size {
                    for y in 0..table.size {
                        let signs = u32::from(x != 0) + u32::from(y != 0);
                        lens[16 * x + y][set] = u8::try_from(table.codeword(x, y).len + signs)
                            .expect("code words are short");
                    }
                }
            }
            let largest: [u32; CODE_SETS] = std::array::from_fn(|set| {
                (sets[set].1.iter().map(|e| e.largest).max()).expect("a table per set")
            });
            let mut choices = [Choice::default(); MAGNITUDE_CLASSES];
            for magnitude in 0..=MAX_MAGNITUDE {
                let mut choice = Choice::default();
                for (set, (_, entries)) in sets.iter().enumerate() {
                    if let Some(first) = entries.iter().find(|entry| magnitude <= entry.largest) {
                        choice.codes[set] = true;
                        choice.linbits[set] = i16::try_from(first.linbits).expect("linbits");
                        choice.table_select[set] = first.table_select;
                    }
                }
                let class = &mut choices[magnitude_class(magnitude)];
                if magnitude == 0 || magnitude_class(magnitude) != magnitude_class(magnitude - 1) {
                    *class = choice;
                }
                assert!(
                    class.codes == choice.codes
                        && class.linbits == choice.linbits
                        && class.table_select == choice.table_select,
                    "magnitude {magnitude} chooses as its class does"
                );
            }
            Self {
                largest,
                choices,
                lens,
            }
        })
    }

    /// Of the tables that can code `largest`, the first of those needing
    /// the fewest bits, and its bits, given the bits but for escape fields
    /// with each set of code words, `bits`, and the number of `escapes`.
    fn cheapest(&self, bits: &[u16; CODE_SETS], escapes: u16, largest: u32) -> (u8, usize) {
        // Each set's bits with its first table that can code `largest`,
        // which of the set needs the fewest, or i16::MAX where none can,
        // laid out so that the compiler can vectorise it; of the fewest,
        // the table first in order is kept.
        let choice = &self.choices[magnitude_class(largest)];
        let lanes = Self::lanes(choice, bits, escapes);
        let least = lanes.iter().copied().fold(i16::MAX, i16::min);
        let set = lanes
            .iter()
            .position(|&bits| bits == least)
            .expect("table 31 codes every allowed magnitude");
        (choice.table_select[set], least as usize)
    }

    /// The bits of [`cheapest`](Self::cheapest), without its table.
    fn fewest_bits(&self, bits: &[u16; CODE_SETS], escapes: u16, largest: u32) -> usize {
        let choice = &self.choices[magnitude_class(largest)];
        let lanes = Self::lanes(choice, bits, escapes);
        lanes.iter().copied().fold(i16::MAX, i16::min) as usize
    }

    /// Each set's bits with its first table that can code the largest
    /// magnitude of `choice`, or i16::MAX where none can.
    fn lanes(choice: &Choice, bits: &[u16; CODE_SETS], escapes: u16) -> [i16; CODE_SETS] {
        let mut lanes = [0i16; CODE_SETS];
        for set in 0..CODE_SETS {
            lanes[set] = if choice.codes[set] {
                bits[set] as i16 + escapes as i16 * choice.linbits[set]
            } else {
                i16::MAX
            };
        }
        lanes
    }
}

/// Chooses a table for a region's `values`: of every table that can code
/// their largest magnitude, the one needing the fewest bits. Spec gap:
/// clause C.1.5.4.4.7 gives choosing by that magnitude as "a simple way",
/// and "additional coding gain" from "trying all" the tables of the same
/// size. Spec reading: tables of a larger size are tried too, as one can
/// need fewer bits.
///
/// Returns the table and the bits of `values` with it.
fn cheapest_table(values: &[i32]) -> (u8, usize) {
    let counts = PairCounts::new(values);
    PairTables::get().cheapest(&counts.codes, counts.escapes, counts.largest)
}

fn can_code(table: &PairTable, magnitude: u32) -> bool {
    if table.linbits == 0 {
        (magnitude as usize) < table.size
    } else {
        magnitude - magnitude.min(15) < 1 << table.linbits
    }
}

/// The bits needed to code `values` as pairs with `table`, sign and escape
/// bits included (clause C.1.5.4.4.8).
fn pair_bits(table: &PairTable, values: &[i32]) -> usize {
    let value_bits = |v: i32| {
        let magnitude = v.unsigned_abs();
        let escape = if magnitude >= 15 { table.linbits } else { 0 };
        (escape + u32::from(magnitude != 0)) as usize
    };
    values
        .chunks_exact(2)
        .map(|pair| {
            let [x, y] = [pair[0], pair[1]].map(|v| v.unsigned_abs().min(15) as usize);
            table.codeword(x, y).len as usize + value_bits(pair[0]) + value_bits(pair[1])
        })
        .sum()
}

/// The bits needed to code `values` as quadruples with `table`, sign bits
/// included (clause C.1.5.4.4.5).
fn quad_bits(table: QuadTable, values: &[i32]) -> usize {
    values
        .chunks_exact(4)
        .map(|quad| {
            let signs = quad.iter().filter(|&&v| v != 0).count();
            table.codeword(vwxy(quad)).len as usize + signs
        })
        .sum()
}

/// [`quad_bits`] with tables A and B, in that order, indexing each
/// quadruple once.
fn quad_bits_with_both(values: &[i32]) -> [usize; 2] {
    static LENS: std::sync::OnceLock<[[u8; 2]; 16]> = std::sync::OnceLock::new();
    // For each index, the length of its code word in each table with its
    // sign bits.
    let lens = LENS.get_or_init(|| {
        std::array::from_fn(|vwxy| {
            let vwxy = vwxy as u8;
            [QuadTable::A, QuadTable::B].map(|table| {
                u8::try_from(table.codeword(vwxy).len + vwxy.count_ones())
                    .expect("code words are short")
            })
        })
    });
    values.chunks_exact(4).fold([0, 0], |[a, b], quad| {
        let [len_a, len_b] = lens[usize::from(vwxy(quad))];
        [a + usize::from(len_a), b + usize::from(len_b)]
    })
}

/// The table index of a quadruple. The normative syntax indexes
/// `hcod[|v|][|w|][|x|][|y|]` and Table B.7 labels its rows `vwxy`, so `v`, the
/// first value, is the leftmost bit. (The bit counting formula of clause
/// C.1.5.4.4.5 weights the first value 1 and the last 8; that informative
/// formula indexes its own table and is not followed here.)
fn vwxy(quad: &[i32]) -> u8 {
    quad.iter().fold(0, |index, &v| {
        let magnitude = v.unsigned_abs();
        assert!(magnitude <= 1, "count1 values are limited to magnitude 1");
        index << 1 | magnitude as u8
    })
}

/// One `big_values` pair: the code word, then for each of `x` and `y` its
/// escape field (if it is 15 or more and the table has `linbits`) and its sign
/// (if it is not zero).
fn write_pair(table: &PairTable, x: i32, y: i32, w: &mut BitWriter) {
    let [mx, my] = [x, y].map(|v| v.unsigned_abs());
    for magnitude in [mx, my] {
        assert!(
            can_code(table, magnitude),
            "magnitude {magnitude} is out of range for a table of size {} with linbits {}",
            table.size,
            table.linbits
        );
    }
    let code = table.codeword(mx.min(15) as usize, my.min(15) as usize);
    w.write(code.bits, code.len);
    for (value, magnitude) in [(x, mx), (y, my)] {
        if magnitude >= 15 && table.linbits > 0 {
            w.write(magnitude - 15, table.linbits);
        }
        if value != 0 {
            w.write_bool(value < 0);
        }
    }
}

/// One `count1` quadruple: the code word, then the signs of the nonzero values
/// in order.
fn write_quad(table: QuadTable, quad: &[i32], w: &mut BitWriter) {
    let code = table.codeword(vwxy(quad));
    w.write(code.bits, code.len);
    for &value in quad {
        if value != 0 {
            w.write_bool(value < 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn select_table(values: &[i32]) -> u8 {
        cheapest_table(values).0
    }
    use crate::mp3::header::SamplingFrequency;
    use crate::mp3::scalefactor_bands::{coding_bands, long_bands};

    const FS: SamplingFrequency = SamplingFrequency::Hz44100;

    fn bands() -> &'static [Band] {
        long_bands(FS)
    }

    fn spectrum(prefix: &[i32]) -> [i32; GRANULE_LINES] {
        let mut values = [0; GRANULE_LINES];
        values[..prefix.len()].copy_from_slice(prefix);
        values
    }

    fn written(values: &[i32; GRANULE_LINES], coding: &SpectrumCoding) -> String {
        let mut w = BitWriter::new();
        coding.write(values, bands(), &mut w);
        let len = w.bit_len();
        let bits: String = w.into_bytes().iter().map(|b| format!("{b:08b}")).collect();
        bits[..len].to_owned()
    }

    #[test]
    fn silence_codes_to_nothing() {
        let values = [0; GRANULE_LINES];
        let coding = SpectrumCoding::choose(&values, BlockType::Normal, bands(), FS);
        assert_eq!(coding.big_values, 0);
        assert_eq!(coding.count1, 0);
        assert_eq!(coding.table_select, [0; 3]);
        assert_eq!((coding.region0_count, coding.region1_count), (0, 0));
        assert_eq!(written(&values, &coding), "");
    }

    #[test]
    fn partitions_into_big_values_count1_and_zeros() {
        // Pairs: (5, 0) (0, 1); quadruples: (1, -1, 0, 0) (0, 0, 0, 1); then zeros.
        let values = spectrum(&[5, 0, 0, 1, 1, -1, 0, 0, 0, 0, 0, 1]);
        let coding = SpectrumCoding::choose(&values, BlockType::Normal, bands(), FS);
        assert_eq!(coding.big_values, 2);
        assert_eq!(coding.count1, 2);
    }

    #[test]
    fn zeros_inside_the_top_nonzero_pair_stay_in_count1() {
        // Zero pairs are counted whole: line 9 is zero but line 8 is not.
        let values = spectrum(&[2, 0, 0, 0, 0, 0, 0, 0, 1, 0]);
        let coding = SpectrumCoding::choose(&values, BlockType::Normal, bands(), FS);
        assert_eq!((coding.big_values, coding.count1), (1, 2));
    }

    /// Spectra whose statistics change from band to band, so that the
    /// split matters: large values at the bottom, an escape among them, and
    /// sparse small ones above.
    fn varied_spectra() -> Vec<[i32; GRANULE_LINES]> {
        let decaying: [i32; GRANULE_LINES] = std::array::from_fn(|i| {
            let magnitude = (40.0 * (-(i as f64) / 50.0).exp()) as i32;
            if i % 3 == 0 { -magnitude } else { magnitude }
        });
        let mut sparse = [0; GRANULE_LINES];
        for i in (0..400).step_by(7) {
            sparse[i] = if i < 60 { 9 } else { 2 };
        }
        sparse[3] = 200;
        vec![decaying, sparse, spectrum(&[3; 40]), spectrum(&[1, 2])]
    }

    /// Pairs counted from the bands that shorter pairs counted whole count
    /// as they do counted afresh.
    #[test]
    fn takes_over_the_whole_bands_of_shorter_pairs() {
        for values in varied_spectra() {
            for short in (0..=GRANULE_LINES).step_by(10) {
                let prefix = BandBits::new(&values[..short], bands(), None);
                for long in (short..=GRANULE_LINES).step_by(14) {
                    let fresh = BandBits::new(&values[..long], bands(), None);
                    let taken = BandBits::new(&values[..long], bands(), Some(&prefix));
                    assert_eq!(
                        (fresh.count, fresh.whole, fresh.below, fresh.escapes_below),
                        (taken.count, taken.whole, taken.below, taken.escapes_below),
                        "{short} to {long}"
                    );
                    assert_eq!(
                        (fresh.least_below, fresh.max),
                        (taken.least_below, taken.max),
                        "{short} to {long}"
                    );
                }
            }
        }
    }

    /// No split the fields allow, each region with the table that codes it
    /// in the fewest bits, needs fewer bits than the split chosen.
    #[test]
    fn subdivides_where_the_fewest_bits_are_needed() {
        for values in varied_spectra() {
            let coding = SpectrumCoding::choose(&values, BlockType::Normal, bands(), FS);
            let chosen = coding.bit_count(&values, bands());
            for region0_count in 0..16 {
                for region1_count in 0..8 {
                    let mut other = SpectrumCoding {
                        region0_count,
                        region1_count,
                        ..coding.clone()
                    };
                    let mut start = 0;
                    for (region, end) in other.region_ends(bands()).into_iter().enumerate() {
                        other.table_select[region] = select_table(&values[start..end]);
                        start = end;
                    }
                    assert!(
                        other.bit_count(&values, bands()) >= chosen,
                        "({region0_count}, {region1_count}) beats {:?}",
                        (coding.region0_count, coding.region1_count)
                    );
                }
            }
        }
    }

    /// A lone value of 1 above many zeros: the zeros go to `big_values`,
    /// where table 0 codes them in no bits, rather than to `count1` as the
    /// run of clause C.1.5.4.4.4 would have them.
    #[test]
    fn zeros_below_a_lone_value_go_to_big_values() {
        let mut values = [0; GRANULE_LINES];
        values[201] = 1;
        let coding = SpectrumCoding::choose(&values, BlockType::Normal, bands(), FS);
        // Line 201 lies in the quadruple of lines 198 to 201.
        assert_eq!((coding.big_values, coding.count1), (99, 1));
        // The run: from line 2, as lines 0 and 1 are left a pair.
        let run = SpectrumCoding {
            big_values: 1,
            count1: 50,
            ..coding.clone()
        };
        assert!(coding.bit_count(&values, bands()) < run.bit_count(&values, bands()));
    }

    #[test]
    fn region_ends_stop_at_big_values() {
        let values = spectrum(&[3; 40]);
        let coding = SpectrumCoding::choose(&values, BlockType::Normal, bands(), FS);
        assert_eq!(coding.big_values, 20);
        let ends = coding.region_ends(bands());
        assert_eq!(ends[2], 40);
        assert!(ends[0] <= ends[1] && ends[1] <= 40);
    }

    /// With window switching the regions are fixed: region 0 ends after 8
    /// long bands, or for short blocks after short band 2 in all three
    /// windows, and region 1 takes the rest of `big_values`.
    #[test]
    fn switched_windows_use_the_default_regions() {
        let fs = SamplingFrequency::Hz44100;
        let values = spectrum(&[3; 200]);
        for block_type in [BlockType::Start, BlockType::Stop] {
            let coding = SpectrumCoding::choose(&values, block_type, long_bands(fs), fs);
            assert_eq!(
                coding.region_ends(long_bands(fs)),
                [long_bands(fs)[8].start, 200, 200]
            );
            assert_eq!(coding.table_select[2], 0);
        }
        let short = coding_bands(BlockType::Short, fs);
        let coding = SpectrumCoding::choose(&values, BlockType::Short, &short, fs);
        assert_eq!(coding.region_ends(&short), [36, 200, 200]);
    }

    /// Region 1 of a mixed block begins at line 36, where its short bands
    /// begin, at the lower sampling frequencies too; at MPEG-2.5, where
    /// decoders disagree on it, both regions have the same table.
    #[test]
    fn mixed_blocks_begin_region_1_below_the_short_bands() {
        // Large values below line 36 and small ones above.
        let mut values = spectrum(&[9; 36]);
        values[36..200].fill(1);
        values[198] = 2;
        for fs in [
            SamplingFrequency::Hz44100,
            SamplingFrequency::Hz22050,
            SamplingFrequency::Hz11025,
        ] {
            let mixed = coding_bands(BlockType::Mixed, fs);
            let coding = SpectrumCoding::choose(&values, BlockType::Mixed, &mixed, fs);
            assert_eq!((coding.region0_count, coding.region1_count), (7, 36));
            assert_eq!(coding.region_ends(&mixed), [36, 200, 200], "{fs:?}");
            let [region0, region1, _] = coding.table_select;
            assert_eq!(region0 == region1, fs.mpeg25(), "{fs:?}");
        }
    }

    /// Of the tables that can code the largest magnitude, the one chosen
    /// needs the fewest bits, of whatever size.
    #[test]
    fn selects_the_table_needing_the_fewest_bits() {
        assert_eq!(select_table(&[0, 0, 0, 0]), 0);
        assert_eq!(select_table(&[1, 0, 0, -1]), 1);
        // 16 needs an escape table with at least 1 linbit; 8191 needs 13.
        assert!(select_table(&[16, 0]) >= 16);
        assert!([23, 31].contains(&select_table(&[8191, 0])));
        let mut values = vec![2, 0, 1, 1, 15, 0, 1, 1];
        values.extend(varied_spectra()[0][..64].iter());
        for values in values.chunks(4).chain([&values[..]]) {
            let max = values.iter().map(|v| v.unsigned_abs()).max().unwrap();
            let chosen = pair_table(select_table(values)).unwrap();
            assert!(can_code(chosen, max));
            for other in (0..32).filter_map(pair_table) {
                if can_code(other, max) {
                    assert!(pair_bits(chosen, values) <= pair_bits(other, values));
                }
            }
        }
    }

    /// The table chosen is the first of those needing the fewest bits, as
    /// trying every table in order finds, and its bits are theirs.
    #[test]
    fn chooses_the_first_table_needing_the_fewest_bits() {
        let mut state = 12345u32;
        let mut next = |bound: u32| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (state >> 8) % bound
        };
        for case in 0..3000 {
            let largest = [1, 2, 3, 5, 7, 15, 16, 40, 300, 8191][case % 10];
            let len = 2 * (1 + next(40) as usize);
            let values: Vec<i32> = (0..len)
                .map(|_| {
                    let magnitude = if next(4) == 0 {
                        next(largest + 1)
                    } else {
                        next(largest.min(3) + 1)
                    };
                    if next(2) == 0 {
                        magnitude as i32
                    } else {
                        -(magnitude as i32)
                    }
                })
                .collect();
            let max = values.iter().map(|v| v.unsigned_abs()).max().unwrap();
            let expected = (0..32u8)
                .filter_map(|t| pair_table(t).map(|table| (t, table)))
                .filter(|(_, table)| can_code(table, max))
                .map(|(t, table)| (t, pair_bits(table, &values)))
                .min_by_key(|&(_, bits)| bits)
                .unwrap();
            assert_eq!(cheapest_table(&values), expected, "{values:?}");
        }
    }

    #[test]
    fn writes_pairs_with_signs_and_escapes() {
        // Table 1, x = 1, y = 0: "01", then signx = 1 for -1.
        let mut w = BitWriter::new();
        write_pair(pair_table(1).unwrap(), -1, 0, &mut w);
        assert_eq!(w.into_bytes(), [0b0110_0000]);

        // Table 16 (linbits 1), x = 16, y = -1: row "15 1 8 00001010", then
        // linbitsx = 16 - 15 = 1, signx = 0, signy = 1.
        let mut w = BitWriter::new();
        write_pair(pair_table(16).unwrap(), 16, -1, &mut w);
        assert_eq!(w.bit_len(), 11);
        assert_eq!(w.into_bytes(), [0b0000_1010, 0b1010_0000]);
    }

    #[test]
    fn writes_quadruples_with_v_leftmost() {
        // Table A row "1000 4 0111" is v = 1: code "0111", then signv.
        let mut w = BitWriter::new();
        write_quad(QuadTable::A, &[-1, 0, 0, 0], &mut w);
        assert_eq!(w.into_bytes(), [0b0111_1000]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn rejects_values_the_table_cannot_code() {
        write_pair(pair_table(13).unwrap(), 16, 0, &mut BitWriter::new());
    }

    #[test]
    fn written_length_matches_the_counted_bits() {
        let mut values = [0; GRANULE_LINES];
        for (i, v) in values.iter_mut().enumerate() {
            *v = match i {
                0..100 => [0, 1, -2, 7, 0, -15, 40, 1, 0, -1][i % 10],
                100..300 => [1, 0, -1, 0, 0][i % 5],
                _ => 0,
            };
        }
        let coding = SpectrumCoding::choose(&values, BlockType::Normal, bands(), FS);
        assert_eq!(
            written(&values, &coding).len(),
            coding.bit_count(&values, bands())
        );
    }
    /// The bits below a line grow from none to all, count the pairs before
    /// the quadruples, and count a quadruple that starts below the line
    /// whole.
    #[test]
    fn bits_below_a_line_count_what_starts_below_it() {
        let mut values = [0; GRANULE_LINES];
        for (i, v) in values.iter_mut().enumerate() {
            *v = match i {
                0..100 => [0, 1, -2, 7, 0, -15, 40, 1, 0, -1][i % 10],
                100..300 => [1, 0, -1, 0, 0][i % 5],
                _ => 0,
            };
        }
        let coding = SpectrumCoding::choose(&values, BlockType::Normal, bands(), FS);
        let below = |end| coding.bit_count_below(&values, bands(), end);
        let total = coding.bit_count(&values, bands());
        assert_eq!(below(0), 0);
        assert_eq!(below(GRANULE_LINES), total);
        let counts: Vec<usize> = (0..=GRANULE_LINES).step_by(2).map(below).collect();
        assert!(counts.windows(2).all(|w| w[0] <= w[1]));
        let count1_start = 2 * coding.big_values;
        let quads = &values[count1_start..count1_start + 4 * coding.count1];
        let quad_total = quad_bits(coding.count1_table, quads);
        assert_eq!(below(count1_start), total - quad_total);
        let first_quad = quad_bits(coding.count1_table, &quads[..4]);
        assert_eq!(below(count1_start + 2), total - quad_total + first_quad);
        assert_eq!(below(count1_start + 4 * coding.count1), total);
    }
}
