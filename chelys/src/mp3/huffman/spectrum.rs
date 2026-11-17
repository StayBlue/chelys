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
        assert!(
            values.iter().all(|v| v.unsigned_abs() <= MAX_MAGNITUDE),
            "quantised magnitudes are limited to {MAX_MAGNITUDE}"
        );

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
        let run = Self::partitioned(values, start, end, block_type, bands, sampling_frequency);
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
            return run;
        }
        let zeros_end = zero_quads.next_back().expect("quadruples of zeros") + 4;
        let shorter = Self::partitioned(
            values,
            zeros_end,
            end,
            block_type,
            bands,
            sampling_frequency,
        );
        if shorter.bit_count(values, bands) < run.bit_count(values, bands) {
            shorter
        } else {
            run
        }
    }

    /// The coding of `values` with the `big_values` partition below line
    /// `start` and the `count1` partition from it to line `end`.
    fn partitioned(
        values: &[i32; GRANULE_LINES],
        start: usize,
        end: usize,
        block_type: BlockType,
        bands: &[Band],
        sampling_frequency: SamplingFrequency,
    ) -> Self {
        let (big_values, count1) = (start / 2, (end - start) / 4);
        let end = start;

        // C.1.5.4.4.5: whichever quadruple table needs fewer bits.
        let quads = &values[end..end + 4 * count1];
        let count1_table = if quad_bits(QuadTable::B, quads) < quad_bits(QuadTable::A, quads) {
            QuadTable::B
        } else {
            QuadTable::A
        };

        // With window_switching_flag set, region0_count and region1_count are
        // not sent but set by default: region 0 holds 8 long bands, or 9
        // short bands counted once per window, and region 1 the rest (clause
        // 2.4.2.7). A mixed block has the count of long blocks.
        let (region0_count, region1_count) = match block_type {
            BlockType::Normal => {
                let (counts, table_select) = subdivide(&values[..end], bands);
                return Self {
                    big_values,
                    count1,
                    table_select,
                    region0_count: counts.0,
                    region1_count: counts.1,
                    mixed_region1: None,
                    count1_table,
                };
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
            let table = select_table(&values[..2 * big_values]);
            coding.table_select = [table, table, 0];
            return coding;
        }
        let mut start = 0;
        for (region, end) in coding.region_ends(bands).into_iter().enumerate() {
            coding.table_select[region] = select_table(&values[start..end]);
            start = end;
        }
        coding
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

/// SUBDIVIDE (clause C.1.5.4.4.6) and the choice of tables (clause
/// C.1.5.4.4.7) for the `big_values` pairs `values` of a normal block:
/// returns `(region0_count, region1_count)` and the table of each region.
///
/// Spec gap: "The split strategy is up to the implementation" (clause
/// C.1.5.4.4.6), which gives assigning a third of the bands to the first
/// region and a quarter to the last only as "a very simple one for
/// instance". Here every split the fields allow is tried, each region with
/// the table that codes it in the fewest bits (see [`select_table`]), and
/// the split needing the fewest bits overall is kept. The bits it saves let
/// the inner loop take a finer step. Graded with PEAQ (GstPEAQ 0.6.1, black
/// box), 22 tracks of pop music at 44.1 kHz in stereo rose from a mean of
/// -1.88 to -1.80 at 96 kbit/s, from -0.69 to -0.66 at 128 kbit/s and from
/// -0.18 to -0.17 at 192 kbit/s, and 14 of rock from -0.86 to -0.82 at
/// 128 kbit/s (from -1.57 to -1.47 for the worst track) and from -0.22 to
/// -0.21 at 192 kbit/s; at `--vbr` the streams came out 1 to 2% smaller at
/// the same grade. Encoding takes about 1.4 times as long.
fn subdivide(values: &[i32], bands: &[Band]) -> ((u8, u8), [u8; 3]) {
    let lines = values.len();
    let count = bands.len();
    assert!(count <= LONG_BANDS, "a normal block has {LONG_BANDS} bands");
    let tables = PairTables::get();
    // For each table, its bits for the bands below each band, a band it
    // cannot code counting nothing, as no region with it is given to it;
    // and the largest magnitude of each band.
    let mut below = [[0usize; LONG_BANDS + 1]; 32];
    let mut max = [0u32; LONG_BANDS];
    // How often each pair of magnitudes, capped at 15, occurs in a band, as
    // `16 * x + y`, and the pairs that occur.
    let mut pairs = [0u16; 256];
    let mut occurring = [0u8; 256];
    let boundary = |band: usize| bands.get(band).map_or(lines, |b| b.start.min(lines));
    for band in 0..count {
        let (mut signs, mut escapes, mut distinct) = (0, 0, 0);
        for pair in values[boundary(band)..boundary(band + 1)].chunks_exact(2) {
            let [x, y] = [pair[0], pair[1]].map(i32::unsigned_abs);
            max[band] = max[band].max(x).max(y);
            signs += usize::from(x != 0) + usize::from(y != 0);
            escapes += usize::from(x >= 15) + usize::from(y >= 15);
            let index = (16 * x.min(15) + y.min(15)) as u8;
            if pairs[usize::from(index)] == 0 {
                occurring[distinct] = index;
                distinct += 1;
            }
            pairs[usize::from(index)] += 1;
        }
        let occurring = &occurring[..distinct];
        // Tables that share their code words differ only in `linbits`, so
        // their code words are counted once.
        let mut code_bits = [None::<usize>; 32];
        for entry in &tables.entries {
            let t = usize::from(entry.table_select);
            let mut bits = 0;
            if max[band] <= entry.largest {
                let table = entry.table;
                let codes = *code_bits[entry.first].get_or_insert_with(|| {
                    (occurring.iter())
                        .map(|&index| {
                            let (x, y) = (usize::from(index / 16), usize::from(index % 16));
                            usize::from(pairs[usize::from(index)])
                                * table.codes[x * table.size + y].len as usize
                        })
                        .sum()
                });
                bits = codes + signs + escapes * table.linbits as usize;
            }
            below[t][band + 1] = below[t][band] + bits;
        }
        for &index in occurring {
            pairs[usize::from(index)] = 0;
        }
    }
    // The table that codes the bands `range` in the fewest bits, and those
    // bits.
    let region = |range: std::ops::Range<usize>| -> (u8, usize) {
        let largest = max[range.clone()].iter().copied().max().unwrap_or(0);
        let mut best = (31, usize::MAX);
        for entry in tables
            .entries
            .iter()
            .filter(|entry| largest <= entry.largest)
        {
            let below = &below[usize::from(entry.table_select)];
            let bits = below[range.end] - below[range.start];
            if bits < best.1 {
                best = (entry.table_select, bits);
            }
        }
        best
    };
    // The last region, for each band it may start at.
    let mut last = [(0u8, 0usize); LONG_BANDS + 1];
    for (start, last) in last.iter_mut().enumerate().take(count + 1) {
        *last = region(start..count);
    }
    let mut best: Option<(usize, (u8, u8), [u8; 3])> = None;
    // region0_count and region1_count are 4- and 3-bit fields.
    for region0 in 0..16u8 {
        let end0 = (usize::from(region0) + 1).min(count);
        let (table0, bits0) = region(0..end0);
        for region1 in 0..8u8 {
            let end1 = (end0 + usize::from(region1) + 1).min(count);
            let (table1, bits1) = region(end0..end1);
            let (table2, bits2) = last[end1];
            let bits = bits0 + bits1 + bits2;
            if best.as_ref().is_none_or(|&(least, ..)| bits < least) {
                best = Some((bits, (region0, region1), [table0, table1, table2]));
            }
            if end1 == count {
                break;
            }
        }
        if end0 == count {
            break;
        }
    }
    let (_, counts, tables) = best.expect("at least one split");
    (counts, tables)
}

/// What [`subdivide`] needs of each table that is used, worked out once.
struct PairTables {
    entries: Vec<TableEntry>,
}

struct TableEntry {
    table_select: u8,
    table: &'static PairTable,
    /// The largest magnitude it can code.
    largest: u32,
    /// The first table with the same code words.
    first: usize,
}

impl PairTables {
    fn get() -> &'static Self {
        static TABLES: std::sync::OnceLock<PairTables> = std::sync::OnceLock::new();
        TABLES.get_or_init(|| {
            let used = || (0..32u8).filter_map(|t| pair_table(t).map(|table| (t, table)));
            let entries = used()
                .map(|(table_select, table)| TableEntry {
                    table_select,
                    table,
                    largest: if table.linbits == 0 {
                        table.size as u32 - 1
                    } else {
                        15 + (1 << table.linbits) - 1
                    },
                    first: used()
                        .find(|(_, other)| std::ptr::eq(other.codes, table.codes))
                        .map_or(0, |(t, _)| usize::from(t)),
                })
                .collect();
            Self { entries }
        })
    }
}

/// Chooses a table for a region's `values`: of every table that can code
/// their largest magnitude, the one needing the fewest bits. Spec gap:
/// clause C.1.5.4.4.7 gives choosing by that magnitude as "a simple way",
/// and "additional coding gain" from "trying all" the tables of the same
/// size. Spec reading: tables of a larger size are tried too, as one can
/// need fewer bits.
fn select_table(values: &[i32]) -> u8 {
    let max = values.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
    (0..32)
        .filter_map(|t| pair_table(t).map(|table| (t, table)))
        .filter(|(_, table)| can_code(table, max))
        .min_by_key(|&(_, table)| pair_bits(table, values))
        .expect("table 31 codes every allowed magnitude")
        .0
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
