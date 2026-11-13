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
use crate::mp3::scalefactor_bands::Band;
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
    pub count1_table: QuadTable,
}

impl SpectrumCoding {
    /// Partitions `values` of a granule of `block_type` and chooses tables
    /// following clause C.1.5.4.4.
    ///
    /// # Panics
    ///
    /// If a magnitude exceeds [`MAX_MAGNITUDE`].
    pub fn choose(values: &[i32; GRANULE_LINES], block_type: BlockType, bands: &[Band]) -> Self {
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
        let mut count1 = 0;
        while end >= 4 && values[end - 4..end].iter().all(|v| v.unsigned_abs() <= 1) {
            end -= 4;
            count1 += 1;
        }
        let big_values = end / 2;

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
        // 2.4.2.7).
        let (region0_count, region1_count) = match block_type {
            BlockType::Normal => subdivide(big_values, bands),
            BlockType::Start | BlockType::Stop => (7, 36),
            BlockType::Short => (8, 36),
        };
        let mut coding = Self {
            big_values,
            count1,
            table_select: [0; 3],
            region0_count,
            region1_count,
            count1_table,
        };
        let mut start = 0;
        for (region, end) in coding.region_ends(bands).into_iter().enumerate() {
            coding.table_select[region] = select_table(&values[start..end]);
            start = end;
        }
        coding
    }

    /// The line just past each of the three `big_values` regions. Region 0
    /// holds `region0_count + 1` scalefactor bands and region 1 the next
    /// `region1_count + 1` (clause 2.4.2.7); all three end at `big_values`
    /// pairs at the latest.
    pub fn region_ends(&self, bands: &[Band]) -> [usize; 3] {
        let lines = 2 * self.big_values;
        let region0_bands = usize::from(self.region0_count) + 1;
        let region1_bands = usize::from(self.region1_count) + 1;
        let boundary = |band: usize| bands.get(band).map_or(lines, |b| b.start.min(lines));
        [
            boundary(region0_bands),
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

/// SUBDIVIDE (clause C.1.5.4.4.6). Spec gap: the clause leaves the split to
/// the implementation; this is the simple strategy it suggests: a third of the
/// scalefactor bands spanned by `big_values` go to region 0 and a quarter to
/// region 2, the last, partly covered band counting as a whole one. Returns
/// `(region0_count, region1_count)`.
fn subdivide(big_values: usize, bands: &[Band]) -> (u8, u8) {
    let lines = 2 * big_values;
    if lines == 0 {
        return (0, 0);
    }
    let spanned = bands.iter().take_while(|b| b.start < lines).count();
    // region0_count and region1_count are 4- and 3-bit fields.
    let region0 = (spanned / 3).clamp(1, 16);
    let region2 = spanned / 4;
    let region1 = spanned.saturating_sub(region0 + region2).clamp(1, 8);
    ((region0 - 1) as u8, (region1 - 1) as u8)
}

/// Chooses a table for a region's `values`. Spec gap: clause C.1.5.4.4.7
/// leaves the choice open; this is what it suggests: the smallest table size
/// that can code their maximum magnitude, and among the tables of that size
/// the one needing the fewest bits.
fn select_table(values: &[i32]) -> u8 {
    let max = values.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
    let candidates =
        || (0..32).filter(|&t| pair_table(t).is_some_and(|table| can_code(table, max)));
    let size = candidates()
        .map(|t| pair_table(t).unwrap().size)
        .min()
        .expect("table 31 codes every allowed magnitude");
    candidates()
        .filter(|&t| pair_table(t).unwrap().size == size)
        .min_by_key(|&t| pair_bits(pair_table(t).unwrap(), values))
        .unwrap()
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

    fn bands() -> &'static [Band] {
        long_bands(SamplingFrequency::Hz44100)
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
        let coding = SpectrumCoding::choose(&values, BlockType::Normal, bands());
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
        let coding = SpectrumCoding::choose(&values, BlockType::Normal, bands());
        assert_eq!(coding.big_values, 2);
        assert_eq!(coding.count1, 2);
    }

    #[test]
    fn zeros_inside_the_top_nonzero_pair_stay_in_count1() {
        // Zero pairs are counted whole: line 9 is zero but line 8 is not.
        let values = spectrum(&[2, 0, 0, 0, 0, 0, 0, 0, 1, 0]);
        let coding = SpectrumCoding::choose(&values, BlockType::Normal, bands());
        assert_eq!((coding.big_values, coding.count1), (1, 2));
    }

    #[test]
    fn subdivides_by_thirds_and_quarters_of_the_spanned_bands() {
        // Lines 0..200 reach into band 17 (196..238) at 44.1 kHz.
        let spanned = bands().iter().take_while(|b| b.start < 200).count();
        assert_eq!(spanned, 18);
        // 18 / 3 = 6 bands in region 0, 18 / 4 = 4 in region 2, 8 in region 1.
        assert_eq!(subdivide(100, bands()), (5, 7));
        // One band: region 0 takes it, region 1 is empty but must still be one
        // band long in the syntax.
        assert_eq!(subdivide(1, bands()), (0, 0));
        // All 21 bands: 7 / 9 / 5, with region 1 clamped to its 3-bit maximum.
        assert_eq!(subdivide(288, bands()), (6, 7));
    }

    #[test]
    fn region_ends_stop_at_big_values() {
        let values = spectrum(&[3; 40]);
        let coding = SpectrumCoding::choose(&values, BlockType::Normal, bands());
        assert_eq!(coding.big_values, 20);
        // 40 lines span bands 0 to 8 (band 9 starts at 44): 3 / 4 / 2 bands.
        assert_eq!((coding.region0_count, coding.region1_count), (2, 3));
        assert_eq!(coding.region_ends(bands()), [12, 30, 40]);
    }

    /// With window switching the regions are fixed: region 0 ends after 8
    /// long bands, or for short blocks after short band 2 in all three
    /// windows, and region 1 takes the rest of `big_values`.
    #[test]
    fn switched_windows_use_the_default_regions() {
        let fs = SamplingFrequency::Hz44100;
        let values = spectrum(&[3; 200]);
        for block_type in [BlockType::Start, BlockType::Stop] {
            let coding = SpectrumCoding::choose(&values, block_type, long_bands(fs));
            assert_eq!(
                coding.region_ends(long_bands(fs)),
                [long_bands(fs)[8].start, 200, 200]
            );
            assert_eq!(coding.table_select[2], 0);
        }
        let short = coding_bands(BlockType::Short, fs);
        let coding = SpectrumCoding::choose(&values, BlockType::Short, &short);
        assert_eq!(coding.region_ends(&short), [36, 200, 200]);
    }

    #[test]
    fn selects_the_smallest_table_size_that_fits() {
        assert_eq!(select_table(&[0, 0, 0, 0]), 0);
        assert_eq!(select_table(&[1, 0, 0, -1]), 1);
        // Size 3 holds tables 2 and 3. Size 16 holds 13, 15 and the escape
        // tables, all of which can code 15.
        assert!([2, 3].contains(&select_table(&[2, 0, 1, 1])));
        assert_eq!(pair_table(select_table(&[15, 0, 1, 1])).unwrap().size, 16);
        // 16 needs an escape table with at least 1 linbit; 8191 needs 13.
        assert!(select_table(&[16, 0]) >= 16);
        assert!([23, 31].contains(&select_table(&[8191, 0])));
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
        let coding = SpectrumCoding::choose(&values, BlockType::Normal, bands());
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
        let coding = SpectrumCoding::choose(&values, BlockType::Normal, bands());
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
