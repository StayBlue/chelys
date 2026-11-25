// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The model's partitions of the lines of a long and of a short block at
//! any sample rate, from the threshold calculation partitions of ISO/IEC
//! 13818-7 (Tables C.1 to C.24), which are printed for twelve sampling
//! frequencies.
//!
//! Each line of the FFT of 2048 samples that the tables partition lies at
//! the same frequency as the line of a long Vorbis block of the same rate,
//! and each of the FFT of 256 samples as that of a short block, so at those
//! twelve rates the tables are taken as printed. At any other
//! rate, the table of the rate nearest it, in ratio, is taken by
//! frequency: each partition keeps its frequencies, its `bval(b)` and the
//! threshold in quiet of each of its lines, its edges moved to the nearest
//! line of the block. A partition left without lines is dropped. Above the
//! highest frequency of a lower rate's table, partitions as wide in lines
//! as its last, or as the last row of the table where that is wider, are
//! added, with its `bval(b)` and threshold, up to the block's last line.

use std::ops::Range;

use chelys_core::psychoacoustic::partitions::{self, Row, SAMPLE_RATES};

pub use partitions::Fft;

/// A partition of the lines of a block.
#[derive(Clone, Debug, PartialEq)]
pub struct Partition {
    pub lines: Range<usize>,
    /// Its median Bark value.
    pub bval: f64,
    /// The threshold in quiet of each of its lines, in dB, relative to the
    /// level of a sine wave of +-1/2 LSB of 16-bit PCM in the line it is
    /// centered on: the table's `qsthr(b)`, the level of a line times the
    /// partition's lines, less those lines in dB.
    pub quiet_db: f64,
}

/// The partitions of the lines of `fft`'s block, a long block's 1024 or a
/// short block's 128, at `sample_rate` Hz.
pub fn partitions(sample_rate: u32, fft: Fft) -> Vec<Partition> {
    let lines = fft.lines();
    let rate = nearest(sample_rate);
    let table = partitions::table(rate, fft).expect("a table for each of SAMPLE_RATES");
    let ratio = f64::from(rate) / f64::from(sample_rate);
    let line = |w: usize| ((w as f64 * ratio).round() as usize).min(lines);
    let partition = |lines: Range<usize>, row: &Row| Partition {
        lines,
        bval: row.bval,
        quiet_db: row.qsthr - 10.0 * (row.lines().len() as f64).log10(),
    };
    let mut out: Vec<Partition> = Vec::with_capacity(table.len());
    let mut start = 0;
    for row in table {
        let end = line(row.lines().end);
        if end > start {
            out.push(partition(start..end, row));
            start = end;
        }
    }
    if out.is_empty() {
        // The whole table lies below the first line's edge.
        out.push(partition(0..1, &table[0]));
        start = 1;
    }
    let last = out.last().expect("a partition").clone();
    let printed = table.last().expect("a table has rows").lines().len();
    let width = last.lines.len().max(printed);
    while start < lines {
        let end = (start + width).min(lines);
        out.push(Partition {
            lines: start..end,
            ..last.clone()
        });
        start = end;
    }
    out
}

/// The rate of [`SAMPLE_RATES`] nearest `sample_rate` in ratio.
fn nearest(sample_rate: u32) -> u32 {
    let distance = |rate: u32| (f64::from(rate) / f64::from(sample_rate)).ln().abs();
    SAMPLE_RATES
        .into_iter()
        .min_by(|&a, &b| distance(a).total_cmp(&distance(b)))
        .expect("tables are printed")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_printed_rates_take_their_tables_as_printed() {
        for rate in SAMPLE_RATES {
            for fft in [Fft::Long, Fft::Short] {
                let table = partitions::table(rate, fft).unwrap();
                let ours = partitions(rate, fft);
                assert_eq!(ours.len(), table.len(), "{rate}");
                for (p, row) in ours.iter().zip(table) {
                    assert_eq!(p.lines, row.lines(), "{rate}");
                    assert_eq!(p.bval, row.bval, "{rate}");
                }
            }
        }
    }

    #[test]
    fn partitions_cover_the_lines_in_order_at_any_rate() {
        for rate in [
            1,
            1_000,
            7_000,
            9_000,
            40_000,
            46_000,
            50_000,
            100_000,
            192_000,
            384_000,
            u32::MAX,
        ] {
            for fft in [Fft::Long, Fft::Short] {
                let ours = partitions(rate, fft);
                assert_eq!(ours[0].lines.start, 0, "{rate}");
                assert_eq!(ours.last().unwrap().lines.end, fft.lines(), "{rate}");
                for pair in ours.windows(2) {
                    assert_eq!(pair[1].lines.start, pair[0].lines.end, "{rate}");
                    assert!(pair[1].bval >= pair[0].bval, "{rate}");
                }
                assert!(ours.iter().all(|p| !p.lines.is_empty()), "{rate}");
            }
        }
    }

    #[test]
    fn a_rate_takes_the_table_nearest_it_by_frequency() {
        assert_eq!(nearest(47_000), 48_000);
        assert_eq!(nearest(40_000), 44_100);
        assert_eq!(nearest(4_000), 8_000);
        assert_eq!(nearest(192_000), 96_000);
        // At 96 kHz, line 300 lies at 14.1 kHz; at 192 kHz, line 150 does.
        let at = |rate, w| {
            partitions(rate, Fft::Long)
                .into_iter()
                .find(|p| p.lines.contains(&w))
                .unwrap()
                .bval
        };
        assert_eq!(at(192_000, 150), at(96_000, 300));
        // Above 48 kHz, the highest partition of 96 kHz carries on.
        assert_eq!(at(192_000, 1000), at(96_000, 1023));
    }
}
