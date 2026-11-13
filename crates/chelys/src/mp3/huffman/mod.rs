// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Layer III Huffman coding: the code tables (Table B.7) and their use by
//! `Huffmancodebits()` (clauses 2.4.1.7 and 2.4.2.7).

mod spectrum;
mod tables;

pub use spectrum::{MAX_MAGNITUDE, SpectrumCoding};

/// A Huffman code word: the low `len` bits of `bits`, written most significant
/// first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Codeword {
    pub bits: u32,
    pub len: u32,
}

/// A table for the `big_values` region, coding pairs of magnitudes.
#[derive(Debug)]
pub struct PairTable {
    /// The width of the escape fields `linbitsx`/`linbitsy` for magnitudes of
    /// 15 or more; zero for tables that cannot code them.
    pub linbits: u32,
    /// Magnitudes `x` and `y` range over `0..size`.
    pub size: usize,
    codes: &'static [Codeword],
}

impl PairTable {
    /// The code word for the magnitudes `x` and `y`, with any magnitude of 15 or
    /// more already reduced to 15 for an escape table.
    pub fn codeword(&self, x: usize, y: usize) -> Codeword {
        assert!(
            x < self.size && y < self.size,
            "({x}, {y}) is outside a {0}x{0} table",
            self.size
        );
        self.codes[x * self.size + y]
    }
}

/// The table selected by `table_select`, or `None` for tables 4 and 14, which
/// are marked "not used".
pub fn pair_table(table_select: u8) -> Option<&'static PairTable> {
    PAIR_TABLES.get(usize::from(table_select))?.as_ref()
}

/// The two tables for the `count1` region, coding quadruples of magnitudes up to
/// 1, chosen by `count1table_select`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuadTable {
    A,
    B,
}

impl QuadTable {
    /// The code word for the magnitudes `v`, `w`, `x` and `y`, packed as the
    /// bits `vwxy`.
    pub fn codeword(self, vwxy: u8) -> Codeword {
        let codes = match self {
            Self::A => &tables::QUAD_A,
            Self::B => &tables::QUAD_B,
        };
        codes[usize::from(vwxy)]
    }
}

const fn pair(linbits: u32, size: usize, codes: &'static [Codeword]) -> Option<PairTable> {
    Some(PairTable {
        linbits,
        size,
        codes,
    })
}

static PAIR_TABLES: [Option<PairTable>; 32] = {
    use tables::*;
    [
        pair(0, 1, &TABLE_0),
        pair(0, 2, &TABLE_1),
        pair(0, 3, &TABLE_2),
        pair(0, 3, &TABLE_3),
        None,
        pair(0, 4, &TABLE_5),
        pair(0, 4, &TABLE_6),
        pair(0, 6, &TABLE_7),
        pair(0, 6, &TABLE_8),
        pair(0, 6, &TABLE_9),
        pair(0, 8, &TABLE_10),
        pair(0, 8, &TABLE_11),
        pair(0, 8, &TABLE_12),
        pair(0, 16, &TABLE_13),
        None,
        pair(0, 16, &TABLE_15),
        pair(1, 16, &TABLE_16),
        pair(2, 16, &TABLE_16),
        pair(3, 16, &TABLE_16),
        pair(4, 16, &TABLE_16),
        pair(6, 16, &TABLE_16),
        pair(8, 16, &TABLE_16),
        pair(10, 16, &TABLE_16),
        pair(13, 16, &TABLE_16),
        pair(4, 16, &TABLE_24),
        pair(5, 16, &TABLE_24),
        pair(6, 16, &TABLE_24),
        pair(7, 16, &TABLE_24),
        pair(8, 16, &TABLE_24),
        pair(9, 16, &TABLE_24),
        pair(11, 16, &TABLE_24),
        pair(13, 16, &TABLE_24),
    ]
};

/// Builds a pair table from `(x, y, hlen, hcod)` rows, which must run over `x`
/// and then `y` in order.
const fn pair_codes<const N: usize>(rows: [(usize, usize, u32, &str); N]) -> [Codeword; N] {
    let size = N.isqrt();
    assert!(size * size == N, "pair tables are square");
    let mut codes = [Codeword { bits: 0, len: 0 }; N];
    let mut i = 0;
    while i < N {
        let (x, y, hlen, hcod) = rows[i];
        assert!(x == i / size && y == i % size, "row out of order");
        codes[i] = codeword(hlen, hcod);
        i += 1;
    }
    codes
}

/// Builds a quadruple table from `(vwxy, hlen, hcod)` rows in `vwxy` order.
const fn quad_codes(rows: [(&str, u32, &str); 16]) -> [Codeword; 16] {
    let mut codes = [Codeword { bits: 0, len: 0 }; 16];
    let mut i = 0;
    while i < 16 {
        let (vwxy, hlen, hcod) = rows[i];
        assert!(binary(vwxy).bits == i as u32, "row out of order");
        codes[i] = codeword(hlen, hcod);
        i += 1;
    }
    codes
}

const fn codeword(hlen: u32, hcod: &str) -> Codeword {
    let code = binary(hcod);
    assert!(code.len == hlen, "hlen does not match the length of hcod");
    code
}

const fn binary(digits: &str) -> Codeword {
    let digits = digits.as_bytes();
    let mut bits = 0;
    let mut i = 0;
    while i < digits.len() {
        bits = bits << 1
            | match digits[i] {
                b'0' => 0,
                b'1' => 1,
                _ => panic!("not a binary digit"),
            };
        i += 1;
    }
    Codeword {
        bits,
        len: digits.len() as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Checks that `codes` form a complete prefix code: no code word begins
    /// another, and the Kraft sum of `2^-len` is exactly 1, so every bit
    /// string decodes. Either property fails if any single code word is
    /// transcribed wrongly.
    fn assert_complete_prefix_code(codes: &[Codeword], name: &str) {
        const MAX_LEN: u32 = 32;
        let kraft: u64 = codes.iter().map(|c| 1u64 << (MAX_LEN - c.len)).sum();
        assert_eq!(kraft, 1 << MAX_LEN, "{name}: Kraft sum is not 1");
        for (i, a) in codes.iter().enumerate() {
            for b in &codes[i + 1..] {
                let (short, long) = if a.len <= b.len { (a, b) } else { (b, a) };
                assert_ne!(
                    long.bits >> (long.len - short.len),
                    short.bits,
                    "{name}: {short:?} is a prefix of {long:?}"
                );
            }
        }
    }

    #[test]
    fn every_table_is_a_complete_prefix_code() {
        for table_select in 1..32 {
            if let Some(table) = pair_table(table_select) {
                assert_complete_prefix_code(table.codes, &format!("table {table_select}"));
            }
        }
        for table in [QuadTable::A, QuadTable::B] {
            let codes: Vec<_> = (0..16).map(|vwxy| table.codeword(vwxy)).collect();
            assert_complete_prefix_code(&codes, &format!("quadruple table {table:?}"));
        }
    }

    /// Clause 2.4.1.7 bounds `hcod` at 19 bits for pairs and 6 for quadruples.
    #[test]
    fn code_lengths_respect_the_syntax() {
        for table_select in 0..32 {
            if let Some(table) = pair_table(table_select) {
                assert!(table.codes.iter().all(|c| c.len <= 19));
            }
        }
        for table in [QuadTable::A, QuadTable::B] {
            assert!((0..16).all(|vwxy| (1..=6).contains(&table.codeword(vwxy).len)));
        }
    }

    #[test]
    fn table_0_codes_only_zeros_in_zero_bits() {
        let table = pair_table(0).unwrap();
        assert_eq!(table.size, 1);
        assert_eq!(table.codeword(0, 0), Codeword { bits: 0, len: 0 });
    }

    #[test]
    fn unused_and_out_of_range_tables_are_absent() {
        assert!(pair_table(4).is_none());
        assert!(pair_table(14).is_none());
        assert!(pair_table(32).is_none());
    }

    #[test]
    fn only_tables_from_16_up_have_escapes() {
        for table_select in 0..32 {
            if let Some(table) = pair_table(table_select) {
                assert_eq!(
                    table.linbits > 0,
                    table_select >= 16,
                    "table {table_select}"
                );
            }
        }
    }

    #[test]
    fn looks_up_rows_as_printed() {
        // Table 7, row "1 3 7 0001010".
        assert_eq!(
            pair_table(7).unwrap().codeword(1, 3),
            Codeword {
                bits: 0b0001010,
                len: 7
            }
        );
        // Table 24 shares its codes with tables 25 to 31; row "15 15 4 0011".
        assert_eq!(
            pair_table(31).unwrap().codeword(15, 15),
            Codeword {
                bits: 0b0011,
                len: 4
            }
        );
        // Quadruple table B, row "0110 4 1001".
        assert_eq!(
            QuadTable::B.codeword(0b0110),
            Codeword {
                bits: 0b1001,
                len: 4
            }
        );
    }
}
