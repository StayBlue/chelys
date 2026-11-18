// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The `spectral_data()` of a section (Table 20): its lines as n-tuples of
//! the section's codebook, from the lowest line up, each codeword followed
//! by the sign bits of an unsigned codebook and, in `ESC_HCB`, by the
//! escape sequences of its lines (clause 9.3).

use chelys_core::bitstream::BitWriter;

use super::codebooks::{Codebook, ESC_FLAG, SpectrumCodebook, escape_bits, escape_sequence};

/// The bits of `lines` coded with `codebook`, as [`write()`] writes them.
///
/// The lines are ones the codebook can code (see
/// [`Codebook::largest_magnitude`]), and divide into its n-tuples.
pub fn bits(codebook: &SpectrumCodebook, lines: &[i32]) -> u32 {
    let mut bits = 0;
    for tuple in lines.chunks_exact(codebook.dimension()) {
        bits += codebook.tuple_bits(codebook.index(tuple));
    }
    if codebook.has_escapes() {
        bits += escapes_bits(lines);
    }
    bits
}

/// The bits of the escape sequences of `lines` in `ESC_HCB`.
fn escapes_bits(lines: &[i32]) -> u32 {
    (lines.iter())
        .map(|line| line.unsigned_abs())
        .filter(|&magnitude| magnitude >= ESC_FLAG)
        .map(escape_bits)
        .sum()
}

/// A count of bits that stands for a codebook unable to code some lines:
/// more than any coding of a window's lines takes, and small enough that
/// the counts of all its bands add up without overflow.
pub(super) const UNCODABLE: u32 = 1 << 24;

/// The bits of `lines` in each codebook, by codebook number, as [`bits`]
/// counts them, with [`UNCODABLE`] for the codebooks that cannot code
/// them: `ZERO_HCB` unless they are all zero, and those whose largest
/// magnitude is below `largest`, the largest of the lines.
///
/// `largest` is at most [`MAX_MAGNITUDE`](super::codebooks::MAX_MAGNITUDE),
/// and the lines divide into quadruples.
pub(super) fn bits_by_codebook(lines: &[i32], largest: u32) -> [u32; 12] {
    debug_assert_eq!(
        largest,
        lines
            .iter()
            .map(|line| line.unsigned_abs())
            .max()
            .unwrap_or(0)
    );
    let mut bits = [UNCODABLE; 12];
    if largest == 0 {
        // Every codebook codes nothing but its tuple of zeros.
        bits[0] = 0;
        for (number, bits) in (1..).zip(&mut bits[1..]) {
            let book = spectrum_codebook(number);
            let tuples = lines.len() / book.dimension();
            *bits = tuples as u32 * book.tuple_bits(book.index(&[0; 4][..book.dimension()]));
        }
        return bits;
    }
    // Codebooks 1 and 2, 3 and 4 and so on to 9 and 10 share their
    // dimension, signedness and LAV (Table 59), and so their codeword
    // indexes, which are worked out once for both.
    for first in [1, 3, 5, 7, 9] {
        let (a, b) = (spectrum_codebook(first), spectrum_codebook(first + 1));
        if a.lav() < largest {
            continue;
        }
        let (mut bits_a, mut bits_b) = (0, 0);
        for tuple in lines.chunks_exact(a.dimension()) {
            let index = a.index(tuple);
            bits_a += a.tuple_bits(index);
            bits_b += b.tuple_bits(index);
        }
        bits[usize::from(first)] = bits_a;
        bits[usize::from(first) + 1] = bits_b;
    }
    bits[usize::from(Codebook::ESC.number())] = self::bits(spectrum_codebook(11), lines);
    bits
}

fn spectrum_codebook(number: u8) -> &'static SpectrumCodebook {
    (Codebook::new(number).and_then(Codebook::spectrum)).expect("a spectrum codebook")
}

/// Writes `lines` in `codebook`: for each n-tuple its codeword, then, in an
/// unsigned codebook, a sign bit for each nonzero line, '1' for a negative
/// one, then, in `ESC_HCB`, the escape sequence of each line of
/// [`ESC_FLAG`] or more, lower lines first in each.
///
/// # Panics
///
/// If a line is one the codebook cannot code, or the lines do not divide
/// into its n-tuples: a caller's error, which [`bits`] does not check.
pub fn write(codebook: &SpectrumCodebook, lines: &[i32], w: &mut BitWriter) {
    let dimension = codebook.dimension();
    assert!(
        lines.len() % dimension == 0,
        "{} lines do not divide into tuples of {dimension}",
        lines.len()
    );
    let largest = Codebook::new(codebook.number())
        .expect("a codebook")
        .largest_magnitude();
    for tuple in lines.chunks_exact(dimension) {
        assert!(
            tuple.iter().all(|line| line.unsigned_abs() <= largest),
            "codebook {} cannot code {tuple:?}",
            codebook.number()
        );
        let codeword = codebook.codeword(codebook.index(tuple));
        w.write(codeword.bits, codeword.len);
        if codebook.is_unsigned() {
            for &line in tuple.iter().filter(|&&line| line != 0) {
                w.write_bool(line < 0);
            }
        }
        if codebook.has_escapes() {
            for line in tuple {
                let magnitude = line.unsigned_abs();
                if magnitude >= ESC_FLAG {
                    let escape = escape_sequence(magnitude);
                    w.write(escape.bits, escape.len);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use chelys_core::bitstream::BitWriter;

    use super::super::codebooks::{Codebook, MAX_MAGNITUDE};
    use super::super::decoder::{BitReader, Noise, spectral_data};
    use super::*;

    /// Lines that codebook `number` codes, drawn from its whole range.
    fn lines_for(noise: &mut Noise, number: u8, len: usize) -> Vec<i32> {
        let largest = Codebook::new(number).unwrap().largest_magnitude();
        (0..len)
            .map(|_| {
                // In ESC_HCB, mostly lines below the escapes.
                let reach = if largest == MAX_MAGNITUDE && noise.below(4) > 0 {
                    17
                } else {
                    largest
                };
                let magnitude = noise.below(reach + 1) as i32;
                if noise.below(2) == 0 {
                    magnitude
                } else {
                    -magnitude
                }
            })
            .collect()
    }

    #[test]
    fn lines_decode_as_written_in_as_many_bits_as_counted() {
        let mut noise = Noise::new(1);
        for number in 1..=11 {
            let book = spectrum_codebook(number);
            for _ in 0..50 {
                let len = 4 * (1 + noise.below(24) as usize);
                let lines = lines_for(&mut noise, number, len);
                let mut w = BitWriter::new();
                write(book, &lines, &mut w);
                assert_eq!(w.bit_len() as u32, bits(book, &lines), "codebook {number}");
                let bytes = w.into_bytes();
                let mut r = BitReader::new(&bytes);
                let offsets = [0, lines.len() as u16];
                let decoded = spectral_data(&mut r, &[(number, 0, 1)], &offsets, lines.len());
                assert_eq!(decoded, lines, "codebook {number}");
            }
        }
    }

    #[test]
    fn escapes_reach_the_largest_magnitude() {
        let book = spectrum_codebook(11);
        let lines = [8191, -8191, 16, -15, 0, 31, -32, 4095];
        let mut w = BitWriter::new();
        write(book, &lines, &mut w);
        // The codewords, 7 sign bits, and escapes of 21, 21, 5, 5, 7 and 19
        // bits.
        let codewords: u32 = (lines.chunks(2))
            .map(|pair| book.codeword(book.index(pair)).len)
            .sum();
        assert_eq!(w.bit_len() as u32, codewords + 7 + 21 + 21 + 5 + 5 + 7 + 19);
        let bytes = w.into_bytes();
        let decoded = spectral_data(&mut BitReader::new(&bytes), &[(11, 0, 1)], &[0, 8], 8);
        assert_eq!(decoded, lines);
    }

    #[test]
    fn each_codebook_is_counted_where_it_codes_the_lines() {
        let mut noise = Noise::new(2);
        for number in 0..=11 {
            for _ in 0..20 {
                let lines = if number == 0 {
                    vec![0; 16]
                } else {
                    lines_for(&mut noise, number, 16)
                };
                let largest = lines.iter().map(|line| line.unsigned_abs()).max().unwrap();
                let counted = bits_by_codebook(&lines, largest);
                for (other, &counted) in (0..).zip(&counted) {
                    let codebook = Codebook::new(other).unwrap();
                    let expected = match codebook.spectrum() {
                        _ if codebook.largest_magnitude() < largest => UNCODABLE,
                        None => 0,
                        Some(book) => bits(book, &lines),
                    };
                    assert_eq!(counted, expected, "{lines:?} in codebook {other}");
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "cannot code")]
    fn a_line_beyond_the_codebook_is_refused() {
        write(spectrum_codebook(5), &[5, 0], &mut BitWriter::new());
    }
}
