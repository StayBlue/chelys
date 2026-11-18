// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The `spectral_data()` of a section (Table 20): its lines as n-tuples of
//! the section's codebook, from the lowest line up, each codeword followed
//! by the sign bits of an unsigned codebook and, in `ESC_HCB`, by the
//! escape sequences of its lines (clause 9.3).

use chelys_core::bitstream::BitWriter;

use super::codebooks::{
    CLASSES, Codebook, ESC_FLAG, SpectrumCodebook, escape_bits, escape_sequence,
};

/// A count of bits that stands for a codebook not counted for some lines:
/// more than any coding of a window's lines takes, and small enough that
/// the counts of all its bands add up without overflow.
pub(super) const UNCODABLE: u32 = 1 << 24;

/// The bits of `lines` in each codebook of [`CLASSES`] `class..=top`, by
/// codebook number, as [`write`] writes them, and [`UNCODABLE`] for the
/// others. `class` is that of the lines' largest magnitude, the first whose
/// codebooks code them, and the lines divide into quadruples.
pub(super) fn bits_by_codebook(lines: &[i32], class: usize, top: usize) -> [u32; 12] {
    let mut bits = [UNCODABLE; 12];
    if class == 0 {
        bits[0] = 0;
    }
    for books in &CLASSES[class.max(1)..=top] {
        let a = spectrum_codebook(books, 0);
        let b = (books.len() > 1).then(|| spectrum_codebook(books, 1));
        let dimension = a.dimension();
        let (mut bits_a, mut bits_b) = (0, 0);
        if class == 0 {
            // Lines of zeros, all one tuple.
            let index = a.index(&[0; 4][..dimension]);
            let tuples = (lines.len() / dimension) as u32;
            bits_a = tuples * a.tuple_bits(index);
            bits_b = b.map_or(0, |b| tuples * b.tuple_bits(index));
        } else if let Some(b) = b {
            // The two codebooks of a class share their codeword indexes.
            for tuple in lines.chunks_exact(dimension) {
                let index = a.index(tuple);
                bits_a += a.tuple_bits(index);
                bits_b += b.tuple_bits(index);
            }
        } else {
            debug_assert!(a.has_escapes());
            for tuple in lines.chunks_exact(dimension) {
                bits_a += a.tuple_bits(a.index(tuple));
                for magnitude in tuple.iter().map(|line| line.unsigned_abs()) {
                    if magnitude >= ESC_FLAG {
                        bits_a += escape_bits(magnitude);
                    }
                }
            }
        }
        bits[usize::from(a.number())] = bits_a;
        if let Some(b) = b {
            bits[usize::from(b.number())] = bits_b;
        }
    }
    bits
}

fn spectrum_codebook(books: &[Codebook], i: usize) -> &'static SpectrumCodebook {
    books[i].spectrum().expect("a spectrum codebook")
}

/// Writes `lines` in `codebook`: for each n-tuple its codeword, then, in an
/// unsigned codebook, a sign bit for each nonzero line, '1' for a negative
/// one, then, in `ESC_HCB`, the escape sequence of each line of
/// [`ESC_FLAG`] or more, lower lines first in each.
///
/// # Panics
///
/// If a line is one the codebook cannot code, or the lines do not divide
/// into its n-tuples: a caller's error.
pub(super) fn write(codebook: &SpectrumCodebook, lines: &[i32], w: &mut BitWriter) {
    let dimension = codebook.dimension();
    assert!(
        lines.len() % dimension == 0,
        "{} lines do not divide into tuples of {dimension}",
        lines.len()
    );
    let largest = codebook.largest_magnitude();
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

    use super::super::codebooks::{MAX_MAGNITUDE, class};
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

    /// The bits of `lines` written in codebook `number`, checking that they
    /// decode as written.
    fn written_bits(number: u8, lines: &[i32]) -> u32 {
        let book = Codebook::new(number).unwrap().spectrum().unwrap();
        let mut w = BitWriter::new();
        write(book, lines, &mut w);
        let bits = w.bit_len() as u32;
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        let offsets = [0, lines.len() as u16];
        let decoded = spectral_data(&mut r, &[(number, 0, 1)], &offsets, lines.len());
        assert_eq!(decoded, lines, "codebook {number}");
        assert_eq!(r.position() as u32, bits);
        bits
    }

    #[test]
    fn lines_are_counted_in_every_codebook_that_codes_them_as_written() {
        let mut noise = Noise::new(1);
        for number in 0..=11 {
            for _ in 0..50 {
                let len = 4 * (1 + noise.below(24) as usize);
                let lines = if number == 0 {
                    vec![0; len]
                } else {
                    lines_for(&mut noise, number, len)
                };
                let largest = lines.iter().map(|line| line.unsigned_abs()).max().unwrap();
                let class = class(largest).unwrap();
                for top in class..CLASSES.len() {
                    let counted = bits_by_codebook(&lines, class, top);
                    for (other, &counted) in (0..).zip(&counted) {
                        let codebook = Codebook::new(other).unwrap();
                        let in_range = (class..=top).any(|c| CLASSES[c].contains(&codebook));
                        let expected = match codebook.spectrum() {
                            _ if !in_range => UNCODABLE,
                            None => 0,
                            Some(_) => written_bits(other, &lines),
                        };
                        assert_eq!(counted, expected, "{lines:?} in codebook {other}");
                    }
                }
            }
        }
    }

    #[test]
    fn escapes_reach_the_largest_magnitude() {
        let lines = [8191, -8191, 16, -15, 0, 31, -32, 4095];
        let book = Codebook::ESC.spectrum().unwrap();
        // The codewords, 7 sign bits, and escapes of 21, 21, 5, 5, 7 and 19
        // bits.
        let codewords: u32 = (lines.chunks(2))
            .map(|pair| book.codeword(book.index(pair)).len)
            .sum();
        let bits = codewords + 7 + 21 + 21 + 5 + 5 + 7 + 19;
        assert_eq!(written_bits(11, &lines), bits);
        assert_eq!(bits_by_codebook(&lines, 6, 6)[11], bits);
    }

    #[test]
    #[should_panic(expected = "cannot code")]
    fn a_line_beyond_the_codebook_is_refused() {
        let book = Codebook::new(5).unwrap().spectrum().unwrap();
        write(book, &[5, 0], &mut BitWriter::new());
    }

    #[test]
    #[should_panic(expected = "cannot code")]
    fn a_line_beyond_the_escapes_is_refused() {
        write(
            Codebook::ESC.spectrum().unwrap(),
            &[MAX_MAGNITUDE as i32 + 1, 0],
            &mut BitWriter::new(),
        );
    }
}
