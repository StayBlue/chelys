// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The `spectral_data()` of a section (Table 20): its lines as n-tuples of
//! the section's codebook, from the lowest line up, each codeword followed
//! by the sign bits of an unsigned codebook and, in `ESC_HCB`, by the
//! escape sequences of its lines (clause 9.3).

use chelys_core::bitstream::BitWriter;

use super::codebooks::{
    CLASSES, Codebook, ESC_FLAG, PAIRED_BITS_1, PAIRED_BITS_2, PAIRED_BITS_3, PAIRED_BITS_4,
    PAIRED_BITS_5, SpectrumCodebook, escape_bits, escape_sequence,
};

/// A count of bits that stands for a codebook not counted for some lines:
/// more than any coding of a window's lines takes, and small enough that
/// the counts of all its bands add up without overflow.
pub(super) const UNCODABLE: u32 = 1 << 24;

/// The bits of `lines` in each codebook of [`CLASSES`] `class..=top`, by
/// codebook number, as [`write`] writes them, and [`UNCODABLE`] for the
/// others, and the least of them. `class` is that of the lines' largest
/// magnitude, the first whose codebooks code them, and the lines divide
/// into quadruples.
pub(super) fn bits_by_codebook(lines: &[i32], class: usize, top: usize) -> ([u32; 12], u32) {
    let mut bits = [UNCODABLE; 12];
    // The least of those counted, as the others are UNCODABLE.
    let mut least = UNCODABLE;
    if class == 0 {
        (bits[0], least) = (0, 0);
    }
    let zeros = class == 0;
    let classes = CLASSES.iter().enumerate().take(top + 1);
    for (class, books) in classes.skip(class.max(1)) {
        // The two codebooks of a class share their codeword indexes, and
        // are counted together (see `paired_bits`).
        let paired = match class {
            1 => paired::<4, true, 1>(lines, &PAIRED_BITS_1, zeros),
            2 => paired::<4, false, 2>(lines, &PAIRED_BITS_2, zeros),
            3 => paired::<2, true, 4>(lines, &PAIRED_BITS_3, zeros),
            4 => paired::<2, false, 7>(lines, &PAIRED_BITS_4, zeros),
            5 => paired::<2, false, 12>(lines, &PAIRED_BITS_5, zeros),
            _ => {
                let escaped = escaped(lines, zeros);
                bits[usize::from(Codebook::ESC.number())] = escaped;
                least = least.min(escaped);
                continue;
            }
        };
        let [a, b] = books else {
            unreachable!("a class of two codebooks")
        };
        let (a_bits, b_bits) = (paired & 0xffff, paired >> 16);
        bits[usize::from(a.number())] = a_bits;
        bits[usize::from(b.number())] = b_bits;
        least = least.min(a_bits).min(b_bits);
    }
    (bits, least)
}

/// The codeword index of `tuple` in a codebook of `DIMENSION`, signed or
/// not, and `LAV` (see [`SpectrumCodebook::index`]), for lines it codes.
fn index<const DIMENSION: usize, const SIGNED: bool, const LAV: u32>(tuple: &[i32]) -> usize {
    let modulus = if SIGNED { 2 * LAV + 1 } else { LAV + 1 };
    tuple.iter().fold(0, |index, &line| {
        let digit = if SIGNED {
            line.wrapping_add_unsigned(LAV) as u32
        } else {
            line.unsigned_abs()
        };
        index * modulus as usize + digit as usize
    })
}

/// The bits of `lines` in the two codebooks of a class, by `bits` (see
/// `paired_bits`), the first's in the low 16 bits and the second's in the
/// high 16; all tuples of zeros if `zeros`.
fn paired<const DIMENSION: usize, const SIGNED: bool, const LAV: u32>(
    lines: &[i32],
    bits: &[u32],
    zeros: bool,
) -> u32 {
    if zeros {
        let tuples = (lines.len() / DIMENSION) as u32;
        return tuples * bits[index::<DIMENSION, SIGNED, LAV>(&[0; DIMENSION])];
    }
    (lines.chunks_exact(DIMENSION))
        .map(|tuple| bits[index::<DIMENSION, SIGNED, LAV>(tuple)])
        .sum()
}

/// The bits of `lines` in `ESC_HCB`, with their escape sequences; all
/// pairs of zeros if `zeros`.
fn escaped(lines: &[i32], zeros: bool) -> u32 {
    let book = Codebook::ESC.spectrum().expect("a spectrum codebook");
    if zeros {
        return (lines.len() / 2) as u32 * book.tuple_bits(0);
    }
    let mut bits = 0;
    for pair in lines.chunks_exact(2) {
        let (a, b) = (pair[0].unsigned_abs(), pair[1].unsigned_abs());
        let lav = ESC_FLAG as usize;
        let index = (lav + 1) * (a as usize).min(lav) + (b as usize).min(lav);
        bits += book.tuple_bits(index);
        for magnitude in [a, b] {
            if magnitude >= ESC_FLAG {
                bits += escape_bits(magnitude);
            }
        }
    }
    bits
}

/// The fewest bits that `lines`, a scalefactor band's quantized lines, take
/// in `spectral_data()` in any codebook that codes them, or `None` if a
/// line's magnitude exceeds what any codes.
pub fn least_bits(lines: &[i32]) -> Option<u32> {
    let largest = lines
        .iter()
        .map(|line| line.unsigned_abs())
        .max()
        .unwrap_or(0);
    let class = super::codebooks::class(largest)?;
    Some(bits_by_codebook(lines, class, class).1)
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
                    let (counted, _) = bits_by_codebook(&lines, class, top);
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
        assert_eq!(bits_by_codebook(&lines, 6, 6).0[11], bits);
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
