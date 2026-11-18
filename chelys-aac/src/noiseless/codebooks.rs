// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The Huffman codebooks of Annex A with the parameters of Table 59: the
//! codebook a section names, and how a spectrum codebook codes n-tuples of
//! quantized lines (clause 9.3).

use super::tables;

/// A Huffman codeword: the low `len` bits of `bits`, sent most significant
/// first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Codeword {
    pub bits: u32,
    pub len: u32,
}

/// The largest magnitude of a quantized line (clause 10.3), which the
/// longest escape sequence of `ESC_HCB` reaches.
pub const MAX_MAGNITUDE: u32 = 8191;

/// The magnitude from which `ESC_HCB` codes a line with an escape sequence,
/// `ESC_FLAG` (clause 9.2.2).
pub(super) const ESC_FLAG: u32 = 16;

/// A section's codebook, `sect_cb` (Table 17): `ZERO_HCB`, for a section
/// whose lines are all zero and send nothing; one of the spectrum Huffman
/// codebooks 1 to `ESC_HCB` (Table 59); or, in the right channel of a pair
/// that shares its window, one of the intensity codebooks, which code no
/// spectrum but the band's intensity stereo position (clause 12.2.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Codebook(u8);

impl Codebook {
    /// `ZERO_HCB`.
    pub const ZERO: Self = Self(0);
    /// `ESC_HCB`.
    pub const ESC: Self = Self(11);
    /// `INTENSITY_HCB2`: intensity stereo out of phase.
    pub const INTENSITY_OUT_OF_PHASE: Self = Self(14);
    /// `INTENSITY_HCB`: intensity stereo in phase.
    pub const INTENSITY: Self = Self(15);

    /// The codebook numbered `number`, if it is `ZERO_HCB` or a spectrum
    /// codebook.
    #[cfg(any(test, feature = "unstable"))]
    pub fn new(number: u8) -> Option<Self> {
        (number <= Self::ESC.0).then_some(Self(number))
    }

    /// The number sent as `sect_cb`.
    pub fn number(self) -> u8 {
        self.0
    }

    /// Whether the bands of a section with this codebook send a scalefactor
    /// (Table 18): those of a spectrum codebook.
    pub fn sends_scalefactor(self) -> bool {
        self != Self::ZERO && !self.is_intensity()
    }

    /// Whether this is an intensity codebook, whose bands send an intensity
    /// stereo position in place of a scalefactor (Table 18).
    pub fn is_intensity(self) -> bool {
        self == Self::INTENSITY || self == Self::INTENSITY_OUT_OF_PHASE
    }

    /// The largest magnitude of a line the codebook codes: zero for
    /// `ZERO_HCB`.
    pub fn largest_magnitude(self) -> u32 {
        self.spectrum()
            .map_or(0, SpectrumCodebook::largest_magnitude)
    }

    /// The spectrum Huffman codebook, or `None` for `ZERO_HCB` and the
    /// intensity codebooks.
    pub(super) fn spectrum(self) -> Option<&'static SpectrumCodebook> {
        usize::from(self.0)
            .checked_sub(1)
            .and_then(|book| SPECTRUM_CODEBOOKS.get(book))
    }
}

/// The codebooks by the largest magnitude they code (Table 59):
/// `ZERO_HCB`, the pairs that share a LAV, and `ESC_HCB`. The two of a pair
/// share their codeword indexes too.
pub(super) const CLASSES: [&[Codebook]; 7] = [
    &[Codebook(0)],
    &[Codebook(1), Codebook(2)],
    &[Codebook(3), Codebook(4)],
    &[Codebook(5), Codebook(6)],
    &[Codebook(7), Codebook(8)],
    &[Codebook(9), Codebook(10)],
    &[Codebook(11)],
];

/// The first of [`CLASSES`] whose codebooks code lines of up to `largest`,
/// or `None` above [`MAX_MAGNITUDE`].
pub(super) fn class(largest: u32) -> Option<usize> {
    CLASSES
        .iter()
        .position(|books| books[0].largest_magnitude() >= largest)
}

/// A spectrum Huffman codebook (Table 59 and Tables A.2 to A.12), which
/// codes an n-tuple of quantized lines by a codeword index.
#[derive(Debug)]
pub(super) struct SpectrumCodebook {
    number: u8,
    /// `unsigned_cb`: whether the codeword codes magnitudes, each nonzero
    /// one followed by its sign bit.
    unsigned: bool,
    dimension: usize,
    /// As Table 59 gives it: for `ESC_HCB`, `ESC_FLAG`.
    lav: u32,
    codewords: &'static [Codeword],
    /// Each codeword's length with the sign bits that follow it: the bits
    /// of the n-tuple it codes, but for escape sequences. For counting
    /// bits without looking at each line's sign.
    tuple_bits: &'static [u8],
}

impl SpectrumCodebook {
    pub(super) fn number(&self) -> u8 {
        self.number
    }

    pub(super) fn is_unsigned(&self) -> bool {
        self.unsigned
    }

    /// The lines of each n-tuple: 4 for codebooks 1 to 4, 2 from
    /// `FIRST_PAIR_HCB` on.
    pub(super) fn dimension(&self) -> usize {
        self.dimension
    }

    /// Whether lines of [`ESC_FLAG`] and more are coded with escape
    /// sequences: `ESC_HCB` alone.
    pub(super) fn has_escapes(&self) -> bool {
        self.number == Codebook::ESC.0
    }

    /// The largest magnitude of a line the codebook codes: its LAV, or with
    /// escape sequences [`MAX_MAGNITUDE`] for `ESC_HCB`.
    pub(super) fn largest_magnitude(&self) -> u32 {
        if self.has_escapes() {
            MAX_MAGNITUDE
        } else {
            self.lav
        }
    }

    /// The codeword index of `tuple`, by clause 9.3 run backwards: each line
    /// offset by the LAV in a signed codebook, or its magnitude in an
    /// unsigned one, with a magnitude of `ESC_FLAG` or more taken as
    /// `ESC_FLAG` in `ESC_HCB`, as the digits of a number in base
    /// `2 * LAV + 1` or `LAV + 1`, first line most significant.
    ///
    /// `tuple` holds [`dimension`](Self::dimension) lines that the codebook
    /// codes (see [`largest_magnitude`](Self::largest_magnitude)); for
    /// others the index is another tuple's or out of range.
    pub(super) fn index(&self, tuple: &[i32]) -> usize {
        debug_assert_eq!(tuple.len(), self.dimension);
        let modulus = modulus(self.unsigned, self.lav) as usize;
        tuple.iter().fold(0, |index, &line| {
            let digit = if self.unsigned {
                line.unsigned_abs().min(self.lav)
            } else {
                line.saturating_add_unsigned(self.lav) as u32
            };
            index * modulus + digit as usize
        })
    }

    /// The codeword with codeword index `index`.
    pub(super) fn codeword(&self, index: usize) -> Codeword {
        self.codewords[index]
    }

    /// The bits of the codeword with index `index` and of the sign bits
    /// that follow it, short of any escape sequences.
    pub(super) fn tuple_bits(&self, index: usize) -> u32 {
        u32::from(self.tuple_bits[index])
    }
}

/// The base of the codeword index of a codebook (clause 9.3): `mod`.
const fn modulus(unsigned: bool, lav: u32) -> u32 {
    if unsigned { lav + 1 } else { 2 * lav + 1 }
}

/// The bits of the escape sequence of a magnitude of [`ESC_FLAG`] to
/// [`MAX_MAGNITUDE`], as [`escape_sequence`] gives them: `2 * N + 5`.
pub(super) fn escape_bits(magnitude: u32) -> u32 {
    debug_assert!((ESC_FLAG..=MAX_MAGNITUDE).contains(&magnitude));
    2 * magnitude.ilog2() - 3
}

/// The escape sequence of a magnitude of [`ESC_FLAG`] to [`MAX_MAGNITUDE`]
/// (clause 9.3): an `escape_prefix` of N ones, an `escape_separator` of a
/// zero, and an `escape_word` of N + 4 bits, where N is such that the
/// magnitude is `2^(N + 4) + escape_word`. As `escape_word` is less than
/// `2^(N + 4)`, N is the integer part of `log2(magnitude)` less 4.
///
/// # Panics
///
/// If the magnitude is out of that range: a caller's error.
pub(super) fn escape_sequence(magnitude: u32) -> Codeword {
    assert!(
        (ESC_FLAG..=MAX_MAGNITUDE).contains(&magnitude),
        "{magnitude} has no escape sequence"
    );
    let n = magnitude.ilog2() - 4;
    let prefix = (1 << n) - 1;
    let word = magnitude - (1 << (n + 4));
    // The separator's zero lies between the prefix and the word.
    Codeword {
        bits: (prefix << (n + 5)) | word,
        len: 2 * n + 5,
    }
}

/// The scalefactor codebook (Table A.1), whose codeword index is a
/// difference of scalefactors less `index_offset` (Table 58).
pub(super) static SCALEFACTOR_CODEWORDS: [Codeword; 121] = tables::SCALEFACTOR;

/// Builds a codebook from `(index, length, codeword)` rows, which must run
/// over the indexes in order.
pub(super) const fn codewords<const N: usize>(rows: [(usize, u32, u32); N]) -> [Codeword; N] {
    let mut codewords = [Codeword { bits: 0, len: 0 }; N];
    let mut i = 0;
    while i < N {
        let (index, len, bits) = rows[i];
        assert!(index == i, "row out of order");
        // Table 18 and Table 20 bound codewords at 19 and 16 bits.
        assert!(len >= 1 && len <= 19, "length out of range");
        assert!(bits >> len == 0, "codeword longer than its length");
        codewords[i] = Codeword { bits, len };
        i += 1;
    }
    codewords
}

/// The bits of each codeword of a codebook with its sign bits: one for each
/// nonzero digit of its index if the codebook is unsigned.
const fn tuple_bits<const N: usize>(
    codewords: &[Codeword; N],
    unsigned: bool,
    dimension: usize,
    lav: u32,
) -> [u8; N] {
    let modulus = modulus(unsigned, lav) as usize;
    assert!(
        N == modulus.pow(dimension as u32),
        "a codeword for each tuple"
    );
    let mut bits = [0; N];
    let mut index = 0;
    while index < N {
        let mut signs = 0;
        if unsigned {
            let mut rest = index;
            let mut line = 0;
            while line < dimension {
                if rest % modulus != 0 {
                    signs += 1;
                }
                rest /= modulus;
                line += 1;
            }
        }
        bits[index] = (codewords[index].len + signs) as u8;
        index += 1;
    }
    bits
}

/// A spectrum codebook from its row of Table 59 and its table of Annex A.
macro_rules! spectrum_codebook {
    ($number:literal, $unsigned:literal, $dimension:literal, $lav:literal, $codewords:ident) => {{
        const TUPLE_BITS: [u8; tables::$codewords.len()] =
            tuple_bits(&tables::$codewords, $unsigned, $dimension, $lav);
        SpectrumCodebook {
            number: $number,
            unsigned: $unsigned,
            dimension: $dimension,
            lav: $lav,
            codewords: &tables::$codewords,
            tuple_bits: &TUPLE_BITS,
        }
    }};
}

/// Table 59: codebook number, `unsigned_cb`, dimension and LAV, with the
/// codebook's table.
static SPECTRUM_CODEBOOKS: [SpectrumCodebook; 11] = [
    spectrum_codebook!(1, false, 4, 1, SPECTRUM_1),
    spectrum_codebook!(2, false, 4, 1, SPECTRUM_2),
    spectrum_codebook!(3, true, 4, 2, SPECTRUM_3),
    spectrum_codebook!(4, true, 4, 2, SPECTRUM_4),
    spectrum_codebook!(5, false, 2, 4, SPECTRUM_5),
    spectrum_codebook!(6, false, 2, 4, SPECTRUM_6),
    spectrum_codebook!(7, true, 2, 7, SPECTRUM_7),
    spectrum_codebook!(8, true, 2, 7, SPECTRUM_8),
    spectrum_codebook!(9, true, 2, 12, SPECTRUM_9),
    spectrum_codebook!(10, true, 2, 12, SPECTRUM_10),
    spectrum_codebook!(11, true, 2, 16, SPECTRUM_11),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Checks that `codewords` form a complete prefix code: no codeword
    /// begins another, and the Kraft sum of `2^-len` is exactly 1, so that
    /// every bit string decodes. Either fails if any one codeword is
    /// transcribed wrongly.
    fn assert_complete_prefix_code(codewords: &[Codeword], name: &str) {
        const MAX_LEN: u32 = 32;
        let kraft: u64 = codewords.iter().map(|c| 1 << (MAX_LEN - c.len)).sum();
        assert_eq!(kraft, 1 << MAX_LEN, "{name}: Kraft sum is not 1");
        for (i, a) in codewords.iter().enumerate() {
            for b in &codewords[i + 1..] {
                let (short, long) = if a.len <= b.len { (a, b) } else { (b, a) };
                assert_ne!(
                    long.bits >> (long.len - short.len),
                    short.bits,
                    "{name}: {short:?} is a prefix of {long:?}"
                );
            }
        }
    }

    fn spectrum(number: u8) -> &'static SpectrumCodebook {
        Codebook::new(number).unwrap().spectrum().unwrap()
    }

    #[test]
    fn every_codebook_is_a_complete_prefix_code() {
        assert_complete_prefix_code(&SCALEFACTOR_CODEWORDS, "scalefactor codebook");
        for book in &SPECTRUM_CODEBOOKS {
            assert_complete_prefix_code(book.codewords, &format!("codebook {}", book.number));
        }
    }

    #[test]
    fn codebooks_follow_table_59() {
        assert!(Codebook::ZERO.spectrum().is_none());
        assert!(!Codebook::ZERO.sends_scalefactor());
        assert_eq!(Codebook::ZERO.largest_magnitude(), 0);
        assert_eq!(Codebook::new(12), None);
        // Number, unsigned_cb, dimension, LAV, codewords, largest magnitude.
        let rows = [
            (1, false, 4, 1, 81, 1),
            (2, false, 4, 1, 81, 1),
            (3, true, 4, 2, 81, 2),
            (4, true, 4, 2, 81, 2),
            (5, false, 2, 4, 81, 4),
            (6, false, 2, 4, 81, 4),
            (7, true, 2, 7, 64, 7),
            (8, true, 2, 7, 64, 7),
            (9, true, 2, 12, 169, 12),
            (10, true, 2, 12, 169, 12),
            (11, true, 2, 16, 289, MAX_MAGNITUDE),
        ];
        for (number, unsigned, dimension, lav, len, largest) in rows {
            let book = spectrum(number);
            assert_eq!(
                (
                    book.number(),
                    book.is_unsigned(),
                    book.dimension(),
                    book.lav,
                    book.codewords.len(),
                ),
                (number, unsigned, dimension, lav, len)
            );
            let codebook = Codebook::new(number).unwrap();
            assert_eq!(codebook.number(), number);
            assert_eq!(codebook.largest_magnitude(), largest);
            assert_eq!(book.largest_magnitude(), largest);
            assert_eq!(codebook.sends_scalefactor(), number > 0);
            assert_eq!(book.has_escapes(), number == 11);
        }
    }

    #[test]
    fn classes_rise_with_the_largest_magnitude() {
        let classes = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 3), (5, 4), (7, 4)];
        let classes = classes.into_iter().chain([(8, 5), (12, 5), (13, 6)]);
        for (largest, class) in classes.chain([(16, 6), (MAX_MAGNITUDE, 6)]) {
            assert_eq!(super::class(largest), Some(class), "{largest}");
        }
        assert_eq!(super::class(MAX_MAGNITUDE + 1), None);
        for (class, books) in CLASSES.iter().enumerate() {
            for book in *books {
                assert_eq!(super::class(book.largest_magnitude()), Some(class));
            }
        }
    }

    #[test]
    fn spot_entries_read_as_printed() {
        let cw = |bits, len| Codeword { bits, len };
        // Table A.1, index 60 (a difference of 0) and 120.
        assert_eq!(SCALEFACTOR_CODEWORDS[60], cw(0x0, 1));
        assert_eq!(SCALEFACTOR_CODEWORDS[120], cw(0x7fff3, 19));
        // Table A.2, index 40, the tuple of zeros; Table A.5, index 80;
        // Table A.12, indexes 0 and 288.
        assert_eq!(spectrum(1).codeword(40), cw(0x0, 1));
        assert_eq!(spectrum(4).codeword(80), cw(0x7fc, 11));
        assert_eq!(spectrum(11).codeword(0), cw(0x0, 4));
        assert_eq!(spectrum(11).codeword(288), cw(0x4, 5));
    }

    #[test]
    fn indexes_invert_clause_9_3() {
        // Clause 9.3's translation of a codeword index to its n-tuple.
        fn tuple(book: &SpectrumCodebook, mut idx: usize) -> Vec<i32> {
            let (modulus, off) = if book.unsigned {
                (book.lav as usize + 1, 0)
            } else {
                (2 * book.lav as usize + 1, book.lav as i32)
            };
            let mut lines = vec![0; book.dimension];
            for line in lines.iter_mut().rev() {
                *line = (idx % modulus) as i32 - off;
                idx /= modulus;
            }
            lines
        }
        for book in &SPECTRUM_CODEBOOKS {
            for index in 0..book.codewords.len() {
                let lines = tuple(book, index);
                assert_eq!(book.index(&lines), index, "codebook {}", book.number);
                let signs = if book.unsigned {
                    lines.iter().filter(|&&line| line != 0).count() as u32
                } else {
                    0
                };
                assert_eq!(book.tuple_bits(index), book.codeword(index).len + signs);
            }
        }
        // Magnitudes from ESC_FLAG up share the escape flag's index.
        let esc = spectrum(11);
        assert_eq!(esc.index(&[-16, 3]), esc.index(&[8191, -3]));
        assert_eq!(esc.index(&[16, 16]), 288);
    }

    #[test]
    fn escape_sequences_follow_clause_9_3() {
        // Clause 9.3: "an escape_sequence of 00000 would decode as 16, an
        // escape_sequence of 01111 as 31, an escape_sequence of 1000000 as
        // 32, one of 1011111 as 63".
        let cw = |bits, len| Codeword { bits, len };
        assert_eq!(escape_sequence(16), cw(0b00000, 5));
        assert_eq!(escape_sequence(31), cw(0b01111, 5));
        assert_eq!(escape_sequence(32), cw(0b1000000, 7));
        assert_eq!(escape_sequence(63), cw(0b1011111, 7));
        // The longest: N = 8, "less than 22 bits": eight ones, a zero,
        // and 12 ones.
        assert_eq!(escape_sequence(8191), cw((0x1fe << 12) | 0xfff, 21));
        for magnitude in ESC_FLAG..=MAX_MAGNITUDE {
            assert_eq!(escape_bits(magnitude), escape_sequence(magnitude).len);
        }
    }
}
