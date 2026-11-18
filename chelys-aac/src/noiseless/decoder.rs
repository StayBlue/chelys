// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! A decoder of `section_data()`, `scale_factor_data()` and
//! `spectral_data()` for one window group, written from the syntax of
//! Tables 17, 18 and 20 and clauses 9.3 and 11.3.2 apart from the encoder,
//! for tests to decode what it writes. It shares only the codebooks.

use super::codebooks::{Codebook, Codeword, SCALEFACTOR_CODEWORDS};
use super::tables;
use crate::scalefactor_bands::Transform;

pub struct BitReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn read(&mut self, bits: u32) -> u32 {
        (0..bits).fold(0, |value, _| {
            let byte = self.bytes[self.position / 8];
            let bit = (byte >> (7 - self.position % 8)) & 1;
            self.position += 1;
            (value << 1) | u32::from(bit)
        })
    }

    /// Reads a codeword of `codewords` bit by bit, and returns its index.
    fn decode(&mut self, codewords: &[Codeword]) -> usize {
        let (mut bits, mut len) = (0, 0);
        loop {
            bits = (bits << 1) | self.read(1);
            len += 1;
            assert!(len <= 19, "no codeword matches");
            if let Some(index) = codewords
                .iter()
                .position(|c| c.len == len && c.bits == bits)
            {
                return index;
            }
        }
    }
}

/// `sect_cb[g][i]`, `sect_start[g][i]` and `sect_end[g][i]` of one group
/// (Table 17).
pub fn section_data(
    r: &mut BitReader,
    transform: Transform,
    max_sfb: usize,
) -> Vec<(u8, usize, usize)> {
    let (bits, sect_esc_val) = match transform {
        Transform::Short => (3, (1 << 3) - 1),
        Transform::Long => (5, (1 << 5) - 1),
    };
    let mut sections = Vec::new();
    let mut k = 0;
    while k < max_sfb {
        let sect_cb = r.read(4) as u8;
        let mut sect_len = 0;
        loop {
            let sect_len_incr = r.read(bits);
            if sect_len_incr != sect_esc_val {
                sect_len += sect_len_incr as usize;
                break;
            }
            sect_len += sect_esc_val as usize;
        }
        sections.push((sect_cb, k, k + sect_len));
        k += sect_len;
    }
    assert_eq!(k, max_sfb, "the sections end at max_sfb");
    sections
}

/// The scalefactors `sf[g][sfb]` of one group (Table 18, clause 11.3.2),
/// from `last_sf`, which they leave at the last one decoded.
pub fn scale_factor_data(
    r: &mut BitReader,
    sections: &[(u8, usize, usize)],
    last_sf: &mut i32,
) -> Vec<i32> {
    let mut sf = Vec::new();
    for &(sect_cb, start, end) in sections {
        for _ in start..end {
            if sect_cb == 0 {
                sf.push(0);
            } else {
                // Table 58: index_offset -60.
                let dpcm_sf = r.decode(&SCALEFACTOR_CODEWORDS) as i32 - 60;
                *last_sf += dpcm_sf;
                assert!((0..=255).contains(last_sf));
                sf.push(*last_sf);
            }
        }
    }
    sf
}

/// The quantized lines of one group (Table 20, clause 9.3), `len` of them,
/// zero above the sections.
pub fn spectral_data(
    r: &mut BitReader,
    sections: &[(u8, usize, usize)],
    offsets: &[u16],
    len: usize,
) -> Vec<i32> {
    // Table 59: unsigned_cb, dimension, LAV and codebook.
    let books: [(bool, usize, i32, &[Codeword]); 11] = [
        (false, 4, 1, &tables::SPECTRUM_1),
        (false, 4, 1, &tables::SPECTRUM_2),
        (true, 4, 2, &tables::SPECTRUM_3),
        (true, 4, 2, &tables::SPECTRUM_4),
        (false, 2, 4, &tables::SPECTRUM_5),
        (false, 2, 4, &tables::SPECTRUM_6),
        (true, 2, 7, &tables::SPECTRUM_7),
        (true, 2, 7, &tables::SPECTRUM_8),
        (true, 2, 12, &tables::SPECTRUM_9),
        (true, 2, 12, &tables::SPECTRUM_10),
        (true, 2, 16, &tables::SPECTRUM_11),
    ];
    let mut lines = vec![0; len];
    for &(sect_cb, start, end) in sections {
        if sect_cb == 0 || sect_cb > Codebook::ESC.number() {
            continue;
        }
        let (unsigned, dim, lav, codewords) = books[usize::from(sect_cb) - 1];
        let (modulus, off) = if unsigned {
            (lav + 1, 0)
        } else {
            (2 * lav + 1, lav)
        };
        let mut k = usize::from(offsets[start]);
        while k < usize::from(offsets[end]) {
            let mut idx = r.decode(codewords) as i32;
            let mut tuple = [0; 4];
            for (i, line) in tuple[..dim].iter_mut().enumerate() {
                let place = modulus.pow((dim - 1 - i) as u32);
                *line = idx / place - off;
                idx -= (*line + off) * place;
            }
            if unsigned {
                for line in &mut tuple[..dim] {
                    if *line != 0 && r.read(1) == 1 {
                        *line = -*line;
                    }
                }
            }
            if sect_cb == Codebook::ESC.number() {
                for line in &mut tuple[..dim] {
                    if line.abs() == 16 {
                        let mut n = 0;
                        while r.read(1) == 1 {
                            n += 1;
                        }
                        let escape_word = r.read(n + 4) as i32;
                        *line = line.signum() * ((1 << (n + 4)) + escape_word);
                    }
                }
            }
            lines[k..k + dim].copy_from_slice(&tuple[..dim]);
            k += dim;
        }
    }
    lines
}

/// A deterministic stream of pseudo-random numbers (xorshift), for tests to
/// make spectra of.
#[cfg(test)]
pub struct Noise(u64);

#[cfg(test)]
impl Noise {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    fn draw(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }

    /// A number in `0..n`.
    pub fn below(&mut self, n: u32) -> u32 {
        self.draw() % n
    }

    /// Lines for `offsets`, whose bands are each of zeros or reach a
    /// largest magnitude drawn from every codebook's range, escapes to
    /// 8191 included, and are sparse at random.
    pub fn spectrum(&mut self, offsets: &[u16], len: usize) -> Vec<i32> {
        const LARGEST: [u32; 12] = [0, 0, 1, 2, 3, 4, 7, 12, 15, 16, 300, 8191];
        let mut lines = vec![0; len];
        for band in offsets.windows(2) {
            let largest = LARGEST[self.below(12) as usize];
            let density = 1 + self.below(4);
            for line in &mut lines[usize::from(band[0])..usize::from(band[1])] {
                if largest > 0 && self.below(4) < density {
                    let magnitude = self.below(largest + 1) as i32;
                    *line = if self.below(2) == 0 {
                        magnitude
                    } else {
                        -magnitude
                    };
                }
            }
        }
        lines
    }
}
