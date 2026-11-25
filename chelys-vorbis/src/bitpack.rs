// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The bitpacking convention (clause 2): each field written least
//! significant bit first, into each byte from its least significant bit.
//! The bits are written in the opposite order from chelys's other codecs,
//! which `chelys_core::bitstream` writes most significant bit first.

/// A packet being written, bit by bit (clause 2.1.4).
#[derive(Debug, Default)]
pub struct BitPacker {
    bytes: Vec<u8>,
    bit_len: usize,
}

impl BitPacker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Writes `value` in `bits` bits, least significant first. A signed
    /// field is written as its two's complement in `bits` bits (clause
    /// 2.1.5).
    ///
    /// # Panics
    ///
    /// If `bits` is above 32, or, in builds with debug assertions, if
    /// `value` does not fit in `bits` bits: a caller's error, of which
    /// other builds write the low `bits` bits.
    pub fn write(&mut self, value: u32, bits: u32) {
        assert!(bits <= 32, "cannot write {bits} bits at once");
        debug_assert!(
            bits == 32 || value >> bits == 0,
            "value {value} does not fit in {bits} bits"
        );
        // As many of the bits left as the last byte has room for at a time.
        let mut value = u64::from(value);
        let mut left = bits;
        while left > 0 {
            let used = (self.bit_len % 8) as u32;
            if used == 0 {
                self.bytes.push(0);
            }
            let taken = (8 - used).min(left);
            let low = (value & ((1 << taken) - 1)) as u8;
            *self.bytes.last_mut().expect("a byte was pushed") |= low << used;
            value >>= taken;
            left -= taken;
            self.bit_len += taken as usize;
        }
    }

    pub fn write_bool(&mut self, flag: bool) {
        self.write(u32::from(flag), 1);
    }

    /// Writes each byte in 8 bits, in order: octets, as the headers' strings
    /// are coded (clause 5.2.3).
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write(u32::from(byte), 8);
        }
    }

    #[cfg(test)]
    pub fn bit_len(&self) -> usize {
        self.bit_len
    }

    /// The packet's bytes, the unused bits of the last zero (clause 2.1.8).
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// The position of the highest bit set in `x`, from 1, or 0 for 0 (clause
/// 9.2.1). The clause also gives 0 for negative numbers, which the encoder
/// never asks of it.
pub fn ilog(x: u32) -> u32 {
    u32::BITS - x.leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_the_example_of_clause_2_1_6() {
        let mut packer = BitPacker::new();
        packer.write(12, 4);
        assert_eq!(packer.bit_len(), 4);
        packer.write(0b111, 3); // -1 in three bits
        packer.write(17, 7);
        assert_eq!(packer.bit_len(), 14); // a bit cursor of 6 in byte 1
        packer.write(6969, 13);
        assert_eq!(
            packer.into_bytes(),
            [0b1111_1100, 0b0100_1000, 0b1100_1110, 0b0000_0110]
        );
    }

    #[test]
    fn packs_32_bits_at_any_offset() {
        for offset in 0..8 {
            let mut packer = BitPacker::new();
            packer.write(0, offset);
            packer.write(0x8765_4321, 32);
            let mut expected = (0x8765_4321u64 << offset).to_le_bytes().to_vec();
            expected.truncate(if offset == 0 { 4 } else { 5 });
            assert_eq!(packer.into_bytes(), expected, "offset {offset}");
        }
    }

    #[test]
    fn writes_octets_whole_at_a_byte_boundary() {
        let mut packer = BitPacker::new();
        packer.write_bytes(b"vorbis");
        assert_eq!(packer.into_bytes(), b"vorbis");
    }

    #[test]
    fn ilog_gives_the_examples_of_clause_9_2_1() {
        for (x, expected) in [(0, 0), (1, 1), (2, 2), (3, 2), (4, 3), (7, 3)] {
            assert_eq!(ilog(x), expected, "ilog({x})");
        }
        assert_eq!(ilog(u32::MAX), 32);
    }
}
