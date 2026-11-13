// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Most-significant-bit-first bit output, as used by the `bslbf` and `uimsbf`
//! mnemonics (clause 2.2.6).

#[derive(Debug, Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    bit_len: usize,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Writes the low `bits` bits of `value`, most significant first.
    pub fn write(&mut self, value: u32, bits: u32) {
        assert!(bits <= 32, "cannot write {bits} bits at once");
        debug_assert!(
            bits == 32 || value >> bits == 0,
            "value {value} does not fit in {bits} bits"
        );
        for shift in (0..bits).rev() {
            if self.bit_len.is_multiple_of(8) {
                self.bytes.push(0);
            }
            if (value >> shift) & 1 == 1 {
                let last = self.bytes.len() - 1;
                self.bytes[last] |= 0x80 >> (self.bit_len % 8);
            }
            self.bit_len += 1;
        }
    }

    pub fn write_bool(&mut self, flag: bool) {
        self.write(u32::from(flag), 1);
    }

    pub fn bit_len(&self) -> usize {
        self.bit_len
    }

    /// Returns the written bytes; a trailing partial byte is zero-filled.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_msb_first_across_byte_boundaries() {
        let mut w = BitWriter::new();
        w.write(0b101, 3);
        w.write(0b1_1110_0001, 9);
        w.write_bool(true);
        assert_eq!(w.bit_len(), 13);
        assert_eq!(w.into_bytes(), [0b1011_1110, 0b0001_1000]);
    }

    #[test]
    fn writes_full_words() {
        let mut w = BitWriter::new();
        w.write(0xDEAD_BEEF, 32);
        assert_eq!(w.into_bytes(), [0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn zero_width_write_is_a_no_op() {
        let mut w = BitWriter::new();
        w.write(0, 0);
        assert_eq!(w.bit_len(), 0);
        assert!(w.into_bytes().is_empty());
    }
}
