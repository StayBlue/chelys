// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The CRC-16 of ISO/IEC 11172-3 (clause 2.4.3.1, figure A.9), which an MP3
//! frame's `crc_check` word carries, and which ISO/IEC 13818-7 takes for
//! the `crc_check` of ADTS (clause 8.1.3.2.2).

/// `G(X) = X^16 + X^15 + X^2 + 1`, without the implicit X^16 term.
const GENERATOR: u16 = 0x8005;

/// The shift register's initial state, `1111 1111 1111 1111`.
pub const INITIAL: u16 = 0xFFFF;

/// Clocks `bytes` through the figure A.9 shift register, most significant bit
/// of each byte first, starting from the all-ones state. The result is
/// `b15..b0`, transmitted highest order first (`rpchof`).
pub fn crc16(bytes: impl IntoIterator<Item = u8>) -> u16 {
    update(INITIAL, bytes)
}

/// Clocks `bytes` through the figure A.9 shift register from the state
/// `register`, and returns the state after them; a CRC of bytes that arrive
/// in parts is the state after the last part.
pub fn update(register: u16, bytes: impl IntoIterator<Item = u8>) -> u16 {
    update_bits(
        register,
        bytes
            .into_iter()
            .flat_map(|byte| (0..8).rev().map(move |shift| (byte >> shift) & 1 == 1)),
    )
}

/// Clocks `bits` through the figure A.9 shift register from the state
/// `register`, one at a time, and returns the state after them: for fields
/// that do not fill whole bytes.
pub fn update_bits(mut register: u16, bits: impl IntoIterator<Item = bool>) -> u16 {
    for bit in bits {
        let feedback = u16::from(bit) ^ (register >> 15);
        register <<= 1;
        if feedback == 1 {
            register ^= GENERATOR;
        }
    }
    register
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_leaves_the_initial_state() {
        assert_eq!(crc16([]), 0xFFFF);
    }

    /// This parameter set (poly 0x8005, init 0xFFFF, unreflected, no final
    /// XOR) is catalogued generically as CRC-16/CMS, whose published check
    /// value over ASCII "123456789" is 0xAEE7.
    #[test]
    fn matches_the_catalogued_check_value() {
        assert_eq!(crc16(*b"123456789"), 0xAEE7);
    }

    /// From the all-zero state this is catalogued as CRC-16/UMTS (or
    /// CRC-16/BUYPASS), whose check value is 0xFEE8; a CRC of bytes in parts
    /// is that of the whole.
    #[test]
    fn updates_in_parts_from_any_state() {
        assert_eq!(update(0, *b"123456789"), 0xFEE8);
        assert_eq!(update(update(0, *b"1234"), *b"56789"), 0xFEE8);
    }

    /// Clocked by hand through figure A.9 from the all-ones state: a 1
    /// matches b15 and only shifts, giving 0xFFFE; a 0 differs from it and
    /// shifts with the feedback into b0, b2 and b15, giving 0xFFFE ^ 0x8005;
    /// and from zero, a 1 sets those three bits.
    #[test]
    fn single_bits_follow_the_register() {
        assert_eq!(update_bits(INITIAL, [true]), 0xFFFE);
        assert_eq!(update_bits(INITIAL, [false]), 0x7FFB);
        assert_eq!(update_bits(0, [true]), 0x8005);
        // Then a 1 against b15 of 0x8005, 1: a shift alone.
        assert_eq!(update_bits(0, [true, true]), 0x000A);
    }

    #[test]
    fn bits_are_the_bytes_most_significant_first() {
        let bits = b"123456789"
            .iter()
            .flat_map(|&byte| (0..8).rev().map(move |shift| (byte >> shift) & 1 == 1));
        assert_eq!(update_bits(INITIAL, bits), 0xAEE7);
        // A string that ends within a byte: 0x31 then the three bits 101.
        let partial = update_bits(crc16([0x31]), [true, false, true]);
        assert_eq!(
            partial,
            update_bits(
                INITIAL,
                [
                    false, false, true, true, false, false, false, true, true, false, true
                ]
            )
        );
    }
}
