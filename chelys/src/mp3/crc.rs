// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The optional `crc_check` word (clauses 2.4.1.4 and 2.4.3.1, figure A.9).

/// `G(X) = X^16 + X^15 + X^2 + 1`, without the implicit X^16 term.
const GENERATOR: u16 = 0x8005;

/// Clocks `bytes` through the figure A.9 shift register, most significant bit
/// of each byte first, starting from the all-ones state. The result is
/// `b15..b0`, transmitted highest order first (`rpchof`).
pub fn crc16(bytes: impl IntoIterator<Item = u8>) -> u16 {
    let mut register = 0xFFFF_u16;
    for byte in bytes {
        for shift in (0..8).rev() {
            let data_bit = u16::from(byte >> shift) & 1;
            let feedback = data_bit ^ (register >> 15);
            register <<= 1;
            if feedback == 1 {
                register ^= GENERATOR;
            }
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
}
