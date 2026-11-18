// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The `fill_element()` (Table 26), with an `extension_payload()` (Table
//! 28) of fill data: what pads a frame to the length the bitrate needs
//! once the bit reservoir is full (clauses 8.8.2.2 and C.7.3).

use chelys_core::bitstream::BitWriter;

use super::ID_FIL;

/// The bits of a `fill_element()` before its payload, with its
/// `id_syn_ele`: `id_syn_ele` and `count`.
const FILL_HEADER_BITS: usize = 3 + 4;

/// The most bytes of `extension_payload()` one `fill_element()` is given:
/// as many as `count` gives without `esc_count`.
///
/// Spec reading: the standard gives two lengths for a `count` of 15. Table
/// 26 reads `cnt = count; if (cnt == 15) cnt += esc_count - 1;`, and
/// `extension_payload(cnt)` (Table 28) is then `cnt` bytes, its
/// `extension_type` and `fill_nibble` among them; but clause 8.7.2 says
/// that `count` "is incremented with the value of esc_count if count
/// equals 15", one more, and that "the resulting number gives the number
/// of fill_bytes to be read". The escape is not used: each `fill_element()` holds at
/// most 14 bytes, which both readings give the same length, and more
/// elements fill more.
const MAX_FILL_BYTES: usize = 14;

/// `extension_type` (Table 40).
///
/// Spec reading: of the two types for filling, `EXT_FILL` leaves its bits
/// undefined (it falls to the `other_bits` of Table 28), and
/// `EXT_FILL_DATA` defines them, as a pattern for receivers that recover
/// their clock from the stream (clause 8.8.2.2); that one is taken.
const EXT_FILL_DATA: u32 = 0b0001;

/// `fill_nibble` and `fill_byte`: "normatively defined" (clause 8.8.2.2).
const FILL_NIBBLE: u32 = 0b0000;
const FILL_BYTE: u32 = 0b1010_0101;

/// Writes as many `fill_element()`s, each with its `id_syn_ele`, as fill
/// `bits` to within the 7 bits that byte alignment pads: none if `bits` is
/// below 8. Gives the bits written.
///
/// A `fill_element()` takes 7 bits and from 0 to 14 bytes after them, so
/// one of them reaches each length of 7 bits more than a byte multiple up
/// to its longest, and more take each a part. Fill elements may come
/// anywhere in a `raw_data_block()`, any number of them (clause 8.2.1.1).
pub fn write_fill(w: &mut BitWriter, bits: usize) -> usize {
    let mut left = bits;
    while left >= 8 {
        let cnt = ((left - FILL_HEADER_BITS) / 8).min(MAX_FILL_BYTES);
        w.write(ID_FIL, 3);
        w.write(cnt as u32, 4); // count, below 15: no esc_count
        // extension_payload(cnt), cnt bytes, while cnt > 0.
        if cnt > 0 {
            w.write(EXT_FILL_DATA, 4); // extension_type
            w.write(FILL_NIBBLE, 4);
            for _ in 0..cnt - 1 {
                w.write(FILL_BYTE, 8);
            }
        }
        left -= FILL_HEADER_BITS + 8 * cnt;
    }
    bits - left
}

#[cfg(test)]
mod tests {
    use chelys_core::bitstream::binary_digits;

    use super::*;

    fn filled(bits: usize) -> String {
        let mut w = BitWriter::new();
        let written = write_fill(&mut w, bits);
        assert_eq!(written, w.bit_len());
        assert!(bits - written < 8, "{bits} bits filled with {written}");
        binary_digits(&w.into_bytes())[..written].to_owned()
    }

    #[test]
    fn bits_that_alignment_pads_are_left() {
        assert_eq!(filled(7), "");
    }

    #[test]
    fn the_shortest_element_has_no_payload() {
        // id_syn_ele ID_FIL, count 0.
        assert_eq!(filled(8), ["110", "0000"].concat());
        assert_eq!(filled(14), ["110", "0000"].concat());
    }

    #[test]
    fn a_payload_is_fill_data() {
        // count 2: extension_type EXT_FILL_DATA, fill_nibble, a fill_byte.
        assert_eq!(
            filled(23),
            ["110", "0010", "0001", "0000", "10100101"].concat()
        );
    }

    #[test]
    fn the_longest_element_has_14_bytes_and_no_escape() {
        let longest = ["110", "1110", "0001", "0000", &"10100101".repeat(13)].concat();
        assert_eq!(filled(7 + 8 * 14), longest);
        // The next byte starts a second element.
        assert_eq!(filled(7 + 8 * 15), [&longest, "110", "0000"].concat());
    }

    #[test]
    fn every_length_is_reached_with_elements_enough() {
        for bits in 0..8 * 2000 {
            let fill = filled(bits);
            // No element's count is the escape, 15.
            assert!(!fill.contains("1101111"), "{bits}: {fill}");
        }
    }
}
