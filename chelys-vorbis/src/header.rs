// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The identification and comment headers (clauses 4.2.1 to 4.2.3 and 5),
//! and the fields that begin every header.

use super::bitpack::{BitPacker, ilog};

/// The vendor string of the comment header: chelys, and its version.
pub const VENDOR: &str = concat!("chelys ", env!("CARGO_PKG_VERSION"));

/// Each header's `packet_type` (clause 4.2.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeaderType {
    Identification = 1,
    Comment = 3,
    Setup = 5,
}

/// A header packet's `packet_type` and the six octets "vorbis" that follow
/// it (clause 4.2.1).
pub fn begin(header: HeaderType) -> BitPacker {
    let mut out = BitPacker::new();
    out.write(header as u32, 8);
    out.write_bytes(b"vorbis");
    out
}

/// The three bitrate fields of the identification header (clause 4.2.2),
/// in bits per second, each 0 where unset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bitrates {
    pub maximum: u32,
    pub nominal: u32,
    pub minimum: u32,
}

impl Bitrates {
    /// None set.
    pub const NONE: Self = Self {
        maximum: 0,
        nominal: 0,
        minimum: 0,
    };
}

/// The identification header (clause 4.2.2) of a stream of `channels`
/// channels at `sample_rate` Hz, of the bitrates `bitrates`, whose blocks
/// are of `blocksizes[0]` and `blocksizes[1]` samples.
///
/// Spec reading: the three bitrate fields "are used only as hints", and
/// "meaningful only when greater than zero". A stream at a constant rate
/// gives it as all three: "All three fields set to the same value implies
/// a fixed rate, or tightly bounded, nearly fixed-rate bitstream", as its
/// buffer bounds it (see [`Rate::Constant`](crate::Rate::Constant)),
/// though its packets' lengths vary from block to block. A stream at a
/// mean rate gives it as `bitrate_nominal` alone, the maximum and minimum
/// 0: "Only nominal set implies a VBR or ABR stream that averages the
/// nominal bitrate", as the steering only aims at it (see
/// [`Rate::Mean`](crate::Rate::Mean)), and "the nominal bitrate field
/// especially may be considerably off". A stream at any other rate gives
/// none: "None set indicates the encoder does not care to speculate", as
/// its rate follows its content.
///
/// # Panics
///
/// If `channels` or `sample_rate` is 0 or does not fit its field, or a
/// bitrate does not fit its signed field, or a block size is not one that
/// clause 4.2.2 allows, or the first is above the second.
pub fn identification(
    channels: usize,
    sample_rate: u32,
    bitrates: Bitrates,
    blocksizes: [usize; 2],
) -> Vec<u8> {
    assert!((1..=255).contains(&channels), "{channels} channels");
    assert!(sample_rate > 0, "a sample rate of 0");
    for bitrate in [bitrates.maximum, bitrates.nominal, bitrates.minimum] {
        assert!(i32::try_from(bitrate).is_ok(), "a bitrate of {bitrate}");
    }
    for blocksize in blocksizes {
        assert!(
            blocksize.is_power_of_two() && (64..=8192).contains(&blocksize),
            "a block of {blocksize} samples"
        );
    }
    assert!(blocksizes[0] <= blocksizes[1], "blocks of {blocksizes:?}");
    let mut out = begin(HeaderType::Identification);
    out.write(0, 32); // vorbis_version
    out.write(channels as u32, 8);
    out.write(sample_rate, 32);
    out.write(bitrates.maximum, 32);
    out.write(bitrates.nominal, 32);
    out.write(bitrates.minimum, 32);
    for blocksize in blocksizes {
        // The exponent of 2.
        out.write(ilog(blocksize as u32) - 1, 4);
    }
    out.write_bool(true); // framing_flag
    out.into_bytes()
}

/// The comment header (clause 5.2), with [`VENDOR`] and no comments.
pub fn comment() -> Vec<u8> {
    let mut out = begin(HeaderType::Comment);
    out.write(VENDOR.len() as u32, 32);
    out.write_bytes(VENDOR.as_bytes());
    out.write(0, 32); // user_comment_list_length
    out.write_bool(true); // framing_bit
    out.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_identification_header_is_30_bytes() {
        // Appendix A.2 puts it alone in the first Ogg page, "of exactly 58
        // bytes": a header of 27 bytes and a segment table of one.
        let header = identification(2, 44_100, Bitrates::NONE, [256, 2048]);
        let mut expected = vec![1];
        expected.extend(b"vorbis");
        expected.extend(0u32.to_le_bytes());
        expected.push(2);
        expected.extend(44_100u32.to_le_bytes());
        expected.extend([0; 12]);
        expected.push(0xB8); // 2^8, then 2^11 in the high bits
        expected.push(1);
        assert_eq!(header, expected);
        assert_eq!(header.len(), 30);
    }

    /// Each bitrate is set in its field, the maximum, nominal and minimum
    /// in order, as a signed integer, little-endian.
    #[test]
    fn the_bitrates_are_set_in_their_fields() {
        let nominal = Bitrates {
            nominal: 128_000,
            ..Bitrates::NONE
        };
        let header = identification(1, 48_000, nominal, [256, 2048]);
        assert_eq!(header[16..20], [0; 4]);
        assert_eq!(header[20..24], 128_000i32.to_le_bytes());
        assert_eq!(header[24..28], [0; 4]);
        let all = Bitrates {
            maximum: 96_000,
            nominal: 64_000,
            minimum: i32::MAX as u32,
        };
        let header = identification(1, 48_000, all, [256, 2048]);
        assert_eq!(header[16..20], 96_000i32.to_le_bytes());
        assert_eq!(header[20..24], 64_000i32.to_le_bytes());
        assert_eq!(header[24..28], i32::MAX.to_le_bytes());
    }

    #[test]
    #[should_panic(expected = "a bitrate of 2147483648")]
    fn a_bitrate_past_the_signed_field_panics() {
        let maximum = Bitrates {
            maximum: 1 << 31,
            ..Bitrates::NONE
        };
        identification(1, 48_000, maximum, [256, 2048]);
    }

    #[test]
    fn the_comment_header_has_the_vendor_alone() {
        let header = comment();
        let mut expected = vec![3];
        expected.extend(b"vorbis");
        expected.extend((VENDOR.len() as u32).to_le_bytes());
        expected.extend(VENDOR.as_bytes());
        expected.extend(0u32.to_le_bytes());
        expected.push(1);
        assert_eq!(header, expected);
        assert!(VENDOR.starts_with("chelys "));
    }
}
