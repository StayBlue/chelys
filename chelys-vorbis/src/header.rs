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

/// The identification header (clause 4.2.2) of a stream of `channels`
/// channels at `sample_rate` Hz, whose blocks are of `blocksizes[0]` and
/// `blocksizes[1]` samples.
///
/// The three bitrate fields are 0: "None set indicates the encoder does not
/// care to speculate".
///
/// # Panics
///
/// If `channels` or `sample_rate` is 0 or does not fit its field, or a
/// block size is not one that clause 4.2.2 allows, or the first is above
/// the second.
pub fn identification(channels: usize, sample_rate: u32, blocksizes: [usize; 2]) -> Vec<u8> {
    assert!((1..=255).contains(&channels), "{channels} channels");
    assert!(sample_rate > 0, "a sample rate of 0");
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
    out.write(0, 32); // bitrate_maximum
    out.write(0, 32); // bitrate_nominal
    out.write(0, 32); // bitrate_minimum
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
        let header = identification(2, 44_100, [256, 2048]);
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
