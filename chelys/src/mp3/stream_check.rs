// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Walks a stream frame by frame, as a decoder would find its frames and
//! main data, and checks the rules of the bit reservoir on it: decoders that
//! read a whole file need not enforce them.

use super::header::{Bitrate, SamplingFrequency, Version};

/// The decoder's input buffer (clause 2.4.3.4.4).
const BUFFER_BITS: usize = 7680;

/// A frame as the stream has it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Frame {
    pub bitrate: Bitrate,
    pub padding: bool,
    pub len: usize,
    pub main_data_begin: usize,
}

/// The frames of `stream`, after checking that each frame's main data
/// begins in bytes of earlier main data slots that no earlier main data
/// used, within `main_data_begin`'s field and what the buffer leaves beside
/// the frame at its bitrate, padded (see
/// [`Reservoir::new`](super::reservoir::Reservoir::new)), and ends within
/// its own slot.
///
/// # Panics
///
/// If the stream breaks one of those rules, or does not end with a whole
/// frame.
pub fn frames(stream: &[u8]) -> Vec<Frame> {
    let mut frames = Vec::new();
    let mut at = 0;
    // Main data slot bytes before the frame, and the bits of main data
    // that end the main data before it.
    let (mut slots, mut used_bits) = (0, 0);
    while at < stream.len() {
        let header = &stream[at..at + 4];
        assert!(
            header[0] == 0xFF && header[1] & 0xE0 == 0xE0,
            "no sync word at {at}"
        );
        assert_eq!(header[1] >> 1 & 0b11, 0b01, "Layer III at {at}");
        let version = match header[1] >> 3 & 1 {
            1 => Version::Mpeg1,
            _ => Version::Mpeg2,
        };
        let family = match (version, header[1] & 0x10 != 0) {
            (Version::Mpeg1, _) => [44_100, 48_000, 32_000],
            (Version::Mpeg2, true) => [22_050, 24_000, 16_000],
            (Version::Mpeg2, false) => [11_025, 12_000, 8_000],
        };
        let hz = family[usize::from(header[2] >> 2 & 0b11)];
        let fs = SamplingFrequency::ALL
            .into_iter()
            .find(|fs| fs.hz() == hz)
            .unwrap();
        let bitrate = Bitrate::all(version)
            .nth(usize::from(header[2] >> 4) - 1)
            .expect("a bitrate of the table");
        let padding = header[2] >> 1 & 1 == 1;
        let len = bitrate.frame_len(fs, padding);
        let crc = if header[1] & 1 == 0 { 2 } else { 0 };
        let channels = if header[3] >> 6 == 0b11 { 1 } else { 2 };
        let side_info_len = match (version, channels) {
            (Version::Mpeg1, 1) => 17,
            (Version::Mpeg1, _) => 32,
            (Version::Mpeg2, 1) => 9,
            (Version::Mpeg2, _) => 17,
        };
        let side_info = &stream[at + 4 + crc..at + 4 + crc + side_info_len];
        let bits = |from: usize, count: usize| {
            (from..from + count).fold(0, |value, bit| {
                value << 1 | usize::from(side_info[bit / 8] >> (7 - bit % 8) & 1)
            })
        };
        // main_data_begin, private_bits and scfsi, then the fields of each
        // granule of each channel, part2_3_length first (clause 2.4.1.7, and
        // clause 2.4.1.7 of ISO/IEC 13818-3).
        let (begin_bits, before, per_channel) = match (version, channels) {
            (Version::Mpeg1, 1) => (9, 9 + 5 + 4, 59),
            (Version::Mpeg1, _) => (9, 9 + 3 + 8, 59),
            (Version::Mpeg2, 1) => (8, 8 + 1, 63),
            (Version::Mpeg2, _) => (8, 8 + 2, 63),
        };
        let main_data_begin = bits(0, begin_bits);
        let main_data_bits: usize = (0..version.granules() * channels)
            .map(|i| bits(before + i * per_channel, 12))
            .sum();

        let slot = len - 4 - crc - side_info_len;
        assert!(
            main_data_begin <= slots,
            "frame {}: main data begins before the stream",
            frames.len()
        );
        let begin = slots - main_data_begin;
        assert!(
            8 * begin >= used_bits,
            "frame {}: main data begins in bytes the frame before used",
            frames.len()
        );
        let largest = bitrate.frame_len(fs, true);
        assert!(
            8 * (main_data_begin + largest) <= BUFFER_BITS || main_data_begin == 0,
            "frame {}: {main_data_begin} bytes and a frame of {largest} overflow the buffer",
            frames.len()
        );
        used_bits = 8 * begin + main_data_bits;
        assert!(
            used_bits <= 8 * (slots + slot),
            "frame {}: main data ends beyond the frame",
            frames.len()
        );
        slots += slot;
        frames.push(Frame {
            bitrate,
            padding,
            len,
            main_data_begin,
        });
        at += len;
    }
    assert_eq!(at, stream.len(), "the stream ends within a frame");
    frames
}
