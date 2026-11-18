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
    /// Where the frame's main data slot lies in the stream.
    slot: (usize, usize),
    /// Where the frame's main data begins among the bytes of all slots.
    begin: usize,
    /// The bits of its scalefactors and Huffman code.
    pub audio_bits: usize,
    /// The bytes of earlier main data slots after the main data before it,
    /// where its main data could have begun.
    pub free_before: usize,
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
/// frame, or has a frame in free format.
pub fn frames(stream: &[u8]) -> Vec<Frame> {
    walk(stream, None)
}

/// The frames of a stream in free format at `bitrate`, after the same
/// checks as [`frames`], which a decoder learns from the distance between
/// headers.
///
/// # Panics
///
/// As [`frames`], or if a frame is not in free format.
pub fn free_frames(stream: &[u8], bitrate: Bitrate) -> Vec<Frame> {
    assert!(bitrate.is_free(), "{bitrate:?} is not free format");
    walk(stream, Some(bitrate))
}

fn walk(stream: &[u8], free: Option<Bitrate>) -> Vec<Frame> {
    let mut frames = Vec::new();
    let mut at = 0;
    // Main data slot bytes before the frame, and the bits of main data
    // that end the main data before it.
    let (mut slots, mut used_bits) = (0, 0usize);
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
        let bitrate = match (header[2] >> 4, free) {
            (0, Some(free)) => free,
            (0, None) => panic!("free format at {at}"),
            (_, Some(_)) => panic!("not free format at {at}"),
            (index, None) => Bitrate::all(version)
                .nth(usize::from(index) - 1)
                .expect("a bitrate of the table"),
        };
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
        let free_before = slots - used_bits.div_ceil(8);
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
            slot: (at + len - slot, at + len),
            begin,
            audio_bits: main_data_bits,
            free_before,
        });
        at += len;
    }
    assert_eq!(at, stream.len(), "the stream ends within a frame");
    frames
}

/// The ancillary bits of each frame of `stream`, after the checks of
/// [`frames`]: from the end of its Huffman code bits to where the next
/// frame's main data begins, or for the last frame to the end of the stream
/// (clause 2.4.2.8).
pub fn ancillary(stream: &[u8]) -> Vec<Vec<bool>> {
    let frames = frames(stream);
    let slots: Vec<u8> = frames
        .iter()
        .flat_map(|f| stream[f.slot.0..f.slot.1].iter().copied())
        .collect();
    let bit = |at: usize| slots[at / 8] >> (7 - at % 8) & 1 == 1;
    let ends = frames.iter().skip(1).map(|f| 8 * f.begin);
    frames
        .iter()
        .zip(ends.chain([8 * slots.len()]))
        .map(|(f, end)| (8 * f.begin + f.audio_bits..end).map(bit).collect())
        .collect()
}
