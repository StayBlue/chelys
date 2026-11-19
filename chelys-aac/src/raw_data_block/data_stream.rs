// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The `data_stream_element()` (Table 24, clause 8.6): bytes of data that
//! are "not part of the audio information itself", which a frame carries
//! beside its audio and which take bits from it; and the bytes queued for
//! the frames to carry.

use std::collections::VecDeque;

use chelys_core::bitstream::BitWriter;

use super::{ID_DSE, Protected};

/// The `element_instance_tag` of every `data_stream_element()`.
///
/// Spec gap: the standard leaves the tag of a data stream to the encoder,
/// asking only that the elements of one stream share it (clause 8.2.4,
/// "All data_stream_element()'s have the same element_instance_tag if they
/// are part of the same data stream", and clause 8.2.1.1), and exempting
/// data stream elements from the tags' being unique in a block (clause
/// 8.2.2). The encoder sends one stream, which is given the first tag.
/// Settled by decoding with ffmpeg and FAAD2 (black box): they decode
/// streams whose data stream elements are so tagged, after the audio
/// element, without complaint, and their audio the same whatever the data
/// they carry; the conformance tests are
/// `ffmpeg_ignores_data_stream_elements` and
/// `faad_ignores_data_stream_elements`.
pub const ELEMENT_INSTANCE_TAG: u32 = 0;

/// The most `data_stream_byte`s one element holds: a `count` of 255 and an
/// `esc_count` of 255 (Table 24; clause 8.6.3, "If count equals 255, its
/// value is incremented by a second 8 bit value, esc_count, this final
/// value represents the number of bytes"). More bytes take more elements,
/// of which a block may hold any number (clause 8.2.1.1).
const MAX_BYTES: usize = 255 + 255;

/// The bits of an element of `bytes` bytes, with its `id_syn_ele`:
/// `id_syn_ele`, `element_instance_tag`, `data_byte_align_flag`, `count`,
/// `esc_count` from 255 bytes, and the bytes.
fn element_bits(bytes: usize) -> u32 {
    let esc_count = if bytes >= 255 { 8 } else { 0 };
    3 + 4 + 1 + 8 + esc_count + 8 * bytes as u32
}

/// The bits of the elements that carry `bytes` bytes, as [`write()`] writes
/// them: none for none.
pub fn bits(bytes: usize) -> u32 {
    let full = (bytes / MAX_BYTES) as u32 * element_bits(MAX_BYTES);
    match bytes % MAX_BYTES {
        0 => full,
        rest => full + element_bits(rest),
    }
}

/// Writes `bytes` in as few `data_stream_element()`s as hold them, each
/// with its `id_syn_ele`, in the bits [`bits`] counts; and gives the bits
/// of each that an `adts_error_check()` protects: "All information in any
/// ... data_stream_element()" (clause 8.1.3.2.1), its `id_syn_ele` aside.
///
/// Spec reading: the clause's "The id_syn_ele bits shall be excluded from
/// CRC protection" follows its sentences on the channel elements, but is
/// not limited to them: a data stream element's `id_syn_ele` is excluded
/// too, and its other bits are protected, from `element_instance_tag` on.
///
/// The bytes are not aligned: `data_byte_align_flag` is 0, which leaves
/// out the `byte_alignment()` that would cost up to 7 bits an element.
pub fn write(w: &mut BitWriter, bytes: &[u8]) -> Vec<Protected> {
    let mut protected = Vec::new();
    for chunk in bytes.chunks(MAX_BYTES) {
        let start = w.bit_len();
        w.write(ID_DSE, 3);
        let element_start = w.bit_len();
        w.write(ELEMENT_INSTANCE_TAG, 4);
        w.write(0, 1); // data_byte_align_flag
        if chunk.len() < 255 {
            w.write(chunk.len() as u32, 8); // count
        } else {
            w.write(255, 8); // count
            w.write((chunk.len() - 255) as u32, 8); // esc_count
        }
        for &byte in chunk {
            w.write(u32::from(byte), 8); // data_stream_byte
        }
        debug_assert_eq!(w.bit_len() - start, element_bits(chunk.len()) as usize);
        protected.push(Protected {
            bits: element_start..w.bit_len(),
            len: None,
        });
    }
    protected
}

/// The bytes queued for frames not yet coded, and how many of them each
/// frame may carry.
#[derive(Debug, Default)]
pub struct DataStreamQueue {
    /// The most bytes a frame carries.
    bytes_per_frame: usize,
    bytes: VecDeque<u8>,
}

impl DataStreamQueue {
    /// A queue from which each frame takes up to `bytes_per_frame` bytes.
    pub fn new(bytes_per_frame: usize) -> Self {
        Self {
            bytes_per_frame,
            bytes: VecDeque::new(),
        }
    }

    pub fn push(&mut self, bytes: &[u8]) {
        self.bytes.extend(bytes);
    }

    /// The bytes queued that no frame has carried.
    pub fn pending(&self) -> usize {
        self.bytes.len()
    }

    /// The bytes the next frame carries: as many as are queued, up to the
    /// frame's share.
    pub fn next_frame_len(&self) -> usize {
        self.pending().min(self.bytes_per_frame)
    }

    /// Takes the next frame's bytes from the queue.
    pub fn take_next_frame(&mut self) -> Vec<u8> {
        let len = self.next_frame_len();
        self.bytes.drain(..len).collect()
    }
}

#[cfg(test)]
mod tests {
    use chelys_core::bitstream::binary_digits;

    use super::*;

    fn written(bytes: &[u8]) -> (String, Vec<Protected>) {
        let mut w = BitWriter::new();
        let protected = write(&mut w, bytes);
        let len = w.bit_len();
        assert_eq!(len, bits(bytes.len()) as usize);
        (binary_digits(&w.into_bytes())[..len].to_owned(), protected)
    }

    #[test]
    fn no_bytes_take_no_element() {
        assert_eq!(written(&[]), (String::new(), Vec::new()));
    }

    #[test]
    fn an_element_spells_its_fields() {
        let (bits, protected) = written(&[0xA5, 0x0F]);
        // id_syn_ele ID_DSE, element_instance_tag, data_byte_align_flag,
        // count, then the bytes.
        let fields = ["100", "0000", "0", "00000010", "10100101", "00001111"];
        assert_eq!(bits, fields.concat());
        assert_eq!(
            protected,
            [Protected {
                bits: 3..bits.len(),
                len: None
            }]
        );
    }

    #[test]
    fn count_escapes_from_255_bytes() {
        let (bits, _) = written(&[0xFF; 254]);
        assert_eq!(&bits[8..16], "11111110");
        assert_eq!(bits.len(), 16 + 8 * 254);
        for (len, esc_count) in [(255, "00000000"), (300, "00101101"), (510, "11111111")] {
            let (bits, protected) = written(&vec![0x5A; len]);
            assert_eq!(&bits[8..24], ["11111111", esc_count].concat(), "{len}");
            assert_eq!(bits.len(), 24 + 8 * len, "{len}");
            assert_eq!(protected.len(), 1);
        }
    }

    #[test]
    fn more_bytes_than_an_element_holds_take_more() {
        let bytes: Vec<u8> = (0..1100).map(|i| i as u8).collect();
        let (bits, protected) = written(&bytes);
        // 510, 510 and 80 bytes.
        let first = 24 + 8 * 510;
        assert_eq!(bits.len(), 2 * first + 16 + 8 * 80);
        assert_eq!(&bits[first..first + 24], "100000001111111111111111");
        assert_eq!(&bits[2 * first..2 * first + 16], "1000000001010000");
        assert_eq!(
            protected.iter().map(|p| p.bits.clone()).collect::<Vec<_>>(),
            [3..first, first + 3..2 * first, 2 * first + 3..bits.len()]
        );
    }

    #[test]
    fn frames_take_the_bytes_in_order_up_to_their_share() {
        let mut queue = DataStreamQueue::new(3);
        queue.push(&[1, 2, 3, 4]);
        queue.push(&[5]);
        assert_eq!(queue.pending(), 5);
        assert_eq!(queue.take_next_frame(), [1, 2, 3]);
        assert_eq!(queue.next_frame_len(), 2);
        assert_eq!(queue.take_next_frame(), [4, 5]);
        assert_eq!(queue.take_next_frame(), []);
        assert_eq!(queue.pending(), 0);
    }
}
