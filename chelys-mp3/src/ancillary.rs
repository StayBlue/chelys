// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Ancillary data (clauses 2.4.1.7 and 2.4.2.8): "user definable" bits that
//! a frame carries after its audio. In Layer III they end its main data,
//! after the Huffman code bits of every granule and channel, and run to
//! where the next frame's `main_data_begin` points, so that they "will
//! reduce the number of bits available for audio" (clause C.1.5.3.6).

use std::collections::VecDeque;

use crate::bitstream::BitWriter;

/// The ancillary data queued for frames not yet coded, and how many of its
/// bits each frame may carry.
#[derive(Debug, Default)]
pub(super) struct AncillaryQueue {
    /// The most bits a frame carries.
    bits_per_frame: usize,
    bytes: VecDeque<u8>,
    /// The bits of the front byte that frames have carried already.
    taken: u32,
}

impl AncillaryQueue {
    pub(super) fn push(&mut self, bytes: &[u8]) {
        self.bytes.extend(bytes);
    }

    /// Lets each frame from the next on carry up to `bits` bits.
    pub(super) fn set_bits_per_frame(&mut self, bits: usize) {
        self.bits_per_frame = bits;
    }

    /// The bits queued that no frame has carried.
    pub(super) fn pending(&self) -> usize {
        8 * self.bytes.len() - self.taken as usize
    }

    /// The bits the next frame carries: as many as are queued, up to the
    /// frame's share.
    pub(super) fn next_frame_bits(&self) -> usize {
        self.pending().min(self.bits_per_frame)
    }

    /// Writes the first `bits` queued bits to `w`, each byte's most
    /// significant bit first, and takes them from the queue.
    ///
    /// Spec gap: the standard sends ancillary data as single
    /// `ancillary_bit`s, and has no bytes of it. A byte is taken as a bit
    /// string of eight, its most significant bit to the left, and sent left
    /// bit first as every `bslbf` field is (clause 2.2.6).
    ///
    /// # Panics
    ///
    /// If fewer than `bits` bits are queued.
    pub(super) fn write(&mut self, bits: usize, w: &mut BitWriter) {
        assert!(
            bits <= self.pending(),
            "{bits} bits of ancillary data are not queued"
        );
        for _ in 0..bits {
            let byte = self.bytes[0];
            w.write(u32::from((byte >> (7 - self.taken)) & 1), 1);
            self.taken += 1;
            if self.taken == 8 {
                self.bytes.pop_front();
                self.taken = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_take_the_bits_in_order() {
        let mut queue = AncillaryQueue::default();
        queue.set_bits_per_frame(12);
        queue.push(&[0xA5, 0x0F]);
        queue.push(&[0xC3]);
        assert_eq!(queue.pending(), 24);
        let mut w = BitWriter::new();
        for _ in 0..3 {
            let bits = queue.next_frame_bits();
            queue.write(bits, &mut w);
        }
        assert_eq!(queue.pending(), 0);
        assert_eq!(queue.next_frame_bits(), 0);
        assert_eq!(w.into_bytes(), [0xA5, 0x0F, 0xC3]);
    }
}
