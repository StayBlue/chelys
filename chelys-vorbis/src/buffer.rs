// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The buffer of a stream at a constant rate (see
//! [`Rate::Constant`](crate::Rate::Constant)): how far its audio packets
//! may run ahead of the rate, or fall behind it.
//!
//! The specification describes no encoder, and no buffer: Vorbis has no
//! bit reservoir of its own, as AAC has, and its packets may be of any
//! length (clause 1.1.3). The buffer is chelys's own design. It counts the
//! audio packets alone, as the library writes no container, and times them
//! by their granule positions, which are exact.

use super::setup::BLOCKSIZES;

/// The bits the buffer keeps after every packet but the last: two bytes,
/// the longest packet that decodes to silence, every channel unused, of up
/// to 8 channels (see [`audio::packet`](super::audio::packet)). So the
/// next packet can always be one, whatever the buffer held: each but the
/// first passes 128 samples at least, a short block's hop, for which the
/// rate brings 16 bits at least.
const KEPT_BITS: u64 = 16;

/// The state of the buffer, kept exactly: in bits times the sample rate, as
/// the bits the rate brings for a sample are a fraction of a bit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Buffer {
    sample_rate: u64,
    bits_per_second: u64,
    /// The bits the buffer holds at most, times the sample rate.
    size: u64,
    /// The bits it holds, times the sample rate.
    held: u64,
    /// The granule position of the last packet spent, 0 before the first.
    granule_position: u64,
}

impl Buffer {
    /// The buffer of a stream at `bits_per_second` and `sample_rate`: of a
    /// second of the rate, or of the samples of a long block where they are
    /// more, half full.
    pub(crate) fn new(bits_per_second: u32, sample_rate: u32) -> Self {
        let sample_rate = u64::from(sample_rate);
        let bits_per_second = u64::from(bits_per_second);
        let size = bits_per_second * sample_rate.max(BLOCKSIZES[1] as u64);
        Self {
            sample_rate,
            bits_per_second,
            size,
            held: size / 2,
            granule_position: 0,
        }
    }

    /// The bits the buffer would hold, times the sample rate, once the rate
    /// has brought those of the samples up to `granule_position`, before
    /// the packet's bits are taken.
    fn filled(&self, granule_position: u64) -> u64 {
        let samples = granule_position
            .checked_sub(self.granule_position)
            .expect("granule positions in order");
        self.held + self.bits_per_second * samples
    }

    /// The lengths in bytes the packet of `granule_position` may have: from
    /// the shortest that keeps the buffer within its size to the longest
    /// that keeps it from running below empty, or below [`KEPT_BITS`] if
    /// the packet is not the `last`. The shortest is never longer than the
    /// longest, as the buffer is longer than three bytes, and the longest
    /// is two bytes at least.
    pub(crate) fn len_range(&self, granule_position: u64, last: bool) -> (usize, usize) {
        let filled = self.filled(granule_position);
        let byte = 8 * self.sample_rate;
        let kept = if last {
            0
        } else {
            KEPT_BITS * self.sample_rate
        };
        let shortest = filled.saturating_sub(self.size).div_ceil(byte);
        let longest = filled.saturating_sub(kept) / byte;
        (shortest as usize, longest as usize)
    }

    /// Takes a packet of `len` bytes, of `granule_position`, within
    /// [`Self::len_range`].
    pub(crate) fn spend(&mut self, len: usize, granule_position: u64) {
        let filled = self.filled(granule_position);
        let bits = 8 * len as u64 * self.sample_rate;
        assert!(
            bits <= filled && filled - bits <= self.size,
            "a packet of {len} bytes past the buffer"
        );
        self.held = filled - bits;
        self.granule_position = granule_position;
    }

    /// The share of its size the buffer holds, from 0 to 1.
    pub(crate) fn fullness(&self) -> f64 {
        self.held as f64 / self.size as f64
    }

    /// The bits the buffer holds at most.
    #[cfg(test)]
    pub(crate) fn size_bits(&self) -> f64 {
        self.size as f64 / self.sample_rate as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A second of the rate, half full: at 128 kbit/s, 8000 bytes, of which
    /// a long block's packet, 1024 samples on, may take those and the 371.5
    /// the rate brings, but for the two kept.
    #[test]
    fn the_buffer_holds_a_second_half_full() {
        let buffer = Buffer::new(128_000, 44_100);
        assert_eq!(buffer.size_bits(), 128_000.0);
        assert_eq!(buffer.fullness(), 0.5);
        // The first packet, of granule position 0, has no samples.
        assert_eq!(buffer.len_range(0, false), (0, 7998));
        assert_eq!(buffer.len_range(1024, false), (0, 8369));
        assert_eq!(buffer.len_range(1024, true), (0, 8371));
    }

    /// Below 2048 Hz, the buffer is of a long block's samples.
    #[test]
    fn below_2048_hz_the_buffer_holds_a_long_block() {
        assert_eq!(Buffer::new(1_000, 1_000).size_bits(), 2_048.0);
        assert_eq!(Buffer::new(1, 1).size_bits(), 2_048.0);
        assert_eq!(Buffer::new(6_000, 2_048).size_bits(), 6_000.0);
    }

    /// Packets of the longest length keep it from running below what it
    /// keeps, and those of the shortest, padded, from running over; the
    /// bits taken are those the rate brought, less what the buffer then
    /// holds beyond what it held.
    #[test]
    fn the_buffer_keeps_within_its_bounds() {
        // 96 kbit/s in stereo at 44.1 kHz: 2229.1156... bits a hop.
        for take_longest in [true, false] {
            let mut buffer = Buffer::new(96_000, 44_100);
            let mut taken = 0;
            for k in 0..200 {
                let granule_position = 1024 * k;
                let (shortest, longest) = buffer.len_range(granule_position, false);
                assert!(shortest <= longest && longest >= 2);
                let len = if take_longest { longest } else { shortest };
                buffer.spend(len, granule_position);
                taken += 8 * len as u64;
                let held = buffer.held / buffer.sample_rate;
                if take_longest {
                    assert!((16..24).contains(&held), "{held}");
                }
            }
            if !take_longest {
                // Full, but for less than a byte.
                assert!(buffer.size - buffer.held < 8 * 44_100);
            }
            let brought = 96_000 * 199 * 1024 / 44_100;
            let held = buffer.held / buffer.sample_rate;
            assert!((taken + held).abs_diff(brought + 48_000) <= 1);
        }
    }

    /// The last packet may take all the buffer holds; one of a granule
    /// position as the packet before it, as the first is of 0, none of
    /// what the rate brings.
    #[test]
    fn the_last_packet_may_empty_the_buffer() {
        let mut buffer = Buffer::new(64_000, 48_000);
        let (_, longest) = buffer.len_range(0, false);
        buffer.spend(longest, 0);
        assert_eq!(buffer.len_range(0, true), (0, 2));
        assert_eq!(buffer.len_range(0, false), (0, 0));
        buffer.spend(2, 0);
        assert_eq!(buffer.held, 0);
    }

    /// At the lowest rate, 1/8 of a bit a sample in mono, a short block's
    /// 128 samples bring 16 bits, so that the buffer, kept at 16, lets the
    /// next packet take two bytes.
    #[test]
    fn at_the_lowest_rate_a_short_block_may_take_two_bytes() {
        for sample_rate in [1u32, 8_000, 44_100, 192_000] {
            let bits_per_second = sample_rate.div_ceil(8);
            let mut buffer = Buffer::new(bits_per_second, sample_rate);
            let (_, longest) = buffer.len_range(0, false);
            buffer.spend(longest, 0);
            for k in 1..100 {
                let (_, longest) = buffer.len_range(128 * k, false);
                assert!(longest >= 2, "{sample_rate} Hz");
                buffer.spend(longest, 128 * k);
            }
        }
    }

    #[test]
    #[should_panic(expected = "a packet of 8001 bytes past the buffer")]
    fn a_packet_past_the_buffer_panics() {
        Buffer::new(128_000, 44_100).spend(8001, 0);
    }
}
