// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The bit reservoir of a stream at a constant rate (clause 8.2.3.2): how
//! far each frame may fall short of the mean length, or run past it, and
//! the state ADTS reports as `adts_buffer_fullness`.

/// The bits of the decoder's input buffer per channel (clause 8.2.3.1).
const BUFFER_BITS_PER_CHANNEL: u64 = 6144;

/// The samples of each channel a frame codes.
const FRAME_SAMPLES: u64 = 1024;

/// The state of the bit reservoir, kept exactly: in bits times the
/// sampling frequency, as the mean length of a frame is a fraction of a
/// bit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitReservoir {
    sample_rate: u64,
    channels: u64,
    /// `mean_framelength`, times the sampling frequency.
    mean: u64,
    /// `max_bit_reservoir`, times the sampling frequency.
    max: u64,
    /// `bit_reservoir_state`, times the sampling frequency.
    state: u64,
}

impl BitReservoir {
    /// The reservoir of a stream of `channels` (the NCC) at
    /// `bits_per_second`, if a frame can be given its mean length within a
    /// byte: if `max_bit_reservoir` is 8 bits at least.
    ///
    /// Spec reading: the maximum bitrate (clause 8.2.3.3) is that of a
    /// frame as long as the decoder's input buffer, where
    /// `max_bit_reservoir` is 0 and every frame must be exactly the mean
    /// length, which is a fraction of a bit where the rate does not divide
    /// evenly. Frames are whole bytes, so the reservoir is kept a byte at
    /// least, and the highest bitrate is below the maximum by a byte a
    /// frame.
    ///
    /// The mean is that of the stream as it is written: a stream in ADTS
    /// counts its headers, as the decoder's input buffer holds them too
    /// (clause 8.2.3.1).
    ///
    /// Spec gap: the standard does not give the state before the first
    /// frame. It is taken as empty, so that the first frame is no longer
    /// than the mean, and no frame needs more bits than the rate has
    /// brought by its end.
    pub fn new(sample_rate: u32, channels: usize, bits_per_second: u32) -> Option<Self> {
        let sample_rate = u64::from(sample_rate);
        let channels = channels as u64;
        let mean = u64::from(bits_per_second) * FRAME_SAMPLES;
        let buffer = BUFFER_BITS_PER_CHANNEL * channels * sample_rate;
        // max_bit_reservoir = INT(buffer - mean_framelength), as in the
        // example of clause 8.2.3.2.
        let max_bits = buffer.checked_sub(mean)? / sample_rate;
        (max_bits >= 8).then_some(Self {
            sample_rate,
            channels,
            mean,
            max: max_bits * sample_rate,
            state: 0,
        })
    }

    /// The highest bitrate whose reservoir holds a byte, by [`Self::new`].
    pub fn max_bits_per_second(sample_rate: u32, channels: usize) -> u32 {
        let buffer = BUFFER_BITS_PER_CHANNEL * channels as u64;
        ((buffer - 8) * u64::from(sample_rate) / FRAME_SAMPLES) as u32
    }

    /// The lengths in bytes the next frame may have, from the shortest
    /// that keeps the reservoir within `max_bit_reservoir` to the longest
    /// that keeps it from running below empty (clause 8.2.3.2):
    ///
    /// `0 ≤ bit_reservoir_state[frame] ≤ max_bit_reservoir`, where
    /// `bit_reservoir_state[frame] = bit_reservoir_state[frame - 1] +
    /// mean_framelength - framelength[frame]`.
    pub fn frame_len_range(&self) -> (usize, usize) {
        let available = self.state + self.mean;
        let byte = 8 * self.sample_rate;
        let shortest = available.saturating_sub(self.max).div_ceil(byte);
        let longest = available / byte;
        (shortest as usize, longest as usize)
    }

    /// Takes a frame of `len` bytes, within [`Self::frame_len_range`].
    pub fn spend(&mut self, len: usize) {
        let (shortest, longest) = self.frame_len_range();
        assert!(
            (shortest..=longest).contains(&len),
            "a frame of {len} bytes, not {shortest} to {longest}"
        );
        self.state = self.state + self.mean - 8 * len as u64 * self.sample_rate;
    }

    /// The state after the last frame spent, in whole bits.
    #[cfg(any(test, feature = "unstable"))]
    pub fn state_bits(&self) -> u64 {
        self.state / self.sample_rate
    }

    /// `max_bit_reservoir`.
    #[cfg(any(test, feature = "unstable"))]
    pub fn max_bits(&self) -> u64 {
        self.max / self.sample_rate
    }

    /// `adts_buffer_fullness`: the state "divided by NCC divided by 32 and
    /// truncated to an integer value" (clause 8.1.3.2.2).
    ///
    /// Spec reading: the state is that after the frame the header is in,
    /// its bits spent: "the state of the bit reservoir in the course of
    /// encoding the ADTS frame, up to and including the first
    /// raw_data_block()", as `adif_buffer_fullness` is "the state of the
    /// bit reservoir after encoding the first raw_data_block()".
    pub fn adts_buffer_fullness(&self) -> u32 {
        (self.state / self.sample_rate / self.channels / 32) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_maximum_follows_the_example_of_the_standard() {
        // 96 kbit/s, stereo, 44.1 kHz: mean_framelength 2229.1156...,
        // max_bit_reservoir 10058 (clause 8.2.3.2).
        let reservoir = BitReservoir::new(44_100, 2, 96_000).unwrap();
        assert_eq!(reservoir.max_bits(), 10_058);
        assert_eq!(reservoir.frame_len_range(), (0, 278));
    }

    #[test]
    fn short_frames_fill_the_reservoir_to_its_maximum() {
        let mut reservoir = BitReservoir::new(48_000, 1, 64_000).unwrap();
        // mean_framelength 1365.33... bits.
        let mut total = 0;
        for _ in 0..100 {
            let (shortest, _) = reservoir.frame_len_range();
            let len = shortest.max(10);
            reservoir.spend(len);
            total += 8 * len as u64;
            assert!(reservoir.state_bits() <= reservoir.max_bits());
        }
        let mean_total = 100 * 64_000 * 1024 / 48_000;
        assert_eq!(total + reservoir.state_bits(), mean_total);
        assert!(reservoir.max_bits() - reservoir.state_bits() <= 8);
        assert_eq!(
            reservoir.adts_buffer_fullness(),
            (reservoir.state_bits() / 32) as u32
        );
    }

    #[test]
    fn bitrates_up_to_the_buffer_less_a_byte() {
        for (sample_rate, channels) in [(8_000, 1), (44_100, 2), (96_000, 2)] {
            let max = BitReservoir::max_bits_per_second(sample_rate, channels);
            let reservoir = BitReservoir::new(sample_rate, channels, max).unwrap();
            assert!(reservoir.max_bits() >= 8);
            assert!(BitReservoir::new(sample_rate, channels, max + 1).is_none());
            let (shortest, longest) = reservoir.frame_len_range();
            assert!(shortest <= longest);
        }
        assert!(BitReservoir::new(8_000, 1, u32::MAX).is_none());
    }
}
