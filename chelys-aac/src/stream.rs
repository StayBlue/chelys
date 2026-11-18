// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The frames of a stream: one per block of 1024 samples of each channel,
//! each as long as the bit reservoir lets it be, framed as asked.

use chelys_core::bitstream::BitWriter;

use super::adts::{self, HEADER_LEN};
use super::channel_configuration::ChannelConfiguration;
use super::config::Framing;
use super::encoder::Output;
use super::raw_data_block;
use super::reservoir::BitReservoir;
use super::sampling_frequency::SamplingFrequency;

/// The samples of each channel a frame codes (clause 8.2.1.1).
pub const FRAME_SAMPLES: usize = 1024;

/// What the stream is: what its headers say, its rate and its framing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamConfig {
    pub sampling_frequency: SamplingFrequency,
    pub channel_configuration: ChannelConfiguration,
    pub bits_per_second: u32,
    pub framing: Framing,
}

impl StreamConfig {
    /// The bytes each frame adds before its `raw_data_block()`.
    fn header_len(&self) -> usize {
        match self.framing {
            Framing::Adts => HEADER_LEN,
            Framing::Raw => 0,
        }
    }

    /// The shortest frame: its channels silent, without fill.
    fn shortest_frame_len(&self) -> usize {
        self.header_len() + raw_data_block::silent_len(self.channel_configuration)
    }

    /// The lowest bitrate that brings each frame the bits of the shortest,
    /// so that the reservoir never runs below empty.
    pub fn min_bits_per_second(&self) -> u32 {
        let bits = 8 * self.shortest_frame_len() as u64;
        let sample_rate = u64::from(self.sampling_frequency.hz());
        (bits * sample_rate).div_ceil(FRAME_SAMPLES as u64) as u32
    }

    /// The highest bitrate, at which the reservoir holds a byte.
    pub fn max_bits_per_second(&self) -> u32 {
        BitReservoir::max_bits_per_second(
            self.sampling_frequency.hz(),
            self.channel_configuration.channels(),
        )
    }

    fn reservoir(&self) -> Option<BitReservoir> {
        if self.bits_per_second < self.min_bits_per_second() {
            return None;
        }
        BitReservoir::new(
            self.sampling_frequency.hz(),
            self.channel_configuration.channels(),
            self.bits_per_second,
        )
    }

    /// Whether every frame of a stream so configured can be written.
    pub fn is_valid(&self) -> bool {
        self.reservoir().is_some()
    }
}

/// Codes a stream frame by frame.
#[derive(Clone, Debug)]
pub struct StreamEncoder {
    config: StreamConfig,
    reservoir: BitReservoir,
    frames: usize,
}

impl StreamEncoder {
    /// # Panics
    ///
    /// If the configuration is not [valid](StreamConfig::is_valid).
    pub fn new(config: StreamConfig) -> Self {
        let reservoir = config
            .reservoir()
            .unwrap_or_else(|| panic!("a stream that cannot be written: {config:?}"));
        Self {
            config,
            reservoir,
            frames: 0,
        }
    }

    /// The frames coded so far.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Codes the frame whose window ends with `block`, the next
    /// [`FRAME_SAMPLES`] of each channel, and gives it to `out`.
    ///
    /// The encoder does not analyse its input yet: every frame codes each
    /// channel's spectrum as all zero, so the stream decodes to silence.
    pub fn encode<O: Output + ?Sized>(&mut self, block: &[&[f64]], out: &mut O) {
        debug_assert!(block.iter().all(|channel| channel.len() == FRAME_SAMPLES));
        let header_len = self.config.header_len();
        // As short as the reservoir allows: the silent channels, filled
        // only where the reservoir is full (clause 8.8.2.2).
        let (shortest, _) = self.reservoir.frame_len_range();
        let frame_len = shortest.max(self.config.shortest_frame_len());
        self.reservoir.spend(frame_len);
        let block =
            raw_data_block::silent(self.config.channel_configuration, frame_len - header_len);
        match self.config.framing {
            Framing::Adts => {
                let header = adts::Header {
                    sampling_frequency: self.config.sampling_frequency,
                    channel_configuration: self.config.channel_configuration,
                };
                let mut w = BitWriter::new();
                header.write(&mut w, frame_len, self.reservoir.adts_buffer_fullness());
                let mut frame = w.into_bytes();
                frame.extend_from_slice(&block);
                out.frame(&frame);
            }
            Framing::Raw => out.frame(&block),
        }
        self.frames += 1;
    }
}

/// A stream of `frames` silent frames, each apart, for tests.
#[cfg(any(test, feature = "unstable"))]
pub fn encode_silence(config: StreamConfig, frames: usize) -> Vec<Vec<u8>> {
    let silence = vec![0.0; FRAME_SAMPLES];
    let block = vec![silence.as_slice(); config.channel_configuration.channels()];
    let mut encoder = StreamEncoder::new(config);
    let mut out = Vec::new();
    for _ in 0..frames {
        encoder.encode(&block, &mut |frame: &[u8]| out.push(frame.to_vec()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(framing: Framing, bits_per_second: u32) -> StreamConfig {
        StreamConfig {
            sampling_frequency: SamplingFrequency::Hz44100,
            channel_configuration: ChannelConfiguration::Stereo,
            bits_per_second,
            framing,
        }
    }

    #[test]
    fn frames_keep_the_rate_once_the_reservoir_is_full() {
        for framing in [Framing::Adts, Framing::Raw] {
            let config = config(framing, 128_000);
            let frames = encode_silence(config, 200);
            let total: usize = frames.iter().map(Vec::len).sum();
            // What the rate brings, less what the reservoir holds at most.
            let mean_total = 200 * 128_000 * 1024 / 44_100 / 8;
            let max_bit_reservoir = 2 * 6144 - 128_000 * 1024 / 44_100;
            assert!(total <= mean_total, "{framing:?}");
            assert!(
                total + max_bit_reservoir / 8 + 1 >= mean_total,
                "{framing:?}"
            );
            // The first frames are the shortest, filling the reservoir.
            assert_eq!(frames[0].len(), config.shortest_frame_len(), "{framing:?}");
        }
    }

    #[test]
    fn adts_frames_give_their_length_and_the_reservoir() {
        let config = config(Framing::Adts, 128_000);
        let frames = encode_silence(config, 200);
        let mut reservoir = config.reservoir().unwrap();
        for frame in &frames {
            reservoir.spend(frame.len());
            let length = (usize::from(frame[3] & 0b11) << 11)
                | (usize::from(frame[4]) << 3)
                | usize::from(frame[5] >> 5);
            let fullness = (u32::from(frame[5] & 0b1_1111) << 6) | u32::from(frame[6] >> 2);
            assert_eq!(length, frame.len());
            assert_eq!(fullness, reservoir.adts_buffer_fullness());
        }
        assert!(frames.last().unwrap().len() > config.shortest_frame_len());
    }

    #[test]
    fn the_bitrate_bounds_admit_every_frame() {
        for fs in SamplingFrequency::ALL {
            for channel_configuration in [ChannelConfiguration::Mono, ChannelConfiguration::Stereo]
            {
                for framing in [Framing::Adts, Framing::Raw] {
                    let mut config = StreamConfig {
                        sampling_frequency: fs,
                        channel_configuration,
                        bits_per_second: 0,
                        framing,
                    };
                    let (min, max) = (config.min_bits_per_second(), config.max_bits_per_second());
                    for bits_per_second in [min - 1, min, max, max + 1] {
                        config.bits_per_second = bits_per_second;
                        let valid = (min..=max).contains(&bits_per_second);
                        assert_eq!(config.is_valid(), valid, "{config:?}");
                        if valid {
                            encode_silence(config, 50);
                        }
                    }
                }
            }
        }
    }
}
