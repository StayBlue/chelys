// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The frames of a stream: one per block of 1024 samples of each channel,
//! each as long as the bit reservoir lets it be, framed as asked.

mod frame;

use super::DELAY;
use super::adts::HEADER_LEN;
use super::audio_specific_config::audio_specific_config;
use super::channel::CodedChannel;
use super::channel_configuration::ChannelConfiguration;
use super::config::Framing;
use super::filterbank::Filterbank;
use super::loops;
use super::raw_data_block;
use super::reservoir::BitReservoir;
use super::sampling_frequency::SamplingFrequency;
use super::window::Windows;

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

    /// The coded channels of a frame of silence.
    fn silent_channels(&self) -> Vec<CodedChannel> {
        let channel = CodedChannel::silent(Windows::LONG_SINE, self.sampling_frequency);
        vec![channel; self.channel_configuration.channels()]
    }

    /// The shortest frame: its channels silent, without fill.
    fn shortest_frame_len(&self) -> usize {
        self.header_len()
            + raw_data_block::min_len(self.channel_configuration, &self.silent_channels())
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
        BitReservoir::max_bits_per_second(self.sampling_frequency, self.channel_configuration)
    }

    fn reservoir(&self) -> Option<BitReservoir> {
        if self.bits_per_second < self.min_bits_per_second() {
            return None;
        }
        BitReservoir::new(
            self.sampling_frequency,
            self.channel_configuration,
            self.bits_per_second,
        )
    }

    /// Whether every frame of a stream so configured can be written.
    pub fn is_valid(&self) -> bool {
        self.reservoir().is_some()
    }
}

/// Where an [`Encoder`](crate::Encoder) gives the frames it codes, one call each, in their
/// order.
pub trait FrameSink {
    /// Takes a frame: an `adts_frame()` or a `raw_data_block()`, as
    /// [`Config::framing`](crate::Config::framing) asked.
    fn frame(&mut self, frame: &[u8]);
}

/// Appends each frame, as to a stream in ADTS.
impl FrameSink for Vec<u8> {
    fn frame(&mut self, frame: &[u8]) {
        self.extend_from_slice(frame);
    }
}

/// Calls the closure with each frame, as to keep raw frames apart:
///
/// ```
/// # use chelys_aac::{Config, Encoder, Framing, Rate};
/// let config = Config::new(48_000, 1, Rate::Constant { bits_per_second: 64_000 })
///     .framing(Framing::Raw);
/// let mut encoder = Encoder::new(config).unwrap();
/// let mut frames = Vec::new();
/// let mut out = |frame: &[u8]| frames.push(frame.to_vec());
/// encoder.encode_interleaved(&[0i16; 4096], &mut out);
/// encoder.finish(&mut out);
/// assert_eq!(frames.len(), 5);
/// ```
impl<F: FnMut(&[u8])> FrameSink for F {
    fn frame(&mut self, frame: &[u8]) {
        self(frame);
    }
}

/// Codes a stream from PCM as it arrives, in any lengths, a frame for each
/// block of [`FRAME_SAMPLES`] of each channel.
#[derive(Debug)]
pub struct StreamEncoder {
    config: StreamConfig,
    reservoir: BitReservoir,
    /// Each channel's samples that no frame has coded yet: fewer than a
    /// block's between calls.
    pcm: Vec<Vec<f64>>,
    /// One for each channel.
    filterbanks: Vec<Filterbank>,
    /// The samples each channel was given.
    samples: u64,
    frames: u64,
}

/// What a [`StreamEncoder`] coded, once it has finished.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamEnd {
    pub frames: u64,
    /// The samples each channel was given.
    pub samples: u64,
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
            pcm: vec![Vec::with_capacity(FRAME_SAMPLES); config.channel_configuration.channels()],
            filterbanks: vec![Filterbank::new(); config.channel_configuration.channels()],
            samples: 0,
            frames: 0,
        }
    }

    /// The `AudioSpecificConfig()` of ISO/IEC 14496-3 that describes the
    /// stream.
    pub fn audio_specific_config(&self) -> Vec<u8> {
        audio_specific_config(
            self.config.sampling_frequency,
            self.config.channel_configuration,
        )
    }

    /// Takes `channels`, the next samples of each channel at full scale
    /// 1.0, and gives `out` the frames they complete.
    ///
    /// # Panics
    ///
    /// If there is not one slice for each channel, all of one length: a
    /// caller's error.
    pub fn encode<O: FrameSink + ?Sized>(&mut self, channels: &[&[f64]], out: &mut O) {
        assert_eq!(channels.len(), self.pcm.len(), "one slice per channel");
        let len = channels[0].len();
        assert!(
            channels.iter().all(|c| c.len() == len),
            "every channel has the same number of samples"
        );
        for (pcm, samples) in self.pcm.iter_mut().zip(channels) {
            pcm.extend_from_slice(samples);
        }
        self.samples += len as u64;
        let mut start = 0;
        while self.pcm[0].len() - start >= FRAME_SAMPLES {
            self.encode_frame(start, out);
            start += FRAME_SAMPLES;
        }
        for pcm in &mut self.pcm {
            pcm.drain(..start);
        }
    }

    /// Ends the stream: pads the input with silence to whole frames, at
    /// least [`DELAY`] samples past it, so that the decoded output holds
    /// all of the input, and gives `out` the frames left.
    pub fn finish<O: FrameSink + ?Sized>(mut self, out: &mut O) -> StreamEnd {
        let frames = (self.samples + DELAY as u64).div_ceil(FRAME_SAMPLES as u64);
        // Few: those of the samples held and of DELAY.
        let remaining = (frames - self.frames) as usize;
        for pcm in &mut self.pcm {
            pcm.resize(remaining * FRAME_SAMPLES, 0.0);
        }
        for frame in 0..remaining {
            self.encode_frame(frame * FRAME_SAMPLES, out);
        }
        StreamEnd {
            frames: self.frames,
            samples: self.samples,
        }
    }

    /// Codes the frame whose block starts at `start` in [`Self::pcm`], and
    /// gives it to `out`: each channel through its filterbank, and its
    /// lines through the iteration loops, in the bits the reservoir makes
    /// available.
    fn encode_frame<O: FrameSink + ?Sized>(&mut self, start: usize, out: &mut O) {
        let windows = Windows::LONG_SINE;
        let available_bits = frame::bits_per_channel(&self.config, &self.reservoir);
        let channels: Vec<CodedChannel> = (self.pcm.iter())
            .zip(&mut self.filterbanks)
            .map(|(pcm, filterbank)| {
                let block = pcm[start..start + FRAME_SAMPLES]
                    .try_into()
                    .expect("a block of samples");
                let spectrum = filterbank.spectrum(block, windows);
                loops::quantize_channel(
                    &spectrum,
                    windows,
                    self.config.sampling_frequency,
                    available_bits,
                )
            })
            .collect();
        frame::write(&self.config, &mut self.reservoir, &channels, out);
        self.frames += 1;
    }
}

/// A stream of `frames` silent frames, each apart, for tests.
#[cfg(any(test, feature = "unstable"))]
pub fn encode_silence(config: StreamConfig, frames: usize) -> Vec<Vec<u8>> {
    let silence = vec![0.0; frames * FRAME_SAMPLES];
    let channels = vec![silence.as_slice(); config.channel_configuration.channels()];
    let mut encoder = StreamEncoder::new(config);
    let mut out = Vec::new();
    encoder.encode(&channels, &mut |frame: &[u8]| out.push(frame.to_vec()));
    assert_eq!(out.len(), frames);
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
