// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The frames of a stream: one per block of 1024 samples of each channel,
//! each as long as the bit reservoir lets it be, framed as asked.

mod element;
mod frame;

use super::DELAY;
use super::adts::HEADER_LEN;
use super::audio_specific_config::audio_specific_config;
use super::channel::CodedChannel;
use super::channel_configuration::ChannelConfiguration;
use super::config::Framing;
use super::raw_data_block;
use super::raw_data_block::element::CodedElement;
use super::reservoir::BitReservoir;
use super::sampling_frequency::SamplingFrequency;
use super::window::{WindowSequence, Windows};
use element::ElementEncoder;

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

    /// The shortest frame of channels with `windows`: its channels
    /// silent, each with its own `ics_info()`, without fill.
    fn shortest_frame_len(&self, windows: Windows) -> usize {
        let channel = CodedChannel::silent(windows, self.sampling_frequency);
        let element = match self.channel_configuration {
            ChannelConfiguration::Mono => CodedElement::single(channel),
            ChannelConfiguration::Stereo => CodedElement::pair(channel.clone(), channel),
        };
        self.header_len() + raw_data_block::min_len(&element)
    }

    /// The lowest bitrate that brings each frame the bits of the shortest,
    /// that of long windows, so that the reservoir never runs below empty.
    /// A frame that cannot afford the longer `ics_info()` of eight short
    /// windows is coded with long ones.
    pub fn min_bits_per_second(&self) -> u32 {
        let bits = 8 * self.shortest_frame_len(Windows::LONG_SINE) as u64;
        let sample_rate = u64::from(self.sampling_frequency.hz());
        (bits * sample_rate).div_ceil(FRAME_SAMPLES as u64) as u32
    }

    /// The bits of the shortest frame of eight short windows, which a
    /// frame must be able to take to be coded with them (see
    /// [`StreamEncoder::affords_short`]).
    fn shortest_short_frame_bits(&self) -> u32 {
        let windows = Windows {
            sequence: WindowSequence::EIGHT_SHORT,
            ..Windows::LONG_SINE
        };
        8 * self.shortest_frame_len(windows) as u32
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
/// block of [`FRAME_SAMPLES`] of each channel, once the block after it has
/// arrived, from which the frame's windows are chosen.
#[derive(Debug)]
pub struct StreamEncoder {
    config: StreamConfig,
    reservoir: BitReservoir,
    /// The one element of each frame, which `channel_configuration` gives.
    element: ElementEncoder,
    /// Whether frames may be coded with short windows.
    block_switching: bool,
    /// [`StreamConfig::shortest_short_frame_bits`].
    shortest_short_frame_bits: u32,
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
        let channels = config.channel_configuration.channels();
        Self {
            config,
            reservoir,
            element: ElementEncoder::new(config.sampling_frequency, channels),
            block_switching: true,
            shortest_short_frame_bits: config.shortest_short_frame_bits(),
            samples: 0,
            frames: 0,
        }
    }

    /// The encoder, but coding every frame with long windows, as if no
    /// frame had an attack, for comparing with block switching in tests.
    #[cfg(any(test, feature = "unstable"))]
    pub fn without_block_switching(mut self) -> Self {
        self.block_switching = false;
        self
    }

    /// Whether the next frame's bits, those the reservoir makes available
    /// to it when it asks for none beyond the mean, pay for the shortest
    /// frame of eight short windows in every channel, and so for any
    /// windows: a frame is given no fewer when it asks for more. Near the
    /// lowest bitrate they may not, as it is set by the frames of long
    /// windows (see [`StreamConfig::min_bits_per_second`]).
    fn affords_short(&self) -> bool {
        self.shortest_short_frame_bits <= self.reservoir.available_bits(0)
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
    /// 1.0, and gives `out` the frames they complete, each once the block
    /// after it has arrived.
    ///
    /// # Panics
    ///
    /// If there is not one slice for each channel, all of one length: a
    /// caller's error.
    pub fn encode<O: FrameSink + ?Sized>(&mut self, channels: &[&[f64]], out: &mut O) {
        assert_eq!(
            channels.len(),
            self.element.channels(),
            "one slice per channel"
        );
        let len = channels[0].len();
        assert!(
            channels.iter().all(|c| c.len() == len),
            "every channel has the same number of samples"
        );
        self.element.push(channels);
        self.samples += len as u64;
        self.encode_buffered(out);
    }

    /// Codes the frames whose blocks, and the block after each, are
    /// buffered, and drops their samples.
    fn encode_buffered<O: FrameSink + ?Sized>(&mut self, out: &mut O) {
        if !self.element.is_primed() && self.element.buffered() >= FRAME_SAMPLES {
            self.element.prime();
        }
        let mut start = 0;
        while self.element.is_primed() && self.element.buffered() - start >= 2 * FRAME_SAMPLES {
            self.encode_frame(start, out);
            start += FRAME_SAMPLES;
        }
        self.element.drain(start);
    }

    /// Ends the stream: pads the input with silence to whole frames, at
    /// least [`DELAY`] samples past it, so that the decoded output holds
    /// all of the input, and gives `out` the frames left. The block after
    /// the last frame, from which its windows are chosen, is silence too.
    pub fn finish<O: FrameSink + ?Sized>(mut self, out: &mut O) -> StreamEnd {
        let frames = (self.samples + DELAY as u64).div_ceil(FRAME_SAMPLES as u64);
        // Few: those of the samples held and of DELAY, and the block after
        // the last.
        let remaining = (frames - self.frames) as usize;
        self.element.pad((remaining + 1) * FRAME_SAMPLES);
        self.encode_buffered(out);
        debug_assert_eq!(self.frames, frames);
        StreamEnd {
            frames: self.frames,
            samples: self.samples,
        }
    }

    /// Codes the frame whose block starts at `start` in the samples
    /// buffered, the block after it following, and gives it to `out`, in
    /// four steps: has the element's psychoacoustic models analyze the block
    /// after it, for the attacks that choose the frame's windows; prepares
    /// the element's channels for quantization (see
    /// [`ElementEncoder::prepare`]); has the bit reservoir make bits
    /// available to the frame for the bits the model asks for; and
    /// quantizes the element in them.
    fn encode_frame<O: FrameSink + ?Sized>(&mut self, start: usize, out: &mut O) {
        self.element
            .look_ahead(start + FRAME_SAMPLES, self.block_switching);
        let afford_short = self.affords_short();
        self.element.prepare(start, afford_short);
        let demand = self.element.demand();
        let available_bits = frame::element_bits(&self.config, &self.reservoir, &demand);
        let element = self.element.quantize(available_bits);
        frame::write(&self.config, &mut self.reservoir, &element, out);
        self.frames += 1;
    }
}

/// A stream of `frames` silent frames, each apart, for tests: the
/// frames of as many blocks of silence, and of the block after them.
#[cfg(any(test, feature = "unstable"))]
pub fn encode_silence(config: StreamConfig, frames: usize) -> Vec<Vec<u8>> {
    let silence = vec![0.0; (frames + 1) * FRAME_SAMPLES];
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
            assert_eq!(
                frames[0].len(),
                config.shortest_frame_len(Windows::LONG_SINE),
                "{framing:?}"
            );
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
        assert!(frames.last().unwrap().len() > config.shortest_frame_len(Windows::LONG_SINE));
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

    /// A click after silence is coded with short windows, but for an
    /// encoder without block switching.
    #[test]
    fn block_switching_can_be_turned_off() {
        let config = config(Framing::Raw, 64_000);
        let mut input = vec![0.0; 8 * FRAME_SAMPLES];
        input[5 * FRAME_SAMPLES + 300] = 0.9;
        for switching in [true, false] {
            let mut encoder = StreamEncoder::new(config);
            if !switching {
                encoder = encoder.without_block_switching();
            }
            let mut sequences = Vec::new();
            encoder.encode(&[&input, &input], &mut |frame: &[u8]| {
                // id_syn_ele, element_instance_tag, common_window,
                // global_gain, ics_reserved_bit, then window_sequence.
                sequences.push((frame[2] >> 5) & 0b11);
            });
            assert_eq!(sequences.len(), 7);
            assert_eq!(sequences.contains(&2), switching, "{sequences:?}");
        }
    }

    /// At the lowest bitrate, which is set by frames of long windows, a
    /// frame cannot afford the side information of eight short windows,
    /// and an attack is coded with long ones; a little above it, it is
    /// coded short.
    #[test]
    fn a_frame_without_bits_for_short_windows_stays_long() {
        let mut config = StreamConfig {
            sampling_frequency: SamplingFrequency::Hz44100,
            channel_configuration: ChannelConfiguration::Mono,
            bits_per_second: 0,
            framing: Framing::Adts,
        };
        let min = config.min_bits_per_second();
        let mut input = vec![0.0; 12 * FRAME_SAMPLES];
        for click in [3 * FRAME_SAMPLES + 300, 8 * FRAME_SAMPLES + 900] {
            input[click] = 0.9;
        }
        for (bits_per_second, short) in [(min, false), (min + 1000, true)] {
            config.bits_per_second = bits_per_second;
            let mut sequences = Vec::new();
            let mut encoder = StreamEncoder::new(config);
            encoder.encode(&[&input], &mut |frame: &[u8]| {
                // After the header, id_syn_ele, element_instance_tag,
                // global_gain and ics_reserved_bit.
                sequences.push((frame[9] >> 6) & 0b11);
            });
            assert_eq!(
                sequences.contains(&2),
                short,
                "{bits_per_second}: {sequences:?}"
            );
            assert!(sequences.contains(&1), "{bits_per_second}: {sequences:?}");
        }
    }

    #[test]
    fn noise_after_silence_borrows_from_the_reservoir() {
        let config = StreamConfig {
            sampling_frequency: SamplingFrequency::Hz44100,
            channel_configuration: ChannelConfiguration::Mono,
            bits_per_second: 8_000,
            framing: Framing::Adts,
        };
        // Silence, which takes the shortest frames and fills the
        // reservoir, then white noise, whose perceptual entropy asks for
        // more than the mean.
        let mut input = vec![0.0; 40 * FRAME_SAMPLES];
        let mut state = 3u64;
        input.extend((0..10 * FRAME_SAMPLES).map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            0.25 * ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
        }));
        let mut encoder = StreamEncoder::new(config);
        let mut frames = Vec::new();
        encoder.encode(&[&input], &mut |frame: &[u8]| frames.push(frame.len()));
        let mean = 8_000.0 * FRAME_SAMPLES as f64 / 44_100.0 / 8.0;
        let longest = frames[40..].iter().copied().max().unwrap();
        assert!(
            longest as f64 > 1.5 * mean,
            "{longest} bytes against {mean}"
        );
    }
}
