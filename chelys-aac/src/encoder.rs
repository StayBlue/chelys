// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The encoder as the library offers it: PCM in, frames out, as the PCM
//! arrives.

use chelys_core::sample::{self, Sample};

use super::config::{Config, ConfigError};
use super::stream::{FRAME_SAMPLES, FrameSink, StreamEncoder};

/// Converts a sample to the encoder's scale, clamped to
/// [`MAX_MAGNITUDE`](chelys_core::sample::MAX_MAGNITUDE), and reads one that
/// is NaN or infinite as silence.
fn convert<S: Sample>(sample: S) -> f64 {
    sample::convert(sample).unwrap_or(0.0)
}

/// Encodes PCM as MPEG-2 AAC in the Low Complexity profile (ISO/IEC
/// 13818-7), as it arrives.
///
/// The encoder is being built: for now every frame codes each channel as
/// silence, whatever the input, in a stream that is valid as it will be.
///
/// A frame is coded once its 1024 samples of each channel have arrived; the
/// stream does not depend on how the input is divided between calls, even
/// within a sample frame. [`finish`](Self::finish) codes the rest. The
/// decoded output lags the input by [`DELAY`](super::DELAY) samples, and
/// ends with [`Finished::padding`] samples of silence.
///
/// No input makes it panic: a sample that is NaN or infinite is read as
/// silence, and one beyond 65536 times full scale is clamped there.
pub struct Encoder {
    stream: StreamEncoder,
    /// The input of each channel, converted, that no frame has coded yet.
    pcm: Vec<Vec<f64>>,
    /// The samples each channel was given.
    given: Vec<usize>,
    /// The channel of the next interleaved sample: where the last call
    /// ended within a sample frame.
    next_channel: usize,
}

/// What [`Encoder::finish`] gives.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Finished {
    /// The frames of the stream.
    pub frames: usize,
    /// The samples each channel was given: of a channel given fewer than
    /// another, as many as that one, as it was padded with silence to them.
    pub samples: usize,
    /// The samples of silence that the decoded output of each channel ends
    /// with, after [`DELAY`](super::DELAY) samples and the input: the
    /// frames' samples, `1024 * frames`, are all three together.
    pub padding: usize,
}

impl Encoder {
    /// # Errors
    ///
    /// If the configuration asks for a stream the encoder cannot write, as
    /// [`ConfigError`] lists.
    pub fn new(config: Config) -> Result<Self, ConfigError> {
        let config = config.lower()?;
        let channels = config.channel_configuration.channels();
        Ok(Self {
            stream: StreamEncoder::new(config),
            pcm: vec![Vec::new(); channels],
            given: vec![0; channels],
            next_channel: 0,
        })
    }

    /// The number of channels the input has.
    pub fn channels(&self) -> usize {
        self.stream.config().channel_configuration.channels()
    }

    /// The samples of each channel a frame codes: 1024.
    pub fn samples_per_frame(&self) -> usize {
        FRAME_SAMPLES
    }

    /// The `AudioSpecificConfig()` of ISO/IEC 14496-3 that describes the
    /// stream to a decoder of its raw frames, as an MPEG-4 file carries it.
    pub fn audio_specific_config(&self) -> Vec<u8> {
        self.stream.audio_specific_config()
    }

    /// Encodes interleaved samples, one of each channel in turn, and gives
    /// `out` the frames they complete.
    ///
    /// The samples need not be whole sample frames: a call may end within
    /// one, and the next go on from there.
    pub fn encode_interleaved<S: Sample, O: FrameSink + ?Sized>(
        &mut self,
        samples: &[S],
        out: &mut O,
    ) {
        let channels = self.channels();
        for &sample in samples {
            self.pcm[self.next_channel].push(convert(sample));
            self.given[self.next_channel] += 1;
            self.next_channel = (self.next_channel + 1) % channels;
        }
        self.encode_blocks(out);
    }

    /// Encodes one slice of samples per channel, and gives `out` the frames
    /// they complete.
    ///
    /// The slices need not be of one length: a channel given more than
    /// another waits for it. Slices past the encoder's channels are
    /// ignored, and a channel without one is given nothing.
    pub fn encode_planar<S: Sample, O: FrameSink + ?Sized>(
        &mut self,
        channels: &[&[S]],
        out: &mut O,
    ) {
        for ((pcm, given), samples) in self.pcm.iter_mut().zip(&mut self.given).zip(channels) {
            pcm.extend(samples.iter().map(|&s| convert(s)));
            *given += samples.len();
        }
        self.encode_blocks(out);
    }

    /// Codes every frame whose samples have all arrived.
    fn encode_blocks<O: FrameSink + ?Sized>(&mut self, out: &mut O) {
        let ready = self.pcm.iter().map(Vec::len).min().unwrap_or(0) / FRAME_SAMPLES;
        for i in 0..ready {
            let block: Vec<&[f64]> = (self.pcm.iter())
                .map(|pcm| &pcm[i * FRAME_SAMPLES..][..FRAME_SAMPLES])
                .collect();
            self.stream.encode(&block, out);
        }
        for pcm in &mut self.pcm {
            pcm.drain(..ready * FRAME_SAMPLES);
        }
    }

    /// Ends the stream: pads the input with silence to whole frames, at
    /// least [`DELAY`](super::DELAY) samples past it, so that the decoded
    /// output holds all of the input, and gives `out` the frames left.
    pub fn finish<O: FrameSink + ?Sized>(mut self, out: &mut O) -> Finished {
        let samples = self.given.iter().copied().max().unwrap_or(0);
        let frames = (samples + super::DELAY).div_ceil(FRAME_SAMPLES);
        let left = (frames - self.stream.frames()) * FRAME_SAMPLES;
        for pcm in &mut self.pcm {
            pcm.resize(left, 0.0);
        }
        self.encode_blocks(out);
        debug_assert_eq!(self.stream.frames(), frames);
        Finished {
            frames,
            samples,
            padding: frames * FRAME_SAMPLES - super::DELAY - samples,
        }
    }
}
