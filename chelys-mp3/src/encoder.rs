// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The encoder as the library offers it: PCM in, the stream out, as the
//! PCM arrives.

use chelys_core::sample::{self, Sample};

use super::config::{Config, ConfigError};
use super::stream::StreamEncoder;

/// Converts a sample to the encoder's scale, clamped to
/// [`MAX_MAGNITUDE`](chelys_core::sample::MAX_MAGNITUDE). Lines that need
/// more bits than their granule has even at the largest step, as from some
/// hundreds of times full scale at low bitrates, lose their highest
/// frequencies (see the outer loop).
///
/// # Panics
///
/// If the sample is NaN or infinite.
fn convert<S: Sample>(sample: S) -> f64 {
    sample::convert(sample).expect("a sample is not a finite number")
}

/// Encodes PCM as MPEG-1 Layer III, or at the lower sampling frequencies
/// as MPEG-2 Layer III (ISO/IEC 13818-3) or MPEG-2.5, as it arrives.
///
/// Psychoacoustic model 2 chooses each granule's block type and the noise
/// each band allows, and the iteration loops of Annex C shape the
/// quantisation noise to it within the bits the rate gives. The stream does
/// not depend on how the input is divided between calls.
///
/// Output lags input: a frame is coded once the two granules after it have
/// arrived, and its bytes are written once no later frame's main data can
/// begin in them, up to 511 bytes after (255 at the lower sampling
/// frequencies). [`finish`](Self::finish) writes the rest. There is no
/// flushing short of that: a frame's bytes may yet hold a later frame's
/// main data.
///
/// The decoded output lags the input by [`DELAY`](super::DELAY) samples,
/// and ends with the silence that fills the last frame.
pub struct Encoder {
    stream: StreamEncoder,
    /// The input of each channel, converted, reused between calls.
    pcm: Vec<Vec<f64>>,
}

/// What [`Encoder::finish`] gives.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Finished {
    /// The Xing/Info tag's frame, if [`Config::info_tag`] asked for one and
    /// the stream has one, to be written over the first
    /// [`info_tag_len`](Encoder::info_tag_len) bytes of the stream.
    pub info_tag: Option<Vec<u8>>,
    /// The frames of audio, after the tag's frame if there is one.
    pub frames: usize,
    /// The samples each channel was given.
    pub samples: usize,
    /// The bits of ancillary data that the last frames had no room for,
    /// dropped.
    pub ancillary_dropped: usize,
}

impl Encoder {
    /// # Errors
    ///
    /// If the configuration asks for a stream the encoder cannot write, as
    /// [`ConfigError`] lists.
    pub fn new(config: Config) -> Result<Self, ConfigError> {
        let stream_config = config.lower()?;
        let mut stream = match config.wants_info_tag() {
            true => StreamEncoder::with_info_tag(stream_config),
            false => StreamEncoder::new(stream_config),
        };
        if config.wants_mixed_blocks() {
            stream = stream.with_mixed_blocks();
        }
        if config.ancillary_bits() > 0 {
            stream = stream.with_ancillary(config.ancillary_bits());
        }
        Ok(Self {
            stream,
            pcm: vec![Vec::new(); stream_config.mode.channels()],
        })
    }

    /// The number of channels the input has.
    pub fn channels(&self) -> usize {
        self.pcm.len()
    }

    /// The samples of each channel a frame codes: 1152, or 576 at the lower
    /// sampling frequencies.
    pub fn samples_per_frame(&self) -> usize {
        self.stream.samples_per_frame()
    }

    /// The length of the Xing/Info tag's frame at the start of the stream,
    /// if [`Config::info_tag`] asked for one and the stream has one.
    pub fn info_tag_len(&self) -> Option<usize> {
        self.stream.info_tag_len()
    }

    /// Encodes interleaved samples, one of each channel in turn, and
    /// appends to `out` the bytes of the stream that are final.
    ///
    /// # Panics
    ///
    /// If the number of samples is not a multiple of the number of
    /// channels, or a sample is NaN or infinite. Samples beyond 65536 times
    /// full scale are clamped there.
    pub fn encode_interleaved<S: Sample>(&mut self, samples: &[S], out: &mut Vec<u8>) {
        let channels = self.channels();
        assert!(
            samples.len() % channels == 0,
            "{} samples are not a whole number of {channels}-channel frames",
            samples.len(),
        );
        for (c, pcm) in self.pcm.iter_mut().enumerate() {
            pcm.clear();
            pcm.extend(
                samples
                    .iter()
                    .skip(c)
                    .step_by(channels)
                    .map(|&s| convert(s)),
            );
        }
        self.encode_converted(out);
    }

    /// Encodes one slice of samples per channel, and appends to `out` the
    /// bytes of the stream that are final.
    ///
    /// # Panics
    ///
    /// If the number of slices is not the number of channels, the slices
    /// are of different lengths, or a sample is NaN or infinite. Samples
    /// beyond 65536 times full scale are clamped there.
    pub fn encode_planar<S: Sample>(&mut self, channels: &[&[S]], out: &mut Vec<u8>) {
        assert_eq!(
            channels.len(),
            self.channels(),
            "one slice of samples per channel"
        );
        for (pcm, samples) in self.pcm.iter_mut().zip(channels) {
            pcm.clear();
            pcm.extend(samples.iter().map(|&s| convert(s)));
        }
        self.encode_converted(out);
    }

    fn encode_converted(&mut self, out: &mut Vec<u8>) {
        let slices: Vec<&[f64]> = self.pcm.iter().map(Vec::as_slice).collect();
        self.stream.encode(&slices, out);
    }

    /// Queues `bytes` of ancillary data for the frames coded from now on,
    /// as many bits a frame as [`Config::ancillary_bits_per_frame`] allows,
    /// each byte's most significant bit first.
    pub fn push_ancillary(&mut self, bytes: &[u8]) {
        self.stream.push_ancillary(bytes);
    }

    /// The bits of ancillary data queued that no frame has carried yet.
    pub fn ancillary_pending(&self) -> usize {
        self.stream.ancillary_pending()
    }

    /// Ends the stream: pads the input with silence to whole frames, at
    /// least [`DELAY`](super::DELAY) samples past it, so that the decoded
    /// output holds all of the input, and appends the rest of the stream to
    /// `out`.
    pub fn finish(self, out: &mut Vec<u8>) -> Finished {
        let end = self.stream.finish(out);
        Finished {
            info_tag: end.info_tag,
            frames: end.frames,
            samples: end.samples,
            ancillary_dropped: end.ancillary_dropped,
        }
    }
}
