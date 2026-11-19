// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The encoder as the library offers it: PCM in, frames out, as the PCM
//! arrives.

use chelys_core::sample::{self, Sample};

use super::config::{Config, ConfigError};
use super::stream::{FRAME_SAMPLES, FrameSink, StreamEncoder};

/// Converts a sample to the encoder's scale, clamped to
/// [`MAX_MAGNITUDE`](chelys_core::sample::MAX_MAGNITUDE).
///
/// # Panics
///
/// If the sample is NaN or infinite.
fn convert<S: Sample>(sample: S) -> f64 {
    sample::convert(sample).expect("a sample is not a finite number")
}

/// Refuses input with a sample that is NaN or infinite before any of it is
/// coded, so that a refused call writes nothing.
fn assert_finite<S: Sample>(samples: &[S]) {
    assert!(
        samples.iter().all(|&s| sample::convert(s).is_some()),
        "a sample is not a finite number"
    );
}

/// Encodes PCM as MPEG-2 AAC in the Low Complexity profile (ISO/IEC
/// 13818-7), as it arrives.
///
/// The encoder is being built: for what it codes so far, see the [crate
/// documentation](crate).
///
/// Output lags input: a frame is coded once its samples, and any the
/// encoder looks ahead to, have arrived, and [`finish`](Self::finish)
/// codes the rest. The input is converted a frame at a time, so a call of
/// any length needs no more memory than a few frames. The stream does not
/// depend on how the input is divided between calls. The decoded output
/// lags the input by [`DELAY`](super::DELAY) samples, and ends with
/// [`Finished::padding`] samples past it, all of which are to be
/// discarded: they code silence, but decode only to near silence, as the
/// quantization of the input next to them spreads into them.
pub struct Encoder {
    stream: StreamEncoder,
    /// Each channel's samples of the part of a call being converted.
    pcm: Vec<Vec<f64>>,
}

/// What [`Encoder::finish`] gives.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Finished {
    /// The frames of the stream.
    pub frames: u64,
    /// The samples each channel was given.
    pub samples: u64,
    /// The samples that the decoded output of each channel ends with after
    /// [`DELAY`](super::DELAY) samples and the input, to be discarded as
    /// those of `DELAY` are: they code the silence that pads the input,
    /// which decodes to near silence only. The frames' samples, `1024 *
    /// frames`, are all three together.
    pub padding: u64,
    /// The bytes of data streams that the last frames had no room for,
    /// dropped.
    pub data_stream_dropped: usize,
}

impl Encoder {
    /// # Errors
    ///
    /// If the configuration asks for a stream the encoder cannot write, as
    /// [`ConfigError`] lists.
    pub fn new(config: Config) -> Result<Self, ConfigError> {
        let stream_config = config.lower()?;
        let channels = stream_config.channel_configuration.channels();
        let mut stream = StreamEncoder::new(stream_config);
        if config.data_stream_bytes() > 0 {
            stream = stream.with_data_stream(config.data_stream_bytes());
        }
        Ok(Self {
            stream,
            pcm: vec![Vec::with_capacity(FRAME_SAMPLES); channels],
        })
    }

    /// The number of channels the input has.
    pub fn channels(&self) -> usize {
        self.pcm.len()
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
    /// # Panics
    ///
    /// If the number of samples is not a multiple of the number of
    /// channels, or a sample is NaN or infinite. Samples beyond 65536 times
    /// full scale are clamped there.
    pub fn encode_interleaved<S: Sample, O: FrameSink + ?Sized>(
        &mut self,
        samples: &[S],
        out: &mut O,
    ) {
        let channels = self.channels();
        assert!(
            samples.len() % channels == 0,
            "{} samples are not a whole number of {channels}-channel frames",
            samples.len(),
        );
        assert_finite(samples);
        for part in samples.chunks(FRAME_SAMPLES * channels) {
            for (c, pcm) in self.pcm.iter_mut().enumerate() {
                pcm.clear();
                pcm.extend(part.iter().skip(c).step_by(channels).map(|&s| convert(s)));
            }
            self.encode_converted(out);
        }
    }

    /// Encodes one slice of samples per channel, and gives `out` the frames
    /// they complete.
    ///
    /// # Panics
    ///
    /// If the number of slices is not the number of channels, the slices
    /// are of different lengths, or a sample is NaN or infinite. Samples
    /// beyond 65536 times full scale are clamped there.
    pub fn encode_planar<S: Sample, O: FrameSink + ?Sized>(
        &mut self,
        channels: &[&[S]],
        out: &mut O,
    ) {
        assert_eq!(
            channels.len(),
            self.channels(),
            "one slice of samples per channel"
        );
        let len = channels[0].len();
        assert!(
            channels.iter().all(|c| c.len() == len),
            "every channel has the same number of samples"
        );
        for samples in channels {
            assert_finite(samples);
        }
        for start in (0..len).step_by(FRAME_SAMPLES) {
            let end = len.min(start + FRAME_SAMPLES);
            for (pcm, samples) in self.pcm.iter_mut().zip(channels) {
                pcm.clear();
                pcm.extend(samples[start..end].iter().map(|&s| convert(s)));
            }
            self.encode_converted(out);
        }
    }

    fn encode_converted<O: FrameSink + ?Sized>(&mut self, out: &mut O) {
        let channels: Vec<&[f64]> = self.pcm.iter().map(Vec::as_slice).collect();
        self.stream.encode(&channels, out);
    }

    /// Queues `bytes` for the data streams of the frames coded from now on,
    /// as many a frame as [`Config::data_stream_bytes_per_frame`] allows.
    ///
    /// [`Config::data_stream_bytes_per_frame`]: super::Config::data_stream_bytes_per_frame
    pub fn push_data_stream(&mut self, bytes: &[u8]) {
        self.stream.push_data_stream(bytes);
    }

    /// The bytes of data streams queued that no frame has carried yet.
    pub fn data_stream_pending(&self) -> usize {
        self.stream.data_stream_pending()
    }

    /// Ends the stream: pads the input with silence to whole frames, at
    /// least [`DELAY`](super::DELAY) samples past it, so that the decoded
    /// output holds all of the input, and gives `out` the frames left.
    pub fn finish<O: FrameSink + ?Sized>(self, out: &mut O) -> Finished {
        let end = self.stream.finish(out);
        Finished {
            frames: end.frames,
            samples: end.samples,
            padding: end.frames * FRAME_SAMPLES as u64 - super::DELAY as u64 - end.samples,
            data_stream_dropped: end.data_stream_dropped,
        }
    }
}
