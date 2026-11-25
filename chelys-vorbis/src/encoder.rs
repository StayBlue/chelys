// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The encoder as the library offers it: PCM in, packets out, as the PCM
//! arrives.

use chelys_core::sample::{self, NonFiniteSample, Sample};

use super::audio;
use super::config::{Config, ConfigError};
use super::header;
use super::packet::{Packet, PacketSink};
use super::setup::{self, BLOCKSIZES, Mode};

/// The samples from one long block's center to the next's: half a long
/// block, by which the blocks overlap.
const HOP: u64 = BLOCKSIZES[1] as u64 / 2;

/// The three header packets that begin a stream, in their order (clause
/// 4.2), which a container gives a decoder before the audio packets.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Headers {
    /// The identification header (clause 4.2.2).
    pub identification: Vec<u8>,
    /// The comment header (clause 5), which gives chelys's vendor string
    /// and no comments.
    pub comment: Vec<u8>,
    /// The setup header (clause 4.2.4).
    pub setup: Vec<u8>,
}

/// Encodes PCM as Vorbis I, as it arrives. For what it codes, see the
/// [crate documentation](crate).
///
/// A stream is its [`headers`](Self::headers), then the audio packets that
/// [`encode_interleaved`](Self::encode_interleaved),
/// [`encode_planar`](Self::encode_planar) and [`finish`](Self::finish)
/// give a [`PacketSink`], each with its granule position. Output lags
/// input: a packet is coded once all of its block's samples have arrived,
/// and [`finish`](Self::finish) codes the rest. The stream does not depend
/// on how the input is divided between calls.
pub struct Encoder {
    config: Config,
    /// The samples each channel has been given.
    samples: u64,
    /// The audio packets given.
    packets: u64,
}

/// What [`Encoder::finish`] gives.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Finished {
    /// The audio packets of the stream.
    pub packets: u64,
    /// The samples each channel was given: the last packet's granule
    /// position.
    pub samples: u64,
}

impl Encoder {
    /// # Errors
    ///
    /// If the configuration asks for a stream the encoder cannot write, as
    /// [`ConfigError`] lists.
    pub fn new(config: Config) -> Result<Self, ConfigError> {
        config.check()?;
        Ok(Self {
            config,
            samples: 0,
            packets: 0,
        })
    }

    /// The number of channels the input has.
    pub fn channels(&self) -> usize {
        self.config.channels()
    }

    /// The sample rate in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate()
    }

    /// The header packets that begin the stream.
    pub fn headers(&self) -> Headers {
        Headers {
            identification: header::identification(self.channels(), self.sample_rate(), BLOCKSIZES),
            comment: header::comment(),
            setup: setup::setup(),
        }
    }

    /// Encodes interleaved samples, one of each channel in turn, and gives
    /// `out` the packets they complete.
    ///
    /// # Errors
    ///
    /// If a sample is NaN or infinite, [`NonFiniteSample`] names the first
    /// to be played, and none of the call is taken: `out` is given nothing,
    /// and the encoder is as it was before the call. The samples may then
    /// be given again once made finite, or left out, or the stream
    /// finished. `i16` samples are never refused.
    ///
    /// # Panics
    ///
    /// If the number of samples is not a multiple of the number of
    /// channels.
    pub fn encode_interleaved<S: Sample, O: PacketSink + ?Sized>(
        &mut self,
        samples: &[S],
        out: &mut O,
    ) -> Result<(), NonFiniteSample> {
        let channels = self.channels();
        assert!(
            samples.len() % channels == 0,
            "{} samples are not a whole number of {channels}-channel frames",
            samples.len(),
        );
        sample::check_interleaved(samples, channels)?;
        self.take((samples.len() / channels) as u64, out);
        Ok(())
    }

    /// Encodes one slice of samples per channel, and gives `out` the
    /// packets they complete.
    ///
    /// # Errors
    ///
    /// As [`encode_interleaved`](Self::encode_interleaved): if a sample is
    /// NaN or infinite, none of the call is taken.
    ///
    /// # Panics
    ///
    /// If the number of slices is not the number of channels, or the
    /// slices are of different lengths.
    pub fn encode_planar<S: Sample, O: PacketSink + ?Sized>(
        &mut self,
        channels: &[&[S]],
        out: &mut O,
    ) -> Result<(), NonFiniteSample> {
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
        sample::check_planar(channels)?;
        self.take(len as u64, out);
        Ok(())
    }

    /// Takes `samples` more of each channel, and gives `out` the packets
    /// whose blocks they complete.
    fn take<O: PacketSink + ?Sized>(&mut self, samples: u64, out: &mut O) {
        self.samples += samples;
        // Packet k's block spans the samples from HOP * (k - 1) to
        // HOP * (k + 1).
        while HOP * (self.packets + 1) <= self.samples {
            self.code(HOP * self.packets, out);
        }
    }

    /// Codes the next packet, whose granule position is `granule_position`.
    ///
    /// Each packet is of a long block, centered on the sample `HOP * k` of
    /// the input for packet k, the first block's on the input's first. A
    /// decoder returns, for each packet after the first, the samples from
    /// the center of the block before it to the center of its own (clause
    /// 4.3.8): `HOP * k` samples in all once it has decoded packet k.
    fn code<O: PacketSink + ?Sized>(&mut self, granule_position: u64, out: &mut O) {
        // Spec gap: the first block has no block before it to lap with,
        // nor the last one after it, and the specification does not say
        // what their window flags are then. Both are set, as for long
        // blocks, as every block is: no decoder returns the half of either
        // block that the flag shapes.
        let data = audio::silence(Mode::Long, true, true, self.channels());
        out.packet(Packet {
            data: &data,
            granule_position,
        });
        self.packets += 1;
    }

    /// Ends the stream, and gives `out` the packets left: up to the first
    /// whose block is centered at or past the end of the input, so that the
    /// decoder returns all of the input. That packet's granule position is
    /// the end of the input, which tells the decoder to return no samples
    /// past it (Appendix A.2).
    pub fn finish<O: PacketSink + ?Sized>(mut self, out: &mut O) -> Finished {
        let last = self.samples.div_ceil(HOP);
        while self.packets <= last {
            self.code((HOP * self.packets).min(self.samples), out);
        }
        Finished {
            packets: self.packets,
            samples: self.samples,
        }
    }
}
