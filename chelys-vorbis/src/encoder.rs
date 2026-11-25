// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The encoder as the library offers it: PCM in, packets out, as the PCM
//! arrives.

use chelys_core::mdct::Mdct;
use chelys_core::sample::{self, NonFiniteSample, Sample};

use super::audio::{self, Channel};
use super::config::{Config, ConfigError};
use super::header;
use super::packet::{Packet, PacketSink};
use super::psychoacoustic::{self, Model, Tables};
use super::residue::MAX_VALUE;
use super::setup::{BLOCKSIZES, Mode, Setup};
use super::window::window;

/// The samples of a long block.
const BLOCK: usize = BLOCKSIZES[1];

/// The samples from one long block's center to the next's: half a long
/// block, by which the blocks overlap.
const HOP: u64 = BLOCK as u64 / 2;

/// The scale of the spectrum that a decoder takes, against the MDCT of
/// `chelys_core::mdct`, `X_k = 2 * sum z_n cos(2 pi / N (n + n_0)(k +
/// 1/2))` of the windowed samples `z_n`, at full scale 1.0.
///
/// Spec gap: the specification does not define the inverse MDCT, and so
/// neither its scale, its phase nor its sign: it refers to a paper (its
/// reference \[1\]) that is not among chelys's sources. The inverse of
/// `chelys_core::mdct`, with the window and the overlap and add of clauses
/// 4.3.1 and 4.3.8, is `2 / N` times the sum of `X_k cos(2 pi / N (n +
/// n_0)(k + 1/2))`; the decoder's inverse is taken to be that sum alone,
/// and the spectrum is scaled by `2 / N` to match it. Settled by decoding
/// with ffmpeg (black box): it decodes a tone at its level, within 0.5 dB,
/// neither delayed, advanced nor inverted; the conformance test is
/// `ffmpeg_decodes_a_tone_at_its_level_and_phase`.
const SPECTRUM_SCALE: f64 = 2.0 / BLOCK as f64;

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
    setup: Setup,
    mdct: Mdct,
    tables: Tables,
    /// The psychoacoustic model of each channel.
    models: Vec<Model>,
    /// The window of a long block between long blocks.
    window: Vec<f64>,
    /// Each channel's samples from the start of the next packet's block,
    /// fewer than a block's between calls.
    pcm: Vec<Vec<f64>>,
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
        let tables = Tables::new(config.sample_rate());
        Ok(Self {
            config,
            setup: Setup::new(),
            mdct: Mdct::new(BLOCK),
            models: (0..config.channels())
                .map(|_| Model::new(&tables))
                .collect(),
            tables,
            window: window(BLOCK, BLOCKSIZES[0], true, true, true),
            // The first block's first half is before the input, of silence.
            pcm: vec![vec![0.0; HOP as usize]; config.channels()],
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
            setup: self.setup.header(),
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
        self.take(samples.len() / channels, out, |pcm, start, end| {
            for (c, pcm) in pcm.iter_mut().enumerate() {
                pcm.extend(
                    samples[start * channels + c..end * channels]
                        .iter()
                        .step_by(channels)
                        .map(|&s| sample::convert(s)),
                );
            }
        });
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
        self.take(len, out, |pcm, start, end| {
            for (pcm, samples) in pcm.iter_mut().zip(channels) {
                pcm.extend(samples[start..end].iter().map(|&s| sample::convert(s)));
            }
        });
        Ok(())
    }

    /// Takes `samples` more of each channel, which `append` appends to each
    /// channel's from the `start`th to the `end`th, and gives `out` the
    /// packets whose blocks they complete: no more at a time than complete
    /// the next block, so that a call of any length needs no more memory
    /// than a block's.
    fn take<O: PacketSink + ?Sized>(
        &mut self,
        samples: usize,
        out: &mut O,
        mut append: impl FnMut(&mut [Vec<f64>], usize, usize),
    ) {
        let mut start = 0;
        while start < samples {
            let end = samples.min(start + BLOCK - self.pcm[0].len());
            append(&mut self.pcm, start, end);
            self.samples += (end - start) as u64;
            start = end;
            // Packet k's block spans the samples from HOP * (k - 1) to
            // HOP * (k + 1).
            if self.pcm[0].len() == BLOCK {
                self.code(HOP * self.packets, out);
            }
        }
    }

    /// Codes the next packet, whose granule position is `granule_position`.
    ///
    /// Each packet is of a long block, centered on the sample `HOP * k` of
    /// the input for packet k, the first block's on the input's first. A
    /// decoder returns, for each packet after the first, the samples from
    /// the center of the block before it to the center of its own (clause
    /// 4.3.8): `HOP * k` samples in all once it has decoded packet k.
    ///
    /// Each channel's spectrum is given the floor that
    /// [`psychoacoustic`] sets, and its residue is each line divided by
    /// the floor there and rounded. A channel whose residue is all zeros
    /// is coded as unused.
    fn code<O: PacketSink + ?Sized>(&mut self, granule_position: u64, out: &mut O) {
        let lines = BLOCK / 2;
        let mut windowed = vec![0.0; BLOCK];
        let mut spectrum = vec![0.0; lines];
        let mut noise = vec![0.0; lines];
        let coded: Vec<Option<(Vec<u32>, Vec<i32>)>> = (self.pcm.iter())
            .zip(&mut self.models)
            .map(|(pcm, model)| {
                model.analyze(&self.tables, pcm, &mut noise);
                for ((z, &x), &w) in windowed.iter_mut().zip(pcm).zip(&self.window) {
                    *z = x * w;
                }
                self.mdct.transform(&windowed, &mut spectrum);
                for line in &mut spectrum {
                    *line *= SPECTRUM_SCALE;
                }
                let floor = &self.setup.floor;
                let values = floor.values(&psychoacoustic::floor_targets(&noise, floor));
                let curve = floor.curve(&values, lines);
                let residue: Vec<i32> = (spectrum.iter().zip(&curve))
                    .map(|(&x, &f)| {
                        let q = (x / f64::from(f)).round();
                        q.clamp(-f64::from(MAX_VALUE), f64::from(MAX_VALUE)) as i32
                    })
                    .collect();
                residue.iter().any(|&q| q != 0).then_some((values, residue))
            })
            .collect();
        let channels: Vec<Channel<'_>> = coded
            .iter()
            .map(|channel| match channel {
                None => Channel::Unused,
                Some((floor, residue)) => Channel::Coded { floor, residue },
            })
            .collect();
        // Spec gap: the first block has no block before it to lap with,
        // nor the last one after it, and the specification does not say
        // what their window flags are then. Both are set, as for long
        // blocks, as every block is: no decoder returns the half of either
        // block that the flag shapes.
        let data = audio::packet(&self.setup, Mode::Long, true, true, &channels);
        out.packet(Packet {
            data: &data,
            granule_position,
        });
        for pcm in &mut self.pcm {
            pcm.drain(..HOP as usize);
        }
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
            // The blocks past the input are of silence.
            for pcm in &mut self.pcm {
                pcm.resize(BLOCK, 0.0);
            }
            self.code((HOP * self.packets).min(self.samples), out);
        }
        Finished {
            packets: self.packets,
            samples: self.samples,
        }
    }
}
