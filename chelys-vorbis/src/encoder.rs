// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The encoder as the library offers it: PCM in, packets out, as the PCM
//! arrives.

use chelys_core::mdct::Mdct;
use chelys_core::sample::{self, NonFiniteSample, Sample};

use super::audio;
use super::config::{Config, ConfigError};
use super::coupling;
use super::floor::Floor;
use super::header;
use super::packet::{Packet, PacketSink};
use super::psychoacoustic::{self, Analysis, FIRST_SHORT, Model, SHORT_BLOCKS, Tables};
use super::residue::MAX_VALUE;
use super::setup::{BLOCKSIZES, Mode, Setup};
use super::window::window;

/// The samples of a long block.
const BLOCK: usize = BLOCKSIZES[1];

/// The samples of a short block.
const SHORT_BLOCK: usize = BLOCKSIZES[0];

/// The samples from one long block's center to the next's: half a long
/// block, by which long blocks overlap. Short blocks take the place of a
/// long block as many at a time as keep this (see [`SHORT_BLOCKS`]).
const HOP: u64 = BLOCK as u64 / 2;

/// The samples the encoder keeps of each channel to code the next long
/// block or the short blocks that take its place: those of the long block,
/// and the next hop, from which the model finds whether the next block has
/// an attack, as a long block's window depends on whether the block after
/// it is short (clause 4.3.1).
const KEPT: usize = BLOCK + HOP as usize;

/// The scale of the spectrum that a decoder takes, against the MDCT of
/// `chelys_core::mdct`, `X_k = 2 * sum z_n cos(2 pi / N (n + n_0)(k +
/// 1/2))` of the windowed samples `z_n`, at full scale 1.0, for a block of
/// `n` samples, short or long: `N` is `n`.
///
/// Spec gap: the specification does not define the inverse MDCT, and so
/// neither its scale, its phase nor its sign: it refers to a paper (its
/// reference \[1\]) that is not among chelys's sources. The inverse of
/// `chelys_core::mdct`, with the window and the overlap and add of clauses
/// 4.3.1 and 4.3.8, is `2 / N` times the sum of `X_k cos(2 pi / N (n +
/// n_0)(k + 1/2))`; the decoder's inverse is taken to be that sum alone,
/// for blocks of either size, and the spectrum is scaled by `2 / N` to
/// match it. Settled by decoding with ffmpeg (black box): it decodes a
/// tone at its level, within 0.5 dB, neither delayed, advanced nor
/// inverted, in long blocks and in short ones; the conformance tests are
/// `ffmpeg_decodes_a_tone_at_its_level_and_phase` and
/// `ffmpeg_decodes_a_tone_in_short_blocks_at_its_level_and_phase`.
fn spectrum_scale(n: usize) -> f64 {
    2.0 / n as f64
}

/// The lines of a block of `n` samples below [`coupling::PROTECTED_HZ`] at
/// `sample_rate`: line `k` is centered at `(k + 1/2) sample_rate / n` Hz.
fn protected_lines(n: usize, sample_rate: u32) -> usize {
    let below = coupling::PROTECTED_HZ * n as f64 / f64::from(sample_rate) - 0.5;
    (below.ceil().max(0.0) as usize).min(n / 2)
}

/// How much nearer zero than the nearest integer each line of the residue
/// is rounded, in steps of the floor: a line is coded as `n` from `n -
/// 1/2 + DEAD_ZONE` steps to `n + 1/2 + DEAD_ZONE`, and as 0 below 1/2 +
/// DEAD_ZONE.
///
/// More lines are coded as 0, and the rest a little low, for fewer bits.
/// Graded with PEAQ (GstPEAQ 0.6.1, black box) on the 51 tracks of the
/// tuning set at 44.1 kHz in stereo, interpolated between noise offsets
/// to the same mean bitrates, with codebooks counted as in [`frequencies`]
/// and the threshold in quiet lowered by 30 dB, the streams graded -0.952
/// at 140 kbit/s and -0.769 at 150, against -1.076 and -0.842 rounded to
/// the nearest. With it lowered by 40 dB, a dead zone of 0.15 graded
/// -0.879 and -0.696, against -0.866 and -0.717 at 0.1, no better.
///
/// [`frequencies`]: super::frequencies
const DEAD_ZONE: f64 = 0.1;

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
/// input: a long block's packet, or those of the short blocks that take
/// its place, are coded once all of its samples and the next 1024 have
/// arrived, and [`finish`](Self::finish) codes the rest. The stream does
/// not depend on how the input is divided between calls.
pub struct Encoder {
    config: Config,
    setup: Setup,
    /// The MDCT of each block size, by blockflag.
    mdcts: [Mdct; 2],
    tables: Tables,
    /// The psychoacoustic model's partition of each line of a block, by
    /// blockflag.
    partitions: [Vec<usize>; 2],
    /// The psychoacoustic model of each channel.
    models: Vec<Model>,
    /// Each channel's analysis of the long block coded next, or whose
    /// place short blocks take, once the next one is analyzed too.
    current: Vec<Analysis>,
    /// Each channel's analysis of the long block after it.
    next: Vec<Analysis>,
    /// The window of a short block.
    short_window: Vec<f64>,
    /// The window of a long block, by its `previous_window_flag` and
    /// `next_window_flag`.
    long_windows: [[Vec<f64>; 2]; 2],
    /// Each channel's samples from the start of the next long block, fewer
    /// than [`KEPT`] between calls.
    pcm: Vec<Vec<f64>>,
    /// The samples each channel has been given.
    samples: u64,
    /// The audio packets given.
    packets: u64,
    /// The long blocks coded, or whose place short blocks took: the next is
    /// centered on the sample `HOP` times this of the input.
    hops: u64,
    /// Whether the last block coded is long.
    last_long: bool,
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
        let noise_offset_db = config.noise_offset_db()?;
        let tables = Tables::new(config.sample_rate());
        let channels = config.channels();
        let long_window =
            |previous_long, next_long| window(BLOCK, SHORT_BLOCK, true, previous_long, next_long);
        Ok(Self {
            config,
            setup: Setup::new(channels),
            mdcts: [Mdct::new(SHORT_BLOCK), Mdct::new(BLOCK)],
            models: (0..channels)
                .map(|_| Model::new(&tables, noise_offset_db))
                .collect(),
            partitions: [false, true].map(|long| tables.partition_of_lines(long)),
            tables,
            current: vec![Analysis::default(); channels],
            next: vec![Analysis::default(); channels],
            short_window: window(SHORT_BLOCK, SHORT_BLOCK, false, true, true),
            long_windows: [
                [long_window(false, false), long_window(false, true)],
                [long_window(true, false), long_window(true, true)],
            ],
            // The first block's first half is before the input, of silence.
            pcm: vec![vec![0.0; HOP as usize]; channels],
            samples: 0,
            packets: 0,
            hops: 0,
            last_long: true,
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
    /// the next long block and the hop after it, so that a call of any
    /// length needs no more memory than that.
    fn take<O: PacketSink + ?Sized>(
        &mut self,
        samples: usize,
        out: &mut O,
        mut append: impl FnMut(&mut [Vec<f64>], usize, usize),
    ) {
        let mut start = 0;
        while start < samples {
            let end = samples.min(start + KEPT - self.pcm[0].len());
            append(&mut self.pcm, start, end);
            self.samples += (end - start) as u64;
            start = end;
            if self.pcm[0].len() == KEPT {
                self.code(None, out);
            }
        }
    }

    /// Codes the next long block, or the short blocks that take its place,
    /// from the samples kept, the next hop's after it too; up to the first
    /// block centered at or past `end`, if there is one, the end of the
    /// input, and then gives whether it did stop there.
    ///
    /// Long block k is centered on the sample `HOP * k` of the input, the
    /// first on the input's first, and spans the samples from `HOP * (k -
    /// 1)` to `HOP * (k + 1)`. Where the model finds an attack in a long
    /// block but the first, [`SHORT_BLOCKS`] short blocks take its place,
    /// centered from `HOP * k - 448` to `HOP * k + 448`, 128 apart, which
    /// keeps the long blocks before and after them where they would be
    /// (clause 4.3.8). A long block's window is shaped for the blocks
    /// beside it (clause 4.3.1), and so the next long block is analyzed
    /// before this one is coded.
    ///
    /// Each packet's granule position is the center of its block: a decoder
    /// returns, for each packet after the first, the samples from the
    /// center of the block before it to the center of its own (clause
    /// 4.3.8). The first packet is of a long block, so that the decoder
    /// returns the input from its first sample.
    ///
    /// Spec reading: Appendix A.2 gives a page the granule position of "the
    /// last complete sample returned by decode", and then notes that where a
    /// long block comes before a short one, "the last decodable sample"
    /// of the long block's packet lies at `3 n / 4 - blocksize_0 / 4` of
    /// it, past its center: where the short block's window begins, the
    /// last sample that the long block's packet finishes without the next
    /// packet. Clause 4.3.8 returns no sample past a block's center,
    /// whatever the block after. The granule position is taken as the
    /// samples returned by clause 4.3.8, as the appendix's first sentence
    /// reads, not as the samples finished. Settled by decoding with ffmpeg
    /// (black box): with each packet on a page of its own, ffprobe times
    /// each packet after the first two from the granule position of the
    /// one before it to its own, long blocks before short ones too; the
    /// conformance test is
    /// `ffmpeg_times_each_packet_from_its_granule_position`.
    fn code<O: PacketSink + ?Sized>(&mut self, end: Option<u64>, out: &mut O) -> bool {
        if self.hops == 0 {
            for ((model, pcm), analysis) in
                self.models.iter_mut().zip(&self.pcm).zip(&mut self.current)
            {
                model.analyze(&self.tables, &pcm[..BLOCK], analysis);
            }
        }
        for ((model, pcm), analysis) in self.models.iter_mut().zip(&self.pcm).zip(&mut self.next) {
            model.analyze(&self.tables, &pcm[HOP as usize..KEPT], analysis);
        }
        let short = self.hops > 0 && self.current.iter().any(Analysis::attack);
        let mut ended = false;
        if short {
            for k in 0..SHORT_BLOCKS {
                ended = self.block(Some(k), end, true, out);
                if ended {
                    break;
                }
            }
        } else {
            let next_long = !self.next.iter().any(Analysis::attack);
            ended = self.block(None, end, next_long, out);
        }
        for pcm in &mut self.pcm {
            pcm.drain(..HOP as usize);
        }
        std::mem::swap(&mut self.current, &mut self.next);
        self.hops += 1;
        ended
    }

    /// Codes the `short`th short block that takes the current long block's
    /// place, or the long block itself, as the last if its center is at or
    /// past `end`; whose block after it is long if `next_long`. Gives
    /// whether it was the last.
    ///
    /// Each channel's spectrum is given the floor that
    /// [`psychoacoustic`] sets, and its residue is quantized to it (see
    /// [`Quantized`]). A channel whose residue is all zeros is coded as
    /// unused. In a stereo stream, the two channels' residues are then
    /// coupled (see [`coupling::couple`]).
    fn block<O: PacketSink + ?Sized>(
        &mut self,
        short: Option<usize>,
        end: Option<u64>,
        next_long: bool,
        out: &mut O,
    ) -> bool {
        let mode = if short.is_some() {
            Mode::Short
        } else {
            Mode::Long
        };
        let n = BLOCKSIZES[usize::from(mode.blockflag())];
        // The block's first sample among those kept, which start a hop
        // before the long block's center, and its center in the input.
        let start = short.map_or(0, |k| FIRST_SHORT + k * SHORT_BLOCK / 2);
        let center = HOP * self.hops + (start + n / 2) as u64 - HOP;
        let last = end.is_some_and(|end| center >= end);
        // Spec gap: the first block has no block before it to lap with,
        // nor the last one after it, and the specification does not say
        // what their window flags are then. Each is set, as if the block it
        // has not were long: no decoder returns the half of either block
        // that the flag shapes.
        let previous_long = self.last_long;
        let next_long = next_long || last;
        let window = match mode {
            Mode::Short => &self.short_window,
            Mode::Long => &self.long_windows[usize::from(previous_long)][usize::from(next_long)],
        };
        let mdct = &mut self.mdcts[usize::from(mode.blockflag())];
        let floor = self.setup.floor(mode);
        // Coupled, an angle is as large as the sum of the two channels'
        // values (see `coupling::square_polar`), and is to be within the
        // residue's reach too.
        let limit = if self.setup.coupled {
            MAX_VALUE / 2
        } else {
            MAX_VALUE
        };
        let noises: Vec<&[f64]> = (self.current.iter())
            .map(|analysis| match short {
                Some(k) => &analysis.short[k][..],
                None => &analysis.long[..],
            })
            .collect();
        let lines = n / 2;
        let mut windowed = vec![0.0; n];
        let mut spectra = vec![vec![0.0; lines]; self.pcm.len()];
        let mut channels: Vec<Quantized> = (self.pcm.iter())
            .zip(&noises)
            .zip(&mut spectra)
            .map(|((pcm, noise), spectrum)| {
                for ((z, &x), &w) in windowed.iter_mut().zip(&pcm[start..start + n]).zip(window) {
                    *z = x * w;
                }
                mdct.transform(&windowed, spectrum);
                for line in spectrum.iter_mut() {
                    *line *= spectrum_scale(n);
                }
                Quantized::new(floor, noise, spectrum, limit)
            })
            .collect();
        let used: Vec<bool> = channels.iter().map(Quantized::used).collect();
        if let ([left, right], true) = (&mut channels[..], self.setup.coupled) {
            coupling::couple(
                [&mut left.residue, &mut right.residue],
                [&left.curve, &right.curve],
                [&spectra[0], &spectra[1]],
                [noises[0], noises[1]],
                &self.partitions[usize::from(mode.blockflag())],
                protected_lines(n, self.config.sample_rate()),
            );
        }
        let floors: Vec<Option<&[u32]>> = (channels.iter().zip(used))
            .map(|(channel, used)| used.then_some(&channel.values[..]))
            .collect();
        let residues: Vec<&[i32]> = channels.iter().map(|c| &c.residue[..]).collect();
        let data = audio::packet(
            &self.setup,
            mode,
            previous_long,
            next_long,
            &floors,
            &residues,
        );
        out.packet(Packet {
            data: &data,
            granule_position: match end {
                Some(end) if last => end,
                _ => center,
            },
        });
        self.packets += 1;
        self.last_long = mode.blockflag();
        last
    }

    /// Ends the stream, and gives `out` the packets left: up to the first
    /// whose block is centered at or past the end of the input, so that the
    /// decoder returns all of the input. That packet's granule position is
    /// the end of the input, which tells the decoder to return no samples
    /// past it (Appendix A.2).
    pub fn finish<O: PacketSink + ?Sized>(mut self, out: &mut O) -> Finished {
        let end = self.samples;
        loop {
            // The blocks past the input are of silence.
            for pcm in &mut self.pcm {
                pcm.resize(KEPT, 0.0);
            }
            if self.code(Some(end), out) {
                break;
            }
        }
        Finished {
            packets: self.packets,
            samples: self.samples,
        }
    }
}

/// A channel's floor, and its residue, before any coupling.
struct Quantized {
    /// The floor's values (see [`Floor::values`]).
    values: Vec<u32>,
    /// The floor's curve: the step each line of the residue is of.
    curve: Vec<f32>,
    residue: Vec<i32>,
}

impl Quantized {
    /// The floor that the model sets for the noise `noise` allows each
    /// line, raised where a line would be more than `limit` steps of it,
    /// and the residue of `spectrum`: each line divided by the floor there
    /// and rounded, nearer zero by [`DEAD_ZONE`], and held within `limit`,
    /// which only a line far past full scale passes.
    fn new(floor: &Floor, noise: &[f64], spectrum: &[f64], limit: i32) -> Self {
        // The floor the model sets, raised where it would put a line past
        // the residue's reach, as at an attack after silence, where the
        // threshold is held near the threshold in quiet.
        let (values, curve) = floor.values_within(
            &psychoacoustic::floor_targets(noise, floor),
            spectrum,
            f64::from(limit),
        );
        let residue = (spectrum.iter().zip(&curve))
            .map(|(&x, &f)| {
                let steps = x / f64::from(f);
                let q = steps.signum() * (steps.abs() + 0.5 - DEAD_ZONE).floor();
                q.clamp(-f64::from(limit), f64::from(limit)) as i32
            })
            .collect();
        Self {
            values,
            curve,
            residue,
        }
    }

    /// Whether any residue value is not 0.
    fn used(&self) -> bool {
        self.residue.iter().any(|&q| q != 0)
    }
}
