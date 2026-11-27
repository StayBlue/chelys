// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The encoder as the library offers it: PCM in, packets out, as the PCM
//! arrives.

use chelys_core::mdct::Mdct;
use chelys_core::sample::{self, NonFiniteSample, Sample};
use chelys_core::steering::{MeanSteering, OFFSETS_DB};

use super::audio;
use super::buffer::Buffer;
use super::config::{Config, ConfigError, Control};
use super::coupling;
use super::energy;
use super::floor::Floor;
use super::header;
use super::packet::{Packet, PacketSink};
use super::psychoacoustic::{
    self, Allowance, Analysis, FIRST_SHORT, Model, Noise, SHORT_BLOCKS, Tables,
};
use super::rate::{self, ConstantSteering};
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
pub(crate) const HOP: u64 = BLOCK as u64 / 2;

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

/// The lines of a block of `n` samples below `hz` at `sample_rate`: line
/// `k` is centered at `(k + 1/2) sample_rate / n` Hz.
fn lines_below(hz: f64, n: usize, sample_rate: u32) -> usize {
    let below = hz * n as f64 / f64::from(sample_rate) - 0.5;
    (below.ceil().max(0.0) as usize).min(n / 2)
}

/// How much nearer zero than the nearest integer each line of the residue
/// is rounded where bits are scarce, in steps of the floor: a line is
/// coded as `n` from `n - 1/2 + d` steps to `n + 1/2 + d`, and as 0 below
/// 1/2 + `d`, for a dead zone `d` from 0 where bits are plentiful to this
/// where they are scarce (see [`scarcity`]).
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
/// Where bits are plentiful, the noise it adds, and the level it takes from
/// each band, cost more than the bits it saves buy: once the model
/// weighed each band's noise (see the psychoacoustic model's
/// `BAND_EXPONENT`), at noise offsets matched to each bitrate, the streams
/// graded -0.904 with it and -0.921 with 0.05 at 96.5 kbit/s, and, with the
/// angles narrowed within twice each channel's allowance where bits are
/// plentiful, as [`coupling::LEAST_ANGLE_ALLOWANCE`] then allowed, -0.295
/// with it, -0.284 with 0.05 and -0.282 with none at 159.2 kbit/s, where
/// PEAQ's `ADBB` fell from 0.82 to 0.57.
///
/// [`frequencies`]: super::frequencies
const DEAD_ZONE: f64 = 0.1;

/// The noise offsets, in dB, at and above which bits are scarce, and at
/// and below which they are plentiful (see [`scarcity`]): about 95 and 168
/// kbit/s in stereo at 44.1 kHz.
const SCARCE_OFFSET_DB: f64 = -4.0;
const PLENTIFUL_OFFSET_DB: f64 = -9.0;

/// How scarce bits are at the noise offset `noise_offset_db`: 1 at and
/// above [`SCARCE_OFFSET_DB`], 0 at and below [`PLENTIFUL_OFFSET_DB`], and
/// in a straight line between, by which the dead zone, the narrowing of
/// angles, the share of energy that the balance aims for, which trade
/// noise for bits, and in stereo the noise allowed in the treble and the
/// narrowing of the image are tapered.
///
/// chelys's own design: each was tuned where bits were scarce, and where
/// they are plentiful the noise it adds costs more than the bits it saves
/// buy (see [`DEAD_ZONE`], [`coupling::LEAST_ANGLE_ALLOWANCE`] and
/// [`energy::BALANCED_SHARE_SCARCE`]). Each takes more bits as the offset
/// falls, as the offset itself does, so that a lower offset never gives a
/// shorter stream. Graded with PEAQ (GstPEAQ 0.6.1, black box) on the 51
/// tracks of the tuning set at 44.1 kHz in stereo at 128.4 kbit/s, where
/// the offset was then about -6.5 dB, the streams graded -0.479 with the
/// dead zone and the narrowing tapered so, against -0.476 with neither
/// tapered and -0.486 with both as where bits are plentiful.
fn scarcity(noise_offset_db: f64) -> f64 {
    ((noise_offset_db - PLENTIFUL_OFFSET_DB) / (SCARCE_OFFSET_DB - PLENTIFUL_OFFSET_DB))
        .clamp(0.0, 1.0)
}

/// The trades of noise for bits that a block is coded with, which
/// [`scarcity`] tapers with its noise offset.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Trades {
    /// The dead zone (see [`DEAD_ZONE`]).
    dead_zone: f64,
    /// The multiple of each channel's allowance that the narrowing of
    /// angles may leave in a band (see [`coupling::angle_allowance`]).
    angle_allowance: f64,
    /// The most by which the narrowing of angles may lower a band's ratio
    /// of side to mid energy, in dB (see [`coupling::side_loss_db`]).
    side_loss_db: f64,
    /// The share of each group's energy that its balance aims for (see
    /// [`energy::balanced_share`]).
    balanced_share: f64,
}

impl Trades {
    fn new(noise_offset_db: f64) -> Self {
        let scarcity = scarcity(noise_offset_db);
        Self {
            dead_zone: DEAD_ZONE * scarcity,
            angle_allowance: coupling::angle_allowance(scarcity),
            side_loss_db: coupling::side_loss_db(scarcity),
            balanced_share: energy::balanced_share(scarcity),
        }
    }
}

/// How each hop's noise offset is set.
enum Steering {
    /// At the encoder's `noise_offset_db`, a variable rate's.
    Fixed,
    /// By the steering of a mean rate.
    Mean(MeanSteering),
    /// By the steering of a constant rate, each packet kept within its
    /// buffer.
    Constant {
        steering: ConstantSteering,
        buffer: Buffer,
        /// The bits the packets coded since the last hop would have taken
        /// at 0 dB (see [`rate::at_0_db`]).
        hop_at_0_db: f64,
    },
}

/// The least by which a block that takes more bits than a constant rate's
/// buffer holds is coded again at a higher noise offset, in dB, where the
/// bits it took and those the buffer holds would raise it less.
const RAISE_STEP_DB: f64 = 1.5;

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
    /// The noise each line of the current long block, and of the short
    /// blocks that may take its place, may take in each channel, at the
    /// hop's noise offset.
    noise: Vec<Noise>,
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
    /// The noise offset of a variable rate, and at a mean or constant rate
    /// the steering that sets each hop's.
    noise_offset_db: f64,
    steering: Steering,
    /// The noise offset the current long block, or the short blocks that
    /// take its place, are coded at: the hop's, or at a constant rate one
    /// higher, where a block took more bits than the buffer held.
    offset_db: f64,
    /// The trades of noise for bits that the current long block, or the
    /// short blocks that take its place, are coded with, at `offset_db`.
    trades: Trades,
    /// The bytes of the packets coded since the last hop.
    hop_bytes: usize,
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
        let channels = config.channels();
        let sample_rate = config.sample_rate();
        let (noise_offset_db, steering) = match config.control()? {
            Control::Variable { noise_offset_db } => (noise_offset_db, Steering::Fixed),
            Control::Mean { bits_per_second } => {
                let steering = rate::mean_steering(bits_per_second, sample_rate, channels);
                (steering.noise_offset_db(), Steering::Mean(steering))
            }
            Control::Constant { bits_per_second } => {
                let steering = ConstantSteering::new(bits_per_second, sample_rate, channels);
                let noise_offset_db = steering.noise_offset_db();
                let steering = Steering::Constant {
                    steering,
                    buffer: Buffer::new(bits_per_second, sample_rate),
                    hop_at_0_db: 0.0,
                };
                (noise_offset_db, steering)
            }
        };
        let setup = Setup::new(channels);
        let tables = Tables::new(sample_rate);
        let long_window =
            |previous_long, next_long| window(BLOCK, SHORT_BLOCK, true, previous_long, next_long);
        Ok(Self {
            config,
            setup,
            mdcts: [Mdct::new(SHORT_BLOCK), Mdct::new(BLOCK)],
            models: (0..channels).map(|_| Model::new(&tables)).collect(),
            partitions: [false, true].map(|long| tables.partition_of_lines(long)),
            tables,
            current: vec![Analysis::default(); channels],
            next: vec![Analysis::default(); channels],
            noise: vec![Noise::default(); channels],
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
            noise_offset_db,
            steering,
            offset_db: noise_offset_db,
            trades: Trades::new(noise_offset_db),
            hop_bytes: 0,
        })
    }

    /// The noise offset of the next hop: the variable rate's, or that the
    /// hops coded so far steer the mean or constant rate to.
    fn hop_offset_db(&self) -> f64 {
        match &self.steering {
            Steering::Fixed => self.noise_offset_db,
            Steering::Mean(steering) => steering.noise_offset_db(),
            Steering::Constant { steering, .. } => steering.noise_offset_db(),
        }
    }

    /// Codes the current long block, or the short blocks that take its
    /// place, at `noise_offset_db` from here on: the noise each line may
    /// take, and the trades of noise for bits.
    fn set_offset(&mut self, noise_offset_db: f64) {
        self.offset_db = noise_offset_db;
        let allowance = self.allowance(noise_offset_db);
        for (analysis, noise) in self.current.iter().zip(&mut self.noise) {
            analysis.noise(&self.tables, allowance, noise);
        }
        self.trades = Trades::new(noise_offset_db);
    }

    /// The model's allowance at `noise_offset_db`: its factor, and the
    /// treble's weight at the scarcity of bits it brings.
    fn allowance(&self, noise_offset_db: f64) -> Allowance {
        let treble_weight =
            psychoacoustic::treble_weight(self.setup.coupled, scarcity(noise_offset_db));
        Allowance::new(noise_offset_db, treble_weight)
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
            identification: header::identification(
                self.channels(),
                self.sample_rate(),
                self.config.bitrates(),
                BLOCKSIZES,
            ),
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
        // The hop is coded at its offset, which at a mean or constant rate
        // the hops coded so far steer it to, and spent at it.
        self.set_offset(self.hop_offset_db());
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
        match &mut self.steering {
            Steering::Fixed => {}
            Steering::Mean(steering) => steering.spend(self.hop_bytes),
            Steering::Constant {
                steering,
                buffer,
                hop_at_0_db,
            } => {
                steering.spend(*hop_at_0_db, buffer.fullness());
                *hop_at_0_db = 0.0;
            }
        }
        self.hop_bytes = 0;
        self.hops += 1;
        ended
    }

    /// Codes the `short`th short block that takes the current long block's
    /// place, or the long block itself, as the last if its center is at or
    /// past `end`; whose block after it is long if `next_long`. Gives
    /// whether it was the last.
    ///
    /// At a constant rate, the packet is kept within the buffer (see
    /// [`Buffer::len_range`]): a block that takes more bits than the buffer
    /// holds is coded again at a higher noise offset, by
    /// [`RAISE_STEP_DB`] at least, up to the highest the steering keeps, and
    /// the rest of the hop with it; failing that, its residue is cut (see
    /// [`Coded::packet_within`]). A packet that would overfill the buffer
    /// is padded with zero bytes.
    ///
    /// Spec reading: clause 1.1.3 lets a packet be "padded" and "remain
    /// decodable", and a decoder reads nothing of an audio packet past its
    /// residue (clause 4.3), whose decoding ends with its last partition
    /// (clause 8.6.2). The specification does not say what padding
    /// contains; it is zero bytes, as the unused bits of a packet's last
    /// byte are zeroed (clause 2.1.8).
    ///
    /// Spec reading: clause 1.1.3 lets a packet be "truncated" too, and a
    /// packet that ends early decodes, in its floors, as silence (clause
    /// 4.3.2), and in its residue as the vectors decoded up to the end
    /// (clause 8.6.2), which would fit any block to the buffer. No packet
    /// is truncated: a block is cut to fit as a whole packet instead, so
    /// that the stream does not rest on how a decoder reads a packet that
    /// ends early.
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
        let granule_position = match end {
            Some(end) if last => end,
            _ => center,
        };
        // Spec gap: the first block has no block before it to lap with,
        // nor the last one after it, and the specification does not say
        // what their window flags are then. Each is set, as if the block it
        // has not were long: no decoder returns the half of either block
        // that the flag shapes.
        let previous_long = self.last_long;
        let next_long = next_long || last;
        let mut coded = self.code_block(short, previous_long, next_long);
        let mut data = coded.packet(&self.setup);
        if let Steering::Constant { buffer, .. } = &self.steering {
            let (shortest, longest) = buffer.len_range(granule_position, last);
            let highest = *OFFSETS_DB.end();
            while data.len() > longest && self.offset_db < highest {
                let raised =
                    rate::raised_offset_db(self.offset_db, data.len(), longest, RAISE_STEP_DB);
                self.set_offset(raised);
                coded = self.code_block(short, previous_long, next_long);
                data = coded.packet(&self.setup);
            }
            let at_0_db = rate::at_0_db(data.len(), self.offset_db);
            if data.len() > longest {
                data = coded.packet_within(&self.setup, longest);
            }
            data.resize(data.len().max(shortest), 0);
            if let Steering::Constant {
                buffer,
                hop_at_0_db,
                ..
            } = &mut self.steering
            {
                buffer.spend(data.len(), granule_position);
                *hop_at_0_db += at_0_db;
            }
        }
        out.packet(Packet {
            data: &data,
            granule_position,
        });
        self.packets += 1;
        self.hop_bytes += data.len();
        self.last_long = mode.blockflag();
        last
    }

    /// Codes the `short`th short block that takes the current long block's
    /// place, or the long block itself, whose window has `previous_long`
    /// and, for a long block, `next_long` (clause 4.3.1), at `offset_db`.
    ///
    /// Each channel's spectrum is given the floor that
    /// [`psychoacoustic`] sets, and its residue is quantized to it (see
    /// [`Quantized`]). A channel whose residue is all zeros is coded as
    /// unused. In a stereo stream, where both are used, the two channels'
    /// floors are then matched where the channels are alike, and their
    /// residues quantized to them again (see [`coupling::match_floors`]),
    /// and the residues coupled (see [`coupling::couple`]); in any other,
    /// the energy of each used channel's lines is balanced (see
    /// [`energy::balance`]) and kept (see [`energy::keep`]), as the
    /// coupling balances and keeps it in stereo.
    fn code_block(&mut self, short: Option<usize>, previous_long: bool, next_long: bool) -> Coded {
        let mode = if short.is_some() {
            Mode::Short
        } else {
            Mode::Long
        };
        let n = BLOCKSIZES[usize::from(mode.blockflag())];
        let start = short.map_or(0, |k| FIRST_SHORT + k * SHORT_BLOCK / 2);
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
        let noises: Vec<&[f64]> = (self.noise.iter())
            .map(|noise| match short {
                Some(k) => &noise.short[k][..],
                None => &noise.long[..],
            })
            .collect();
        let lines = n / 2;
        let mut windowed = vec![0.0; n];
        let mut spectra = vec![vec![0.0; lines]; self.pcm.len()];
        let (mut targets, mut channels): (Vec<Vec<i32>>, Vec<Quantized>) = (self.pcm.iter())
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
                let targets = psychoacoustic::floor_targets(noise, floor);
                let quantized =
                    Quantized::new(floor, &targets, spectrum, limit, self.trades.dead_zone);
                (targets, quantized)
            })
            .unzip();
        // Coupled, where both channels are used, their floors are matched
        // where the channels are alike (see `coupling::match_floors`).
        let both_used = channels.iter().all(Quantized::used);
        if let ([left, right], true) = (&mut targets[..], self.setup.coupled && both_used) {
            coupling::match_floors([left, right], [&spectra[0], &spectra[1]], floor);
            for ((channel, targets), spectrum) in channels.iter_mut().zip(&targets).zip(&spectra) {
                *channel = Quantized::new(floor, targets, spectrum, limit, self.trades.dead_zone);
            }
        }
        let used: Vec<bool> = channels.iter().map(Quantized::used).collect();
        let kept = energy::kept_lines(lines, self.config.sample_rate());
        let protected = lines_below(coupling::PROTECTED_HZ, n, self.config.sample_rate());
        let partitions = &self.partitions[usize::from(mode.blockflag())];
        if let ([left, right], true) = (&mut channels[..], self.setup.coupled) {
            coupling::couple(
                [&mut left.residue, &mut right.residue],
                [&left.curve, &right.curve],
                [&spectra[0], &spectra[1]],
                [noises[0], noises[1]],
                partitions,
                coupling::Narrowing {
                    protected,
                    allowance: self.trades.angle_allowance,
                    side_loss_db: self.trades.side_loss_db,
                },
                kept,
                limit,
                self.trades.balanced_share,
            );
        } else {
            for (((channel, spectrum), noise), &used) in
                (channels.iter_mut().zip(&spectra).zip(&noises)).zip(&used)
            {
                if used {
                    energy::balance(
                        &mut channel.residue,
                        spectrum,
                        &channel.curve,
                        limit,
                        |_| false,
                        kept,
                        self.trades.balanced_share,
                    );
                    let protected = energy::Protected {
                        lines: protected,
                        noise,
                        partitions,
                    };
                    energy::keep(
                        &mut channel.residue,
                        spectrum,
                        &channel.curve,
                        protected,
                        kept,
                        None,
                    );
                }
            }
        }
        let floors = (channels.iter_mut().zip(used))
            .map(|(channel, used)| used.then(|| std::mem::take(&mut channel.values)))
            .collect();
        let residues = channels.into_iter().map(|c| c.residue).collect();
        Coded {
            mode,
            previous_long,
            next_long,
            floors,
            residues,
        }
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
    /// The floor of the Y values `targets`, raised where a line would be
    /// more than `limit` steps of it, and the residue of `spectrum`: each
    /// line divided by the floor there and rounded, nearer zero by
    /// `dead_zone` (see [`DEAD_ZONE`]), and held within `limit`, which only
    /// a line far past full scale passes.
    fn new(floor: &Floor, targets: &[i32], spectrum: &[f64], limit: i32, dead_zone: f64) -> Self {
        // The floor the model sets, raised where it would put a line past
        // the residue's reach, as at an attack after silence, where the
        // threshold is held near the threshold in quiet.
        let (values, curve) = floor.values_within(targets, spectrum, f64::from(limit));
        let residue = (spectrum.iter().zip(&curve))
            .map(|(&x, &f)| {
                let steps = x / f64::from(f);
                let q = steps.signum() * (steps.abs() + 0.5 - dead_zone).floor();
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

/// A block coded, before its packet is written: what [`audio::packet`]
/// takes.
struct Coded {
    mode: Mode,
    previous_long: bool,
    next_long: bool,
    /// Each channel's floor's values, `None` where unused.
    floors: Vec<Option<Vec<u32>>>,
    /// Each channel's residue, after any coupling.
    residues: Vec<Vec<i32>>,
}

impl Coded {
    /// The block's packet.
    fn packet(&self, setup: &Setup) -> Vec<u8> {
        self.write(setup, &self.floors, &self.residues)
    }

    /// The block's packet, each residue's lines from `cut` on coded as 0.
    fn packet_below(&self, setup: &Setup, cut: usize) -> Vec<u8> {
        let residues: Vec<Vec<i32>> = (self.residues.iter())
            .map(|residue| {
                let mut residue = residue.clone();
                let cut = cut.min(residue.len());
                residue[cut..].fill(0);
                residue
            })
            .collect();
        self.write(setup, &self.floors, &residues)
    }

    /// The block's packet in `longest` bytes at most, where it is longer:
    /// each residue's lines coded as 0 from a line found by bisection, the
    /// highest at which the packet fits where its length rises with the
    /// line, as it does but for a few bits; or, where even every line coded
    /// as 0 leaves its floors too long, the block coded as silence, every
    /// channel unused, in two bytes at most.
    fn packet_within(&self, setup: &Setup, longest: usize) -> Vec<u8> {
        let fits = |cut| {
            let data = self.packet_below(setup, cut);
            (data.len() <= longest).then_some(data)
        };
        let Some(mut within) = fits(0) else {
            let silent = vec![None; self.floors.len()];
            let zeros = vec![vec![0; self.residues[0].len()]; self.residues.len()];
            return self.write(setup, &silent, &zeros);
        };
        // The packet fits with its residues cut from `low` on, and not from
        // `high` on.
        let (mut low, mut high) = (0, self.residues[0].len());
        while high - low > 1 {
            let mid = (low + high) / 2;
            match fits(mid) {
                Some(data) => (low, within) = (mid, data),
                None => high = mid,
            }
        }
        within
    }

    fn write(&self, setup: &Setup, floors: &[Option<Vec<u32>>], residues: &[Vec<i32>]) -> Vec<u8> {
        let floors: Vec<Option<&[u32]>> = floors.iter().map(Option::as_deref).collect();
        let residues: Vec<&[i32]> = residues.iter().map(Vec::as_slice).collect();
        audio::packet(
            setup,
            self.mode,
            self.previous_long,
            self.next_long,
            &floors,
            &residues,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stereo pair takes the treble's noise and the bound of its image as
    /// bits are scarce or plentiful; one channel the treble's noise whatever
    /// the rate.
    #[test]
    fn the_encoder_weighs_the_treble_and_bounds_the_image_by_the_rate() {
        let encoder = |channels: usize, noise_offset_db: f64| {
            let rate = crate::Rate::Variable { noise_offset_db };
            Encoder::new(Config::new(44_100, channels, rate)).unwrap()
        };
        let (plentiful, scarce) = (
            encoder(2, PLENTIFUL_OFFSET_DB),
            encoder(2, SCARCE_OFFSET_DB),
        );
        let treble = |encoder: &Encoder, offset| encoder.allowance(offset).treble_weight;
        assert_eq!(
            treble(&plentiful, PLENTIFUL_OFFSET_DB),
            psychoacoustic::treble_weight(true, 0.0)
        );
        assert_eq!(treble(&scarce, SCARCE_OFFSET_DB), 1.0);
        assert_eq!(plentiful.trades.side_loss_db, coupling::SIDE_LOSS_DB);
        assert_eq!(scarce.trades.side_loss_db, f64::INFINITY);
        for offset in [PLENTIFUL_OFFSET_DB, SCARCE_OFFSET_DB] {
            let mono = encoder(1, offset);
            assert_eq!(
                treble(&mono, offset),
                psychoacoustic::treble_weight(false, 0.0)
            );
            assert!(treble(&mono, offset) > 1.0);
        }
        assert!(treble(&plentiful, PLENTIFUL_OFFSET_DB) > 1.0);
    }

    /// The packets of `pcm`, stereo at 44.1 kHz, given a hop at a time,
    /// coded at `rate`, and the noise offset of each hop coded; with
    /// `offsets`, each hop is coded at its offset there in place of the
    /// rate's.
    fn encode(pcm: &[f32], rate: crate::Rate, offsets: Option<&[f64]>) -> (Vec<Vec<u8>>, Vec<f64>) {
        let mut encoder = Encoder::new(Config::new(44_100, 2, rate)).unwrap();
        let (mut packets, mut coded) = (Vec::new(), Vec::new());
        for chunk in pcm.chunks(2 * HOP as usize) {
            let hops = encoder.hops as usize;
            if let Some(&offset) = offsets.and_then(|offsets| offsets.get(hops)) {
                encoder.noise_offset_db = offset;
            }
            let offset = encoder.hop_offset_db();
            let mut out = |packet: Packet<'_>| packets.push(packet.data.to_vec());
            encoder.encode_interleaved(chunk, &mut out).unwrap();
            // A hop of input completes at most one hop of blocks.
            match encoder.hops as usize - hops {
                0 => {}
                1 => coded.push(offset),
                n => panic!("{n} hops at once"),
            }
        }
        (packets, coded)
    }

    /// Ten seconds of a chord and of noise, louder in its second half.
    fn music() -> Vec<f32> {
        let mut seed = 0x9e37_79b9u32;
        (0..2 * 10 * 44_100)
            .map(|i| {
                let t = (i / 2) as f64 / 44_100.0;
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = f64::from(seed >> 8) / f64::from(1u32 << 24) - 0.5;
                let level = if t < 5.0 { 0.02 } else { 0.3 };
                let chord: f64 = [220.0, 277.2, 329.6]
                    .iter()
                    .map(|hz| (std::f64::consts::TAU * hz * t).sin())
                    .sum();
                (0.1 * chord + level * noise) as f32
            })
            .collect()
    }

    /// At a mean rate, the packets keep near it whatever it is, the offset
    /// moving from hop to hop; and each hop is coded as the variable rate of
    /// its offset codes it, byte for byte.
    #[test]
    fn a_mean_rate_codes_each_hop_as_the_variable_rate_of_its_offset() {
        let pcm = music();
        for bits_per_second in [96_000, 160_000] {
            let (packets, offsets) = encode(&pcm, crate::Rate::Mean { bits_per_second }, None);
            let bits = 8.0 * packets.iter().map(Vec::len).sum::<usize>() as f64;
            let achieved = bits * 44_100.0 / (offsets.len() as u64 * HOP) as f64;
            let error = achieved / f64::from(bits_per_second) - 1.0;
            assert!(error.abs() < 0.05, "{bits_per_second}: {achieved:.0}");
            let start = rate::mean_steering(bits_per_second, 44_100, 2).noise_offset_db();
            assert_eq!(offsets[0], start);
            assert!(offsets.iter().any(|&offset| (offset - start).abs() > 1.0));
            let variable = crate::Rate::Variable {
                noise_offset_db: start,
            };
            let (replayed, replayed_offsets) = encode(&pcm, variable, Some(&offsets));
            assert_eq!(replayed_offsets, offsets);
            assert!(replayed == packets, "{bits_per_second}");
        }
    }

    /// At a variable rate, every hop is coded at the rate's offset.
    #[test]
    fn a_variable_rate_keeps_its_offset() {
        let rate = crate::Rate::Variable {
            noise_offset_db: -5.0,
        };
        let (_, offsets) = encode(&music(), rate, None);
        assert!(offsets.iter().all(|&offset| offset == -5.0));
    }

    /// The dead zone and the narrowing of angles are whole where bits are
    /// scarce, and fall, never rising, as the offset falls.
    #[test]
    fn the_trades_of_noise_for_bits_taper_as_the_offset_falls() {
        assert_eq!(scarcity(SCARCE_OFFSET_DB), 1.0);
        assert_eq!(scarcity(10.0), 1.0);
        assert_eq!(scarcity(PLENTIFUL_OFFSET_DB), 0.0);
        assert_eq!(scarcity(-30.0), 0.0);
        let offsets: Vec<f64> = (-120..=20).map(|o| f64::from(o) / 10.0).collect();
        for pair in offsets.windows(2) {
            assert!(scarcity(pair[0]) <= scarcity(pair[1]));
        }
    }

    /// The packets of `pcm`, interleaved, of `channels` channels at
    /// `sample_rate`, coded at `rate`, each with its granule position.
    fn packets(
        pcm: &[f32],
        sample_rate: u32,
        channels: usize,
        rate: crate::Rate,
    ) -> Vec<(Vec<u8>, u64)> {
        let mut encoder = Encoder::new(Config::new(sample_rate, channels, rate)).unwrap();
        let mut packets = Vec::new();
        let mut out = |packet: Packet<'_>| {
            packets.push((packet.data.to_vec(), packet.granule_position));
        };
        encoder.encode_interleaved(pcm, &mut out).unwrap();
        encoder.finish(&mut out);
        packets
    }

    /// What the buffer of a constant rate of `bits_per_second` at
    /// `sample_rate` holds after each of `packets`, in bits times the
    /// sample rate, counted afresh from the definition of
    /// [`crate::Rate::Constant`], and its size.
    fn held(
        packets: &[(Vec<u8>, u64)],
        bits_per_second: u32,
        sample_rate: u32,
    ) -> (Vec<i128>, i128) {
        let (rate, fs) = (i128::from(bits_per_second), i128::from(sample_rate));
        let size = rate * fs.max(2048);
        let (mut held, mut position) = (size / 2, 0);
        let held = (packets.iter())
            .map(|(data, granule_position)| {
                let samples = i128::from(*granule_position) - position;
                assert!(samples >= 0);
                position = i128::from(*granule_position);
                held += rate * samples - 8 * data.len() as i128 * fs;
                held
            })
            .collect();
        (held, size)
    }

    /// At a constant rate, the buffer holds from none to its size after
    /// every packet, in stereo and mono, at rates from the lowest to high,
    /// of music loud and quiet; its packets' lengths vary, and the
    /// identification header gives the rate as all three bitrates.
    #[test]
    fn a_constant_rate_keeps_its_packets_within_the_buffer() {
        let stereo = music();
        let mono: Vec<f32> = stereo.chunks(2).map(|pair| pair[0]).collect();
        for (pcm, channels, bits_per_second) in [
            (&stereo, 2, 96_000),
            (&stereo, 2, 160_000),
            (&stereo, 2, 11_025),
            (&mono, 1, 64_000),
            (&mono, 1, 5_513),
        ] {
            let rate = crate::Rate::Constant { bits_per_second };
            let packets = packets(pcm, 44_100, channels, rate);
            let (held, size) = held(&packets, bits_per_second, 44_100);
            for (k, &held) in held.iter().enumerate() {
                assert!((0..=size).contains(&held), "{bits_per_second}: packet {k}");
            }
            let lengths: Vec<usize> = packets.iter().map(|(data, _)| data.len()).collect();
            assert!(lengths.iter().min() < lengths.iter().max());
            let encoder = Encoder::new(Config::new(44_100, channels, rate)).unwrap();
            let header = encoder.headers().identification;
            let rate = (bits_per_second as i32).to_le_bytes();
            assert_eq!(header[16..28], [rate, rate, rate].concat());
        }
    }

    /// Silence takes far fewer bits than the rate: its packets are padded
    /// with zero bytes, once the buffer is full, so that the stream takes
    /// the rate's bits less half the buffer, within a byte.
    #[test]
    fn silence_is_padded_to_the_rate() {
        let pcm = vec![0.0; 2 * 10 * 44_100];
        let packets = packets(
            &pcm,
            44_100,
            2,
            crate::Rate::Constant {
                bits_per_second: 128_000,
            },
        );
        let bits: usize = packets.iter().map(|(data, _)| 8 * data.len()).sum();
        let expected = 128_000 * 10 - 64_000;
        assert!(bits <= expected && expected - bits < 8, "{bits}");
        // The first packets, while the buffer fills, are of one byte;
        // the rest padded with zeros.
        assert_eq!(packets[1].0, [0b0_1110]);
        let last = &packets[packets.len() - 2].0;
        assert!(last.len() > 300 && last[1..].iter().all(|&b| b == 0));
    }

    /// Bursts of loud noise after silence, 64 samples of every 512, take
    /// more bits than 128 kbit/s in stereo, even at the highest offset the
    /// steering keeps, as the floor of each short block after silence is
    /// raised to keep the burst's lines within the residue's reach. Each
    /// block is cut to fit the buffer, which holds from none to its size
    /// after every packet: at 128 kbit/s in its residue alone, and at the
    /// lowest rate, 11 kbit/s, where the floors of many blocks take more
    /// than the buffer holds, those blocks are coded as silence.
    #[test]
    fn bursts_past_the_rate_at_any_offset_are_cut_to_fit() {
        let mut seed = 0x1234_5678u32;
        let pcm: Vec<f32> = (0..2 * 5 * 44_100)
            .map(|i| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = f64::from(seed >> 8) / f64::from(1u32 << 23) - 1.0;
                if (i / 2) % 512 < 64 {
                    noise as f32
                } else {
                    0.0
                }
            })
            .collect();
        let variable = crate::Rate::Variable {
            noise_offset_db: *OFFSETS_DB.end(),
        };
        let uncut = packets(&pcm, 44_100, 2, variable);
        let bytes: usize = uncut.iter().map(|(data, _)| data.len()).sum();
        assert!(8 * bytes > 5 * 128_000, "{bytes}");
        let coded = uncut.iter().filter(|(data, _)| data.len() > 2).count();
        for (bits_per_second, silenced) in [(128_000, false), (11_025, true)] {
            let rate = crate::Rate::Constant { bits_per_second };
            let packets = packets(&pcm, 44_100, 2, rate);
            let (held, size) = held(&packets, bits_per_second, 44_100);
            assert!(held.iter().all(|held| (0..=size).contains(held)));
            // The blocks are those of the variable rate, which the offset
            // does not choose.
            assert_eq!(packets.len(), uncut.len());
            let silent = (packets.iter().zip(&uncut))
                .filter(|((data, _), (uncut, _))| data.len() <= 2 && uncut.len() > 2)
                .count();
            assert_eq!(
                silent > 0,
                silenced,
                "{bits_per_second}: {silent} of {coded}"
            );
        }
    }
}
