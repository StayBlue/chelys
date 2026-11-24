// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Runs each channel's PCM through its filterbank granule by granule as it
//! arrives, has a [`Coding`] quantise each frame at the bitrate the rate
//! chooses, from the bit reservoir, and writes the frames.

use super::StreamConfig;
use super::coding::{CodedFrame, Coding, Filtered};
use super::layout::{FrameWriter, frame_contents, main_data_bits};
use crate::DELAY;
use crate::ancillary::AncillaryQueue;
use crate::granule::{BlockType, GRANULE_LINES, MAX_PART2_3_LENGTH};
use crate::header::{Bitrate, PaddingSchedule, Version};
use crate::rate::{MeanRateSchedule, StreamRate};
use crate::reservoir::Reservoir;
use crate::{filterbank, rate, scalefactor_bands};
use chelys_core::FrameSink;

/// The PCM a [`FrameEncoder`] needs past a frame before coding it: two
/// granules, as a granule's block type depends on whether the granule after
/// it has an attack, and the filterbank needs the block type of the granule
/// after the one it filters.
const LOOK_AHEAD: usize = 2 * GRANULE_LINES;

/// How [`FrameEncoder`] chooses the bitrate of each frame.
enum Schedule {
    Constant(Bitrate, PaddingSchedule),
    Mean(MeanRateSchedule),
    Variable {
        /// The bitrates a frame may have, lowest first.
        bitrates: Vec<Bitrate>,
        /// The index in `bitrates` of the frame before.
        previous: usize,
    },
}

impl Schedule {
    fn new(config: &StreamConfig) -> Self {
        let fs = config.sampling_frequency;
        assert_eq!(
            config.rate.version(),
            fs.version(),
            "{:?} is not a rate of {fs:?}",
            config.rate
        );
        match config.rate {
            StreamRate::Constant(bitrate) => {
                Self::Constant(bitrate, PaddingSchedule::new(bitrate, fs))
            }
            StreamRate::Mean(mean) => Self::Mean(MeanRateSchedule::new(mean, fs)),
            StreamRate::Variable(variable) => Self::Variable {
                bitrates: variable.bitrates(),
                previous: 0,
            },
        }
    }
}

/// A frame quantised at one bitrate, with the reservoir after it.
struct FrameAtBitrate {
    bitrate: Bitrate,
    padding: bool,
    coded: CodedFrame,
    main_data_begin: usize,
    /// The bits of the frame's main data slot beyond the granules' shares
    /// and its ancillary data.
    unshared_bits: usize,
    reservoir: Reservoir,
}

/// Runs each channel through its filterbank granule by granule as its PCM
/// arrives, quantises each frame as `coding` does, and writes the frames.
pub(super) struct FrameEncoder<C: Coding> {
    pub(super) config: StreamConfig,
    pub(super) coding: C,
    /// The reservoir between frames.
    reservoir: Reservoir,
    /// One for each channel.
    filterbanks: Vec<filterbank::Filterbank>,
    schedule: Schedule,
    pub(super) writer: FrameWriter,
    /// The bytes the reservoir holds for the next frame.
    pub(super) reservoir_bytes: usize,
    /// Each channel's PCM from the first granule not yet filtered.
    pcm: Vec<Vec<f64>>,
    /// The samples each channel has been given.
    samples: u64,
    /// The length of each frame written.
    frame_lens: Vec<usize>,
    pub(super) ancillary: AncillaryQueue,
    /// For each channel, the block type and analysis of the granule after
    /// the next to be filtered.
    ahead: Vec<Option<(BlockType, C::Analysis)>>,
}

/// What a [`FrameEncoder`] wrote, once it has finished.
pub(super) struct Ended {
    /// The length of each frame.
    pub(super) frame_lens: Vec<usize>,
    /// The samples each channel was given.
    pub(super) samples: u64,
    /// The bits of ancillary data that the last frames had no room for.
    pub(super) ancillary_dropped: usize,
}

impl<C: Coding> FrameEncoder<C> {
    pub(super) fn new(config: StreamConfig, coding: C, reservoir: Reservoir) -> Self {
        let channels = config.mode.channels();
        Self {
            schedule: Schedule::new(&config),
            config,
            coding,
            reservoir,
            filterbanks: (0..channels)
                .map(|_| filterbank::Filterbank::new())
                .collect(),
            writer: FrameWriter::default(),
            reservoir_bytes: 0,
            pcm: vec![Vec::new(); channels],
            samples: 0,
            frame_lens: Vec::new(),
            ancillary: AncillaryQueue::default(),
            ahead: (0..channels).map(|_| None).collect(),
        }
    }

    /// Codes the frames whose PCM, and the granule after it, have arrived,
    /// and gives `out` the frames that are final.
    pub(super) fn encode<O: FrameSink + ?Sized>(&mut self, channels: &[&[f64]], out: &mut O) {
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
        let frame_samples = self.samples_per_frame();
        let mut start = 0;
        while self.pcm[0].len() - start >= frame_samples + LOOK_AHEAD {
            self.encode_frame(start);
            start += frame_samples;
        }
        for pcm in &mut self.pcm {
            pcm.drain(..start);
        }
        self.writer.take_final(out);
    }

    /// Pads the input with silence to whole frames, at least [`DELAY`]
    /// samples past it, so that the decoded output holds all of the input,
    /// codes them, and gives `out` the frames left.
    pub(super) fn finish<O: FrameSink + ?Sized>(mut self, out: &mut O) -> Ended {
        let frame_samples = self.samples_per_frame();
        let frames = (self.samples + DELAY as u64).div_ceil(frame_samples as u64);
        // Few: the frames of the PCM held and of DELAY.
        let remaining = (frames - self.frame_lens.len() as u64) as usize;
        for pcm in &mut self.pcm {
            pcm.resize(remaining * frame_samples + LOOK_AHEAD, 0.0);
        }
        for frame in 0..remaining {
            self.encode_frame(frame * frame_samples);
        }
        self.writer.finish(out);
        Ended {
            frame_lens: self.frame_lens,
            samples: self.samples,
            ancillary_dropped: self.ancillary.pending(),
        }
    }

    pub(super) fn samples_per_frame(&self) -> usize {
        self.config.sampling_frequency.version().samples_per_frame()
    }

    /// Codes the frame whose PCM starts at `start` in [`Self::pcm`].
    fn encode_frame(&mut self, start: usize) {
        let Self {
            config,
            coding,
            filterbanks,
            pcm,
            ahead,
            ..
        } = self;
        let fs = config.sampling_frequency;
        let granule_pcm = |channel: usize, granule: usize| -> &[f64; GRANULE_LINES] {
            let from = start + granule * GRANULE_LINES;
            pcm[channel][from..from + GRANULE_LINES]
                .try_into()
                .expect("a granule of PCM")
        };
        if self.frame_lens.is_empty() {
            for (channel, ahead) in ahead.iter_mut().enumerate() {
                assert!(
                    coding
                        .look_ahead(channel, granule_pcm(channel, 0))
                        .is_none()
                );
                *ahead = coding.look_ahead(channel, granule_pcm(channel, 1));
            }
        }
        let granules = fs.version().granules();
        let mut filtered: Vec<Vec<Filtered<C::Analysis>>> = (0..granules)
            .map(|gr| {
                (filterbanks.iter_mut().enumerate())
                    .map(|(channel, filterbank)| {
                        let next = coding.look_ahead(channel, granule_pcm(channel, gr + 2));
                        let next_type = next.as_ref().map(|(block_type, _)| *block_type);
                        let (block_type, analysis) = std::mem::replace(&mut ahead[channel], next)
                            .expect("the granule has been analysed");
                        let next_type = next_type.expect("the granule after has been analysed");
                        let xr =
                            filterbank.granule(granule_pcm(channel, gr), block_type, next_type);
                        let xr = match block_type {
                            BlockType::Short => scalefactor_bands::short_coding_order(&xr, fs),
                            BlockType::Mixed => scalefactor_bands::mixed_coding_order(&xr, fs),
                            _ => xr,
                        };
                        Filtered {
                            block_type,
                            analysis,
                            xr,
                        }
                    })
                    .collect()
            })
            .collect();
        coding.prepare(&mut filtered);
        let block_types: Vec<BlockType> = filtered.iter().flatten().map(|f| f.block_type).collect();
        let ancillary_bits = self.ancillary.next_frame_bits();
        let kept = match &mut self.schedule {
            Schedule::Constant(bitrate, padding) => {
                let (bitrate, padding) = (*bitrate, padding.next_padding());
                self.quantize_at(&filtered, bitrate, padding, true, ancillary_bits, false)
                    .expect("a frame coded in full")
            }
            Schedule::Mean(schedule) => {
                let (bitrate, padding) = schedule.next_frame();
                self.quantize_at(&filtered, bitrate, padding, true, ancillary_bits, false)
                    .expect("a frame coded in full")
            }
            Schedule::Variable { bitrates, previous } => {
                // From the bitrate below the last frame's, at which most
                // frames within their allowance at the last frame's are
                // within it too (see `rate::search_variable`).
                let (mut bitrates, start) = (bitrates.clone(), previous.saturating_sub(1));
                bitrates.truncate(self.useful_bitrates(&bitrates, ancillary_bits));
                let start = start.min(bitrates.len() - 1);
                let once = (self.config.sampling_frequency.version() == Version::Mpeg1
                    && block_types.iter().all(|&b| b == BlockType::Normal))
                .then(|| self.quantize_within(&filtered, &bitrates, ancillary_bits))
                .flatten();
                let (index, kept) = once.unwrap_or_else(|| {
                    let highest = bitrates.len() - 1;
                    // The search keeps a frame that is not within its
                    // allowance only at the highest bitrate, so below it,
                    // coding stops once a frame is known not to be.
                    let (index, kept) = rate::search_variable(bitrates.len(), start, |index| {
                        let at = self.quantize_at(
                            &filtered,
                            bitrates[index],
                            false,
                            index == highest,
                            ancillary_bits,
                            index != highest,
                        );
                        let within = at.as_ref().is_some_and(|at| at.coded.excess <= 1.0);
                        (at, within)
                    });
                    let kept =
                        kept.expect("a frame within its allowance or at the highest bitrate");
                    (index, kept)
                });
                if let Schedule::Variable { previous, .. } = &mut self.schedule {
                    *previous = index;
                }
                kept
            }
        };
        let FrameAtBitrate {
            bitrate,
            padding,
            coded,
            main_data_begin,
            unshared_bits,
            reservoir,
        } = kept;
        assert!(
            (coded.granules.iter().flatten())
                .map(|q| q.block_type)
                .eq(block_types),
            "the granules keep their block types"
        );
        let header =
            self.config
                .frame_header(bitrate, padding, coded.ms_stereo, coded.intensity_stereo);
        let (side_info, main_data) = frame_contents(
            &header,
            &coded.granules,
            main_data_begin,
            Some((&mut self.ancillary, ancillary_bits)),
        );
        self.writer.write(&header, &side_info, &main_data);
        self.reservoir = reservoir;
        self.reservoir_bytes = self.reservoir.end_frame(unshared_bits);
        self.writer.keep_unused(self.reservoir_bytes);
        self.frame_lens.push(header.frame_len());
    }

    /// How many of `bitrates`, lowest first, a frame at a variable bitrate
    /// is coded at: those up to the lowest whose main data slot, less
    /// `ancillary_bits`, holds [`MAX_PART2_3_LENGTH`] bits for every granule
    /// of every channel; or all of them if none does.
    ///
    /// Spec gap: neither standard says how to choose a frame's bitrate (see
    /// [`rate::search_variable`]). A granule of a channel has at most 4095
    /// bits, as `part2_3_length` has 12 (clause 2.4.1.7), and at that
    /// bitrate the slot alone gives every granule as many: above it no
    /// granule can be given more, and the rest of the slot goes to the bit
    /// reservoir, which keeps at most 511 bytes, as `main_data_begin` has 9
    /// bits, or 255 at the lower sampling frequencies, where it has 8; or it
    /// is stuffing. The search keeps a frame that is not within its
    /// allowance at the highest bitrate it tries, which is then this one.
    /// In one channel it is 256 kbit/s at 32 kHz, 128 at 16 kHz, 96 at
    /// 11.025 and 12 kHz and 64 at 8 kHz, and in two 128 at 8 kHz; at the
    /// other sampling frequencies and modes no bitrate of the table holds
    /// that much (at 44.1 kHz, 320 kbit/s gives one channel 8184 bits), and
    /// every bitrate is kept.
    ///
    /// At 8 kHz, where the bands fixed at zero (see
    /// [`fixed_at_zero`](scalefactor_bands::fixed_at_zero)) are within their
    /// allowance only at a fine step, many frames are not: over one minute
    /// of each of nine tracks, one of each kind of music, at level 100 in
    /// two channels, 28% of the frames were kept at 160 kbit/s, and the mean
    /// fell from 84.9 kbit/s to 78.9 with them at 128. Of the same tracks'
    /// streams at levels 0, 80 and 100 in one channel at 32 kHz, 6 of 27
    /// kept frames at 320 kbit/s, 1 to 7 each, at levels 80 and 100; at 256
    /// the frame after each but two took 64 to 80 kbit/s more, from the
    /// reservoir's smaller store, and the streams came out the same length
    /// to 288 bytes shorter. At 16 kHz none kept a frame above 128 kbit/s.
    /// Quiet tones at the edges of the subbands, whose allowance in some band
    /// no step met, kept 98% of their frames at the highest bitrate at every
    /// level; with the partitions converted to the bands at their own
    /// frequencies and the threshold in quiet as the least a band is allowed
    /// (see [`Masking::allowed_distortion`](crate::psychoacoustic::Masking::allowed_distortion)),
    /// they take 8 to 38 kbit/s. Since then, of one minute of each of
    /// those nine tracks at 32 kHz in one channel, 3 streams kept 4 frames
    /// at 256 kbit/s at level 80 and 5 kept 9 at level 100, none at 16 kHz,
    /// and at 8 kHz in two channels at level 100, 15% of the frames were
    /// kept at 128. Where no bitrate is dropped, at 44.1 and 48 kHz and
    /// in two channels at 32 and 16 kHz, those tracks and tones came out
    /// byte for byte as with every bitrate kept.
    fn useful_bitrates(&self, bitrates: &[Bitrate], ancillary_bits: usize) -> usize {
        let fs = self.config.sampling_frequency;
        let most = fs.version().granules() * self.pcm.len() * MAX_PART2_3_LENGTH;
        (bitrates.iter())
            .position(|&bitrate| {
                let header = self.config.frame_header(bitrate, false, false, false);
                main_data_bits(&header).saturating_sub(ancillary_bits) >= most
            })
            .map_or(bitrates.len(), |index| index + 1)
    }

    /// Quantises a frame once, each granule given all the bits it may have
    /// and free to leave any of them, and keeps it at the lowest of
    /// `bitrates` whose frame, with the bit reservoir, holds what it uses;
    /// with the index of that bitrate. `None` if the frame's noise is not
    /// within its allowance (see [`loops::excess`](crate::loops::excess)), or no frame holds it.
    ///
    /// Spec gap: neither standard says how to choose a frame's bitrate (see
    /// [`rate::search_variable`]). Coding a frame at each bitrate the search
    /// tries took about 3.3 codings of it, two thirds of them discarded.
    /// Given more bits than it needs, a granule's outer loop amplifies no
    /// band, and the saving search leaves it at the coarsest step at which
    /// every band is within its allowance (see
    /// [`loops::outer_loop`](crate::loops::outer_loop)): one
    /// band's noise is at its allowance and the rest below theirs, where a
    /// tight budget would have the loop shape the noise up to the allowance
    /// of more bands for fewer bits. Graded with PEAQ (GstPEAQ 0.6.1, black
    /// box) against the search, 22 tracks of pop music and 14 of rock, in
    /// stereo, came out 2.6% and 2.9% larger at 44.1 kHz and graded 0.048
    /// and 0.045 higher (their worst tracks 0.059 and 0.099), and 1.7% and
    /// 2.7% larger at 48 kHz for 0.045 and 0.046 (0.056 and 0.086); matched
    /// in size at 44.1 kHz, they graded 0.027 and 0.020 higher (0.024 and
    /// 0.069). Encoding took 37% less time. The bits a frame leaves beyond
    /// the reservoir's room are stuffing, under 0.05% of a stream's.
    ///
    /// Spec gap: frames with short, start or stop blocks, which spend all
    /// the bits they are given, keep the search, as do the lower sampling
    /// frequencies. There a frame has one granule and the highest bitrate
    /// is 160 kbit/s; coded once, streams came out 22% larger at 22.05 kHz
    /// and 39% at 11.025 kHz, and half the frames were not held at any
    /// bitrate.
    fn quantize_within(
        &mut self,
        frame: &[Vec<Filtered<C::Analysis>>],
        bitrates: &[Bitrate],
        ancillary_bits: usize,
    ) -> Option<(usize, FrameAtBitrate)> {
        let fs = self.config.sampling_frequency;
        let coded = self.coding.quantize(
            frame,
            MAX_PART2_3_LENGTH,
            &mut Reservoir::unbounded(),
            false,
            true,
        )?;
        let used: usize = (coded.granules.iter().flatten())
            .map(|granule| granule.bit_len(fs))
            .sum();
        let slot_bits = |bitrate: Bitrate| {
            main_data_bits(&self.config.frame_header(bitrate, false, false, false))
                .checked_sub(ancillary_bits)
                .expect("the ancillary data fits in the frame")
        };
        let reservoir = |bitrate: Bitrate| self.reservoir.for_frame(bitrate.frame_len(fs, true));
        let index = (bitrates.iter())
            .position(|&bitrate| used <= 8 * reservoir(bitrate).bytes() + slot_bits(bitrate))?;
        let bitrate = bitrates[index];
        let (mut reservoir, slot_bits) = (reservoir(bitrate), slot_bits(bitrate));
        let main_data_begin = reservoir.bytes();
        let shares = frame.len() * self.pcm.len();
        let mean_bits = slot_bits / shares;
        for granule in coded.granules.iter().flatten() {
            reservoir.granule_done(mean_bits, granule.bit_len(fs));
        }
        Some((
            index,
            FrameAtBitrate {
                bitrate,
                padding: false,
                coded,
                main_data_begin,
                unshared_bits: slot_bits - shares * mean_bits,
                reservoir,
            },
        ))
    }

    /// Quantises a frame at `bitrate`, with or without `padding`, from the
    /// reservoir as the frame finds it, trying intensity stereo if
    /// `intensity` is set and the mode allows it, with `ancillary_bits` of
    /// its main data slot kept for ancillary data; or `None` if `only_within`
    /// is set and the frame's noise is not within its allowance (see
    /// [`Coding::quantize`]).
    fn quantize_at(
        &mut self,
        frame: &[Vec<Filtered<C::Analysis>>],
        bitrate: Bitrate,
        padding: bool,
        intensity: bool,
        ancillary_bits: usize,
        only_within: bool,
    ) -> Option<FrameAtBitrate> {
        let fs = self.config.sampling_frequency;
        let mut reservoir = self.reservoir.for_frame(bitrate.frame_len(fs, true));
        let main_data_begin = reservoir.bytes();
        // mode_extension does not change the frame's length.
        let slot_bits = main_data_bits(&self.config.frame_header(bitrate, padding, false, false))
            .checked_sub(ancillary_bits)
            .expect("the ancillary data fits in the frame");
        let shares = frame.len() * self.pcm.len();
        let mean_bits = slot_bits / shares;
        let coded =
            self.coding
                .quantize(frame, mean_bits, &mut reservoir, intensity, only_within)?;
        Some(FrameAtBitrate {
            bitrate,
            padding,
            coded,
            main_data_begin,
            unshared_bits: slot_bits - shares * mean_bits,
            reservoir,
        })
    }
}
