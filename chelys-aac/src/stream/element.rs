// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The frame's element: its channels as they arrive, the decisions made
//! for it each frame, and its coding in the bits the frame gives it.

use super::FRAME_SAMPLES;
use super::frame::Demand;
use crate::channel::{self, CodedChannel, FRAME_LINES};
use crate::filterbank::Filterbank;
use crate::loops::{self, ChannelCoding};
use crate::psychoacoustic::switching::{self, Attacks, BlockSwitching};
use crate::raw_data_block::element::CodedElement;
use crate::sampling_frequency::SamplingFrequency;
use crate::scalefactor_bands::MAX_BANDS;
use crate::window::{SHORT_WINDOWS, WindowSequence, Windows};

/// One channel of the element: its samples not yet coded, and what codes
/// them.
#[derive(Debug)]
struct ChannelEncoder {
    /// The samples that no frame has coded yet: the block of the frame
    /// coded next, and fewer than a block's after it, between calls.
    pcm: Vec<f64>,
    filterbank: Filterbank,
    /// The psychoacoustic model, a block ahead of the filterbank.
    switching: BlockSwitching,
    /// The attacks in the frame being coded and the one after.
    attacks: Attacks,
}

impl ChannelEncoder {
    fn new(sampling_frequency: SamplingFrequency) -> Self {
        Self {
            pcm: Vec::with_capacity(2 * FRAME_SAMPLES),
            filterbank: Filterbank::new(sampling_frequency),
            switching: BlockSwitching::new(sampling_frequency),
            attacks: Attacks {
                this: false,
                next: false,
            },
        }
    }

    /// The block of samples at `start` in [`Self::pcm`].
    fn block(&self, start: usize) -> &[f64; FRAME_SAMPLES] {
        self.pcm[start..start + FRAME_SAMPLES]
            .try_into()
            .expect("a block of samples")
    }
}

/// A channel of a frame as it is to be quantized: its lines and what each
/// band may take as noise, and what it asks of the frame.
#[derive(Debug)]
pub struct PreparedChannel {
    pub windows: Windows,
    /// The lines of the frame's transform, in the order they are sent.
    pub spectrum: Box<[f64; FRAME_LINES]>,
    /// The allowed distortion `xmin` of each band of each window group.
    pub xmin: Vec<[f64; MAX_BANDS]>,
    /// `bit_allocation` (clause C.1.4, step 15).
    pub bit_allocation: f64,
    /// The perceptual entropy (clause C.1.4, steps 12 and 15).
    pub perceptual_entropy: f64,
    /// Whether every line is zero, so that the channel is coded without
    /// any.
    pub silent: bool,
}

impl PreparedChannel {
    fn new() -> Self {
        Self {
            windows: Windows::LONG_SINE,
            spectrum: Box::new([0.0; FRAME_LINES]),
            xmin: Vec::with_capacity(SHORT_WINDOWS),
            bit_allocation: 0.0,
            perceptual_entropy: 0.0,
            silent: true,
        }
    }

    /// The bits of the channel without lines, as its element sends it.
    fn silent_bits(&self) -> u32 {
        channel::bits(self.windows, false, 0)
    }
}

/// The element of a stream: a `single_channel_element()` of one channel,
/// or a `channel_pair_element()` of two.
#[derive(Debug)]
pub struct ElementEncoder {
    sampling_frequency: SamplingFrequency,
    channels: Vec<ChannelEncoder>,
    /// Each channel of the frame being coded, kept from one frame to the
    /// next for its memory.
    prepared: Vec<PreparedChannel>,
}

impl ElementEncoder {
    pub fn new(sampling_frequency: SamplingFrequency, channels: usize) -> Self {
        Self {
            sampling_frequency,
            channels: (0..channels)
                .map(|_| ChannelEncoder::new(sampling_frequency))
                .collect(),
            prepared: (0..channels).map(|_| PreparedChannel::new()).collect(),
        }
    }

    pub fn channels(&self) -> usize {
        self.channels.len()
    }

    /// Appends the next samples of each channel.
    pub fn push(&mut self, samples: &[&[f64]]) {
        for (channel, samples) in self.channels.iter_mut().zip(samples) {
            channel.pcm.extend_from_slice(samples);
        }
    }

    /// The samples of each channel that no frame has coded yet.
    pub fn buffered(&self) -> usize {
        self.channels[0].pcm.len()
    }

    /// Pads each channel with silence to `len` samples.
    pub fn pad(&mut self, len: usize) {
        for channel in &mut self.channels {
            channel.pcm.resize(len, 0.0);
        }
    }

    /// Drops the first `len` samples of each channel.
    pub fn drain(&mut self, len: usize) {
        for channel in &mut self.channels {
            channel.pcm.drain(..len);
        }
    }

    /// Whether the models have analyzed the block of the frame coded next.
    pub fn is_primed(&self) -> bool {
        self.channels[0].switching.is_primed()
    }

    /// Has the models analyze the first block, that of the first frame.
    pub fn prime(&mut self) {
        for channel in &mut self.channels {
            let block = *channel.block(0);
            let attacks = channel.switching.look_ahead(&block);
            debug_assert!(attacks.is_none(), "the first block");
        }
    }

    /// Has the models analyze the block at `next`, after that of the frame
    /// coded next, and keeps what they find of the attacks of the frame,
    /// none if `block_switching` is off.
    pub fn look_ahead(&mut self, next: usize, block_switching: bool) {
        let mut attacks = [Attacks {
            this: false,
            next: false,
        }; 2];
        let attacks = &mut attacks[..self.channels.len()];
        for (channel, attacks) in self.channels.iter_mut().zip(attacks.iter_mut()) {
            let block = *channel.block(next);
            let found = channel.switching.look_ahead(&block).expect("primed");
            if block_switching {
                *attacks = found;
            }
        }
        // The channels of a pair do not yet share their windows.
        switching::agree(attacks, false);
        for (channel, &attacks) in self.channels.iter_mut().zip(attacks.iter()) {
            channel.attacks = attacks;
        }
    }

    /// Prepares the frame whose block is at `start`: chooses each channel's
    /// windows, short ones only if the frame can `afford_short` them,
    /// transforms its block with them in its filterbank, and from its
    /// spectrum and the model's thresholds finds what each scalefactor
    /// band may take as noise.
    pub fn prepare(&mut self, start: usize, afford_short: bool) {
        for (channel, prepared) in self.channels.iter_mut().zip(&mut self.prepared) {
            let mut sequence = channel.switching.decide(channel.attacks, afford_short);
            if let WindowSequence::EightShort { .. } = sequence {
                let grouping = switching::grouping(&[channel.switching.current()]);
                sequence = WindowSequence::EightShort { grouping };
            }
            let windows = Windows {
                sequence,
                shape: switching::shape(sequence),
            };
            let block = channel.pcm[start..start + FRAME_SAMPLES]
                .try_into()
                .expect("a block of samples");
            prepared.windows = windows;
            (channel.filterbank).spectrum(block, windows, &mut prepared.spectrum);
            prepared.silent = prepared.spectrum.iter().all(|&x| x == 0.0);
            let analysis = channel.switching.current();
            analysis.xmin(windows, &prepared.spectrum, &mut prepared.xmin);
            prepared.bit_allocation = analysis.bit_allocation(windows.sequence);
            prepared.perceptual_entropy = analysis.perceptual_entropy(windows.sequence);
        }
    }

    /// What the frame prepared asks of the frame.
    pub fn demand(&self) -> Demand {
        let channels: u32 = self.prepared.iter().map(PreparedChannel::silent_bits).sum();
        Demand {
            bit_allocation: self.prepared.iter().map(|c| c.bit_allocation).sum(),
            side_info_bits: CodedElement::fields_bits(self.channels()) + channels,
        }
    }

    /// Quantizes the frame prepared in `available_bits`, the bits of the
    /// element.
    ///
    /// Spec gap: Annex C does not say how the bits of a frame are shared
    /// between its channels, whose loops each fit one channel's lines to the
    /// bits they are given (clause C.7.1). A silent channel is given what it
    /// costs, the bits of a channel without lines, and the others equal
    /// shares of the rest, so that a channel codes alike whichever of a
    /// pair it is. Shares in proportion to each channel's `bit_allocation`
    /// were tried, and starved channels: the allocation is 0 in most frames
    /// of music, where the perceptual entropy is negative (see
    /// [`Analysis::bit_allocation`](crate::psychoacoustic::Analysis::bit_allocation)),
    /// and a channel whose allocation was 0 beside one whose was not was
    /// left the bits of a channel without lines.
    ///
    /// Each channel is quantized in its share and the bits that those
    /// quantized before it left: a channel's loops take the finest step
    /// that fits its bits, one of 1.5 dB, which leaves about 90 bits unused
    /// on average at 128 kbit/s in stereo (figure C.6). Spec gap: Annex C
    /// does not say what becomes of them within a frame. The channels are
    /// quantized in order of their perceptual entropy, the least first, so
    /// that what one leaves goes to one that asks for more, whatever their
    /// order in the frame; a channel of zeros, given just what it costs,
    /// leaves nothing. Graded with PEAQ (GstPEAQ 0.6.1, black box), 8
    /// tracks of pop music at 44.1 kHz in stereo at 128 kbit/s gave a mean
    /// of -2.19 so, and -2.35 with the bits left to the reservoir.
    ///
    /// # Panics
    ///
    /// If `available_bits` do not pay for the element's channels without
    /// lines: a caller's error.
    pub fn quantize(&self, available_bits: u32) -> CodedElement {
        let fs = self.sampling_frequency;
        let fields = CodedElement::fields_bits(self.channels());
        let reserved: u32 = (self.prepared.iter())
            .filter(|c| c.silent)
            .map(PreparedChannel::silent_bits)
            .sum();
        let coded = self.prepared.iter().filter(|c| !c.silent).count() as u32;
        let share = (available_bits.saturating_sub(fields))
            .saturating_sub(reserved)
            .checked_div(coded)
            .unwrap_or(0);
        let mut order: Vec<usize> = (0..self.prepared.len()).collect();
        order.sort_by(|&a, &b| {
            let pe = |c: usize| self.prepared[c].perceptual_entropy;
            pe(a).total_cmp(&pe(b))
        });
        let mut channels: Vec<Option<CodedChannel>> = vec![None; order.len()];
        let mut left = 0;
        for c in order {
            let prepared = &self.prepared[c];
            let channel = if prepared.silent {
                CodedChannel::silent(prepared.windows, fs)
            } else {
                let available = share + left;
                let coding = ChannelCoding::single(prepared.windows, fs, available);
                let channel = loops::quantize_channel(&prepared.spectrum, &prepared.xmin, coding);
                left = available - channel.bits(false);
                channel
            };
            channels[c] = Some(channel);
        }
        let mut channels = channels.into_iter().map(|c| c.expect("coded"));
        let first = channels.next().expect("a channel");
        match channels.next() {
            None => CodedElement::single(first),
            Some(second) => CodedElement::pair(first, second),
        }
    }
}
