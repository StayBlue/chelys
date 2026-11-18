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
use crate::stereo::middle_side::MsMask;
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
    /// Whether the frame being coded is a pair that shares its window.
    common_window: bool,
}

impl ElementEncoder {
    pub fn new(sampling_frequency: SamplingFrequency, channels: usize) -> Self {
        Self {
            sampling_frequency,
            channels: (0..channels)
                .map(|_| ChannelEncoder::new(sampling_frequency))
                .collect(),
            prepared: (0..channels).map(|_| PreparedChannel::new()).collect(),
            common_window: false,
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
    /// none if `block_switching` is off. The channels of a pair share
    /// their attacks, so that they can share their windows (see
    /// [`prepare`](Self::prepare)).
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
        switching::agree(attacks, SHARED_ATTACKS);
        for (channel, &attacks) in self.channels.iter_mut().zip(attacks.iter()) {
            channel.attacks = attacks;
        }
    }

    /// Prepares the frame whose block is at `start`: chooses each channel's
    /// windows, short ones only if the frame can `afford_short` them, and
    /// whether a pair shares them; transforms each channel's block with
    /// them in its filterbank, and from its spectrum and the model's
    /// thresholds finds what each scalefactor band may take as noise.
    ///
    /// Spec gap: Annex C does not say when the two channels of a pair share
    /// their `ics_info()`, `common_window` 1, which makes them share their
    /// windows and `max_sfb` and lets them be coded as middle and side
    /// (clause 12.1.1). A pair shares it in every frame whose windows agree.
    /// Its channels share their attacks (see [`SHARED_ATTACKS`]), so that
    /// their window sequences, chosen alike from the same state, are the
    /// same, and where they are short, their grouping is chosen from both
    /// (see [`switching::grouping`]).
    pub fn prepare(&mut self, start: usize, afford_short: bool) {
        let mut sequences = [WindowSequence::OnlyLong; 2];
        let sequences = &mut sequences[..self.channels.len()];
        for (channel, sequence) in self.channels.iter_mut().zip(sequences.iter_mut()) {
            *sequence = channel.switching.decide(channel.attacks, afford_short);
        }
        let all_short = (sequences.iter()).all(|s| matches!(s, WindowSequence::EightShort { .. }));
        let joint = all_short.then(|| {
            let analyses: Vec<_> = self
                .channels
                .iter()
                .map(|c| c.switching.current())
                .collect();
            switching::grouping(&analyses)
        });
        for ((channel, prepared), &sequence) in (self.channels.iter_mut())
            .zip(&mut self.prepared)
            .zip(sequences.iter())
        {
            let sequence = match sequence {
                WindowSequence::EightShort { .. } => WindowSequence::EightShort {
                    grouping: joint
                        .unwrap_or_else(|| switching::grouping(&[channel.switching.current()])),
                },
                sequence => sequence,
            };
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
        self.common_window = match &self.prepared[..] {
            [left, right] => left.windows == right.windows,
            _ => false,
        };
    }

    /// The windows the frame's pair shares, if it does.
    fn common_windows(&self) -> Option<Windows> {
        self.common_window.then(|| self.prepared[0].windows)
    }

    /// The bits of a channel of the frame without lines, as its element
    /// sends it.
    fn silent_bits(&self, channel: &PreparedChannel) -> u32 {
        channel::bits(channel.windows, self.common_window, 0)
    }

    /// What the frame prepared asks of the frame.
    pub fn demand(&self) -> Demand {
        let channels: u32 = self.prepared.iter().map(|c| self.silent_bits(c)).sum();
        let fields = CodedElement::fields_bits(self.channels(), self.common_windows());
        Demand {
            bit_allocation: self.prepared.iter().map(|c| c.bit_allocation).sum(),
            side_info_bits: fields + channels,
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
    /// leaves nothing, and is quantized first. Graded with PEAQ (GstPEAQ
    /// 0.6.1, black box), 8 tracks of pop music at 44.1 kHz in stereo at
    /// 128 kbit/s gave a mean of -2.19 so, and -2.35 with the bits left to
    /// the reservoir.
    ///
    /// A pair that shares its window sends one `max_sfb`: the channel
    /// quantized second spans at least the bands of the first, and counts
    /// the bits that extend the first to its own, and the `ms_used` flags,
    /// with its own (see [`ChannelCoding::element_bits`]). Where its bits
    /// cannot pay even for its bands without lines, the first is quantized
    /// again in fewer bits.
    ///
    /// # Panics
    ///
    /// If `available_bits` do not pay for the element's channels without
    /// lines: a caller's error.
    pub fn quantize(&self, available_bits: u32) -> CodedElement {
        let fs = self.sampling_frequency;
        let fields = CodedElement::fields_bits(self.channels(), self.common_windows());
        let reserved: u32 = (self.prepared.iter())
            .filter(|c| c.silent)
            .map(|c| self.silent_bits(c))
            .sum();
        let coded = self.prepared.iter().filter(|c| !c.silent).count() as u32;
        let share = (available_bits.saturating_sub(fields))
            .saturating_sub(reserved)
            .checked_div(coded)
            .unwrap_or(0);
        let budget = |c: &PreparedChannel| {
            if c.silent { self.silent_bits(c) } else { share }
        };
        let mut order: Vec<usize> = (0..self.prepared.len()).collect();
        order.sort_by(|&a, &b| {
            let key = |c: usize| {
                (
                    !self.prepared[c].silent,
                    self.prepared[c].perceptual_entropy,
                )
            };
            let (a, b) = (key(a), key(b));
            a.0.cmp(&b.0).then(a.1.total_cmp(&b.1))
        });
        match order[..] {
            [only] => {
                let prepared = &self.prepared[only];
                let coding = ChannelCoding::single(prepared.windows, fs, budget(prepared));
                CodedElement::single(self.code(prepared, coding))
            }
            [first, second] if self.common_window => {
                let (a, b) = (&self.prepared[first], &self.prepared[second]);
                let windows = a.windows;
                let groups = windows.groups(fs).count();
                let num_swb = windows.groups(fs).next().expect("a group").num_swb();
                let ms = MsMask::NONE;
                let mut coding = ChannelCoding {
                    common_window: true,
                    ..ChannelCoding::single(windows, fs, budget(a))
                };
                let (mut a_channel, b_coding) = loop {
                    let a_channel = self.code(a, coding);
                    let left = coding.available_bits - a_channel.bits(true);
                    let least_max_sfb = a_channel.max_sfb();
                    let mut b_coding = ChannelCoding {
                        least_max_sfb,
                        common_window: true,
                        ..ChannelCoding::single(windows, fs, budget(b) + left)
                    };
                    for m in least_max_sfb..=num_swb {
                        b_coding.element_bits[m] =
                            a_channel.extension_bits(m) + ms.flag_bits(groups, m);
                    }
                    let silent = CodedChannel::silent(windows, fs);
                    let b_least = silent.bits(true)
                        + silent.extension_bits(least_max_sfb)
                        + b_coding.element_bits[least_max_sfb];
                    let floor = self.silent_bits(a);
                    if b_least <= b_coding.available_bits || coding.available_bits <= floor {
                        break (a_channel, b_coding);
                    }
                    let short = b_least - b_coding.available_bits;
                    coding.available_bits = coding.available_bits.saturating_sub(short).max(floor);
                };
                let mut b_channel = self.code(b, b_coding);
                let max_sfb = a_channel.max_sfb().max(b_channel.max_sfb());
                a_channel.extend(max_sfb);
                b_channel.extend(max_sfb);
                let mut channels = [Some(a_channel), Some(b_channel)];
                if first == 1 {
                    channels.reverse();
                }
                let [left, right] = channels.map(|c| c.expect("coded"));
                CodedElement::common_pair(left, right, ms)
            }
            [first, second] => {
                let mut channels = [None, None];
                let mut left = 0;
                for c in [first, second] {
                    let prepared = &self.prepared[c];
                    let available = budget(prepared) + left;
                    let coding = ChannelCoding::single(prepared.windows, fs, available);
                    let channel = self.code(prepared, coding);
                    left = available - channel.bits(false);
                    channels[c] = Some(channel);
                }
                let [left, right] = channels.map(|c| c.expect("coded"));
                CodedElement::pair(left, right)
            }
            _ => unreachable!("an element of one channel or two"),
        }
    }

    /// Codes `prepared` as `coding` asks: without lines where it is
    /// silent, or through the iteration loops.
    fn code(&self, prepared: &PreparedChannel, coding: ChannelCoding) -> CodedChannel {
        if prepared.silent {
            CodedChannel::silent(prepared.windows, self.sampling_frequency)
        } else {
            loops::quantize_channel(&prepared.spectrum, &prepared.xmin, coding)
        }
    }
}

/// Whether the channels of a pair share their attacks, so that a frame
/// with an attack in either is coded with short windows in both, and the
/// pair can share its window (see [`ElementEncoder::prepare`]).
///
/// Spec gap: Annex C chooses each channel's block type from its own model
/// (clause C.1.4, step 13), and does not say whether the channels of a
/// pair are to agree. Graded with PEAQ (GstPEAQ 0.6.1, black box), on 8
/// tracks of pop music at 44.1 kHz in stereo, and on 3 tracks of rock
/// music and their 3 windows of 3 s with the most frames coded short, at
/// 64, 128 and 192 kbit/s: with shared attacks, so that the pair shares its
/// window in every frame, the means were -3.670, -1.858 and -0.470 for pop
/// and -3.636, -1.292 and -0.217 for rock; with each channel's own, and
/// the window shared where the windows happen to agree, -3.671, -1.870 and
/// -0.476, and -3.642, -1.327 and -0.230; each channel with its own window
/// in every frame, -3.673, -1.878 and -0.477, and -3.643, -1.336 and
/// -0.232.
const SHARED_ATTACKS: bool = true;

#[cfg(test)]
mod tests {
    use chelys_core::bitstream::BitWriter;

    use super::*;
    use crate::noiseless::decoder::{BitReader, Element, element};

    /// Codes `frames` frames of `input`, one slice of samples for each
    /// channel, at `bits` for each frame's element, and gives each frame's
    /// element as decoded with the spectra the encoder quantized.
    fn round_trip(
        fs: SamplingFrequency,
        input: &[Vec<f64>],
        bits: u32,
    ) -> Vec<(Element, Vec<Vec<f64>>)> {
        let mut encoder = ElementEncoder::new(fs, input.len());
        let channels: Vec<&[f64]> = input.iter().map(Vec::as_slice).collect();
        encoder.push(&channels);
        encoder.prime();
        let mut frames = Vec::new();
        let mut start = 0;
        while encoder.buffered() - start >= 2 * FRAME_SAMPLES {
            encoder.look_ahead(start + FRAME_SAMPLES, true);
            encoder.prepare(start, true);
            let coded = encoder.quantize(bits);
            let mut w = BitWriter::new();
            w.write(u32::from(coded.is_pair()), 3);
            coded.write(&mut w);
            assert!(coded.bits() <= bits, "{} bits in {bits}", coded.bits());
            let len = w.bit_len();
            w.write(0x7, 3);
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            let decoded = element(&mut r, fs);
            assert_eq!(r.position(), len, "the bits written are those read");
            let spectra = (encoder.prepared.iter())
                .map(|prepared| prepared.spectrum.to_vec())
                .collect();
            frames.push((decoded, spectra));
            start += FRAME_SAMPLES;
        }
        frames
    }

    /// The SNR in dB of `decoded` against `original`.
    fn snr(original: &[f64], decoded: &[f64]) -> f64 {
        let signal: f64 = original.iter().map(|x| x * x).sum();
        let noise: f64 = (original.iter().zip(decoded))
            .map(|(x, y)| (x - y).powi(2))
            .sum();
        10.0 * (signal / noise).log10()
    }

    fn noise(len: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                0.25 * ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
            })
            .collect()
    }

    /// Pairs of channels alike and unlike, silent and not, with attacks in
    /// one channel, share their window in every frame and decode to the
    /// spectra quantized.
    #[test]
    fn pairs_share_their_window_and_decode() {
        let fs = SamplingFrequency::Hz44100;
        let len = 12 * FRAME_SAMPLES;
        let tone: Vec<f64> = (0..len).map(|t| 0.3 * (t as f64 * 0.05).sin()).collect();
        let mut clicks = vec![0.0; len];
        clicks[5 * FRAME_SAMPLES + 300] = 0.9;
        let pairs = [
            vec![tone.clone(), tone.clone()],
            vec![tone.clone(), noise(len, 1)],
            vec![noise(len, 2), vec![0.0; len]],
            vec![vec![0.0; len], tone.clone()],
            vec![tone.clone(), clicks],
        ];
        for (i, pair) in pairs.iter().enumerate() {
            for (f, (decoded, spectra)) in round_trip(fs, pair, 6000).iter().enumerate() {
                assert!(decoded.common_window, "pair {i}, frame {f}");
                let [left, right] = &decoded.channels[..] else {
                    panic!("a pair")
                };
                assert_eq!(left.info, right.info);
                for (c, decoded) in decoded.spectra(fs).iter().enumerate() {
                    let original = &spectra[c];
                    if original.iter().any(|&x| x != 0.0) {
                        let snr = snr(original, decoded);
                        assert!(snr > 10.0, "pair {i}, frame {f}, channel {c}: {snr} dB");
                    } else {
                        assert!(decoded.iter().all(|&x| x == 0.0));
                    }
                }
            }
        }
    }

    /// Where one channel's bits cannot pay for the bands of the other, the
    /// other is quantized again in fewer.
    #[test]
    fn a_pair_fits_its_bits_however_few() {
        let fs = SamplingFrequency::Hz48000;
        let len = 6 * FRAME_SAMPLES;
        let pair = [
            noise(len, 3),
            noise(len, 4).iter().map(|x| x * 1e-3).collect(),
        ];
        // From the bits of a pair of silent channels of short windows each
        // with its own window.
        for bits in [57, 60, 100, 200, 400] {
            for (decoded, _) in round_trip(fs, &pair, bits) {
                let [left, right] = &decoded.channels[..] else {
                    panic!("a pair")
                };
                assert_eq!(left.info.max_sfb, right.info.max_sfb);
            }
        }
    }
}
