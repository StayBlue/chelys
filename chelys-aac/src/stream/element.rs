// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! An element of the frame: its channels as they arrive, the decisions
//! made for it each frame, and its coding in the bits the frame gives it.

use super::FRAME_SAMPLES;
use super::frame::Demand;
use crate::channel::{self, CodedChannel, CodedGroup, FRAME_LINES};
use crate::channel_configuration::ElementKind;
use crate::filterbank::Filterbank;
use crate::loops::{self, ChannelCoding};
use crate::noise_substitution::{self, BandSet, Noise};
use crate::noiseless::pulses::Pulses;
use crate::noiseless::scalefactors;
use crate::noiseless::sections::Sections;
use crate::psychoacoustic::Analysis;
use crate::psychoacoustic::imaging::MidSideModel;
use crate::psychoacoustic::switching::{self, Attacks, BlockSwitching};
use crate::raw_data_block::element::{CodedElement, LFE_LINES};
use crate::sampling_frequency::SamplingFrequency;
use crate::scalefactor_bands::{MAX_BANDS, Transform, first_band_above};
use crate::stereo::intensity::{self, Intensity};
use crate::stereo::middle_side::{self, MsMask};
use crate::stereo::{self, BandCoding};
use crate::tns::analysis::{Shaping, target_bands};
use crate::tns::{NoiseGains, Tns};
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
    /// A channel of an element of `kind`: a low frequency enhancement
    /// channel's model finds no attacks (see [`ElementEncoder`]).
    fn new(sampling_frequency: SamplingFrequency, kind: ElementKind) -> Self {
        let switching = match kind {
            ElementKind::Lfe => BlockSwitching::long_only(sampling_frequency),
            _ => BlockSwitching::new(sampling_frequency),
        };
        Self {
            pcm: Vec::with_capacity(2 * FRAME_SAMPLES),
            filterbank: Filterbank::new(sampling_frequency),
            switching,
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
    /// The lines of the frame's transform, in the order they are sent,
    /// shaped by the filters of `tns`.
    pub spectrum: Box<[f64; FRAME_LINES]>,
    /// The filters of the channel's temporal noise shaping.
    pub tns: Tns,
    /// The lines of the frame's transform before temporal noise shaping.
    pub unshaped: Box<[f64; FRAME_LINES]>,
    /// The weight of each line's quantization error in its band's
    /// distortion (see
    /// [`Lines::weighted`](crate::loops::inner::Lines::weighted)).
    pub weights: Box<[f64; FRAME_LINES]>,
    /// How far the decoder's filters raise the noise of each window's
    /// bands, and the bands they span.
    pub gains: NoiseGains,
    /// The allowed distortion `xmin` of each band of each window group,
    /// raised by [`XMIN_OFFSET_DB`](loops::XMIN_OFFSET_DB) as the loops take
    /// it.
    pub xmin: Vec<[f64; MAX_BANDS]>,
    /// `bit_allocation` (clause C.1.4, step 15).
    pub bit_allocation: f64,
    /// The perceptual entropy (clause C.1.4, steps 12 and 15).
    pub perceptual_entropy: f64,
    /// The bands coded by perceptual noise substitution, whose lines in
    /// `spectrum` are zero.
    pub noise: Noise,
    /// Whether every line is zero and no band is coded by perceptual noise
    /// substitution, so that the channel is coded without any.
    pub silent: bool,
}

impl PreparedChannel {
    fn new() -> Self {
        Self {
            windows: Windows::LONG_SINE,
            spectrum: Box::new([0.0; FRAME_LINES]),
            tns: Tns::NONE,
            unshaped: Box::new([0.0; FRAME_LINES]),
            weights: Box::new([1.0; FRAME_LINES]),
            gains: NoiseGains::NONE,
            xmin: Vec::with_capacity(SHORT_WINDOWS),
            bit_allocation: 0.0,
            perceptual_entropy: 0.0,
            noise: Noise::NONE,
            silent: true,
        }
    }

    /// The channel's allowed distortions, perceptual entropy and
    /// `bit_allocation`, from its lines before temporal noise shaping and
    /// `analysis`, its model's thresholds.
    fn thresholds(&mut self, analysis: &Analysis) {
        (self.bit_allocation, self.perceptual_entropy) =
            thresholds(analysis, self.windows, &self.unshaped, &mut self.xmin);
    }
}

/// A pair's middle or side channel in the frame being coded, of the
/// pair's windows: its lines and what each band may take as noise, which
/// take the place of the left or right channel's in the bands coded as
/// middle and side (see [`ElementEncoder::middle_side_bands`]).
#[derive(Debug)]
struct MidSideChannel {
    /// The lines, shaped by the left and right channels' filters.
    spectrum: Box<[f64; FRAME_LINES]>,
    /// The lines before temporal noise shaping.
    unshaped: Box<[f64; FRAME_LINES]>,
    /// The weight of each line's quantization error in its band's
    /// distortion, as for a [`PreparedChannel`].
    weights: Box<[f64; FRAME_LINES]>,
    /// The allowed distortion `xmin` of each band of each window group, as
    /// for a [`PreparedChannel`].
    xmin: Vec<[f64; MAX_BANDS]>,
    /// `bit_allocation` (clause C.1.4, step 15).
    bit_allocation: f64,
    /// The perceptual entropy (clause C.1.4, steps 12 and 15).
    perceptual_entropy: f64,
}

impl MidSideChannel {
    fn new() -> Self {
        Self {
            spectrum: Box::new([0.0; FRAME_LINES]),
            unshaped: Box::new([0.0; FRAME_LINES]),
            weights: Box::new([1.0; FRAME_LINES]),
            xmin: Vec::with_capacity(SHORT_WINDOWS),
            bit_allocation: 0.0,
            perceptual_entropy: 0.0,
        }
    }

    /// The channel's allowed distortions, perceptual entropy and
    /// `bit_allocation` in `windows`, from its lines before temporal noise
    /// shaping and `analysis`, its model's thresholds.
    fn thresholds(&mut self, windows: Windows, analysis: &Analysis) {
        (self.bit_allocation, self.perceptual_entropy) =
            thresholds(analysis, windows, &self.unshaped, &mut self.xmin);
    }
}

/// From a channel's lines `unshaped` of `windows`, before temporal noise
/// shaping, and `analysis`, its model's thresholds: the allowed distortion
/// of each band of each window group, raised as the loops take it, in
/// `xmin`, and the channel's `bit_allocation` and perceptual entropy.
fn thresholds(
    analysis: &Analysis,
    windows: Windows,
    unshaped: &[f64; FRAME_LINES],
    xmin: &mut Vec<[f64; MAX_BANDS]>,
) -> (f64, f64) {
    analysis.xmin(windows, unshaped, xmin);
    loops::raise_xmin(xmin);
    (
        analysis.bit_allocation(windows.sequence),
        analysis.perceptual_entropy(windows.sequence),
    )
}

/// An element of a stream: a `single_channel_element()` or
/// `lfe_channel_element()` of one channel, or a `channel_pair_element()` of
/// two.
///
/// Spec gap: Annex C does not mention the low frequency enhancement
/// channel. It is coded as a single channel within the restrictions of
/// clause 8.4.1 (see [`CodedElement::lfe`]): its model analyzes the
/// channel's samples as they are given, as those of a channel that holds
/// low frequencies only, for the thresholds of its bands; it finds no
/// attacks, so that the channel is coded with long windows, of the sine
/// shape; it is not shaped by temporal noise shaping; and its lines from
/// [`LFE_LINES`] up are set to zero before it is quantized, so that they
/// cost no bits and the loops measure no distortion there. Its
/// `bit_allocation` is not counted in the frame's demand (see
/// [`demand`](Self::demand)).
#[derive(Debug)]
pub struct ElementEncoder {
    sampling_frequency: SamplingFrequency,
    kind: ElementKind,
    channels: Vec<ChannelEncoder>,
    /// Where the input ends among the samples buffered, once the stream is
    /// padded past it.
    input_end: Option<usize>,
    /// Each channel of the frame being coded, kept from one frame to the
    /// next for its memory.
    prepared: Vec<PreparedChannel>,
    /// Whether the frame being coded is a pair that shares its window.
    common_window: bool,
    /// For a pair, the model of its middle and side channels, and their
    /// lines and allowances in the frame being coded.
    mid_side: Option<Box<MidSide>>,
    /// The bands of the frame being coded that a pair that shares its
    /// window codes as middle and side.
    ms: MsMask,
    /// The lower border frequency above which a pair that shares its window
    /// is coded in intensity stereo, if it is (see [`intensity::code`]).
    intensity_border: Option<f64>,
    /// The bands of the frame being coded that a pair that shares its
    /// window codes in intensity stereo.
    intensity: Intensity,
    /// The bands of noise of the frame being coded that a pair that shares
    /// its window codes in intensity stereo in its right channel, from the
    /// left channel's noise.
    noise_intensity: Intensity,
    /// Whether such bands are sent as noise in both channels with
    /// `ms_used` set instead, and those of the frame being coded, for tests.
    noise_by_ms_used: bool,
    noise_ms_used: BandSet,
    /// The lower border frequency above which bands are coded by perceptual
    /// noise substitution, if they are (see [`noise_substitution`]).
    noise_border: Option<f64>,
    /// Whether the channels' lines are shaped by temporal noise shaping,
    /// and the memory it works in.
    tns: bool,
    /// Whether the loops clip the lines of long windows where that saves
    /// bits (see [`ChannelCoding::spectrum_clipping`]).
    spectrum_clipping: bool,
    shaping: Shaping,
    /// The bits of the frame being coded beside its lines.
    lineless: LinelessBits,
    /// How the second channel of the last pair that shared its window was
    /// given its bits, for tests.
    #[cfg(test)]
    second_channel: Option<SecondChannel>,
}

/// How the second channel of a pair that shares its window was given its
/// bits: whether the first was quantized again in fewer, the bits the
/// first took, those the second was given, and the two channels' shares.
#[cfg(test)]
#[derive(Clone, Copy, Debug)]
struct SecondChannel {
    retried: bool,
    first_bits: u32,
    available_bits: u32,
    budgets: u32,
}

/// The bits of the frame prepared beside its lines: the element's fields,
/// and each channel coded without lines, with and without its filters of
/// temporal noise shaping. Counted once the frame is prepared, and kept as
/// [`ElementEncoder::unshape`] drops filters.
#[derive(Clone, Copy, Debug, Default)]
struct LinelessBits {
    /// The element's fields beside its channels (see
    /// [`CodedElement::fields_bits`]).
    fields: u32,
    /// Each channel without lines, with its filters as they stand.
    channels_with_tns: [u32; 2],
    /// Each channel without lines or filters.
    channels_without_tns: [u32; 2],
}

impl LinelessBits {
    /// The element's fields and its channels without lines, with their
    /// filters as they stand.
    fn element_with_tns(&self) -> u32 {
        self.fields + self.channels_with_tns.iter().sum::<u32>()
    }

    /// The element's fields and its channels without lines or filters.
    fn element_without_tns(&self) -> u32 {
        self.fields + self.channels_without_tns.iter().sum::<u32>()
    }
}

/// What a pair keeps of its middle and side channels.
#[derive(Debug)]
struct MidSide {
    model: MidSideModel,
    channels: [MidSideChannel; 2],
}

impl ElementEncoder {
    /// The element `kind` at `sampling_frequency`, a pair coded in
    /// intensity stereo above `intensity_border` Hz, if given, and its
    /// channels by perceptual noise substitution above `noise_border` Hz,
    /// if given.
    pub fn new(
        sampling_frequency: SamplingFrequency,
        kind: ElementKind,
        intensity_border: Option<f64>,
        noise_border: Option<f64>,
    ) -> Self {
        let channels = kind.channels();
        Self {
            sampling_frequency,
            kind,
            input_end: None,
            channels: (0..channels)
                .map(|_| ChannelEncoder::new(sampling_frequency, kind))
                .collect(),
            prepared: (0..channels).map(|_| PreparedChannel::new()).collect(),
            common_window: false,
            mid_side: (channels == 2).then(|| {
                Box::new(MidSide {
                    model: MidSideModel::new(sampling_frequency),
                    channels: [MidSideChannel::new(), MidSideChannel::new()],
                })
            }),
            ms: MsMask::NONE,
            intensity_border,
            intensity: Intensity::NONE,
            noise_intensity: Intensity::NONE,
            noise_by_ms_used: false,
            noise_ms_used: [0; SHORT_WINDOWS],
            noise_border,
            tns: true,
            spectrum_clipping: loops::SPECTRUM_CLIPPING,
            shaping: Shaping::default(),
            lineless: LinelessBits::default(),
            #[cfg(test)]
            second_channel: None,
        }
    }

    pub fn channels(&self) -> usize {
        self.channels.len()
    }

    pub fn kind(&self) -> ElementKind {
        self.kind
    }

    /// The element, but without temporal noise shaping, for comparing with
    /// it in tests.
    #[cfg(any(test, feature = "unstable"))]
    pub fn without_tns(&mut self) {
        self.tns = false;
    }

    /// The element, but with the noise of a pair's bands that is to be the
    /// same in both channels sent as noise in both, with `ms_used` set, for
    /// comparing with intensity stereo in tests (see
    /// [`noise_substitution`](Self::noise_substitution)).
    #[cfg(feature = "unstable")]
    pub fn noise_shared_by_ms_used(&mut self) {
        self.noise_by_ms_used = true;
    }

    /// The element, but with spectrum clipping on or off, for comparing the
    /// two in tests.
    #[cfg(any(test, feature = "unstable"))]
    pub fn spectrum_clipping(&mut self, on: bool) {
        self.spectrum_clipping = on;
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
        self.input_end = Some(self.input_end.unwrap_or(self.buffered()));
        for channel in &mut self.channels {
            channel.pcm.resize(len, 0.0);
        }
    }

    /// Drops the first `len` samples of each channel.
    pub fn drain(&mut self, len: usize) {
        self.input_end = self.input_end.map(|end| end.saturating_sub(len));
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
        let attacks = self.analyze(0);
        debug_assert!(attacks.is_none(), "the first block");
    }

    /// Has the models analyze the block at `next`, after that of the frame
    /// coded next, and keeps what they find of the attacks of the frame,
    /// none if `block_switching` is off or the element is a low frequency
    /// enhancement channel. The channels of a pair share their attacks, so
    /// that they can share their windows (see [`prepare`](Self::prepare)).
    pub fn look_ahead(&mut self, next: usize, block_switching: bool) {
        let mut attacks = self.analyze(next).expect("primed");
        let attacks = &mut attacks[..self.channels.len()];
        if !block_switching || self.kind == ElementKind::Lfe {
            attacks.fill(Attacks {
                this: false,
                next: false,
            });
        }
        switching::agree(attacks, SHARED_ATTACKS);
        for (channel, &attacks) in self.channels.iter_mut().zip(attacks.iter()) {
            channel.attacks = attacks;
        }
    }

    /// Has each channel's model, and a pair's model of its middle and side
    /// channels, analyze the block at `start`, and gives each channel's
    /// attacks in the frame before and in this one, but for the first
    /// block.
    fn analyze(&mut self, start: usize) -> Option<[Attacks; 2]> {
        match (&mut self.channels[..], &mut self.mid_side) {
            ([left, right], Some(mid_side)) => {
                let blocks = [*left.block(start), *right.block(start)];
                BlockSwitching::look_ahead_pair(
                    [&mut left.switching, &mut right.switching],
                    &mut mid_side.model,
                    [&blocks[0], &blocks[1]],
                )
            }
            ([channel], _) => {
                let block = *channel.block(start);
                let attacks = channel.switching.look_ahead(&block)?;
                Some([attacks; 2])
            }
            _ => unreachable!("an element of one channel or two"),
        }
    }

    /// Prepares the frame whose block is at `start` in stages: its
    /// [spectra](Self::spectra), each channel's windows chosen, short ones
    /// only if the frame can `afford_short` them, and its block transformed
    /// with them; their [temporal noise
    /// shaping](Self::temporal_noise_shaping); their
    /// [thresholds](Self::thresholds), what each scalefactor band may take
    /// as noise; a pair's [joint](Self::joint) coding; and which channels
    /// are [silent](Self::find_silence). The stages follow the decoder's
    /// tools in reverse (clause 4, figure 2), as the encoder of figure 1
    /// orders them: the decoder restores the temporal noise shaping of each
    /// channel after it decodes the pair's joint coding, so the encoder
    /// shapes each channel's lines before it codes them jointly.
    ///
    /// A pair shares its `ics_info()`, `common_window` 1, where "each
    /// individual channel in the pair window switch together" (clause
    /// 8.3.2.3), which makes them share their windows and `max_sfb` and lets
    /// them be coded as middle and side (clause 12.1.1): in every frame
    /// whose windows agree. Spec gap: Annex C does not say whether a pair's
    /// channels are to switch together. Its channels share their attacks
    /// (see [`SHARED_ATTACKS`]), so that their window sequences, chosen
    /// alike from the same state, are the same, and where they are short,
    /// their grouping is chosen from both (see [`switching::grouping`]).
    pub fn prepare(&mut self, start: usize, afford_short: bool) {
        self.spectra(start, afford_short);
        let padded = self
            .input_end
            .is_some_and(|end| end < start + FRAME_SAMPLES);
        self.prepare_spectra(!padded);
    }

    /// The stages of [`prepare`](Self::prepare) after the
    /// [spectra](Self::spectra), each channel's lines shaped by temporal
    /// noise shaping if `shape` is set.
    fn prepare_spectra(&mut self, shape: bool) {
        self.temporal_noise_shaping(shape);
        self.thresholds();
        self.noise_substitution();
        self.joint();
        self.find_silence();
        self.count_lineless();
    }

    /// Each channel's windows and the lines of its block transformed with
    /// them, and whether a pair shares its window.
    fn spectra(&mut self, start: usize, afford_short: bool) {
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
            let windows = match self.kind {
                ElementKind::Lfe => {
                    debug_assert_eq!(sequence, WindowSequence::OnlyLong);
                    Windows::LONG_SINE
                }
                _ => Windows {
                    sequence,
                    shape: switching::shape(sequence),
                },
            };
            let block = channel.pcm[start..start + FRAME_SAMPLES]
                .try_into()
                .expect("a block of samples");
            prepared.windows = windows;
            (channel.filterbank).spectrum(block, windows, &mut prepared.spectrum);
            if self.kind == ElementKind::Lfe {
                prepared.spectrum[LFE_LINES..].fill(0.0);
            }
        }
        self.common_window = match &self.prepared[..] {
            [left, right] => left.windows == right.windows,
            _ => false,
        };
    }

    /// The first band of the frame's pair coded in intensity stereo, where
    /// it is: where the pair shares a window of long blocks and its stream
    /// codes it so, the first above the border frequency (see
    /// [`intensity_bands`](Self::intensity_bands)).
    ///
    /// Spec reading: this deliberately departs from clause C.6.2, which
    /// codes every window group in intensity stereo, for frames of eight
    /// short windows, which are not: they are chosen for attacks, whose
    /// timing intensity stereo would give up in the right channel, moving
    /// the right channel's attack into the left channel's intensity signal
    /// and the left's into the right. Coded so, a one-sample impulse in one
    /// channel of a pair at 64 kHz and 96 kbit/s decoded at about half its
    /// level. Music does not settle it either way: graded with PEAQ
    /// (GstPEAQ 0.6.1, black box) at 48 and 64 kbit/s, short windows coded
    /// in intensity stereo too gave means of -3.653 and -3.537 on 8 tracks
    /// of pop music at 44.1 kHz in stereo, against -3.657 and -3.543, and
    /// -3.589 and -3.561 on 3 tracks of rock music and their 3 windows of 3
    /// s with the most frames coded short, against -3.591 and -3.563. In
    /// the short windows of the 3 rock tracks at 64 kbit/s, the difference
    /// of the channels' levels above 6 kHz, in blocks of 128 samples, was
    /// 0.72, 1.14 and 1.93 dB off on average, weighed by their energy,
    /// against 0.92, 1.41 and 1.87 dB.
    fn intensity_from(&self) -> Option<usize> {
        let windows = self.prepared[0].windows;
        let long = windows.sequence.transform() == Transform::Long;
        let border = self
            .intensity_border
            .filter(|_| self.common_window && long)?;
        Some(first_band_above(
            self.sampling_frequency,
            Transform::Long,
            border,
        ))
    }

    /// Each channel's lines shaped in time by the filters of temporal noise
    /// shaping (see [`Shaping::shape`]), and the filters that shaped them,
    /// which [`quantize`](Self::quantize) may yet drop.
    ///
    /// Spec reading: this deliberately departs from clause C.5, which
    /// shapes every window whose prediction gain exceeds the threshold,
    /// for windows none of whose bands in the target range exceeds its
    /// allowed distortion (see [`Analysis::audible`]): their lines quantize
    /// to zero within their allowance, which leaves no noise to shape, and
    /// their filters would take bits for nothing. The lines above 1.5 kHz
    /// of a steady tone below it, the leakage of its window, are well
    /// predicted and inaudible: a tone of 1 kHz beside silence, at 6.5
    /// kbit/s in stereo, so shaped in most frames, decoded at an SNR of 0
    /// dB, against more than 10 dB unshaped (the conformance test
    /// `a_tone_beside_silence_decodes_alike_in_either_channel`).
    ///
    /// Spec reading: this deliberately departs from clause C.5 in frames
    /// whose block holds samples past the end of the input, the silence
    /// that the stream is padded with (see `StreamEncoder::finish`), which
    /// are not shaped. The input's end, a fall to silence in mid-block,
    /// raises the prediction gain of their lines, and their filters put
    /// noise after it, into the output past the input: in pairs of tones at
    /// 48 kHz and a bit a sample, the output past the input took 7.8 times
    /// the noise power of the input's last block, and 2.1 times unshaped;
    /// a sweep at 44.1 kHz in stereo at 96 kbit/s decoded at an SNR of 46.3
    /// dB, 50.8 dB with its last frames unshaped (the conformance tests
    /// `stereo_pairs_decode_each_channel_in_place` and
    /// `snr_rises_with_the_bitrate`).
    ///
    /// Spec gap: the decoder applies the right channel's filters to its
    /// bands in intensity stereo, after it has taken them from the left
    /// channel's lines as they were sent (clause 4, figure 2), which the
    /// left channel's filters have shaped; Annex C does not say how
    /// temporal noise shaping and intensity stereo combine. Where the pair
    /// is coded in intensity stereo, the filters of both channels end below
    /// its first band in intensity stereo: the left channel sends the
    /// bands above unfiltered, and the right channel takes them so.
    fn temporal_noise_shaping(&mut self, shape: bool) {
        let fs = self.sampling_frequency;
        let top = self.intensity_from();
        for (channel, prepared) in self.channels.iter().zip(&mut self.prepared) {
            prepared.unshaped.copy_from_slice(&*prepared.spectrum);
            prepared.tns = Tns::NONE;
            let mut gains = NoiseGains::NONE;
            if self.tns && shape && self.kind != ElementKind::Lfe {
                let windows = prepared.windows;
                let bands = target_bands(fs, windows.sequence.transform(), top);
                let analysis = channel.switching.current();
                let active = analysis.audible(windows, &prepared.spectrum, bands);
                let spectrum = &mut prepared.spectrum;
                prepared.tns = (self.shaping).shape(windows, fs, top, spectrum, active, &mut gains);
            }
            prepared.gains = gains;
            gains.weights(prepared.windows, fs, &mut prepared.weights);
        }
    }

    /// From each channel's lines and its model's thresholds, what each of
    /// its scalefactor bands may take as noise, and its perceptual entropy
    /// and `bit_allocation`.
    ///
    /// Spec gap: Annex C computes `xmin(n)` from the energy of each band's
    /// MDCT lines (clause C.1.4, step 14), and measures the distortion of
    /// the lines it quantizes (clause C.7.4.3), without saying how either
    /// combines with temporal noise shaping, which comes between the
    /// filterbank and the quantization (clause 4, figure 1). The allowance
    /// is that of the lines as the filterbank gave them, which the decoder
    /// restores; the loops measure the distortion of the lines as shaped,
    /// each line's error weighed by how far the decoder's filter raises it
    /// (see [`NoiseGains`]). With the allowance of the shaped lines' own
    /// energy and their errors unweighed, a sweep from 100 Hz to 6 kHz at
    /// 12 kHz in mono at 12 kbit/s decoded at an SNR of 3.3 dB, against
    /// 13.0 dB so weighed and 18.3 dB unshaped; graded with PEAQ (GstPEAQ
    /// 0.6.1, black box), 8 tracks of pop music at 128 kbit/s gave means of
    /// -1.197 and -1.181. Measuring the distortion of the lines as the
    /// decoder restores them, through its filters, rather than weighing
    /// them, graded -1.200 and took half as long again to encode.
    fn thresholds(&mut self) {
        for (channel, prepared) in self.channels.iter().zip(&mut self.prepared) {
            prepared.thresholds(channel.switching.current());
        }
    }

    /// Each channel's bands coded by perceptual noise substitution, where
    /// the stream codes them so (see [`noise_substitution`]): those that
    /// are noise-like, their lines set to zero; and of a pair that shares
    /// its window, the bands whose noise is to be the same in both channels,
    /// which the right channel takes from the left in intensity stereo.
    ///
    /// Spec gap: Annex 4.B.11 of ISO/IEC 14496-3 describes the choice for
    /// one channel. A pair that shares its window codes by perceptual noise
    /// substitution only the bands that are noise-like in both of its
    /// channels, which are then not coded as middle and side, nor in
    /// intensity stereo as [`intensity_bands`](Self::intensity_bands) codes
    /// a pair: in a band of noise in one channel only, `ms_used` is "not
    /// evaluated" (clause 4.6.13.3), so that the other channel's band could
    /// not be coded as middle and side either.
    ///
    /// Spec reading: where a pair's noise is to be the same in both
    /// channels, clause 4.6.13.3 has both channels send the band as noise
    /// and sets its `ms_used`. Such a band is instead sent as noise in the
    /// left channel alone, and in intensity stereo in the right, which the
    /// decoder takes from the left channel's noise, scaled by the band's
    /// position, as it applies perceptual noise substitution before
    /// intensity stereo (clause 4.1.1.2 and Figure 4.2): the same noise in
    /// both channels by either coding. Settled by decoding with ffmpeg and
    /// FAAD2 (black box): sent with `ms_used`, FAAD2 decoded the two
    /// channels' noise the same, as the clause says, and ffmpeg each
    /// channel's its own, uncorrelated; sent in intensity stereo, both
    /// decoded it the same, each channel at its level. Of the bands of
    /// music noise-like in both channels, 74 to 93% were so correlated.
    /// The conformance test is
    /// `ffmpeg_and_faad_share_a_pairs_noise_through_intensity_stereo`.
    ///
    /// Spec gap: ISO/IEC 14496-3 does not say how perceptual noise
    /// substitution and temporal noise shaping combine, whose filters the
    /// decoder applies after it generates the noise (clause 4, figure 2 of
    /// ISO/IEC 13818-7). A band spanned by a filter is substituted by
    /// noise of its lines' energy as the filter shaped them, which the
    /// decoder's filter shapes back as it shapes their quantization noise;
    /// where the right channel takes the left channel's noise, its own
    /// filters shape it.
    ///
    /// No low frequency enhancement channel is coded so: it sends only its
    /// lowest lines, below any border.
    fn noise_substitution(&mut self) {
        self.noise_intensity = Intensity::NONE;
        self.noise_ms_used = [0; SHORT_WINDOWS];
        for prepared in &mut self.prepared {
            prepared.noise = Noise::NONE;
        }
        let Some(border) = self.noise_border.filter(|_| self.kind != ElementKind::Lfe) else {
            return;
        };
        let fs = self.sampling_frequency;
        let mut sets = [[0; SHORT_WINDOWS]; 2];
        for ((channel, prepared), set) in self.channels.iter().zip(&self.prepared).zip(&mut sets) {
            *set = noise_substitution::noise_like(
                prepared.windows,
                fs,
                &prepared.spectrum,
                channel.switching.current(),
                border,
            );
        }
        if let ([left, right], true) = (&mut self.prepared[..], self.common_window) {
            let windows = left.windows;
            let mut set: BandSet = std::array::from_fn(|g| sets[0][g] & sets[1][g]);
            let mut shared =
                noise_substitution::shared(windows, fs, &set, [&left.spectrum, &right.spectrum]);
            if self.noise_by_ms_used {
                let mut set = set;
                loop {
                    let l = noise_substitution::reachable(windows, fs, &set, &left.spectrum);
                    let both = noise_substitution::reachable(windows, fs, &l, &right.spectrum);
                    if both == set {
                        break;
                    }
                    set = both;
                }
                shared.keep(&set);
                self.noise_ms_used = shared.set;
                for (prepared, own) in self.prepared.iter_mut().zip([set, set]) {
                    prepared.noise =
                        noise_substitution::substitute(windows, fs, &own, &mut prepared.spectrum);
                }
                return;
            }
            // Bands within reach in both channels, the right channel's
            // without those it takes from the left, until none is dropped.
            loop {
                let own: BandSet = std::array::from_fn(|g| set[g] & !shared.set[g]);
                let l = noise_substitution::reachable(windows, fs, &set, &left.spectrum);
                let r = noise_substitution::reachable(windows, fs, &own, &right.spectrum);
                let both = std::array::from_fn(|g| l[g] & (r[g] | shared.set[g]));
                if both == set {
                    break;
                }
                set = both;
                shared.keep(&set);
            }
            for (g, (start, bands)) in windows.groups_with_start(fs).enumerate() {
                for sfb in (0..bands.num_swb()).filter(|&sfb| (shared.set[g] >> sfb) & 1 == 1) {
                    let band = bands.band(sfb);
                    right.spectrum[start + band.start..start + band.end].fill(0.0);
                }
            }
            sets = [set, std::array::from_fn(|g| set[g] & !shared.set[g])];
            self.noise_intensity = shared.intensity;
        }
        for (prepared, set) in self.prepared.iter_mut().zip(&sets) {
            prepared.noise =
                noise_substitution::substitute(prepared.windows, fs, set, &mut prepared.spectrum);
        }
    }

    /// A pair that shares its window coded in intensity stereo and as
    /// middle and side, with the bands of noise that its right channel
    /// takes from the left in intensity stereo.
    fn joint(&mut self) {
        self.ms = MsMask::NONE;
        self.intensity = Intensity::NONE;
        if self.common_window {
            if let Some(first) = self.intensity_from() {
                self.intensity_bands(first);
            }
            self.intensity.merge(&self.noise_intensity);
            self.middle_side_bands();
            for g in 0..SHORT_WINDOWS {
                for sfb in 0..MAX_BANDS {
                    if (self.noise_ms_used[g] >> sfb) & 1 == 1 {
                        self.ms.set(g, sfb, true);
                    }
                }
            }
        }
    }

    /// Which channels are silent, as they are coded.
    fn find_silence(&mut self) {
        // A right channel with bands in intensity stereo sends them, and a
        // channel with bands of noise sends those, even where its lines are
        // all zero.
        let in_intensity = self.in_intensity();
        for (c, prepared) in self.prepared.iter_mut().enumerate() {
            let sends_positions = c == 1 && in_intensity;
            prepared.silent = !sends_positions
                && !prepared.noise.is_present()
                && prepared.spectrum.iter().all(|&x| x == 0.0);
        }
    }

    /// Whether the frame's pair codes bands in intensity stereo.
    fn in_intensity(&self) -> bool {
        self.intensity.bands.iter().any(|bands| bands.end() > 0)
    }

    /// Codes the frame's pair, which shares its window, in intensity stereo
    /// from the band `first` up (see [`intensity::code`]).
    fn intensity_bands(&mut self, first: usize) {
        let fs = self.sampling_frequency;
        let [left, right] = &mut self.prepared[..] else {
            unreachable!("a pair")
        };
        let windows = left.windows;
        self.intensity = intensity::code(
            windows,
            fs,
            first,
            &left.noise,
            [&mut left.spectrum, &mut right.spectrum],
            [&mut left.xmin, &mut right.xmin],
        );
    }

    /// Chooses the bands of the frame's pair, which shares its window, to
    /// code as middle and side (see [`middle_side::choose`]), and puts the
    /// middle channel's lines and allowances in place of the left
    /// channel's in those bands, and the side channel's in place of the
    /// right's.
    ///
    /// The allowance of the middle and side channels' bands, `xmin(n)`
    /// (step 14), is that of their own thresholds, as the imaging control
    /// adjusts them (see [`MidSideModel::look_ahead`]); the left and right
    /// channels' are adjusted too, whichever bands are coded so.
    ///
    /// Spec gap: Annex C gives the perceptual entropy and `bit_allocation`
    /// of each channel's block (clause C.1.4, step 15), and does not say
    /// what they are for a channel coded as left in some bands and middle
    /// in others. They are taken as those of the middle and side channels
    /// where those code at least half of the frame's lines, and else as
    /// those of the left and right channels.
    fn middle_side_bands(&mut self) {
        let fs = self.sampling_frequency;
        let Some(mid_side) = &mut self.mid_side else {
            unreachable!("a pair")
        };
        let [left, right] = &mut self.prepared[..] else {
            unreachable!("a pair")
        };
        let windows = left.windows;
        let [mid, side] = &mut mid_side.channels;
        middle_side::matrix(
            &left.spectrum,
            &right.spectrum,
            &mut mid.spectrum,
            &mut side.spectrum,
        );
        middle_side::matrix(
            &left.unshaped,
            &right.unshaped,
            &mut mid.unshaped,
            &mut side.unshaped,
        );
        // The noise of the middle and side channels reaches both channels,
        // and is raised by the filters of both.
        for (i, (&l, &r)) in left.weights.iter().zip(&*right.weights).enumerate() {
            (mid.weights[i], side.weights[i]) = (l.max(r), l.max(r));
        }
        let analyses = mid_side.model.current(windows.sequence.transform());
        for (channel, analysis) in [&mut *mid, &mut *side].into_iter().zip(analyses) {
            channel.thresholds(windows, analysis);
        }
        self.ms = middle_side::choose(
            windows,
            fs,
            [
                &*left.spectrum,
                &*right.spectrum,
                &*mid.spectrum,
                &*side.spectrum,
            ],
            [
                &*left.weights,
                &*right.weights,
                &*mid.weights,
                &*side.weights,
            ],
            [
                &left.xmin[..],
                &right.xmin[..],
                &mid.xmin[..],
                &side.xmin[..],
            ],
            &self.intensity,
            &left.noise,
        );
        let mut lines_mid_side = 0;
        for band in stereo::bands(windows, fs, self.ms, &self.intensity, &left.noise) {
            if band.coding == BandCoding::MidSide {
                let (g, sfb, range) = (band.group, band.sfb, band.lines);
                lines_mid_side += range.len();
                left.spectrum[range.clone()].copy_from_slice(&mid.spectrum[range.clone()]);
                right.spectrum[range.clone()].copy_from_slice(&side.spectrum[range.clone()]);
                left.weights[range.clone()].copy_from_slice(&mid.weights[range.clone()]);
                right.weights[range.clone()].copy_from_slice(&side.weights[range]);
                left.xmin[g][sfb] = mid.xmin[g][sfb];
                right.xmin[g][sfb] = side.xmin[g][sfb];
            }
        }
        if 2 * lines_mid_side >= FRAME_LINES {
            for (coded, mid_side) in [left, right].into_iter().zip([&*mid, &*side]) {
                coded.bit_allocation = mid_side.bit_allocation;
                coded.perceptual_entropy = mid_side.perceptual_entropy;
            }
        }
    }

    /// The windows the frame's pair shares, if it does.
    fn common_windows(&self) -> Option<Windows> {
        self.common_window.then(|| self.prepared[0].windows)
    }

    /// Counts the bits of the frame prepared beside its lines.
    fn count_lineless(&mut self) {
        let common = self.common_window;
        let mut lineless = LinelessBits {
            fields: CodedElement::fields_bits(self.channels(), self.common_windows()),
            ..LinelessBits::default()
        };
        for (c, prepared) in self.prepared.iter().enumerate() {
            let windows = prepared.windows;
            lineless.channels_with_tns[c] =
                channel::bits(windows, common, &Pulses::NONE, &prepared.tns, 0);
            lineless.channels_without_tns[c] =
                channel::bits(windows, common, &Pulses::NONE, &Tns::NONE, 0);
        }
        self.lineless = lineless;
    }

    /// The fewest bits the frame prepared can be coded in: its fields, and
    /// its channels without lines or temporal noise shaping, which
    /// [`quantize`](Self::quantize) drops where they do not fit.
    pub fn least_bits(&self) -> u32 {
        self.lineless.element_without_tns()
    }

    /// The channels of the frame prepared that are coded with lines: those
    /// that are not silent.
    pub fn coded_channels(&self) -> u32 {
        self.prepared.iter().filter(|c| !c.silent).count() as u32
    }

    /// The mean perceptual entropy of the channels of the frame prepared
    /// that are coded with lines, or minus infinity if none is.
    pub fn perceptual_entropy(&self) -> f64 {
        let coded = self.prepared.iter().filter(|c| !c.silent);
        let sum: f64 = coded.clone().map(|c| c.perceptual_entropy).sum();
        match coded.count() {
            0 => f64::NEG_INFINITY,
            n => sum / n as f64,
        }
    }

    /// What the frame prepared asks of the frame.
    ///
    /// A low frequency enhancement channel asks for no `bit_allocation`:
    /// its model's perceptual entropy is that of all its lines, of which it
    /// sends [`LFE_LINES`] at most.
    pub fn demand(&self) -> Demand {
        let bit_allocation = match self.kind {
            ElementKind::Lfe => 0.0,
            _ => self.prepared.iter().map(|c| c.bit_allocation).sum(),
        };
        Demand {
            bit_allocation,
            side_info_bits: self.lineless.element_with_tns(),
        }
    }

    /// Quantizes the frame prepared in `available_bits`, the bits of the
    /// element, keeping the filters of each channel's temporal noise
    /// shaping where it pays for them.
    ///
    /// Spec gap: Annex C does not count the bits of `tns_data()` against
    /// the channel's, nor say what to do where they leave its lines no room.
    /// The filters of a channel are dropped, and its lines coded unshaped,
    /// where the channels without lines, with their filters, do not fit in
    /// the element's bits, or the channel's share of them (see
    /// [`code_element`](Self::code_element)); and where, coded, the filters
    /// shape no line that the channel sends: where its loops left all the
    /// lines of the bands they span zero, or sent none of those bands. One
    /// such channel, a tone of 2131 Hz at 96 kHz in a pair at 19.5 kbit/s,
    /// whose filters took most of its bits, decoded to silence in most
    /// frames (the conformance test
    /// `a_tone_keeps_its_lines_beside_filters_it_cannot_afford`).
    pub fn quantize(&mut self, available_bits: u32) -> CodedElement {
        if self.lineless.element_with_tns() > available_bits {
            for c in 0..self.prepared.len() {
                self.unshape(c);
            }
        }
        for c in 0..self.prepared.len() {
            if !self.prepared[c].silent
                && self.share(available_bits) < self.lineless.channels_with_tns[c]
            {
                self.unshape(c);
            }
        }
        let element = self.code_element(available_bits);
        let mut idle = [false; 2];
        for (c, idle) in idle.iter_mut().enumerate().take(self.prepared.len()) {
            *idle = self.prepared[c].tns.is_present() && !self.shapes_lines(&element, c);
        }
        if !idle.contains(&true) {
            return element;
        }
        for c in (0..self.prepared.len()).filter(|&c| idle[c]) {
            self.unshape(c);
        }
        self.code_element(available_bits)
    }

    /// The bits of each channel of the frame that is not silent (see
    /// [`code_element`](Self::code_element)).
    fn share(&self, available_bits: u32) -> u32 {
        let lineless = &self.lineless;
        let reserved: u32 = (self.prepared.iter().zip(lineless.channels_with_tns))
            .filter(|(c, _)| c.silent)
            .map(|(_, bits)| bits)
            .sum();
        (available_bits.saturating_sub(lineless.fields))
            .saturating_sub(reserved)
            .checked_div(self.coded_channels())
            .unwrap_or(0)
    }

    /// Whether channel `c`'s filters, as `element` codes it, shape any line
    /// it sends: whether a band they span, below its `max_sfb`, has a line
    /// that is not zero, its own or, in a band coded as middle and side,
    /// its pair's.
    fn shapes_lines(&self, element: &CodedElement, c: usize) -> bool {
        let fs = self.sampling_frequency;
        let channels = element.channels();
        let gains = &self.prepared[c].gains;
        let windows = channels[c].windows();
        let max_sfb = channels[c].max_sfb();
        let noise = &self.prepared[c].noise;
        for band in stereo::bands(windows, fs, self.ms, &self.intensity, noise) {
            if band.sfb >= max_sfb {
                continue;
            }
            let paired = band.coding == BandCoding::MidSide;
            for (w, range) in band.windows() {
                if !gains.bands(w).contains(&band.sfb) {
                    continue;
                }
                let sends =
                    |channel: &CodedChannel| channel.lines()[range.clone()].iter().any(|&q| q != 0);
                if sends(&channels[c]) || (paired && sends(&channels[1 - c])) {
                    return true;
                }
            }
        }
        false
    }

    /// Drops channel `c`'s filters of temporal noise shaping, and puts its
    /// lines as the filterbank gave them in place of the shaped ones: in a
    /// band coded as middle and side, the middle and side of its unshaped
    /// lines and its pair's as they are coded. The bands in intensity
    /// stereo, which no filter spans, are left as they are.
    fn unshape(&mut self, c: usize) {
        if !self.prepared[c].tns.is_present() {
            return;
        }
        let fs = self.sampling_frequency;
        let windows = self.prepared[c].windows;
        let noise = self.prepared[c].noise.clone();
        for band in stereo::bands(windows, fs, self.ms, &self.intensity, &noise) {
            let range = band.lines;
            match band.coding {
                BandCoding::Intensity | BandCoding::Noise => {}
                BandCoding::MidSide => {
                    let [left, right] = &mut self.prepared[..] else {
                        unreachable!("a pair")
                    };
                    for i in range {
                        let (m, s) = (left.spectrum[i], right.spectrum[i]);
                        let mut lr = [m + s, m - s];
                        lr[c] = if c == 0 {
                            left.unshaped[i]
                        } else {
                            right.unshaped[i]
                        };
                        left.spectrum[i] = (lr[0] + lr[1]) / 2.0;
                        right.spectrum[i] = (lr[0] - lr[1]) / 2.0;
                    }
                }
                BandCoding::Plain => {
                    let prepared = &mut self.prepared[c];
                    prepared.spectrum[range.clone()].copy_from_slice(&prepared.unshaped[range]);
                }
            }
        }
        let prepared = &mut self.prepared[c];
        let unshaped = &*prepared.unshaped;
        noise_substitution::measure(windows, fs, &mut prepared.noise, unshaped);
        prepared.tns = Tns::NONE;
        prepared.gains = NoiseGains::NONE;
        self.lineless.channels_with_tns[c] = self.lineless.channels_without_tns[c];
        self.weigh();
        self.find_silence();
    }

    /// Each channel's weights from its filters' gains, those of a band
    /// coded as middle and side the larger of the pair's.
    fn weigh(&mut self) {
        let fs = self.sampling_frequency;
        for prepared in &mut self.prepared {
            let gains = prepared.gains;
            gains.weights(prepared.windows, fs, &mut prepared.weights);
        }
        if let ([left, right], true) = (&mut self.prepared[..], self.common_window) {
            let noise = &left.noise;
            for band in stereo::bands(left.windows, fs, self.ms, &self.intensity, noise) {
                if band.coding == BandCoding::MidSide {
                    for i in band.lines {
                        let weight = left.weights[i].max(right.weights[i]);
                        (left.weights[i], right.weights[i]) = (weight, weight);
                    }
                }
            }
        }
    }

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
    /// lines, with their filters: a caller's error.
    fn code_element(&mut self, available_bits: u32) -> CodedElement {
        let fs = self.sampling_frequency;
        let share = self.share(available_bits);
        let lineless = self.lineless;
        let budget = |c: usize| {
            if self.prepared[c].silent {
                lineless.channels_with_tns[c]
            } else {
                share
            }
        };
        let mut order: Vec<usize> = (0..self.prepared.len()).collect();
        // A pair in intensity stereo codes its left channel first, which the
        // right channel's bands in intensity stereo scale, and whose
        // `max_sfb` sets which of them it sends; but not for the bands of
        // noise alone, which the left channel sends whatever its lines.
        let in_intensity = self.in_intensity();
        let left_first = (0..SHORT_WINDOWS).any(|g| {
            let noise = self.noise_intensity.bands[g];
            (0..MAX_BANDS).any(|sfb| {
                self.intensity.bands[g].codebook(sfb).is_some() && noise.codebook(sfb).is_none()
            })
        });
        order.sort_by(|&a, &b| {
            if left_first {
                return a.cmp(&b);
            }
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
                let coding = ChannelCoding::new(prepared.windows, fs, budget(only), &prepared.tns);
                let channel = self.code(prepared, coding);
                match self.kind {
                    ElementKind::Lfe => CodedElement::lfe(channel),
                    _ => CodedElement::single(channel),
                }
            }
            [first, second] if self.common_window => {
                let (a, b) = (&self.prepared[first], &self.prepared[second]);
                let windows = a.windows;
                let groups = windows.groups(fs).count();
                let num_swb = windows.groups(fs).next().expect("a group").num_swb();
                let ms = self.ms;
                let mut coding = ChannelCoding {
                    common_window: true,
                    intensity: (first == 1 && in_intensity).then_some(&self.intensity),
                    ..ChannelCoding::new(windows, fs, budget(first), &a.tns)
                };
                let (mut a_channel, intensity, b_coding) = loop {
                    let a_channel = self.code(a, coding);
                    // What the first leaves of its share, which a retry
                    // has lowered its bits within.
                    let left = budget(first) - a_channel.bits(true);
                    let least_max_sfb = a_channel.max_sfb();
                    let mut b_coding = ChannelCoding {
                        common_window: true,
                        least_max_sfb,
                        ..ChannelCoding::new(windows, fs, budget(second) + left, &b.tns)
                    };
                    for m in least_max_sfb..=num_swb {
                        b_coding.element_bits[m] =
                            a_channel.extension_bits(m) + ms.flag_bits(groups, m);
                    }
                    // The bands in intensity stereo that the left channel
                    // sends, where the right channel is second, and the
                    // second channel's bands of noise, which it sends
                    // whatever the first sends.
                    let intensity = if second == 1 {
                        self.intensity.below(least_max_sfb)
                    } else {
                        Intensity::NONE
                    };
                    let least_sfb = least_max_sfb.max(b.noise.end());
                    let b_least =
                        least_channel(windows, fs, least_sfb, &intensity, &b.noise, &b.tns)
                            .bits(true)
                            + b_coding.element_bits[least_sfb];
                    let floor = lineless.channels_with_tns[first];
                    if b_least <= b_coding.available_bits || coding.available_bits <= floor {
                        break (a_channel, intensity, b_coding);
                    }
                    let short = b_least - b_coding.available_bits;
                    coding.available_bits = coding.available_bits.saturating_sub(short).max(floor);
                };
                #[cfg(test)]
                let second_channel = SecondChannel {
                    retried: coding.available_bits < budget(first),
                    first_bits: a_channel.bits(true),
                    available_bits: b_coding.available_bits,
                    budgets: budget(first) + budget(second),
                };
                let b_coding = ChannelCoding {
                    intensity: (second == 1 && in_intensity).then_some(&intensity),
                    ..b_coding
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
                #[cfg(test)]
                {
                    self.second_channel = Some(second_channel);
                }
                CodedElement::common_pair(left, right, ms)
            }
            [first, second] => {
                let mut channels = [None, None];
                let mut left = 0;
                for c in [first, second] {
                    let prepared = &self.prepared[c];
                    let available = budget(c) + left;
                    let coding = ChannelCoding::new(prepared.windows, fs, available, &prepared.tns);
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
    fn code(&self, prepared: &PreparedChannel, coding: ChannelCoding<'_>) -> CodedChannel {
        if prepared.silent {
            CodedChannel::without_lines(prepared.windows, self.sampling_frequency, prepared.tns)
        } else {
            let coding = ChannelCoding {
                spectrum_clipping: self.spectrum_clipping,
                noise: prepared.noise.is_present().then_some(&prepared.noise),
                ..coding
            };
            loops::quantize_channel(
                &prepared.spectrum,
                &prepared.weights,
                &prepared.xmin,
                coding,
            )
        }
    }
}

/// The fewest bits a channel of `windows` shaped by `tns` can be coded in,
/// as the second of a pair that shares its window: without lines, its
/// groups spanning `max_sfb` bands, with the bands of `intensity` and of
/// `noise`, all below it.
fn least_channel(
    windows: Windows,
    sampling_frequency: SamplingFrequency,
    max_sfb: usize,
    intensity: &Intensity,
    noise: &Noise,
    tns: &Tns,
) -> CodedChannel {
    let lines = [0; FRAME_LINES];
    let groups = (windows.groups(sampling_frequency).enumerate())
        .map(|(g, bands)| {
            let sections = Sections::count_with_pseudo(
                &lines[..bands.lines()],
                bands,
                max_sfb,
                intensity.bands[g].union(noise.bands[g]),
            )
            .expect("zeros are codable")
            .choose();
            CodedGroup {
                sections,
                scalefactors: vec![0; max_sfb],
                positions: intensity.positions[g][..max_sfb].to_vec(),
                noise_energies: noise.energies[g][..max_sfb].to_vec(),
            }
        })
        .collect();
    let first_noise = (noise.bands.iter().zip(&noise.energies)).find_map(|(bands, energies)| {
        (0..MAX_BANDS)
            .find(|&sfb| bands.is_noise(sfb))
            .map(|sfb| energies[sfb])
    });
    let global_gain = first_noise.map_or(0, scalefactors::noise_global_gain);
    CodedChannel::new(
        windows,
        sampling_frequency,
        global_gain,
        *tns,
        Pulses::NONE,
        lines,
        groups,
    )
    .expect("positions and noise energies within the codebook's reach")
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
pub(super) const SHARED_ATTACKS: bool = true;

#[cfg(test)]
mod tests {
    use chelys_core::bitstream::BitWriter;

    use super::*;
    use crate::noiseless::decoder::{BitReader, Element, element_of};
    use crate::scalefactor_bands::{GroupBands, Transform};

    /// `INTENSITY_HCB` and `INTENSITY_HCB2`.
    const INTENSITY_HCB: u8 = 15;
    const INTENSITY_HCB2: u8 = 14;

    /// The element of `channels` channels.
    fn kind(channels: usize) -> ElementKind {
        match channels {
            1 => ElementKind::Single,
            _ => ElementKind::Pair,
        }
    }

    /// Codes the frames of `input`, one slice of samples for each channel,
    /// at `bits` for each frame's element, and gives each frame's element
    /// as decoded with the spectra the filterbanks gave.
    fn round_trip(
        fs: SamplingFrequency,
        input: &[Vec<f64>],
        bits: u32,
    ) -> Vec<(Element, Vec<Vec<f64>>)> {
        round_trip_with(fs, input, bits, None)
    }

    /// [`round_trip`], coding a pair in intensity stereo above
    /// `intensity_border` Hz, if given; the spectra are given as the
    /// filterbanks gave them, before temporal noise shaping and before the
    /// pair was coded jointly.
    fn round_trip_with(
        fs: SamplingFrequency,
        input: &[Vec<f64>],
        bits: u32,
        intensity_border: Option<f64>,
    ) -> Vec<(Element, Vec<Vec<f64>>)> {
        round_trip_switching(fs, input, bits, intensity_border, true)
    }

    /// [`round_trip_with`], with `block_switching` or every frame long.
    fn round_trip_switching(
        fs: SamplingFrequency,
        input: &[Vec<f64>],
        bits: u32,
        intensity_border: Option<f64>,
        block_switching: bool,
    ) -> Vec<(Element, Vec<Vec<f64>>)> {
        let encoder = ElementEncoder::new(fs, kind(input.len()), intensity_border, None);
        round_trip_encoder(encoder, input, bits, block_switching)
    }

    /// [`round_trip_switching`] with `encoder`.
    fn round_trip_encoder(
        mut encoder: ElementEncoder,
        input: &[Vec<f64>],
        bits: u32,
        block_switching: bool,
    ) -> Vec<(Element, Vec<Vec<f64>>)> {
        let fs = encoder.sampling_frequency;
        let channels: Vec<&[f64]> = input.iter().map(Vec::as_slice).collect();
        encoder.push(&channels);
        encoder.prime();
        let mut frames = Vec::new();
        let mut start = 0;
        while encoder.buffered() - start >= 2 * FRAME_SAMPLES {
            encoder.look_ahead(start + FRAME_SAMPLES, block_switching);
            encoder.spectra(start, true);
            let original: Vec<Vec<f64>> = (encoder.prepared.iter())
                .map(|prepared| prepared.spectrum.to_vec())
                .collect();
            encoder.prepare_spectra(true);
            let coded = encoder.quantize(bits);
            let mut w = BitWriter::new();
            w.write(u32::from(coded.is_pair()), 3);
            coded.write(&mut w, 0);
            assert!(coded.bits() <= bits, "{} bits in {bits}", coded.bits());
            let len = w.bit_len();
            w.write(0x7, 3);
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            let decoded = element_of(&mut r, fs, encoder.noise_border.is_some());
            assert_eq!(r.position(), len, "the bits written are those read");
            frames.push((decoded, original));
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
                let energy = |lines: &[f64]| lines.iter().map(|x| x * x).sum::<f64>();
                let pair_energy = energy(&spectra[0]) + energy(&spectra[1]);
                for (c, decoded) in decoded.spectra(fs).iter().enumerate() {
                    let original = &spectra[c];
                    if original.iter().any(|&x| x != 0.0) {
                        let snr = snr(original, decoded);
                        assert!(snr > 10.0, "pair {i}, frame {f}, channel {c}: {snr} dB");
                    } else {
                        // Coded as middle and side, a silent channel takes
                        // the difference of their noise, which they may
                        // take up to their thresholds in quiet: where the
                        // other channel's energy is near its own, the
                        // middle and side channels, each with half its
                        // amplitude, are below theirs.
                        let noise = energy(decoded);
                        assert!(
                            noise <= 1e-2 * pair_energy,
                            "pair {i}, frame {f}, channel {c}: {noise} against {pair_energy}"
                        );
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

    /// Single channels and pairs with spectrum clipping fit their bits, in
    /// long windows and short, and send pulses in long windows only.
    #[test]
    fn clipped_channels_fit_their_bits() {
        let fs = SamplingFrequency::Hz44100;
        let len = 12 * FRAME_SAMPLES;
        // Noise, and clicks for short windows.
        let clicks = |seed| -> Vec<f64> {
            (noise(len, seed).iter().enumerate())
                .map(|(i, x)| 0.2 * x + if i % 3000 == 1500 { 0.5 } else { 0.0 })
                .collect()
        };
        let pair = [clicks(5), clicks(6)];
        let (mut pulses, mut short) = (0, 0);
        for (input, bits) in [(&pair[..1], 500), (&pair[..1], 1500), (&pair[..], 3000)] {
            let mut encoder = ElementEncoder::new(fs, kind(input.len()), None, None);
            encoder.spectrum_clipping(true);
            // round_trip_encoder checks the bits and decodes each frame.
            for (element, _) in round_trip_encoder(encoder, input, bits, true) {
                for ics in &element.channels {
                    if ics.info.window_sequence == 2 {
                        short += 1;
                        assert_eq!(ics.pulses, None);
                    }
                    pulses += usize::from(ics.pulses.is_some());
                }
            }
        }
        assert!(pulses > 0 && short > 0, "{pulses} pulses, {short} short");
    }

    /// A pair coded in intensity stereo above 6 kHz decodes with each band
    /// of its right channel there at about its energy, the left channel's
    /// signal scaled, in phase or out of phase as the channels were; below,
    /// both channels decode as they would without it.
    #[test]
    fn intensity_stereo_keeps_each_bands_level_and_phase() {
        let fs = SamplingFrequency::Hz44100;
        let len = 10 * FRAME_SAMPLES;
        let source = noise(len, 5);
        let other = noise(len, 6);
        // The right channel half the left, with a little of its own, in
        // phase and out of phase, and the same as the left, which codes
        // its lines below the border as middle and side, leaving the right
        // channel only its bands in intensity stereo.
        let cases = [
            (1.0, 0.5, INTENSITY_HCB),
            (-1.0, 0.5, INTENSITY_HCB2),
            (1.0, 1.0, INTENSITY_HCB),
        ];
        for (sign, gain, codebook) in cases {
            let own = if gain == 1.0 { 0.0 } else { 0.05 };
            let right: Vec<f64> = (source.iter().zip(&other))
                .map(|(s, o)| sign * gain * s + own * o)
                .collect();
            let input = [source.clone(), right];
            let border = Some(intensity::BORDER_HZ);
            let frames = round_trip_with(fs, &input, 8000, border);
            let mut bands_in_intensity = 0;
            for (f, (decoded, spectra)) in frames.iter().enumerate().skip(1) {
                let right_ics = &decoded.channels[1];
                let info = &right_ics.info;
                // The onset of the noise is coded short.
                if info.window_sequence == 2 {
                    continue;
                }
                let first = crate::scalefactor_bands::first_band_above(fs, Transform::Long, 6000.0);
                let bands = GroupBands::long(fs);
                let out = decoded.spectra(fs);
                // The lines of the bands below the border.
                let mut below = Vec::new();
                for sfb in 0..info.max_sfb {
                    let range = bands.offset(sfb)..bands.offset(sfb + 1);
                    let energy = |lines: &[f64]| lines.iter().map(|x| x * x).sum::<f64>();
                    if right_ics.codebook(0, sfb) == codebook {
                        assert!(sfb >= first, "frame {f}, band {sfb}");
                        bands_in_intensity += 1;
                        let (original, decoded) = (
                            energy(&spectra[1][range.clone()]),
                            energy(&out[1][range.clone()]),
                        );
                        let db = 10.0 * (decoded / original).log10();
                        assert!(db.abs() < 1.5, "frame {f}, band {sfb}: {db:.2} dB");
                        let product: f64 = (out[0][range.clone()].iter())
                            .zip(&out[1][range])
                            .map(|(l, r)| l * r)
                            .sum();
                        assert_eq!(product.signum(), sign, "frame {f}, band {sfb}");
                    } else {
                        assert!(sfb < first, "frame {f}, band {sfb}");
                        below.extend(range);
                    }
                }
                let lines =
                    |spectrum: &[f64]| below.iter().map(|&i| spectrum[i]).collect::<Vec<_>>();
                let snr = snr(&lines(&spectra[1]), &lines(&out[1]));
                assert!(snr > 10.0, "frame {f}: {snr:.1} dB");
            }
            assert!(bands_in_intensity > 100, "{bands_in_intensity}");
        }
    }

    /// Bands of a pair's short windows trimmed to the left channel's
    /// `max_sfb` keep positions that the scalefactor codebook can send.
    /// Clicks over quiet noise are shaped in time in their frames, long and
    /// short, in one channel and in a pair, and the decoder's filters
    /// restore their lines; in a pair coded in intensity stereo, the
    /// filters stop below its first band in intensity stereo, and the
    /// right channel's bands there keep their level.
    #[test]
    fn clicks_are_shaped_in_time_and_restored() {
        let fs = SamplingFrequency::Hz44100;
        let len = 16 * FRAME_SAMPLES;
        let clicks = |seed: u64, every: usize| -> Vec<f64> {
            let mut input: Vec<f64> = noise(len, seed).iter().map(|x| 0.01 * x).collect();
            for at in (700..len).step_by(every) {
                input[at] += 0.7;
            }
            input
        };
        let first = first_band_above(fs, Transform::Long, intensity::BORDER_HZ);
        let cases = [
            (vec![clicks(1, 1301)], 6000, None),
            (vec![clicks(2, 1301), clicks(3, 1777)], 12000, None),
            (
                vec![clicks(4, 1301), clicks(4, 1301)],
                12000,
                Some(intensity::BORDER_HZ),
            ),
        ];
        for (i, (input, bits, border)) in cases.iter().enumerate() {
            let mut frames = round_trip_with(fs, input, *bits, *border);
            frames.extend(round_trip_switching(fs, input, *bits, *border, false));
            let (mut long, mut short) = (0, 0);
            for (f, (decoded, spectra)) in frames.iter().enumerate() {
                for (c, (ics, decoded)) in
                    decoded.channels.iter().zip(decoded.spectra(fs)).enumerate()
                {
                    let Some(tns) = &ics.tns else { continue };
                    let context = format!("case {i}, frame {f}, channel {c}");
                    if ics.info.window_sequence == 2 {
                        short += tns.iter().filter(|w| !w.filters.is_empty()).count();
                    } else {
                        long += 1;
                        let filters = &tns[0].filters;
                        if border.is_some() {
                            // A filter of order 0 from the first band in
                            // intensity stereo up, then the filter.
                            assert_eq!(filters.len(), 2, "{context}");
                            assert_eq!(filters[0].length, 49 - first, "{context}");
                            assert_eq!(filters[0].order, 0, "{context}");
                        } else {
                            assert_eq!(filters.len(), 1, "{context}");
                        }
                    }
                    // Below the border, where the pair is coded in
                    // intensity stereo, as the right channel's bands above
                    // it are not its own.
                    let lines = match (border, ics.info.window_sequence) {
                        (Some(_), 0 | 1 | 3) => GroupBands::long(fs).offset(first),
                        _ => FRAME_LINES,
                    };
                    if spectra[c][..lines].iter().all(|&x| x == 0.0) {
                        continue;
                    }
                    let snr = snr(&spectra[c][..lines], &decoded[..lines]);
                    assert!(snr > 10.0, "{context}: {snr:.1} dB");
                }
            }
            assert!(long > 3, "case {i}: {long} long windows shaped");
            assert!(
                border.is_some() || short > 3,
                "case {i}: {short} short windows shaped"
            );
        }
    }

    #[test]
    fn trimmed_positions_stay_within_reach_of_each_other() {
        use crate::noiseless::codebooks::Codebook;
        use crate::noiseless::scalefactors;
        let mut intensity = Intensity::NONE;
        let most = intensity::MAX_POSITION;
        for (g, sfb, position) in [(0, 3, most), (0, 8, 0), (1, 3, -most), (2, 4, most)] {
            intensity.bands[g].set(sfb, Some(Codebook::INTENSITY));
            intensity.positions[g][sfb] = position;
        }
        let below = intensity.below(5);
        assert_eq!(below.bands[0].end(), 4);
        assert_eq!(below.bands[1].end(), 4);
        let fs = SamplingFrequency::Hz48000;
        let grouping = crate::window::Grouping::from_starts([
            true, false, true, false, true, false, false, false,
        ]);
        let windows = Windows {
            sequence: WindowSequence::EightShort { grouping },
            shape: crate::window::WindowShape::Sine,
        };
        let channel = least_channel(windows, fs, 5, &below, &Noise::NONE, &Tns::NONE);
        let groups: Vec<_> = channel
            .groups()
            .map(|(group, _)| group.scalefactor_data())
            .collect();
        assert!(scalefactors::bits(0, groups).is_some());
    }

    /// Where the first channel of a pair is quantized again in fewer bits,
    /// for the second to pay for its bands, the second is given every bit
    /// of the two channels' shares that the first leaves, the bits taken
    /// from the first among them: at 64 kHz and 23 kbit/s, with noise in
    /// each channel in turn, coded with short windows.
    #[test]
    fn a_retry_gives_the_second_channel_what_the_first_leaves() {
        let fs = SamplingFrequency::Hz64000;
        let len = 24 * FRAME_SAMPLES;
        // Each channel noisy for 200 samples in turn.
        let burst = |c: usize, seed: u64| -> Vec<f64> {
            (noise(len, seed).into_iter().enumerate())
                .map(|(t, x)| if (t / 200) % 2 == c { x } else { 0.0 })
                .collect()
        };
        let input = [burst(0, 7), burst(1, 8)];
        // The bits of each frame's element at 23 kbit/s, less the header,
        // id_syn_ele and ID_END.
        let bits = 23_000 * 1024 / 64_000 - 56 - 6;
        let mut encoder = ElementEncoder::new(fs, ElementKind::Pair, None, None);
        let channels: Vec<&[f64]> = input.iter().map(Vec::as_slice).collect();
        encoder.push(&channels);
        encoder.prime();
        let mut retries = 0;
        let mut start = 0;
        while encoder.buffered() - start >= 2 * FRAME_SAMPLES {
            encoder.look_ahead(start + FRAME_SAMPLES, true);
            // Short windows in every frame, whose side information leaves
            // the channels the fewest bits for their lines.
            for channel in &mut encoder.channels {
                channel.attacks = Attacks {
                    this: true,
                    next: true,
                };
            }
            encoder.prepare(start, true);
            encoder.second_channel = None;
            let coded = encoder.quantize(bits);
            assert!(coded.bits() <= bits);
            if let Some(second) = encoder.second_channel {
                assert_eq!(second.first_bits + second.available_bits, second.budgets);
                retries += usize::from(second.retried);
            }
            start += FRAME_SAMPLES;
        }
        assert!(retries > 0, "no frame quantized its first channel again");
    }

    /// The energy of each band of each group of `spectrum`, a channel's
    /// lines of `windows` in the order they are sent, in dB.
    fn band_levels(fs: SamplingFrequency, windows: Windows, spectrum: &[f64]) -> Vec<Vec<f64>> {
        (windows.groups_with_start(fs))
            .map(|(start, bands)| {
                (0..bands.num_swb())
                    .map(|sfb| {
                        let band = bands.band(sfb);
                        let lines = &spectrum[start + band.start..start + band.end];
                        10.0 * lines.iter().map(|x| x * x).sum::<f64>().log10()
                    })
                    .collect()
            })
            .collect()
    }

    /// Noise is coded by perceptual noise substitution above the border,
    /// and decodes at the level of the lines it stands for.
    #[test]
    fn noise_is_substituted_above_the_border_at_its_level() {
        let fs = SamplingFrequency::Hz44100;
        let input = [noise(16 * FRAME_SAMPLES, 7)];
        let encoder = ElementEncoder::new(fs, kind(1), None, Some(noise_substitution::BORDER_HZ));
        let frames = round_trip_encoder(encoder, &input, 1500, true);
        let first = first_band_above(fs, Transform::Long, noise_substitution::BORDER_HZ);
        let mut substituted = 0;
        for (decoded, original) in &frames[2..] {
            let ics = &decoded.channels[0];
            let windows = Windows {
                sequence: match ics.info.window_sequence {
                    2 => WindowSequence::EIGHT_SHORT,
                    _ => WindowSequence::OnlyLong,
                },
                ..Windows::LONG_SINE
            };
            if windows.sequence != WindowSequence::OnlyLong {
                continue;
            }
            let decoded_levels = band_levels(fs, windows, &decoded.spectra(fs)[0]);
            let original_levels = band_levels(fs, windows, &original[0]);
            for sfb in 0..ics.info.max_sfb {
                if ics.codebook(0, sfb) != 13 {
                    continue;
                }
                assert!(sfb >= first, "band {sfb} below the border");
                substituted += 1;
                let error = decoded_levels[0][sfb] - original_levels[0][sfb];
                assert!(error.abs() <= 0.76, "band {sfb} off by {error:.2} dB");
            }
        }
        assert!(substituted > 100, "{substituted} bands substituted");
    }

    /// A pair that shares its window makes its noise the same in both
    /// channels where they are alike, the right channel taking the left
    /// channel's in intensity stereo, and not where they are not.
    #[test]
    fn a_pair_shares_its_noise_where_its_channels_are_alike() {
        let fs = SamplingFrequency::Hz48000;
        let len = 12 * FRAME_SAMPLES;
        let left = noise(len, 11);
        let half: Vec<f64> = left.iter().map(|x| 0.5 * x).collect();
        let other = noise(len, 12);
        for (right, alike) in [(half, true), (other, false)] {
            let encoder =
                ElementEncoder::new(fs, kind(2), None, Some(noise_substitution::BORDER_HZ));
            let frames = round_trip_encoder(encoder, &[left.clone(), right], 3000, false);
            let (mut substituted, mut shared) = (0, 0);
            for (decoded, _) in &frames[2..] {
                assert!(decoded.common_window);
                let [l, r] = &decoded.channels[..] else {
                    unreachable!("a pair")
                };
                let spectra = decoded.spectra(fs);
                let offsets = crate::scalefactor_bands::swb_offsets(fs, Transform::Long);
                for sfb in 0..l.info.max_sfb {
                    let codebooks = (l.codebook(0, sfb), r.codebook(0, sfb));
                    match codebooks {
                        (13, 13) => substituted += 1,
                        (13, INTENSITY_HCB) => {
                            substituted += 1;
                            shared += 1;
                            let band = usize::from(offsets[sfb])..usize::from(offsets[sfb + 1]);
                            let (a, b) = (&spectra[0][band.clone()], &spectra[1][band]);
                            let product: f64 = a.iter().zip(b).map(|(a, b)| a * b).sum();
                            let norm = (a.iter().map(|x| x * x).sum::<f64>()
                                * b.iter().map(|x| x * x).sum::<f64>())
                            .sqrt();
                            assert!((product / norm - 1.0).abs() < 1e-9);
                        }
                        // Only bands of noise in both channels, or noise
                        // the right channel takes from the left.
                        (a, b) => assert!(a != 13 && b != 13, "band {sfb}: {codebooks:?}"),
                    }
                }
            }
            assert!(substituted > 50, "{substituted} bands substituted");
            if alike {
                assert_eq!(shared, substituted);
            } else {
                assert!(
                    20 * shared < substituted,
                    "{shared} of {substituted} shared"
                );
            }
        }
    }
}
