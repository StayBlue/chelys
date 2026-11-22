// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The frames of a stream: one per block of 1024 samples of each channel,
//! each as long as the bit reservoir lets it be, framed as asked.

mod element;
mod frame;

use super::DELAY;
use super::adts::{self, Flags};
use super::audio_specific_config::audio_specific_config;
use super::channel::CodedChannel;
use super::channel_configuration::{ChannelConfiguration, ElementKind};
use super::config::Framing;
use super::noise_substitution;
use super::raw_data_block;
use super::raw_data_block::data_stream::{self, DataStreamQueue};
use super::raw_data_block::element::CodedElement;
use super::reservoir::{self, BitReservoir};
use super::sampling_frequency::SamplingFrequency;
use super::stereo::intensity;
use super::stereo::middle_side::MsMask;
use super::window::{WindowSequence, Windows};
use element::ElementEncoder;

/// The samples of each channel a frame codes (clause 8.2.1.1).
pub const FRAME_SAMPLES: usize = 1024;

/// The most bits for each sample of each channel at which a pair is coded
/// in intensity stereo (see [`intensity::code`]).
///
/// Spec gap: clause C.6.2 codes the bands above `f0` in intensity stereo,
/// "one possible implementation" among several, and does not say when.
/// Intensity stereo gives up the right channel's waveform in those bands
/// for the bits it saves, which pays only where bits are few. Graded with
/// PEAQ (GstPEAQ 0.6.1, black box), on 8 tracks of pop music at 44.1 kHz
/// in stereo and on 3 tracks of rock music and their 3 windows of 3 s
/// with the most frames coded short, with M/S stereo, intensity stereo
/// from 6 kHz gave means of -3.711 and -3.611 at 48 kbit/s against -3.754
/// and -3.636 without, and -3.648 and -3.596 at 64 kbit/s against -3.703
/// and -3.579. Before bands where a channel is silent and frames of short
/// windows were left out of it, which changed no grade at 64 kbit/s by
/// more than 0.001, it gave -3.163 and -2.497 at 96 kbit/s against -2.992
/// and -2.166, -1.965 and -1.403 at 128 kbit/s against -1.485 and -0.938,
/// and -0.622 and -0.355 at 192 kbit/s against -0.411 and -0.197; and at
/// 64 kbit/s from 9 and 12 kHz, -3.657 and -3.591, and -3.673 and -3.591.
/// The limit lies between 64 and 96 kbit/s, 0.73 and 1.09 bits a sample;
/// it is set per sample, untested at other sampling frequencies, so as to
/// hold where a band's lines are as few for its bits. Below about 16 kHz
/// sampling, the border of 6 kHz leaves few bands above it or none, so the
/// limit matters little there. At 88.2 and 96 kHz, where a frame's bits
/// are spread over a band twice as wide, the limit is generous, and
/// untested.
pub const INTENSITY_BITS_PER_SAMPLE: f64 = 0.75;

/// The most bits for each sample of each channel at which channels are
/// coded by perceptual noise substitution, where the stream codes them so
/// (see [`noise_substitution::noise_like`]).
///
/// Spec gap: Annex 4.B.11 of ISO/IEC 14496-3 does not say when. Perceptual
/// noise substitution gives up a band's waveform for the bits it saves,
/// which pays only where bits are few. Graded as for the limits of
/// [`noise_substitution::noise_like`] (PEAQ, black box, with the encoder's
/// bandwidth cut alike with and without it, at 10, 13, 14 and 15 kHz at
/// 64, 96, 112 and 128 kbit/s), the 10 excerpts on which they were chosen
/// gained 0.226 at 112 kbit/s, 9 of them, and lost 0.129 at 128 kbit/s,
/// all but one; 10 others, of other tracks of the same albums, gained
/// 0.300, 0.477 and 0.253 at 64, 96 and 112 kbit/s, every one at each.
/// The limit lies between 112 and 128 kbit/s, 1.27 and 1.45 bits a
/// sample, set per sample as for [`INTENSITY_BITS_PER_SAMPLE`], untested
/// at other sampling frequencies.
///
/// With the bandwidth uncut, PEAQ grades it far higher, as it fills with
/// noise the treble that is otherwise lost, which PEAQ's measures of
/// bandwidth weigh heavily: the 10 other excerpts graded -2.117, -0.960 and
/// -0.805 at 64, 96 and 112 kbit/s, against -3.564, -2.451 and -1.615
/// without it, their bandwidth 20.7 to 20.9 kHz against 12.3, 15.0 and
/// 15.9 kHz. That gain is not counted here: on the excerpts whose measures
/// were examined, PEAQ's measures of noise grew with it.
///
/// Of 5 synthetic signals of 15 s, with the bandwidth uncut, its tonal
/// ones lost 0.15 at most at 64, 96 and 112 kbit/s: the chord of
/// [`noise_substitution::FLATNESS_LIMIT`], a harpsichord of plucked
/// harmonic notes and castanets of clicks; its noise-like ones gained,
/// cymbals of decaying noise by 0.82, 2.78 and 2.84, and sibilants of
/// noise bursts over a tone by 0.61, 0.95 and 0.31.
///
/// The grade at 128 kbit/s, and those that chose the limits of
/// [`noise_substitution::noise_like`], were taken while a pair's noise
/// that is to be the same in both channels was sent with `ms_used`, which
/// ffmpeg, which decoded them, takes as noise of each channel's own (see
/// `ElementEncoder::noise_substitution`); sent in intensity stereo, as it
/// is, the music's grades moved by 0.02 at most.
pub const NOISE_BITS_PER_SAMPLE: f64 = 1.3;

/// What the stream is: what its headers say, its rate and its framing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamConfig {
    pub sampling_frequency: SamplingFrequency,
    pub channel_configuration: ChannelConfiguration,
    pub bits_per_second: u32,
    pub framing: Framing,
    /// The flags of each ADTS header; raw frames have none.
    pub flags: Flags,
    /// Whether each ADTS frame has an `adts_error_check()`; raw frames
    /// have none.
    pub crc_protected: bool,
    /// Whether channels are coded by perceptual noise substitution, at
    /// rates of up to [`NOISE_BITS_PER_SAMPLE`], which makes the stream
    /// MPEG-4 AAC (see [`noise_substitution::noise_like`]).
    pub noise_substitution: bool,
}

impl StreamConfig {
    /// The header of each frame in ADTS.
    pub fn adts_header(&self) -> Option<adts::Header> {
        (self.framing == Framing::Adts).then_some(adts::Header {
            sampling_frequency: self.sampling_frequency,
            channel_configuration: self.channel_configuration,
            flags: self.flags,
            protected: self.crc_protected,
            mpeg4: self.noise_border().is_some(),
        })
    }

    /// The bytes each frame adds before its `raw_data_block()`: its header
    /// and `adts_error_check()` in ADTS.
    pub fn header_len(&self) -> usize {
        self.adts_header().map_or(0, |header| header.bytes())
    }

    /// The shortest frame of channels with `windows` that carries
    /// `data_bytes` of data streams, as the encoder writes it: its channels
    /// silent, without fill, a pair sharing its `ics_info()` where its
    /// channels share their attacks, and so their windows, in every frame
    /// (see [`element::SHARED_ATTACKS`]), and each with its own otherwise,
    /// and a low frequency enhancement channel with long windows, its only
    /// ones (see [`CodedElement::lfe`]).
    fn shortest_frame_len(&self, windows: Windows, data_bytes: usize) -> usize {
        let fs = self.sampling_frequency;
        let channel = CodedChannel::silent(windows, fs);
        let configuration = self.channel_configuration;
        let elements: Vec<CodedElement> = (configuration.elements().iter())
            .map(|kind| match kind {
                ElementKind::Single => CodedElement::single(channel.clone()),
                ElementKind::Pair if element::SHARED_ATTACKS => {
                    CodedElement::common_pair(channel.clone(), channel.clone(), MsMask::NONE)
                }
                ElementKind::Pair => CodedElement::pair(channel.clone(), channel.clone()),
                ElementKind::Lfe => CodedElement::lfe(CodedChannel::silent(Windows::LONG_SINE, fs)),
            })
            .collect();
        self.header_len() + raw_data_block::min_len(configuration, &elements, data_bytes)
    }

    /// The lowest bitrate that brings each frame the bits of the shortest,
    /// that of long windows, so that the reservoir never runs below empty.
    /// A frame that cannot afford the longer `ics_info()` of eight short
    /// windows is coded with long ones.
    pub fn min_bits_per_second(&self) -> u32 {
        let bits = 8 * self.shortest_frame_len(Windows::LONG_SINE, 0) as u64;
        let sample_rate = u64::from(self.sampling_frequency.hz());
        (bits * sample_rate).div_ceil(FRAME_SAMPLES as u64) as u32
    }

    /// The most bytes of data streams that each frame can carry: as many as
    /// the shortest frame that carries them, that of long windows, keeps
    /// within the whole bytes the rate brings each frame, so that the
    /// reservoir never runs below empty (see [`Self::min_bits_per_second`]).
    /// 0 if the stream is not [valid](Self::is_valid).
    pub fn max_data_stream_bytes(&self) -> usize {
        let mean = u64::from(self.bits_per_second) * FRAME_SAMPLES as u64
            / (8 * u64::from(self.sampling_frequency.hz()));
        let fits = |bytes| self.shortest_frame_len(Windows::LONG_SINE, bytes) as u64 <= mean;
        // The frame grows with the bytes.
        let (mut fitting, mut over) = (0, mean as usize + 1);
        if !fits(fitting) {
            return 0;
        }
        while over - fitting > 1 {
            let mid = fitting + (over - fitting) / 2;
            if fits(mid) {
                fitting = mid;
            } else {
                over = mid;
            }
        }
        fitting
    }

    /// The bits of the shortest frame of eight short windows, those of a
    /// low frequency enhancement channel long, without data streams, which
    /// a frame must be able to take with its data streams to be coded with
    /// them (see [`StreamEncoder::affords_short`]).
    fn shortest_short_frame_bits(&self) -> u32 {
        let windows = Windows {
            sequence: WindowSequence::EIGHT_SHORT,
            ..Windows::LONG_SINE
        };
        8 * self.shortest_frame_len(windows, 0) as u32
    }

    /// The longest frame in bytes: [`reservoir::MAX_ADTS_FRAME_LEN`] in
    /// ADTS, and none but the buffer's for raw frames.
    fn max_frame_len(&self) -> Option<usize> {
        (self.framing == Framing::Adts).then_some(reservoir::MAX_ADTS_FRAME_LEN)
    }

    /// The highest bitrate, at which the reservoir holds a byte, and a
    /// mean frame is a byte shorter than the longest at least (see
    /// [`BitReservoir::new`]).
    pub fn max_bits_per_second(&self) -> u32 {
        BitReservoir::max_bits_per_second(
            self.sampling_frequency,
            self.channel_configuration,
            self.max_frame_len(),
        )
    }

    fn reservoir(&self) -> Option<BitReservoir> {
        if self.bits_per_second < self.min_bits_per_second() {
            return None;
        }
        BitReservoir::new(
            self.sampling_frequency,
            self.channel_configuration,
            self.bits_per_second,
            self.max_frame_len(),
        )
    }

    /// The lower border frequency above which a pair is coded in intensity
    /// stereo, if it is: Annex C's (see [`intensity::BORDER_HZ`]), where
    /// the stream has at most [`INTENSITY_BITS_PER_SAMPLE`] bits for each
    /// sample of each channel of the NCC.
    fn intensity_border(&self) -> Option<f64> {
        (self.bits_per_sample() <= INTENSITY_BITS_PER_SAMPLE).then_some(intensity::BORDER_HZ)
    }

    /// The lower border frequency above which channels are coded by
    /// perceptual noise substitution, if they are: Annex 4.B.11's (see
    /// [`noise_substitution::BORDER_HZ`]), where the stream codes them so
    /// and has at most [`NOISE_BITS_PER_SAMPLE`] bits for each sample of
    /// each channel of the NCC.
    fn noise_border(&self) -> Option<f64> {
        (self.noise_substitution && self.bits_per_sample() <= NOISE_BITS_PER_SAMPLE)
            .then_some(noise_substitution::BORDER_HZ)
    }

    /// The bits for each sample of each channel of the NCC.
    fn bits_per_sample(&self) -> f64 {
        let ncc = self.channel_configuration.ncc() as f64;
        f64::from(self.bits_per_second) / (ncc * f64::from(self.sampling_frequency.hz()))
    }

    /// Whether every frame of a stream so configured can be written.
    pub fn is_valid(&self) -> bool {
        self.reservoir().is_some()
    }
}

/// Where an [`Encoder`](crate::Encoder) gives the frames it codes, one call each, in their
/// order.
pub trait FrameSink {
    /// Takes a frame: an `adts_frame()` or a `raw_data_block()`, as
    /// [`Config::framing`](crate::Config::framing) asked.
    fn frame(&mut self, frame: &[u8]);
}

/// Appends each frame, as to a stream in ADTS.
impl FrameSink for Vec<u8> {
    fn frame(&mut self, frame: &[u8]) {
        self.extend_from_slice(frame);
    }
}

/// Calls the closure with each frame, as to keep raw frames apart:
///
/// ```
/// # use chelys_aac::{Config, Encoder, Framing, Rate};
/// let config = Config::new(48_000, 1, Rate::Constant { bits_per_second: 64_000 })
///     .framing(Framing::Raw);
/// let mut encoder = Encoder::new(config).unwrap();
/// let mut frames = Vec::new();
/// let mut out = |frame: &[u8]| frames.push(frame.to_vec());
/// encoder.encode_interleaved(&[0i16; 4096], &mut out);
/// encoder.finish(&mut out);
/// assert_eq!(frames.len(), 5);
/// ```
impl<F: FnMut(&[u8])> FrameSink for F {
    fn frame(&mut self, frame: &[u8]) {
        self(frame);
    }
}

/// Codes a stream from PCM as it arrives, in any lengths, a frame for each
/// block of [`FRAME_SAMPLES`] of each channel, once the block after it has
/// arrived, from which the frame's windows are chosen.
#[derive(Debug)]
pub struct StreamEncoder {
    config: StreamConfig,
    reservoir: BitReservoir,
    /// The elements of each frame, which `channel_configuration` gives, in
    /// order.
    elements: Vec<ElementEncoder>,
    /// Whether frames may be coded with short windows.
    block_switching: bool,
    /// [`StreamConfig::shortest_short_frame_bits`].
    shortest_short_frame_bits: u32,
    /// The bytes queued for the frames' data streams.
    data: DataStreamQueue,
    /// The samples each channel was given.
    samples: u64,
    frames: u64,
}

/// What a [`StreamEncoder`] coded, once it has finished.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamEnd {
    pub frames: u64,
    /// The samples each channel was given.
    pub samples: u64,
    /// The bytes of data streams that the last frames had no room for,
    /// dropped.
    pub data_stream_dropped: usize,
}

impl StreamEncoder {
    /// # Panics
    ///
    /// If the configuration is not [valid](StreamConfig::is_valid).
    pub fn new(config: StreamConfig) -> Self {
        let reservoir = config
            .reservoir()
            .unwrap_or_else(|| panic!("a stream that cannot be written: {config:?}"));
        let elements = (config.channel_configuration.elements().iter())
            .map(|&kind| {
                ElementEncoder::new(
                    config.sampling_frequency,
                    kind,
                    config.intensity_border(),
                    config.noise_border(),
                )
            })
            .collect();
        Self {
            config,
            reservoir,
            elements,
            block_switching: true,
            shortest_short_frame_bits: config.shortest_short_frame_bits(),
            data: DataStreamQueue::default(),
            samples: 0,
            frames: 0,
        }
    }

    /// The encoder, with each frame carrying up to `bytes_per_frame` bytes
    /// of those [pushed](Self::push_data_stream) in data stream elements,
    /// which take bits from its audio.
    ///
    /// # Panics
    ///
    /// If `bytes_per_frame` is above
    /// [`StreamConfig::max_data_stream_bytes`]: a caller's error.
    pub fn with_data_stream(mut self, bytes_per_frame: usize) -> Self {
        let max = self.config.max_data_stream_bytes();
        assert!(
            bytes_per_frame <= max,
            "{bytes_per_frame} bytes of data streams a frame, not at most {max}"
        );
        self.data = DataStreamQueue::new(bytes_per_frame);
        self
    }

    /// Queues `bytes` for the data streams of the frames coded from now on.
    pub fn push_data_stream(&mut self, bytes: &[u8]) {
        self.data.push(bytes);
    }

    /// The bytes queued that no frame has carried yet.
    pub fn data_stream_pending(&self) -> usize {
        self.data.pending()
    }

    /// The encoder, but coding every frame with long windows, as if no
    /// frame had an attack, for comparing with block switching in tests.
    #[cfg(any(test, feature = "unstable"))]
    pub fn without_block_switching(mut self) -> Self {
        self.block_switching = false;
        self
    }

    /// The encoder, but without temporal noise shaping, for comparing with
    /// it in tests.
    #[cfg(any(test, feature = "unstable"))]
    pub fn without_tns(mut self) -> Self {
        for element in &mut self.elements {
            element.without_tns();
        }
        self
    }

    /// The encoder, but with spectrum clipping on or off, for comparing
    /// the two in tests.
    #[cfg(any(test, feature = "unstable"))]
    pub fn spectrum_clipping(mut self, on: bool) -> Self {
        for element in &mut self.elements {
            element.spectrum_clipping(on);
        }
        self
    }

    /// The encoder, but with the noise of a pair's bands that is to be the
    /// same in both channels sent as noise in both, with `ms_used` set, as
    /// clause 4.6.13.3 of ISO/IEC 14496-3 has it, rather than in intensity
    /// stereo in the right channel, for comparing the two in tests (see
    /// `ElementEncoder::noise_substitution`).
    #[cfg(any(test, feature = "unstable"))]
    pub fn noise_shared_by_ms_used(mut self) -> Self {
        for element in &mut self.elements {
            element.noise_shared_by_ms_used();
        }
        self
    }

    /// Whether the next frame's bits, those the reservoir makes available
    /// to it when it asks for none beyond the mean, or offers it whatever it
    /// asks for (see [`BitReservoir::offered_bits`]), pay for the shortest
    /// frame of eight short windows in every channel but a low frequency
    /// enhancement channel, with `data_bits` of data stream elements, and
    /// so for any windows: a frame is given no fewer when it asks for more.
    /// Near the lowest bitrate they may not, as it is set by the frames of
    /// long windows (see [`StreamConfig::min_bits_per_second`]).
    fn affords_short(&self, data_bits: u32) -> bool {
        let bits = (self.reservoir.available_bits(0)).max(self.reservoir.offered_bits());
        self.shortest_short_frame_bits + data_bits <= bits
    }

    /// The `AudioSpecificConfig()` of ISO/IEC 14496-3 that describes the
    /// stream.
    pub fn audio_specific_config(&self) -> Vec<u8> {
        audio_specific_config(
            self.config.sampling_frequency,
            self.config.channel_configuration,
        )
    }

    /// The sampling frequency in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.config.sampling_frequency.hz()
    }

    /// The bits of the decoder's input buffer (clause 8.2.3.1), within
    /// which the bit reservoir keeps the stream.
    pub fn decoder_buffer_bits(&self) -> u32 {
        reservoir::buffer_bits(self.config.channel_configuration)
    }

    /// Takes `channels`, the next samples of each channel at full scale
    /// 1.0, in the order of the elements that carry them (see
    /// [`ChannelConfiguration::elements`]), and gives `out` the frames they
    /// complete, each once the block after it has arrived.
    ///
    /// # Panics
    ///
    /// If there is not one slice for each channel, all of one length: a
    /// caller's error.
    pub fn encode<O: FrameSink + ?Sized>(&mut self, channels: &[&[f64]], out: &mut O) {
        assert_eq!(
            channels.len(),
            self.config.channel_configuration.channels(),
            "one slice per channel"
        );
        let len = channels[0].len();
        assert!(
            channels.iter().all(|c| c.len() == len),
            "every channel has the same number of samples"
        );
        let mut rest = channels;
        for element in &mut self.elements {
            let (own, after) = rest.split_at(element.channels());
            element.push(own);
            rest = after;
        }
        self.samples += len as u64;
        self.encode_buffered(out);
    }

    /// The first element, whose buffers and models are in step with the
    /// others'.
    fn first(&self) -> &ElementEncoder {
        &self.elements[0]
    }

    /// Codes the frames whose blocks, and the block after each, are
    /// buffered, and drops their samples.
    fn encode_buffered<O: FrameSink + ?Sized>(&mut self, out: &mut O) {
        if !self.first().is_primed() && self.first().buffered() >= FRAME_SAMPLES {
            for element in &mut self.elements {
                element.prime();
            }
        }
        let mut start = 0;
        while self.first().is_primed() && self.first().buffered() - start >= 2 * FRAME_SAMPLES {
            self.encode_frame(start, out);
            start += FRAME_SAMPLES;
        }
        for element in &mut self.elements {
            element.drain(start);
        }
    }

    /// Ends the stream: pads the input with silence to whole frames, at
    /// least [`DELAY`] samples past it, so that the decoded output holds
    /// all of the input, and gives `out` the frames left. The block after
    /// the last frame, from which its windows are chosen, is silence too.
    pub fn finish<O: FrameSink + ?Sized>(mut self, out: &mut O) -> StreamEnd {
        let frames = (self.samples + DELAY as u64).div_ceil(FRAME_SAMPLES as u64);
        // Few: those of the samples held and of DELAY, and the block after
        // the last.
        let remaining = (frames - self.frames) as usize;
        for element in &mut self.elements {
            element.pad((remaining + 1) * FRAME_SAMPLES);
        }
        self.encode_buffered(out);
        debug_assert_eq!(self.frames, frames);
        StreamEnd {
            frames: self.frames,
            samples: self.samples,
            data_stream_dropped: self.data.pending(),
        }
    }

    /// Codes the frame whose block starts at `start` in the samples
    /// buffered, the block after it following, and gives it to `out`, in
    /// four steps: has the elements' psychoacoustic models analyze the
    /// block after it, for the attacks that choose the frame's windows;
    /// prepares the elements' channels for quantization (see
    /// [`ElementEncoder::prepare`]); has the bit reservoir make bits
    /// available to the frame for the bits the models ask for; and
    /// quantizes the elements in them, less those of the data the frame
    /// carries (see [`Self::quantize`]).
    fn encode_frame<O: FrameSink + ?Sized>(&mut self, start: usize, out: &mut O) {
        for element in &mut self.elements {
            element.look_ahead(start + FRAME_SAMPLES, self.block_switching);
        }
        let data = self.data.take_next_frame();
        let data_bits = data_stream::bits(data.len());
        let afford_short = self.affords_short(data_bits);
        for element in &mut self.elements {
            element.prepare(start, afford_short);
        }
        let demand = self.elements.iter().map(ElementEncoder::demand).sum();
        let available_bits =
            frame::elements_bits(&self.config, &self.reservoir, &demand, data_bits);
        let elements = self.quantize(available_bits);
        frame::write(&self.config, &mut self.reservoir, &elements, &data, out);
        self.frames += 1;
    }

    /// Quantizes the frame's elements, prepared, in `available_bits`.
    ///
    /// No element is given more than its share of the decoder's input
    /// buffer (see [`reservoir::element_max_bits`]); what it cannot take
    /// goes to those quantized after it.
    ///
    /// Spec gap: Annex C does not say how the bits of a frame are shared
    /// between its elements, any more than between its channels (see
    /// [`ElementEncoder::quantize`]). Each element is given the fewest bits
    /// it can be coded in (see [`ElementEncoder::least_bits`]), and the
    /// bits beyond them are shared by the elements in proportion to the
    /// channels each codes with lines, so that each channel has about an
    /// equal share, as within an element. The elements are quantized one
    /// after another, each in its share of the bits the elements before it
    /// left, so that what one leaves goes to those quantized after it: a
    /// low frequency enhancement channel first, which sends a few lines
    /// and leaves most of its share, then the others in order of their
    /// perceptual entropy (see [`ElementEncoder::perceptual_entropy`]), the
    /// least first, as the channels of an element are.
    ///
    /// Carried forward alone, the bits an element leaves can come too late:
    /// an element quantized early in a share too small for any line is
    /// coded without lines, while one after it codes its own in the bits
    /// the first left, and the frame ends with bits unspent. In 4 channels
    /// at 48 kHz and 11 kbit/s with the CRC, the center channel carrying
    /// the same tone of 79 Hz as the rear surround so decoded to silence
    /// throughout, the rear surround at its level. An element with lines
    /// to code that its share left without any is therefore quantized
    /// again, once, after all the others, in its bits and those the frame
    /// has left, within its share of the buffer. With one element, the
    /// frame's bits are its share, and the element is not quantized again.
    fn quantize(&mut self, available_bits: u32) -> Vec<CodedElement> {
        let mut order: Vec<usize> = (0..self.elements.len()).collect();
        order.sort_by(|&a, &b| {
            let key = |e: usize| {
                let element = &self.elements[e];
                (
                    element.kind() != ElementKind::Lfe,
                    element.perceptual_entropy(),
                )
            };
            let (a, b) = (key(a), key(b));
            a.0.cmp(&b.0).then(a.1.total_cmp(&b.1))
        });
        let mut coded: Vec<Option<CodedElement>> = vec![None; self.elements.len()];
        let mut budgets = vec![0; self.elements.len()];
        let mut remaining = available_bits;
        for (k, &e) in order.iter().enumerate() {
            let rest = &order[k..];
            let element = &self.elements[e];
            let budget = if rest.len() == 1 {
                remaining
            } else {
                let least: u32 = rest.iter().map(|&i| self.elements[i].least_bits()).sum();
                let channels: u32 = (rest.iter())
                    .map(|&i| self.elements[i].coded_channels())
                    .sum();
                let beyond = u64::from(remaining.saturating_sub(least));
                let share = (beyond * u64::from(element.coded_channels()))
                    .checked_div(u64::from(channels))
                    .unwrap_or(0);
                element.least_bits() + share as u32
            };
            let budget =
                reservoir::element_max_bits(element.kind()).map_or(budget, |max| budget.min(max));
            let element = self.elements[e].quantize(budget);
            remaining = remaining.saturating_sub(element.bits());
            budgets[e] = budget;
            coded[e] = Some(element);
        }
        // An element with lines to code that its share left without any is
        // quantized again, once, with the bits the frame has left.
        for &e in &order {
            let element = coded[e].as_ref().expect("quantized");
            let silenced = self.elements[e].coded_channels() > 0
                && (element.channels().iter()).all(|c| c.lines().iter().all(|&q| q == 0));
            let budget = element.bits() + remaining;
            let budget =
                reservoir::element_max_bits(element.kind()).map_or(budget, |max| budget.min(max));
            if silenced && budget > budgets[e] {
                remaining += element.bits();
                let element = self.elements[e].quantize(budget);
                remaining = remaining.saturating_sub(element.bits());
                coded[e] = Some(element);
            }
        }
        coded.into_iter().map(|e| e.expect("quantized")).collect()
    }
}

/// A stream of `frames` silent frames, each apart, for tests: the
/// frames of as many blocks of silence, and of the block after them.
#[cfg(any(test, feature = "unstable"))]
pub fn encode_silence(config: StreamConfig, frames: usize) -> Vec<Vec<u8>> {
    let silence = vec![0.0; (frames + 1) * FRAME_SAMPLES];
    let channels = vec![silence.as_slice(); config.channel_configuration.channels()];
    let mut encoder = StreamEncoder::new(config);
    let mut out = Vec::new();
    encoder.encode(&channels, &mut |frame: &[u8]| out.push(frame.to_vec()));
    assert_eq!(out.len(), frames);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(framing: Framing, bits_per_second: u32) -> StreamConfig {
        StreamConfig {
            sampling_frequency: SamplingFrequency::Hz44100,
            channel_configuration: ChannelConfiguration::Stereo,
            bits_per_second,
            framing,
            flags: Flags::default(),
            crc_protected: false,
            noise_substitution: false,
        }
    }

    #[test]
    fn frames_keep_the_rate_once_the_reservoir_is_full() {
        for framing in [Framing::Adts, Framing::Raw] {
            let config = config(framing, 128_000);
            let frames = encode_silence(config, 200);
            let total: usize = frames.iter().map(Vec::len).sum();
            // What the rate brings, less what the reservoir holds at most.
            let mean_total = 200 * 128_000 * 1024 / 44_100 / 8;
            let max_bit_reservoir = 2 * 6144 - 128_000 * 1024 / 44_100;
            assert!(total <= mean_total, "{framing:?}");
            assert!(
                total + max_bit_reservoir / 8 + 1 >= mean_total,
                "{framing:?}"
            );
            // The first frames are the shortest, filling the reservoir: a
            // pair of silent channels that share their window.
            let silent = CodedChannel::silent(Windows::LONG_SINE, config.sampling_frequency);
            let element = CodedElement::common_pair(silent.clone(), silent, MsMask::NONE);
            let shortest = config.header_len()
                + raw_data_block::min_len(config.channel_configuration, &[element], 0);
            assert_eq!(frames[0].len(), shortest, "{framing:?}");
            assert_eq!(shortest, config.shortest_frame_len(Windows::LONG_SINE, 0));
        }
    }

    #[test]
    fn the_crc_raises_the_lowest_bitrate_by_its_bits() {
        // The shortest frame of a pair that shares its window, 13 bytes of
        // which 7 are the header, and with the CRC 15, at 44.1 kHz,
        // rounded up.
        let mut config = config(Framing::Adts, 0);
        assert_eq!(config.min_bits_per_second(), 4479);
        config.crc_protected = true;
        assert_eq!(config.min_bits_per_second(), 5168);
    }

    #[test]
    fn adts_frames_give_their_length_and_the_reservoir() {
        let mut config = config(Framing::Adts, 128_000);
        config.crc_protected = true;
        let frames = encode_silence(config, 200);
        let mut reservoir = config.reservoir().unwrap();
        for frame in &frames {
            reservoir.spend(frame.len());
            let length = (usize::from(frame[3] & 0b11) << 11)
                | (usize::from(frame[4]) << 3)
                | usize::from(frame[5] >> 5);
            let fullness = (u32::from(frame[5] & 0b1_1111) << 6) | u32::from(frame[6] >> 2);
            assert_eq!(length, frame.len());
            assert_eq!(fullness, reservoir.adts_buffer_fullness());
        }
        assert!(frames.last().unwrap().len() > config.shortest_frame_len(Windows::LONG_SINE, 0));
    }

    #[test]
    fn the_bitrate_bounds_admit_every_frame() {
        for fs in SamplingFrequency::ALL {
            for channel_configuration in [ChannelConfiguration::Mono, ChannelConfiguration::Stereo]
            {
                for (framing, crc_protected) in [
                    (Framing::Adts, false),
                    (Framing::Adts, true),
                    (Framing::Raw, false),
                ] {
                    let mut config = StreamConfig {
                        sampling_frequency: fs,
                        channel_configuration,
                        bits_per_second: 0,
                        framing,
                        flags: Flags::default(),
                        crc_protected,
                        noise_substitution: false,
                    };
                    let (min, max) = (config.min_bits_per_second(), config.max_bits_per_second());
                    for bits_per_second in [min - 1, min, max, max + 1] {
                        config.bits_per_second = bits_per_second;
                        let valid = (min..=max).contains(&bits_per_second);
                        assert_eq!(config.is_valid(), valid, "{config:?}");
                        if valid {
                            encode_silence(config, 50);
                        }
                    }
                }
            }
        }
    }

    /// A click after silence is coded with short windows, but for an
    /// encoder without block switching.
    #[test]
    fn block_switching_can_be_turned_off() {
        let config = config(Framing::Raw, 64_000);
        let mut input = vec![0.0; 8 * FRAME_SAMPLES];
        input[5 * FRAME_SAMPLES + 300] = 0.9;
        for switching in [true, false] {
            let mut encoder = StreamEncoder::new(config);
            if !switching {
                encoder = encoder.without_block_switching();
            }
            let mut sequences = Vec::new();
            encoder.encode(&[&input, &input], &mut |frame: &[u8]| {
                // id_syn_ele, element_instance_tag, common_window 1,
                // ics_reserved_bit, then window_sequence.
                assert_eq!(frame[0] & 1, 1, "common_window");
                sequences.push((frame[1] >> 5) & 0b11);
            });
            assert_eq!(sequences.len(), 7);
            assert_eq!(sequences.contains(&2), switching, "{sequences:?}");
        }
    }

    /// At the lowest bitrate, which is set by frames of long windows, a
    /// frame cannot afford the side information of eight short windows,
    /// and an attack is coded with long ones; a little above it, it is
    /// coded short.
    #[test]
    fn tns_can_be_turned_off() {
        use crate::noiseless::decoder::{BitReader, element};
        let config = config(Framing::Raw, 192_000);
        // A click in each of the frames, coded with long windows, in bits
        // enough to code some of its lines.
        let mut input = vec![0.0; 8 * FRAME_SAMPLES];
        for frame in 0..8 {
            input[frame * FRAME_SAMPLES + 300] = 0.9;
        }
        for tns in [true, false] {
            let mut encoder = StreamEncoder::new(config).without_block_switching();
            if !tns {
                encoder = encoder.without_tns();
            }
            let mut shaped = 0;
            encoder.encode(&[&input, &input], &mut |frame: &[u8]| {
                let mut r = BitReader::new(frame);
                let element = element(&mut r, SamplingFrequency::Hz44100);
                shaped += element.channels.iter().filter(|c| c.tns.is_some()).count();
            });
            assert_eq!(shaped > 0, tns, "{shaped}");
        }
    }

    #[test]
    fn spectrum_clipping_is_off_unless_asked_for() {
        use crate::noiseless::decoder::{BitReader, element};
        let config = config(Framing::Raw, 128_000);
        // Noise falling by 6 dB an octave, whose highest lines quantize to
        // few above 1.
        let (mut state, mut y) = (1u64, 0.0);
        let input: Vec<f64> = (0..24 * FRAME_SAMPLES)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                y = 0.98 * y + ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5);
                0.2 * y
            })
            .collect();
        let right: Vec<f64> = input.iter().map(|x| -0.7 * x).collect();
        for on in [None, Some(false), Some(true)] {
            let mut encoder = StreamEncoder::new(config);
            if let Some(on) = on {
                encoder = encoder.spectrum_clipping(on);
            }
            let mut pulses = 0;
            encoder.encode(&[&input, &right], &mut |frame: &[u8]| {
                let mut r = BitReader::new(frame);
                let element = element(&mut r, SamplingFrequency::Hz44100);
                pulses += element
                    .channels
                    .iter()
                    .filter(|c| c.pulses.is_some())
                    .count();
            });
            assert_eq!(pulses > 0, on == Some(true), "{on:?}: {pulses}");
        }
    }

    #[test]
    fn the_blocks_past_the_input_are_not_shaped() {
        use crate::noiseless::decoder::{BitReader, element};
        let config = config(Framing::Raw, 192_000);
        // A click in each block, the last block half filled.
        let len = 5 * FRAME_SAMPLES + FRAME_SAMPLES / 2;
        let mut input = vec![0.0; len];
        for at in (300..len).step_by(FRAME_SAMPLES) {
            input[at] = 0.9;
        }
        let mut encoder = StreamEncoder::new(config).without_block_switching();
        let mut shaped = Vec::new();
        let mut sink = |frame: &[u8]| {
            let mut r = BitReader::new(frame);
            let element = element(&mut r, SamplingFrequency::Hz44100);
            shaped.push(element.channels.iter().any(|c| c.tns.is_some()));
        };
        encoder.encode(&[&input, &input], &mut sink);
        let end = encoder.finish(&mut sink);
        assert_eq!(end.frames, 7);
        // Blocks 5, the input's last, and 6, after it, hold padding.
        assert!(shaped[1..5].iter().any(|&shaped| shaped), "{shaped:?}");
        assert_eq!(shaped[5..], [false, false]);
    }

    #[test]
    fn a_frame_without_bits_for_short_windows_stays_long() {
        let mut config = StreamConfig {
            sampling_frequency: SamplingFrequency::Hz44100,
            channel_configuration: ChannelConfiguration::Mono,
            bits_per_second: 0,
            framing: Framing::Adts,
            flags: Flags::default(),
            crc_protected: false,
            noise_substitution: false,
        };
        let min = config.min_bits_per_second();
        let mut input = vec![0.0; 12 * FRAME_SAMPLES];
        for click in [3 * FRAME_SAMPLES + 300, 8 * FRAME_SAMPLES + 900] {
            input[click] = 0.9;
        }
        for (bits_per_second, short) in [(min, false), (min + 1000, true)] {
            config.bits_per_second = bits_per_second;
            let mut sequences = Vec::new();
            let mut encoder = StreamEncoder::new(config);
            encoder.encode(&[&input], &mut |frame: &[u8]| {
                // After the header, id_syn_ele, element_instance_tag,
                // global_gain and ics_reserved_bit.
                sequences.push((frame[9] >> 6) & 0b11);
            });
            assert_eq!(
                sequences.contains(&2),
                short,
                "{bits_per_second}: {sequences:?}"
            );
            assert!(sequences.contains(&1), "{bits_per_second}: {sequences:?}");
        }
    }

    #[test]
    fn noise_after_silence_borrows_from_the_reservoir() {
        let config = StreamConfig {
            sampling_frequency: SamplingFrequency::Hz44100,
            channel_configuration: ChannelConfiguration::Mono,
            bits_per_second: 8_000,
            framing: Framing::Adts,
            flags: Flags::default(),
            crc_protected: false,
            noise_substitution: false,
        };
        // Silence, which takes the shortest frames and fills the
        // reservoir, then white noise, whose perceptual entropy asks for
        // more than the mean.
        let mut input = vec![0.0; 40 * FRAME_SAMPLES];
        let mut state = 3u64;
        input.extend((0..10 * FRAME_SAMPLES).map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            0.25 * ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
        }));
        let mut encoder = StreamEncoder::new(config);
        let mut frames = Vec::new();
        encoder.encode(&[&input], &mut |frame: &[u8]| frames.push(frame.len()));
        let mean = 8_000.0 * FRAME_SAMPLES as f64 / 44_100.0 / 8.0;
        let longest = frames[40..].iter().copied().max().unwrap();
        assert!(
            longest as f64 > 1.5 * mean,
            "{longest} bytes against {mean}"
        );
    }

    #[test]
    fn the_bits_frames_leave_come_back_to_their_audio() {
        use crate::raw_data_block::decoder::{adts_frame, raw_data_block};

        let config = config(Framing::Adts, 128_000);
        let reservoir = config.reservoir().unwrap();
        // Noise falling with frequency, whose perceptual entropy asks for no
        // bits beyond the mean, and which takes all it is given, but for
        // what the loops leave.
        let red = |seed| {
            let mut y = 0.0;
            (noise(301 * FRAME_SAMPLES, seed).into_iter())
                .map(|x| {
                    y = 0.99 * y + 0.1 * x;
                    y
                })
                .collect::<Vec<f64>>()
        };
        let (left, right) = (red(5), red(6));
        let mut encoder = StreamEncoder::new(config);
        let mut stream = Vec::new();
        encoder.encode(&[&left, &right], &mut stream);
        let (mut frame_bits, mut element_bits, mut fullness) = (0, 0, Vec::new());
        let mut rest = stream.as_slice();
        for k in 0..300 {
            let frame = adts_frame(rest);
            rest = &rest[frame.aac_frame_length..];
            if k >= 100 {
                let block = raw_data_block(frame.raw_data_block, config.sampling_frequency);
                frame_bits += 8 * frame.aac_frame_length;
                element_bits += block.elements[0].bits.len();
                fullness.push(frame.adts_buffer_fullness);
            }
        }
        // The audio takes all but the header, the element's id_syn_ele and
        // ID_END, 62 of the 2972 bits of a mean frame, and a few the loops
        // leave; filled out to the mean, it took 92.2%.
        let share = element_bits as f64 / frame_bits as f64;
        assert!(share > 0.975, "{share}");
        // The reservoir is never full, which would have frames filled out,
        // and keeps what lies within half a mean frame of its maximum, for
        // frames that ask for more.
        let mean = 128_000 * FRAME_SAMPLES as u64 / 44_100;
        let full = (reservoir.max_bits() / 2 / 32) as u32;
        let kept = ((reservoir.max_bits() - mean / 2) / 2 / 32) as u32;
        assert!(
            fullness.iter().all(|&f| (kept..full).contains(&f)),
            "{fullness:?}"
        );
    }

    /// A mono stream at 44.1 kHz whose reservoir is as large as a mean
    /// frame, 3072 bits, with the CRC and data streams that leave its audio
    /// a few bits a frame: 38 blocks of silence fill the reservoir, then
    /// white noise sets in, with a second attack after a gap in block 44.
    /// Neither the first attack nor the frames after it take all that the
    /// reservoir holds, which leaves the second bits to borrow: every frame
    /// codes lines, the second attack's in eight short windows, in more
    /// than the 436 bits it took before frames were offered any.
    #[test]
    fn an_attack_finds_bits_in_the_reservoir_after_another() {
        use crate::raw_data_block::decoder::{adts_frame, raw_data_block};

        let fs = SamplingFrequency::Hz44100;
        let config = StreamConfig {
            sampling_frequency: fs,
            channel_configuration: ChannelConfiguration::Mono,
            bits_per_second: 132_300,
            framing: Framing::Adts,
            flags: Flags::default(),
            crc_protected: true,
            noise_substitution: false,
        };
        let mut input = vec![0.0; 38 * FRAME_SAMPLES];
        let gap = 6 * FRAME_SAMPLES..6 * FRAME_SAMPLES + 600;
        let noise = noise(12 * FRAME_SAMPLES, 3);
        input.extend(
            (noise.iter().enumerate()).map(|(t, &x)| if gap.contains(&t) { 0.0 } else { x }),
        );
        let mut encoder = StreamEncoder::new(config).with_data_stream(368);
        let mut stream = Vec::new();
        encoder.encode(&[&input[..38 * FRAME_SAMPLES]], &mut stream);
        encoder.push_data_stream(&[0x33; 368 * 60]);
        encoder.encode(&[&input[38 * FRAME_SAMPLES..]], &mut stream);
        let mut rest = stream.as_slice();
        // The frames from the first attack's on.
        let frames: Vec<_> = (0..49)
            .map(|_| {
                let frame = adts_frame(rest);
                rest = &rest[frame.aac_frame_length..];
                raw_data_block(frame.raw_data_block, fs).elements[0].clone()
            })
            .skip(38)
            .collect();
        for (k, element) in frames.iter().enumerate() {
            let lines = &element.element.channels[0].lines;
            assert!(lines.iter().flatten().any(|&q| q != 0), "frame {}", 38 + k);
        }
        let second = &frames[45 - 38];
        assert_eq!(second.element.channels[0].info.window_sequence, 2);
        assert!(second.bits.len() > 436, "{}", second.bits.len());
    }

    /// A mono stream at 44.1 kHz and 48 kbit/s, with the CRC and 124 bytes
    /// of data streams a frame, whose mean frame does not pay for eight
    /// short windows, codes an attack with them once the reservoir offers
    /// the bits.
    #[test]
    fn an_attack_is_coded_short_in_the_bits_the_reservoir_offers() {
        use crate::raw_data_block::decoder::{adts_frame, raw_data_block};

        let fs = SamplingFrequency::Hz44100;
        let config = StreamConfig {
            sampling_frequency: fs,
            channel_configuration: ChannelConfiguration::Mono,
            bits_per_second: 48_000,
            framing: Framing::Adts,
            flags: Flags::default(),
            crc_protected: true,
            noise_substitution: false,
        };
        let mut encoder = StreamEncoder::new(config).with_data_stream(124);
        let data_bits = data_stream::bits(124);
        assert!(!encoder.affords_short(data_bits));
        // Silence, without data, fills the reservoir; then noise sets in
        // within block 40, with data in every frame.
        let mut input = vec![0.0; 40 * FRAME_SAMPLES + 512];
        input.extend(noise(4 * FRAME_SAMPLES, 7));
        let mut stream = Vec::new();
        encoder.encode(&[&input[..36 * FRAME_SAMPLES]], &mut stream);
        encoder.push_data_stream(&[0x5A; 124 * 50]);
        encoder.encode(&[&input[36 * FRAME_SAMPLES..]], &mut stream);
        let mut rest = stream.as_slice();
        let sequences: Vec<u32> = (0..43)
            .map(|k| {
                let frame = adts_frame(rest);
                rest = &rest[frame.aac_frame_length..];
                let block = raw_data_block(frame.raw_data_block, fs);
                let carried: usize = block.data_streams.iter().map(|d| d.bytes.len()).sum();
                assert!(k < 36 || carried == 124, "{k}: {carried}");
                u32::from(block.elements[0].element.channels[0].info.window_sequence)
            })
            .collect();
        // Block 40's frame is coded short, between a start and a stop.
        assert_eq!(sequences[39..42], [1, 2, 3], "{sequences:?}");
        assert!(sequences[..39].iter().all(|&s| s == 0), "{sequences:?}");
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

    /// Frames carry the bytes pushed in order, each up to its share, in
    /// raw frames and in ADTS, those pushed after the frames were coded
    /// among them, and the rest are dropped at the end; the stream keeps
    /// its rate.
    #[test]
    fn data_streams_carry_the_bytes_pushed_in_order() {
        use crate::raw_data_block::decoder::{adts_frame, raw_data_block};

        let fs = SamplingFrequency::Hz44100;
        let input = noise(12 * FRAME_SAMPLES, 5);
        let data: Vec<u8> = (0..20_000u32).map(|i| (i * 7 + i / 255) as u8).collect();
        // At 320 kbit/s, two elements a frame.
        for (framing, bits_per_second, per_frame) in
            [(Framing::Raw, 128_000, 100), (Framing::Adts, 320_000, 600)]
        {
            let config = config(framing, bits_per_second);
            let mut encoder = StreamEncoder::new(config).with_data_stream(per_frame);
            let mut frames = Vec::new();
            let mut sink = |frame: &[u8]| frames.push(frame.to_vec());
            let (first, second) = data.split_at(300);
            encoder.push_data_stream(first);
            encoder.encode(&[&input[..6 * FRAME_SAMPLES]; 2], &mut sink);
            encoder.push_data_stream(second);
            encoder.encode(&[&input[6 * FRAME_SAMPLES..]; 2], &mut sink);
            let end = encoder.finish(&mut sink);
            // Each frame's bytes, from its data stream elements.
            let carried: Vec<Vec<u8>> = (frames.iter())
                .map(|frame| {
                    let block = match framing {
                        Framing::Adts => adts_frame(frame).raw_data_block,
                        Framing::Raw => frame,
                    };
                    (raw_data_block(block, fs).data_streams.iter())
                        .flat_map(|stream| stream.bytes.clone())
                        .collect()
                })
                .collect();
            // The first five frames, coded before the second push, carry
            // the first push; the rest carry their share.
            let lens: Vec<usize> = carried.iter().map(Vec::len).collect();
            let first_lens: Vec<usize> = (0..5)
                .map(|k| 300usize.saturating_sub(k * per_frame).min(per_frame))
                .collect();
            assert_eq!(lens[..5], first_lens, "{framing:?}");
            assert!(lens[5..].iter().all(|&len| len == per_frame), "{lens:?}");
            let carried = carried.concat();
            assert_eq!(carried, data[..carried.len()], "{framing:?}");
            assert_eq!(carried.len() + end.data_stream_dropped, data.len());
            let lens: Vec<usize> = frames.iter().map(Vec::len).collect();
            let total: usize = lens.iter().sum();
            let mean_total = lens.len() * bits_per_second as usize * 1024 / 44_100 / 8;
            assert!(total <= mean_total, "{framing:?}");
        }
    }

    /// At the lowest bitrates and above, a frame of noise or silence
    /// carries the most bytes of data streams allowed, and one more is not.
    #[test]
    fn the_most_data_fits_every_frame() {
        let input = noise(10 * FRAME_SAMPLES, 9);
        for fs in [SamplingFrequency::Hz8000, SamplingFrequency::Hz48000] {
            for channel_configuration in [ChannelConfiguration::Mono, ChannelConfiguration::Stereo]
            {
                let mut config = StreamConfig {
                    sampling_frequency: fs,
                    channel_configuration,
                    bits_per_second: 0,
                    framing: Framing::Adts,
                    flags: Flags::default(),
                    crc_protected: true,
                    noise_substitution: false,
                };
                let min = config.min_bits_per_second();
                assert_eq!(config.max_data_stream_bytes(), 0);
                for bits_per_second in [min, min + 200, min + 5_000, 4 * min + 30_000] {
                    config.bits_per_second = bits_per_second;
                    let max = config.max_data_stream_bytes();
                    // The shortest frame with them is within the mean, and
                    // with one more byte beyond it.
                    let mean = bits_per_second as usize * FRAME_SAMPLES / (8 * fs.hz() as usize);
                    assert!(config.shortest_frame_len(Windows::LONG_SINE, max) <= mean);
                    assert!(config.shortest_frame_len(Windows::LONG_SINE, max + 1) > mean);
                    let mut encoder = StreamEncoder::new(config).with_data_stream(max);
                    encoder.push_data_stream(&vec![0xFF; 20 * max]);
                    let channels = vec![input.as_slice(); channel_configuration.channels()];
                    let mut stream = Vec::new();
                    encoder.encode(&channels, &mut stream);
                    let end = encoder.finish(&mut stream);
                    assert_eq!(end.data_stream_dropped, (20 - end.frames as usize) * max);
                }
            }
        }
    }

    /// A pair at 48 kHz and 96 kbit/s, with the CRC, carries 239 bytes in
    /// every frame of noise: as many as its shortest frame, of a common
    /// window, leaves of the 256 a frame the rate brings.
    #[test]
    fn a_pair_carries_what_its_common_window_leaves() {
        use crate::raw_data_block::decoder::{adts_frame, raw_data_block};

        let mut config = config(Framing::Adts, 96_000);
        config.sampling_frequency = SamplingFrequency::Hz48000;
        config.crc_protected = true;
        assert_eq!(config.max_data_stream_bytes(), 239);
        let input = noise(31 * FRAME_SAMPLES, 11);
        let mut encoder = StreamEncoder::new(config).with_data_stream(239);
        encoder.push_data_stream(&[0x5A; 32 * 239]);
        let mut stream = Vec::new();
        encoder.encode(&[&input, &input], &mut stream);
        let end = encoder.finish(&mut stream);
        assert_eq!((end.frames, end.data_stream_dropped), (32, 0));
        let mut rest = stream.as_slice();
        while !rest.is_empty() {
            let frame = adts_frame(rest);
            rest = &rest[frame.aac_frame_length..];
            let block = raw_data_block(frame.raw_data_block, SamplingFrequency::Hz48000);
            assert!(block.elements[0].element.common_window);
            let carried: usize = block.data_streams.iter().map(|d| d.bytes.len()).sum();
            assert_eq!(carried, 239);
        }
    }

    /// The channel configurations of more than two channels.
    const MULTICHANNEL: [ChannelConfiguration; 5] = [
        ChannelConfiguration::Three,
        ChannelConfiguration::Four,
        ChannelConfiguration::Five,
        ChannelConfiguration::FiveOne,
        ChannelConfiguration::SevenOne,
    ];

    /// Each frame of a configuration holds its elements in order, each kind
    /// tagged from 0, and each channel's tone in its own place: the input's
    /// channels in the order of the elements that send them.
    #[test]
    fn each_configuration_sends_its_channels_in_its_elements_in_order() {
        use crate::raw_data_block::decoder::raw_data_block;
        use crate::raw_data_block::element_instance_tags;

        let fs = SamplingFrequency::Hz48000;
        for channel_configuration in MULTICHANNEL {
            let channels = channel_configuration.channels();
            let config = StreamConfig {
                sampling_frequency: fs,
                channel_configuration,
                bits_per_second: 96_000 * channel_configuration.ncc() as u32,
                framing: Framing::Raw,
                flags: Flags::default(),
                crc_protected: false,
                noise_substitution: false,
            };
            // Channel c a steady tone at the centre of line 40 + 37c, and a
            // low frequency enhancement channel, last, at line 5, among the
            // 12 it sends.
            let lfe = channel_configuration.elements().last() == Some(&ElementKind::Lfe);
            let line = |c: usize| match c {
                c if lfe && c == channels - 1 => 5,
                c => 40 + 37 * c,
            };
            let input: Vec<Vec<f64>> = (0..channels)
                .map(|c| {
                    let k = line(c) as f64 + 0.5;
                    (0..8 * FRAME_SAMPLES)
                        .map(|t| 0.3 * (std::f64::consts::PI * k * t as f64 / 1024.0).sin())
                        .collect()
                })
                .collect();
            let input: Vec<&[f64]> = input.iter().map(Vec::as_slice).collect();
            let mut frames = Vec::new();
            StreamEncoder::new(config).encode(&input, &mut |frame: &[u8]| {
                frames.push(frame.to_vec());
            });
            assert_eq!(frames.len(), 7);
            let kinds = channel_configuration.elements();
            for frame in &frames[2..] {
                let block = raw_data_block(frame, fs);
                let mut ids: Vec<u32> = kinds.iter().map(|k| k.id_syn_ele()).collect();
                ids.push(7);
                let sent: Vec<u32> = block.ids.iter().copied().filter(|&id| id != 6).collect();
                assert_eq!(sent, ids, "{channel_configuration:?}");
                let tags: Vec<u32> = (block.elements.iter())
                    .map(|e| e.element.element_instance_tag)
                    .collect();
                let expected: Vec<u32> = element_instance_tags(channel_configuration).collect();
                assert_eq!(tags, expected, "{channel_configuration:?}");
                let spectra: Vec<Vec<f64>> = (block.elements.iter())
                    .flat_map(|e| e.element.spectra(fs))
                    .collect();
                assert_eq!(spectra.len(), channels);
                // A tone at the centre of a line spreads to the lines beside
                // it with its phase.
                for (c, spectrum) in spectra.iter().enumerate() {
                    let energy = |lines: &[f64]| lines.iter().map(|x| x * x).sum::<f64>();
                    let near = energy(&spectrum[line(c) - 2..=line(c) + 2]);
                    assert!(
                        near > 0.99 * energy(spectrum),
                        "{channel_configuration:?}, channel {c}"
                    );
                }
            }
        }
    }

    /// A low frequency enhancement channel keeps to its restrictions
    /// (clause 8.4.1) whatever it is given: clicks over loud noise, which
    /// the other channels code with short windows and temporal noise
    /// shaping, it codes with long windows of the sine shape, unshaped, in
    /// its 12 lowest lines, which carry its tone.
    #[test]
    fn a_low_frequency_enhancement_channel_keeps_to_its_restrictions() {
        use crate::raw_data_block::decoder::raw_data_block;
        use crate::raw_data_block::element::LFE_LINES;

        let fs = SamplingFrequency::Hz48000;
        for (channel_configuration, clipping) in [
            (ChannelConfiguration::FiveOne, false),
            (ChannelConfiguration::SevenOne, true),
        ] {
            let channels = channel_configuration.channels();
            let config = StreamConfig {
                sampling_frequency: fs,
                channel_configuration,
                bits_per_second: 64_000 * channel_configuration.ncc() as u32,
                framing: Framing::Raw,
                flags: Flags::default(),
                crc_protected: false,
                noise_substitution: false,
            };
            let mut input = noise(10 * FRAME_SAMPLES, 13);
            for (t, x) in input.iter_mut().enumerate() {
                *x += 0.5 * (std::f64::consts::PI * 5.5 * t as f64 / 1024.0).sin();
                if t % 1500 == 700 {
                    *x += 0.9;
                }
            }
            let mut encoder = StreamEncoder::new(config).spectrum_clipping(clipping);
            let (mut short, mut shaped, mut lfe_frames) = (false, false, 0);
            encoder.encode(&vec![input.as_slice(); channels], &mut |frame: &[u8]| {
                let block = raw_data_block(frame, fs);
                for element in &block.elements {
                    for ics in &element.element.channels {
                        if element.id_syn_ele != 3 {
                            short |= ics.info.window_sequence == 2;
                            shaped |= ics.tns.is_some();
                            continue;
                        }
                        lfe_frames += 1;
                        assert_eq!((ics.info.window_sequence, ics.info.window_shape), (0, 0));
                        assert!(ics.tns.is_none());
                        assert!(ics.lines[0][LFE_LINES..].iter().all(|&q| q == 0));
                        assert!(ics.lines[0][4..8].iter().any(|&q| q != 0));
                    }
                }
            });
            assert_eq!(lfe_frames, 9, "{channel_configuration:?}");
            assert!(short && shaped, "{channel_configuration:?}");
        }
    }

    /// No element is longer than its share of the decoder's input buffer,
    /// 6144 bits a channel with its `id_syn_ele`, however many bits the
    /// frame has and however few of its elements ask for them: noise in
    /// one element of 5.1 and 7.1 beside silence, up to the highest
    /// bitrate.
    #[test]
    fn no_element_outgrows_its_buffer() {
        use crate::raw_data_block::decoder::raw_data_block;

        for (fs, channel_configuration, loud, bits_per_second) in [
            (
                SamplingFrequency::Hz48000,
                ChannelConfiguration::FiveOne,
                0,
                320_000,
            ),
            (
                SamplingFrequency::Hz48000,
                ChannelConfiguration::FiveOne,
                1,
                640_000,
            ),
            (
                SamplingFrequency::Hz96000,
                ChannelConfiguration::SevenOne,
                1,
                0,
            ),
        ] {
            let mut config = StreamConfig {
                sampling_frequency: fs,
                channel_configuration,
                bits_per_second,
                framing: Framing::Raw,
                flags: Flags::default(),
                crc_protected: false,
                noise_substitution: false,
            };
            if bits_per_second == 0 {
                config.bits_per_second = config.max_bits_per_second();
            }
            // Loud noise in the channels of element `loud`, silence in the
            // rest.
            let noise = noise(12 * FRAME_SAMPLES, 21)
                .iter()
                .map(|x| 3.0 * x)
                .collect::<Vec<_>>();
            let silence = vec![0.0; noise.len()];
            let mut channels = Vec::new();
            for (e, kind) in channel_configuration.elements().iter().enumerate() {
                let samples = if e == loud { &noise } else { &silence };
                channels.extend(std::iter::repeat_n(samples.as_slice(), kind.channels()));
            }
            let mut longest = 0;
            StreamEncoder::new(config).encode(&channels, &mut |frame: &[u8]| {
                for element in raw_data_block(frame, fs).elements {
                    let channels = element.element.channels.len();
                    let bits = 3 + element.bits.len();
                    if element.id_syn_ele != 3 {
                        assert!(bits <= 6144 * channels, "{channel_configuration:?}: {bits}");
                    }
                    longest = longest.max(bits);
                }
            });
            // The loud element takes all its buffer allows.
            let kind = channel_configuration.elements()[loud];
            assert!(
                longest > 6144 * kind.channels() - 100,
                "{channel_configuration:?}: {longest}"
            );
        }
    }

    /// An element that its share leaves without lines is quantized again
    /// with the bits the frame has left: in 4 channels at 48 kHz and 11
    /// kbit/s with the CRC, the center and rear surround carrying the same
    /// tone of 79 Hz, and the front pair one of 1.3 kHz, the center sends
    /// lines in every frame of the tone as the rear surround does, where
    /// it once sent none.
    #[test]
    fn an_element_left_without_lines_takes_the_bits_left() {
        use crate::raw_data_block::decoder::{adts_frame, raw_data_block};

        let fs = SamplingFrequency::Hz48000;
        let config = StreamConfig {
            sampling_frequency: fs,
            channel_configuration: ChannelConfiguration::Four,
            bits_per_second: 11_000,
            framing: Framing::Adts,
            flags: Flags::default(),
            crc_protected: true,
            noise_substitution: false,
        };
        let tone = |frequency: f64, amplitude: f64| -> Vec<f64> {
            (0..3 * 48_000)
                .map(|t| {
                    amplitude * (std::f64::consts::TAU * frequency * t as f64 / 48_000.0).sin()
                })
                .collect()
        };
        let (low, mid) = (tone(79.0, 0.8), tone(1300.0, 0.5));
        let mut stream = Vec::new();
        StreamEncoder::new(config).encode(&[&low, &mid, &mid, &low], &mut stream);
        let mut sending = Vec::new();
        let mut rest = stream.as_slice();
        while !rest.is_empty() {
            let frame = adts_frame(rest);
            rest = &rest[frame.aac_frame_length..];
            let block = raw_data_block(frame.raw_data_block, fs);
            let sends = |e: usize| {
                let lines = &block.elements[e].element.channels[0].lines;
                lines.iter().flatten().any(|&q| q != 0)
            };
            sending.push((sends(0), sends(2)));
        }
        assert!(sending.len() > 100);
        // From the third frame, the first of the tone throughout.
        assert!(
            sending[2..].iter().all(|&s| s == (true, true)),
            "{sending:?}"
        );
    }

    /// The bitrate of a configuration ranges from that of its shortest
    /// frame to that of its NCC's input buffer, or of 4608 bytes where that
    /// is less (see [`reservoir::MAX_ADTS_FRAME_LEN`]), and every frame of
    /// silence and noise fits it at both ends, with the CRC and without.
    #[test]
    fn every_configuration_codes_at_both_ends_of_its_range() {
        let fs = SamplingFrequency::Hz44100;
        for channel_configuration in MULTICHANNEL {
            let input = noise(4 * FRAME_SAMPLES, 3);
            for crc_protected in [false, true] {
                let mut config = StreamConfig {
                    sampling_frequency: fs,
                    channel_configuration,
                    bits_per_second: 0,
                    framing: Framing::Adts,
                    flags: Flags::default(),
                    crc_protected,
                    noise_substitution: false,
                };
                let max = config.max_bits_per_second();
                let ncc = channel_configuration.ncc() as u64;
                assert_eq!(max as u64, ((6144 * ncc).min(8 * 4608) - 8) * 44_100 / 1024);
                for bits_per_second in [config.min_bits_per_second(), max] {
                    config.bits_per_second = bits_per_second;
                    assert!(config.is_valid());
                    let channels = vec![input.as_slice(); channel_configuration.channels()];
                    let mut encoder = StreamEncoder::new(config);
                    let mut stream = Vec::new();
                    encoder.encode(&channels, &mut stream);
                    encoder.finish(&mut stream);
                }
            }
        }
    }

    /// No frame of 7.1 in ADTS is longer than 4608 bytes (see
    /// [`reservoir::MAX_ADTS_FRAME_LEN`]): of loud noise at the highest
    /// bitrate at 8, 48 and 96 kHz, nor of loud noise after the silence
    /// that fills the reservoir at 1700 kbit/s at 48 kHz, below the
    /// highest, 1727, whose raw frames, which keep the standard's limits,
    /// are longer. At lower bitrates, frames borrow too little from the
    /// reservoir to reach 4608 bytes.
    #[test]
    fn adts_frames_of_7_1_are_no_longer_than_4608_bytes() {
        use crate::raw_data_block::decoder::adts_frame;

        let configuration = ChannelConfiguration::SevenOne;
        let loud = |len| noise(len, 5).iter().map(|x| 3.0 * x).collect::<Vec<_>>();
        let mut silence_then_noise = vec![0.0; 8 * FRAME_SAMPLES];
        silence_then_noise.extend(loud(6 * FRAME_SAMPLES));
        let at_the_highest = loud(6 * FRAME_SAMPLES);
        for (fs, bits_per_second, input) in [
            (SamplingFrequency::Hz8000, 0, &at_the_highest),
            (SamplingFrequency::Hz48000, 0, &at_the_highest),
            (SamplingFrequency::Hz96000, 0, &at_the_highest),
            (SamplingFrequency::Hz48000, 1_700_000, &silence_then_noise),
        ] {
            let channels = vec![input.as_slice(); configuration.channels()];
            let mut longest = [0; 2];
            for (i, framing) in [Framing::Adts, Framing::Raw].into_iter().enumerate() {
                let mut config = StreamConfig {
                    sampling_frequency: fs,
                    channel_configuration: configuration,
                    bits_per_second,
                    framing,
                    flags: Flags::default(),
                    crc_protected: false,
                    noise_substitution: false,
                };
                if bits_per_second == 0 {
                    config.bits_per_second = StreamConfig {
                        framing: Framing::Adts,
                        ..config
                    }
                    .max_bits_per_second();
                }
                let mut encoder = StreamEncoder::new(config);
                let mut sink = |frame: &[u8]| {
                    let len = match framing {
                        Framing::Adts => adts_frame(frame).aac_frame_length,
                        Framing::Raw => frame.len(),
                    };
                    longest[i] = longest[i].max(len);
                };
                encoder.encode(&channels, &mut sink);
                encoder.finish(&mut sink);
            }
            let [adts, raw] = longest;
            assert!(adts <= 4608, "{fs:?} {bits_per_second}: {adts}");
            if bits_per_second != 0 {
                assert!(raw > 4608, "{fs:?} {bits_per_second}: {raw}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "bytes of data streams a frame")]
    fn a_share_beyond_the_most_is_refused() {
        let config = config(Framing::Raw, 64_000);
        StreamEncoder::new(config).with_data_stream(config.max_data_stream_bytes() + 1);
    }
}
