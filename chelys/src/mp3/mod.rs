// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! MPEG-1 Audio Layer III, and its extension to lower sampling frequencies in
//! ISO/IEC 13818-3.

mod ancillary;
mod crc;
pub mod filterbank;
pub mod header;
pub mod huffman;
pub mod info_tag;
pub mod loops;
pub mod psychoacoustic;
pub mod rate;
pub mod reservoir;
pub mod scalefactor_bands;
pub mod scalefactors;
pub mod side_info;
pub mod stereo;
#[cfg(test)]
mod stream_check;

use std::collections::VecDeque;
use std::ops::Range;

use crate::bitstream::BitWriter;
use ancillary::AncillaryQueue;
use header::{Bitrate, Emphasis, Flags, Header, Mode, PaddingSchedule, SamplingFrequency, Version};
use huffman::{QuadTable, SpectrumCoding};
use loops::scfsi::{self, GranuleSummary};
use psychoacoustic::switching::{BlockSwitching, SwitchedMasking};
use rate::{MeanRateSchedule, Rate};
use reservoir::Reservoir;
use scalefactors::Scalefactors;
use side_info::{GranuleChannel, SideInfo, Window};

/// Frequency lines in a granule (clause 2.1.70).
pub const GRANULE_LINES: usize = 576;

/// The PCM a [`FrameEncoder`] needs past a frame before coding it: two
/// granules, as a granule's block type depends on whether the granule after
/// it has an attack, and the filterbank needs the block type of the granule
/// after the one it filters.
const LOOK_AHEAD: usize = 2 * GRANULE_LINES;

/// The samples by which a decoder's output lags the encoder's input: the
/// delays of the analysis and synthesis polyphase filterbanks and of the
/// overlapping blocks of the hybrid filterbank, together. An [`Encoder`]
/// ends the stream at least this far past the input, so that the decoded
/// output holds all of it.
///
/// Spec gap: the standard does not give the delay. Settled by decoding
/// with mpg123 and LAME (black box): given an encoder delay of this less
/// their own (see [`info_tag`]), both give the input sample for sample; the
/// conformance test is `decoders_read_the_tag_at_every_sampling_frequency`.
/// ffmpeg's output, untrimmed, lags the input by as much.
pub const DELAY: usize = 1057;

/// The window and transform of a granule's block (clause C.1.5.3.3, figure
/// C.7): `block_type` 0 to 3 of clause 2.4.2.7, and `block_type` 2 with
/// `mixed_block_flag` set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockType {
    Normal,
    Start,
    Short,
    /// A short block whose two lowest subbands are transformed with the
    /// normal window, and sent with long scalefactor bands (clause 2.4.2.7):
    /// `block_type` 2 with `mixed_block_flag` set. It takes the place of a
    /// short block among the others.
    Mixed,
    Stop,
}

impl BlockType {
    /// Whether the granule is `block_type` 2, with short windows in all its
    /// subbands or all but the two lowest.
    pub fn short_windows(self) -> bool {
        matches!(self, Self::Short | Self::Mixed)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreamConfig {
    pub rate: Rate,
    pub sampling_frequency: SamplingFrequency,
    pub mode: Mode,
    pub crc_protected: bool,
    /// The `private_bit`, `copyright` and `original/copy` of every frame.
    pub flags: Flags,
}

/// Appends one frame (clause 2.4.1.2): the header, the `crc_check` word if
/// the header asks for one, the side information, and then `main_data`
/// zero-filled to the frame length. The main data starts in this frame, so
/// `side_info.main_data_begin` should be 0.
///
/// # Panics
///
/// If `main_data` does not fit in the frame.
pub fn write_frame(header: &Header, side_info: &SideInfo, main_data: &[u8], out: &mut Vec<u8>) {
    let slot = write_frame_start(header, side_info, out);
    assert!(
        main_data.len() <= slot.len(),
        "{} bytes of main data exceed the {} available in the frame",
        main_data.len(),
        slot.len(),
    );
    out[slot.start..slot.start + main_data.len()].copy_from_slice(main_data);
}

/// Appends a frame's header, `crc_check` word and side information, and
/// zeros for the rest of the frame, whose range in `out` it returns: the
/// frame's main data slot.
fn write_frame_start(header: &Header, side_info: &SideInfo, out: &mut Vec<u8>) -> Range<usize> {
    let start = out.len();
    let mut w = BitWriter::new();
    header.write(&mut w);
    side_info.write(header.mode.channels(), &mut w);
    let bytes = w.into_bytes();
    let (header_bytes, side_info_bytes) = bytes.split_at(Header::LEN_BYTES);

    out.extend_from_slice(header_bytes);
    if header.crc_protected {
        // Table B.5: bits 16..31 of the header and all of the side information.
        let protected = header_bytes[2..].iter().chain(side_info_bytes).copied();
        out.extend_from_slice(&crc::crc16(protected).to_be_bytes());
    }
    out.extend_from_slice(side_info_bytes);

    let slot = out.len()..start + header.frame_len();
    out.resize(slot.end, 0);
    slot
}

/// Writes frames whose main data may start in earlier frames (clause
/// 2.4.3.4.3, figure A.7). The main data of all frames forms one stream of
/// bytes, laid out over the frames' main data slots in order, and each
/// frame's `main_data_begin` counts back into the slot bytes before its
/// header that earlier frames left unused.
#[derive(Debug, Default)]
struct FrameWriter {
    out: Vec<u8>,
    /// The positions in `out` of slot bytes no main data has used, in stream
    /// order.
    unused: VecDeque<usize>,
}

impl FrameWriter {
    /// # Panics
    ///
    /// If fewer unused bytes than `main_data_begin` precede the frame, or
    /// `main_data` does not fit in them and the frame's slot.
    fn write(&mut self, header: &Header, side_info: &SideInfo, main_data: &[u8]) {
        let begin = usize::from(side_info.main_data_begin);
        assert!(
            begin <= self.unused.len(),
            "main_data_begin {begin} reaches past the {} unused bytes before the frame",
            self.unused.len()
        );
        // Unused bytes further back stay zero: stuffing (clause C.1.5.4.2.2).
        self.unused.drain(..self.unused.len() - begin);
        let slot = write_frame_start(header, side_info, &mut self.out);
        self.unused.extend(slot);
        assert!(
            main_data.len() <= self.unused.len(),
            "{} bytes of main data exceed the {} available to the frame",
            main_data.len(),
            self.unused.len()
        );
        for &byte in main_data {
            let at = self.unused.pop_front().expect("checked above");
            self.out[at] = byte;
        }
    }

    /// Keeps only the last `main_data_begin` unused bytes, those the next
    /// frame's main data begins in; those before them stay zero.
    fn keep_unused(&mut self, main_data_begin: usize) {
        let keep = main_data_begin.min(self.unused.len());
        self.unused.drain(..self.unused.len() - keep);
    }

    /// Moves to `out` the bytes before the first that later main data may
    /// still fill.
    fn take_final(&mut self, out: &mut Vec<u8>) {
        let end = self.unused.front().copied().unwrap_or(self.out.len());
        out.extend(self.out.drain(..end));
        for at in &mut self.unused {
            *at -= end;
        }
    }

    /// Moves the rest of the stream to `out`, with no frame to follow.
    fn finish(self, out: &mut Vec<u8>) {
        out.extend(self.out);
    }
}

/// One granule of one channel, quantised and ready for coding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantizedGranule {
    pub block_type: BlockType,
    /// The quantizer step size; see the requantization formula in clause
    /// 2.4.3.4.7.1.
    pub global_gain: u8,
    pub scalefactors: Scalefactors,
    /// The quantised values in the order they are sent: by frequency, or for
    /// short blocks by scalefactor band, window and frequency (clause
    /// 2.4.2.7), and for mixed blocks so above their two lowest subbands.
    pub values: [i32; GRANULE_LINES],
    pub coding: SpectrumCoding,
}

impl QuantizedGranule {
    /// The main data bits of the granule, `part2_3_length`: its scalefactors
    /// and Huffman code bits.
    ///
    /// # Panics
    ///
    /// If a scalefactor is above its limit.
    pub fn bit_len(&self, sampling_frequency: SamplingFrequency) -> usize {
        let compress = self
            .scalefactors
            .compress()
            .expect("scalefactors within their limits");
        let bands = scalefactor_bands::coding_bands(self.block_type, sampling_frequency);
        self.scalefactors.part2_length(compress) + self.coding.bit_count(&self.values, &bands)
    }

    /// Partitions `values` of a granule of `block_type` and chooses Huffman
    /// tables for them (clause C.1.5.4.4). The scalefactors are all zero.
    pub fn new(
        global_gain: u8,
        values: [i32; GRANULE_LINES],
        block_type: BlockType,
        sampling_frequency: SamplingFrequency,
    ) -> Self {
        let bands = scalefactor_bands::coding_bands(block_type, sampling_frequency);
        let coding = SpectrumCoding::choose(&values, block_type, &bands, sampling_frequency);
        Self {
            block_type,
            global_gain,
            scalefactors: Scalefactors::new(block_type, sampling_frequency.version()),
            values,
            coding,
        }
    }

    /// The side information's `window_switching_flag` and the fields that
    /// depend on it.
    ///
    /// Spec gap: Annex C does not say how to choose `subblock_gain`, which
    /// sets a coarser step for one short window. It is always 0: the outer
    /// loop's scalefactors shape the noise of each window instead.
    ///
    fn window(&self) -> Window {
        let coding = &self.coding;
        let block_type = match self.block_type {
            BlockType::Normal => {
                return Window::Normal {
                    table_select: coding.table_select,
                    region0_count: coding.region0_count,
                    region1_count: coding.region1_count,
                };
            }
            BlockType::Start => side_info::BlockType::Start,
            BlockType::Short | BlockType::Mixed => side_info::BlockType::Short,
            BlockType::Stop => side_info::BlockType::End,
        };
        Window::Switched {
            block_type,
            mixed_block_flag: self.block_type == BlockType::Mixed,
            table_select: [coding.table_select[0], coding.table_select[1]],
            subblock_gain: [0; 3],
        }
    }
}

/// The largest `part2_3_length`, a 12-bit field (clause 2.4.1.7).
const MAX_PART2_3_LENGTH: usize = (1 << 12) - 1;

impl StreamConfig {
    /// The bitrate of every frame.
    ///
    /// # Panics
    ///
    /// If the rate is not [`Rate::Constant`].
    fn constant_bitrate(&self) -> Bitrate {
        match self.rate {
            Rate::Constant(bitrate) => bitrate,
            rate => panic!("{rate:?} is not a constant bitrate"),
        }
    }

    /// Whether the configured mode allows MS stereo and intensity stereo:
    /// joint stereo with `ms_stereo` or `intensity_stereo` set, in that order.
    fn joint_stereo(&self) -> (bool, bool) {
        match self.mode {
            Mode::JointStereo {
                intensity_stereo,
                ms_stereo,
            } => (ms_stereo, intensity_stereo),
            _ => (false, false),
        }
    }

    /// The header of a frame at `bitrate`, with `mode_extension` switching
    /// MS stereo and intensity stereo on for this frame alone if `ms_stereo`
    /// and `intensity_stereo` are set.
    fn frame_header(
        &self,
        bitrate: Bitrate,
        padding: bool,
        ms_stereo: bool,
        intensity_stereo: bool,
    ) -> Header {
        let mode = match self.mode {
            Mode::JointStereo { .. } => Mode::JointStereo {
                intensity_stereo,
                ms_stereo,
            },
            mode => {
                assert!(
                    !ms_stereo && !intensity_stereo,
                    "MS and intensity stereo need the joint stereo mode"
                );
                mode
            }
        };
        Header {
            crc_protected: self.crc_protected,
            bitrate,
            sampling_frequency: self.sampling_frequency,
            padding,
            mode,
            flags: self.flags,
            emphasis: Emphasis::None,
        }
    }
}

/// The bits a frame has for main data: all of it but the header, the
/// `crc_check` word and the side information.
fn main_data_bits(header: &Header) -> usize {
    let crc = if header.crc_protected { 2 } else { 0 };
    let side_info =
        SideInfo::len_bytes(header.sampling_frequency.version(), header.mode.channels());
    8 * (header.frame_len() - Header::LEN_BYTES - crc - side_info)
}

/// The side information and main data of one frame of granules, one entry
/// per channel each, under `header`.
///
/// # Panics
///
/// If the right channel's scalefactors give a zero part
/// ([`Scalefactors::intensity`](scalefactors::Scalefactors::intensity)) and
/// `header` does not have intensity stereo on, or the other way round: the
/// decoder would read its zero part, and at the lower sampling frequencies
/// its `scalefac_compress`, otherwise than they were coded.
///
/// With `ancillary`, the main data ends with that many bits taken from the
/// queue, as the frame's ancillary data.
fn frame_contents(
    header: &Header,
    frame: &[Vec<QuantizedGranule>],
    main_data_begin: usize,
    ancillary: Option<(&mut AncillaryQueue, usize)>,
) -> (SideInfo, Vec<u8>) {
    let sampling_frequency = header.sampling_frequency;
    let version = sampling_frequency.version();
    assert!(
        main_data_begin < 1 << version.main_data_begin_bits(),
        "main_data_begin {main_data_begin} overflows its field"
    );
    let intensity_stereo = matches!(
        header.mode,
        Mode::JointStereo {
            intensity_stereo: true,
            ..
        }
    );
    for granule in frame {
        for (ch, channel) in granule.iter().enumerate() {
            assert_eq!(
                channel.scalefactors.intensity.is_some(),
                intensity_stereo && ch == 1,
                "only the right channel of a frame in intensity stereo has a zero part"
            );
        }
    }
    let mut side_info = SideInfo {
        main_data_begin: main_data_begin as u16,
        ..SideInfo::new(version)
    };
    let mut main_data = BitWriter::new();
    for (gr, granule) in frame.iter().enumerate() {
        for (ch, channel) in granule.iter().enumerate() {
            let start = main_data.bit_len();
            let scalefactors = &channel.scalefactors;
            // scfsi is sent once per frame and applies to its second granule.
            if gr == 0 {
                assert!(
                    !scalefactors.scfsi.contains(&true),
                    "the first granule of a frame sends all its scalefactors"
                );
            } else {
                side_info.scfsi[ch] = scalefactors.scfsi;
            }
            let scalefac_compress = scalefactors
                .compress()
                .expect("scalefactors within their limits");
            scalefactors.write(scalefac_compress, &mut main_data);
            let bands = scalefactor_bands::coding_bands(channel.block_type, sampling_frequency);
            channel
                .coding
                .write(&channel.values, &bands, &mut main_data);
            let part2_3_length = main_data.bit_len() - start;
            assert!(
                part2_3_length <= MAX_PART2_3_LENGTH,
                "{part2_3_length} bits overflow the 12-bit part2_3_length"
            );
            let coding = &channel.coding;
            side_info.granules[gr][ch] = GranuleChannel {
                part2_3_length: part2_3_length as u16,
                big_values: coding.big_values as u16,
                global_gain: channel.global_gain,
                scalefac_compress,
                window: channel.window(),
                preflag: scalefactors.preflag,
                scalefac_scale: scalefactors.scalefac_scale,
                count1table_select: coding.count1_table == QuadTable::B,
            };
        }
    }
    if let Some((queue, bits)) = ancillary {
        queue.write(bits, &mut main_data);
    }
    (side_info, main_data.into_bytes())
}

/// Encodes quantised granules, given in order with one entry per channel,
/// into frames of two granules each, or one at the lower sampling
/// frequencies, at a constant bitrate.
///
/// Main data is not shared between frames: `main_data_begin` is always 0. In
/// joint stereo every frame has MS stereo and intensity stereo on as the
/// configured mode sets them; in intensity stereo the right channel's
/// scalefactors must give its zero part
/// ([`Scalefactors::intensity`](scalefactors::Scalefactors::intensity)).
///
/// # Panics
///
/// If the rate is not [`Rate::Constant`], the granules do not divide into
/// frames for the configured channels, or a frame's granules do not fit in
/// it.
pub fn encode_granules(config: &StreamConfig, granules: &[Vec<QuantizedGranule>]) -> Vec<u8> {
    let channels = config.mode.channels();
    let per_frame = config.sampling_frequency.version().granules();
    assert!(
        granules.len().is_multiple_of(per_frame),
        "a frame holds {per_frame} granules"
    );
    assert!(
        granules.iter().all(|g| g.len() == channels),
        "each granule needs one entry per channel"
    );
    let (ms_stereo, intensity_stereo) = config.joint_stereo();
    let bitrate = config.constant_bitrate();
    let mut padding = PaddingSchedule::new(bitrate, config.sampling_frequency);
    let mut writer = FrameWriter::default();
    for frame in granules.chunks_exact(per_frame) {
        let header =
            config.frame_header(bitrate, padding.next_padding(), ms_stereo, intensity_stereo);
        let (side_info, main_data) = frame_contents(&header, frame, 0, None);
        writer.write(&header, &side_info, &main_data);
    }
    writer.out
}

/// Encodes PCM, one slice of samples (nominally -1.0 to 1.0) per channel, in
/// one go, with an [`Encoder`], without a Xing/Info tag.
///
/// # Panics
///
/// If the channel count does not match the configured mode, or the channels
/// have different numbers of samples.
pub fn encode_pcm(config: &StreamConfig, channels: &[&[f64]]) -> Vec<u8> {
    let mut encoder = Encoder::new(*config);
    let mut out = Vec::new();
    encoder.encode(channels, &mut out);
    encoder.finish(&mut out);
    out
}

/// Encodes PCM like [`encode_pcm`], with a Xing/Info tag at the start if
/// the stream has one (see [`Encoder::with_info_tag`]).
///
/// # Panics
///
/// As [`encode_pcm`].
pub fn encode_pcm_with_info_tag(config: &StreamConfig, channels: &[&[f64]]) -> Vec<u8> {
    let mut encoder = Encoder::with_info_tag(*config);
    let mut out = Vec::new();
    encoder.encode(channels, &mut out);
    if let Some(tag) = encoder.finish(&mut out) {
        out[..tag.len()].copy_from_slice(&tag);
    }
    out
}

/// Encodes PCM at the configured rate as it arrives.
///
/// Psychoacoustic model 2 chooses each granule's block type, switching to
/// short blocks for attacks, and sets the distortion each scalefactor band
/// allows and the perceptual entropy of each granule. Each granule of each
/// channel gets an equal share of its frame's main data, the `mean_bits` of
/// clause C.1.5.3.4, plus what the bit reservoir lends it, and the iteration
/// loops shape the quantisation noise to the allowed distortion within those
/// bits.
///
/// In joint stereo, frames whose channels are alike are coded in MS stereo
/// if the mode allows it, and frames short of bits are coded in intensity
/// stereo if the mode allows that (Annex G.2).
///
/// At a [`Rate::Variable`] each frame is coded at bitrates from the one
/// before it, as [`rate::search_variable`] tries them, and kept at the
/// lowest of those at which every band's noise stays within its allowance
/// (see [`loops::excess`]). Frames are not padded. Intensity stereo, which
/// gives up the stereo image above its bound for bits, is tried only at the
/// highest bitrate, where a larger frame is no longer to be had.
///
/// Spec reading: a padded frame is one slot longer to keep a mean bitrate
/// (clause 2.4.2.3: "the frame contains an additional slot to adjust the
/// mean bitrate to the sampling frequency"); at a variable bitrate there is
/// no mean to keep, and the bit reservoir carries what a frame does not
/// use.
///
/// The stream does not depend on how the input is divided between calls to
/// [`encode`](Self::encode).
pub struct Encoder {
    frames: FrameEncoder<ModelCoding>,
    tag: Option<TagState>,
}

/// The Xing/Info tag of an [`Encoder`] that writes one.
struct TagState {
    header: Header,
    /// Whether the tag's frame has been written, as a placeholder.
    written: bool,
    /// The CRC of the stream after the tag's frame so far.
    music_crc: u16,
}

impl Encoder {
    /// # Panics
    ///
    /// If the rate's bitrates are not of the version of the sampling
    /// frequency, or a [`Rate::Variable`] has its lowest bitrate above its
    /// highest.
    pub fn new(config: StreamConfig) -> Self {
        let fs = config.sampling_frequency;
        let (ms_allowed, intensity_allowed) = config.joint_stereo();
        let noise_factor = match config.rate {
            Rate::Variable(variable) => variable.noise_factor(),
            _ => 1.0,
        };
        let coding = ModelCoding {
            models: (0..config.mode.channels())
                .map(|_| BlockSwitching::new(fs))
                .collect(),
            ms_allowed,
            intensity_allowed,
            noise_factor,
            saving_noise_db: match config.rate {
                Rate::Variable(_) => 0.0,
                _ => loops::SAVING_NOISE_DB,
            },
            sampling_frequency: fs,
        };
        let shortest = config.rate.lowest().frame_len(fs, true);
        let reservoir = Reservoir::new(shortest, fs.version());
        Self {
            frames: FrameEncoder::new(config, coding, reservoir),
            tag: None,
        }
    }

    /// An encoder whose stream starts with a frame for a Xing/Info tag (see
    /// [`info_tag`]): the number of frames and bytes, a table to seek by,
    /// and the samples added before and after the input, which make the
    /// decoded output the input exactly for decoders that apply them. As
    /// those are known only at the end, the frame is written as a
    /// placeholder, silent and with no tag, and [`finish`](Self::finish)
    /// returns it with the tag, to be written over it. A stream in free
    /// format whose frames are too short for the tag has none, as its
    /// bitrate cannot change for the tag's frame ([`info_tag::header`]).
    ///
    /// # Panics
    ///
    /// As [`new`](Self::new).
    pub fn with_info_tag(config: StreamConfig) -> Self {
        Self {
            tag: info_tag::header(&config).map(|header| TagState {
                header,
                written: false,
                music_crc: info_tag::MUSIC_CRC_START,
            }),
            ..Self::new(config)
        }
    }

    /// Gives each frame up to `bits_per_frame` bits of ancillary data, from
    /// the bytes queued with [`push_ancillary`](Self::push_ancillary), after
    /// its audio. Its audio then has as many bits fewer, as ancillary data
    /// "will reduce the number of bits available for audio" (clause
    /// C.1.5.3.6); a frame with less data queued keeps the rest for audio.
    ///
    /// The ancillary data of a frame are all its bits from the end of its
    /// audio to where the next frame's main data begins (clause 2.4.2.8):
    /// the queued bits come first, and any after them are zero. As the
    /// standard leaves the bits "user definable", a reader cannot tell the
    /// queued bits from those zeros by the stream alone, and the data needs
    /// a framing of its own. Nor is it kept from imitating a syncword, which
    /// "may hamper synchronization" (clause C.1.5.3.6).
    ///
    /// # Panics
    ///
    /// If `bits_per_frame` is more than the main data slot of the stream's
    /// shortest frame holds.
    pub fn with_ancillary(mut self, bits_per_frame: usize) -> Self {
        let config = &self.frames.config;
        let shortest = config.frame_header(config.rate.lowest(), false, false, false);
        let slot = main_data_bits(&shortest);
        assert!(
            bits_per_frame <= slot,
            "{bits_per_frame} bits of ancillary data exceed the {slot} of the shortest frame"
        );
        self.frames.ancillary.set_bits_per_frame(bits_per_frame);
        self
    }

    /// Codes mixed blocks in place of short ones: the two lowest subbands of
    /// a granule with an attack keep the frequency resolution of the normal
    /// window (clause 2.4.2.7). See
    /// [`BlockSwitching::with_mixed_blocks`].
    ///
    /// Frames with mixed blocks are not coded in intensity stereo (see
    /// `intensity_frame`).
    ///
    /// # Panics
    ///
    /// At MPEG-2.5's 8 kHz, where mixed blocks are not used (see
    /// [`scalefactor_bands::mixed_long_bands`]).
    pub fn with_mixed_blocks(mut self) -> Self {
        let fs = self.frames.config.sampling_frequency;
        let models = std::mem::take(&mut self.frames.coding.models);
        self.frames.coding.models = (models.into_iter())
            .map(|model| model.with_mixed_blocks(fs))
            .collect();
        self
    }

    /// Queues `bytes` of ancillary data for the frames coded from now on,
    /// as many bits a frame as [`with_ancillary`](Self::with_ancillary)
    /// allows, each byte's most significant bit first.
    pub fn push_ancillary(&mut self, bytes: &[u8]) {
        self.frames.ancillary.push(bytes);
    }

    /// The bits of ancillary data queued that no frame has carried yet. A
    /// frame is coded once the granule after it has arrived;
    /// [`finish`](Self::finish) drops what its last frames have no room
    /// for.
    pub fn ancillary_pending(&self) -> usize {
        self.frames.ancillary.pending()
    }

    /// Encodes `channels`, one slice of samples (nominally -1.0 to 1.0) per
    /// channel, and appends to `out` the bytes of the stream that are final.
    ///
    /// A frame is coded once the granule after it has arrived, as its block
    /// type depends on whether that granule has an attack, and its bytes are
    /// final once no later frame's main data can begin in them: the bytes
    /// the bit reservoir holds, at most 511 of main data (255 at the lower
    /// sampling frequencies), and the frames
    /// after them stay behind until later calls.
    ///
    /// # Panics
    ///
    /// If the channel count does not match the configured mode, or the
    /// channels have different numbers of samples.
    pub fn encode(&mut self, channels: &[&[f64]], out: &mut Vec<u8>) {
        let start = self.write_placeholder(out);
        self.frames.encode(channels, out);
        self.add_to_crc(&out[start..]);
    }

    /// Ends the stream: pads the input with silence to whole frames, at
    /// least [`DELAY`] samples past the input, and appends the rest of the
    /// stream to `out`. For an encoder made [`with_info_tag`]
    /// (Self::with_info_tag), returns the tag's frame, if the stream has
    /// one, to be written over the first bytes of the stream.
    pub fn finish(mut self, out: &mut Vec<u8>) -> Option<Vec<u8>> {
        let start = self.write_placeholder(out);
        let config = self.frames.config;
        let ended = self.frames.finish(out);
        let mut tag = self.tag?;
        tag.music_crc = info_tag::music_crc(tag.music_crc, &out[start..]);
        let stream = info_tag::Stream {
            config: &config,
            frame_lens: &ended.frame_lens,
            samples: ended.samples,
            music_crc: tag.music_crc,
        };
        Some(info_tag::frame(&tag.header, &stream))
    }

    /// Writes the tag's placeholder to `out` if it is the first thing
    /// written, and returns where the frames after it start.
    fn write_placeholder(&mut self, out: &mut Vec<u8>) -> usize {
        if let Some(tag) = self.tag.as_mut().filter(|tag| !tag.written) {
            out.extend(info_tag::placeholder(&tag.header));
            tag.written = true;
        }
        out.len()
    }

    fn add_to_crc(&mut self, bytes: &[u8]) {
        if let Some(tag) = &mut self.tag {
            tag.music_crc = info_tag::music_crc(tag.music_crc, bytes);
        }
    }
}

/// The encoding process of Annex C: psychoacoustic model 2 with block
/// switching, the iteration loops and the bit reservoir, with joint stereo as
/// Annex G describes.
struct ModelCoding {
    /// One for each channel.
    models: Vec<BlockSwitching>,
    ms_allowed: bool,
    intensity_allowed: bool,
    /// The factor the allowed distortion of every band is scaled by: 1 but
    /// at a variable bitrate (see [`rate::VariableRate`]).
    noise_factor: f64,
    /// How much more noise than its allowance, in dB, a normal block within
    /// its allowance may be given to leave bits to the reservoir (see
    /// [`loops::outer_loop`]): [`loops::SAVING_NOISE_DB`] but at a variable
    /// bitrate, where it is 0. There the allowance is already raised by
    /// [`noise_factor`](Self::noise_factor), and a frame is kept at a
    /// bitrate only if its noise stays within it; a granule given more
    /// noise than that would push its frame to the highest bitrate. Graded
    /// with PEAQ (GstPEAQ 0.6.1, black box), 22 tracks of pop music at
    /// 44.1 kHz in stereo, with `noise_offset_db` 4, averaged 184 kbit/s and
    /// a grade of -0.35 without saving bits, 298 kbit/s and -0.06 saving
    /// them with the noise raised 4 dB, and 140 kbit/s and -0.50 saving them
    /// down to the allowance.
    saving_noise_db: f64,
    sampling_frequency: SamplingFrequency,
}

impl Coding for ModelCoding {
    type Analysis = Analysis;

    fn look_ahead(
        &mut self,
        channel: usize,
        pcm: &[f64; GRANULE_LINES],
    ) -> Option<(BlockType, Analysis)> {
        let masking = self.models[channel].granule(pcm)?;
        Some((masking.block_type(), (masking, Vec::new())))
    }

    fn prepare(&mut self, frame: &mut [Vec<Filtered<Analysis>>]) {
        let fs = self.sampling_frequency;
        for granule in frame.iter_mut() {
            for filtered in granule.iter_mut() {
                let (masking, xmin) = &mut filtered.analysis;
                *xmin = masking.allowed_distortion(&filtered.xr, fs);
                for xmin in xmin.iter_mut() {
                    *xmin *= self.noise_factor;
                }
            }
        }
    }

    fn quantize(
        &mut self,
        frame: &[Vec<Filtered<Analysis>>],
        mean_bits: usize,
        reservoir: &mut Reservoir,
        intensity: bool,
    ) -> CodedFrame {
        let fs = self.sampling_frequency;
        let ms_stereo = self.ms_allowed && middle_side_frame(frame);
        let steps = if intensity && self.intensity_allowed && intensity_frame(frame) {
            0..stereo::INTENSITY_STEPS
        } else {
            0..0
        };
        let to_quantize = FrameToQuantize {
            frame,
            ms_stereo,
            mean_bits,
            saving_noise_db: self.saving_noise_db,
            sampling_frequency: fs,
        };
        let plain = quantize_frame(&to_quantize, None, reservoir, IfOutsideAllowance::Finish)
            .expect("a finished frame");
        let tries: Vec<_> = if plain.within_allowance {
            Vec::new()
        } else {
            steps
                .filter_map(|step| IntensityTry::new(frame, &plain.granules, step, fs))
                .collect()
        };
        let kept = search_intensity(&tries, |intensity, if_outside| {
            quantize_frame(&to_quantize, Some(intensity), reservoir, if_outside)
        });
        let intensity_stereo = kept.is_some();
        let FrameTry {
            granules,
            excess,
            reservoir: kept_reservoir,
            ..
        } = kept.unwrap_or(plain);
        *reservoir = kept_reservoir;
        CodedFrame {
            granules,
            ms_stereo,
            intensity_stereo,
            excess,
        }
    }
}

/// Annex G.1: if the frame needs more bits than it has, more and more bands
/// are coded in intensity stereo, down to the lowest bound, which is kept
/// even if the frame still needs more. Bounds that would not lower the bits
/// are not tried (see [`IntensityTry::new`]).
///
/// Given the tries worth making, from the fewest bands to the most, and a
/// function that quantises the frame for a try, returns the try kept: `None`
/// if there are no tries, or if the function gives up on every one, as
/// [`quantize_frame`] can whatever the distortion. A try before the last is
/// kept only if the whole frame is within its allowed distortion, so it is
/// given up at the first granule that is not; the frame then comes out as if
/// it had been finished and passed over.
///
/// Spec gap: G.1 does not say what to do when the frame cannot be coded
/// from the lowest bound at all, as when at the lower sampling frequencies
/// the right channel would need a band marked (see [`quantize_frame`]). The
/// tries are then finished from the lowest bound up, and the first that can
/// be coded is kept: the most bands that can be coded in intensity stereo,
/// as G.1 would have from the lowest bound. Tries given up for their
/// distortion are quantised again to the end.
fn search_intensity<Try, Frame>(
    tries: &[Try],
    mut quantize: impl FnMut(&Try, IfOutsideAllowance) -> Option<Frame>,
) -> Option<Frame> {
    let (_, earlier) = tries.split_last()?;
    earlier
        .iter()
        .find_map(|intensity| quantize(intensity, IfOutsideAllowance::GiveUp))
        .or_else(|| {
            (tries.iter().rev())
                .find_map(|intensity| quantize(intensity, IfOutsideAllowance::Finish))
        })
}

/// A frame to quantise, the same for each try at coding it: its granules,
/// indexed `[gr][ch]`, whether it is in MS stereo, and the share of its main
/// data of each granule of each channel.
struct FrameToQuantize<'a> {
    frame: &'a [Vec<Filtered<Analysis>>],
    ms_stereo: bool,
    mean_bits: usize,
    /// How much more noise than its allowance, in dB, a normal block within
    /// its allowance may be given to leave bits to the reservoir (see
    /// [`loops::outer_loop`]).
    saving_noise_db: f64,
    sampling_frequency: SamplingFrequency,
}

/// What [`quantize_frame`] does when a granule comes out outside its allowed
/// distortion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IfOutsideAllowance {
    /// Quantise the rest of the frame.
    Finish,
    /// Stop, and return no frame.
    GiveUp,
}

/// A frame quantised one way, and the bit reservoir after it, which is taken
/// over only if the frame is kept.
struct FrameTry {
    /// The frame's granules, with one entry per channel each.
    granules: Vec<Vec<QuantizedGranule>>,
    /// Whether every granule was within its allowed distortion, as the
    /// outer loop judges it.
    within_allowance: bool,
    /// The largest [`loops::excess`] of its granules.
    excess: f64,
    reservoir: Reservoir,
}

/// A granule's psychoacoustic analysis and the allowed distortion of each
/// of its scalefactors, in the order they are sent.
type Analysis = (SwitchedMasking, Vec<f64>);

/// Whether to code a stereo frame in MS stereo (Annex G.2 a)).
///
/// The normative decoding needs both channels of a granule to have the same
/// block type in MS stereo (clause 2.4.3.4.9). Spec gap: Annex G does not
/// say what to do when they do not; such frames are coded in left/right
/// stereo.
fn middle_side_frame(frame: &[Vec<Filtered<Analysis>>]) -> bool {
    let same_types = frame
        .iter()
        .all(|granule| granule[0].block_type == granule[1].block_type);
    let spectra: Vec<_> = frame
        .iter()
        .map(|granule| {
            (
                granule[0].analysis.0.spectrum(),
                granule[1].analysis.0.spectrum(),
            )
        })
        .collect();
    same_types && stereo::ms_stereo(&spectra)
}

/// Whether a stereo frame may be coded in intensity stereo: if each of its
/// granules has the same block type in both channels, and none is a mixed
/// block. Short blocks find the bound in each window (see
/// [`stereo::IntensityPositions::mark`]).
///
/// Spec gap: the standard asks for the same block type in both channels
/// only for MS stereo (clause 2.4.3.4.9). As intensity stereo moves the left
/// channel's lines into the right channel's synthesis, whose window should
/// then be the same, and short blocks send their lines in another order
/// (clause 2.4.2.7), frames are coded in it only if their channels have the
/// same block types.
///
/// Spec gap: the bound of intensity stereo is found "individually for each
/// window in the case of short blocks (block_type == '10')" (clause 2.4.3.2
/// of ISO/IEC 13818-3), which mixed blocks are too, but neither standard
/// says how a mixed block's long bands, which have one window, and its
/// short bands share a bound. Frames with mixed blocks are not coded in
/// intensity stereo.
fn intensity_frame(frame: &[Vec<Filtered<Analysis>>]) -> bool {
    frame.iter().all(|granule| {
        granule[0].block_type == granule[1].block_type && granule[0].block_type != BlockType::Mixed
    })
}

/// A try at coding a frame in intensity stereo: the positions of each of
/// its granules, and the bits the right channel of each keeps.
///
/// Spec gap: Annex G does not say how to share the bits between the
/// channels in intensity stereo. In Layers I and II a combined subband is
/// allotted bits once, the higher of its two channels' allocations (Annex
/// G.1), but the Layer III iteration loops spend all the bits each channel
/// is given (clause C.1.5.4.4): the right
/// channel would spend the bits of the bands it no longer codes on those it
/// still does, while the left channel codes the sum of both on the bits it
/// had for itself alone. So the right channel keeps only the bits it spent
/// below the bound when the frame was coded without intensity stereo, and
/// those its scalefactors then take, marked with its intensity stereo
/// positions, or the most the marks can take if that is more. The left
/// channel gets what that leaves of the right channel's share of the
/// frame.
struct IntensityTry {
    positions: Vec<stereo::IntensityPositions>,
    right_bits: Vec<usize>,
}

impl IntensityTry {
    /// The try at coding `frame` in intensity stereo from the bounds of
    /// `step` of [`stereo::intensity_bounds`], long or short as each
    /// granule's block type has them, given its granules as coded without
    /// it, `plain`.
    ///
    /// Spec reading: Annex G.1 codes bands in intensity stereo to lower the
    /// bits a frame needs. When the right channel's bits from the bound up
    /// in `plain` are no more than its intensity stereo positions would take,
    /// the bound would not lower them, and there is no try: `None`. This is
    /// so at low bitrates, where the right channel's upper bands often
    /// quantise to zero without intensity stereo.
    fn new(
        frame: &[Vec<Filtered<Analysis>>],
        plain: &[Vec<QuantizedGranule>],
        step: usize,
        sampling_frequency: SamplingFrequency,
    ) -> Option<Self> {
        let fs = sampling_frequency;
        let positions: Vec<_> = frame
            .iter()
            .map(|granule| {
                let block_type = granule[0].block_type;
                let bound = stereo::intensity_bounds(block_type, fs)[step];
                let (left, right) = (&granule[0].xr, &granule[1].xr);
                stereo::IntensityPositions::new(left, right, block_type, bound, fs)
            })
            .collect();
        let right_bits: Vec<usize> = (plain.iter().zip(&positions))
            .map(|(granule, positions)| {
                let right = &granule[1];
                let bands = scalefactor_bands::coding_bands(right.block_type, fs);
                // The lines from the bound up follow those below it in every
                // window.
                let start = positions.first_line();
                let below = right.coding.bit_count_below(&right.values, &bands, start);
                // Its scalefactors as they would be sent with those values:
                // all of them, as it will not use scfsi, and marked.
                let mut values = right.values;
                values[start..].fill(0);
                // At the lower sampling frequencies the right channel's
                // branch of scalefac_compress has no preflag, and sends the
                // scalefactors within their limits without it.
                let mut scalefactors = right.scalefactors.clone();
                scalefactors.scfsi = [false; side_info::SCFSI_BANDS];
                if fs.version() == Version::Mpeg2 {
                    scalefactors.preflag = false;
                }
                positions.mark(&mut scalefactors, &values);
                let compress = scalefactors
                    .compress()
                    .expect("scalefactors within their limits");
                // Enough for the marks if fewer bits leave more bands zero.
                (scalefactors.part2_length(compress) + below).max(positions.silent_bits())
            })
            .collect();
        let plain_bits: usize = plain.iter().map(|granule| granule[1].bit_len(fs)).sum();
        (right_bits.iter().sum::<usize>() < plain_bits).then_some(Self {
            positions,
            right_bits,
        })
    }
}

/// Quantises a frame's granules, in MS stereo if `ms_stereo` is set and in
/// intensity stereo as `intensity` gives, accounting for their bits in a
/// copy of the reservoir, so that a try leaves the encoder's state as it
/// was. `None` if `if_outside` gives up on a granule outside its allowed
/// distortion, or if the reservoir gives the right channel of a granule
/// in intensity stereo fewer bits than its marks take with all its lines
/// zero (see [`stereo::IntensityPositions::silent_bits`]), which can happen
/// at the lowest bitrates of the lower sampling frequencies. `None` too at
/// the lower sampling frequencies if the right channel of a granule in
/// intensity stereo would mark a band.
///
/// Spec reading: at the lower sampling frequencies the illegal intensity
/// position that marks a band of the right channel's zero part as not coded
/// in intensity stereo is the largest value of its field (clause 2.4.3.2 of
/// ISO/IEC 13818-3), but decoders do not agree on it, and no mark is read as
/// one by all of them. So the encoder sends no marks there: the right
/// channel is given a value in the band just below the bound where it has
/// none, in any window of a short block (see
/// [`stereo::IntensityPositions::fill_below_bound`]), and a frame whose
/// right channel would still need a mark is coded from another bound (see
/// [`search_intensity`]) or without intensity stereo.
///
/// Settled by decoding with ffmpeg, mpg123 and LAME (black box): of the
/// marks of fields of 1 to 5 bits, 1, 3, 7, 15 and 31, ffmpeg reads only 31
/// as illegal and the others as positions, and mpg123 and LAME read only 7
/// as illegal and the others as positions; a marked band is then decoded in
/// intensity stereo. The conformance test is
/// `marks_are_read_as_illegal_in_different_fields`. In MS stereo at 22.05
/// kHz, with the side channel all but zero below every bound, frames are
/// coded in intensity stereo mostly from the highest bound, below which the
/// side channel codes what is left of the tones above it, and all three
/// decode every tone at its level; the conformance test is
/// `intensity_stereo_keeps_the_level_of_each_channel_at_the_lower_sampling_frequencies`.
/// At 16 and 24 kHz the tones of that test leave nothing there, and nearly
/// every frame is coded in MS stereo alone.
///
/// The perceptual entropy each channel's bits are reckoned from stays that
/// of the left and right channels; in intensity stereo the two then share
/// them as [`IntensityTry`] describes.
///
/// Spec gap: Annex G does not say how to find the allowed distortion of
/// the channels coded in MS or intensity stereo; see
/// [`stereo::middle_side_distortion`] and
/// [`stereo::IntensityPositions::sum_distortion`].
///
/// Spec gap: scfsi would have the second granule take over the first
/// granule's intensity stereo positions, which follow the signal of each
/// granule (Annex G.2 c)). In intensity stereo the right channel does not
/// use scfsi.
fn quantize_frame(
    to_quantize: &FrameToQuantize,
    intensity: Option<&IntensityTry>,
    reservoir: &Reservoir,
    if_outside: IfOutsideAllowance,
) -> Option<FrameTry> {
    let &FrameToQuantize {
        frame,
        ms_stereo,
        mean_bits,
        saving_noise_db,
        sampling_frequency: fs,
    } = to_quantize;
    let mut reservoir = reservoir.clone();
    // Each channel's first granule, for scfsi in the second.
    let mut first: Vec<Option<(GranuleSummary, Scalefactors)>> = vec![None; frame[0].len()];
    let mut within_allowance = true;
    let mut excess: f64 = 0.0;
    let mut granules = Vec::with_capacity(frame.len());
    for (gr, granule) in frame.iter().enumerate() {
        let mut lines: Vec<_> = granule
            .iter()
            .map(|f| (f.xr, f.analysis.1.clone()))
            .collect();
        let positions = intensity.map(|intensity| &intensity.positions[gr]);
        if ms_stereo || positions.is_some() {
            let [(left, left_xmin), (right, right_xmin)] = &mut lines[..] else {
                unreachable!("joint stereo has two channels")
            };
            if ms_stereo {
                (*left, *right) = stereo::middle_side(left, right);
                stereo::limit_side(right, granule[0].block_type, fs);
                let joint = stereo::middle_side_distortion(left_xmin, right_xmin);
                left_xmin.clone_from(&joint);
                *right_xmin = joint;
            }
            if let Some(positions) = positions {
                positions.code(&granule[0].xr, &granule[1].xr, [left, right]);
                let (left_alone, right_alone) = (&granule[0].analysis.1, &granule[1].analysis.1);
                let coded = positions.first_scalefactor();
                for (index, xmin) in left_xmin.iter_mut().enumerate().skip(coded) {
                    *xmin = positions.sum_distortion(index, left_alone[index], right_alone[index]);
                }
            }
        }
        // The right channel's bits are reckoned before the left channel takes
        // any, and need not all be there after; it gets no more than the
        // reservoir then gives it. The left channel gets only what the right
        // leaves of its share of the frame, which the right then returns.
        let right_bits = intensity.map(|intensity| intensity.right_bits[gr]);
        let mut quantized = Vec::with_capacity(granule.len());
        for (channel, (filtered, (xr, xmin))) in granule.iter().zip(&lines).enumerate() {
            let block_type = filtered.block_type;
            let mut available_bits =
                reservoir.granule_bits(mean_bits, filtered.analysis.0.perceptual_entropy());
            if let Some(right_bits) = right_bits {
                available_bits = match channel {
                    0 => available_bits + mean_bits.saturating_sub(right_bits),
                    _ => available_bits.min(right_bits),
                };
            }
            let available_bits = available_bits.min(MAX_PART2_3_LENGTH);
            let intensity = positions.filter(|_| channel == 1);
            if intensity.is_some_and(|positions| available_bits < positions.silent_bits()) {
                return None;
            }
            let summary = GranuleSummary::new(xr, xmin, block_type, fs);
            let reuse = match (gr, &first[channel]) {
                (1, Some((first_summary, first_scalefactors))) if intensity.is_none() => {
                    Some(loops::Reuse {
                        first: first_scalefactors,
                        scfsi: scfsi::select(first_summary, &summary),
                    })
                }
                _ => None,
            };
            let bits = loops::Bits {
                available: available_bits,
                saving: Some(loops::Saving {
                    least: reservoir.least_use(mean_bits),
                    noise_db: saving_noise_db,
                }),
            };
            let outcome = loops::quantize_granule(xr, xmin, block_type, bits, fs, reuse, intensity);
            if gr == 0 {
                first[channel] = Some((summary, outcome.granule.scalefactors.clone()));
            }
            if fs.version() == Version::Mpeg2 && outcome.granule.scalefactors.marks_any() {
                return None;
            }
            reservoir.granule_done(mean_bits, outcome.granule.bit_len(fs));
            if !outcome.within_allowance && if_outside == IfOutsideAllowance::GiveUp {
                return None;
            }
            within_allowance &= outcome.within_allowance;
            excess = excess.max(loops::excess(xr, xmin, &outcome.granule, fs));
            quantized.push(outcome.granule);
        }
        granules.push(quantized);
    }
    Some(FrameTry {
        granules,
        within_allowance,
        excess,
        reservoir,
    })
}

/// Encodes PCM like [`encode_pcm`], but with long blocks only, quantising
/// every granule with the same `global_gain` instead of fitting it to the
/// bitrate, and without a bit reservoir.
///
/// # Panics
///
/// If the channel count does not match the configured mode, or a granule
/// cannot be quantised with `global_gain` or does not fit in its frame.
pub fn encode_pcm_with_global_gain(
    config: &StreamConfig,
    global_gain: u8,
    channels: &[&[f64]],
) -> Vec<u8> {
    encode_with_block_types(config, global_gain, channels, |_, _| BlockType::Normal)
}

/// Encodes PCM like [`encode_pcm_with_global_gain`], with the block type of
/// each granule given by `block_type(channel, granule)`. The block types must
/// follow each other as figure C.7 allows, with mixed blocks in place of
/// short ones if any, and not at 8 kHz. If the mode allows MS stereo,
/// every frame is coded in it, with no limitation of the side channel, and
/// both channels must have the same block types.
///
/// # Panics
///
/// As [`encode_pcm_with_global_gain`], and if the mode allows intensity
/// stereo.
pub fn encode_with_block_types(
    config: &StreamConfig,
    global_gain: u8,
    channels: &[&[f64]],
    block_type: impl FnMut(usize, usize) -> BlockType,
) -> Vec<u8> {
    let (ms_stereo, intensity_stereo) = config.joint_stereo();
    assert!(!intensity_stereo, "intensity stereo needs encode_pcm");
    let coding = FixedGainCoding {
        global_gain,
        block_type,
        next: vec![0; config.mode.channels()],
        ms_stereo,
        sampling_frequency: config.sampling_frequency,
    };
    let mut encoder = FrameEncoder::new(*config, coding, Reservoir::disabled());
    let mut out = Vec::new();
    encoder.encode(channels, &mut out);
    encoder.finish(&mut out);
    out
}

/// Quantisation of every granule with the same `global_gain`, with the block
/// type of each given by `block_type(channel, granule)`, for
/// [`encode_with_block_types`].
struct FixedGainCoding<F> {
    global_gain: u8,
    block_type: F,
    /// The index of the granule each channel's look-ahead is given next.
    next: Vec<usize>,
    ms_stereo: bool,
    sampling_frequency: SamplingFrequency,
}

impl<F: FnMut(usize, usize) -> BlockType> Coding for FixedGainCoding<F> {
    type Analysis = ();

    fn look_ahead(&mut self, channel: usize, _: &[f64; GRANULE_LINES]) -> Option<(BlockType, ())> {
        // Called once ahead of the first granule, whose type it gives on the
        // second call.
        let granule = self.next[channel].checked_sub(1);
        self.next[channel] += 1;
        granule.map(|g| ((self.block_type)(channel, g), ()))
    }

    fn quantize(
        &mut self,
        frame: &[Vec<Filtered<()>>],
        mean_bits: usize,
        reservoir: &mut Reservoir,
        _intensity: bool,
    ) -> CodedFrame {
        let (global_gain, fs) = (self.global_gain, self.sampling_frequency);
        let mut granules = Vec::with_capacity(frame.len());
        for granule in frame {
            let mut lines: Vec<_> = granule.iter().map(|filtered| filtered.xr).collect();
            if self.ms_stereo {
                let [left, right] = &granule[..] else {
                    unreachable!("MS stereo has two channels")
                };
                assert_eq!(
                    left.block_type, right.block_type,
                    "MS stereo needs one block type"
                );
                let (middle, side) = stereo::middle_side(&left.xr, &right.xr);
                lines = vec![middle, side];
            }
            let quantized: Vec<_> = (granule.iter().zip(&lines))
                .map(|(filtered, xr)| {
                    let values = loops::quantize(xr, global_gain).unwrap_or_else(|| {
                        panic!("a granule is too loud for global_gain {global_gain}")
                    });
                    let granule =
                        QuantizedGranule::new(global_gain, values, filtered.block_type, fs);
                    reservoir.granule_done(mean_bits, granule.bit_len(fs));
                    granule
                })
                .collect();
            granules.push(quantized);
        }
        CodedFrame {
            granules,
            ms_stereo: self.ms_stereo,
            intensity_stereo: false,
            excess: 0.0,
        }
    }
}

/// One granule of one channel, analysed and filtered, before quantisation.
struct Filtered<A> {
    block_type: BlockType,
    analysis: A,
    /// The lines in the order they are sent.
    xr: [f64; GRANULE_LINES],
}

/// A frame's granules, quantised, with one entry per channel each, and the
/// joint stereo coding they use.
struct CodedFrame {
    granules: Vec<Vec<QuantizedGranule>>,
    ms_stereo: bool,
    intensity_stereo: bool,
    /// The largest [`loops::excess`] of its granules: at most 1 if all are
    /// within their allowed distortion.
    excess: f64,
}

/// How [`FrameEncoder`] analyses and quantises granules.
trait Coding {
    /// What the analysis of a granule gives besides its block type.
    type Analysis;

    /// Given each channel's PCM granule by granule, starting with the first,
    /// returns the block type and analysis of the granule before the one it
    /// is given: `None` only for the first. It is given each granule two
    /// granules ahead of the filterbank, which needs the block types of the
    /// granule it filters and of the one after.
    fn look_ahead(
        &mut self,
        channel: usize,
        pcm: &[f64; GRANULE_LINES],
    ) -> Option<(BlockType, Self::Analysis)>;

    /// Readies a frame's granules, indexed `[gr][ch]`, for quantisation,
    /// once however many times the frame is quantised.
    fn prepare(&mut self, _frame: &mut [Vec<Filtered<Self::Analysis>>]) {}

    /// Quantises a frame's granules, indexed `[gr][ch]`, given the share of
    /// the frame's main data of each granule of each channel, accounting in
    /// the reservoir for the bits each uses. Intensity stereo is tried only
    /// if `intensity` is set and the mode allows it.
    fn quantize(
        &mut self,
        frame: &[Vec<Filtered<Self::Analysis>>],
        mean_bits: usize,
        reservoir: &mut Reservoir,
        intensity: bool,
    ) -> CodedFrame;
}

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
            Rate::Constant(bitrate) => Self::Constant(bitrate, PaddingSchedule::new(bitrate, fs)),
            Rate::Mean(mean) => Self::Mean(MeanRateSchedule::new(mean, fs)),
            Rate::Variable(variable) => Self::Variable {
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
struct FrameEncoder<C: Coding> {
    config: StreamConfig,
    coding: C,
    /// The reservoir between frames.
    reservoir: Reservoir,
    /// One for each channel.
    filterbanks: Vec<filterbank::Filterbank>,
    schedule: Schedule,
    writer: FrameWriter,
    /// The bytes the reservoir holds for the next frame.
    reservoir_bytes: usize,
    /// Each channel's PCM from the first granule not yet filtered.
    pcm: Vec<Vec<f64>>,
    /// The samples each channel has been given.
    samples: usize,
    /// The length of each frame written.
    frame_lens: Vec<usize>,
    ancillary: AncillaryQueue,
    /// For each channel, the block type and analysis of the granule after
    /// the next to be filtered.
    ahead: Vec<Option<(BlockType, C::Analysis)>>,
}

/// What a [`FrameEncoder`] wrote, once it has finished.
struct Ended {
    /// The length of each frame.
    frame_lens: Vec<usize>,
    /// The samples each channel was given.
    samples: usize,
}

impl<C: Coding> FrameEncoder<C> {
    fn new(config: StreamConfig, coding: C, reservoir: Reservoir) -> Self {
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
    /// and appends the bytes that are final to `out`.
    fn encode(&mut self, channels: &[&[f64]], out: &mut Vec<u8>) {
        assert_eq!(channels.len(), self.pcm.len(), "one slice per channel");
        let len = channels[0].len();
        assert!(
            channels.iter().all(|c| c.len() == len),
            "every channel has the same number of samples"
        );
        for (pcm, samples) in self.pcm.iter_mut().zip(channels) {
            pcm.extend_from_slice(samples);
        }
        self.samples += len;
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
    /// codes them, and appends the rest of the stream to `out`.
    fn finish(mut self, out: &mut Vec<u8>) -> Ended {
        let frame_samples = self.samples_per_frame();
        let frames = (self.samples + DELAY).div_ceil(frame_samples);
        let remaining = frames - self.frame_lens.len();
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
        }
    }

    fn samples_per_frame(&self) -> usize {
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
                self.quantize_at(&filtered, bitrate, padding, true, ancillary_bits)
            }
            Schedule::Mean(schedule) => {
                let (bitrate, padding) = schedule.next_frame();
                self.quantize_at(&filtered, bitrate, padding, true, ancillary_bits)
            }
            Schedule::Variable { bitrates, previous } => {
                let (bitrates, start) = (bitrates.clone(), *previous);
                let highest = bitrates.len() - 1;
                let (index, kept) = rate::search_variable(bitrates.len(), start, |index| {
                    let at = self.quantize_at(
                        &filtered,
                        bitrates[index],
                        false,
                        index == highest,
                        ancillary_bits,
                    );
                    let within = at.coded.excess <= 1.0;
                    (at, within)
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

    /// Quantises a frame at `bitrate`, with or without `padding`, from the
    /// reservoir as the frame finds it, trying intensity stereo if
    /// `intensity` is set and the mode allows it, with `ancillary_bits` of
    /// its main data slot kept for ancillary data.
    fn quantize_at(
        &mut self,
        frame: &[Vec<Filtered<C::Analysis>>],
        bitrate: Bitrate,
        padding: bool,
        intensity: bool,
        ancillary_bits: usize,
    ) -> FrameAtBitrate {
        let fs = self.config.sampling_frequency;
        let mut reservoir = self.reservoir.for_frame(bitrate.frame_len(fs, true));
        let main_data_begin = reservoir.bytes();
        // mode_extension does not change the frame's length.
        let slot_bits = main_data_bits(&self.config.frame_header(bitrate, padding, false, false))
            .checked_sub(ancillary_bits)
            .expect("the ancillary data fits in the frame");
        let shares = frame.len() * self.pcm.len();
        let mean_bits = slot_bits / shares;
        let coded = self
            .coding
            .quantize(frame, mean_bits, &mut reservoir, intensity);
        FrameAtBitrate {
            bitrate,
            padding,
            coded,
            main_data_begin,
            unshared_bits: slot_bits - shares * mean_bits,
            reservoir,
        }
    }
}

/// Encodes `frames` frames of digital silence. Every granule codes to
/// nothing, so each frame is zero-filled after its side information.
pub fn encode_silence(config: &StreamConfig, frames: usize) -> Vec<u8> {
    let silent = QuantizedGranule::new(
        0,
        [0; GRANULE_LINES],
        BlockType::Normal,
        config.sampling_frequency,
    );
    let granule = vec![silent; config.mode.channels()];
    let granules = frames * config.sampling_frequency.version().granules();
    encode_granules(config, &vec![granule; granules])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(crc_protected: bool) -> StreamConfig {
        StreamConfig {
            rate: Rate::Constant(Bitrate::from_kbps(Version::Mpeg1, 128).unwrap()),
            sampling_frequency: SamplingFrequency::Hz44100,
            mode: Mode::Stereo,
            crc_protected,
            flags: Flags::default(),
        }
    }

    #[test]
    fn silence_frames_are_back_to_back() {
        let bytes = encode_silence(&config(false), 3);
        // 128 kbit/s at 44.1 kHz is 417.96 slots per frame.
        assert_eq!(bytes.len(), 417 + 418 + 418);
        for (offset, padded) in [(0, false), (417, true), (835, true)] {
            assert_eq!(bytes[offset..offset + 2], [0xFF, 0xFB]);
            assert_eq!(bytes[offset + 2] & 0b10 != 0, padded);
        }
    }

    #[test]
    fn crc_word_sits_between_header_and_side_info() {
        let plain = encode_silence(&config(false), 1);
        let protected = encode_silence(&config(true), 1);
        assert_eq!(plain.len(), protected.len());
        assert_eq!(protected[..2], [0xFF, 0xFA]);
        assert_eq!(protected[2..4], plain[2..4]);

        let side_info = &plain[4..4 + SideInfo::len_bytes(Version::Mpeg1, 2)];
        assert_eq!(protected[6..6 + side_info.len()], *side_info);
        let expected = crc::crc16(plain[2..4].iter().chain(side_info).copied());
        assert_eq!(protected[4..6], expected.to_be_bytes());
    }

    fn small_header() -> Header {
        Header {
            crc_protected: false,
            bitrate: Bitrate::from_kbps(Version::Mpeg1, 32).unwrap(),
            sampling_frequency: SamplingFrequency::Hz32000,
            padding: false,
            mode: Mode::SingleChannel,
            flags: Flags::default(),
            emphasis: Emphasis::None,
        }
    }

    /// Figure A.7: a frame's main data starts `main_data_begin` bytes back
    /// in the slot bytes the frames before it left unused, and runs on into
    /// its own slot.
    #[test]
    fn main_data_continues_from_earlier_slots() {
        let header = small_header();
        // 144 slots minus the 4-byte header and 17 bytes of side information.
        let slot = 144 - 4 - 17;
        let mut writer = FrameWriter::default();
        writer.write(&header, &SideInfo::new(Version::Mpeg1), &[1; 100]);
        let side_info = SideInfo {
            main_data_begin: 20,
            ..SideInfo::new(Version::Mpeg1)
        };
        writer.write(&header, &side_info, &[2; 30]);
        let first = &writer.out[21..144];
        assert!(first[..100].iter().all(|&b| b == 1));
        // The last 20 of the 23 bytes the first frame left unused hold the
        // start of the second frame's main data; the 3 before them are
        // stuffing.
        assert_eq!(first[100..103], [0; 3]);
        assert!(first[103..].iter().all(|&b| b == 2));
        let second = &writer.out[144 + 21..];
        assert_eq!(second.len(), slot);
        assert!(second[..10].iter().all(|&b| b == 2));
        assert!(second[10..].iter().all(|&b| b == 0));
        // main_data_begin is the first 9 bits of the side information.
        assert_eq!(writer.out[144 + 4], 20 >> 1);
    }

    #[test]
    #[should_panic(expected = "reaches past")]
    fn main_data_cannot_begin_before_the_stream() {
        let side_info = SideInfo {
            main_data_begin: 1,
            ..SideInfo::new(Version::Mpeg1)
        };
        FrameWriter::default().write(&small_header(), &side_info, &[]);
    }

    #[test]
    #[should_panic(expected = "exceed")]
    fn rejects_main_data_larger_than_the_frame() {
        let header = Header {
            crc_protected: false,
            bitrate: Bitrate::from_kbps(Version::Mpeg1, 32).unwrap(),
            sampling_frequency: SamplingFrequency::Hz32000,
            padding: false,
            mode: Mode::SingleChannel,
            flags: Flags::default(),
            emphasis: Emphasis::None,
        };
        // 144 slots minus the 4-byte header and 17 bytes of side information.
        let main_data = [0; 144 - 4 - 17 + 1];
        write_frame(
            &header,
            &SideInfo::new(Version::Mpeg1),
            &main_data,
            &mut Vec::new(),
        );
    }

    /// How a try at intensity stereo comes out: within its allowed
    /// distortion, outside it, or not coded at all.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum TryOutcome {
        Within,
        Outside,
        Uncodable,
    }

    /// For every pattern of outcomes of tries, up to the most a frame can
    /// have, the search keeps the first try before the last within its
    /// allowed distortion, or else the one with the most bands that can be
    /// coded. It gives up only on tries before the last, finishes tries only
    /// once it has given up on all of those, from the last back, and
    /// quantises none after the one it keeps.
    #[test]
    fn intensity_search_keeps_the_first_try_that_fits_or_else_the_lowest_that_can_be_coded() {
        use TryOutcome::*;
        for count in 0..=stereo::INTENSITY_STEPS {
            for pattern in 0..3usize.pow(count as u32) {
                let outcomes: Vec<_> = (0..count)
                    .map(|index| {
                        [Within, Outside, Uncodable][pattern / 3usize.pow(index as u32) % 3]
                    })
                    .collect();
                let tries: Vec<usize> = (0..count).collect();
                let mut calls = Vec::new();
                let kept = search_intensity(&tries, |&index, if_outside| {
                    calls.push((index, if_outside));
                    match (outcomes[index], if_outside) {
                        (Within, _) | (Outside, IfOutsideAllowance::Finish) => Some(index),
                        _ => None,
                    }
                });

                let earlier = count.saturating_sub(1);
                let within = (0..earlier).find(|&index| outcomes[index] == Within);
                let expected =
                    within.or_else(|| (0..count).rev().find(|&i| outcomes[i] != Uncodable));
                assert_eq!(kept, expected, "{outcomes:?}");
                let gave_up = (0..within.map_or(earlier, |index| index + 1))
                    .map(|index| (index, IfOutsideAllowance::GiveUp));
                let finished = match within {
                    Some(_) => 0..0,
                    None => expected.unwrap_or(0)..count,
                };
                let finished = (finished.rev()).map(|index| (index, IfOutsideAllowance::Finish));
                let expected_calls: Vec<_> = gave_up.chain(finished).collect();
                assert_eq!(calls, expected_calls, "{outcomes:?}");
            }
        }
    }

    /// A deterministic sequence of samples from -1.0 to 1.0 (an LCG).
    fn noise(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (*seed >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    /// However the input is divided between calls, the encoder writes the
    /// same stream, of whole frames for the input padded with silence to at
    /// least [`DELAY`] samples past it, and holds back no more than the next frame's main data can
    /// begin in: its `main_data_begin` bytes, and the header and side
    /// information of each frame after the first of them.
    #[test]
    fn streaming_does_not_depend_on_how_the_input_is_divided() {
        let config = StreamConfig {
            rate: Rate::Constant(Bitrate::from_kbps(Version::Mpeg1, 32).unwrap()),
            sampling_frequency: SamplingFrequency::Hz44100,
            mode: Mode::JointStereo {
                intensity_stereo: false,
                ms_stereo: true,
            },
            crc_protected: false,
            flags: Flags::default(),
        };
        // Silence fills the reservoir to its limit, past which the rest is
        // stuffing, and the noise after it reaches back into it.
        let mut seed = 1;
        let samples = 12 * Version::Mpeg1.samples_per_frame() + 300;
        let pcm: Vec<[f64; 2]> = (0..samples)
            .map(|n| {
                let level = if n < 9 * Version::Mpeg1.samples_per_frame() {
                    0.0
                } else {
                    0.5
                };
                [level * noise(&mut seed), level * noise(&mut seed)]
            })
            .collect();
        let (left, right): (Vec<f64>, Vec<f64>) = pcm.iter().map(|&[l, r]| (l, r)).unzip();
        let whole = encode_pcm(&config, &[&left, &right]);
        stream_check::frames(&whole);
        let bitrate = config.constant_bitrate();
        let fs = config.sampling_frequency;
        let mut padding = PaddingSchedule::new(bitrate, fs);
        let frames = (samples + DELAY).div_ceil(Version::Mpeg1.samples_per_frame());
        let frame_lens = (0..frames).map(|_| bitrate.frame_len(fs, padding.next_padding()));
        assert_eq!(whole.len(), frame_lens.sum::<usize>());

        // Frames without padding have the smallest slots, and so span the
        // most frames for the same bytes.
        let header = config.frame_header(bitrate, false, false, false);
        let slot = main_data_bits(&header) / 8;
        let overhead = header.frame_len() - slot;
        let streams = std::thread::scope(|scope| {
            [
                1,
                100,
                GRANULE_LINES,
                Version::Mpeg1.samples_per_frame() - 1,
                Version::Mpeg1.samples_per_frame() + GRANULE_LINES + 1,
            ]
            .map(|chunk| {
                let (left, right) = (&left, &right);
                scope.spawn(move || {
                    let mut encoder = Encoder::new(config);
                    let mut out = Vec::new();
                    let mut furthest = 0;
                    for (left, right) in left.chunks(chunk).zip(right.chunks(chunk)) {
                        encoder.encode(&[left, right], &mut out);
                        let begin = encoder.frames.reservoir_bytes;
                        furthest = furthest.max(begin);
                        let held = encoder.frames.writer.out.len();
                        let most = begin + begin.div_ceil(slot) * overhead;
                        assert!(held <= most, "{held} bytes held in chunks of {chunk}");
                    }
                    encoder.finish(&mut out);
                    (chunk, out, furthest)
                })
            })
            .map(|stream| stream.join().expect("the encoding passes"))
        });
        for (chunk, stream, furthest) in streams {
            assert!(stream == whole, "chunks of {chunk}");
            assert!(
                furthest > slot,
                "main data reaches back {furthest} bytes, within the frame before"
            );
        }
    }

    /// Silence, full-scale noise, a quiet tone, bursts of noise with sharp
    /// onsets, and silence again, over `seconds` seconds at `fs`, in two
    /// channels that differ in the noise.
    fn varied(fs: SamplingFrequency, seconds: f64) -> [Vec<f64>; 2] {
        let hz = f64::from(fs.hz());
        let mut seed = 7;
        let (mut left, mut right) = (Vec::new(), Vec::new());
        for t in 0..(seconds * hz) as usize {
            let s = t as f64 / hz / seconds;
            let (common, own) = (noise(&mut seed), noise(&mut seed));
            let tone = (std::f64::consts::TAU * 440.0 * t as f64 / hz).sin();
            let (l, r) = match s {
                s if s < 0.15 => (0.0, 0.0),
                s if s < 0.4 => (0.9 * common, 0.9 * (0.5 * common + 0.5 * own)),
                s if s < 0.6 => (0.01 * tone, 0.01 * tone),
                s if s < 0.9 => {
                    let burst = 0.7 * (-(s - 0.6) % 0.1 * seconds / 0.02).exp();
                    (burst * common + 0.05 * tone, burst * own)
                }
                _ => (0.0, 0.0),
            };
            left.push(l);
            right.push(r);
        }
        [left, right]
    }

    const MS: Mode = Mode::JointStereo {
        intensity_stereo: false,
        ms_stereo: true,
    };

    fn variable(fs: SamplingFrequency, kbps: [u32; 2], noise_offset_db: f64) -> Rate {
        let at = |kbps| Bitrate::from_kbps(fs.version(), kbps).unwrap();
        Rate::Variable(rate::VariableRate {
            lowest: at(kbps[0]),
            highest: at(kbps[1]),
            noise_offset_db,
        })
    }

    fn encode(config: StreamConfig, [left, right]: &[Vec<f64>; 2]) -> Vec<u8> {
        match config.mode {
            Mode::SingleChannel => encode_pcm(&config, &[left]),
            _ => encode_pcm(&config, &[left, right]),
        }
    }

    /// A stream in dual channel is the stream in stereo but for the `mode`
    /// of each header, and the `crc_check` that covers it: "the coding
    /// process is the same as for the stereo mode" (clause 2.1.52).
    #[test]
    fn dual_channel_codes_as_stereo() {
        use SamplingFrequency::*;
        let mean = |fs: SamplingFrequency, kbps: u32| {
            Rate::Mean(rate::MeanRate::new(fs.version(), kbps * 1000).unwrap())
        };
        let at = |fs: SamplingFrequency, kbps| {
            Rate::Constant(Bitrate::from_kbps(fs.version(), kbps).unwrap())
        };
        let cases = [
            (Hz44100, at(Hz44100, 128), false),
            (Hz48000, mean(Hz48000, 100), true),
            (Hz32000, variable(Hz32000, [32, 320], 0.0), false),
            (Hz22050, at(Hz22050, 64), true),
            (Hz11025, variable(Hz11025, [8, 64], 4.0), false),
        ];
        for (fs, rate, crc_protected) in cases {
            let pcm = varied(fs, 1.0);
            let config = |mode| StreamConfig {
                rate,
                sampling_frequency: fs,
                mode,
                crc_protected,
                flags: Flags::default(),
            };
            let stereo = encode(config(Mode::Stereo), &pcm);
            let mut dual = encode(config(Mode::DualChannel), &pcm);
            assert_eq!(dual.len(), stereo.len(), "{fs:?}");
            let mut at = 0;
            for frame in stream_check::frames(&stereo) {
                assert_eq!(stereo[at + 3] >> 6, 0b00, "{fs:?}: stereo at {at}");
                assert_eq!(dual[at + 3] >> 6, 0b10, "{fs:?}: dual channel at {at}");
                dual[at + 3] &= 0b0011_1111;
                if crc_protected {
                    dual[at + 4..at + 6].copy_from_slice(&stereo[at + 4..at + 6]);
                }
                at += frame.len;
            }
            assert!(dual == stereo, "{fs:?}");
        }
    }

    /// Each frame's ancillary data, from the end of its Huffman code bits to
    /// where the next frame's main data begins (clause 2.4.2.8), starts with
    /// the next of the queued bits, as many as it may carry, and is zero
    /// after them, at constant, mean and variable bitrates, in both
    /// versions, with a share of a whole number of bytes or not, and with
    /// data that imitates headers. Once the queue runs dry the frames carry
    /// none, and without ancillary data the bits there are all zero.
    #[test]
    fn frames_carry_the_queued_ancillary_data() {
        use SamplingFrequency::*;
        let at = |fs: SamplingFrequency, kbps| {
            Rate::Constant(Bitrate::from_kbps(fs.version(), kbps).unwrap())
        };
        let mean = Rate::Mean(rate::MeanRate::new(Version::Mpeg1, 100_000).unwrap());
        let cases = [
            (Hz44100, at(Hz44100, 128), MS, false, 1000),
            (Hz48000, mean, Mode::Stereo, true, 2000),
            (
                Hz32000,
                variable(Hz32000, [32, 320], 0.0),
                Mode::SingleChannel,
                false,
                300,
            ),
            (Hz22050, at(Hz22050, 64), Mode::Stereo, true, 77),
            (Hz8000, at(Hz8000, 16), Mode::SingleChannel, false, 8),
        ];
        for (fs, rate, mode, crc_protected, per_frame) in cases {
            let case = format!("{fs:?}, {per_frame} bits a frame");
            let config = StreamConfig {
                rate,
                sampling_frequency: fs,
                mode,
                crc_protected,
                flags: Flags::default(),
            };
            let pcm = varied(fs, 1.0);
            let channels: [&[f64]; 2] = [&pcm[0], &pcm[1]];
            let channels = &channels[..mode.channels()];
            let frames = (pcm[0].len() + DELAY).div_ceil(fs.version().samples_per_frame());
            // Enough for two thirds of the frames, headers among it.
            let mut seed = 3;
            let mut data: Vec<u8> = (0..frames * per_frame * 2 / 3 / 8)
                .map(|_| (noise(&mut seed) * 128.0 + 128.0) as u8)
                .collect();
            data[..8].copy_from_slice(&[0xFF, 0xFB, 0x90, 0x64, 0xFF, 0xF3, 0x48, 0xC4]);
            let bits: Vec<bool> = (0..8 * data.len())
                .map(|i| data[i / 8] >> (7 - i % 8) & 1 == 1)
                .collect();

            let mut encoder = Encoder::new(config).with_ancillary(per_frame);
            encoder.push_ancillary(&data);
            let mut stream = Vec::new();
            encoder.encode(channels, &mut stream);
            assert!(encoder.ancillary_pending() < 8 * data.len(), "{case}");
            encoder.finish(&mut stream);

            let ancillary = stream_check::ancillary(&stream);
            assert_eq!(ancillary.len(), frames, "{case}");
            let mut sent = 0;
            for (k, frame) in ancillary.iter().enumerate() {
                let carried = per_frame.min(bits.len() - sent);
                assert!(frame.len() >= carried, "{case}, frame {k}");
                assert_eq!(
                    frame[..carried],
                    bits[sent..sent + carried],
                    "{case}, frame {k}"
                );
                assert!(!frame[carried..].contains(&true), "{case}, frame {k}");
                sent += carried;
            }
            assert_eq!(sent, bits.len(), "{case}");

            let plain = stream_check::ancillary(&encode_pcm(&config, channels));
            assert!(plain.iter().flatten().all(|&bit| !bit), "{case}");
        }
    }

    /// Ancillary data that fills the shortest frame's main data slot is
    /// allowed, and leaves its audio no bits of its own; any more is not.
    #[test]
    fn ancillary_data_fits_the_shortest_frame() {
        let config = StreamConfig {
            rate: Rate::Constant(Bitrate::from_kbps(Version::Mpeg1, 32).unwrap()),
            sampling_frequency: SamplingFrequency::Hz48000,
            mode: Mode::SingleChannel,
            crc_protected: false,
            flags: Flags::default(),
        };
        // 96 bytes, less the header and 17 of side information.
        let slot = 8 * (96 - 4 - 17);
        let pcm = &varied(SamplingFrequency::Hz48000, 0.5)[0];
        let mut encoder = Encoder::new(config).with_ancillary(slot);
        encoder.push_ancillary(&vec![0x55; 200 * slot / 8]);
        let mut stream = Vec::new();
        encoder.encode(&[pcm], &mut stream);
        encoder.finish(&mut stream);
        for frame in stream_check::ancillary(&stream) {
            assert!(frame.len() >= slot);
        }
        let result = std::panic::catch_unwind(|| Encoder::new(config).with_ancillary(slot + 1));
        assert!(result.is_err());
    }

    /// The flags mark every frame, the Xing/Info tag's too, and change
    /// nothing else but the `crc_check` that covers them: `private_bit` is
    /// the last bit of the header's third byte, and `copyright` and
    /// `original/copy` the fifth and sixth of its fourth (clause 2.4.1.3).
    #[test]
    fn flags_mark_every_frame_and_change_nothing_else() {
        use SamplingFrequency::*;
        let at = |fs: SamplingFrequency, kbps| {
            Rate::Constant(Bitrate::from_kbps(fs.version(), kbps).unwrap())
        };
        let cases = [
            (Hz44100, at(Hz44100, 128), MS, true),
            (
                Hz32000,
                variable(Hz32000, [32, 320], 0.0),
                Mode::SingleChannel,
                false,
            ),
            (Hz22050, at(Hz22050, 64), Mode::Stereo, true),
        ];
        let flags = Flags {
            private: true,
            copyright: true,
            original: true,
        };
        for (fs, rate, mode, crc_protected) in cases {
            let pcm = varied(fs, 1.0);
            let config = |flags| StreamConfig {
                rate,
                sampling_frequency: fs,
                mode,
                crc_protected,
                flags,
            };
            let plain = encode(config(Flags::default()), &pcm);
            let mut marked = encode(config(flags), &pcm);
            assert_eq!(marked.len(), plain.len(), "{fs:?}");
            let mut at = 0;
            for frame in stream_check::frames(&plain) {
                assert_eq!(plain[at + 2] & 1, 0, "{fs:?} at {at}");
                assert_eq!(plain[at + 3] >> 2 & 0b11, 0b00, "{fs:?} at {at}");
                assert_eq!(marked[at + 2] & 1, 1, "{fs:?} at {at}");
                assert_eq!(marked[at + 3] >> 2 & 0b11, 0b11, "{fs:?} at {at}");
                marked[at + 2] &= !1;
                marked[at + 3] &= !0b1100;
                if crc_protected {
                    marked[at + 4..at + 6].copy_from_slice(&plain[at + 4..at + 6]);
                }
                at += frame.len;
            }
            assert!(marked == plain, "{fs:?}");

            let channels: [&[f64]; 2] = [&pcm[0], &pcm[1]];
            let tagged = encode_pcm_with_info_tag(&config(flags), &channels[..mode.channels()]);
            assert_eq!(tagged[2] & 1, 1, "{fs:?}");
            assert_eq!(tagged[3] >> 2 & 0b11, 0b11, "{fs:?}");
        }
    }

    /// At a variable bitrate, frames switch among the bitrates allowed,
    /// silence takes the lowest and loud passages more, no frame is padded, and every frame's main data keeps to the rules of the bit
    /// reservoir: where it may begin, and what the buffer leaves beside the
    /// frame, down to none for frames longer than the buffer at 320 kbit/s
    /// and 32 kHz.
    #[test]
    fn variable_bitrate_streams_keep_the_rules_of_the_reservoir() {
        use SamplingFrequency::*;
        let cases = [
            (Hz44100, [32, 320], MS, false),
            (Hz32000, [96, 320], Mode::Stereo, true),
            (Hz48000, [32, 192], Mode::SingleChannel, false),
            (
                Hz22050,
                [8, 160],
                Mode::JointStereo {
                    intensity_stereo: true,
                    ms_stereo: true,
                },
                false,
            ),
            (Hz24000, [32, 64], Mode::Stereo, true),
            (Hz8000, [8, 64], Mode::SingleChannel, false),
        ];
        std::thread::scope(|scope| {
            for (fs, kbps, mode, crc_protected) in cases {
                scope.spawn(move || {
                    let config = StreamConfig {
                        rate: variable(fs, kbps, 0.0),
                        sampling_frequency: fs,
                        mode,
                        crc_protected,
                        flags: Flags::default(),
                    };
                    let input = varied(fs, 1.0);
                    let frames = stream_check::frames(&encode(config, &input));
                    let samples = fs.version().samples_per_frame();
                    let case = format!("{fs:?} from {} to {} kbit/s", kbps[0], kbps[1]);
                    let expected = (input[0].len() + DELAY).div_ceil(samples);
                    assert_eq!(frames.len(), expected, "{case}");
                    assert!(frames.iter().all(|f| !f.padding), "{case}");
                    let kbps_of = |f: &stream_check::Frame| f.bitrate.bits_per_second() / 1000;
                    let used: std::collections::BTreeSet<u32> =
                        frames.iter().map(kbps_of).collect();
                    assert_eq!(used.first(), Some(&kbps[0]), "{case}: {used:?}");
                    assert!(used.len() >= 2, "{case}: {used:?}");
                    assert!(used.last() <= Some(&kbps[1]), "{case}: {used:?}");
                    // The frames of the silence at the start, short of the
                    // one whose block the noise after it reaches into.
                    let silent = (0.15 * f64::from(fs.hz())) as usize / samples - 1;
                    assert!(
                        frames[..silent].iter().all(|f| kbps_of(f) == kbps[0]),
                        "{case}"
                    );
                });
            }
        });
    }

    /// In free format every frame has `bitrate_index` 0 and `N` or `N + 1`
    /// slots, padded as at a bitrate of the table, keeps to the rules of the
    /// bit reservoir, from frames shorter than any of the table to frames
    /// longer than the decoder's buffer. Where its frames hold the
    /// Xing/Info tag, the tag's frame is in free format too; where they do
    /// not, the stream has none.
    #[test]
    fn free_format_streams_keep_their_bitrate() {
        use SamplingFrequency::*;
        let cases = [
            (Hz44100, 100_000, Mode::Stereo, false, true),
            (Hz48000, 12_667, Mode::SingleChannel, true, false),
            (Hz32000, 319_999, MS, false, true),
            (Hz22050, 50_000, MS, true, false),
            (Hz24000, 8_001, Mode::SingleChannel, false, false),
            (Hz16000, 159_999, Mode::Stereo, true, true),
            (Hz11025, 33_333, MS, false, true),
            (Hz8000, 20_000, Mode::SingleChannel, true, true),
        ];
        std::thread::scope(|scope| {
            for (fs, bits_per_second, mode, crc_protected, tagged) in cases {
                scope.spawn(move || {
                    let bitrate = Bitrate::free(fs.version(), bits_per_second).unwrap();
                    let config = StreamConfig {
                        rate: Rate::Constant(bitrate),
                        sampling_frequency: fs,
                        mode,
                        crc_protected,
                        flags: Flags::default(),
                    };
                    let case = format!("{fs:?} at {bits_per_second} bit/s");
                    let input = varied(fs, 1.0);
                    let mut encoder = Encoder::with_info_tag(config);
                    let mut stream = Vec::new();
                    match mode {
                        Mode::SingleChannel => encoder.encode(&[&input[0]], &mut stream),
                        _ => encoder.encode(&[&input[0], &input[1]], &mut stream),
                    }
                    let tag = encoder.finish(&mut stream);
                    assert_eq!(tag.is_some(), tagged, "{case}");
                    if let Some(tag) = tag {
                        stream[..tag.len()].copy_from_slice(&tag);
                    }
                    let frames = stream_check::free_frames(&stream, bitrate);
                    let samples = fs.version().samples_per_frame();
                    let expected = (input[0].len() + DELAY).div_ceil(samples);
                    assert_eq!(frames.len(), expected + usize::from(tagged), "{case}");
                    // After k frames the stream holds floor(k * N) slots,
                    // with N as exact as 144 * bit_rate / fs (72 at the lower
                    // sampling frequencies) can be; the tag's frame, which
                    // is not padded, takes the place of the first.
                    let numerator = samples as u64 / 8 * u64::from(bits_per_second);
                    let hz = u64::from(fs.hz());
                    let mut total = 0;
                    for (k, frame) in frames.iter().enumerate() {
                        total += frame.len as u64;
                        let k = (k + 1 - usize::from(tagged)) as u64;
                        let floor = usize::from(tagged) as u64 * (numerator / hz);
                        assert_eq!(total, floor + k * numerator / hz, "{case}, frame {k}");
                    }
                    if bitrate.frame_len(fs, true) * 8 > 7680 {
                        assert!(frames.iter().all(|f| f.main_data_begin == 0), "{case}");
                    } else {
                        assert!(frames.iter().any(|f| f.main_data_begin > 0), "{case}");
                    }
                });
            }
        });
    }

    /// Allowing more noise gives smaller streams, and less noise larger.
    #[test]
    fn the_noise_offset_trades_size_for_noise() {
        let fs = SamplingFrequency::Hz44100;
        let input = varied(fs, 0.6);
        let lens: Vec<usize> = [-6.0, 0.0, 6.0]
            .map(|offset| {
                let config = StreamConfig {
                    rate: variable(fs, [32, 320], offset),
                    sampling_frequency: fs,
                    mode: Mode::SingleChannel,
                    crc_protected: false,
                    flags: Flags::default(),
                };
                encode(config, &input).len()
            })
            .to_vec();
        assert!(lens.windows(2).all(|w| w[0] > w[1]), "{lens:?}");
    }

    /// A mean bitrate between two of the table switches between them, and
    /// the stream is as long as its schedule makes it.
    #[test]
    fn a_mean_bitrate_switches_between_its_neighbours() {
        for (fs, kbps, neighbours) in [
            (SamplingFrequency::Hz44100, 150, [128, 160]),
            (SamplingFrequency::Hz22050, 100, [96, 112]),
        ] {
            let mean = rate::MeanRate::new(fs.version(), kbps * 1000).unwrap();
            let config = StreamConfig {
                rate: Rate::Mean(mean),
                sampling_frequency: fs,
                mode: MS,
                crc_protected: false,
                flags: Flags::default(),
            };
            let frames = stream_check::frames(&encode(config, &varied(fs, 0.5)));
            let mut schedule = MeanRateSchedule::new(mean, fs);
            for (k, frame) in frames.iter().enumerate() {
                let (bitrate, padding) = schedule.next_frame();
                assert_eq!(
                    (frame.bitrate, frame.padding),
                    (bitrate, padding),
                    "frame {k}"
                );
            }
            let used: std::collections::BTreeSet<u32> = (frames.iter())
                .map(|f| f.bitrate.bits_per_second() / 1000)
                .collect();
            assert_eq!(used.into_iter().collect::<Vec<_>>(), neighbours, "{fs:?}");
        }
    }

    /// At a variable bitrate too, the stream does not depend on how the
    /// input is divided between calls.
    #[test]
    fn variable_streaming_does_not_depend_on_how_the_input_is_divided() {
        let fs = SamplingFrequency::Hz44100;
        let config = StreamConfig {
            rate: variable(fs, [32, 320], 0.0),
            sampling_frequency: fs,
            mode: MS,
            crc_protected: false,
            flags: Flags::default(),
        };
        let [left, right] = varied(fs, 0.5);
        let whole = encode_pcm(&config, &[&left, &right]);
        for chunk in [1000, Version::Mpeg1.samples_per_frame() + 1] {
            let mut encoder = Encoder::new(config);
            let mut out = Vec::new();
            for (left, right) in left.chunks(chunk).zip(right.chunks(chunk)) {
                encoder.encode(&[left, right], &mut out);
            }
            encoder.finish(&mut out);
            assert!(out == whole, "chunks of {chunk}");
        }
    }
}
