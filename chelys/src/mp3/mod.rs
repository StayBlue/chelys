// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! MPEG-1 Audio Layer III, and its extension to lower sampling frequencies in
//! ISO/IEC 13818-3.

mod crc;
pub mod filterbank;
pub mod header;
pub mod huffman;
pub mod loops;
pub mod psychoacoustic;
pub mod reservoir;
pub mod scalefactor_bands;
pub mod scalefactors;
pub mod side_info;
pub mod stereo;

use std::collections::VecDeque;
use std::ops::Range;

use crate::bitstream::BitWriter;
use header::{Bitrate, Emphasis, Header, Mode, PaddingSchedule, SamplingFrequency, Version};
use huffman::{QuadTable, SpectrumCoding};
use loops::scfsi::{self, GranuleSummary};
use psychoacoustic::switching::{BlockSwitching, SwitchedMasking};
use reservoir::Reservoir;
use scalefactors::{Scalefactors, ZeroPart};
use side_info::{GranuleChannel, SideInfo, Window};

/// Frequency lines in a granule (clause 2.1.70).
pub const GRANULE_LINES: usize = 576;

/// The window and transform of a granule's block (clause C.1.5.3.3, figure
/// C.7): `block_type` 0 to 3 of clause 2.4.2.7.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockType {
    Normal,
    Start,
    Short,
    Stop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamConfig {
    pub bitrate: Bitrate,
    pub sampling_frequency: SamplingFrequency,
    pub mode: Mode,
    pub crc_protected: bool,
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
    /// 2.4.2.7).
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
        let coding = SpectrumCoding::choose(&values, block_type, &bands);
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
            BlockType::Short => side_info::BlockType::Short,
            BlockType::Stop => side_info::BlockType::End,
        };
        Window::Switched {
            block_type,
            mixed_block_flag: false,
            table_select: [coding.table_select[0], coding.table_select[1]],
            subblock_gain: [0; 3],
        }
    }
}

/// The largest `part2_3_length`, a 12-bit field (clause 2.4.1.7).
const MAX_PART2_3_LENGTH: usize = (1 << 12) - 1;

impl StreamConfig {
    fn header(&self, padding: bool) -> Header {
        self.frame_header(padding, false, false)
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

    /// The header of a frame, with `mode_extension` switching MS stereo and
    /// intensity stereo on for this frame alone if `ms_stereo` and
    /// `intensity_stereo` are set.
    fn frame_header(&self, padding: bool, ms_stereo: bool, intensity_stereo: bool) -> Header {
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
            bitrate: self.bitrate,
            sampling_frequency: self.sampling_frequency,
            padding,
            private: false,
            mode,
            copyright: false,
            original: false,
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
/// per channel each.
fn frame_contents(
    sampling_frequency: SamplingFrequency,
    frame: &[Vec<QuantizedGranule>],
    main_data_begin: usize,
) -> (SideInfo, Vec<u8>) {
    let version = sampling_frequency.version();
    assert!(
        main_data_begin < 1 << version.main_data_begin_bits(),
        "main_data_begin {main_data_begin} overflows its field"
    );
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
    (side_info, main_data.into_bytes())
}

/// Encodes quantised granules, given in order with one entry per channel,
/// into frames of two granules each, or one at the lower sampling
/// frequencies.
///
/// Main data is not shared between frames: `main_data_begin` is always 0. In
/// joint stereo every frame has MS stereo and intensity stereo on as the
/// configured mode sets them; in intensity stereo the right channel's
/// scalefactors must give its zero part
/// ([`Scalefactors::intensity`](scalefactors::Scalefactors::intensity)).
///
/// # Panics
///
/// If the granules do not divide into frames for the configured channels, or
/// a frame's granules do not fit in it.
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
    let mut padding = PaddingSchedule::new(config.bitrate, config.sampling_frequency);
    let mut writer = FrameWriter::default();
    for frame in granules.chunks_exact(per_frame) {
        let (side_info, main_data) = frame_contents(config.sampling_frequency, frame, 0);
        writer.write(
            &config.frame_header(padding.next_padding(), ms_stereo, intensity_stereo),
            &side_info,
            &main_data,
        );
    }
    writer.out
}

/// Encodes PCM, one slice of samples (nominally -1.0 to 1.0) per channel, in
/// one go, with an [`Encoder`].
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

/// Encodes PCM at the configured bitrate as it arrives.
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
/// The stream does not depend on how the input is divided between calls to
/// [`encode`](Self::encode).
pub struct Encoder(FrameEncoder<ModelCoding>);

impl Encoder {
    pub fn new(config: StreamConfig) -> Self {
        let fs = config.sampling_frequency;
        let (ms_allowed, intensity_allowed) = config.joint_stereo();
        let coding = ModelCoding {
            models: (0..config.mode.channels())
                .map(|_| BlockSwitching::new(fs))
                .collect(),
            ms_allowed,
            intensity_allowed,
            sampling_frequency: fs,
        };
        let reservoir = Reservoir::new(config.header(true).frame_len(), fs.version());
        Self(FrameEncoder::new(config, coding, reservoir))
    }

    /// Encodes `channels`, one slice of samples (nominally -1.0 to 1.0) per
    /// channel, and appends to `out` the bytes of the stream that are final.
    ///
    /// A frame is coded once the granule after it has arrived, as its block
    /// type depends on whether that granule has an attack, and its bytes are
    /// final once no later frame's main data can begin in them: the bytes
    /// the bit reservoir holds, at most 511 of main data, and the frames
    /// after them stay behind until later calls.
    ///
    /// # Panics
    ///
    /// If the channel count does not match the configured mode, or the
    /// channels have different numbers of samples.
    pub fn encode(&mut self, channels: &[&[f64]], out: &mut Vec<u8>) {
        self.0.encode(channels, out);
    }

    /// Ends the stream: pads the input with silence to whole frames, plus
    /// one more frame to flush the filterbank, and appends the rest of the
    /// stream to `out`.
    pub fn finish(self, out: &mut Vec<u8>) {
        self.0.finish(out);
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

    fn quantize(
        &mut self,
        mut frame: Vec<Vec<Filtered<Analysis>>>,
        mean_bits: usize,
        reservoir: &mut Reservoir,
    ) -> CodedFrame {
        let fs = self.sampling_frequency;
        for granule in frame.iter_mut() {
            for filtered in granule.iter_mut() {
                let (masking, xmin) = &mut filtered.analysis;
                *xmin = masking.allowed_distortion(&filtered.xr, fs);
            }
        }
        let ms_stereo = self.ms_allowed && middle_side_frame(&frame);
        let bounds = if self.intensity_allowed && intensity_frame(&frame) {
            stereo::intensity_bounds(fs).to_vec()
        } else {
            Vec::new()
        };
        let to_quantize = FrameToQuantize {
            frame: &frame,
            ms_stereo,
            mean_bits,
            sampling_frequency: fs,
        };
        let plain = quantize_frame(&to_quantize, None, reservoir, IfOutsideAllowance::Finish)
            .expect("a finished frame");
        let tries: Vec<_> = if plain.within_allowance {
            Vec::new()
        } else {
            bounds
                .into_iter()
                .filter_map(|bound| IntensityTry::new(&frame, &plain.granules, bound, fs))
                .collect()
        };
        let kept = search_intensity(&tries, |intensity, if_outside| {
            quantize_frame(&to_quantize, Some(intensity), reservoir, if_outside)
        });
        let intensity_stereo = kept.is_some();
        let FrameTry {
            granules,
            reservoir: kept_reservoir,
            ..
        } = kept.unwrap_or(plain);
        *reservoir = kept_reservoir;
        CodedFrame {
            granules,
            ms_stereo,
            intensity_stereo,
        }
    }
}

/// Annex G.1: if the frame needs more bits than it has, more and more bands
/// are coded in intensity stereo, down to the lowest bound, and the last
/// tried is kept even if the frame still needs more. Bounds that would not
/// lower the bits are not tried (see [`IntensityTry::new`]).
///
/// Given the tries worth making, from the fewest bands to the most, and a
/// function that quantises the frame for a try, returns the try kept: `None`
/// if there are no tries, or if the function gives up on the last, as
/// [`quantize_frame`] can whatever the distortion. A try before the last is kept only if the whole
/// frame is within its allowed distortion, so it is given up at the first
/// granule that is not; the frame then comes out as if it had been finished
/// and passed over.
fn search_intensity<Try, Frame>(
    tries: &[Try],
    mut quantize: impl FnMut(&Try, IfOutsideAllowance) -> Option<Frame>,
) -> Option<Frame> {
    let (last, earlier) = tries.split_last()?;
    earlier
        .iter()
        .find_map(|intensity| quantize(intensity, IfOutsideAllowance::GiveUp))
        .or_else(|| quantize(last, IfOutsideAllowance::Finish))
}

/// A frame to quantise, the same for each try at coding it: its granules,
/// indexed `[gr][ch]`, whether it is in MS stereo, and the share of its main
/// data of each granule of each channel.
struct FrameToQuantize<'a> {
    frame: &'a [Vec<Filtered<Analysis>>],
    ms_stereo: bool,
    mean_bits: usize,
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
    /// Whether every granule was within its allowed distortion.
    within_allowance: bool,
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

/// Whether a stereo frame may be coded in intensity stereo.
///
/// Spec gap: the standard asks for the same block type in both channels
/// only for MS stereo (clause 2.4.3.4.9), and does not say how intensity
/// stereo applies to short blocks: clause 2.4.3.4.9.3 finds the bound from
/// the right channel's zero part, and short blocks are sent by band and
/// then window (clause 2.4.3.4.8), so that the zero part could begin in any
/// window of a band. As intensity stereo moves the left channel's lines
/// into the right channel's synthesis, whose window should then be the
/// same, frames are coded in it only if each granule has the same long
/// block type in both channels.
///
/// At the lower sampling frequencies clause 2.4.3.2 of ISO/IEC 13818-3 does
/// settle the bound of short blocks: it is found for each window. Its 1998
/// edition calls this "like" ISO/IEC 11172-3, 2.4.3.4, where its 1995
/// edition said "unlike", but clause 2.4.3.4 of ISO/IEC 11172-3 says
/// nothing of windows, so it leaves the gap above open. Short blocks are
/// kept out of intensity stereo at the lower sampling frequencies too: the
/// positions, the bits shared between the channels and the bounds tried
/// here are all worked out for long scalefactor bands.
fn intensity_frame(frame: &[Vec<Filtered<Analysis>>]) -> bool {
    frame.iter().all(|granule| {
        let block_type = granule[0].block_type;
        block_type != BlockType::Short && granule[1].block_type == block_type
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
    /// The try at coding `frame` in intensity stereo from long scalefactor
    /// band `bound`, given its granules as coded without it, `plain`.
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
        bound: usize,
        sampling_frequency: SamplingFrequency,
    ) -> Option<Self> {
        let fs = sampling_frequency;
        let bands = scalefactor_bands::long_bands(fs);
        let start = bands[bound].start;
        let positions: Vec<_> = frame
            .iter()
            .map(|granule| {
                stereo::IntensityPositions::new(&granule[0].xr, &granule[1].xr, bound, fs)
            })
            .collect();
        let right_bits: Vec<usize> = (plain.iter().zip(&positions))
            .map(|(granule, positions)| {
                let right = &granule[1];
                let below = right.coding.bit_count_below(&right.values, bands, start);
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
/// one by all of them. So the encoder sends no marks there: a frame whose
/// right channel would need one, with no nonzero value in the band just
/// below the bound, is coded from a lower bound or without intensity stereo.
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
        sampling_frequency: fs,
    } = to_quantize;
    let mut reservoir = reservoir.clone();
    // Each channel's first granule, for scfsi in the second.
    let mut first: Vec<Option<(GranuleSummary, Scalefactors)>> = vec![None; frame[0].len()];
    let mut within_allowance = true;
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
                for (sb, xmin) in left_xmin.iter_mut().enumerate().skip(positions.bound()) {
                    *xmin = positions.sum_distortion(sb, left_alone[sb], right_alone[sb]);
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
            let outcome =
                loops::quantize_granule(xr, xmin, block_type, available_bits, fs, reuse, intensity);
            if gr == 0 {
                first[channel] = Some((summary, outcome.granule.scalefactors.clone()));
            }
            let zero_part = outcome.granule.scalefactors.intensity;
            if fs.version() == Version::Mpeg2 && zero_part.is_some_and(ZeroPart::marks_any) {
                return None;
            }
            reservoir.granule_done(mean_bits, outcome.granule.bit_len(fs));
            if !outcome.within_allowance && if_outside == IfOutsideAllowance::GiveUp {
                return None;
            }
            within_allowance &= outcome.within_allowance;
            quantized.push(outcome.granule);
        }
        granules.push(quantized);
    }
    Some(FrameTry {
        granules,
        within_allowance,
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
/// follow each other as figure C.7 allows. If the mode allows MS stereo,
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
        mut frame: Vec<Vec<Filtered<()>>>,
        mean_bits: usize,
        reservoir: &mut Reservoir,
    ) -> CodedFrame {
        let (global_gain, fs) = (self.global_gain, self.sampling_frequency);
        let mut granules = Vec::with_capacity(frame.len());
        for granule in frame.iter_mut() {
            if self.ms_stereo {
                let [left, right] = &mut granule[..] else {
                    unreachable!("MS stereo has two channels")
                };
                assert_eq!(
                    left.block_type, right.block_type,
                    "MS stereo needs one block type"
                );
                (left.xr, right.xr) = stereo::middle_side(&left.xr, &right.xr);
            }
            let quantized: Vec<_> = granule
                .iter()
                .map(|filtered| {
                    let values = loops::quantize(&filtered.xr, global_gain).unwrap_or_else(|| {
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
}

/// How [`FrameEncoder`] analyses and quantises granules.
trait Coding {
    /// What the analysis of a granule gives besides its block type.
    type Analysis;

    /// Given each channel's PCM one granule ahead of the filterbank,
    /// starting with the first granule before any is filtered, returns the
    /// block type and analysis of the granule before the one it is given:
    /// `None` only for the first.
    fn look_ahead(
        &mut self,
        channel: usize,
        pcm: &[f64; GRANULE_LINES],
    ) -> Option<(BlockType, Self::Analysis)>;

    /// Quantises a frame's granules, indexed `[gr][ch]`, given the share of
    /// the frame's main data of each granule of each channel, accounting in
    /// the reservoir for the bits each uses.
    fn quantize(
        &mut self,
        frame: Vec<Vec<Filtered<Self::Analysis>>>,
        mean_bits: usize,
        reservoir: &mut Reservoir,
    ) -> CodedFrame;
}

/// Runs each channel through its filterbank granule by granule as its PCM
/// arrives, quantises each frame as `coding` does, and writes the frames.
struct FrameEncoder<C> {
    config: StreamConfig,
    coding: C,
    reservoir: Reservoir,
    /// One for each channel.
    filterbanks: Vec<filterbank::Filterbank>,
    padding: PaddingSchedule,
    writer: FrameWriter,
    main_data_begin: usize,
    /// Each channel's PCM from the first granule not yet filtered.
    pcm: Vec<Vec<f64>>,
    /// The samples each channel has been given.
    samples: usize,
    /// The frames written.
    frames: usize,
}

impl<C: Coding> FrameEncoder<C> {
    fn new(config: StreamConfig, coding: C, reservoir: Reservoir) -> Self {
        let channels = config.mode.channels();
        Self {
            config,
            coding,
            reservoir,
            filterbanks: (0..channels)
                .map(|_| filterbank::Filterbank::new())
                .collect(),
            padding: PaddingSchedule::new(config.bitrate, config.sampling_frequency),
            writer: FrameWriter::default(),
            main_data_begin: 0,
            pcm: vec![Vec::new(); channels],
            samples: 0,
            frames: 0,
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
        while self.pcm[0].len() - start >= frame_samples + GRANULE_LINES {
            self.encode_frame(start);
            start += frame_samples;
        }
        for pcm in &mut self.pcm {
            pcm.drain(..start);
        }
        self.writer.take_final(out);
    }

    /// Pads the input with silence to whole frames, plus one more frame to
    /// flush the filterbank, codes them, and appends the rest of the stream
    /// to `out`.
    fn finish(mut self, out: &mut Vec<u8>) {
        let frame_samples = self.samples_per_frame();
        let frames = self.samples.div_ceil(frame_samples) + 1;
        let remaining = frames - self.frames;
        for pcm in &mut self.pcm {
            pcm.resize(remaining * frame_samples + GRANULE_LINES, 0.0);
        }
        for frame in 0..remaining {
            self.encode_frame(frame * frame_samples);
        }
        self.writer.finish(out);
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
            ..
        } = self;
        let fs = config.sampling_frequency;
        let granule_pcm = |channel: usize, granule: usize| -> &[f64; GRANULE_LINES] {
            let from = start + granule * GRANULE_LINES;
            pcm[channel][from..from + GRANULE_LINES]
                .try_into()
                .expect("a granule of PCM")
        };
        if self.frames == 0 {
            for channel in 0..pcm.len() {
                assert!(
                    coding
                        .look_ahead(channel, granule_pcm(channel, 0))
                        .is_none()
                );
            }
        }
        let granules = fs.version().granules();
        let filtered: Vec<Vec<Filtered<C::Analysis>>> = (0..granules)
            .map(|gr| {
                (filterbanks.iter_mut().enumerate())
                    .map(|(channel, filterbank)| {
                        let (block_type, analysis) = coding
                            .look_ahead(channel, granule_pcm(channel, gr + 1))
                            .expect("the granule before has been analysed");
                        let xr = filterbank.granule(granule_pcm(channel, gr), block_type);
                        let xr = match block_type {
                            BlockType::Short => scalefactor_bands::short_coding_order(&xr, fs),
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
        // mode_extension does not change the frame's length.
        let padded = self.padding.next_padding();
        let slot_bits = main_data_bits(&self.config.header(padded));
        let shares = granules * self.pcm.len();
        let mean_bits = slot_bits / shares;
        let block_types: Vec<BlockType> = filtered.iter().flatten().map(|f| f.block_type).collect();
        let coded = self
            .coding
            .quantize(filtered, mean_bits, &mut self.reservoir);
        assert!(
            (coded.granules.iter().flatten())
                .map(|q| q.block_type)
                .eq(block_types),
            "the granules keep their block types"
        );
        let header = self
            .config
            .frame_header(padded, coded.ms_stereo, coded.intensity_stereo);
        let (side_info, main_data) = frame_contents(fs, &coded.granules, self.main_data_begin);
        self.writer.write(&header, &side_info, &main_data);
        self.main_data_begin = self.reservoir.end_frame(slot_bits - shares * mean_bits);
        self.writer.keep_unused(self.main_data_begin);
        self.frames += 1;
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
            bitrate: Bitrate::from_kbps(Version::Mpeg1, 128).unwrap(),
            sampling_frequency: SamplingFrequency::Hz44100,
            mode: Mode::Stereo,
            crc_protected,
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
            private: false,
            mode: Mode::SingleChannel,
            copyright: false,
            original: false,
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
            private: false,
            mode: Mode::SingleChannel,
            copyright: false,
            original: false,
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

    /// For every pattern of tries within their allowed distortion, up to the
    /// most a frame can have, the search keeps the first try within it or
    /// else the last, gives up only on tries before the last, and quantises
    /// none after the one it keeps.
    #[test]
    fn intensity_search_keeps_the_first_try_that_fits_or_else_the_last() {
        for count in 0..=stereo::intensity_bounds(SamplingFrequency::Hz44100).len() {
            for pattern in 0..1u32 << count {
                let fits = |index: usize| pattern & 1 << index != 0;
                let tries: Vec<usize> = (0..count).collect();
                let mut calls = Vec::new();
                let kept = search_intensity(&tries, |&index, if_outside| {
                    calls.push((index, if_outside));
                    (fits(index) || if_outside == IfOutsideAllowance::Finish).then_some(index)
                });

                let expected = (0..count).find(|&index| fits(index) || index + 1 == count);
                assert_eq!(kept, expected, "{count} tries, fits {pattern:b}");
                let expected_calls: Vec<_> = expected
                    .into_iter()
                    .flat_map(|kept| 0..=kept)
                    .map(|index| {
                        let last = index + 1 == count;
                        let if_outside = if last {
                            IfOutsideAllowance::Finish
                        } else {
                            IfOutsideAllowance::GiveUp
                        };
                        (index, if_outside)
                    })
                    .collect();
                assert_eq!(calls, expected_calls, "{count} tries, fits {pattern:b}");
            }
        }
    }

    /// A deterministic sequence of samples from -1.0 to 1.0 (an LCG).
    fn noise(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (*seed >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    /// However the input is divided between calls, the encoder writes the
    /// same stream, of whole frames for the input padded with silence plus
    /// one more, and holds back no more than the next frame's main data can
    /// begin in: its `main_data_begin` bytes, and the header and side
    /// information of each frame after the first of them.
    #[test]
    fn streaming_does_not_depend_on_how_the_input_is_divided() {
        let config = StreamConfig {
            bitrate: Bitrate::from_kbps(Version::Mpeg1, 32).unwrap(),
            sampling_frequency: SamplingFrequency::Hz44100,
            mode: Mode::JointStereo {
                intensity_stereo: false,
                ms_stereo: true,
            },
            crc_protected: false,
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
        let mut padding = PaddingSchedule::new(config.bitrate, config.sampling_frequency);
        let frames = samples.div_ceil(Version::Mpeg1.samples_per_frame()) + 1;
        let frame_lens = (0..frames).map(|_| config.header(padding.next_padding()).frame_len());
        assert_eq!(whole.len(), frame_lens.sum::<usize>());

        // Frames without padding have the smallest slots, and so span the
        // most frames for the same bytes.
        let header = config.header(false);
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
                        let begin = encoder.0.main_data_begin;
                        furthest = furthest.max(begin);
                        let held = encoder.0.writer.out.len();
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
}
