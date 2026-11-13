//! MPEG-1 Audio Layer III.

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
use header::{Bitrate, Emphasis, Header, Mode, PaddingSchedule, SamplingFrequency};
use huffman::{QuadTable, SpectrumCoding};
use loops::scfsi::{self, GranuleSummary};
use psychoacoustic::switching::{BlockSwitching, SwitchedMasking};
use reservoir::Reservoir;
use scalefactors::Scalefactors;
use side_info::{GRANULES, GranuleChannel, SideInfo, Window};

/// Two granules of 576 samples per channel (clause 2.4.2.1).
pub const SAMPLES_PER_FRAME: usize = 1152;

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
            scalefactors: Scalefactors::new(block_type),
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
        self.frame_header(padding, false)
    }

    /// Whether the configured mode allows MS stereo: joint stereo with
    /// `ms_stereo` set.
    ///
    /// # Panics
    ///
    /// If it asks for intensity stereo, which is not implemented.
    fn ms_stereo_allowed(&self) -> bool {
        match self.mode {
            Mode::JointStereo {
                intensity_stereo,
                ms_stereo,
            } => {
                assert!(!intensity_stereo, "intensity stereo is not implemented");
                ms_stereo
            }
            _ => false,
        }
    }

    /// The header of a frame, with `mode_extension` switching MS stereo on
    /// for this frame alone if `ms_stereo` is set.
    fn frame_header(&self, padding: bool, ms_stereo: bool) -> Header {
        let mode = match self.mode {
            Mode::JointStereo { .. } => Mode::JointStereo {
                intensity_stereo: false,
                ms_stereo,
            },
            mode => {
                assert!(!ms_stereo, "MS stereo needs the joint stereo mode");
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
    8 * (header.frame_len() - Header::LEN_BYTES - crc - SideInfo::len_bytes(header.mode.channels()))
}

/// The side information and main data of one frame of granules, one entry
/// per channel each.
fn frame_contents(
    sampling_frequency: SamplingFrequency,
    frame: &[Vec<QuantizedGranule>],
    main_data_begin: usize,
) -> (SideInfo, Vec<u8>) {
    let mut side_info = SideInfo {
        main_data_begin: u16::try_from(main_data_begin).expect("main_data_begin is 9 bits"),
        ..SideInfo::default()
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
/// into frames of two granules each.
///
/// Main data is not shared between frames: `main_data_begin` is always 0.
///
/// # Panics
///
/// If the granules do not divide into frames for the configured channels, or
/// a frame's granules do not fit in it.
pub fn encode_granules(config: &StreamConfig, granules: &[Vec<QuantizedGranule>]) -> Vec<u8> {
    let channels = config.mode.channels();
    assert!(
        granules.len().is_multiple_of(GRANULES),
        "a frame holds {GRANULES} granules"
    );
    assert!(
        granules.iter().all(|g| g.len() == channels),
        "each granule needs one entry per channel"
    );
    let mut padding = PaddingSchedule::new(config.bitrate, config.sampling_frequency);
    let mut writer = FrameWriter::default();
    for frame in granules.chunks_exact(GRANULES) {
        let (side_info, main_data) = frame_contents(config.sampling_frequency, frame, 0);
        writer.write(
            &config.header(padding.next_padding()),
            &side_info,
            &main_data,
        );
    }
    writer.out
}

/// Encodes PCM, one slice of samples (nominally -1.0 to 1.0) per channel, at
/// the configured bitrate.
///
/// Psychoacoustic model 2 chooses each granule's block type, switching to
/// short blocks for attacks, and sets the distortion each scalefactor band
/// allows and the perceptual entropy of each granule. Each granule of each
/// channel gets an equal share of its frame's main data, the `mean_bits` of
/// clause C.1.5.3.4, plus what the bit reservoir lends it, and the iteration
/// loops shape the quantisation noise to the allowed distortion within those
/// bits. The input is padded with silence to whole frames, plus one more
/// frame to flush the filterbank.
///
/// # Panics
///
/// If the channel count does not match the configured mode.
pub fn encode_pcm(config: &StreamConfig, channels: &[&[f64]]) -> Vec<u8> {
    let fs = config.sampling_frequency;
    let ms_allowed = config.ms_stereo_allowed();
    let mut models: Vec<_> = channels.iter().map(|_| BlockSwitching::new(fs)).collect();
    // Each channel's first granule of the frame, for scfsi in the second.
    let mut first: Vec<Option<(GranuleSummary, Scalefactors)>> = vec![None; channels.len()];
    let reservoir = Reservoir::new(config.header(true).frame_len());
    encode_filtered(
        config,
        channels,
        reservoir,
        |channel, pcm| {
            let masking = models[channel].granule(pcm)?;
            Some((masking.block_type(), (masking, Vec::new())))
        },
        |frame| {
            for granule in frame.iter_mut() {
                for filtered in granule.iter_mut() {
                    let (masking, xmin) = &mut filtered.analysis;
                    *xmin = masking.allowed_distortion(&filtered.xr, fs);
                }
            }
            ms_allowed && middle_side_frame(frame, fs)
        },
        |channel, gr, (masking, xmin), xr, mean_bits, reservoir| {
            let block_type = masking.block_type();
            let available_bits = reservoir
                .granule_bits(mean_bits, masking.perceptual_entropy())
                .min(MAX_PART2_3_LENGTH);
            let summary = GranuleSummary::new(xr, &xmin, block_type, fs);
            let reuse = match (gr, &first[channel]) {
                (1, Some((first_summary, first_scalefactors))) => Some(loops::Reuse {
                    first: first_scalefactors,
                    scfsi: scfsi::select(first_summary, &summary),
                }),
                _ => None,
            };
            let granule = loops::quantize_granule(xr, &xmin, block_type, available_bits, fs, reuse);
            first[channel] = (gr == 0).then(|| (summary, granule.scalefactors.clone()));
            granule
        },
    )
}

/// A granule's psychoacoustic analysis and the allowed distortion of each
/// of its scalefactors, in the order they are sent.
type Analysis = (SwitchedMasking, Vec<f64>);

/// Decides whether to code a stereo frame in MS stereo (Annex G.2), and if
/// so turns its left and right lines into middle and side lines, with the
/// allowed distortion of each.
///
/// The normative decoding needs both channels of a granule to have the same
/// block type in MS stereo (clause 2.4.3.4.9). Spec gap: Annex G does not
/// say what to do when they do not; such frames are coded in left/right
/// stereo. The perceptual entropy each channel's bits are reckoned from
/// stays that of the left and right channels.
fn middle_side_frame(
    frame: &mut [Vec<Filtered<Analysis>>],
    sampling_frequency: SamplingFrequency,
) -> bool {
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
    if !same_types || !stereo::ms_stereo(&spectra) {
        return false;
    }
    for granule in frame.iter_mut() {
        let [left, right] = &mut granule[..] else {
            unreachable!("MS stereo has two channels")
        };
        let (middle, mut side) = stereo::middle_side(&left.xr, &right.xr);
        stereo::limit_side(&mut side, left.block_type, sampling_frequency);
        let xmin = stereo::middle_side_distortion(&left.analysis.1, &right.analysis.1);
        (left.xr, right.xr) = (middle, side);
        left.analysis.1.clone_from(&xmin);
        right.analysis.1 = xmin;
    }
    true
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
/// As [`encode_pcm_with_global_gain`].
pub fn encode_with_block_types(
    config: &StreamConfig,
    global_gain: u8,
    channels: &[&[f64]],
    mut block_type: impl FnMut(usize, usize) -> BlockType,
) -> Vec<u8> {
    let mut next = vec![0usize; channels.len()];
    encode_filtered(
        config,
        channels,
        Reservoir::disabled(),
        |channel, _| {
            // Called once ahead of the first granule, whose type it gives
            // on the second call.
            let granule = next[channel].checked_sub(1);
            next[channel] += 1;
            granule.map(|g| {
                let block_type = block_type(channel, g);
                (block_type, block_type)
            })
        },
        |frame| {
            if !config.ms_stereo_allowed() {
                return false;
            }
            for granule in frame.iter_mut() {
                let [left, right] = &mut granule[..] else {
                    unreachable!("MS stereo has two channels")
                };
                assert_eq!(
                    left.block_type, right.block_type,
                    "MS stereo needs one block type"
                );
                (left.xr, right.xr) = stereo::middle_side(&left.xr, &right.xr);
            }
            true
        },
        |_, _, block_type, xr, _, _| {
            let values = loops::quantize(xr, global_gain)
                .unwrap_or_else(|| panic!("a granule is too loud for global_gain {global_gain}"));
            QuantizedGranule::new(global_gain, values, block_type, config.sampling_frequency)
        },
    )
}

/// One granule of one channel, analysed and filtered, before quantisation.
struct Filtered<A> {
    block_type: BlockType,
    analysis: A,
    /// The lines in the order they are sent.
    xr: [f64; GRANULE_LINES],
}

/// Runs each channel through its filterbank granule by granule, quantises
/// each granule with `quantize`, and writes the frames.
///
/// `look_ahead` is given each channel's PCM one granule ahead of the
/// filterbank, starting with the first granule before any is filtered, and
/// returns the block type and analysis of the granule before the one it is
/// given: `None` only for the first. `joint` is given each frame's granules,
/// indexed `[gr][ch]`, once all are filtered, may change them, and returns
/// whether the frame is coded in MS stereo. `quantize` is then given the
/// channel, the granule's index in its frame, its analysis, its lines, its
/// share of the frame's main data and the bit reservoir, which accounts for
/// the bits it then uses.
fn encode_filtered<A>(
    config: &StreamConfig,
    channels: &[&[f64]],
    mut reservoir: Reservoir,
    mut look_ahead: impl FnMut(usize, &[f64; GRANULE_LINES]) -> Option<(BlockType, A)>,
    mut joint: impl FnMut(&mut [Vec<Filtered<A>>]) -> bool,
    mut quantize: impl FnMut(
        usize,
        usize,
        A,
        &[f64; GRANULE_LINES],
        usize,
        &Reservoir,
    ) -> QuantizedGranule,
) -> Vec<u8> {
    assert_eq!(
        channels.len(),
        config.mode.channels(),
        "one slice per channel"
    );
    let samples = channels.iter().map(|c| c.len()).max().unwrap_or(0);
    let frames = samples.div_ceil(SAMPLES_PER_FRAME) + 1;
    let granule_pcm = |channel: usize, granule: usize| -> [f64; GRANULE_LINES] {
        let offset = granule * GRANULE_LINES;
        std::array::from_fn(|t| channels[channel].get(offset + t).copied().unwrap_or(0.0))
    };
    let mut filterbanks: Vec<_> = channels
        .iter()
        .map(|_| filterbank::Filterbank::new())
        .collect();
    for channel in 0..channels.len() {
        assert!(look_ahead(channel, &granule_pcm(channel, 0)).is_none());
    }
    let fs = config.sampling_frequency;
    let mut padding = PaddingSchedule::new(config.bitrate, fs);
    let mut writer = FrameWriter::default();
    let mut main_data_begin = 0;

    for frame in 0..frames {
        let mut filtered: Vec<Vec<Filtered<A>>> = (0..GRANULES)
            .map(|gr| {
                let index = frame * GRANULES + gr;
                (filterbanks.iter_mut().enumerate())
                    .map(|(channel, filterbank)| {
                        let (block_type, analysis) =
                            look_ahead(channel, &granule_pcm(channel, index + 1))
                                .expect("the granule before has been analysed");
                        let xr = filterbank.granule(&granule_pcm(channel, index), block_type);
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
        let ms_stereo = joint(&mut filtered);
        let header = config.frame_header(padding.next_padding(), ms_stereo);
        let slot_bits = main_data_bits(&header);
        let shares = GRANULES * channels.len();
        let mean_bits = slot_bits / shares;
        let mut granules = Vec::with_capacity(GRANULES);
        for (gr, granule) in filtered.into_iter().enumerate() {
            let mut quantized_granule = Vec::with_capacity(channels.len());
            for (channel, filtered) in granule.into_iter().enumerate() {
                let Filtered {
                    block_type,
                    analysis,
                    xr,
                } = filtered;
                let quantized = quantize(channel, gr, analysis, &xr, mean_bits, &reservoir);
                assert_eq!(quantized.block_type, block_type);
                reservoir.granule_done(mean_bits, quantized.bit_len(fs));
                quantized_granule.push(quantized);
            }
            granules.push(quantized_granule);
        }
        let (side_info, main_data) = frame_contents(fs, &granules, main_data_begin);
        writer.write(&header, &side_info, &main_data);
        main_data_begin = reservoir.end_frame(slot_bits - shares * mean_bits);
    }
    writer.out
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
    encode_granules(config, &vec![granule; frames * GRANULES])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(crc_protected: bool) -> StreamConfig {
        StreamConfig {
            bitrate: Bitrate::from_kbps(128).unwrap(),
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

        let side_info = &plain[4..4 + SideInfo::len_bytes(2)];
        assert_eq!(protected[6..6 + side_info.len()], *side_info);
        let expected = crc::crc16(plain[2..4].iter().chain(side_info).copied());
        assert_eq!(protected[4..6], expected.to_be_bytes());
    }

    fn small_header() -> Header {
        Header {
            crc_protected: false,
            bitrate: Bitrate::from_kbps(32).unwrap(),
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
        writer.write(&header, &SideInfo::default(), &[1; 100]);
        let side_info = SideInfo {
            main_data_begin: 20,
            ..SideInfo::default()
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
            ..SideInfo::default()
        };
        FrameWriter::default().write(&small_header(), &side_info, &[]);
    }

    #[test]
    #[should_panic(expected = "exceed")]
    fn rejects_main_data_larger_than_the_frame() {
        let header = Header {
            crc_protected: false,
            bitrate: Bitrate::from_kbps(32).unwrap(),
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
        write_frame(&header, &SideInfo::default(), &main_data, &mut Vec::new());
    }
}
