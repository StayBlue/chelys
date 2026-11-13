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
/// In joint stereo, frames whose channels are alike are coded in MS stereo
/// if the mode allows it, and frames short of bits are coded in intensity
/// stereo if the mode allows that (Annex G.2).
///
/// # Panics
///
/// If the channel count does not match the configured mode.
pub fn encode_pcm(config: &StreamConfig, channels: &[&[f64]]) -> Vec<u8> {
    let fs = config.sampling_frequency;
    let (ms_allowed, intensity_allowed) = config.joint_stereo();
    let mut models: Vec<_> = channels.iter().map(|_| BlockSwitching::new(fs)).collect();
    let reservoir = Reservoir::new(config.header(true).frame_len());
    encode_filtered(
        config,
        channels,
        reservoir,
        |channel, pcm| {
            let masking = models[channel].granule(pcm)?;
            Some((masking.block_type(), (masking, Vec::new())))
        },
        |mut frame, mean_bits, reservoir| {
            for granule in frame.iter_mut() {
                for filtered in granule.iter_mut() {
                    let (masking, xmin) = &mut filtered.analysis;
                    *xmin = masking.allowed_distortion(&filtered.xr, fs);
                }
            }
            let ms_stereo = ms_allowed && middle_side_frame(&frame);
            let bounds = if intensity_allowed && intensity_frame(&frame) {
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
        },
    )
}

/// Annex G.1: if the frame needs more bits than it has, more and more bands
/// are coded in intensity stereo, down to the lowest bound, and the last
/// tried is kept even if the frame still needs more. Bounds that would not
/// lower the bits are not tried (see [`IntensityTry::new`]).
///
/// Given the tries worth making, from the fewest bands to the most, and a
/// function that quantises the frame for a try, returns the try kept: `None`
/// if there are no tries. A try before the last is kept only if the whole
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
                let mut scalefactors = right.scalefactors.clone();
                scalefactors.scfsi = [false; side_info::SCFSI_BANDS];
                positions.mark(&mut scalefactors, &values, fs);
                let compress = scalefactors
                    .compress()
                    .expect("scalefactors within their limits");
                // Enough for the marks if fewer bits leave more bands zero.
                (scalefactors.part2_length(compress) + below).max(stereo::MAX_POSITION_BITS)
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
/// distortion.
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
                positions.code(&granule[0].xr, &granule[1].xr, [left, right], fs);
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
    mut block_type: impl FnMut(usize, usize) -> BlockType,
) -> Vec<u8> {
    let (ms_stereo, intensity_stereo) = config.joint_stereo();
    assert!(!intensity_stereo, "intensity stereo needs encode_pcm");
    let fs = config.sampling_frequency;
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
        |mut frame, mean_bits, reservoir| {
            let mut granules = Vec::with_capacity(frame.len());
            for granule in frame.iter_mut() {
                if ms_stereo {
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
                        let values =
                            loops::quantize(&filtered.xr, global_gain).unwrap_or_else(|| {
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
                ms_stereo,
                intensity_stereo: false,
            }
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

/// A frame's granules, quantised, with one entry per channel each, and the
/// joint stereo coding they use.
struct CodedFrame {
    granules: Vec<Vec<QuantizedGranule>>,
    ms_stereo: bool,
    intensity_stereo: bool,
}

/// Runs each channel through its filterbank granule by granule, quantises
/// each frame with `quantize`, and writes the frames.
///
/// `look_ahead` is given each channel's PCM one granule ahead of the
/// filterbank, starting with the first granule before any is filtered, and
/// returns the block type and analysis of the granule before the one it is
/// given: `None` only for the first. `quantize` is given each frame's
/// granules, indexed `[gr][ch]`, once all are filtered, the share of the
/// frame's main data of each granule of each channel, and the bit
/// reservoir. It quantises the granules, accounting in the reservoir for
/// the bits each uses.
fn encode_filtered<A>(
    config: &StreamConfig,
    channels: &[&[f64]],
    mut reservoir: Reservoir,
    mut look_ahead: impl FnMut(usize, &[f64; GRANULE_LINES]) -> Option<(BlockType, A)>,
    mut quantize: impl FnMut(Vec<Vec<Filtered<A>>>, usize, &mut Reservoir) -> CodedFrame,
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
        let filtered: Vec<Vec<Filtered<A>>> = (0..GRANULES)
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
        // mode_extension does not change the frame's length.
        let padded = padding.next_padding();
        let slot_bits = main_data_bits(&config.header(padded));
        let shares = GRANULES * channels.len();
        let mean_bits = slot_bits / shares;
        let block_types: Vec<BlockType> = filtered.iter().flatten().map(|f| f.block_type).collect();
        let coded = quantize(filtered, mean_bits, &mut reservoir);
        assert!(
            (coded.granules.iter().flatten())
                .map(|q| q.block_type)
                .eq(block_types),
            "the granules keep their block types"
        );
        let header = config.frame_header(padded, coded.ms_stereo, coded.intensity_stereo);
        let (side_info, main_data) = frame_contents(fs, &coded.granules, main_data_begin);
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
}
