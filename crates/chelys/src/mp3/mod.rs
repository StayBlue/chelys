//! MPEG-1 Audio Layer III.

mod crc;
pub mod filterbank;
pub mod header;
pub mod huffman;
pub mod quantizer;
pub mod scalefactor_bands;
pub mod side_info;

use crate::bitstream::BitWriter;
use header::{Bitrate, Emphasis, Header, Mode, PaddingSchedule, SamplingFrequency};
use huffman::{QuadTable, SpectrumCoding};
use side_info::{GRANULES, GranuleChannel, SideInfo, Window};

/// Two granules of 576 samples per channel (clause 2.4.2.1).
pub const SAMPLES_PER_FRAME: usize = 1152;

/// Frequency lines in a granule (clause 2.1.70).
pub const GRANULE_LINES: usize = 576;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamConfig {
    pub bitrate: Bitrate,
    pub sampling_frequency: SamplingFrequency,
    pub mode: Mode,
    pub crc_protected: bool,
}

/// Appends one frame (clause 2.4.1.2): the header, the `crc_check` word if
/// the header asks for one, the side information, and then `main_data`
/// zero-filled to the frame length.
///
/// # Panics
///
/// If `main_data` does not fit in the frame.
pub fn write_frame(header: &Header, side_info: &SideInfo, main_data: &[u8], out: &mut Vec<u8>) {
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

    let end = start + header.frame_len();
    assert!(
        out.len() + main_data.len() <= end,
        "{} bytes of main data exceed the {} available in the frame",
        main_data.len(),
        end - out.len(),
    );
    out.extend_from_slice(main_data);
    out.resize(end, 0);
}

/// One granule of one channel, quantised and ready for Huffman coding. Long
/// blocks only, without scalefactors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantizedGranule {
    /// The quantizer step size; see the requantization formula in clause
    /// 2.4.3.4.7.1.
    pub global_gain: u8,
    pub values: [i32; GRANULE_LINES],
    pub coding: SpectrumCoding,
}

impl QuantizedGranule {
    /// Partitions `values` and chooses Huffman tables for them (clause
    /// C.1.5.4.4).
    pub fn new(
        global_gain: u8,
        values: [i32; GRANULE_LINES],
        sampling_frequency: SamplingFrequency,
    ) -> Self {
        let coding =
            SpectrumCoding::choose(&values, scalefactor_bands::long_bands(sampling_frequency));
        Self {
            global_gain,
            values,
            coding,
        }
    }
}

/// The largest `part2_3_length`, a 12-bit field (clause 2.4.1.7).
const MAX_PART2_3_LENGTH: usize = (1 << 12) - 1;

impl StreamConfig {
    fn header(&self, padding: bool) -> Header {
        Header {
            crc_protected: self.crc_protected,
            bitrate: self.bitrate,
            sampling_frequency: self.sampling_frequency,
            padding,
            private: false,
            mode: self.mode,
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

/// Writes one frame of granules, one entry per channel each, with
/// `main_data_begin` 0.
fn write_granule_frame(header: &Header, frame: &[Vec<QuantizedGranule>], out: &mut Vec<u8>) {
    let bands = scalefactor_bands::long_bands(header.sampling_frequency);
    let mut side_info = SideInfo::default();
    let mut main_data = BitWriter::new();
    for (gr, granule) in frame.iter().enumerate() {
        for (ch, channel) in granule.iter().enumerate() {
            let start = main_data.bit_len();
            channel.coding.write(&channel.values, bands, &mut main_data);
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
                scalefac_compress: 0,
                window: Window::Normal {
                    table_select: coding.table_select,
                    region0_count: coding.region0_count,
                    region1_count: coding.region1_count,
                },
                preflag: false,
                scalefac_scale: false,
                count1table_select: coding.count1_table == QuadTable::B,
            };
        }
    }
    write_frame(header, &side_info, &main_data.into_bytes(), out);
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
    let mut out = Vec::new();
    for frame in granules.chunks_exact(GRANULES) {
        write_granule_frame(&config.header(padding.next_padding()), frame, &mut out);
    }
    out
}

/// Encodes PCM, one slice of samples (nominally -1.0 to 1.0) per channel, at
/// the configured bitrate.
///
/// Each granule of each channel gets an equal share of its frame's main data,
/// the `mean_bits` of clause C.1.5.3.4 (there is no bit reservoir yet), and
/// the inner iteration loop picks the finest quantizer step that fits it. The
/// input is padded with silence to whole frames, plus one more frame to flush
/// the filterbank.
///
/// # Panics
///
/// If the channel count does not match the configured mode.
pub fn encode_pcm(config: &StreamConfig, channels: &[&[f64]]) -> Vec<u8> {
    encode_filtered(config, channels, |xr, available_bits| {
        quantizer::inner_loop(xr, available_bits, config.sampling_frequency)
    })
}

/// Encodes PCM like [`encode_pcm`], but quantises every granule with the same
/// `global_gain` instead of fitting it to the bitrate.
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
    encode_filtered(config, channels, |xr, _| {
        let values = quantizer::quantize(xr, global_gain)
            .unwrap_or_else(|| panic!("a granule is too loud for global_gain {global_gain}"));
        QuantizedGranule::new(global_gain, values, config.sampling_frequency)
    })
}

/// Runs each channel through its filterbank granule by granule, quantises
/// each granule with `quantize`, given the bits available to it, and writes
/// the frames.
fn encode_filtered(
    config: &StreamConfig,
    channels: &[&[f64]],
    mut quantize: impl FnMut(&[f64; GRANULE_LINES], usize) -> QuantizedGranule,
) -> Vec<u8> {
    assert_eq!(
        channels.len(),
        config.mode.channels(),
        "one slice per channel"
    );
    let samples = channels.iter().map(|c| c.len()).max().unwrap_or(0);
    let frames = samples.div_ceil(SAMPLES_PER_FRAME) + 1;
    let mut filterbanks: Vec<_> = channels
        .iter()
        .map(|_| filterbank::Filterbank::new())
        .collect();
    let mut padding = PaddingSchedule::new(config.bitrate, config.sampling_frequency);
    let mut out = Vec::new();

    for frame in 0..frames {
        let header = config.header(padding.next_padding());
        let available_bits =
            (main_data_bits(&header) / (GRANULES * channels.len())).min(MAX_PART2_3_LENGTH);
        let mut granules = Vec::with_capacity(GRANULES);
        for gr in 0..GRANULES {
            let offset = (frame * GRANULES + gr) * GRANULE_LINES;
            let mut granule = Vec::with_capacity(channels.len());
            for (pcm, filterbank) in channels.iter().zip(&mut filterbanks) {
                let pcm = std::array::from_fn(|t| pcm.get(offset + t).copied().unwrap_or(0.0));
                granule.push(quantize(&filterbank.granule(&pcm), available_bits));
            }
            granules.push(granule);
        }
        write_granule_frame(&header, &granules, &mut out);
    }
    out
}

/// Encodes `frames` frames of digital silence. Every granule codes to
/// nothing, so each frame is zero-filled after its side information.
pub fn encode_silence(config: &StreamConfig, frames: usize) -> Vec<u8> {
    let silent = QuantizedGranule::new(0, [0; GRANULE_LINES], config.sampling_frequency);
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
