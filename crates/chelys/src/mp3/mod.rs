//! MPEG-1 Audio Layer III.

mod crc;
pub mod header;
pub mod huffman;
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
    let bands = scalefactor_bands::long_bands(config.sampling_frequency);
    let mut padding = PaddingSchedule::new(config.bitrate, config.sampling_frequency);
    let mut out = Vec::new();

    for frame in granules.chunks_exact(GRANULES) {
        let header = Header {
            crc_protected: config.crc_protected,
            bitrate: config.bitrate,
            sampling_frequency: config.sampling_frequency,
            padding: padding.next_padding(),
            private: false,
            mode: config.mode,
            copyright: false,
            original: false,
            emphasis: Emphasis::None,
        };
        let mut side_info = SideInfo::default();
        let mut main_data = BitWriter::new();
        for (gr, granule) in frame.iter().enumerate() {
            for (ch, channel) in granule.iter().enumerate() {
                let start = main_data.bit_len();
                channel.coding.write(&channel.values, bands, &mut main_data);
                let part2_3_length = main_data.bit_len() - start;
                assert!(
                    part2_3_length < 1 << 12,
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
        write_frame(&header, &side_info, &main_data.into_bytes(), &mut out);
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
