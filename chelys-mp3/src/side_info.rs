// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Layer III side information (clauses 2.4.1.7 and 2.4.2.7, and clause
//! 2.4.1.7 of ISO/IEC 13818-3 for the lower sampling frequencies).

use crate::header::Version;
use chelys_core::bitstream::BitWriter;

pub const SCFSI_BANDS: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SideInfo {
    pub version: Version,
    pub main_data_begin: u16,
    /// 5 bits for single channel, 3 bits otherwise; at the lower sampling
    /// frequencies 1 and 2 bits.
    pub private_bits: u8,
    /// Indexed `[ch][scfsi_band]`. Not sent at the lower sampling
    /// frequencies.
    pub scfsi: [[bool; SCFSI_BANDS]; 2],
    /// Indexed `[gr][ch]`, one entry for each of the version's granules.
    pub granules: Vec<[GranuleChannel; 2]>,
}

impl SideInfo {
    /// Side information for frames of `version`, all zero.
    pub fn new(version: Version) -> Self {
        Self {
            version,
            main_data_begin: 0,
            private_bits: 0,
            scfsi: [[false; SCFSI_BANDS]; 2],
            granules: vec![Default::default(); version.granules()],
        }
    }

    /// The fixed side information length stated under `part2_3_length` in
    /// clause 2.4.2.7, or for the lower sampling frequencies implied by the
    /// bits of clause 2.4.2.4 of ISO/IEC 13818-3 that the error check covers.
    pub fn len_bytes(version: Version, channels: usize) -> usize {
        match (version, channels) {
            (Version::Mpeg1, 1) => 17,
            (Version::Mpeg1, 2) => 32,
            (Version::Mpeg2, 1) => 9,
            (Version::Mpeg2, 2) => 17,
            _ => panic!("Layer III carries one or two channels, not {channels}"),
        }
    }

    /// # Panics
    ///
    /// If there is not one entry in `granules` for each of the version's
    /// granules.
    pub fn write(&self, channels: usize, w: &mut BitWriter) {
        let version = self.version;
        assert_eq!(
            self.granules.len(),
            version.granules(),
            "one entry per granule"
        );
        let start = w.bit_len();
        w.write(
            u32::from(self.main_data_begin),
            version.main_data_begin_bits(),
        );
        let private_bits = match (version, channels) {
            (Version::Mpeg1, 1) => 5,
            (Version::Mpeg1, _) => 3,
            (Version::Mpeg2, 1) => 1,
            (Version::Mpeg2, _) => 2,
        };
        w.write(u32::from(self.private_bits), private_bits);
        if version == Version::Mpeg1 {
            for scfsi in &self.scfsi[..channels] {
                for &flag in scfsi {
                    w.write_bool(flag);
                }
            }
        }
        for granule in &self.granules {
            for channel in &granule[..channels] {
                channel.write(version, w);
            }
        }
        debug_assert_eq!(w.bit_len() - start, Self::len_bytes(version, channels) * 8);
    }
}

/// Side information for one granule of one channel.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GranuleChannel {
    pub part2_3_length: u16,
    pub big_values: u16,
    pub global_gain: u8,
    /// 4 bits, or 9 at the lower sampling frequencies.
    pub scalefac_compress: u16,
    pub window: Window,
    /// Not sent at the lower sampling frequencies, where `scalefac_compress`
    /// implies it (clause 2.4.3.2 of ISO/IEC 13818-3).
    pub preflag: bool,
    pub scalefac_scale: bool,
    pub count1table_select: bool,
}

impl GranuleChannel {
    fn write(&self, version: Version, w: &mut BitWriter) {
        w.write(u32::from(self.part2_3_length), 12);
        w.write(u32::from(self.big_values), 9);
        w.write(u32::from(self.global_gain), 8);
        let compress_bits = match version {
            Version::Mpeg1 => 4,
            Version::Mpeg2 => 9,
        };
        w.write(u32::from(self.scalefac_compress), compress_bits);
        match &self.window {
            Window::Normal {
                table_select,
                region0_count,
                region1_count,
            } => {
                w.write_bool(false); // window_switching_flag
                for &table in table_select {
                    w.write(u32::from(table), 5);
                }
                w.write(u32::from(*region0_count), 4);
                w.write(u32::from(*region1_count), 3);
            }
            Window::Switched {
                block_type,
                mixed_block_flag,
                table_select,
                subblock_gain,
            } => {
                w.write_bool(true); // window_switching_flag
                w.write(*block_type as u32, 2);
                w.write_bool(*mixed_block_flag);
                for &table in table_select {
                    w.write(u32::from(table), 5);
                }
                for &gain in subblock_gain {
                    w.write(u32::from(gain), 3);
                }
            }
        }
        if version == Version::Mpeg1 {
            w.write_bool(self.preflag);
        }
        w.write_bool(self.scalefac_scale);
        w.write_bool(self.count1table_select);
    }
}

/// The `window_switching_flag` and the fields that depend on it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Window {
    Normal {
        table_select: [u8; 3],
        region0_count: u8,
        region1_count: u8,
    },
    Switched {
        block_type: BlockType,
        mixed_block_flag: bool,
        table_select: [u8; 2],
        subblock_gain: [u8; 3],
    },
}

impl Default for Window {
    fn default() -> Self {
        Self::Normal {
            table_select: [0; 3],
            region0_count: 0,
            region1_count: 0,
        }
    }
}

/// Block types valid when `window_switching_flag` is set; type 0 is reserved there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockType {
    Start = 1,
    Short = 2,
    End = 3,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelys_core::bitstream::binary_digits;

    fn written_bits(side_info: &SideInfo, channels: usize) -> Vec<u8> {
        let mut w = BitWriter::new();
        side_info.write(channels, &mut w);
        assert_eq!(
            w.bit_len(),
            SideInfo::len_bytes(side_info.version, channels) * 8
        );
        w.into_bytes()
    }

    #[test]
    fn length_matches_clause_2_4_2_7_for_both_window_layouts() {
        for version in [Version::Mpeg1, Version::Mpeg2] {
            let mut side_info = SideInfo::new(version);
            for channels in [1, 2] {
                written_bits(&side_info, channels);
            }
            side_info.granules.last_mut().unwrap()[1].window = Window::Switched {
                block_type: BlockType::Short,
                mixed_block_flag: true,
                table_select: [31, 31],
                subblock_gain: [7, 7, 7],
            };
            for channels in [1, 2] {
                written_bits(&side_info, channels);
            }
        }
    }

    #[test]
    fn single_channel_field_layout() {
        let mut side_info = SideInfo {
            main_data_begin: 0x1FF,
            private_bits: 0b10101,
            ..SideInfo::new(Version::Mpeg1)
        };
        side_info.scfsi[0] = [true, false, false, true];
        side_info.granules[0][0] = GranuleChannel {
            part2_3_length: 0xABC,
            big_values: 0x123,
            global_gain: 0xF0,
            scalefac_compress: 0b0110,
            window: Window::Normal {
                table_select: [1, 2, 3],
                region0_count: 0b1001,
                region1_count: 0b101,
            },
            preflag: true,
            scalefac_scale: false,
            count1table_select: true,
        };

        let bits = binary_digits(&written_bits(&side_info, 1));
        let expected = [
            "111111111",    // main_data_begin
            "10101",        // private_bits
            "1001",         // scfsi
            "101010111100", // part2_3_length
            "100100011",    // big_values
            "11110000",     // global_gain
            "0110",         // scalefac_compress
            "0",            // window_switching_flag
            "00001",        // table_select[0]
            "00010",        // table_select[1]
            "00011",        // table_select[2]
            "1001",         // region0_count
            "101",          // region1_count
            "1",            // preflag
            "0",            // scalefac_scale
            "1",            // count1table_select
        ]
        .concat();
        assert_eq!(&bits[..expected.len()], expected);
        // Granule 1 is all zeros, including its window_switching_flag.
        assert!(bits[expected.len()..].bytes().all(|b| b == b'0'));
    }

    #[test]
    fn window_switching_field_layout() {
        let mut side_info = SideInfo::new(Version::Mpeg1);
        side_info.granules[0][0].window = Window::Switched {
            block_type: BlockType::End,
            mixed_block_flag: true,
            table_select: [0b10001, 0b01110],
            subblock_gain: [0b001, 0b010, 0b100],
        };
        let bits = binary_digits(&written_bits(&side_info, 1));
        // Skip main_data_begin, private_bits, scfsi and the first four fields.
        let offset = 9 + 5 + 4 + 12 + 9 + 8 + 4;
        let expected = ["1", "11", "1", "10001", "01110", "001", "010", "100", "000"].concat();
        assert_eq!(&bits[offset..offset + expected.len()], expected);
    }

    /// At the lower sampling frequencies, `main_data_begin` and
    /// `private_bits` are narrower, there is no `scfsi` and no `preflag`, and
    /// `scalefac_compress` has 9 bits (clause 2.4.1.7 of ISO/IEC 13818-3).
    #[test]
    fn lower_sampling_frequency_field_layout() {
        let mut side_info = SideInfo {
            main_data_begin: 0xA5,
            private_bits: 0b10,
            ..SideInfo::new(Version::Mpeg2)
        };
        side_info.scfsi = [[true; SCFSI_BANDS]; 2];
        let channel = GranuleChannel {
            part2_3_length: 0xABC,
            big_values: 0x123,
            global_gain: 0xF0,
            scalefac_compress: 0b1_0110_1001,
            window: Window::Normal {
                table_select: [1, 2, 3],
                region0_count: 0b1001,
                region1_count: 0b101,
            },
            preflag: true,
            scalefac_scale: true,
            count1table_select: false,
        };
        side_info.granules[0] = [channel.clone(), channel];

        let bits = binary_digits(&written_bits(&side_info, 2));
        let channel = [
            "101010111100", // part2_3_length
            "100100011",    // big_values
            "11110000",     // global_gain
            "101101001",    // scalefac_compress
            "0",            // window_switching_flag
            "00001",        // table_select[0]
            "00010",        // table_select[1]
            "00011",        // table_select[2]
            "1001",         // region0_count
            "101",          // region1_count
            "1",            // scalefac_scale
            "0",            // count1table_select
        ]
        .concat();
        let expected = ["10100101", "10", &channel, &channel].concat();
        assert_eq!(bits, expected);
    }
}
