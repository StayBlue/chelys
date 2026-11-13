// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Layer III side information (clauses 2.4.1.7 and 2.4.2.7).

use crate::bitstream::BitWriter;

pub const GRANULES: usize = 2;
pub const SCFSI_BANDS: usize = 4;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SideInfo {
    pub main_data_begin: u16,
    /// 5 bits for single channel, 3 bits otherwise.
    pub private_bits: u8,
    /// Indexed `[ch][scfsi_band]`.
    pub scfsi: [[bool; SCFSI_BANDS]; 2],
    /// Indexed `[gr][ch]`.
    pub granules: [[GranuleChannel; 2]; GRANULES],
}

impl SideInfo {
    /// The fixed side information length stated under `part2_3_length` in
    /// clause 2.4.2.7.
    pub fn len_bytes(channels: usize) -> usize {
        match channels {
            1 => 17,
            2 => 32,
            _ => panic!("Layer III carries one or two channels, not {channels}"),
        }
    }

    pub fn write(&self, channels: usize, w: &mut BitWriter) {
        let start = w.bit_len();
        w.write(u32::from(self.main_data_begin), 9);
        w.write(
            u32::from(self.private_bits),
            if channels == 1 { 5 } else { 3 },
        );
        for scfsi in &self.scfsi[..channels] {
            for &flag in scfsi {
                w.write_bool(flag);
            }
        }
        for granule in &self.granules {
            for channel in &granule[..channels] {
                channel.write(w);
            }
        }
        debug_assert_eq!(w.bit_len() - start, Self::len_bytes(channels) * 8);
    }
}

/// Side information for one granule of one channel.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GranuleChannel {
    pub part2_3_length: u16,
    pub big_values: u16,
    pub global_gain: u8,
    pub scalefac_compress: u8,
    pub window: Window,
    pub preflag: bool,
    pub scalefac_scale: bool,
    pub count1table_select: bool,
}

impl GranuleChannel {
    fn write(&self, w: &mut BitWriter) {
        w.write(u32::from(self.part2_3_length), 12);
        w.write(u32::from(self.big_values), 9);
        w.write(u32::from(self.global_gain), 8);
        w.write(u32::from(self.scalefac_compress), 4);
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
        w.write_bool(self.preflag);
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

    fn written_bits(side_info: &SideInfo, channels: usize) -> Vec<u8> {
        let mut w = BitWriter::new();
        side_info.write(channels, &mut w);
        assert_eq!(w.bit_len(), SideInfo::len_bytes(channels) * 8);
        w.into_bytes()
    }

    #[test]
    fn length_matches_clause_2_4_2_7_for_both_window_layouts() {
        let mut side_info = SideInfo::default();
        for channels in [1, 2] {
            written_bits(&side_info, channels);
        }
        side_info.granules[1][1].window = Window::Switched {
            block_type: BlockType::Short,
            mixed_block_flag: true,
            table_select: [31, 31],
            subblock_gain: [7, 7, 7],
        };
        for channels in [1, 2] {
            written_bits(&side_info, channels);
        }
    }

    #[test]
    fn single_channel_field_layout() {
        let mut side_info = SideInfo {
            main_data_begin: 0x1FF,
            private_bits: 0b10101,
            ..SideInfo::default()
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

        let bits: String = written_bits(&side_info, 1)
            .iter()
            .map(|b| format!("{b:08b}"))
            .collect();
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
        let mut side_info = SideInfo::default();
        side_info.granules[0][0].window = Window::Switched {
            block_type: BlockType::End,
            mixed_block_flag: true,
            table_select: [0b10001, 0b01110],
            subblock_gain: [0b001, 0b010, 0b100],
        };
        let bits: String = written_bits(&side_info, 1)
            .iter()
            .map(|b| format!("{b:08b}"))
            .collect();
        // Skip main_data_begin, private_bits, scfsi and the first four fields.
        let offset = 9 + 5 + 4 + 12 + 9 + 8 + 4;
        let expected = ["1", "11", "1", "10001", "01110", "001", "010", "100", "000"].concat();
        assert_eq!(&bits[offset..offset + expected.len()], expected);
    }
}
