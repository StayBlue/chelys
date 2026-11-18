// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The Audio Data Transport Stream (clause 8.1.3): each `raw_data_block()`
//! in an `adts_frame()` (Table 5) of its own, after a header that a
//! decoder can start from.

use chelys_core::bitstream::BitWriter;

use super::channel_configuration::ChannelConfiguration;
use super::sampling_frequency::SamplingFrequency;

/// The bytes of `adts_fixed_header()` (Table 8) and
/// `adts_variable_header()` (Table 9) together: 28 bits each.
pub const HEADER_LEN: usize = 7;

/// The largest `aac_frame_length`, in its 13 bits.
pub const MAX_FRAME_LEN: usize = (1 << 13) - 1;

/// `profile` (Table 31).
const LOW_COMPLEXITY: u32 = 1;

/// What every header of a stream holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub sampling_frequency: SamplingFrequency,
    pub channel_configuration: ChannelConfiguration,
}

impl Header {
    /// Writes the header of an `adts_frame()` of `frame_len` bytes, headers
    /// included, with `buffer_fullness` as `adts_buffer_fullness`.
    ///
    /// The frame holds one `raw_data_block()`
    /// (`number_of_raw_data_blocks_in_frame` 0) and no `adts_error_check()`
    /// (`protection_absent` 1), so the block follows the header directly.
    pub fn write(&self, w: &mut BitWriter, frame_len: usize, buffer_fullness: u32) {
        assert!(
            (HEADER_LEN..=MAX_FRAME_LEN).contains(&frame_len),
            "an ADTS frame of {frame_len} bytes"
        );
        debug_assert!(buffer_fullness < 0x7FF, "buffer fullness {buffer_fullness}");
        // The private bit, original_copy and home, and the copyright
        // identification, are the user's to set; none are offered, and
        // they are 0.
        // adts_fixed_header()
        w.write(0xFFF, 12); // syncword
        w.write(1, 1); // ID: "set to '1'" (clause 8.1.3.2.2)
        w.write(0b00, 2); // layer: "Set to '00'"
        w.write(1, 1); // protection_absent
        w.write(LOW_COMPLEXITY, 2); // profile
        w.write(self.sampling_frequency.index(), 4); // sampling_frequency_index
        w.write(0, 1); // private_bit
        w.write(self.channel_configuration.value(), 3); // channel_configuration
        w.write(0, 1); // original_copy
        w.write(0, 1); // home
        // adts_variable_header()
        w.write(0, 1); // copyright_identification_bit
        // copyright_identification_start: "If no copyright identification
        // is transmitted, this bit should be kept '0'."
        w.write(0, 1);
        w.write(frame_len as u32, 13); // aac_frame_length
        w.write(buffer_fullness, 11); // adts_buffer_fullness
        w.write(0, 2); // number_of_raw_data_blocks_in_frame
    }
}

#[cfg(test)]
mod tests {
    use chelys_core::bitstream::binary_digits;

    use super::*;

    #[test]
    fn header_spells_its_fields() {
        let header = Header {
            sampling_frequency: SamplingFrequency::Hz44100,
            channel_configuration: ChannelConfiguration::Stereo,
        };
        let mut w = BitWriter::new();
        header.write(&mut w, 0x1ABC, 0x123);
        assert_eq!(w.bit_len(), 8 * HEADER_LEN);
        let fixed = [
            "111111111111", // syncword
            "1",            // ID
            "00",           // layer
            "1",            // protection_absent
            "01",           // profile
            "0100",         // sampling_frequency_index
            "0",            // private_bit
            "010",          // channel_configuration
            "0",            // original_copy
            "0",            // home
        ];
        let variable = [
            "0",             // copyright_identification_bit
            "0",             // copyright_identification_start
            "1101010111100", // aac_frame_length
            "00100100011",   // adts_buffer_fullness
            "00",            // number_of_raw_data_blocks_in_frame
        ];
        assert_eq!(
            binary_digits(&w.into_bytes()),
            [fixed.concat(), variable.concat()].concat()
        );
    }
}
