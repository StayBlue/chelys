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

/// The bits of the header that describe the stream and leave its decoding
/// alone, each set as the user asks.
///
/// Spec reading: clause 8.1.3.2.2 defines `original_copy` and `home` by
/// those of ADIF (clause 8.1.2.2.2), which refer `original_copy` to the
/// definition of ISO/IEC 11172-3's `copyright` ("'1' means copyright
/// protected") and `home` to that of its `original/copy` ("'0' if the
/// bitstream is a copy, '1' if it is an original"), against what the name
/// `original_copy` says. Neither meaning is chosen here: each bit is named
/// for its field and set as asked, and [`Config`](crate::Config) gives
/// both references.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Flags {
    /// `private_bit`: as ISO/IEC 11172-3 defines it, a "bit for private
    /// use", which the standard does not use.
    pub private_bit: bool,
    pub original_copy: bool,
    pub home: bool,
}

/// What every header of a stream holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub sampling_frequency: SamplingFrequency,
    pub channel_configuration: ChannelConfiguration,
    pub flags: Flags,
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
        // The copyright identification is not offered (see below).
        // adts_fixed_header()
        w.write(0xFFF, 12); // syncword
        w.write(1, 1); // ID: "set to '1'" (clause 8.1.3.2.2)
        w.write(0b00, 2); // layer: "Set to '00'"
        w.write(1, 1); // protection_absent
        w.write(LOW_COMPLEXITY, 2); // profile
        w.write(self.sampling_frequency.index(), 4); // sampling_frequency_index
        w.write_bool(self.flags.private_bit);
        w.write(self.channel_configuration.value(), 3); // channel_configuration
        w.write_bool(self.flags.original_copy);
        w.write_bool(self.flags.home);
        // adts_variable_header()
        // copyright_identification_bit: one bit a frame of a 72-bit field,
        // an identifier that a registration authority gives and a number
        // that identifies the material (clause 8.1.3.2.2). It is not
        // offered, and is 0: the clause does not say in which frame the
        // field starts or whether it repeats, and refers to ISO/IEC
        // 13818-3, subclause 2.5.2.13, for it.
        w.write(0, 1);
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
            flags: Flags::default(),
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

    #[test]
    fn each_flag_sets_its_bit() {
        let set = |flags| {
            let header = Header {
                sampling_frequency: SamplingFrequency::Hz8000,
                channel_configuration: ChannelConfiguration::Mono,
                flags,
            };
            let mut w = BitWriter::new();
            header.write(&mut w, HEADER_LEN, 0);
            binary_digits(&w.into_bytes())
        };
        let none = set(Flags::default());
        // private_bit, channel_configuration 1, original_copy, home.
        assert_eq!(&none[22..28], "000100");
        let flips = |flags| {
            let bits = set(flags);
            (0..bits.len())
                .filter(|&i| bits.as_bytes()[i] != none.as_bytes()[i])
                .collect::<Vec<_>>()
        };
        let private_bit = Flags {
            private_bit: true,
            ..Flags::default()
        };
        let original_copy = Flags {
            original_copy: true,
            ..Flags::default()
        };
        let home = Flags {
            home: true,
            ..Flags::default()
        };
        assert_eq!(flips(private_bit), [22]);
        assert_eq!(flips(original_copy), [26]);
        assert_eq!(flips(home), [27]);
        let all = Flags {
            private_bit: true,
            original_copy: true,
            home: true,
        };
        assert_eq!(flips(all), [22, 26, 27]);
    }
}
