// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The Audio Data Transport Stream (clause 8.1.3): each `raw_data_block()`
//! in an `adts_frame()` (Table 5) of its own, after a header that a
//! decoder can start from, and optionally a CRC that protects them.

use chelys_core::bitstream::BitWriter;
use chelys_core::crc;

use super::channel_configuration::ChannelConfiguration;
use super::raw_data_block::Block;
use super::sampling_frequency::SamplingFrequency;

/// The bytes of `adts_fixed_header()` (Table 8) and
/// `adts_variable_header()` (Table 9) together: 28 bits each.
pub const HEADER_LEN: usize = 7;

/// The bytes of an `adts_error_check()` (Table 10) that is present: its
/// `crc_check`.
pub const CRC_LEN: usize = 2;

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
    /// `private_bit`: as ISO/IEC 11172-3 defines it, a "Bit for private
    /// use", which "will not be used in the future by ISO/IEC".
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
    /// Whether an `adts_error_check()` follows the header:
    /// `protection_absent` 0.
    pub protected: bool,
}

impl Header {
    /// The bytes of the header and its `adts_error_check()`, which come
    /// before the frame's `raw_data_block()`.
    pub fn bytes(&self) -> usize {
        HEADER_LEN + if self.protected { CRC_LEN } else { 0 }
    }

    /// Writes the header of an `adts_frame()` of `frame_len` bytes, headers
    /// included, with `buffer_fullness` as `adts_buffer_fullness`.
    ///
    /// The frame holds one `raw_data_block()`
    /// (`number_of_raw_data_blocks_in_frame` 0), which follows the
    /// `adts_error_check()` (Table 5), present if the header is
    /// [`protected`](Self::protected): see [`crc_check`].
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
        w.write_bool(!self.protected); // protection_absent
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

/// The `crc_check` of the `adts_error_check()` of an `adts_frame()` whose
/// header is `header`, its 7 bytes, and whose one `raw_data_block()` is
/// `block`: the CRC of ISO/IEC 11172-3, subclause 2.4.3.1 (clause
/// 8.1.3.2.2), of all the bits of `adts_fixed_header()` and
/// `adts_variable_header()` and then those of the block that
/// `adts_error_check()` protects, in order (clause 8.1.3.2.1; see
/// [`Block::protected`]).
pub fn crc_check(header: &[u8], block: &Block) -> u16 {
    assert_eq!(header.len(), HEADER_LEN, "the bytes of a header");
    let register = crc::crc16(header.iter().copied());
    (block.protected.iter()).fold(register, |register, protected| {
        crc::update_bits(register, protected.feed(&block.bytes))
    })
}

#[cfg(test)]
mod tests {
    use chelys_core::bitstream::binary_digits;

    use super::*;
    use crate::channel::CodedChannel;
    use crate::raw_data_block::{self, element::CodedElement};
    use crate::window::Windows;

    /// The CRC of a string of binary digits, from the initial state.
    fn crc_of(digits: &str) -> u16 {
        crc::update_bits(crc::INITIAL, digits.chars().map(|c| c == '1'))
    }

    /// The header and block of a frame of `element`, `len` bytes of block.
    fn frame(
        element: &CodedElement,
        configuration: ChannelConfiguration,
        len: usize,
    ) -> (Vec<u8>, Block) {
        let header = Header {
            sampling_frequency: SamplingFrequency::Hz48000,
            channel_configuration: configuration,
            flags: Flags::default(),
            protected: true,
        };
        let block = raw_data_block::write(configuration, element, &[], len);
        let mut w = BitWriter::new();
        header.write(&mut w, header.bytes() + block.bytes.len(), 0x55);
        (w.into_bytes(), block)
    }

    #[test]
    fn protection_clears_protection_absent_and_adds_the_word() {
        let header = Header {
            sampling_frequency: SamplingFrequency::Hz48000,
            channel_configuration: ChannelConfiguration::Mono,
            flags: Flags::default(),
            protected: true,
        };
        assert_eq!(header.bytes(), 9);
        let mut w = BitWriter::new();
        header.write(&mut w, 9, 0);
        assert_eq!(&binary_digits(&w.into_bytes())[12..16], "1000");
    }

    /// A silent single channel's element is 26 bits after its `id_syn_ele`,
    /// zero padded to 192; the fill after it is not protected.
    #[test]
    fn a_short_single_channel_element_is_padded_to_192_bits() {
        let channel = CodedChannel::silent(Windows::LONG_SINE, SamplingFrequency::Hz48000);
        let element = CodedElement::single(channel);
        let ics = ["00000000", "00000000000", "000"].concat();
        let sce = ["0000", &ics].concat();
        for len in [4, 30] {
            let (header, block) = frame(&element, ChannelConfiguration::Mono, len);
            let protected = [
                binary_digits(&header),
                sce.clone(),
                "0".repeat(192 - sce.len()),
            ]
            .concat();
            assert_eq!(protected.len(), 56 + 192);
            assert_eq!(crc_check(&header, &block), crc_of(&protected), "{len}");
        }
    }

    /// A pair of silent channels is 49 bits, padded to 192, its second
    /// stream starting at its 28th, whose 22 bits are protected again,
    /// padded to 128.
    #[test]
    fn a_short_pair_protects_its_second_stream_twice() {
        let channel = CodedChannel::silent(Windows::LONG_SINE, SamplingFrequency::Hz48000);
        let element = CodedElement::pair(channel.clone(), channel);
        let ics = ["00000000", "00000000000", "000"].concat();
        let cpe = ["0000", "0", &ics, &ics].concat();
        let (header, block) = frame(&element, ChannelConfiguration::Stereo, 0);
        let protected = [
            binary_digits(&header),
            cpe.clone(),
            "0".repeat(192 - cpe.len()),
            ics.clone(),
            "0".repeat(128 - ics.len()),
        ]
        .concat();
        assert_eq!(crc_check(&header, &block), crc_of(&protected));
    }

    /// In frames of noise and silence, in one channel and two, the
    /// `crc_check` of each frame is that of the bits the test decoder finds
    /// in it: the headers, the first 192 bits of the element after its
    /// `id_syn_ele`, zero padded, and of a pair the first 128 of its second
    /// stream, zero padded, then each data stream element after its
    /// `id_syn_ele`; elements long and short among them, and frames with
    /// data streams and without.
    #[test]
    fn each_frame_protects_the_bits_its_elements_start_with() {
        use crate::config::Framing;
        use crate::raw_data_block::decoder::{adts_frame, raw_data_block};
        use crate::stream::{StreamConfig, StreamEncoder};

        let fs = SamplingFrequency::Hz44100;
        let mut state = 7u64;
        let mut noise = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            0.3 * ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
        };
        let mut input: Vec<f64> = vec![0.0; 3 * 1024];
        input.extend((0..6 * 1024).map(|_| noise()));
        for (channel_configuration, bits_per_second) in [
            (ChannelConfiguration::Mono, 64_000),
            (ChannelConfiguration::Stereo, 128_000),
            (ChannelConfiguration::Stereo, 32_000),
        ] {
            let config = StreamConfig {
                sampling_frequency: fs,
                channel_configuration,
                bits_per_second,
                framing: Framing::Adts,
                flags: Flags::default(),
                crc_protected: true,
            };
            let channels = vec![input.as_slice(); channel_configuration.channels()];
            let mut stream = Vec::new();
            let mut encoder = StreamEncoder::new(config).with_data_stream(20);
            encoder.push_data_stream(&[0xC3; 150]);
            encoder.encode(&channels, &mut stream);
            encoder.finish(&mut stream);
            let (mut short, mut long, mut data) = (false, false, 0);
            let mut rest = stream.as_slice();
            while !rest.is_empty() {
                let frame = adts_frame(rest);
                rest = &rest[frame.aac_frame_length..];
                assert!(!frame.protection_absent);
                let block = raw_data_block(frame.raw_data_block, fs);
                let bits = binary_digits(frame.raw_data_block);
                let first = |range: std::ops::Range<usize>, len: usize| {
                    let mut part = bits[range.start..range.end.min(range.start + len)].to_owned();
                    part.push_str(&"0".repeat(len - part.len()));
                    part
                };
                let mut protected = binary_digits(frame.header);
                protected.push_str(&first(block.element_bits.clone(), 192));
                if let Some(second) = block.second_channel_start {
                    protected.push_str(&first(second..block.element_bits.end, 128));
                }
                for stream in &block.data_streams {
                    protected.push_str(&bits[stream.bits.clone()]);
                    data += stream.bytes.len();
                }
                assert_eq!(frame.crc_check, Some(crc_of(&protected)));
                short |= block.element_bits.len() < 192;
                long |= block.element_bits.len() > 192 + 128;
            }
            assert!(short && long, "{channel_configuration:?}");
            assert_eq!(data, 150);
        }
    }

    #[test]
    fn header_spells_its_fields() {
        let header = Header {
            sampling_frequency: SamplingFrequency::Hz44100,
            channel_configuration: ChannelConfiguration::Stereo,
            flags: Flags::default(),
            protected: false,
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
                protected: false,
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
