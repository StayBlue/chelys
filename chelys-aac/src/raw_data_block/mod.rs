// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The `raw_data_block()` (Table 12): a frame's coded channels in the
//! elements `channel_configuration` gives (Table 42), the fill that pads it
//! to its length, and `ID_END`, byte aligned; and the bits of it that an
//! `adts_error_check()` protects.

/// For chelys's tests, and through `internals` for its conformance tests.
#[cfg(any(test, feature = "unstable"))]
pub mod decoder;
pub mod element;
pub mod fill;
pub mod ics;

use std::ops::Range;

use chelys_core::bitstream::BitWriter;

use super::channel_configuration::ChannelConfiguration;
use element::CodedElement;
use fill::write_fill;

/// `id_syn_ele` (Table 36).
const ID_SCE: u32 = 0x0;
const ID_CPE: u32 = 0x1;
const ID_FIL: u32 = 0x6;
const ID_END: u32 = 0x7;

/// The bits of a `raw_data_block()` of one element besides the element
/// itself, fill and byte alignment: its `id_syn_ele` and `ID_END`.
pub const OVERHEAD_BITS: u32 = 3 + 3;

/// The bits of an audio channel element that an `adts_error_check()`
/// protects: its first 192 (clause 8.1.3.2.1).
const ELEMENT_PROTECTED_BITS: usize = 192;

/// The bits of the second `individual_channel_stream()` of a
/// `channel_pair_element()` that an `adts_error_check()` protects: its
/// first 128 (clause 8.1.3.2.1).
const SECOND_ICS_PROTECTED_BITS: usize = 128;

/// Bits of a `raw_data_block()` that a CRC protects, as they are fed to
/// it: `bits` of the block, cut to `len` bits or padded with zeros to them
/// where it has a length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Protected {
    pub bits: Range<usize>,
    pub len: Option<usize>,
}

impl Protected {
    /// The bits fed to the CRC, of the `block` they are bits of.
    pub fn feed<'a>(&'a self, block: &'a [u8]) -> impl Iterator<Item = bool> + 'a {
        let len = self.len.unwrap_or(self.bits.len());
        let bit = |i: usize| (block[i / 8] >> (7 - i % 8)) & 1 == 1;
        (self.bits.start..self.bits.start + len).map(move |i| self.bits.contains(&i) && bit(i))
    }
}

/// A written `raw_data_block()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub bytes: Vec<u8>,
    /// The bits of it that an `adts_error_check()` protects, in the order
    /// they are fed to the CRC.
    pub protected: Vec<Protected>,
}

/// Writes `element` with its `id_syn_ele`, and gives the bits of it that an
/// `adts_error_check()` protects (clause 8.1.3.2.1): its first 192 bits
/// after the `id_syn_ele`, which "shall be excluded", and of a pair, the
/// first 128 of its second `individual_channel_stream()`, each padded with
/// zeros where the element or the stream is shorter.
///
/// Spec reading: the clause lists the bits protected "in order of their
/// appearance", and a pair's second stream, whose first 128 bits are
/// protected apart from the element's first 192, appears within them or
/// after them. Its 128 bits are fed after the element's 192, each part in
/// order, which the clause says of the bits it protects twice: "if the
/// first ICS of the CPE ends at the Nth bit (N<192), the first (192 – N)
/// bits of the second ICS are protected twice, each time in order of their
/// appearance".
///
/// # Panics
///
/// If the element is not the one `configuration` gives: a caller's error.
fn write_element(
    w: &mut BitWriter,
    configuration: ChannelConfiguration,
    element: &CodedElement,
) -> Vec<Protected> {
    let id = match (configuration, element.is_pair()) {
        (ChannelConfiguration::Mono, false) => ID_SCE,
        (ChannelConfiguration::Stereo, true) => ID_CPE,
        _ => panic!("{element:?} in a {configuration:?} frame"),
    };
    w.write(id, 3);
    let start = w.bit_len();
    element.write(w);
    let end = w.bit_len();
    let mut protected = vec![Protected {
        bits: start..end,
        len: Some(ELEMENT_PROTECTED_BITS),
    }];
    if let Some(offset) = element.second_channel_offset() {
        protected.push(Protected {
            bits: start + offset as usize..end,
            len: Some(SECOND_ICS_PROTECTED_BITS),
        });
    }
    protected
}

/// The bytes of the shortest `raw_data_block()` of `element`: the element
/// and `ID_END`, byte aligned, without fill.
pub fn min_len(element: &CodedElement) -> usize {
    (OVERHEAD_BITS + element.bits()).div_ceil(8) as usize
}

/// A `raw_data_block()` of `element`, padded with fill elements to `len`
/// bytes, or to [`min_len`] if that is longer. Fill elements are not
/// protected (clause 8.1.3.2.1), nor is `ID_END`, nor the bits that align.
///
/// # Panics
///
/// If the element is not the one `configuration` gives: a caller's error.
pub fn write(configuration: ChannelConfiguration, element: &CodedElement, len: usize) -> Block {
    let mut w = BitWriter::new();
    let protected = write_element(&mut w, configuration, element);
    // Fill before ID_END to within the bits that byte alignment pads. A
    // block longer than the shortest has room for a byte of fill at least.
    let fill = (8 * len).saturating_sub(w.bit_len() + 3);
    write_fill(&mut w, fill);
    w.write(ID_END, 3);
    // byte_alignment(), to the block's first bit, which is the writer's.
    // Spec gap: the standard does not give the value of the bits that
    // align; they are zero, as the writer leaves them. Settled by decoding
    // with ffmpeg (black box): it decodes blocks so aligned without
    // complaint; the conformance test is
    // `ffmpeg_decodes_silence_at_every_sample_rate`.
    let bytes = w.into_bytes();
    debug_assert_eq!(bytes.len(), len.max(min_len(element)));
    Block { bytes, protected }
}

#[cfg(test)]
mod tests {
    use chelys_core::bitstream::binary_digits;

    use super::*;
    use crate::channel::CodedChannel;
    use crate::sampling_frequency::SamplingFrequency;
    use crate::window::Windows;

    fn silent(configuration: ChannelConfiguration) -> CodedElement {
        let channel = CodedChannel::silent(Windows::LONG_SINE, SamplingFrequency::Hz44100);
        match configuration {
            ChannelConfiguration::Mono => CodedElement::single(channel),
            ChannelConfiguration::Stereo => CodedElement::pair(channel.clone(), channel),
        }
    }

    #[test]
    fn a_silent_mono_block_is_an_sce_and_end() {
        let element = silent(ChannelConfiguration::Mono);
        let block = write(ChannelConfiguration::Mono, &element, 0).bytes;
        let sce = ["0000", "00000000", "00000000000", "000"].concat();
        // ID_SCE, the element and ID_END end on a byte.
        let bits = ["000", &sce, "111"].concat();
        assert_eq!(binary_digits(&block), bits);
        assert_eq!(min_len(&element), 4);
    }

    #[test]
    fn a_silent_stereo_block_is_a_cpe_and_end() {
        let element = silent(ChannelConfiguration::Stereo);
        let block = write(ChannelConfiguration::Stereo, &element, 0).bytes;
        let ics = ["00000000", "00000000000", "000"].concat();
        let cpe = ["0000", "0", &ics, &ics].concat();
        // ID_CPE, the element, ID_END, then zeros to the byte.
        let bits = ["001", &cpe, "111", "0"].concat();
        assert_eq!(binary_digits(&block), bits);
        assert_eq!(min_len(&element), 7);
    }

    #[test]
    fn fill_pads_a_block_to_every_longer_length() {
        for configuration in [ChannelConfiguration::Mono, ChannelConfiguration::Stereo] {
            let element = silent(configuration);
            let shortest = min_len(&element);
            let unfilled = write(configuration, &element, 0).bytes;
            for len in shortest..shortest + 600 {
                let block = write(configuration, &element, len).bytes;
                assert_eq!(block.len(), len, "{configuration:?}");
                assert_eq!(block[..shortest - 1], unfilled[..shortest - 1]);
            }
        }
    }

    #[test]
    #[should_panic(expected = "in a Mono frame")]
    fn the_channels_are_those_of_the_configuration() {
        write(
            ChannelConfiguration::Mono,
            &silent(ChannelConfiguration::Stereo),
            0,
        );
    }
}
