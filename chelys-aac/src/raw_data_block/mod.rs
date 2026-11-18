// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The `raw_data_block()` (Table 12): a frame's coded channels in the
//! elements `channel_configuration` gives (Table 42), the fill that pads it
//! to its length, and `ID_END`, byte aligned.

pub mod element;
pub mod fill;
pub mod ics;

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

/// Writes `element` with its `id_syn_ele`.
///
/// # Panics
///
/// If the element is not the one `configuration` gives: a caller's error.
fn write_element(w: &mut BitWriter, configuration: ChannelConfiguration, element: &CodedElement) {
    let id = match (configuration, element.is_pair()) {
        (ChannelConfiguration::Mono, false) => ID_SCE,
        (ChannelConfiguration::Stereo, true) => ID_CPE,
        _ => panic!("{element:?} in a {configuration:?} frame"),
    };
    w.write(id, 3);
    element.write(w);
}

/// The bytes of the shortest `raw_data_block()` of `element`: the element
/// and `ID_END`, byte aligned, without fill.
pub fn min_len(element: &CodedElement) -> usize {
    (OVERHEAD_BITS + element.bits()).div_ceil(8) as usize
}

/// A `raw_data_block()` of `element`, padded with fill elements to `len`
/// bytes, or to [`min_len`] if that is longer.
///
/// # Panics
///
/// If the element is not the one `configuration` gives: a caller's error.
pub fn write(configuration: ChannelConfiguration, element: &CodedElement, len: usize) -> Vec<u8> {
    let mut w = BitWriter::new();
    write_element(&mut w, configuration, element);
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
    let block = w.into_bytes();
    debug_assert_eq!(block.len(), len.max(min_len(element)));
    block
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
        let block = write(ChannelConfiguration::Mono, &element, 0);
        let sce = ["0000", "00000000", "00000000000", "000"].concat();
        // ID_SCE, the element and ID_END end on a byte.
        let bits = ["000", &sce, "111"].concat();
        assert_eq!(binary_digits(&block), bits);
        assert_eq!(min_len(&element), 4);
    }

    #[test]
    fn a_silent_stereo_block_is_a_cpe_and_end() {
        let element = silent(ChannelConfiguration::Stereo);
        let block = write(ChannelConfiguration::Stereo, &element, 0);
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
            let unfilled = write(configuration, &element, 0);
            for len in shortest..shortest + 600 {
                let block = write(configuration, &element, len);
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
