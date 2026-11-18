// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The `raw_data_block()` (Table 12): a frame's coded channels in the
//! elements `channel_configuration` gives (Table 42), the fill that pads it
//! to its length, and `ID_END`, byte aligned.

pub mod element;
pub mod fill;
pub mod ics;

use chelys_core::bitstream::BitWriter;

use super::channel::CodedChannel;
use super::channel_configuration::ChannelConfiguration;
use element::{
    CHANNEL_PAIR_ELEMENT_BITS, SINGLE_CHANNEL_ELEMENT_BITS, channel_bits,
    write_channel_pair_element, write_single_channel_element,
};
use fill::write_fill;

/// `id_syn_ele` (Table 36).
const ID_SCE: u32 = 0x0;
const ID_CPE: u32 = 0x1;
const ID_FIL: u32 = 0x6;
const ID_END: u32 = 0x7;

/// The bits of a `raw_data_block()` of `configuration` besides its
/// channels' `individual_channel_stream()`s, fill and byte alignment: its
/// element's `id_syn_ele` and fields, and `ID_END`.
pub fn overhead_bits(configuration: ChannelConfiguration) -> u32 {
    let element = match configuration {
        ChannelConfiguration::Mono => SINGLE_CHANNEL_ELEMENT_BITS,
        ChannelConfiguration::Stereo => CHANNEL_PAIR_ELEMENT_BITS,
    };
    3 + element + 3
}

/// Writes the element of `channels`, which `configuration` gives, with its
/// `id_syn_ele`.
///
/// # Panics
///
/// If there are not as many channels as `configuration` has: a caller's
/// error.
fn write_elements(
    w: &mut BitWriter,
    configuration: ChannelConfiguration,
    channels: &[CodedChannel],
) {
    match (configuration, channels) {
        (ChannelConfiguration::Mono, [channel]) => {
            w.write(ID_SCE, 3);
            write_single_channel_element(channel, w);
        }
        (ChannelConfiguration::Stereo, [left, right]) => {
            w.write(ID_CPE, 3);
            write_channel_pair_element(left, right, w);
        }
        _ => panic!("{} channels in a {configuration:?} frame", channels.len()),
    }
}

/// The bytes of the shortest `raw_data_block()` of `channels`: its
/// elements and `ID_END`, byte aligned, without fill.
pub fn min_len(configuration: ChannelConfiguration, channels: &[CodedChannel]) -> usize {
    let channels: u32 = channels.iter().map(channel_bits).sum();
    (overhead_bits(configuration) + channels).div_ceil(8) as usize
}

/// A `raw_data_block()` of `channels`, padded with fill elements to `len`
/// bytes, or to [`min_len`] if that is longer.
///
/// # Panics
///
/// If there are not as many channels as `configuration` has: a caller's
/// error.
pub fn write(
    configuration: ChannelConfiguration,
    channels: &[CodedChannel],
    len: usize,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    write_elements(&mut w, configuration, channels);
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
    debug_assert_eq!(block.len(), len.max(min_len(configuration, channels)));
    block
}

#[cfg(test)]
mod tests {
    use chelys_core::bitstream::binary_digits;

    use super::*;
    use crate::sampling_frequency::SamplingFrequency;
    use crate::window::Windows;

    fn silent(configuration: ChannelConfiguration) -> Vec<CodedChannel> {
        let channel = CodedChannel::silent(Windows::LONG_SINE, SamplingFrequency::Hz44100);
        vec![channel; configuration.channels()]
    }

    #[test]
    fn a_silent_mono_block_is_an_sce_and_end() {
        let channels = silent(ChannelConfiguration::Mono);
        let block = write(ChannelConfiguration::Mono, &channels, 0);
        let sce = ["0000", "00000000", "00000000000", "000"].concat();
        // ID_SCE, the element and ID_END end on a byte.
        let bits = ["000", &sce, "111"].concat();
        assert_eq!(binary_digits(&block), bits);
        assert_eq!(min_len(ChannelConfiguration::Mono, &channels), 4);
    }

    #[test]
    fn a_silent_stereo_block_is_a_cpe_and_end() {
        let channels = silent(ChannelConfiguration::Stereo);
        let block = write(ChannelConfiguration::Stereo, &channels, 0);
        let ics = ["00000000", "00000000000", "000"].concat();
        let cpe = ["0000", "0", &ics, &ics].concat();
        // ID_CPE, the element, ID_END, then zeros to the byte.
        let bits = ["001", &cpe, "111", "0"].concat();
        assert_eq!(binary_digits(&block), bits);
        assert_eq!(min_len(ChannelConfiguration::Stereo, &channels), 7);
    }

    #[test]
    fn fill_pads_a_block_to_every_longer_length() {
        for configuration in [ChannelConfiguration::Mono, ChannelConfiguration::Stereo] {
            let channels = silent(configuration);
            let shortest = min_len(configuration, &channels);
            let unfilled = write(configuration, &channels, 0);
            for len in shortest..shortest + 600 {
                let block = write(configuration, &channels, len);
                assert_eq!(block.len(), len, "{configuration:?}");
                assert_eq!(block[..shortest - 1], unfilled[..shortest - 1]);
            }
        }
    }

    #[test]
    #[should_panic(expected = "2 channels in a Mono frame")]
    fn the_channels_are_those_of_the_configuration() {
        write(
            ChannelConfiguration::Mono,
            &silent(ChannelConfiguration::Stereo),
            0,
        );
    }
}
