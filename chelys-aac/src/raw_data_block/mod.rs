// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The `raw_data_block()` (Table 12): a frame's 1024 samples of each
//! channel in the elements `channel_configuration` gives (Table 42), the
//! fill that pads it to its length, and `ID_END`, byte aligned.

pub mod element;
pub mod fill;
pub mod ics;

use chelys_core::bitstream::BitWriter;

use super::channel_configuration::ChannelConfiguration;
use element::{write_silent_channel_pair_element, write_silent_single_channel_element};
use fill::write_fill;

/// `id_syn_ele` (Table 36).
const ID_SCE: u32 = 0x0;
const ID_CPE: u32 = 0x1;
const ID_FIL: u32 = 0x6;
const ID_END: u32 = 0x7;

/// Writes the elements of a frame whose channels are all silent, each
/// with its `id_syn_ele`.
fn write_silent_elements(w: &mut BitWriter, configuration: ChannelConfiguration) {
    match configuration {
        ChannelConfiguration::Mono => {
            w.write(ID_SCE, 3);
            write_silent_single_channel_element(w);
        }
        ChannelConfiguration::Stereo => {
            w.write(ID_CPE, 3);
            write_silent_channel_pair_element(w);
        }
    }
}

/// The bytes of the shortest `raw_data_block()` of silent channels: its
/// elements and `ID_END`, byte aligned, without fill.
pub fn silent_len(configuration: ChannelConfiguration) -> usize {
    let mut w = BitWriter::new();
    write_silent_elements(&mut w, configuration);
    w.write(ID_END, 3);
    w.bit_len().div_ceil(8)
}

/// A `raw_data_block()` of silent channels, padded with fill elements to
/// `len` bytes, which is at least [`silent_len`].
pub fn silent(configuration: ChannelConfiguration, len: usize) -> Vec<u8> {
    let mut w = BitWriter::new();
    write_silent_elements(&mut w, configuration);
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
    debug_assert_eq!(block.len(), len.max(silent_len(configuration)));
    block
}

#[cfg(test)]
mod tests {
    use chelys_core::bitstream::binary_digits;

    use super::*;

    #[test]
    fn a_silent_mono_block_is_an_sce_and_end() {
        let block = silent(ChannelConfiguration::Mono, 0);
        let sce = ["0000", "00000000", "00000000000", "000"].concat();
        // ID_SCE, the element and ID_END end on a byte.
        let bits = ["000", &sce, "111"].concat();
        assert_eq!(binary_digits(&block), bits);
        assert_eq!(silent_len(ChannelConfiguration::Mono), 4);
    }

    #[test]
    fn a_silent_stereo_block_is_a_cpe_and_end() {
        let block = silent(ChannelConfiguration::Stereo, 0);
        let stream = ["00000000", "000"].concat();
        let cpe = ["0000", "1", "00000000000", "00", &stream, &stream].concat();
        // ID_CPE, the element, ID_END, then zeros to the byte.
        let bits = ["001", &cpe, "111", "00"].concat();
        assert_eq!(binary_digits(&block), bits);
        assert_eq!(silent_len(ChannelConfiguration::Stereo), 6);
    }

    #[test]
    fn fill_pads_a_block_to_every_longer_length() {
        for configuration in [ChannelConfiguration::Mono, ChannelConfiguration::Stereo] {
            let shortest = silent_len(configuration);
            for len in shortest..shortest + 600 {
                let block = silent(configuration, len);
                assert_eq!(block.len(), len, "{configuration:?}");
                assert_eq!(
                    block[..shortest - 1],
                    silent(configuration, 0)[..shortest - 1]
                );
            }
        }
    }
}
