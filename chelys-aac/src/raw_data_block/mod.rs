// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The `raw_data_block()` (Table 12): a frame's coded channels in the
//! elements `channel_configuration` gives (Table 42), in its order, any
//! data stream elements, the fill that pads it to its length, and
//! `ID_END`, byte aligned, in the order of the examples of clause 8.2.4,
//! `<CPE><DSE><FIL><TERM>` and `<SCE><CPE><CPE><LFE><TERM>`; and the bits of
//! it that an `adts_error_check()` protects.

pub mod data_stream;
/// For chelys's tests, and through `internals` for its conformance tests.
#[cfg(any(test, feature = "unstable"))]
pub mod decoder;
pub mod element;
pub mod fill;
pub mod ics;

use std::ops::Range;

use chelys_core::bitstream::BitWriter;

use super::channel_configuration::{ChannelConfiguration, ElementKind};
use element::CodedElement;
use fill::write_fill;

/// `id_syn_ele` (Table 36), besides those of the audio channel elements
/// (see [`ElementKind::id_syn_ele`]).
const ID_DSE: u32 = 0x4;
const ID_FIL: u32 = 0x6;
const ID_END: u32 = 0x7;

/// The bits of a `raw_data_block()` of `configuration`'s elements besides
/// the elements themselves, data stream elements, fill and byte alignment:
/// each element's `id_syn_ele`, and `ID_END`.
pub fn overhead_bits(configuration: ChannelConfiguration) -> u32 {
    3 * configuration.elements().len() as u32 + 3
}

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

/// Writes `element` with its `id_syn_ele` and `element_instance_tag`, and
/// gives the bits of it that an `adts_error_check()` protects (clause 8.1.3.2.1): its first 192 bits
/// after the `id_syn_ele`, which "shall be excluded", and of a pair, the
/// first 128 of its second `individual_channel_stream()`, each padded with
/// zeros where the element or the stream is shorter.
///
/// Spec reading: the 192 bits are counted from the element's first bit
/// after its `id_syn_ele`, `element_instance_tag`, where its syntax starts
/// (Tables 13 and 14), rather than from the `id_syn_ele`, whose bits would
/// then be left out of them: the clause's example counts the bits "of
/// CPE", and "the 190th bit of CPE" leaves 3 bits of 192 after it.
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
/// Spec gap: where the first stream ends at the Nth bit with N below 64,
/// the element's 192 bits hold 192 - N bits of the second stream, more
/// than the 128 of its own pass, and the clause's "the first (192 – N)
/// bits of the second ICS are protected twice" would protect all of them
/// twice; its example, of the 190th bit, fits either. The rules of 192 and
/// 128 bits are taken as what is protected, and the sentence as their
/// consequence where N is 64 or more: the second pass is 128 bits, so that
/// only they are protected twice. For example, in a pair at 48 kHz and 320
/// kbit/s whose left channel is silent and right one a tone of 1 kHz at
/// half of full scale, the second frame's second stream starts after 43
/// bits of the element, and its `crc_check` is D458, where feeding all 149
/// bits of the second stream within the 192 again would give FC83. No
/// decoder settles it, as neither ffmpeg nor FAAD2 checks the CRC (see
/// [`adts::crc_check`](crate::adts::crc_check)); the test of this reading
/// is `a_second_stream_from_before_bit_64_is_protected_again_for_128_bits`.
///
fn write_element(
    w: &mut BitWriter,
    element: &CodedElement,
    element_instance_tag: u32,
) -> [Option<Protected>; 2] {
    w.write(element.kind().id_syn_ele(), 3);
    let start = w.bit_len();
    element.write(w, element_instance_tag);
    let end = w.bit_len();
    let first = Protected {
        bits: start..end,
        len: Some(ELEMENT_PROTECTED_BITS),
    };
    let second = (element.second_channel_offset()).map(|offset| Protected {
        bits: start + offset as usize..end,
        len: Some(SECOND_ICS_PROTECTED_BITS),
    });
    [Some(first), second]
}

/// The `element_instance_tag` of each of `configuration`'s elements, in
/// order: the elements of each kind numbered from 0.
///
/// Spec gap: where `channel_configuration` gives the elements (Table 42),
/// the standard does not say which tags they have, only that the elements
/// of one kind in a `raw_data_block()` differ in tag (clause 8.2.1.1), and
/// that each element keeps its tag from one block to the next (clause
/// 8.2.4): "for the 5.1 channel signal each SCE must have the same value in
/// its element_instance_tag, each CPE associated with the front channel
/// pair must have the same value ..., and each CPE associated with the back
/// channel pair must have the same value". The first of each kind is given
/// tag 0, the next 1, and so on. Settled by decoding with ffmpeg (black
/// box): it decodes mono and stereo streams so tagged, at every sampling
/// frequency, without complaint, and streams of channel configurations 3
/// to 7 so tagged, each channel in its place; the conformance tests are
/// `ffmpeg_decodes_silence_at_every_sample_rate` and
/// `ffmpeg_maps_each_channel_of_configurations_3_to_7`.
///
/// Spec reading: clause 8.2.2 says that elements with tags "must have a
/// unique element_instance_tag in each raw_data_block()", which could be
/// read across kinds, so that an SCE and a CPE could not both have tag 0.
/// It is read within each kind, as clause 8.2.1.1 puts it: "several
/// instances of the same syntactic element may occur, but must have a
/// different 4 bit element_instance_tag. Therefore, in one
/// raw_data_block(), there can be from 0 to at most 16 instances of any
/// syntactic element", which tags unique across kinds would not allow; and
/// a `program_config_element()` tells an SCE from a CPE of the same tag by
/// its `is_cpe` flags (clause 8.5.2). Settled by decoding with ffmpeg (black
/// box): it decodes mono and stereo streams so tagged, at every sampling
/// frequency, without complaint; the conformance test is
/// `ffmpeg_decodes_silence_at_every_sample_rate`.
pub fn element_instance_tags(configuration: ChannelConfiguration) -> impl Iterator<Item = u32> {
    let elements = configuration.elements();
    (0..elements.len()).map(move |i| {
        let kind = elements[i];
        elements[..i].iter().filter(|&&e| e == kind).count() as u32
    })
}

/// The bytes of the shortest `raw_data_block()` of `elements`, those of
/// `configuration`, and `data_bytes` bytes of data streams: the elements,
/// data stream elements and `ID_END`, byte aligned, without fill.
pub fn min_len(
    configuration: ChannelConfiguration,
    elements: &[CodedElement],
    data_bytes: usize,
) -> usize {
    let elements: u32 = elements.iter().map(CodedElement::bits).sum();
    let bits = overhead_bits(configuration) + elements + data_stream::bits(data_bytes);
    bits.div_ceil(8) as usize
}

/// A `raw_data_block()` of `elements` and data stream elements of `data`,
/// padded with fill elements to `len` bytes, or to [`min_len`] if that is
/// longer.
///
/// Spec reading: clause 8.1.3.2.1 lists the bits that `adts_error_check()`
/// protects, and the list is taken as whole: the fill elements, which it
/// does not name, are not protected, nor is `ID_END`, nor the bits that
/// align the block.
///
/// Spec reading: with several elements, the bits that
/// `adts_error_check()` protects are fed "in order of their appearance"
/// (clause 8.1.3.2.1): element by element, each element's 192 bits and a
/// pair's 128 together, as `write_element` gives them, rather than the
/// 192 bits of every element before the 128 of any pair, an order of the
/// clause's list rather than of the bits' appearance.
///
/// # Panics
///
/// If the elements are not those `configuration` gives, in its order: a
/// caller's error.
pub fn write(
    configuration: ChannelConfiguration,
    elements: &[CodedElement],
    data: &[u8],
    len: usize,
) -> Block {
    let kinds: Vec<ElementKind> = elements.iter().map(CodedElement::kind).collect();
    assert_eq!(
        kinds,
        configuration.elements(),
        "the elements of a {configuration:?} frame"
    );
    let mut w = BitWriter::new();
    let mut protected = Vec::new();
    for (element, tag) in elements.iter().zip(element_instance_tags(configuration)) {
        protected.extend(write_element(&mut w, element, tag).into_iter().flatten());
    }
    protected.extend(data_stream::write(&mut w, data));
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
    debug_assert_eq!(
        bytes.len(),
        len.max(min_len(configuration, elements, data.len()))
    );
    Block { bytes, protected }
}

#[cfg(test)]
mod tests {
    use chelys_core::bitstream::binary_digits;

    use super::*;
    use crate::channel::CodedChannel;
    use crate::sampling_frequency::SamplingFrequency;
    use crate::window::Windows;

    fn silent(configuration: ChannelConfiguration) -> Vec<CodedElement> {
        let channel = CodedChannel::silent(Windows::LONG_SINE, SamplingFrequency::Hz44100);
        (configuration.elements().iter())
            .map(|kind| match kind {
                ElementKind::Single => CodedElement::single(channel.clone()),
                ElementKind::Pair => CodedElement::pair(channel.clone(), channel.clone()),
                ElementKind::Lfe => CodedElement::lfe(channel.clone()),
            })
            .collect()
    }

    #[test]
    fn a_silent_mono_block_is_an_sce_and_end() {
        let elements = silent(ChannelConfiguration::Mono);
        let block = write(ChannelConfiguration::Mono, &elements, &[], 0).bytes;
        let sce = ["0000", "00000000", "00000000000", "000"].concat();
        // ID_SCE, the element and ID_END end on a byte.
        let bits = ["000", &sce, "111"].concat();
        assert_eq!(binary_digits(&block), bits);
        assert_eq!(min_len(ChannelConfiguration::Mono, &elements, 0), 4);
    }

    #[test]
    fn a_silent_stereo_block_is_a_cpe_and_end() {
        let elements = silent(ChannelConfiguration::Stereo);
        let block = write(ChannelConfiguration::Stereo, &elements, &[], 0).bytes;
        let ics = ["00000000", "00000000000", "000"].concat();
        let cpe = ["0000", "0", &ics, &ics].concat();
        // ID_CPE, the element, ID_END, then zeros to the byte.
        let bits = ["001", &cpe, "111", "0"].concat();
        assert_eq!(binary_digits(&block), bits);
        assert_eq!(min_len(ChannelConfiguration::Stereo, &elements, 0), 7);
    }

    #[test]
    fn fill_pads_a_block_to_every_longer_length() {
        for configuration in [ChannelConfiguration::Mono, ChannelConfiguration::Stereo] {
            let elements = silent(configuration);
            let shortest = min_len(configuration, &elements, 0);
            let unfilled = write(configuration, &elements, &[], 0).bytes;
            for len in shortest..shortest + 600 {
                let block = write(configuration, &elements, &[], len).bytes;
                assert_eq!(block.len(), len, "{configuration:?}");
                assert_eq!(block[..shortest - 1], unfilled[..shortest - 1]);
            }
        }
    }

    #[test]
    #[should_panic(expected = "the elements of a Mono frame")]
    fn the_channels_are_those_of_the_configuration() {
        write(
            ChannelConfiguration::Mono,
            &silent(ChannelConfiguration::Stereo),
            &[],
            0,
        );
    }

    /// The test decoder reads the element, the data streams' bytes after
    /// it, the fill after them and the end, from blocks of every length,
    /// and each data stream element's bits are protected whole.
    #[test]
    fn data_streams_follow_the_element() {
        use crate::sampling_frequency::SamplingFrequency;
        use decoder::raw_data_block;

        let data: Vec<u8> = (0..700u32).map(|i| (i * 37) as u8).collect();
        for configuration in [ChannelConfiguration::Mono, ChannelConfiguration::Stereo] {
            let elements = silent(configuration);
            let element = &elements[0];
            for data_len in [0, 1, 254, 255, 510, 511, 700] {
                let data = &data[..data_len];
                let shortest = min_len(configuration, &elements, data_len);
                for len in [0, shortest, shortest + 1, shortest + 40] {
                    let block = write(configuration, &elements, data, len);
                    assert_eq!(block.bytes.len(), len.max(shortest));
                    let read = raw_data_block(&block.bytes, SamplingFrequency::Hz44100);
                    let carried: Vec<u8> = (read.data_streams.iter())
                        .flat_map(|stream| {
                            assert_eq!(stream.element_instance_tag, 0);
                            assert!(!stream.data_byte_align_flag);
                            stream.bytes.clone()
                        })
                        .collect();
                    assert_eq!(carried, data);
                    // The element, its data streams, any fill, ID_END.
                    let streams = read.data_streams.len();
                    assert_eq!(read.ids[0], u32::from(element.is_pair()));
                    assert!(read.ids[1..=streams].iter().all(|&id| id == ID_DSE));
                    assert!(
                        read.ids[streams + 1..]
                            .iter()
                            .all(|&id| id == ID_FIL || id == ID_END)
                    );
                    assert_eq!(read.ids.last(), Some(&ID_END));
                    let element_protected = if element.is_pair() { 2 } else { 1 };
                    let protected: Vec<_> = (block.protected[element_protected..].iter())
                        .map(|p| (p.bits.clone(), p.len))
                        .collect();
                    let expected: Vec<_> = (read.data_streams.iter())
                        .map(|stream| (stream.bits.clone(), None))
                        .collect();
                    assert_eq!(protected, expected, "{data_len}, {len}");
                }
            }
        }
    }
}
