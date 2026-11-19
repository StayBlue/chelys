// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! A reader of `adts_frame()`s (Tables 5 to 10) and `raw_data_block()`s
//! (Table 12), written from their syntax apart from the encoder, for tests
//! to read what it writes: the fields of the headers, the `crc_check`, and
//! where the elements of a block lie, with the bytes of its
//! `data_stream_element()`s (Table 24). It reads an element's channels with
//! [`noiseless::decoder`](crate::noiseless::decoder).

use std::ops::Range;

use crate::noiseless::decoder::{BitReader, Element, element_after_id};
use crate::sampling_frequency::SamplingFrequency;

/// An `adts_frame()` of one `raw_data_block()` as read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdtsFrame<'a> {
    /// The 7 bytes of `adts_fixed_header()` and `adts_variable_header()`.
    pub header: &'a [u8],
    pub protection_absent: bool,
    pub profile: u32,
    pub sampling_frequency_index: u32,
    pub private_bit: bool,
    pub channel_configuration: u32,
    pub original_copy: bool,
    pub home: bool,
    pub copyright_identification_bit: bool,
    pub copyright_identification_start: bool,
    pub aac_frame_length: usize,
    pub adts_buffer_fullness: u32,
    /// `crc_check` where `protection_absent` is 0.
    pub crc_check: Option<u16>,
    pub raw_data_block: &'a [u8],
}

/// Reads the `adts_frame()` at the start of `stream`, which must hold one
/// `raw_data_block()` (`number_of_raw_data_blocks_in_frame` 0).
pub fn adts_frame(stream: &[u8]) -> AdtsFrame<'_> {
    let mut r = BitReader::new(stream);
    // adts_fixed_header()
    assert_eq!(r.read(12), 0xFFF, "syncword");
    assert_eq!(r.read(1), 1, "ID");
    assert_eq!(r.read(2), 0, "layer");
    let protection_absent = r.read(1) == 1;
    let profile = r.read(2);
    let sampling_frequency_index = r.read(4);
    let private_bit = r.read(1) == 1;
    let channel_configuration = r.read(3);
    let original_copy = r.read(1) == 1;
    let home = r.read(1) == 1;
    // adts_variable_header()
    let copyright_identification_bit = r.read(1) == 1;
    let copyright_identification_start = r.read(1) == 1;
    let aac_frame_length = r.read(13) as usize;
    let adts_buffer_fullness = r.read(11);
    assert_eq!(r.read(2), 0, "number_of_raw_data_blocks_in_frame");
    // adts_error_check()
    let crc_check = (!protection_absent).then(|| r.read(16) as u16);
    let start = r.position() / 8;
    AdtsFrame {
        header: &stream[..7],
        protection_absent,
        profile,
        sampling_frequency_index,
        private_bit,
        channel_configuration,
        original_copy,
        home,
        copyright_identification_bit,
        copyright_identification_start,
        aac_frame_length,
        adts_buffer_fullness,
        crc_check,
        raw_data_block: &stream[start..aac_frame_length],
    }
}

/// A `data_stream_element()` as read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DataStream {
    pub element_instance_tag: u32,
    pub data_byte_align_flag: bool,
    /// `data_stream_byte[element_instance_tag][i]`.
    pub bytes: Vec<u8>,
    /// Its bits in the block, after its `id_syn_ele`.
    pub bits: Range<usize>,
}

/// A `raw_data_block()` of one audio channel element as read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub element: Element,
    /// The element's bits in the block, after its `id_syn_ele`.
    pub element_bits: Range<usize>,
    /// Where a pair's second `individual_channel_stream()` starts.
    pub second_channel_start: Option<usize>,
    /// The block's `data_stream_element()`s, in order.
    pub data_streams: Vec<DataStream>,
    /// The `id_syn_ele` of each element in order, `ID_END` last.
    pub ids: Vec<u32>,
}

/// Reads a `raw_data_block()` at `fs` that holds one
/// `single_channel_element()` or `channel_pair_element()`, any
/// `data_stream_element()`s and `fill_element()`s, and `ID_END`, then aligns
/// to its byte (Table 12), which must be its last.
pub fn raw_data_block(block: &[u8], fs: SamplingFrequency) -> Block {
    let mut r = BitReader::new(block);
    let mut element = None;
    let mut data_streams = Vec::new();
    let mut ids = Vec::new();
    loop {
        let id = r.read(3);
        ids.push(id);
        let start = r.position();
        match id {
            0 | 1 => {
                assert!(element.is_none(), "one audio channel element");
                let (read, second) = element_after_id(&mut r, fs, id);
                element = Some((read, start..r.position(), second));
            }
            4 => {
                let element_instance_tag = r.read(4);
                let data_byte_align_flag = r.read(1) == 1;
                let mut cnt = r.read(8) as usize; // count
                if cnt == 255 {
                    cnt += r.read(8) as usize; // esc_count
                }
                if data_byte_align_flag {
                    // byte_alignment(), to the first bit of the block.
                    while r.position() % 8 != 0 {
                        r.read(1);
                    }
                }
                let bytes = (0..cnt).map(|_| r.read(8) as u8).collect();
                data_streams.push(DataStream {
                    element_instance_tag,
                    data_byte_align_flag,
                    bytes,
                    bits: start..r.position(),
                });
            }
            6 => {
                // fill_element(): count, esc_count, then cnt bytes.
                let mut cnt = r.read(4);
                if cnt == 15 {
                    cnt += r.read(8) - 1;
                }
                for _ in 0..cnt {
                    r.read(8);
                }
            }
            7 => break,
            other => panic!("element {other}"),
        }
    }
    while r.position() % 8 != 0 {
        assert_eq!(r.read(1), 0, "byte_alignment()");
    }
    assert_eq!(r.position(), 8 * block.len(), "the block ends aligned");
    let (element, element_bits, second_channel_start) = element.expect("an element");
    Block {
        element,
        element_bits,
        second_channel_start,
        data_streams,
        ids,
    }
}
