// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The audio channel elements: the `single_channel_element()` (Table 13)
//! and the `channel_pair_element()` (Table 14), of channels whose spectra
//! are all zero.

use chelys_core::bitstream::BitWriter;

use super::ics::{write_silent_ics_info, write_silent_individual_channel_stream};

/// The `element_instance_tag` of every element.
///
/// Spec gap: where `channel_configuration` gives the elements (Table 42),
/// the standard does not say which tags they have, only that the elements
/// of one kind in a raw_data_block() differ in tag and that a mono or
/// stereo stream keeps its element's tag throughout (clause 8.2.4). Each
/// stream here has one element, which is given the first tag.
const ELEMENT_INSTANCE_TAG: u32 = 0;

/// `ms_mask_present` (clause 12.1.2.1): no band is coded as middle and
/// side.
const MS_MASK_ALL_ZEROS: u32 = 0b00;

/// Writes a `single_channel_element()` (Table 13), after its `id_syn_ele`.
pub fn write_silent_single_channel_element(w: &mut BitWriter) {
    w.write(ELEMENT_INSTANCE_TAG, 4);
    write_silent_individual_channel_stream(w, false);
}

/// Writes a `channel_pair_element()` (Table 14), after its `id_syn_ele`.
/// The two channels share their `ics_info()`, as their windows are alike.
pub fn write_silent_channel_pair_element(w: &mut BitWriter) {
    w.write(ELEMENT_INSTANCE_TAG, 4);
    w.write(1, 1); // common_window
    write_silent_ics_info(w);
    w.write(MS_MASK_ALL_ZEROS, 2);
    // ms_used[][]: none, as ms_mask_present is not 1.
    write_silent_individual_channel_stream(w, true);
    write_silent_individual_channel_stream(w, true);
}

#[cfg(test)]
mod tests {
    use chelys_core::bitstream::binary_digits;

    use super::*;

    fn written(write: impl FnOnce(&mut BitWriter)) -> String {
        let mut w = BitWriter::new();
        write(&mut w);
        let len = w.bit_len();
        binary_digits(&w.into_bytes())[..len].to_owned()
    }

    #[test]
    fn single_channel_element_is_a_tag_and_a_stream() {
        let ics = ["00000000", "00000000000", "000"].concat();
        assert_eq!(
            written(write_silent_single_channel_element),
            ["0000", &ics].concat()
        );
    }

    #[test]
    fn channel_pair_element_shares_its_window() {
        let stream = ["00000000", "000"].concat();
        assert_eq!(
            written(write_silent_channel_pair_element),
            // element_instance_tag, common_window, ics_info(),
            // ms_mask_present, then the two streams.
            ["0000", "1", "00000000000", "00", &stream, &stream].concat()
        );
    }
}
