// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The audio channel elements: the `single_channel_element()` (Table 13)
//! and the `channel_pair_element()` (Table 14).

use chelys_core::bitstream::BitWriter;

use super::ics::write_individual_channel_stream;
use crate::channel::CodedChannel;

/// The `element_instance_tag` of every element.
///
/// Spec gap: where `channel_configuration` gives the elements (Table 42),
/// the standard does not say which tags they have, only that the elements
/// of one kind in a raw_data_block() differ in tag and that a mono or
/// stereo stream keeps its element's tag throughout (clause 8.2.4). Each
/// stream here has one element, which is given the first tag. Settled by
/// decoding with ffmpeg (black box): it decodes mono and stereo streams so
/// tagged, at every sampling frequency, without complaint; the conformance
/// test is `ffmpeg_decodes_silence_at_every_sample_rate`.
const ELEMENT_INSTANCE_TAG: u32 = 0;

/// `common_window` of every `channel_pair_element()`.
///
/// Spec gap: Annex C does not say when the two channels of a pair share
/// their `ics_info()`, which makes them share their windows and `max_sfb`,
/// and lets them be coded as middle and side (clause 12.1). While each
/// channel is quantized apart, with its own `max_sfb`, and middle and side
/// are not coded, the channels do not share it.
pub const COMMON_WINDOW: bool = false;

/// The bits of a `single_channel_element()` besides its channel's
/// `individual_channel_stream()`: `element_instance_tag`.
pub const SINGLE_CHANNEL_ELEMENT_BITS: u32 = 4;

/// The bits of a `channel_pair_element()` besides its channels'
/// `individual_channel_stream()`s, with `common_window` 0:
/// `element_instance_tag` and `common_window`.
pub const CHANNEL_PAIR_ELEMENT_BITS: u32 = 4 + 1;

/// Writes a `single_channel_element()` (Table 13), after its `id_syn_ele`.
pub fn write_single_channel_element(channel: &CodedChannel, w: &mut BitWriter) {
    w.write(ELEMENT_INSTANCE_TAG, 4);
    write_individual_channel_stream(channel, false, w);
}

/// Writes a `channel_pair_element()` (Table 14), after its `id_syn_ele`.
pub fn write_channel_pair_element(left: &CodedChannel, right: &CodedChannel, w: &mut BitWriter) {
    // The ics_info() sent once for both: their windows and max_sfb.
    if COMMON_WINDOW {
        debug_assert_eq!(
            (left.windows(), left.max_sfb()),
            (right.windows(), right.max_sfb()),
            "a common window shared by channels of different windows"
        );
    }
    w.write(ELEMENT_INSTANCE_TAG, 4);
    w.write_bool(COMMON_WINDOW);
    write_individual_channel_stream(left, COMMON_WINDOW, w);
    write_individual_channel_stream(right, COMMON_WINDOW, w);
}

/// The bits of `channel` in its element, which is single or one of a
/// pair with `common_window` 0.
pub fn channel_bits(channel: &CodedChannel) -> u32 {
    channel.bits(COMMON_WINDOW)
}

#[cfg(test)]
mod tests {
    use chelys_core::bitstream::binary_digits;

    use super::*;
    use crate::sampling_frequency::SamplingFrequency;
    use crate::window::Windows;

    fn written(write: impl FnOnce(&mut BitWriter)) -> String {
        let mut w = BitWriter::new();
        write(&mut w);
        let len = w.bit_len();
        binary_digits(&w.into_bytes())[..len].to_owned()
    }

    fn silent() -> CodedChannel {
        CodedChannel::silent(Windows::LONG_SINE, SamplingFrequency::Hz48000)
    }

    #[test]
    fn single_channel_element_is_a_tag_and_a_stream() {
        let ics = ["00000000", "00000000000", "000"].concat();
        let bits = written(|w| write_single_channel_element(&silent(), w));
        assert_eq!(bits, ["0000", &ics].concat());
        assert_eq!(
            bits.len() as u32,
            SINGLE_CHANNEL_ELEMENT_BITS + channel_bits(&silent())
        );
    }

    #[test]
    fn channel_pair_element_has_a_window_for_each_channel() {
        let ics = ["00000000", "00000000000", "000"].concat();
        let bits = written(|w| write_channel_pair_element(&silent(), &silent(), w));
        // element_instance_tag, common_window, then the two streams.
        assert_eq!(bits, ["0000", "0", &ics, &ics].concat());
        assert_eq!(
            bits.len() as u32,
            CHANNEL_PAIR_ELEMENT_BITS + 2 * channel_bits(&silent())
        );
    }
}
