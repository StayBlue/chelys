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

/// The bits of a `single_channel_element()` besides its channel's
/// `individual_channel_stream()`: `element_instance_tag`.
const SINGLE_CHANNEL_ELEMENT_BITS: u32 = 4;

/// The bits of a `channel_pair_element()` besides its channels'
/// `individual_channel_stream()`s and the fields of a common window:
/// `element_instance_tag` and `common_window`.
const CHANNEL_PAIR_ELEMENT_BITS: u32 = 4 + 1;

/// One element of a frame as it is written: a `single_channel_element()`
/// of one channel, or a `channel_pair_element()` of two, each with its own
/// `ics_info()` (`common_window` 0).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodedElement {
    channels: Vec<CodedChannel>,
}

impl CodedElement {
    pub fn single(channel: CodedChannel) -> Self {
        Self {
            channels: vec![channel],
        }
    }

    pub fn pair(left: CodedChannel, right: CodedChannel) -> Self {
        Self {
            channels: vec![left, right],
        }
    }

    /// Whether the element is a `channel_pair_element()`.
    pub fn is_pair(&self) -> bool {
        self.channels.len() == 2
    }

    /// The bits of the element besides its `id_syn_ele`.
    pub fn bits(&self) -> u32 {
        let channels: u32 = self.channels.iter().map(|c| c.bits(false)).sum();
        Self::fields_bits(self.channels.len()) + channels
    }

    /// The bits of an element of `channels` channels besides their
    /// `individual_channel_stream()`s.
    pub fn fields_bits(channels: usize) -> u32 {
        if channels == 1 {
            SINGLE_CHANNEL_ELEMENT_BITS
        } else {
            CHANNEL_PAIR_ELEMENT_BITS
        }
    }

    /// Writes the element (Table 13 or 14), after its `id_syn_ele`.
    pub fn write(&self, w: &mut BitWriter) {
        let start = w.bit_len();
        w.write(ELEMENT_INSTANCE_TAG, 4);
        if self.is_pair() {
            w.write_bool(false); // common_window
        }
        for channel in &self.channels {
            write_individual_channel_stream(channel, false, w);
        }
        debug_assert_eq!(w.bit_len() - start, self.bits() as usize);
    }
}

#[cfg(test)]
mod tests {
    use chelys_core::bitstream::binary_digits;

    use super::*;
    use crate::sampling_frequency::SamplingFrequency;
    use crate::window::Windows;

    fn written(element: &CodedElement) -> String {
        let mut w = BitWriter::new();
        element.write(&mut w);
        let len = w.bit_len();
        assert_eq!(len as u32, element.bits());
        binary_digits(&w.into_bytes())[..len].to_owned()
    }

    fn silent() -> CodedChannel {
        CodedChannel::silent(Windows::LONG_SINE, SamplingFrequency::Hz48000)
    }

    #[test]
    fn single_channel_element_is_a_tag_and_a_stream() {
        let ics = ["00000000", "00000000000", "000"].concat();
        let bits = written(&CodedElement::single(silent()));
        assert_eq!(bits, ["0000", &ics].concat());
    }

    #[test]
    fn channel_pair_element_has_a_window_for_each_channel() {
        let ics = ["00000000", "00000000000", "000"].concat();
        let bits = written(&CodedElement::pair(silent(), silent()));
        // element_instance_tag, common_window, then the two streams.
        assert_eq!(bits, ["0000", "0", &ics, &ics].concat());
    }
}
