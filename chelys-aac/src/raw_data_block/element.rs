// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The audio channel elements: the `single_channel_element()` (Table 13)
//! and the `channel_pair_element()` (Table 14).

use chelys_core::bitstream::BitWriter;

use super::ics::write_individual_channel_stream;
use crate::channel::CodedChannel;
use crate::channel_configuration::ElementKind;
use crate::stereo::middle_side::MsMask;
use crate::window::Windows;

/// The bits of a `single_channel_element()` besides its channel's
/// `individual_channel_stream()`: `element_instance_tag`.
const SINGLE_CHANNEL_ELEMENT_BITS: u32 = 4;

/// The bits of a `channel_pair_element()` besides its channels'
/// `individual_channel_stream()`s and the fields of a common window:
/// `element_instance_tag` and `common_window`.
const CHANNEL_PAIR_ELEMENT_BITS: u32 = 4 + 1;

/// One element of a frame as it is written: a `single_channel_element()`
/// of one channel, or a `channel_pair_element()` of two.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodedElement {
    kind: ElementKind,
    channels: Vec<CodedChannel>,
    /// For a pair that shares its window, `common_window` 1, the bands it
    /// codes as middle and side.
    common: Option<MsMask>,
}

impl CodedElement {
    pub fn single(channel: CodedChannel) -> Self {
        Self {
            kind: ElementKind::Single,
            channels: vec![channel],
            common: None,
        }
    }

    /// A pair whose channels each send their own `ics_info()`:
    /// `common_window` 0.
    pub fn pair(left: CodedChannel, right: CodedChannel) -> Self {
        Self {
            kind: ElementKind::Pair,
            channels: vec![left, right],
            common: None,
        }
    }

    /// A pair that shares one `ics_info()`, `common_window` 1, with the
    /// bands `ms` codes as middle and side.
    ///
    /// # Panics
    ///
    /// If the channels differ in their windows or `max_sfb`: a caller's
    /// error.
    pub fn common_pair(left: CodedChannel, right: CodedChannel, ms: MsMask) -> Self {
        assert_eq!(
            (left.windows(), left.max_sfb()),
            (right.windows(), right.max_sfb()),
            "a common window shared by channels of different windows"
        );
        Self {
            kind: ElementKind::Pair,
            channels: vec![left, right],
            common: Some(ms),
        }
    }

    /// Which element it is.
    pub fn kind(&self) -> ElementKind {
        self.kind
    }

    /// The element's channels.
    pub fn channels(&self) -> &[CodedChannel] {
        &self.channels
    }

    /// Whether the element is a `channel_pair_element()`.
    pub fn is_pair(&self) -> bool {
        self.kind == ElementKind::Pair
    }

    /// The bits of the element besides its `id_syn_ele`.
    pub fn bits(&self) -> u32 {
        let common = self.common.is_some();
        let channels: u32 = self.channels.iter().map(|c| c.bits(common)).sum();
        let mask = match (&self.common, self.channels.first()) {
            (Some(ms), Some(channel)) => {
                let groups = channel.groups().count();
                ms.flag_bits(groups, channel.max_sfb())
            }
            _ => 0,
        };
        let windows = self.channels.first().map(CodedChannel::windows);
        Self::fields_bits(self.channels.len(), windows.filter(|_| common)) + mask + channels
    }

    /// Where a pair's second `individual_channel_stream()` starts, in bits
    /// from the start of the element after its `id_syn_ele`; `None` for a
    /// single channel.
    pub fn second_channel_offset(&self) -> Option<u32> {
        // The second stream ends the element.
        let [_, right] = self.channels.as_slice() else {
            return None;
        };
        Some(self.bits() - right.bits(self.common.is_some()))
    }

    /// The bits of an element of `channels` channels besides their
    /// `individual_channel_stream()`s and `ms_used` flags: where a pair
    /// shares its `windows`, they include their `ics_info()` and
    /// `ms_mask_present`.
    pub fn fields_bits(channels: usize, common_window: Option<Windows>) -> u32 {
        match (channels, common_window) {
            (1, _) => SINGLE_CHANNEL_ELEMENT_BITS,
            (_, None) => CHANNEL_PAIR_ELEMENT_BITS,
            (_, Some(windows)) => CHANNEL_PAIR_ELEMENT_BITS + windows.ics_info_bits() + 2,
        }
    }

    /// Writes the element (Table 13 or 14), after its `id_syn_ele`, with
    /// `element_instance_tag`.
    pub fn write(&self, w: &mut BitWriter, element_instance_tag: u32) {
        let start = w.bit_len();
        w.write(element_instance_tag, 4);
        if self.is_pair() {
            w.write_bool(self.common.is_some());
            if let (Some(ms), Some(channel)) = (&self.common, self.channels.first()) {
                // The ics_info() sent once for both: their windows and
                // max_sfb.
                let max_sfb = channel.max_sfb();
                channel.windows().write_ics_info(max_sfb, w);
                ms.write(channel.groups().count(), max_sfb, w);
            }
        }
        for channel in &self.channels {
            write_individual_channel_stream(channel, self.common.is_some(), w);
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
        element.write(&mut w, 0);
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
        let element = CodedElement::single(silent());
        assert_eq!(written(&element), ["0000", &ics].concat());
        assert_eq!(element.second_channel_offset(), None);
    }

    #[test]
    fn channel_pair_element_has_a_window_for_each_channel() {
        let ics = ["00000000", "00000000000", "000"].concat();
        let element = CodedElement::pair(silent(), silent());
        // element_instance_tag, common_window, then the two streams.
        assert_eq!(written(&element), ["0000", "0", &ics, &ics].concat());
        assert_eq!(element.second_channel_offset(), Some(5 + 22));
    }

    #[test]
    fn a_common_window_is_sent_once_with_the_mask() {
        let fs = SamplingFrequency::Hz48000;
        let silent = CodedChannel::silent(Windows::LONG_SINE, fs);
        let ics = ["00000000", "000"].concat();
        let ics_info = "00000000000";
        let element = CodedElement::common_pair(silent.clone(), silent.clone(), MsMask::NONE);
        // element_instance_tag, common_window, ics_info(),
        // ms_mask_present, then the two streams without their ics_info().
        assert_eq!(
            written(&element),
            ["0000", "1", ics_info, "00", &ics, &ics].concat()
        );
        assert_eq!(element.second_channel_offset(), Some(5 + 11 + 2 + 11));
        // A mask of max_sfb bands of each group.
        let mut left = silent.clone();
        let mut right = silent;
        left.extend(2);
        right.extend(2);
        let mut ms = MsMask::NONE;
        ms.set(0, 1, true);
        let bits = written(&CodedElement::common_pair(left, right, ms));
        assert_eq!(
            &bits[..5 + 11 + 2 + 2],
            ["0000", "1", "00000000100", "01", "01"].concat()
        );
    }

    #[test]
    #[should_panic(expected = "different windows")]
    fn a_common_window_is_one_window() {
        let fs = SamplingFrequency::Hz48000;
        let mut right = CodedChannel::silent(Windows::LONG_SINE, fs);
        right.extend(1);
        CodedElement::common_pair(
            CodedChannel::silent(Windows::LONG_SINE, fs),
            right,
            MsMask::NONE,
        );
    }
}
