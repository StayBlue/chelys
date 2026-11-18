// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The `individual_channel_stream()` (Table 16) of a coded channel.

use chelys_core::bitstream::BitWriter;

use crate::channel::CodedChannel;
use crate::noiseless::scalefactors;

/// Writes the `individual_channel_stream()` (Table 16) of `channel`, with
/// its `ics_info()` (Table 15) unless the window is `common_window`, in
/// the bits [`CodedChannel::bits`] counts.
pub fn write_individual_channel_stream(
    channel: &CodedChannel,
    common_window: bool,
    w: &mut BitWriter,
) {
    let start = w.bit_len();
    w.write(u32::from(channel.global_gain()), 8);
    if !common_window {
        channel.windows().write_ics_info(channel.max_sfb(), w);
    }
    for (group, _) in channel.groups() {
        group.sections.write_section_data(w);
    }
    scalefactors::write(
        channel.global_gain(),
        (channel.groups()).map(|(group, _)| (&group.sections, group.scalefactors.as_slice())),
        w,
    );
    w.write(0, 1); // pulse_data_present
    w.write(0, 1); // tns_data_present
    // gain_control_data_present: gain control is not permitted in the Low
    // Complexity profile (clause 7.1.2).
    w.write(0, 1);
    for (group, lines) in channel.groups() {
        group.sections.write_spectral_data(lines, w);
    }
    debug_assert_eq!(w.bit_len() - start, channel.bits(common_window) as usize);
}

#[cfg(test)]
mod tests {
    use chelys_core::bitstream::binary_digits;

    use super::*;
    use crate::channel::{CodedGroup, FRAME_LINES};
    use crate::noiseless::sections::Sections;
    use crate::sampling_frequency::SamplingFrequency;
    use crate::scalefactor_bands::GroupBands;
    use crate::window::Windows;

    fn written(channel: &CodedChannel, common_window: bool) -> String {
        let mut w = BitWriter::new();
        write_individual_channel_stream(channel, common_window, &mut w);
        let len = w.bit_len();
        assert_eq!(len as u32, channel.bits(common_window));
        binary_digits(&w.into_bytes())[..len].to_owned()
    }

    #[test]
    fn a_silent_stream_holds_its_ics_info_unless_the_window_is_common() {
        let silent = CodedChannel::silent(Windows::LONG_SINE, SamplingFrequency::Hz44100);
        // global_gain, ics_info(), pulse, tns and gain control flags.
        assert_eq!(
            written(&silent, false),
            ["00000000", "00000000000", "000"].concat()
        );
        assert_eq!(written(&silent, true), ["00000000", "000"].concat());
    }

    #[test]
    fn a_stream_sends_its_sections_scalefactors_and_lines_in_order() {
        let fs = SamplingFrequency::Hz48000;
        let mut lines = [0; FRAME_LINES];
        lines[1] = 1;
        let sections = Sections::choose(&lines, GroupBands::long(fs), 1).unwrap();
        let group = CodedGroup {
            sections,
            scalefactors: vec![101],
        };
        let channel = CodedChannel::new(Windows::LONG_SINE, fs, 100, lines, vec![group]).unwrap();
        let bits = written(&channel, false);
        // global_gain 100, then ics_info() with max_sfb 1.
        assert!(bits.starts_with(&["01100100", "0", "00", "0", "000001", "0"].concat()));
        // A section of one band (sect_cb, sect_len_incr), then the
        // scalefactor's difference of +1 (Table A.1: index 61).
        let sections = &bits[19..19 + 4 + 5];
        assert_eq!(&sections[4..], "00001");
        assert_eq!(&bits[28..32], "1010");
        // The three flags follow the scalefactor, then the lines.
        assert_eq!(&bits[32..35], "000");
    }
}
