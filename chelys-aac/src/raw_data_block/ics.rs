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
        (channel.groups()).map(|(group, _)| group.scalefactor_data()),
        w,
    );
    channel.pulses().write(w);
    channel.tns().write(w);
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
    use crate::noiseless::pulses::Pulses;
    use crate::noiseless::sections::Sections;
    use crate::sampling_frequency::SamplingFrequency;
    use crate::scalefactor_bands::GroupBands;
    use crate::tns::Tns;
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
            positions: Vec::new(),
            noise_energies: Vec::new(),
        };
        let channel = CodedChannel::new(
            Windows::LONG_SINE,
            fs,
            100,
            Tns::NONE,
            Pulses::NONE,
            lines,
            vec![group],
        )
        .unwrap();
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

    /// A channel of `lines` with the pulses at `at` clipped, all its bands
    /// sent with the scalefactor 100.
    fn with_pulses(fs: SamplingFrequency, lines: [i32; FRAME_LINES], at: &[usize]) -> CodedChannel {
        let bands = GroupBands::long(fs);
        let amps: Vec<_> = (at.iter())
            .map(|&line| (line, lines[line].unsigned_abs() - 1))
            .collect();
        let pulses = Pulses::new(bands, &amps).expect("pulses the syntax sends");
        let mut sent = lines;
        pulses.clip(&mut sent);
        let max_sfb = crate::noiseless::sections::max_sfb(&sent, bands);
        let group = CodedGroup {
            sections: Sections::choose(&sent, bands, max_sfb).unwrap(),
            scalefactors: vec![100; max_sfb],
            positions: Vec::new(),
            noise_energies: Vec::new(),
        };
        CodedChannel::new(
            Windows::LONG_SINE,
            fs,
            100,
            Tns::NONE,
            pulses,
            sent,
            vec![group],
        )
        .unwrap()
    }

    /// The decoder reads `pulse_data()` after the scalefactors, and restores
    /// the lines it clips as clause 9.3 does, in the bits counted.
    #[test]
    fn pulses_are_sent_after_the_scalefactors_and_restore_the_lines() {
        use crate::noiseless::decoder::{BitReader, PulseData, individual_channel_stream};
        let fs = SamplingFrequency::Hz44100;
        let mut lines = [0; FRAME_LINES];
        lines[1] = 1;
        lines[450] = -2;
        lines[481] = 16;
        lines[490] = 8;
        lines[491] = -1;
        let channel = with_pulses(fs, lines, &[450, 481, 490]);
        assert_eq!(channel.quantized_lines(), lines);
        assert_eq!((channel.lines()[450], channel.lines()[481]), (-1, 1));
        let bits = written(&channel, false);
        let mut w = BitWriter::new();
        write_individual_channel_stream(&channel, false, &mut w);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        let ics = individual_channel_stream(&mut r, fs, None, false, false);
        assert_eq!(r.position(), bits.len());
        assert_eq!(
            ics.pulses,
            Some(PulseData {
                number_pulse: 2,
                pulse_start_sfb: 33,
                pulse_offset: vec![2, 31, 9],
                pulse_amp: vec![1, 15, 7],
            })
        );
        assert_eq!(ics.lines, [lines.to_vec()]);
    }

    /// Pulses anywhere their fields reach, at every sampling frequency,
    /// decode to the lines they were clipped from.
    #[test]
    fn pulses_restore_their_lines_at_every_sampling_frequency() {
        use crate::noiseless::decoder::{BitReader, Noise, individual_channel_stream};
        let mut noise = Noise::new(10);
        for fs in SamplingFrequency::ALL {
            let bands = GroupBands::long(fs);
            for _ in 0..50 {
                let mut lines = [0; FRAME_LINES];
                for line in &mut lines {
                    if noise.below(3) == 0 {
                        *line = noise.below(5) as i32 - 2;
                    }
                }
                // 1 to 4 pulses, the first within reach of its band's start.
                let sfb = noise.below(bands.num_swb() as u32) as usize;
                let band = bands.band(sfb);
                let mut line = band.start + noise.below(band.len().min(32) as u32) as usize;
                let mut at = Vec::new();
                for _ in 0..=noise.below(4) {
                    if line >= FRAME_LINES {
                        break;
                    }
                    let magnitude = 2 + noise.below(15) as i32;
                    lines[line] = if noise.below(2) == 0 {
                        magnitude
                    } else {
                        -magnitude
                    };
                    at.push(line);
                    line += 1 + noise.below(31) as usize;
                }
                let channel = with_pulses(fs, lines, &at);
                let mut w = BitWriter::new();
                write_individual_channel_stream(&channel, false, &mut w);
                assert_eq!(w.bit_len() as u32, channel.bits(false));
                let bytes = w.into_bytes();
                let ics =
                    individual_channel_stream(&mut BitReader::new(&bytes), fs, None, false, false);
                assert_eq!(ics.lines, [lines.to_vec()], "{fs:?}, pulses at {at:?}");
            }
        }
    }
}
