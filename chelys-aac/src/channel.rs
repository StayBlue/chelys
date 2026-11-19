// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! One channel of one frame as its `individual_channel_stream()` (Table
//! 16) codes it: its windows, `global_gain`, its pulses and temporal noise
//! shaping, and for each window group its sections, scalefactors and
//! quantized lines.

use crate::noiseless::pulses::Pulses;
use crate::noiseless::scalefactors;
use crate::noiseless::sections::Sections;
use crate::sampling_frequency::SamplingFrequency;
use crate::scalefactor_bands::Transform;
use crate::stream::FRAME_SAMPLES;
use crate::tns::Tns;
use crate::window::Windows;

/// The spectral lines of a channel in a frame: one for each sample it
/// codes, in one long window or eight short ones (clause 8.3.3).
pub const FRAME_LINES: usize = FRAME_SAMPLES;

/// The bits of an `individual_channel_stream()` (Table 16) besides its
/// `ics_info()`, pulses, temporal noise shaping and noiseless coding:
/// `global_gain`, and the `gain_control_data_present` flag, which is 0.
const FIXED_BITS: u32 = 8 + 1;

/// The `global_gain` of a channel without spectrum.
///
/// Spec gap: `global_gain` is the start value of the scalefactors (clause
/// 11.3.2), of which a channel that transmits no scalefactor band has
/// none, and Annex C's start value of the quantiser step (C.7.2.1) is not
/// defined for a spectrum without a line above zero. Any value decodes
/// alike; 0 is taken. Settled by decoding with ffmpeg (black box): it
/// decodes channels so coded, at every sampling frequency, to digital
/// silence without complaint; the conformance test is
/// `ffmpeg_decodes_silence_at_every_sample_rate`.
const SILENT_GLOBAL_GAIN: u8 = 0;

/// A window group as a channel codes it: its sections, and the scalefactor
/// and intensity stereo position of each of its bands below `max_sfb`, of
/// which those of bands that send none are not used (see
/// [`scalefactors`]); a group without bands in intensity stereo may have
/// no positions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodedGroup {
    pub sections: Sections,
    pub scalefactors: Vec<u8>,
    pub positions: Vec<i8>,
}

impl CodedGroup {
    /// What `scale_factor_data()` sends of the group.
    pub fn scalefactor_data(&self) -> scalefactors::Group<'_> {
        (&self.sections, &self.scalefactors, &self.positions)
    }
}

/// One channel of one frame, quantized and ready for writing: everything
/// its `individual_channel_stream()` sends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodedChannel {
    windows: Windows,
    global_gain: u8,
    /// The filters that shaped its lines, with which the decoder restores
    /// them.
    tns: Tns,
    /// The lines of its long window that `lines` hold clipped, which the
    /// decoder restores.
    pulses: Pulses,
    /// The lines in the order they are sent (clause 8.3.5), as
    /// `spectral_data()` sends them: the quantized lines, those of the
    /// pulses clipped.
    lines: [i32; FRAME_LINES],
    groups: Vec<CodedGroup>,
    /// The bits of the scalefactors of all groups.
    scalefactor_bits: u32,
}

impl CodedChannel {
    /// A channel that sends the lines `lines`, in the window groups of
    /// `windows`, shaped by the filters of `tns`, with the lines of
    /// `pulses` clipped, each group with its sections chosen for its lines
    /// and its scalefactors; or `None` if a difference of scalefactors has
    /// no codeword (see [`scalefactors::bits`]).
    ///
    /// # Panics
    ///
    /// If the groups are not those of `windows` at the sections' sampling
    /// frequency, or do not share one `max_sfb`, or there are pulses in
    /// short windows: a caller's error. Lines other than those the sections
    /// were chosen for panic when written.
    pub fn new(
        windows: Windows,
        sampling_frequency: SamplingFrequency,
        global_gain: u8,
        tns: Tns,
        pulses: Pulses,
        lines: [i32; FRAME_LINES],
        groups: Vec<CodedGroup>,
    ) -> Option<Self> {
        assert!(
            groups
                .iter()
                .map(|group| group.sections.bands())
                .eq(windows.groups(sampling_frequency)),
            "the groups of the windows"
        );
        let max_sfb = groups.first().map_or(0, |group| group.sections.max_sfb());
        assert!(
            groups
                .iter()
                .all(|group| group.sections.max_sfb() == max_sfb),
            "one max_sfb for every group"
        );
        assert!(
            !pulses.is_present() || windows.sequence.transform() == Transform::Long,
            "pulses in short windows"
        );
        let scalefactor_bits =
            scalefactors::bits(global_gain, groups.iter().map(CodedGroup::scalefactor_data))?;
        Some(Self {
            windows,
            global_gain,
            tns,
            pulses,
            lines,
            groups,
            scalefactor_bits,
        })
    }

    /// A channel whose lines are all zero, without temporal noise shaping
    /// (see [`without_lines`](Self::without_lines)).
    pub fn silent(windows: Windows, sampling_frequency: SamplingFrequency) -> Self {
        Self::without_lines(windows, sampling_frequency, Tns::NONE)
    }

    /// A channel whose lines are all zero: `max_sfb` 0, so that its
    /// `section_data()`, `scale_factor_data()` and `spectral_data()` are
    /// empty. "The spectral information for all scalefactor bands at and
    /// above max_sfb, for which there is no section data, is zero" (clause
    /// 9.3). It sends the filters of `tns`, which shape the lines the
    /// decoder gives it where it is the channel of a pair coded as middle
    /// and side, its lines and the other's combined (clause 12.1.3).
    pub fn without_lines(
        windows: Windows,
        sampling_frequency: SamplingFrequency,
        tns: Tns,
    ) -> Self {
        let lines = [0; FRAME_LINES];
        let groups = (windows.groups(sampling_frequency))
            .map(|bands| CodedGroup {
                sections: Sections::choose(&lines[..bands.lines()], bands, 0)
                    .expect("zeros are codable"),
                scalefactors: Vec::new(),
                positions: Vec::new(),
            })
            .collect();
        Self::new(
            windows,
            sampling_frequency,
            SILENT_GLOBAL_GAIN,
            tns,
            Pulses::NONE,
            lines,
            groups,
        )
        .expect("no scalefactor is sent")
    }

    pub fn windows(&self) -> Windows {
        self.windows
    }

    pub fn global_gain(&self) -> u8 {
        self.global_gain
    }

    pub fn tns(&self) -> &Tns {
        &self.tns
    }

    pub fn pulses(&self) -> &Pulses {
        &self.pulses
    }

    /// The lines in the order they are sent, as `spectral_data()` sends
    /// them, with the lines of the [`pulses`](Self::pulses) clipped. Only
    /// their magnitudes differ from the quantized lines: none is zero
    /// that is not.
    pub fn lines(&self) -> &[i32; FRAME_LINES] {
        &self.lines
    }

    /// The quantized lines in the order they are sent, those of the pulses
    /// restored, as the decoder restores them (clause 9.3).
    #[cfg(test)]
    pub fn quantized_lines(&self) -> [i32; FRAME_LINES] {
        let mut lines = self.lines;
        for (line, amp) in self.pulses.iter() {
            let q = &mut lines[line];
            *q += q.signum() * amp as i32;
        }
        lines
    }

    /// `max_sfb`: the scalefactor bands each group sends.
    pub fn max_sfb(&self) -> usize {
        self.groups[0].sections.max_sfb()
    }

    /// The bits that [`extend`](Self::extend) adds to the channel to make
    /// its groups span `max_sfb` bands.
    ///
    /// # Panics
    ///
    /// If `max_sfb` is below the channel's or beyond its bands: a caller's
    /// error.
    pub fn extension_bits(&self, max_sfb: usize) -> u32 {
        (self.groups.iter())
            .map(|group| group.sections.extension_bits(max_sfb))
            .sum()
    }

    /// Makes every group of the channel span `max_sfb` bands, those of its
    /// pair where the pair shares its window: the bands added send nothing
    /// (see [`Sections::extend`]).
    ///
    /// # Panics
    ///
    /// As [`extension_bits`](Self::extension_bits).
    pub fn extend(&mut self, max_sfb: usize) {
        for group in &mut self.groups {
            group.sections.extend(max_sfb);
            group
                .scalefactors
                .resize(max_sfb.max(group.scalefactors.len()), 0);
        }
    }

    /// Each window group with its lines, in the order and as they are sent
    /// (see [`lines`](Self::lines)).
    pub fn groups(&self) -> impl Iterator<Item = (&CodedGroup, &[i32])> + Clone + '_ {
        let mut start = 0;
        self.groups.iter().map(move |group| {
            let end = start + group.sections.bands().lines();
            let lines = &self.lines[start..end];
            start = end;
            (group, lines)
        })
    }

    /// The bits of the channel's `individual_channel_stream()`, with its
    /// `ics_info()` unless the window is `common_window`.
    pub fn bits(&self, common_window: bool) -> u32 {
        let sections: u32 = (self.groups.iter())
            .map(|group| group.sections.section_data_bits() + group.sections.spectral_data_bits())
            .sum();
        bits(
            self.windows,
            common_window,
            &self.pulses,
            &self.tns,
            sections + self.scalefactor_bits,
        )
    }
}

/// The bits of an `individual_channel_stream()` of `windows` with
/// `common_window`, the pulses `pulses`, the temporal noise shaping `tns`
/// and the given noiseless coding, as [`CodedChannel::bits`] counts them,
/// for counting a coding without making a channel of it.
pub fn bits(
    windows: Windows,
    common_window: bool,
    pulses: &Pulses,
    tns: &Tns,
    noiseless_bits: u32,
) -> u32 {
    let ics_info = if common_window {
        0
    } else {
        windows.ics_info_bits()
    };
    FIXED_BITS + ics_info + pulses.bits() + tns.bits() + noiseless_bits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalefactor_bands::GroupBands;

    #[test]
    fn a_silent_channel_sends_no_band() {
        let fs = SamplingFrequency::Hz32000;
        let channel = CodedChannel::silent(Windows::LONG_SINE, fs);
        assert_eq!((channel.max_sfb(), channel.global_gain()), (0, 0));
        // global_gain, ics_info() and the three flags.
        assert_eq!(channel.bits(false), 8 + 11 + 3);
        assert_eq!(channel.bits(true), 8 + 3);
        let groups: Vec<_> = channel.groups().collect();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].1.len(), FRAME_LINES);
    }

    #[test]
    fn a_silent_channel_of_short_windows_sends_no_band_in_any_group() {
        use crate::window::{Grouping, WindowSequence, WindowShape};
        let fs = SamplingFrequency::Hz48000;
        let grouping = Grouping::from_starts([true, false, true, true, false, false, true, true]);
        let windows = Windows {
            sequence: WindowSequence::EightShort { grouping },
            shape: WindowShape::Kbd,
        };
        let channel = CodedChannel::silent(windows, fs);
        assert_eq!(channel.max_sfb(), 0);
        // global_gain, an ics_info() of 15 bits and the three flags: no
        // group sends a section.
        assert_eq!(channel.bits(false), 8 + 15 + 3);
        let lens: Vec<usize> = channel.groups().map(|(_, lines)| lines.len()).collect();
        assert_eq!(lens, [256, 128, 384, 128, 128]);
    }

    #[test]
    fn bits_count_every_part() {
        let fs = SamplingFrequency::Hz48000;
        let bands = GroupBands::long(fs);
        let mut lines = [0; FRAME_LINES];
        lines[0] = 3;
        lines[9] = -1;
        let sections = Sections::choose(&lines, bands, 3).unwrap();
        let noiseless = sections.section_data_bits() + sections.spectral_data_bits();
        let group = CodedGroup {
            sections,
            scalefactors: vec![120, 118, 121],
            positions: Vec::new(),
        };
        let sf_bits = scalefactors::bits(120, [group.scalefactor_data()]).unwrap();
        let channel = CodedChannel::new(
            Windows::LONG_SINE,
            fs,
            120,
            Tns::NONE,
            Pulses::NONE,
            lines,
            vec![group.clone()],
        )
        .unwrap();
        assert_eq!(channel.max_sfb(), 3);
        assert_eq!(
            channel.bits(false),
            bits(
                Windows::LONG_SINE,
                false,
                &Pulses::NONE,
                &Tns::NONE,
                noiseless + sf_bits
            )
        );
        assert_eq!(channel.bits(false), 8 + 11 + 3 + noiseless + sf_bits);
        // A filter adds its tns_data(): n_filt, coef_res, length, order,
        // direction, coef_compress and two coefficients of 4 bits.
        let mut tns = Tns::new(crate::scalefactor_bands::Transform::Long);
        let window = tns.window_mut(0);
        window.n_filt = 1;
        window.filters[0].length = 30;
        window.filters[0].order = 2;
        let channel = CodedChannel::new(
            Windows::LONG_SINE,
            fs,
            120,
            tns,
            Pulses::NONE,
            lines,
            vec![group],
        );
        let channel = channel.unwrap();
        let tns_data = 2 + 1 + 6 + 5 + 1 + 1 + 2 * 4;
        assert_eq!(
            channel.bits(false),
            8 + 11 + 3 + tns_data + noiseless + sf_bits
        );
    }

    #[test]
    fn a_difference_without_codeword_is_refused() {
        let fs = SamplingFrequency::Hz48000;
        let mut lines = [0; FRAME_LINES];
        lines[0] = 1;
        let sections = Sections::choose(&lines, GroupBands::long(fs), 1).unwrap();
        let group = CodedGroup {
            sections,
            scalefactors: vec![200],
            positions: Vec::new(),
        };
        assert_eq!(
            CodedChannel::new(
                Windows::LONG_SINE,
                fs,
                100,
                Tns::NONE,
                Pulses::NONE,
                lines,
                vec![group]
            ),
            None
        );
    }
}
