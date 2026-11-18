// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! One channel of one frame as its `individual_channel_stream()` (Table
//! 16) codes it: its windows, `global_gain`, and for each window group its
//! sections, scalefactors and quantized lines.

use crate::noiseless::scalefactors;
use crate::noiseless::sections::Sections;
use crate::sampling_frequency::SamplingFrequency;
use crate::stream::FRAME_SAMPLES;
use crate::window::Windows;

/// The spectral lines of a channel in a frame: one for each sample it
/// codes, in one long window or eight short ones (clause 8.3.3).
pub const FRAME_LINES: usize = FRAME_SAMPLES;

/// The bits of an `individual_channel_stream()` (Table 16) besides its
/// `ics_info()` and noiseless coding: `global_gain`, and the
/// `pulse_data_present`, `tns_data_present` and
/// `gain_control_data_present` flags, which are all 0.
const FIXED_BITS: u32 = 8 + 3;

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
    /// The quantized lines in the order they are sent (clause 8.3.5).
    lines: [i32; FRAME_LINES],
    groups: Vec<CodedGroup>,
    /// The bits of the scalefactors of all groups.
    scalefactor_bits: u32,
}

impl CodedChannel {
    /// A channel whose lines are `lines`, in the window groups of
    /// `windows`, each group with its sections chosen for its lines and its
    /// scalefactors; or `None` if a difference of scalefactors has no
    /// codeword (see [`scalefactors::bits`]).
    ///
    /// # Panics
    ///
    /// If the groups are not those of `windows` at the sections' sampling
    /// frequency, or do not share one `max_sfb`: a caller's error. Lines
    /// other than those the sections were chosen for panic when written.
    pub fn new(
        windows: Windows,
        sampling_frequency: SamplingFrequency,
        global_gain: u8,
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
        let scalefactor_bits =
            scalefactors::bits(global_gain, groups.iter().map(CodedGroup::scalefactor_data))?;
        Some(Self {
            windows,
            global_gain,
            lines,
            groups,
            scalefactor_bits,
        })
    }

    /// A channel whose lines are all zero: `max_sfb` 0, so that its
    /// `section_data()`, `scale_factor_data()` and `spectral_data()` are
    /// empty. "The spectral information for all scalefactor bands at and
    /// above max_sfb, for which there is no section data, is zero" (clause
    /// 9.3).
    pub fn silent(windows: Windows, sampling_frequency: SamplingFrequency) -> Self {
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

    /// Each window group with its quantized lines, in the order they are
    /// sent.
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
            sections + self.scalefactor_bits,
        )
    }
}

/// The bits of an `individual_channel_stream()` of `windows` with
/// `common_window` and the given noiseless coding, as
/// [`CodedChannel::bits`] counts them, for counting a coding without
/// making a channel of it.
pub fn bits(windows: Windows, common_window: bool, noiseless_bits: u32) -> u32 {
    let ics_info = if common_window {
        0
    } else {
        windows.ics_info_bits()
    };
    FIXED_BITS + ics_info + noiseless_bits
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
        let channel = CodedChannel::new(Windows::LONG_SINE, fs, 120, lines, vec![group]).unwrap();
        assert_eq!(channel.max_sfb(), 3);
        assert_eq!(
            channel.bits(false),
            bits(Windows::LONG_SINE, false, noiseless + sf_bits)
        );
        assert_eq!(channel.bits(false), 8 + 11 + 3 + noiseless + sf_bits);
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
            CodedChannel::new(Windows::LONG_SINE, fs, 100, lines, vec![group]),
            None
        );
    }
}
