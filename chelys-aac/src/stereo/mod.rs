// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Joint coding of a channel pair (clause 12): M/S stereo, sent as the
//! pair's `ms_mask_present` and `ms_used` flags, and intensity stereo, sent
//! in the right channel's sections and scalefactors.

pub mod intensity;
pub mod middle_side;

use std::array;
use std::iter::{Enumerate, Take};
use std::ops::Range;

use intensity::Intensity;
use middle_side::MsMask;

use crate::sampling_frequency::SamplingFrequency;
use crate::scalefactor_bands::GroupBands;
use crate::window::{SHORT_WINDOWS, Windows};

/// How a pair codes a band of a window group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BandCoding {
    /// Each channel its own lines, as left and right.
    Plain,
    /// As middle and side (clause 12.1).
    MidSide,
    /// In intensity stereo (clause 12.2), whatever its `ms_used` flag.
    Intensity,
}

/// A scalefactor band of a window group, as [`bands`] gives it.
#[derive(Clone, Debug)]
pub struct Band {
    /// The window group, `g`.
    pub group: usize,
    pub sfb: usize,
    /// The band's lines among the frame's, in the order they are sent.
    pub lines: Range<usize>,
    pub coding: BandCoding,
    /// The group's first line among the frame's.
    start: usize,
    bands: GroupBands,
    /// The group's first window among the frame's.
    first_window: usize,
}

impl Band {
    /// Each of the group's windows, as its index among the frame's windows,
    /// with its lines in the band among the frame's.
    pub fn windows(&self) -> impl Iterator<Item = (usize, Range<usize>)> + '_ {
        (0..self.bands.windows()).map(move |w| {
            let lines = self.bands.window_band(w, self.sfb);
            (
                self.first_window + w,
                self.start + lines.start..self.start + lines.end,
            )
        })
    }
}

/// Each scalefactor band of each window group of `windows`, in the order
/// they are sent, with how a pair codes it: in intensity stereo where
/// `intensity` codes it, else as middle and side where `ms` sets its flag,
/// else plain.
pub fn bands(
    windows: Windows,
    sampling_frequency: SamplingFrequency,
    ms: MsMask,
    intensity: &Intensity,
) -> Bands<'_> {
    Bands {
        groups: windows.groups(sampling_frequency).enumerate(),
        ms,
        intensity,
        group: None,
        start: 0,
        first_window: 0,
        sfb: 0,
    }
}

/// The iterator [`bands`] gives.
pub struct Bands<'a> {
    groups: Enumerate<Take<array::IntoIter<GroupBands, SHORT_WINDOWS>>>,
    ms: MsMask,
    intensity: &'a Intensity,
    /// The group whose bands are being given, `g` and its bands, if any.
    group: Option<(usize, GroupBands)>,
    /// The group's first line among the frame's.
    start: usize,
    /// The group's first window among the frame's.
    first_window: usize,
    /// The group's next band.
    sfb: usize,
}

impl Iterator for Bands<'_> {
    type Item = Band;

    fn next(&mut self) -> Option<Band> {
        loop {
            if let Some((g, bands)) = self.group {
                if self.sfb < bands.num_swb() {
                    let sfb = self.sfb;
                    self.sfb += 1;
                    let band = bands.band(sfb);
                    let coding = if self.intensity.bands[g].codebook(sfb).is_some() {
                        BandCoding::Intensity
                    } else if self.ms.is_used(g, sfb) {
                        BandCoding::MidSide
                    } else {
                        BandCoding::Plain
                    };
                    return Some(Band {
                        group: g,
                        sfb,
                        lines: self.start + band.start..self.start + band.end,
                        coding,
                        start: self.start,
                        bands,
                        first_window: self.first_window,
                    });
                }
                self.start += bands.lines();
                self.first_window += bands.windows();
            }
            self.group = Some(self.groups.next()?);
            self.sfb = 0;
        }
    }
}
