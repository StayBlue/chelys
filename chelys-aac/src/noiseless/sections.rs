// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Sections (clause 9.1): runs of scalefactor bands whose lines one
//! codebook codes, sent in `section_data()` (Table 17), and their choice by
//! the greedy merge of clause C.8.3.

use chelys_core::bitstream::BitWriter;

use super::codebooks::{CLASSES, Codebook, class};
use super::scalefactors;
use super::spectrum::{self, UNCODABLE};
use crate::scalefactor_bands::{GroupBands, MAX_BANDS, Transform};

/// A section: the scalefactor bands `start..end` of its window group, and
/// the codebook that codes their lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Section {
    pub codebook: Codebook,
    pub start: usize,
    pub end: usize,
}

/// The sections of one window group, of its bands 0 to `max_sfb`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sections {
    bands: GroupBands,
    section_count: usize,
    codebooks: [Codebook; MAX_BANDS],
    /// The band each section ends before.
    ends: [u8; MAX_BANDS],
    /// Bit `sfb` is set where band `sfb` has no line but zero.
    zero_bands: u64,
    /// The bits of the group's `spectral_data()`.
    spectral_bits: u32,
}

/// The bands of a window group of the right channel of a pair that are
/// coded in intensity stereo (clause 12.2), each with its intensity
/// codebook: their lines are not sent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IntensityBands {
    in_phase: u64,
    out_of_phase: u64,
}

impl IntensityBands {
    /// No band coded in intensity stereo.
    pub const NONE: Self = Self {
        in_phase: 0,
        out_of_phase: 0,
    };

    /// Codes band `sfb` in intensity stereo, in phase or out of phase, or,
    /// with `None`, not.
    ///
    /// # Panics
    ///
    /// If `codebook` is not an intensity codebook: a caller's error.
    pub fn set(&mut self, sfb: usize, codebook: Option<Codebook>) {
        let bit = 1 << sfb;
        self.in_phase &= !bit;
        self.out_of_phase &= !bit;
        match codebook {
            Some(Codebook::INTENSITY) => self.in_phase |= bit,
            Some(Codebook::INTENSITY_OUT_OF_PHASE) => self.out_of_phase |= bit,
            None => {}
            Some(other) => panic!("codebook {other:?} is not an intensity codebook"),
        }
    }

    /// The intensity codebook of band `sfb`, if it is coded in intensity
    /// stereo.
    pub fn codebook(self, sfb: usize) -> Option<Codebook> {
        if (self.in_phase >> sfb) & 1 == 1 {
            Some(Codebook::INTENSITY)
        } else if (self.out_of_phase >> sfb) & 1 == 1 {
            Some(Codebook::INTENSITY_OUT_OF_PHASE)
        } else {
            None
        }
    }

    /// One past the highest band coded in intensity stereo, or 0.
    pub fn end(self) -> usize {
        64 - (self.in_phase | self.out_of_phase).leading_zeros() as usize
    }

    /// Those of the bands below `max_sfb`.
    pub fn below(self, max_sfb: usize) -> Self {
        let bands = if max_sfb >= 64 {
            u64::MAX
        } else {
            (1 << max_sfb) - 1
        };
        Self {
            in_phase: self.in_phase & bands,
            out_of_phase: self.out_of_phase & bands,
        }
    }
}

impl Sections {
    /// The sections that [`Counted::choose`] chooses for the lines that
    /// [`count`](Self::count) counts.
    pub fn choose(lines: &[i32], bands: GroupBands, max_sfb: usize) -> Option<Self> {
        Self::count(lines, bands, max_sfb).map(|counted| counted.choose())
    }

    /// [`count`](Self::count), but for the bands `intensity` codes in
    /// intensity stereo, whose lines are to be zero.
    pub fn count_with_intensity(
        lines: &[i32],
        bands: GroupBands,
        max_sfb: usize,
        intensity: IntensityBands,
    ) -> Option<Counted> {
        Self::count_bands(lines, bands, max_sfb, intensity)
    }

    /// Counts the bits of a window group's bands `0..max_sfb` in each
    /// codebook a section of them could take, for [`Counted::choose`] to
    /// choose sections from.
    ///
    /// `lines` are the group's quantized lines in the order they are sent
    /// (clause 8.3.5), [`GroupBands::lines`] of them, and `max_sfb` the one
    /// of the `ics_info()`: for the groups of an EIGHT_SHORT_SEQUENCE, the
    /// same for all of them (see [`max_sfb`]). The lines of the bands from
    /// `max_sfb` on are not sent, and so are to be zero.
    ///
    /// Returns `None` if a line's magnitude exceeds
    /// [`MAX_MAGNITUDE`](super::codebooks::MAX_MAGNITUDE), which no codebook
    /// codes.
    ///
    /// # Panics
    ///
    /// If there are not as many lines as the bands hold, or `max_sfb`
    /// exceeds their `num_swb`: a caller's error.
    pub fn count(lines: &[i32], bands: GroupBands, max_sfb: usize) -> Option<Counted> {
        Self::count_bands(lines, bands, max_sfb, IntensityBands::NONE)
    }

    fn count_bands(
        lines: &[i32],
        bands: GroupBands,
        max_sfb: usize,
        intensity: IntensityBands,
    ) -> Option<Counted> {
        let mut counted = Counted::new(bands);
        counted
            .count(lines, bands, max_sfb, intensity)
            .then_some(counted)
    }

    pub fn bands(&self) -> GroupBands {
        self.bands
    }

    /// The bands the sections span: `max_sfb`.
    pub fn max_sfb(&self) -> usize {
        self.section_count
            .checked_sub(1)
            .map_or(0, |last| usize::from(self.ends[last]))
    }

    pub fn iter(&self) -> impl Iterator<Item = Section> + '_ {
        (0..self.section_count).map(|i| Section {
            codebook: self.codebooks[i],
            start: i
                .checked_sub(1)
                .map_or(0, |before| usize::from(self.ends[before])),
            end: usize::from(self.ends[i]),
        })
    }

    /// Whether band `sfb` has no line but zero.
    pub fn is_zero_band(&self, sfb: usize) -> bool {
        sfb < self.max_sfb() && (self.zero_bands >> sfb) & 1 == 1
    }

    /// The bands in sections whose codebook sends a scalefactor (Table 18),
    /// in order.
    pub fn bands_with_scalefactors(&self) -> impl Iterator<Item = usize> + '_ {
        (self.iter())
            .filter(|section| section.codebook.sends_scalefactor())
            .flat_map(|section| section.start..section.end)
    }

    /// The bands that send a scalefactor or an intensity stereo position
    /// (Table 18), in order, each with whether it sends a position.
    pub fn bands_with_scalefactors_or_positions(&self) -> impl Iterator<Item = (usize, bool)> + '_ {
        (self.iter())
            .filter(|section| section.codebook != Codebook::ZERO)
            .flat_map(|section| {
                let position = section.codebook.is_intensity();
                (section.start..section.end).map(move |sfb| (sfb, position))
            })
    }

    /// The bits that [`extend`](Self::extend) adds to the sections'
    /// `section_data()` to make them span `max_sfb` bands.
    ///
    /// # Panics
    ///
    /// If `max_sfb` is below the bands they span or beyond the group's:
    /// a caller's error.
    pub fn extension_bits(&self, max_sfb: usize) -> u32 {
        let own = self.max_sfb();
        assert!(
            (own..=self.bands.num_swb()).contains(&max_sfb),
            "sections of {own} bands extended to {max_sfb}"
        );
        let transform = self.bands.transform();
        match self.zero_tail() {
            _ if max_sfb == own => 0,
            Some(start) => {
                section_bits(transform, max_sfb - start) - section_bits(transform, own - start)
            }
            None => section_bits(transform, max_sfb - own),
        }
    }

    /// Makes the sections span `max_sfb` bands, the pair's `max_sfb` where
    /// the pair shares its window (clause 8.3.1): the bands added, which
    /// have no lines, join a last section of `ZERO_HCB`, or make one.
    ///
    /// # Panics
    ///
    /// As [`extension_bits`](Self::extension_bits).
    pub fn extend(&mut self, max_sfb: usize) {
        self.extension_bits(max_sfb);
        let own = self.max_sfb();
        if max_sfb == own {
            return;
        }
        if self.zero_tail().is_none() {
            self.codebooks[self.section_count] = Codebook::ZERO;
            self.section_count += 1;
        }
        self.ends[self.section_count - 1] = max_sfb as u8;
        for sfb in own..max_sfb {
            self.zero_bands |= 1 << sfb;
        }
    }

    /// The first band of the last section, if its codebook is `ZERO_HCB`.
    fn zero_tail(&self) -> Option<usize> {
        let last = self.iter().last()?;
        (last.codebook == Codebook::ZERO).then_some(last.start)
    }

    /// The bits [`write_section_data`](Self::write_section_data) writes.
    pub fn section_data_bits(&self) -> u32 {
        (self.iter())
            .map(|section| section_bits(self.bands.transform(), section.end - section.start))
            .sum()
    }

    /// The bits [`write_spectral_data`](Self::write_spectral_data) writes.
    pub fn spectral_data_bits(&self) -> u32 {
        self.spectral_bits
    }

    /// The bits that the choice of sections changes: those of
    /// `section_data()` and `spectral_data()`, and the scalefactors that
    /// the bands of zeros in sections with a spectrum codebook send (see
    /// [`Counted::choose`]).
    pub fn coding_bits(&self) -> u32 {
        let repeated = (self.bands_with_scalefactors())
            .filter(|&sfb| self.is_zero_band(sfb))
            .count() as u32;
        self.section_data_bits()
            + self.spectral_data_bits()
            + repeated * scalefactors::REPEATED_BITS
    }

    /// Writes the group's part of `section_data()` (Table 17): for each
    /// section its `sect_cb`, then its length in bands as `sect_len_incr`
    /// fields, each `sect_esc_val` but the last, which is less.
    pub fn write_section_data(&self, w: &mut BitWriter) {
        let (bits, esc) = sect_len_incr(self.bands.transform());
        for section in self.iter() {
            w.write(u32::from(section.codebook.number()), 4);
            let mut len = section.end - section.start;
            while len >= esc {
                w.write(esc as u32, bits);
                len -= esc;
            }
            w.write(len as u32, bits);
        }
    }

    /// Writes the group's part of `spectral_data()` (Table 20): the lines
    /// of each section with a spectrum codebook.
    ///
    /// # Panics
    ///
    /// If `lines` are not those the sections were chosen for, as far as
    /// that shows: in their number, a line that a section's codebook cannot
    /// code, one that is not zero where none is sent, or bits other than
    /// those counted. A caller's error, which would otherwise write lines
    /// other than those given or in other bits than counted.
    pub fn write_spectral_data(&self, lines: &[i32], w: &mut BitWriter) {
        let stale = "lines other than those the sections were chosen for";
        assert_eq!(lines.len(), self.bands.lines(), "{stale}");
        let start_bits = w.bit_len();
        for section in self.iter() {
            let lines = &lines[self.bands.offset(section.start)..self.bands.offset(section.end)];
            match section.codebook.spectrum() {
                Some(book) => spectrum::write(book, lines, w),
                None => assert!(lines.iter().all(|&line| line == 0), "{stale}"),
            }
        }
        let unsent = &lines[self.bands.offset(self.max_sfb())..];
        assert!(unsent.iter().all(|&line| line == 0), "{stale}");
        assert_eq!(
            w.bit_len() - start_bits,
            self.spectral_bits as usize,
            "{stale}"
        );
    }
}

/// `max_sfb` for a window group whose bands' lines have the largest
/// magnitudes `largest` (as for [`Counted::count_with`]): the bands
/// up to the highest with a line that is not zero, so that no section has
/// to code the bands of zeros above it. Over the groups of an
/// EIGHT_SHORT_SEQUENCE, `max_sfb` is the largest of theirs.
///
/// Spec reading: `max_sfb` is "one greater than the highest active
/// scalefactor band in all groups" (clause 8.3.4), where a band is active
/// unless its codebook is `ZERO_HCB` (clause 8.3.2.5). A band of zeros
/// above the highest with a line that is not zero would cost a section or
/// a longer one, where outside `max_sfb` it costs nothing, so it is left
/// out.
pub fn max_sfb_of_largest(largest: &[u32]) -> usize {
    largest
        .iter()
        .rposition(|&q| q != 0)
        .map_or(0, |sfb| sfb + 1)
}

/// [`max_sfb_of_largest`] for a window group's `lines` (as for
/// [`Sections::count`]).
///
/// # Panics
///
/// If there are not as many lines as the bands hold: a caller's error.
#[cfg(any(test, feature = "unstable"))]
pub fn max_sfb(lines: &[i32], bands: GroupBands) -> usize {
    assert_eq!(lines.len(), bands.lines(), "a group's lines fill its bands");
    lines
        .iter()
        .rposition(|&line| line != 0)
        .map_or(0, |highest| {
            (1..=bands.num_swb())
                .find(|&sfb| bands.offset(sfb) > highest)
                .expect("the bands end past every line")
        })
}

/// A window group's lines counted in each codebook a section could take
/// (see [`Sections::count`]).
#[derive(Clone, Debug)]
pub struct Counted {
    bands: GroupBands,
    max_sfb: usize,
    /// The highest class of a band, the highest whose codebooks are
    /// counted.
    top: usize,
    /// `bits_below[b][cb]`: the bits of bands `0..b` in codebook `cb`, or
    /// [`UNCODABLE`] where it is not counted.
    bits_below: [[u32; 12]; MAX_BANDS + 1],
    /// The bands of zeros in bands `0..b`.
    zero_bands_below: [u8; MAX_BANDS + 1],
    /// The class of each band: the first of [`CLASSES`] that codes it.
    classes: [usize; MAX_BANDS],
    least_bits: u32,
}

impl Counted {
    /// A window group of `bands` with no band counted, to
    /// [`count`](Self::count) into.
    pub fn new(bands: GroupBands) -> Self {
        Self {
            bands,
            max_sfb: 0,
            top: 0,
            bits_below: [[0; 12]; MAX_BANDS + 1],
            zero_bands_below: [0; MAX_BANDS + 1],
            classes: [0; MAX_BANDS],
            least_bits: 0,
        }
    }

    /// Counts a window group's bands as [`Sections::count_with_intensity`]
    /// does, in place of what was counted before, whose memory it reuses:
    /// whether each line is one that a codebook codes. Where one is not,
    /// what is left is to be counted again before it is used.
    ///
    /// # Panics
    ///
    /// As [`Sections::count`].
    pub fn count(
        &mut self,
        lines: &[i32],
        bands: GroupBands,
        max_sfb: usize,
        intensity: IntensityBands,
    ) -> bool {
        assert_eq!(lines.len(), bands.lines(), "a group's lines fill its bands");
        assert!(
            max_sfb <= bands.num_swb(),
            "max_sfb {max_sfb} exceeds num_swb {}",
            bands.num_swb()
        );
        let mut largest = [0; MAX_BANDS];
        for (sfb, largest) in largest[..max_sfb].iter_mut().enumerate() {
            let lines = &lines[bands.offset(sfb)..bands.offset(sfb + 1)];
            *largest = lines
                .iter()
                .map(|line| line.unsigned_abs())
                .max()
                .unwrap_or(0);
        }
        let band_bits =
            |_, class, top, lines: &[i32]| spectrum::bits_by_codebook(lines, class, top);
        self.count_with(
            lines,
            bands,
            max_sfb,
            intensity,
            &largest,
            u32::MAX,
            band_bits,
        )
    }

    /// [`count`](Self::count), given the largest magnitude of each band's
    /// lines in `largest`, and each band's bits in each codebook, and the
    /// least of them, by `band_bits(sfb, class, top, lines)`, as
    /// `spectrum::bits_by_codebook` gives them; but giving `false` too
    /// once the [`least_bits`](Self::least_bits) of the bands counted
    /// exceed `budget`, without counting the rest.
    #[allow(clippy::too_many_arguments)]
    pub fn count_with(
        &mut self,
        lines: &[i32],
        bands: GroupBands,
        max_sfb: usize,
        intensity: IntensityBands,
        largest: &[u32],
        budget: u32,
        mut band_bits: impl FnMut(usize, usize, usize, &[i32]) -> ([u32; 12], u32),
    ) -> bool {
        assert_eq!(lines.len(), bands.lines(), "a group's lines fill its bands");
        assert!(
            max_sfb <= bands.num_swb(),
            "max_sfb {max_sfb} exceeds num_swb {}",
            bands.num_swb()
        );
        debug_assert!(
            lines[bands.offset(max_sfb)..].iter().all(|&line| line == 0),
            "lines above max_sfb are not sent"
        );
        assert!(intensity.end() <= max_sfb, "intensity bands above max_sfb");
        (self.bands, self.max_sfb, self.least_bits) = (bands, max_sfb, 0);
        for (sfb, class) in self.classes[..max_sfb].iter_mut().enumerate() {
            let largest = largest[sfb];
            debug_assert_eq!(
                Some(largest),
                (lines[bands.offset(sfb)..bands.offset(sfb + 1)].iter())
                    .map(|line| line.unsigned_abs())
                    .max(),
                "the largest magnitude of band {sfb}"
            );
            *class = match intensity.codebook(sfb) {
                Some(codebook) => {
                    debug_assert!(largest == 0, "lines in an intensity band");
                    intensity_class(codebook)
                }
                None => match self::class(largest) {
                    Some(class) => class,
                    None => return false,
                },
            };
        }
        // A merged section takes the higher class of the two, so no section
        // takes a class above the group's highest.
        self.top = (self.classes[..max_sfb].iter().copied())
            .filter(|&class| class < CLASSES.len())
            .max()
            .unwrap_or(0);
        if max_sfb > 0 {
            // At least one section, of all the bands.
            self.least_bits = section_bits(bands.transform(), max_sfb);
        }
        // The first row of `bits_below` and of `zero_bands_below`, of no
        // bands, is never written, and stays 0.
        for sfb in 0..max_sfb {
            let class = self.classes[sfb];
            let lines = &lines[bands.offset(sfb)..bands.offset(sfb + 1)];
            let bits = if class < CLASSES.len() {
                let (bits, least) = band_bits(sfb, class, self.top, lines);
                self.least_bits += least;
                if self.least_bits > budget {
                    return false;
                }
                bits
            } else {
                // No spectrum codebook codes an intensity band.
                [UNCODABLE; 12]
            };
            let [below, row] = &mut self.bits_below[sfb..sfb + 2] else {
                unreachable!("two rows")
            };
            for ((row, below), bits) in row.iter_mut().zip(&*below).zip(bits) {
                *row = below + bits;
            }
            self.zero_bands_below[sfb + 1] = self.zero_bands_below[sfb] + u8::from(class == 0);
        }
        true
    }

    /// The fewest bits that the `section_data()` and `spectral_data()` of
    /// any sections of the group can take: each band in the codebook that
    /// codes it in the fewest bits, in one section. The sections that
    /// [`choose`](Self::choose) chooses take at least this many, so that
    /// where it is already too many, they need not be chosen.
    pub fn least_bits(&self) -> u32 {
        self.least_bits
    }

    /// Counts band `sfb` again for its lines `lines`, which hold the same
    /// lines of zero as those counted, and no larger magnitudes: those of
    /// a long window that [spectrum clipping](super::clipping) clips. The
    /// band is counted in the codebooks up to those of the highest class of
    /// the lines counted before, as any section that merges it can take.
    ///
    /// # Panics
    ///
    /// If the band is not one counted, or its lines are not as many as it
    /// holds, or have other zeros or larger magnitudes than those counted:
    /// a caller's error.
    pub fn recount_band(&mut self, sfb: usize, lines: &[i32]) {
        assert!(sfb < self.max_sfb, "band {sfb} is not counted");
        assert_eq!(lines.len(), self.bands.band(sfb).len(), "the band's lines");
        let largest = lines.iter().map(|line| line.unsigned_abs()).max();
        let class = class(largest.expect("a band holds lines")).expect("a codable line");
        assert!(
            (class == 0) == (self.classes[sfb] == 0) && class <= self.classes[sfb],
            "lines of the band's zeros and magnitudes"
        );
        if class == 0 {
            return;
        }
        let (new, _) = spectrum::bits_by_codebook(lines, class, self.top);
        let old: [u32; 12] =
            std::array::from_fn(|cb| self.bits_below[sfb + 1][cb] - self.bits_below[sfb][cb]);
        for below in &mut self.bits_below[sfb + 1..=self.max_sfb] {
            for (bits, (new, old)) in below.iter_mut().zip(new.iter().zip(&old)) {
                *bits = *bits - old + new;
            }
        }
        let least = |bits: &[u32; 12]| *bits.iter().min().expect("a codebook");
        self.least_bits = self.least_bits - least(&old) + least(&new);
        self.classes[sfb] = class;
    }

    /// Divides the bands into sections and chooses their codebooks, by the
    /// greedy merge of clause C.8.3: one section for each band first, each
    /// with the codebook of the smallest index that codes it, then, while
    /// merging two neighbouring sections saves bits, the merge that saves
    /// the most, with the higher of their codebooks.
    ///
    /// Spec reading: clause C.8.3 starts each section with "the Huffman
    /// codebook with the smallest possible index", and a merged section
    /// takes "the codebook with the higher index". Table 59 pairs the
    /// codebooks by largest magnitude, 1 and 2, 3 and 4 and so on to 9 and
    /// 10, the two of a pair "representing a distinct probability
    /// distribution function", of which "The best fit is always chosen"
    /// (clause C.8.6). So the index is read as that of the pair: a section
    /// takes the pair of the smallest largest magnitude that codes its
    /// lines, or a merged section the higher pair of the two, and of the
    /// pair the codebook that codes the section in fewer bits, the lower on
    /// a tie, whichever the two sections had.
    ///
    /// Spec reading: the "total bit count" that merging lowers is that of
    /// the sections' `section_data()`, `scale_factor_data()` and
    /// `spectral_data()`. Merging changes the scalefactors only where it
    /// puts a band with no line but zero in a section with a spectrum
    /// codebook, which sends its scalefactor; that costs the bits of a
    /// difference of 0 (see [`scalefactors::bits`]).
    ///
    /// Spec gap: clause C.8.3 does not say which merge comes first where
    /// two save the same bits; the lower in frequency is taken.
    pub fn choose(&self) -> Sections {
        let max_sfb = self.max_sfb;
        // One section for each band.
        let mut sections = [Candidate::default(); MAX_BANDS];
        for (sfb, section) in sections[..max_sfb].iter_mut().enumerate() {
            *section = self.section(sfb, sfb + 1, self.classes[sfb]);
        }
        // merged[i]: sections i and i + 1 merged, and saved[i] the bits
        // that merging them saves.
        let mut merged = [Candidate::default(); MAX_BANDS];
        let mut saved = [0; MAX_BANDS];
        let saving = |a: Candidate, b: Candidate, merged: Candidate| {
            (a.bits + b.bits).saturating_sub(merged.bits)
        };
        for i in 0..max_sfb.saturating_sub(1) {
            merged[i] = self.merge(sections[i], sections[i + 1]);
            saved[i] = saving(sections[i], sections[i + 1], merged[i]);
        }
        let mut section_count = max_sfb;
        loop {
            // The first merge of those that save the most, if any saves.
            let pairs = &saved[..section_count.saturating_sub(1)];
            let most = pairs.iter().copied().max().unwrap_or(0);
            if most == 0 {
                break;
            }
            let i = pairs
                .iter()
                .position(|&saved| saved == most)
                .expect("the most");
            sections[i] = merged[i];
            sections.copy_within(i + 2..section_count, i + 1);
            merged.copy_within(i + 1..section_count - 1, i);
            saved.copy_within(i + 1..section_count - 1, i);
            section_count -= 1;
            if i > 0 {
                merged[i - 1] = self.merge(sections[i - 1], sections[i]);
                saved[i - 1] = saving(sections[i - 1], sections[i], merged[i - 1]);
            }
            if i + 1 < section_count {
                merged[i] = self.merge(sections[i], sections[i + 1]);
                saved[i] = saving(sections[i], sections[i + 1], merged[i]);
            }
        }

        let mut chosen = Sections {
            bands: self.bands,
            section_count,
            codebooks: [Codebook::ZERO; MAX_BANDS],
            ends: [0; MAX_BANDS],
            zero_bands: 0,
            spectral_bits: 0,
        };
        for (i, section) in sections[..section_count].iter().enumerate() {
            chosen.codebooks[i] = section.codebook;
            chosen.ends[i] = section.end;
            if !section.codebook.is_intensity() {
                let (start, end) = (usize::from(section.start), usize::from(section.end));
                chosen.spectral_bits += self.spectral_bits(start, end, section.codebook);
            }
        }
        for sfb in 0..max_sfb {
            if self.classes[sfb] == 0 {
                chosen.zero_bands |= 1 << sfb;
            }
        }
        chosen
    }

    /// The bits of the lines of bands `start..end` in `codebook`.
    fn spectral_bits(&self, start: usize, end: usize, codebook: Codebook) -> u32 {
        let codebook = usize::from(codebook.number());
        let bits = self.bits_below[end][codebook] - self.bits_below[start][codebook];
        debug_assert!(
            bits < UNCODABLE,
            "codebook {codebook} is not counted for bands {start}..{end}"
        );
        bits
    }

    /// The section of bands `start..end` with the codebook of `class` that
    /// costs the fewest bits, the lower on a tie: those of its
    /// `section_data()` and `spectral_data()`, and of the scalefactors of
    /// its bands of zeros if the codebook sends them; for an intensity class,
    /// with its intensity codebook, which sends no lines (the positions its
    /// bands send are counted with the scalefactors).
    fn section(&self, start: usize, end: usize, class: usize) -> Candidate {
        let header = section_bits(self.bands.transform(), end - start);
        let candidate = |codebook, bits| Candidate {
            start: start as u8,
            end: end as u8,
            class: class as u8,
            codebook,
            bits,
        };
        if let Some(codebook) = class.checked_sub(CLASSES.len()).map(intensity_codebook) {
            return candidate(codebook, header);
        }
        // The codebooks of a class differ only in their spectral bits.
        let shared = header + self.repeated_bits(start, end, CLASSES[class][0]);
        match *CLASSES[class] {
            [a] => candidate(a, shared + self.spectral_bits(start, end, a)),
            [a, b] => {
                let (a_bits, b_bits) = (
                    self.spectral_bits(start, end, a),
                    self.spectral_bits(start, end, b),
                );
                if b_bits < a_bits {
                    candidate(b, shared + b_bits)
                } else {
                    candidate(a, shared + a_bits)
                }
            }
            _ => unreachable!("a class of one codebook or two"),
        }
    }

    /// Sections `a` and `b`, neighbours, merged into one. Bands in
    /// intensity stereo merge only with bands of the same intensity
    /// codebook: any other merge costs more than any coding.
    fn merge(&self, a: Candidate, b: Candidate) -> Candidate {
        debug_assert_eq!(a.end, b.start);
        let intensity = |class: u8| usize::from(class) >= CLASSES.len();
        if (intensity(a.class) || intensity(b.class)) && a.class != b.class {
            return Candidate {
                start: a.start,
                end: b.end,
                class: a.class.max(b.class),
                codebook: a.codebook,
                bits: UNCODABLE,
            };
        }
        let class = usize::from(a.class.max(b.class));
        self.section(usize::from(a.start), usize::from(b.end), class)
    }

    /// The bits of the scalefactors of the bands of zeros among bands
    /// `start..end`, in a section with `codebook`, if it sends them.
    fn repeated_bits(&self, start: usize, end: usize, codebook: Codebook) -> u32 {
        if codebook.sends_scalefactor() {
            let zero_bands = self.zero_bands_below[end] - self.zero_bands_below[start];
            u32::from(zero_bands) * scalefactors::REPEATED_BITS
        } else {
            0
        }
    }
}

/// A section as the merge sees it: its bands, the class of its codebook
/// and the codebook, and its bits in all.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    start: u8,
    end: u8,
    class: u8,
    codebook: Codebook,
    bits: u32,
}

impl Default for Candidate {
    fn default() -> Self {
        Self {
            start: 0,
            end: 0,
            class: 0,
            codebook: Codebook::ZERO,
            bits: 0,
        }
    }
}

/// The class [`Counted`] gives a band of `codebook`, an intensity codebook:
/// one past [`CLASSES`] for each.
fn intensity_class(codebook: Codebook) -> usize {
    CLASSES.len() + usize::from(codebook == Codebook::INTENSITY_OUT_OF_PHASE)
}

/// The intensity codebook of the class `intensity_class` gives, less
/// [`CLASSES`]`.len()`.
fn intensity_codebook(class: usize) -> Codebook {
    if class == 0 {
        Codebook::INTENSITY
    } else {
        Codebook::INTENSITY_OUT_OF_PHASE
    }
}

/// `sect_len_incr`'s bits and `sect_esc_val` (Table 17).
const fn sect_len_incr(transform: Transform) -> (u32, usize) {
    match transform {
        Transform::Long => (5, (1 << 5) - 1),
        Transform::Short => (3, (1 << 3) - 1),
    }
}

/// The bits of a section of `len` bands in `section_data()`: `sect_cb`,
/// and a `sect_len_incr` for each `sect_esc_val` bands and one more; from
/// a table of every length.
fn section_bits(transform: Transform, len: usize) -> u32 {
    /// The bits of each length of section, for `sect_len_incr`'s `bits` and
    /// `sect_esc_val` `esc`.
    const fn table(bits: u32, esc: usize) -> [u32; MAX_BANDS + 1] {
        let mut table = [0; MAX_BANDS + 1];
        let mut len = 0;
        while len <= MAX_BANDS {
            table[len] = 4 + bits * (len / esc + 1) as u32;
            len += 1;
        }
        table
    }
    static LONG: [u32; MAX_BANDS + 1] = {
        let (bits, esc) = sect_len_incr(Transform::Long);
        table(bits, esc)
    };
    static SHORT: [u32; MAX_BANDS + 1] = {
        let (bits, esc) = sect_len_incr(Transform::Short);
        table(bits, esc)
    };
    match transform {
        Transform::Long => LONG[len],
        Transform::Short => SHORT[len],
    }
}

#[cfg(test)]
mod tests {
    use super::super::codebooks::MAX_MAGNITUDE;
    use super::super::decoder::{BitReader, Noise, section_data, spectral_data};
    use super::*;
    use crate::sampling_frequency::SamplingFrequency;

    fn offsets(bands: GroupBands) -> Vec<u16> {
        (0..=bands.num_swb())
            .map(|sfb| bands.offset(sfb) as u16)
            .collect()
    }

    /// Lines for `bands` with no line from band `max_sfb` up.
    fn spectrum(noise: &mut Noise, bands: GroupBands, max_sfb: usize) -> Vec<i32> {
        let mut lines = noise.spectrum(&offsets(bands), bands.lines());
        lines[bands.offset(max_sfb)..].fill(0);
        lines
    }

    /// Chooses sections for `lines`, writes them and their lines, and
    /// checks that they decode as written, in the bits counted.
    fn round_trip(lines: &[i32], bands: GroupBands, max_sfb: usize) -> Sections {
        let counted = Sections::count(lines, bands, max_sfb).expect("codable lines");
        let least = counted.least_bits();
        let sections = counted.choose();
        assert_eq!(sections.max_sfb(), max_sfb);
        let mut w = BitWriter::new();
        sections.write_section_data(&mut w);
        assert_eq!(w.bit_len() as u32, sections.section_data_bits());
        sections.write_spectral_data(lines, &mut w);
        let bits = sections.section_data_bits() + sections.spectral_data_bits();
        assert_eq!(w.bit_len() as u32, bits);
        assert!(least <= bits);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        let decoded = section_data(&mut r, bands.transform(), max_sfb);
        let written: Vec<_> = (sections.iter())
            .map(|s| (s.codebook.number(), s.start, s.end))
            .collect();
        assert_eq!(decoded, written);
        assert_eq!(
            spectral_data(&mut r, &decoded, &offsets(bands), lines.len()),
            lines
        );
        sections
    }

    #[test]
    fn spectra_decode_as_written_at_every_sampling_frequency() {
        let mut noise = Noise::new(3);
        for fs in SamplingFrequency::ALL {
            let long = GroupBands::long(fs);
            for _ in 0..40 {
                let max_sfb = noise.below(long.num_swb() as u32 + 1) as usize;
                round_trip(&spectrum(&mut noise, long, max_sfb), long, max_sfb);
            }
            for windows in 1..=8 {
                let group = GroupBands::short(fs, windows);
                let max_sfb = noise.below(group.num_swb() as u32 + 1) as usize;
                round_trip(&spectrum(&mut noise, group, max_sfb), group, max_sfb);
            }
        }
    }

    #[test]
    fn long_sections_escape_their_length() {
        let bands = GroupBands::long(SamplingFrequency::Hz32000);
        // Ones in the first and last bands, zeros between: a section of
        // zeros of 49 bands, sent as 31 and 18.
        let mut lines = vec![0; 1024];
        lines[0] = 1;
        lines[1023] = -1;
        let sections = round_trip(&lines, bands, 51);
        let zeros: Vec<_> = (sections.iter())
            .filter(|section| section.codebook == Codebook::ZERO)
            .map(|section| section.end - section.start)
            .collect();
        assert_eq!(zeros, [49]);
        assert_eq!(sections.section_data_bits(), 3 * 4 + 5 + 2 * 5 + 5);
        // A section of 31 bands exactly ends with a zero increment.
        let mut lines = vec![0; 1024];
        lines[bands.offset(31)] = 1;
        let sections = round_trip(&lines, bands, 32);
        assert_eq!(sections.iter().next().map(|s| s.end), Some(31));
        assert_eq!(sections.section_data_bits(), 4 + 2 * 5 + 4 + 5);
    }

    #[test]
    fn short_sections_escape_their_length() {
        let bands = GroupBands::short(SamplingFrequency::Hz8000, 1);
        // Zeros below the last of 15 bands: a section of 14, sent as 7, 7
        // and 0.
        let mut lines = vec![0; 128];
        lines[127] = 3;
        let sections = round_trip(&lines, bands, 15);
        assert_eq!(sections.iter().next().map(|s| s.end), Some(14));
        assert_eq!(sections.section_data_bits(), 4 + 3 * 3 + 4 + 3);
    }

    #[test]
    fn zeros_up_to_max_sfb_are_one_section_of_zero_hcb() {
        let bands = GroupBands::long(SamplingFrequency::Hz48000);
        let sections = round_trip(&[0; 1024], bands, 49);
        let only = Section {
            codebook: Codebook::ZERO,
            start: 0,
            end: 49,
        };
        assert_eq!(sections.iter().collect::<Vec<_>>(), [only]);
        assert_eq!(sections.spectral_data_bits(), 0);
        assert!((0..49).all(|sfb| sections.is_zero_band(sfb)));
        assert_eq!(sections.bands_with_scalefactors().count(), 0);
    }

    #[test]
    fn no_bands_make_no_sections() {
        let sections = round_trip(&[0; 1024], GroupBands::long(SamplingFrequency::Hz48000), 0);
        assert_eq!(sections.iter().count(), 0);
        assert_eq!(sections.section_data_bits(), 0);
    }

    /// C4: no codebook codes a line above MAX_MAGNITUDE, which is not
    /// counted but refused.
    #[test]
    fn lines_beyond_the_largest_magnitude_are_refused() {
        let bands = GroupBands::long(SamplingFrequency::Hz48000);
        let mut lines = vec![0; 1024];
        lines[500] = -(MAX_MAGNITUDE as i32);
        assert!(Sections::choose(&lines, bands, 49).is_some());
        for line in [MAX_MAGNITUDE as i32 + 1, i32::MIN] {
            lines[500] = line;
            assert!(Sections::count(&lines, bands, 49).is_none());
        }
        // The first quadruple of [8192, 0, 0, 0].
        let mut lines = vec![0; 1024];
        lines[0] = 8192;
        assert!(Sections::count(&lines, bands, 1).is_none());
    }

    /// C2: the bands are those of the sampling frequency, which at 96 kHz
    /// are 41, not the 51 of 32 kHz.
    #[test]
    #[should_panic(expected = "exceeds num_swb")]
    fn max_sfb_is_bounded_by_the_bands_of_the_sampling_frequency() {
        Sections::count(&[0; 1024], GroupBands::long(SamplingFrequency::Hz96000), 42);
    }

    /// C2: every group of an EIGHT_SHORT_SEQUENCE is sectioned up to the
    /// one max_sfb, a group of zeros with a section of ZERO_HCB.
    #[test]
    fn short_groups_are_sectioned_up_to_the_shared_max_sfb() {
        let fs = SamplingFrequency::Hz44100;
        let (first, second) = (GroupBands::short(fs, 1), GroupBands::short(fs, 7));
        let silent = vec![0; first.lines()];
        let mut loud = vec![0; second.lines()];
        loud[0] = 1;
        let max_sfb = max_sfb(&silent, first).max(max_sfb(&loud, second));
        assert_eq!(max_sfb, 1);
        let sections = round_trip(&silent, first, max_sfb);
        let zero = Section {
            codebook: Codebook::ZERO,
            start: 0,
            end: 1,
        };
        assert_eq!(sections.iter().collect::<Vec<_>>(), [zero]);
        assert_eq!(sections.section_data_bits(), 4 + 3);
        round_trip(&loud, second, max_sfb);
    }

    #[test]
    #[should_panic(expected = "fill its bands")]
    fn a_group_has_the_lines_of_its_bands() {
        Sections::count(
            &[0; 1024],
            GroupBands::short(SamplingFrequency::Hz48000, 7),
            1,
        );
    }

    /// C3: sections write no lines but those they were chosen for.
    #[test]
    fn sections_refuse_lines_they_were_not_chosen_for() {
        let bands = GroupBands::long(SamplingFrequency::Hz48000);
        let mut lines = vec![0; 1024];
        lines[..8].copy_from_slice(&[1, 0, -1, 0, 0, 0, 0, 1]);
        let sections = Sections::choose(&lines, bands, 2).unwrap();
        let refused = |other: &[i32]| {
            let other = other.to_vec();
            let sections = sections.clone();
            std::panic::catch_unwind(move || {
                sections.write_spectral_data(&other, &mut BitWriter::new())
            })
            .is_err()
        };
        assert!(!refused(&lines));
        // Other lines in as many bits; in more bits; where none are sent,
        // in a band of zeros or above max_sfb; too few.
        let mut other = lines.clone();
        other[..4].copy_from_slice(&[0, 1, 0, -1]);
        assert!(!refused(&other));
        other[..4].copy_from_slice(&[1, 1, 1, 1]);
        assert!(refused(&other));
        for line in [9, 500] {
            let mut other = lines.clone();
            other[line] = 1;
            assert!(refused(&other), "line {line}");
        }
        assert!(refused(&lines[..128]));
    }

    /// Where the merge stops, no two neighbouring sections merge into
    /// fewer bits, and the sections cost no more than one for each band.
    #[test]
    fn merging_stops_where_no_merge_saves_bits() {
        let mut noise = Noise::new(4);
        for fs in SamplingFrequency::ALL {
            let bands = GroupBands::long(fs);
            for _ in 0..20 {
                let max_sfb = bands.num_swb();
                let lines = spectrum(&mut noise, bands, max_sfb);
                let counted = Sections::count(&lines, bands, max_sfb).unwrap();
                let sections = counted.clone().choose();
                let candidates: Vec<_> = (sections.iter())
                    .map(|s| {
                        let class = (s.start..s.end).map(|b| counted.classes[b]).max().unwrap();
                        let candidate = counted.section(s.start, s.end, class);
                        assert_eq!(candidate.codebook, s.codebook);
                        candidate
                    })
                    .collect();
                for pair in candidates.windows(2) {
                    let merged = counted.merge(pair[0], pair[1]);
                    assert!(merged.bits >= pair[0].bits + pair[1].bits);
                }
                let total: u32 = candidates.iter().map(|c| c.bits).sum();
                let unmerged: u32 = (0..max_sfb)
                    .map(|b| counted.section(b, b + 1, counted.classes[b]).bits)
                    .sum();
                assert!(total <= unmerged);
                let scalefactors = (sections.bands_with_scalefactors())
                    .filter(|&b| sections.is_zero_band(b))
                    .count() as u32;
                assert_eq!(
                    total,
                    sections.section_data_bits()
                        + sections.spectral_data_bits()
                        + scalefactors * scalefactors::REPEATED_BITS
                );
            }
        }
    }

    #[test]
    fn neighbours_merge_into_the_higher_pair_of_codebooks() {
        let bands = GroupBands::long(SamplingFrequency::Hz48000);
        // Bands 0 to 3 of ones, 4 of a 7: the 7 alone would cost a
        // section, and coding the ones in codebook 7 or 8 costs less than
        // that.
        let mut lines = vec![0; 1024];
        lines[..16].fill(1);
        lines[16] = 7;
        let sections = round_trip(&lines, bands, 5);
        let books: Vec<_> = sections.iter().map(|s| s.codebook.number()).collect();
        assert_eq!(books.len(), 1, "{books:?}");
        assert!([7, 8].contains(&books[0]));
    }

    /// Extended to a pair's max_sfb, sections decode as before, the bands
    /// added zero, in the bits counted.
    #[test]
    fn sections_extend_to_a_shared_max_sfb() {
        let mut noise = Noise::new(9);
        for fs in SamplingFrequency::ALL {
            for bands in [GroupBands::long(fs), GroupBands::short(fs, 3)] {
                for _ in 0..20 {
                    let own = noise.below(bands.num_swb() as u32 + 1) as usize;
                    let lines = spectrum(&mut noise, bands, own);
                    let shared = own + noise.below((bands.num_swb() - own) as u32 + 1) as usize;
                    let mut sections = Sections::choose(&lines, bands, own).unwrap();
                    let before = sections.section_data_bits();
                    let added = sections.extension_bits(shared);
                    sections.extend(shared);
                    assert_eq!(sections.max_sfb(), shared);
                    assert_eq!(sections.section_data_bits(), before + added);
                    assert!((own..shared).all(|sfb| sections.is_zero_band(sfb)));
                    let mut w = BitWriter::new();
                    sections.write_section_data(&mut w);
                    sections.write_spectral_data(&lines, &mut w);
                    let bytes = w.into_bytes();
                    let mut r = BitReader::new(&bytes);
                    let decoded = section_data(&mut r, bands.transform(), shared);
                    assert_eq!(
                        spectral_data(&mut r, &decoded, &offsets(bands), lines.len()),
                        lines
                    );
                }
            }
        }
    }

    #[test]
    fn max_sfb_ends_at_the_highest_band_with_a_line() {
        let bands = GroupBands::long(SamplingFrequency::Hz44100);
        let mut lines = vec![0; 1024];
        assert_eq!(max_sfb(&lines, bands), 0);
        lines[0] = 1;
        assert_eq!(max_sfb(&lines, bands), 1);
        // Line 196 opens band 24.
        lines[196] = -2;
        assert_eq!(max_sfb(&lines, bands), 25);
        lines[1023] = 1;
        assert_eq!(max_sfb(&lines, bands), 49);
        // In a group of two short windows, band 13 starts at line 224.
        let group = GroupBands::short(SamplingFrequency::Hz44100, 2);
        let mut lines = vec![0; 256];
        lines[223] = 1;
        assert_eq!(max_sfb(&lines, group), 13);
        lines[224] = 1;
        assert_eq!(max_sfb(&lines, group), 14);
    }
}
