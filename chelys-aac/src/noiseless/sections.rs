// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Sections (clause 9.1): runs of scalefactor bands whose lines one
//! codebook codes, sent in `section_data()` (Table 17), and their choice by
//! the greedy merge of clause C.8.3.

use chelys_core::bitstream::BitWriter;

use super::codebooks::{Codebook, MAX_MAGNITUDE};
use super::scalefactors;
use super::spectrum::{self, UNCODABLE};
use crate::scalefactor_bands::{MAX_BANDS, Transform};

/// A section: the scalefactor bands `start..end` of its window group, and
/// the codebook that codes their lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Section {
    pub codebook: Codebook,
    pub start: usize,
    pub end: usize,
}

/// The sections of one window group, of bands 0 to `max_sfb`: of a long
/// window's scalefactor bands, or of the scalefactor bands of a group of
/// short windows, each of which holds the group's windows' lines of a band
/// (clause 8.3.4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sections {
    transform: Transform,
    len: usize,
    codebooks: [Codebook; MAX_BANDS],
    /// The band each section ends before.
    ends: [u8; MAX_BANDS],
    /// Bit `sfb` is set where band `sfb` has no line but zero.
    zero_bands: u64,
    /// The bits of the group's `spectral_data()`.
    spectral_bits: u32,
}

impl Sections {
    /// Divides the scalefactor bands of a window group into sections and
    /// chooses their codebooks, by the greedy merge of clause C.8.3: one
    /// section for each band first, each with the codebook of the smallest
    /// index that codes it, then, while merging two neighbouring sections
    /// saves bits, the merge that saves the most, with the higher of their
    /// codebooks.
    ///
    /// `lines` are the group's quantized lines in the order they are sent
    /// (clause 8.3.5), and `offsets` its `sect_sfb_offset` (clause 8.3.4):
    /// the first line of each band, then the end of the last, `max_sfb + 1`
    /// in all. For a long window they are the lines of its spectrum and
    /// [`swb_offsets`](crate::scalefactor_bands::swb_offsets) up to
    /// `max_sfb`. `transform` is [`Transform::Short`] for the groups of an
    /// EIGHT_SHORT_SEQUENCE, where `sect_len_incr` has 3 bits rather than
    /// 5.
    ///
    /// Returns `None` if a line's magnitude exceeds [`MAX_MAGNITUDE`],
    /// which no codebook codes.
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
    /// a tie.
    ///
    /// Spec reading: the "total bit count" that merging lowers is that of
    /// the sections' `section_data()`, `scale_factor_data()` and
    /// `spectral_data()`. Merging changes the scalefactors only where it
    /// puts a band with no line but zero in a section with a spectrum
    /// codebook, which sends its scalefactor; that costs the bits of a
    /// difference of 0 (see [`scalefactors::Dpcm`]).
    ///
    /// Spec gap: clause C.8.3 does not say which merge comes first where
    /// two save the same bits; the lower in frequency is taken.
    ///
    /// # Panics
    ///
    /// If `offsets` do not run from 0, rising by a multiple of 4 lines
    /// each, to at most the end of `lines`, or count more than
    /// [`MAX_BANDS`] bands: a caller's error.
    pub fn choose(lines: &[i32], offsets: &[u16], transform: Transform) -> Option<Self> {
        let bands = Bands::count(lines, offsets, transform)?;
        let max_sfb = offsets.len() - 1;
        let mut zero_bands = 0;
        for band in 0..max_sfb {
            if bands.zero_bands_below[band + 1] > bands.zero_bands_below[band] {
                zero_bands |= 1 << band;
            }
        }

        // One section for each band.
        let mut sections = [Candidate::default(); MAX_BANDS];
        for (band, section) in sections[..max_sfb].iter_mut().enumerate() {
            *section = bands.section(band, band + 1, bands.classes[band]);
        }
        // merged[i]: sections i and i + 1 merged.
        let mut merged = [Candidate::default(); MAX_BANDS];
        for i in 0..max_sfb.saturating_sub(1) {
            merged[i] = bands.merge(sections[i], sections[i + 1]);
        }
        let mut len = max_sfb;
        loop {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..len.saturating_sub(1) {
                let separate = sections[i].bits + sections[i + 1].bits;
                let saved = separate.saturating_sub(merged[i].bits);
                if saved > best.map_or(0, |(_, saved)| saved) {
                    best = Some((i, saved));
                }
            }
            let Some((i, _)) = best else { break };
            sections[i] = merged[i];
            sections.copy_within(i + 2..len, i + 1);
            merged.copy_within(i + 1..len - 1, i);
            len -= 1;
            if i > 0 {
                merged[i - 1] = bands.merge(sections[i - 1], sections[i]);
            }
            if i + 1 < len {
                merged[i] = bands.merge(sections[i], sections[i + 1]);
            }
        }

        let mut chosen = Self {
            transform,
            len,
            codebooks: [Codebook::ZERO; MAX_BANDS],
            ends: [0; MAX_BANDS],
            zero_bands,
            spectral_bits: 0,
        };
        for (i, section) in sections[..len].iter().enumerate() {
            chosen.codebooks[i] = section.codebook;
            chosen.ends[i] = section.end as u8;
            chosen.spectral_bits +=
                bands.spectral_bits(section.start, section.end, section.codebook);
        }
        Some(chosen)
    }

    pub fn transform(&self) -> Transform {
        self.transform
    }

    /// The bands the sections span: `max_sfb`.
    pub fn max_sfb(&self) -> usize {
        self.len
            .checked_sub(1)
            .map_or(0, |last| usize::from(self.ends[last]))
    }

    pub fn iter(&self) -> impl Iterator<Item = Section> + '_ {
        (0..self.len).map(|i| Section {
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

    /// The bands in sections with a spectrum codebook, which send a
    /// scalefactor (Table 18), in order.
    pub fn bands_with_scalefactors(&self) -> impl Iterator<Item = usize> + '_ {
        (self.iter())
            .filter(|section| section.codebook != Codebook::ZERO)
            .flat_map(|section| section.start..section.end)
    }

    /// The bits [`write_section_data`](Self::write_section_data) writes.
    pub fn section_data_bits(&self) -> u32 {
        self.iter()
            .map(|section| section_bits(self.transform, section.end - section.start))
            .sum()
    }

    /// The bits [`write_spectral_data`](Self::write_spectral_data) writes
    /// for the lines the sections were chosen for.
    pub fn spectral_data_bits(&self) -> u32 {
        self.spectral_bits
    }

    /// Writes the group's part of `section_data()` (Table 17): for each
    /// section its `sect_cb`, then its length in bands as `sect_len_incr`
    /// fields, each `sect_esc_val` but the last, which is less.
    pub fn write_section_data(&self, w: &mut BitWriter) {
        let (bits, esc) = sect_len_incr(self.transform);
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
    /// of each section with a spectrum codebook (see [`spectrum::write`]).
    ///
    /// `lines` and `offsets` are those the sections were chosen for.
    ///
    /// # Panics
    ///
    /// If they are not, and a section's codebook cannot code its lines: a
    /// caller's error.
    pub fn write_spectral_data(&self, lines: &[i32], offsets: &[u16], w: &mut BitWriter) {
        for section in self.iter() {
            if let Some(book) = section.codebook.spectrum() {
                let start = usize::from(offsets[section.start]);
                let end = usize::from(offsets[section.end]);
                spectrum::write(book, &lines[start..end], w);
            }
        }
    }
}

/// `max_sfb` for `lines` in the bands `offsets` (as for
/// [`Sections::choose`]): the bands up to the highest with a line that is
/// not zero, so that no section has to code the bands of zeros above it.
///
/// Spec reading: `max_sfb` is "one greater than the highest active
/// scalefactor band in all groups" (clause 8.3.4), where a band is active
/// unless its codebook is `ZERO_HCB` (clause 8.3.2.5). A band of zeros
/// above the highest with a line that is not zero would cost a section or
/// a longer one, where outside `max_sfb` it costs nothing, so it is left
/// out. Over the groups of an EIGHT_SHORT_SEQUENCE, `max_sfb` is the
/// largest of theirs.
pub fn max_sfb(lines: &[i32], offsets: &[u16]) -> usize {
    let end = offsets.last().map_or(0, |&end| usize::from(end));
    lines[..end]
        .iter()
        .rposition(|&line| line != 0)
        .map_or(0, |highest| {
            offsets.partition_point(|&offset| usize::from(offset) <= highest)
        })
}

/// `sect_len_incr`'s bits and `sect_esc_val` (Table 17).
fn sect_len_incr(transform: Transform) -> (u32, usize) {
    match transform {
        Transform::Long => (5, (1 << 5) - 1),
        Transform::Short => (3, (1 << 3) - 1),
    }
}

/// The bits of a section of `len` bands in `section_data()`: `sect_cb`,
/// and a `sect_len_incr` for each `sect_esc_val` bands and one more.
fn section_bits(transform: Transform, len: usize) -> u32 {
    let (bits, esc) = sect_len_incr(transform);
    4 + bits * (len / esc + 1) as u32
}

/// The codebooks by largest magnitude (Table 59): `ZERO_HCB`, the pairs
/// that share a LAV, and `ESC_HCB`.
const CLASSES: [&[u8]; 7] = [&[0], &[1, 2], &[3, 4], &[5, 6], &[7, 8], &[9, 10], &[11]];

/// The first of [`CLASSES`] whose codebooks code `largest`.
fn class(largest: u32) -> usize {
    CLASSES
        .iter()
        .position(|books| codebook(books[0]).largest_magnitude() >= largest)
        .expect("ESC_HCB codes up to MAX_MAGNITUDE")
}

fn codebook(number: u8) -> Codebook {
    Codebook::new(number).expect("a codebook")
}

/// A section as the merge sees it: its bands, the class of its codebook
/// and the codebook, and its bits in all.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    start: usize,
    end: usize,
    class: usize,
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

/// The bits of a window group's bands in each codebook, summed from the
/// lowest band, so that a run of bands is counted by a difference.
struct Bands {
    transform: Transform,
    /// `bits_below[b][cb]`: the bits of bands `0..b` in codebook `cb`.
    bits_below: [[u32; 12]; MAX_BANDS + 1],
    /// The bands of zeros in bands `0..b`.
    zero_bands_below: [u8; MAX_BANDS + 1],
    /// The class of each band: the first that codes it.
    classes: [usize; MAX_BANDS],
}

impl Bands {
    fn count(lines: &[i32], offsets: &[u16], transform: Transform) -> Option<Self> {
        assert!(
            offsets.first() == Some(&0),
            "the first band starts at line 0"
        );
        let max_sfb = offsets.len() - 1;
        assert!(max_sfb <= MAX_BANDS, "{max_sfb} bands are too many");
        let mut bands = Self {
            transform,
            bits_below: [[0; 12]; MAX_BANDS + 1],
            zero_bands_below: [0; MAX_BANDS + 1],
            classes: [0; MAX_BANDS],
        };
        for band in 0..max_sfb {
            let (start, end) = (usize::from(offsets[band]), usize::from(offsets[band + 1]));
            assert!(
                start < end && (end - start) % 4 == 0,
                "band {band} holds lines {start}..{end}, not a multiple of 4"
            );
            let lines = &lines[start..end];
            let largest = lines.iter().map(|line| line.unsigned_abs()).max();
            let largest = largest.expect("a band holds lines");
            if largest > MAX_MAGNITUDE {
                return None;
            }
            let bits = spectrum::bits_by_codebook(lines, largest);
            for (codebook, bits) in bits.into_iter().enumerate() {
                bands.bits_below[band + 1][codebook] = bands.bits_below[band][codebook] + bits;
            }
            bands.zero_bands_below[band + 1] =
                bands.zero_bands_below[band] + u8::from(largest == 0);
            bands.classes[band] = class(largest);
        }
        Some(bands)
    }

    /// The bits of the lines of bands `start..end` in `codebook`.
    fn spectral_bits(&self, start: usize, end: usize, codebook: Codebook) -> u32 {
        let codebook = usize::from(codebook.number());
        let bits = self.bits_below[end][codebook] - self.bits_below[start][codebook];
        debug_assert!(
            bits < UNCODABLE,
            "codebook {codebook} cannot code bands {start}..{end}"
        );
        bits
    }

    /// The section of bands `start..end` with the codebook of `class` that
    /// costs the fewest bits, the lower on a tie.
    fn section(&self, start: usize, end: usize, class: usize) -> Candidate {
        let (codebook, bits) = (CLASSES[class].iter())
            .map(|&number| {
                let codebook = codebook(number);
                (codebook, self.bits(start, end, codebook))
            })
            .min_by_key(|&(_, bits)| bits)
            .expect("a class has a codebook");
        Candidate {
            start,
            end,
            class,
            codebook,
            bits,
        }
    }

    /// Sections `a` and `b`, neighbours, merged into one.
    fn merge(&self, a: Candidate, b: Candidate) -> Candidate {
        debug_assert_eq!(a.end, b.start);
        self.section(a.start, b.end, a.class.max(b.class))
    }

    /// The bits of a section of bands `start..end` with `codebook`: its
    /// `section_data()`, its `spectral_data()`, and the scalefactors of its
    /// bands of zeros, which a section with a spectrum codebook sends.
    fn bits(&self, start: usize, end: usize, codebook: Codebook) -> u32 {
        let scalefactors = if codebook == Codebook::ZERO {
            0
        } else {
            let zero_bands = self.zero_bands_below[end] - self.zero_bands_below[start];
            u32::from(zero_bands) * scalefactors::REPEATED_BITS
        };
        section_bits(self.transform, end - start)
            + self.spectral_bits(start, end, codebook)
            + scalefactors
    }
}

#[cfg(test)]
mod tests {
    use super::super::decoder::{BitReader, Noise, section_data, spectral_data};
    use super::*;
    use crate::sampling_frequency::SamplingFrequency;
    use crate::scalefactor_bands::{num_swb, swb_offsets};

    /// The `sect_sfb_offset` of a group of `windows` short windows (clause
    /// 8.3.4).
    fn group_offsets(fs: SamplingFrequency, windows: u16) -> Vec<u16> {
        (swb_offsets(fs, Transform::Short).iter())
            .map(|offset| offset * windows)
            .collect()
    }

    /// Chooses sections for `lines`, writes them and their lines, and
    /// checks that they decode as written, in the bits counted.
    fn round_trip(lines: &[i32], offsets: &[u16], transform: Transform) -> Sections {
        let sections = Sections::choose(lines, offsets, transform).expect("codable lines");
        let max_sfb = offsets.len() - 1;
        assert_eq!(sections.max_sfb(), max_sfb);
        let mut w = BitWriter::new();
        sections.write_section_data(&mut w);
        assert_eq!(w.bit_len() as u32, sections.section_data_bits());
        sections.write_spectral_data(lines, offsets, &mut w);
        assert_eq!(
            w.bit_len() as u32,
            sections.section_data_bits() + sections.spectral_data_bits()
        );
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        let decoded = section_data(&mut r, transform, max_sfb);
        let written: Vec<_> = (sections.iter())
            .map(|s| (s.codebook.number(), s.start, s.end))
            .collect();
        assert_eq!(decoded, written);
        let end = usize::from(offsets[max_sfb]);
        assert_eq!(spectral_data(&mut r, &decoded, offsets, end), lines[..end]);
        sections
    }

    #[test]
    fn spectra_decode_as_written_at_every_sampling_frequency() {
        let mut noise = Noise::new(3);
        for fs in SamplingFrequency::ALL {
            let long = swb_offsets(fs, Transform::Long);
            for _ in 0..40 {
                let max_sfb = noise.below(num_swb(fs, Transform::Long) as u32 + 1) as usize;
                let lines = noise.spectrum(long, 1024);
                round_trip(&lines, &long[..=max_sfb], Transform::Long);
            }
            for windows in 1..=8 {
                let offsets = group_offsets(fs, windows);
                let lines = noise.spectrum(&offsets, 128 * usize::from(windows));
                round_trip(&lines, &offsets, Transform::Short);
            }
        }
    }

    #[test]
    fn long_sections_escape_their_length() {
        let fs = SamplingFrequency::Hz32000;
        let offsets = swb_offsets(fs, Transform::Long);
        // Ones in the first and last bands, zeros between: a section of
        // zeros of 49 bands, sent as 31 and 18.
        let mut lines = vec![0; 1024];
        lines[0] = 1;
        lines[1023] = -1;
        let sections = round_trip(&lines, offsets, Transform::Long);
        let zeros: Vec<_> = (sections.iter())
            .filter(|section| section.codebook == Codebook::ZERO)
            .map(|section| section.end - section.start)
            .collect();
        assert_eq!(zeros, [49]);
        assert_eq!(sections.section_data_bits(), 3 * 4 + 5 + 2 * 5 + 5);
        // A section of 31 bands exactly ends with a zero increment.
        let mut lines = vec![0; 1024];
        lines[usize::from(offsets[31])] = 1;
        let sections = round_trip(&lines, &offsets[..=32], Transform::Long);
        assert_eq!(sections.iter().next().map(|s| s.end), Some(31));
        assert_eq!(sections.section_data_bits(), 4 + 2 * 5 + 4 + 5);
    }

    #[test]
    fn short_sections_escape_their_length() {
        let fs = SamplingFrequency::Hz8000;
        let offsets = group_offsets(fs, 1);
        // Zeros below the last of 15 bands: a section of 14, sent as 7, 7
        // and 0.
        let mut lines = vec![0; 128];
        lines[127] = 3;
        let sections = round_trip(&lines, &offsets, Transform::Short);
        assert_eq!(sections.iter().next().map(|s| s.end), Some(14));
        assert_eq!(sections.section_data_bits(), 4 + 3 * 3 + 4 + 3);
    }

    #[test]
    fn zeros_up_to_max_sfb_are_one_section_of_zero_hcb() {
        let offsets = swb_offsets(SamplingFrequency::Hz48000, Transform::Long);
        let sections = round_trip(&[0; 1024], offsets, Transform::Long);
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
        let sections = round_trip(&[0; 1024], &[0], Transform::Long);
        assert_eq!(sections.iter().count(), 0);
        assert_eq!(sections.section_data_bits(), 0);
    }

    #[test]
    fn lines_beyond_the_largest_magnitude_are_refused() {
        let offsets = swb_offsets(SamplingFrequency::Hz48000, Transform::Long);
        let mut lines = vec![0; 1024];
        lines[500] = -8191;
        assert!(Sections::choose(&lines, offsets, Transform::Long).is_some());
        lines[500] = -8192;
        assert_eq!(Sections::choose(&lines, offsets, Transform::Long), None);
    }

    /// Where the merge stops, no two neighbouring sections merge into
    /// fewer bits, and the sections cost no more than one for each band.
    #[test]
    fn merging_stops_where_no_merge_saves_bits() {
        let mut noise = Noise::new(4);
        for fs in SamplingFrequency::ALL {
            let offsets = swb_offsets(fs, Transform::Long);
            for _ in 0..20 {
                let lines = noise.spectrum(offsets, 1024);
                let sections = Sections::choose(&lines, offsets, Transform::Long).unwrap();
                let bands = Bands::count(&lines, offsets, Transform::Long).unwrap();
                let candidates: Vec<_> = (sections.iter())
                    .map(|s| {
                        let class = (s.start..s.end).map(|b| bands.classes[b]).max().unwrap();
                        let candidate = bands.section(s.start, s.end, class);
                        assert_eq!(candidate.codebook, s.codebook);
                        candidate
                    })
                    .collect();
                for pair in candidates.windows(2) {
                    let merged = bands.merge(pair[0], pair[1]);
                    assert!(merged.bits >= pair[0].bits + pair[1].bits);
                }
                let total: u32 = candidates.iter().map(|c| c.bits).sum();
                let unmerged: u32 = (0..offsets.len() - 1)
                    .map(|b| bands.section(b, b + 1, bands.classes[b]).bits)
                    .sum();
                assert!(total <= unmerged);
                let scalefactors = (sections.iter())
                    .filter(|s| s.codebook != Codebook::ZERO)
                    .flat_map(|s| s.start..s.end)
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
        let offsets = swb_offsets(SamplingFrequency::Hz48000, Transform::Long);
        // Bands 0 to 3 of ones, 4 of a 7, then zeros: the 7 alone would
        // cost a section, and coding the ones in codebook 7 or 8 costs less
        // than that.
        let mut lines = vec![0; 1024];
        lines[..16].fill(1);
        lines[16] = 7;
        let sections = round_trip(&lines, &offsets[..=5], Transform::Long);
        let books: Vec<_> = sections.iter().map(|s| s.codebook.number()).collect();
        assert_eq!(books.len(), 1, "{books:?}");
        assert!([7, 8].contains(&books[0]));
    }

    #[test]
    fn max_sfb_ends_at_the_highest_band_with_a_line() {
        let offsets = swb_offsets(SamplingFrequency::Hz44100, Transform::Long);
        let mut lines = vec![0; 1024];
        assert_eq!(max_sfb(&lines, offsets), 0);
        lines[0] = 1;
        assert_eq!(max_sfb(&lines, offsets), 1);
        // Line 196 opens band 24.
        lines[196] = -2;
        assert_eq!(max_sfb(&lines, offsets), 25);
        lines[1023] = 1;
        assert_eq!(max_sfb(&lines, offsets), 49);
        // Lines past the bands given are not looked at.
        assert_eq!(max_sfb(&lines, &offsets[..=10]), 1);
    }
}
