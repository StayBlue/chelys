// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Spectrum clipping (clause C.8.2), "a method of noiseless dynamic range
//! limiting" that "may be applied" as the first step of noiseless coding:
//! up to four lines of a long window sent with a magnitude of 1, the rest
//! of each sent as a pulse that the decoder adds back (see [`Pulses`]),
//! "only if it results in a net savings of bits".
//!
//! Spec reading: clause C.8.2 says that for the clipped lines "the long
//! block scalefactor bands and coefficient ordering within those bands are
//! used regardless of the window sequence", while "the pulse escape method
//! is illegal for a block whose window_sequence is EIGHT_SHORT_SEQUENCE"
//! (clause 9.3). The normative clause is followed: only long windows are
//! clipped.
//!
//! Spec reading: clause C.8.2 codes each clipped line as "an offset from
//! the base of the previously indicated scalefactor band", but clause 9.3
//! decodes each `pulse_offset` after the first from the line of the pulse
//! before (`k += pulse_offset[j]`). The normative clause is followed (see
//! [`Pulses::new`]). Settled by decoding with ffmpeg (black box): it decodes
//! streams whose pulses are so sent, four of them 31 lines apart, as the
//! same lines sent unclipped, at 8, 44.1 and 96 kHz; the conformance test
//! is `ffmpeg_restores_clipped_lines_as_sent_unclipped`.

use super::pulses::{MAX_AMP, MAX_OFFSET, MAX_PULSES, Pulses};
use super::sections::{Counted, Sections};
use crate::channel::FRAME_LINES;
use crate::scalefactor_bands::{GroupBands, Transform};

/// The largest magnitude that a line is clipped from: "Up to four
/// coefficients can be coded separately as magnitudes in excess of one,
/// with a value of +-1 left in the quantized coefficient array to carry
/// the sign" (clause C.8.2), the excess a `pulse_amp` of at most
/// [`MAX_AMP`].
const MAX_CLIPPED: u32 = MAX_AMP + 1;

/// A clipping of a long window's lines: its pulses, and the lines it
/// leaves, counted for sectioning (see [`Sections::count`]).
#[derive(Clone, Debug)]
pub struct Clipping {
    pub pulses: Pulses,
    pub counted: Counted,
}

impl Clipping {
    /// The fewest bits the clipping takes beyond the
    /// `pulse_data_present` flag that every coding sends: those of its
    /// `pulse_data()`, and the least of its sections (see
    /// [`Counted::least_bits`]).
    pub fn least_bits(&self) -> u32 {
        self.counted.least_bits() + self.pulses.bits() - Pulses::NONE.bits()
    }
}

/// The lines that clause C.8.2's strategy clips, highest first: "One
/// strategy for applying spectrum clipping is to clip high-frequency
/// coefficients whose absolute amplitudes are larger than one."
///
/// Spec gap: clause C.8.2 does not say which of them, nor how many. The
/// highest lines of magnitudes from 2 to [`MAX_CLIPPED`] are taken, up to
/// [`MAX_PULSES`] of them, each within [`MAX_OFFSET`] lines of the one
/// above it, as `pulse_data()` can send no others (clause 9.3); a line of a
/// larger magnitude, which cannot be clipped to 1, is passed over.
fn candidates(lines: &[i32]) -> ([usize; MAX_PULSES], usize) {
    let mut chosen = [0; MAX_PULSES];
    let mut count = 0;
    for (i, line) in lines.iter().enumerate().rev() {
        if !(2..=MAX_CLIPPED).contains(&line.unsigned_abs()) {
            continue;
        }
        if count > 0 && chosen[count - 1] - i > MAX_OFFSET {
            break;
        }
        chosen[count] = i;
        count += 1;
        if count == MAX_PULSES {
            break;
        }
    }
    (chosen, count)
}

/// The scalefactor band of `bands` that holds `line`.
fn band_of(bands: GroupBands, line: usize) -> usize {
    (1..=bands.num_swb())
        .find(|&sfb| bands.offset(sfb) > line)
        .expect("a line of the bands")
        - 1
}

/// The clippings of a long window's quantized `lines`, counted unclipped as
/// `counted`, among which [`choose`] chooses, into `clippings`: one of the
/// highest `n` of the lines that clause C.8.2's strategy clips, for each
/// `n` whose lowest is within [`MAX_OFFSET`] lines of the start of its
/// band, as `pulse_data()` can send no other (clause 9.3).
///
/// Spec gap: clause C.8.2 clips "Up to four coefficients", and does not say
/// how many. Each number is counted, and [`choose`] takes the one that
/// saves the most bits.
///
/// # Panics
///
/// If `bands` are not those of a long window, or `counted` does not count
/// `lines` in them: a caller's error.
pub fn clippings(
    lines: &[i32],
    bands: GroupBands,
    counted: &Counted,
    clippings: &mut Vec<Clipping>,
) {
    assert_eq!(
        bands.transform(),
        Transform::Long,
        "only long windows are clipped"
    );
    clippings.clear();
    let (chosen, count) = candidates(lines);
    if count == 0 {
        return;
    }
    let mut sent = [0; FRAME_LINES];
    sent.copy_from_slice(lines);
    let mut clipped = counted.clone();
    let mut pulses = [(0, 0); MAX_PULSES];
    for n in 1..=count {
        // Each clipping clips the lines of the one before, and one more.
        let line = chosen[n - 1];
        sent[line] = sent[line].signum();
        let sfb = band_of(bands, line);
        clipped.recount_band(sfb, &sent[bands.band(sfb)]);
        // The highest n lines, in ascending order.
        for (pulse, &line) in pulses[..n].iter_mut().zip(chosen[..n].iter().rev()) {
            *pulse = (line, lines[line].unsigned_abs() - 1);
        }
        if let Some(pulses) = Pulses::new(bands, &pulses[..n]) {
            clippings.push(Clipping {
                pulses,
                counted: clipped.clone(),
            });
        }
    }
}

/// The coding of a long window's lines, counted unclipped as `counted`, or
/// of one of `clippings` of them (see [`clippings`]), that takes the
/// fewest bits: its pulses and its sections, each chosen by the greedy
/// merge of clause C.8.3 (see [`Counted::choose`]). A clipping is chosen
/// "only if it results in a net savings of bits" (clause C.8.2).
///
/// Spec gap: clause C.8.2 does not say what bits the saving is counted
/// in. They are those that the clipping changes: of the sections'
/// `section_data()` and `spectral_data()`, the scalefactors they send for
/// bands of zeros (see [`Sections::coding_bits`]), and `pulse_data()`.
/// Of codings of as many bits, the one of fewer pulses is taken.
pub fn choose(counted: &Counted, clippings: &mut Vec<Clipping>) -> (Pulses, Sections) {
    let mut sections = counted.choose();
    let mut pulses = Pulses::NONE;
    let mut bits = sections.coding_bits();
    for clipping in clippings.drain(..) {
        // Sections take at least their least bits.
        if clipping.least_bits() >= bits {
            continue;
        }
        let extra = clipping.pulses.bits() - Pulses::NONE.bits();
        let clipped = clipping.counted.choose();
        if clipped.coding_bits() + extra < bits {
            bits = clipped.coding_bits() + extra;
            (pulses, sections) = (clipping.pulses, clipped);
        }
    }
    (pulses, sections)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling_frequency::SamplingFrequency;

    #[test]
    fn the_highest_lines_within_reach_of_each_other_are_clipped() {
        let mut lines = [0; FRAME_LINES];
        // 17 cannot be clipped to 1, nor 1 any lower.
        for (line, q) in [
            (900, 17),
            (899, -1),
            (880, 3),
            (850, -16),
            (840, 2),
            (800, 5),
        ] {
            lines[line] = q;
        }
        let (chosen, count) = candidates(&lines);
        assert_eq!(chosen[..count], [880, 850, 840]);
        lines[830] = -2;
        lines[820] = 4;
        let (chosen, count) = candidates(&lines);
        assert_eq!(chosen[..count], [880, 850, 840, 830]);
        assert_eq!(candidates(&[1; FRAME_LINES]).1, 0);
    }

    /// The lines the clipping sends, counted, and its pulses restore those
    /// clipped.
    #[test]
    fn each_clipping_counts_the_lines_it_sends() {
        let fs = SamplingFrequency::Hz44100;
        let bands = GroupBands::long(fs);
        let mut lines = [0; FRAME_LINES];
        for (i, line) in lines[..640].iter_mut().enumerate() {
            *line = [1, -1, 0, 1][i % 4];
        }
        lines[600] = 9;
        lines[620] = -2;
        lines[630] = 16;
        let max_sfb = super::super::sections::max_sfb(&lines, bands);
        let counted = Sections::count(&lines, bands, max_sfb).unwrap();
        let mut found = Vec::new();
        clippings(&lines, bands, &counted, &mut found);
        assert_eq!(found.len(), 3);
        for (n, clipping) in found.iter().enumerate() {
            let mut sent = lines;
            clipping.pulses.clip(&mut sent);
            let expected = Sections::count(&sent, bands, max_sfb).unwrap();
            // As many sections, codebooks and bits; the least bits are those
            // of codebooks up to the escape codebook, of the lines counted
            // before they were clipped, which can be fewer.
            let chosen = clipping.counted.clone().choose();
            assert!(clipping.counted.least_bits() <= expected.least_bits());
            assert!(clipping.counted.least_bits() <= chosen.coding_bits());
            assert_eq!(chosen, expected.choose(), "{n}");
            let clipped: Vec<_> = clipping.pulses.iter().map(|(line, _)| line).collect();
            assert_eq!(clipped, [600, 620, 630][2 - n..]);
            assert!(clipped.iter().all(|&line| sent[line].abs() == 1));
        }
        // One pulse saves its band of ones the escape codebook, which
        // costs more than the pulse.
        let (pulses, sections) = choose(&counted, &mut found);
        assert!(pulses.is_present());
        assert!(found.is_empty());
        let unclipped = counted.choose();
        assert!(sections.coding_bits() + pulses.bits() < unclipped.coding_bits() + 1);
    }

    #[test]
    fn a_clipping_that_saves_nothing_is_not_taken() {
        let bands = GroupBands::long(SamplingFrequency::Hz48000);
        let mut lines = [0; FRAME_LINES];
        // Pulses in lines of 2 cost more than they save.
        lines[..8].fill(2);
        let counted = Sections::count(&lines, bands, 2).unwrap();
        let mut found = Vec::new();
        clippings(&lines, bands, &counted, &mut found);
        assert_eq!(found.len(), 4);
        let (pulses, sections) = choose(&counted, &mut found);
        assert_eq!(pulses, Pulses::NONE);
        assert_eq!(sections, counted.choose());
    }

    #[test]
    fn a_lowest_line_out_of_reach_of_its_band_is_not_sent() {
        // At 8 kHz, band 39 spans lines 944 to 1024 (Table 48): line 1000
        // is beyond the reach of its start, 970 within it.
        let bands = GroupBands::long(SamplingFrequency::Hz8000);
        let mut lines = [0; FRAME_LINES];
        lines[1000] = 3;
        lines[970] = 3;
        let counted = Sections::count(&lines, bands, 40).unwrap();
        let mut found = Vec::new();
        clippings(&lines, bands, &counted, &mut found);
        assert_eq!(found.len(), 1);
        let clipped: Vec<_> = found[0].pulses.iter().collect();
        assert_eq!(clipped, [(970, 2), (1000, 2)]);
    }
}
