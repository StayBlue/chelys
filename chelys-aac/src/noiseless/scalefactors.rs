// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! `scale_factor_data()` (Table 18): the scalefactor of each band in a
//! section whose codebook sends one, as its difference from the one sent
//! before it, or from `global_gain` for the first (clause 11.3.2), in the
//! scalefactor codebook (Table A.1), through the window groups in order.
//!
//! The scalefactors of a group are given for its bands `0..max_sfb`, of
//! which those of bands that send none are not used.
//!
//! Spec gap: a band of zeros in a section with a spectrum codebook sends a
//! scalefactor (Table 18), which scales nothing, and the standard leaves
//! its value to the encoder. It is sent as the scalefactor before it, a
//! difference of 0 that costs the fewest bits and leaves the next
//! difference as it would be without it; the scalefactor given for the
//! band is not used.

use chelys_core::bitstream::BitWriter;

use super::codebooks::{Codeword, SCALEFACTOR_CODEWORDS};
use super::sections::Sections;

/// `index_offset` (Table 58): a difference of scalefactors is its codeword
/// index plus this.
///
/// Spec reading: clause 9.3 and Table 58 add `index_offset` to the index,
/// for differences of -60 to +60, but the pseudo code of clause 11.3.2
/// subtracts it ("decode_huffman() - index_offset"), which would make them
/// 60 to 180. Clause 9.3 and Table 58 are followed.
const INDEX_OFFSET: i32 = -60;

/// The codeword of a difference of scalefactors, or `None` outside the
/// codebook's range of -60 to +60 (Table 58).
fn codeword(difference: i32) -> Option<Codeword> {
    let index = usize::try_from(difference.checked_sub(INDEX_OFFSET)?).ok()?;
    SCALEFACTOR_CODEWORDS.get(index).copied()
}

/// The bits of the codeword of a difference of 0, which a band of zeros
/// costs in a section with a spectrum codebook.
pub(super) const REPEATED_BITS: u32 = SCALEFACTOR_CODEWORDS[(0 - INDEX_OFFSET) as usize].len;

/// The bits of `scale_factor_data()` for the window groups `groups`, each
/// with its sections and scalefactors, from `global_gain`; or `None` if a
/// difference falls outside the codebook's range.
///
/// # Panics
///
/// If a group has fewer scalefactors than its sections span bands: a
/// caller's error.
pub fn bits<'a>(
    global_gain: u8,
    groups: impl IntoIterator<Item = (&'a Sections, &'a [u8])>,
) -> Option<u32> {
    let mut bits = 0;
    for_each_difference(global_gain, groups, |difference| {
        bits += codeword(difference)?.len;
        Some(())
    })?;
    Some(bits)
}

/// Writes `scale_factor_data()` for the window groups `groups`, as
/// [`bits`] counts it.
///
/// # Panics
///
/// Where [`bits`] would return `None` or panic, before writing anything: a
/// caller's error.
pub fn write<'a>(
    global_gain: u8,
    groups: impl IntoIterator<Item = (&'a Sections, &'a [u8])> + Clone,
    w: &mut BitWriter,
) {
    assert!(
        bits(global_gain, groups.clone()).is_some(),
        "a difference of scalefactors has no codeword"
    );
    for_each_difference(global_gain, groups, |difference| {
        let codeword = codeword(difference)?;
        w.write(codeword.bits, codeword.len);
        Some(())
    })
    .expect("differences with codewords");
}

/// Calls `f` with each difference sent, from the running scalefactor that
/// starts at `global_gain` (clause 11.3.2), until it returns `None`.
fn for_each_difference<'a>(
    global_gain: u8,
    groups: impl IntoIterator<Item = (&'a Sections, &'a [u8])>,
    mut f: impl FnMut(i32) -> Option<()>,
) -> Option<()> {
    let mut last = global_gain;
    for (sections, scalefactors) in groups {
        assert!(
            scalefactors.len() >= sections.max_sfb(),
            "a scalefactor for each band"
        );
        for sfb in sections.bands_with_scalefactors() {
            let difference = if sections.is_zero_band(sfb) {
                0
            } else {
                let difference = i32::from(scalefactors[sfb]) - i32::from(last);
                last = scalefactors[sfb];
                difference
            };
            f(difference)?;
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::super::codebooks::Codebook;
    use super::super::decoder::{BitReader, Noise, scale_factor_data, section_data};
    use super::*;
    use crate::sampling_frequency::SamplingFrequency;
    use crate::scalefactor_bands::GroupBands;

    /// C5: every difference has a codeword or none, without overflow.
    #[test]
    fn differences_have_codewords_from_minus_60_to_60() {
        for difference in [-61, 61, i32::MIN, i32::MAX] {
            assert_eq!(codeword(difference), None);
        }
        // Table A.1, indexes 0, 60 and 120.
        assert_eq!(codeword(-60), Some(SCALEFACTOR_CODEWORDS[0]));
        assert_eq!(codeword(0), Some(Codeword { bits: 0, len: 1 }));
        assert_eq!(codeword(60), Some(SCALEFACTOR_CODEWORDS[120]));
        assert_eq!(REPEATED_BITS, 1);
    }

    /// Writes the sections and scalefactors of each group, then decodes
    /// them, checking the bits counted and the scalefactors decoded.
    fn round_trip(global_gain: u8, groups: &[(Sections, Vec<u8>)]) {
        let groups_ref = || groups.iter().map(|(s, sf)| (s, sf.as_slice()));
        let mut w = BitWriter::new();
        let mut counted = 0;
        for (sections, _) in groups {
            sections.write_section_data(&mut w);
            counted += sections.section_data_bits();
        }
        counted += bits(global_gain, groups_ref()).expect("differences in range");
        write(global_gain, groups_ref(), &mut w);
        assert_eq!(w.bit_len() as u32, counted);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        let decoded: Vec<_> = (groups.iter())
            .map(|(sections, _)| {
                section_data(&mut r, sections.bands().transform(), sections.max_sfb())
            })
            .collect();
        let mut last_sf = i32::from(global_gain);
        let mut expected_last = i32::from(global_gain);
        for ((sections, scalefactors), decoded) in groups.iter().zip(&decoded) {
            let sf = scale_factor_data(&mut r, decoded, &mut last_sf);
            for (sfb, &sf) in sf.iter().enumerate() {
                let codebook = (sections.iter())
                    .find(|s| (s.start..s.end).contains(&sfb))
                    .unwrap()
                    .codebook;
                // A band of zeros repeats the scalefactor before it.
                if codebook == Codebook::ZERO {
                    assert_eq!(sf, 0);
                } else if sections.is_zero_band(sfb) {
                    assert_eq!(sf, expected_last);
                } else {
                    assert_eq!(sf, i32::from(scalefactors[sfb]));
                    expected_last = sf;
                }
            }
        }
        assert_eq!(r.position(), counted as usize);
    }

    #[test]
    fn scalefactors_decode_as_written_through_the_groups() {
        let mut noise = Noise::new(5);
        let fs = SamplingFrequency::Hz44100;
        for _ in 0..200 {
            let global_gain = 80 + noise.below(40) as u8;
            // Scalefactors that wander within the codebook's reach from one
            // that is sent to the next, and anything where none is sent.
            let mut sf = i32::from(global_gain);
            let mut groups = Vec::new();
            let max_sfb = noise.below(15) as usize;
            for _ in 0..1 + noise.below(4) {
                let bands = GroupBands::short(fs, 1 + noise.below(3) as usize);
                let offsets: Vec<_> = (0..=bands.num_swb())
                    .map(|sfb| bands.offset(sfb) as u16)
                    .collect();
                let mut lines = noise.spectrum(&offsets, bands.lines());
                lines[bands.offset(max_sfb)..].fill(0);
                let sections = Sections::choose(&lines, bands, max_sfb).unwrap();
                let mut scalefactors: Vec<u8> =
                    (0..max_sfb).map(|_| noise.below(256) as u8).collect();
                for sfb in sections.bands_with_scalefactors() {
                    if !sections.is_zero_band(sfb) {
                        sf = (sf + noise.below(121) as i32 - 60).clamp(0, 255);
                        scalefactors[sfb] = sf as u8;
                    }
                }
                groups.push((sections, scalefactors));
            }
            round_trip(global_gain, &groups);
        }
    }

    fn two_bands_of_ones() -> Sections {
        let mut lines = vec![0; 1024];
        lines[0] = 1;
        lines[4] = 1;
        Sections::choose(&lines, GroupBands::long(SamplingFrequency::Hz48000), 2).unwrap()
    }

    #[test]
    fn a_difference_out_of_range_is_counted_as_none() {
        let sections = two_bands_of_ones();
        let sixty = codeword(60).unwrap().len;
        assert_eq!(bits(100, [(&sections, &[160, 220][..])]), Some(2 * sixty));
        assert_eq!(bits(100, [(&sections, &[161, 161][..])]), None);
    }

    #[test]
    #[should_panic(expected = "has no codeword")]
    fn a_difference_out_of_range_is_not_written() {
        let sections = two_bands_of_ones();
        write(100, [(&sections, &[161, 161][..])], &mut BitWriter::new());
    }

    /// C1: counting leaves nothing behind that writing would start from.
    #[test]
    fn bits_counted_are_the_bits_written() {
        let sections = two_bands_of_ones();
        let groups = [(&sections, &[220, 180][..])];
        let counted = bits(240, groups).unwrap();
        let mut w = BitWriter::new();
        write(240, groups, &mut w);
        assert_eq!(w.bit_len() as u32, counted);
        let bytes = w.into_bytes();
        let decoded = [(1, 0, 2)];
        let mut last_sf = 240;
        let sf = scale_factor_data(&mut BitReader::new(&bytes), &decoded, &mut last_sf);
        assert_eq!(sf, [220, 180]);
    }
}
