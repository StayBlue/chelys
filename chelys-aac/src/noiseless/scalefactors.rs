// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! `scale_factor_data()` (Table 18): the scalefactor of each band in a
//! section whose codebook sends one, as its difference from the one sent
//! before it, or from `global_gain` for the first (clause 11.3.2), in the
//! scalefactor codebook (Table A.1), through the window groups in order;
//! and among them, in the bands coded in intensity stereo, the intensity
//! stereo position of each, as its difference from the position sent
//! before it, or from 0 for the first (clause 12.2.3), in the same
//! codebook; and in the bands coded by perceptual noise substitution, the
//! noise energy of each, as its difference from the energy sent before it
//! in the same codebook, but for the first, sent in 9 bits from
//! `global_gain` (ISO/IEC 14496-3, clause 4.6.13.3 and Table 4.55).
//!
//! The scalefactors, positions and noise energies of a group are given for
//! its bands `0..max_sfb`, of which those of bands that send none are not
//! used; a group without bands in intensity stereo may give no positions,
//! and one without bands of noise no energies.
//!
//! Spec gap: a band of zeros in a section with a spectrum codebook sends a
//! scalefactor (Table 18), which scales nothing, and the standard leaves
//! its value to the encoder. It is sent as the scalefactor before it, a
//! difference of 0 that costs the fewest bits and leaves the next
//! difference as it would be without it; the scalefactor given for the
//! band is not used.

use chelys_core::bitstream::BitWriter;

use super::codebooks::{Codeword, SCALEFACTOR_CODEWORDS};
use super::sections::{BandValue, Sections};

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

/// `NOISE_OFFSET` (ISO/IEC 14496-3, clause 4.6.13.3).
const NOISE_OFFSET: i32 = 90;

/// The bits of the first noise energy sent, `dpcm_noise_nrg` where
/// `noise_pcm_flag` is set (ISO/IEC 14496-3, Table 4.55).
const NOISE_PCM_BITS: u32 = 9;

/// The noise energy from which the first one sent is counted, less
/// `global_gain`: the decoder starts its noise energies at `global_gain -
/// NOISE_OFFSET - 256` and adds the first one sent, unsigned (ISO/IEC
/// 14496-3, clause 4.6.13.3).
const NOISE_PCM_START: i32 = -NOISE_OFFSET - 256;

/// The `global_gain` of a channel that sends no scalefactor, whose first
/// noise energy sent is `energy`: the one that sends it in the middle of
/// the range of its 9 bits.
///
/// Spec gap: a channel whose bands send no scalefactor leaves `global_gain`
/// to the encoder, as the start of its noise energies alone (ISO/IEC
/// 14496-3, clause 4.6.13.3).
pub fn noise_global_gain(energy: i16) -> u8 {
    let centred = i32::from(energy) - NOISE_PCM_START - (1 << (NOISE_PCM_BITS - 1));
    centred.clamp(0, 255) as u8
}

/// A window group as `scale_factor_data()` sends it: its sections, and the
/// scalefactors, intensity stereo positions and noise energies of its
/// bands.
#[derive(Clone, Copy, Debug)]
pub struct Group<'a> {
    pub sections: &'a Sections,
    pub scalefactors: &'a [u8],
    pub positions: &'a [i8],
    /// `noise_nrg[g][sfb]`: each band's noise energy, in steps of 1.5 dB
    /// (ISO/IEC 14496-3, clause 4.6.13.3).
    pub noise_energies: &'a [i16],
}

/// A field of `scale_factor_data()`.
#[derive(Clone, Copy)]
enum Field {
    /// A difference in the scalefactor codebook.
    Difference(i32),
    /// The first noise energy, in [`NOISE_PCM_BITS`].
    NoisePcm(u32),
}

/// The bits of `scale_factor_data()` for the window groups `groups`, each
/// with its sections, scalefactors, positions and noise energies, from
/// `global_gain`; or `None` if a difference falls outside the codebook's
/// range, or the first noise energy outside the range of its 9 bits.
///
/// # Panics
///
/// If a group has fewer scalefactors, or positions or noise energies where
/// it sends any, than its sections span bands: a caller's error.
pub fn bits<'a>(global_gain: u8, groups: impl IntoIterator<Item = Group<'a>>) -> Option<u32> {
    let mut bits = 0;
    for_each_field(global_gain, groups, |field| {
        bits += match field {
            Field::Difference(difference) => codeword(difference)?.len,
            Field::NoisePcm(_) => NOISE_PCM_BITS,
        };
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
    groups: impl IntoIterator<Item = Group<'a>> + Clone,
    w: &mut BitWriter,
) {
    assert!(
        bits(global_gain, groups.clone()).is_some(),
        "a difference of scalefactors has no codeword"
    );
    for_each_field(global_gain, groups, |field| {
        match field {
            Field::Difference(difference) => {
                let codeword = codeword(difference)?;
                w.write(codeword.bits, codeword.len);
            }
            Field::NoisePcm(value) => w.write(value, NOISE_PCM_BITS),
        }
        Some(())
    })
    .expect("differences with codewords");
}

/// Calls `f` with each field sent, from the running scalefactor that
/// starts at `global_gain` (clause 11.3.2), the running position that
/// starts at 0 (clause 12.2.3) and the running noise energy that starts at
/// `global_gain - NOISE_OFFSET - 256` (ISO/IEC 14496-3, clause 4.6.13.3),
/// until it returns `None`; or returns `None` itself where the first noise
/// energy lies beyond the reach of its 9 bits.
fn for_each_field<'a>(
    global_gain: u8,
    groups: impl IntoIterator<Item = Group<'a>>,
    mut f: impl FnMut(Field) -> Option<()>,
) -> Option<()> {
    let mut last = global_gain;
    let mut last_position = 0;
    let mut last_noise: Option<i32> = None;
    for group in groups {
        let sections = group.sections;
        for (sfb, value) in sections.bands_with_values() {
            let field = match value {
                BandValue::Position => {
                    let position = i32::from(group.positions[sfb]);
                    let difference = position - last_position;
                    last_position = position;
                    Field::Difference(difference)
                }
                BandValue::NoiseEnergy => {
                    let energy = i32::from(group.noise_energies[sfb]);
                    let field = match last_noise {
                        Some(last) => Field::Difference(energy - last),
                        None => {
                            let start = i32::from(global_gain) + NOISE_PCM_START;
                            let value = u32::try_from(energy - start).ok()?;
                            if value >= 1 << NOISE_PCM_BITS {
                                return None;
                            }
                            Field::NoisePcm(value)
                        }
                    };
                    last_noise = Some(energy);
                    field
                }
                BandValue::Scalefactor if sections.is_zero_band(sfb) => Field::Difference(0),
                BandValue::Scalefactor => {
                    let difference = i32::from(group.scalefactors[sfb]) - i32::from(last);
                    last = group.scalefactors[sfb];
                    Field::Difference(difference)
                }
            };
            f(field)?;
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::super::codebooks::Codebook;
    use super::super::decoder::{BitReader, Noise, Running, scale_factor_data, section_data};
    use super::super::sections::PseudoBands;
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
        let groups_ref = || groups.iter().map(|(s, sf)| group(s, sf));
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
        let mut running = Running::new(i32::from(global_gain));
        let mut expected_last = i32::from(global_gain);
        for ((sections, scalefactors), decoded) in groups.iter().zip(&decoded) {
            let sf = scale_factor_data(&mut r, decoded, &mut running, false, false);
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

    /// A group of `sections` and `scalefactors`, without positions or noise
    /// energies.
    fn group<'a>(sections: &'a Sections, scalefactors: &'a [u8]) -> Group<'a> {
        Group {
            sections,
            scalefactors,
            positions: &[],
            noise_energies: &[],
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
        assert_eq!(
            bits(100, [group(&sections, &[160, 220][..])]),
            Some(2 * sixty)
        );
        assert_eq!(bits(100, [group(&sections, &[161, 161][..])]), None);
    }

    #[test]
    #[should_panic(expected = "has no codeword")]
    fn a_difference_out_of_range_is_not_written() {
        let sections = two_bands_of_ones();
        write(
            100,
            [group(&sections, &[161, 161][..])],
            &mut BitWriter::new(),
        );
    }

    /// C1: counting leaves nothing behind that writing would start from.
    #[test]
    fn bits_counted_are_the_bits_written() {
        let sections = two_bands_of_ones();
        let groups = [group(&sections, &[220, 180][..])];
        let counted = bits(240, groups).unwrap();
        let mut w = BitWriter::new();
        write(240, groups, &mut w);
        assert_eq!(w.bit_len() as u32, counted);
        let bytes = w.into_bytes();
        let decoded = [(1, 0, 2)];
        let sf = scale_factor_data(
            &mut BitReader::new(&bytes),
            &decoded,
            &mut Running::new(240),
            false,
            false,
        );
        assert_eq!(sf, [220, 180]);
    }

    /// Sections of a long window at 48 kHz whose bands 0 and 4 have a line
    /// of 1 each, and bands 2, 3 and 6 noise.
    fn sections_with_noise() -> Sections {
        let mut lines = vec![0; 1024];
        lines[0] = 1;
        lines[16] = 1;
        let mut pseudo = PseudoBands::NONE;
        for sfb in [2, 3, 6] {
            pseudo.set(sfb, Some(Codebook::NOISE));
        }
        let bands = GroupBands::long(SamplingFrequency::Hz48000);
        Sections::count_with_pseudo(&lines, bands, 7, pseudo)
            .unwrap()
            .choose()
    }

    #[test]
    fn noise_energies_follow_their_first_sent_in_9_bits() {
        let sections = sections_with_noise();
        assert_eq!(
            (sections.iter())
                .map(|s| (s.codebook.number(), s.start, s.end))
                .filter(|&(cb, _, _)| cb == 13)
                .collect::<Vec<_>>(),
            [(13, 2, 4), (13, 6, 7)]
        );
        // From global_gain 100, the energies start at 100 - 90 - 256 =
        // -246, so 30 is sent as 276.
        let scalefactors = [100, 0, 0, 0, 104, 0, 0];
        let energies = [0, 0, 30, 20, 0, 0, 79];
        let group = Group {
            sections: &sections,
            scalefactors: &scalefactors,
            positions: &[],
            noise_energies: &energies,
        };
        let counted = bits(100, [group]).unwrap();
        let mut w = BitWriter::new();
        write(100, [group], &mut w);
        assert_eq!(w.bit_len() as u32, counted);
        // Scalefactors 0 and +4, the first energy in 9 bits, then -10 and
        // +59, and a difference of 0 for each band of zeros in a section
        // with a spectrum codebook.
        let codewords = [0, 4, -10, 59].map(|d| codeword(d).unwrap().len);
        let repeated = (sections.bands_with_scalefactors())
            .filter(|&sfb| sections.is_zero_band(sfb))
            .count() as u32;
        assert_eq!(
            counted,
            codewords.iter().sum::<u32>() + 9 + repeated * REPEATED_BITS
        );
        let bytes = w.into_bytes();
        let decoded: Vec<_> = (sections.iter())
            .map(|s| (s.codebook.number(), s.start, s.end))
            .collect();
        let values = scale_factor_data(
            &mut BitReader::new(&bytes),
            &decoded,
            &mut Running::new(100),
            false,
            true,
        );
        // A band of zeros repeats the scalefactor before it, or has 0.
        let zero = |sfb: usize, before: i32| {
            if sections.bands_with_scalefactors().any(|b| b == sfb) {
                before
            } else {
                0
            }
        };
        assert_eq!(values, [100, zero(1, 100), 30, 20, 104, zero(5, 104), 79]);
    }

    #[test]
    fn a_first_noise_energy_beyond_9_bits_is_counted_as_none() {
        let sections = sections_with_noise();
        let scalefactors = [100, 0, 0, 0, 100, 0, 0];
        let group = |first: i16| {
            let energies = [0, 0, first, first, 0, 0, first];
            bits(
                100,
                [Group {
                    sections: &sections,
                    scalefactors: &scalefactors,
                    positions: &[],
                    noise_energies: &energies,
                }],
            )
        };
        // 0 to 511 from -246.
        assert!(group(-246).is_some() && group(265).is_some());
        assert_eq!(group(-247), None);
        assert_eq!(group(266), None);
    }

    #[test]
    fn a_global_gain_for_noise_alone_sends_it_mid_range() {
        assert_eq!(noise_global_gain(30), 120);
        assert_eq!(noise_global_gain(-400), 0);
        assert_eq!(noise_global_gain(400), 255);
    }
}
