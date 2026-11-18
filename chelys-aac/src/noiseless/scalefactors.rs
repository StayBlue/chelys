// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! `scale_factor_data()` (Table 18): the scalefactor of each band in a
//! section with a spectrum codebook, as its difference from the one sent
//! before it, or from `global_gain` for the first (clause 11.3.2), in the
//! scalefactor codebook (Table A.1).

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
pub const INDEX_OFFSET: i32 = -60;

/// The codeword of a difference of scalefactors, or `None` outside the
/// codebook's range of -60 to +60 (Table 58).
pub fn codeword(difference: i32) -> Option<Codeword> {
    let index = usize::try_from(difference - INDEX_OFFSET).ok()?;
    SCALEFACTOR_CODEWORDS.get(index).copied()
}

/// The bits of the codeword of a difference of 0, which a band of zeros
/// costs in a section with a spectrum codebook (see [`Dpcm`]).
pub const REPEATED_BITS: u32 = SCALEFACTOR_CODEWORDS[(0 - INDEX_OFFSET) as usize].len;

/// The running scalefactor of `scale_factor_data()`, from which each is
/// coded as a difference (clause 11.3.2). It starts at `global_gain`, and
/// runs through the window groups in order.
///
/// Scalefactors are given for bands `0..max_sfb` of a group, of which those
/// of bands in sections with `ZERO_HCB` are not used.
///
/// Spec gap: a band of zeros in a section with a spectrum codebook sends a
/// scalefactor (Table 18), which scales nothing, and the standard leaves
/// its value to the encoder. It is sent as the scalefactor before it, a
/// difference of 0 that costs the fewest bits and leaves the next
/// difference as it would be without it; the scalefactor given for the
/// band is not used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dpcm {
    last: u8,
}

impl Dpcm {
    pub fn new(global_gain: u8) -> Self {
        Self { last: global_gain }
    }

    /// Counts the bits of a group's scalefactors, and runs on past them;
    /// or returns `None`, and does not, if a difference falls outside the
    /// codebook's range.
    ///
    /// # Panics
    ///
    /// If fewer scalefactors are given than the sections span bands: a
    /// caller's error.
    pub fn count(&mut self, sections: &Sections, scalefactors: &[u8]) -> Option<u32> {
        let mut running = *self;
        let mut bits = 0;
        for band in sections.bands_with_scalefactors() {
            bits += codeword(running.difference(sections, scalefactors, band))?.len;
        }
        *self = running;
        Some(bits)
    }

    /// Writes a group's part of `scale_factor_data()`, as
    /// [`count`](Self::count) counts it.
    ///
    /// # Panics
    ///
    /// Where [`count`](Self::count) would return `None` or panic: a caller's
    /// error.
    pub fn write(&mut self, sections: &Sections, scalefactors: &[u8], w: &mut BitWriter) {
        for band in sections.bands_with_scalefactors() {
            let difference = self.difference(sections, scalefactors, band);
            let codeword = codeword(difference)
                .unwrap_or_else(|| panic!("a difference of {difference} has no codeword"));
            w.write(codeword.bits, codeword.len);
        }
    }

    /// The difference sent for band `sfb`'s scalefactor, which becomes the
    /// running one.
    fn difference(&mut self, sections: &Sections, scalefactors: &[u8], sfb: usize) -> i32 {
        if sections.is_zero_band(sfb) {
            return 0;
        }
        let scalefactor = scalefactors[sfb];
        let difference = i32::from(scalefactor) - i32::from(self.last);
        self.last = scalefactor;
        difference
    }
}

#[cfg(test)]
mod tests {
    use super::super::decoder::{BitReader, Noise, scale_factor_data, section_data};
    use super::*;
    use crate::noiseless::codebooks::Codebook;
    use crate::sampling_frequency::SamplingFrequency;
    use crate::scalefactor_bands::{Transform, swb_offsets};

    #[test]
    fn differences_have_codewords_from_minus_60_to_60() {
        assert_eq!(codeword(-61), None);
        assert_eq!(codeword(61), None);
        // Table A.1, indexes 0, 60 and 120.
        assert_eq!(codeword(-60), Some(SCALEFACTOR_CODEWORDS[0]));
        assert_eq!(codeword(0), Some(Codeword { bits: 0, len: 1 }));
        assert_eq!(codeword(60), Some(SCALEFACTOR_CODEWORDS[120]));
        assert_eq!(REPEATED_BITS, 1);
    }

    /// Writes the sections and scalefactors of each group, then decodes
    /// them, checking the bits counted and the scalefactors decoded.
    fn round_trip(global_gain: u8, groups: &[(Sections, Vec<u8>)]) {
        let mut w = BitWriter::new();
        let mut counted = 0;
        let (mut counting, mut writing) = (Dpcm::new(global_gain), Dpcm::new(global_gain));
        for (sections, _) in groups {
            sections.write_section_data(&mut w);
            counted += sections.section_data_bits();
        }
        for (sections, scalefactors) in groups {
            counted += counting
                .count(sections, scalefactors)
                .expect("differences in range");
            writing.write(sections, scalefactors, &mut w);
        }
        assert_eq!(w.bit_len() as u32, counted);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        let decoded: Vec<_> = (groups.iter())
            .map(|(sections, _)| section_data(&mut r, sections.transform(), sections.max_sfb()))
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
            for _ in 0..1 + noise.below(4) {
                let windows = 1 + noise.below(3) as u16;
                let offsets: Vec<_> = (swb_offsets(fs, Transform::Short).iter())
                    .map(|offset| offset * windows)
                    .collect();
                let lines = noise.spectrum(&offsets, 128 * usize::from(windows));
                let sections = Sections::choose(&lines, &offsets, Transform::Short).unwrap();
                let mut scalefactors: Vec<u8> =
                    (1..offsets.len()).map(|_| noise.below(256) as u8).collect();
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

    #[test]
    fn a_difference_out_of_range_is_counted_as_none() {
        let offsets = swb_offsets(SamplingFrequency::Hz48000, Transform::Long);
        let mut lines = vec![0; 1024];
        lines[0] = 1;
        lines[4] = 1;
        let sections = Sections::choose(&lines, &offsets[..=2], Transform::Long).unwrap();
        let mut dpcm = Dpcm::new(100);
        assert_eq!(
            dpcm.count(&sections, &[160, 220]),
            Some(2 * codeword(60).unwrap().len)
        );
        assert_eq!(dpcm, Dpcm::new(220));
        let mut dpcm = Dpcm::new(100);
        assert_eq!(dpcm.count(&sections, &[161, 161]), None);
        // The running scalefactor is left as it was.
        assert_eq!(dpcm, Dpcm::new(100));
    }
}
