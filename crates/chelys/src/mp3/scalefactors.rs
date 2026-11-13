//! Layer III scalefactors for long blocks: their meaning in the
//! requantization formula (clause 2.4.3.4.7.1), their transmission with
//! `scalefac_compress` (clause 2.4.2.7), and preemphasis (Table B.6).

use crate::bitstream::BitWriter;
use crate::mp3::scalefactor_bands::LONG_BANDS;

/// Scalefactor bands 0 to 10 are sent with `slen1` bits, 11 to 20 with
/// `slen2` (clause 2.4.2.7).
const SLEN1_BANDS: usize = 11;

/// `(slen1, slen2)` for each `scalefac_compress` (clause 2.4.2.7).
const SCALEFAC_COMPRESS: [(u32, u32); 16] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (3, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 1),
    (3, 2),
    (3, 3),
    (4, 2),
    (4, 3),
];

/// Table B.6: `pretab`, added to the scalefactors when `preflag` is set.
pub const PRETAB: [u8; LONG_BANDS] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 2,
];

/// The scalefactors of one granule of one channel with long blocks.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Scalefactors {
    /// `scalefac_l` for each scalefactor band.
    pub scalefac_l: [u8; LONG_BANDS],
    pub preflag: bool,
    pub scalefac_scale: bool,
}

impl Scalefactors {
    /// The largest scalefactor that can be sent for `band`: 15 in bands 0 to
    /// 10 and 7 above, the widths of the largest `slen1` and `slen2` (clause
    /// C.1.5.4.3.6).
    pub fn limit(band: usize) -> u8 {
        if band < SLEN1_BANDS { 15 } else { 7 }
    }

    /// `scalefac_multiplier` (clause 2.4.2.7).
    fn multiplier(&self) -> f64 {
        if self.scalefac_scale { 1.0 } else { 0.5 }
    }

    /// The factor by which the decoder divides the lines of `band`,
    /// `2^(scalefac_multiplier * (scalefac_l + preflag * pretab))` (clause
    /// 2.4.3.4.7.1), and by which the encoder amplifies them before
    /// quantisation.
    pub fn amplification(&self, band: usize) -> f64 {
        let pretab = if self.preflag { PRETAB[band] } else { 0 };
        let steps = f64::from(self.scalefac_l[band] + pretab);
        (self.multiplier() * steps).exp2()
    }

    /// The `scalefac_compress` whose field widths hold the scalefactors in
    /// the fewest bits, or `None` if a scalefactor is above its
    /// [`limit`](Self::limit).
    pub fn compress(&self) -> Option<u8> {
        let bits = |values: &[u8]| {
            let max = values.iter().copied().max().unwrap_or(0);
            u8::BITS - max.leading_zeros()
        };
        let (low, high) = self.scalefac_l.split_at(SLEN1_BANDS);
        let (need1, need2) = (bits(low), bits(high));
        (0..16u8)
            .filter(|&c| {
                let (slen1, slen2) = SCALEFAC_COMPRESS[usize::from(c)];
                slen1 >= need1 && slen2 >= need2
            })
            .min_by_key(|&c| part2_length(c))
    }

    /// Writes the scalefactors of a granule without `scfsi` (clause 2.4.1.7)
    /// with the field widths of `scalefac_compress`.
    ///
    /// # Panics
    ///
    /// If a scalefactor does not fit its field.
    pub fn write(&self, scalefac_compress: u8, w: &mut BitWriter) {
        let (slen1, slen2) = SCALEFAC_COMPRESS[usize::from(scalefac_compress)];
        for (band, &scalefac) in self.scalefac_l.iter().enumerate() {
            let slen = if band < SLEN1_BANDS { slen1 } else { slen2 };
            assert!(
                u32::from(scalefac) >> slen == 0,
                "scalefactor {scalefac} of band {band} does not fit in {slen} bits"
            );
            w.write(u32::from(scalefac), slen);
        }
    }
}

/// `part2_length` for long blocks without `scfsi`, `11 * slen1 + 10 * slen2`
/// (clause 2.4.2.7).
pub fn part2_length(scalefac_compress: u8) -> usize {
    let (slen1, slen2) = SCALEFAC_COMPRESS[usize::from(scalefac_compress)];
    (SLEN1_BANDS as u32 * slen1 + (LONG_BANDS - SLEN1_BANDS) as u32 * slen2) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every pair of widths appears at most once, and the table holds no
    /// width above the 4 bits of `slen1` or the 3 of `slen2`.
    #[test]
    fn scalefac_compress_widths_are_distinct_and_bounded() {
        for (i, &(slen1, slen2)) in SCALEFAC_COMPRESS.iter().enumerate() {
            assert!(slen1 <= 4 && slen2 <= 3, "{i}");
            assert!(!SCALEFAC_COMPRESS[..i].contains(&(slen1, slen2)), "{i}");
        }
    }

    #[test]
    fn compress_picks_the_shortest_fitting_widths() {
        let mut scalefactors = Scalefactors::default();
        assert_eq!(scalefactors.compress(), Some(0));
        scalefactors.scalefac_l[3] = 5;
        // slen1 = 3, slen2 = 0.
        assert_eq!(scalefactors.compress(), Some(4));
        scalefactors.scalefac_l[12] = 1;
        // (3, 1) needs 43 bits.
        assert_eq!(scalefactors.compress(), Some(11));
        assert_eq!(part2_length(11), 43);
        scalefactors.scalefac_l[0] = 15;
        scalefactors.scalefac_l[20] = 7;
        assert_eq!(scalefactors.compress(), Some(15));
        scalefactors.scalefac_l[20] = 8;
        assert_eq!(scalefactors.compress(), None);
    }

    #[test]
    fn writes_each_band_with_its_width() {
        let mut scalefactors = Scalefactors::default();
        scalefactors.scalefac_l[0] = 0b101;
        scalefactors.scalefac_l[20] = 0b11;
        let compress = scalefactors.compress().unwrap();
        assert_eq!(SCALEFAC_COMPRESS[usize::from(compress)], (3, 2));
        let mut w = BitWriter::new();
        scalefactors.write(compress, &mut w);
        assert_eq!(w.bit_len(), part2_length(compress));
        let bits: String = w.into_bytes().iter().map(|b| format!("{b:08b}")).collect();
        assert_eq!(&bits[..3], "101");
        assert_eq!(&bits[3 * 11 + 2 * 9..3 * 11 + 2 * 10], "11");
    }

    /// With `scalefac_scale` set each step is a factor of 2, otherwise of
    /// the square root of 2, and preemphasis adds `pretab` steps.
    #[test]
    fn amplification_follows_the_requantization_formula() {
        let mut scalefactors = Scalefactors::default();
        scalefactors.scalefac_l[18] = 2;
        assert_eq!(scalefactors.amplification(18), 2.0);
        scalefactors.preflag = true;
        assert!((scalefactors.amplification(18) - 2f64.powf(2.5)).abs() < 1e-12);
        scalefactors.scalefac_scale = true;
        assert_eq!(scalefactors.amplification(18), 32.0);
        assert_eq!(scalefactors.amplification(0), 1.0);
    }
}
