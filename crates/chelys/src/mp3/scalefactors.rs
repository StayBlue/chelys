//! Layer III scalefactors: their meaning in the requantization formula
//! (clause 2.4.3.4.7.1), their transmission with `scalefac_compress` (clause
//! 2.4.2.7), and preemphasis (Table B.6). Mixed blocks are not used.

use crate::bitstream::BitWriter;
use crate::mp3::BlockType;
use crate::mp3::scalefactor_bands::{LONG_BANDS, SHORT_BANDS, SHORT_WINDOWS};

/// Long scalefactor bands 0 to 10 are sent with `slen1` bits, 11 to 20 with
/// `slen2`; short bands 0 to 5 with `slen1`, 6 to 11 with `slen2` (clause
/// 2.4.2.7).
const SLEN1_LONG: usize = 11;
const SLEN1_SHORT: usize = 6 * SHORT_WINDOWS;

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

/// The scalefactors of one granule of one channel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Scalefac {
    /// `scalefac_l[sfb]`, for block types 0, 1 and 3.
    Long([u8; LONG_BANDS]),
    /// `scalefac_s[sfb][window]`, for short blocks.
    Short([[u8; SHORT_WINDOWS]; SHORT_BANDS]),
}

/// The scalefactors of one granule of one channel and the flags that go
/// with them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scalefactors {
    pub scalefac: Scalefac,
    /// Never set for short blocks (clause 2.4.2.7).
    pub preflag: bool,
    pub scalefac_scale: bool,
}

impl Default for Scalefactors {
    fn default() -> Self {
        Self::new(BlockType::Normal)
    }
}

impl Scalefactors {
    /// Zero scalefactors for a granule of `block_type`.
    pub fn new(block_type: BlockType) -> Self {
        let scalefac = match block_type {
            BlockType::Short => Scalefac::Short([[0; SHORT_WINDOWS]; SHORT_BANDS]),
            _ => Scalefac::Long([0; LONG_BANDS]),
        };
        Self {
            scalefac,
            preflag: false,
            scalefac_scale: false,
        }
    }

    /// The scalefactors in the order they are sent: `scalefac_l[sfb]` at
    /// `sfb`, or `scalefac_s[sfb][window]` at `3 * sfb + window`, the order
    /// of [`coding_bands`](crate::mp3::scalefactor_bands::coding_bands).
    pub fn values(&self) -> &[u8] {
        match &self.scalefac {
            Scalefac::Long(scalefac) => scalefac,
            Scalefac::Short(scalefac) => scalefac.as_flattened(),
        }
    }

    pub fn values_mut(&mut self) -> &mut [u8] {
        match &mut self.scalefac {
            Scalefac::Long(scalefac) => scalefac,
            Scalefac::Short(scalefac) => scalefac.as_flattened_mut(),
        }
    }

    /// How many of [`values`](Self::values) are sent with `slen1`.
    fn slen1_count(&self) -> usize {
        match self.scalefac {
            Scalefac::Long(_) => SLEN1_LONG,
            Scalefac::Short(_) => SLEN1_SHORT,
        }
    }

    /// The largest value that can be sent for scalefactor `index` of
    /// [`values`](Self::values): 15 for those sent with `slen1`, and 7 for
    /// those sent with `slen2`, the widths of the largest `slen1` and `slen2`
    /// (clause C.1.5.4.3.6).
    ///
    /// Spec reading: for short blocks clause C.1.5.4.3.6 gives the limit of
    /// 15 to "scalefactors 0 through 18". `slen1` covers short bands 0 to 5
    /// (clause 2.4.2.7), whose scalefactors are 0 through 17 in the order
    /// sent, so that is the range used: scalefactor 18 is sent with `slen2`,
    /// at most 3 bits, and could not hold 15. The clause is one off for mixed
    /// blocks in the same way, giving "0 through 17" where `slen1` covers 17
    /// scalefactors, 0 through 16.
    pub fn limit(&self, index: usize) -> u8 {
        if index < self.slen1_count() { 15 } else { 7 }
    }

    /// `scalefac_multiplier` (clause 2.4.2.7).
    fn multiplier(&self) -> f64 {
        if self.scalefac_scale { 1.0 } else { 0.5 }
    }

    /// The factor by which the decoder divides the lines of scalefactor
    /// `index` of [`values`](Self::values),
    /// `2^(scalefac_multiplier * (scalefac_l + preflag * pretab))` for long
    /// blocks or `2^(scalefac_multiplier * scalefac_s)` for short ones (clause
    /// 2.4.3.4.7.1), and by which the encoder amplifies them before
    /// quantisation. `subblock_gain` is always 0 in this encoder.
    pub fn amplification(&self, index: usize) -> f64 {
        let pretab = match self.scalefac {
            Scalefac::Long(_) if self.preflag => PRETAB[index],
            _ => 0,
        };
        let steps = f64::from(self.values()[index] + pretab);
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
        let (low, high) = self.values().split_at(self.slen1_count());
        let (need1, need2) = (bits(low), bits(high));
        (0..16u8)
            .filter(|&c| {
                let (slen1, slen2) = SCALEFAC_COMPRESS[usize::from(c)];
                slen1 >= need1 && slen2 >= need2
            })
            .min_by_key(|&c| self.part2_length(c))
    }

    /// Writes the scalefactors of a granule without `scfsi` (clause 2.4.1.7)
    /// with the field widths of `scalefac_compress`.
    ///
    /// # Panics
    ///
    /// If a scalefactor does not fit its field.
    pub fn write(&self, scalefac_compress: u8, w: &mut BitWriter) {
        let (slen1, slen2) = SCALEFAC_COMPRESS[usize::from(scalefac_compress)];
        for (index, &scalefac) in self.values().iter().enumerate() {
            let slen = if index < self.slen1_count() {
                slen1
            } else {
                slen2
            };
            assert!(
                u32::from(scalefac) >> slen == 0,
                "scalefactor {scalefac} at {index} does not fit in {slen} bits"
            );
            w.write(u32::from(scalefac), slen);
        }
    }

    /// `part2_length` without `scfsi`: the scalefactors sent with `slen1`
    /// times `slen1` plus the rest times `slen2`, `11 * slen1 + 10 * slen2`
    /// for long blocks and `18 * slen1 + 18 * slen2` for short ones (clause
    /// 2.4.2.7).
    pub fn part2_length(&self, scalefac_compress: u8) -> usize {
        let (slen1, slen2) = SCALEFAC_COMPRESS[usize::from(scalefac_compress)];
        let count1 = self.slen1_count();
        let count2 = self.values().len() - count1;
        count1 * slen1 as usize + count2 * slen2 as usize
    }
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
        scalefactors.values_mut()[3] = 5;
        // slen1 = 3, slen2 = 0.
        assert_eq!(scalefactors.compress(), Some(4));
        scalefactors.values_mut()[12] = 1;
        // (3, 1) needs 43 bits.
        assert_eq!(scalefactors.compress(), Some(11));
        assert_eq!(scalefactors.part2_length(11), 43);
        scalefactors.values_mut()[0] = 15;
        scalefactors.values_mut()[20] = 7;
        assert_eq!(scalefactors.compress(), Some(15));
        scalefactors.values_mut()[20] = 8;
        assert_eq!(scalefactors.compress(), None);
    }

    #[test]
    fn writes_each_band_with_its_width() {
        let mut scalefactors = Scalefactors::default();
        scalefactors.values_mut()[0] = 0b101;
        scalefactors.values_mut()[20] = 0b11;
        let compress = scalefactors.compress().unwrap();
        assert_eq!(SCALEFAC_COMPRESS[usize::from(compress)], (3, 2));
        let mut w = BitWriter::new();
        scalefactors.write(compress, &mut w);
        assert_eq!(w.bit_len(), scalefactors.part2_length(compress));
        let bits: String = w.into_bytes().iter().map(|b| format!("{b:08b}")).collect();
        assert_eq!(&bits[..3], "101");
        assert_eq!(&bits[3 * 11 + 2 * 9..3 * 11 + 2 * 10], "11");
    }

    /// With `scalefac_scale` set each step is a factor of 2, otherwise of
    /// the square root of 2, and preemphasis adds `pretab` steps.
    #[test]
    fn amplification_follows_the_requantization_formula() {
        let mut scalefactors = Scalefactors::default();
        scalefactors.values_mut()[18] = 2;
        assert_eq!(scalefactors.amplification(18), 2.0);
        scalefactors.preflag = true;
        assert!((scalefactors.amplification(18) - 2f64.powf(2.5)).abs() < 1e-12);
        scalefactors.scalefac_scale = true;
        assert_eq!(scalefactors.amplification(18), 32.0);
        assert_eq!(scalefactors.amplification(0), 1.0);
    }

    /// Short blocks send 36 scalefactors, `scalefac_s[sfb][window]` with
    /// the windows of a band together, bands 0 to 5 with `slen1`, and
    /// have no preemphasis.
    #[test]
    fn short_blocks_send_three_scalefactors_per_band() {
        let mut scalefactors = Scalefactors::new(BlockType::Short);
        assert_eq!(scalefactors.values().len(), 36);
        assert_eq!((scalefactors.limit(17), scalefactors.limit(18)), (15, 7));
        if let Scalefac::Short(scalefac_s) = &mut scalefactors.scalefac {
            scalefac_s[5][2] = 15;
            scalefac_s[6][0] = 3;
        }
        assert_eq!(scalefactors.values()[17], 15);
        assert_eq!(scalefactors.values()[18], 3);
        let compress = scalefactors.compress().unwrap();
        assert_eq!(SCALEFAC_COMPRESS[usize::from(compress)], (4, 2));
        assert_eq!(scalefactors.part2_length(compress), 18 * 4 + 18 * 2);
        let mut w = BitWriter::new();
        scalefactors.write(compress, &mut w);
        assert_eq!(w.bit_len(), 108);

        scalefactors.preflag = true;
        assert_eq!(scalefactors.amplification(18), 2f64.powf(1.5));
    }
}
