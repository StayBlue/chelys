// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Layer III scalefactors: their meaning in the requantization formula
//! (clause 2.4.3.4.7.1), their transmission with `scalefac_compress` (clause
//! 2.4.2.7, and clause 2.4.3.2 of ISO/IEC 13818-3 for the lower sampling
//! frequencies), and preemphasis (Table B.6). Mixed blocks are not used.

use crate::bitstream::BitWriter;
use crate::mp3::BlockType;
use crate::mp3::header::Version;
use crate::mp3::scalefactor_bands::{LONG_BANDS, SHORT_BANDS, SHORT_WINDOWS};
use crate::mp3::side_info::SCFSI_BANDS;

/// The long scalefactor bands of each scfsi band: `SCFSI_STARTS[band]` up to
/// `SCFSI_STARTS[band + 1]` (clause 2.4.2.7).
pub const SCFSI_STARTS: [usize; SCFSI_BANDS + 1] = [0, 6, 11, 16, 21];

/// The scfsi band of long scalefactor band `sb`.
pub fn scfsi_band(sb: usize) -> usize {
    SCFSI_STARTS[1..]
        .iter()
        .position(|&end| sb < end)
        .expect("a long scalefactor band")
}

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

/// The values of the 9-bit `scalefac_compress` of the lower sampling
/// frequencies.
const LSF_SCALEFAC_COMPRESS: u16 = 512;

/// How the scalefactors are sent at the lower sampling frequencies for a
/// value of `scalefac_compress`: in four partitions of `nr_of_sfb`
/// scalefactors in the order sent, each scalefactor of a partition in
/// `slen` bits, and whether `preflag` is set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LsfPartitions {
    slen: [u32; 4],
    nr_of_sfb: [usize; 4],
    preflag: bool,
}

impl LsfPartitions {
    /// Decodes `scalefac_compress` as clause 2.4.3.2 of ISO/IEC 13818-3
    /// prints it for channels not coded in intensity stereo, with the rows
    /// for block types 0, 1 and 3 or for block type 2 without mixed blocks.
    fn decode(scalefac_compress: u16, short: bool) -> Self {
        let sc = u32::from(scalefac_compress);
        let (slen, preflag, long_sfb, short_sfb) = if sc < 400 {
            let slen = [(sc >> 4) / 5, (sc >> 4) % 5, (sc % 16) >> 2, sc % 4];
            (slen, false, [6, 5, 5, 5], [9, 9, 9, 9])
        } else if sc < 500 {
            let sc = sc - 400;
            let slen = [(sc >> 2) / 5, (sc >> 2) % 5, sc % 4, 0];
            (slen, false, [6, 5, 7, 3], [9, 9, 12, 6])
        } else {
            assert!(sc < 512, "scalefac_compress {sc} has more than 9 bits");
            let sc = sc - 500;
            ([sc / 3, sc % 3, 0, 0], true, [11, 10, 0, 0], [18, 18, 0, 0])
        };
        Self {
            slen,
            nr_of_sfb: if short { short_sfb } else { long_sfb },
            preflag,
        }
    }

    /// The `slen` of each of `count` scalefactors in the order sent.
    fn widths(self, count: usize) -> impl Iterator<Item = u32> {
        let widths = (self.slen.into_iter().zip(self.nr_of_sfb))
            .flat_map(|(slen, n)| std::iter::repeat_n(slen, n));
        assert_eq!(
            self.nr_of_sfb.iter().sum::<usize>(),
            count,
            "the partitions hold every scalefactor"
        );
        widths
    }
}

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
    /// The version whose `scalefac_compress` sends them.
    pub version: Version,
    pub scalefac: Scalefac,
    /// Never set for short blocks (clause 2.4.2.7).
    pub preflag: bool,
    pub scalefac_scale: bool,
    /// The scfsi bands whose scalefactors the second granule of a frame takes
    /// over from the first, and so does not send (clause 2.4.2.7). Always
    /// clear in the first granule, for short blocks, and at the lower
    /// sampling frequencies, which have no scfsi.
    pub scfsi: [bool; SCFSI_BANDS],
}

impl Default for Scalefactors {
    fn default() -> Self {
        Self::new(BlockType::Normal, Version::Mpeg1)
    }
}

impl Scalefactors {
    /// Zero scalefactors for a granule of `block_type`, sent as `version`
    /// sends them.
    pub fn new(block_type: BlockType, version: Version) -> Self {
        let scalefac = match block_type {
            BlockType::Short => Scalefac::Short([[0; SHORT_WINDOWS]; SHORT_BANDS]),
            _ => Scalefac::Long([0; LONG_BANDS]),
        };
        Self {
            version,
            scalefac,
            preflag: false,
            scalefac_scale: false,
            scfsi: [false; SCFSI_BANDS],
        }
    }

    /// Whether scalefactor `index` of [`values`](Self::values) is sent, rather
    /// than taken over from the first granule.
    pub fn sent(&self, index: usize) -> bool {
        match self.scalefac {
            Scalefac::Long(_) => !self.scfsi[scfsi_band(index)],
            Scalefac::Short(_) => true,
        }
    }

    /// The scalefactors sent with `slen1` and with `slen2`.
    fn sent_values(&self) -> (Vec<u8>, Vec<u8>) {
        let sent = (self.values().iter().enumerate()).filter(|&(index, _)| self.sent(index));
        let (low, high): (Vec<_>, Vec<_>) =
            sent.partition(|&(index, _)| index < self.slen1_count());
        let values = |pairs: Vec<(usize, &u8)>| pairs.into_iter().map(|(_, &v)| v).collect();
        (values(low), values(high))
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
    ///
    /// Spec gap: Annex C of ISO/IEC 13818-3 gives no limits for the lower
    /// sampling frequencies. They are taken, in the same way, as the widest
    /// fields a `scalefac_compress` with the same `preflag` gives each
    /// scalefactor (clause 2.4.3.2 of ISO/IEC 13818-3). Without `preflag`
    /// these are 4 bits for the first two partitions of values below 400,
    /// which cover the same scalefactors as `slen1` above, and 3 bits for the
    /// rest, so the limits are the same; with `preflag`, values 500 and up
    /// give 3 bits to the scalefactors `slen1` covers above and 2 to the
    /// rest, limits of 7 and 3.
    pub fn limit(&self, index: usize) -> u8 {
        let first = index < self.slen1_count();
        match (self.version, self.preflag, first) {
            (Version::Mpeg2, true, true) => 7,
            (Version::Mpeg2, true, false) => 3,
            (_, _, true) => 15,
            (_, _, false) => 7,
        }
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

    /// The `scalefac_compress` whose field widths hold the scalefactors sent
    /// in the fewest bits, or `None` if one is above its
    /// [`limit`](Self::limit). At the lower sampling frequencies it also
    /// sets `preflag` as the scalefactors have it.
    pub fn compress(&self) -> Option<u16> {
        let fits = |c: u16| match self.version {
            Version::Mpeg1 => {
                let (slen1, slen2) = SCALEFAC_COMPRESS[usize::from(c)];
                let (low, high) = self.sent_values();
                low.iter().all(|&v| fits_in(v, slen1)) && high.iter().all(|&v| fits_in(v, slen2))
            }
            Version::Mpeg2 => {
                let partitions = self.lsf_partitions(c);
                partitions.preflag == self.preflag
                    && (self
                        .values()
                        .iter()
                        .zip(partitions.widths(self.values().len())))
                    .all(|(&v, slen)| fits_in(v, slen))
            }
        };
        let values = match self.version {
            Version::Mpeg1 => SCALEFAC_COMPRESS.len() as u16,
            Version::Mpeg2 => LSF_SCALEFAC_COMPRESS,
        };
        (0..values)
            .filter(|&c| fits(c))
            .min_by_key(|&c| self.part2_length(c))
    }

    /// How `scalefac_compress` sends these scalefactors at the lower sampling
    /// frequencies.
    fn lsf_partitions(&self, scalefac_compress: u16) -> LsfPartitions {
        assert_eq!(self.version, Version::Mpeg2);
        assert!(
            !self.scfsi.contains(&true),
            "the lower sampling frequencies have no scfsi"
        );
        let short = matches!(self.scalefac, Scalefac::Short(_));
        LsfPartitions::decode(scalefac_compress, short)
    }

    /// The width of each scalefactor of [`values`](Self::values) sent with
    /// `scalefac_compress`, or `None` for those not sent.
    fn widths(&self, scalefac_compress: u16) -> Vec<Option<u32>> {
        match self.version {
            Version::Mpeg1 => {
                let (slen1, slen2) = SCALEFAC_COMPRESS[usize::from(scalefac_compress)];
                (0..self.values().len())
                    .map(|index| {
                        let slen = if index < self.slen1_count() {
                            slen1
                        } else {
                            slen2
                        };
                        self.sent(index).then_some(slen)
                    })
                    .collect()
            }
            Version::Mpeg2 => self
                .lsf_partitions(scalefac_compress)
                .widths(self.values().len())
                .map(Some)
                .collect(),
        }
    }

    /// Writes the scalefactors sent (clause 2.4.1.7) with the field widths of
    /// `scalefac_compress`.
    ///
    /// # Panics
    ///
    /// If a scalefactor sent does not fit its field.
    pub fn write(&self, scalefac_compress: u16, w: &mut BitWriter) {
        let widths = self.widths(scalefac_compress);
        for (index, (&scalefac, slen)) in self.values().iter().zip(widths).enumerate() {
            let Some(slen) = slen else { continue };
            assert!(
                fits_in(scalefac, slen),
                "scalefactor {scalefac} at {index} does not fit in {slen} bits"
            );
            w.write(u32::from(scalefac), slen);
        }
    }

    /// `part2_length`: the sum of the widths of the scalefactors sent. Without
    /// `scfsi` this is `11 * slen1 + 10 * slen2` for long blocks and
    /// `18 * slen1 + 18 * slen2` for short ones (clause 2.4.2.7), and at the
    /// lower sampling frequencies the sum of `nr_of_sfb * slen` over the
    /// partitions (clause 2.4.3.2 of ISO/IEC 13818-3).
    pub fn part2_length(&self, scalefac_compress: u16) -> usize {
        (self.widths(scalefac_compress).into_iter().flatten())
            .map(|slen| slen as usize)
            .sum()
    }
}

/// Whether `value` fits in `bits` bits.
fn fits_in(value: u8, bits: u32) -> bool {
    u32::from(value) >> bits == 0
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

    /// With scfsi, the scfsi bands' scalefactors are neither sent nor
    /// counted, nor do they limit the field widths.
    #[test]
    fn scfsi_bands_are_not_sent() {
        let mut scalefactors = Scalefactors::default();
        scalefactors.values_mut()[2] = 15;
        scalefactors.values_mut()[12] = 1;
        scalefactors.scfsi = [true, false, false, false];
        // Bands 0 to 5 are not sent: slen1 needs no bits for bands 6 to 10.
        let compress = scalefactors.compress().unwrap();
        assert_eq!(SCALEFAC_COMPRESS[usize::from(compress)], (0, 1));
        assert_eq!(scalefactors.part2_length(compress), 10);
        let mut w = BitWriter::new();
        scalefactors.write(compress, &mut w);
        assert_eq!(w.bit_len(), 10);
        assert_eq!((scfsi_band(5), scfsi_band(6), scfsi_band(20)), (0, 1, 3));
    }

    /// Short blocks send 36 scalefactors, `scalefac_s[sfb][window]` with
    /// the windows of a band together, bands 0 to 5 with `slen1`, and
    /// have no preemphasis.
    #[test]
    fn short_blocks_send_three_scalefactors_per_band() {
        let mut scalefactors = Scalefactors::new(BlockType::Short, Version::Mpeg1);
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
