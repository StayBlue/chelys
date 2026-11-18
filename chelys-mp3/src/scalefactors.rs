// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Layer III scalefactors: their meaning in the requantization formula
//! (clause 2.4.3.4.7.1), their transmission with `scalefac_compress` (clause
//! 2.4.2.7, and clause 2.4.3.2 of ISO/IEC 13818-3 for the lower sampling
//! frequencies), with the intensity stereo positions the right channel sends
//! in their place (clause 2.4.3.4.9.3), and preemphasis (Table B.6).

use crate::bitstream::BitWriter;
use crate::granule::BlockType;
use crate::header::Version;
use crate::scalefactor_bands::{LONG_BANDS, MIXED_SHORT_START, SHORT_BANDS, SHORT_WINDOWS};
use crate::side_info::SCFSI_BANDS;
use crate::stereo::INTENSITY_SCALE;

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
/// `slen2`; short bands 0 to 5 with `slen1`, 6 to 11 with `slen2`; and in a
/// mixed block its long bands and short bands 3 to 5 with `slen1`, the rest
/// with `slen2` (clause 2.4.2.7).
const SLEN1_LONG: usize = 11;
const SLEN1_SHORT: usize = 6 * SHORT_WINDOWS;
const SLEN1_MIXED_SHORT: usize = (6 - MIXED_SHORT_START) * SHORT_WINDOWS;

/// The scalefactors a mixed block sends at most: 8 long bands and short
/// bands 3 to 11 in each window (clause 2.4.2.7).
const MIXED_SCALEFACTORS: usize = 8 + (SHORT_BANDS - MIXED_SHORT_START) * SHORT_WINDOWS;

/// The rows of the tables of clause 2.4.3.2 of ISO/IEC 13818-3, which give
/// the number of scalefactors in each partition by block type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rows {
    /// Block types 0, 1 and 3.
    Long,
    /// Block type 2 without `mixed_block_flag`.
    Short,
    /// Block type 2 with `mixed_block_flag`.
    Mixed,
}

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
    /// prints it for channels not coded in intensity stereo, with `rows`.
    fn decode(scalefac_compress: u16, rows: Rows) -> Self {
        let sc = u32::from(scalefac_compress);
        let (slen, preflag, nr_of_sfb) = if sc < 400 {
            let slen = [(sc >> 4) / 5, (sc >> 4) % 5, (sc % 16) >> 2, sc % 4];
            (slen, false, [[6, 5, 5, 5], [9, 9, 9, 9], [6, 9, 9, 9]])
        } else if sc < 500 {
            let sc = sc - 400;
            let slen = [(sc >> 2) / 5, (sc >> 2) % 5, sc % 4, 0];
            (slen, false, [[6, 5, 7, 3], [9, 9, 12, 6], [6, 9, 12, 6]])
        } else {
            assert!(sc < 512, "scalefac_compress {sc} has more than 9 bits");
            let sc = sc - 500;
            let nr_of_sfb = [[11, 10, 0, 0], [18, 18, 0, 0], [15, 18, 0, 0]];
            ([sc / 3, sc % 3, 0, 0], true, nr_of_sfb)
        };
        Self {
            slen,
            nr_of_sfb: nr_of_sfb[rows as usize],
            preflag,
        }
    }

    /// Decodes `scalefac_compress` as clause 2.4.3.2 of ISO/IEC 13818-3
    /// prints it for the right channel of a frame coded in intensity stereo,
    /// `intensity_scale` in its lowest bit and `int_scalefac_compress` in the
    /// rest, with `rows`.
    fn decode_intensity(scalefac_compress: u16, rows: Rows) -> Self {
        let isc = u32::from(scalefac_compress >> 1);
        let (slen, nr_of_sfb) = if isc < 180 {
            let slen = [isc / 36, (isc % 36) / 6, (isc % 36) % 6, 0];
            (slen, [[7, 7, 7, 0], [12, 12, 12, 0], [6, 15, 12, 0]])
        } else if isc < 244 {
            let isc = isc - 180;
            let slen = [(isc % 64) >> 4, (isc % 16) >> 2, isc % 4, 0];
            (slen, [[6, 6, 6, 3], [12, 9, 9, 6], [6, 12, 9, 6]])
        } else {
            assert!(
                isc <= 255,
                "scalefac_compress {scalefac_compress} has more than 9 bits"
            );
            let isc = isc - 244;
            let nr_of_sfb = [[8, 8, 5, 0], [15, 12, 9, 0], [6, 18, 9, 0]];
            ([isc / 3, isc % 3, 0, 0], nr_of_sfb)
        };
        Self {
            slen,
            nr_of_sfb: nr_of_sfb[rows as usize],
            preflag: false,
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
    /// For mixed blocks, the first `long` of `scalefac`: `scalefac_l[sfb]`
    /// of the long bands, at `sfb`; and after them `scalefac_s[sfb][window]`
    /// from short band 3 up, at `long + 3 * (sfb - 3) + window`. `long` is
    /// 8, or 6 at the lower sampling frequencies (see
    /// [`mixed_long_bands`](crate::scalefactor_bands::mixed_long_bands)),
    /// and the scalefactors past those sent are unused.
    Mixed {
        long: usize,
        scalefac: [u8; MIXED_SCALEFACTORS],
    },
}

/// The scalefactors of one granule of one channel and the flags that go
/// with them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scalefactors {
    /// The version whose `scalefac_compress` sends them.
    pub version: Version,
    pub scalefac: Scalefac,
    /// Never set for short blocks, mixed or not (clause 2.4.2.7).
    pub preflag: bool,
    pub scalefac_scale: bool,
    /// The scfsi bands whose scalefactors the second granule of a frame takes
    /// over from the first, and so does not send (clause 2.4.2.7). Always
    /// clear in the first granule, for short blocks, and at the lower
    /// sampling frequencies, which have no scfsi.
    pub scfsi: [bool; SCFSI_BANDS],
    /// Set for the right channel of a frame coded in intensity stereo, whose
    /// zero part sends intensity stereo positions in place of scalefactors
    /// (clause 2.4.3.4.9.3). At the lower sampling frequencies it also sends
    /// all its scalefactors with the other branch of `scalefac_compress`
    /// (clause 2.4.3.2 of ISO/IEC 13818-3).
    pub intensity: Option<ZeroPart>,
}

/// Where the zero part of the right channel of a frame coded in intensity
/// stereo begins, in each window: in window `w`, the scalefactor bands from
/// `start[w]` up to `bound` are marked as not coded in intensity stereo, and
/// those from `bound` up are intensity stereo positions. Long blocks have
/// one window, the first, and leave the other starts unused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZeroPart {
    pub start: [usize; SHORT_WINDOWS],
    pub bound: usize,
}

impl ZeroPart {
    /// No zero part yet: the scalefactors of the right channel before its
    /// positions are marked (see
    /// [`IntensityPositions::mark`](crate::stereo::IntensityPositions::mark)).
    pub const UNMARKED: Self = Self {
        start: [usize::MAX; SHORT_WINDOWS],
        bound: usize::MAX,
    };

    /// Whether scalefactor band `sfb` of `window` is marked as not coded in
    /// intensity stereo.
    fn marks(self, sfb: usize, window: usize) -> bool {
        (self.start[window]..self.bound).contains(&sfb)
    }
}

/// The value that marks a scalefactor band of the right channel's zero part
/// as not coded in intensity stereo, the "illegal" position, when sent in
/// `slen` bits, or `None` if the field cannot send it: 7, sent in 3 bits
/// (clause 2.4.3.4.9.3, Annex G.2 c)), or at the lower sampling frequencies
/// the largest value of the field (clause 2.4.3.2 of ISO/IEC 13818-3).
///
/// Spec reading: clause 2.4.3.2 of ISO/IEC 13818-3 gives the scalefactors of
/// a band sent in no bits the intensity stereo position zero, so a field of
/// no bits has no illegal position.
///
/// Settled by decoding with ffmpeg, mpg123 and LAME (black box): all three
/// decode a band whose position is sent in no bits at position 0, the same
/// level in both channels. The conformance test is
/// `positions_7_and_16_are_read_as_illegal`.
///
/// The encoder never sends these marks at the lower sampling frequencies: it
/// codes such a frame without intensity stereo (see `quantize_frame`). They
/// are still given here, as the standard has them, so that the bits the right
/// channel's scalefactors take, and the values its positions must avoid,
/// follow it.
fn not_intensity(version: Version, slen: u32) -> Option<u8> {
    match version {
        Version::Mpeg1 => (slen >= 3).then_some(7),
        Version::Mpeg2 => (slen >= 1).then(|| ((1u32 << slen) - 1) as u8),
    }
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
            BlockType::Mixed => Scalefac::Mixed {
                long: match version {
                    Version::Mpeg1 => 8,
                    Version::Mpeg2 => 6,
                },
                scalefac: [0; MIXED_SCALEFACTORS],
            },
            _ => Scalefac::Long([0; LONG_BANDS]),
        };
        Self {
            version,
            scalefac,
            preflag: false,
            scalefac_scale: false,
            scfsi: [false; SCFSI_BANDS],
            intensity: None,
        }
    }

    /// Whether scalefactor `index` of [`values`](Self::values) is sent, rather
    /// than taken over from the first granule.
    pub fn sent(&self, index: usize) -> bool {
        match self.scalefac {
            Scalefac::Long(_) => !self.scfsi[scfsi_band(index)],
            Scalefac::Short(_) | Scalefac::Mixed { .. } => true,
        }
    }

    /// The scalefactors in the order they are sent: `scalefac_l[sfb]` at
    /// `sfb`, or `scalefac_s[sfb][window]` at `3 * sfb + window`, the order
    /// of [`coding_bands`](crate::scalefactor_bands::coding_bands), and
    /// for mixed blocks as [`Scalefac::Mixed`] has them.
    pub fn values(&self) -> &[u8] {
        match &self.scalefac {
            Scalefac::Long(scalefac) => scalefac,
            Scalefac::Short(scalefac) => scalefac.as_flattened(),
            Scalefac::Mixed { long, scalefac } => &scalefac[..mixed_len(*long)],
        }
    }

    pub fn values_mut(&mut self) -> &mut [u8] {
        match &mut self.scalefac {
            Scalefac::Long(scalefac) => scalefac,
            Scalefac::Short(scalefac) => scalefac.as_flattened_mut(),
            Scalefac::Mixed { long, scalefac } => &mut scalefac[..mixed_len(*long)],
        }
    }

    /// How many of [`values`](Self::values) are sent with `slen1`.
    fn slen1_count(&self) -> usize {
        match self.scalefac {
            Scalefac::Long(_) => SLEN1_LONG,
            Scalefac::Short(_) => SLEN1_SHORT,
            Scalefac::Mixed { long, .. } => long + SLEN1_MIXED_SHORT,
        }
    }

    /// The rows of clause 2.4.3.2 of ISO/IEC 13818-3 for these scalefactors.
    fn rows(&self) -> Rows {
        match self.scalefac {
            Scalefac::Long(_) => Rows::Long,
            Scalefac::Short(_) => Rows::Short,
            Scalefac::Mixed { .. } => Rows::Mixed,
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
    /// rest, limits of 7 and 3. The right channel of a frame coded in
    /// intensity stereo, which has no `preflag`, gets the widest fields of
    /// its own branch: 4 bits for the first partition of
    /// `int_scalefac_compress` below 180, and 5 for the rest, limits of 15
    /// and 31.
    pub fn limit(&self, index: usize) -> u8 {
        if self.version == Version::Mpeg2 && self.intensity.is_some() {
            let first = LsfPartitions::decode_intensity(0, self.rows()).nr_of_sfb[0];
            return if index < first { 15 } else { 31 };
        }
        let first = index < self.slen1_count();
        match (self.version, self.preflag, first) {
            (Version::Mpeg2, true, true) => 7,
            (Version::Mpeg2, true, false) => 3,
            (_, _, true) => 15,
            (_, _, false) => 7,
        }
    }

    /// Whether `preflag` can be sent: not for short blocks, mixed or not
    /// (clause 2.4.2.7), nor for the right channel of a frame coded in
    /// intensity stereo at the lower sampling frequencies, whose branch of
    /// `scalefac_compress` always clears it (clause 2.4.3.2 of ISO/IEC
    /// 13818-3).
    ///
    /// Spec reading: "If block_type==2 (short blocks) preflag is never used"
    /// (clause 2.4.2.7) is taken to cover mixed blocks, which are
    /// `block_type` 2 too. Their long bands would gain nothing from it:
    /// `pretab` is 0 in long bands 0 to 7.
    pub fn can_preemphasise(&self) -> bool {
        !self.short() && (self.version != Version::Mpeg2 || self.intensity.is_none())
    }

    fn short(&self) -> bool {
        self.rows() != Rows::Long
    }

    /// The scalefactor band and window of scalefactor `index` of
    /// [`values`](Self::values): the long bands of a mixed block are in
    /// window 0.
    fn band_window(&self, index: usize) -> (usize, usize) {
        match self.scalefac {
            Scalefac::Long(_) => (index, 0),
            Scalefac::Short(_) => (index / SHORT_WINDOWS, index % SHORT_WINDOWS),
            Scalefac::Mixed { long, .. } if index < long => (index, 0),
            Scalefac::Mixed { long, .. } => (
                MIXED_SHORT_START + (index - long) / SHORT_WINDOWS,
                (index - long) % SHORT_WINDOWS,
            ),
        }
    }

    /// Whether any scalefactor is marked as not coded in intensity stereo.
    pub fn marks_any(&self) -> bool {
        self.intensity.is_some_and(|zero_part| {
            (0..self.values().len()).any(|index| {
                let (sfb, window) = self.band_window(index);
                zero_part.marks(sfb, window)
            })
        })
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
        let steps = usize::from(self.values()[index] + pretab);
        // `scalefac_multiplier * steps`, in halves.
        let halves = if self.scalefac_scale {
            2 * steps
        } else {
            steps
        };
        half_power(halves)
    }

    /// The `scalefac_compress` whose field widths hold the scalefactors sent
    /// in the fewest bits, or `None` if one is above its
    /// [`limit`](Self::limit). At the lower sampling frequencies it also
    /// sets `preflag` as the scalefactors have it, and in the right channel
    /// of a frame coded in intensity stereo [`INTENSITY_SCALE`].
    pub fn compress(&self) -> Option<u16> {
        if self.version == Version::Mpeg1 && self.intensity.is_none() {
            return self.long_compress();
        }
        self.tried_compress()
    }

    /// [`compress`](Self::compress) by trying every field width for every
    /// scalefactor sent.
    fn tried_compress(&self) -> Option<u16> {
        let (count, step, offset) = match (self.version, self.intensity) {
            (Version::Mpeg1, _) => (SCALEFAC_COMPRESS.len() as u16, 1, 0),
            (Version::Mpeg2, None) => (LSF_SCALEFAC_COMPRESS, 1, 0),
            (Version::Mpeg2, Some(_)) => (LSF_SCALEFAC_COMPRESS / 2, 2, INTENSITY_SCALE),
        };
        let fits = |c: u16| {
            let preflag_sent = match self.version {
                Version::Mpeg1 => true,
                Version::Mpeg2 => self.lsf_partitions(c).preflag == self.preflag,
            };
            preflag_sent
                && self.widths(c).enumerate().all(|(index, slen)| {
                    slen.is_none_or(|slen| self.sent_value(index, slen).is_some())
                })
        };
        (0..count)
            .map(|c| c * step + offset)
            .filter(|&c| fits(c))
            .min_by_key(|&c| self.part2_length(c))
    }

    /// [`compress`](Self::compress) in ISO/IEC 11172-3 without intensity
    /// stereo, where a field holds a scalefactor if it holds the largest sent
    /// with its width, and the outer loop asks for it in each iteration.
    fn long_compress(&self) -> Option<u16> {
        let slen1_count = self.slen1_count();
        // The largest scalefactor sent with `slen1` and with `slen2`, and
        // how many are.
        let (mut largest, mut sent) = ([0u8; 2], [0usize; 2]);
        for (index, &value) in self.values().iter().enumerate() {
            if self.sent(index) {
                let part = usize::from(index >= slen1_count);
                largest[part] = largest[part].max(value);
                sent[part] += 1;
            }
        }
        (0..SCALEFAC_COMPRESS.len())
            .filter(|&c| {
                let (slen1, slen2) = SCALEFAC_COMPRESS[c];
                fits_in(largest[0], slen1) && fits_in(largest[1], slen2)
            })
            .min_by_key(|&c| {
                let (slen1, slen2) = SCALEFAC_COMPRESS[c];
                sent[0] * slen1 as usize + sent[1] * slen2 as usize
            })
            .map(|c| c as u16)
    }

    /// The value sent for scalefactor `index` of [`values`](Self::values) in
    /// a field of `slen` bits, or `None` if the field cannot send it: the
    /// scalefactor, or in the right channel of a frame coded in intensity
    /// stereo the mark of a band not coded in it. An intensity stereo
    /// position must not be sent as the mark.
    fn sent_value(&self, index: usize, slen: u32) -> Option<u8> {
        let Some(zero_part) = self.intensity else {
            let value = self.values()[index];
            return fits_in(value, slen).then_some(value);
        };
        let mark = not_intensity(self.version, slen);
        let (sfb, window) = self.band_window(index);
        if zero_part.marks(sfb, window) {
            return mark;
        }
        let value = self.values()[index];
        let position = sfb >= zero_part.bound;
        (fits_in(value, slen) && !(position && mark == Some(value))).then_some(value)
    }

    /// How `scalefac_compress` sends these scalefactors at the lower sampling
    /// frequencies.
    fn lsf_partitions(&self, scalefac_compress: u16) -> LsfPartitions {
        assert_eq!(self.version, Version::Mpeg2);
        assert!(
            !self.scfsi.contains(&true),
            "the lower sampling frequencies have no scfsi"
        );
        match self.intensity {
            Some(_) => LsfPartitions::decode_intensity(scalefac_compress, self.rows()),
            None => LsfPartitions::decode(scalefac_compress, self.rows()),
        }
    }

    /// The width of each scalefactor of [`values`](Self::values) sent with
    /// `scalefac_compress`, or `None` for those not sent.
    fn widths(&self, scalefac_compress: u16) -> impl Iterator<Item = Option<u32>> + '_ {
        // One of the two, without collecting them, as `compress` asks for
        // the widths of every `scalefac_compress` in the inner loop.
        let (mpeg1, mpeg2) = match self.version {
            Version::Mpeg1 => {
                let (slen1, slen2) = SCALEFAC_COMPRESS[usize::from(scalefac_compress)];
                let slen1_count = self.slen1_count();
                let widths = (0..self.values().len()).map(move |index| {
                    let slen = if index < slen1_count { slen1 } else { slen2 };
                    self.sent(index).then_some(slen)
                });
                (Some(widths), None)
            }
            Version::Mpeg2 => {
                let widths = self
                    .lsf_partitions(scalefac_compress)
                    .widths(self.values().len())
                    .map(Some);
                (None, Some(widths))
            }
        };
        mpeg1
            .into_iter()
            .flatten()
            .chain(mpeg2.into_iter().flatten())
    }

    /// Writes the scalefactors sent (clause 2.4.1.7) with the field widths of
    /// `scalefac_compress`.
    ///
    /// # Panics
    ///
    /// If a scalefactor sent does not fit its field.
    pub fn write(&self, scalefac_compress: u16, w: &mut BitWriter) {
        let widths = self.widths(scalefac_compress);
        for (index, slen) in widths.enumerate() {
            let Some(slen) = slen else { continue };
            let value = self
                .sent_value(index, slen)
                .unwrap_or_else(|| panic!("scalefactor {index} cannot be sent in {slen} bits"));
            w.write(u32::from(value), slen);
        }
    }

    /// `part2_length`: the sum of the widths of the scalefactors sent. Without
    /// `scfsi` this is `11 * slen1 + 10 * slen2` for long blocks,
    /// `18 * slen1 + 18 * slen2` for short ones and `17 * slen1 + 18 * slen2`
    /// for mixed ones (clause 2.4.3.4.5), and at the
    /// lower sampling frequencies the sum of `nr_of_sfb * slen` over the
    /// partitions (clause 2.4.3.2 of ISO/IEC 13818-3).
    pub fn part2_length(&self, scalefac_compress: u16) -> usize {
        (self.widths(scalefac_compress).flatten())
            .map(|slen| slen as usize)
            .sum()
    }
}

/// The scalefactors a mixed block with `long` long bands sends.
fn mixed_len(long: usize) -> usize {
    long + (SHORT_BANDS - MIXED_SHORT_START) * SHORT_WINDOWS
}

/// Whether `value` fits in `bits` bits.
fn fits_in(value: u8, bits: u32) -> bool {
    u32::from(value) >> bits == 0
}

/// `2^(halves / 2)`, from a table of every power that a scalefactor with its
/// preemphasis, which fit in a byte, can give, as the outer loop takes the
/// amplification of every band in each iteration.
fn half_power(halves: usize) -> f64 {
    const TABLE_HALVES: usize = 2 * (u8::MAX as usize + 1);
    static TABLE: std::sync::OnceLock<[f64; TABLE_HALVES]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| std::array::from_fn(|halves| (0.5 * halves as f64).exp2()));
    table[halves]
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

    /// For every block type, `compress` without intensity stereo chooses as
    /// trying every width for every scalefactor does.
    #[test]
    fn long_compress_chooses_as_trying_every_width_does() {
        let mut state = 7u32;
        let mut next = |bound: u32| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            ((state >> 8) % bound) as u8
        };
        for case in 0..20_000 {
            let block_type = [BlockType::Normal, BlockType::Short, BlockType::Mixed][case % 3];
            let mut scalefactors = Scalefactors::new(block_type, Version::Mpeg1);
            let largest = [1, 2, 4, 8, 16, 20][case % 6];
            for value in scalefactors.values_mut() {
                *value = next(largest);
            }
            if block_type == BlockType::Normal {
                scalefactors.scfsi = std::array::from_fn(|_| next(3) == 0);
            }
            assert_eq!(
                scalefactors.long_compress(),
                scalefactors.tried_compress(),
                "{scalefactors:?}"
            );
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

    /// A mixed block sends 8 long and 27 short scalefactors, 17 of them with
    /// `slen1` and 18 with `slen2` (clause 2.4.3.4.5), never with preflag,
    /// and has the limits of clause C.1.5.4.3.6, 15 for the first 17.
    #[test]
    fn mixed_blocks_send_17_scalefactors_with_slen1() {
        let mut scalefactors = Scalefactors::new(BlockType::Mixed, Version::Mpeg1);
        assert_eq!(scalefactors.values().len(), 35);
        assert!(!scalefactors.can_preemphasise());
        assert_eq!((scalefactors.limit(16), scalefactors.limit(17)), (15, 7));
        scalefactors.values_mut()[16] = 9;
        scalefactors.values_mut()[34] = 5;
        let compress = scalefactors.compress().unwrap();
        assert_eq!(SCALEFAC_COMPRESS[usize::from(compress)], (4, 3));
        assert_eq!(scalefactors.part2_length(compress), 17 * 4 + 18 * 3);
        let mut w = BitWriter::new();
        scalefactors.write(compress, &mut w);
        assert_eq!(w.bit_len(), 17 * 4 + 18 * 3);
        // Short band 3, window 1, follows the 8 long bands.
        assert_eq!(scalefactors.band_window(9), (3, 1));
        assert_eq!(scalefactors.band_window(34), (11, 2));
    }

    /// At the lower sampling frequencies a mixed block sends 6 long and 27
    /// short scalefactors in the partitions of the mixed rows of clause
    /// 2.4.3.2 of ISO/IEC 13818-3.
    #[test]
    fn lsf_mixed_blocks_use_the_mixed_rows() {
        let cases = [
            (0, [6, 9, 9, 9]),
            (399, [6, 9, 9, 9]),
            (400, [6, 9, 12, 6]),
            (499, [6, 9, 12, 6]),
            (500, [15, 18, 0, 0]),
            (511, [15, 18, 0, 0]),
        ];
        for (sc, nr_of_sfb) in cases {
            let partitions = LsfPartitions::decode(sc, Rows::Mixed);
            assert_eq!(partitions.nr_of_sfb, nr_of_sfb, "{sc}");
            assert_eq!(partitions.nr_of_sfb.iter().sum::<usize>(), 33);
            // The other rows are unchanged.
            assert_eq!(LsfPartitions::decode(sc, Rows::Short).slen, partitions.slen);
        }
        let mut scalefactors = Scalefactors::new(BlockType::Mixed, Version::Mpeg2);
        assert_eq!(scalefactors.values().len(), 33);
        scalefactors.values_mut()[5] = 15;
        scalefactors.values_mut()[32] = 7;
        let compress = scalefactors.compress().unwrap();
        let partitions = LsfPartitions::decode(compress, Rows::Mixed);
        assert_eq!(partitions.slen, [4, 0, 0, 3]);
        assert_eq!(scalefactors.part2_length(compress), 6 * 4 + 9 * 3);
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

    /// The right channel's branch of `scalefac_compress` (clause 2.4.3.2 of
    /// ISO/IEC 13818-3), at the first value of each of its three ranges and
    /// at the last.
    #[test]
    fn intensity_branch_decodes_int_scalefac_compress() {
        let cases = [
            (
                0,
                [0, 0, 0, 0],
                [7, 7, 7, 0],
                [12, 12, 12, 0],
                [6, 15, 12, 0],
            ),
            (
                2 * 179 + 1,
                [4, 5, 5, 0],
                [7, 7, 7, 0],
                [12, 12, 12, 0],
                [6, 15, 12, 0],
            ),
            (
                2 * 180,
                [0, 0, 0, 0],
                [6, 6, 6, 3],
                [12, 9, 9, 6],
                [6, 12, 9, 6],
            ),
            (
                2 * 243,
                [3, 3, 3, 0],
                [6, 6, 6, 3],
                [12, 9, 9, 6],
                [6, 12, 9, 6],
            ),
            (
                2 * 244,
                [0, 0, 0, 0],
                [8, 8, 5, 0],
                [15, 12, 9, 0],
                [6, 18, 9, 0],
            ),
            (
                2 * 255 + 1,
                [3, 2, 0, 0],
                [8, 8, 5, 0],
                [15, 12, 9, 0],
                [6, 18, 9, 0],
            ),
        ];
        for (sc, slen, long, short, mixed) in cases {
            for (rows, nr_of_sfb) in [
                (Rows::Long, long),
                (Rows::Short, short),
                (Rows::Mixed, mixed),
            ] {
                let partitions = LsfPartitions::decode_intensity(sc, rows);
                assert_eq!(partitions.slen, slen, "{sc}");
                assert_eq!(partitions.nr_of_sfb, nr_of_sfb, "{sc}");
                assert!(!partitions.preflag);
            }
        }
    }

    /// At the lower sampling frequencies the right channel of a frame in
    /// intensity stereo sends `intensity_scale` 1, has no preemphasis, and
    /// limits of 15 and 31.
    #[test]
    fn intensity_branch_sends_its_scale_and_no_preflag() {
        let mut scalefactors = Scalefactors::new(BlockType::Normal, Version::Mpeg2);
        assert!(scalefactors.can_preemphasise());
        scalefactors.intensity = Some(ZeroPart::UNMARKED);
        assert!(!scalefactors.can_preemphasise());
        assert_eq!((scalefactors.limit(6), scalefactors.limit(7)), (15, 31));
        assert_eq!(scalefactors.compress(), Some(1));
        scalefactors.values_mut()[0] = 15;
        scalefactors.values_mut()[20] = 31;
        let compress = scalefactors.compress().unwrap();
        assert_eq!(compress % 2, 1);
        assert_eq!(
            LsfPartitions::decode_intensity(compress, Rows::Long).slen,
            [4, 0, 5, 0]
        );
        scalefactors.preflag = true;
        assert_eq!(scalefactors.compress(), None);

        let short = Scalefactors {
            intensity: Some(ZeroPart::UNMARKED),
            ..Scalefactors::new(BlockType::Short, Version::Mpeg2)
        };
        assert_eq!((short.limit(11), short.limit(12)), (15, 31));
    }

    /// At the lower sampling frequencies a mark is sent as the largest value
    /// of its field, a position never is, and a position sent in no bits is
    /// zero.
    #[test]
    fn lsf_marks_are_the_largest_value_of_their_field() {
        let mut scalefactors = Scalefactors::new(BlockType::Normal, Version::Mpeg2);
        scalefactors.values_mut()[..7].copy_from_slice(&[0, 2, 0, 0, 0, 0, 0]);
        // Bands 7 to 13 marked, 14 to 20 positions.
        scalefactors.intensity = Some(ZeroPart {
            start: [7; SHORT_WINDOWS],
            bound: 14,
        });
        // Positions 0: the marks need one bit, the positions none.
        let compress = scalefactors.compress().unwrap();
        assert_eq!(
            LsfPartitions::decode_intensity(compress, Rows::Long).slen,
            [2, 1, 0, 0]
        );
        assert_eq!(scalefactors.part2_length(compress), 7 * 2 + 7);
        let mut w = BitWriter::new();
        scalefactors.write(compress, &mut w);
        let bits: String = w.into_bytes().iter().map(|b| format!("{b:08b}")).collect();
        assert_eq!(&bits[..21], "00100000000000".to_owned() + "1111111");

        // Position 3 would be the mark of 2 bits, and so takes 3.
        scalefactors.values_mut()[20] = 3;
        let compress = scalefactors.compress().unwrap();
        let slen = LsfPartitions::decode_intensity(compress, Rows::Long).slen;
        assert_eq!(slen[2], 3);
        let mut w = BitWriter::new();
        scalefactors.write(compress, &mut w);
        assert_eq!(w.bit_len(), scalefactors.part2_length(compress));
        let bits: String = w.into_bytes().iter().map(|b| format!("{b:08b}")).collect();
        let marks = 7 * slen[0] as usize..7 * (slen[0] + slen[1]) as usize;
        let mark = "1".repeat(slen[1] as usize);
        assert_eq!(bits[marks], mark.repeat(7));

        // MPEG-1 marks with 7 in 3 bits.
        let mpeg1 = Scalefactors {
            intensity: Some(ZeroPart {
                start: [7; SHORT_WINDOWS],
                bound: 14,
            }),
            ..Scalefactors::default()
        };
        let compress = mpeg1.compress().unwrap();
        assert_eq!(SCALEFAC_COMPRESS[usize::from(compress)], (3, 3));
    }
}
