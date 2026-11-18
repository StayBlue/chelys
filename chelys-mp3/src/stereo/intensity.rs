// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Intensity stereo (Annex G.2 c), and G.1 for its bounds): the bands it
//! codes and their positions. The decoder's inverse is clause 2.4.3.4.9.3,
//! and clause 2.4.3.2 of ISO/IEC 13818-3 at the lower sampling frequencies.

use std::f64::consts::PI;
use std::iter::Chain;
use std::ops::Range;

use crate::granule::{BlockType, GRANULE_LINES};
use crate::header::{SamplingFrequency, Version};
use crate::scalefactor_bands::{self, SHORT_WINDOWS};
use crate::scalefactors::{Scalefactors, ZeroPart};

/// `intensity_scale`, sent in the right channel's `scalefac_compress` at the
/// lower sampling frequencies, which sets the step between intensity stereo
/// positions: `i0 = 1/sqrt(2)` for 1, `1/sqrt(sqrt(2))` for 0 (clause
/// 2.4.3.2 of ISO/IEC 13818-3).
///
/// Spec gap: neither standard says how to choose it. It is always 1: each
/// step of position then halves the energy of the quieter channel, 3 dB,
/// which keeps every level within 1.5 dB of its ratio, closer than the
/// positions of ISO/IEC 11172-3 at their closest (0 and 4.8 dB at positions
/// 3 and 4), and in the 3 bits those positions take reaches 9 dB either way,
/// where steps of 1.5 dB would reach 4.5 dB.
pub const SCALE: u16 = 1;

/// `i0` for [`SCALE`].
const I0: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// The most steps of `i0` that a position at the lower sampling frequencies
/// gives the quieter channel: positions 13 and 14, 21 dB with
/// [`SCALE`] 1.
///
/// Spec reading: clause 2.4.3.2 of ISO/IEC 13818-3 allows positions up to
/// 30, in a field of 5 bits, but not every decoder reads them so. Positions
/// are kept below 16.
///
/// Settled by decoding with ffmpeg, mpg123 and LAME (black box): ffmpeg
/// reads position 16 as illegal, and mpg123 and LAME read it as a position;
/// all three read 13, 14 and 15 as positions. The conformance test is
/// `positions_7_and_16_are_read_as_illegal`.
const MAX_STEPS: i32 = 7;

/// The steps of `i0` that a position at the lower sampling frequencies never
/// gives: position 7, the left channel four steps quieter than the right.
///
/// Spec reading: position 7 is a position in a field of 4 or 5 bits (clause
/// 2.4.3.2 of ISO/IEC 13818-3), but not every decoder reads it so. A band
/// whose nearest position is 7 takes the nearer of 5 and 9.
///
/// Settled by decoding with ffmpeg, mpg123 and LAME (black box): mpg123 and
/// LAME read position 7 as illegal in a field of 4 bits, and ffmpeg reads it
/// as a position; all three read 5 as a position. The conformance test is
/// `positions_7_and_16_are_read_as_illegal`, and
/// `marks_are_read_as_illegal_in_different_fields` shows mpg123 and LAME
/// reading 7 as illegal in a field of 3 bits too.
const SKIPPED_STEPS: i32 = -4;

/// The steps by which more and more bands are coded in intensity stereo
/// (see [`bounds`]).
pub const STEPS: usize = 4;

/// The `bound` of Layers I and II in intensity stereo: the subband from
/// which they code it (Annex G.1, clause 2.4.2.3), from the fewest subbands
/// to the most.
const LAYER_I_II_BOUNDS: [usize; STEPS] = [16, 12, 8, 4];

/// The frequency lines of a granule in each polyphase subband (clause
/// 2.4.3.4.10): 18 in a long block, 6 in each window of a short one.
const SUBBAND_LINES: usize = 18;

/// The scalefactor bands of `block_type`, long or short, from which a
/// granule may be coded in intensity stereo, from the fewest bands to the
/// most.
///
/// Spec gap: Annex G.2 c) does not say which bands to code in intensity
/// stereo. Annex G.1 does for Layers I and II: from subband 16, 12, 8 or 4,
/// depending on the bits needed. The same four steps are taken here, each
/// from the first scalefactor band that starts at or above the subband, in
/// each window of a short block, so that intensity stereo never reaches
/// below where those layers would let it. For long blocks the lowest, from
/// subband 4, starts at 2.3 to 3.0 kHz, above the "about 2 kHz" where G.1
/// says the temporal envelope, which intensity stereo keeps, takes over
/// localisation from the fine structure, which it loses; for short blocks,
/// whose bands are coarser, at 2.5 to 3.5 kHz. At the lower sampling
/// frequencies, whose subbands are half as wide, it starts at 1.1 to 1.7 kHz
/// for long blocks and 1.1 to 1.6 kHz for short ones, below that, as it
/// does for Layers I and II there, whose `bound` is that of ISO/IEC 11172-3
/// (clause 2.4.2.3 of ISO/IEC 13818-3).
pub fn bounds(block_type: BlockType, sampling_frequency: SamplingFrequency) -> [usize; STEPS] {
    let starts: Vec<usize> = match block_type {
        BlockType::Short => scalefactor_bands::short_bands(sampling_frequency)
            .iter()
            .map(|band| band.start)
            .collect(),
        _ => scalefactor_bands::long_bands(sampling_frequency)
            .iter()
            .map(|band| band.start)
            .collect(),
    };
    let lines = SUBBAND_LINES / scalefactor_bands::windows(block_type);
    LAYER_I_II_BOUNDS.map(|subband| {
        starts
            .iter()
            .position(|&start| start >= lines * subband)
            .expect("every bound lies below the highest band")
    })
}

/// The intensity stereo position of a scalefactor band,
/// `is_pos = nint(12 / pi * arctan(sqrt(L_energy / R_energy)))` (Annex
/// G.2 c)), from the energies of its left and right lines: 0 for the right
/// channel alone, 6 for the left alone.
///
/// Spec reading: a band with no energy in the right channel divides by
/// zero; it takes the limit, position 6, the left channel alone.
///
/// Spec gap: a band with no energy in either channel has no position. It
/// takes the middle one, 3; its lines are zero in both channels anyway.
pub fn position(left_energy: f64, right_energy: f64) -> u8 {
    if left_energy == 0.0 && right_energy == 0.0 {
        return 3;
    }
    // A silent right channel makes the ratio infinite, and the position 6.
    let angle = (left_energy / right_energy).sqrt().atan();
    (12.0 / PI * angle).round() as u8
}

/// The intensity stereo position of a scalefactor band at the lower
/// sampling frequencies, from the energies of its left and right lines.
///
/// Spec reading: Annex C.1.3 of ISO/IEC 13818-3 has the encoder take every
/// basic step of ISO/IEC 11172-3, but the position of Annex G.2 c) is the
/// one by which that standard's decoder splits the sum, and clause 2.4.3.2
/// of ISO/IEC 13818-3 decodes positions otherwise: it multiplies the lines
/// sent by `k_l` and `k_r`, both 1 at position 0, and at odd positions
/// `k_l = i0^((is_pos + 1) / 2)`, at even ones `k_r = i0^(is_pos / 2)`, the
/// other 1. The position taken is the one whose `k_l / k_r` lies nearest the
/// ratio `sqrt(L_energy / R_energy)` that Annex G.2 c) takes the arctangent
/// of, nearest in steps of `i0`, of which there are up to `MAX_STEPS`,
/// other than `SKIPPED_STEPS`.
///
/// Spec gap: a band with no energy in either channel has no position. It
/// takes position 0, the same level in both; its lines are zero in both
/// channels anyway.
pub fn lsf_position(left_energy: f64, right_energy: f64) -> u8 {
    if left_energy == 0.0 && right_energy == 0.0 {
        return 0;
    }
    // Steps of i0 in amplitude; a silent channel makes them infinite.
    let exact = (left_energy / right_energy).sqrt().ln() / -I0.ln();
    let steps = exact
        .round()
        .clamp(f64::from(-MAX_STEPS), f64::from(MAX_STEPS)) as i32;
    let steps = match steps {
        SKIPPED_STEPS if exact >= f64::from(SKIPPED_STEPS) => SKIPPED_STEPS + 1,
        SKIPPED_STEPS => SKIPPED_STEPS - 1,
        steps => steps,
    };
    match steps {
        0 => 0,
        n if n > 0 => (2 * n) as u8,
        n => (-2 * n - 1) as u8,
    }
}

/// The factors by which the decoder multiplies the lines sent in the left
/// channel for a band at intensity stereo `position`, to give the left and
/// the right channel: `is_ratio / (1 + is_ratio)` and `1 / (1 + is_ratio)`
/// with `is_ratio = tan(is_pos * pi / 12)` (clause 2.4.3.4.9.3), or at the
/// lower sampling frequencies `k_l` and `k_r` (clause 2.4.3.2 of ISO/IEC
/// 13818-3).
///
/// Spec reading: step 5 of clause 2.4.3.4.9.3 computes `R_i` from `L_i`
/// after step 4 has set `L_i` to the left channel's share. It is read as the
/// lines sent, as before step 4: so the two shares add up to them, which
/// Annex G.2 c) sends as `L_i + R_i`. Clause 2.4.3.2 of ISO/IEC 13818-3
/// swaps the two steps so that this is so.
fn shares(version: Version, position: u8) -> (f64, f64) {
    match version {
        Version::Mpeg1 => {
            // sin / (sin + cos) and cos / (sin + cos), exact at the ends,
            // where tan is 0 or infinite.
            let (sin, cos) = match position {
                0 => (0.0, 1.0),
                6 => (1.0, 0.0),
                _ => (f64::from(position) * PI / 12.0).sin_cos(),
            };
            (sin / (sin + cos), cos / (sin + cos))
        }
        Version::Mpeg2 => match position {
            0 => (1.0, 1.0),
            p if p % 2 == 1 => (I0.powi(i32::from(p.div_ceil(2))), 1.0),
            p => (1.0, I0.powi(i32::from(p / 2))),
        },
    }
}

/// The positions of the bands of the right channel that a granule codes in
/// intensity stereo, for the scalefactors of its zero part (clause
/// 2.4.3.4.9.3).
///
/// For short blocks each window has its own position in each band, and the
/// encoder codes the same bands of all three in intensity stereo: the bound
/// is the same short scalefactor band in each window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Positions {
    block_type: BlockType,
    /// The first scalefactor band coded in intensity stereo, long or short
    /// as `block_type` has them.
    bound: usize,
    /// `is_pos` of each scalefactor from the bound, in the order of
    /// [`Scalefactors::values`].
    positions: Vec<u8>,
    sampling_frequency: SamplingFrequency,
}

impl Positions {
    /// The positions of one granule of `block_type` coded from scalefactor
    /// band `bound` up in intensity stereo, each found from the energies of
    /// the band's `left` and `right` lines (Annex G.2 c), and
    /// [`lsf_position`] at the lower sampling frequencies), in
    /// each window of a short block.
    ///
    /// Spec gap: the lines above the highest scalefactor band have no
    /// scalefactor, and so no position, and the standard does not say how a
    /// decoder treats them in intensity stereo. They lie in the right
    /// channel's zero part, which the figure of clause 2.4.3.4.9.3 shows
    /// coded in intensity stereo up to the end of the spectrum, and are
    /// coded like the bands (see [`code`](Self::code)). They are taken to
    /// share the position of the highest band, in each window of a short
    /// block that of the window's highest band, which is found from the
    /// energies of that band's lines and theirs together.
    ///
    /// Settled by decoding with ffmpeg (black box): at 64 kbit/s and
    /// 44.1 kHz, an 18 kHz tone with the right channel at half the left
    /// decodes at 0.0948 and 0.0547 for the 0.0951 and 0.0549 its position
    /// gives. With the highest band's position found from that band's lines
    /// alone, the left channel decodes at 0.0800: ffmpeg splits the lines
    /// above the highest band by its position. The conformance test
    /// `intensity_stereo_keeps_the_level_of_each_channel` fails then.
    ///
    /// Settled by decoding with ffmpeg, mpg123 and LAME (black box): at
    /// 22.05 kHz and 48 kbit/s, a 10.5 kHz tone, above the highest band, with
    /// the right channel at half the left decodes within 10% of its level in
    /// each channel in all three, as the highest band's position gives. The
    /// conformance test is
    /// `intensity_stereo_keeps_the_level_of_each_channel_at_the_lower_sampling_frequencies`.
    ///
    /// Settled by decoding with ffmpeg, mpg123 and LAME (black box): at 44.1
    /// and 22.05 kHz, a line of a short block above band 11, the highest
    /// short band, decodes in all three at the position sent for band 11 of
    /// its own window, with a different position in each window. The
    /// conformance test is
    /// `lines_above_the_short_bands_take_the_position_of_the_highest_band`.
    ///
    /// # Panics
    ///
    /// If `bound` is not a scalefactor band of `block_type`.
    pub fn new(
        left: &[f64; GRANULE_LINES],
        right: &[f64; GRANULE_LINES],
        block_type: BlockType,
        bound: usize,
        sampling_frequency: SamplingFrequency,
    ) -> Self {
        let position_of = match sampling_frequency.version() {
            Version::Mpeg1 => position,
            Version::Mpeg2 => lsf_position,
        };
        let scalefactors = scalefactor_bands::coding_bands(block_type, sampling_frequency).len();
        assert!(
            bound < scalefactors / scalefactor_bands::windows(block_type),
            "bound {bound} is not a scalefactor band"
        );
        let mut positions = vec![0; scalefactors];
        for (index, lines) in coded_bands(block_type, bound, sampling_frequency) {
            let left_energy = energy(lines.clone().map(|i| left[i]));
            positions[index] = position_of(left_energy, energy(lines.map(|i| right[i])));
        }
        Self {
            block_type,
            bound,
            positions,
            sampling_frequency,
        }
    }

    /// The first scalefactor coded in intensity stereo, in the order of
    /// [`Scalefactors::values`]: the bound in its first window.
    pub fn first_scalefactor(&self) -> usize {
        self.bound * scalefactor_bands::windows(self.block_type)
    }

    /// The first line coded in intensity stereo, in the order the lines are
    /// sent: all those from it on are, in every window.
    pub fn first_line(&self) -> usize {
        scalefactor_bands::coding_bands(self.block_type, self.sampling_frequency)
            [self.first_scalefactor()]
        .start
    }

    /// Codes a granule's lines from the bound up in intensity stereo (Annex
    /// G.2 c)): sends the sum of its `left` and `right` lines in the left
    /// channel, `L_i = L_i + R_i`, and zero in the right. `coded` holds the
    /// lines sent in the left and right channels, which below the bound are
    /// the left and right lines or, in MS stereo, the middle and side lines.
    /// The decoder splits the sum by the positions (clause 2.4.3.4.9.3).
    ///
    /// Spec reading: the decoder of ISO/IEC 11172-3 splits the sum into
    /// shares that add up to it, but at the lower sampling frequencies it
    /// multiplies the lines sent by `k_l` and `k_r`, the larger of which is 1
    /// (clause 2.4.3.2 of ISO/IEC 13818-3), so that the sum itself would
    /// decode too loud: twice the level of each channel where the two are
    /// alike. There the sum is scaled in each band so that the two channels
    /// decode with the energy of its left and right lines together. For a
    /// signal panned between the channels, as intensity stereo takes it to
    /// be, at the level difference its position gives, that sends the
    /// louder channel's lines.
    ///
    /// Settled by decoding with ffmpeg, mpg123 and LAME (black box): at
    /// 22.05 kHz and 48 kbit/s, tones at positions 0 and 4 decode within 10%
    /// of their level in each channel in all three, where the sum unscaled
    /// would decode 1.5 or 2 times too loud. The conformance test is
    /// `intensity_stereo_keeps_the_level_of_each_channel_at_the_lower_sampling_frequencies`.
    pub fn code(
        &self,
        left: &[f64; GRANULE_LINES],
        right: &[f64; GRANULE_LINES],
        coded: [&mut [f64; GRANULE_LINES]; 2],
    ) {
        let fs = self.sampling_frequency;
        let [coded_left, coded_right] = coded;
        for (index, lines) in coded_bands(self.block_type, self.bound, fs) {
            let sum: Vec<f64> = lines.clone().map(|i| left[i] + right[i]).collect();
            let scale = match fs.version() {
                Version::Mpeg1 => 1.0,
                Version::Mpeg2 => {
                    let (k_l, k_r) = shares(fs.version(), self.positions[index]);
                    let wanted = energy(lines.clone().map(|i| left[i]))
                        + energy(lines.clone().map(|i| right[i]));
                    let decoded = (k_l * k_l + k_r * k_r) * energy(sum.iter().copied());
                    if decoded > 0.0 {
                        (wanted / decoded).sqrt()
                    } else {
                        0.0
                    }
                }
            };
            for (i, x) in lines.zip(sum) {
                coded_left[i] = scale * x;
                coded_right[i] = 0.0;
            }
        }
    }

    /// The allowed distortion of the sum of the left and right channels
    /// sent for scalefactor `index` of [`Scalefactors::values`], from or
    /// above the bound, given the allowed distortion of the left and right
    /// channels.
    ///
    /// Spec gap: Annex G does not say how to find it. The decoder gives each
    /// channel the lines sent times its `shares`, and so those factors,
    /// squared, of their noise energy. The sum is allowed the most noise
    /// that keeps each channel within its allowance. A channel whose share
    /// is zero sets no limit, and nor does one that allows no noise because
    /// it has no energy in the band, as in
    /// [`middle_side::distortion`](super::middle_side::distortion).
    pub fn sum_distortion(&self, index: usize, left: f64, right: f64) -> f64 {
        assert!(
            index >= self.first_scalefactor(),
            "scalefactor {index} is below the bound"
        );
        let (left_share, right_share) =
            shares(self.sampling_frequency.version(), self.positions[index]);
        [(left, left_share), (right, right_share)]
            .into_iter()
            .filter(|&(xmin, share)| xmin > 0.0 && share > 0.0)
            .map(|(xmin, share)| xmin / (share * share))
            .reduce(f64::min)
            .unwrap_or(0.0)
    }

    /// Sets the scalefactors of the right channel's zero part, given its
    /// quantised `values`: the position of each band from the bound, and a
    /// mark in each band below it that lies above the last nonzero value, as
    /// Annex G.2 c) asks so that the decoder does not decode those bands in
    /// intensity stereo. The mark is the "illegal" position, sent as
    /// [`Scalefactors::write`] gives it. At the lower sampling frequencies a
    /// granule with marks is not kept (see `quantize_frame`), but they still
    /// count towards the bits of its scalefactors while it is quantised.
    ///
    /// Spec reading: clause 2.4.3.4.9.1 has the zero part begin above the
    /// band of the last nonzero value, and also at line
    /// `big_values * 2 + count1 * 4`. Here the `count1` partition ends with
    /// the pair holding the last nonzero value (clause C.1.5.4.4.4), and
    /// scalefactor bands start on even lines, so the two agree.
    ///
    /// Spec reading: for short blocks, clause 2.4.3.2 of ISO/IEC 13818-3 finds
    /// the bound in each window apart, from the window's own last nonzero
    /// value, and says this is done "like" ISO/IEC 11172-3 (its 1995 edition
    /// said "unlike"). Clause 2.4.3.4.9.3 of ISO/IEC 11172-3 says nothing of
    /// windows, and is read as the 1998 edition of ISO/IEC 13818-3 describes
    /// it. So each window has its own zero part, and its bands below the
    /// bound are marked from above its own last nonzero value. A decoder
    /// that found one zero part for the granule, from the last nonzero value
    /// in the order the values are sent, would read the same bands as
    /// coded in intensity stereo, as the bound is the same in every window
    /// and the bands marked would at most be read as scalefactors of zeros.
    ///
    /// Settled by decoding with ffmpeg, mpg123 and LAME (black box): at 44.1
    /// and 22.05 kHz, with the right channel's last nonzero value in band 6
    /// of its third window, a line in band 3 of the first window decodes in
    /// all three at the position sent for it, and in band 3 of the third
    /// window in the left channel alone. At 44.1 kHz, with the bound at band
    /// 8, a line in a band below it marked in its window decodes in the left
    /// channel alone, and a line from the bound at its position. The
    /// conformance tests are `short_blocks_find_the_bound_in_each_window`
    /// and `short_blocks_mark_bands_in_each_window`.
    ///
    /// Spec reading: Annex G.2 c) asks for the mark in every band of zeros
    /// outside the intensity coded part. Only the bands of the zero part
    /// could be decoded in intensity stereo, so the bands of zeros below the
    /// last nonzero value keep their scalefactors.
    ///
    /// Spec reading: in MS stereo the right channel is the side channel,
    /// which Annex G.2 c) names, as the "right/difference channel", among
    /// those whose marked bands are kept from intensity stereo decoding.
    /// Clause 2.4.3.4.9.1 has MS stereo end at the zero part, but a marked
    /// band is taken to be decoded as though outside it, in MS stereo, as
    /// its middle channel would otherwise be decoded as the left channel
    /// alone.
    ///
    /// Settled by decoding with ffmpeg (black box): in MS and intensity
    /// stereo, with the side channel zero below the bound and so every band
    /// there marked, a 440 Hz tone the same in both channels at 0.2 decodes
    /// at 0.1997 in each, where decoding the marked bands in left/right
    /// stereo would give 0.283 in the left and nothing in the right. The
    /// conformance test is `intensity_stereo_keeps_the_level_of_each_channel`.
    ///
    /// # Panics
    ///
    /// If a value from the bound is not zero.
    pub fn mark(&self, scalefactors: &mut Scalefactors, values: &[i32; GRANULE_LINES]) {
        let bands = scalefactor_bands::coding_bands(self.block_type, self.sampling_frequency);
        let windows = scalefactor_bands::windows(self.block_type);
        let first = self.first_scalefactor();
        assert!(
            values[bands[first].start..].iter().all(|&v| v == 0),
            "the right channel is zero from the intensity stereo bound"
        );
        // The band above the last nonzero value of each window.
        let mut start = [0; SHORT_WINDOWS];
        for (index, band) in bands[..first].iter().enumerate() {
            if values[band.start..band.end()].iter().any(|&v| v != 0) {
                start[index % windows] = index / windows + 1;
            }
        }
        for (index, scalefactor) in scalefactors.values_mut().iter_mut().enumerate() {
            let (sfb, window) = (index / windows, index % windows);
            if sfb >= start[window] {
                // The marked bands send no value of their own.
                *scalefactor = if sfb < self.bound {
                    0
                } else {
                    self.positions[index]
                };
            }
        }
        scalefactors.intensity = Some(ZeroPart {
            start,
            bound: self.bound,
        });
    }

    /// At the lower sampling frequencies, gives the right channel's
    /// quantised `values` a value of 1, with the sign of the line, at the
    /// largest of the `amplified` lines of the band just below the bound,
    /// in each window where that band has no nonzero value. Returns whether
    /// it gave any.
    ///
    /// Spec reading: the encoder sends no marks there (see
    /// `quantize_frame`), and a right channel whose last nonzero value lies
    /// below that band would need every band from above it to the bound
    /// marked, so that the frame could not be coded in intensity stereo
    /// from this bound. With a value in the band just below the bound, the
    /// zero part begins at the bound and the bands below are sent as zeros.
    /// The value adds the line's quantisation noise to the band, which the
    /// iteration loops judge against its allowed distortion as any other.
    /// The bands below with no nonzero value decode as zeros whatever their
    /// scalefactors, so those are sent as 0, as [`mark`](Self::mark) sends
    /// those of the bands it marks.
    pub fn fill_below_bound(
        &self,
        values: &mut [i32; GRANULE_LINES],
        amplified: &[f64; GRANULE_LINES],
    ) -> bool {
        if self.sampling_frequency.version() != Version::Mpeg2 || self.bound == 0 {
            return false;
        }
        let bands = scalefactor_bands::coding_bands(self.block_type, self.sampling_frequency);
        let windows = scalefactor_bands::windows(self.block_type);
        let mut filled = false;
        for window in 0..windows {
            let band = &bands[(self.bound - 1) * windows + window];
            let lines = band.start..band.end();
            if values[lines.clone()].iter().any(|&v| v != 0) {
                continue;
            }
            let strongest = lines
                .max_by(|&a, &b| amplified[a].abs().total_cmp(&amplified[b].abs()))
                .expect("a band has lines");
            values[strongest] = if amplified[strongest] < 0.0 { -1 } else { 1 };
            filled = true;
        }
        filled
    }

    /// Sets to 0 the scalefactors of the bands below the bound that have no
    /// nonzero value in `values` (see [`fill_below_bound`](Self::fill_below_bound)).
    pub fn clear_empty_bands(
        &self,
        scalefactors: &mut Scalefactors,
        values: &[i32; GRANULE_LINES],
    ) {
        let bands = scalefactor_bands::coding_bands(self.block_type, self.sampling_frequency);
        let first = self.first_scalefactor();
        for (band, scalefactor) in bands[..first].iter().zip(scalefactors.values_mut()) {
            if values[band.start..band.end()].iter().all(|&v| v == 0) {
                *scalefactor = 0;
            }
        }
    }

    /// The bits the right channel takes with all its lines quantised to
    /// zero, every band below the bound marked and every band from it
    /// sending its position: the bits the iteration loops can always fit it
    /// in, by quantising more coarsely, and so the fewest it must be given.
    ///
    /// In ISO/IEC 11172-3 these are 63 for long blocks and 108 for short
    /// ones, every band in 3 bits; at the lower sampling frequencies they
    /// depend on the positions, and a granule quantised so is not kept, as
    /// its marks are not sent there (see `quantize_frame`).
    pub fn silent_bits(&self) -> usize {
        let mut scalefactors =
            Scalefactors::new(self.block_type, self.sampling_frequency.version());
        self.mark(&mut scalefactors, &[0; GRANULE_LINES]);
        let compress = scalefactors.compress().expect("marks that fit a field");
        scalefactors.part2_length(compress)
    }
}

/// The scalefactors of a granule of `block_type` from band `bound` up, in
/// the order of [`Scalefactors::values`], each with the lines it codes in
/// intensity stereo in the order they are sent: the highest band of each
/// window also those above it (see [`Positions::new`]).
fn coded_bands(
    block_type: BlockType,
    bound: usize,
    sampling_frequency: SamplingFrequency,
) -> impl Iterator<Item = (usize, Chain<Range<usize>, Range<usize>>)> {
    let bands = scalefactor_bands::coding_bands(block_type, sampling_frequency);
    let windows = scalefactor_bands::windows(block_type);
    let highest = bands.len() / windows - 1;
    (bound * windows..bands.len()).map(move |index| {
        let (sfb, window) = (index / windows, index % windows);
        let above = if sfb == highest {
            scalefactor_bands::lines_above_bands(block_type, sampling_frequency, window)
        } else {
            0..0
        };
        (index, (bands[index].start..bands[index].end()).chain(above))
    })
}

fn energy(lines: impl Iterator<Item = f64>) -> f64 {
    lines.map(|x| x * x).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitWriter;
    use crate::bitstream::binary_digits;
    use crate::scalefactor_bands::LONG_BANDS;
    use crate::stereo::middle_side;

    /// The Layer I and II bounds, subbands 16, 12, 8 and 4, start at lines
    /// 288, 216, 144 and 72 of a long block, and 96, 72, 48 and 24 of each
    /// window of a short one; each step is the first band starting there or
    /// above (Table B.8, and Table B.2 of ISO/IEC 13818-3).
    #[test]
    fn intensity_bounds_follow_the_layer_i_and_ii_subbands() {
        use SamplingFrequency::*;
        let cases = [
            (Hz16000, [18, 16, 14, 10], [10, 9, 7, 5]),
            (Hz22050, [18, 16, 14, 10], [10, 9, 8, 5]),
            (Hz24000, [18, 16, 14, 10], [10, 9, 7, 5]),
            (Hz32000, [18, 17, 15, 12], [10, 9, 8, 6]),
            (Hz44100, [19, 18, 16, 12], [11, 10, 8, 6]),
            (Hz48000, [20, 18, 16, 12], [11, 10, 8, 6]),
        ];
        for (fs, long, short) in cases {
            assert_eq!(bounds(BlockType::Normal, fs), long, "{fs:?}");
            assert_eq!(bounds(BlockType::Short, fs), short, "{fs:?}");
        }
    }

    /// `nint(12 / pi * arctan(sqrt(L / R)))`: amplitudes 2 to 1 lie at
    /// 63.4 degrees, nearest position 4 (60 degrees), and 1 to 2 at 26.6
    /// degrees, nearest 2 (30 degrees).
    #[test]
    fn positions_follow_the_energy_ratio() {
        assert_eq!(position(1.0, 1.0), 3);
        assert_eq!(position(4.0, 1.0), 4);
        assert_eq!(position(1.0, 4.0), 2);
        assert_eq!(position(1.0, 0.0), 6);
        assert_eq!(position(0.0, 1.0), 0);
        assert_eq!(position(0.0, 0.0), 3);
    }

    /// From the bound up, the left channel sends the sum and the right
    /// zero; below it the lines sent are left as they are. The highest
    /// band's position counts the lines above it.
    #[test]
    fn intensity_sends_the_sum_from_the_bound() {
        let fs = SamplingFrequency::Hz44100;
        let bands = scalefactor_bands::long_bands(fs);
        let left: [f64; GRANULE_LINES] = std::array::from_fn(|i| 1.0 + (i % 3) as f64);
        let mut right = [0.5; GRANULE_LINES];
        // The highest band is silent; the lines above it are in the left
        // channel alone.
        right[bands[20].start..].fill(0.0);
        let mut left_top = left;
        left_top[bands[20].start..bands[20].end()].fill(0.0);
        let positions = Positions::new(&left_top, &right, BlockType::Normal, 12, fs);
        assert_eq!(positions.first_scalefactor(), 12);
        assert_eq!(positions.positions[20], 6);
        let (mut coded_left, mut coded_right) = middle_side::matrix(&left_top, &right);
        let (middle, side) = (coded_left, coded_right);
        positions.code(&left_top, &right, [&mut coded_left, &mut coded_right]);
        let start = bands[12].start;
        assert_eq!(coded_left[..start], middle[..start]);
        assert_eq!(coded_right[..start], side[..start]);
        for i in start..GRANULE_LINES {
            assert_eq!(coded_left[i], left_top[i] + right[i]);
            assert_eq!(coded_right[i], 0.0);
        }
    }

    /// The zero part of the right channel sends the positions from the
    /// bound and 7 below it; the bands up to the last nonzero value keep
    /// their scalefactors.
    #[test]
    fn the_zero_part_is_marked() {
        let fs = SamplingFrequency::Hz44100;
        let bands = scalefactor_bands::long_bands(fs);
        let left: [f64; GRANULE_LINES] = std::array::from_fn(|i| 1.0 + (i % 3) as f64);
        let positions = Positions::new(&left, &[0.5; GRANULE_LINES], BlockType::Normal, 12, fs);
        let mut values = [0; GRANULE_LINES];
        values[bands[5].start + 1] = -2;
        let mut scalefactors = Scalefactors::default();
        scalefactors.values_mut()[2] = 9;
        positions.mark(&mut scalefactors, &values);
        let marked = scalefactors.values();
        assert_eq!(marked[..6], [0, 0, 9, 0, 0, 0]);
        assert_eq!(marked[12..], positions.positions[12..]);
        assert_eq!(
            scalefactors.intensity,
            Some(ZeroPart {
                start: [6, 0, 0],
                bound: 12
            })
        );
        // Bands 0 to 10 in 4 bits, 11 to 20 in 3: 9, then the marks.
        let compress = scalefactors.compress().unwrap();
        let mut w = BitWriter::new();
        scalefactors.write(compress, &mut w);
        let bits = binary_digits(&w.into_bytes());
        assert_eq!(&bits[..4 * 6], "000000001001000000000000");
        assert_eq!(&bits[4 * 6..4 * 11], "01110111011101110111");
        assert_eq!(&bits[4 * 11..4 * 11 + 3], "111");

        let mut scalefactors = Scalefactors::default();
        positions.mark(&mut scalefactors, &[0; GRANULE_LINES]);
        assert_eq!(scalefactors.intensity.unwrap().start, [0; SHORT_WINDOWS]);
        // Marks in 3 bits for all 21 bands.
        assert_eq!(positions.silent_bits(), 3 * LONG_BANDS);
    }

    /// The right channel of a short block at the level of the left in its
    /// first window, at half in its second and silent in its third: each
    /// window has its own positions, 3, 4 and 6. Band 11 of the second
    /// window is silent in both channels, so its position is found from the
    /// window's lines above it alone. From the bound up, in every window,
    /// the left channel sends the sum and the right zero.
    #[test]
    fn short_blocks_have_positions_in_each_window() {
        let fs = SamplingFrequency::Hz44100;
        let bands = scalefactor_bands::coding_bands(BlockType::Short, fs);
        let mut left: [f64; GRANULE_LINES] = std::array::from_fn(|i| 1.0 + (i % 3) as f64);
        let mut right = left;
        for window in 0..SHORT_WINDOWS {
            let above = scalefactor_bands::lines_above_bands(BlockType::Short, fs, window);
            let lines = (window..bands.len()).step_by(SHORT_WINDOWS);
            let lines = lines.flat_map(|index| bands[index].start..bands[index].end());
            for i in lines.chain(above) {
                right[i] *= [1.0, 0.5, 0.0][window];
            }
        }
        let band_11 = bands[3 * 11 + 1];
        left[band_11.start..band_11.end()].fill(0.0);
        right[band_11.start..band_11.end()].fill(0.0);

        let positions = Positions::new(&left, &right, BlockType::Short, 8, fs);
        assert_eq!(positions.first_scalefactor(), 24);
        assert_eq!(
            positions.first_line(),
            3 * scalefactor_bands::short_bands(fs)[8].start
        );
        for index in 24..bands.len() {
            assert_eq!(positions.positions[index], [3, 4, 6][index % 3], "{index}");
        }
        let (mut coded_left, mut coded_right) = (left, right);
        positions.code(&left, &right, [&mut coded_left, &mut coded_right]);
        let start = positions.first_line();
        assert_eq!(coded_left[..start], left[..start]);
        assert_eq!(coded_right[..start], right[..start]);
        for i in start..GRANULE_LINES {
            assert_eq!(coded_left[i], left[i] + right[i]);
            assert_eq!(coded_right[i], 0.0);
        }
    }

    /// Each window of a short block has its own zero part: the bands of a
    /// window above its last nonzero value are marked with 7 up to the
    /// bound, and the others keep their scalefactors.
    #[test]
    fn short_blocks_mark_the_zero_part_of_each_window() {
        let fs = SamplingFrequency::Hz44100;
        let bands = scalefactor_bands::coding_bands(BlockType::Short, fs);
        let left: [f64; GRANULE_LINES] = std::array::from_fn(|i| 1.0 + (i % 3) as f64);
        let positions = Positions::new(&left, &[0.5; GRANULE_LINES], BlockType::Short, 8, fs);
        // Nonzero in band 5 of the first window and band 2 of the third.
        let mut values = [0; GRANULE_LINES];
        values[bands[3 * 5].start] = 1;
        values[bands[3 * 2 + 2].end() - 1] = -1;
        let mut scalefactors = Scalefactors::new(BlockType::Short, Version::Mpeg1);
        // Band 1 in the first and second windows.
        scalefactors.values_mut()[3] = 9;
        scalefactors.values_mut()[4] = 9;
        positions.mark(&mut scalefactors, &values);
        assert_eq!(
            scalefactors.intensity,
            Some(ZeroPart {
                start: [6, 0, 3],
                bound: 8
            })
        );
        assert!(scalefactors.marks_any());
        assert_eq!(scalefactors.values()[3..5], [9, 0]);
        assert_eq!(scalefactors.values()[24..], positions.positions[24..]);
        // Bands 0 to 5 in 4 bits, 6 to 11 in 3.
        let compress = scalefactors.compress().unwrap();
        let mut w = BitWriter::new();
        scalefactors.write(compress, &mut w);
        let bits = binary_digits(&w.into_bytes());
        let sent = |index: usize| {
            let at = if index < 18 {
                4 * index
            } else {
                72 + 3 * (index - 18)
            };
            let width = if index < 18 { 4 } else { 3 };
            &bits[at..at + width]
        };
        let expected = [
            (3, "1001"),
            (4, "0111"),
            (8, "0000"),
            (11, "0111"),
            (15, "0000"),
            (16, "0111"),
            (18, "111"),
            (23, "111"),
        ];
        for (index, value) in expected {
            assert_eq!(sent(index), value, "scalefactor {index}");
        }
        for index in 24..bands.len() {
            let position = format!("{:03b}", positions.positions[index]);
            assert_eq!(sent(index), position, "scalefactor {index}");
        }

        // Marks in 3 bits for all 36 scalefactors.
        assert_eq!(positions.silent_bits(), 3 * 36);
    }

    /// At the lower sampling frequencies each step of position halves the
    /// energy of the quieter channel: the louder channel at twice the
    /// amplitude of the other is 2 steps, positions 4 (left) and 3 (right).
    #[test]
    fn lsf_positions_step_by_i0() {
        assert_eq!(lsf_position(1.0, 1.0), 0);
        assert_eq!(lsf_position(4.0, 1.0), 4);
        assert_eq!(lsf_position(1.0, 4.0), 3);
        assert_eq!(lsf_position(2.0, 1.0), 2);
        assert_eq!(lsf_position(1.0, 2.0), 1);
        // Nearest in steps: 2^1.4 is 1 step, 2^1.6 is 2.
        assert_eq!(lsf_position(1.4f64.exp2(), 1.0), 2);
        assert_eq!(lsf_position(1.6f64.exp2(), 1.0), 4);
        assert_eq!(lsf_position(1.0, 0.0), 14);
        assert_eq!(lsf_position(0.0, 1.0), 13);
        assert_eq!(lsf_position(40f64.exp2(), 1.0), 14);
        // Never position 7, the left channel 4 steps quieter: 3 or 5.
        assert_eq!(lsf_position(1.0, 3.6f64.exp2()), 5);
        assert_eq!(lsf_position(1.0, 3.9f64.exp2()), 5);
        assert_eq!(lsf_position(1.0, 4.1f64.exp2()), 9);
        assert_eq!(lsf_position(1.0, 4.4f64.exp2()), 9);
        assert_eq!(lsf_position(4f64.exp2(), 1.0), 8);
        assert_eq!(lsf_position(0.0, 0.0), 0);
        for (position, expected) in [(0, (1.0, 1.0)), (3, (0.5, 1.0)), (4, (1.0, 0.5))] {
            let (k_l, k_r) = shares(Version::Mpeg2, position);
            assert!((k_l - expected.0).abs() < 1e-12, "{position}");
            assert!((k_r - expected.1).abs() < 1e-12, "{position}");
        }
    }

    /// At the lower sampling frequencies the sum is scaled so that the
    /// channels decode with their energy: alike channels send either, and a
    /// tone with the right channel at half the left sends the left.
    #[test]
    fn lsf_intensity_sends_the_louder_channel() {
        let fs = SamplingFrequency::Hz22050;
        let start = scalefactor_bands::long_bands(fs)[10].start;
        let tone: [f64; GRANULE_LINES] = std::array::from_fn(|i| ((i * 7) % 5) as f64 - 2.0);
        let half = tone.map(|x| 0.5 * x);
        for (right, position) in [(tone, 0), (half, 4)] {
            let positions = Positions::new(&tone, &right, BlockType::Normal, 10, fs);
            assert!(positions.positions[10..].iter().all(|&p| p == position));
            let (mut coded_left, mut coded_right) = (tone, right);
            positions.code(&tone, &right, [&mut coded_left, &mut coded_right]);
            for i in start..GRANULE_LINES {
                assert!((coded_left[i] - tone[i]).abs() < 1e-12, "{position}, {i}");
                assert_eq!(coded_right[i], 0.0);
            }
            assert_eq!(coded_left[..start], tone[..start]);
        }
    }

    /// At the lower sampling frequencies a mark is the largest value of its
    /// field, so a right channel all zero needs a bit for each band below
    /// the bound, and its positions a field that sends them below the mark.
    #[test]
    fn lsf_marks_take_a_field_of_at_least_one_bit() {
        let fs = SamplingFrequency::Hz22050;
        let tone: [f64; GRANULE_LINES] = std::array::from_fn(|i| ((i * 7) % 5) as f64 - 2.0);
        let alike = Positions::new(&tone, &tone, BlockType::Normal, 14, fs);
        // Bands 0 to 13 marked in 1 bit; positions 0 in bands 14 to 20, the
        // third partition of int_scalefac_compress below 180, in no bits.
        assert_eq!(alike.silent_bits(), 14);
        let half = tone.map(|x| 0.5 * x);
        let panned = Positions::new(&tone, &half, BlockType::Normal, 14, fs);
        // Position 4 needs a field of 3 bits, below its mark 7.
        assert_eq!(panned.silent_bits(), 7 + 7 + 3 * 7);
    }

    /// In the middle position each channel gets half the sum, and a quarter
    /// of its noise energy; at the ends only one channel limits the noise.
    #[test]
    fn the_sum_is_allowed_what_keeps_both_channels_within_theirs() {
        let positions = Positions {
            block_type: BlockType::Normal,
            bound: 10,
            positions: (0..LONG_BANDS)
                .map(|sb| match sb {
                    10 => 3,
                    11 => 6,
                    12 => 0,
                    _ => 2,
                })
                .collect(),
            sampling_frequency: SamplingFrequency::Hz44100,
        };
        assert!((positions.sum_distortion(10, 2.0, 1.0) - 4.0).abs() < 1e-12);
        assert_eq!(positions.sum_distortion(11, 2.0, 1.0), 2.0);
        assert_eq!(positions.sum_distortion(12, 2.0, 1.0), 1.0);
        // A silent channel sets no limit.
        assert!((positions.sum_distortion(10, 0.0, 1.0) - 4.0).abs() < 1e-12);
        assert_eq!(positions.sum_distortion(10, 0.0, 0.0), 0.0);
        // Position 2, 30 degrees: the right channel gets cos / (sin + cos)
        // of the sum, the larger share, and so limits it.
        let (sin, cos) = (PI / 6.0).sin_cos();
        let expected = 1.0 * ((sin + cos) / cos).powi(2);
        assert!((positions.sum_distortion(13, 1.0, 1.0) - expected).abs() < 1e-12);
        // At the lower sampling frequencies position 4 gives the right
        // channel half the lines sent, and a quarter of their noise.
        let lsf = Positions {
            sampling_frequency: SamplingFrequency::Hz22050,
            positions: vec![4; LONG_BANDS],
            ..positions
        };
        assert!((lsf.sum_distortion(10, 2.0, 1.0) - 2.0).abs() < 1e-12);
        assert!((lsf.sum_distortion(10, 2.0, 0.25) - 1.0).abs() < 1e-12);
    }
}
