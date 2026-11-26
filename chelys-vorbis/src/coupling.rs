// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Square polar mapping: the two channels of a stereo stream coded as a
//! magnitude and an angle, which a decoder maps back to them (clause
//! 4.3.5).
//!
//! The specification defines only the inverse, which acts on the residue
//! vectors, before each channel's floor scales its own; the forward
//! mapping here is chelys's own, the one that inverts it exactly, value
//! for value. So is the matching of the two channels' floors where the
//! channels are alike (see [`match_floors`]), which makes their residues
//! alike too.

use super::energy;
use super::floor::{self, Floor};

/// How much noise, as a multiple of each channel's allowance in a band,
/// [`narrow_angles`] may leave in the band, above [`PROTECTED_HZ`], where
/// bits are scarce: up to 3 times, about 4.8 dB, over the noise the
/// psychoacoustic model allows the band at the rate's offset. Where they
/// are plentiful, the multiple is [`LEAST_ANGLE_ALLOWANCE`] (see
/// [`angle_allowance`]).
///
/// chelys's own design, a deliberate excess over the model's mask: the
/// noise spent in making the channels alike buys more, in bits saved, than
/// the same noise spread over every line by a higher offset, as PEAQ grades
/// it at the same size. It is bounded in each channel and band, so that
/// neither channel's noise grows without limit where the other is louder,
/// and stays within the allowance itself below [`PROTECTED_HZ`].
///
/// Graded with PEAQ (GstPEAQ 0.6.1, black box) on the 51 tracks of the
/// tuning set at 44.1 kHz in stereo, each multiple with codebooks counted
/// for it, against the streams coupled without narrowing interpolated to
/// the same mean bitrate, the streams narrowed within the allowance itself,
/// 1, graded 0.01, 0.07 and 0.24 higher at 154, 127 and 90 kbit/s; and
/// within 2, 0.15, 0.30 and 0.64 higher at 133, 119 and 106. Interpolated
/// to 96.5, 128.4 and 159.2 kbit/s, the streams graded -2.437, -0.919 and
/// -0.488 unnarrowed, -1.653, -0.731 and -0.461 within 2, -1.547, -0.692
/// and -0.449 within 3, -1.471, -0.695 and -0.452 within 4, and -1.493,
/// -0.715 and -0.468 within 6; with [`PROTECTED_HZ`], -1.454, -0.677 and
/// -0.424 within 3. Within 3, with [`PROTECTED_HZ`], at 209 and 245
/// kbit/s, the streams graded 0.025 and 0.027 higher than unnarrowed.
pub const ANGLE_ALLOWANCE: f64 = 3.0;

/// The multiple of each channel's allowance that [`narrow_angles`] may
/// leave in a band where bits are plentiful (see [`angle_allowance`]).
///
/// Once the model weighed each band's noise (see the psychoacoustic
/// model's `BAND_EXPONENT`), graded as for [`ANGLE_ALLOWANCE`] at noise
/// offsets matched to each bitrate, the streams graded -0.931, -0.904 and
/// -0.922 within 2, 3 and 5 at 96.5 kbit/s; -0.553, -0.477 and -0.476
/// within 1, 2 and 3 at 128.4; and -0.310, -0.295 and -0.320 within 1, 2
/// and 3 at 159.2. PEAQ's `ADBB`, which measures the distortion of the
/// frames it finds distorted against the steps above threshold of all of
/// them, fell from 1.05 to 0.82 within 2 at 159.2: each line narrowed
/// changes the level of each channel's band a little, and where bits are
/// plentiful, the bits that saves buy less than the noise costs.
///
/// Once the energy of each band was balanced (see [`energy::balance`]),
/// which leaves the narrowed lines as they are, the streams graded -0.464
/// and -0.264 within 2 at 128.4 and 159.2 kbit/s, -0.465 and -0.253 within
/// 1, and -0.465 and -0.257 within 1/2; within 1, PEAQ's `ADBB` fell from
/// 0.45 to 0.31 at 159.2, and its `AvgModDiff1B` from 14.8 and 11.7 to
/// 14.6 and 11.5.
pub const LEAST_ANGLE_ALLOWANCE: f64 = 1.0;

/// The multiple of each channel's allowance that [`narrow_angles`] may
/// leave in a band above [`PROTECTED_HZ`], where bits are as scarce as
/// `scarcity`, from 0 where they are plentiful to 1 where they are
/// scarce: from [`LEAST_ANGLE_ALLOWANCE`] to [`ANGLE_ALLOWANCE`].
pub fn angle_allowance(scarcity: f64) -> f64 {
    LEAST_ANGLE_ALLOWANCE + (ANGLE_ALLOWANCE - LEAST_ANGLE_ALLOWANCE) * scarcity.clamp(0.0, 1.0)
}

/// The frequency below which [`narrow_angles`] leaves no more noise in a
/// band than each channel's allowance.
///
/// Below about 1.5 kHz, hearing follows the difference in phase between
/// the ears, by which noise whose phase between the channels is unlike
/// the signal's may be heard where the model, which takes each channel on
/// its own, finds it masked; and narrowing angles makes the channels' lines
/// alike in phase. PEAQ takes each channel on its own too, and does not
/// measure it, but grades the limit no lower: graded as for
/// [`ANGLE_ALLOWANCE`], the streams narrowed within 3 times the allowance
/// above 1.5 kHz and within the allowance below graded -1.470, -0.635 and
/// -0.423 at 95.9, 130.1 and 159.3 kbit/s; with 3 kHz for 1.5, -1.479,
/// -0.633 and -0.418 at 96.3, 130.2 and 159.0; with no narrowing below 1.5
/// kHz, -1.444, -0.625 and -0.418 at 96.5, 130.8 and 159.9; and within 4
/// times above 1.5 kHz, -1.374, -0.620 and -0.421 at 96.9, 130.5 and
/// 159.6.
pub const PROTECTED_HZ: f64 = 1500.0;

/// The most, in dB, by which [`narrow_angles`] may lower the ratio of the
/// energy of the two channels' difference to that of their sum in a band,
/// the side to the mid, as decoded, below the lines' own, where bits are
/// plentiful; as they grow scarce, the more (see [`side_loss_db`]).
///
/// chelys's own design. Setting two channels' values of a line alike makes
/// the line decode in phase in both, each scaled by its own floor, and so
/// narrows the stereo image, which neither the model's allowance nor PEAQ,
/// which takes each channel on its own, measures. With the floors matched
/// (see [`match_floors`]), more lines are set alike, and with more noise
/// allowed in the treble (see the psychoacoustic model's `TREBLE_HZ`), more
/// still there. Measured on the 51 tracks of the tuning set at 44.1 kHz in
/// stereo, decoded by ffmpeg (black box), at 128.4 kbit/s, the decoded
/// ratio from 16 to 18 kHz lay 4.0 dB below the lines' on average before
/// the floors were matched, and more than 6 dB below on 11 of the tracks;
/// 4.6 dB and 16 tracks with them matched; and 5.0 dB and 18 tracks with
/// the treble's noise too. So bounded, 3.4 dB and 5 tracks, and from 14 to
/// 16 kHz 2.6 dB below on average, against 2.1 before; at 159.2 kbit/s,
/// 1.4 and 1.1 dB below, against 1.8 and 0.9 before; and at 96.5 kbit/s,
/// where [`side_loss_db`] bounds little, 5.4 and 4.0 dB below, against 6.3
/// and 3.7 before. Graded with PEAQ (GstPEAQ 0.6.1, black box) at noise
/// offsets matched to 96.5, 128.4 and 159.2 kbit/s, the streams graded
/// -0.847, -0.410 and -0.161 unbounded, -0.931, -0.420 and -0.157 bounded
/// at 6 dB at every rate, and -0.857, -0.413 and -0.157 bounded as
/// [`side_loss_db`] widens the bound.
pub const SIDE_LOSS_DB: f64 = 6.0;

/// The most, in dB, by which [`narrow_angles`] may lower a band's ratio of
/// side to mid energy where bits are as scarce as `scarcity`, from 0 where
/// they are plentiful to 1 where they are scarce: [`SIDE_LOSS_DB`] divided
/// by what is left of 1 by the scarcity, without bound where it is 1, as
/// where bits are scarce the bits narrowing saves count for most.
pub fn side_loss_db(scarcity: f64) -> f64 {
    SIDE_LOSS_DB / (1.0 - scarcity.clamp(0.0, 1.0))
}

/// The lines and bounds within which [`narrow_angles`] narrows a pair's
/// angles.
#[derive(Clone, Copy, Debug)]
pub struct Narrowing {
    /// The lines below [`PROTECTED_HZ`], the first of the block, where a
    /// band may take its allowance itself.
    pub protected: usize,
    /// The multiple of its allowance a band may take above them (see
    /// [`angle_allowance`]).
    pub allowance: f64,
    /// The most by which a band's ratio of side to mid energy may fall, in
    /// dB (see [`side_loss_db`]).
    pub side_loss_db: f64,
}

/// The least correlation between the two channels' lines about a post of
/// the floor at which [`match_floors`] matches their floors there.
///
/// Graded with PEAQ (GstPEAQ 0.6.1, black box) on the 51 tracks of the
/// tuning set at 44.1 kHz in stereo, at noise offsets matched to 96.5,
/// 128.4 and 159.2 kbit/s, the streams graded -0.890, -0.451 and -0.230
/// unmatched, -0.863, -0.431 and -0.205 matched from a correlation of 0.8,
/// and -0.853, -0.428 and -0.203 from 0.5, higher on 43, 44 and 47 of the
/// 51 tracks; PEAQ's `AvgModDiff1B` fell from 18.8, 14.3 and 11.2 to 18.1,
/// 13.7 and 10.8, and its `RmsNoiseLoudB` from 0.392, 0.280 and 0.213 to
/// 0.373, 0.267 and 0.204. Setting the two floors alike where their Y
/// values were within one or two of each other, whatever the levels,
/// graded lower than matching them.
pub const MATCHED_CORRELATION: f64 = 0.5;

/// Matches the floors of a stereo pair to the ratio of the channels'
/// levels where the channels are alike: the Y values `targets` that the
/// model sets for each channel, in the list order of `floor`, whose lines
/// are `spectra`.
///
/// chelys's own design. A channel's residue is its lines divided by its
/// floor (see [`residue`](super::residue)), and so the two channels of a
/// sound panned between them have alike residues, which cost the fewest
/// bits coupled, where their floors differ as their levels do. But each
/// floor is set for its own channel's noise, in Y values about 1.6 dB
/// apart, and the two differ by chance, by a Y value or two, so that the
/// residues differ by as much at their larger values; [`narrow_angles`]
/// makes them alike again only within the noise each channel may take, and
/// the noise it adds changes their levels from one block to the next.
///
/// So where the lines about a post correlate by at least
/// [`MATCHED_CORRELATION`] between the channels, the two Y values are set
/// the whole Y values apart nearest the ratio of the channels' energies
/// there: one stays as the model sets it, and the other is lowered to lie
/// that far from it. Neither floor is raised, and neither channel takes
/// more noise than the model allows it.
pub fn match_floors(targets: [&mut [i32]; 2], spectra: [&[f64]; 2], floor: &Floor) {
    let [left, right] = targets;
    let step_db = floor::y_step_db();
    for (i, lines) in floor.post_lines(spectra[0].len()).into_iter().enumerate() {
        let (mut el, mut er, mut cross) = (0.0, 0.0, 0.0);
        for k in lines {
            let (l, r) = (spectra[0][k], spectra[1][k]);
            el += l * l;
            er += r * r;
            cross += l * r;
        }
        if el == 0.0 || er == 0.0 || cross < MATCHED_CORRELATION * (el * er).sqrt() {
            continue;
        }
        let apart = (10.0 * (er / el).log10() / step_db).round() as i32;
        let l = left[i].min(right[i] - apart);
        let r = l + apart;
        if (0..floor::RANGE).contains(&l) && (0..floor::RANGE).contains(&r) {
            (left[i], right[i]) = (l, r);
        }
    }
}

/// Maps the residues of a stereo pair, channel 0 and channel 1, in place,
/// to the magnitude and angle that clause 4.3.5 maps back to them: each
/// channel's residue quantized to its floor `curves`, of its lines
/// `spectra`, which may take the noise `noises`, and which lie in the
/// psychoacoustic model's partitions `partitions`, one for each line.
///
/// Where both channels are used, their angles are first narrowed where
/// that leaves each channel's noise in each band within the multiple of
/// its allowance that `narrowing` gives the band, and the band's image
/// within its bound (see [`narrow_angles`]). Then the energy of each used
/// channel's first `kept` lines is balanced toward `share` of their
/// energy, as no more than `limit` (see [`energy::balance`]), but on the
/// lines whose values the two channels share, whose angles are 0; the
/// balance may take a band past its allowance below [`PROTECTED_HZ`] too.
/// Last, the energy is kept (see [`energy::keep`]), the left channel's
/// first, each preferring the lines that the other codes as it would,
/// which adds noise above [`PROTECTED_HZ`] and none beyond a band's
/// allowance below. Both come after the narrowing, which would code the
/// lines that keeping adds as 0 again, as that makes less noise, and would
/// take back much of the energy that the balance gives.
///
/// Neither is held to the narrowing's bound on the image (see
/// [`SIDE_LOSS_DB`]), and either may take a band past it: on the tuning
/// set at 44.1 kHz in stereo, the balance took 1.4% of the bands that the
/// narrowing left within it past it at -9 dB, and 1.1% at -7 dB, by 2.7 to
/// 3.5 dB at the median, about 11 dB at the 90th percentile and at most
/// 58 dB, a third of them below 1.5 kHz and a third in short blocks; the
/// keeping of energy, about 0.06%. Held to the bound too, both rejecting
/// what would take a band below the bound or below the ratio the
/// narrowing left it with, the streams graded 0.006, 0.004 and 0.003 lower
/// at noise offsets matched to 96.5, 128.4 and 159.2 kbit/s, on every
/// track.
///
/// Spec reading: a channel whose residue is all zeros is coded as unused,
/// its floor unused. Clause 4.3.3 then unsets its `no_residue` flag, as it
/// is coupled with a channel whose floor is used, so that its residue is
/// decoded all the same, and clause 4.3.5 decouples the two. But an unused
/// floor "indicates that that final output vector is all-zero values",
/// which clause 4.3.3 says is to be remembered "at floor curve synthesis
/// time", and clause 4.3.6 multiplies each channel's residue by its floor
/// curve, of which an unused floor has none (clause 7.2.3: "this floor
/// curve (and thus this channel) is unused"). The channel is taken to
/// decode to silence whatever its residue after decoupling, and its
/// residue is taken as the other's, so that every angle is 0, which takes
/// the fewest bits. Settled by decoding with ffmpeg (black box): a silent
/// channel beside a coded one decodes to silence, every sample of it, and a
/// channel silent in some blocks only decodes to silence there; the
/// conformance tests are
/// `a_silent_channel_beside_a_coded_one_decodes_to_silence` and
/// `a_channel_silent_in_some_blocks_decodes_to_silence_there`.
#[allow(clippy::too_many_arguments)]
pub fn couple(
    residues: [&mut Vec<i32>; 2],
    curves: [&[f32]; 2],
    spectra: [&[f64]; 2],
    noises: [&[f64]; 2],
    partitions: &[usize],
    narrowing: Narrowing,
    kept: usize,
    limit: i32,
    share: f64,
) {
    let [left, right] = residues;
    let used = [left.iter().any(|&v| v != 0), right.iter().any(|&v| v != 0)];
    let guard = |c: usize| energy::Protected {
        lines: narrowing.protected,
        noise: noises[c],
        partitions,
    };
    match used {
        [true, true] => {
            narrow_angles(
                [&mut *left, &mut *right],
                curves,
                spectra,
                noises,
                partitions,
                narrowing,
            );
            let alike: Vec<bool> = left.iter().zip(right.iter()).map(|(l, r)| l == r).collect();
            for (c, residue) in [&mut *left, &mut *right].into_iter().enumerate() {
                energy::balance(
                    residue,
                    spectra[c],
                    curves[c],
                    limit,
                    |k| alike[k],
                    kept,
                    share,
                );
            }
            energy::keep(left, spectra[0], curves[0], guard(0), kept, Some(right));
            energy::keep(right, spectra[1], curves[1], guard(1), kept, Some(left));
        }
        [true, false] => {
            energy::balance(left, spectra[0], curves[0], limit, |_| false, kept, share);
            energy::keep(left, spectra[0], curves[0], guard(0), kept, None);
            right.clone_from(left);
        }
        [false, true] => {
            energy::balance(right, spectra[1], curves[1], limit, |_| false, kept, share);
            energy::keep(right, spectra[1], curves[1], guard(1), kept, None);
            left.clone_from(right);
        }
        [false, false] => {}
    }
    for (l, r) in left.iter_mut().zip(right.iter_mut()) {
        (*l, *r) = square_polar(*l, *r);
    }
}

/// The magnitude and angle that clause 4.3.5 maps back to the residue
/// values `magnitude_channel` and `angle_channel` of the two channels: the
/// value of greater magnitude, the angle channel's where they are as great,
/// and their difference, signed as the magnitude, whose magnitude is at
/// most the sum of theirs. Where the two are equal, the angle is 0.
pub fn square_polar(magnitude_channel: i32, angle_channel: i32) -> (i32, i32) {
    let magnitude = if magnitude_channel.abs() > angle_channel.abs() {
        magnitude_channel
    } else {
        angle_channel
    };
    let angle = if magnitude > 0 {
        magnitude_channel - angle_channel
    } else {
        angle_channel - magnitude_channel
    };
    (magnitude, angle)
}

/// Narrows the angles of a pair's residues, `residues`, whose channels'
/// floors are `curves`, of the lines `spectra`, which may take the noise
/// `noises`, as [`couple`] gives them: sets the two channels' values of a
/// line alike, so that its angle is 0, where each channel's noise stays
/// within a multiple of its allowance.
///
/// A coupled pair costs fewest bits where its channels' values are alike,
/// as their angle is then 0 or near it; but each channel's values are
/// rounded on their own, and two channels much alike, but for a fraction
/// of a step, round to values a step apart. The noise of each channel is
/// accounted for on its own, as the channel decodes once decoupled: line
/// `k` coded as `v` decodes to `f v`, and makes the noise `(x - f v)^2`, in
/// a channel whose line is `x` and whose floor is `f` there. A band is a
/// partition of the psychoacoustic model, `partitions` giving each line's,
/// split at `narrowing`'s `protected` first line: the model allows each of
/// its lines the same share of the partition's threshold, which it takes
/// as masked together. In each band, each channel may take `narrowing`'s
/// `allowance` times the noise its lines are allowed, or below the
/// `protected` first line the allowance itself, less the noise its
/// rounding makes. Each line whose values differ may take the value
/// between them, either included, of the least noise in the two channels
/// together; the lines are taken in the order of that noise, the least
/// first, each where it fits the room left to both channels in its band,
/// and where it leaves the band's ratio of side to mid energy, as decoded,
/// no more than `side_loss_db` below the lines' own, or no lower than it
/// was (see [`SIDE_LOSS_DB`]).
///
/// A band whose lines are all coded as 0 in both channels decodes with
/// neither side nor mid, and is taken as within the bound: it is so coded
/// only where its noise fits the room left to it, and whether its image
/// is heard is then the noise budget's to say.
///
/// The values stay between the channels' own, and so within the residue's
/// reach.
fn narrow_angles(
    residues: [&mut Vec<i32>; 2],
    curves: [&[f32]; 2],
    spectra: [&[f64]; 2],
    noises: [&[f64]; 2],
    partitions: &[usize],
    narrowing: Narrowing,
) {
    let Narrowing {
        protected,
        allowance,
        side_loss_db,
    } = narrowing;
    let [left, right] = residues;
    let lines = left.len();
    let noise = |c: usize, k: usize, v: i32| {
        (spectra[c][k] - f64::from(curves[c][k]) * f64::from(v)).powi(2)
    };
    // Each line's band, and the noise each band may yet take in each
    // channel.
    let mut bands = Vec::with_capacity(lines);
    let mut room: Vec<[f64; 2]> = Vec::new();
    for k in 0..lines {
        if k == 0 || k == protected || partitions[k] != partitions[k - 1] {
            room.push([0.0; 2]);
        }
        bands.push(room.len() - 1);
        let multiple = if k < protected { 1.0 } else { allowance };
        let room = room.last_mut().expect("a band was pushed");
        room[0] += multiple * noises[0][k] - noise(0, k, left[k]);
        room[1] += multiple * noises[1][k] - noise(1, k, right[k]);
    }
    // For each line whose values differ, the value of the least noise, and
    // the noise it adds in each channel.
    let mut changes: Vec<(usize, i32, [f64; 2])> = (0..lines)
        .filter(|&k| left[k] != right[k])
        .map(|k| {
            let (l, r) = (left[k], right[k]);
            (l.min(r)..=l.max(r))
                .map(|v| {
                    (
                        k,
                        v,
                        [
                            noise(0, k, v) - noise(0, k, l),
                            noise(1, k, v) - noise(1, k, r),
                        ],
                    )
                })
                .min_by(|a, b| (a.2[0] + a.2[1]).total_cmp(&(b.2[0] + b.2[1])))
                .expect("a value between the two")
        })
        .collect();
    changes.sort_by(|a, b| (a.2[0] + a.2[1]).total_cmp(&(b.2[0] + b.2[1])));
    // Each band's energy of the channels' difference and of their sum, of
    // the lines and as decoded.
    let decoded = |c: usize, k: usize, v: i32| f64::from(curves[c][k]) * f64::from(v);
    let mut image = vec![Image::default(); room.len()];
    for k in 0..lines {
        let (x, y) = (spectra[0][k], spectra[1][k]);
        image[bands[k]].add(x, y, decoded(0, k, left[k]), decoded(1, k, right[k]));
    }
    let least = 10f64.powf(-side_loss_db / 10.0);
    for (k, v, added) in changes {
        let band = bands[k];
        if !(0..2).all(|c| added[c] <= 0.0 || added[c] <= room[band][c]) {
            continue;
        }
        let (old, new) = (
            (decoded(0, k, left[k]), decoded(1, k, right[k])),
            (decoded(0, k, v), decoded(1, k, v)),
        );
        let after = image[band].replaced(old, new);
        if !after.within(least, &image[band]) {
            continue;
        }
        image[band] = after;
        room[band][0] -= added[0];
        room[band][1] -= added[1];
        left[k] = v;
        right[k] = v;
    }
}

/// A band's energy of the difference of a pair's channels, the side, and
/// of their sum, the mid: of their lines, and as decoded.
#[derive(Clone, Copy, Debug, Default)]
struct Image {
    side: f64,
    mid: f64,
    decoded_side: f64,
    decoded_mid: f64,
}

impl Image {
    /// Adds the line whose channels are `x` and `y`, and decode to `dx` and
    /// `dy`.
    fn add(&mut self, x: f64, y: f64, dx: f64, dy: f64) {
        self.side += (x - y).powi(2);
        self.mid += (x + y).powi(2);
        self.decoded_side += (dx - dy).powi(2);
        self.decoded_mid += (dx + dy).powi(2);
    }

    /// The band with a line decoded as `old` decoded as `new` instead.
    /// The sums are kept from falling below 0 by rounding.
    fn replaced(&self, old: (f64, f64), new: (f64, f64)) -> Self {
        let side = self.decoded_side - (old.0 - old.1).powi(2) + (new.0 - new.1).powi(2);
        let mid = self.decoded_mid - (old.0 + old.1).powi(2) + (new.0 + new.1).powi(2);
        Self {
            decoded_side: side.max(0.0),
            decoded_mid: mid.max(0.0),
            ..*self
        }
    }

    /// Whether the band's decoded ratio of side to mid is at least `least`
    /// times its lines' own, or no lower than `before`'s; always where
    /// `least` is 0, without bound.
    fn within(&self, least: f64, before: &Self) -> bool {
        least == 0.0
            || self.decoded_side * self.mid >= least * self.side * self.decoded_mid
            || self.decoded_side * before.decoded_mid >= before.decoded_side * self.decoded_mid
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use super::*;

    /// Clause 4.3.5: the residue values of the magnitude channel and the
    /// angle channel that `magnitude` and `angle` decode to.
    fn decouple(magnitude: i32, angle: i32) -> (i32, i32) {
        if magnitude > 0 {
            if angle > 0 {
                (magnitude, magnitude - angle)
            } else {
                (magnitude + angle, magnitude)
            }
        } else if angle > 0 {
            (magnitude, magnitude + angle)
        } else {
            (magnitude - angle, magnitude)
        }
    }

    #[test]
    fn decoupling_inverts_square_polar_mapping() {
        for l in -40..=40 {
            for r in -40..=40 {
                let (m, a) = square_polar(l, r);
                assert_eq!(decouple(m, a), (l, r), "{l}, {r}");
                assert_eq!(m.abs(), l.abs().max(r.abs()));
                assert!(a.abs() <= l.abs() + r.abs());
            }
        }
    }

    #[test]
    fn equal_values_have_no_angle() {
        for v in -40..=40 {
            assert_eq!(square_polar(v, v), (v, 0));
        }
    }

    /// The noise of a channel's lines `range` coded as `residue`.
    fn noise_of(spectrum: &[f64], curve: &[f32], residue: &[i32], range: Range<usize>) -> f64 {
        range
            .map(|k| (spectrum[k] - f64::from(curve[k]) * f64::from(residue[k])).powi(2))
            .sum()
    }

    /// Two channels much alike, of lines a fraction of a step apart, each
    /// line allowed a little more noise than rounding makes, 1/12.
    fn alike(lines: usize) -> ([Vec<f64>; 2], Vec<f32>, Vec<f64>) {
        let left: Vec<f64> = (0..lines)
            .map(|k| ((k * 37 % 23) as f64 - 11.0) * 0.7)
            .collect();
        let right: Vec<f64> = (left.iter().enumerate())
            .map(|(k, &x)| x + ((k * 13 % 7) as f64 - 3.0) * 0.15)
            .collect();
        let curve = vec![1.0; lines];
        let noise = vec![0.086; lines];
        ([left, right], curve, noise)
    }

    fn round(spectrum: &[f64]) -> Vec<i32> {
        spectrum.iter().map(|x| x.round() as i32).collect()
    }

    /// Narrows `spectra`, rounded, in partitions of 16 lines, and checks that
    /// each channel's noise stays within the allowance in each partition's
    /// lines below `protected` and within [`ANGLE_ALLOWANCE`] times it in
    /// those above, or within what rounding made, and that each line narrowed
    /// takes a value between the two. Gives the lines narrowed.
    fn narrow_and_check(
        spectra: &[Vec<f64>; 2],
        curve: &[f32],
        noise: &[f64],
        protected: usize,
    ) -> usize {
        let lines = curve.len();
        let partitions: Vec<usize> = (0..lines).map(|k| k / 16).collect();
        let mut residues = [round(&spectra[0]), round(&spectra[1])];
        let rounded = residues.clone();
        let [left, right] = &mut residues;
        narrow_angles(
            [left, right],
            [curve, curve],
            [&spectra[0], &spectra[1]],
            [noise, noise],
            &partitions,
            Narrowing {
                protected,
                allowance: ANGLE_ALLOWANCE,
                side_loss_db: f64::INFINITY,
            },
        );
        for start in (0..lines).step_by(16) {
            let split = protected.clamp(start, start + 16);
            for (range, multiple) in [(start..split, 1.0), (split..start + 16, ANGLE_ALLOWANCE)] {
                let allowed: f64 = multiple * noise[range.clone()].iter().sum::<f64>();
                for c in 0..2 {
                    let before = noise_of(&spectra[c], curve, &rounded[c], range.clone());
                    let after = noise_of(&spectra[c], curve, &residues[c], range.clone());
                    assert!(
                        after <= allowed.max(before) + 1e-9,
                        "channel {c}, lines {range:?}: {after} against {allowed}"
                    );
                }
            }
        }
        let mut narrowed = 0;
        for k in 0..lines {
            let (l, r) = (residues[0][k], residues[1][k]);
            let (l0, r0) = (rounded[0][k], rounded[1][k]);
            if (l, r) != (l0, r0) {
                assert_eq!(l, r);
                assert!(l0.min(r0) <= l && l <= l0.max(r0));
                narrowed += 1;
            }
        }
        narrowed
    }

    #[test]
    fn narrowed_angles_leave_each_band_within_its_allowance() {
        let lines = 256;
        let (spectra, curve, noise) = alike(lines);
        let counts: Vec<usize> = [0, 100, lines]
            .into_iter()
            .map(|protected| narrow_and_check(&spectra, &curve, &noise, protected))
            .collect();
        // More room is left above the protected lines than below.
        assert!(
            counts[0] > counts[1] && counts[1] > counts[2] && counts[2] > 0,
            "{counts:?}"
        );
    }

    /// The lines below the protected one take no room from those above it,
    /// in the same partition: here, the left channel's first 8 lines, a
    /// step from the right's, would all be narrowed with the room of the 8
    /// above, coded alike already, for noise far beyond their allowance.
    #[test]
    fn the_protected_lines_take_no_room_from_those_above() {
        let left: Vec<f64> = (0..16)
            .map(|k| if k < 8 { 0.5 + 0.02 * k as f64 } else { 3.0 })
            .collect();
        let right: Vec<f64> = (0..16).map(|k| if k < 8 { -0.49 } else { 3.0 }).collect();
        narrow_and_check(&[left, right], &[1.0; 16], &[0.1; 16], 8);
    }

    /// Two channels of wide bands of many lines a little unlike, on unlike
    /// floors, each line allowed far more noise than setting it alike adds:
    /// unbounded, every line is narrowed; bounded, each band keeps its
    /// decoded ratio of side to mid within the bound, or as rounding left
    /// it, and some band comes close to the bound, so that the bound sets
    /// how far each is narrowed.
    #[test]
    fn narrowing_keeps_each_bands_side_within_its_bound() {
        let (lines, width) = (512, 128);
        let left: Vec<f64> = (0..lines)
            .map(|k| 3.0 + (k * 7 % 13) as f64 / 6.5)
            .collect();
        let right: Vec<f64> = (left.iter().enumerate())
            .map(|(k, &x)| x + 0.9 * ((k * 37 % 29) as f64 / 14.0 - 1.0))
            .collect();
        let curves: [Vec<f32>; 2] = [
            vec![1.0; lines],
            (0..lines)
                .map(|k| if k % 3 == 0 { 1.1 } else { 0.95 })
                .collect(),
        ];
        let spectra = [left, right];
        let noise = vec![100.0; lines];
        let partitions: Vec<usize> = (0..lines).map(|k| k / width).collect();
        let rounded: [Vec<i32>; 2] = [0, 1].map(|c| {
            (spectra[c].iter().zip(&curves[c]))
                .map(|(&x, &f)| (x / f64::from(f)).round() as i32)
                .collect()
        });
        let ratio = |residues: &[Vec<i32>; 2], band: Range<usize>| {
            let (mut side, mut mid) = (0.0, 0.0);
            for k in band {
                let l = f64::from(curves[0][k]) * f64::from(residues[0][k]);
                let r = f64::from(curves[1][k]) * f64::from(residues[1][k]);
                side += (l - r).powi(2);
                mid += (l + r).powi(2);
            }
            side / mid
        };
        let narrow = |side_loss_db: f64| {
            let mut residues = rounded.clone();
            let [left, right] = &mut residues;
            narrow_angles(
                [left, right],
                [&curves[0], &curves[1]],
                [&spectra[0], &spectra[1]],
                [&noise, &noise],
                &partitions,
                Narrowing {
                    protected: 0,
                    allowance: ANGLE_ALLOWANCE,
                    side_loss_db,
                },
            );
            residues
        };
        let unbounded = narrow(f64::INFINITY);
        assert_eq!(unbounded[0], unbounded[1]);
        let (three, six) = (narrow(3.0), narrow(6.0));
        assert_ne!(three, six, "the bound sets how far bands are narrowed");
        for (residues, side_loss_db) in [(&three, 3.0), (&six, 6.0)] {
            let least = 10f64.powf(-side_loss_db / 10.0);
            let mut nearest = f64::INFINITY;
            for start in (0..lines).step_by(width) {
                let band = start..start + width;
                let (x, y) = (&spectra[0][band.clone()], &spectra[1][band.clone()]);
                let side: f64 = x.iter().zip(y).map(|(a, b)| (a - b).powi(2)).sum();
                let mid: f64 = x.iter().zip(y).map(|(a, b)| (a + b).powi(2)).sum();
                let bound = least * side / mid;
                let decoded = ratio(residues, band.clone());
                let floor = bound.min(ratio(&rounded, band.clone()));
                assert!(decoded >= floor * (1.0 - 1e-9), "lines {band:?}");
                nearest = nearest.min(decoded / bound);
            }
            // Some band is narrowed to within 0.5 dB of the bound.
            assert!(nearest < 10f64.powf(0.05), "{side_loss_db} dB: {nearest}");
        }
    }

    /// A band already narrower than its bound may change only so as to
    /// widen; one within it, so as to stay within it; and without bound,
    /// in any way.
    #[test]
    fn a_band_is_held_within_its_bound_or_no_narrower() {
        let least = 10f64.powf(-0.3);
        // Lines whose side is a quarter of their mid, decoded with a side of
        // a hundredth of it: far past a bound of 3 dB.
        let mut narrow = Image::default();
        narrow.add(3.0, 1.0, 1.0, 1.0);
        narrow.add(1.0, 1.0, 2.0, 1.8);
        // Narrower still: refused; wider: admitted, though still past it.
        assert!(
            !narrow
                .replaced((2.0, 1.8), (1.9, 1.9))
                .within(least, &narrow)
        );
        assert!(
            narrow
                .replaced((2.0, 1.8), (2.0, 1.5))
                .within(least, &narrow)
        );
        // Within the bound, narrowed as far as it allows.
        let mut wide = Image::default();
        wide.add(2.0, 1.0, 2.0, 1.0);
        let within = wide.replaced((2.0, 1.0), (2.0, 1.15));
        assert!(within.within(least, &wide));
        assert!(
            !within
                .replaced((2.0, 1.15), (2.0, 1.4))
                .within(least, &within)
        );
        // Without bound, narrowed to nothing, rounding notwithstanding.
        let near = wide.replaced((2.0, 1.0), (1.0 + 1e-9, 1.0));
        let alike = near.replaced((1.0 + 1e-9, 1.0), (1.0, 1.0));
        assert!(alike.decoded_side >= 0.0);
        assert!(alike.within(0.0, &near));
    }

    #[test]
    fn the_side_loss_widens_as_bits_grow_scarce() {
        assert_eq!(side_loss_db(0.0), SIDE_LOSS_DB);
        assert_eq!(side_loss_db(-1.0), SIDE_LOSS_DB);
        assert_eq!(side_loss_db(0.5), 2.0 * SIDE_LOSS_DB);
        assert_eq!(side_loss_db(1.0), f64::INFINITY);
    }

    #[test]
    fn the_angle_allowance_falls_as_bits_grow_plentiful() {
        assert_eq!(angle_allowance(1.0), ANGLE_ALLOWANCE);
        assert_eq!(angle_allowance(2.0), ANGLE_ALLOWANCE);
        assert_eq!(angle_allowance(0.0), LEAST_ANGLE_ALLOWANCE);
        assert_eq!(angle_allowance(-1.0), LEAST_ANGLE_ALLOWANCE);
        let halfway = (LEAST_ANGLE_ALLOWANCE + ANGLE_ALLOWANCE) / 2.0;
        assert!((angle_allowance(0.5) - halfway).abs() < 1e-12);
    }

    #[test]
    fn an_unused_channel_takes_the_others_residue_and_no_angle() {
        let lines = 64;
        let (spectra, curve, noise) = alike(lines);
        let zeros = vec![0.0; lines];
        for silent in 0..2 {
            let mut residues = [round(&spectra[0]), round(&spectra[1])];
            residues[silent].fill(0);
            let coded = residues[1 - silent].clone();
            let mut sources = [&spectra[0][..], &spectra[1][..]];
            sources[silent] = &zeros;
            let [left, right] = &mut residues;
            couple(
                [left, right],
                [&curve, &curve],
                sources,
                [&noise, &noise],
                &vec![0; lines],
                Narrowing {
                    protected: 0,
                    allowance: ANGLE_ALLOWANCE,
                    side_loss_db: SIDE_LOSS_DB,
                },
                0,
                i32::MAX,
                1.0,
            );
            let (magnitude, angle) = (&residues[0], &residues[1]);
            assert_eq!(magnitude, &coded);
            assert!(angle.iter().all(|&a| a == 0));
        }
    }

    /// Lines of a long block, some loud and some quiet, none 0.
    fn lines_of_a_long_block() -> Vec<f64> {
        (0..1024)
            .map(|k| ((k * 37 % 23) as f64 - 11.5) * 1e-3 * (1.0 + (k % 5) as f64))
            .collect()
    }

    #[test]
    fn matched_floors_follow_the_ratio_of_alike_channels() {
        let floor = Floor::long(0);
        let posts = floor.xs().len();
        let left = lines_of_a_long_block();
        // 6 dB lower: nearest 4 Y values lower.
        let right: Vec<f64> = left.iter().map(|x| 0.5 * x).collect();
        let (mut l, mut r) = (vec![40; posts], vec![37; posts]);
        match_floors([&mut l, &mut r], [&left, &right], &floor);
        for i in 0..posts {
            assert_eq!(r[i] - l[i], -4, "post {i}");
            assert!(l[i] <= 40 && r[i] <= 37, "post {i}: neither is raised");
        }
        // The louder channel on the right, its floor the lower.
        let (mut l, mut r) = (vec![30; posts], vec![36; posts]);
        match_floors([&mut l, &mut r], [&right, &left], &floor);
        assert!((0..posts).all(|i| (l[i], r[i]) == (30, 34)));
    }

    #[test]
    fn unlike_channels_keep_their_floors() {
        let floor = Floor::long(0);
        let posts = floor.xs().len();
        let left = lines_of_a_long_block();
        let inverted: Vec<f64> = left.iter().map(|x| -x).collect();
        let silent = vec![0.0; left.len()];
        for right in [&inverted, &silent] {
            let (mut l, mut r) = (vec![40; posts], vec![35; posts]);
            match_floors([&mut l, &mut r], [&left, right], &floor);
            assert_eq!((l, r), (vec![40; posts], vec![35; posts]));
        }
    }
}
