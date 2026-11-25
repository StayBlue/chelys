// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Square polar mapping: the two channels of a stereo stream coded as a
//! magnitude and an angle, which a decoder maps back to them (clause
//! 4.3.5).
//!
//! The specification defines only the inverse, which acts on the residue
//! vectors, before each channel's floor scales its own; the forward
//! mapping here is chelys's own, the one that inverts it exactly, value
//! for value.

use super::energy;

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
pub const LEAST_ANGLE_ALLOWANCE: f64 = 2.0;

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

/// Maps the residues of a stereo pair, channel 0 and channel 1, in place,
/// to the magnitude and angle that clause 4.3.5 maps back to them: each
/// channel's residue quantized to its floor `curves`, of its lines
/// `spectra`, which may take the noise `noises`, and which lie in the
/// psychoacoustic model's partitions `partitions`, one for each line. The
/// lines below the `protected` first are those below [`PROTECTED_HZ`].
///
/// Where both channels are used, their angles are first narrowed where
/// that leaves each channel's noise in each band within `allowance` times
/// its allowance above [`PROTECTED_HZ`] (see [`narrow_angles`]). Then the
/// energy of each used channel's first `kept` lines is kept (see
/// [`energy::keep`]), which adds noise above [`PROTECTED_HZ`] and keeps
/// each band within its allowance below: after the narrowing, which would
/// code the lines it adds as 0 again, as that makes less noise.
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
    protected: usize,
    allowance: f64,
    kept: usize,
) {
    let [left, right] = residues;
    let used = [left.iter().any(|&v| v != 0), right.iter().any(|&v| v != 0)];
    let guard = |c: usize| energy::Protected {
        lines: protected,
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
                protected,
                allowance,
            );
            energy::keep(left, spectra[0], curves[0], guard(0), kept);
            energy::keep(right, spectra[1], curves[1], guard(1), kept);
        }
        [true, false] => {
            energy::keep(left, spectra[0], curves[0], guard(0), kept);
            right.clone_from(left);
        }
        [false, true] => {
            energy::keep(right, spectra[1], curves[1], guard(1), kept);
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
/// split at the `protected` first line: the model allows each of its lines
/// the same share of the partition's threshold, which it takes as masked
/// together. In each band, each channel may take `allowance` times the
/// noise its lines are allowed, or below the `protected` first line
/// the allowance itself, less the noise its rounding makes. Each line
/// whose values differ may take the value between them, either included,
/// of the least noise in the two channels together; the lines are taken in
/// the order of that noise, the least first, each where it fits the room
/// left to both channels in its band.
///
/// The values stay between the channels' own, and so within the residue's
/// reach.
fn narrow_angles(
    residues: [&mut Vec<i32>; 2],
    curves: [&[f32]; 2],
    spectra: [&[f64]; 2],
    noises: [&[f64]; 2],
    partitions: &[usize],
    protected: usize,
    allowance: f64,
) {
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
    for (k, v, added) in changes {
        let room = &mut room[bands[k]];
        if (0..2).all(|c| added[c] <= 0.0 || added[c] <= room[c]) {
            room[0] -= added[0];
            room[1] -= added[1];
            left[k] = v;
            right[k] = v;
        }
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
            protected,
            ANGLE_ALLOWANCE,
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

    #[test]
    fn the_angle_allowance_falls_as_bits_grow_plentiful() {
        assert_eq!(angle_allowance(1.0), ANGLE_ALLOWANCE);
        assert_eq!(angle_allowance(2.0), ANGLE_ALLOWANCE);
        assert_eq!(angle_allowance(0.0), LEAST_ANGLE_ALLOWANCE);
        assert_eq!(angle_allowance(-1.0), LEAST_ANGLE_ALLOWANCE);
        assert!((angle_allowance(0.5) - 2.5).abs() < 1e-12);
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
                0,
                ANGLE_ALLOWANCE,
                0,
            );
            let (magnitude, angle) = (&residues[0], &residues[1]);
            assert_eq!(magnitude, &coded);
            assert!(angle.iter().all(|&a| a == 0));
        }
    }
}
