// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The energy of each band of lines, as rounding to the floor leaves it:
//! where lines lie near the point at which their rounding changes, some are
//! rounded the other way, so that the band decodes with its energy (see
//! [`balance`]); and where rounding codes most of a band's lines as 0, some
//! of them are coded as 1 instead, so that the band keeps part of its
//! energy (see [`keep`]).
//!
//! The specification describes no encoder, and this is chelys's own
//! design. The floor allows each line noise of the power that rounding to
//! it makes, and most lines of most blocks lie below half a step and are
//! coded as 0: the noise of a band is then the energy of its lines, and
//! the band decodes as a hole, lower than the signal by more than the noise
//! allowed it is heard to be, and one whose lines come and go from one
//! block to the next. A line coded as 1 rather than 0 makes more noise,
//! measured as the difference between the two, but brings back energy
//! where it was.

/// The lines of a channel in a group whose energy is kept: as many as a
/// partition of the residue holds where channels are coded each on its
/// own. Where a stereo pair is coupled, a partition holds half as many of
/// each channel, interleaved.
pub const GROUP: usize = super::residue::PARTITION_SIZE;

/// The share of a group's energy below which its lines coded as 0 are
/// coded as 1 instead, largest first, until it decodes with this share.
///
/// Graded with PEAQ (GstPEAQ 0.6.1, black box) on the 51 tracks of the
/// tuning set at 44.1 kHz in stereo, at noise offsets matched to 128.4
/// kbit/s, kept below [`TOP_HZ`] once the angles are narrowed, the streams
/// graded -0.537 with no energy kept, and -0.525, -0.519 and -0.534 with
/// 0.3, 0.5 and 0.7. The gain is in the coded band: PEAQ's `EHSB`, which
/// measures the error's harmonic structure, fell from 0.295 to 0.275, and
/// its `ADBB`, the distortion of the blocks it finds distorted, from 1.29
/// to 1.26, while the bandwidth it measured stayed at 838 lines. Kept
/// before the narrowing instead, which codes most of the lines it adds as 0
/// again, at 0.8, with the noise tilted by 2 dB per octave rather than
/// weighted as the model weighs it, the streams graded -0.534 against
/// -0.542 with none kept.
pub const KEPT_SHARE: f64 = 0.5;

/// The least line, in steps of the floor, that is coded as 1 rather than 0:
/// a line lower still would make more noise than its energy.
pub const LEAST_STEPS: f64 = 0.15;

/// How many times as large [`keep`] counts a line of one channel of a
/// coupled pair whose value in the other channel is the one keeping would
/// give it, so that the two channels' values of the line are alike, and
/// its angle 0, which takes fewer bits (see the coupling's
/// `square_polar`).
///
/// Graded with PEAQ (GstPEAQ 0.6.1, black box) on the 51 tracks of the
/// tuning set at 44.1 kHz in stereo, at noise offsets matched to 96.5,
/// 128.4 and 159.2 kbit/s, with the floors matched (see the coupling's
/// `match_floors`) and up to twice the noise allowed in the treble where
/// bits were scarce (see the psychoacoustic model's `TREBLE_HZ`), the
/// streams graded -0.879, -0.416 and -0.193 with each channel's energy
/// kept on its own, and -0.870, -0.416 and -0.193 so; with twice the noise
/// in the treble at every rate and the two channels' values also set alike
/// above 14 kHz, -0.877, -0.394 and -0.156 against -0.868, -0.393 and
/// -0.157, higher on 51 of the 51 tracks at 96.5 kbit/s.
pub const SHARED_PREFERENCE: f64 = 4.0;

/// The frequency above which no energy is kept.
///
/// Above it, lines coded as 1 among lines coded as 0 grade higher with
/// PEAQ, as PEAQ measures the bandwidth of the decoded signal from the
/// level of its highest lines, and credits a wider one; but that gain is
/// not one of lower noise, and the bits it takes are taken from the band
/// below. Graded as for [`KEPT_SHARE`], the streams graded -0.442 with the
/// energy kept over every line, against -0.519 below 16 kHz; but the
/// bandwidth PEAQ measured rose from 838 lines to 878, while its
/// `TotalNMRB` rose from -10.7 to -10.3 dB, its `AvgModDiff1B` from 14.9
/// to 15.6 and its `EHSB` from 0.275 to 0.293.
pub const TOP_HZ: f64 = 16_000.0;

/// How near, in steps of the floor, to the point halfway between two
/// values a line is to lie for [`balance`] to round it to the other: a line
/// rounded to `q` but `1/2 - BALANCE_WINDOW` steps or less from `q + 1`, or
/// from `q - 1`, may be rounded to it instead, which adds at most `2
/// BALANCE_WINDOW` times the step squared to its noise.
///
/// chelys's own design. Rounding each line to the nearest value, or nearer
/// zero (see the encoder's `DEAD_ZONE`), decodes a band with more or less
/// energy than its lines have, by chance, and from one block to the next:
/// mostly less, as the lines of most bands lie below a step or two and are
/// rounded nearer zero. Its level then strays from the signal's by more than
/// its noise alone, and wavers. PEAQ measures both, from the bands' levels
/// rather than from the noise itself: its `ADBB` from the steps by which the
/// decoded signal is heard to be lower than the reference, and its
/// `AvgModDiff1B` from how much faster its level changes. Each line rounded
/// the other way within the window costs little noise and few bits, or
/// saves them, as a line rounded nearer zero does.
///
/// Graded with PEAQ (GstPEAQ 0.6.1, black box) on the 51 tracks of the
/// tuning set at 44.1 kHz in stereo, at noise offsets matched to 96.5, 128.4
/// and 159.2 kbit/s, balanced once the angles are narrowed, the streams
/// graded -0.904, -0.464 and -0.264, against -0.905, -0.474 and -0.284
/// unbalanced; within 0.1, -0.908, -0.466 and -0.267, and within 0.25,
/// -0.907, -0.464 and -0.262. `AvgModDiff1B` fell from 19.8, 15.4 and 12.2
/// to 19.2, 14.8 and 11.7, and `ADBB` from 1.66, 1.22 and 0.61 to 1.55, 1.07
/// and 0.45. Balanced before the narrowing instead, which undoes much of it,
/// -0.914, -0.473 and -0.287 within 1/2; and with the lines that the
/// narrowing set alike also moved, each on its own, which gives their
/// angles back, the bits that takes cost more: -1.342, -0.562 and -0.272.
/// In one channel, at 65.2 and 81.0 kbit/s, the streams graded -0.552 and
/// -0.230, against -0.589 and -0.259 unbalanced.
///
/// Below [`coupling::PROTECTED_HZ`], the narrowing and [`keep`] add no
/// noise to a band beyond what the model allows it, though rounding alone
/// may leave it more; the balance may add some: a deliberate excess, of at
/// most `2 BALANCE_WINDOW` times the step squared in each line it moves, as
/// it grades higher so. Balanced only above it, the streams graded -0.913,
/// -0.467 and -0.267. On the tuning set at 44.1 kHz, the balance takes 3.9
/// to 8.3% of the bands below it past their allowance, by 1.1 to 1.9 dB at
/// the median, about 5 dB at the 99th percentile and at most 5.4 dB; about
/// a fifth of the bands so taken are past it already from rounding alone.
///
/// [`coupling::PROTECTED_HZ`]: super::coupling::PROTECTED_HZ
pub const BALANCE_WINDOW: f64 = 0.15;

/// The share of a group's energy that [`balance`] brings the group's
/// decoded energy nearest where bits are scarce (see the encoder's
/// `scarcity`); where they are plentiful, all of it.
///
/// Aimed a little low, as the encoder's dead zone rounds lines, the
/// balance rounds fewer lines up and more down, for fewer bits. Graded as
/// for [`BALANCE_WINDOW`], with the encoder's other choices as they are
/// now, the streams graded -0.894, -0.460 and -0.248 at 96.5, 128.4 and
/// 159.2 kbit/s aiming for 0.95 where bits are scarce, tapered to 1 where
/// they are plentiful, against -0.905, -0.463 and -0.250 aiming for all of
/// it; higher on 51, 39 and 38 of the 51 tracks, and on none lower but 4
/// at 128.4. In one channel at 65.2 and 81.0 kbit/s, -0.535 and -0.221,
/// against -0.550 and -0.227.
pub const BALANCED_SHARE_SCARCE: f64 = 0.95;

/// The share of a group's energy that [`balance`] aims for where bits are
/// as scarce as `scarcity`, from 0 where they are plentiful to 1 where
/// they are scarce: from all of it to [`BALANCED_SHARE_SCARCE`].
pub fn balanced_share(scarcity: f64) -> f64 {
    1.0 - (1.0 - BALANCED_SHARE_SCARCE) * scarcity.clamp(0.0, 1.0)
}

/// The lines of a block of `lines` lines at `sample_rate` Hz below
/// [`TOP_HZ`]: line `k` of a block of `n` samples is centered at `(k +
/// 1/2) sample_rate / n` Hz.
pub fn kept_lines(lines: usize, sample_rate: u32) -> usize {
    let below = TOP_HZ * 2.0 * lines as f64 / f64::from(sample_rate) - 0.5;
    (below.ceil().max(0.0) as usize).min(lines)
}

/// The lines below the frequency below which no band of the
/// psychoacoustic model takes more noise than it allows (see
/// [`coupling::PROTECTED_HZ`]), and what that allowance is.
///
/// [`coupling::PROTECTED_HZ`]: super::coupling::PROTECTED_HZ
#[derive(Clone, Copy, Debug)]
pub struct Protected<'a> {
    /// The protected lines, the first of the block.
    pub lines: usize,
    /// The noise each line of the block is allowed.
    pub noise: &'a [f64],
    /// The model's partition of each line of the block: the bands in which
    /// the noise is allowed together.
    pub partitions: &'a [usize],
}

/// Keeps the energy of the first `kept` lines of `residue`, the lines
/// `spectrum` quantized to the floor `curve`: in each group of [`GROUP`]
/// lines, while the group decodes with less than [`KEPT_SHARE`] of the
/// energy of its lines, the largest of its lines coded as 0 against its
/// step, if it is at least [`LEAST_STEPS`] of a step, is coded as 1, signed
/// as the line.
///
/// A line of the `protected` lines is coded so only where its band, the
/// protected lines of its partition, stays within the noise it is allowed,
/// or where coding it so makes less noise than coding it as 0. A line
/// coded as `v` makes the noise `(x - f v)^2`, where it is `x` and the
/// floor `f`.
///
/// In a coupled pair, `other` is the other channel's residue: a line whose
/// value there is the one this channel's would take counts as
/// [`SHARED_PREFERENCE`] times as large.
pub fn keep(
    residue: &mut [i32],
    spectrum: &[f64],
    curve: &[f32],
    protected: Protected<'_>,
    kept: usize,
    other: Option<&[i32]>,
) {
    let kept = kept.min(residue.len());
    let noise = |k: usize, v: i32| (spectrum[k] - f64::from(curve[k]) * f64::from(v)).powi(2);
    // The noise each partition's protected lines may yet take.
    let protected_lines = protected.lines.min(residue.len());
    let bands = protected.partitions[..protected_lines]
        .last()
        .map_or(0, |&b| b + 1);
    let mut room = vec![0.0; bands];
    for k in 0..protected_lines {
        room[protected.partitions[k]] += protected.noise[k] - noise(k, residue[k]);
    }
    for start in (0..kept).step_by(GROUP) {
        let group = start..(start + GROUP).min(kept);
        let energy: f64 = spectrum[group.clone()].iter().map(|x| x * x).sum();
        let mut decoded: f64 = (group.clone())
            .map(|k| (f64::from(residue[k]) * f64::from(curve[k])).powi(2))
            .sum();
        while decoded < KEPT_SHARE * energy {
            let sign = |k: usize| if spectrum[k] < 0.0 { -1 } else { 1 };
            let added = |k: usize| noise(k, sign(k)) - noise(k, 0);
            let fits = |k: usize| {
                k >= protected_lines || added(k) <= 0.0 || added(k) <= room[protected.partitions[k]]
            };
            let largest = (group.clone())
                .filter(|&k| residue[k] == 0)
                .map(|k| (k, spectrum[k].abs() / f64::from(curve[k])))
                .filter(|&(k, steps)| steps >= LEAST_STEPS && fits(k))
                .map(|(k, steps)| match other {
                    Some(other) if other[k] == sign(k) => (k, steps * SHARED_PREFERENCE),
                    _ => (k, steps),
                })
                .max_by(|a, b| a.1.total_cmp(&b.1));
            let Some((k, _)) = largest else { break };
            if k < protected_lines {
                room[protected.partitions[k]] -= added(k);
            }
            residue[k] = sign(k);
            decoded += f64::from(curve[k]).powi(2);
        }
    }
}

/// Balances the energy of the first `kept` lines of `residue`, the lines
/// `spectrum` quantized to the floor `curve`, as no more than `limit`: in
/// each group of [`GROUP`] lines, lines within [`BALANCE_WINDOW`] of the
/// point halfway to the next value, above or below, are rounded to it, one
/// step, while that brings the energy the group decodes with nearer `share`
/// of that of its lines (see [`balanced_share`]). Rounding up adds energy,
/// and down takes it away; the lines that add or take the most energy for
/// the least noise go first, each line at most once. Lines for which
/// `locked` holds are left as they are: in a coupled pair, those whose
/// values the two channels share, where the angles are narrowed or the
/// values are alike already, both 0 among them.
///
/// Rounded up, a line coded as 0 takes the sign of the line; any other
/// keeps its own.
pub fn balance(
    residue: &mut [i32],
    spectrum: &[f64],
    curve: &[f32],
    limit: i32,
    locked: impl Fn(usize) -> bool,
    kept: usize,
    share: f64,
) {
    let kept = kept.min(residue.len());
    for start in (0..kept).step_by(GROUP) {
        let group = start..(start + GROUP).min(kept);
        let target: f64 = share * spectrum[group.clone()].iter().map(|x| x * x).sum::<f64>();
        let mut decoded: f64 = (group.clone())
            .map(|k| (f64::from(residue[k]) * f64::from(curve[k])).powi(2))
            .sum();
        let mut moved = [false; GROUP];
        loop {
            let up = decoded < target;
            // The line whose move adds the least noise for the energy it
            // adds or takes away, its new magnitude and that energy.
            let mut best: Option<(usize, i32, f64, f64)> = None;
            for k in group.clone() {
                let step = f64::from(curve[k]);
                if moved[k - start] || locked(k) || step <= 0.0 {
                    continue;
                }
                let steps = spectrum[k].abs() / step;
                let q = residue[k].abs();
                let to = if up { q + 1 } else { q - 1 };
                let within = if up {
                    to <= limit && steps >= f64::from(q) + 0.5 - BALANCE_WINDOW
                } else {
                    q > 0 && steps <= f64::from(q) - 0.5 + BALANCE_WINDOW
                };
                if !within {
                    continue;
                }
                let energy = f64::from(to * to - q * q) * step * step;
                let noise = ((f64::from(to) - steps).powi(2) - (f64::from(q) - steps).powi(2))
                    * step
                    * step;
                let cost = noise / energy.abs();
                if best.is_none_or(|(_, _, _, least)| cost < least) {
                    best = Some((k, to, energy, cost));
                }
            }
            let Some((k, to, energy, _)) = best else {
                break;
            };
            if (decoded + energy - target).abs() >= (decoded - target).abs() {
                break;
            }
            decoded += energy;
            moved[k - start] = true;
            let sign = match residue[k].signum() {
                0 if spectrum[k] < 0.0 => -1,
                0 => 1,
                sign => sign,
            };
            residue[k] = sign * to;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No protected lines.
    const NONE: Protected<'static> = Protected {
        lines: 0,
        noise: &[],
        partitions: &[],
    };

    /// The decoded energy of `residue`'s lines `range` on the floor `curve`.
    fn decoded(residue: &[i32], curve: &[f32], range: std::ops::Range<usize>) -> f64 {
        range
            .map(|k| (f64::from(residue[k]) * f64::from(curve[k])).powi(2))
            .sum()
    }

    #[test]
    fn a_line_alike_in_the_other_channel_is_kept_first() {
        // Lines from 0.1 to 0.475 steps, all coded as 0; the other channel
        // codes the line of 0.4 steps as this channel would, and the
        // largest with the other sign.
        let spectrum: Vec<f64> = (0..GROUP).map(|k| 0.1 + 0.025 * k as f64).collect();
        let curve = vec![1.0; GROUP];
        let mut other = vec![0; GROUP];
        other[12] = 1;
        other[GROUP - 1] = -1;
        let mut residue = vec![0; GROUP];
        keep(&mut residue, &spectrum, &curve, NONE, GROUP, Some(&other));
        let filled: Vec<usize> = (0..GROUP).filter(|&k| residue[k] != 0).collect();
        assert_eq!(filled, [12]);
        assert_eq!(residue[12], 1);
        // Without the other channel, the largest.
        let mut residue = vec![0; GROUP];
        keep(&mut residue, &spectrum, &curve, NONE, GROUP, None);
        assert_eq!(residue.iter().position(|&v| v != 0), Some(GROUP - 1));
    }

    #[test]
    fn a_hole_is_filled_with_its_largest_lines_to_the_share() {
        // A group of lines from 0.1 to 0.475 steps, all coded as 0.
        let spectrum: Vec<f64> = (0..GROUP)
            .map(|k| (0.1 + 0.025 * k as f64) * if k % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        let curve = vec![1.0; GROUP];
        let mut residue = vec![0; GROUP];
        keep(&mut residue, &spectrum, &curve, NONE, GROUP, None);
        let energy: f64 = spectrum.iter().map(|x| x * x).sum();
        assert!(decoded(&residue, &curve, 0..GROUP) >= KEPT_SHARE * energy);
        // The fewest lines that do, the largest, signed as the lines.
        let filled: Vec<usize> = (0..GROUP).filter(|&k| residue[k] != 0).collect();
        assert_eq!(filled, [GROUP - 1]);
        assert_eq!(
            residue[GROUP - 1].signum(),
            spectrum[GROUP - 1].signum() as i32
        );
    }

    #[test]
    fn groups_that_keep_their_share_and_lines_past_the_kept_are_left() {
        let mut spectrum = vec![0.4; 3 * GROUP];
        // The first group keeps its share as coded.
        let mut residue = vec![0; 3 * GROUP];
        residue[..GROUP].fill(1);
        spectrum[..GROUP].fill(0.9);
        let curve = vec![1.0; 3 * GROUP];
        let before = residue.clone();
        keep(&mut residue, &spectrum, &curve, NONE, 2 * GROUP, None);
        assert_eq!(residue[..GROUP], before[..GROUP]);
        assert!(residue[GROUP..2 * GROUP].iter().any(|&v| v != 0));
        assert!(residue[2 * GROUP..].iter().all(|&v| v == 0));
    }

    #[test]
    fn lines_below_the_least_are_never_coded() {
        let spectrum = vec![0.9 * LEAST_STEPS; GROUP];
        let mut residue = vec![0; GROUP];
        keep(&mut residue, &spectrum, &[1.0; GROUP], NONE, GROUP, None);
        assert!(residue.iter().all(|&v| v == 0));
    }

    /// In protected bands, lines are coded as 1 only while the band stays
    /// within the noise it is allowed.
    #[test]
    fn protected_bands_stay_within_their_allowance() {
        // Two bands of 8 lines in a group, of lines 0.3 steps high, all
        // coded as 0: each line coded as 1 adds 0.49 - 0.09 = 0.4.
        let spectrum = vec![0.3; GROUP];
        let curve = vec![1.0; GROUP];
        let partitions: Vec<usize> = (0..GROUP).map(|k| k / 8).collect();
        // The noise each line is allowed: the lines' own 0.09, and room
        // for one line of each band but the second's, which has none.
        let mut noise = vec![0.09; GROUP];
        noise[0] += 0.45;
        let noise_of = |residue: &[i32], range: std::ops::Range<usize>| -> f64 {
            range
                .map(|k| (spectrum[k] - f64::from(residue[k])).powi(2))
                .sum()
        };
        for lines in [0, 8, GROUP] {
            let mut residue = vec![0; GROUP];
            let protected = Protected {
                lines,
                noise: &noise,
                partitions: &partitions,
            };
            keep(&mut residue, &spectrum, &curve, protected, GROUP, None);
            let coded = residue.iter().filter(|&&v| v != 0).count();
            for band in [0..8, 8..GROUP] {
                if band.start < lines {
                    let allowed: f64 = noise[band.clone()].iter().sum();
                    assert!(noise_of(&residue, band.clone()) <= allowed + 1e-12);
                }
            }
            // Unprotected, the group is filled to its share, 0.5 of 1.44,
            // by 1 line; protected, the first band takes it, and only one.
            assert_eq!(coded, 1, "{lines} protected lines");
            if lines == GROUP {
                assert!(residue[..8].iter().any(|&v| v != 0));
            }
        }
        // With no room at all, none is coded.
        let mut residue = vec![0; GROUP];
        let protected = Protected {
            lines: GROUP,
            noise: &[0.09; GROUP],
            partitions: &partitions,
        };
        keep(&mut residue, &spectrum, &curve, protected, GROUP, None);
        assert!(residue.iter().all(|&v| v == 0));
    }

    /// A group of lines 0.4 steps high, all coded as 0: the fewest rounded
    /// up that bring its energy nearest, each a step, signed as the line.
    #[test]
    fn a_group_rounded_low_is_balanced_up() {
        let spectrum: Vec<f64> = (0..GROUP)
            .map(|k| if k % 2 == 0 { 0.4 } else { -0.4 })
            .collect();
        let curve = vec![1.0; GROUP];
        let mut residue = vec![0; GROUP];
        balance(
            &mut residue,
            &spectrum,
            &curve,
            i32::MAX,
            |_| false,
            GROUP,
            1.0,
        );
        let energy: f64 = spectrum.iter().map(|x| x * x).sum();
        // 16 lines of 0.16 are 2.56: 3 lines coded as 1 decode with 3.
        assert_eq!(residue.iter().filter(|&&v| v != 0).count(), 3);
        assert!((decoded(&residue, &curve, 0..GROUP) - energy).abs() <= 0.5);
        for (&v, &x) in residue.iter().zip(&spectrum) {
            assert!(v == 0 || v == x.signum() as i32);
        }
    }

    /// Lines 0.62 steps high, coded as 1, decode with far more energy than
    /// they have: some are rounded down to 0.
    #[test]
    fn a_group_rounded_high_is_balanced_down() {
        let spectrum = vec![-0.62; GROUP];
        let curve = vec![1.0; GROUP];
        let mut residue = vec![-1; GROUP];
        balance(
            &mut residue,
            &spectrum,
            &curve,
            i32::MAX,
            |_| false,
            GROUP,
            1.0,
        );
        let energy: f64 = spectrum.iter().map(|x| x * x).sum();
        let before = GROUP as f64 - energy;
        let after = (decoded(&residue, &curve, 0..GROUP) - energy).abs();
        assert!(after < before && after <= 0.5, "{after}");
        assert!(residue.iter().all(|&v| v == 0 || v == -1));
    }

    /// Lines further from halfway than the window, locked lines and lines
    /// past the kept are never moved, nor any past the limit.
    #[test]
    fn only_lines_within_the_window_move() {
        let curve = vec![1.0; 3 * GROUP];
        // Below the window: 0.3 steps, coded as 0.
        let mut spectrum = vec![0.3; 3 * GROUP];
        let mut residue = vec![0; 3 * GROUP];
        balance(
            &mut residue,
            &spectrum,
            &curve,
            i32::MAX,
            |_| false,
            3 * GROUP,
            1.0,
        );
        assert!(residue.iter().all(|&v| v == 0));
        // Within it, but locked in the first group, past the kept in the
        // third.
        spectrum.fill(0.45);
        balance(
            &mut residue,
            &spectrum,
            &curve,
            i32::MAX,
            |k| k < GROUP,
            2 * GROUP,
            1.0,
        );
        assert!(residue[..GROUP].iter().all(|&v| v == 0));
        assert!(residue[GROUP..2 * GROUP].iter().any(|&v| v != 0));
        assert!(residue[2 * GROUP..].iter().all(|&v| v == 0));
        // At the limit, a line is not rounded up past it.
        let spectrum = vec![2.45; GROUP];
        let mut residue = vec![2; GROUP];
        balance(
            &mut residue,
            &spectrum,
            &curve[..GROUP],
            2,
            |_| false,
            GROUP,
            1.0,
        );
        assert!(residue.iter().all(|&v| v == 2));
    }

    /// Aimed for a share of the energy, the balance takes less of it.
    #[test]
    fn the_balance_aims_for_its_share() {
        let spectrum = vec![0.45; GROUP];
        let curve = vec![1.0; GROUP];
        let coded = |share| {
            let mut residue = vec![0; GROUP];
            balance(
                &mut residue,
                &spectrum,
                &curve,
                i32::MAX,
                |_| false,
                GROUP,
                share,
            );
            decoded(&residue, &curve, 0..GROUP)
        };
        assert!(coded(0.5) < coded(1.0));
        assert_eq!(balanced_share(0.0), 1.0);
        assert_eq!(balanced_share(1.0), BALANCED_SHARE_SCARCE);
        assert!(balanced_share(0.5) > BALANCED_SHARE_SCARCE && balanced_share(0.5) < 1.0);
    }

    #[test]
    fn the_kept_lines_stop_below_the_top() {
        assert_eq!(kept_lines(1024, 44_100), 743);
        assert_eq!(kept_lines(128, 44_100), 93);
        assert_eq!(kept_lines(1024, 16_000), 1024);
    }
}
