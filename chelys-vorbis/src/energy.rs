// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Holes: where rounding to the floor codes most of a band's lines as 0,
//! some of them are coded as 1 instead, so that the band keeps part of its
//! energy.
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
pub fn keep(
    residue: &mut [i32],
    spectrum: &[f64],
    curve: &[f32],
    protected: Protected<'_>,
    kept: usize,
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
    fn a_hole_is_filled_with_its_largest_lines_to_the_share() {
        // A group of lines from 0.1 to 0.475 steps, all coded as 0.
        let spectrum: Vec<f64> = (0..GROUP)
            .map(|k| (0.1 + 0.025 * k as f64) * if k % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        let curve = vec![1.0; GROUP];
        let mut residue = vec![0; GROUP];
        keep(&mut residue, &spectrum, &curve, NONE, GROUP);
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
        keep(&mut residue, &spectrum, &curve, NONE, 2 * GROUP);
        assert_eq!(residue[..GROUP], before[..GROUP]);
        assert!(residue[GROUP..2 * GROUP].iter().any(|&v| v != 0));
        assert!(residue[2 * GROUP..].iter().all(|&v| v == 0));
    }

    #[test]
    fn lines_below_the_least_are_never_coded() {
        let spectrum = vec![0.9 * LEAST_STEPS; GROUP];
        let mut residue = vec![0; GROUP];
        keep(&mut residue, &spectrum, &[1.0; GROUP], NONE, GROUP);
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
            keep(&mut residue, &spectrum, &curve, protected, GROUP);
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
        keep(&mut residue, &spectrum, &curve, protected, GROUP);
        assert!(residue.iter().all(|&v| v == 0));
    }

    #[test]
    fn the_kept_lines_stop_below_the_top() {
        assert_eq!(kept_lines(1024, 44_100), 743);
        assert_eq!(kept_lines(128, 44_100), 93);
        assert_eq!(kept_lines(1024, 16_000), 1024);
    }
}
