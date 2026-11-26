// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! How a mean rate is kept (see [`Rate::Mean`](crate::Rate::Mean)): the
//! noise offset steered from one long block to the next.
//!
//! The specification describes no encoder, and no rate control: the
//! steering is chelys's own design, that of chelys's AAC encoder (see
//! [`MeanSteering`]), whose frames are of 1024 samples as the hops from one
//! long block to the next are. Each hop's packets, a long block's or those
//! of the short blocks that take its place, are counted together.
//!
//! The trades of noise for bits that the offset tapers (see the encoder's
//! `scarcity`) follow the steered offset, hop by hop, rather than being
//! fixed for the stream as chelys's AAC encoder fixes the tools its mean
//! rate chooses: they taper smoothly, without switching a tool on or off,
//! and the offset moves slowly, so that each block is coded as the
//! variable rate of its offset codes it.

use chelys_core::steering::{self, MeanSteering};

use super::encoder::HOP;

/// The steering of a stream of `channels` channels at a mean rate of
/// `bits_per_second` at `sample_rate`, a step for each hop from one long
/// block to the next (see [`HOP`]): from the offset at which music
/// takes the mean on average (see [`NOMINAL_STEREO`] and
/// [`NOMINAL_UNCOUPLED`]), each hop's bits taken to fall by
/// [`BITS_PER_DB`] for each dB of offset.
pub(crate) fn mean_steering(
    bits_per_second: u32,
    sample_rate: u32,
    channels: usize,
) -> MeanSteering {
    let nominal: &[(f64, f64)] = if channels == 2 {
        &NOMINAL_STEREO
    } else {
        &NOMINAL_UNCOUPLED
    };
    let bits_per_sample = f64::from(bits_per_second) / f64::from(sample_rate) / channels as f64;
    MeanSteering::new(
        bits_per_second,
        sample_rate,
        HOP as usize,
        steering::nominal_noise_offset_db(nominal, bits_per_sample),
        BITS_PER_DB,
    )
}

/// How far the logarithm of a hop's bits falls for each dB of noise
/// offset, about: on the tuning set of 51 tracks at 44.1 kHz, the audio
/// packets fell from 212 to 65 kbit/s from -12 to 0 dB in stereo (0.098),
/// and from 115 to 41 in mono (0.085); in stereo by 0.097 for each dB
/// about 90 kbit/s, 0.115 about 125 and 0.139 about 150, where the trades
/// of noise for bits taper in (see `scarcity`).
const BITS_PER_DB: f64 = 0.09;

/// Pairs of a noise offset in dB and the bits for each sample of each
/// channel that music takes at it in stereo, whose two channels are
/// coupled, as [`steering::nominal_bits_per_sample`] takes them: the mean
/// of the tuning set of 51 tracks at 44.1 kHz, the audio packets alone.
const NOMINAL_STEREO: [(f64, f64); 23] = [
    (-30.0, 6.307),
    (-25.0, 5.209),
    (-20.0, 4.054),
    (-16.5, 3.284),
    (-15.0, 2.977),
    (-13.5, 2.685),
    (-12.0, 2.406),
    (-10.5, 2.141),
    (-9.0, 1.889),
    (-7.5, 1.534),
    (-6.0, 1.291),
    (-4.5, 1.102),
    (-3.0, 0.953),
    (-1.5, 0.837),
    (0.0, 0.742),
    (1.5, 0.668),
    (3.0, 0.611),
    (4.5, 0.569),
    (6.0, 0.538),
    (7.5, 0.516),
    (10.0, 0.481),
    (15.0, 0.293),
    (20.0, 0.106),
];

/// As [`NOMINAL_STEREO`], for channels coded each on its own: the tuning
/// set in mono, its two channels mixed.
const NOMINAL_UNCOUPLED: [(f64, f64); 23] = [
    (-30.0, 6.168),
    (-25.0, 5.240),
    (-20.0, 4.177),
    (-16.5, 3.447),
    (-15.0, 3.153),
    (-13.5, 2.870),
    (-12.0, 2.600),
    (-10.5, 2.341),
    (-9.0, 2.095),
    (-7.5, 1.840),
    (-6.0, 1.604),
    (-4.5, 1.391),
    (-3.0, 1.213),
    (-1.5, 1.061),
    (0.0, 0.935),
    (1.5, 0.831),
    (3.0, 0.749),
    (4.5, 0.684),
    (6.0, 0.632),
    (7.5, 0.588),
    (10.0, 0.515),
    (15.0, 0.288),
    (20.0, 0.105),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Each table's bits fall as the offset rises.
    #[test]
    fn the_nominal_rates_fall_with_the_offset() {
        for nominal in [NOMINAL_STEREO, NOMINAL_UNCOUPLED] {
            for pair in nominal.windows(2) {
                assert!(pair[0].0 < pair[1].0 && pair[0].1 > pair[1].1, "{pair:?}");
            }
        }
    }

    /// The steering starts from the offset at which music takes the mean
    /// on average: in stereo at 44.1 kHz, 128 kbit/s lies between -7.5 and
    /// -6 dB; in mono at 48 kHz, 64 kbit/s between -4.5 and -3.
    #[test]
    fn the_steering_starts_from_the_nominal_offset() {
        let start = |bits_per_second, sample_rate, channels| {
            mean_steering(bits_per_second, sample_rate, channels).noise_offset_db()
        };
        assert!((-7.5..-6.0).contains(&start(128_000, 44_100, 2)));
        assert!((-4.5..-3.0).contains(&start(64_000, 48_000, 1)));
        // Six channels coded each on their own, at 64 kbit/s each.
        assert!((-4.5..-3.0).contains(&start(384_000, 48_000, 6)));
        // A higher mean starts lower.
        assert!(start(160_000, 44_100, 2) < start(128_000, 44_100, 2));
    }
}
