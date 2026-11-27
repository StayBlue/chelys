// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! How a mean rate and a constant rate are kept (see
//! [`Rate::Mean`](crate::Rate::Mean) and
//! [`Rate::Constant`](crate::Rate::Constant)): the noise offset steered
//! from one long block to the next.
//!
//! The specification describes no encoder, and no rate control: the
//! steering of a mean rate is chelys's own design, that of chelys's AAC
//! encoder (see [`MeanSteering`]), whose frames are of 1024 samples as the
//! hops from one long block to the next are; that of a constant rate (see
//! [`ConstantSteering`]) is its own design too. Each hop's packets, a long
//! block's or those of the short blocks that take its place, are counted
//! together.
//!
//! The trades of noise for bits that the offset tapers (see the encoder's
//! `scarcity`) follow the steered offset, hop by hop, rather than being
//! fixed for the stream as chelys's AAC encoder fixes the tools its mean
//! rate chooses: they taper smoothly, without switching a tool on or off,
//! and the offset moves slowly, so that each block is coded as the
//! variable rate of its offset codes it.

use chelys_core::steering::{self, MeanSteering, OFFSETS_DB};

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
    MeanSteering::new(
        bits_per_second,
        sample_rate,
        HOP as usize,
        nominal_noise_offset_db(bits_per_second, sample_rate, channels),
        BITS_PER_DB,
    )
}

/// The offset at which music takes `bits_per_second` on average in a stream
/// of `channels` channels at `sample_rate` (see [`NOMINAL_STEREO`] and
/// [`NOMINAL_UNCOUPLED`]).
fn nominal_noise_offset_db(bits_per_second: u32, sample_rate: u32, channels: usize) -> f64 {
    let nominal: &[(f64, f64)] = if channels == 2 {
        &NOMINAL_STEREO
    } else {
        &NOMINAL_UNCOUPLED
    };
    let bits_per_sample = f64::from(bits_per_second) / f64::from(sample_rate) / channels as f64;
    steering::nominal_noise_offset_db(nominal, bits_per_sample)
}

/// The noise offset of each hop of a stream at a constant rate: that at
/// which the recent hops would have taken the bits the rate brings a hop,
/// more where the buffer holds more than half of its size, and fewer where
/// it holds less (see [`Buffer`](super::buffer::Buffer)).
///
/// As [`MeanSteering`] does, each hop's bits are taken back to what they
/// would have been at 0 dB, each packet's at the offset it was coded at,
/// and their mean over the last [`MEMORY_HOPS`] or so gives the offset at
/// which the next hop would take the bits it is to take: those the rate
/// brings, times `1 + GAIN * (2 * fullness - 1)` for the share of its size
/// the buffer holds, but no fewer than a quarter of them. The bits are
/// counted as coded, before any padding, so that a hop padded because its
/// content takes few bits is not taken for one that takes many.
#[derive(Clone, Debug)]
pub(crate) struct ConstantSteering {
    /// The bits the rate brings a hop.
    mean: f64,
    /// The bits the recent hops would have taken at 0 dB, a mean that
    /// forgets each hop over [`MEMORY_HOPS`].
    at_0_db: f64,
    /// The hops taken so far.
    hops: f64,
    /// The noise offset of the next hop.
    noise_offset_db: f64,
}

/// The hops over which [`ConstantSteering`] forgets a hop's bits: it
/// starts out as if the hops before the first had taken the rate's bits at
/// the offset it starts from, forgetting them over 8 hops more than those
/// taken while those are fewer, as [`MeanSteering`] does.
///
/// Graded with PEAQ (GstPEAQ 0.6.1, black box) on the 51 tracks of the
/// tuning set at 44.1 kHz in stereo at 128 kbit/s, the streams graded
/// -0.401 forgetting over 64 hops with a [`GAIN`] of 0.25, against -0.404
/// over 32 with 0.5 and -0.409 over 16 with 1, and -0.398 at a mean rate:
/// the more slowly the steering follows the content and the buffer, the
/// better they graded, and the more of the buffer they used, to within
/// 0.1% of empty on 2 tracks, against 1 and none.
const MEMORY_HOPS: f64 = 64.0;
const START_MEMORY_HOPS: f64 = 8.0;

/// How far the bits a hop is to take follow the buffer: at 1, twice the
/// rate's where the buffer is full and none where it is empty (see
/// [`MEMORY_HOPS`]).
const GAIN: f64 = 0.25;

impl ConstantSteering {
    /// The steering of a stream of `channels` channels at a constant rate
    /// of `bits_per_second` at `sample_rate`, from the offset at which
    /// music takes the rate on average.
    pub(crate) fn new(bits_per_second: u32, sample_rate: u32, channels: usize) -> Self {
        let mean = f64::from(bits_per_second) * HOP as f64 / f64::from(sample_rate);
        let start_db = nominal_noise_offset_db(bits_per_second, sample_rate, channels);
        Self {
            mean,
            at_0_db: mean * (BITS_PER_DB * start_db).exp(),
            hops: 0.0,
            noise_offset_db: start_db,
        }
    }

    /// The noise offset of the next hop.
    pub(crate) fn noise_offset_db(&self) -> f64 {
        self.noise_offset_db
    }

    /// Takes a hop whose packets would have taken `at_0_db` bits at 0 dB
    /// (see [`at_0_db`]), after which the buffer holds `fullness` of its
    /// size.
    pub(crate) fn spend(&mut self, at_0_db: f64, fullness: f64) {
        self.hops += 1.0;
        let memory = MEMORY_HOPS.min(self.hops + START_MEMORY_HOPS);
        self.at_0_db += (at_0_db - self.at_0_db) / memory;
        let wanted = self.mean * (1.0 + GAIN * (2.0 * fullness - 1.0)).max(0.25);
        let offset = (self.at_0_db / wanted).ln() / BITS_PER_DB;
        self.noise_offset_db = offset.clamp(*OFFSETS_DB.start(), *OFFSETS_DB.end());
    }
}

/// The bits a packet of `len` bytes coded at `noise_offset_db` would have
/// taken at 0 dB, about.
pub(crate) fn at_0_db(len: usize, noise_offset_db: f64) -> f64 {
    8.0 * len as f64 * (BITS_PER_DB * noise_offset_db).exp()
}

/// The offset at which a block that took `len` bytes at `noise_offset_db`
/// would take `longest`, about, and at least `step` dB higher; no higher
/// than the highest the steering keeps.
pub(crate) fn raised_offset_db(noise_offset_db: f64, len: usize, longest: usize, step: f64) -> f64 {
    let needed = (len as f64 / longest.max(1) as f64).ln() / BITS_PER_DB;
    (noise_offset_db + needed.max(step)).min(*OFFSETS_DB.end())
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
        // A constant rate starts where a mean rate does.
        let constant = ConstantSteering::new(128_000, 44_100, 2);
        assert_eq!(constant.noise_offset_db(), start(128_000, 44_100, 2));
    }

    /// Hops that take the rate's bits at the offset keep it while the
    /// buffer is half full; the offset falls as the buffer fills, and rises
    /// as it empties, the more so the farther, but no further than the
    /// steering keeps.
    #[test]
    fn the_constant_steering_follows_the_buffer() {
        let steering = ConstantSteering::new(128_000, 44_100, 2);
        let start = steering.noise_offset_db();
        let after = |fullness: f64| {
            let mut steering = steering.clone();
            let at_0_db = steering.mean * (BITS_PER_DB * start).exp();
            steering.spend(at_0_db, fullness);
            steering.noise_offset_db()
        };
        assert!((after(0.5) - start).abs() < 1e-9);
        assert!(after(1.0) < after(0.75) && after(0.75) < start);
        assert!(after(0.0) > after(0.25) && after(0.25) > start);
        // Hops of silence, a few bits each, drive it to the lowest.
        let mut steering = steering.clone();
        for _ in 0..1000 {
            steering.spend(8.0, 1.0);
        }
        assert_eq!(steering.noise_offset_db(), *OFFSETS_DB.start());
    }

    /// A block raised to fit is raised by what its bits ask, or by the
    /// step where they ask less, and no higher than the steering keeps.
    #[test]
    fn a_block_is_raised_by_what_its_bits_ask() {
        // Twice the bits: ln 2 / 0.09, 7.7 dB.
        let raised = raised_offset_db(-6.0, 200, 100, 1.5);
        assert!((raised - (-6.0 + 2f64.ln() / BITS_PER_DB)).abs() < 1e-9);
        assert_eq!(raised_offset_db(-6.0, 101, 100, 1.5), -4.5);
        assert_eq!(raised_offset_db(39.0, 200, 100, 1.5), *OFFSETS_DB.end());
    }
}
