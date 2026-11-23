// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! How the frames' lengths are chosen at a variable or mean rate: the
//! noise each band is allowed, and for a mean rate how it is steered.

/// How the frames' lengths are chosen, as a [`StreamConfig`] has it: a
/// [`Rate`](crate::Rate) with what it fixes for the stream.
///
/// [`StreamConfig`]: crate::stream::StreamConfig
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Control {
    /// The bit reservoir keeps the stream at its bitrate (clause 8.2.3.2).
    Constant,
    /// Each frame within its allowance, raised by `noise_offset_db`.
    Variable { noise_offset_db: f64 },
    /// Each frame within its allowance, raised by an offset steered toward
    /// the stream's bitrate (see [`MeanSteering`]).
    Mean,
}

impl Control {
    /// Whether the frames are coded within their allowance, as long as that
    /// takes, rather than as long as a constant rate brings.
    pub fn is_variable(self) -> bool {
        self != Self::Constant
    }
}

/// The noise offsets over which, at a variable or mean rate, the model's
/// reading of the threshold in quiet goes from the threshold as noise has
/// it, at the lower end and below, to the threshold as printed, at the
/// upper end and above (see [`StreamConfig::quiet_threshold`]), and the
/// offset is lowered by as much of [`PRINTED_QUIET_SHIFT_DB`]: 3 dB either
/// side of 6.1 dB, where the tuning set of 51 tracks at 44.1 kHz in stereo
/// takes [`PRINTED_QUIET_BITS_PER_SAMPLE`] with the threshold as noise has
/// it.
///
/// The reading as printed raises the allowance of every band by 3 dB at
/// least (see [`raise_xmin`]), and the shift lowers it by 6.2 dB: over 6
/// dB of the offset, the allowance of a band that masking governs rises by
/// 1 - 3.2 / 6 dB for each dB, and that of a band the threshold in quiet
/// governs by more. Over 3.2 dB or less it would not rise.
///
/// [`StreamConfig::quiet_threshold`]: crate::stream::StreamConfig::quiet_threshold
/// [`PRINTED_QUIET_BITS_PER_SAMPLE`]: crate::stream::PRINTED_QUIET_BITS_PER_SAMPLE
/// [`raise_xmin`]: crate::loops::raise_xmin
pub const BLENDED_QUIET_DB: std::ops::RangeInclusive<f64> = 3.1..=9.1;

/// The noise offsets over which, at a variable or mean rate, the lower
/// border of a pair's bands in intensity stereo falls from half the
/// sampling frequency to Annex C's (see
/// [`StreamConfig::intensity_border`]): those at which the tuning set of
/// 51 tracks at 44.1 kHz in stereo took 1.0 and 0.75 bits a sample, the
/// rates between which, at constant rates, the limit of
/// [`INTENSITY_BITS_PER_SAMPLE`] lies, before the threshold in quiet was
/// lowered at low offsets (see [`LOWERED_QUIET_DB`]); with it lowered, 1.11
/// and 0.74.
///
/// [`StreamConfig::intensity_border`]: crate::stream::StreamConfig::intensity_border
/// [`INTENSITY_BITS_PER_SAMPLE`]: crate::stream::INTENSITY_BITS_PER_SAMPLE
pub const INTENSITY_DB: std::ops::RangeInclusive<f64> = 3.8..=7.0;

/// How far, in dB, the threshold in quiet is lowered at a variable or mean
/// rate, at noise offsets up to the start of [`LOWERED_QUIET_DB`] (see
/// [`StreamConfig::quiet_lowered_db`]).
///
/// [`StreamConfig::quiet_lowered_db`]: crate::stream::StreamConfig::quiet_lowered_db
pub const QUIET_LOWERING_DB: f64 = 20.0;

/// The noise offsets over which the lowering of the threshold in quiet
/// falls from [`QUIET_LOWERING_DB`], at the lower end and below, to none,
/// at the upper end and above, evenly in dB.
pub const LOWERED_QUIET_DB: std::ops::RangeInclusive<f64> = 0.0..=6.0;

/// How far the threshold in quiet is lowered, in dB, at a variable or mean
/// rate whose tools are chosen by `noise_offset_db` (see
/// [`LOWERED_QUIET_DB`]).
pub fn quiet_lowered_db(noise_offset_db: f64) -> f64 {
    let (low, high) = LOWERED_QUIET_DB.into_inner();
    QUIET_LOWERING_DB * ((high - noise_offset_db) / (high - low)).clamp(0.0, 1.0)
}

/// How far the noise offset of a variable or mean rate is lowered before it
/// raises the allowed distortion where the model takes the threshold in
/// quiet as printed, and by as much of it in dB as it takes it so where it
/// takes it between the two (see [`BLENDED_QUIET_DB`]).
///
/// The threshold in quiet as printed, with the offset of clause C.7.4.4
/// that the loops then add, allows far more noise in the bands it governs
/// than the threshold as noise has it: the tuning set took 0.79 bits a
/// sample with the threshold as printed at 0 dB, and 0.81 as noise has it
/// at 6 dB. Lowered by this much, the offsets at which either reading
/// takes a rate are about the same, and the rate falls with the offset at
/// about the same pace through the blend as on either side of it.
pub const PRINTED_QUIET_SHIFT_DB: f64 = 6.2;

/// The fewest bits a frame takes at a variable or mean rate, as a share of
/// those of a frame at the stream's nominal rate or mean: a frame that
/// takes fewer within its allowance is fitted to these by the loops, as at
/// a constant rate, and the bits they leave spent (see
/// [`spend`](crate::loops::allowance::spend)).
///
/// Spec gap: within its allowance, a frame codes nothing that the model
/// takes as inaudible, and of a quiet passage, nothing below the threshold
/// in quiet, which the model puts at the level of an LSB of 16-bit PCM
/// (step 11 of clause C.1.4), whatever the level it is played at; nor, of
/// a pair coded as middle and side, a side channel below its allowance. At
/// a constant rate, the bits the rate brings still code such content; at
/// a variable rate, nothing would, and fades and the tails of notes would
/// end early, their stereo image narrowing as they do. Of the tuning set of
/// 51 tracks at 44.1 kHz in stereo, before the threshold in quiet was
/// lowered (see [`LOWERED_QUIET_DB`]), the frames so raised took 0.2% more
/// bits at 0 dB.
///
/// Settled by decoding with ffmpeg (black box): pink noise panned 8 dB to
/// the left and fading out from -30 dB below full scale, coded at 0 dB
/// without a least length, fell to silence from -70 dB, and its two
/// channels to one level from -52 dB, where at a constant 128 and 64
/// kbit/s it was coded to -115 dB, its panning within 1 dB. With frames of
/// at least a quarter of the nominal rate's bits, it was coded to -115 dB
/// too, its panning within 1.2 dB at 0 dB and 2 dB at 10 dB. The
/// conformance test is `a_panned_fade_keeps_its_image`.
pub const LEAST_BITS_OF_THE_MEAN: f64 = 0.25;

/// The factor `noise_offset_db` raises the allowed distortion by.
pub fn noise_factor(noise_offset_db: f64) -> f64 {
    10f64.powf(noise_offset_db / 10.0)
}

/// The bits for each sample of each channel of the NCC that a stream
/// coded within its allowance raised by `noise_offset_db` takes, about, by
/// which perceptual noise substitution is chosen at a variable rate, and
/// its frames' least bits are set; [`nominal_noise_offset_db`] is the
/// inverse, from which a mean rate is first steered, and which chooses its
/// other tools.
///
/// Measured with each offset of `NOMINAL` on the tuning set of 51 tracks
/// of 9 kinds of music at 44.1 kHz in stereo: the mean of the tracks'
/// bitrates, with the tools so chosen, before the bits the loops leave
/// were spent (see [`spend`](crate::loops::allowance::spend)), which
/// raised them by 0.2% to 0.5% from -15 to 7.6 dB. It is taken as the
/// same at every sampling frequency and in every channel configuration.
///
/// Spec gap: Annex C chooses no tool by the rate, as it has only one
/// rate. The limits on the bits for each sample by which chelys turns
/// intensity stereo, perceptual noise substitution and the threshold in
/// quiet as noise has it on and off were chosen at constant rates (see
/// [`INTENSITY_BITS_PER_SAMPLE`]); at a variable rate the bits of each
/// frame follow its content, and keying the tools to them would turn them
/// on and off from frame to frame, and with them the stereo image and the
/// bandwidth. They are keyed instead to the stream's offset, fixed for the
/// stream, or to its nominal rate, that which streams take on average at
/// the offset: intensity stereo and the reading of the threshold in quiet
/// come in over ranges of the offset, so that the length falls with the
/// offset without a break (see [`StreamConfig::intensity_border`] and
/// [`StreamConfig::quiet_threshold`]); perceptual noise substitution, which
/// is asked for, switches at a nominal rate, and was not graded so.
///
/// [`INTENSITY_BITS_PER_SAMPLE`]: crate::stream::INTENSITY_BITS_PER_SAMPLE
/// [`StreamConfig::quiet_threshold`]: crate::stream::StreamConfig::quiet_threshold
/// [`StreamConfig::intensity_border`]: crate::stream::StreamConfig::intensity_border
pub fn nominal_bits_per_sample(noise_offset_db: f64) -> f64 {
    let log = NOMINAL.map(|(offset, bits)| (offset, bits.ln()));
    interpolate(&log, noise_offset_db).exp()
}

/// The offset whose [`nominal_bits_per_sample`] is `bits_per_sample`.
pub fn nominal_noise_offset_db(bits_per_sample: f64) -> f64 {
    // The bits fall as the offset rises: the inverse, its logarithm
    // rising.
    let inverse = NOMINAL.map(|(offset, bits)| (-bits.ln(), offset));
    interpolate(&inverse, -bits_per_sample.ln())
}

/// Pairs of a noise offset in dB and the bits a sample it takes (see
/// [`nominal_bits_per_sample`]), the offset rising: the logarithm of the
/// bits is taken to be linear between them, and beyond the ends.
const NOMINAL: [(f64, f64); 17] = [
    (-15.0, 3.504),
    (-12.0, 3.079),
    (-9.0, 2.687),
    (-6.0, 2.322),
    (-3.0, 1.963),
    (0.0, 1.615),
    (3.0, 1.200),
    (3.8, 1.107),
    (4.6, 1.009),
    (5.4, 0.904),
    (6.1, 0.814),
    (7.0, 0.736),
    (7.6, 0.698),
    (9.0, 0.614),
    (12.0, 0.428),
    (15.0, 0.255),
    (18.0, 0.153),
];

/// `y` at `x` on the line through the pairs `points`, `x` rising: between
/// the two around `x`, or beyond the ends, the two at that end.
fn interpolate(points: &[(f64, f64)], x: f64) -> f64 {
    let i = (points[1..points.len() - 1].iter())
        .take_while(|&&(px, _)| px < x)
        .count();
    let ((x0, y0), (x1, y1)) = (points[i], points[i + 1]);
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

/// The noise offset of each frame of a stream at a mean rate: that at
/// which the recent frames would have taken the bits the mean brings the
/// next, less a share of what the frames so far took beyond it.
///
/// Spec gap: Annex C has no variable rate, and so no mean rate kept by
/// one (see [`Variable`](crate::Rate::Variable)). A frame's bits fall
/// with its noise offset by about `BITS_PER_DB` for each dB, in the
/// logarithm, whatever its content, while what the content takes at a
/// given offset varies from stream to stream by a factor of 3 or more. So
/// each frame's bits are taken back to what they would have been at 0 dB,
/// `bits * e^(BITS_PER_DB * offset)`, and their mean over the last
/// `MEMORY_FRAMES` or so gives the offset at which the next frame would
/// take the bits it is to take: those of the mean, less the bits the
/// frames so far took beyond it spread over `REPAYMENT_FRAMES` frames.
///
/// Within music, the frames so far keep within some 20 frames' bits of
/// the mean. The steering sets only the frames to come, and nothing makes
/// up at the end for what the frames so far took beyond the mean or
/// short of it. Frames that take few bits at any offset, of silence or of
/// a quiet end, leave the stream short by what they could not take. The
/// first frames are steered from the nominal rate's offset, which tells
/// little of the content, and can run over or under: a stream of a few
/// seconds can miss the mean by much more than a long one. One second of
/// loud white noise in stereo at 44.1 kHz and 128 kbit/s took 133 kbit/s,
/// and its first five frames 160 kbit/s; 168 kbit/s before the start was
/// steered faster (see `START_MEMORY_FRAMES`).
///
/// On the tuning set of 51 tracks at 44.1 kHz in stereo, at means of 64,
/// 128 and 192 kbit/s, the tracks were 2.1%, 2.0% and 2.0% short on
/// average, 7.7% at most, the one ending in some 13 s of a quiet fade, and
/// none over. Steered faster, forgetting over 100 frames and giving back
/// over 250, they were 1.6% to 1.8% short, the offset varying within a
/// track by 3.1 dB, a standard deviation, against 2.2 dB; slower, over 400
/// and 1000, 2.2% to 2.4% short, varying by 1.8 dB.
#[derive(Clone, Debug)]
pub struct MeanSteering {
    /// The bits of a frame at the mean.
    mean: f64,
    /// The bits the recent frames would have taken at 0 dB, a mean that
    /// forgets each frame over `MEMORY_FRAMES`.
    at_0_db: f64,
    /// The frames taken so far.
    frames: f64,
    /// The bits the frames so far took beyond the mean; below it if
    /// negative.
    excess: f64,
    /// The noise offset of the next frame.
    noise_offset_db: f64,
}

/// How far the logarithm of a frame's bits falls for each dB of noise
/// offset, about: on the tuning set of 51 tracks at 44.1 kHz in stereo,
/// their mean bitrate fell from 151 to 72 kbit/s from -3 to 6 dB (0.083),
/// and from 124 to 72 from 0 to 6 dB (0.091).
const BITS_PER_DB: f64 = 0.09;

/// The frames over which [`MeanSteering`] forgets a frame's bits: the
/// steering starts out as if the frames before the first had taken the
/// nominal rate's bits at its offset.
const MEMORY_FRAMES: f64 = 200.0;

/// The frames over which the bits the stream took beyond its mean, or
/// short of it, are given back.
const REPAYMENT_FRAMES: f64 = 500.0;

/// The frames, beyond those taken so far, over which the steering forgets
/// a frame's bits and gives back what the frames took beyond the mean,
/// where they are fewer than [`MEMORY_FRAMES`] and [`REPAYMENT_FRAMES`]: at
/// the start of a stream, of which the nominal rate's offset tells little.
/// The tuning set's means were as before, within 0.1%.
const START_MEMORY_FRAMES: f64 = 8.0;
const START_REPAYMENT_FRAMES: f64 = 40.0;

/// The noise offsets a mean rate steers within.
const OFFSETS_DB: std::ops::RangeInclusive<f64> = -30.0..=40.0;

impl MeanSteering {
    /// The steering of a stream at `bits_per_second` and `sample_rate`,
    /// which has `bits_per_sample` for each sample of each channel of its
    /// NCC.
    pub fn new(bits_per_second: u32, sample_rate: u32, bits_per_sample: f64) -> Self {
        let mean = f64::from(bits_per_second) * crate::stream::FRAME_SAMPLES as f64
            / f64::from(sample_rate);
        let start_db = nominal_noise_offset_db(bits_per_sample);
        Self {
            mean,
            at_0_db: mean * (BITS_PER_DB * start_db).exp(),
            frames: 0.0,
            excess: 0.0,
            noise_offset_db: start_db,
        }
    }

    /// The noise offset of the next frame.
    pub fn noise_offset_db(&self) -> f64 {
        self.noise_offset_db
    }

    /// Takes a frame of `len` bytes, coded at [`Self::noise_offset_db`].
    pub fn spend(&mut self, len: usize) {
        let bits = 8.0 * len as f64;
        let at_0_db = bits * (BITS_PER_DB * self.noise_offset_db).exp();
        self.frames += 1.0;
        let memory = MEMORY_FRAMES.min(self.frames + START_MEMORY_FRAMES);
        self.at_0_db += (at_0_db - self.at_0_db) / memory;
        self.excess += bits - self.mean;
        let repayment = REPAYMENT_FRAMES.min(self.frames + START_REPAYMENT_FRAMES);
        let wanted = (self.mean - self.excess / repayment).max(self.mean / 4.0);
        let offset = (self.at_0_db / wanted).ln() / BITS_PER_DB;
        self.noise_offset_db = offset.clamp(*OFFSETS_DB.start(), *OFFSETS_DB.end());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_nominal_rate_falls_with_the_offset_and_inverts() {
        let mut last = f64::INFINITY;
        for tenth in -150..=300 {
            let offset = f64::from(tenth) / 10.0;
            let bits = nominal_bits_per_sample(offset);
            assert!(bits < last, "{offset}");
            assert!(
                (nominal_noise_offset_db(bits) - offset).abs() < 1e-9,
                "{offset}"
            );
            last = bits;
        }
    }

    /// Frames whose bits fall with the offset more slowly or faster than
    /// the steering takes them to, of passages loud and quiet, keep the
    /// mean over a stream of a few minutes within 3%; and a quiet end,
    /// whose frames take few bits at any offset, leaves it short by what
    /// those frames could not take.
    #[test]
    fn a_mean_rate_keeps_its_mean() {
        let mean = 128_000.0 * 1024.0 / 44_100.0;
        for slope in [0.05, 0.09, 0.14] {
            for quiet_end in [false, true] {
                let mut steering = MeanSteering::new(128_000, 44_100, 1.45);
                let frames = 8000;
                let (mut total, mut least) = (0.0, 0.0);
                for k in 0..frames {
                    // Passages of 20 s, a third at 3 times the bits of the
                    // rest.
                    let level = if (k / 860) % 3 == 0 { 3.0 } else { 1.0 };
                    let at_0_db = 1.2 * mean * level;
                    let offset = steering.noise_offset_db();
                    let bits = if quiet_end && k >= frames - 400 {
                        100.0
                    } else {
                        (at_0_db * (-slope * offset).exp()).max(100.0)
                    };
                    let len = (bits / 8.0).round() as usize;
                    steering.spend(len);
                    total += 8.0 * len as f64;
                    if bits == 100.0 {
                        least += mean - 8.0 * len as f64;
                    }
                }
                let error = total / (frames as f64 * mean) - 1.0;
                let short = least / (frames as f64 * mean);
                assert!(
                    (error + short).abs() < 0.03,
                    "slope {slope}, quiet end {quiet_end}: {error:+.4}, {short:+.4}"
                );
            }
        }
    }
}
