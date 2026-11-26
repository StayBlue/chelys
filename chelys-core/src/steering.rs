// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! How a mean rate is kept: the noise offset of an encoder's variable rate,
//! steered frame by frame toward the bits the mean brings.

/// The noise offset of each frame of a stream at a mean rate: that at
/// which the recent frames would have taken the bits the mean brings the
/// next, less a share of what the frames so far took beyond it.
///
/// A frame's bits fall with its noise offset by about `bits_per_db` for
/// each dB, in the logarithm, whatever its content, while what the content
/// takes at a given offset varies from stream to stream by a factor of 3
/// or more. So each frame's bits are taken back to what they would have
/// been at 0 dB, `bits * e^(bits_per_db * offset)`, and their mean over
/// the last `MEMORY_FRAMES` or so gives the offset at which the next
/// frame would take the bits it is to take: those of the mean, less the
/// bits the frames so far took beyond it spread over `REPAYMENT_FRAMES`
/// frames, but no fewer than a quarter of the mean.
///
/// The steering sets only the frames to come, and nothing makes up at the
/// end for what the frames so far took beyond the mean or short of it.
/// Frames that take few bits at any offset, of silence or of a quiet end,
/// leave the stream short by what they could not take. The first frames
/// are steered from an offset the encoder guesses, which tells little of
/// the content, and can run over or under: a stream of a few seconds can
/// miss the mean by much more than a long one.
///
/// The frames' counts were chosen with chelys's AAC encoder, whose frames
/// are of 1024 samples; each encoder that steers so gives its own
/// measures.
#[derive(Clone, Debug)]
pub struct MeanSteering {
    /// The bits of a frame at the mean.
    mean: f64,
    /// How far the logarithm of a frame's bits falls for each dB of noise
    /// offset, about.
    bits_per_db: f64,
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

/// The frames over which [`MeanSteering`] forgets a frame's bits: the
/// steering starts out as if the frames before the first had taken the
/// mean's bits at the offset it starts from.
///
/// With chelys's AAC encoder, on the tuning set of 51 tracks at 44.1 kHz
/// in stereo, at means of 64, 128 and 192 kbit/s: steered faster,
/// forgetting over 100 frames and giving back over 250, the tracks were
/// 1.6% to 1.8% short, the offset varying within a track by 3.1 dB, a
/// standard deviation, against 2.0% to 2.1% short and 2.2 dB; slower, over
/// 400 and 1000, 2.2% to 2.4% short, varying by 1.8 dB.
const MEMORY_FRAMES: f64 = 200.0;

/// The frames over which the bits the stream took beyond its mean, or
/// short of it, are given back (see `MEMORY_FRAMES`).
const REPAYMENT_FRAMES: f64 = 500.0;

/// The frames, beyond those taken so far, over which the steering forgets
/// a frame's bits and gives back what the frames took beyond the mean,
/// where they are fewer than `MEMORY_FRAMES` and `REPAYMENT_FRAMES`: at
/// the start of a stream, of which the offset it starts from tells little.
/// With chelys's AAC encoder, one second of loud white noise in stereo at
/// 44.1 kHz and 128 kbit/s took 133 kbit/s, and its first five frames 160
/// kbit/s, against 168 before the start was steered so; the tuning set's
/// means were as before, within 0.1%.
const START_MEMORY_FRAMES: f64 = 8.0;
const START_REPAYMENT_FRAMES: f64 = 40.0;

/// The noise offsets the steering keeps within, in dB.
pub const OFFSETS_DB: std::ops::RangeInclusive<f64> = -30.0..=40.0;

impl MeanSteering {
    /// The steering of a stream at `bits_per_second` and `sample_rate`, of
    /// frames of `frame_samples` samples, from the offset `start_db`, at
    /// which the encoder guesses the stream takes the mean, of an encoder
    /// whose frames' bits fall by about `bits_per_db` in the logarithm for
    /// each dB of offset.
    pub fn new(
        bits_per_second: u32,
        sample_rate: u32,
        frame_samples: usize,
        start_db: f64,
        bits_per_db: f64,
    ) -> Self {
        let mean = f64::from(bits_per_second) * frame_samples as f64 / f64::from(sample_rate);
        Self {
            mean,
            bits_per_db,
            at_0_db: mean * (bits_per_db * start_db).exp(),
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
        let at_0_db = bits * (self.bits_per_db * self.noise_offset_db).exp();
        self.frames += 1.0;
        let memory = MEMORY_FRAMES.min(self.frames + START_MEMORY_FRAMES);
        self.at_0_db += (at_0_db - self.at_0_db) / memory;
        self.excess += bits - self.mean;
        let repayment = REPAYMENT_FRAMES.min(self.frames + START_REPAYMENT_FRAMES);
        let wanted = (self.mean - self.excess / repayment).max(self.mean / 4.0);
        let offset = (self.at_0_db / wanted).ln() / self.bits_per_db;
        self.noise_offset_db = offset.clamp(*OFFSETS_DB.start(), *OFFSETS_DB.end());
    }
}

/// The bits for each sample of each channel that streams coded at
/// `noise_offset_db` take on average, about, from `nominal`: pairs of a
/// noise offset in dB and those bits, measured on music, the offset rising
/// and the bits falling. The logarithm of the bits is taken to be linear
/// between them, and beyond the ends.
pub fn nominal_bits_per_sample(nominal: &[(f64, f64)], noise_offset_db: f64) -> f64 {
    let point = |k: usize| (nominal[k].0, nominal[k].1.ln());
    interpolate(nominal.len(), point, noise_offset_db).exp()
}

/// The offset whose [`nominal_bits_per_sample`] of `nominal` is
/// `bits_per_sample`: an offset from which a mean rate may be steered.
pub fn nominal_noise_offset_db(nominal: &[(f64, f64)], bits_per_sample: f64) -> f64 {
    // The bits fall as the offset rises: the inverse, its logarithm
    // rising.
    let point = |k: usize| (-nominal[k].1.ln(), nominal[k].0);
    interpolate(nominal.len(), point, -bits_per_sample.ln())
}

/// `y` at `x` on the line through the `n` pairs `point(k)`, `x` rising:
/// between the two around `x`, or beyond the ends, the two at that end.
fn interpolate(n: usize, point: impl Fn(usize) -> (f64, f64), x: f64) -> f64 {
    let i = (1..n - 1).take_while(|&k| point(k).0 < x).count();
    let ((x0, y0), (x1, y1)) = (point(i), point(i + 1));
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

#[cfg(test)]
mod tests {
    use super::*;

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
                let mut steering = MeanSteering::new(128_000, 44_100, 1024, 1.5, 0.09);
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

    #[test]
    fn the_nominal_rate_is_interpolated_in_its_logarithm_and_inverts() {
        let nominal = [(-6.0, 4.0), (0.0, 2.0), (6.0, 0.5)];
        assert!((nominal_bits_per_sample(&nominal, -3.0) - 8f64.sqrt()).abs() < 1e-12);
        assert!((nominal_bits_per_sample(&nominal, 3.0) - 1.0).abs() < 1e-12);
        // Beyond the ends, the line through the two at that end.
        assert!((nominal_bits_per_sample(&nominal, -12.0) - 8.0).abs() < 1e-12);
        assert!((nominal_bits_per_sample(&nominal, 12.0) - 0.125).abs() < 1e-12);
        for tenth in -150..=150 {
            let offset = f64::from(tenth) / 10.0;
            let bits = nominal_bits_per_sample(&nominal, offset);
            let inverse = nominal_noise_offset_db(&nominal, bits);
            assert!((inverse - offset).abs() < 1e-9, "{offset}");
        }
    }

    /// Frames of silence, which take the same few bits at any offset, drive
    /// the offset to the lowest the steering keeps, and no further.
    #[test]
    fn the_offset_keeps_within_its_range() {
        let mut steering = MeanSteering::new(128_000, 44_100, 1024, 0.0, 0.09);
        for _ in 0..2000 {
            steering.spend(2);
        }
        assert_eq!(steering.noise_offset_db(), *OFFSETS_DB.start());
        let mut steering = MeanSteering::new(8_000, 44_100, 1024, 0.0, 0.09);
        for _ in 0..2000 {
            steering.spend(100_000);
        }
        assert_eq!(steering.noise_offset_db(), *OFFSETS_DB.end());
    }
}
