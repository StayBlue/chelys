// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! How the frames' lengths are chosen at a variable or mean rate: the
//! noise each band is allowed, and for a mean rate how it is steered.

use chelys_core::steering;
pub use chelys_core::steering::MeanSteering;

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
    steering::nominal_bits_per_sample(&NOMINAL, noise_offset_db)
}

/// The offset whose [`nominal_bits_per_sample`] is `bits_per_sample`.
pub fn nominal_noise_offset_db(bits_per_sample: f64) -> f64 {
    steering::nominal_noise_offset_db(&NOMINAL, bits_per_sample)
}

/// Pairs of a noise offset in dB and the bits a sample it takes (see
/// [`nominal_bits_per_sample`]), as
/// [`steering::nominal_bits_per_sample`] takes them.
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

/// The steering of a stream at a mean rate of `bits_per_second` at
/// `sample_rate`, which has `bits_per_sample` for each sample of each
/// channel of its NCC: from the offset of its nominal rate (see
/// [`nominal_noise_offset_db`]), each frame's bits taken to fall by
/// `BITS_PER_DB` for each dB of offset.
///
/// Spec gap: Annex C has no variable rate, and so no mean rate kept by
/// one (see [`Variable`](crate::Rate::Variable)). The offset is steered as
/// [`MeanSteering`] describes. Within music, the frames so far keep within
/// some 20 frames' bits of the mean. One second of loud white noise in
/// stereo at 44.1 kHz and 128 kbit/s took 133 kbit/s, and its first five
/// frames 160 kbit/s.
///
/// On the tuning set of 51 tracks at 44.1 kHz in stereo, at means of 64,
/// 128 and 192 kbit/s, the tracks were 2.1%, 2.0% and 2.0% short on
/// average, 7.7% at most, the one ending in some 13 s of a quiet fade, and
/// none over.
pub fn mean_steering(bits_per_second: u32, sample_rate: u32, bits_per_sample: f64) -> MeanSteering {
    MeanSteering::new(
        bits_per_second,
        sample_rate,
        crate::stream::FRAME_SAMPLES,
        nominal_noise_offset_db(bits_per_sample),
        BITS_PER_DB,
    )
}

/// How far the logarithm of a frame's bits falls for each dB of noise
/// offset, about: on the tuning set of 51 tracks at 44.1 kHz in stereo,
/// their mean bitrate fell from 151 to 72 kbit/s from -3 to 6 dB (0.083),
/// and from 124 to 72 from 0 to 6 dB (0.091).
const BITS_PER_DB: f64 = 0.09;

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
}
