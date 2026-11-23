// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The PCM samples the encoders take, and their conversion to the scale the
//! encoders work at.

use core::fmt;

/// A PCM sample the encoder takes: `f32` and `f64` at a nominal full scale
/// of -1.0 to 1.0, and `i16` at its own, read as `x / 32768`.
pub trait Sample: Copy + sealed::Sealed {}

impl Sample for f32 {}
impl Sample for f64 {}
impl Sample for i16 {}

mod sealed {
    pub trait Sealed: Copy {
        fn to_f64(self) -> f64;

        fn is_finite(self) -> bool;
    }

    impl Sealed for f32 {
        fn to_f64(self) -> f64 {
            f64::from(self)
        }

        fn is_finite(self) -> bool {
            f32::is_finite(self)
        }
    }

    impl Sealed for f64 {
        fn to_f64(self) -> f64 {
            self
        }

        fn is_finite(self) -> bool {
            f64::is_finite(self)
        }
    }

    impl Sealed for i16 {
        fn to_f64(self) -> f64 {
            f64::from(self) / 32_768.0
        }

        fn is_finite(self) -> bool {
            true
        }
    }
}

/// A sample that is NaN or infinite, for which an encoder refuses the
/// whole of the call that holds it.
///
/// Where a call holds more than one, it names the first to be played: the
/// one with the lowest `sample`, and of those the one with the lowest
/// `channel`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct NonFiniteSample {
    /// The sample's channel, from 0.
    pub channel: usize,
    /// The sample's index within its channel, from 0 at the start of the
    /// call: `channel + sample * channels` in interleaved input,
    /// `channels[channel][sample]` in planar.
    pub sample: usize,
}

impl fmt::Display for NonFiniteSample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "sample {} of channel {} is not a finite number",
            self.sample, self.channel
        )
    }
}

impl std::error::Error for NonFiniteSample {}

/// Checks that every sample of interleaved input, one of each of
/// `channels` in turn, is finite.
///
/// # Errors
///
/// The first sample to be played that is NaN or infinite.
///
/// # Panics
///
/// If `channels` is 0.
pub fn check_interleaved<S: Sample>(samples: &[S], channels: usize) -> Result<(), NonFiniteSample> {
    assert!(channels > 0, "no channels");
    match samples.iter().position(|&s| !sealed::Sealed::is_finite(s)) {
        None => Ok(()),
        Some(i) => Err(NonFiniteSample {
            channel: i % channels,
            sample: i / channels,
        }),
    }
}

/// Checks that every sample of planar input, one slice per channel, is
/// finite.
///
/// # Errors
///
/// The first sample to be played that is NaN or infinite.
pub fn check_planar<S: Sample>(channels: &[&[S]]) -> Result<(), NonFiniteSample> {
    let mut first: Option<NonFiniteSample> = None;
    for (channel, samples) in channels.iter().enumerate() {
        // Only a sample before the first found so far can take its place.
        let end = first.map_or(samples.len(), |f| f.sample.min(samples.len()));
        if let Some(sample) = samples[..end]
            .iter()
            .position(|&s| !sealed::Sealed::is_finite(s))
        {
            first = Some(NonFiniteSample { channel, sample });
        }
    }
    first.map_or(Ok(()), Err)
}

/// The magnitude past which a sample is clamped: 65536 times full scale,
/// 96 dB above it.
///
/// Spec gap: the standards take PCM within full scale, and say nothing of
/// samples beyond it. Those somewhat beyond, as floating-point PCM may
/// have, are coded as given, and a decoder that does not clip gives them
/// back; but each codec's quantisation reaches only so far. In MP3, past
/// some ten million times full scale a granule no longer quantises within
/// the range of `global_gain`; in AAC, past some hundred million times
/// full scale the quantizer's first step (clause C.7.2.1 of ISO/IEC
/// 13818-7) is beyond the range of its `global_gain`. The clamp keeps
/// every finite sample far below both and far above any real signal.
pub const MAX_MAGNITUDE: f64 = 65_536.0;

/// The sample at the encoders' scale, full scale at 1.0, clamped to
/// [`MAX_MAGNITUDE`].
///
/// The sample is to be finite, as [`check_interleaved`] and
/// [`check_planar`] find it. What a sample that is not gives is
/// unspecified; a debug build panics on it.
pub fn convert<S: Sample>(sample: S) -> f64 {
    let x = sealed::Sealed::to_f64(sample);
    debug_assert!(x.is_finite(), "{x} is not a finite number");
    x.clamp(-MAX_MAGNITUDE, MAX_MAGNITUDE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scales_each_type_to_full_scale_one() {
        assert_eq!(convert(-32_768i16), -1.0);
        assert_eq!(convert(16_384i16), 0.5);
        assert_eq!(convert(0.25f32), 0.25);
        assert_eq!(convert(-0.75f64), -0.75);
    }

    #[test]
    fn clamps_far_beyond_full_scale() {
        assert_eq!(convert(1e12f64), MAX_MAGNITUDE);
        assert_eq!(convert(-f32::MAX), -MAX_MAGNITUDE);
        assert_eq!(convert(MAX_MAGNITUDE), MAX_MAGNITUDE);
    }

    fn at(channel: usize, sample: usize) -> Result<(), NonFiniteSample> {
        Err(NonFiniteSample { channel, sample })
    }

    #[test]
    fn accepts_what_is_finite() {
        assert_eq!(
            check_interleaved(&[0.0f32, -f32::MAX, f32::MIN_POSITIVE], 1),
            Ok(())
        );
        assert_eq!(check_interleaved(&[i16::MIN, i16::MAX], 2), Ok(()));
        assert_eq!(check_planar::<f64>(&[&[1e300], &[-1e300]]), Ok(()));
        assert_eq!(check_planar::<f64>(&[&[], &[]]), Ok(()));
    }

    #[test]
    fn refuses_what_is_not_a_finite_number() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(check_interleaved(&[0.0, bad], 1), at(0, 1));
            assert_eq!(check_planar(&[&[0.0, bad][..]]), at(0, 1));
        }
        assert_eq!(check_interleaved(&[0.0, 0.0, f32::NAN], 3), at(2, 0));
    }

    /// Both layouts name the first sample to be played, and of the samples
    /// played at once the lowest channel, whichever channel is scanned
    /// first.
    #[test]
    fn names_the_first_to_be_played() {
        let nan = f64::NAN;
        let planar: [&[f64]; 3] = [
            &[0.0, 0.0, 0.0, nan],
            &[0.0, nan, nan, 0.0],
            &[0.0, nan, 0.0, 0.0],
        ];
        let mut interleaved = Vec::new();
        for sample in 0..4 {
            interleaved.extend(planar.iter().map(|c| c[sample]));
        }
        assert_eq!(check_planar(&planar), at(1, 1));
        assert_eq!(check_interleaved(&interleaved, 3), at(1, 1));
    }
}
