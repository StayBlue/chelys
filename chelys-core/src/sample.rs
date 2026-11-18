// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The PCM samples the encoders take, and their conversion to the scale the
//! encoders work at.

/// A PCM sample the encoder takes: `f32` and `f64` at a nominal full scale
/// of -1.0 to 1.0, and `i16` at its own, read as `x / 32768`.
pub trait Sample: Copy + sealed::Sealed {}

impl Sample for f32 {}
impl Sample for f64 {}
impl Sample for i16 {}

mod sealed {
    pub trait Sealed {
        fn to_f64(self) -> f64;
    }

    impl Sealed for f32 {
        fn to_f64(self) -> f64 {
            f64::from(self)
        }
    }

    impl Sealed for f64 {
        fn to_f64(self) -> f64 {
            self
        }
    }

    impl Sealed for i16 {
        fn to_f64(self) -> f64 {
            f64::from(self) / 32_768.0
        }
    }
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
/// [`MAX_MAGNITUDE`]; `None` if it is NaN or infinite, which each codec
/// treats as its API says.
pub fn convert<S: Sample>(sample: S) -> Option<f64> {
    let x = sealed::Sealed::to_f64(sample);
    x.is_finite()
        .then(|| x.clamp(-MAX_MAGNITUDE, MAX_MAGNITUDE))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scales_each_type_to_full_scale_one() {
        assert_eq!(convert(-32_768i16), Some(-1.0));
        assert_eq!(convert(16_384i16), Some(0.5));
        assert_eq!(convert(0.25f32), Some(0.25));
        assert_eq!(convert(-0.75f64), Some(-0.75));
    }

    #[test]
    fn clamps_far_beyond_full_scale() {
        assert_eq!(convert(1e12f64), Some(MAX_MAGNITUDE));
        assert_eq!(convert(-f32::MAX), Some(-MAX_MAGNITUDE));
        assert_eq!(convert(MAX_MAGNITUDE), Some(MAX_MAGNITUDE));
    }

    #[test]
    fn refuses_what_is_not_a_finite_number() {
        assert_eq!(convert(f64::NAN), None);
        assert_eq!(convert(f32::INFINITY), None);
        assert_eq!(convert(f64::NEG_INFINITY), None);
    }
}
