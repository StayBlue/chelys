// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! An original and its encoding, decoded and lined up sample for sample.

use chelys_conformance::ffmpeg;

pub struct Clip {
    pub sample_rate: u32,
    pub channels: usize,
    /// Interleaved samples.
    pub original: Vec<f32>,
    /// Interleaved samples, as many as in `original`, each at the time of
    /// the original's sample in the same place.
    pub encoded: Vec<f32>,
}

/// The longest delay [`lag`] tries, in samples per channel.
const MAX_LAG: usize = 8192;

/// The length of the part of the original [`lag`] lines up, in seconds.
const LAG_SECONDS: usize = 5;

impl Clip {
    /// Reads the original's raw PCM, decodes its encoding with ffmpeg, and
    /// drops the first `delay` samples of each channel of the decoding.
    ///
    /// ffmpeg applies no delay from chelys's Xing/Info tag, so its output
    /// lags the input by the encoder's delay, with or without the tag (see
    /// [`DELAY`](chelys::mp3::DELAY)); another encoder's encoding may lag by
    /// another delay, which [`lag`] finds.
    pub fn load(
        sample_rate: u32,
        channels: usize,
        original: &str,
        encoded: &str,
        delay: usize,
    ) -> Result<Self, String> {
        let (original_samples, decoded) = read(sample_rate, channels, original, encoded)?;
        let start = delay * channels;
        let end = start + original_samples.len();
        let encoded_samples = decoded
            .get(start..end)
            .ok_or(format!(
                "{encoded} decodes to fewer samples than {original} holds after a delay of \
                 {delay}; is it an encoding of it, and is the delay right?"
            ))?
            .to_vec();
        Ok(Self {
            sample_rate,
            channels,
            original: original_samples,
            encoded: encoded_samples,
        })
    }

    /// The part from `from` seconds to `to`, each the clip's start or end
    /// if not given.
    pub fn section(self, from: Option<f64>, to: Option<f64>) -> Result<Self, String> {
        let frames = self.original.len() / self.channels;
        let seconds = frames as f64 / f64::from(self.sample_rate);
        let (from_s, to_s) = (from.unwrap_or(0.0), to.unwrap_or(seconds));
        if !(0.0 <= from_s && from_s < to_s && to_s <= seconds) {
            return Err(format!(
                "--from must come before --to, both within the clip's {seconds:.2} s"
            ));
        }
        let frame = |s: f64| (s * f64::from(self.sample_rate)).round() as usize * self.channels;
        let range = frame(from_s)..frame(to_s).min(self.original.len());
        Ok(Self {
            original: self.original[range.clone()].to_vec(),
            encoded: self.encoded[range].to_vec(),
            ..self
        })
    }

    /// The encoding less the original: the noise that coding added.
    pub fn difference(&self) -> Vec<f32> {
        self.encoded
            .iter()
            .zip(&self.original)
            .map(|(encoded, original)| encoded - original)
            .collect()
    }
}

/// The original's samples and ffmpeg's decoding of its encoding, both
/// interleaved.
fn read(
    sample_rate: u32,
    channels: usize,
    original: &str,
    encoded: &str,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let bytes = std::fs::read(original).map_err(|e| format!("reading {original}: {e}"))?;
    if bytes.len() % (4 * channels) != 0 {
        return Err(format!(
            "{original} is not a whole number of {}-byte sample frames",
            4 * channels
        ));
    }
    let original_samples = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let stream = std::fs::read(encoded).map_err(|e| format!("reading {encoded}: {e}"))?;
    let aac = is_aac(encoded);
    let info = if aac {
        ffmpeg::probe_aac(&stream).0
    } else {
        ffmpeg::probe_mp3(&stream)
    };
    if info.sample_rate != sample_rate || info.channels != channels {
        return Err(format!(
            "{encoded} has {} channel(s) at {} Hz, not {channels} at {sample_rate} Hz",
            info.channels, info.sample_rate
        ));
    }
    let decoded = if aac {
        ffmpeg::decode_aac(&stream)
    } else {
        ffmpeg::decode_mp3(&stream)
    };
    Ok((original_samples, decoded.samples))
}

/// Whether `encoded` is to be read as AAC in ADTS, by its extension `.aac`,
/// rather than as MP3.
pub fn is_aac(encoded: &str) -> bool {
    std::path::Path::new(encoded)
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("aac"))
}

/// The delay, in samples per channel from 0 to [`MAX_LAG`], by which
/// ffmpeg's decoding of the encoding lags the original: where the two,
/// mixed to mono, correlate best over the loudest [`LAG_SECONDS`] of the
/// original.
pub fn lag(
    sample_rate: u32,
    channels: usize,
    original: &str,
    encoded: &str,
) -> Result<usize, String> {
    let (original, decoded) = read(sample_rate, channels, original, encoded)?;
    let mono = |samples: &[f32]| -> Vec<f32> {
        samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum())
            .collect()
    };
    let (original, decoded) = (mono(&original), mono(&decoded));
    best_lag(&original, &decoded, LAG_SECONDS * sample_rate as usize).ok_or(format!(
        "too short to line up: needs {LAG_SECONDS} s of the original and \
         {MAX_LAG} samples more of the encoding"
    ))
}

/// The lag, from 0 to [`MAX_LAG`], at which `decoded` correlates best with
/// the `len` samples of `original` that hold the most energy, starting on a
/// multiple of `len / LAG_SECONDS`; `None` if either is too short.
fn best_lag(original: &[f32], decoded: &[f32], len: usize) -> Option<usize> {
    let span = original.len().min(decoded.len().checked_sub(MAX_LAG)?);
    if len == 0 || span < len {
        return None;
    }
    let step = (len / LAG_SECONDS).max(1);
    let energy = |start: usize| -> f64 {
        original[start..start + len]
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum()
    };
    let start = (0..=(span - len) / step)
        .map(|s| s * step)
        .max_by(|&a, &b| energy(a).total_cmp(&energy(b)))?;
    let part = &original[start..start + len];
    let correlation = |lag: usize| -> f64 {
        let decoded = &decoded[start + lag..start + lag + len];
        let (mut dot, mut power) = (0.0, 0.0);
        for (&x, &y) in part.iter().zip(decoded) {
            dot += f64::from(x) * f64::from(y);
            power += f64::from(y) * f64::from(y);
        }
        if power == 0.0 {
            0.0
        } else {
            dot / power.sqrt()
        }
    };
    (0..=MAX_LAG).max_by(|&a, &b| correlation(a).total_cmp(&correlation(b)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Noise from a linear congruential generator, in [-1, 1).
    fn noise(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    /// A quieter, noisier copy delayed by a known lag is found at that lag,
    /// at either end of the range too.
    #[test]
    fn the_lag_of_a_delayed_copy_is_found() {
        let len = 2000;
        let original = noise(3 * len, 1);
        let added = noise(3 * len + MAX_LAG, 2);
        for lag in [0, 1, 1057, MAX_LAG] {
            let mut decoded = vec![0.0; lag];
            decoded.extend(original.iter().map(|x| 0.5 * x));
            decoded.resize(3 * len + MAX_LAG, 0.0);
            for (y, n) in decoded.iter_mut().zip(&added) {
                *y += 0.1 * n;
            }
            assert_eq!(best_lag(&original, &decoded, len), Some(lag));
        }
        assert_eq!(best_lag(&original, &original, len), None);
    }
}
