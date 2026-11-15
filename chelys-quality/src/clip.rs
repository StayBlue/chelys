// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! An original and its encoding, decoded and lined up sample for sample.

use chelys::mp3::DELAY;
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

impl Clip {
    /// Reads the original's raw PCM, and decodes its encoding with ffmpeg.
    pub fn load(
        sample_rate: u32,
        channels: usize,
        original: &str,
        encoded: &str,
    ) -> Result<Self, String> {
        let bytes = std::fs::read(original).map_err(|e| format!("reading {original}: {e}"))?;
        if bytes.len() % (4 * channels) != 0 {
            return Err(format!(
                "{original} is not a whole number of {}-byte sample frames",
                4 * channels
            ));
        }
        let original_samples: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        let stream = std::fs::read(encoded).map_err(|e| format!("reading {encoded}: {e}"))?;
        let info = ffmpeg::probe_mp3(&stream);
        if info.sample_rate != sample_rate || info.channels != channels {
            return Err(format!(
                "{encoded} has {} channel(s) at {} Hz, not {channels} at {sample_rate} Hz",
                info.channels, info.sample_rate
            ));
        }
        // ffmpeg applies no delay from chelys's Xing/Info tag, so its output
        // lags the input by the encoder's delay, with or without the tag
        // (see `DELAY`).
        let decoded = ffmpeg::decode_mp3(&stream).samples;
        let start = DELAY * channels;
        let end = start + original_samples.len();
        let encoded_samples = decoded
            .get(start..end)
            .ok_or(format!(
                "{encoded} decodes to fewer samples than {original} holds; \
                 is it chelys's encoding of it?"
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
