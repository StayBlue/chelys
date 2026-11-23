// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! MP3 at a variable bitrate, of pure tones at the edges of the polyphase
//! filter's subbands, as ffmpeg decodes them.
//!
//! Tones above the highest scalefactor band, from 16 kHz at 44.1 and
//! 48 kHz, are not checked: that band has no allowance, and at a variable
//! bitrate they are lost (see `partitions::band_weights` in the
//! psychoacoustic model).

use std::f64::consts::TAU;

use chelys::mp3::{ChannelMode, Config, DELAY, Encoder, Rate};
use chelys_conformance::ffmpeg;

const SECONDS: usize = 2;

/// `SECONDS` of a tone of `frequency` Hz at `db` below full scale, at
/// `sample_rate`.
fn tone(sample_rate: u32, frequency: f64, db: f64) -> Vec<f64> {
    let amplitude = 10f64.powf(db / 20.0) * 2f64.sqrt();
    let fs = f64::from(sample_rate);
    (0..SECONDS * sample_rate as usize)
        .map(|t| amplitude * (TAU * frequency * t as f64 / fs).sin())
        .collect()
}

/// `input` coded in one channel at level 80 of the quality scale.
fn encode(sample_rate: u32, input: &[f64]) -> Vec<u8> {
    let rate = Rate::Quality {
        level: 80,
        min_kbps: None,
        max_kbps: None,
    };
    let mut encoder = Encoder::new(Config::new(sample_rate, ChannelMode::Mono, rate)).unwrap();
    let mut stream = Vec::new();
    encoder.encode_planar(&[input], &mut stream).unwrap();
    encoder.finish(&mut stream);
    stream
}

/// The level in dB of `samples` over the `len` samples from `start`.
fn level(samples: impl Fn(usize) -> f64, start: usize, len: usize) -> f64 {
    let power = (start..start + len)
        .map(|t| samples(t).powi(2))
        .sum::<f64>()
        / len as f64;
    10.0 * (power + 1e-30).log10()
}

/// A tone at the edge of a subband, at `fs / 8`, `fs / 16` or `fs / 32`,
/// at -40 dB below full scale, takes no more bits at level 80 than one in
/// the middle of a subband: below 40 kbit/s at 8, 16, 24 and 44.1 kHz, in
/// one channel. With the partitions of Tables C.8 and D.4, the bands beside
/// such a tone allowed almost no noise, and every frame took the highest
/// bitrate: 160 kbit/s at 16 kHz, at all three edges, and at 24 kHz at
/// `fs / 8` and `fs / 16`, and 320 kbit/s at 44.1 kHz at `fs / 8`. With
/// the partitions at the bands' own frequencies, all but the tone of 2 kHz
/// at 16 kHz took 8 to 38 kbit/s; with the threshold in quiet as the least
/// a band is allowed, that one takes 11 kbit/s too.
///
/// A tone at an edge well above the threshold in quiet, at -20 dB, and one
/// nearer it, at -70 dB, keep their level as ffmpeg decodes them, within
/// 0.5 dB in each window of 0.1 s but the last; they were 0.3 dB off at
/// worst.
#[test]
fn tones_at_subband_edges_take_few_bits_and_keep_their_level() {
    std::thread::scope(|scope| {
        for sample_rate in [8_000, 16_000, 24_000, 44_100] {
            for divisor in [8.0, 16.0, 32.0] {
                scope.spawn(move || {
                    let frequency = f64::from(sample_rate) / divisor;
                    for db in [-20.0, -40.0, -70.0] {
                        let input = tone(sample_rate, frequency, db);
                        let stream = encode(sample_rate, &input);
                        let case = format!("{sample_rate} Hz, {frequency} Hz at {db} dB");
                        let kbps = stream.len() as f64 * 8.0 / SECONDS as f64 / 1000.0;
                        if db == -40.0 {
                            assert!(kbps < 40.0, "{case}: {kbps:.1} kbit/s");
                            continue;
                        }
                        let decoded = ffmpeg::decode_mp3(&stream);
                        assert_eq!(decoded.diagnostics, "", "{case}");
                        let window = sample_rate as usize / 10;
                        let samples = &decoded.samples;
                        for start in (0..input.len() - window).step_by(window) {
                            let output = level(|t| f64::from(samples[t + DELAY]), start, window);
                            let expected = level(|t| input[t], start, window);
                            assert!(
                                (output - expected).abs() < 0.5,
                                "{case}, from {start}: {output:.2} dB"
                            );
                        }
                    }
                });
            }
        }
    });
}
