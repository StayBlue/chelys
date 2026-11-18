// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Audio through the AAC encoder, decoded by ffmpeg and compared with the
//! input.

use std::f64::consts::TAU;

use chelys::aac::{Config, ConfigError, DELAY, Encoder, Finished, Rate, SAMPLE_RATES};
use chelys_conformance::ffmpeg;

/// A stream in ADTS, with the length of each frame.
struct Encoded {
    stream: Vec<u8>,
    frame_lens: Vec<usize>,
    finished: Finished,
}

fn config(sample_rate: u32, channels: usize, bits_per_second: u32) -> Config {
    Config::new(sample_rate, channels, Rate::Constant { bits_per_second })
}

/// Encodes planar `input` at `bits_per_second`.
fn encode(sample_rate: u32, input: &[Vec<f64>], bits_per_second: u32) -> Encoded {
    let mut encoder = Encoder::new(config(sample_rate, input.len(), bits_per_second)).unwrap();
    let mut stream = Vec::new();
    let mut frame_lens = Vec::new();
    let mut out = |frame: &[u8]| {
        frame_lens.push(frame.len());
        stream.extend_from_slice(frame);
    };
    let channels: Vec<&[f64]> = input.iter().map(Vec::as_slice).collect();
    encoder.encode_planar(&channels, &mut out);
    let finished = encoder.finish(&mut out);
    Encoded {
        stream,
        frame_lens,
        finished,
    }
}

/// Decodes `encoded` with ffmpeg, checking that it complains of nothing and
/// gives every frame's samples; gives them interleaved.
fn decode(encoded: &Encoded, channels: usize, context: &str) -> Vec<f32> {
    let decoded = ffmpeg::decode_aac(&encoded.stream);
    assert_eq!(decoded.diagnostics, "", "{context}");
    assert_eq!(
        decoded.samples.len(),
        encoded.finished.frames as usize * 1024 * channels,
        "{context}"
    );
    decoded.samples
}

/// The bitrates the encoder takes at `sample_rate` for `channels`.
fn bitrate_range(sample_rate: u32, channels: usize) -> (u32, u32) {
    match config(sample_rate, channels, 0).check() {
        Err(ConfigError::UnsupportedBitrate { min, max, .. }) => (min, max),
        other => panic!("{other:?}"),
    }
}

/// Six tones spread over the band below `top` Hz, from different phases
/// for each `channel`.
fn tones(sample_rate: u32, len: usize, top: f64, channel: usize) -> Vec<f64> {
    let fs = f64::from(sample_rate);
    let frequencies = [0.011, 0.07, 0.19, 0.33, 0.52, 0.71].map(|f| f * top);
    (0..len)
        .map(|t| {
            (frequencies.iter().enumerate())
                .map(|(i, f)| 0.1 * (TAU * f * t as f64 / fs + (channel * i) as f64).sin())
                .sum()
        })
        .collect()
}

/// White noise, uniform within a quarter of full scale.
fn noise(len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = (state.wrapping_mul(6_364_136_223_846_793_005))
                .wrapping_add(1_442_695_040_888_963_407);
            0.25 * ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
        })
        .collect()
}

/// A sine sweeping exponentially from `from` to `to` Hz, at half of full
/// scale.
fn sweep(sample_rate: u32, len: usize, from: f64, to: f64) -> Vec<f64> {
    let fs = f64::from(sample_rate);
    let duration = len as f64 / fs;
    let k = (to / from).ln();
    (0..len)
        .map(|t| {
            let t = t as f64 / fs;
            0.5 * (TAU * from * duration / k * ((k * t / duration).exp() - 1.0)).sin()
        })
        .collect()
}

/// The signal-to-noise ratio in dB of `output`, interleaved over
/// `channels`, against `input` of `channel` at `lag`, over `span` of the
/// input.
fn snr_over(
    input: &[f64],
    output: &[f32],
    (channels, channel): (usize, usize),
    lag: usize,
    span: std::ops::Range<usize>,
) -> f64 {
    let (signal, noise) = span.fold((0.0, 0.0), |(s, n), t| {
        let x = input[t];
        let y = output
            .get((t + lag) * channels + channel)
            .map_or(0.0, |&y| f64::from(y));
        (s + x * x, n + (x - y) * (x - y))
    });
    10.0 * (signal / noise).log10()
}

/// [`snr_over`] all of the input but its first and last frames.
fn snr(input: &[f64], output: &[f32], channels: usize, channel: usize, lag: usize) -> f64 {
    let span = 1024..input.len() - 1024;
    snr_over(input, output, (channels, channel), lag, span)
}

/// The lag of `output` behind `input`, up to three frames, at which they
/// line up best over four frames of the input, and the SNR there over all
/// of it.
fn align(input: &[f64], output: &[f32], channels: usize, channel: usize) -> (usize, f64) {
    let span = 2048..6 * 1024;
    let (lag, _) = (0..3 * 1024)
        .map(|lag| {
            let snr = snr_over(input, output, (channels, channel), lag, span.clone());
            (lag, snr)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    (lag, snr(input, output, channels, channel, lag))
}

/// ffmpeg gives the input back [`DELAY`] samples late, at a bit a sample
/// closely enough to line up: the first frame decodes to the silence
/// before the input, and each frame after it to the block before its own,
/// at the level the encoder takes its input at.
///
/// This is the evidence for the gap in the standard that [`DELAY`] marks
/// as settled by decoding with ffmpeg: where the input lands in the
/// decoded output.
#[test]
fn ffmpeg_decodes_the_input_a_frame_late_at_every_sample_rate() {
    for sample_rate in SAMPLE_RATES {
        for channels in [1, 2] {
            let context = format!("{sample_rate} Hz, {channels} channels");
            let len = sample_rate as usize;
            let top = (f64::from(sample_rate) / 2.0).min(16_000.0);
            // Sweeps, which line up with themselves at one lag only.
            let input: Vec<Vec<f64>> = (0..channels)
                .map(|c| sweep(sample_rate, len, 100.0 + 50.0 * c as f64, top))
                .collect();
            let (_, max) = bitrate_range(sample_rate, channels);
            let bits_per_second = (sample_rate * channels as u32).min(max);
            let encoded = encode(sample_rate, &input, bits_per_second);
            let output = decode(&encoded, channels, &context);
            for (c, input) in input.iter().enumerate() {
                let (lag, snr) = align(input, &output, channels, c);
                assert_eq!(lag, DELAY, "{context}, channel {c}: {snr:.1} dB");
                assert!(snr > 15.0, "{context}, channel {c}: {snr:.1} dB");
            }
        }
    }
}

/// The SNR rises with the bitrate, for signals whose spectra differ:
/// tones, a sweep over the whole band, and white noise, which no
/// quantizer codes well at these rates.
#[test]
fn snr_rises_with_the_bitrate() {
    let sample_rate = 44_100;
    let len = 3 * sample_rate as usize;
    let signals = [
        ("tones", [(64_000, 33.0), (128_000, 47.0), (256_000, 60.0)]),
        ("sweep", [(64_000, 40.0), (128_000, 57.0), (256_000, 70.0)]),
        ("noise", [(64_000, 0.5), (128_000, 3.5), (256_000, 11.0)]),
    ];
    for (name, thresholds) in signals {
        let input = match name {
            "tones" => vec![
                tones(sample_rate, len, 16_000.0, 0),
                tones(sample_rate, len, 16_000.0, 1),
            ],
            "sweep" => vec![
                sweep(sample_rate, len, 20.0, 20_000.0),
                sweep(sample_rate, len, 30.0, 18_000.0),
            ],
            _ => vec![noise(len, 1), noise(len, 2)],
        };
        for (bits_per_second, threshold) in thresholds {
            let context = format!("{name} at {bits_per_second} bit/s");
            let encoded = encode(sample_rate, &input, bits_per_second);
            let output = decode(&encoded, 2, &context);
            for (c, input) in input.iter().enumerate() {
                let snr = snr(input, &output, 2, c, DELAY);
                assert!(snr > threshold, "{context}, channel {c}: {snr:.1} dB");
            }
        }
    }
}

/// A tone at a low bitrate decodes at its frequency and level, with little
/// else.
///
/// This is the evidence for the reading of the standard that the
/// filterbank's `PCM_SCALE` marks as settled by decoding with ffmpeg: the
/// scale of the decoder's output. At twice or half the scale, the level
/// would be 6 dB off.
#[test]
fn a_tone_keeps_its_frequency_and_level_at_a_low_bitrate() {
    let sample_rate = 44_100;
    let fs = f64::from(sample_rate);
    let (frequency, amplitude) = (997.0, 0.5);
    let len = 2 * sample_rate as usize;
    let input: Vec<f64> = (0..len)
        .map(|t| amplitude * (TAU * frequency * t as f64 / fs).sin())
        .collect();
    let encoded = encode(sample_rate, &[input.clone()], 16_000);
    let output = decode(&encoded, 1, "a tone at 16 kbit/s");
    // The decoded tone's level and phase, by correlation over whole frames
    // past the first.
    let span = 2048..len - 1024;
    let (mut sin, mut cos, mut energy) = (0.0, 0.0, 0.0);
    for t in span.clone() {
        let y = f64::from(output[t + DELAY]);
        let phase = TAU * frequency * t as f64 / fs;
        sin += y * phase.sin();
        cos += y * phase.cos();
        energy += y * y;
    }
    let n = span.len() as f64;
    let level = 2.0 * (sin * sin + cos * cos).sqrt() / n;
    assert!(
        (20.0 * (level / amplitude).log10()).abs() < 0.1,
        "level {level}"
    );
    assert!(cos.abs() < 0.01 * sin.abs(), "phase: {sin}, {cos}");
    // What the tone does not account for.
    let rest = energy - level * level / 2.0 * n;
    assert!(rest < 1e-3 * energy, "{rest} of {energy}");
}

/// Every frame keeps within the bit reservoir (clause 8.2.3.2): from the
/// first, the frames take no more bits than the rate has brought, and no
/// fewer than it has brought less the reservoir's size, so that the stream
/// keeps the bitrate; and each is within the decoder's input buffer.
#[test]
fn frames_keep_the_bitrate() {
    for (sample_rate, channels, bits_per_second) in [
        (44_100, 2, 128_000),
        (48_000, 1, 32_000),
        (16_000, 2, 24_000),
    ] {
        let context = format!("{sample_rate} Hz, {channels} channels, {bits_per_second} bit/s");
        let len = 5 * sample_rate as usize;
        let input: Vec<Vec<f64>> = (0..channels)
            .map(|c| {
                let tones = tones(sample_rate, len, f64::from(sample_rate) / 2.0, c);
                let noise = noise(len, c as u64);
                tones.iter().zip(noise).map(|(t, n)| t + n).collect()
            })
            .collect();
        let encoded = encode(sample_rate, &input, bits_per_second);
        decode(&encoded, channels, &context);
        // In bits times the sampling frequency, as the mean length of a
        // frame is a fraction of a bit.
        let mean = u64::from(bits_per_second) * 1024;
        let buffer = 6144 * channels as u64;
        let max_bit_reservoir = (buffer * u64::from(sample_rate) - mean) / u64::from(sample_rate);
        let mut taken = 0;
        let mut frame_start = 0;
        for (i, &len) in encoded.frame_lens.iter().enumerate() {
            // The ADTS header gives the frame's length.
            let header = &encoded.stream[frame_start..frame_start + 7];
            let frame_length = (usize::from(header[3] & 0b11) << 11)
                | (usize::from(header[4]) << 3)
                | usize::from(header[5] >> 5);
            assert_eq!(frame_length, len, "{context}, frame {i}");
            frame_start += len;

            assert!(8 * len as u64 <= buffer, "{context}, frame {i}");
            taken += 8 * len as u64 * u64::from(sample_rate);
            let brought = (i as u64 + 1) * mean;
            assert!(taken <= brought, "{context}, frame {i}");
            assert!(
                taken + max_bit_reservoir * u64::from(sample_rate) >= brought,
                "{context}, frame {i}"
            );
        }
    }
}

/// Input at the edges of what the encoder takes decodes without
/// complaint, at the lowest and highest bitrates and one between: square
/// waves at full scale and at the clamp, 65536 times full scale, noise
/// beyond the clamp, the highest frequency at the clamp, full-scale DC and
/// silence.
#[test]
fn inputs_at_the_edges_decode_without_complaint() {
    for sample_rate in [8_000, 44_100, 96_000] {
        let len = sample_rate as usize / 2;
        let square: Vec<f64> = (0..len)
            .map(|t| if (t / 37) % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let signals = [
            ("square", square.clone()),
            (
                "clamped square",
                square.iter().map(|x| x * 65_536.0).collect(),
            ),
            (
                "noise beyond the clamp",
                noise(len, 3).iter().map(|x| x * 1e9).collect(),
            ),
            (
                "clamped Nyquist",
                (0..len)
                    .map(|t| if t % 2 == 0 { 1e6 } else { -1e6 })
                    .collect(),
            ),
            ("DC", vec![1.0; len]),
            ("silence", vec![0.0; len]),
        ];
        for channels in [1, 2] {
            let (min, max) = bitrate_range(sample_rate, channels);
            for (name, signal) in &signals {
                for bits_per_second in [min, (min + max) / 8, max] {
                    let context = format!(
                        "{name}, {sample_rate} Hz, {channels} channels, {bits_per_second} bit/s"
                    );
                    let input = vec![signal.clone(); channels];
                    let encoded = encode(sample_rate, &input, bits_per_second);
                    decode(&encoded, channels, &context);
                }
            }
        }
    }
}
