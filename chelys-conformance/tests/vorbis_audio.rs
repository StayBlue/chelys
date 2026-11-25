// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

use std::f64::consts::TAU;

use chelys::vorbis::{Config, Encoder, MAX_CHANNELS, Packet};
use chelys_conformance::ffmpeg;
use chelys_conformance::ogg::{self, AudioPacket};

/// Sample rates of the MPEG standards and beyond, as Vorbis has no table
/// of them.
const SAMPLE_RATES: [u32; 9] = [
    8_000, 11_025, 16_000, 22_050, 32_000, 44_100, 48_000, 96_000, 192_000,
];

/// The Ogg stream of `pcm`, interleaved.
fn encode(pcm: &[f32], sample_rate: u32, channels: usize) -> Vec<u8> {
    let mut encoder = Encoder::new(Config::new(sample_rate, channels)).unwrap();
    let headers = encoder.headers();
    let mut packets = Vec::new();
    let mut out = |packet: Packet<'_>| {
        packets.push(AudioPacket {
            data: packet.data.to_vec(),
            granule_position: packet.granule_position,
        })
    };
    encoder.encode_interleaved(pcm, &mut out).unwrap();
    encoder.finish(&mut out);
    ogg::vorbis(&headers, &packets, 4)
}

/// `pcm` encoded and decoded by ffmpeg, which is to report nothing and to
/// give back as many samples.
fn round_trip(pcm: &[f32], sample_rate: u32, channels: usize) -> Vec<f32> {
    let decoded = ffmpeg::decode_ogg(&encode(pcm, sample_rate, channels));
    assert_eq!(decoded.diagnostics, "");
    assert_eq!(decoded.samples.len(), pcm.len());
    decoded.samples
}

/// The ratio in dB of the power of `reference` to that of what `decoded`
/// differs from it by, over every `step`th sample from `start`.
fn snr(reference: &[f32], decoded: &[f32], start: usize, step: usize) -> f64 {
    let (mut signal, mut noise) = (0.0, 0.0);
    for (&r, &d) in reference.iter().zip(decoded).skip(start).step_by(step) {
        signal += f64::from(r).powi(2);
        noise += (f64::from(r) - f64::from(d)).powi(2);
    }
    10.0 * (signal / noise).log10()
}

/// Uniform noise from -0.5 to 0.5, the same each run.
fn noise(len: usize) -> Vec<f64> {
    let mut state = 1u32;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f64::from(state >> 8) / f64::from(1 << 24) - 0.5
        })
        .collect()
}

/// ffmpeg decodes a tone at its level and in phase: at the encoder's input
/// scale, and neither delayed, advanced nor inverted. So it takes the
/// spectrum at the scale, phase and sign that the encoder gives it.
#[test]
fn ffmpeg_decodes_a_tone_at_its_level_and_phase() {
    let sample_rate = 48_000;
    let pcm: Vec<f32> = (0..sample_rate as usize)
        .map(|t| (0.5 * (TAU * 1000.0 * t as f64 / f64::from(sample_rate)).sin()) as f32)
        .collect();
    let decoded = round_trip(&pcm, sample_rate, 1);
    let middle = 4096..pcm.len() - 4096;
    let correlation = |lag: isize| -> f64 {
        middle
            .clone()
            .map(|i| f64::from(pcm[i]) * f64::from(decoded[i.wrapping_add_signed(lag)]))
            .sum()
    };
    let power: f64 = middle.clone().map(|i| f64::from(pcm[i]).powi(2)).sum();
    let gain_db = 20.0 * (correlation(0) / power).log10();
    assert!(gain_db.abs() < 0.5, "a gain of {gain_db:.2} dB");
    for lag in [-2, -1, 1, 2] {
        assert!(correlation(lag) < correlation(0), "lag {lag}");
    }
    assert!(snr(&pcm, &decoded, middle.start, 1) > 15.0);
}

/// ffmpeg decodes each channel, at any sample rate, close to its own
/// input: a tone of its own in each. ffmpeg gives the channels in its own
/// order of speakers, not that of clause 4.3.9 above two channels, so each
/// channel it gives is matched to the input it is nearest.
#[test]
fn ffmpeg_decodes_each_channel_at_any_sample_rate() {
    let samples = 5 * 1024 + 300;
    for sample_rate in SAMPLE_RATES {
        for channels in 1..=MAX_CHANNELS {
            let context = format!("{sample_rate} Hz, {channels} channels");
            let pcm: Vec<f32> = (0..samples * channels)
                .map(|i| {
                    let (t, c) = (i / channels, i % channels);
                    let cycles = (c + 1) as f64 * 0.017;
                    (0.4 * (TAU * cycles * t as f64).sin()) as f32
                })
                .collect();
            let decoded = round_trip(&pcm, sample_rate, channels);
            let channel = |pcm: &[f32], c: usize| -> Vec<f32> {
                pcm.iter().skip(c).step_by(channels).copied().collect()
            };
            let mut matched = Vec::new();
            for out in 0..channels {
                let decoded = channel(&decoded, out);
                let (input, snr) = (0..channels)
                    .map(|c| (c, snr(&channel(&pcm, c), &decoded, 0, 1)))
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .unwrap();
                assert!(snr > 15.0, "{context}: channel {out}, {snr:.1} dB");
                matched.push(input);
            }
            matched.sort_unstable();
            matched.dedup();
            assert_eq!(matched.len(), channels, "{context}");
        }
    }
}

/// ffmpeg decodes noise, quiet or loud, and a mix of tones and noise, with
/// no more noise than the floor allows.
#[test]
fn ffmpeg_decodes_noise_and_tones_within_the_floor() {
    let sample_rate = 44_100;
    let noise = noise(2 * sample_rate as usize);
    for level in [0.01, 0.3, 1.5] {
        let pcm: Vec<f32> = noise.iter().map(|&x| (level * x) as f32).collect();
        let snr = snr(&pcm, &round_trip(&pcm, sample_rate, 1), 0, 1);
        assert!(snr > 15.0, "noise at {level}: {snr:.1} dB");
    }
    let mut low = 0.0;
    let pcm: Vec<f32> = (noise.iter().enumerate())
        .map(|(t, &x)| {
            low = 0.95 * low + 0.05 * x;
            let t = t as f64 / f64::from(sample_rate);
            (0.3 * (TAU * 220.0 * t).sin() + 0.1 * (TAU * 3520.0 * t).sin() + 0.5 * low) as f32
        })
        .collect();
    let snr = snr(&pcm, &round_trip(&pcm, sample_rate, 1), 0, 1);
    assert!(snr > 15.0, "the mix: {snr:.1} dB");
}

/// ffmpeg decodes a square wave at full scale, whose lines are the largest
/// the encoder meets in music, without clipping their residue.
#[test]
fn ffmpeg_decodes_a_square_wave_at_full_scale() {
    let pcm: Vec<f32> = (0..88_200)
        .map(|t| if t / 50 % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let snr = snr(&pcm, &round_trip(&pcm, 44_100, 1), 0, 1);
    assert!(snr > 15.0, "{snr:.1} dB");
}
