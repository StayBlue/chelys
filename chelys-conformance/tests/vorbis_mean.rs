// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

use std::f64::consts::TAU;

use chelys::vorbis::{Config, Encoder, Headers, Packet, Rate};
use chelys_conformance::ffmpeg;
use chelys_conformance::ogg::{self, AudioPacket};

/// The headers and audio packets of `pcm`, interleaved, coded at `rate`.
fn encode(
    pcm: &[f32],
    sample_rate: u32,
    channels: usize,
    rate: Rate,
) -> (Headers, Vec<AudioPacket>) {
    let mut encoder = Encoder::new(Config::new(sample_rate, channels, rate)).unwrap();
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
    (headers, packets)
}

/// The bits a second of the audio packets of a stream of `samples`
/// samples a channel at `sample_rate`.
fn bits_per_second(packets: &[AudioPacket], samples: usize, sample_rate: u32) -> f64 {
    let bytes: usize = packets.iter().map(|p| p.data.len()).sum();
    8.0 * bytes as f64 * f64::from(sample_rate) / samples as f64
}

/// `seconds` of passages of 5 s, in each channel a chord of three tones and
/// noise, of a level and a balance of their own, all scaled by `level`.
fn passages(sample_rate: u32, channels: usize, seconds: usize, level: f64) -> Vec<f32> {
    let mut seed = 0x2545_f491u32;
    let samples = seconds * sample_rate as usize;
    (0..samples * channels)
        .map(|i| {
            let (n, c) = (i / channels, i % channels);
            let passage = n / (5 * sample_rate as usize);
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = f64::from(seed >> 8) / f64::from(1u32 << 24) - 0.5;
            let noise_level = [0.02, 0.4, 0.005, 0.15][passage % 4];
            let t = n as f64 / f64::from(sample_rate);
            let root = 110.0 * (1.0 + (passage % 3) as f64 / 4.0) * (1.0 + c as f64 / 50.0);
            let chord: f64 = [1.0, 1.25, 1.5]
                .iter()
                .map(|ratio| (TAU * root * ratio * t).sin())
                .sum();
            (level * (0.15 * chord + noise_level * noise)) as f32
        })
        .collect()
}

/// ffmpeg decodes a stream at a mean rate to the input's length, and over
/// 30 s its audio packets keep the mean within 4%, in stereo and mono, of
/// passages at full scale and 40 dB below it, whose quietest the model
/// allows more noise as the threshold in quiet nears: at the offset the
/// steering starts from, at which music takes the mean on average, a
/// variable rate takes those 40 dB below full scale 15% to 30% short.
#[test]
fn ffmpeg_decodes_a_mean_rate_near_its_bitrate() {
    for (sample_rate, channels, mean, start_db) in
        [(44_100, 2, 128_000, -7.0), (48_000, 1, 64_000, -4.0)]
    {
        for level in [1.0, 0.01] {
            let context = format!("{sample_rate} Hz, {channels} channels, {mean} bit/s, {level}");
            let pcm = passages(sample_rate, channels, 30, level);
            let samples = pcm.len() / channels;
            let rate = Rate::Mean {
                bits_per_second: mean,
            };
            let (headers, packets) = encode(&pcm, sample_rate, channels, rate);
            let decoded = ffmpeg::decode_ogg(&ogg::vorbis(&headers, &packets, 4));
            assert_eq!(decoded.diagnostics, "", "{context}");
            assert_eq!(decoded.samples.len(), pcm.len(), "{context}");
            let achieved = bits_per_second(&packets, samples, sample_rate);
            let error = achieved / f64::from(mean) - 1.0;
            assert!(error.abs() < 0.04, "{context}: {achieved:.0} bit/s");
            if level < 1.0 {
                let variable = Rate::Variable {
                    noise_offset_db: start_db,
                };
                let (_, variable) = encode(&pcm, sample_rate, channels, variable);
                let variable = bits_per_second(&variable, samples, sample_rate);
                let miss = variable / f64::from(mean) - 1.0;
                assert!(
                    miss.abs() > 0.1,
                    "{context}: {variable:.0} bit/s at {start_db} dB"
                );
            }
        }
    }
}

/// A steady sine, which takes few bits at any offset, falls far short of
/// any mean, the steering's offset held at its lowest; the header still
/// gives the mean as nominal.
#[test]
fn a_sine_falls_short_of_any_mean() {
    let pcm: Vec<f32> = (0..2 * 30 * 44_100)
        .map(|i| (0.25 * (TAU * 440.0 * (i / 2) as f64 / 44_100.0).sin()) as f32)
        .collect();
    let mut rates = Vec::new();
    for mean in [128_000, 529_000] {
        let rate = Rate::Mean {
            bits_per_second: mean,
        };
        let (headers, packets) = encode(&pcm, 44_100, 2, rate);
        let achieved = bits_per_second(&packets, pcm.len() / 2, 44_100);
        assert!(achieved < 0.5 * 128_000.0, "{mean} bit/s: {achieved:.0}");
        let stream = ogg::vorbis(&headers, &packets, 4);
        assert_eq!(ffmpeg::bit_rate_ogg(&stream), Some(mean));
        rates.push(achieved);
    }
    // The two alike, within 2%.
    assert!((rates[1] / rates[0] - 1.0).abs() < 0.02, "{rates:?}");
}

/// ffprobe reads the nominal bitrate of the identification header as the
/// stream's bitrate: the mean of a mean rate, and none at a variable rate.
#[test]
fn ffprobe_reads_the_nominal_bitrate_of_a_mean_rate() {
    let pcm = passages(44_100, 2, 1, 1.0);
    for (rate, expected) in [
        (
            Rate::Mean {
                bits_per_second: 96_000,
            },
            Some(96_000),
        ),
        (Rate::Quality { level: 50 }, None),
        (
            Rate::Variable {
                noise_offset_db: 0.0,
            },
            None,
        ),
    ] {
        let (headers, packets) = encode(&pcm, 44_100, 2, rate);
        let stream = ogg::vorbis(&headers, &packets, 4);
        assert_eq!(ffmpeg::bit_rate_ogg(&stream), expected, "{rate:?}");
    }
}
