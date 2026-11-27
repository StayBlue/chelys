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

/// What the buffer of a constant rate of `bits_per_second` at
/// `sample_rate` holds after each of `packets`, in bits times the sample
/// rate, as [`Rate::Constant`] defines it: half of its size, a second of
/// the rate, before the first packet, then filled at the rate by the
/// samples each packet's granule position passes, and emptied by its bits.
/// Gives it with its size.
fn held(packets: &[AudioPacket], bits_per_second: u32, sample_rate: u32) -> (Vec<i128>, i128) {
    let (rate, fs) = (i128::from(bits_per_second), i128::from(sample_rate));
    let size = rate * fs;
    let (mut held, mut position) = (size / 2, 0);
    let held = (packets.iter())
        .map(|packet| {
            let granule_position = i128::from(packet.granule_position);
            assert!(granule_position >= position);
            held += rate * (granule_position - position) - 8 * packet.data.len() as i128 * fs;
            position = granule_position;
            held
        })
        .collect();
    (held, size)
}

/// The ratio of the power of `a - b` to that of `b`, in dB.
fn error_db(a: &[f32], b: &[f32]) -> f64 {
    let error: f64 = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| f64::from(a - b).powi(2))
        .sum();
    let power: f64 = b.iter().map(|&b| f64::from(b).powi(2)).sum();
    10.0 * (error / power).log10()
}

/// ffmpeg decodes a stream at a constant rate to the input's length,
/// without a diagnostic, in stereo and mono, of passages at full scale
/// and 40 dB below it, whose quietest are padded; its audio packets keep
/// the buffer from none to its size after every packet, and their bits are
/// the rate's over the stream, give or take half of it. The decoded
/// samples are as near the input as those of a mean rate, within 1 dB.
#[test]
fn ffmpeg_decodes_a_constant_rate_within_its_buffer() {
    for (sample_rate, channels, bits_per_second) in [(44_100, 2, 128_000), (48_000, 1, 64_000)] {
        for level in [1.0, 0.01] {
            let context =
                format!("{sample_rate} Hz, {channels} channels, {bits_per_second} bit/s, {level}");
            let pcm = passages(sample_rate, channels, 30, level);
            let rate = Rate::Constant { bits_per_second };
            let (headers, packets) = encode(&pcm, sample_rate, channels, rate);
            let decoded = ffmpeg::decode_ogg(&ogg::vorbis(&headers, &packets, 4));
            assert_eq!(decoded.diagnostics, "", "{context}");
            assert_eq!(decoded.samples.len(), pcm.len(), "{context}");
            let (held, size) = held(&packets, bits_per_second, sample_rate);
            for (k, held) in held.iter().enumerate() {
                assert!((0..=size).contains(held), "{context}: packet {k}");
            }
            let lengths: Vec<usize> = packets.iter().map(|p| p.data.len()).collect();
            assert!(lengths.iter().min() < lengths.iter().max(), "{context}");
            let mean = Rate::Mean { bits_per_second };
            let (headers, packets) = encode(&pcm, sample_rate, channels, mean);
            let mean = ffmpeg::decode_ogg(&ogg::vorbis(&headers, &packets, 4));
            let (constant, mean) = (
                error_db(&decoded.samples, &pcm),
                error_db(&mean.samples, &pcm),
            );
            assert!(
                constant < mean + 1.0,
                "{context}: {constant:.2} {mean:.2} dB"
            );
        }
    }
}

/// A packet padded with zero bytes decodes as it does unpadded: ffmpeg
/// decodes a stream whose packets are padded, each by up to some 2000
/// bytes, to the same samples, without a diagnostic.
#[test]
fn ffmpeg_decodes_padded_packets_as_unpadded() {
    let pcm = passages(44_100, 2, 10, 1.0);
    let (headers, packets) = encode(&pcm, 44_100, 2, Rate::Quality { level: 50 });
    let unpadded = ffmpeg::decode_ogg(&ogg::vorbis(&headers, &packets, 4));
    assert_eq!(unpadded.diagnostics, "");
    let padded: Vec<AudioPacket> = (packets.iter().enumerate())
        .map(|(k, packet)| {
            let mut data = packet.data.clone();
            data.resize(data.len() + (k * 7919) % 2000, 0);
            AudioPacket {
                data,
                granule_position: packet.granule_position,
            }
        })
        .collect();
    let decoded = ffmpeg::decode_ogg(&ogg::vorbis(&headers, &padded, 4));
    assert_eq!(decoded.diagnostics, "");
    assert!(decoded.samples == unpadded.samples);
}

/// A packet that ends early decodes, in its floors, as silence (clause
/// 4.3.2), and in its residue as the vectors decoded up to the end (clause
/// 8.6.2): those of the codewords read whole, which leave out what the
/// rest would add. ffmpeg decodes the first so, without a diagnostic, but
/// reads a packet cut in its residue past its end, reporting each such
/// read, and decodes what it reads there as if the packet went on: cut at
/// 24 bytes, the stream decodes to samples up to 9 times full scale, where
/// the whole decodes to 0.38 at most.
#[test]
fn ffmpeg_decodes_truncated_packets_unlike_the_specification() {
    let pcm = passages(44_100, 2, 10, 0.5);
    let (headers, packets) = encode(&pcm, 44_100, 2, Rate::Quality { level: 50 });
    let peak = |samples: &[f32]| samples.iter().fold(0.0f32, |peak, s| peak.max(s.abs()));
    let whole = ffmpeg::decode_ogg(&ogg::vorbis(&headers, &packets, 4));
    assert!(peak(&whole.samples) < 0.5);
    let cut = |len: usize| {
        let cut: Vec<AudioPacket> = (packets.iter())
            .map(|packet| AudioPacket {
                data: packet.data[..len.min(packet.data.len())].to_vec(),
                granule_position: packet.granule_position,
            })
            .collect();
        ffmpeg::decode_ogg(&ogg::vorbis(&headers, &cut, 4))
    };
    // Two bytes hold no more than the start of the first floor.
    let floors = cut(2);
    assert_eq!(floors.diagnostics, "");
    assert!(floors.samples.iter().all(|&s| s == 0.0));
    let peaks: Vec<f32> = (24..=48)
        .step_by(4)
        .map(|len| {
            let decoded = cut(len);
            assert!(decoded.diagnostics.contains("Overread"), "{len}");
            peak(&decoded.samples)
        })
        .collect();
    assert!(peaks.iter().any(|&peak| peak > 1.0), "{peaks:?}");
}

/// ffprobe reads the bitrate of the identification header of a constant
/// rate, all three of its bitrates the rate, as the stream's bitrate.
#[test]
fn ffprobe_reads_the_bitrate_of_a_constant_rate() {
    let pcm = passages(44_100, 2, 1, 1.0);
    let rate = Rate::Constant {
        bits_per_second: 96_000,
    };
    let (headers, packets) = encode(&pcm, 44_100, 2, rate);
    let stream = ogg::vorbis(&headers, &packets, 4);
    assert_eq!(ffmpeg::bit_rate_ogg(&stream), Some(96_000));
}
