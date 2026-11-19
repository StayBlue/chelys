// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! AAC streams in ADTS with the options of their headers, the CRC of
//! `adts_error_check()` and the header's flags, and with data stream
//! elements, as ffmpeg and FAAD2 decode them.

use std::f64::consts::TAU;

use chelys::aac::{Config, Encoder, Finished, Rate};
use chelys_conformance::{Decoded, faad, ffmpeg};

/// Tones at `sample_rate`, `seconds` long, interleaved in `channels`.
fn tones(sample_rate: u32, channels: usize, seconds: f64) -> Vec<f32> {
    let fs = f64::from(sample_rate);
    (0..(seconds * fs) as usize)
        .flat_map(|t| {
            (0..channels).map(move |c| {
                [220.0, 997.0, 3_100.0]
                    .iter()
                    .map(|f| 0.15 * (TAU * f * (1.0 + 0.1 * c as f64) * t as f64 / fs).sin())
                    .sum::<f64>() as f32
            })
        })
        .collect()
}

/// Encodes `pcm` as `config` asks, with `data` queued for its data streams.
fn encode(config: Config, channels: usize, pcm: &[f32], data: &[u8]) -> (Vec<u8>, Finished) {
    let mut encoder = Encoder::new(config).unwrap();
    assert_eq!(encoder.channels(), channels);
    encoder.push_data_stream(data);
    let mut stream = Vec::new();
    encoder.encode_interleaved(pcm, &mut stream);
    let finished = encoder.finish(&mut stream);
    (stream, finished)
}

/// Each `adts_frame()` of `stream`, by its `aac_frame_length`.
fn frames(stream: &[u8]) -> Vec<&[u8]> {
    let mut frames = Vec::new();
    let mut rest = stream;
    while !rest.is_empty() {
        let len = (usize::from(rest[3] & 0b11) << 11)
            | (usize::from(rest[4]) << 3)
            | usize::from(rest[5] >> 5);
        frames.push(&rest[..len]);
        rest = &rest[len..];
    }
    frames
}

/// `stream` with each frame's `crc_check` taken out, `protection_absent`
/// set to 1 and `aac_frame_length` two bytes shorter.
fn without_crc(stream: &[u8]) -> Vec<u8> {
    frames(stream)
        .into_iter()
        .flat_map(|frame| {
            assert_eq!(frame[1] & 1, 0, "protection_absent 0");
            let mut header = frame[..7].to_vec();
            header[1] |= 1;
            let len = frame.len() - 2;
            header[3] = (header[3] & !0b11) | (len >> 11) as u8;
            header[4] = (len >> 3) as u8;
            header[5] = (header[5] & 0b1_1111) | ((len & 0b111) << 5) as u8;
            [header, frame[9..].to_vec()].concat()
        })
        .collect()
}

/// ffmpeg decodes streams with the CRC and every header flag set, at a
/// rate low and high, in mono and stereo, without complaint and as it
/// decodes them with the CRC taken out: it finds the `crc_check` where
/// Table 5 puts it, between the header and the `raw_data_block()`.
#[test]
fn ffmpeg_decodes_streams_with_the_crc_and_flags() {
    for (sample_rate, channels, bits_per_second) in [
        (44_100, 2, 128_000),
        (48_000, 1, 24_000),
        (8_000, 2, 16_000),
    ] {
        let context = format!("{sample_rate} Hz, {channels} channels, {bits_per_second} bit/s");
        let config = Config::new(sample_rate, channels, Rate::Constant { bits_per_second })
            .crc(true)
            .private_bit(true)
            .original_copy(true)
            .home(true);
        let (stream, finished) = encode(config, channels, &tones(sample_rate, channels, 1.0), &[]);
        let decoded = ffmpeg::decode_aac(&stream);
        assert_eq!(decoded.diagnostics, "", "{context}");
        assert_eq!(
            decoded.samples.len(),
            finished.frames as usize * 1024 * channels,
            "{context}"
        );
        assert!(decoded.samples.iter().any(|&s| s.abs() > 0.1), "{context}");
        let plain = ffmpeg::decode_aac(&without_crc(&stream));
        assert_eq!(plain.diagnostics, "", "{context}");
        assert_eq!(plain.samples, decoded.samples, "{context}");
    }
}

/// ffmpeg does not check the `crc_check` of ADTS, even when asked to check
/// CRCs: a stream whose every `crc_check` is wrong decodes without
/// complaint and as the stream does. So no decoder here settles which bits
/// the CRC protects; that rests on the reading of clause 8.1.3.2.1 and the
/// unit tests of `chelys-aac`. Should a later ffmpeg check it, this test
/// fails, and it becomes the evidence.
#[test]
fn ffmpeg_skips_the_crc_unchecked() {
    let config = Config::new(
        44_100,
        2,
        Rate::Constant {
            bits_per_second: 96_000,
        },
    )
    .crc(true);
    let (stream, _) = encode(config, 2, &tones(44_100, 2, 1.0), &[]);
    let wrong = with_wrong_crc(&stream);
    let (right, wrong) = (ffmpeg::decode_aac(&stream), ffmpeg::decode_aac(&wrong));
    assert_eq!(wrong.diagnostics, "");
    assert_eq!(wrong.samples, right.samples);
}

/// `stream` with a bit of each frame's `crc_check` flipped at either end.
fn with_wrong_crc(stream: &[u8]) -> Vec<u8> {
    frames(stream)
        .into_iter()
        .flat_map(|frame| {
            let mut frame = frame.to_vec();
            frame[7] ^= 0x80;
            frame[8] ^= 0x01;
            frame
        })
        .collect()
}

/// FAAD2 decodes the streams that ffmpeg does above, with the CRC and
/// every header flag set, without complaint and as it decodes them with
/// the CRC taken out; at 8 kHz, at twice the sampling frequency.
#[test]
fn faad_decodes_streams_with_the_crc_and_flags() {
    for (sample_rate, channels, bits_per_second) in [
        (44_100, 2, 128_000),
        (48_000, 1, 24_000),
        (8_000, 2, 16_000),
    ] {
        let context = format!("{sample_rate} Hz, {channels} channels, {bits_per_second} bit/s");
        let config = Config::new(sample_rate, channels, Rate::Constant { bits_per_second })
            .crc(true)
            .private_bit(true)
            .original_copy(true)
            .home(true);
        let (stream, finished) = encode(config, channels, &tones(sample_rate, channels, 1.0), &[]);
        let factor = if sample_rate <= 24_000 { 2 } else { 1 };
        let decoded = faad::decode_aac(&stream);
        assert_eq!(decoded.diagnostics, "", "{context}");
        assert_eq!(
            decoded.samples.len(),
            finished.frames as usize * 1024 * channels * factor,
            "{context}"
        );
        assert!(decoded.samples.iter().any(|&s| s.abs() > 0.1), "{context}");
        let plain = faad::decode_aac(&without_crc(&stream));
        assert_eq!(plain.diagnostics, "", "{context}");
        assert_eq!(plain.samples, decoded.samples, "{context}");
    }
}

/// FAAD2 does not check the `crc_check` of ADTS either, and `faad` has no
/// option to: a stream whose every `crc_check` is wrong decodes without
/// complaint and as the stream does. So neither
/// decoder settles which bits the CRC protects. Should a later FAAD2 check
/// it, this test fails, and it becomes the evidence.
#[test]
fn faad_skips_the_crc_unchecked() {
    let config = Config::new(
        44_100,
        2,
        Rate::Constant {
            bits_per_second: 96_000,
        },
    )
    .crc(true);
    let (stream, _) = encode(config, 2, &tones(44_100, 2, 1.0), &[]);
    let wrong = with_wrong_crc(&stream);
    let (right, wrong) = (faad::decode_aac(&stream), faad::decode_aac(&wrong));
    assert_eq!(wrong.diagnostics, "");
    assert_eq!(wrong.samples, right.samples);
}

/// ffmpeg decodes streams whose frames carry data stream elements, two or
/// one a frame, with the CRC and without, without complaint, and the data
/// they carry leaves the audio alone: streams of the same input whose data
/// differ but not in length, zeros in one and copies of ADTS headers and
/// syncwords in the other, decode to the same samples, as many as the
/// stream without data gives, and to the input, coded in the bits the data
/// leaves it.
#[test]
fn ffmpeg_ignores_data_stream_elements() {
    ignores_data_stream_elements(ffmpeg::decode_aac);
}

/// FAAD2 decodes the streams with data stream elements as ffmpeg does
/// above.
#[test]
fn faad_ignores_data_stream_elements() {
    ignores_data_stream_elements(faad::decode_aac);
}

fn ignores_data_stream_elements(decode: fn(&[u8]) -> Decoded) {
    for (sample_rate, channels, bits_per_second, per_frame, crc) in [
        (44_100, 2, 320_000, 600, true),
        (48_000, 1, 64_000, 40, false),
    ] {
        let context = format!("{sample_rate} Hz, {channels} channels, {per_frame} bytes a frame");
        let config =
            Config::new(sample_rate, channels, Rate::Constant { bits_per_second }).crc(crc);
        let pcm = tones(sample_rate, channels, 2.0);
        let (plain, _) = encode(config, channels, &pcm, &[]);
        let config = config.data_stream_bytes_per_frame(per_frame);
        let len = 200 * per_frame;
        let (zeros, finished) = encode(config, channels, &pcm, &vec![0; len]);
        let header = &plain[..9];
        let imitations: Vec<u8> = (0..)
            .flat_map(|i| match i % 3 {
                0 => header.to_vec(),
                1 => vec![0xFF, 0xF1, 0x50, 0x80],
                _ => vec![0xFF, 0xF9, 0x4C, 0x40, 0x12, 0x3F, 0xFC],
            })
            .take(len)
            .collect();
        let (imitating, _) = encode(config, channels, &pcm, &imitations);
        assert_eq!(
            finished.data_stream_dropped,
            len - (finished.frames as usize * per_frame).min(len)
        );
        assert_ne!(zeros, imitating, "{context}");
        assert_eq!(zeros.len(), imitating.len(), "{context}");
        let [plain, zeros, imitating] = [&plain, &zeros, &imitating].map(|stream| decode(stream));
        for decoded in [&zeros, &imitating] {
            assert_eq!(decoded.diagnostics, "", "{context}");
            assert_eq!(decoded.samples.len(), plain.samples.len(), "{context}");
        }
        assert_eq!(zeros.samples, imitating.samples, "{context}");
        // The SNR of the input in the decoded output, a frame late.
        let snr = |decoded: &[f32]| -> f64 {
            let delay = 1024 * channels;
            let (signal, noise) =
                (pcm.iter().zip(&decoded[delay..])).fold((0.0, 0.0), |(s, n), (&x, &y)| {
                    let (x, y) = (f64::from(x), f64::from(y));
                    (s + x * x, n + (x - y) * (x - y))
                });
            10.0 * (signal / noise).log10()
        };
        let (with, without) = (snr(&zeros.samples), snr(&plain.samples));
        assert!(
            with > 20.0 && with < without,
            "{context}: {with:.1} dB with the data, {without:.1} dB without"
        );
    }
}
