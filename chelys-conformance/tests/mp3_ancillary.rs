// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Streams that carry ancillary data (clause 2.4.2.8), as ffmpeg, mpg123
//! and LAME decode them.

use std::f64::consts::TAU;

use chelys::mp3::internals::header::{Bitrate, Flags, Mode, SamplingFrequency};
use chelys::mp3::internals::rate::StreamRate;
use chelys::mp3::internals::{StreamConfig, StreamEncoder};
use chelys_conformance::{DECODERS, Decoder};

fn tones(fs: f64, seconds: f64) -> Vec<f64> {
    let frequencies = [220.0, 997.0, 3_100.0, 7_000.0];
    (0..(seconds * fs) as usize)
        .map(|t| {
            frequencies
                .iter()
                .map(|f| 0.15 * (TAU * f * t as f64 / fs).sin())
                .sum()
        })
        .collect()
}

/// The SNR in dB of the first channel of `output` against `input`, at the
/// lag that lines them up best.
fn snr(input: &[f64], output: &[f32], channels: usize) -> f64 {
    let at = |lag: usize| {
        let (signal, noise) = (4000..input.len() - 4000).fold((0.0, 0.0), |(s, n), t| {
            let x = input[t];
            let y = f64::from(output[(t + lag) * channels]);
            (s + x * x, n + (x - y) * (x - y))
        });
        10.0 * (signal / noise).log10()
    };
    (0..2000).map(at).fold(f64::MIN, f64::max)
}

/// The stream of `config` for `input` in every channel, with `per_frame`
/// bits of each frame given to `data`.
fn encode(config: StreamConfig, input: &[f64], per_frame: usize, data: &[u8]) -> Vec<u8> {
    let mut encoder = StreamEncoder::new(config).with_ancillary(per_frame);
    encoder.push_ancillary(data);
    let mut stream = Vec::new();
    encoder.encode(&vec![input; config.mode.channels()], &mut stream);
    encoder.finish(&mut stream);
    stream
}

/// Each decoder decodes a stream whose frames carry ancillary data, made of
/// copies of the stream's own headers and of others, as many samples as the
/// same stream without it, and as well but for the bits the data takes from
/// the audio, within 10 dB of its SNR (about 6 dB less here, as all three
/// decode it): the data imitating headers does not lead them astray, as a
/// decoder that reads a whole stream finds each frame from the one before.
/// In MPEG-1 at 44.1 kHz and 128 kbit/s, and at 22.05 kHz and 64 kbit/s,
/// with a quarter of each frame's main data slot given to the data.
#[test]
fn decoders_decode_streams_with_ancillary_data() {
    // 417-byte frames, 381 of main data; and 208, of which 195.
    for (fs, kbps, mode, per_frame) in [
        (SamplingFrequency::Hz44100, 128, Mode::Stereo, 8 * 95),
        (SamplingFrequency::Hz22050, 64, Mode::SingleChannel, 8 * 48),
    ] {
        let config = StreamConfig {
            rate: StreamRate::Constant(Bitrate::from_kbps(fs.version(), kbps).unwrap()),
            sampling_frequency: fs,
            mode,
            crc_protected: false,
            flags: Flags::default(),
        };
        let hz = f64::from(fs.hz());
        let input = tones(hz, 2.0);
        let plain = encode(config, &input, 0, &[]);
        let header = &plain[..4];
        let data: Vec<u8> = (0..2000)
            .flat_map(|i| match i % 3 {
                0 => header.to_vec(),
                1 => vec![0xFF, 0xFB, 0x90, 0x00],
                _ => vec![0xFF, 0xF3, 0x48, 0xC4, 0x12],
            })
            .collect();
        let carried = encode(config, &input, per_frame, &data);
        assert_ne!(carried, plain);
        let channels = mode.channels();
        for Decoder { name, decode, .. } in DECODERS {
            let case = format!("{name}, {fs:?}");
            let (plain, carried) = (decode(&plain), decode(&carried));
            assert_eq!(carried.diagnostics, "", "{case}");
            assert_eq!(carried.samples.len(), plain.samples.len(), "{case}");
            let (plain, carried) = (
                snr(&input, &plain.samples, channels),
                snr(&input, &carried.samples, channels),
            );
            assert!(
                carried > plain - 10.0,
                "{case}: {carried:.1} dB with the data, {plain:.1} dB without"
            );
        }
    }
}
