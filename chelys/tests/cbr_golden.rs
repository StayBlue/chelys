// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The streams of constant-bitrate encoding, pinned by their length and an
//! FNV-1a hash, so that changes meant to leave them alone are seen to.
//!
//! A deliberate change to the coding updates the table; the input mixes
//! tones, silence, attacks and channels that differ, to reach block
//! switching, the bit reservoir, MS and intensity stereo, scfsi and the
//! error check. The hashes depend on the platform's floating point, and
//! were recorded on x86-64 Linux.

use std::f64::consts::TAU;

use chelys::mp3::{ChannelMode, Config, Encoder, Rate};

/// One and a half seconds: a tone in each channel, apart by a fifth, a
/// quiet passage, then bursts of noise with sharp onsets.
fn input(sample_rate: u32) -> [Vec<f64>; 2] {
    let hz = f64::from(sample_rate);
    let len = 3 * sample_rate as usize / 2;
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut noise = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    };
    let mut left = Vec::with_capacity(len);
    let mut right = Vec::with_capacity(len);
    for t in 0..len {
        let s = t as f64 / hz;
        let (l, r) = if s < 0.5 {
            (
                0.3 * (TAU * 440.0 * s).sin() + 0.05 * (TAU * 2_500.0 * s).sin(),
                0.25 * (TAU * 660.0 * s).sin() + 0.05 * (TAU * 2_500.0 * s).sin(),
            )
        } else if s < 0.8 {
            let tone = 0.002 * (TAU * 330.0 * s).sin();
            (tone, tone)
        } else {
            let since = (s - 0.8) % 0.2;
            let burst = 0.6 * (-since / 0.03).exp();
            let common = noise();
            (
                burst * common + 0.1 * (TAU * 1_000.0 * s).sin(),
                burst * (0.7 * common + 0.3 * noise()),
            )
        };
        left.push(l);
        right.push(r);
    }
    [left, right]
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xCBF2_9CE4_8422_2325, |hash, &b| {
        (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3)
    })
}

const MS: ChannelMode = ChannelMode::JointStereo {
    ms_stereo: true,
    intensity_stereo: false,
};
const MS_IS: ChannelMode = ChannelMode::JointStereo {
    ms_stereo: true,
    intensity_stereo: true,
};

/// Sample rate, kbit/s, mode, error check, and the stream's length and
/// hash.
const STREAMS: [(u32, u32, ChannelMode, bool, usize, u64); 8] = [
    (44_100, 128, MS, false, 24659, 0xA57F_DF71_1E69_7196),
    (44_100, 64, MS_IS, false, 12329, 0x5699_6067_A462_D2D9),
    (
        32_000,
        320,
        ChannelMode::Stereo,
        true,
        61920,
        0x84DF_5785_62E0_68F2,
    ),
    (
        48_000,
        64,
        ChannelMode::Mono,
        false,
        12288,
        0xD709_F742_FFB8_AAD3,
    ),
    (22_050, 32, MS_IS, false, 6269, 0x64B3_3378_78DE_AC83),
    (
        16_000,
        8,
        ChannelMode::Mono,
        true,
        1584,
        0xD8FA_70F9_662C_4B46,
    ),
    (8_000, 16, MS, false, 3312, 0x33A2_5E5A_B2C0_DB6B),
    (
        11_025,
        24,
        ChannelMode::Stereo,
        false,
        4858,
        0x9B27_2AC3_C953_7FE9,
    ),
];

#[test]
fn constant_bitrate_streams_are_unchanged() {
    let results = std::thread::scope(|scope| {
        STREAMS
            .map(|(sample_rate, kbps, mode, crc, ..)| {
                scope.spawn(move || {
                    let config = Config::new(sample_rate, mode, Rate::Constant { kbps }).crc(crc);
                    let mut encoder = Encoder::new(config).unwrap();
                    let [left, right] = input(sample_rate);
                    let mut stream = Vec::new();
                    match mode {
                        ChannelMode::Mono => encoder.encode_planar(&[&left], &mut stream),
                        _ => encoder.encode_planar(&[&left, &right], &mut stream),
                    }
                    encoder.finish(&mut stream);
                    (stream.len(), fnv1a(&stream))
                })
            })
            .map(|handle| handle.join().expect("the encoding passes"))
    });
    let mismatches: Vec<String> = (STREAMS.iter().zip(results))
        .filter(|((.., len, hash), got)| (*len, *hash) != *got)
        .map(|((sample_rate, kbps, mode, crc, ..), (len, hash))| {
            format!("({sample_rate}, {kbps}, {mode:?}, {crc}): {len}, {hash:#018X}")
        })
        .collect();
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}
