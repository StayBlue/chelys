// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Streams whose bitrate changes from frame to frame (clause 2.4.2.3),
//! decoded by ffmpeg, mpg123 and LAME.

use std::f64::consts::TAU;
use std::ops::Range;

use chelys::mp3::header::{Bitrate, Mode, SamplingFrequency};
use chelys::mp3::rate::{MeanRate, Rate, VariableRate};
use chelys::mp3::{StreamConfig, encode_pcm};
use chelys_conformance::{DECODERS, Decoder};

/// Loud noise and a quiet tone in turn, each for a fifth of `seconds`,
/// starting and ending with the noise: at a variable bitrate the tone's
/// frames are small and the noise's large.
fn input(fs: SamplingFrequency, seconds: f64) -> Vec<f64> {
    let hz = f64::from(fs.hz());
    let mut state = 0x9E37_79B9u32;
    let mut noise = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        f64::from(state) / f64::from(u32::MAX) * 2.0 - 1.0
    };
    (0..(seconds * hz) as usize)
        .map(|t| {
            let s = t as f64 / hz;
            if ((5.0 * s / seconds) as usize).is_multiple_of(2) {
                0.5 * noise()
            } else {
                0.02 * (TAU * 440.0 * s).sin() + 0.005 * (TAU * 1_000.0 * s).sin()
            }
        })
        .collect()
}

/// The tone's passages, away from the noise around them, whose coding
/// noise spreads over a granule and the filterbank's span.
fn tone_passages(fs: SamplingFrequency, seconds: f64) -> [Range<usize>; 2] {
    let fifth = (seconds * f64::from(fs.hz()) / 5.0) as usize;
    [
        fifth + 2304..2 * fifth - 2304,
        3 * fifth + 2304..4 * fifth - 2304,
    ]
}

fn snr(input: &[f64], output: &[f32], range: &Range<usize>, delay: usize) -> f64 {
    let (signal, noise) = range.clone().fold((0.0, 0.0), |(s, n), t| {
        let x = input[t];
        let y = f64::from(output[t + delay]);
        (s + x * x, n + (x - y) * (x - y))
    });
    10.0 * (signal / noise).log10()
}

/// Every decoder finds each frame and its main data however the bitrate
/// changes, at a variable bitrate and at a mean between two of the table:
/// it reports nothing, and the quiet tone decodes cleanly, which it would
/// not from misplaced main data. The output is 1057 samples behind the input, less what the
/// decoder trims from its start; the delay is the one at which the first
/// passage of the tone decodes best.
#[test]
fn every_decoder_follows_the_bitrate_from_frame_to_frame() {
    use SamplingFrequency::*;
    let at = |fs: SamplingFrequency, kbps| Bitrate::from_kbps(fs.version(), kbps).unwrap();
    let variable = |fs, lowest, highest| {
        Rate::Variable(VariableRate {
            lowest: at(fs, lowest),
            highest: at(fs, highest),
            noise_offset_db: 0.0,
        })
    };
    let mean = |fs: SamplingFrequency, kbps: u32| {
        Rate::Mean(MeanRate::new(fs.version(), kbps * 1000).unwrap())
    };
    let ms = Mode::JointStereo {
        intensity_stereo: false,
        ms_stereo: true,
    };
    let cases = [
        (Hz44100, variable(Hz44100, 32, 320), ms, false),
        (Hz32000, variable(Hz32000, 32, 320), Mode::Stereo, true),
        (Hz44100, mean(Hz44100, 150), ms, false),
        (Hz22050, variable(Hz22050, 8, 160), ms, false),
        (Hz24000, mean(Hz24000, 50), Mode::SingleChannel, true),
        (
            Hz11025,
            variable(Hz11025, 8, 64),
            Mode::SingleChannel,
            false,
        ),
    ];
    let seconds = 3.0;
    std::thread::scope(|scope| {
        for (fs, rate, mode, crc_protected) in cases {
            scope.spawn(move || {
                let config = StreamConfig {
                    rate,
                    sampling_frequency: fs,
                    mode,
                    crc_protected,
                };
                let input = input(fs, seconds);
                let channels = mode.channels();
                let stream = match channels {
                    1 => encode_pcm(&config, &[&input]),
                    _ => encode_pcm(&config, &[&input, &input]),
                };
                let passages = tone_passages(fs, seconds);
                for Decoder { name, decode, .. } in DECODERS {
                    let case = format!("{name}, {fs:?}, {rate:?}");
                    let decoded = decode(&stream);
                    assert_eq!(decoded.diagnostics, "", "{case}");
                    for channel in 0..channels {
                        let output: Vec<f32> = decoded.samples[channel..]
                            .iter()
                            .step_by(channels)
                            .copied()
                            .collect();
                        let delay = (0..=1057)
                            .max_by(|&a, &b| {
                                let snr = |delay| snr(&input, &output, &passages[0], delay);
                                snr(a).total_cmp(&snr(b))
                            })
                            .unwrap();
                        for passage in &passages {
                            let snr = snr(&input, &output, passage, delay);
                            assert!(snr > 30.0, "{case}, channel {channel}: {snr:.1} dB");
                        }
                    }
                }
            });
        }
    });
}
