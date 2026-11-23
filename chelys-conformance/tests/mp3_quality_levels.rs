// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! MP3 at levels of the quality scale, at the sampling frequencies of
//! MPEG-2.5 and, to compare, those of ISO/IEC 13818-3, as ffmpeg and mpg123
//! decode them.

use std::f64::consts::TAU;

use chelys::mp3::{ChannelMode, Config, DELAY, Encoder, Finished, Rate};
use chelys_conformance::{ffmpeg, mpg123};

const MPEG25: [u32; 3] = [8_000, 11_025, 12_000];
const LOWER: [u32; 3] = [16_000, 22_050, 24_000];

const JOINT: ChannelMode = ChannelMode::JointStereo {
    ms_stereo: true,
    intensity_stereo: false,
};

fn quality(level: u8) -> Rate {
    Rate::Quality {
        level,
        min_kbps: None,
        max_kbps: None,
    }
}

/// Encodes planar `input` as `config` asks, the tag written over its
/// placeholder if there is one.
fn encode(config: Config, input: &[Vec<f64>]) -> (Vec<u8>, Finished) {
    let mut encoder = Encoder::new(config).unwrap();
    let mut stream = Vec::new();
    let channels: Vec<&[f64]> = input.iter().map(Vec::as_slice).collect();
    encoder.encode_planar(&channels, &mut stream);
    let finished = encoder.finish(&mut stream);
    if let Some(tag) = &finished.info_tag {
        stream[..tag.len()].copy_from_slice(tag);
    }
    (stream, finished)
}

/// The bitrate in kbit/s of each frame of `stream`, from `bitrate_index`
/// (clause 2.4.2.3, and Table 2.4.2.3 of ISO/IEC 13818-3, which MPEG-2.5
/// shares); the frame of the tag, if any, among them.
fn frame_kbps(stream: &[u8]) -> Vec<u32> {
    const KBPS: [u32; 15] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160];
    let mut kbps = Vec::new();
    let mut at = 0;
    while at + 4 <= stream.len() {
        let header = &stream[at..at + 4];
        assert_eq!(
            (header[0], header[1] & 0xE0),
            (0xFF, 0xE0),
            "a frame at {at}"
        );
        let bitrate = KBPS[usize::from(header[2] >> 4)];
        let hz = match ((header[1] >> 3) & 0b11, (header[2] >> 2) & 0b11) {
            (0b00, 0b00) => 11_025,
            (0b00, 0b01) => 12_000,
            (0b00, 0b10) => 8_000,
            (0b10, 0b00) => 22_050,
            (0b10, 0b01) => 24_000,
            (0b10, 0b10) => 16_000,
            other => panic!("not a lower sampling frequency: {other:?}"),
        };
        let padding = usize::from((header[2] >> 1) & 1);
        kbps.push(bitrate);
        at += 72_000 * bitrate as usize / hz + padding;
    }
    assert_eq!(at, stream.len());
    kbps
}

/// `seconds` of a signal at `sample_rate` in `channels`, planar: tones,
/// noise that swells and fades, and a click a second, the right channel
/// unlike the left, all below 4 kHz.
fn music(sample_rate: u32, channels: usize, seconds: f64) -> Vec<Vec<f64>> {
    let fs = f64::from(sample_rate);
    let len = (seconds * fs) as usize;
    let mut state = 11u64;
    let mut noise = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let noise: Vec<f64> = (0..len).map(|_| noise()).collect();
    (0..channels)
        .map(|c| {
            (0..len)
                .map(|t| {
                    let time = t as f64 / fs;
                    let tones: f64 = [196.0, 523.0, 1_760.0, 2_900.0]
                        .iter()
                        .map(|f| 0.08 * (TAU * f * (1.0 + 0.01 * c as f64) * time).sin())
                        .sum();
                    let swell = 0.3 * (0.5 - 0.5 * (TAU * time / 2.0).cos());
                    let click = if t % sample_rate as usize == 300 + 50 * c {
                        0.8
                    } else {
                        0.0
                    };
                    tones + swell * noise[(t + 997 * c) % len] + click
                })
                .collect()
        })
        .collect()
}

/// The SNR in dB of channel `c` of `output`, interleaved in `channels`,
/// against `input`, the output `lag` samples late, away from the ends.
fn snr(input: &[f64], output: &[f32], channels: usize, c: usize, lag: usize) -> f64 {
    let (signal, noise) = (2_000..input.len() - 2_000).fold((0.0, 0.0), |(s, n), t| {
        let x = input[t];
        let y = f64::from(output[(t + lag) * channels + c]);
        (s + x * x, n + (x - y) * (x - y))
    });
    10.0 * (signal / noise).log10()
}

/// Whether channel `c` of `output` lines up with `input` at `lag` and not
/// at the lags beside it: at low levels the signal decodes noisily, but a
/// lag one sample off is worse still.
fn lines_up(input: &[f64], output: &[f32], channels: usize, c: usize, lag: usize) -> bool {
    let exact = snr(input, output, channels, c, lag);
    let beside = [lag.checked_sub(1), Some(lag + 1)];
    (beside.into_iter().flatten()).all(|lag| exact > snr(input, output, channels, c, lag) + 3.0)
}

/// At every level from 0 to 100 in steps of 10, at each sampling frequency
/// of MPEG-2.5, in one channel and in two, with a Xing/Info tag:
///
/// - ffmpeg and mpg123 decode the stream without complaint.
/// - mpg123 gives the input exactly, as many samples lined up to the
///   sample, which it could not were the tag's delays or frame count
///   wrong; ffmpeg every frame's samples, the input [`DELAY`] late (see
///   `decoders_read_the_tag_at_every_sampling_frequency`).
/// - The stream is shorter at each level than at the level above.
/// - At 8 kHz no frame is above 64 kbit/s in one channel, nor 128 in two,
///   the lowest bitrates whose slots hold every granule's 4095 bits.
#[test]
fn every_level_decodes_at_the_mpeg25_sampling_frequencies() {
    std::thread::scope(|scope| {
        for sample_rate in MPEG25 {
            for (mode, channels) in [(ChannelMode::Mono, 1), (JOINT, 2)] {
                scope.spawn(move || {
                    let input = music(sample_rate, channels, 2.0);
                    let mut lens = Vec::new();
                    for level in (0..=100).step_by(10) {
                        let case = format!("{sample_rate} Hz, {channels} channels, level {level}");
                        let config = Config::new(sample_rate, mode, quality(level)).info_tag(true);
                        let (stream, finished) = encode(config, &input);
                        let highest = frame_kbps(&stream).into_iter().max().unwrap();
                        if sample_rate == 8_000 {
                            assert!(highest <= 64 * channels as u32, "{case}: {highest} kbit/s");
                        }
                        let decoded = mpg123::decode_mp3(&stream);
                        assert_eq!(decoded.diagnostics, "", "mpg123, {case}");
                        assert_eq!(
                            decoded.samples.len(),
                            input[0].len() * channels,
                            "mpg123, {case}"
                        );
                        let decoded_ffmpeg = ffmpeg::decode_mp3(&stream);
                        assert_eq!(decoded_ffmpeg.diagnostics, "", "ffmpeg, {case}");
                        let samples = finished.frames as usize * 576 * channels;
                        assert_eq!(decoded_ffmpeg.samples.len(), samples, "ffmpeg, {case}");
                        for (c, input) in input.iter().enumerate() {
                            assert!(
                                lines_up(input, &decoded.samples, channels, c, 0),
                                "mpg123, {case}"
                            );
                            let lag = DELAY;
                            assert!(
                                lines_up(input, &decoded_ffmpeg.samples, channels, c, lag),
                                "ffmpeg, {case}"
                            );
                        }
                        lens.push(stream.len());
                    }
                    let context = format!("{sample_rate} Hz, {channels} channels: {lens:?}");
                    assert!(lens.windows(2).all(|pair| pair[0] < pair[1]), "{context}");
                });
            }
        }
    });
}

/// At the sampling frequencies of MPEG-2.5, every frame keeps to the lowest
/// and highest bitrates asked for, as a level is coded within them, though
/// the lowest be above those whose slots hold every granule's 4095 bits:
/// in one channel at 11.025 kHz, from 112 kbit/s up, every frame is at 112.
#[test]
fn frames_keep_to_the_bitrates_asked_for_at_the_mpeg25_sampling_frequencies() {
    let cases = [
        (8_000, JOINT, 32, 64, 32..=64),
        (8_000, ChannelMode::Mono, 144, 160, 144..=144),
        (11_025, ChannelMode::Mono, 112, 160, 112..=112),
        (11_025, ChannelMode::Stereo, 40, 40, 40..=40),
        (12_000, JOINT, 8, 24, 8..=24),
        (12_000, ChannelMode::Mono, 24, 96, 24..=96),
    ];
    std::thread::scope(|scope| {
        for (sample_rate, mode, min_kbps, max_kbps, expected) in cases {
            scope.spawn(move || {
                let rate = Rate::Quality {
                    level: 80,
                    min_kbps: Some(min_kbps),
                    max_kbps: Some(max_kbps),
                };
                let input = music(sample_rate, mode.channels(), 2.0);
                let (stream, _) = encode(Config::new(sample_rate, mode, rate), &input);
                let kbps = frame_kbps(&stream);
                let case = format!("{sample_rate} Hz, {mode:?}, {min_kbps} to {max_kbps} kbit/s");
                assert!(
                    kbps.iter().all(|kbps| expected.contains(kbps)),
                    "{case}: {kbps:?}"
                );
                let decoded = ffmpeg::decode_mp3(&stream);
                assert_eq!(decoded.diagnostics, "", "{case}");
            });
        }
    });
}

/// Noise falling by 3 dB an octave from 60 Hz to 3.6 kHz, the same at every
/// sampling frequency, `seconds` of it at `sample_rate`: sinusoids at random
/// frequencies and phases, each of an amplitude of `0.42 / sqrt(f)`, at
/// about -10 dB below full scale.
fn pink_below_3_6_khz(sample_rate: u32, seconds: f64) -> Vec<f64> {
    let mut seed = 7u64;
    let mut next = || {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let components: Vec<(f64, f64, f64)> = (0..300)
        .map(|_| {
            let f = 60.0 * 60f64.powf(next());
            (f, 0.42 / f.sqrt(), TAU * next())
        })
        .collect();
    let rate = f64::from(sample_rate);
    (0..(seconds * rate) as usize)
        .map(|t| {
            let time = t as f64 / rate;
            (components.iter())
                .map(|(f, a, phase)| a * (TAU * f * time + phase).sin())
                .sum()
        })
        .collect()
}

/// The level in dB of `samples(t)` over the `len` samples from `start`.
fn level(samples: impl Fn(usize) -> f64, start: usize, len: usize) -> f64 {
    let power = (start..start + len)
        .map(|t| samples(t).powi(2))
        .sum::<f64>()
        / len as f64;
    10.0 * (power + 1e-30).log10()
}

/// Noise below 3.6 kHz, the same at every sampling frequency, panned to
/// the left, 8 dB louder there than in the right channel, fading out over
/// 4 s by 80 dB, from -22 dB below full scale in the left channel, is
/// coded at the sampling frequencies of MPEG-2.5 as at those of ISO/IEC
/// 13818-3, as ffmpeg decodes it. In each window of 0.1 s whose left
/// channel is above -70 dB, each channel keeps its level within 2.5 dB and
/// the left channel is louder by 8 dB within 3.5 dB at level 0, and within
/// 1.5 and 2.5 dB at levels 30, 50, 80 and 100; at 80 and 100 within 1.5
/// and 2 dB down to -75 dB; and no window above -80 dB decodes to silence.
/// The worst windows above -70 dB were 1.6 and 2.0 dB off at MPEG-2.5's
/// rates, at level 0, and 2.2 and 3.0 dB at 24 kHz; above level 0, 1.2 and
/// 1.9 dB, at 12 kHz. The first window decoded to silence was below -85 dB
/// at MPEG-2.5's rates, and at -81 dB at 16 kHz at level 0. Below the
/// fade's last 10 dB, where the threshold in quiet lies, the image narrows,
/// and the right channel, then the left, are lost at every sampling
/// frequency.
#[test]
fn a_panned_fade_keeps_its_image_as_at_the_lower_sampling_frequencies() {
    std::thread::scope(|scope| {
        for sample_rate in MPEG25.into_iter().chain(LOWER) {
            for q in [0, 30, 50, 80, 100] {
                scope.spawn(move || {
                    let noise = pink_below_3_6_khz(sample_rate, 4.0);
                    let len = noise.len();
                    let fade: Vec<f64> = (noise.iter().enumerate())
                        .map(|(t, x)| {
                            let db = -10.0 - 80.0 * t as f64 / len as f64;
                            x * 10f64.powf(db / 20.0)
                        })
                        .collect();
                    let input = [
                        fade.iter().map(|x| 0.93 * x).collect::<Vec<f64>>(),
                        fade.iter().map(|x| 0.37 * x).collect(),
                    ];
                    let (stream, _) = encode(Config::new(sample_rate, JOINT, quality(q)), &input);
                    let decoded = ffmpeg::decode_mp3(&stream);
                    let case = format!("{sample_rate} Hz, level {q}");
                    assert_eq!(decoded.diagnostics, "", "{case}");
                    let window = sample_rate as usize / 10;
                    let samples = &decoded.samples;
                    for start in (0..len - window).step_by(window) {
                        let output =
                            |c: usize| move |t: usize| f64::from(samples[(t + DELAY) * 2 + c]);
                        let [left, right] = [0, 1].map(|c| level(output(c), start, window));
                        let [left_in, right_in] =
                            [0, 1].map(|c| level(|t| input[c][t], start, window));
                        let context = format!(
                            "{case}, from {start}: left {left:.1} dB ({left_in:.1}), \
                             right {right:.1} dB ({right_in:.1})"
                        );
                        let (level_db, image_db) = match (q, left_in) {
                            (_, -70.0..) if q == 0 => (2.5, 3.5),
                            (_, -70.0..) => (1.5, 2.5),
                            (80.., -75.0..) => (1.5, 2.0),
                            (_, -80.0..) => {
                                assert!(left > -200.0, "{context}");
                                continue;
                            }
                            _ => continue,
                        };
                        assert!((left - left_in).abs() < level_db, "{context}");
                        assert!((right - right_in).abs() < level_db, "{context}");
                        assert!((left - right - 8.0).abs() < image_db, "{context}");
                    }
                });
            }
        }
    });
}

/// A quiet tone, and sparse bursts of a tone in digital silence, keep their
/// level at the sampling frequencies of MPEG-2.5 as at those of ISO/IEC
/// 13818-3, at levels 0, 50, 80 and 100, as ffmpeg decodes them. A tone of
/// 700 Hz at -60 and at -80 dB below full scale, in one channel, keeps its
/// level within 1 dB in each window of 0.1 s away from its ends. Bursts of
/// 50 ms of a tone of 440 Hz every half second, at -40 dB in the left
/// channel and 6 dB less in the right, keep each channel's level within
/// 0.3 dB, and the silence between them, away from the filterbank's span,
/// decodes below -120 dB. The worst tone was 0.63 dB off, at 11.025 kHz
/// at level 0, the worst burst 0.12 dB, at 16 kHz, and the loudest silence
/// -145 dB.
#[test]
fn quiet_and_sparse_signals_keep_their_level_as_at_the_lower_sampling_frequencies() {
    std::thread::scope(|scope| {
        for sample_rate in MPEG25.into_iter().chain(LOWER) {
            for q in [0, 50, 80, 100] {
                scope.spawn(move || {
                    let fs = f64::from(sample_rate);
                    let len = 3 * sample_rate as usize;
                    let window = sample_rate as usize / 10;
                    for db in [-60.0, -80.0] {
                        let amplitude = 10f64.powf(db / 20.0) * 2f64.sqrt();
                        let tone: Vec<f64> = (0..len)
                            .map(|t| amplitude * (TAU * 700.0 * t as f64 / fs).sin())
                            .collect();
                        let config = Config::new(sample_rate, ChannelMode::Mono, quality(q));
                        let (stream, _) = encode(config, std::slice::from_ref(&tone));
                        let decoded = ffmpeg::decode_mp3(&stream);
                        let case = format!("{sample_rate} Hz, level {q}, tone at {db} dB");
                        assert_eq!(decoded.diagnostics, "", "{case}");
                        for start in (2 * window..len - 2 * window).step_by(window) {
                            let output =
                                level(|t| f64::from(decoded.samples[t + DELAY]), start, window);
                            let input = level(|t| tone[t], start, window);
                            assert!(
                                (output - input).abs() < 1.0,
                                "{case}, from {start}: {output:.2} dB"
                            );
                        }
                    }
                    let (burst, period) = ((0.05 * fs) as usize, sample_rate as usize / 2);
                    let amplitude = 10f64.powf(-40.0 / 20.0) * 2f64.sqrt();
                    let left: Vec<f64> = (0..len)
                        .map(|t| match t % period < burst {
                            true => amplitude * (TAU * 440.0 * t as f64 / fs).sin(),
                            false => 0.0,
                        })
                        .collect();
                    let input = [left.clone(), left.iter().map(|x| 0.5 * x).collect()];
                    let (stream, _) = encode(Config::new(sample_rate, JOINT, quality(q)), &input);
                    let decoded = ffmpeg::decode_mp3(&stream);
                    let case = format!("{sample_rate} Hz, level {q}, bursts");
                    assert_eq!(decoded.diagnostics, "", "{case}");
                    let samples = &decoded.samples;
                    let output = |c: usize| move |t: usize| f64::from(samples[(t + DELAY) * 2 + c]);
                    for start in (0..=len - period).step_by(period) {
                        for (c, input) in input.iter().enumerate() {
                            let decoded = level(output(c), start, burst);
                            let expected = level(|t| input[t], start, burst);
                            let context =
                                format!("{case}, channel {c}, from {start}: {decoded:.2} dB");
                            assert!((decoded - expected).abs() < 0.3, "{context}");
                        }
                        let gap = start + burst + 1152;
                        let between = level(output(0), gap, period - burst - 2 * 1152);
                        assert!(between < -120.0, "{case}, from {gap}: {between:.1} dB");
                    }
                });
            }
        }
    });
}
