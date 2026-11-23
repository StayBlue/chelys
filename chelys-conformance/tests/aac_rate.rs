// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! AAC streams at variable and mean rates, in ADTS, as ffmpeg and FAAD2
//! decode them.

use std::f64::consts::TAU;

use chelys::aac::{Config, DELAY, Encoder, Finished, Rate};
use chelys_conformance::{faad, ffmpeg};

/// `seconds` of a signal at `sample_rate` in `channels`, planar: tones,
/// noise that swells and fades, and a click a second, the right channel
/// unlike the left.
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
                    let tones: f64 = [196.0, 523.0, 1_760.0, 4_400.0]
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

/// Encodes planar `input` as `config` asks, giving the stream and each
/// frame's length.
fn encode(config: Config, input: &[Vec<f64>]) -> (Vec<u8>, Vec<usize>, Finished) {
    let mut encoder = Encoder::new(config).unwrap();
    let mut stream = Vec::new();
    let mut lens = Vec::new();
    let mut out = |frame: &[u8]| {
        lens.push(frame.len());
        stream.extend_from_slice(frame);
    };
    let channels: Vec<&[f64]> = input.iter().map(Vec::as_slice).collect();
    encoder.encode_planar(&channels, &mut out);
    let finished = encoder.finish(&mut out);
    (stream, lens, finished)
}

/// The SNR in dB of channel `c` of `output`, interleaved in `channels`,
/// against `input`, the output [`DELAY`] samples late.
fn snr(input: &[f64], output: &[f32], channels: usize, c: usize) -> f64 {
    let (signal, noise) = (input.iter().enumerate()).fold((0.0, 0.0), |(s, n), (t, &x)| {
        let y = f64::from(output[(t + DELAY) * channels + c]);
        (s + x * x, n + (x - y) * (x - y))
    });
    10.0 * (signal / noise).log10()
}

/// ffmpeg and FAAD2 decode streams at variable and mean rates without
/// complaint, every frame's samples, to the input within an SNR that falls
/// as the noise offset rises, in mono and stereo, at sampling frequencies
/// high and low, with the threshold in quiet read either way, in intensity
/// stereo, and in frames as long as the decoder's input buffer; and FAAD2
/// to what ffmpeg decodes. Every header's `adts_buffer_fullness` is 0x7FF,
/// which says that the stream is of variable rate (clause 8.1.3.2.2).
#[test]
fn ffmpeg_and_faad_decode_variable_and_mean_rates() {
    let variable = |noise_offset_db| Rate::Variable { noise_offset_db };
    let mean = |bits_per_second| Rate::Mean { bits_per_second };
    for (sample_rate, channels, rate, least_snr) in [
        (44_100, 2, variable(0.0), 9.0),
        (44_100, 2, variable(10.0), 5.0),
        (48_000, 1, variable(-6.0), 11.0),
        (32_000, 2, mean(96_000), 12.0),
        (44_100, 2, mean(48_000), 5.0),
        (8_000, 1, variable(-40.0), 30.0),
    ] {
        let context = format!("{sample_rate} Hz, {channels} channels, {rate:?}");
        let input = music(sample_rate, channels, 3.0);
        let (stream, lens, finished) = encode(Config::new(sample_rate, channels, rate), &input);
        let mut at = 0;
        for len in &lens {
            let frame = &stream[at..at + len];
            let fullness = (u32::from(frame[5] & 0b1_1111) << 6) | u32::from(frame[6] >> 2);
            assert_eq!(fullness, 0x7FF, "{context}");
            at += len;
        }
        if sample_rate == 8_000 {
            // As long as the decoder's input buffer, 6144 bits.
            assert!(lens.iter().all(|&len| len <= 768), "{context}");
            assert!(lens.iter().any(|&len| len > 750), "{context}");
        }
        let decoded = ffmpeg::decode_aac(&stream);
        assert_eq!(decoded.diagnostics, "", "{context}");
        let samples = finished.frames as usize * 1024 * channels;
        assert_eq!(decoded.samples.len(), samples, "{context}");
        for (c, input) in input.iter().enumerate() {
            let snr = snr(input, &decoded.samples, channels, c);
            assert!(snr > least_snr, "{context}, channel {c}: {snr:.1} dB");
        }
        let faad = faad::decode_aac(&stream);
        assert_eq!(faad.diagnostics, "", "{context}");
        if sample_rate > 24_000 {
            assert_eq!(faad.samples.len(), samples, "{context}");
            let (signal, difference) =
                (decoded.samples.iter().zip(&faad.samples)).fold((0.0, 0.0), |(s, d), (&x, &y)| {
                    let (x, y) = (f64::from(x), f64::from(y));
                    (s + x * x, d + (x - y) * (x - y))
                });
            let db = 10.0 * (signal / difference).log10();
            assert!(db > 100.0, "{context}: FAAD2 against ffmpeg, {db:.1} dB");
        }
    }
}

/// Pink noise panned to the left, 8 dB louder there than in the right
/// channel, fading out over 4 s by 80 dB, from -20 dB below full scale in
/// the left channel, is coded to -100 dB at variable and mean rates, as
/// ffmpeg decodes it: in each window of 0.1 s, at 0 dB, each channel
/// within 1.5 dB of its level and the left channel louder than the right
/// by 8 dB within 2 dB; at rates of about 0.45 bit a sample, a noise
/// offset of 12 dB and a mean of 40 kbit/s, each channel within 4 dB of
/// its level and the left channel louder by 8 dB within 5.5 dB, the image
/// narrowed where the side channel of the pair is allowed most of its
/// energy. Coded within their allowance, without a least length, the
/// frames coded the fade without lines from about -70 dB at 0 dB, and the
/// side channel without lines from about -50 dB, so that the two channels
/// fell to the same level (see `chelys_aac`'s rate module,
/// `LEAST_BITS_OF_THE_MEAN`).
#[test]
fn a_panned_fade_keeps_its_image() {
    let sample_rate = 44_100;
    let len = 4 * sample_rate as usize;
    let mut state = 5u64;
    // White noise through first-order filters whose poles are spread
    // over the band, which sum to noise falling by 3 dB an octave.
    let poles = [0.99886, 0.99332, 0.969, 0.8665, 0.55, -0.7616];
    let gains = [
        0.0555179, 0.0750759, 0.153852, 0.3104856, -0.5329522, -0.016898,
    ];
    let mut filters = [0.0; 6];
    let fade: Vec<f64> = (0..len)
        .map(|t| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let x = 2.0 * ((state >> 11) as f64 / (1u64 << 53) as f64) - 1.0;
            for ((y, pole), gain) in filters.iter_mut().zip(poles).zip(gains) {
                *y = pole * *y + gain * x;
            }
            let pink = 0.15 * (filters.iter().sum::<f64>() + 0.5362 * x);
            let db = -10.0 - 80.0 * t as f64 / len as f64;
            pink * 10f64.powf(db / 20.0)
        })
        .collect();
    let input = [
        fade.iter().map(|x| 0.93 * x).collect::<Vec<f64>>(),
        fade.iter().map(|x| 0.37 * x).collect(),
    ];
    let window = sample_rate as usize / 10;
    let level = |samples: &dyn Fn(usize) -> f64, w: usize| {
        let power = (w * window..(w + 1) * window)
            .map(|t| samples(t).powi(2))
            .sum::<f64>()
            / window as f64;
        10.0 * (power + 1e-30).log10()
    };
    for (rate, level_db, image_db) in [
        (
            Rate::Variable {
                noise_offset_db: 0.0,
            },
            1.5,
            2.0,
        ),
        (
            Rate::Variable {
                noise_offset_db: 12.0,
            },
            4.0,
            5.5,
        ),
        (
            Rate::Mean {
                bits_per_second: 40_000,
            },
            4.0,
            5.5,
        ),
    ] {
        let (stream, _, _) = encode(Config::new(sample_rate, 2, rate), &input);
        let decoded = ffmpeg::decode_aac(&stream);
        assert_eq!(decoded.diagnostics, "", "{rate:?}");
        for w in 0..len / window {
            let [left, right] =
                [0, 1].map(|c| level(&|t| f64::from(decoded.samples[(t + DELAY) * 2 + c]), w));
            let [left_in, right_in] = [0, 1].map(|c| level(&|t| input[c][t], w));
            let context = format!(
                "{rate:?}, window {w}: left {left:.1} dB ({left_in:.1}), \
                 right {right:.1} dB ({right_in:.1})"
            );
            if left_in < -100.0 {
                continue;
            }
            assert!((left - left_in).abs() < level_db, "{context}");
            assert!((right - right_in).abs() < level_db, "{context}");
            assert!((left - right - 8.0).abs() < image_db, "{context}");
        }
    }
}

/// ffmpeg decodes MP4 files of streams at variable and mean rates to the
/// input's length, trimmed by the edit list, as it decodes their frames.
#[test]
fn ffmpeg_decodes_mp4_files_at_variable_and_mean_rates() {
    for (sample_rate, channels, rate) in [
        (
            44_100,
            2,
            Rate::Variable {
                noise_offset_db: 3.0,
            },
        ),
        (
            48_000,
            1,
            Rate::Mean {
                bits_per_second: 64_000,
            },
        ),
    ] {
        let context = format!("{sample_rate} Hz, {channels} channels, {rate:?}");
        let input = music(sample_rate, channels, 2.0);
        let len = input[0].len();
        let pcm: Vec<f32> = (0..len)
            .flat_map(|t| input.iter().map(move |c| c[t] as f32))
            .collect();
        let mut encoder =
            chelys::mp4::aac::Encoder::new(Config::new(sample_rate, channels, rate)).unwrap();
        encoder.encode_interleaved(&pcm);
        let mut file = Vec::new();
        let finished = encoder.finish(&mut file).unwrap();
        let all = ffmpeg::decode_mp4_ignoring_edits(&file);
        assert_eq!(all.diagnostics, "", "{context}");
        assert_eq!(
            all.samples.len(),
            finished.frames as usize * 1024 * channels,
            "{context}"
        );
        let edited = ffmpeg::decode_mp4(&file);
        let from = DELAY * channels;
        assert!(
            edited.samples == all.samples[from..from + pcm.len()],
            "{context}"
        );
    }
}
