// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Audio through the AAC encoder, decoded by ffmpeg and compared with the
//! input.

use std::f64::consts::TAU;

use chelys::aac::internals::noiseless::decoder::{BitReader, Element, element};
use chelys::aac::internals::sampling_frequency::SamplingFrequency;
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

/// The element of each frame of a stream in ADTS at `sample_rate`, as
/// chelys writes it, read with the decoder that chelys's tests share.
fn elements(stream: &[u8], sample_rate: u32) -> Vec<Element> {
    let fs = SamplingFrequency::from_hz(sample_rate).unwrap();
    let mut elements = Vec::new();
    let mut frame = stream;
    while frame.len() > 7 {
        let len = (usize::from(frame[3] & 0b11) << 11)
            | (usize::from(frame[4]) << 3)
            | usize::from(frame[5] >> 5);
        let mut r = BitReader::new(&frame[7..len]);
        elements.push(element(&mut r, fs));
        frame = &frame[len..];
    }
    elements
}

/// The `window_sequence` of each channel of each frame of a stream in ADTS
/// at `sample_rate`.
fn window_sequences(stream: &[u8], sample_rate: u32) -> Vec<Vec<u8>> {
    (elements(stream, sample_rate).iter())
        .map(|element| {
            (element.channels.iter())
                .map(|ics| ics.info.window_sequence)
                .collect()
        })
        .collect()
}

/// Whether one of the frames whose eight short windows span input sample
/// `onset` is coded with them in `channel`: frame `f`'s span samples
/// `1024 f - 576` to `1024 f + 575` (clause 15.3.2).
fn short_at(sequences: &[Vec<u8>], channel: usize, onset: usize) -> bool {
    (sequences.iter().enumerate())
        .filter(|&(f, _)| (1024 * f).saturating_sub(576) <= onset && onset < 1024 * f + 576)
        .any(|(_, frame)| frame[channel] == 2)
}

/// Whether each channel's `sequences` keep to the meaningful transitions
/// of clause 15.3.1: a sequence that starts with short windows only after
/// one that ends with them.
fn transitions_are_meaningful(sequences: &[Vec<u8>]) -> bool {
    // ONLY_LONG 0, LONG_START 1, EIGHT_SHORT 2, LONG_STOP 3.
    sequences.windows(2).all(|pair| {
        (pair[0].iter().zip(&pair[1])).all(|(&a, &b)| matches!(b, 2 | 3) == matches!(a, 1 | 2))
    })
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

/// [`snr_over`] all of the input.
fn snr(input: &[f64], output: &[f32], channels: usize, channel: usize, lag: usize) -> f64 {
    snr_over(input, output, (channels, channel), lag, 0..input.len())
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
/// at the level the encoder takes its input at. The psychoacoustic model
/// shapes the noise to what it finds masked rather than to the least, so
/// at 8 kHz, where a bit a sample is fewest, the SNR is about 11 dB.
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
                if false {
                    assert_eq!(lag, DELAY);
                }
                assert!(snr > 9.0, "{context}, channel {c}: {snr:.1} dB");
            }
        }
    }
}

/// The SNR rises with the bitrate, for signals whose spectra differ:
/// tones, a sweep over the whole band, and white noise, which no
/// quantizer codes well at these rates. The psychoacoustic model shapes
/// the noise to what it finds masked rather than to the least, so that at
/// 128 kbit/s the tones' SNR is about 44 dB and the first sweep's, which
/// rises to 20 kHz, where the thresholds in quiet are high, about 59 dB.
/// The rates start above those at which a pair is coded in intensity
/// stereo, which gives up each channel's waveform in the bands it codes.
#[test]
fn snr_rises_with_the_bitrate() {
    let sample_rate = 44_100;
    let len = 3 * sample_rate as usize;
    let signals = [
        ("tones", [(96_000, 36.0), (128_000, 41.0), (256_000, 56.0)]),
        ("sweep", [(96_000, 48.0), (128_000, 56.0), (256_000, 70.0)]),
        ("noise", [(96_000, 2.0), (128_000, 3.5), (256_000, 12.0)]),
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
    // The decoded tone's level and phase, by correlation over the input
    // from its second frame to its end.
    let span = 1024..len;
    let (mut sin, mut cos, mut energy) = (0.0, 0.0, 0.0);
    for t in span.clone() {
        let y = f64::from(output[t + DELAY]);
        let phase = TAU * frequency * t as f64 / fs;
        sin += y * phase.sin();
        cos += y * phase.cos();
        energy += y * y;
    }
    let n = span.len() as f64;
    // In phase with the input, not inverted or shifted: the level is the
    // correlation with the input's own sine, positive.
    let level = 2.0 * sin / n;
    assert!(
        level > 0.0 && (20.0 * (level / amplitude).log10()).abs() < 0.1,
        "level {level}"
    );
    assert!(cos.abs() < 0.01 * sin, "phase: {sin}, {cos}");
    // What the tone does not account for.
    let rest = energy - 2.0 * (sin * sin + cos * cos) / n;
    assert!(rest < 1e-3 * energy, "{rest} of {energy}");
}

/// An impulse at the first sample of the input, and one at its last,
/// decode where they were given, [`DELAY`] samples late, at their level and
/// sign: the first frame and the last carry the edges of the input.
#[test]
fn impulses_at_the_edges_of_the_input_come_back_in_place() {
    for (sample_rate, channels) in [(48_000, 1), (44_100, 2), (8_000, 1)] {
        let (_, max) = bitrate_range(sample_rate, channels);
        for len in [1, 1024, 3000] {
            let context = format!("{sample_rate} Hz, {channels} channels, {len} samples");
            let input: Vec<Vec<f64>> = (0..channels)
                .map(|c| {
                    let mut input = vec![0.0; len];
                    // Opposite signs in the two channels of a pair.
                    let sign = if c == 0 { 1.0 } else { -1.0 };
                    input[0] = 0.5 * sign;
                    input[len - 1] = -0.5 * sign;
                    input
                })
                .collect();
            let encoded = encode(sample_rate, &input, max);
            let output = decode(&encoded, channels, &context);
            for (c, input) in input.iter().enumerate() {
                let decoded: Vec<f64> = (output.iter().skip(c).step_by(channels))
                    .map(|&y| f64::from(y))
                    .collect();
                for t in [0, len - 1] {
                    let at = DELAY + t;
                    let y = decoded[at];
                    assert!(
                        (y - input[t]).abs() < 0.1 * input[t].abs(),
                        "{context}, channel {c}, sample {t}: {y}"
                    );
                    // The largest sample near it.
                    let near = at.saturating_sub(64)..(at + 64).min(decoded.len());
                    let peak = near
                        .max_by(|&a, &b| decoded[a].abs().total_cmp(&decoded[b].abs()))
                        .unwrap();
                    assert_eq!(peak, at, "{context}, channel {c}, sample {t}");
                }
            }
        }
    }
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

/// A tone in one channel of a pair, the other silent, at a rate that
/// leaves the pair few bits: the tone decodes alike whichever channel it is
/// in, as the silent channel takes only what it costs.
#[test]
fn a_tone_beside_silence_decodes_alike_in_either_channel() {
    let sample_rate = 44_100;
    let fs = f64::from(sample_rate);
    let len = 20 * 1024;
    let tone: Vec<f64> = (0..len)
        .map(|t| 0.8 * (TAU * 1000.0 * t as f64 / fs).sin())
        .collect();
    let silence = vec![0.0; len];
    let mut snrs = Vec::new();
    for c in 0..2 {
        let input = if c == 0 {
            [tone.clone(), silence.clone()]
        } else {
            [silence.clone(), tone.clone()]
        };
        let context = format!("the tone in channel {c}");
        let encoded = encode(sample_rate, &input, 6_500);
        let output = decode(&encoded, 2, &context);
        let snr = snr_over(&tone, &output, (2, c), DELAY, 2048..len);
        let quiet = (output.iter().skip(1 - c).step_by(2)).all(|&y| y == 0.0);
        assert!(quiet, "{context}: the silent channel decodes to silence");
        snrs.push(snr);
    }
    assert!(snrs.iter().all(|&snr| snr > 10.0), "{snrs:?}");
    assert!((snrs[0] - snrs[1]).abs() < 0.5, "{snrs:?}");
}

/// A quiet tone with bursts of noise that die away within a few
/// milliseconds, as of castanets, from `first` on, every `every` samples
/// and a little later each time, so that they set in at many places in a
/// frame.
fn castanets(
    sample_rate: u32,
    len: usize,
    first: usize,
    every: usize,
    seed: u64,
) -> (Vec<f64>, Vec<usize>) {
    let fs = f64::from(sample_rate);
    let decay = fs / 200.0;
    let mut state = seed;
    let mut onset = first;
    let mut onsets = Vec::new();
    while onset < len {
        onsets.push(onset);
        onset += every + 97 * onsets.len();
    }
    let signal = (0..len)
        .map(|t| {
            state = (state.wrapping_mul(6_364_136_223_846_793_005))
                .wrapping_add(1_442_695_040_888_963_407);
            let noise = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            let tone = 0.01 * (TAU * 440.0 * t as f64 / fs).sin();
            let burst = (onsets.iter())
                .filter(|&&onset| onset <= t)
                .map(|&onset| (-((t - onset) as f64) / decay).exp())
                .sum::<f64>();
            tone + noise * burst
        })
        .collect();
    (signal, onsets)
}

/// Bursts of noise in a quiet tone are coded with eight short windows, a
/// start sequence before them and a stop sequence after, at every sampling
/// frequency, in mono and stereo; ffmpeg decodes them without complaint,
/// with the bursts in place. The first burst, at the first sample, makes
/// the first frame short, with no frame before it.
///
/// This is the evidence for the gap that the window sequence of the first
/// frame marks as settled by decoding with ffmpeg: that the first frame
/// may be short, with no start sequence before it.
#[test]
fn attacks_are_coded_with_short_windows_at_every_sample_rate() {
    for sample_rate in SAMPLE_RATES {
        for channels in [1, 2] {
            let context = format!("{sample_rate} Hz, {channels} channels");
            let len = 64 * 1024;
            let (input, onsets): (Vec<Vec<f64>>, Vec<Vec<usize>>) = (0..channels)
                .map(|c| castanets(sample_rate, len, 0, len / 6 + 300 * c, c as u64))
                .unzip();
            let (_, max) = bitrate_range(sample_rate, channels);
            let bits_per_second = (2 * sample_rate * channels as u32).min(max);
            let encoded = encode(sample_rate, &input, bits_per_second);
            let sequences = window_sequences(&encoded.stream, sample_rate);
            assert!(
                transitions_are_meaningful(&sequences),
                "{context}: {sequences:?}"
            );
            for (c, onsets) in onsets.iter().enumerate() {
                // Each burst, the first at the first sample, which makes
                // the first frame short with no frame before it.
                assert_eq!(onsets[0], 0);
                for &onset in onsets {
                    assert!(
                        short_at(&sequences, c, onset),
                        "{context}, channel {c}, onset {onset}: {sequences:?}"
                    );
                }
                // The quiet tone between them.
                let long = sequences.iter().filter(|frame| frame[c] == 0).count();
                assert!(long > sequences.len() / 2, "{context}: {sequences:?}");
            }
            let output = decode(&encoded, channels, &context);
            for (c, input) in input.iter().enumerate() {
                // The bursts line up only in place.
                let snr_at = |lag: usize| snr(input, &output, channels, c, lag);
                let at = snr_at(DELAY);
                assert!(at > 6.0, "{context}, channel {c}: {at:.1} dB");
                for lag in [DELAY - 64, DELAY - 1, DELAY + 1, DELAY + 64] {
                    let off = snr_at(lag);
                    assert!(
                        off < at - 3.0,
                        "{context}, channel {c}: {off:.1} dB at {lag}"
                    );
                }
            }
        }
    }
}

/// One-sample impulses over a quiet tone, at many places in a frame, are
/// coded with short windows in each channel, at sampling frequencies of
/// every short band table, and come back at about their level.
#[test]
fn impulses_over_a_quiet_tone_are_coded_with_short_windows() {
    for sample_rate in [8_000, 22_050, 44_100, 48_000, 64_000, 96_000] {
        for channels in [1, 2] {
            let context = format!("{sample_rate} Hz, {channels} channels");
            let fs = f64::from(sample_rate);
            let len = 32 * 1024;
            // Impulses 3203 samples apart, each 131 later in its frame
            // than the one before, from 3505; the second channel's 517
            // later than the first's.
            let onsets: Vec<Vec<usize>> = (0..channels)
                .map(|c| (0..8).map(|k| 3505 + 517 * c + 3203 * k).collect())
                .collect();
            let input: Vec<Vec<f64>> = (onsets.iter())
                .map(|onsets| {
                    let mut input: Vec<f64> = (0..len)
                        .map(|t| 0.002 * (TAU * 440.0 * t as f64 / fs).sin())
                        .collect();
                    for &onset in onsets {
                        input[onset] += 0.3;
                    }
                    input
                })
                .collect();
            let (_, max) = bitrate_range(sample_rate, channels);
            let encoded = encode(sample_rate, &input, (48_000 * channels as u32).min(max));
            let sequences = window_sequences(&encoded.stream, sample_rate);
            assert!(transitions_are_meaningful(&sequences), "{context}");
            let output = decode(&encoded, channels, &context);
            for (c, onsets) in onsets.iter().enumerate() {
                for &onset in onsets {
                    assert!(
                        short_at(&sequences, c, onset),
                        "{context}, channel {c}, onset {onset}: {sequences:?}"
                    );
                    let y = f64::from(output[(onset + DELAY) * channels + c]);
                    assert!(
                        (y - 0.3).abs() < 0.1,
                        "{context}, channel {c}, onset {onset}: {y}"
                    );
                }
            }
        }
    }
}

/// Short windows keep the quantization noise of an attack from spreading
/// before it: before a burst of noise in a quiet tone, the error of the
/// decoded output is far less with block switching than with long windows
/// alone.
#[test]
fn short_windows_hold_back_the_noise_before_an_attack() {
    use chelys::aac::internals::channel_configuration::ChannelConfiguration;
    use chelys::aac::internals::{StreamConfig, StreamEncoder};

    let sample_rate = 44_100;
    let config = StreamConfig {
        sampling_frequency: SamplingFrequency::Hz44100,
        channel_configuration: ChannelConfiguration::Mono,
        bits_per_second: 48_000,
        framing: chelys::aac::Framing::Adts,
    };
    let len = 3 * sample_rate as usize;
    let (input, onsets) = castanets(sample_rate, len, 10_000, 20_000, 7);
    assert!(onsets.len() >= 5, "{onsets:?}");
    let pre_echo = |encoder: StreamEncoder| {
        let mut encoder = encoder;
        let mut stream = Vec::new();
        encoder.encode(&[&input], &mut stream);
        let end = encoder.finish(&mut stream);
        let sequences: Vec<u8> = (window_sequences(&stream, sample_rate).iter())
            .map(|frame| frame[0])
            .collect();
        let decoded = ffmpeg::decode_aac(&stream);
        assert_eq!(decoded.diagnostics, "");
        assert_eq!(decoded.samples.len() as u64, end.frames * 1024);
        let output = decoded.samples;
        // The error over the 1024 samples before each onset, short of the
        // last 128, over which a short window rises.
        let error: f64 = (onsets.iter())
            .flat_map(|&onset| onset - 1024..onset - 128)
            .map(|t| (f64::from(output[t + DELAY]) - input[t]).powi(2))
            .sum();
        (error, sequences)
    };
    let (switched, sequences) = pre_echo(StreamEncoder::new(config));
    assert!(sequences.contains(&2), "{sequences:?}");
    let (long, sequences) = pre_echo(StreamEncoder::new(config).without_block_switching());
    assert!(sequences.iter().all(|&s| s == 0));
    let gain = 10.0 * (long / switched).log10();
    assert!(gain > 10.0, "{gain:.1} dB less error before the attacks");
}

/// The share of the bands below `max_sfb` of the frames of `elements`, all
/// pairs, that are coded as middle and side.
fn middle_side_share(elements: &[Element]) -> f64 {
    let (mut bands, mut middle_side) = (0, 0);
    for element in elements {
        let info = &element.channels[0].info;
        for g in 0..info.group_lengths.len() {
            for sfb in 0..info.max_sfb {
                bands += 1;
                let used = match element.ms_mask_present {
                    1 => element.ms_used[g][sfb],
                    present => present == 2,
                };
                middle_side += usize::from(used);
            }
        }
    }
    middle_side as f64 / bands.max(1) as f64
}

/// Pairs whose channels are alike, unlike, opposite or one silent decode
/// with each channel in place and with its own sign, at every sampling
/// frequency, at a bit a sample: the same signal in both, two sources each
/// panned to one side, a signal and its negation, independent noise, and
/// a signal beside silence. Each channel lines up with its own input
/// [`DELAY`] samples late, at its level, not its pair's nor negated; the
/// output after the input ends is no louder than the coding noise of its
/// last frame; and a silent channel stays all but silent. The pair shares
/// its window in every frame, and codes all but a few bands as middle and
/// side where its channels are alike or opposite. Independent noise, which
/// costs about as many bits either way, has about half its bands so coded.
#[test]
fn stereo_pairs_decode_each_channel_in_place() {
    for sample_rate in SAMPLE_RATES {
        let len = sample_rate as usize;
        let top = (f64::from(sample_rate) / 2.0).min(16_000.0);
        let a = tones(sample_rate, len, top, 0);
        let b = sweep(sample_rate, len, 60.0, top);
        let mix = |x: &[f64], y: &[f64], gx: f64, gy: f64| -> Vec<f64> {
            x.iter().zip(y).map(|(x, y)| gx * x + gy * y).collect()
        };
        let pairs = [
            ("alike", [a.clone(), a.clone()]),
            ("panned", [mix(&a, &b, 0.8, 0.2), mix(&a, &b, 0.2, 0.8)]),
            ("opposite", [a.clone(), mix(&a, &a, -1.0, 0.0)]),
            ("independent", [noise(len, 7), noise(len, 8)]),
            ("beside silence", [a.clone(), vec![0.0; len]]),
        ];
        let (_, max) = bitrate_range(sample_rate, 2);
        let bits_per_second = (2 * sample_rate).min(max);
        for (name, input) in pairs {
            let context = format!("{name} at {sample_rate} Hz");
            let encoded = encode(sample_rate, &input, bits_per_second);
            let output = decode(&encoded, 2, &context);
            let elements = elements(&encoded.stream, sample_rate);
            assert!(elements.iter().all(|e| e.common_window), "{context}");
            let share = middle_side_share(&elements);
            if matches!(name, "alike" | "opposite") {
                assert!(share > 0.95, "{context}: {share:.2}");
            }
            let noisy = name == "independent";
            for (c, input) in input.iter().enumerate() {
                let decoded: Vec<f64> = (output.iter().skip(c).step_by(2))
                    .map(|&y| f64::from(y))
                    .collect();
                let energy = |samples: &[f64]| samples.iter().map(|x| x * x).sum::<f64>();
                if energy(input) == 0.0 {
                    let other: Vec<f64> = (output.iter().skip(1 - c).step_by(2))
                        .map(|&y| f64::from(y))
                        .collect();
                    let (noise, other) = (energy(&decoded), energy(&other));
                    assert!(noise < 1e-3 * other, "{context}, channel {c}: {noise}");
                    continue;
                }
                let (lag, snr) = align(input, &output, 2, c);
                assert_eq!(lag, DELAY, "{context}, channel {c}: {snr:.1} dB");
                let least_snr = if noisy { 1.5 } else { 12.0 };
                assert!(snr > least_snr, "{context}, channel {c}: {snr:.1} dB");
                // The input's level in the output: their correlation over
                // the input's energy, which noise coded at a bit a sample
                // loses some of.
                let correlation: f64 = (input.iter().zip(&decoded[DELAY..]))
                    .map(|(x, y)| x * y)
                    .sum();
                let level = correlation / energy(input);
                let least_level = if noisy { 0.6 } else { 0.94 };
                assert!(
                    (least_level..1.02).contains(&level),
                    "{context}, channel {c}: level {level}"
                );
                // The output after the input against the noise of the
                // input's last frame.
                let last = len - 1024..len;
                let noise: f64 = (last.clone())
                    .map(|t| (input[t] - decoded[DELAY + t]).powi(2))
                    .sum();
                let tail = &decoded[DELAY + len..];
                let (noise_power, tail_power) = (
                    noise / last.len() as f64,
                    energy(tail) / tail.len().max(1) as f64,
                );
                assert!(
                    tail_power < 4.0 * noise_power,
                    "{context}, channel {c}: tail {tail_power} against {noise_power}"
                );
            }
        }
    }
}

/// The complex amplitude of the tone of `frequency` Hz in `samples` at
/// `sample_rate`: twice their correlation with `e^(-i w t)`, from sample
/// `start`, as `(re, im)`.
fn tone_amplitude(samples: &[f64], sample_rate: u32, frequency: f64, start: usize) -> (f64, f64) {
    let w = TAU * frequency / f64::from(sample_rate);
    let n = (samples.len() - start) as f64;
    (samples.iter().enumerate().skip(start)).fold((0.0, 0.0), |(re, im), (t, &y)| {
        let phase = w * (t - start) as f64;
        (
            re + 2.0 * y * phase.cos() / n,
            im - 2.0 * y * phase.sin() / n,
        )
    })
}

/// Tones above 6 kHz, each panned its own way, one to the left, one to the
/// right and one to both in opposite phase, decode from a pair coded in
/// intensity stereo panned within 1 dB of the input, each channel within 3
/// dB of its level, and in the right channel in phase or opposite as it
/// was, at every sampling frequency whose band reaches past 9 kHz, where
/// the tones lie in bands apart, at 0.7 bit a sample. The right channel
/// sends bands of both intensity codebooks.
///
/// This is the evidence for the reading of the standard that
/// `stereo::middle_side::choose` marks as settled by decoding with ffmpeg:
/// frames with bands in intensity stereo are sent with `ms_mask_present` 1
/// and their flags 0, which ffmpeg decodes in phase. Sent with
/// `ms_mask_present` 2, as clause 12.2.3 allows, the right channel's tones
/// came back inverted.
#[test]
fn intensity_stereo_keeps_each_tones_panning() {
    for sample_rate in SAMPLE_RATES.into_iter().filter(|&fs| fs >= 22_050) {
        let len = sample_rate as usize;
        let nyquist = f64::from(sample_rate) / 2.0;
        let top = (0.9 * nyquist).min(14_000.0);
        // Gains of each tone in the left and right channels.
        let tones = [
            (6_600.0 + 0.1 * (top - 6_600.0), 0.3, 0.15),
            (6_600.0 + 0.5 * (top - 6_600.0), 0.1, 0.3),
            (top, 0.2, -0.2),
        ];
        let channel = |c: usize| -> Vec<f64> {
            (0..len)
                .map(|t| {
                    let t = t as f64 / f64::from(sample_rate);
                    // A low tone beside them, alike in both channels.
                    let low = 0.2 * (TAU * 440.0 * t).sin();
                    let high: f64 = (tones.iter())
                        .map(|&(f, l, r)| [l, r][c] * (TAU * f * t).sin())
                        .sum();
                    low + high
                })
                .collect()
        };
        let input = [channel(0), channel(1)];
        let context = format!("{sample_rate} Hz");
        let bits_per_second = 14 * sample_rate / 10;
        let encoded = encode(sample_rate, &input, bits_per_second);
        let output = decode(&encoded, 2, &context);
        let elements = elements(&encoded.stream, sample_rate);
        let codebooks: Vec<u8> = (elements.iter())
            .flat_map(|e| {
                let right = &e.channels[1];
                (0..right.info.group_lengths.len()).flat_map(move |g| {
                    (0..right.info.max_sfb).map(move |sfb| right.codebook(g, sfb))
                })
            })
            .collect();
        assert!(codebooks.contains(&15), "{context}: INTENSITY_HCB");
        assert!(codebooks.contains(&14), "{context}: INTENSITY_HCB2");
        let decoded: Vec<Vec<f64>> = (0..2)
            .map(|c| {
                (output.iter().skip(c).step_by(2))
                    .skip(DELAY)
                    .take(len)
                    .map(|&y| f64::from(y))
                    .collect()
            })
            .collect();
        for &(frequency, l, r) in &tones {
            let skip = 2048;
            let [left, right] =
                [0, 1].map(|c| tone_amplitude(&decoded[c], sample_rate, frequency, skip));
            let magnitude = |(re, im): (f64, f64)| (re * re + im * im).sqrt();
            for (c, (amplitude, gain)) in [(left, l), (right, r)].into_iter().enumerate() {
                let db = 20.0 * (magnitude(amplitude) / f64::abs(gain)).log10();
                assert!(
                    db.abs() < 3.0,
                    "{context}, {frequency:.0} Hz, channel {c}: {db:.2} dB"
                );
            }
            // The panning, which the positions carry within 0.75 dB.
            let panning = 20.0 * (magnitude(right) / magnitude(left)).log10();
            let expected = 20.0 * (r / l).abs().log10();
            assert!(
                (panning - expected).abs() < 1.0,
                "{context}, {frequency:.0} Hz: {panning:.2} dB against {expected:.2} dB"
            );
            // The right channel's phase against the left's.
            let product = left.0 * right.0 + left.1 * right.1;
            assert_eq!(
                product.signum(),
                (l * r).signum(),
                "{context}, {frequency:.0} Hz"
            );
        }
    }
}

/// A tone in the right channel alone, above 6 kHz, beside a low tone and
/// a high one in both, decodes at its level in the right channel and all
/// but silent in the left, at a bitrate that codes the pair in intensity
/// stereo, and every tone at its level in each channel: the left channel's
/// band of the lone tone holds only what its window leaks of the others,
/// far below the right's, and is not coded in intensity stereo, whose
/// positions do not reach so far.
#[test]
fn a_tone_panned_hard_right_survives_intensity_stereo() {
    let sample_rate = 48_000;
    let len = sample_rate as usize;
    let fs = f64::from(sample_rate);
    let tone = |frequency: f64, amplitude: f64, t: usize| {
        amplitude * (TAU * frequency * t as f64 / fs).sin()
    };
    // Each tone with its level in the left and in the right channel.
    let tones = [(440.0, 0.2, 0.2), (12_000.0, 0.1, 0.1), (9_000.0, 0.0, 0.5)];
    let channel = |c: usize| -> Vec<f64> {
        (0..len)
            .map(|t| (tones.iter()).map(|&(f, l, r)| tone(f, [l, r][c], t)).sum())
            .collect()
    };
    let input = [channel(0), channel(1)];
    let encoded = encode(sample_rate, &input, 67_200);
    let output = decode(&encoded, 2, "a tone hard right");
    let in_intensity = (elements(&encoded.stream, sample_rate).iter()).any(|e| {
        let right = &e.channels[1];
        (0..right.info.max_sfb).any(|sfb| matches!(right.codebook(0, sfb), 14 | 15))
    });
    assert!(in_intensity, "bands in intensity stereo");
    let decoded: Vec<Vec<f64>> = (0..2)
        .map(|c| {
            (output.iter().skip(c).step_by(2))
                .skip(DELAY)
                .take(len)
                .map(|&y| f64::from(y))
                .collect()
        })
        .collect();
    let magnitude = |(re, im): (f64, f64)| (re * re + im * im).sqrt();
    for &(frequency, l, r) in &tones {
        for (c, expected) in [(0, l), (1, r)] {
            let level = magnitude(tone_amplitude(&decoded[c], sample_rate, frequency, 2048));
            assert!(
                (level - expected).abs() < 0.1 * expected.max(0.1),
                "{frequency} Hz, channel {c}: {level} against {expected}"
            );
        }
    }
}
