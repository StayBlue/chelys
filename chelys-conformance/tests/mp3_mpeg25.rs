// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! MPEG-2.5 (US patent 6,185,539 B1), decoded by ffmpeg, mpg123 and LAME.

use std::f64::consts::TAU;

use chelys::mp3::header::{Bitrate, Mode, SamplingFrequency, Version};
use chelys::mp3::rate::Rate;
use chelys::mp3::scalefactor_bands;
use chelys::mp3::scalefactors::Scalefactors;
use chelys::mp3::{
    BlockType, GRANULE_LINES, QuantizedGranule, StreamConfig, encode_granules, encode_pcm,
    encode_pcm_with_global_gain, encode_silence, encode_with_block_types,
};
use chelys_conformance::DECODERS;

fn config(sampling_frequency: SamplingFrequency, mode: Mode, kbps: u32) -> StreamConfig {
    StreamConfig {
        rate: Rate::Constant(Bitrate::from_kbps(Version::Mpeg2, kbps).unwrap()),
        sampling_frequency,
        mode,
        crc_protected: false,
    }
}

/// Each decoder reads the sampling frequency of every MPEG-2.5 stream from
/// its sync word and `sampling_frequency` code.
///
/// This is the evidence for the gap in US 6,185,539 B1 that the header marks
/// as settled by decoding with ffmpeg, mpg123 and LAME: the
/// `sampling_frequency` codes of MPEG-2.5.
#[test]
fn decoders_read_the_mpeg25_sampling_frequencies() {
    for fs in SamplingFrequency::MPEG25 {
        let stream = encode_silence(&config(fs, Mode::SingleChannel, 32), 8);
        for decoder in DECODERS {
            assert_eq!((decoder.sample_rate)(&stream), fs.hz(), "{}", decoder.name);
        }
    }
}

/// The granules between the probes, silent, so that the decoded probes do
/// not overlap.
const PROBE_PERIOD: usize = 4;

/// The scalefactors of the probes, repeated over the scalefactors in the
/// order sent: each band decodes 0, 9 or 18 dB below its neighbours'
/// levels, so that every boundary between bands shows.
const PROBE_SCALEFACTORS: [u8; 3] = [0, 3, 6];

/// A stream of one probe granule for each line in the order sent, with only
/// that line nonzero, divided by the scalefactors of [`PROBE_SCALEFACTORS`]
/// if `scaled`.
///
/// The lines are 1 or 0, and so are sent as quadruples, but for at most the
/// pair of lines 0 and 1, which lie in region 0 whatever the bands: their
/// Huffman coding does not depend on where the decoder puts the bands.
fn probe_stream(fs: SamplingFrequency, block_type: BlockType, scaled: bool) -> Vec<u8> {
    let silent = QuantizedGranule::new(200, [0; GRANULE_LINES], block_type, fs);
    let mut granules = Vec::new();
    for line in 0..GRANULE_LINES {
        let mut values = [0; GRANULE_LINES];
        values[line] = 1;
        let mut probe = QuantizedGranule::new(200, values, block_type, fs);
        assert!(probe.coding.big_values <= 1, "line {line}");
        let mut scalefactors = Scalefactors::new(block_type, Version::Mpeg2);
        if scaled {
            for (index, s) in scalefactors.values_mut().iter_mut().enumerate() {
                *s = PROBE_SCALEFACTORS[index % PROBE_SCALEFACTORS.len()];
            }
        }
        probe.scalefactors = scalefactors;
        granules.push(vec![probe]);
        granules.extend(std::iter::repeat_n(vec![silent.clone()], PROBE_PERIOD - 1));
    }
    encode_granules(&config(fs, Mode::SingleChannel, 32), &granules)
}

/// The energy of each decoded probe, in the samples from `offset` past the
/// start of its granule on, as long as a probe and its overlap with the
/// next granule last; `cumulative` sums the squared samples.
fn probe_energies(cumulative: &[f64], offset: isize) -> Vec<f64> {
    let at = |t: isize| cumulative[t.clamp(0, cumulative.len() as isize - 1) as usize];
    (0..GRANULE_LINES)
        .map(|line| {
            let start = (line * PROBE_PERIOD * GRANULE_LINES) as isize + offset;
            at(start + 3 * GRANULE_LINES as isize) - at(start)
        })
        .collect()
}

fn cumulative_energy(samples: &[f32]) -> Vec<f64> {
    let mut sum = 0.0;
    std::iter::once(0.0)
        .chain(samples.iter().map(|&s| {
            sum += f64::from(s) * f64::from(s);
            sum
        }))
        .collect()
}

/// The widths of the bands a decoder puts the lines in, in the order sent,
/// with the lines above the last band as one more: from where the decoded
/// level of a probe changes, as the scalefactor of [`PROBE_SCALEFACTORS`]
/// dividing it changes. `streams` are the probe streams without and with
/// the scalefactors.
fn decoded_widths(
    decode: fn(&[u8]) -> chelys_conformance::Decoded,
    streams: &[Vec<u8>; 2],
) -> Vec<usize> {
    let plain = decode(&streams[0]);
    let scaled = decode(&streams[1]);
    assert_eq!(plain.diagnostics, "");
    assert_eq!(scaled.diagnostics, "");
    let (plain, scaled) = (
        cumulative_energy(&plain.samples),
        cumulative_energy(&scaled.samples),
    );
    // The decoder's delay: where the windows hold the most of the probes.
    let offset = (-(2 * GRANULE_LINES as isize)..2 * GRANULE_LINES as isize)
        .step_by(8)
        .max_by(|&a, &b| {
            let total = |offset| probe_energies(&plain, offset).iter().sum::<f64>();
            total(a).total_cmp(&total(b))
        })
        .unwrap();
    let step_db = 10.0 * 2f64.log10() * f64::from(PROBE_SCALEFACTORS[1]);
    let levels: Vec<usize> = probe_energies(&plain, offset)
        .iter()
        .zip(probe_energies(&scaled, offset))
        .enumerate()
        .map(|(line, (plain, scaled))| {
            let db = 10.0 * (plain / scaled).log10();
            let level = (db / step_db).round();
            assert!(
                (db - level * step_db).abs() < 1.0,
                "line {line}: {db:.2} dB below the unscaled probe"
            );
            level as usize
        })
        .collect();
    let mut widths = vec![1];
    for pair in levels.windows(2) {
        if pair[0] == pair[1] {
            *widths.last_mut().unwrap() += 1;
        } else {
            widths.push(1);
        }
    }
    widths
}

/// Each decoder puts the lines of every MPEG-2.5 stream in the scalefactor
/// bands that the encoder sends them in, for long and short blocks.
///
/// This is the evidence for the gap in US 6,185,539 B1 that the scalefactor
/// bands mark as settled by decoding with ffmpeg, mpg123 and LAME: the bands
/// of 11.025 and 12 kHz.
#[test]
fn decoders_use_the_mpeg25_scalefactor_bands() {
    for fs in SamplingFrequency::MPEG25 {
        for block_type in [BlockType::Normal, BlockType::Short] {
            let bands = scalefactor_bands::coding_bands(block_type, fs);
            let top = GRANULE_LINES - bands.last().unwrap().end();
            let expected: Vec<usize> = (bands.iter().map(|band| band.width)).chain([top]).collect();
            let streams = [false, true].map(|scaled| probe_stream(fs, block_type, scaled));
            for decoder in DECODERS {
                assert_eq!(
                    decoded_widths(decoder.decode, &streams),
                    expected,
                    "{} at {fs:?}, {block_type:?}",
                    decoder.name
                );
            }
        }
    }
}

/// Tones in even and odd subbands, mid-band and near subband edges, fading
/// in and out over 20 ms, outside the part [`align`] compares.
fn tones(sampling_frequency: SamplingFrequency, seconds: f64, offset: f64) -> Vec<f64> {
    let fs = f64::from(sampling_frequency.hz());
    let band = fs / 64.0;
    let frequencies = [
        0.5 * band,
        1.5 * band,
        3.97 * band,
        7.03 * band,
        12.5 * band,
        21.2 * band,
    ];
    let len = (seconds * fs) as usize;
    let fade = (0.02 * fs) as usize;
    (0..len)
        .map(|t| {
            let edge = t.min(len - 1 - t);
            let gain = if edge < fade {
                0.5 - 0.5 * (TAU / 2.0 * edge as f64 / fade as f64).cos()
            } else {
                1.0
            };
            let sum: f64 = (frequencies.iter().enumerate())
                .map(|(i, f)| 0.08 * (TAU * f * t as f64 / fs + offset * i as f64).sin())
                .sum();
            gain * sum
        })
        .collect()
}

/// The signal-to-noise ratio in dB of channel `channel` of `output`,
/// interleaved over `channels`, against `input`, over the steady middle
/// part, at the lag that lines them up best. Samples past the end of the
/// output, which LAME shortens by its delay, count as zero.
fn align(input: &[f64], output: &[f32], channels: usize, channel: usize) -> f64 {
    let span = 1152 * 2..input.len() - 1152 * 2;
    let snr = |lag: usize| {
        let (signal, noise) = span.clone().fold((0.0, 0.0), |(s, n), t| {
            let x = input[t];
            let y = output
                .get((t + lag) * channels + channel)
                .map_or(0.0, |&y| f64::from(y));
            (s + x * x, n + (x - y) * (x - y))
        });
        10.0 * (signal / noise).log10()
    };
    (0..3000).map(snr).fold(f64::NEG_INFINITY, f64::max)
}

/// Every MPEG-2.5 sampling frequency decodes close to its input: with a
/// fixed quantizer, in long blocks, in short ones, and switching between
/// them (with a tone above the last short band, in the lines that have no
/// scalefactor); and through the whole encoder, in one channel and in two.
#[test]
fn mpeg25_round_trips() {
    use BlockType::*;
    const CYCLE: [BlockType; 13] = [
        Normal, Normal, Start, Short, Short, Stop, Normal, Start, Short, Stop, Start, Short, Stop,
    ];
    for fs in SamplingFrequency::MPEG25 {
        let rate = f64::from(fs.hz());
        let left = tones(fs, 1.5, 0.0);
        let right: Vec<f64> = tones(fs, 1.5, 1.7).iter().map(|x| -0.7 * x).collect();
        let with_top: Vec<f64> = (left.iter().enumerate())
            .map(|(t, x)| x + 0.08 * (TAU * 0.95 * rate / 2.0 * t as f64 / rate).sin())
            .collect();
        let fixed = config(fs, Mode::SingleChannel, 160);
        let cases = [
            (
                "fixed quantizer, long blocks",
                encode_pcm_with_global_gain(&fixed, 150, &[&left]),
                vec![&left],
                50.0,
            ),
            (
                "fixed quantizer, every block transition",
                encode_with_block_types(&fixed, 154, &[&with_top], |_, g| CYCLE[g % CYCLE.len()]),
                vec![&with_top],
                45.0,
            ),
            (
                "fixed quantizer, short blocks",
                encode_with_block_types(&fixed, 154, &[&with_top], |_, _| Short),
                vec![&with_top],
                45.0,
            ),
            (
                "encoder, one channel at 32 kbit/s",
                encode_pcm(&config(fs, Mode::SingleChannel, 32), &[&left]),
                vec![&left],
                40.0,
            ),
            (
                "encoder, two channels at 64 kbit/s",
                encode_pcm(&config(fs, Mode::Stereo, 64), &[&left, &right]),
                vec![&left, &right],
                40.0,
            ),
        ];
        for (label, stream, inputs, min_snr) in &cases {
            for decoder in DECODERS {
                let decoded = (decoder.decode)(stream);
                assert_eq!(decoded.diagnostics, "", "{} at {fs:?}", decoder.name);
                for (channel, input) in inputs.iter().enumerate() {
                    let snr = align(input, &decoded.samples, inputs.len(), channel);
                    assert!(
                        snr > *min_snr,
                        "{} at {fs:?}, {label}, channel {channel}: {snr:.1} dB",
                        decoder.name
                    );
                }
            }
        }
    }
}

/// A steady signal below 3.2 kHz, `seconds` long and fading in and out over
/// 20 ms, the same at every sampling frequency: four tones over many quieter
/// sinusoids, spread like noise.
fn below_3_2_khz(fs: SamplingFrequency, seconds: f64) -> Vec<f64> {
    let mut seed = 11u64;
    let mut next = || {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut components = vec![(300.0, 0.2, 0.0), (900.0, 0.1, 1.0), (1700.0, 0.05, 2.0)];
    components.push((2600.0, 0.1, 3.0));
    for _ in 0..150 {
        components.push((100.0 + 3100.0 * next(), 0.003, TAU * next()));
    }
    let rate = f64::from(fs.hz());
    let len = (seconds * rate) as usize;
    let fade = (0.02 * rate) as usize;
    (0..len)
        .map(|t| {
            let edge = t.min(len - 1 - t);
            let gain = if edge < fade {
                0.5 - 0.5 * (TAU / 2.0 * edge as f64 / fade as f64).cos()
            } else {
                1.0
            };
            let time = t as f64 / rate;
            let sum: f64 = (components.iter())
                .map(|(f, a, phase)| a * (TAU * f * time + phase).sin())
                .sum();
            gain * sum
        })
        .collect()
}

/// `samples` at twice their sampling frequency: every other sample is
/// interpolated with a Blackman-windowed sinc of 64 taps, which passes the
/// band below 3.2 kHz at the MPEG-2.5 frequencies within 0.01 dB.
fn upsample(samples: &[f64]) -> Vec<f64> {
    const HALF_TAPS: isize = 32;
    let taps: Vec<f64> = (-HALF_TAPS..HALF_TAPS)
        .map(|k| {
            let x = k as f64 + 0.5;
            let n = (k + HALF_TAPS) as f64 + 0.5;
            let window = 0.42 - 0.5 * (TAU * n / (2 * HALF_TAPS) as f64).cos()
                + 0.08 * (2.0 * TAU * n / (2 * HALF_TAPS) as f64).cos();
            (x * TAU / 2.0).sin() / (x * TAU / 2.0) * window
        })
        .collect();
    let at = |t: isize| {
        usize::try_from(t)
            .ok()
            .and_then(|t| samples.get(t))
            .copied()
    };
    (0..samples.len() as isize)
        .flat_map(|t| {
            let between = (-HALF_TAPS..HALF_TAPS)
                .zip(&taps)
                .map(|(k, tap)| at(t - k).unwrap_or(0.0) * tap)
                .sum();
            [samples[t as usize], between]
        })
        .collect()
}

/// How far the coding noise of `stream`, which codes `input` at an MPEG-2.5
/// sampling frequency, rises above the masking threshold that the model of
/// `judge`, the frequency twice as high, puts on the input, in its long
/// scalefactor bands below 3.2 kHz: the noise-to-mask ratio in dB, averaged
/// over every band of every steady granule, and the share of those bands in
/// which the noise exceeds the threshold.
fn noise_to_mask(
    input: &[f64],
    stream: &[u8],
    judge: SamplingFrequency,
    input_at_judge: &[f64],
) -> (f64, f64) {
    use chelys::mp3::filterbank::Filterbank;
    use chelys::mp3::psychoacoustic::PsychoacousticModel;

    let decoded = chelys_conformance::ffmpeg::decode_mp3(stream);
    assert_eq!(decoded.diagnostics, "");
    let lag = (0..3000)
        .max_by(|&a, &b| {
            let error = |lag: usize| -> f64 {
                (input.iter().enumerate())
                    .map(|(t, x)| {
                        let y = decoded.samples.get(t + lag).map_or(0.0, |&y| f64::from(y));
                        (x - y).powi(2)
                    })
                    .sum()
            };
            error(b).total_cmp(&error(a))
        })
        .unwrap();
    let error: Vec<f64> = (input.iter().enumerate())
        .map(|(t, x)| decoded.samples.get(t + lag).map_or(0.0, |&y| f64::from(y)) - x)
        .collect();
    let error = upsample(&error);

    let granule = |samples: &[f64], g: usize| -> [f64; GRANULE_LINES] {
        std::array::from_fn(|t| samples.get(g * GRANULE_LINES + t).copied().unwrap_or(0.0))
    };
    let mut model = PsychoacousticModel::new(judge);
    let (mut signal_bank, mut error_bank) = (Filterbank::new(), Filterbank::new());
    model.granule(&granule(input_at_judge, 0));
    let bands = scalefactor_bands::long_bands(judge);
    let cutoff = 3200.0 / f64::from(judge.hz()) * 2.0 * GRANULE_LINES as f64;
    let granules = input_at_judge.len() / GRANULE_LINES;
    let mut nmr = Vec::new();
    for g in 0..granules - 1 {
        let masking = model.granule(&granule(input_at_judge, g + 1));
        let xr = signal_bank.granule(&granule(input_at_judge, g), BlockType::Normal);
        let er = error_bank.granule(&granule(&error, g), BlockType::Normal);
        if g < 4 || g + 4 > granules {
            continue;
        }
        let xmin = masking.allowed_distortion(&xr, judge);
        for (band, xmin) in bands.iter().zip(xmin) {
            if band.end() as f64 > cutoff || xmin == 0.0 {
                continue;
            }
            let noise = er[band.start..band.end()]
                .iter()
                .map(|e| e * e)
                .sum::<f64>()
                / band.width as f64;
            nmr.push(10.0 * (noise / xmin).log10());
        }
    }
    let mean = nmr.iter().sum::<f64>() / nmr.len() as f64;
    let audible = nmr.iter().filter(|&&db| db > 0.0).count() as f64 / nmr.len() as f64;
    (mean, audible)
}

/// The coding noise of each MPEG-2.5 sampling frequency, at 16 and 24
/// kbit/s, stays below the threshold that the model of the frequency twice
/// as high puts on the input in most bands below 3.2 kHz. With the tables
/// derived from that frequency's, the noise exceeds it in 37%, 52% and 48%
/// of those bands at 16 kbit/s, and in 19%, 24% and 23% at 24 kbit/s, at
/// 8, 11.025 and 12 kHz. The tables of 16 kHz, which MPEG-2.5 used before,
/// gave 35%, 60% and 52%, and 18%, 23% and 22%: by this measure the two
/// are not far apart.
#[test]
fn mpeg25_noise_to_mask_ratio() {
    use SamplingFrequency::*;
    for (half, double) in [(Hz8000, Hz16000), (Hz11025, Hz22050), (Hz12000, Hz24000)] {
        let input = below_3_2_khz(half, 4.0);
        let input_at_judge = below_3_2_khz(double, 4.0);
        for (kbps, most_audible) in [(16, 0.55), (24, 0.27)] {
            let stream = encode_pcm(&config(half, Mode::SingleChannel, kbps), &[&input]);
            let (mean, audible) = noise_to_mask(&input, &stream, double, &input_at_judge);
            assert!(
                audible < most_audible,
                "{half:?} at {kbps} kbit/s: {:.1}% of bands audible, mean NMR {mean:.2} dB",
                100.0 * audible
            );
        }
    }
}
