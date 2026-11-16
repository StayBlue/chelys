// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! PCM through the analysis filterbank, decoded by ffmpeg and compared with
//! the input.

use std::f64::consts::TAU;

use chelys::mp3::header::{Bitrate, Flags, Mode, SamplingFrequency};
use chelys::mp3::rate::Rate;
use chelys::mp3::{BlockType, StreamConfig, encode_pcm_with_global_gain, encode_with_block_types};
use chelys_conformance::{DECODERS, ffmpeg};

fn config(sampling_frequency: SamplingFrequency, mode: Mode) -> StreamConfig {
    StreamConfig {
        // The highest: 320 kbit/s, or 160 at the lower sampling frequencies.
        rate: Rate::Constant(Bitrate::all(sampling_frequency.version()).last().unwrap()),
        sampling_frequency,
        mode,
        crc_protected: false,
        flags: Flags::default(),
    }
}

/// Tones in even and odd subbands, mid-band and near subband edges, where the
/// frequency inversion and the aliasing butterflies matter.
///
/// They fade in and out over 20 ms, outside the part [`align`] compares:
/// a granule where they start or stop abruptly takes several times the
/// bits of the others, more than a frame of one granule holds at the lower
/// sampling frequencies with a fixed quantizer and no bit reservoir.
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
            let sum: f64 = frequencies
                .iter()
                .enumerate()
                .map(|(i, f)| 0.08 * (TAU * f * t as f64 / fs + offset * i as f64).sin())
                .sum();
            gain * sum
        })
        .collect()
}

/// The lag of `output` behind `input` that best lines them up, and the
/// signal-to-noise ratio in dB at that lag over the steady middle part.
/// `output` is interleaved over `channels`, of which `input` is `channel`.
fn align(input: &[f64], output: &[f32], channels: usize, channel: usize) -> (usize, f64) {
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
    (0..3000)
        .map(|lag| (lag, snr(lag)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
}

/// With a fine quantizer the decoded output matches the input closely, which
/// needs the filterbank, the encoder's frequency inversion and aliasing
/// butterflies, and the scaling to the decoder's `xr` all to be right.
/// (Leaving out the inversion drops this to about 1 dB, the butterflies to
/// about 10 dB, and doubling the scaling to 6 dB.)
///
/// This is the evidence for two gaps in the standard that the filterbank
/// marks as settled by decoding with ffmpeg: the gain of the polyphase
/// filters and the encoder's frequency inversion.
#[test]
fn pcm_round_trips_at_every_sampling_frequency() {
    for fs in SamplingFrequency::STANDARD {
        let input = tones(fs, 1.0, 0.0);
        let stream = encode_pcm_with_global_gain(&config(fs, Mode::SingleChannel), 150, &[&input]);
        let decoded = ffmpeg::decode_mp3(&stream);
        assert_eq!(decoded.diagnostics, "");
        let (lag, snr) = align(&input, &decoded.samples, 1, 0);
        assert!(snr > 50.0, "{fs:?}: {snr:.1} dB at lag {lag}");
    }
}

/// Short blocks round trip nearly as well as long ones (with a slightly
/// coarser quantizer, so that the extra tone fits), which needs the short
/// windows and MDCT, their scaling, the reordering into scalefactor band
/// order (above the last band too), and the default Huffman regions of
/// switched windows all to agree with the decoder. Every block type and every change between them that
/// figure C.7 allows is covered; so is a stream of short blocks only, which
/// starts with one. (Interleaving the lines above the last short band
/// across the windows drops this to between 2 and 12 dB.)
///
/// This is the evidence for two gaps in the standard that the scalefactor
/// bands mark as settled by decoding with ffmpeg: the order of the lines
/// above the last short scalefactor band, and at the lower sampling
/// frequencies the fixed scalefactor of the last band of Table B.2 of
/// ISO/IEC 13818-3.
#[test]
fn short_blocks_round_trip_at_every_sampling_frequency() {
    use BlockType::*;
    const CYCLE: [BlockType; 13] = [
        Normal, Normal, Start, Short, Short, Stop, Normal, Start, Short, Stop, Start, Short, Stop,
    ];
    for fs in SamplingFrequency::STANDARD {
        // A tone above the last short scalefactor band at every sampling
        // frequency (line 182 of 192, above band 11's end at 180 or below),
        // in the lines that have no scalefactor.
        let rate = f64::from(fs.hz());
        let input: Vec<f64> = tones(fs, 1.0, 0.0)
            .iter()
            .enumerate()
            .map(|(t, x)| x + 0.08 * (TAU * 0.95 * rate / 2.0 * t as f64 / rate).sin())
            .collect();
        let config = config(fs, Mode::SingleChannel);
        for (label, pattern) in [
            (
                "every transition",
                &(|_, g: usize| CYCLE[g % CYCLE.len()]) as &dyn Fn(usize, usize) -> BlockType,
            ),
            ("short only", &|_, _| Short),
        ] {
            let stream = encode_with_block_types(&config, 154, &[&input], pattern);
            let decoded = ffmpeg::decode_mp3(&stream);
            assert_eq!(decoded.diagnostics, "");
            let (lag, snr) = align(&input, &decoded.samples, 1, 0);
            assert!(snr > 45.0, "{fs:?} {label}: {snr:.1} dB at lag {lag}");
        }
    }
}

/// Mixed blocks round trip as well as short ones, through ffmpeg, mpg123
/// and LAME, at every sampling frequency at which they are used: in a
/// stream that changes block type as figure C.7 allows, with mixed blocks
/// in place of short ones, and in a stream of mixed blocks only. The tones
/// lie in the middle of subbands 0 and 1, which a mixed block transforms
/// as long blocks, near the edge between them, where their aliasing
/// butterfly works, and near the edge with subband 2, the first short one.
///
/// This is the evidence for three gaps in the standard that the encoder
/// marks as settled by decoding with ffmpeg, mpg123 and LAME. Without the
/// butterfly between subbands 0 and 1, every decoder decodes these streams
/// at 12 to 17 dB. With the decoder's windows in place of the analysis
/// windows matched to them, the stream that changes block type decodes at
/// about 16 dB. With region 1 of a mixed block beginning at line 54, the
/// end of 8 long bands, rather than at line 36, all three decode at 2 to 8
/// dB at 16, 22.05 and 24 kHz; and at MPEG-2.5's 11.025 and 12 kHz, with
/// different tables in the two regions, mpg123 and LAME decode at under 4
/// dB with it at line 36, and ffmpeg at under 10 dB with it at line 54.
#[test]
fn mixed_blocks_round_trip() {
    use BlockType::*;
    const CYCLE: [BlockType; 13] = [
        Normal, Normal, Start, Mixed, Mixed, Stop, Normal, Start, Mixed, Stop, Start, Mixed, Stop,
    ];
    for fs in SamplingFrequency::ALL {
        if fs == SamplingFrequency::Hz8000 {
            continue;
        }
        let rate = f64::from(fs.hz());
        let band = rate / 64.0;
        let input: Vec<f64> = tones(fs, 1.0, 0.0)
            .iter()
            .enumerate()
            .map(|(t, x)| {
                let t = t as f64 / rate;
                x + [0.97, 1.03, 1.97]
                    .iter()
                    .map(|f| 0.08 * (TAU * f * band * t).sin())
                    .sum::<f64>()
            })
            .collect();
        let config = config(fs, Mode::SingleChannel);
        for (label, pattern) in [
            (
                "every transition",
                &(|_, g: usize| CYCLE[g % CYCLE.len()]) as &dyn Fn(usize, usize) -> BlockType,
            ),
            ("mixed only", &|_, _| Mixed),
        ] {
            let stream = encode_with_block_types(&config, 154, &[&input], pattern);
            for decoder in DECODERS {
                let decoded = (decoder.decode)(&stream);
                assert_eq!(decoded.diagnostics, "", "{}", decoder.name);
                let (lag, snr) = align(&input, &decoded.samples, 1, 0);
                assert!(
                    snr > 45.0,
                    "{}, {fs:?} {label}: {snr:.1} dB at lag {lag}",
                    decoder.name
                );
            }
        }
    }
}

/// A slightly coarser quantizer than the mono test, so that two channels fit
/// in a frame without rate control or a bit reservoir.
#[test]
fn stereo_channels_round_trip_separately() {
    let fs = SamplingFrequency::Hz32000;
    let left = tones(fs, 1.0, 0.0);
    let right: Vec<f64> = tones(fs, 1.0, 1.7).iter().map(|x| -0.7 * x).collect();
    let stream = encode_pcm_with_global_gain(&config(fs, Mode::Stereo), 160, &[&left, &right]);
    let decoded = ffmpeg::decode_mp3(&stream);
    assert_eq!(decoded.diagnostics, "");
    for (channel, input) in [&left, &right].into_iter().enumerate() {
        let (lag, snr) = align(input, &decoded.samples, 2, channel);
        assert!(snr > 40.0, "channel {channel}: {snr:.1} dB at lag {lag}");
    }
}

/// In MS stereo the middle and side channels are sent in place of the left
/// and right, and the decoder's matrix (clause 2.4.3.4.9.2) restores them:
/// with long blocks, and with short or mixed ones in both channels.
#[test]
fn ms_stereo_round_trips() {
    let fs = SamplingFrequency::Hz32000;
    let left = tones(fs, 1.0, 0.0);
    let right: Vec<f64> = tones(fs, 1.0, 1.7).iter().map(|x| -0.7 * x).collect();
    let mode = Mode::JointStereo {
        intensity_stereo: false,
        ms_stereo: true,
    };
    let config = config(fs, mode);
    for (label, block_type) in [
        ("long blocks", BlockType::Normal),
        ("short blocks", BlockType::Short),
        ("mixed blocks", BlockType::Mixed),
    ] {
        let stream = encode_with_block_types(&config, 160, &[&left, &right], |_, _| block_type);
        let decoded = ffmpeg::decode_mp3(&stream);
        assert_eq!(decoded.diagnostics, "");
        for (channel, input) in [&left, &right].into_iter().enumerate() {
            let (lag, snr) = align(input, &decoded.samples, 2, channel);
            assert!(
                snr > 40.0,
                "{label}, channel {channel}: {snr:.1} dB at lag {lag}"
            );
        }
    }
}

/// Each step of `global_gain` scales the quantizer step by 2^(1/4), so the
/// noise grows steadily as the gain rises.
#[test]
fn noise_follows_the_quantizer_step() {
    let fs = SamplingFrequency::Hz44100;
    let input = tones(fs, 1.0, 0.0);
    let snrs: Vec<f64> = [150, 160, 170, 180]
        .into_iter()
        .map(|gain| {
            let stream =
                encode_pcm_with_global_gain(&config(fs, Mode::SingleChannel), gain, &[&input]);
            align(&input, &ffmpeg::decode_mp3(&stream).samples, 1, 0).1
        })
        .collect();
    for pair in snrs.windows(2) {
        let drop = pair[0] - pair[1];
        assert!((7.0..14.0).contains(&drop), "SNR by gain: {snrs:?}");
    }
}
