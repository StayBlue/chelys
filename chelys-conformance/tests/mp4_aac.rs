// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! AAC-LC in MP4 files, as ffmpeg reads, decodes and seeks in them.

use std::f64::consts::TAU;

use chelys::aac::{Config, DELAY, Rate, SAMPLE_RATES};
use chelys::mp4::aac::Encoder;
use chelys_conformance::ffmpeg;

/// A sweep from 100 Hz up, `len` samples at `sample_rate`, in each of
/// `channels`, interleaved, each channel's a little higher and quieter
/// than the one before: a signal that lines up with itself at one lag
/// only.
fn sweeps(sample_rate: u32, channels: usize, len: usize) -> Vec<f32> {
    let fs = f64::from(sample_rate);
    let top = (fs / 2.0).min(12_000.0);
    (0..len)
        .flat_map(|t| {
            (0..channels).map(move |c| {
                let from = 100.0 * (1.0 + 0.2 * c as f64);
                let rise = (top - from) / (2.0 * len as f64);
                let phase = TAU * (from + rise * t as f64) * t as f64 / fs;
                (0.4 / (1.0 + 0.1 * c as f64) * phase.sin()) as f32
            })
        })
        .collect()
}

fn config(sample_rate: u32, channels: usize, bits_per_second: u32) -> Config {
    Config::new(sample_rate, channels, Rate::Constant { bits_per_second })
}

/// The MP4 file of `pcm`, and the frames of its stream.
fn encode(config: Config, pcm: &[f32]) -> (Vec<u8>, u64) {
    let mut encoder = Encoder::new(config).unwrap();
    encoder.encode_interleaved(pcm);
    let mut file = Vec::new();
    let finished = encoder.finish(&mut file).unwrap();
    (file, finished.frames)
}

/// A bitrate the sample rate and channels allow, near 1.5 bits a sample.
fn bitrate(sample_rate: u32, channels: usize) -> u32 {
    let ncc = if channels >= 6 {
        channels - 1
    } else {
        channels
    } as u32;
    3 * sample_rate / 2 * ncc
}

/// The lag, from `-max` to `max`, at which `output` lines up best with
/// `input`, channel 0 of `channels` interleaved: the square error of
/// `output[t + lag]` against `input[t]` is least, over up to 8192 samples
/// `t` that both have at every lag. A negative lag is an output that
/// starts late in the input.
fn lag(input: &[f32], output: &[f32], channels: usize, max: usize) -> isize {
    let channel = |pcm: &[f32]| -> Vec<f32> { pcm.iter().step_by(channels).copied().collect() };
    let (input, output) = (channel(input), channel(output));
    let span = max..input
        .len()
        .min(output.len())
        .saturating_sub(max)
        .min(max + 8192);
    assert!(!span.is_empty(), "too short to search lags of {max}");
    let error = |lag: isize| -> f64 {
        (span.clone())
            .map(|t| f64::from(input[t] - output[t.wrapping_add_signed(lag)]).powi(2))
            .sum()
    };
    let max = max as isize;
    (-max..=max)
        .min_by(|&a, &b| error(a).total_cmp(&error(b)))
        .unwrap()
}

/// Guards the helper: it finds a shift either way, and none where there
/// is none.
#[test]
fn lag_finds_an_output_early_or_late() {
    let channels = 2;
    let pcm = sweeps(8_000, channels, 8_000);
    for shift in [-1024isize, -1, 0, 1, 1024] {
        let output: Vec<f32> = if shift >= 0 {
            [vec![0.0; shift as usize * channels], pcm.clone()].concat()
        } else {
            pcm[shift.unsigned_abs() * channels..].to_vec()
        };
        assert_eq!(lag(&pcm, &output, channels, DELAY + 64), shift);
    }
}

/// ffmpeg reads the files at every sample rate, in mono and stereo, as
/// AAC LC of the input's channels and rate, and decodes each to exactly
/// the input's length, the input in place, early or late by no sample: the
/// edit list leaves out the [`DELAY`] samples before the input and the
/// padding after it, sample for sample what ffmpeg decodes from the frames
/// with the edit list ignored. At 88.2 and 96 kHz, whose `samplerate`
/// field cannot hold them, the file also gives them in a
/// `SamplingRateBox`.
#[test]
fn ffmpeg_decodes_mp4_files_to_the_input_at_every_sample_rate() {
    for sample_rate in SAMPLE_RATES {
        for channels in [1, 2] {
            let context = format!("{sample_rate} Hz, {channels} channels");
            let len = sample_rate as usize + 123;
            let pcm = sweeps(sample_rate, channels, len);
            let config = config(sample_rate, channels, bitrate(sample_rate, channels));
            let (file, frames) = encode(config, &pcm);
            let stream = ffmpeg::probe_mp4(&file);
            assert_eq!(stream.info.codec_name, "aac", "{context}");
            assert_eq!(stream.profile, "LC", "{context}");
            assert_eq!(stream.info.sample_rate, sample_rate, "{context}");
            assert_eq!(stream.info.channels, channels, "{context}");
            assert_eq!(stream.time_base, format!("1/{sample_rate}"), "{context}");
            assert_eq!(stream.duration_ts, len as u64, "{context}");
            let decoded = ffmpeg::decode_mp4(&file);
            assert_eq!(decoded.diagnostics, "", "{context}");
            assert_eq!(decoded.samples.len(), pcm.len(), "{context}");
            assert_eq!(
                lag(&pcm, &decoded.samples, channels, DELAY + 64),
                0,
                "{context}"
            );
            let all = ffmpeg::decode_mp4_ignoring_edits(&file);
            assert_eq!(all.diagnostics, "", "{context}");
            assert_eq!(
                all.samples.len(),
                frames as usize * 1024 * channels,
                "{context}"
            );
            let from = DELAY * channels;
            assert!(
                decoded.samples == all.samples[from..from + pcm.len()],
                "{context}"
            );
        }
    }
}

/// The edit list is all that ffmpeg trims: with it ignored, ffmpeg
/// decodes every sample of every frame, and with it applied, those from
/// the [`DELAY`]'th on, as many as the input has, sample for sample.
#[test]
fn ffmpeg_trims_the_delay_and_padding_by_the_edit_list() {
    for (sample_rate, channels) in [(44_100, 2), (48_000, 1), (8_000, 2)] {
        let context = format!("{sample_rate} Hz, {channels} channels");
        let len = 2 * sample_rate as usize + 500;
        let pcm = sweeps(sample_rate, channels, len);
        let config = config(sample_rate, channels, bitrate(sample_rate, channels));
        let (file, frames) = encode(config, &pcm);
        let all = ffmpeg::decode_mp4_ignoring_edits(&file);
        assert_eq!(all.diagnostics, "", "{context}");
        assert_eq!(
            all.samples.len(),
            frames as usize * 1024 * channels,
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

/// ffmpeg reads the files of every channel configuration, in the layout
/// it gives the same configuration in ADTS, and decodes each to the
/// input's length; with perceptual noise substitution too.
#[test]
fn ffmpeg_decodes_mp4_files_of_every_channel_configuration() {
    let sample_rate = 48_000;
    for channels in [3, 4, 5, 6, 8] {
        for noise in [false, true] {
            let context = format!("{channels} channels, noise substitution {noise}");
            let len = sample_rate as usize / 2 + 7;
            let pcm = sweeps(sample_rate, channels, len);
            let bits_per_second = bitrate(sample_rate, channels) * 2 / 3;
            let config = config(sample_rate, channels, bits_per_second).noise_substitution(noise);
            let (file, _) = encode(config, &pcm);
            let mut adts = Vec::new();
            let mut encoder = chelys::aac::Encoder::new(config).unwrap();
            encoder.encode_interleaved(&pcm, &mut adts);
            encoder.finish(&mut adts);
            let (layout, _) = ffmpeg::channel_layout_aac(&adts);
            let stream = ffmpeg::probe_mp4(&file);
            assert_eq!(stream.info.channels, channels, "{context}");
            assert_eq!(stream.channel_layout, layout, "{context}");
            assert_eq!(stream.duration_ts, len as u64, "{context}");
            let decoded = ffmpeg::decode_mp4(&file);
            assert_eq!(decoded.diagnostics, "", "{context}");
            assert_eq!(decoded.samples.len(), pcm.len(), "{context}");
        }
    }
}

/// ffmpeg n9.0.2 seeks in the files to the start of the first frame's
/// samples at or after the time asked for, and decodes from there exactly
/// as it decodes the whole file: it decodes the frame before, whose
/// samples the first overlaps, though it presents none of them.
#[test]
fn ffmpeg_seeks_in_mp4_files_to_a_frame_and_decodes_from_it_exactly() {
    let (sample_rate, channels) = (44_100, 2);
    let len = 3 * sample_rate as usize;
    let pcm = sweeps(sample_rate, channels, len);
    let (file, _) = encode(config(sample_rate, channels, 128_000), &pcm);
    let whole = ffmpeg::decode_mp4(&file).samples;
    for from in [0.5, 1.0, 1.37, 2.9] {
        let seeked = ffmpeg::decode_mp4_seeking(&file, from);
        assert_eq!(seeked.diagnostics, "", "{from} s");
        let at = (from * f64::from(sample_rate)).round() as usize;
        let start = len - seeked.samples.len() / channels;
        assert!((at..at + 1024).contains(&start), "{from} s: from {start}");
        assert!(
            seeked.samples == whole[start * channels..],
            "{from} s: from {start}"
        );
    }
}

/// ffmpeg decodes a file of no samples to none, and files a sample or so
/// either side of a frame's length to exactly their samples.
#[test]
fn ffmpeg_reads_empty_and_short_mp4_files() {
    let (sample_rate, channels) = (48_000, 1);
    for len in [0, 1, 2, 1023, 1024, 1025, 2047, 2048] {
        let pcm = sweeps(sample_rate, channels, len);
        let (file, frames) = encode(config(sample_rate, channels, 64_000), &pcm);
        assert_eq!(frames as usize, (len + DELAY).div_ceil(1024), "{len}");
        assert_eq!(ffmpeg::probe_mp4(&file).duration_ts, len as u64, "{len}");
        let decoded = ffmpeg::decode_mp4(&file);
        assert_eq!(decoded.diagnostics, "", "{len}");
        assert_eq!(decoded.samples.len(), len, "{len}");
    }
}
