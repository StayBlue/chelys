// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! FLAC in MP4 files, of frames that the `flac` program made, as ffmpeg
//! reads and decodes them.

use std::f64::consts::TAU;

use chelys::mp4::{FragmentInfo, FragmentOptions, FragmentedWriter, Writer, flac};
use chelys_conformance::{ffmpeg, flac as flac_encoder, gstreamer};

/// A sweep from 100 Hz up, as `bits`-bit integers, `len` samples in each
/// of `channels`, interleaved, each channel's a little higher than the one
/// before.
fn sweeps(sample_rate: u32, channels: usize, len: usize, bits: u32) -> Vec<i32> {
    let fs = f64::from(sample_rate);
    let peak = f64::from((1u32 << (bits - 1)) - 1);
    (0..len)
        .flat_map(|t| {
            (0..channels).map(move |c| {
                let from = 100.0 * (1.0 + 0.2 * c as f64);
                let rise = (fs.min(48_000.0) / 4.0 - from) / (2.0 * len as f64);
                let phase = TAU * (from + rise * t as f64) * t as f64 / fs;
                (0.5 * peak * phase.sin()).round() as i32
            })
        })
        .collect()
}

/// The FLAC frames of `pcm`, as `flac` makes them and ffprobe splits them,
/// and the configuration of their stream.
fn frames(
    pcm: &[i32],
    sample_rate: u32,
    channels: usize,
    bits: u32,
    block_size: u32,
) -> (flac::Config, Vec<Vec<u8>>) {
    let file = flac_encoder::encode(pcm, sample_rate, channels, bits, block_size);
    let stream = ffmpeg::packets(&file, "flac");
    let config = flac::Config::from_streaminfo(&stream.config).unwrap();
    (config, stream.packets)
}

/// The MP4 file of `frames`, written whole.
fn whole(config: &flac::Config, frames: &[Vec<u8>], length: Option<u64>) -> Vec<u8> {
    let mut writer = Writer::new(config.clone());
    for frame in frames {
        writer.push(frame).unwrap();
    }
    let mut file = Vec::new();
    writer.finish(length, &mut file).unwrap();
    file
}

/// The fragmented MP4 file of `frames`, of fragments of `fragment_samples`.
fn fragmented(
    config: &flac::Config,
    frames: &[Vec<u8>],
    options: FragmentOptions,
) -> (Vec<u8>, usize) {
    let mut writer = FragmentedWriter::new(config.clone(), options);
    let mut file = writer.init_segment().to_vec();
    let mut fragments = 0;
    let mut out = |fragment: &[u8], _: FragmentInfo| {
        file.extend_from_slice(fragment);
        fragments += 1;
    };
    for frame in frames {
        writer.push(frame, &mut out).unwrap();
    }
    writer.finish(&mut out).unwrap();
    (file, fragments)
}

/// `pcm` as ffmpeg's decoder gives it, as floats: it divides each sample
/// by a power of two, which is exact at 24 bits or fewer.
fn as_decoded(pcm: &[i32], bits: u32) -> Vec<f32> {
    let scale = (1u32 << (bits - 1)) as f32;
    pcm.iter().map(|&s| s as f32 / scale).collect()
}

/// ffmpeg decodes the files to the input exactly: in 16 and 24 bits, in
/// mono, stereo and 5.1, at rates from 8 to 192 kHz, those above 65535 Hz
/// among them, whose `samplerate` field holds them halved, in blocks of a
/// common size and of an uncommon one, whose size follows the frame's
/// coded number, and with a last frame shorter than the others. ffprobe
/// gives the stream its rate, channels and length.
#[test]
fn ffmpeg_decodes_flac_in_mp4_files_to_the_input_exactly() {
    for (sample_rate, bits) in [
        (8_000, 16),
        (44_100, 16),
        (48_000, 24),
        (96_000, 24),
        (176_400, 24),
        (192_000, 24),
    ] {
        for channels in [1, 2, 6] {
            for block_size in [4096, 1000] {
                let context = format!(
                    "{sample_rate} Hz, {bits} bits, {channels} channels, blocks of {block_size}"
                );
                let len = sample_rate as usize / 4 + 123;
                let pcm = sweeps(sample_rate, channels, len, bits);
                let (config, frames) = frames(&pcm, sample_rate, channels, bits, block_size);
                assert_eq!(frames.len(), len.div_ceil(block_size as usize), "{context}");
                let file = whole(&config, &frames, Some(len as u64));
                let stream = ffmpeg::probe_mp4(&file);
                assert_eq!(stream.info.codec_name, "flac", "{context}");
                assert_eq!(stream.info.sample_rate, sample_rate, "{context}");
                assert_eq!(stream.info.channels, channels, "{context}");
                assert_eq!(stream.time_base, format!("1/{sample_rate}"), "{context}");
                assert_eq!(stream.duration_ts, len as u64, "{context}");
                let decoded = ffmpeg::decode_mp4(&file);
                assert_eq!(decoded.diagnostics, "", "{context}");
                assert!(decoded.samples == as_decoded(&pcm, bits), "{context}");
            }
        }
    }
}

/// The fragmented files decode to the input exactly too, with the length
/// declared or not, in fragments of a second and of a frame.
#[test]
fn ffmpeg_decodes_fragmented_flac_in_mp4_files_to_the_input_exactly() {
    for (sample_rate, channels) in [(44_100, 2), (96_000, 1)] {
        let len = 3 * sample_rate as usize + 500;
        let pcm = sweeps(sample_rate, channels, len, 24);
        let (config, frames) = frames(&pcm, sample_rate, channels, 24, 4096);
        for length in [None, Some(len as u64)] {
            for fragment_samples in [None, Some(4096)] {
                let context = format!("{sample_rate} Hz, {length:?}, {fragment_samples:?}");
                let mut options = FragmentOptions::default();
                if let Some(length) = length {
                    options = options.length(length);
                }
                if let Some(samples) = fragment_samples {
                    options = options.fragment_samples(std::num::NonZeroU32::new(samples).unwrap());
                }
                let (file, fragments) = fragmented(&config, &frames, options);
                let expected = match fragment_samples {
                    None => 4,
                    Some(_) => frames.len(),
                };
                assert_eq!(fragments, expected, "{context}");
                let decoded = ffmpeg::decode_mp4(&file);
                assert_eq!(decoded.diagnostics, "", "{context}");
                assert!(decoded.samples == as_decoded(&pcm, 24), "{context}");
            }
        }
    }
}

/// GStreamer, through `qtdemux` and `flacdec`, decodes the files written
/// whole and the fragmented ones to the input exactly too, at rates above
/// 65535 Hz among them, as ffmpeg does, both taking the rate from the
/// STREAMINFO, not the `samplerate` field, which holds it halved. Of a
/// fragmented file of a declared length, the segment `qtdemux` marks to
/// the `MovieExtendsHeaderBox`'s duration can end a nanosecond short of
/// the last sample, which it then leaves out, as it can of AAC (see the
/// `gstreamer` module).
#[test]
fn gstreamer_decodes_flac_in_mp4_files_to_the_input_exactly() {
    for (sample_rate, channels) in [(44_100, 2), (96_000, 6), (192_000, 1)] {
        let len = sample_rate as usize + 777;
        let pcm = sweeps(sample_rate, channels, len, 24);
        let (config, frames) = frames(&pcm, sample_rate, channels, 24, 4096);
        let expected = as_decoded(&pcm, 24);
        let files = [
            whole(&config, &frames, None),
            fragmented(&config, &frames, FragmentOptions::default()).0,
            fragmented(
                &config,
                &frames,
                FragmentOptions::default().length(len as u64),
            )
            .0,
        ];
        for (i, file) in files.iter().enumerate() {
            let context = format!("{sample_rate} Hz, {channels} channels, file {i}");
            let decoded = gstreamer::decode_flac_mp4(file);
            assert_eq!(decoded.diagnostics, "", "{context}");
            let short = expected.len() - decoded.samples.len();
            let declared = i == 2;
            assert!(
                short == 0 || (declared && short == channels),
                "{context}: {short}"
            );
            assert!(expected.starts_with(&decoded.samples), "{context}");
        }
    }
}
