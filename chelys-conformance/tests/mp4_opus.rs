// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Opus in MP4 files, of packets that ffmpeg's libopus encoder made, as
//! ffmpeg and GStreamer read and decode them, compared with the Ogg Opus
//! files the packets came in, which ffmpeg presents from the end of the
//! pre-skip to the end of the input.

use std::f64::consts::TAU;

use chelys::mp4::{FragmentInfo, FragmentOptions, FragmentedWriter, Writer, opus};
use chelys_conformance::{Decoded, ffmpeg, gstreamer};

/// A sweep from 100 Hz up, `len` samples at 48 kHz in each of
/// `channels`, interleaved, each channel's a little higher and quieter
/// than the one before.
fn sweeps(channels: usize, len: usize) -> Vec<f32> {
    let fs = 48_000.0;
    (0..len)
        .flat_map(|t| {
            (0..channels).map(move |c| {
                let from = 100.0 * (1.0 + 0.2 * c as f64);
                let rise = (12_000.0 - from) / (2.0 * len as f64);
                let phase = TAU * (from + rise * t as f64) * t as f64 / fs;
                (0.4 / (1.0 + 0.1 * c as f64) * phase.sin()) as f32
            })
        })
        .collect()
}

/// A stream of `len` samples of `channels`: its Ogg Opus file as ffmpeg's
/// libopus encoder makes it, decoded as ffmpeg presents it, and its packets
/// as ffprobe splits them, with the configuration of their stream.
struct Stream {
    reference: Vec<f32>,
    config: opus::Config,
    packets: Vec<Vec<u8>>,
}

impl Stream {
    fn new(channels: usize, len: usize) -> Self {
        let pcm = sweeps(channels, len);
        let ogg = ffmpeg::encode_opus(&pcm, channels, 64 * channels as u32);
        let reference = ffmpeg::decode_file(&ogg, "opus");
        assert_eq!(reference.diagnostics, "");
        assert_eq!(reference.samples.len(), pcm.len());
        let packets = ffmpeg::packets(&ogg, "opus");
        Self {
            reference: reference.samples,
            config: opus::Config::from_id_header(&packets.config).unwrap(),
            packets: packets.packets,
        }
    }

    /// The MP4 file of the packets, written whole, of `length` samples.
    fn whole(&self, length: Option<u64>) -> Vec<u8> {
        let mut writer = Writer::new(self.config.clone());
        for packet in &self.packets {
            writer.push(packet).unwrap();
        }
        let mut file = Vec::new();
        writer.finish(length, &mut file).unwrap();
        file
    }

    /// The fragmented MP4 file of the packets.
    fn fragmented(&self, options: FragmentOptions) -> Vec<u8> {
        let mut writer = FragmentedWriter::new(self.config.clone(), options);
        let mut file = writer.init_segment().to_vec();
        let mut out = |fragment: &[u8], _: FragmentInfo| file.extend_from_slice(fragment);
        for packet in &self.packets {
            writer.push(packet, &mut out).unwrap();
        }
        writer.finish(&mut out).unwrap();
        file
    }
}

/// The lengths of the tests, in samples at 48 kHz: from a sample, within
/// the pre-skip of 312, either side of the first packet's end after it and
/// of a packet's length, to a few seconds.
const LENGTHS: [usize; 9] = [1, 312, 313, 648, 649, 959, 961, 48_000 + 123, 3 * 48_000];

/// Whether GStreamer's output is ffmpeg's `reference`, as near as their
/// decoders come: of the same length, and each sample within 2^-11, where
/// libopus, which `opusdec` wraps and which decodes to 16 bits, and
/// ffmpeg's decoder differ by as much as 2^-12. A shift of a sample moves
/// the sweeps by ten times that and more, as at their start, of 100 Hz.
fn as_near(decoded: &[f32], reference: &[f32]) -> bool {
    decoded.len() == reference.len()
        && (decoded.iter().zip(reference)).all(|(a, b)| (a - b).abs() <= 1.0 / 2048.0)
}

/// Guards [`as_near`]: the sweeps a sample early or late are not near
/// them, at any length of the tests.
#[test]
fn as_near_tells_a_shift_of_a_sample() {
    for channels in [1, 2, 6] {
        for len in LENGTHS.into_iter().filter(|&len| len > 1) {
            let pcm = sweeps(channels, len);
            let early = [&pcm[channels..], &vec![0.0; channels]].concat();
            let late = [&vec![0.0; channels], &pcm[..pcm.len() - channels]].concat();
            assert!(as_near(&pcm, &pcm));
            assert!(!as_near(&early, &pcm), "{channels} channels, {len}");
            assert!(!as_near(&late, &pcm), "{channels} channels, {len}");
        }
    }
}

fn decoded_ok(decoded: Decoded, context: &str) -> Vec<f32> {
    assert_eq!(decoded.diagnostics, "", "{context}");
    decoded.samples
}

/// GStreamer, through `qtdemux` and `opusdec`, decodes the files written
/// whole to what ffmpeg decodes from the Ogg Opus files, as near as their
/// decoders come (see [`as_near`]): the input's length, from the end of
/// the pre-skip, which the edit list leaves out, and which
/// is not left out again for the `OpusSpecificBox`'s PreSkip, to the
/// input's end, where the edit ends. In mono and stereo, of channel
/// mapping family 0, and in 5.1, of family 1, whose samples hold a packet
/// of each of four streams.
#[test]
fn gstreamer_decodes_opus_in_mp4_files_as_the_ogg_opus_files() {
    for channels in [1, 2, 6] {
        for len in LENGTHS {
            let context = format!("{channels} channels, {len} samples");
            let stream = Stream::new(channels, len);
            let file = stream.whole(Some(len as u64));
            let decoded = decoded_ok(gstreamer::decode_opus_mp4(&file), &context);
            assert!(as_near(&decoded, &stream.reference), "{context}");
        }
    }
}

/// ffmpeg n9.0.2 reads the files as Opus at 48 kHz of the input's
/// channels, and decodes them to exactly what it decodes from the Ogg Opus
/// files, but for an input no longer than the
/// pre-skip, of which it presents nothing.
#[test]
fn ffmpeg_decodes_opus_in_mp4_files_as_the_ogg_opus_files() {
    for channels in [1, 2, 6] {
        for len in LENGTHS {
            let context = format!("{channels} channels, {len} samples");
            let stream = Stream::new(channels, len);
            let file = stream.whole(Some(len as u64));
            let probed = ffmpeg::probe_mp4(&file);
            assert_eq!(probed.info.codec_name, "opus", "{context}");
            assert_eq!(probed.info.sample_rate, 48_000, "{context}");
            assert_eq!(probed.info.channels, channels, "{context}");
            assert_eq!(probed.time_base, "1/48000", "{context}");
            let decoded = decoded_ok(ffmpeg::decode_mp4(&file), &context);
            if len <= usize::from(stream.config.pre_skip()) {
                assert!(decoded.is_empty(), "{context}");
            } else {
                assert!(decoded == stream.reference, "{context}");
            }
        }
    }
}

/// Every sample the packets decode to after the pre-skip, the padding
/// included, counted from the packets' TOC bytes and frame counts as
/// RFC 6716 codes them, independently of the writer.
fn after_pre_skip(stream: &Stream) -> usize {
    let decodes_to: usize = stream.packets.iter().map(|p| opus_duration(p)).sum();
    decodes_to - usize::from(stream.config.pre_skip())
}

/// The duration of an Opus packet, from its TOC byte and frame count.
fn opus_duration(packet: &[u8]) -> usize {
    let config = packet[0] >> 3;
    let frame = match config {
        0..=11 => [480, 960, 1920, 2880][usize::from(config % 4)],
        12..=15 => [480, 960][usize::from(config % 2)],
        _ => [120, 240, 480, 960][usize::from(config % 4)],
    };
    let frames = match packet[0] & 3 {
        0 => 1,
        1 | 2 => 2,
        _ => usize::from(packet[1] & 0x3f),
    };
    frame * frames
}

/// The fragmented files of a declared length present the input alone to
/// GStreamer, as the files written whole do, though their edit does not
/// end before the media, whose last packet lasts only to the input's end;
/// without it, the padding after the input too. ffmpeg presents the
/// padding of both, as it does of AAC's fragmented files (see the
/// `mp4_fragmented` test).
#[test]
fn fragmented_opus_in_mp4_files_decode_as_the_files_written_whole() {
    for channels in [1, 2, 6] {
        let len = 3 * 48_000 + 500;
        let context = format!("{channels} channels");
        let stream = Stream::new(channels, len);
        let all = decoded_ok(ffmpeg::decode_mp4(&stream.whole(None)), &context);
        assert!(all.starts_with(&stream.reference), "{context}");
        assert_eq!(all.len(), after_pre_skip(&stream) * channels, "{context}");
        let declared = stream.fragmented(FragmentOptions::default().length(len as u64));
        let undeclared = stream.fragmented(FragmentOptions::default());
        let gst = |file: &[u8]| decoded_ok(gstreamer::decode_opus_mp4(file), &context);
        let gst_all = gst(&stream.whole(None));
        assert!(as_near(&gst_all, &all), "{context}");
        assert!(as_near(&gst(&declared), &stream.reference), "{context}");
        assert!(gst(&undeclared) == gst_all, "{context}");
        let ffmpeg = |file| decoded_ok(ffmpeg::decode_mp4(file), &context);
        assert!(ffmpeg(&declared) == all, "{context}");
        assert!(ffmpeg(&undeclared) == all, "{context}");
    }
}
