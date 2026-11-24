// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! AAC-LC in MP4 files, of packets that ffmpeg's native AAC encoder made,
//! as ffmpeg and GStreamer read and decode them, compared with the MP4
//! files the packets came in, which ffmpeg presents from the end of the
//! encoder's delay to the end of the input.

use std::f64::consts::TAU;

use chelys::mp4::{FragmentInfo, FragmentOptions, FragmentedWriter, Writer, aac};
use chelys_conformance::{Decoded, ffmpeg, gstreamer};

/// A sweep from 100 Hz up, `len` samples at `sample_rate` in each of
/// `channels`, interleaved, each channel's a little higher and quieter
/// than the one before.
fn sweeps(sample_rate: u32, channels: usize, len: usize) -> Vec<f32> {
    let fs = f64::from(sample_rate);
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

/// A stream of `len` samples of `channels`: its MP4 file as ffmpeg's
/// native AAC encoder makes it, decoded as ffmpeg presents it, and its
/// packets as ffprobe splits them, with the configuration of their stream
/// and the encoder's delay.
struct Stream {
    channels: usize,
    reference: Vec<f32>,
    asc: Vec<u8>,
    delay: u32,
    packets: Vec<Vec<u8>>,
}

impl Stream {
    fn new(sample_rate: u32, channels: usize, len: usize) -> Self {
        let pcm = sweeps(sample_rate, channels, len);
        let m4a = ffmpeg::encode_aac(&pcm, sample_rate, channels, 64 * channels as u32);
        let reference = ffmpeg::decode_file(&m4a, "m4a");
        assert_eq!(reference.diagnostics, "");
        assert_eq!(reference.samples.len(), pcm.len());
        let packets = ffmpeg::packets(&m4a, "m4a");
        Self {
            channels,
            reference: reference.samples,
            asc: packets.config,
            delay: u32::try_from(packets.initial_padding).unwrap(),
            packets: packets.packets,
        }
    }

    /// The configuration of the packets, of their delay if `delayed`.
    fn config(&self, delayed: bool) -> aac::Config {
        let delay = delayed.then_some(self.delay);
        aac::Config::from_audio_specific_config(&self.asc, delay).unwrap()
    }

    /// The MP4 file of the packets, written whole, of `length` samples.
    fn whole(&self, delayed: bool, length: Option<u64>) -> Vec<u8> {
        let mut writer = Writer::new(self.config(delayed));
        for packet in &self.packets {
            writer.push(packet).unwrap();
        }
        let mut file = Vec::new();
        writer.finish(length, &mut file).unwrap();
        file
    }

    /// The fragmented MP4 file of the packets, of their delay.
    fn fragmented(&self, options: FragmentOptions) -> Vec<u8> {
        let mut writer = FragmentedWriter::new(self.config(true), options);
        let mut file = writer.init_segment().to_vec();
        let mut out = |fragment: &[u8], _: FragmentInfo| file.extend_from_slice(fragment);
        for packet in &self.packets {
            writer.push(packet, &mut out).unwrap();
        }
        writer.finish(&mut out).unwrap();
        file
    }

    /// Every sample of each channel that the packets decode to.
    fn decodes_to(&self) -> usize {
        1024 * self.packets.len()
    }
}

/// The lengths of the tests: from a sample, either side of a frame, to a
/// few seconds.
const LENGTHS: [usize; 6] = [1, 1023, 1024, 1025, 44_100 + 123, 3 * 44_100];

/// The channels of the tests: mono, stereo, 5.1, of `channelConfiguration`
/// 6, and 7.1, of 12, each with ffmpeg's extension of the configuration
/// that signals SBR absent.
const CHANNELS: [usize; 4] = [1, 2, 6, 8];

fn decoded_ok(decoded: Decoded, context: &str) -> Vec<f32> {
    assert_eq!(decoded.diagnostics, "", "{context}");
    decoded.samples
}

/// ffmpeg n9.0.2 reads the files as AAC LC of the input's rate and
/// channels, and decodes them to exactly what it decodes from its own MP4
/// files: the input's length, from the end of the encoder's delay, which
/// the edit list leaves out, to the input's end, where the edit ends.
#[test]
fn ffmpeg_decodes_aac_packets_in_mp4_files_as_its_own_files() {
    for sample_rate in [44_100, 48_000] {
        for channels in CHANNELS {
            for len in LENGTHS {
                let context = format!("{sample_rate} Hz, {channels} channels, {len} samples");
                let stream = Stream::new(sample_rate, channels, len);
                let file = stream.whole(true, Some(len as u64));
                let probed = ffmpeg::probe_mp4(&file);
                assert_eq!(probed.info.codec_name, "aac", "{context}");
                assert_eq!(probed.profile, "LC", "{context}");
                assert_eq!(probed.info.sample_rate, sample_rate, "{context}");
                assert_eq!(probed.info.channels, channels, "{context}");
                let decoded = decoded_ok(ffmpeg::decode_mp4(&file), &context);
                assert!(decoded == stream.reference, "{context}");
            }
        }
    }
}

/// Without a delay, the file has no edit list, and ffmpeg presents every
/// sample the packets decode to, the delay and the padding included, with
/// the input in its place after the delay.
#[test]
fn aac_packets_without_a_delay_present_every_sample() {
    for channels in CHANNELS {
        let len = 44_100 + 123;
        let context = format!("{channels} channels");
        let stream = Stream::new(44_100, channels, len);
        let decoded = decoded_ok(ffmpeg::decode_mp4(&stream.whole(false, None)), &context);
        assert_eq!(decoded.len(), stream.decodes_to() * channels, "{context}");
        let start = stream.delay as usize * channels;
        assert!(
            decoded[start..start + len * channels] == stream.reference,
            "{context}"
        );
    }
}

/// The fragmented files of a declared length present the input alone to
/// GStreamer, through `qtdemux`, `avdec_aac` and `audiosegmentclip`, as
/// the files written whole do; without it, the padding after the input
/// too. ffmpeg presents the padding of both, as it does of chelys's own
/// fragmented AAC files (see the `mp4_fragmented` test).
#[test]
fn fragmented_aac_packets_in_mp4_files_decode_as_the_files_written_whole() {
    for channels in CHANNELS {
        let len = 3 * 44_100 + 500;
        let context = format!("{channels} channels");
        let stream = Stream::new(44_100, channels, len);
        let all = decoded_ok(ffmpeg::decode_mp4(&stream.whole(true, None)), &context);
        assert!(all.starts_with(&stream.reference), "{context}");
        let after_delay = stream.decodes_to() - stream.delay as usize;
        assert_eq!(all.len(), after_delay * stream.channels, "{context}");
        let declared = stream.fragmented(FragmentOptions::default().length(len as u64));
        let undeclared = stream.fragmented(FragmentOptions::default());
        let gst = |file: &[u8]| decoded_ok(gstreamer::decode_mp4(file), &context);
        let whole = stream.whole(true, Some(len as u64));
        assert!(gst(&declared) == gst(&whole), "{context}");
        assert!(
            gst(&undeclared) == gst(&stream.whole(true, None)),
            "{context}"
        );
        let ffmpeg = |file| decoded_ok(ffmpeg::decode_mp4(file), &context);
        assert!(ffmpeg(&declared) == all, "{context}");
        assert!(ffmpeg(&undeclared) == all, "{context}");
    }
}
