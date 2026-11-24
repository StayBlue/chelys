// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! ALAC in MP4 files, of packets that ffmpeg's ALAC encoder made, as ffmpeg
//! and GStreamer read and decode them.

use std::f64::consts::TAU;
use std::num::NonZeroU32;

use chelys::mp4::{FragmentInfo, FragmentOptions, FragmentedWriter, Writer, alac};
use chelys_conformance::{ffmpeg, gstreamer};

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
                let rise = (fs.min(48_000.0) / 4.0 - from) / (2.0 * len.max(1) as f64);
                let phase = TAU * (from + rise * t as f64) * t as f64 / fs;
                (0.5 * peak * phase.sin()).round() as i32
            })
        })
        .collect()
}

/// The packets of `pcm`, as ffmpeg's ALAC encoder makes them and ffprobe
/// splits them, each with its duration, and the configuration of their
/// stream.
///
/// ffprobe gives the stream's extradata as the box that holds the cookie in
/// an MP4 file, as the cookie document describes it: its size, `'alac'`
/// and its version and flags, 12 bytes, then the cookie.
fn packets(
    pcm: &[i32],
    sample_rate: u32,
    channels: usize,
    bits: u32,
) -> (alac::Config, Vec<(Vec<u8>, u32)>) {
    let file = ffmpeg::encode_alac(pcm, sample_rate, channels, bits);
    let stream = ffmpeg::packets(&file, "m4a");
    let (header, cookie) = stream.config.split_at(12);
    assert_eq!(&header[..4], (stream.config.len() as u32).to_be_bytes());
    assert_eq!(&header[4..], b"alac\0\0\0\0");
    assert_eq!(stream.initial_padding, 0);
    let config = alac::Config::from_magic_cookie(cookie).unwrap();
    let durations = stream.durations.iter().map(|&d| u32::try_from(d).unwrap());
    (config, stream.packets.into_iter().zip(durations).collect())
}

/// The MP4 file of `packets`, written whole, each given with its duration
/// where it is not the frame length.
fn whole(config: &alac::Config, packets: &[(Vec<u8>, u32)], length: Option<u64>) -> Vec<u8> {
    let mut writer = Writer::new(config.clone());
    for (packet, duration) in packets {
        if *duration == config.frame_length() {
            writer.push(packet).unwrap();
        } else {
            writer.push_with_duration(packet, *duration).unwrap();
        }
    }
    let mut file = Vec::new();
    writer.finish(length, &mut file).unwrap();
    file
}

/// The fragmented MP4 file of `packets`, each given with its duration, and
/// the number of its fragments.
fn fragmented(
    config: &alac::Config,
    packets: &[(Vec<u8>, u32)],
    options: FragmentOptions,
) -> (Vec<u8>, usize) {
    let mut writer = FragmentedWriter::new(config.clone(), options);
    let mut file = writer.init_segment().to_vec();
    let mut fragments = 0;
    let mut out = |fragment: &[u8], _: FragmentInfo| {
        file.extend_from_slice(fragment);
        fragments += 1;
    };
    for (packet, duration) in packets {
        writer
            .push_with_duration(packet, *duration, &mut out)
            .unwrap();
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
/// among them, whose `samplerate` field holds an integer division of the
/// rate and whose `SamplingRateBox` the rate itself, with a last packet
/// shorter than the others. ffprobe gives the stream its rate, channels
/// and length.
#[test]
fn ffmpeg_decodes_alac_in_mp4_files_to_the_input_exactly() {
    for (sample_rate, bits) in [
        (8_000, 16),
        (44_100, 16),
        (48_000, 24),
        (96_000, 24),
        (176_400, 24),
        (192_000, 24),
    ] {
        for channels in [1, 2, 6] {
            let context = format!("{sample_rate} Hz, {bits} bits, {channels} channels");
            let len = sample_rate as usize / 4 + 123;
            let pcm = sweeps(sample_rate, channels, len, bits);
            let (config, packets) = packets(&pcm, sample_rate, channels, bits);
            assert_eq!(config.sample_rate(), sample_rate, "{context}");
            assert_eq!(packets.len(), len.div_ceil(4096), "{context}");
            assert!(packets.last().unwrap().1 < 4096, "{context}");
            let file = whole(&config, &packets, Some(len as u64));
            let stream = ffmpeg::probe_mp4(&file);
            assert_eq!(stream.info.codec_name, "alac", "{context}");
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

/// A cookie with an `ALACChannelLayoutInfo`, of 48 bytes, in a box of 60:
/// ffmpeg reads the layout it gives, 5.1 (`kALACChannelLayoutTag_MPEG_5_1_D`,
/// "C L R Ls Rs LFE"), and decodes the file as the one whose cookie has
/// none, of 6 channels.
#[test]
fn ffmpeg_reads_the_channel_layout_of_the_cookie() {
    let (sample_rate, channels, bits) = (48_000, 6, 16);
    let len = 10_000;
    let pcm = sweeps(sample_rate, channels, len, bits);
    let (config, packets) = packets(&pcm, sample_rate, channels, bits);
    let file = ffmpeg::encode_alac(&pcm, sample_rate, channels, bits);
    let cookie = &ffmpeg::packets(&file, "m4a").config[12..];
    let layout = [
        &24u32.to_be_bytes()[..],
        b"chan",
        &[0; 4],
        &((124u32 << 16) | 6).to_be_bytes(),
        &[0; 8],
    ]
    .concat();
    let with_layout = alac::Config::from_magic_cookie(&[cookie, &layout].concat()).unwrap();
    let plain = whole(&config, &packets, None);
    let file = whole(&with_layout, &packets, None);
    let stream = ffmpeg::probe_mp4(&file);
    assert_eq!(stream.info.channels, channels);
    assert_eq!(stream.channel_layout, "5.1");
    let decoded = ffmpeg::decode_mp4(&file);
    assert_eq!(decoded.diagnostics, "");
    assert!(decoded.samples == ffmpeg::decode_mp4(&plain).samples);
    assert!(decoded.samples == as_decoded(&pcm, bits));
}

/// The fragmented files decode to the input exactly too, with the length
/// declared or not, in fragments of a second and of a packet.
#[test]
fn ffmpeg_decodes_fragmented_alac_in_mp4_files_to_the_input_exactly() {
    for (sample_rate, channels) in [(44_100, 2), (96_000, 1)] {
        let len = 3 * sample_rate as usize + 500;
        let pcm = sweeps(sample_rate, channels, len, 24);
        let (config, packets) = packets(&pcm, sample_rate, channels, 24);
        for length in [None, Some(len as u64)] {
            for fragment_samples in [None, Some(4096)] {
                let context = format!("{sample_rate} Hz, {length:?}, {fragment_samples:?}");
                let mut options = FragmentOptions::default();
                if let Some(length) = length {
                    options = options.length(length);
                }
                if let Some(samples) = fragment_samples {
                    options = options.fragment_samples(NonZeroU32::new(samples).unwrap());
                }
                let (file, fragments) = fragmented(&config, &packets, options);
                let expected = match fragment_samples {
                    None => 4,
                    Some(_) => packets.len(),
                };
                assert_eq!(fragments, expected, "{context}");
                let decoded = ffmpeg::decode_mp4(&file);
                assert_eq!(decoded.diagnostics, "", "{context}");
                assert!(decoded.samples == as_decoded(&pcm, 24), "{context}");
            }
        }
    }
}

/// ffmpeg, seeking in the files, written whole or fragmented, without
/// decoding from an earlier sample, starts decoding at the packet that
/// holds the time it seeks to, and decodes from it the input exactly: each
/// packet decodes without those before it.
#[test]
fn ffmpeg_decodes_alac_in_mp4_files_from_where_it_seeks() {
    let (sample_rate, channels, bits) = (44_100, 2, 16);
    let len = 4 * sample_rate as usize + 300;
    let pcm = sweeps(sample_rate, channels, len, bits);
    let (config, packets) = packets(&pcm, sample_rate, channels, bits);
    let expected = as_decoded(&pcm, bits);
    let files = [
        whole(&config, &packets, None),
        fragmented(&config, &packets, FragmentOptions::default()).0,
    ];
    for (i, file) in files.iter().enumerate() {
        for from in [0.5, 1.0, 1.37, 2.9, 3.5] {
            let context = format!("file {i}, {from} s");
            let seeked = ffmpeg::decode_mp4_from_seek_point(file, from);
            assert_eq!(seeked.diagnostics, "", "{context}");
            let start = len - seeked.samples.len() / channels;
            let at = (from * f64::from(sample_rate)).round() as usize;
            assert_eq!(start, at / 4096 * 4096, "{context}");
            assert!(
                seeked.samples == expected[start * channels..],
                "{context}: from {start}"
            );
        }
    }
}

/// ffmpeg decodes files of a sample, and of a sample either side of a
/// packet's length, to exactly their samples.
#[test]
fn ffmpeg_reads_short_alac_in_mp4_files() {
    let (sample_rate, channels, bits) = (48_000, 1, 16);
    for len in [1, 4095, 4096, 4097] {
        let pcm = sweeps(sample_rate, channels, len, bits);
        let (config, packets) = packets(&pcm, sample_rate, channels, bits);
        assert_eq!(packets.len(), len.div_ceil(4096), "{len}");
        let file = whole(&config, &packets, Some(len as u64));
        assert_eq!(ffmpeg::probe_mp4(&file).duration_ts, len as u64, "{len}");
        let decoded = ffmpeg::decode_mp4(&file);
        assert_eq!(decoded.diagnostics, "", "{len}");
        assert!(decoded.samples == as_decoded(&pcm, bits), "{len}");
    }
}

/// GStreamer, through `qtdemux` and `avdec_alac`, which wraps ffmpeg's
/// decoder, decodes the files written whole and the fragmented ones to the
/// input exactly too, at their rate, those above 65535 Hz among them. Of
/// a fragmented file of a declared length, the segment `qtdemux` marks to
/// the `MovieExtendsHeaderBox`'s duration can end a nanosecond short of the
/// last sample, which it then leaves out, as it can of FLAC and AAC (see
/// the `gstreamer` module).
#[test]
fn gstreamer_decodes_alac_in_mp4_files_to_the_input_exactly() {
    for (sample_rate, channels) in [(44_100, 2), (96_000, 6), (192_000, 1)] {
        let len = sample_rate as usize + 777;
        let pcm = sweeps(sample_rate, channels, len, 24);
        let (config, packets) = packets(&pcm, sample_rate, channels, 24);
        let expected = as_decoded(&pcm, 24);
        let files = [
            whole(&config, &packets, None),
            fragmented(&config, &packets, FragmentOptions::default()).0,
            fragmented(
                &config,
                &packets,
                FragmentOptions::default().length(len as u64),
            )
            .0,
        ];
        for (i, file) in files.iter().enumerate() {
            let context = format!("{sample_rate} Hz, {channels} channels, file {i}");
            let decoded = gstreamer::decode_alac_mp4(file, sample_rate);
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
