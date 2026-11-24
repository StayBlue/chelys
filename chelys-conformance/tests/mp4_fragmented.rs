// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! AAC-LC in fragmented MP4 files, as ffmpeg reads, decodes and seeks in
//! them, against the files written whole of the same input.

use std::f64::consts::TAU;

use chelys::aac::{Config, DELAY, Rate};
use chelys::mp4::FragmentInfo;
use chelys::mp4::aac::{Encoder, FragmentOptions, FragmentedEncoder};
use chelys_conformance::{ffmpeg, gstreamer};

/// A sweep from 100 Hz up, `len` samples at `sample_rate`, in each of
/// `channels`, interleaved, each channel's a little higher and quieter
/// than the one before.
fn sweeps(sample_rate: u32, channels: usize, len: usize) -> Vec<f32> {
    let fs = f64::from(sample_rate);
    let top = (fs / 2.0).min(12_000.0);
    (0..len)
        .flat_map(|t| {
            (0..channels).map(move |c| {
                let from = 100.0 * (1.0 + 0.2 * c as f64);
                let rise = (top - from) / (2.0 * len.max(1) as f64);
                let phase = TAU * (from + rise * t as f64) * t as f64 / fs;
                (0.4 / (1.0 + 0.1 * c as f64) * phase.sin()) as f32
            })
        })
        .collect()
}

/// The file written whole of `pcm`.
fn whole(config: Config, pcm: &[f32]) -> Vec<u8> {
    let mut encoder = Encoder::new(config).unwrap();
    encoder.encode_interleaved(pcm).unwrap();
    let mut file = Vec::new();
    encoder.finish(&mut file).unwrap();
    file
}

/// The fragmented file of `pcm`, in `channels`, its init segment and each
/// fragment apart, with what each holds; its length declared if `known`.
fn fragmented(
    config: Config,
    channels: usize,
    pcm: &[f32],
    known: bool,
) -> (Vec<u8>, Vec<(Vec<u8>, FragmentInfo)>) {
    let mut options = FragmentOptions::default();
    if known {
        options = options.length((pcm.len() / channels) as u64);
    }
    let mut encoder = FragmentedEncoder::new(config, options).unwrap();
    let init = encoder.init_segment().to_vec();
    let mut fragments = Vec::new();
    let mut out = |fragment: &[u8], info: FragmentInfo| fragments.push((fragment.to_vec(), info));
    // In calls of a tenth of a second or so, as from a live source.
    let chunk = 4410 * encoder.channels();
    for part in pcm.chunks(chunk) {
        encoder.encode_interleaved(part, &mut out).unwrap();
    }
    encoder.finish(&mut out).unwrap();
    (init, fragments)
}

/// The init segment and the fragments, one after another.
fn file(init: &[u8], fragments: &[(Vec<u8>, FragmentInfo)]) -> Vec<u8> {
    let mut file = init.to_vec();
    for (fragment, _) in fragments {
        file.extend_from_slice(fragment);
    }
    file
}

/// ffmpeg decodes a fragmented file to what it decodes of the file
/// written whole of the same input, sample for sample, at the rates of
/// every kind and in mono and stereo, whether the length was declared or
/// not; but where the file written whole ends with the input, ffmpeg
/// n9.0.2 decodes the fragmented file on to the end of its last frame:
/// it starts at the edit's `media_time`, after the [`DELAY`] samples, and
/// goes on past a declared length as where there is none. With its edit
/// list ignored, it decodes every sample of every frame, as of the file
/// written whole.
#[test]
fn ffmpeg_decodes_fragmented_files_to_the_frames_of_the_file_written_whole() {
    for (sample_rate, channels, rate) in [
        (
            44_100,
            2,
            Rate::Constant {
                bits_per_second: 128_000,
            },
        ),
        (48_000, 1, Rate::Quality { level: 50 }),
        (
            22_050,
            2,
            Rate::Mean {
                bits_per_second: 48_000,
            },
        ),
    ] {
        let len = 3 * sample_rate as usize + 123;
        let pcm = sweeps(sample_rate, channels, len);
        let config = Config::new(sample_rate, channels, rate);
        let whole = whole(config, &pcm);
        let all = ffmpeg::decode_mp4_ignoring_edits(&whole);
        assert_eq!(all.diagnostics, "");
        let edited = ffmpeg::decode_mp4(&whole).samples;
        assert_eq!(edited.len(), pcm.len());
        for known in [true, false] {
            let context = format!("{sample_rate} Hz, {channels} channels, {rate:?}, {known}");
            let (init, fragments) = fragmented(config, channels, &pcm, known);
            let file = file(&init, &fragments);
            let decoded = ffmpeg::decode_mp4(&file);
            assert_eq!(decoded.diagnostics, "", "{context}");
            assert!(
                decoded.samples == all.samples[DELAY * channels..],
                "{context}"
            );
            assert!(decoded.samples[..pcm.len()] == edited, "{context}");
            let ignoring = ffmpeg::decode_mp4_ignoring_edits(&file);
            assert_eq!(ignoring.diagnostics, "", "{context}");
            assert!(ignoring.samples == all.samples, "{context}");
        }
    }
}

/// ffprobe n9.0.2 gives the stream of a fragmented file the duration of
/// its media, which ends with the input, as its last frame lasts only to
/// the end of the input: the [`DELAY`] and the input, whether its length
/// was declared or not, though ffmpeg decodes it on to the end of its last
/// frame (see
/// `ffmpeg_decodes_fragmented_files_to_the_frames_of_the_file_written_whole`);
/// and it reads it as AAC LC of the input's channels and rate.
#[test]
fn ffprobe_reads_fragmented_files_as_long_as_their_media() {
    let (sample_rate, channels) = (44_100, 2);
    let len = 2 * sample_rate as usize + 500;
    let pcm = sweeps(sample_rate, channels, len);
    let config = Config::new(
        sample_rate,
        channels,
        Rate::Constant {
            bits_per_second: 96_000,
        },
    );
    for known in [true, false] {
        let (init, fragments) = fragmented(config, channels, &pcm, known);
        let duration: u64 = fragments.iter().map(|(_, info)| info.duration).sum();
        assert_eq!(duration, (DELAY + len) as u64, "{known}");
        let stream = ffmpeg::probe_mp4(&file(&init, &fragments));
        assert_eq!(stream.info.codec_name, "aac", "{known}");
        assert_eq!(stream.profile, "LC", "{known}");
        assert_eq!(stream.info.sample_rate, sample_rate, "{known}");
        assert_eq!(stream.info.channels, channels, "{known}");
        assert_eq!(stream.time_base, format!("1/{sample_rate}"), "{known}");
        assert_eq!(stream.duration_ts, duration, "{known}");
    }
}

/// The init segment and the fragments up to any one decode, as a file cut
/// short or a stream joined late would, to the start of what the whole
/// file decodes to: every sample of their frames after the [`DELAY`].
#[test]
fn ffmpeg_decodes_the_first_fragments_alone_to_the_start_of_the_file() {
    let (sample_rate, channels) = (48_000, 2);
    let pcm = sweeps(sample_rate, channels, 4 * sample_rate as usize);
    let config = Config::new(
        sample_rate,
        channels,
        Rate::Constant {
            bits_per_second: 128_000,
        },
    );
    let (init, fragments) = fragmented(config, channels, &pcm, false);
    let all = ffmpeg::decode_mp4(&file(&init, &fragments)).samples;
    assert!(fragments.len() >= 4);
    for k in 1..fragments.len() {
        let decoded = ffmpeg::decode_mp4(&file(&init, &fragments[..k]));
        assert_eq!(decoded.diagnostics, "", "{k} fragments");
        let info = fragments[k - 1].1;
        let samples = (info.decode_time + info.duration) as usize - DELAY;
        assert_eq!(decoded.samples.len(), samples * channels, "{k} fragments");
        assert!(
            decoded.samples == all[..samples * channels],
            "{k} fragments"
        );
    }
}

/// ffmpeg n9.0.2 seeks in fragmented files to the sample asked for, in
/// any fragment, and decodes from there exactly as it decodes the whole
/// file, the frame before decoded first where it is in the fragment
/// before.
#[test]
fn ffmpeg_seeks_in_fragmented_files_and_decodes_from_there_exactly() {
    let (sample_rate, channels) = (44_100, 2);
    let len = 4 * sample_rate as usize;
    let pcm = sweeps(sample_rate, channels, len);
    let config = Config::new(
        sample_rate,
        channels,
        Rate::Constant {
            bits_per_second: 128_000,
        },
    );
    let (init, fragments) = fragmented(config, channels, &pcm, true);
    let file = file(&init, &fragments);
    let whole = ffmpeg::decode_mp4(&file).samples;
    let decoded_len = whole.len() / channels;
    // The second fragment's first frame starts 43 frames into the media,
    // 42 into the presentation; the times about it seek across the
    // boundary.
    let boundary = 42.0 * 1024.0 / f64::from(sample_rate);
    for from in [
        0.5,
        boundary - 0.01,
        boundary,
        boundary + 0.01,
        1.37,
        2.9,
        3.5,
    ] {
        let seeked = ffmpeg::decode_mp4_seeking(&file, from);
        assert_eq!(seeked.diagnostics, "", "{from} s");
        let start = decoded_len - seeked.samples.len() / channels;
        let at = (from * f64::from(sample_rate)).round() as usize;
        assert!((at..at + 1024).contains(&start), "{from} s: from {start}");
        assert!(
            seeked.samples == whole[start * channels..],
            "{from} s: from {start}"
        );
    }
}

/// ffmpeg decodes fragmented files of no samples, and of a sample or so
/// either side of a frame's length, to every sample of their frames after
/// the [`DELAY`], whether their length was declared or not: none for no
/// samples, whose one frame the delay takes.
#[test]
fn ffmpeg_reads_empty_and_short_fragmented_files() {
    let (sample_rate, channels) = (48_000, 1);
    let config = Config::new(
        sample_rate,
        channels,
        Rate::Constant {
            bits_per_second: 64_000,
        },
    );
    for len in [0, 1, 2, 1023, 1024, 1025, 2047, 2048] {
        let pcm = sweeps(sample_rate, channels, len);
        let frames = (len + DELAY).div_ceil(1024);
        let edited = ffmpeg::decode_mp4(&whole(config, &pcm)).samples;
        for known in [true, false] {
            let context = format!("{len} samples, {known}");
            let (init, fragments) = fragmented(config, channels, &pcm, known);
            assert_eq!(fragments.len(), 1, "{context}");
            assert_eq!(fragments[0].1.frames as usize, frames, "{context}");
            let decoded = ffmpeg::decode_mp4(&file(&init, &fragments));
            assert_eq!(decoded.diagnostics, "", "{context}");
            assert_eq!(decoded.samples.len(), frames * 1024 - DELAY, "{context}");
            assert!(decoded.samples[..len] == edited, "{context}");
        }
    }
}

/// `samples` at `sample_rate` in nanoseconds, rounded down.
fn nanoseconds(samples: u64, sample_rate: u32) -> u64 {
    (u128::from(samples) * 1_000_000_000 / u128::from(sample_rate)) as u64
}

/// Whether `time` in nanoseconds is that of `samples` at `sample_rate`, to
/// the nanosecond either way.
fn at_samples(time: u64, samples: u64, sample_rate: u32) -> bool {
    time.abs_diff(nanoseconds(samples, sample_rate)) <= 1
}

/// GStreamer 1.28.9 (`qtdemux`, `avdec_aac`, `audiosegmentclip`) decodes a
/// fragmented file whose length was declared to what it decodes of the
/// file written whole of the same input, sample for sample, the padding
/// left out, where the padding is none, a frame's less one, and in
/// between, and for no input.
///
/// `qtdemux` marks segments from the [`DELAY`] to the end of the input,
/// where the last frame's duration ends the media: of the fragmented file
/// one as it reads the `MovieBox`, and one again at some fragments, the
/// last of which ends at the end of the input to the nanosecond, and none
/// past it; of the file written whole one, which ends there to the
/// nanosecond. Where the file written whole's ends a nanosecond short of a
/// sample's time, as at 48 kHz, GStreamer decodes one sample fewer of it,
/// at the end. Of both files it decodes the input and at some rates one
/// sample more at the start, and of no input, whose edit lasts no time,
/// one sample, `qtdemux` marking a segment with no end.
#[test]
fn gstreamer_decodes_fragmented_files_of_a_declared_length_as_the_files_written_whole() {
    let cases = [
        (
            44_100,
            2,
            Rate::Constant {
                bits_per_second: 128_000,
            },
            88_700,
        ),
        (48_000, 1, Rate::Quality { level: 50 }, 2 * 48_000 + 77),
        (
            22_050,
            2,
            Rate::Mean {
                bits_per_second: 48_000,
            },
            30 * 1024,
        ),
        (
            44_100,
            1,
            Rate::Constant {
                bits_per_second: 64_000,
            },
            40 * 1024 + 1,
        ),
        (
            44_100,
            2,
            Rate::Constant {
                bits_per_second: 128_000,
            },
            1,
        ),
        (
            44_100,
            2,
            Rate::Constant {
                bits_per_second: 128_000,
            },
            0,
        ),
    ];
    for (sample_rate, channels, rate, len) in cases {
        let context = format!("{sample_rate} Hz, {channels} channels, {rate:?}, {len} samples");
        let pcm = sweeps(sample_rate, channels, len);
        let config = Config::new(sample_rate, channels, rate);
        let whole = self::whole(config, &pcm);
        let whole_decoded = gstreamer::decode_mp4(&whole);
        assert_eq!(whole_decoded.diagnostics, "", "{context}");
        let whole_len = whole_decoded.samples.len() / channels;
        let (init, fragments) = fragmented(config, channels, &pcm, true);
        let file = file(&init, &fragments);
        let decoded = gstreamer::decode_mp4(&file);
        assert_eq!(decoded.diagnostics, "", "{context}");
        let decoded_len = decoded.samples.len() / channels;
        assert!(
            decoded.samples.starts_with(&whole_decoded.samples),
            "{context}: {decoded_len} and {whole_len} samples"
        );
        let end = (DELAY + len) as u64;
        let segments = gstreamer::segments_mp4(&file);
        let whole_segments = gstreamer::segments_mp4(&whole);
        assert_eq!(whole_segments.len(), 1, "{context}");
        if len == 0 {
            assert_eq!(decoded_len, 1, "{context}");
            assert_eq!(whole_len, 1, "{context}");
            assert!(segments.iter().all(|s| s.stop.is_none()), "{context}");
            assert_eq!(whole_segments[0].stop, None, "{context}");
            continue;
        }
        assert!(
            (len..=len + 1).contains(&whole_len),
            "{context}: {whole_len}"
        );
        let stop = |segment: &gstreamer::Segment| segment.stop.expect("a stop");
        let last = stop(segments.last().unwrap());
        assert!(
            at_samples(last, end, sample_rate),
            "{context}: {segments:?}"
        );
        let exact = nanoseconds(end, sample_rate);
        assert!(
            segments.iter().all(|s| stop(s) <= exact + 1),
            "{context}: {segments:?}"
        );
        let whole_stop = stop(&whole_segments[0]);
        assert!(
            at_samples(whole_stop, end, sample_rate),
            "{context}: {whole_segments:?}"
        );
        // One sample more of the fragmented file only where the file
        // written whole's segment ends earlier.
        match decoded_len - whole_len {
            0 => {}
            1 => assert!(
                whole_stop < last,
                "{context}: {segments:?}, {whole_segments:?}"
            ),
            more => panic!("{context}: {more} samples more"),
        }
    }
}

/// GStreamer 1.28.9 (`qtdemux`, `avdec_aac`, `audiosegmentclip`) presents
/// the padding of a fragmented file whose length was not declared:
/// `qtdemux` marks a segment from the [`DELAY`] with no end, so that it
/// decodes every sample of every frame from there, as of the same file
/// with the length declared and then the padding.
#[test]
fn gstreamer_presents_the_padding_of_fragmented_files_of_no_length() {
    let (sample_rate, channels) = (44_100, 2);
    let len = 88_700;
    let pcm = sweeps(sample_rate, channels, len);
    let config = Config::new(
        sample_rate,
        channels,
        Rate::Constant {
            bits_per_second: 128_000,
        },
    );
    let (init, fragments) = fragmented(config, channels, &pcm, false);
    let frames: usize = fragments.iter().map(|(_, info)| info.frames as usize).sum();
    let file = file(&init, &fragments);
    let segments = gstreamer::segments_mp4(&file);
    assert_eq!(segments.len(), 1);
    assert!(
        at_samples(segments[0].start, DELAY as u64, sample_rate),
        "{segments:?}"
    );
    assert_eq!(segments[0].stop, None);
    let decoded = gstreamer::decode_mp4(&file);
    assert_eq!(decoded.diagnostics, "");
    assert_eq!(
        decoded.samples.len(),
        (frames * 1024 - DELAY + 1) * channels
    );
    let (init, fragments) = fragmented(config, channels, &pcm, true);
    let known = gstreamer::decode_mp4(&self::file(&init, &fragments)).samples;
    assert!(decoded.samples[..known.len()] == known);
}

/// GStreamer 1.28.9's `qtdemux` (black box), given a fragmented file whose
/// edit has the input's length but whose `MovieExtendsHeaderBox` is taken
/// out, as chelys never writes it, ends each segment it marks at the end
/// of the fragments it has read, until the last, which ends at the end of
/// the input; with the box, at the end of the input from the first. Its
/// samples decode alike either way, here.
#[test]
fn gstreamer_ends_a_finite_edit_without_mehd_with_the_fragments_read() {
    let (sample_rate, channels) = (44_100, 2);
    let len = 88_700;
    let pcm = sweeps(sample_rate, channels, len);
    let config = Config::new(
        sample_rate,
        channels,
        Rate::Constant {
            bits_per_second: 128_000,
        },
    );
    let (init, fragments) = fragmented(config, channels, &pcm, true);
    let with = file(&init, &fragments);
    let mut without = with.clone();
    let at = without.windows(4).position(|w| w == b"mehd").unwrap();
    without[at..at + 4].copy_from_slice(b"free");
    let end = (DELAY + len) as u64;
    let ends: Vec<u64> = fragments
        .iter()
        .map(|(_, info)| info.decode_time + info.duration)
        .collect();
    assert_eq!(*ends.last().unwrap(), end);
    let segments = gstreamer::segments_mp4(&without);
    assert_eq!(segments.len(), ends.len());
    for (segment, &fragment_end) in segments.iter().zip(&ends) {
        let stop = segment.stop.expect("a stop");
        assert!(at_samples(stop, fragment_end, sample_rate), "{segments:?}");
    }
    for segment in gstreamer::segments_mp4(&with) {
        assert!(at_samples(segment.stop.unwrap(), end, sample_rate));
    }
    let decoded = gstreamer::decode_mp4(&without);
    assert_eq!(decoded.diagnostics, "");
    assert!(decoded.samples == gstreamer::decode_mp4(&with).samples);
}
