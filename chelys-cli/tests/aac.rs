// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! `chelys aac` run as a program: its help, its errors, and the ADTS
//! stream it writes from raw PCM on stdin.

use std::io::Write;
use std::process::{Command, Output, Stdio};

use chelys::aac::{Config, DELAY, Encoder, Rate};

/// Runs `chelys` with `args`, `stdin` as its input.
fn chelys(args: &[&str], stdin: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_chelys"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("chelys runs");
    let mut input = child.stdin.take().unwrap();
    let stdin = stdin.to_vec();
    // Written apart from the reading of stdout, which a long input would
    // otherwise fill while chelys waits for the rest of it.
    let writer = std::thread::spawn(move || {
        // chelys may refuse its arguments without reading its input.
        let _ = input.write_all(&stdin);
    });
    let output = child.wait_with_output().expect("chelys ends");
    writer.join().unwrap();
    output
}

fn text(bytes: &[u8]) -> &str {
    std::str::from_utf8(bytes).expect("UTF-8")
}

/// `seconds` of a 440 Hz tone in the left channel and 660 Hz in the right,
/// interleaved.
fn stereo_tone(sample_rate: u32, seconds: f32) -> Vec<f32> {
    let len = (sample_rate as f32 * seconds) as usize;
    (0..len)
        .flat_map(|t| {
            let t = t as f32 / sample_rate as f32;
            [440.0, 660.0].map(|hz| 0.5 * (std::f32::consts::TAU * hz * t).sin())
        })
        .collect()
}

fn le_bytes(samples: &[f32]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

/// The fields of an `adts_fixed_header()` and `adts_variable_header()`
/// (clause 8.1.3.2.2) that the test checks.
#[derive(Debug, PartialEq, Eq)]
struct AdtsHeader {
    id: u8,
    layer: u8,
    protection_absent: u8,
    profile: u8,
    sampling_frequency_index: u8,
    channel_configuration: u8,
    frame_length: usize,
    number_of_raw_data_blocks_in_frame: u8,
}

fn adts_header(frame: &[u8]) -> AdtsHeader {
    assert!(frame.len() >= 7, "a header is 7 bytes");
    assert_eq!(
        (frame[0], frame[1] >> 4),
        (0xFF, 0xF),
        "syncword at a frame's start"
    );
    AdtsHeader {
        id: (frame[1] >> 3) & 1,
        layer: (frame[1] >> 1) & 3,
        protection_absent: frame[1] & 1,
        profile: frame[2] >> 6,
        sampling_frequency_index: (frame[2] >> 2) & 0xF,
        channel_configuration: ((frame[2] & 1) << 2) | (frame[3] >> 6),
        frame_length: (usize::from(frame[3] & 3) << 11)
            | (usize::from(frame[4]) << 3)
            | (usize::from(frame[5]) >> 5),
        number_of_raw_data_blocks_in_frame: frame[6] & 3,
    }
}

#[test]
fn encodes_raw_pcm_to_the_adts_stream_the_library_writes() {
    // Longer than the 64 KiB chelys reads at a time, and not a whole
    // number of frames.
    let pcm = stereo_tone(44_100, 1.3);
    let output = chelys(
        &[
            "aac",
            "--sample-rate",
            "44100",
            "--channels",
            "2",
            "--bitrate",
            "96",
        ],
        &le_bytes(&pcm),
    );
    assert!(output.status.success(), "{}", text(&output.stderr));
    assert_eq!(text(&output.stderr), "");

    let config = Config::new(
        44_100,
        2,
        Rate::Constant {
            bits_per_second: 96_000,
        },
    );
    let mut expected = Vec::new();
    let mut encoder = Encoder::new(config).unwrap();
    encoder.encode_interleaved(&pcm, &mut expected);
    let finished = encoder.finish(&mut expected);
    assert!(output.stdout == expected, "the library's stream");

    let samples = (pcm.len() / 2) as u64;
    let frames = (samples + DELAY as u64).div_ceil(1024);
    assert_eq!(finished.frames, frames);
    let mut stream = output.stdout.as_slice();
    let mut count = 0;
    while !stream.is_empty() {
        let header = adts_header(stream);
        assert_eq!(
            header,
            AdtsHeader {
                id: 1,
                layer: 0,
                protection_absent: 1,
                profile: 1,                  // Low Complexity
                sampling_frequency_index: 4, // 44.1 kHz
                channel_configuration: 2,
                frame_length: header.frame_length,
                number_of_raw_data_blocks_in_frame: 0,
            },
            "frame {count}"
        );
        assert!(header.frame_length <= stream.len(), "frame {count} whole");
        stream = &stream[header.frame_length..];
        count += 1;
    }
    assert_eq!(count, frames);
}

/// Each option of the header gives the stream the library writes with it,
/// and every option together their stream.
#[test]
fn the_header_options_are_the_library_s() {
    let pcm: Vec<f32> = stereo_tone(48_000, 0.3).into_iter().step_by(2).collect();
    let base = ["aac", "--sample-rate", "48000", "--channels", "1"];
    type Set = fn(Config) -> Config;
    let options: [(&str, Set); 4] = [
        ("--crc", |config| config.crc(true)),
        ("--private-bit", |config| config.private_bit(true)),
        ("--original-copy", |config| config.original_copy(true)),
        ("--home", |config| config.home(true)),
    ];
    let encode = |config: Config| {
        let mut expected = Vec::new();
        let mut encoder = Encoder::new(config).unwrap();
        encoder.encode_interleaved(&pcm, &mut expected);
        encoder.finish(&mut expected);
        expected
    };
    let plain = Config::new(
        48_000,
        1,
        Rate::Constant {
            bits_per_second: 128_000,
        },
    );
    let mut all_args = base.to_vec();
    let mut all = plain;
    for (flag, set) in options {
        let output = chelys(&[&base[..], &[flag]].concat(), &le_bytes(&pcm));
        assert_eq!(text(&output.stderr), "", "{flag}");
        assert!(output.stdout == encode(set(plain)), "{flag}");
        assert!(output.stdout != encode(plain), "{flag}");
        all_args.push(flag);
        all = set(all);
    }
    let output = chelys(&all_args, &le_bytes(&pcm));
    assert!(output.stdout == encode(all));
    assert_eq!(adts_header(&output.stdout).protection_absent, 0);
}

#[test]
fn encodes_mono_at_the_default_bitrate() {
    let pcm: Vec<f32> = stereo_tone(8_000, 0.5).into_iter().step_by(2).collect();
    let output = chelys(
        &["aac", "--sample-rate", "8000", "--channels", "1"],
        &le_bytes(&pcm),
    );
    assert!(output.status.success(), "{}", text(&output.stderr));
    let header = adts_header(&output.stdout);
    assert_eq!(header.sampling_frequency_index, 11);
    assert_eq!(header.channel_configuration, 1);
}

/// Interleaved input of 6 and 8 channels gives the stream the library
/// writes of it, in channel configurations 6 and 7.
#[test]
fn encodes_5_1_and_7_1_as_the_library_does() {
    for (channels, channel_configuration) in [(6, 6), (8, 7)] {
        let pcm: Vec<f32> = (stereo_tone(48_000, 0.2).chunks(2))
            .flat_map(|pair| (0..channels).map(move |c| pair[c % 2] / (1 + c) as f32))
            .collect();
        let args = [
            "--sample-rate",
            "48000",
            "--channels",
            &channels.to_string(),
        ];
        let output = chelys(
            &[&["aac"][..], &args, &["--bitrate", "320"]].concat(),
            &le_bytes(&pcm),
        );
        assert!(output.status.success(), "{}", text(&output.stderr));
        let config = Config::new(
            48_000,
            channels,
            Rate::Constant {
                bits_per_second: 320_000,
            },
        );
        let mut expected = Vec::new();
        let mut encoder = Encoder::new(config).unwrap();
        encoder.encode_interleaved(&pcm, &mut expected);
        encoder.finish(&mut expected);
        assert!(output.stdout == expected, "{channels} channels");
        let header = adts_header(&output.stdout);
        assert_eq!(header.channel_configuration, channel_configuration);
    }
}

#[test]
fn encodes_empty_input_to_a_frame_of_silence() {
    let output = chelys(&["aac", "--sample-rate", "48000", "--channels", "2"], &[]);
    assert!(output.status.success(), "{}", text(&output.stderr));
    let header = adts_header(&output.stdout);
    assert_eq!(header.frame_length, output.stdout.len(), "one frame");
}

#[test]
fn prints_its_usage_when_asked() {
    let output = chelys(&["aac", "--help"], &[]);
    assert!(output.status.success());
    let usage = text(&output.stdout);
    assert!(usage.starts_with("usage: chelys aac --sample-rate HZ --channels N"));
    assert!(usage.contains("--bitrate KBPS"));

    let output = chelys(&["--help"], &[]);
    assert!(output.status.success());
    assert!(
        text(&output.stdout).contains("\n  aac "),
        "aac among the codecs"
    );
}

#[test]
fn refuses_arguments_with_its_usage() {
    let output = chelys(&["aac", "--channels", "2"], &[]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let stderr = text(&output.stderr);
    assert!(
        stderr.starts_with("chelys: --sample-rate is required\n\nusage: chelys aac"),
        "{stderr}"
    );

    let output = chelys(
        &[
            "aac",
            "--sample-rate",
            "8000",
            "--channels",
            "2",
            "--bitrate",
            "128",
        ],
        &[],
    );
    assert_eq!(output.status.code(), Some(2));
    assert!(
        text(&output.stderr).starts_with(
            "chelys: unsupported bitrate at 8000 Hz in stereo: 128; it takes 1 to 95\n"
        )
    );
}

#[test]
fn refuses_input_that_ends_within_a_sample_frame() {
    let mut input = le_bytes(&stereo_tone(44_100, 0.1));
    input.extend(0.5f32.to_le_bytes());
    let output = chelys(
        &["aac", "--sample-rate", "44100", "--channels", "2"],
        &input,
    );
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        text(&output.stderr),
        "chelys: input is not a whole number of 8-byte sample frames\n"
    );
}

#[test]
fn refuses_input_that_is_not_a_number() {
    let input = le_bytes(&[0.0, f32::NAN]);
    let output = chelys(
        &["aac", "--sample-rate", "44100", "--channels", "2"],
        &input,
    );
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        text(&output.stderr),
        "chelys: input holds a sample that is not a finite number: NaN\n"
    );
}
