// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The `ffmpeg` and `ffprobe` programs.

use std::process::Command;

pub use crate::Decoded;
use crate::{TempFile, run};

#[derive(Debug, PartialEq, Eq)]
pub struct StreamInfo {
    pub codec_name: String,
    pub sample_rate: u32,
    pub channels: usize,
}

/// Decodes an MP3 stream with CRC checking enabled.
pub fn decode_mp3(stream: &[u8]) -> Decoded {
    let file = TempFile::new("mp3", stream);
    let output = run(Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-err_detect", "crccheck"])
        .args(["-f", "mp3", "-i"])
        .arg(file.path())
        .args(["-f", "f32le", "-acodec", "pcm_f32le", "-"]));
    Decoded {
        samples: output
            .stdout
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        diagnostics: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// Describes the first audio stream, as detected by ffprobe.
pub fn probe_mp3(stream: &[u8]) -> StreamInfo {
    let file = TempFile::new("mp3", stream);
    let output = run(Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a:0"])
        .args(["-show_entries", "stream=codec_name,sample_rate,channels"])
        .args(["-of", "default=noprint_wrappers=1"])
        .arg(file.path()));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let field = |key: &str| {
        stdout
            .lines()
            .find_map(|line| line.strip_prefix(key)?.strip_prefix('='))
            .unwrap_or_else(|| panic!("ffprobe did not report {key}: {stdout:?}"))
    };
    StreamInfo {
        codec_name: field("codec_name").to_owned(),
        sample_rate: field("sample_rate").parse().unwrap(),
        channels: field("channels").parse().unwrap(),
    }
}

/// The duration in seconds that ffprobe gives a stream.
pub fn duration_mp3(stream: &[u8]) -> f64 {
    let file = TempFile::new("mp3", stream);
    let output = run(Command::new("ffprobe")
        .args(["-v", "error", "-show_entries", "format=duration"])
        .args(["-of", "default=noprint_wrappers=1:nokey=1"])
        .arg(file.path()));
    let stdout = String::from_utf8(output.stdout).unwrap();
    stdout
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("ffprobe gave no duration: {stdout:?}"))
}

/// Decodes `seconds` of an MP3 stream from where ffmpeg seeks to for
/// `from` seconds by the stream's Xing table of contents (its `usetoc`
/// option), without decoding from earlier to get there exactly.
pub fn decode_mp3_seeking(stream: &[u8], from: f64, seconds: f64) -> Decoded {
    let file = TempFile::new("mp3", stream);
    let output = run(Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-v",
            "error",
            "-noaccurate_seek",
            "-usetoc",
            "1",
        ])
        .args(["-ss", &from.to_string(), "-f", "mp3", "-i"])
        .arg(file.path())
        .args(["-t", &seconds.to_string()])
        .args(["-f", "f32le", "-acodec", "pcm_f32le", "-"]));
    Decoded {
        samples: output
            .stdout
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        diagnostics: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// Decodes an AAC stream in ADTS, with CRC checking enabled, as MP3 is
/// decoded; though ffmpeg n9.0.2 does not check the `crc_check` of ADTS
/// (see `ffmpeg_skips_the_crc_unchecked` in the `aac_adts` test).
pub fn decode_aac(stream: &[u8]) -> Decoded {
    let file = TempFile::new("aac", stream);
    let output = run(Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-err_detect", "crccheck"])
        .args(["-f", "aac", "-i"])
        .arg(file.path())
        .args(["-f", "f32le", "-acodec", "pcm_f32le", "-"]));
    Decoded {
        samples: output
            .stdout
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        diagnostics: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// Describes the first audio stream of an AAC stream in ADTS, read as
/// ADTS, as ffprobe reports it, with the profile it names. Left to detect
/// the format, ffprobe takes a stream of a few frames for FLAC.
pub fn probe_aac(stream: &[u8]) -> (StreamInfo, String) {
    let file = TempFile::new("aac", stream);
    let output = run(Command::new("ffprobe")
        .args(["-v", "error", "-f", "aac", "-select_streams", "a:0"])
        .args([
            "-show_entries",
            "stream=codec_name,profile,sample_rate,channels",
        ])
        .args(["-of", "default=noprint_wrappers=1"])
        .arg(file.path()));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let field = |key: &str| {
        stdout
            .lines()
            .find_map(|line| line.strip_prefix(key)?.strip_prefix('='))
            .unwrap_or_else(|| panic!("ffprobe did not report {key}: {stdout:?}"))
    };
    let info = StreamInfo {
        codec_name: field("codec_name").to_owned(),
        sample_rate: field("sample_rate").parse().unwrap(),
        channels: field("channels").parse().unwrap(),
    };
    (info, field("profile").to_owned())
}

/// The channel layout that ffprobe reports for the first audio stream of
/// an AAC stream in ADTS, as ffmpeg names it, such as "5.1", and the
/// channels of ffmpeg's decoded output in their order, such as
/// "FL+FR+FC+LFE+BL+BR".
pub fn channel_layout_aac(stream: &[u8]) -> (String, String) {
    let file = TempFile::new("aac", stream);
    let output = run(Command::new("ffprobe")
        .args(["-v", "error", "-f", "aac", "-select_streams", "a:0"])
        .args(["-show_entries", "stream=channel_layout"])
        .args(["-of", "default=noprint_wrappers=1:nokey=1"])
        .arg(file.path()));
    let layout = String::from_utf8(output.stdout).unwrap().trim().to_owned();
    let channels = channels_of_layout(&layout);
    (layout, channels)
}

/// The channels of a channel layout that ffmpeg names, in order, as
/// `ffmpeg -layouts` lists them.
fn channels_of_layout(layout: &str) -> String {
    let output = run(Command::new("ffmpeg").args(["-hide_banner", "-layouts"]));
    let stdout = String::from_utf8(output.stdout).unwrap();
    stdout
        .lines()
        .skip_while(|line| !line.starts_with("Standard channel layouts"))
        .filter_map(|line| line.split_once(char::is_whitespace))
        .find(|(name, _)| *name == layout)
        .map(|(_, channels)| channels.trim().to_owned())
        .unwrap_or_else(|| panic!("ffmpeg lists no layout {layout}"))
}

/// The `AudioSpecificConfig()` that ffmpeg gives an AAC stream in ADTS
/// when it copies the stream into an MPEG-4 file, as ffprobe reports it
/// there.
pub fn audio_specific_config_from_adts(stream: &[u8]) -> Vec<u8> {
    let adts = TempFile::new("aac", stream);
    let mp4 = TempFile::new("mp4", &[]);
    run(Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-y", "-f", "aac", "-i"])
        .arg(adts.path())
        .args(["-c", "copy", "-f", "mp4"])
        .arg(mp4.path()));
    let output = run(Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a:0", "-show_data"])
        .args(["-show_entries", "stream=extradata"])
        .args(["-of", "default=noprint_wrappers=1"])
        .arg(mp4.path()));
    // A hex dump of lines such as "00000000: 1210  ..": an offset, the
    // bytes in groups of two, and the bytes as text.
    let stdout = String::from_utf8(output.stdout).unwrap();
    stdout
        .lines()
        .filter_map(|line| line.split_once(": "))
        .flat_map(|(_, dump)| {
            let hex = dump.get(..40).unwrap_or(dump);
            hex.split_whitespace()
                .flat_map(|group| (0..group.len()).step_by(2).map(move |i| &group[i..i + 2]))
                .map(|byte| u8::from_str_radix(byte, 16).unwrap())
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Decodes the first audio stream of an MP4 file, with `input_options`
/// before the file, as `-ss` or `-ignore_editlist`.
fn decode_mp4_with(file: &[u8], input_options: &[&str]) -> Decoded {
    let file = TempFile::new("mp4", file);
    let output = run(Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-err_detect", "crccheck"])
        .args(input_options)
        .arg("-i")
        .arg(file.path())
        .args(["-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", "-"]));
    Decoded {
        samples: output
            .stdout
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        diagnostics: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// Decodes the first audio stream of an MP4 file, as a player would: its
/// edit list applied.
pub fn decode_mp4(file: &[u8]) -> Decoded {
    decode_mp4_with(file, &[])
}

/// Decodes the first audio stream of an MP4 file with its edit list
/// ignored (ffmpeg's `ignore_editlist` option): every sample of every
/// frame.
pub fn decode_mp4_ignoring_edits(file: &[u8]) -> Decoded {
    decode_mp4_with(file, &["-ignore_editlist", "1"])
}

/// Decodes the first audio stream of an MP4 file from where ffmpeg seeks
/// to for `from` seconds, given as an input option, so that ffmpeg seeks
/// in the file rather than decode from its start.
pub fn decode_mp4_seeking(file: &[u8], from: f64) -> Decoded {
    decode_mp4_with(file, &["-ss", &from.to_string()])
}

/// What ffprobe reports of the first audio stream of an MP4 file.
#[derive(Debug, PartialEq, Eq)]
pub struct Mp4Stream {
    pub info: StreamInfo,
    pub profile: String,
    /// The channel layout, as ffmpeg names it, such as "5.1".
    pub channel_layout: String,
    /// The stream's duration in its time base's units.
    pub duration_ts: u64,
    /// The stream's time base, such as "1/44100".
    pub time_base: String,
}

/// Describes the first audio stream of an MP4 file, as ffprobe reports
/// it.
pub fn probe_mp4(file: &[u8]) -> Mp4Stream {
    let file = TempFile::new("mp4", file);
    let output = run(Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a:0", "-show_entries"])
        .arg("stream=codec_name,profile,sample_rate,channels,channel_layout,duration_ts,time_base")
        .args(["-of", "default=noprint_wrappers=1"])
        .arg(file.path()));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let field = |key: &str| {
        stdout
            .lines()
            .find_map(|line| line.strip_prefix(key)?.strip_prefix('='))
            .unwrap_or_else(|| panic!("ffprobe did not report {key}: {stdout:?}"))
    };
    Mp4Stream {
        info: StreamInfo {
            codec_name: field("codec_name").to_owned(),
            sample_rate: field("sample_rate").parse().unwrap(),
            channels: field("channels").parse().unwrap(),
        },
        profile: field("profile").to_owned(),
        channel_layout: field("channel_layout").to_owned(),
        duration_ts: field("duration_ts").parse().unwrap(),
        time_base: field("time_base").to_owned(),
    }
}
