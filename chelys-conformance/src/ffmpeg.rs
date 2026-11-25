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

/// Decodes the first audio stream of an MP4 file from the sample ffmpeg
/// seeks to for `from` seconds, without decoding from an earlier one and
/// discarding samples to reach `from` exactly (`-noaccurate_seek`): its
/// output starts with the first sample it decodes after seeking.
pub fn decode_mp4_from_seek_point(file: &[u8], from: f64) -> Decoded {
    decode_mp4_with(file, &["-noaccurate_seek", "-ss", &from.to_string()])
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

/// A stream's packets as ffprobe splits them, each with its duration in
/// the stream's time base, and the stream's extradata, the codec's
/// configuration: the data of the STREAMINFO block for FLAC, the
/// identification header for Opus, the `AudioSpecificConfig()` for AAC,
/// and for ALAC the box that holds the magic cookie in an MP4 file.
pub struct Packets {
    pub config: Vec<u8>,
    pub packets: Vec<Vec<u8>>,
    pub durations: Vec<u64>,
    /// The samples a decoder's output starts with before the input, as
    /// ffprobe reports them (`initial_padding`): the encoder's delay.
    pub initial_padding: u64,
}

/// The packets of the first audio stream of `file`, a file of the format
/// `extension` names, from ffprobe's hex dumps of them. Only that text is
/// parsed, never the file's format.
pub fn packets(file: &[u8], extension: &str) -> Packets {
    let file = TempFile::new(extension, file);
    let output = run(Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a:0", "-show_data"])
        .args([
            "-show_entries",
            "packet=duration,data:stream=extradata,initial_padding",
        ])
        .arg(file.path()));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut config = Vec::new();
    let mut packets: Vec<Vec<u8>> = Vec::new();
    let mut durations = Vec::new();
    let mut initial_padding = None;
    let mut in_stream = false;
    for line in stdout.lines() {
        if line == "[PACKET]" {
            packets.push(Vec::new());
        } else if line == "[STREAM]" {
            in_stream = true;
        } else if let Some(duration) = line.strip_prefix("duration=") {
            durations.push(duration.parse().unwrap());
        } else if let Some(padding) = line.strip_prefix("initial_padding=") {
            initial_padding = Some(padding.parse().unwrap());
        } else if let Some((_offset, dump)) = line.split_once(": ") {
            // The hex in groups of four digits, then two spaces and the
            // bytes as text.
            let hex: String = dump.split("  ").next().unwrap().split(' ').collect();
            let bytes = (0..hex.len())
                .step_by(2)
                .map(|at| u8::from_str_radix(&hex[at..at + 2], 16).unwrap());
            if in_stream {
                config.extend(bytes);
            } else {
                packets.last_mut().unwrap().extend(bytes);
            }
        }
    }
    assert_eq!(packets.len(), durations.len(), "{stdout}");
    Packets {
        config,
        packets,
        durations,
        initial_padding: initial_padding.expect("ffprobe reports initial_padding"),
    }
}

/// An Ogg Opus file of `pcm`, interleaved samples of `channels` at 48 kHz,
/// as ffmpeg's libopus encoder makes it at `kbps`, run as a black box to
/// make the packets that `chelys-mp4`'s tests write in MP4 files.
pub fn encode_opus(pcm: &[f32], channels: usize, kbps: u32) -> Vec<u8> {
    let raw: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
    let input = TempFile::new("f32", &raw);
    let output = run(Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-f", "f32le", "-ar", "48000"])
        .args(["-ac", &channels.to_string(), "-i"])
        .arg(input.path())
        .args([
            "-c:a",
            "libopus",
            "-b:a",
            &format!("{kbps}k"),
            "-f",
            "ogg",
            "-",
        ]));
    output.stdout
}

/// An MP4 file of `pcm`, interleaved samples of `channels` at
/// `sample_rate`, as ffmpeg's native AAC encoder makes it at `kbps`, run as
/// a black box to make the packets that `chelys-mp4`'s tests write in MP4
/// files.
pub fn encode_aac(pcm: &[f32], sample_rate: u32, channels: usize, kbps: u32) -> Vec<u8> {
    let raw: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
    let input = TempFile::new("f32", &raw);
    // ffmpeg writes an MP4 file whole only where it can seek.
    let output = TempFile::new("m4a", &[]);
    run(Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-y", "-f", "f32le"])
        .args(["-ar", &sample_rate.to_string()])
        .args(["-ac", &channels.to_string(), "-i"])
        .arg(input.path())
        .args(["-c:a", "aac", "-b:a", &format!("{kbps}k")])
        .arg(output.path()));
    std::fs::read(output.path()).unwrap()
}

/// An MP4 file of `pcm`, interleaved `bits`-bit integers, 16 or 24, of
/// `channels` at `sample_rate`, as ffmpeg's ALAC encoder makes it, run as a
/// black box to make the packets that `chelys-mp4`'s tests write in MP4
/// files.
pub fn encode_alac(pcm: &[i32], sample_rate: u32, channels: usize, bits: u32) -> Vec<u8> {
    let (raw, format, sample_format): (Vec<u8>, _, _) = match bits {
        16 => (
            pcm.iter().flat_map(|&s| (s as i16).to_le_bytes()).collect(),
            "s16le",
            "s16p",
        ),
        24 => (
            pcm.iter().flat_map(|&s| (s << 8).to_le_bytes()).collect(),
            "s32le",
            "s32p",
        ),
        _ => panic!("ffmpeg's ALAC encoder takes 16 or 24 bits, not {bits}"),
    };
    let input = TempFile::new("pcm", &raw);
    // ffmpeg writes an MP4 file whole only where it can seek.
    let output = TempFile::new("m4a", &[]);
    run(Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-y", "-f", format])
        .args(["-ar", &sample_rate.to_string()])
        .args(["-ac", &channels.to_string(), "-i"])
        .arg(input.path())
        .args(["-c:a", "alac", "-sample_fmt", sample_format])
        .arg(output.path()));
    std::fs::read(output.path()).unwrap()
}

/// Decodes the first audio stream of a file of the format `extension`
/// names, as ffmpeg presents it.
pub fn decode_file(file: &[u8], extension: &str) -> Decoded {
    let file = TempFile::new(extension, file);
    let output = run(Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-i"])
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

/// Decodes the first audio stream of an Ogg stream, read as Ogg. ffmpeg
/// n9.0.2 checks the CRC of each page whatever its `err_detect` option (see
/// `ffmpeg_reports_a_page_whose_crc_is_wrong` in the `vorbis_silence`
/// test). Where ffmpeg fails, as on a stream it cannot open, its diagnostics
/// end with its exit status, rather than the call panicking, so that tests
/// can show what it refuses.
pub fn decode_ogg(stream: &[u8]) -> Decoded {
    let file = TempFile::new("ogg", stream);
    let output = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-f", "ogg", "-i"])
        .arg(file.path())
        .args(["-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", "-"])
        .output()
        .unwrap_or_else(|e| panic!("could not run ffmpeg (is it on PATH?): {e}"));
    let mut diagnostics = String::from_utf8_lossy(&output.stderr).into_owned();
    if !output.status.success() {
        diagnostics.push_str(&format!("ffmpeg failed: {}", output.status));
    }
    Decoded {
        samples: output
            .stdout
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        diagnostics,
    }
}

/// Describes the first audio stream of an Ogg stream, read as Ogg, as
/// ffprobe reports it.
pub fn probe_ogg(stream: &[u8]) -> StreamInfo {
    let file = TempFile::new("ogg", stream);
    let output = run(Command::new("ffprobe")
        .args(["-v", "error", "-f", "ogg", "-select_streams", "a:0"])
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
