// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The `ffmpeg` and `ffprobe` programs.

use std::process::{Command, Output};

use crate::TempFile;

pub struct Decoded {
    /// Interleaved 32-bit float PCM, as the decoder produced it.
    pub samples: Vec<f32>,
    /// Anything the decoder logged at error level.
    pub diagnostics: String,
}

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

fn run(command: &mut Command) -> Output {
    let program = command.get_program().to_string_lossy().into_owned();
    let output = command
        .output()
        .unwrap_or_else(|e| panic!("could not run {program} (is it on PATH?): {e}"));
    assert!(
        output.status.success(),
        "{program} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}
