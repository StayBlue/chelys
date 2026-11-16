// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The `mpg123` program.

use std::process::Command;

use crate::{Decoded, TempFile, run, wav_sample_rate};

/// Decodes an MP3 stream to 32-bit float PCM.
pub fn decode_mp3(stream: &[u8]) -> Decoded {
    let file = TempFile::new("mp3", stream);
    let output = run(Command::new("mpg123")
        .args(["-q", "-e", "f32", "-s"])
        .arg(file.path()));
    Decoded {
        samples: output
            .stdout
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        diagnostics: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// The sampling frequency in Hz of the WAV file mpg123 decodes a stream to.
pub fn sample_rate(stream: &[u8]) -> u32 {
    let file = TempFile::new("mp3", stream);
    let output = run(Command::new("mpg123")
        .args(["-q", "-w", "-"])
        .arg(file.path()));
    wav_sample_rate(&output.stdout)
}

/// mpg123's account of a stream's first frame header, from its most
/// verbose output: the lines that begin "Channels:" and "Bitrate:", which
/// give `copyright`, `original/copy` and, as the "Extension value",
/// `private_bit`.
pub fn describe_header(stream: &[u8]) -> String {
    let file = TempFile::new("mp3", stream);
    let output = run(Command::new("mpg123")
        .args(["-vvv", "-t", "-n", "1"])
        .arg(file.path()));
    String::from_utf8_lossy(&output.stderr)
        .lines()
        .filter(|line| line.starts_with("Channels:") || line.starts_with("Bitrate:"))
        .collect::<Vec<_>>()
        .join("\n")
}
