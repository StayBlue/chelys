// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The `faad` program, FAAD2's decoder.
//!
//! Observed of FAAD2 2.11.4 as a black box:
//!
//! - Above 24 kHz it gives every frame's 1024 samples from the first frame
//!   on, as ffmpeg does, and trims nothing from a stream in ADTS, gapless
//!   or not. At 24 kHz and below it gives twice as many, at twice the
//!   sampling frequency, which its WAV header gives, and later: every other
//!   sample of them, from the first, gives the input 481 samples later than
//!   ffmpeg does, and at some rates less closely (see
//!   `faad_decodes_the_input_a_frame_late_at_every_sample_rate`).
//! - It writes its banner, a description of the stream and its progress to
//!   standard error, and, at a frame it cannot decode, a line that begins
//!   "Error:", after which it decodes no more of the stream; it still exits
//!   with status 0. Its option `-q` silences the errors with the rest, so
//!   it is not given, and the rest is taken out of what it writes.
//! - It does not check the `crc_check` of ADTS, and has no option to (see
//!   `faad_skips_the_crc_unchecked`).
//! - It decodes no `adts_frame()` longer than 4608 bytes (see
//!   `faad_decodes_no_frame_longer_than_4608_bytes`), to which the encoder
//!   keeps every frame in ADTS, those of 7.1 shorter than the standard
//!   allows.
//! - It gives the channels of 5.1 in ffmpeg's order, and those of the other
//!   channel configurations in the order the stream sends them (see
//!   `faad_maps_each_channel_of_configurations_3_to_7`).

use std::process::Command;

use crate::{Decoded, TempFile, run, wav_sample_rate};

/// Decodes an AAC stream in ADTS to 32-bit float PCM.
pub fn decode_aac(stream: &[u8]) -> Decoded {
    let file = TempFile::new("aac", stream);
    let output = run(Command::new("faad")
        .args(["-w", "-f", "2", "-b", "4"])
        .arg(file.path()));
    Decoded {
        samples: output
            .stdout
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        diagnostics: diagnostics(&String::from_utf8_lossy(&output.stderr)),
    }
}

/// The sampling frequency in Hz of the WAV file faad decodes a stream to.
pub fn sample_rate(stream: &[u8]) -> u32 {
    let file = TempFile::new("aac", stream);
    let output = run(Command::new("faad").args(["-q", "-w"]).arg(file.path()));
    wav_sample_rate(&output.stdout)
}

/// What faad wrote to standard error besides its banner, between lines of
/// asterisks, its description of the stream, after a line that ends "file
/// info:" and in a table of the channels, and its progress, each update
/// ended by a carriage return: the lines left, one to a line.
fn diagnostics(stderr: &str) -> String {
    let mut lines = stderr.split(['\r', '\n']).map(str::trim_end);
    let mut kept = Vec::new();
    let mut in_banner = false;
    while let Some(line) = lines.next() {
        if line.trim_start().starts_with("****") {
            in_banner = !in_banner;
        } else if line.ends_with("file info:") {
            lines.next();
        } else if !(in_banner
            || line.is_empty()
            || line.starts_with(" |")
            || line.starts_with("  ---")
            || is_progress(line))
        {
            kept.push(line);
        }
    }
    kept.join("\n")
}

/// Whether `line` is one of faad's reports of its progress: "12% decoding
/// file.aac." or "Decoding file.aac took: ...".
fn is_progress(line: &str) -> bool {
    let percent = line.trim_start_matches(|c: char| c.is_ascii_digit());
    percent.len() < line.len() && percent.starts_with("% decoding ")
        || line.starts_with("Decoding ") && line.contains(" took: ")
}
