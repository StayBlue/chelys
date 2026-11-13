// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The `lame` program, as a decoder.

use std::process::Command;

use crate::{Decoded, TempFile, run};

/// Decodes an MP3 stream. LAME writes 16-bit PCM, which is scaled to the
/// range -1.0 to 1.0.
pub fn decode_mp3(stream: &[u8]) -> Decoded {
    let file = TempFile::new("mp3", stream);
    let output = run(Command::new("lame")
        .args(["--decode", "-t", "--quiet"])
        .arg(file.path())
        .arg("-"));
    Decoded {
        samples: output
            .stdout
            .chunks_exact(2)
            .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0)
            .collect(),
        diagnostics: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}
