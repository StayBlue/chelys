// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The `mpg123` program.

use std::process::Command;

use crate::{Decoded, TempFile, run};

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
