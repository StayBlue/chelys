// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Harness for checking chelys output against external decoders.
//!
//! The decoders are run as separate programs and treated strictly as black
//! boxes: only their observable output is used. A missing decoder is a test
//! failure, not a skip.
//!
//! Where the standard leaves a gap that a decoded test settled, the library
//! says "Settled by decoding with" the decoders that settled it and names
//! the test. That evidence was recorded with ffmpeg n9.0.2, mpg123 1.33.7
//! and LAME 4.0 (64-bit); other versions could decode differently.

pub mod ffmpeg;
pub mod lame;
pub mod mpg123;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A decoder, run as a black box, with the name it is reported by.
#[derive(Clone, Copy)]
pub struct Decoder {
    pub name: &'static str,
    pub decode: fn(&[u8]) -> Decoded,
}

/// Every decoder that MP3 streams are checked against.
pub const DECODERS: [Decoder; 3] = [
    Decoder {
        name: "ffmpeg",
        decode: ffmpeg::decode_mp3,
    },
    Decoder {
        name: "mpg123",
        decode: mpg123::decode_mp3,
    },
    Decoder {
        name: "LAME",
        decode: lame::decode_mp3,
    },
];

/// A decoded stream.
pub struct Decoded {
    /// Interleaved 32-bit float PCM, as the decoder produced it, or converted
    /// to it from the decoder's integer samples.
    pub samples: Vec<f32>,
    /// Anything the decoder logged, at error level where it has levels.
    pub diagnostics: String,
}

/// A file in the system temporary directory, removed on drop.
struct TempFile(PathBuf);

impl TempFile {
    fn new(extension: &str, contents: &[u8]) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let name = format!(
            "chelys-{}-{}.{extension}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, contents).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Runs `command` to completion, and panics if it cannot be run or fails.
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
