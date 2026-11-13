//! Harness for checking chelys output against external decoders.
//!
//! The decoders are run as separate programs and treated strictly as black
//! boxes: only their observable output is used. A missing decoder is a test
//! failure, not a skip.
//!
//! Where the standard leaves a gap that a decoded test settled, the library
//! says "Settled by decoding with ffmpeg" and names the test. That evidence
//! was recorded with ffmpeg n9.0.2; another version could decode
//! differently.

pub mod ffmpeg;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

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
