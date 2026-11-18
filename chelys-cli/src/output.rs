// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Where every codec writes its stream: stdout, and where stdout is a file,
//! back over the stream's start once it has ended, for what is known only
//! then.

use std::fs::File;
use std::io::{Seek, SeekFrom, StdoutLock, Write};

pub struct Output {
    stdout: StdoutLock<'static>,
    /// stdout as a file, if it is a regular file at its start.
    file: Option<File>,
}

impl Output {
    pub fn stdout() -> Self {
        Self {
            file: seekable_stdout(),
            stdout: std::io::stdout().lock(),
        }
    }

    /// Whether [`rewrite_start`](Self::rewrite_start) can write over the
    /// start of the stream.
    pub fn can_rewrite_start(&self) -> bool {
        self.file.is_some()
    }

    /// Writes `stream` and empties it.
    pub fn write(&mut self, stream: &mut Vec<u8>) -> Result<(), String> {
        self.stdout
            .write_all(stream)
            .map_err(|e| format!("writing stdout: {e}"))?;
        stream.clear();
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), String> {
        self.stdout
            .flush()
            .map_err(|e| format!("writing stdout: {e}"))
    }

    /// Writes `bytes`, `what` they are, over the start of the stream, if
    /// stdout is a file. A file opened for appending takes the write at its
    /// end instead; that is undone, and the stream keeps what it started
    /// with.
    pub fn rewrite_start(&mut self, bytes: &[u8], what: &str) -> Result<(), String> {
        let Some(file) = &mut self.file else {
            return Ok(());
        };
        let error = |e: std::io::Error| format!("writing {what}: {e}");
        let len = file.metadata().map_err(error)?.len();
        file.seek(SeekFrom::Start(0)).map_err(error)?;
        file.write_all(bytes).map_err(error)?;
        if file.stream_position().map_err(error)? != bytes.len() as u64 {
            file.set_len(len).map_err(error)?;
            eprintln!("chelys: stdout is appended to, so {what} is left out");
        }
        Ok(())
    }
}

/// stdout, if it is a regular file at its start.
#[cfg(unix)]
fn seekable_stdout() -> Option<File> {
    use std::os::fd::AsFd;
    let fd = std::io::stdout().as_fd().try_clone_to_owned().ok()?;
    let mut file = File::from(fd);
    let at_start = file.stream_position().ok()? == 0;
    (file.metadata().ok()?.is_file() && at_start).then_some(file)
}

#[cfg(not(unix))]
fn seekable_stdout() -> Option<File> {
    None
}
