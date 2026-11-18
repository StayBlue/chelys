// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The input every codec takes: raw interleaved 32-bit little-endian float
//! PCM on stdin.

use std::io::{ErrorKind, Read};

/// Reads stdin to its end, giving each whole number of sample frames of
/// `channels` to `encode` as it arrives.
///
/// # Errors
///
/// If stdin cannot be read, holds a sample that is not a finite number, or
/// ends within a sample frame, or `encode` fails.
pub fn read(
    channels: usize,
    mut encode: impl FnMut(&[f32]) -> Result<(), String>,
) -> Result<(), String> {
    let frame_bytes = 4 * channels;
    let mut stdin = std::io::stdin().lock();
    let mut input = vec![0; 64 * 1024];
    // Bytes read but not yet decoded: the start of an incomplete sample frame.
    let mut held = 0;
    let mut pcm = Vec::new();
    loop {
        let read = match stdin.read(&mut input[held..]) {
            Ok(0) => break,
            Ok(read) => read,
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => return Err(format!("reading stdin: {e}")),
        };
        let filled = held + read;
        let whole = filled - filled % frame_bytes;
        pcm.clear();
        pcm.extend(
            (input[..whole].chunks_exact(4))
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())),
        );
        if let Some(sample) = pcm.iter().find(|s| !s.is_finite()) {
            return Err(format!(
                "input holds a sample that is not a finite number: {sample}"
            ));
        }
        input.copy_within(whole..filled, 0);
        held = filled - whole;
        encode(&pcm)?;
    }
    match held {
        0 => Ok(()),
        _ => Err(format!(
            "input is not a whole number of {frame_bytes}-byte sample frames"
        )),
    }
}
