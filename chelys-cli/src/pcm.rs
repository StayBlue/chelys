// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The input every codec takes: raw interleaved 32-bit little-endian float
//! PCM on stdin.

use std::io::{ErrorKind, Read};

// The codecs' encoders refuse the same type, whichever names it.
use chelys::mp3::NonFiniteSample;

/// Why `encode` stopped: an encoder refused a sample, which [`read`] names
/// in the whole of the input, or anything else.
pub enum Error {
    /// The sample's position must be relative to the whole slice `encode`
    /// was given, as one encoder call on it reports it.
    NonFinite(NonFiniteSample),
    Other(String),
}

impl From<NonFiniteSample> for Error {
    fn from(e: NonFiniteSample) -> Self {
        Self::NonFinite(e)
    }
}

impl From<String> for Error {
    fn from(message: String) -> Self {
        Self::Other(message)
    }
}

/// Reads stdin to its end, giving each whole number of sample frames of
/// `channels` to `encode` as it arrives. Whether the samples are finite is
/// left to the encoder `encode` gives them to.
///
/// # Errors
///
/// If stdin cannot be read or ends within a sample frame, or `encode`
/// fails: where it fails on a sample that is not a finite number, the
/// message gives the sample's value and its place in the whole input.
pub fn read(
    channels: usize,
    mut encode: impl FnMut(&[f32]) -> Result<(), Error>,
) -> Result<(), String> {
    let frame_bytes = 4 * channels;
    let mut stdin = std::io::stdin().lock();
    let mut input = vec![0; 64 * 1024];
    // Bytes read but not yet decoded: the start of an incomplete sample frame.
    let mut held = 0;
    // Sample frames given to `encode` before `pcm`: a `u64`, as a long input
    // passes `usize::MAX` on a 32-bit target.
    let mut frames: u64 = 0;
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
        input.copy_within(whole..filled, 0);
        held = filled - whole;
        match encode(&pcm) {
            Ok(()) => {}
            Err(Error::NonFinite(e)) => {
                let value = pcm[e.sample * channels + e.channel];
                return Err(format!(
                    "input sample {} of channel {} is not a finite number: {value}",
                    frames + e.sample as u64,
                    e.channel,
                ));
            }
            Err(Error::Other(message)) => return Err(message),
        }
        frames += (pcm.len() / channels) as u64;
    }
    match held {
        0 => Ok(()),
        _ => Err(format!(
            "input is not a whole number of {frame_bytes}-byte sample frames"
        )),
    }
}
