// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The `flac` program, the reference FLAC encoder, run as a black box to
//! make the FLAC frames that `chelys-mp4`'s tests write in MP4 files: only
//! its output is used, split into frames by ffprobe (see
//! [`ffmpeg::packets`](crate::ffmpeg::packets)).

use std::process::Command;

use crate::{TempFile, run};

/// A FLAC file of `pcm`, interleaved samples of `channels` of `bits` bits
/// each, a whole number of bytes, at `sample_rate`, in blocks of
/// `block_size` samples. It is written to a file, so that `flac` can go
/// back and write its STREAMINFO's length and MD5 checksum, which it
/// cannot on a pipe.
pub fn encode(
    pcm: &[i32],
    sample_rate: u32,
    channels: usize,
    bits: u32,
    block_size: u32,
) -> Vec<u8> {
    assert!(bits % 8 == 0, "a whole number of bytes a sample");
    let width = bits as usize / 8;
    let raw: Vec<u8> = (pcm.iter())
        .flat_map(|sample| sample.to_le_bytes()[..width].to_vec())
        .collect();
    let input = TempFile::new("raw", &raw);
    let output = TempFile::new("flac", &[]);
    run(Command::new("flac")
        .args(["--silent", "--force", "--force-raw-format"])
        .args(["--endian=little", "--sign=signed"])
        .arg(format!("--channels={channels}"))
        .arg(format!("--bps={bits}"))
        .arg(format!("--sample-rate={sample_rate}"))
        .arg(format!("--blocksize={block_size}"))
        .arg("-o")
        .arg(output.path())
        .arg(input.path()));
    std::fs::read(output.path()).unwrap()
}
