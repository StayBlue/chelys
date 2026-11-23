// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! `chelys mp3` run as a program: its errors on raw PCM on stdin.

use std::io::Write;
use std::process::{Command, Output, Stdio};

/// Runs `chelys` with `args`, `stdin` as its input.
fn chelys(args: &[&str], stdin: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_chelys"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("chelys runs");
    let mut input = child.stdin.take().unwrap();
    let stdin = stdin.to_vec();
    // Written apart from the reading of stdout, which a long input would
    // otherwise fill while chelys waits for the rest of it.
    let writer = std::thread::spawn(move || {
        // chelys stops reading its input at the sample it refuses.
        let _ = input.write_all(&stdin);
    });
    let output = child.wait_with_output().expect("chelys ends");
    writer.join().unwrap();
    output
}

#[test]
fn refuses_input_that_is_not_a_number() {
    // Past the first 64 KiB chelys reads, where the encoder is given the
    // input in pieces: the sample is named in the whole of it.
    let mut pcm = vec![0.25f32; 44_100];
    pcm[30_000] = f32::NEG_INFINITY;
    let input: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
    let output = chelys(
        &["mp3", "--sample-rate", "44100", "--channels", "1"],
        &input,
    );
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        std::str::from_utf8(&output.stderr).unwrap(),
        "chelys: input sample 30000 of channel 0 is not a finite number: -inf\n"
    );
}
