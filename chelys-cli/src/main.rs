// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Encodes raw PCM from stdin on stdout, with the codec named first. Input
//! formats are left to other tools, for example:
//!
//! ```sh
//! ffmpeg -i song.flac -f f32le -ar 44100 -ac 2 - | chelys mp3 --sample-rate 44100 --channels 2 > song.mp3
//! ```

mod aac;
mod mp3;
mod output;
mod pcm;

use std::process::ExitCode;
use std::str::FromStr;

const USAGE: &str = "\
usage: chelys CODEC [OPTIONS] < input.f32 > output

Reads interleaved 32-bit little-endian float PCM from stdin and writes it
to stdout, encoded with CODEC:

  aac   MPEG-2 AAC in the Low Complexity profile, in ADTS or an MP4 file
  mp3   MPEG-1 Layer III, MPEG-2 Layer III at its lower sampling
        frequencies, and MPEG-2.5

Run chelys CODEC --help for its options.";

/// The value after the flag `name`, parsed.
fn value<T: FromStr>(args: &mut impl Iterator<Item = String>, name: &str) -> Result<T, String> {
    let value = args.next().ok_or(format!("{name} needs a value"))?;
    value
        .parse()
        .map_err(|_| format!("{name}: not a number: {value}"))
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("aac") => run(aac::USAGE, aac::parse(args), |o| aac::encode(&o)),
        Some("mp3") => run(mp3::USAGE, mp3::parse(args), |o| mp3::encode(&o)),
        Some("-h" | "--help") => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some(flag) if flag.starts_with('-') => usage_error(&format!(
            "the codec comes first, as in: chelys mp3 {flag} …"
        )),
        Some(other) => usage_error(&format!("unknown codec: {other}")),
        None => usage_error("a codec is required"),
    }
}

fn usage_error(message: &str) -> ExitCode {
    eprintln!("chelys: {message}\n\n{USAGE}");
    ExitCode::from(2)
}

/// Encodes with `encode` the options a codec parsed, or reports why it did
/// not: an empty message asks for the codec's usage text alone.
fn run<O>(
    usage: &str,
    parsed: Result<O, String>,
    encode: impl FnOnce(O) -> Result<(), String>,
) -> ExitCode {
    let options = match parsed {
        Ok(options) => options,
        Err(message) if message.is_empty() => {
            println!("{usage}");
            return ExitCode::SUCCESS;
        }
        Err(message) => {
            eprintln!("chelys: {message}\n\n{usage}");
            return ExitCode::from(2);
        }
    };
    match encode(options) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("chelys: {message}");
            ExitCode::FAILURE
        }
    }
}
