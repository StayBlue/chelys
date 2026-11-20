// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Judges chelys's output against its input: by blind listening tests, in
//! which an ABX session plays the input (A), its encoding (B) and, in each
//! trial, one of the two at random (X), and counts how often the listener
//! tells which X is; or by PEAQ's grade. The encoding is decoded with
//! ffmpeg, played with ffplay and graded with GstPEAQ, all as black boxes.
//! Another encoder's output, such as LAME's or ffmpeg's AAC encoder's when
//! it is the benchmark, is judged the same way, lined up by its own delay.

mod abx;
mod clip;
mod player;

use std::io::Write;
use std::process::ExitCode;

use chelys_conformance::peaq;
use clip::Clip;

const USAGE: &str = "\
usage: chelys-quality abx --sample-rate HZ --channels N [--trials N]
                          [--from SECONDS] [--to SECONDS] [--delay SAMPLES]
                          ORIGINAL.f32 ENCODED.mp3
       chelys-quality diff --sample-rate HZ --channels N
                           [--from SECONDS] [--to SECONDS] [--delay SAMPLES]
                           ORIGINAL.f32 ENCODED.mp3 > difference.f32
       chelys-quality grade --sample-rate HZ --channels N
                            [--from SECONDS] [--to SECONDS] [--delay SAMPLES]
                            ORIGINAL.f32 ENCODED.mp3
       chelys-quality lag --sample-rate HZ --channels N
                          ORIGINAL.f32 ENCODED.mp3

ORIGINAL.f32 is interleaved 32-bit little-endian float PCM, as the chelys
encoder reads it, and ENCODED.mp3 is what an encoder made of it: chelys, or
another encoder judged beside it; an encoding named .aac is read as AAC in
ADTS instead. ffmpeg decodes the encoding, ffplay plays the
sound, and GstPEAQ grades it, through gst-launch-1.0; each must be installed
for the commands that use it.

  abx   a blind ABX test: in each trial, play A (the original), B (the
        encoding) and X (one of the two at random) as often as needed, then
        say which X is; the score, and how likely it is to come from
        guessing, are given at the end
  diff  write the encoding less the original, the noise that coding added,
        as interleaved 32-bit float PCM to stdout
  grade the encoding with PEAQ (ITU-R BS.1387, basic version): the
        Objective Difference Grade, from 0 (imperceptible) down to -4 (very
        annoying), and the bandwidths of the original and the encoding,
        which weigh heavily in it
  lag   print the delay, in samples per channel from 0 to 8192, by which
        ffmpeg's decoding of the encoding lags the original, found where
        the two correlate best; give it to the others with --delay

  --sample-rate HZ  the sampling frequency of both
  --channels N      1 or 2
  --trials N        with abx, the number of trials (default 16)
  --from SECONDS    use only the part from this time on (default: the start)
  --to SECONDS      use only the part up to this time (default: the end)
  --delay SAMPLES   the samples per channel by which ffmpeg's decoding of
                    the encoding lags the original (default: chelys's
                    1057 for MP3 and 1024 for AAC, which ffmpeg does not
                    remove)

To learn what coding noise sounds like, first take the test with an encoding
at a low bitrate, such as 32 kbit/s, and raise the bitrate from there.";

enum Command {
    Abx { trials: usize },
    Diff,
    Grade,
    Lag,
}

struct Options {
    command: Command,
    sample_rate: u32,
    channels: usize,
    original: String,
    encoded: String,
    from: Option<f64>,
    to: Option<f64>,
    delay: usize,
}

/// The value after the flag `name`, parsed.
fn value<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<T, String> {
    let value = args.next().ok_or(format!("{name} needs a value"))?;
    value
        .parse()
        .map_err(|_| format!("{name}: not a number: {value}"))
}

fn parse(args: impl IntoIterator<Item = String>) -> Result<Options, String> {
    let mut args = args.into_iter();
    let command = args.next().ok_or("no command given")?;
    let (mut sample_rate, mut channels, mut trials) = (None, None, None);
    let (mut from, mut to, mut delay) = (None, None, None);
    let mut paths = Vec::new();
    while let Some(arg) = args.next() {
        let args = &mut args;
        match arg.as_str() {
            "--sample-rate" => sample_rate = Some(value::<u32>(args, &arg)?),
            "--channels" => channels = Some(value::<usize>(args, &arg)?),
            "--trials" => trials = Some(value::<usize>(args, &arg)?),
            "--from" => from = Some(value::<f64>(args, &arg)?),
            "--to" => to = Some(value::<f64>(args, &arg)?),
            "--delay" => delay = Some(value::<usize>(args, &arg)?),
            "-h" | "--help" => return Err(String::new()),
            flag if flag.starts_with("--") => return Err(format!("unknown argument: {flag}")),
            _ => paths.push(arg),
        }
    }
    let command = match command.as_str() {
        "abx" => Command::Abx {
            trials: match trials.unwrap_or(16) {
                0 => return Err("--trials must be at least 1".to_owned()),
                n => n,
            },
        },
        "diff" | "grade" | "lag" if trials.is_some() => {
            return Err("--trials needs abx".to_owned());
        }
        "lag" if from.is_some() || to.is_some() || delay.is_some() => {
            return Err("lag takes no --from, --to or --delay".to_owned());
        }
        "diff" => Command::Diff,
        "grade" => Command::Grade,
        "lag" => Command::Lag,
        "-h" | "--help" => return Err(String::new()),
        other => return Err(format!("unknown command: {other}")),
    };
    let sample_rate = sample_rate.ok_or("--sample-rate is required")?;
    let channels = match channels.ok_or("--channels is required")? {
        n @ (1 | 2) => n,
        n => return Err(format!("unsupported channel count: {n}")),
    };
    let [original, encoded] = <[String; 2]>::try_from(paths)
        .map_err(|_| "give the original and the encoding, in that order".to_owned())?;
    let delay = delay.unwrap_or(if clip::is_aac(&encoded) {
        chelys::aac::DELAY
    } else {
        chelys::mp3::DELAY
    });
    Ok(Options {
        command,
        sample_rate,
        channels,
        original,
        encoded,
        from,
        to,
        delay,
    })
}

fn main() -> ExitCode {
    let options = match parse(std::env::args().skip(1)) {
        Ok(options) => options,
        Err(message) if message.is_empty() => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(message) => {
            eprintln!("chelys-quality: {message}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let result = if let Command::Lag = options.command {
        clip::lag(
            options.sample_rate,
            options.channels,
            &options.original,
            &options.encoded,
        )
        .map(|lag| println!("{lag}"))
    } else {
        Clip::load(
            options.sample_rate,
            options.channels,
            &options.original,
            &options.encoded,
            options.delay,
        )
        .and_then(|clip| clip.section(options.from, options.to))
        .and_then(|clip| match options.command {
            Command::Abx { trials } => abx::run(&clip, trials),
            Command::Diff => write_difference(&clip),
            Command::Grade => {
                print_grade(&clip);
                Ok(())
            }
            Command::Lag => unreachable!("lag needs no clip"),
        })
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("chelys-quality: {message}");
            ExitCode::FAILURE
        }
    }
}

fn write_difference(clip: &Clip) -> Result<(), String> {
    let bytes: Vec<u8> = clip
        .difference()
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect();
    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(&bytes)
        .and_then(|()| stdout.flush())
        .map_err(|e| format!("writing stdout: {e}"))
}

fn print_grade(clip: &Clip) {
    let grade = peaq::grade(
        &clip.original,
        &clip.encoded,
        clip.sample_rate,
        clip.channels,
    );
    let khz = |lines: f64| lines * peaq::LINE_HZ / 1000.0;
    println!(
        "Objective Difference Grade: {:.3}\nBandwidth: {:.1} kHz in the original, {:.1} kHz in the encoding",
        grade.odg,
        khz(grade.bandwidth_reference),
        khz(grade.bandwidth_test),
    );
}
