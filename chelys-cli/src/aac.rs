// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! `chelys aac`: MPEG-2 AAC in the Low Complexity profile, in ADTS.

use chelys::aac::{Config, ConfigError, Encoder, Rate};

use crate::output::Output;
use crate::{pcm, value};

pub const USAGE: &str = "\
usage: chelys aac --sample-rate HZ --channels N [--bitrate KBPS]
                  < input.f32 > output.aac

Reads interleaved 32-bit little-endian float PCM from stdin and writes MPEG-2
AAC in the Low Complexity profile (AAC-LC) to stdout, each frame in ADTS
(experimental: the encoder is still being built).

  --sample-rate HZ   96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050,
                     16000, 12000, 11025 or 8000
  --channels N       1 or 2
  --bitrate KBPS     the stream's constant bitrate, ADTS headers included:
                     a whole number within the range of the sample rate and
                     channels, at 48 kHz 5 to 287 in mono and 6 to 575 in
                     stereo, at 44.1 kHz 4 to 264 and 5 to 528, and lower
                     with the sample rate, to 1 to 47 and 1 to 95 at 8 kHz
                     (default 128, or the highest the sample rate and
                     channels allow where that is lower)

The decoded output starts with 1024 samples before the input and ends with
fewer than 1024 after it, near silence to be discarded: ADTS has no field
that tells a decoder so.";

/// The bitrate without `--bitrate`, in kbit/s, where the sample rate and
/// channels allow it: that of `chelys mp3`.
const DEFAULT_KBPS: u32 = 128;

pub struct Options {
    sample_rate: u32,
    channels: usize,
    kbps: u32,
}

impl Options {
    fn config(&self) -> Config {
        config(self.sample_rate, self.channels, self.kbps)
    }
}

fn config(sample_rate: u32, channels: usize, kbps: u32) -> Config {
    // A bitrate beyond a u32 of bit/s saturates, which the encoder refuses
    // as out of range like any other, and `describe` gives in kbit/s.
    let bits_per_second = kbps.saturating_mul(1000);
    Config::new(sample_rate, channels, Rate::Constant { bits_per_second })
}

pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Options, String> {
    let (mut sample_rate, mut channels, mut bitrate) = (None, None, None);
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        let args = &mut args;
        match flag.as_str() {
            "--sample-rate" => sample_rate = Some(value::<u32>(args, &flag)?),
            "--channels" => channels = Some(value::<u32>(args, &flag)?),
            "--bitrate" => bitrate = Some(value::<u32>(args, &flag)?),
            // An empty message asks for the usage text alone.
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    let sample_rate = sample_rate.ok_or("--sample-rate is required")?;
    let channels = match channels.ok_or("--channels is required")? {
        n @ (1 | 2) => n as usize,
        n => return Err(format!("unsupported channel count: {n}")),
    };
    let kbps = bitrate.unwrap_or_else(|| default_kbps(sample_rate, channels));
    let options = Options {
        sample_rate,
        channels,
        kbps,
    };
    options
        .config()
        .check()
        .map_err(|e| describe(e, &options))?;
    Ok(options)
}

/// [`DEFAULT_KBPS`], or the highest whole bitrate the sample rate and
/// channels allow where that is lower.
fn default_kbps(sample_rate: u32, channels: usize) -> u32 {
    match config(sample_rate, channels, DEFAULT_KBPS).check() {
        Err(ConfigError::UnsupportedBitrate { max, .. }) if max < DEFAULT_KBPS * 1000 => max / 1000,
        _ => DEFAULT_KBPS,
    }
}

/// The message for a configuration the encoder does not take, in terms of
/// the command line's flags.
fn describe(error: ConfigError, options: &Options) -> String {
    match error {
        ConfigError::UnsupportedBitrate { min, max, .. } => {
            let channels = if options.channels == 1 {
                "mono"
            } else {
                "stereo"
            };
            format!(
                "unsupported bitrate at {} Hz in {channels}: {}; it takes {} to {}",
                options.sample_rate,
                options.kbps,
                min.div_ceil(1000),
                max / 1000,
            )
        }
        error => error.to_string(),
    }
}

/// Encodes stdin to stdout as it arrives.
pub fn encode(options: &Options) -> Result<(), String> {
    let mut output = Output::stdout();
    let mut encoder = Encoder::new(options.config()).map_err(|e| e.to_string())?;
    let mut stream = Vec::new();
    pcm::read(encoder.channels(), |samples| {
        encoder.encode_interleaved(samples, &mut stream);
        output.write(&mut stream)
    })?;
    encoder.finish(&mut stream);
    output.write(&mut stream)?;
    output.flush()
}

#[cfg(test)]
mod tests {
    use chelys::aac::SAMPLE_RATES;

    use super::*;

    fn parse(args: &[&str]) -> Result<Options, String> {
        super::parse(args.iter().map(|&a| a.to_owned()))
    }

    fn error(args: &[&str]) -> String {
        parse(args).err().expect("the arguments are refused")
    }

    #[test]
    fn help_asks_for_the_usage_alone() {
        assert_eq!(error(&["--help"]), "");
        assert_eq!(error(&["--sample-rate", "44100", "-h"]), "");
    }

    #[test]
    fn the_usage_lists_every_sample_rate() {
        let line = USAGE
            .lines()
            .skip_while(|l| !l.starts_with("  --sample-rate HZ"))
            .take(2)
            .collect::<String>();
        for rate in SAMPLE_RATES {
            assert!(line.contains(&rate.to_string()), "{rate} missing");
        }
    }

    #[test]
    fn the_sample_rate_and_channels_are_required() {
        assert_eq!(error(&["--channels", "2"]), "--sample-rate is required");
        assert_eq!(error(&["--sample-rate", "44100"]), "--channels is required");
    }

    #[test]
    fn refuses_what_it_cannot_parse() {
        assert_eq!(error(&["--vbr"]), "unknown argument: --vbr");
        assert_eq!(error(&["--bitrate"]), "--bitrate needs a value");
        assert_eq!(
            error(&["--bitrate", "128k"]),
            "--bitrate: not a number: 128k"
        );
    }

    #[test]
    fn refuses_what_the_encoder_does_not_take() {
        let args = |rate, channels, kbps| {
            error(&[
                "--sample-rate",
                rate,
                "--channels",
                channels,
                "--bitrate",
                kbps,
            ])
        };
        assert_eq!(args("44100", "3", "128"), "unsupported channel count: 3");
        assert_eq!(args("44000", "2", "128"), "unsupported sample rate: 44000");
        assert_eq!(
            args("44100", "2", "4"),
            "unsupported bitrate at 44100 Hz in stereo: 4; it takes 5 to 528"
        );
        assert_eq!(
            args("8000", "1", "48"),
            "unsupported bitrate at 8000 Hz in mono: 48; it takes 1 to 47"
        );
        // Beyond what a u32 holds in bit/s.
        assert_eq!(
            args("44100", "1", "4294968"),
            "unsupported bitrate at 44100 Hz in mono: 4294968; it takes 4 to 264"
        );
    }

    #[test]
    fn takes_the_ends_of_the_range() {
        for (rate, channels, kbps) in [("44100", "2", "5"), ("44100", "2", "528")] {
            let args = [
                "--sample-rate",
                rate,
                "--channels",
                channels,
                "--bitrate",
                kbps,
            ];
            assert!(parse(&args).is_ok(), "{args:?}");
        }
    }

    #[test]
    fn the_default_bitrate_is_128_where_the_sample_rate_allows() {
        let kbps = |rate: &str, channels| {
            parse(&["--sample-rate", rate, "--channels", channels])
                .unwrap()
                .kbps
        };
        assert_eq!(kbps("44100", "1"), 128);
        assert_eq!(kbps("44100", "2"), 128);
        assert_eq!(kbps("11025", "2"), 128);
        assert_eq!(kbps("8000", "1"), 47);
        assert_eq!(kbps("8000", "2"), 95);
        assert_eq!(kbps("16000", "1"), 95);
    }
}
