// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! `chelys vorbis`: Vorbis I in an Ogg stream.

use std::hash::{BuildHasher, RandomState};

use chelys::ogg::vorbis::Encoder;
use chelys::vorbis::quality::MAX_LEVEL;
use chelys::vorbis::{Config, ConfigError, MAX_CHANNELS, Rate};

use crate::output::Output;
use crate::{pcm, value};

pub const USAGE: &str = "\
usage: chelys vorbis --sample-rate HZ --channels N
                     [--quality LEVEL | --vbr-noise DB]
                     [--serial-number N]
                     < input.f32 > output.ogg

Reads interleaved 32-bit little-endian float PCM from stdin and writes
Vorbis I to stdout, in an Ogg stream written as the input arrives, which
tells a player the input's exact length.

  --sample-rate HZ   any rate in Hz but 0
  --channels N       1 to 8, interleaved in the order the stream sends them:
                       1  mono
                       2  left, right
                       3  left, center, right
                       4  front left, front right, rear left, rear right
                       5  front left, center, front right, rear left,
                          rear right
                       6  as 5, then LFE
                       7  front left, center, front right, side left,
                          side right, rear center, LFE
                       8  front left, center, front right, side left,
                          side right, rear left, rear right, LFE
  --quality LEVEL    the level of quality, from 0 to 100, higher for better
                     quality and mostly larger files, on a scale shared
                     with chelys mp3 --vbr and chelys aac --vbr: for music
                     in stereo at 44.1 kHz about 102 kbit/s at 50, 166 at
                     80 and 189 at 100 (default 80)
  --vbr-noise DB     in place of --quality, allow each line this many dB
                     more noise than the psychoacoustic model does, or less
                     if negative: about 137 kbit/s at -7.5 and 67 at 0 in
                     stereo at 44.1 kHz
  --serial-number N  the Ogg stream's serial number, from 0 to 4294967295,
                     which streams chained one after another must not share
                     (default: drawn at random)";

/// The level of quality coded at unless `--quality` or `--vbr-noise` says
/// otherwise: that of `chelys mp3 --vbr` and `chelys aac --vbr`.
const DEFAULT_QUALITY: u8 = 80;

pub struct Options {
    sample_rate: u32,
    channels: usize,
    rate: Rate,
    serial_number: u32,
}

impl Options {
    fn config(&self) -> Config {
        Config::new(self.sample_rate, self.channels, self.rate)
    }
}

pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Options, String> {
    let (mut sample_rate, mut channels, mut quality, mut vbr_noise) = (None, None, None, None);
    let mut serial_number = None;
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        let args = &mut args;
        match flag.as_str() {
            "--sample-rate" => sample_rate = Some(value::<u32>(args, &flag)?),
            "--channels" => channels = Some(value::<u32>(args, &flag)?),
            "--quality" => quality = Some(value::<u32>(args, &flag)?),
            "--vbr-noise" => vbr_noise = Some(value::<f64>(args, &flag)?),
            "--serial-number" => serial_number = Some(value::<u32>(args, &flag)?),
            // An empty message asks for the usage text alone.
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    let sample_rate = sample_rate.ok_or("--sample-rate is required")?;
    let channels = channels.ok_or("--channels is required")? as usize;
    let rate = match (quality, vbr_noise) {
        (Some(_), Some(_)) => {
            return Err("--quality and --vbr-noise exclude each other".to_owned());
        }
        (_, Some(noise_offset_db)) => Rate::Variable { noise_offset_db },
        (None, None) => Rate::Quality {
            level: DEFAULT_QUALITY,
        },
        (Some(level), None) => match u8::try_from(level) {
            Ok(level) if level <= MAX_LEVEL => Rate::Quality { level },
            _ => return Err(format!("--quality: not from 0 to {MAX_LEVEL}: {level}")),
        },
    };
    let options = Options {
        sample_rate,
        channels,
        rate,
        serial_number: serial_number.unwrap_or_else(random_serial_number),
    };
    options.config().check().map_err(describe)?;
    Ok(options)
}

/// A serial number drawn at random, as RFC 3533 asks (clause 4), from the
/// keys that the standard library draws for its hash maps.
fn random_serial_number() -> u32 {
    RandomState::new().hash_one(std::process::id()) as u32
}

/// The message for a configuration the encoder does not take, in terms of
/// the command line's flags.
fn describe(error: ConfigError) -> String {
    match error {
        ConfigError::UnsupportedSampleRate { .. } => "--sample-rate: not a rate: 0".to_owned(),
        ConfigError::UnsupportedChannels { channels } => {
            format!("unsupported channel count: {channels}; it takes 1 to {MAX_CHANNELS}")
        }
        ConfigError::NoiseOffset { noise_offset_db } => {
            format!("--vbr-noise: not a finite number: {noise_offset_db}")
        }
        error => error.to_string(),
    }
}

/// Encodes stdin into an Ogg stream, written to stdout as it is coded: its
/// header pages first, then each page as it is ended, each flushed at
/// once, so that a reader of stdout has it.
pub fn encode(options: &Options) -> Result<(), String> {
    let mut encoder =
        Encoder::new(options.config(), options.serial_number).map_err(|e| e.to_string())?;
    let mut output = Output::stdout();
    let mut stream = encoder.header_pages().to_vec();
    let mut write = |stream: &mut Vec<u8>| {
        if stream.is_empty() {
            return Ok(());
        }
        output.write(stream)?;
        output.flush()
    };
    write(&mut stream)?;
    pcm::read(encoder.channels(), |samples| {
        encoder.encode_interleaved(samples, &mut stream)?;
        Ok(write(&mut stream)?)
    })?;
    encoder.finish(&mut stream);
    write(&mut stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Options, String> {
        super::parse(args.iter().map(|&a| a.to_owned()))
    }

    fn error(args: &[&str]) -> String {
        parse(args).err().expect("the arguments are refused")
    }

    const STEREO: [&str; 4] = ["--sample-rate", "44100", "--channels", "2"];

    #[test]
    fn help_asks_for_the_usage_alone() {
        assert_eq!(error(&["--help"]), "");
        assert_eq!(error(&["--sample-rate", "44100", "-h"]), "");
    }

    #[test]
    fn the_sample_rate_and_channels_are_required() {
        assert_eq!(error(&["--channels", "2"]), "--sample-rate is required");
        assert_eq!(error(&["--sample-rate", "44100"]), "--channels is required");
    }

    #[test]
    fn refuses_what_the_encoder_does_not_take() {
        assert_eq!(
            error(&["--sample-rate", "0", "--channels", "2"]),
            "--sample-rate: not a rate: 0"
        );
        assert_eq!(
            error(&["--sample-rate", "44100", "--channels", "9"]),
            "unsupported channel count: 9; it takes 1 to 8"
        );
        assert_eq!(
            error(&["--sample-rate", "44100", "--channels", "0"]),
            "unsupported channel count: 0; it takes 1 to 8"
        );
        assert_eq!(
            error(&[&STEREO[..], &["--bitrate", "128"]].concat()),
            "unknown argument: --bitrate"
        );
    }

    #[test]
    fn codes_at_a_level_of_quality_or_a_noise_offset() {
        let rate = |args: &[&str]| parse(&[&STEREO[..], args].concat()).unwrap().rate;
        assert_eq!(rate(&[]), Rate::Quality { level: 80 });
        assert_eq!(rate(&["--quality", "35"]), Rate::Quality { level: 35 });
        assert_eq!(
            rate(&["--vbr-noise", "-3"]),
            Rate::Variable {
                noise_offset_db: -3.0
            }
        );
        let error = |args: &[&str]| error(&[&STEREO[..], args].concat());
        assert_eq!(
            error(&["--quality", "101"]),
            "--quality: not from 0 to 100: 101"
        );
        assert_eq!(
            error(&["--quality", "50", "--vbr-noise", "0"]),
            "--quality and --vbr-noise exclude each other"
        );
        assert_eq!(
            error(&["--vbr-noise", "inf"]),
            "--vbr-noise: not a finite number: inf"
        );
    }

    #[test]
    fn the_serial_number_is_given() {
        let options = parse(&[&STEREO[..], &["--serial-number", "4294967295"]].concat()).unwrap();
        assert_eq!(options.serial_number, u32::MAX);
        assert_eq!(
            error(&[&STEREO[..], &["--serial-number", "4294967296"]].concat()),
            "--serial-number: not a number: 4294967296"
        );
    }
}
