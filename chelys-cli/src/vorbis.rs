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
                     [--quality LEVEL | --vbr-noise DB |
                     [--bitrate KBPS] --abr]
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
  --abr              steer the stream toward --bitrate as its mean, each
                     packet as long as its content needs: silence leaves
                     it short, and a stream of a few seconds can miss the
                     mean either way
  --bitrate KBPS     with --abr, the mean bitrate of the audio packets, to
                     which Ogg adds about 1%: a whole number from 1/8 to 6
                     bits for each sample of each channel, at 44.1 kHz 6
                     to 264 in mono and 12 to 529 in stereo
                     (default 128 in mono and stereo, and 64 for each
                     channel but the LFE in more: 192 in 3 channels, 320
                     in 5.1, 448 in 7.1; or the nearest the sample rate
                     and channels allow); Vorbis has no constant bitrate,
                     so --bitrate needs --abr
  --serial-number N  the Ogg stream's serial number, from 0 to 4294967295,
                     which streams chained one after another must not share
                     (default: drawn at random)";

/// The level of quality coded at unless `--quality`, `--vbr-noise` or
/// `--abr` says otherwise: that of `chelys mp3 --vbr` and `chelys aac
/// --vbr`.
const DEFAULT_QUALITY: u8 = 80;

/// The mean bitrate of `--abr` without `--bitrate` in mono and stereo, in
/// kbit/s: that of `chelys aac`.
const DEFAULT_KBPS: u32 = 128;

/// The mean bitrate of `--abr` without `--bitrate`, in kbit/s, for each
/// channel but the LFE, where there are more than two: that of `chelys
/// aac`.
const DEFAULT_KBPS_PER_CHANNEL: u32 = 64;

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
    let (mut bitrate, mut abr, mut serial_number) = (None, false, None);
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        let args = &mut args;
        match flag.as_str() {
            "--sample-rate" => sample_rate = Some(value::<u32>(args, &flag)?),
            "--channels" => channels = Some(value::<u32>(args, &flag)?),
            "--quality" => quality = Some(value::<u32>(args, &flag)?),
            "--vbr-noise" => vbr_noise = Some(value::<f64>(args, &flag)?),
            "--bitrate" => bitrate = Some(value::<u32>(args, &flag)?),
            "--abr" => abr = true,
            "--serial-number" => serial_number = Some(value::<u32>(args, &flag)?),
            // An empty message asks for the usage text alone.
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    let sample_rate = sample_rate.ok_or("--sample-rate is required")?;
    let channels = channels.ok_or("--channels is required")? as usize;
    if bitrate.is_some() && !abr {
        return Err(
            "--bitrate needs --abr: Vorbis has no constant bitrate, only a mean one".to_owned(),
        );
    }
    if abr && (quality.is_some() || vbr_noise.is_some()) {
        return Err("--bitrate and --abr exclude --quality and --vbr-noise".to_owned());
    }
    let kbps = abr.then(|| bitrate.unwrap_or_else(|| default_kbps(sample_rate, channels)));
    let rate = match (kbps, quality, vbr_noise) {
        // A bitrate beyond a u32 of bit/s saturates, which the encoder
        // refuses as out of range like any other, and `describe` gives in
        // kbit/s.
        (Some(kbps), ..) => Rate::Mean {
            bits_per_second: kbps.saturating_mul(1000),
        },
        (None, Some(_), Some(_)) => {
            return Err("--quality and --vbr-noise exclude each other".to_owned());
        }
        (None, _, Some(noise_offset_db)) => Rate::Variable { noise_offset_db },
        (None, None, None) => Rate::Quality {
            level: DEFAULT_QUALITY,
        },
        (None, Some(level), None) => match u8::try_from(level) {
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
    (options.config().check()).map_err(|e| describe(e, &options, kbps.unwrap_or_default()))?;
    Ok(options)
}

/// [`DEFAULT_KBPS`] in mono and stereo, and [`DEFAULT_KBPS_PER_CHANNEL`]
/// for each channel but the LFE in more, which 6 to 8 channels have last;
/// or the nearest whole bitrate the sample rate and channels allow.
fn default_kbps(sample_rate: u32, channels: usize) -> u32 {
    let kbps = match channels {
        1 | 2 => DEFAULT_KBPS,
        6..=8 => DEFAULT_KBPS_PER_CHANNEL * (channels as u32 - 1),
        _ => DEFAULT_KBPS_PER_CHANNEL.saturating_mul(channels as u32),
    };
    let rate = Rate::Mean {
        bits_per_second: kbps.saturating_mul(1000),
    };
    match Config::new(sample_rate, channels, rate).check() {
        Err(ConfigError::UnsupportedBitrate { min, max, .. })
            if min.div_ceil(1000) <= max / 1000 =>
        {
            kbps.clamp(min.div_ceil(1000), max / 1000)
        }
        _ => kbps,
    }
}

/// A serial number drawn at random, as RFC 3533 asks (clause 4), from the
/// keys that the standard library draws for its hash maps.
fn random_serial_number() -> u32 {
    RandomState::new().hash_one(std::process::id()) as u32
}

/// The message for a configuration the encoder does not take, in terms of
/// the command line's flags, of which `--bitrate` asked for `kbps`.
fn describe(error: ConfigError, options: &Options, kbps: u32) -> String {
    match error {
        ConfigError::UnsupportedBitrate { min, max, .. } => {
            let (min, max) = (min.div_ceil(1000), max / 1000);
            let channels = match options.channels {
                1 => "mono".to_owned(),
                2 => "stereo".to_owned(),
                6 => "5.1".to_owned(),
                8 => "7.1".to_owned(),
                n => format!("{n} channels"),
            };
            if min > max {
                return format!(
                    "--abr: no whole bitrate in kbit/s is within the range at {} Hz in \
                     {channels}",
                    options.sample_rate,
                );
            }
            format!(
                "unsupported bitrate at {} Hz in {channels}: {kbps}; it takes {min} to {max}",
                options.sample_rate,
            )
        }
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
            error(&[&STEREO[..], &["--vbr"]].concat()),
            "unknown argument: --vbr"
        );
    }

    #[test]
    fn codes_at_a_mean_bitrate_with_abr() {
        let rate = |args: &[&str]| parse(args).unwrap().rate;
        let mean = |kbps: u32| Rate::Mean {
            bits_per_second: kbps * 1000,
        };
        let stereo = |args: &[&str]| rate(&[&STEREO[..], args].concat());
        assert_eq!(stereo(&["--bitrate", "96", "--abr"]), mean(96));
        assert_eq!(stereo(&["--abr", "--bitrate", "529"]), mean(529));
        // Without --bitrate, that of chelys aac.
        assert_eq!(stereo(&["--abr"]), mean(128));
        for (channels, kbps) in [("1", 128), ("3", 192), ("6", 320), ("7", 384), ("8", 448)] {
            let args = ["--sample-rate", "48000", "--channels", channels, "--abr"];
            assert_eq!(rate(&args), mean(kbps), "{channels} channels");
        }
        // Or the nearest the sample rate and channels allow: 6 bits a
        // sample at 8 kHz in mono, 1/8 at 1.6 MHz.
        let at = |hz: &str| rate(&["--sample-rate", hz, "--channels", "1", "--abr"]);
        assert_eq!(at("8000"), mean(48));
        assert_eq!(at("1600000"), mean(200));
    }

    #[test]
    fn a_mean_bitrate_is_refused_out_of_its_range_or_without_abr() {
        let error = |args: &[&str]| error(&[&STEREO[..], args].concat());
        assert_eq!(
            error(&["--bitrate", "128"]),
            "--bitrate needs --abr: Vorbis has no constant bitrate, only a mean one"
        );
        assert_eq!(
            error(&["--bitrate", "11", "--abr"]),
            "unsupported bitrate at 44100 Hz in stereo: 11; it takes 12 to 529"
        );
        assert_eq!(
            error(&["--bitrate", "530", "--abr"]),
            "unsupported bitrate at 44100 Hz in stereo: 530; it takes 12 to 529"
        );
        assert_eq!(
            super::parse(
                [
                    "--sample-rate",
                    "44100",
                    "--channels",
                    "1",
                    "--abr",
                    "--bitrate",
                    "4294968"
                ]
                .map(str::to_owned)
            )
            .err()
            .unwrap(),
            "unsupported bitrate at 44100 Hz in mono: 4294968; it takes 6 to 264"
        );
        assert_eq!(
            super::parse(
                [
                    "--sample-rate",
                    "48000",
                    "--channels",
                    "6",
                    "--abr",
                    "--bitrate",
                    "35"
                ]
                .map(str::to_owned)
            )
            .err()
            .unwrap(),
            "unsupported bitrate at 48000 Hz in 5.1: 35; it takes 36 to 1728"
        );
        // From 1/8 to 6 bit/s at 1 Hz, and 12.5 to 600 at 100 Hz, in
        // mono: no whole kbit/s.
        for hz in ["1", "100"] {
            let args = ["--sample-rate", hz, "--channels", "1", "--abr"];
            assert_eq!(
                super::parse(args.map(str::to_owned)).err().unwrap(),
                format!("--abr: no whole bitrate in kbit/s is within the range at {hz} Hz in mono")
            );
        }
        for args in [
            &["--abr", "--quality", "50"][..],
            &["--bitrate", "128", "--abr", "--vbr-noise", "0"],
            &["--vbr-noise", "0", "--abr"],
        ] {
            assert_eq!(
                error(args),
                "--bitrate and --abr exclude --quality and --vbr-noise"
            );
        }
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
