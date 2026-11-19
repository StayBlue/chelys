// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! `chelys aac`: MPEG-2 AAC in the Low Complexity profile, in ADTS.

use chelys::aac::{Config, ConfigError, Encoder, Rate};

use crate::output::Output;
use crate::{pcm, value};

pub const USAGE: &str = "\
usage: chelys aac --sample-rate HZ --channels N [--bitrate KBPS]
                  [--crc] [--private-bit] [--original-copy] [--home]
                  < input.f32 > output.aac

Reads interleaved 32-bit little-endian float PCM from stdin and writes MPEG-2
AAC in the Low Complexity profile (AAC-LC) to stdout, each frame in ADTS
(experimental: the encoder is still being built).

  --sample-rate HZ   96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050,
                     16000, 12000, 11025 or 8000
  --channels N       1, 2, 3, 4, 5, 6 (5.1) or 8 (7.1), interleaved in the
                     order the stream sends them:
                       1  center
                       2  left, right
                       3  center, left, right
                       4  center, left, right, rear surround
                       5  center, left, right, left surround, right surround
                       6  as 5, then LFE
                       8  center, left center, right center, left, right,
                          left surround, right surround, LFE
                     of which the LFE is coded in its lowest 12 of 1024
                     lines, up to 281 Hz at 48 kHz
  --bitrate KBPS     the stream's constant bitrate, ADTS headers and any CRC
                     included: a whole number within the range of the
                     sample rate and channels, at 48 kHz 5 to 287 in mono,
                     5 to 575 in stereo, 10 to 1439 in 5.1 and 12 to 2015
                     in 7.1, at 44.1 kHz 4 to 264, 5 to 528, 9 to 1322 and
                     11 to 1851; the lowest from 1 in mono at 8 kHz to 24
                     in 7.1 at 96 kHz, and the highest 6144 bits a frame
                     for each channel but the LFE; with --crc, the lowest
                     is up to 2 higher (default 128 in mono and stereo,
                     and 64 for each channel but the LFE in more: 192 in
                     3 channels, 320 in 5.1, 448 in 7.1; or the highest
                     the sample rate and channels allow where that is
                     lower)
  --crc              protect each frame's header and the start of its audio
                     with a CRC-16 check word, 16 bits a frame (default:
                     off)
  --private-bit      set each frame's bit for private use (default: off)
  --original-copy    set each header's original_copy bit, which the
                     standard defines as MPEG-1 Audio's copyright bit, set
                     where the stream is copyright protected (default: off)
  --home             set each header's home bit, which the standard defines
                     as MPEG-1 Audio's original/copy bit, set where the
                     stream is an original (default: off)

The decoded output starts with 1024 samples before the input and ends with
fewer than 1024 after it, near silence to be discarded: ADTS has no field
that tells a decoder so.";

/// The bitrate without `--bitrate` in mono and stereo, in kbit/s, where the
/// sample rate allows it: that of `chelys mp3`.
const DEFAULT_KBPS: u32 = 128;

pub struct Options {
    sample_rate: u32,
    channels: usize,
    kbps: u32,
    crc: bool,
    private_bit: bool,
    original_copy: bool,
    home: bool,
}

impl Options {
    fn config(&self) -> Config {
        config(self.sample_rate, self.channels, self.kbps)
            .crc(self.crc)
            .private_bit(self.private_bit)
            .original_copy(self.original_copy)
            .home(self.home)
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
    let (mut crc, mut private_bit, mut original_copy, mut home) = (false, false, false, false);
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        let args = &mut args;
        match flag.as_str() {
            "--sample-rate" => sample_rate = Some(value::<u32>(args, &flag)?),
            "--channels" => channels = Some(value::<u32>(args, &flag)?),
            "--bitrate" => bitrate = Some(value::<u32>(args, &flag)?),
            "--crc" => crc = true,
            "--private-bit" => private_bit = true,
            "--original-copy" => original_copy = true,
            "--home" => home = true,
            // An empty message asks for the usage text alone.
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    let sample_rate = sample_rate.ok_or("--sample-rate is required")?;
    let channels = match channels.ok_or("--channels is required")? {
        n @ (1..=6 | 8) => n as usize,
        n => return Err(format!("unsupported channel count: {n}")),
    };
    let kbps = bitrate.unwrap_or_else(|| default_kbps(sample_rate, channels, crc));
    let options = Options {
        sample_rate,
        channels,
        kbps,
        crc,
        private_bit,
        original_copy,
        home,
    };
    options
        .config()
        .check()
        .map_err(|e| describe(e, &options))?;
    Ok(options)
}

/// The bitrate without `--bitrate`, in kbit/s, for each channel but the
/// LFE, where there are more than two.
const DEFAULT_KBPS_PER_CHANNEL: u32 = 64;

/// [`DEFAULT_KBPS`] in mono and stereo, and [`DEFAULT_KBPS_PER_CHANNEL`]
/// for each channel but the LFE in more, or the highest whole bitrate the
/// sample rate and channels allow where that is lower, with the CRC if
/// `crc`.
fn default_kbps(sample_rate: u32, channels: usize, crc: bool) -> u32 {
    let kbps = match channels {
        1 | 2 => DEFAULT_KBPS,
        // 5.1 and 7.1, whose last channel is the LFE.
        6 | 8 => DEFAULT_KBPS_PER_CHANNEL * (channels as u32 - 1),
        _ => DEFAULT_KBPS_PER_CHANNEL * channels as u32,
    };
    match config(sample_rate, channels, kbps).crc(crc).check() {
        Err(ConfigError::UnsupportedBitrate { max, .. }) if max < kbps * 1000 => max / 1000,
        _ => kbps,
    }
}

/// The message for a configuration the encoder does not take, in terms of
/// the command line's flags.
fn describe(error: ConfigError, options: &Options) -> String {
    match error {
        ConfigError::UnsupportedBitrate { min, max, .. } => {
            let channels = match options.channels {
                1 => "mono".to_owned(),
                2 => "stereo".to_owned(),
                6 => "5.1".to_owned(),
                8 => "7.1".to_owned(),
                n => format!("{n} channels"),
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
        assert_eq!(args("44100", "7", "128"), "unsupported channel count: 7");
        assert_eq!(args("44100", "9", "128"), "unsupported channel count: 9");
        assert_eq!(
            args("48000", "6", "9"),
            "unsupported bitrate at 48000 Hz in 5.1: 9; it takes 10 to 1439"
        );
        assert_eq!(
            args("48000", "3", "900"),
            "unsupported bitrate at 48000 Hz in 3 channels: 900; it takes 7 to 863"
        );
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
    fn the_crc_raises_the_lowest_bitrate_at_44100_hz() {
        let lowest = |crc: &[&str]| {
            let args = [
                &[
                    "--sample-rate",
                    "44100",
                    "--channels",
                    "1",
                    "--bitrate",
                    "1",
                ],
                crc,
            ];
            error(&args.concat())
        };
        assert!(
            lowest(&[]).ends_with("it takes 4 to 264"),
            "{}",
            lowest(&[])
        );
        assert!(lowest(&["--crc"]).ends_with("it takes 5 to 264"));
        // A pair, which shares its window, at the bitrate its shortest
        // frame takes with the CRC.
        let args = [
            "--sample-rate",
            "88200",
            "--channels",
            "2",
            "--bitrate",
            "11",
        ];
        assert!(parse(&[&args[..], &["--crc"]].concat()).is_ok());
    }

    /// The ranges the usage gives.
    #[test]
    fn the_usage_gives_the_ranges() {
        let range = |rate: &str, channels: &str, crc: &[&str]| {
            let args = [
                &[
                    "--sample-rate",
                    rate,
                    "--channels",
                    channels,
                    "--bitrate",
                    "0",
                ],
                crc,
            ];
            let error = error(&args.concat());
            error.rsplit("it takes ").next().unwrap().to_owned()
        };
        for (rate, channels, without, with) in [
            ("48000", "1", "5 to 287", "5 to 287"),
            ("48000", "2", "5 to 575", "6 to 575"),
            ("44100", "1", "4 to 264", "5 to 264"),
            ("44100", "2", "5 to 528", "6 to 528"),
            ("8000", "1", "1 to 47", "1 to 47"),
            ("8000", "2", "1 to 95", "1 to 95"),
            ("96000", "1", "9 to 575", "10 to 575"),
            ("96000", "2", "10 to 1151", "12 to 1151"),
            ("88200", "2", "9 to 1057", "11 to 1057"),
            ("48000", "6", "10 to 1439", "11 to 1439"),
            ("48000", "8", "12 to 2015", "13 to 2015"),
            ("44100", "6", "9 to 1322", "10 to 1322"),
            ("44100", "8", "11 to 1851", "12 to 1851"),
            ("96000", "8", "24 to 4031", "25 to 4031"),
        ] {
            assert_eq!(range(rate, channels, &[]), without, "{rate} {channels}");
            assert_eq!(range(rate, channels, &["--crc"]), with, "{rate} {channels}");
        }
        for rate in SAMPLE_RATES {
            for channels in ["1", "2", "3", "4", "5", "6", "8"] {
                let lowest = |crc| range(&rate.to_string(), channels, crc);
                let low = |range: String| range.split(' ').next().unwrap().parse::<u32>().unwrap();
                let rise = low(lowest(&["--crc"])) - low(lowest(&[]));
                assert!(rise <= 2, "{rate} {channels}");
            }
        }
    }

    #[test]
    fn takes_the_ends_of_the_range() {
        for (rate, channels, kbps) in [
            ("44100", "2", "5"),
            ("44100", "2", "528"),
            ("48000", "6", "10"),
            ("48000", "6", "1439"),
            ("48000", "8", "2015"),
        ] {
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
    fn the_default_bitrate_is_128_or_64_a_channel_where_the_sample_rate_allows() {
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
        assert_eq!(kbps("44100", "3"), 192);
        assert_eq!(kbps("44100", "4"), 256);
        assert_eq!(kbps("48000", "5"), 320);
        assert_eq!(kbps("48000", "6"), 320);
        assert_eq!(kbps("48000", "8"), 448);
        assert_eq!(kbps("8000", "6"), 239);
        assert_eq!(kbps("8000", "8"), 335);
    }
}
