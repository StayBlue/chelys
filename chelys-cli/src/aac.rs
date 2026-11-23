// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! `chelys aac`: AAC in the Low Complexity profile, in ADTS or in an MP4
//! file.

use std::io::Write;

use chelys::aac::quality::MAX_LEVEL;
use chelys::aac::{Config, ConfigError, Encoder, Framing, Rate};
use chelys::mp4;
use chelys::mp4::aac::{EncodeError, FragmentOptions};

use crate::output::Output;
use crate::{pcm, value};

pub const USAGE: &str = "\
usage: chelys aac --sample-rate HZ --channels N
                  [--bitrate KBPS [--abr] | --vbr [--quality LEVEL |
                  --vbr-noise DB]]
                  [--noise-substitution]
                  [--crc] [--private-bit] [--original-copy] [--home]
                  < input.f32 > output.aac
       chelys aac --mp4 [--fragmented] --sample-rate HZ --channels N
                  [--bitrate KBPS [--abr] | --vbr [--quality LEVEL |
                  --vbr-noise DB]]
                  [--noise-substitution]
                  < input.f32 > output.mp4

Reads interleaved 32-bit little-endian float PCM from stdin and writes AAC in
the Low Complexity profile (AAC-LC) to stdout, each frame in ADTS: MPEG-2 AAC,
or MPEG-4 AAC where --noise-substitution is used. With --mp4, it writes an
MP4 file instead, which tells a player the input's exact length.

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
                     5 to 575 in stereo, 10 to 1439 in 5.1 and 12 to 1727
                     in 7.1, at 44.1 kHz 4 to 264, 5 to 528, 9 to 1322 and
                     11 to 1587; the lowest from 1 in mono at 8 kHz to 24
                     in 7.1 at 96 kHz, and the highest 6144 bits a frame
                     for each channel but the LFE, or 4608 bytes in 7.1;
                     with --crc, the lowest is up to 2 higher; with --mp4,
                     which counts the frames alone, the lowest is up to 6
                     lower, and the highest in 7.1 that of 6144 bits a
                     frame, 2015 at 48 kHz and 1851 at 44.1 kHz (default 128
                     in mono and stereo, and 64 for each channel but the
                     LFE in more: 192 in 3 channels, 320 in 5.1, 448 in
                     7.1; or the highest the sample rate and channels
                     allow where that is lower)
  --abr              steer the stream toward --bitrate as its mean, each
                     frame as long as its content needs: silence leaves it
                     short, and a stream of a few seconds can miss the
                     mean either way (default: a constant bitrate)
  --vbr              give each frame the bits at which the noise of every
                     band stays within what the psychoacoustic model
                     allows, at a level of quality
  --quality LEVEL    with --vbr, the level of quality, from 0 to 100,
                     higher for better quality and mostly larger files, on
                     a scale shared with chelys mp3 --vbr: for music in
                     stereo at 44.1 kHz about 138 kbit/s at 50, 190 at 80
                     and 229 at 100 (default 80)
  --vbr-noise DB     with --vbr, in place of --quality, allow each band
                     this many dB more noise than the model does, or less
                     if negative: about 205 kbit/s at -6, 142 at 0 and 73
                     at 6 in stereo at 44.1 kHz
  --noise-substitution
                     code noise-like bands as the energy of noise that the
                     decoder generates in their place, saving bits for the
                     others, at up to 1.3 bits a sample of each channel but
                     the LFE (about 115 in stereo at 44.1 kHz); a tool of
                     MPEG-4 AAC, which makes the stream MPEG-4 AAC-LC where
                     it is used, and which a decoder of MPEG-2 AAC alone may
                     not decode (default: off)
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
  --mp4              write an MP4 file (ISO/IEC 14496-14) of the stream
                     instead of ADTS; the four options above set fields of
                     ADTS headers, which an MP4 file does not have, and are
                     refused with it; the file is written once the input
                     has ended, so it is held in memory until then, about
                     the bitrate times the duration
  --fragmented       with --mp4, write a fragmented MP4 file as the input
                     arrives: the file's header at once, then a fragment of
                     about a second of the stream as each is coded, so that
                     little is held in memory and the file can be played
                     as it is written; but as the input's length is not
                     known until it ends, the file does not give it, and
                     a player plays the near silence after the input too,
                     fewer than 1024 samples, which --mp4 alone leaves out

The decoded output starts with 1024 samples before the input and ends with
fewer than 1024 after it, near silence to be discarded: ADTS has no field
that tells a decoder so, where an MP4 file's edit list does.";

/// The bitrate without `--bitrate` in mono and stereo, in kbit/s, where the
/// sample rate allows it: that of `chelys mp3`.
const DEFAULT_KBPS: u32 = 128;

pub struct Options {
    sample_rate: u32,
    channels: usize,
    kbps: u32,
    /// With `--abr`, `kbps` is the mean.
    abr: bool,
    /// With `--vbr`, its rate, a level of quality or a noise offset, in
    /// place of `kbps`.
    vbr: Option<Rate>,
    noise_substitution: bool,
    crc: bool,
    private_bit: bool,
    original_copy: bool,
    home: bool,
    mp4: bool,
    /// With `--mp4`, whether the file is fragmented.
    fragmented: bool,
}

impl Options {
    fn config(&self) -> Config {
        let bits_per_second = self.kbps.saturating_mul(1000);
        let rate = match self.vbr {
            Some(rate) => rate,
            None if self.abr => Rate::Mean { bits_per_second },
            None => Rate::Constant { bits_per_second },
        };
        Config::new(self.sample_rate, self.channels, rate)
            .framing(framing(self.mp4))
            .noise_substitution(self.noise_substitution)
            .crc(self.crc)
            .private_bit(self.private_bit)
            .original_copy(self.original_copy)
            .home(self.home)
    }
}

/// Raw frames for an MP4 file, which carries them as they are, and ADTS
/// otherwise.
fn framing(mp4: bool) -> Framing {
    if mp4 { Framing::Raw } else { Framing::Adts }
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
    let (mut noise_substitution, mut mp4, mut fragmented) = (false, false, false);
    let (mut abr, mut vbr, mut quality, mut vbr_noise) = (false, false, None, None);
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        let args = &mut args;
        match flag.as_str() {
            "--sample-rate" => sample_rate = Some(value::<u32>(args, &flag)?),
            "--channels" => channels = Some(value::<u32>(args, &flag)?),
            "--bitrate" => bitrate = Some(value::<u32>(args, &flag)?),
            "--abr" => abr = true,
            "--vbr" => vbr = true,
            "--quality" => quality = Some(value::<u32>(args, &flag)?),
            "--vbr-noise" => vbr_noise = Some(value::<f64>(args, &flag)?),
            "--noise-substitution" => noise_substitution = true,
            "--crc" => crc = true,
            "--private-bit" => private_bit = true,
            "--original-copy" => original_copy = true,
            "--home" => home = true,
            "--mp4" => mp4 = true,
            "--fragmented" => fragmented = true,
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
    if fragmented && !mp4 {
        return Err("--fragmented needs --mp4".to_owned());
    }
    if vbr && (bitrate.is_some() || abr) {
        return Err("--bitrate and --abr exclude --vbr".to_owned());
    }
    if !vbr && (quality.is_some() || vbr_noise.is_some()) {
        return Err("--quality and --vbr-noise need --vbr".to_owned());
    }
    let vbr = if vbr {
        Some(match (quality, vbr_noise) {
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
        })
    } else {
        None
    };
    let kbps = bitrate.unwrap_or_else(|| default_kbps(sample_rate, channels, crc, mp4));
    let options = Options {
        sample_rate,
        channels,
        kbps,
        abr,
        vbr,
        noise_substitution,
        crc,
        private_bit,
        original_copy,
        home,
        mp4,
        fragmented,
    };
    options
        .config()
        .check()
        .map_err(|e| describe(e, &options))?;
    Ok(options)
}

/// The level of quality `--vbr` codes at unless `--quality` or `--vbr-noise`
/// says otherwise: that of `chelys mp3 --vbr`. At 44.1 kHz in stereo, level
/// 80 allows each band 4.6 dB less noise than the psychoacoustic model
/// does, and over 51 tracks of nine kinds of music averages 190 kbit/s and
/// a grade of -0.12 (PEAQ, GstPEAQ 0.6.1, black box), where `chelys mp3`
/// averages 210 kbit/s at the same grade.
const DEFAULT_QUALITY: u8 = 80;

/// The bitrate without `--bitrate`, in kbit/s, for each channel but the
/// LFE, where there are more than two.
const DEFAULT_KBPS_PER_CHANNEL: u32 = 64;

/// [`DEFAULT_KBPS`] in mono and stereo, and [`DEFAULT_KBPS_PER_CHANNEL`]
/// for each channel but the LFE in more, or the highest whole bitrate the
/// sample rate and channels allow where that is lower, with the CRC if
/// `crc`, in an MP4 file if `mp4`.
fn default_kbps(sample_rate: u32, channels: usize, crc: bool, mp4: bool) -> u32 {
    let kbps = match channels {
        1 | 2 => DEFAULT_KBPS,
        // 5.1 and 7.1, whose last channel is the LFE.
        6 | 8 => DEFAULT_KBPS_PER_CHANNEL * (channels as u32 - 1),
        _ => DEFAULT_KBPS_PER_CHANNEL * channels as u32,
    };
    let config = config(sample_rate, channels, kbps).framing(framing(mp4));
    match config.crc(crc).check() {
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
        ConfigError::NoiseOffset { noise_offset_db } => {
            format!("--vbr-noise: not a finite number: {noise_offset_db}")
        }
        ConfigError::AdtsOnly => {
            "--crc, --private-bit, --original-copy and --home set fields of ADTS headers, \
             which --mp4 does not write"
                .to_owned()
        }
        error => error.to_string(),
    }
}

/// Encodes stdin to stdout as it arrives, or with `--mp4` writes the file
/// once stdin has ended, unless it is `--fragmented`.
pub fn encode(options: &Options) -> Result<(), String> {
    match (options.mp4, options.fragmented) {
        (true, false) => return encode_mp4(options),
        (true, true) => return encode_fragmented(options),
        (false, _) => {}
    }
    let mut output = Output::stdout();
    let mut encoder = Encoder::new(options.config()).map_err(|e| e.to_string())?;
    let mut stream = Vec::new();
    pcm::read(encoder.channels(), |samples| {
        encoder.encode_interleaved(samples, &mut stream)?;
        Ok(output.write(&mut stream)?)
    })?;
    encoder.finish(&mut stream);
    output.write(&mut stream)?;
    output.flush()
}

/// Encodes stdin into an MP4 file, which is written to stdout whole once
/// stdin has ended: its sample tables, which come first, describe every
/// frame. Written in one pass, it needs no seekable stdout.
fn encode_mp4(options: &Options) -> Result<(), String> {
    let mut encoder = mp4::aac::Encoder::new(options.config()).map_err(|e| e.to_string())?;
    pcm::read(encoder.channels(), |samples| {
        Ok(encoder.encode_interleaved(samples)?)
    })?;
    let mut stdout = std::io::stdout().lock();
    let error = |e: std::io::Error| format!("writing stdout: {e}");
    encoder.finish(&mut stdout).map_err(error)?;
    stdout.flush().map_err(error)
}

/// Encodes stdin into a fragmented MP4 file, written to stdout as it is
/// coded: its init segment first, then each fragment as it is ended, each
/// flushed at once, so that a reader of stdout has it.
fn encode_fragmented(options: &Options) -> Result<(), String> {
    let mut encoder =
        mp4::aac::FragmentedEncoder::new(options.config(), FragmentOptions::default())
            .map_err(|e| e.to_string())?;
    let mut output = Output::stdout();
    let mut stream = encoder.init_segment().to_vec();
    let mut write = |stream: &mut Vec<u8>| {
        if stream.is_empty() {
            return Ok(());
        }
        output.write(stream)?;
        output.flush()
    };
    write(&mut stream)?;
    pcm::read(encoder.channels(), |samples| {
        match encoder.encode_interleaved(samples, &mut stream) {
            Ok(()) => {}
            Err(EncodeError::NonFiniteSample(e)) => return Err(e.into()),
            Err(e) => return Err(e.to_string().into()),
        }
        Ok(write(&mut stream)?)
    })?;
    encoder.finish(&mut stream).map_err(|e| e.to_string())?;
    write(&mut stream)
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
        assert_eq!(error(&["--cbr"]), "unknown argument: --cbr");
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
            ("48000", "8", "12 to 1727", "13 to 1727"),
            ("44100", "6", "9 to 1322", "10 to 1322"),
            ("44100", "8", "11 to 1587", "12 to 1587"),
            ("96000", "8", "24 to 3455", "25 to 3455"),
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
    fn mp4_refuses_the_options_of_adts_headers() {
        let args = ["--mp4", "--sample-rate", "44100", "--channels", "2"];
        assert!(parse(&args).is_ok());
        for flag in ["--crc", "--private-bit", "--original-copy", "--home"] {
            assert_eq!(
                error(&[&args[..], &[flag]].concat()),
                "--crc, --private-bit, --original-copy and --home set fields of ADTS headers, \
                 which --mp4 does not write",
                "{flag}"
            );
        }
    }

    #[test]
    fn fragmented_needs_mp4() {
        let args = ["--fragmented", "--sample-rate", "44100", "--channels", "2"];
        assert_eq!(error(&args), "--fragmented needs --mp4");
        let options = parse(&[&args[..], &["--mp4"]].concat()).unwrap();
        assert!(options.mp4 && options.fragmented);
    }

    /// The ranges the usage gives with --mp4, and that the lowest is at
    /// most 6 lower than in ADTS.
    #[test]
    fn mp4_ranges_count_the_frames_alone() {
        let range = |rate: &str, channels: &str, mp4: &[&str]| {
            let args = [
                &[
                    "--sample-rate",
                    rate,
                    "--channels",
                    channels,
                    "--bitrate",
                    "0",
                ],
                mp4,
            ];
            let error = error(&args.concat());
            let range = error.rsplit("it takes ").next().unwrap().to_owned();
            let ends: Vec<u32> = range.split(" to ").map(|n| n.parse().unwrap()).collect();
            (ends[0], ends[1])
        };
        assert_eq!(range("48000", "8", &["--mp4"]), (9, 2015));
        assert_eq!(range("44100", "8", &["--mp4"]), (9, 1851));
        for rate in SAMPLE_RATES {
            for channels in ["1", "2", "3", "4", "5", "6", "8"] {
                let rate = rate.to_string();
                let (adts, mp4) = (
                    range(&rate, channels, &[]),
                    range(&rate, channels, &["--mp4"]),
                );
                assert!(adts.0 - mp4.0 <= 6, "{rate} {channels}");
                if channels != "8" {
                    assert_eq!(adts.1, mp4.1, "{rate} {channels}");
                }
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
            ("48000", "8", "1727"),
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
        assert_eq!(kbps("8000", "8"), 287);
    }

    #[test]
    fn the_rate_modes() {
        let rate = |args: &[&str]| {
            let base = ["--sample-rate", "44100", "--channels", "2"];
            parse(&[&base[..], args].concat()).unwrap().config().rate()
        };
        assert_eq!(
            rate(&["--bitrate", "96"]),
            Rate::Constant {
                bits_per_second: 96_000
            }
        );
        assert_eq!(
            rate(&["--bitrate", "96", "--abr"]),
            Rate::Mean {
                bits_per_second: 96_000
            }
        );
        assert_eq!(
            rate(&["--abr"]),
            Rate::Mean {
                bits_per_second: 128_000
            }
        );
        assert_eq!(
            rate(&["--vbr"]),
            Rate::Quality {
                level: DEFAULT_QUALITY
            }
        );
        assert_eq!(
            rate(&["--vbr", "--quality", "0"]),
            Rate::Quality { level: 0 }
        );
        assert_eq!(
            rate(&["--vbr", "--quality", "100"]),
            Rate::Quality { level: 100 }
        );
        assert_eq!(
            rate(&["--vbr", "--vbr-noise", "-2.5"]),
            Rate::Variable {
                noise_offset_db: -2.5
            }
        );
    }

    #[test]
    fn refuses_rate_modes_that_exclude_each_other() {
        let error = |args: &[&str]| {
            let base = ["--sample-rate", "44100", "--channels", "2"];
            error(&[&base[..], args].concat())
        };
        assert_eq!(
            error(&["--vbr", "--bitrate", "96"]),
            "--bitrate and --abr exclude --vbr"
        );
        assert_eq!(
            error(&["--vbr", "--abr"]),
            "--bitrate and --abr exclude --vbr"
        );
        assert_eq!(
            error(&["--vbr-noise", "3"]),
            "--quality and --vbr-noise need --vbr"
        );
        assert_eq!(
            error(&["--quality", "80"]),
            "--quality and --vbr-noise need --vbr"
        );
        assert_eq!(
            error(&["--vbr", "--quality", "80", "--vbr-noise", "3"]),
            "--quality and --vbr-noise exclude each other"
        );
        assert_eq!(
            error(&["--vbr", "--quality", "101"]),
            "--quality: not from 0 to 100: 101"
        );
        assert_eq!(
            error(&["--vbr", "--quality", "256"]),
            "--quality: not from 0 to 100: 256"
        );
        assert_eq!(
            error(&["--vbr", "--quality", "-1"]),
            "--quality: not a number: -1"
        );
        assert_eq!(
            error(&["--vbr", "--vbr-noise", "inf"]),
            "--vbr-noise: not a finite number: inf"
        );
        assert_eq!(
            error(&["--abr", "--bitrate", "4"]),
            "unsupported bitrate at 44100 Hz in stereo: 4; it takes 5 to 528"
        );
    }
}
