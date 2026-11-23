// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! `chelys mp3`: MPEG-1 and MPEG-2 Layer III, and MPEG-2.5.

use chelys::mp3::quality::MAX_LEVEL;
use chelys::mp3::{ChannelMode, Config, ConfigError, Encoder, Rate, bitrates};

use crate::output::Output;
use crate::{pcm, value};

pub const USAGE: &str = "\
usage: chelys mp3 --sample-rate HZ --channels N
                  [--bitrate KBPS [--free-format] | --vbr [--quality LEVEL |
                  --vbr-noise DB] [--min-bitrate KBPS] [--max-bitrate KBPS]]
                  [--no-joint-stereo]
                  [--dual-channel] [--intensity-stereo] [--mixed-blocks]
                  [--crc] [--copyright] [--original] [--private-bit]
                  < input.f32 > output.mp3

Reads interleaved 32-bit little-endian float PCM from stdin and writes MPEG-1
Layer III to stdout, or MPEG-2 Layer III at 16, 22.05 and 24 kHz, or
MPEG-2.5 Layer III at 8, 11.025 and 12 kHz (experimental).

  --sample-rate HZ   32000, 44100 or 48000; or 16000, 22050 or 24000; or
                     8000, 11025 or 12000
  --channels N       1 or 2
  --bitrate KBPS     the bitrate of every frame: 32, 40, 48, 56, 64, 80, 96,
                     112, 128, 160, 192, 224, 256 or 320; at 24 kHz and
                     below 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128,
                     144 or 160 (default 128); or a whole number between
                     two of those, kept as the stream's mean, with each
                     frame at one of the two
  --free-format      code every frame at --bitrate in free format, which
                     need not be in the list: 13 to 319, or 9 to 159 at
                     24 kHz and below; for links of a fixed capacity.
                     mpg123 and LAME decode it; ffmpeg (n9.0.2) does not
                     (default: off)
  --vbr              give each frame the lowest bitrate at which the noise
                     of every band stays within what the psychoacoustic
                     model allows, at a level of quality
  --quality LEVEL    with --vbr, the level of quality, from 0 to 100,
                     higher for better quality and mostly larger files, on
                     a scale shared with chelys aac --vbr (default 80)
  --min-bitrate KBPS with --vbr, the lowest bitrate a frame may have, from
                     the list above (default: the lowest)
  --max-bitrate KBPS with --vbr, the highest (default: the highest)
  --vbr-noise DB     with --vbr, in place of --quality, allow each band
                     this many dB more noise than the model does, or less
                     if negative
  --no-joint-stereo  code two channels as left and right throughout
                     (default: code each frame in MS stereo where the
                     channels are alike)
  --dual-channel     code two channels as two independent programmes, such
                     as two languages: as left and right throughout, and
                     marked as dual channel (default: off)
  --intensity-stereo code the upper bands of each frame of two channels in
                     intensity stereo where the frame is short of bits,
                     keeping only the level of each channel there and not
                     its fine structure; for low bitrates; with --vbr, only
                     at the highest bitrate (default: off)
  --mixed-blocks     where an attack switches to short blocks, keep the
                     normal block in the two lowest subbands (up to 1.4 kHz
                     at 44.1 kHz), for their frequency resolution; frames
                     with them are not coded in intensity stereo; not at
                     8 kHz (default: off)
  --crc              protect each frame's header and side information with
                     a CRC-16 check word (default: off)
  --copyright        mark the stream as copyright protected (default: off)
  --original         mark the stream as an original (default: marked as a
                     copy)
  --private-bit      set each frame's bit for private use (default: off)

Where stdout is a file, the stream starts with a Xing/Info tag: its length,
a table to seek by (left out at a low constant bitrate, which needs none),
and the samples to trim at its start and end. Through a pipe it has none,
as the tag is known only once the stream has ended, and in free format none
where the frames are too short to hold it.";

pub struct Options {
    sample_rate: u32,
    channels: usize,
    rate: Rate,
    /// On unless turned off, following ffmpeg's command line: its MP3
    /// encoder codes two channels in joint stereo by default, as observed
    /// from the frame headers of its output (ffmpeg n9.0.2).
    joint_stereo: bool,
    dual_channel: bool,
    /// Off unless asked for, following ffmpeg's command line: its MP3
    /// encoder offers no switch for intensity stereo, and with its defaults
    /// it coded no frame in it, at 32 to 128 kbit/s, when observed from the
    /// frame headers of its output (ffmpeg n9.0.2). Intensity stereo trades
    /// the stereo image above its bound for bits, a trade for low bitrates,
    /// which content that relies on the image may not want even there.
    intensity_stereo: bool,
    /// Off unless asked for: neither standard says when to use mixed
    /// blocks, and PEAQ (GstPEAQ 0.6.1, black box) finds them no better.
    /// Over 22 tracks of pop music at 44.1 kHz in stereo, where short
    /// blocks code under 1% of the granules, the mean grade moved by at
    /// most 0.004 at 96, 128 and 192 kbit/s and at `--vbr`. In the
    /// 3 seconds with the most short blocks of each of those and of 13
    /// tracks of rock, it moved by +0.017 at 128 kbit/s, -0.011 at 96,
    /// -0.003 at 192 and +0.005 at `--vbr`, better in 24, 13, 12 and 21 of
    /// the 35 windows.
    mixed_blocks: bool,
    crc: bool,
    copyright: bool,
    original: bool,
    private_bit: bool,
}

impl Options {
    /// The encoder's configuration, without a Xing/Info tag.
    fn config(&self) -> Config {
        let mode = match (
            self.channels,
            self.dual_channel,
            self.joint_stereo,
            self.intensity_stereo,
        ) {
            (1, _, _, _) => ChannelMode::Mono,
            (_, true, _, _) => ChannelMode::DualChannel,
            (_, _, false, false) => ChannelMode::Stereo,
            (_, _, ms_stereo, intensity_stereo) => ChannelMode::JointStereo {
                ms_stereo,
                intensity_stereo,
            },
        };
        Config::new(self.sample_rate, mode, self.rate)
            .mixed_blocks(self.mixed_blocks)
            .crc(self.crc)
            .copyright(self.copyright)
            .original(self.original)
            .private_bit(self.private_bit)
    }
}

/// The level of quality `--vbr` codes at unless `--quality` or `--vbr-noise`
/// says otherwise. At 44.1 kHz in stereo, level 80 allows each band 2.2 dB
/// more noise than the psychoacoustic model does, and over 51 tracks of
/// nine kinds of music averages 211 kbit/s and a grade of -0.12 (PEAQ,
/// GstPEAQ 0.6.1, black box). That is close to the 3 dB `--vbr` allowed
/// before the scale (200 kbit/s and -0.160 there), as the model's own
/// allowance spends more bits than listeners need. 3 dB was chosen while
/// every frame's bitrate was searched by coding the frame at each bitrate
/// tried. Graded with PEAQ (GstPEAQ 0.6.1, black box), 22 tracks of pop
/// music at 44.1 kHz in stereo then averaged 178 kbit/s and a grade of
/// -0.19 at 3 dB (-0.26 at worst), above the -0.22 (-0.32) of a constant
/// 178 kbit/s. Above 3 dB `--vbr` gained
/// nothing over a constant bitrate of its mean: 162 kbit/s and -0.27 (-0.39)
/// at 4 dB, against -0.27 (-0.39) at a constant 160 kbit/s; 148 kbit/s and
/// -0.37 (-0.55) at 5 dB, against -0.38 (-0.47) at a constant 148 kbit/s;
/// and 134 kbit/s and -0.50 (-0.78) at 6 dB, the worst track sparse and
/// tonal. Before the threshold in quiet was taken as noise has it, 3 dB gave
/// 153 kbit/s and -0.37 (-0.51 at worst); in a blind ABX test of that
/// track's worst passage at 4 dB before granules within their allowance
/// left their bits to the reservoir (184 kbit/s, -0.35, and -0.65 at worst),
/// a listener could not tell it from the input.
///
/// Since frames of normal blocks are quantised once instead, their noise
/// unshaped (see `chelys::mp3::Encoder`), `--vbr` gains over a constant
/// bitrate of its mean above 3 dB too: the pop music averages 180 kbit/s
/// and -0.15 (-0.21) at 3 dB, against -0.21 (-0.32) at a constant
/// 180 kbit/s, and 161 kbit/s and -0.25 (-0.39) at 4.5 dB, against -0.27
/// (-0.38) at a constant 162 kbit/s. The offset is not raised to win back
/// the 2-3% more bits the single coding spends: it keeps the allowance of
/// the frames still searched, and of the lower sampling frequencies, where
/// nothing changed, and the gain is taken as quality, most of it on the
/// worst tracks. A size matched to the search's would be particular to the
/// tracks it was matched on.
const DEFAULT_QUALITY: u8 = 80;

pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Options, String> {
    let (mut sample_rate, mut channels, mut bitrate) = (None, None, None);
    let (mut vbr, mut min_kbps, mut max_kbps, mut vbr_noise) = (false, None, None, None);
    let mut quality = None;
    let mut free_format = false;
    let (mut copyright, mut original, mut private_bit) = (false, false, false);
    let (mut joint_stereo, mut dual_channel) = (true, false);
    let (mut intensity_stereo, mut crc) = (false, false);
    let mut mixed_blocks = false;
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        let args = &mut args;
        match flag.as_str() {
            "--sample-rate" => sample_rate = Some(value::<u32>(args, &flag)?),
            "--channels" => channels = Some(value::<u32>(args, &flag)?),
            "--bitrate" => bitrate = Some(value::<u32>(args, &flag)?),
            "--free-format" => free_format = true,
            "--vbr" => vbr = true,
            "--min-bitrate" => min_kbps = Some(value::<u32>(args, &flag)?),
            "--max-bitrate" => max_kbps = Some(value::<u32>(args, &flag)?),
            "--quality" => quality = Some(value::<u32>(args, &flag)?),
            "--vbr-noise" => vbr_noise = Some(value::<f64>(args, &flag)?),
            "--no-joint-stereo" => joint_stereo = false,
            "--dual-channel" => dual_channel = true,
            "--intensity-stereo" => intensity_stereo = true,
            "--mixed-blocks" => mixed_blocks = true,
            "--crc" => crc = true,
            "--copyright" => copyright = true,
            "--original" => original = true,
            "--private-bit" => private_bit = true,
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
    let rate = if vbr {
        if bitrate.is_some() || free_format {
            return Err("--bitrate and --free-format exclude --vbr".to_owned());
        }
        match (quality, vbr_noise) {
            (Some(_), Some(_)) => {
                return Err("--quality and --vbr-noise exclude each other".to_owned());
            }
            (_, Some(noise_offset_db)) => Rate::Variable {
                min_kbps,
                max_kbps,
                noise_offset_db,
            },
            (level, None) => Rate::Quality {
                level: match level {
                    None => DEFAULT_QUALITY,
                    Some(level) if level <= u32::from(MAX_LEVEL) => level as u8,
                    Some(level) => {
                        return Err(format!("--quality: not from 0 to {MAX_LEVEL}: {level}"));
                    }
                },
                min_kbps,
                max_kbps,
            },
        }
    } else {
        if min_kbps.is_some() || max_kbps.is_some() || quality.is_some() || vbr_noise.is_some() {
            return Err(
                "--min-bitrate, --max-bitrate, --quality and --vbr-noise need --vbr".to_owned(),
            );
        }
        let kbps = bitrate.unwrap_or(128);
        let in_table = bitrates(sample_rate).is_none_or(|table| table.contains(&kbps));
        match kbps.checked_mul(1000) {
            Some(bits_per_second) if free_format => Rate::Free { bits_per_second },
            None if free_format => {
                return Err(format!(
                    "--free-format: unsupported bitrate at {sample_rate} Hz: {kbps}"
                ));
            }
            Some(bits_per_second) if !in_table => Rate::Mean { bits_per_second },
            _ => Rate::Constant { kbps },
        }
    };
    if intensity_stereo && channels != 2 {
        return Err("--intensity-stereo needs two channels".to_owned());
    }
    if dual_channel && channels != 2 {
        return Err("--dual-channel needs two channels".to_owned());
    }
    if dual_channel && intensity_stereo {
        return Err("--dual-channel and --intensity-stereo exclude each other".to_owned());
    }
    let options = Options {
        sample_rate,
        channels,
        rate,
        joint_stereo,
        dual_channel,
        intensity_stereo,
        mixed_blocks,
        crc,
        copyright,
        original,
        private_bit,
    };
    options
        .config()
        .check()
        .map_err(|e| describe(e, kbps_of(rate)))?;
    Ok(options)
}

/// The bitrate given with `--bitrate`, in kbit/s, if any.
fn kbps_of(rate: Rate) -> Option<u32> {
    match rate {
        Rate::Constant { kbps } => Some(kbps),
        Rate::Mean { bits_per_second } | Rate::Free { bits_per_second } => {
            Some(bits_per_second / 1000)
        }
        _ => None,
    }
}

/// The message for a configuration the encoder does not take, in terms of
/// the command line's flags.
fn describe(error: ConfigError, kbps: Option<u32>) -> String {
    let kbps = kbps.unwrap_or_default();
    match error {
        ConfigError::UnsupportedMeanBitrate { sample_rate, .. } => {
            format!("unsupported bitrate at {sample_rate} Hz: {kbps}")
        }
        ConfigError::UnsupportedBitrate { sample_rate, kbps } => {
            format!("unsupported bitrate at {sample_rate} Hz: {kbps}")
        }
        ConfigError::UnsupportedFreeBitrate { in_table: true, .. } => {
            format!("--free-format: {kbps} kbit/s is in the list; leave out --free-format")
        }
        ConfigError::UnsupportedFreeBitrate { sample_rate, .. } => {
            format!("--free-format: unsupported bitrate at {sample_rate} Hz: {kbps}")
        }
        ConfigError::BitrateRange { .. } => "--min-bitrate is above --max-bitrate".to_owned(),
        ConfigError::NoiseOffset { noise_offset_db } => {
            format!("--vbr-noise: not a finite number: {noise_offset_db}")
        }
        ConfigError::MixedBlocksUnsupported { sample_rate } => {
            format!("--mixed-blocks: not at {sample_rate} Hz")
        }
        error => error.to_string(),
    }
}

/// Encodes stdin to stdout as it arrives, with a Xing/Info tag where stdout
/// is a file whose start can be rewritten once the stream has ended.
pub fn encode(options: &Options) -> Result<(), String> {
    let mut output = Output::stdout();
    let config = options.config().info_tag(output.can_rewrite_start());
    let mut encoder = Encoder::new(config).map_err(|e| e.to_string())?;
    let mut stream = Vec::new();
    pcm::read(encoder.channels(), |samples| {
        encoder.encode_interleaved(samples, &mut stream)?;
        Ok(output.write(&mut stream)?)
    })?;
    let tag = encoder.finish(&mut stream).info_tag;
    output.write(&mut stream)?;
    output.flush()?;
    match tag {
        Some(tag) => output.rewrite_start(&tag, "the Xing/Info tag"),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Options, String> {
        let args = ["--sample-rate", "44100", "--channels", "2"]
            .iter()
            .chain(args);
        super::parse(args.map(|&a| a.to_owned()))
    }

    fn error(args: &[&str]) -> String {
        parse(args).err().expect("the arguments are refused")
    }

    #[test]
    fn vbr_codes_at_a_level_of_quality_or_a_noise_offset() {
        let quality = |level| Rate::Quality {
            level,
            min_kbps: None,
            max_kbps: None,
        };
        assert_eq!(parse(&["--vbr"]).unwrap().rate, quality(DEFAULT_QUALITY));
        assert_eq!(
            parse(&["--vbr", "--quality", "0"]).unwrap().rate,
            quality(0)
        );
        assert_eq!(
            parse(&["--vbr", "--quality", "100", "--min-bitrate", "128"])
                .unwrap()
                .rate,
            Rate::Quality {
                level: 100,
                min_kbps: Some(128),
                max_kbps: None,
            }
        );
        assert_eq!(
            parse(&["--vbr", "--vbr-noise", "-1.5"]).unwrap().rate,
            Rate::Variable {
                min_kbps: None,
                max_kbps: None,
                noise_offset_db: -1.5,
            }
        );
    }

    #[test]
    fn a_level_of_quality_is_refused_out_of_place() {
        assert_eq!(
            error(&["--vbr", "--quality", "101"]),
            "--quality: not from 0 to 100: 101"
        );
        assert_eq!(
            error(&["--vbr", "--quality", "-1"]),
            "--quality: not a number: -1"
        );
        assert_eq!(
            error(&["--vbr", "--quality", "80", "--vbr-noise", "3"]),
            "--quality and --vbr-noise exclude each other"
        );
        assert_eq!(
            error(&["--quality", "80"]),
            "--min-bitrate, --max-bitrate, --quality and --vbr-noise need --vbr"
        );
    }
}
