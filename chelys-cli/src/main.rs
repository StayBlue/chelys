// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Encodes raw PCM from stdin as MPEG-1 Layer III, or MPEG-2 Layer III at the
//! lower sampling frequencies, on stdout. Input formats are left to other
//! tools, for example:
//!
//! ```sh
//! ffmpeg -i song.flac -f f32le -ar 44100 -ac 2 - | chelys --sample-rate 44100 --channels 2 > song.mp3
//! ```
//!
//! Where stdout is a file, as there, the stream starts with a Xing/Info
//! tag, written once the stream has ended.

use std::fs::File;
use std::io::{ErrorKind, Read, Seek, SeekFrom, Write};
use std::process::ExitCode;
use std::str::FromStr;

use chelys::mp3::header::{Bitrate, Mode, SamplingFrequency};
use chelys::mp3::rate::{MeanRate, Rate, VariableRate};
use chelys::mp3::{Encoder, StreamConfig};

const USAGE: &str = "\
usage: chelys --sample-rate HZ --channels N
              [--bitrate KBPS | --vbr [--min-bitrate KBPS] [--max-bitrate KBPS]
              [--vbr-noise DB]] [--no-joint-stereo] [--dual-channel]
              [--intensity-stereo] [--crc] < input.f32 > output.mp3

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
  --vbr              give each frame the lowest bitrate at which the noise
                     of every band stays within what the psychoacoustic
                     model allows (experimental)
  --min-bitrate KBPS with --vbr, the lowest bitrate a frame may have, from
                     the list above (default: the lowest)
  --max-bitrate KBPS with --vbr, the highest (default: the highest)
  --vbr-noise DB     with --vbr, allow each band this many dB more noise
                     than the model does, or less if negative (default 4)
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
  --crc              protect each frame's header and side information with
                     a CRC-16 check word (default: off)

Where stdout is a file, the stream starts with a Xing/Info tag: its length,
a table to seek by, and the samples to trim at its start and end. Through a
pipe it has none, as the tag is known only once the stream has ended.";

struct Options {
    sampling_frequency: SamplingFrequency,
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
    crc_protected: bool,
}

/// The noise `--vbr` allows each band beyond what the psychoacoustic model
/// does, in dB, unless `--vbr-noise` says otherwise. The model's own
/// allowance spends more bits than listeners need. Graded with PEAQ
/// (GstPEAQ 0.6.1, black box), 22 tracks of pop music at 44.1 kHz in stereo
/// averaged 238 kbit/s and a grade of -0.10 at 0 dB, 196 kbit/s and -0.26
/// at 3 dB, and 184 kbit/s and -0.35 at 4 dB, -0.65 at worst. Above 4 dB,
/// the worst track, sparse and tonal, fell quickly: -1.07 at 5 dB, and at
/// 6 dB -1.76 at 117 kbit/s, below its grade at a constant 128 kbit/s. In
/// a blind ABX test of that track's worst passage at 4 dB, a listener could
/// not tell it from the input.
const DEFAULT_VBR_NOISE_DB: f64 = 4.0;

/// The value after the flag `name`, parsed.
fn value<T: FromStr>(args: &mut impl Iterator<Item = String>, name: &str) -> Result<T, String> {
    let value = args.next().ok_or(format!("{name} needs a value"))?;
    value
        .parse()
        .map_err(|_| format!("{name}: not a number: {value}"))
}

fn parse(args: impl IntoIterator<Item = String>) -> Result<Options, String> {
    let (mut sample_rate, mut channels, mut bitrate) = (None, None, None);
    let (mut vbr, mut min_bitrate, mut max_bitrate, mut vbr_noise) = (false, None, None, None);
    let (mut joint_stereo, mut dual_channel) = (true, false);
    let (mut intensity_stereo, mut crc_protected) = (false, false);
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        let args = &mut args;
        match flag.as_str() {
            "--sample-rate" => sample_rate = Some(value::<u32>(args, &flag)?),
            "--channels" => channels = Some(value::<u32>(args, &flag)?),
            "--bitrate" => bitrate = Some(value::<u32>(args, &flag)?),
            "--vbr" => vbr = true,
            "--min-bitrate" => min_bitrate = Some(value::<u32>(args, &flag)?),
            "--max-bitrate" => max_bitrate = Some(value::<u32>(args, &flag)?),
            "--vbr-noise" => vbr_noise = Some(value::<f64>(args, &flag)?),
            "--no-joint-stereo" => joint_stereo = false,
            "--dual-channel" => dual_channel = true,
            "--intensity-stereo" => intensity_stereo = true,
            "--crc" => crc_protected = true,
            // An empty message asks for the usage text alone.
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    let sample_rate = sample_rate.ok_or("--sample-rate is required")?;
    let sampling_frequency = SamplingFrequency::ALL
        .into_iter()
        .find(|fs| fs.hz() == sample_rate)
        .ok_or(format!("unsupported sample rate: {sample_rate}"))?;
    let channels = match channels.ok_or("--channels is required")? {
        n @ (1 | 2) => n as usize,
        n => return Err(format!("unsupported channel count: {n}")),
    };
    let version = sampling_frequency.version();
    let table_bitrate = |kbps: u32| {
        Bitrate::from_kbps(version, kbps)
            .ok_or(format!("unsupported bitrate at {sample_rate} Hz: {kbps}"))
    };
    let rate = if vbr {
        if bitrate.is_some() {
            return Err("--bitrate and --vbr exclude each other".to_owned());
        }
        let mut bitrates = Bitrate::all(version);
        let lowest = match min_bitrate {
            Some(kbps) => table_bitrate(kbps)?,
            None => bitrates.next().expect("a table of bitrates"),
        };
        let highest = match max_bitrate {
            Some(kbps) => table_bitrate(kbps)?,
            None => bitrates.last().expect("a table of bitrates"),
        };
        if lowest.bits_per_second() > highest.bits_per_second() {
            return Err("--min-bitrate is above --max-bitrate".to_owned());
        }
        let noise_offset_db = vbr_noise.unwrap_or(DEFAULT_VBR_NOISE_DB);
        if !noise_offset_db.is_finite() {
            return Err(format!(
                "--vbr-noise: not a finite number: {noise_offset_db}"
            ));
        }
        Rate::Variable(VariableRate {
            lowest,
            highest,
            noise_offset_db,
        })
    } else {
        if min_bitrate.is_some() || max_bitrate.is_some() || vbr_noise.is_some() {
            return Err("--min-bitrate, --max-bitrate and --vbr-noise need --vbr".to_owned());
        }
        let kbps = bitrate.unwrap_or(128);
        match Bitrate::from_kbps(version, kbps) {
            Some(bitrate) => Rate::Constant(bitrate),
            None => kbps
                .checked_mul(1000)
                .and_then(|bps| MeanRate::new(version, bps))
                .map(Rate::Mean)
                .ok_or(format!("unsupported bitrate at {sample_rate} Hz: {kbps}"))?,
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
    Ok(Options {
        sampling_frequency,
        channels,
        rate,
        joint_stereo,
        dual_channel,
        intensity_stereo,
        crc_protected,
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
            eprintln!("chelys: {message}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };

    let config = StreamConfig {
        rate: options.rate,
        sampling_frequency: options.sampling_frequency,
        mode: match (
            options.channels,
            options.dual_channel,
            options.joint_stereo,
            options.intensity_stereo,
        ) {
            (1, _, _, _) => Mode::SingleChannel,
            (_, true, _, _) => Mode::DualChannel,
            (_, _, false, false) => Mode::Stereo,
            (_, _, ms_stereo, intensity_stereo) => Mode::JointStereo {
                intensity_stereo,
                ms_stereo,
            },
        },
        crc_protected: options.crc_protected,
    };
    match encode(config, options.channels) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("chelys: {message}");
            ExitCode::FAILURE
        }
    }
}

/// Encodes stdin to stdout as it arrives.
fn encode(config: StreamConfig, channels: usize) -> Result<(), String> {
    let frame_bytes = 4 * channels;
    let mut stdin = std::io::stdin().lock();
    let file = seekable_stdout();
    let mut stdout = std::io::stdout().lock();
    let mut encoder = match file {
        Some(_) => Encoder::with_info_tag(config),
        None => Encoder::new(config),
    };
    let mut input = vec![0; 64 * 1024];
    // Bytes read but not yet decoded: the start of an incomplete sample frame.
    let mut held = 0;
    let mut pcm = vec![Vec::new(); channels];
    let mut stream = Vec::new();
    loop {
        let read = match stdin.read(&mut input[held..]) {
            Ok(0) => break,
            Ok(read) => read,
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => return Err(format!("reading stdin: {e}")),
        };
        let filled = held + read;
        let whole = filled - filled % frame_bytes;
        for samples in &mut pcm {
            samples.clear();
        }
        for (i, bytes) in input[..whole].chunks_exact(4).enumerate() {
            let sample = f32::from_le_bytes(bytes.try_into().unwrap());
            pcm[i % channels].push(f64::from(sample));
        }
        input.copy_within(whole..filled, 0);
        held = filled - whole;
        let slices: Vec<&[f64]> = pcm.iter().map(Vec::as_slice).collect();
        encoder.encode(&slices, &mut stream);
        write(&mut stdout, &mut stream)?;
    }
    if held != 0 {
        return Err(format!(
            "input is not a whole number of {frame_bytes}-byte sample frames"
        ));
    }
    let tag = encoder.finish(&mut stream);
    write(&mut stdout, &mut stream)?;
    stdout.flush().map_err(|e| format!("writing stdout: {e}"))?;
    match (tag, file) {
        (Some(tag), Some(mut file)) => write_tag(&mut file, &tag),
        _ => Ok(()),
    }
}

/// stdout, if it is a regular file at its start, which the Xing/Info tag
/// can be written back into once the stream has ended.
#[cfg(unix)]
fn seekable_stdout() -> Option<File> {
    use std::os::fd::AsFd;
    let fd = std::io::stdout().as_fd().try_clone_to_owned().ok()?;
    let mut file = File::from(fd);
    let at_start = file.stream_position().ok()? == 0;
    (file.metadata().ok()?.is_file() && at_start).then_some(file)
}

#[cfg(not(unix))]
fn seekable_stdout() -> Option<File> {
    None
}

/// Writes the Xing/Info tag's frame over the start of the stream in `file`.
/// A file opened for appending takes the write at its end instead; that is
/// undone, and the stream keeps the placeholder, a silent frame.
fn write_tag(file: &mut File, tag: &[u8]) -> Result<(), String> {
    let error = |e: std::io::Error| format!("writing the Xing/Info tag: {e}");
    let len = file.metadata().map_err(error)?.len();
    file.seek(SeekFrom::Start(0)).map_err(error)?;
    file.write_all(tag).map_err(error)?;
    if file.stream_position().map_err(error)? != tag.len() as u64 {
        file.set_len(len).map_err(error)?;
        eprintln!("chelys: stdout is appended to, so the Xing/Info tag is left out");
    }
    Ok(())
}

/// Writes `stream` to `stdout` and empties it.
fn write(stdout: &mut impl Write, stream: &mut Vec<u8>) -> Result<(), String> {
    stdout
        .write_all(stream)
        .map_err(|e| format!("writing stdout: {e}"))?;
    stream.clear();
    Ok(())
}
