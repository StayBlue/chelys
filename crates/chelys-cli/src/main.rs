//! Encodes raw PCM from stdin as MPEG-1 Layer III on stdout. Input formats are
//! left to other tools, for example:
//!
//! ```sh
//! ffmpeg -i song.flac -f f32le -ar 44100 -ac 2 - | chelys --sample-rate 44100 --channels 2 > song.mp3
//! ```

use std::io::{Read, Write};
use std::process::ExitCode;

use chelys::mp3::header::{Bitrate, Mode, SamplingFrequency};
use chelys::mp3::{StreamConfig, encode_pcm};

const USAGE: &str = "\
usage: chelys --sample-rate HZ --channels N [--bitrate KBPS] [--joint-stereo]
              < input.f32 > output.mp3

Reads interleaved 32-bit little-endian float PCM from stdin and writes MPEG-1
Layer III to stdout.

  --sample-rate HZ   32000, 44100 or 48000
  --channels N       1 or 2
  --bitrate KBPS     32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256
                     or 320 (default 128)
  --joint-stereo     code each frame of two channels in MS stereo where the
                     channels are alike (default: left/right stereo)";

struct Options {
    sampling_frequency: SamplingFrequency,
    channels: usize,
    bitrate: Bitrate,
    joint_stereo: bool,
}

fn parse(args: impl IntoIterator<Item = String>) -> Result<Options, String> {
    let (mut sample_rate, mut channels, mut bitrate) = (None, None, 128);
    let mut joint_stereo = false;
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        let mut value = |name: &str| -> Result<u32, String> {
            let value = args.next().ok_or(format!("{name} needs a value"))?;
            value
                .parse()
                .map_err(|_| format!("{name}: not a number: {value}"))
        };
        match flag.as_str() {
            "--sample-rate" => sample_rate = Some(value("--sample-rate")?),
            "--channels" => channels = Some(value("--channels")?),
            "--bitrate" => bitrate = value("--bitrate")?,
            "--joint-stereo" => joint_stereo = true,
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
    let bitrate = Bitrate::from_kbps(bitrate).ok_or(format!("unsupported bitrate: {bitrate}"))?;
    if joint_stereo && channels != 2 {
        return Err("--joint-stereo needs two channels".to_owned());
    }
    Ok(Options {
        sampling_frequency,
        channels,
        bitrate,
        joint_stereo,
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

    let mut input = Vec::new();
    if let Err(e) = std::io::stdin().lock().read_to_end(&mut input) {
        eprintln!("chelys: reading stdin: {e}");
        return ExitCode::FAILURE;
    }
    let frame_bytes = 4 * options.channels;
    if input.len() % frame_bytes != 0 {
        eprintln!("chelys: input is not a whole number of {frame_bytes}-byte sample frames");
        return ExitCode::FAILURE;
    }
    let mut channels = vec![Vec::with_capacity(input.len() / frame_bytes); options.channels];
    for (i, bytes) in input.chunks_exact(4).enumerate() {
        let sample = f32::from_le_bytes(bytes.try_into().unwrap());
        channels[i % options.channels].push(f64::from(sample));
    }

    let config = StreamConfig {
        bitrate: options.bitrate,
        sampling_frequency: options.sampling_frequency,
        mode: match (options.channels, options.joint_stereo) {
            (1, _) => Mode::SingleChannel,
            (_, false) => Mode::Stereo,
            (_, true) => Mode::JointStereo {
                intensity_stereo: false,
                ms_stereo: true,
            },
        },
        crc_protected: false,
    };
    let slices: Vec<&[f64]> = channels.iter().map(Vec::as_slice).collect();
    let stream = encode_pcm(&config, &slices);
    if let Err(e) = std::io::stdout().lock().write_all(&stream) {
        eprintln!("chelys: writing stdout: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
