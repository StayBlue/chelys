//! Encodes raw PCM from stdin as MPEG-1 Layer III on stdout. Input formats are
//! left to other tools, for example:
//!
//! ```sh
//! ffmpeg -i song.flac -f f32le -ar 44100 -ac 2 - | chelys --sample-rate 44100 --channels 2 > song.mp3
//! ```

use std::io::{ErrorKind, Read, Write};
use std::process::ExitCode;

use chelys::mp3::header::{Bitrate, Mode, SamplingFrequency};
use chelys::mp3::{Encoder, StreamConfig};

const USAGE: &str = "\
usage: chelys --sample-rate HZ --channels N [--bitrate KBPS] [--joint-stereo]
              [--intensity-stereo] < input.f32 > output.mp3

Reads interleaved 32-bit little-endian float PCM from stdin and writes MPEG-1
Layer III to stdout.

  --sample-rate HZ   32000, 44100 or 48000
  --channels N       1 or 2
  --bitrate KBPS     32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256
                     or 320 (default 128)
  --joint-stereo     code each frame of two channels in MS stereo where the
                     channels are alike (default: left/right stereo)
  --intensity-stereo code the upper bands of each frame of two channels in
                     intensity stereo where the frame is short of bits,
                     keeping only the level of each channel there and not
                     its fine structure; for low bitrates (default: off)";

struct Options {
    sampling_frequency: SamplingFrequency,
    channels: usize,
    bitrate: Bitrate,
    joint_stereo: bool,
    /// Off unless asked for, following ffmpeg's command line: its MP3
    /// encoder offers no switch for intensity stereo, and with its defaults
    /// it coded no frame in it, at 32 to 128 kbit/s, when observed from the
    /// frame headers of its output (ffmpeg n9.0.2). Intensity stereo trades
    /// the stereo image above its bound for bits, a trade for low bitrates,
    /// which content that relies on the image may not want even there.
    intensity_stereo: bool,
}

fn parse(args: impl IntoIterator<Item = String>) -> Result<Options, String> {
    let (mut sample_rate, mut channels, mut bitrate) = (None, None, 128);
    let (mut joint_stereo, mut intensity_stereo) = (false, false);
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
            "--intensity-stereo" => intensity_stereo = true,
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
    if intensity_stereo && channels != 2 {
        return Err("--intensity-stereo needs two channels".to_owned());
    }
    Ok(Options {
        sampling_frequency,
        channels,
        bitrate,
        joint_stereo,
        intensity_stereo,
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
        bitrate: options.bitrate,
        sampling_frequency: options.sampling_frequency,
        mode: match (
            options.channels,
            options.joint_stereo,
            options.intensity_stereo,
        ) {
            (1, _, _) => Mode::SingleChannel,
            (_, false, false) => Mode::Stereo,
            (_, ms_stereo, intensity_stereo) => Mode::JointStereo {
                intensity_stereo,
                ms_stereo,
            },
        },
        crc_protected: false,
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
    let mut stdout = std::io::stdout().lock();
    let mut encoder = Encoder::new(config);
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
    encoder.finish(&mut stream);
    write(&mut stdout, &mut stream)
}

/// Writes `stream` to `stdout` and empties it.
fn write(stdout: &mut impl Write, stream: &mut Vec<u8>) -> Result<(), String> {
    stdout
        .write_all(stream)
        .map_err(|e| format!("writing stdout: {e}"))?;
    stream.clear();
    Ok(())
}
