// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Free format (`bitrate_index` 0, clause 2.4.2.3), as mpg123 and LAME
//! decode it. ffmpeg does not.

use std::f64::consts::TAU;
use std::process::Command;

use chelys::mp3::header::{Bitrate, Flags, Mode, SamplingFrequency};
use chelys::mp3::rate::Rate;
use chelys::mp3::{DELAY, StreamConfig, encode_pcm, encode_pcm_with_info_tag, info_tag};
use chelys_conformance::{Decoder, lame, mpg123};

/// A tone that rises in pitch, so that only one lag lines it up.
fn chirp(fs: SamplingFrequency, len: usize) -> Vec<f64> {
    let hz = f64::from(fs.hz());
    (0..len)
        .map(|t| {
            let s = t as f64 / hz;
            let rise = hz / 16.0 * t as f64 / len as f64;
            0.3 * (TAU * (200.0 + rise) * s).sin()
        })
        .collect()
}

/// The SNR in dB of `output` against `input`, with `output` behind by
/// `lag`, over the input away from its ends.
fn snr(input: &[f64], output: &[f32], lag: isize) -> f64 {
    let (signal, noise) = (2000..input.len() - 2000).fold((0.0, 0.0), |(s, n), t| {
        let x = input[t];
        let y = f64::from(output[(t as isize + lag) as usize]);
        (s + x * x, n + (x - y) * (x - y))
    });
    10.0 * (signal / noise).log10()
}

/// Whether `output` lines up with `input` at `lag` and not at the lags
/// beside it, which it could not do were a frame lost or misread.
fn lines_up(input: &[f64], output: &[f32], lag: isize) -> Result<(), String> {
    let [early, exact, late] = [lag - 1, lag, lag + 1].map(|lag| snr(input, output, lag));
    match exact > early + 3.0 && exact > late + 3.0 {
        true => Ok(()),
        false => Err(format!(
            "{early:.1}, {exact:.1}, {late:.1} dB at lags {}, {lag}, {}",
            lag - 1,
            lag + 1
        )),
    }
}

/// The first channel of interleaved samples.
fn first_channel(samples: &[f32], channels: usize) -> Vec<f32> {
    samples.iter().step_by(channels).copied().collect()
}

const MS: Mode = Mode::JointStereo {
    intensity_stereo: false,
    ms_stereo: true,
};

/// In free format at every sampling frequency, in one channel and two, with
/// and without the error check, from frames shorter than any of the table
/// (38 bytes at 48 kHz) to frames longer than the decoder's buffer (1440
/// at 32 kHz), with padding and without, mpg123 and LAME decode every
/// frame: the input lines up to the sample. Where the frames hold the
/// Xing/Info tag, they read it, and give the input exactly; where they do
/// not, the stream has no tag, and their output is untrimmed, the input
/// lagging by [`DELAY`] in mpg123's and by 529 samples less in LAME's.
///
/// This is the evidence for free format at MPEG-2.5's sampling
/// frequencies, and for the tag's frame in free format, which the bitrate
/// and the tag mark as settled by decoding with mpg123 and LAME.
#[test]
fn mpg123_and_lame_decode_free_format() {
    use SamplingFrequency::*;
    let cases = [
        (Hz44100, 100_000, Mode::Stereo, false),
        (Hz48000, 12_667, Mode::SingleChannel, true),
        (Hz32000, 319_999, MS, false),
        (Hz22050, 50_000, MS, true),
        (Hz24000, 8_001, Mode::SingleChannel, false),
        (Hz16000, 159_999, Mode::Stereo, true),
        (Hz11025, 33_333, MS, false),
        (Hz12000, 9_000, Mode::SingleChannel, false),
        (Hz8000, 20_000, Mode::SingleChannel, true),
    ];
    std::thread::scope(|scope| {
        for (fs, bits_per_second, mode, crc_protected) in cases {
            scope.spawn(move || {
                let config = StreamConfig {
                    rate: Rate::Constant(Bitrate::free(fs.version(), bits_per_second).unwrap()),
                    sampling_frequency: fs,
                    mode,
                    crc_protected,
                    flags: Flags::default(),
                };
                let len = fs.hz() as usize * 3 / 5 + 123;
                let input = chirp(fs, len);
                let channels = mode.channels();
                let pcm = vec![&input[..]; channels];
                let stream = encode_pcm_with_info_tag(&config, &pcm);
                let tagged = info_tag::header(&config).is_some();
                assert_eq!(tagged, stream != encode_pcm(&config, &pcm));
                let case = format!("{fs:?} at {bits_per_second} bit/s, tag {tagged}");

                for (Decoder { name, decode, .. }, untrimmed_lag) in [
                    (
                        Decoder {
                            name: "mpg123",
                            decode: mpg123::decode_mp3,
                            sample_rate: mpg123::sample_rate,
                        },
                        DELAY,
                    ),
                    (
                        Decoder {
                            name: "LAME",
                            decode: lame::decode_mp3,
                            sample_rate: lame::sample_rate,
                        },
                        DELAY - 529,
                    ),
                ] {
                    let decoded = decode(&stream);
                    assert_eq!(decoded.diagnostics, "", "{name}, {case}");
                    let output = first_channel(&decoded.samples, channels);
                    let lag = match tagged {
                        true => {
                            assert_eq!(output.len(), len, "{name}, {case}");
                            0
                        }
                        false => {
                            assert!(output.len() >= len + untrimmed_lag, "{name}, {case}");
                            untrimmed_lag
                        }
                    };
                    if let Err(snrs) = lines_up(&input, &output, lag as isize) {
                        panic!("{name}, {case}: {snrs}");
                    }
                }
            });
        }
    });
}

/// ffmpeg n9.0.2 decodes none of a stream in free format, though it decodes
/// the same stream at a bitrate of the table: its demuxer does not find
/// the frames, and given them in a WAV file of MPEG Layer III, its decoder
/// reports each frame's header missing.
#[test]
fn ffmpeg_does_not_decode_free_format() {
    let fs = SamplingFrequency::Hz44100;
    let input = chirp(fs, 44_100);
    let decodes = |bitrate| {
        let config = StreamConfig {
            rate: Rate::Constant(bitrate),
            sampling_frequency: fs,
            mode: Mode::SingleChannel,
            crc_protected: false,
            flags: Flags::default(),
        };
        let path = std::env::temp_dir().join(format!("chelys-free-{}.mp3", std::process::id()));
        std::fs::write(&path, encode_pcm(&config, &[&input])).unwrap();
        let output = Command::new("ffmpeg")
            .args(["-nostdin", "-v", "error", "-f", "mp3", "-i"])
            .arg(&path)
            .args(["-f", "f32le", "-"])
            .output()
            .expect("ffmpeg runs");
        let _ = std::fs::remove_file(&path);
        output.status.success() && !output.stdout.is_empty()
    };
    assert!(decodes(Bitrate::from_kbps(fs.version(), 96).unwrap()));
    assert!(!decodes(Bitrate::free(fs.version(), 100_000).unwrap()));
}
