// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The Xing/Info tag ("Mp3 Info Tag rev 1 specifications - draft 0"), as
//! ffmpeg, mpg123 and LAME read it.

use std::f64::consts::TAU;

use chelys::mp3::header::{Bitrate, Flags, Mode, SamplingFrequency};
use chelys::mp3::rate::{MeanRate, Rate, VariableRate};
use chelys::mp3::{DELAY, StreamConfig, encode_pcm_with_info_tag};
use chelys_conformance::{Decoder, ffmpeg, lame, mpg123};

/// A tone that rises in pitch, so that only one lag lines it up.
fn chirp(fs: SamplingFrequency, len: usize) -> Vec<f64> {
    let hz = f64::from(fs.hz());
    (0..len)
        .map(|t| {
            let s = t as f64 / hz;
            let rise = hz / 8.0 * t as f64 / len as f64;
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
/// beside it: some of the bitrates are low, and the chirp decodes noisily
/// at them, but a lag one sample off is worse still.
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

/// At every sampling frequency, in one channel and two, at constant, mean
/// and variable bitrates, with and without the error check:
///
/// - mpg123 and LAME give the input exactly: as many samples, lined up to
///   the sample. They skip the tag's frame and the encoder delay and 529
///   samples more, and leave out the padding at the end, which they could
///   not do were the delays or the frame count misplaced or misread.
/// - ffmpeg skips the tag's frame, and gives the duration of the frames
///   after it, but applies neither the delay nor the padding: its output is
///   every frame's samples, the input lagging by [`DELAY`] (but see
///   `ffmpeg_applies_the_delays_for_some_encoders_only`).
#[test]
fn decoders_read_the_tag_at_every_sampling_frequency() {
    use SamplingFrequency::*;
    let at = |fs: SamplingFrequency, kbps| Bitrate::from_kbps(fs.version(), kbps).unwrap();
    let ms = Mode::JointStereo {
        intensity_stereo: false,
        ms_stereo: true,
    };
    let cases = [
        (Hz44100, Rate::Constant(at(Hz44100, 128)), ms, false),
        (Hz48000, Rate::Constant(at(Hz48000, 32)), Mode::Stereo, true),
        (
            Hz32000,
            Rate::Variable(VariableRate {
                lowest: at(Hz32000, 32),
                highest: at(Hz32000, 320),
                noise_offset_db: 0.0,
            }),
            Mode::SingleChannel,
            false,
        ),
        (
            Hz22050,
            Rate::Mean(MeanRate::new(Hz22050.version(), 50_000).unwrap()),
            ms,
            true,
        ),
        (
            Hz24000,
            Rate::Constant(at(Hz24000, 8)),
            Mode::SingleChannel,
            false,
        ),
        (
            Hz16000,
            Rate::Variable(VariableRate {
                lowest: at(Hz16000, 8),
                highest: at(Hz16000, 160),
                noise_offset_db: 0.0,
            }),
            Mode::Stereo,
            false,
        ),
        (Hz11025, Rate::Constant(at(Hz11025, 32)), ms, false),
        (
            Hz12000,
            Rate::Constant(at(Hz12000, 8)),
            Mode::SingleChannel,
            true,
        ),
        (
            Hz8000,
            Rate::Variable(VariableRate {
                lowest: at(Hz8000, 8),
                highest: at(Hz8000, 64),
                noise_offset_db: 0.0,
            }),
            Mode::SingleChannel,
            false,
        ),
    ];
    std::thread::scope(|scope| {
        for (fs, rate, mode, crc_protected) in cases {
            scope.spawn(move || {
                let config = StreamConfig {
                    rate,
                    sampling_frequency: fs,
                    mode,
                    crc_protected,
                    flags: Flags::default(),
                };
                let len = fs.hz() as usize * 3 / 5 + 123;
                let input = chirp(fs, len);
                let channels = mode.channels();
                let stream = match channels {
                    1 => encode_pcm_with_info_tag(&config, &[&input]),
                    _ => encode_pcm_with_info_tag(&config, &[&input, &input]),
                };
                let samples_per_frame = fs.version().samples_per_frame();
                let frames = (len + DELAY).div_ceil(samples_per_frame);
                let case = format!("{fs:?}, {rate:?}, {mode:?}, error check {crc_protected}");

                for Decoder { name, decode, .. } in [
                    Decoder {
                        name: "mpg123",
                        decode: mpg123::decode_mp3,
                        sample_rate: mpg123::sample_rate,
                    },
                    Decoder {
                        name: "LAME",
                        decode: lame::decode_mp3,
                        sample_rate: lame::sample_rate,
                    },
                ] {
                    let decoded = decode(&stream);
                    assert_eq!(decoded.diagnostics, "", "{name}, {case}");
                    let output = first_channel(&decoded.samples, channels);
                    assert_eq!(output.len(), len, "{name}, {case}");
                    if let Err(snrs) = lines_up(&input, &output, 0) {
                        panic!("{name}, {case}: {snrs}");
                    }
                }

                let decoded = ffmpeg::decode_mp3(&stream);
                assert_eq!(decoded.diagnostics, "", "ffmpeg, {case}");
                let output = first_channel(&decoded.samples, channels);
                assert_eq!(output.len(), frames * samples_per_frame, "ffmpeg, {case}");
                if let Err(snrs) = lines_up(&input, &output, DELAY as isize) {
                    panic!("ffmpeg, {case}: {snrs}");
                }
                let duration = ffmpeg::duration_mp3(&stream);
                let frames_duration = (frames * samples_per_frame) as f64 / f64::from(fs.hz());
                assert!(
                    (duration - frames_duration).abs() < 1e-3,
                    "ffmpeg, {case}: {duration} s for {frames_duration} s of frames"
                );
            });
        }
    });
}

/// Seeking by the table, ffmpeg starts decoding within a frame of where it
/// was asked to, in a stream whose frames are small beside the tag's, so
/// that a table whose offsets left out the tag's frame would send it some
/// 2000 samples early. The decoded output, untrimmed, lags by [`DELAY`]
/// from wherever decoding starts.
#[test]
fn ffmpeg_seeks_by_the_table() {
    let fs = SamplingFrequency::Hz8000;
    let config = StreamConfig {
        rate: Rate::Constant(Bitrate::from_kbps(fs.version(), 8).unwrap()),
        sampling_frequency: fs,
        mode: Mode::SingleChannel,
        crc_protected: false,
        flags: Flags::default(),
    };
    let input: Vec<f64> = chirp(fs, 4 * 8000).iter().map(|x| x / 30.0).collect();
    let stream = encode_pcm_with_info_tag(&config, &[&input]);
    for from in [1.0, 2.0, 3.0] {
        let decoded = ffmpeg::decode_mp3_seeking(&stream, from, 0.5);
        assert_eq!(decoded.diagnostics, "", "from {from} s");
        let output = &decoded.samples;
        // The input sample that the decoded output's sample 1000 is,
        // where it lines up best with the input.
        let target = (from * 8000.0) as usize;
        let start = (target - 3000..target + 3000)
            .min_by(|&a, &b| {
                let error = |k: usize| -> f64 {
                    (1000..2600)
                        .map(|t| (input[k + t] - f64::from(output[t])).powi(2))
                        .sum()
                };
                error(a).total_cmp(&error(b))
            })
            .unwrap();
        let error = start as isize - target as isize;
        assert!(
            error.abs() <= 576,
            "from {from} s: decoding starts {error} samples off"
        );
    }
}

/// ffmpeg applies the encoder delay and the padding when the tag's encoder
/// string starts with "LAME" or "Lav", and not otherwise: changed to say so,
/// and nothing else, the tag makes ffmpeg give the input exactly, as mpg123
/// and LAME do whatever the string.
#[test]
fn ffmpeg_applies_the_delays_for_some_encoders_only() {
    let fs = SamplingFrequency::Hz44100;
    let config = StreamConfig {
        rate: Rate::Constant(Bitrate::from_kbps(fs.version(), 128).unwrap()),
        sampling_frequency: fs,
        mode: Mode::SingleChannel,
        crc_protected: false,
        flags: Flags::default(),
    };
    let len = 30_000;
    let input = chirp(fs, len);
    let stream = encode_pcm_with_info_tag(&config, &[&input]);
    // The encoder string, after the header, 17 bytes of side information
    // and 120 of the Xing fields.
    let at = 4 + 17 + 120;
    assert_eq!(&stream[at..at + 6], b"chelys");
    for (encoder, trimmed) in [
        (b"LAME3.100", true),
        (b"Lavc62.1 ", true),
        (b"Lavf62.1 ", true),
        (b"GOGO3.02b", false),
        (b"chelys   ", false),
    ] {
        let mut stream = stream.clone();
        stream[at..at + 9].copy_from_slice(encoder);
        let output = ffmpeg::decode_mp3(&stream).samples;
        let name = String::from_utf8_lossy(encoder);
        if trimmed {
            assert_eq!(output.len(), len, "{name}");
            lines_up(&input, &output, 0).unwrap_or_else(|snrs| panic!("{name}: {snrs}"));
        } else {
            assert_eq!(output.len(), (len + DELAY).div_ceil(1152) * 1152, "{name}");
            lines_up(&input, &output, DELAY as isize)
                .unwrap_or_else(|snrs| panic!("{name}: {snrs}"));
        }
    }
}
