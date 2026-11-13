//! PCM encoded at a fixed bitrate by the inner iteration loop, decoded by
//! ffmpeg.

use std::f64::consts::TAU;

use chelys::mp3::header::{Bitrate, Mode, SamplingFrequency};
use chelys::mp3::{StreamConfig, encode_pcm, encode_silence};
use chelys_conformance::ffmpeg;

fn config(kbps: u32, sampling_frequency: SamplingFrequency, mode: Mode) -> StreamConfig {
    StreamConfig {
        bitrate: Bitrate::from_kbps(kbps).unwrap(),
        sampling_frequency,
        mode,
        crc_protected: false,
    }
}

fn tones(fs: f64, seconds: f64) -> Vec<f64> {
    let frequencies = [220.0, 997.0, 3_100.0, 7_000.0];
    (0..(seconds * fs) as usize)
        .map(|t| {
            frequencies
                .iter()
                .map(|f| 0.15 * (TAU * f * t as f64 / fs).sin())
                .sum()
        })
        .collect()
}

/// The decoder's output delay behind the input is a constant 1057 samples
/// for this filterbank (measured in the filterbank tests); the SNR in dB of
/// the decoded `channel` against `input` over the steady middle part.
fn snr(input: &[f64], output: &[f32], channels: usize, channel: usize) -> f64 {
    const DELAY: usize = 1057;
    let (signal, noise) = (2304..input.len() - 2304).fold((0.0, 0.0), |(s, n), t| {
        let x = input[t];
        let y = f64::from(output[(t + DELAY) * channels + channel]);
        (s + x * x, n + (x - y) * (x - y))
    });
    10.0 * (signal / noise).log10()
}

#[test]
fn quality_rises_with_the_bitrate() {
    let fs = SamplingFrequency::Hz44100;
    let input = tones(44_100.0, 1.0);
    let snrs: Vec<f64> = [32, 64, 128, 320]
        .into_iter()
        .map(|kbps| {
            let stream = encode_pcm(&config(kbps, fs, Mode::SingleChannel), &[&input]);
            let decoded = ffmpeg::decode_mp3(&stream);
            assert_eq!(decoded.diagnostics, "", "{kbps} kbit/s");
            snr(&input, &decoded.samples, 1, 0)
        })
        .collect();
    assert!(
        snrs.windows(2).all(|w| w[1] > w[0]),
        "SNR by bitrate: {snrs:?}"
    );
    assert!(snrs[3] > 40.0, "SNR by bitrate: {snrs:?}");
}

/// Full-scale white noise is the hardest signal to fit; at the lowest bitrate
/// every granule must still squeeze into its share of the frame.
#[test]
fn loud_noise_fits_at_the_lowest_bitrate() {
    let mut state = 0x1234_5678u32;
    let mut noise = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        f64::from(state) / f64::from(u32::MAX) * 2.0 - 1.0
    };
    let left: Vec<f64> = (0..48_000).map(|_| noise()).collect();
    let right: Vec<f64> = (0..48_000).map(|_| noise()).collect();
    for fs in SamplingFrequency::ALL {
        let stream = encode_pcm(&config(32, fs, Mode::Stereo), &[&left, &right]);
        let decoded = ffmpeg::decode_mp3(&stream);
        assert_eq!(decoded.diagnostics, "", "{fs:?}");
        assert!(decoded.samples.iter().any(|&s| s != 0.0));
    }
}

/// Digital silence skips the loops and codes to nothing.
#[test]
fn silence_encodes_like_encode_silence() {
    let config = config(128, SamplingFrequency::Hz48000, Mode::Stereo);
    let zeros = vec![0.0; 10 * 1152];
    assert_eq!(
        encode_pcm(&config, &[&zeros, &zeros]),
        encode_silence(&config, 11)
    );
}
