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

/// Digital silence skips the loops and codes to nothing. Its frames match
/// those of `encode_silence` except for `main_data_begin`, which grows as
/// the bit reservoir saves the unused main data, and they decode to silence.
#[test]
fn silence_codes_to_nothing() {
    let config = config(128, SamplingFrequency::Hz48000, Mode::Stereo);
    let zeros = vec![0.0; 10 * 1152];
    let stream = encode_pcm(&config, &[&zeros, &zeros]);
    let expected = encode_silence(&config, 11);
    assert_eq!(stream.len(), expected.len());
    // 128 kbit/s at 48 kHz: 384-byte frames, never padded. main_data_begin
    // is the first 9 bits after the 4-byte header.
    let mut begins = Vec::new();
    for (frame, (got, want)) in stream.chunks(384).zip(expected.chunks(384)).enumerate() {
        let begin = (usize::from(got[4]) << 1) | usize::from(got[5] >> 7);
        begins.push(begin);
        let masked = |bytes: &[u8]| {
            let mut bytes = bytes.to_vec();
            bytes[4] = 0;
            bytes[5] &= 0x7F;
            bytes
        };
        assert_eq!(masked(got), masked(want), "frame {frame}");
    }
    assert_eq!(begins[0], 0);
    assert!(begins.windows(2).all(|w| w[1] >= w[0]), "{begins:?}");
    let decoded = ffmpeg::decode_mp3(&stream);
    assert_eq!(decoded.diagnostics, "");
    assert!(decoded.samples.iter().all(|&s| s == 0.0));
}

/// The `main_data_begin` of each frame of a stream with unpadded and padded
/// frames of MPEG-1 Layer III, read from the headers (clauses 2.4.2.3 and
/// 2.4.2.7).
fn main_data_begins(stream: &[u8]) -> Vec<usize> {
    const KBPS: [u32; 15] = [
        0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
    ];
    const HZ: [u32; 3] = [44_100, 48_000, 32_000];
    let mut begins = Vec::new();
    let mut at = 0;
    while at + 6 <= stream.len() {
        let header = &stream[at..];
        assert_eq!(header[0], 0xFF, "no syncword at {at}");
        let kbps = KBPS[usize::from(header[2] >> 4)];
        let hz = HZ[usize::from((header[2] >> 2) & 3)];
        let padding = usize::from((header[2] >> 1) & 1);
        let crc = if header[1] & 1 == 0 { 2 } else { 0 };
        let side_info = &header[4 + crc..];
        begins.push((usize::from(side_info[0]) << 1) | usize::from(side_info[1] >> 7));
        at += (144_000 * kbps / hz) as usize + padding;
    }
    begins
}

/// Quiet passages leave bits for the loud passages after them: main data
/// then starts more than a whole slot before its frame, and the decoder must
/// still find it.
#[test]
fn main_data_spans_frames_after_quiet_passages() {
    let fs = 44_100.0;
    let mut state = 0x9E37_79B9u32;
    let mut noise = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        f64::from(state) / f64::from(u32::MAX) * 2.0 - 1.0
    };
    // Half a second of a quiet tone, then half a second of loud noise, twice.
    let input: Vec<f64> = (0..2 * 44_100)
        .map(|t| {
            if (t / 22_050) % 2 == 0 {
                0.01 * (TAU * 440.0 * t as f64 / fs).sin()
            } else {
                0.5 * noise()
            }
        })
        .collect();
    let config = config(64, SamplingFrequency::Hz44100, Mode::SingleChannel);
    let stream = encode_pcm(&config, &[&input]);
    // A 64 kbit/s frame at 44.1 kHz has 208 bytes, 187 of them main data.
    let begins = main_data_begins(&stream);
    let deepest = begins.iter().copied().max().unwrap();
    assert!(
        deepest > 187,
        "main_data_begin never exceeds a slot: {begins:?}"
    );
    let decoded = ffmpeg::decode_mp3(&stream);
    assert_eq!(decoded.diagnostics, "");
    // Main data found in the wrong place would ruin the quiet passages,
    // which otherwise decode almost exactly. Each passage is measured away
    // from its edges, where the loud noise's quantisation noise spreads.
    for (passage, range) in [(0, 2304..19_746), (2, 46_404..63_846)] {
        let (signal, noise) = range.fold((0.0, 0.0), |(s, n), t| {
            let x = input[t];
            let y = f64::from(decoded.samples[t + 1057]);
            (s + x * x, n + (x - y) * (x - y))
        });
        let snr = 10.0 * (signal / noise).log10();
        assert!(snr > 40.0, "quiet passage {passage}: {snr:.1} dB");
    }
}
