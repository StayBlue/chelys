// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! PCM encoded at a fixed bitrate by the inner iteration loop, decoded by
//! ffmpeg.

use std::f64::consts::TAU;

use chelys::mp3::header::{Bitrate, Mode, SamplingFrequency};
use chelys::mp3::{StreamConfig, encode_pcm, encode_silence};
use chelys_conformance::ffmpeg;

fn config(kbps: u32, sampling_frequency: SamplingFrequency, mode: Mode) -> StreamConfig {
    StreamConfig {
        bitrate: Bitrate::from_kbps(sampling_frequency.version(), kbps).unwrap(),
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

/// At the lower sampling frequencies too, with frames of one granule and
/// the bitrates and scalefactor bands of ISO/IEC 13818-3. (From about 64
/// kbit/s up, these tones in one channel reach some 78 dB, the finest the
/// quantizer goes.)
#[test]
fn quality_rises_with_the_bitrate_at_the_lower_sampling_frequencies() {
    for fs in [
        SamplingFrequency::Hz16000,
        SamplingFrequency::Hz22050,
        SamplingFrequency::Hz24000,
    ] {
        let input = tones(f64::from(fs.hz()), 1.0);
        let snrs: Vec<f64> = [16, 24, 32, 48]
            .into_iter()
            .map(|kbps| {
                let stream = encode_pcm(&config(kbps, fs, Mode::SingleChannel), &[&input]);
                let decoded = ffmpeg::decode_mp3(&stream);
                assert_eq!(decoded.diagnostics, "", "{fs:?} at {kbps} kbit/s");
                snr(&input, &decoded.samples, 1, 0)
            })
            .collect();
        assert!(
            snrs.windows(2).all(|w| w[1] > w[0]),
            "{fs:?}: SNR by bitrate: {snrs:?}"
        );
        assert!(snrs[3] > 40.0, "{fs:?}: SNR by bitrate: {snrs:?}");
    }
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

/// The side information of each frame of a stream with unpadded and padded
/// frames of MPEG-1 Layer III, found from the headers (clauses 2.4.2.3 and
/// 2.4.2.7).
fn side_infos(stream: &[u8]) -> Vec<&[u8]> {
    const KBPS: [u32; 15] = [
        0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
    ];
    const HZ: [u32; 3] = [44_100, 48_000, 32_000];
    let mut side_infos = Vec::new();
    let mut at = 0;
    while at + 6 <= stream.len() {
        let header = &stream[at..];
        assert_eq!(header[0], 0xFF, "no syncword at {at}");
        let kbps = KBPS[usize::from(header[2] >> 4)];
        let hz = HZ[usize::from((header[2] >> 2) & 3)];
        let padding = usize::from((header[2] >> 1) & 1);
        let crc = if header[1] & 1 == 0 { 2 } else { 0 };
        side_infos.push(&header[4 + crc..]);
        at += (144_000 * kbps / hz) as usize + padding;
    }
    side_infos
}

/// The `main_data_begin` of each frame.
fn main_data_begins(stream: &[u8]) -> Vec<usize> {
    side_infos(stream)
        .iter()
        .map(|side_info| (usize::from(side_info[0]) << 1) | usize::from(side_info[1] >> 7))
        .collect()
}

/// The four `scfsi` bits of each frame of a single-channel stream, after the
/// 9 bits of `main_data_begin` and 5 of `private_bits` (clause 2.4.1.7).
fn single_channel_scfsi(stream: &[u8]) -> Vec<u8> {
    side_infos(stream)
        .iter()
        .map(|side_info| ((side_info[1] & 0b11) << 2) | (side_info[2] >> 6))
        .collect()
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

/// Short blocks keep the quantisation noise of an attack from spreading
/// back over the long window before it. A quiet tone is interrupted by
/// bursts of noise 30 dB louder, starting early and late in a granule. With
/// long blocks only, the noise rises to about -60 dBFS from 1024 samples
/// before each burst; with window switching it stays far below that until
/// the last few hundred samples, which the short windows and the polyphase
/// filter span.
#[test]
fn attacks_switch_to_short_blocks_and_shorten_pre_echo() {
    const DELAY: usize = 1057;
    let fs = 44_100.0;
    let attacks = [22_050 + 162 - 22_050 % 576, 44_100 + 461 - 44_100 % 576];
    let mut seed = 1u64;
    let input: Vec<f64> = (0..66_150)
        .map(|t| {
            let tone = 0.02 * (TAU * 440.0 * t as f64 / fs).sin();
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let noise = (seed >> 11) as f64 / (1u64 << 52) as f64 - 1.0;
            let burst: f64 = attacks
                .iter()
                .filter(|&&a| t >= a)
                .map(|&a| 0.6 * (-((t - a) as f64) / (0.03 * fs)).exp() * noise)
                .sum();
            tone + burst
        })
        .collect();
    let stream = encode_pcm(
        &config(128, SamplingFrequency::Hz44100, Mode::SingleChannel),
        &[&input],
    );
    let decoded = ffmpeg::decode_mp3(&stream);
    assert_eq!(decoded.diagnostics, "");
    for a in attacks {
        for start in (a - 1024..a - 512).step_by(32) {
            let noise = (start..start + 32)
                .map(|t| (f64::from(decoded.samples[t + DELAY]) - input[t]).powi(2))
                .sum::<f64>()
                / 32.0;
            let db = 10.0 * noise.log10();
            assert!(
                db < -80.0,
                "attack at {a}: {db:.1} dBFS of noise {} samples before it",
                a - start
            );
        }
    }
}

/// In steady passages the second granule of a frame takes over the first
/// granule's scalefactors in some scfsi bands, and the stream still decodes
/// as well as the bitrate allows.
#[test]
fn steady_tones_share_scalefactors_between_granules() {
    let fs = SamplingFrequency::Hz44100;
    let input = tones(44_100.0, 1.0);
    let stream = encode_pcm(&config(128, fs, Mode::SingleChannel), &[&input]);
    let scfsi = single_channel_scfsi(&stream);
    let sharing = scfsi.iter().filter(|&&bits| bits != 0).count();
    assert!(
        sharing > scfsi.len() / 2,
        "{sharing} of {} frames use scfsi: {scfsi:?}",
        scfsi.len()
    );
    let decoded = ffmpeg::decode_mp3(&stream);
    assert_eq!(decoded.diagnostics, "");
    let snr = snr(&input, &decoded.samples, 1, 0);
    assert!(snr > 25.0, "{snr:.1} dB");
}

/// The `mode_extension` of each frame of a stream (clause 2.4.2.3).
fn mode_extensions(stream: &[u8]) -> Vec<u8> {
    let mut frames = Vec::new();
    let mut at = 0;
    while at + 4 <= stream.len() {
        let header = &stream[at..];
        assert_eq!(header[0], 0xFF, "no syncword at {at}");
        frames.push(header[3] >> 4 & 0b11);
        at += frame_len(header);
    }
    frames
}

/// Whether each frame of a stream is coded in MS stereo: bit 1 of its
/// `mode_extension`.
fn ms_stereo_frames(stream: &[u8]) -> Vec<bool> {
    mode_extensions(stream)
        .iter()
        .map(|e| e & 0b10 != 0)
        .collect()
}

/// Whether each frame of a stream is coded in intensity stereo: bit 0 of
/// its `mode_extension`.
fn intensity_stereo_frames(stream: &[u8]) -> Vec<bool> {
    mode_extensions(stream)
        .iter()
        .map(|e| e & 0b01 != 0)
        .collect()
}

/// The length of a frame from its header: MPEG-1 Layer III (clause 2.4.2.3),
/// or with `ID` 0 the lower sampling frequencies, whose frames hold half as
/// many slots (clause 2.4.2.3 of ISO/IEC 13818-3).
fn frame_len(header: &[u8]) -> usize {
    const KBPS: [[u32; 15]; 2] = [
        [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
        [
            0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
        ],
    ];
    const HZ: [[u32; 3]; 2] = [[22_050, 24_000, 16_000], [44_100, 48_000, 32_000]];
    let id = usize::from(header[1] >> 3 & 1);
    let kbps = KBPS[id][usize::from(header[2] >> 4)];
    let hz = HZ[id][usize::from((header[2] >> 2) & 3)];
    let slots_per_kbit = if id == 1 { 144_000 } else { 72_000 };
    (slots_per_kbit * kbps / hz) as usize + usize::from((header[2] >> 1) & 1)
}

/// With joint stereo, frames whose channels are alike are coded in MS
/// stereo (Annex G.2). The same signal in both channels then needs bits for
/// the middle channel only, and decodes better than in left/right stereo at
/// a low bitrate. A signal in one channel only stays in left/right stereo.
///
/// A panned signal is coded in MS stereo too, and keeps the level
/// difference between its channels. (With the side channel sparsed as
/// Annex G.2 b) describes, the channels of tones at full and half level
/// decode at about 13 and 8 dB.)
///
/// This is the evidence for leaving the sparsing out, which the side
/// channel's limitation marks as settled by decoding with ffmpeg.
#[test]
fn alike_channels_use_ms_stereo() {
    let fs = SamplingFrequency::Hz44100;
    let joint = Mode::JointStereo {
        intensity_stereo: false,
        ms_stereo: true,
    };
    let both = tones(44_100.0, 1.0);
    let panned: Vec<f64> = both.iter().map(|x| 0.5 * x).collect();
    let silent = vec![0.0; both.len()];

    let stream = encode_pcm(&config(64, fs, joint), &[&both, &both]);
    let ms = ms_stereo_frames(&stream);
    assert!(
        ms.iter().filter(|&&on| on).count() > ms.len() * 9 / 10,
        "{ms:?}"
    );
    let decoded = ffmpeg::decode_mp3(&stream);
    assert_eq!(decoded.diagnostics, "");
    let joint_snr = [0, 1].map(|channel| snr(&both, &decoded.samples, 2, channel));
    let stream = encode_pcm(&config(64, fs, Mode::Stereo), &[&both, &both]);
    let decoded = ffmpeg::decode_mp3(&stream);
    let stereo_snr = [0, 1].map(|channel| snr(&both, &decoded.samples, 2, channel));
    for channel in 0..2 {
        assert!(
            joint_snr[channel] > stereo_snr[channel] + 3.0,
            "channel {channel}: {:.1} dB in MS against {:.1} dB in left/right",
            joint_snr[channel],
            stereo_snr[channel]
        );
    }

    let stream = encode_pcm(&config(64, fs, joint), &[&both, &panned]);
    let ms = ms_stereo_frames(&stream);
    assert!(
        ms.iter().filter(|&&on| on).count() > ms.len() * 9 / 10,
        "{ms:?}"
    );
    let decoded = ffmpeg::decode_mp3(&stream);
    assert_eq!(decoded.diagnostics, "");
    for (channel, input) in [&both, &panned].into_iter().enumerate() {
        let snr = snr(input, &decoded.samples, 2, channel);
        assert!(snr > 25.0, "panned, channel {channel}: {snr:.1} dB");
    }

    for (label, channels) in [
        ("left only", [&both[..], &silent[..]]),
        ("right only", [&silent[..], &both[..]]),
    ] {
        let stream = encode_pcm(&config(64, fs, joint), &channels);
        assert!(ms_stereo_frames(&stream).iter().all(|&on| !on), "{label}");
    }
}

/// MS stereo works the same at the lower sampling frequencies.
#[test]
fn alike_channels_use_ms_stereo_at_the_lower_sampling_frequencies() {
    let fs = SamplingFrequency::Hz22050;
    let joint = Mode::JointStereo {
        intensity_stereo: false,
        ms_stereo: true,
    };
    let both = tones(22_050.0, 1.0);
    let stream = encode_pcm(&config(32, fs, joint), &[&both, &both]);
    let ms = ms_stereo_frames(&stream);
    assert!(
        ms.iter().filter(|&&on| on).count() > ms.len() * 9 / 10,
        "{ms:?}"
    );
    let decoded = ffmpeg::decode_mp3(&stream);
    assert_eq!(decoded.diagnostics, "");
    for channel in 0..2 {
        let snr = snr(&both, &decoded.samples, 2, channel);
        assert!(snr > 20.0, "channel {channel}: {snr:.1} dB");
    }
}

/// The amplitude of the sine at `frequency` in the decoded `channel` of a
/// stereo stream at 44.1 kHz, over its steady middle part.
fn amplitude(samples: &[f32], channel: usize, frequency: f64) -> f64 {
    let range = 4608..samples.len() / 2 - 4608;
    let n = range.len() as f64;
    let (cos, sin) = range.fold((0.0, 0.0), |(c, s), t| {
        let y = f64::from(samples[2 * t + channel]);
        let phase = TAU * frequency * t as f64 / 44_100.0;
        (c + y * phase.cos(), s + y * phase.sin())
    });
    2.0 * cos.hypot(sin) / n
}

/// With intensity stereo allowed, frames short of bits code their upper
/// bands in intensity stereo (Annex G.2 c)), alone or with MS stereo below
/// them. A low tone the same in both channels, and tones at 12 and 18 kHz
/// with the right channel at half the left, at 64 kbit/s: the low tone
/// stays the same in both channels. The upper tones decode at the levels
/// that the nearest intensity stereo position gives: 2 to 1 in amplitude
/// lies at 63.4 degrees, nearest position 4 (60 degrees), which splits the
/// sum of the channels, 1.5 times the left, as `tan(60) / (1 + tan(60))`
/// and `1 / (1 + tan(60))`. The tone at 12 kHz lies in the highest
/// scalefactor band but one, above every bound, and the tone at 18 kHz
/// above the highest band.
///
/// With MS stereo on as well, the side channel is zero below the bound, so
/// that every band there is marked as not coded in intensity stereo, and
/// the low tone is coded in the middle channel alone.
///
/// Without intensity stereo allowed, no frame is coded in it.
///
/// This is the evidence for decoding the marked bands of the side channel
/// in MS stereo, and for the position of the lines above the highest band,
/// which the intensity stereo positions mark as settled by decoding with
/// ffmpeg.
#[test]
fn intensity_stereo_keeps_the_level_of_each_channel() {
    let fs = SamplingFrequency::Hz44100;
    let tones = |levels: [(f64, f64); 3]| -> Vec<f64> {
        (0..2 * 44_100)
            .map(|t| {
                (levels.iter())
                    .map(|(f, a)| a * (TAU * f * t as f64 / 44_100.0).sin())
                    .sum()
            })
            .collect()
    };
    let left = tones([(440.0, 0.2), (12_000.0, 0.2), (18_000.0, 0.1)]);
    let right = tones([(440.0, 0.2), (12_000.0, 0.1), (18_000.0, 0.05)]);
    let ratio = (60f64).to_radians().tan();
    let (left_share, right_share) = (ratio / (1.0 + ratio), 1.0 / (1.0 + ratio));

    for ms_stereo in [false, true] {
        let mode = Mode::JointStereo {
            intensity_stereo: true,
            ms_stereo,
        };
        let stream = encode_pcm(&config(64, fs, mode), &[&left, &right]);
        let intensity = intensity_stereo_frames(&stream);
        assert!(
            intensity.iter().filter(|&&on| on).count() > intensity.len() * 9 / 10,
            "MS {ms_stereo}: {intensity:?}"
        );
        let decoded = ffmpeg::decode_mp3(&stream);
        assert_eq!(decoded.diagnostics, "", "MS {ms_stereo}");
        let expected = [
            (440.0, 0.2, 0.2),
            (12_000.0, 0.3 * left_share, 0.3 * right_share),
            (18_000.0, 0.15 * left_share, 0.15 * right_share),
        ];
        for (frequency, left_level, right_level) in expected {
            for (channel, level) in [(0, left_level), (1, right_level)] {
                let decoded = amplitude(&decoded.samples, channel, frequency);
                assert!(
                    (decoded / level - 1.0).abs() < 0.1,
                    "MS {ms_stereo}, {frequency} Hz, channel {channel}: \
                     {decoded:.4} for {level:.4}"
                );
            }
        }
    }

    let joint = Mode::JointStereo {
        intensity_stereo: false,
        ms_stereo: true,
    };
    let stream = encode_pcm(&config(64, fs, joint), &[&left, &right]);
    assert!(intensity_stereo_frames(&stream).iter().all(|&on| !on));
}
