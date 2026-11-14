// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! PCM encoded at a fixed bitrate by the inner iteration loop, decoded by
//! ffmpeg.

use std::f64::consts::TAU;

use chelys::mp3::header::{Bitrate, Mode, SamplingFrequency};
use chelys::mp3::{GRANULE_LINES, StreamConfig, encode_pcm, encode_silence};
use chelys_conformance::{DECODERS, Decoder, ffmpeg};

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
/// every granule must still squeeze into its share of the frame. That is 32
/// kbit/s in ISO/IEC 11172-3 and 8 kbit/s in ISO/IEC 13818-3, where a stereo
/// frame at 24 kHz with an error check leaves 1 byte for the main data.
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
    for fs in SamplingFrequency::STANDARD {
        for crc_protected in [false, true] {
            let config = StreamConfig {
                bitrate: Bitrate::all(fs.version()).next().unwrap(),
                sampling_frequency: fs,
                mode: Mode::Stereo,
                crc_protected,
            };
            let stream = encode_pcm(&config, &[&left, &right]);
            let decoded = ffmpeg::decode_mp3(&stream);
            let case = format!("{fs:?}, error check {crc_protected}");
            assert_eq!(decoded.diagnostics, "", "{case}");
            for channel in 0..2 {
                assert!(
                    decoded.samples[channel..]
                        .iter()
                        .step_by(2)
                        .any(|&s| s != 0.0),
                    "{case}: channel {channel} decodes to silence"
                );
            }
        }
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

/// The side information of each frame of a stream, found from the headers
/// (clauses 2.4.2.3 and 2.4.2.7), with whether the frame is MPEG-1.
fn side_infos(stream: &[u8]) -> Vec<(bool, &[u8])> {
    let mut side_infos = Vec::new();
    let mut at = 0;
    while at + 6 <= stream.len() {
        let header = &stream[at..];
        assert_eq!(header[0], 0xFF, "no syncword at {at}");
        let mpeg1 = header[1] >> 3 & 1 == 1;
        let crc = if header[1] & 1 == 0 { 2 } else { 0 };
        side_infos.push((mpeg1, &header[4 + crc..]));
        at += frame_len(header);
    }
    side_infos
}

/// The `main_data_begin` of each frame: 9 bits in MPEG-1, 8 at the lower
/// sampling frequencies (clause 2.4.1.7, and clause 2.4.1.7 of ISO/IEC
/// 13818-3).
fn main_data_begins(stream: &[u8]) -> Vec<usize> {
    side_infos(stream)
        .iter()
        .map(|&(mpeg1, side_info)| {
            if mpeg1 {
                (usize::from(side_info[0]) << 1) | usize::from(side_info[1] >> 7)
            } else {
                usize::from(side_info[0])
            }
        })
        .collect()
}

/// The four `scfsi` bits of each frame of a single-channel stream, after the
/// 9 bits of `main_data_begin` and 5 of `private_bits` (clause 2.4.1.7).
fn single_channel_scfsi(stream: &[u8]) -> Vec<u8> {
    side_infos(stream)
        .iter()
        .map(|&(mpeg1, side_info)| {
            assert!(mpeg1, "scfsi is sent in MPEG-1 only");
            ((side_info[1] & 0b11) << 2) | (side_info[2] >> 6)
        })
        .collect()
}

/// Quiet passages leave bits for the loud passages after them: main data
/// then starts more than a whole slot before its frame, and the decoder must
/// still find it. At 44.1 kHz and 64 kbit/s, where a frame has 208 bytes,
/// 187 of them main data, and at 22.05 kHz and 32 kbit/s, where it has 104
/// or 105 bytes, 95 or 96 of them main data, and `main_data_begin` has 8
/// bits (clause 2.4.1.7 of ISO/IEC 13818-3).
#[test]
fn main_data_spans_frames_after_quiet_passages() {
    for (sampling_frequency, kbps, main_data) in [
        (SamplingFrequency::Hz44100, 64, 187),
        (SamplingFrequency::Hz22050, 32, 96),
    ] {
        let fs = f64::from(sampling_frequency.hz());
        let half = sampling_frequency.hz() as usize / 2;
        let mut state = 0x9E37_79B9u32;
        let mut noise = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            f64::from(state) / f64::from(u32::MAX) * 2.0 - 1.0
        };
        // Half a second of a quiet tone, then half a second of loud noise,
        // twice.
        let input: Vec<f64> = (0..4 * half)
            .map(|t| {
                if (t / half) % 2 == 0 {
                    0.01 * (TAU * 440.0 * t as f64 / fs).sin()
                } else {
                    0.5 * noise()
                }
            })
            .collect();
        let config = config(kbps, sampling_frequency, Mode::SingleChannel);
        let stream = encode_pcm(&config, &[&input]);
        let begins = main_data_begins(&stream);
        let deepest = begins.iter().copied().max().unwrap();
        assert!(
            deepest > main_data,
            "{sampling_frequency:?}: main_data_begin never exceeds a slot: {begins:?}"
        );
        for Decoder { name, decode } in DECODERS {
            let decoded = decode(&stream);
            assert_eq!(decoded.diagnostics, "", "{name}, {sampling_frequency:?}");
            // Main data found in the wrong place would ruin the quiet
            // passages, which otherwise decode almost exactly. Each passage
            // is measured away from its edges, where the loud noise's
            // quantisation noise spreads. The output is 1057 samples behind
            // the input, less what the decoder trims from its start, as LAME
            // does; the delay is the one at which the first passage decodes
            // best, which no delay makes good if its main data is misplaced.
            let passages = [
                (0, 2304..half - 2304),
                (2, 2 * half + 2304..3 * half - 2304),
            ];
            let snr = |range: &std::ops::Range<usize>, delay: usize| {
                let (signal, noise) = range.clone().fold((0.0, 0.0), |(s, n), t| {
                    let x = input[t];
                    let y = f64::from(decoded.samples[t + delay]);
                    (s + x * x, n + (x - y) * (x - y))
                });
                10.0 * (signal / noise).log10()
            };
            let delay = (0..=1057)
                .max_by(|&a, &b| snr(&passages[0].1, a).total_cmp(&snr(&passages[0].1, b)))
                .unwrap();
            for (passage, range) in passages {
                let snr = snr(&range, delay);
                assert!(
                    snr > 40.0,
                    "{name}, {sampling_frequency:?}, quiet passage {passage}: {snr:.1} dB"
                );
            }
        }
    }
}

/// Short blocks keep the quantisation noise of an attack from spreading
/// back over the long window before it. A quiet tone is interrupted by
/// bursts of noise 30 dB louder, starting early and late in a granule. With
/// long blocks only, the noise rises to about -60 dBFS from 1024 samples
/// before each burst; with window switching it stays far below that until
/// the last few hundred samples, which the short windows and the polyphase
/// filter span. At 44.1 kHz and 128 kbit/s, and at 22.05 kHz and 64 kbit/s,
/// where Tables D.3 and D.4 of ISO/IEC 13818-3 drive the model.
#[test]
fn attacks_switch_to_short_blocks_and_shorten_pre_echo() {
    const DELAY: usize = 1057;
    for (sampling_frequency, kbps) in [
        (SamplingFrequency::Hz44100, 128),
        (SamplingFrequency::Hz22050, 64),
    ] {
        let fs = f64::from(sampling_frequency.hz());
        let half = sampling_frequency.hz() as usize / 2;
        let attacks = [half + 162 - half % 576, 2 * half + 461 - 2 * half % 576];
        let mut seed = 1u64;
        let input: Vec<f64> = (0..3 * half)
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
            &config(kbps, sampling_frequency, Mode::SingleChannel),
            &[&input],
        );
        let decoded = ffmpeg::decode_mp3(&stream);
        assert_eq!(decoded.diagnostics, "", "{sampling_frequency:?}");
        for a in attacks {
            for start in (a - 1024..a - 512).step_by(32) {
                let noise = (start..start + 32)
                    .map(|t| (f64::from(decoded.samples[t + DELAY]) - input[t]).powi(2))
                    .sum::<f64>()
                    / 32.0;
                let db = 10.0 * noise.log10();
                assert!(
                    db < -80.0,
                    "{sampling_frequency:?}, attack at {a}: {db:.1} dBFS of noise {} samples before it",
                    a - start
                );
            }
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
/// stereo stream at `fs` Hz, over its steady middle part.
fn amplitude(samples: &[f32], fs: f64, channel: usize, frequency: f64) -> f64 {
    let range = 4608..samples.len() / 2 - 4608;
    let n = range.len() as f64;
    let (cos, sin) = range.fold((0.0, 0.0), |(c, s), t| {
        let y = f64::from(samples[2 * t + channel]);
        let phase = TAU * frequency * t as f64 / fs;
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
                let decoded = amplitude(&decoded.samples, 44_100.0, channel, frequency);
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

/// Intensity stereo at the lower sampling frequencies (clause 2.4.3.2 of
/// ISO/IEC 13818-3), at 22.05 kHz and 48 kbit/s, with a low tone the same
/// in both channels and tones at 1.4, 2.9, 4.2 and 6 kHz with the right
/// channel at half the left: each lies in the band just below a bound, so
/// that the right channel, or in MS stereo the side channel, is not zero
/// there and needs no band marked as not coded in intensity stereo. Tones
/// with the right channel at half the left take position 4, two steps of
/// `i0 = 1/sqrt(2)`, which the decoder gives the left channel the lines
/// sent and the right half of them: so every tone decodes at its own level
/// in each channel. First, tones at 8 and 10.5 kHz at half level in the
/// right channel: the tone at 8 kHz lies in the highest scalefactor band
/// but one, above every bound, and the tone at 10.5 kHz above the highest
/// band. Then tones at 7, 8 and 10.5 kHz the same in both channels, which
/// take position 0 from band 18 up, where the fourth partition of
/// `int_scalefac_compress` from 180 sends it in no bits.
///
/// Last, in MS stereo, the tones below 7 kHz the same in both channels:
/// the side channel is all but zero below every bound, and would need its
/// bands there marked, which the encoder never sends at these sampling
/// frequencies, but for the highest bound: there the right channel, with
/// the bits it keeps and nothing else to spend them on, codes what the
/// filterbank leaves of the tones above it in the band below it, more than
/// a hundred dB down. So the frames are coded in intensity stereo, most of
/// them from band 18, and every tone keeps its level.
///
/// The same at 16 kHz and 40 kbit/s and at 24 kHz and 48 kbit/s, with the
/// tones in the same scalefactor bands (Table B.2 of ISO/IEC 13818-3), and
/// away from the edges of the polyphase subbands, but for the last case:
/// there the filterbank leaves nothing in the band below the highest bound,
/// which would need marking, and nearly every frame is coded in MS stereo
/// alone. (At 16 kHz and 32 kbit/s the right channel's low tone, below
/// every bound, loses up to a fifth of its level to the bitrate; in
/// left/right stereo the tones above lose up to a third.)
///
/// Each stream is decoded with ffmpeg, mpg123 and LAME. This is the evidence
/// for the position of the lines above the highest band and the scaling of
/// the sum at the lower sampling frequencies, and, with
/// `marks_are_read_as_illegal_in_different_fields`, for sending no marks
/// there, which the intensity stereo positions and `quantize_frame` mark as
/// settled by decoding with ffmpeg, mpg123 and LAME.
#[test]
fn intensity_stereo_keeps_the_level_of_each_channel_at_the_lower_sampling_frequencies() {
    // The tones in bands 9, 13, 15 and 17, below the bounds, bands 10, 14,
    // 16 and 18; in bands 18 and 19; and above band 20, the highest.
    let cases = [
        (
            SamplingFrequency::Hz22050,
            48,
            [1_400.0, 2_900.0, 4_200.0, 6_000.0],
            [7_000.0, 8_000.0, 10_500.0],
        ),
        (
            SamplingFrequency::Hz16000,
            40,
            [1_050.0, 2_150.0, 3_050.0, 4_300.0],
            [5_100.0, 6_100.0, 7_600.0],
        ),
        (
            SamplingFrequency::Hz24000,
            48,
            [1_550.0, 3_100.0, 4_400.0, 6_300.0],
            [7_600.0, 9_100.0, 11_450.0],
        ),
    ];
    for (fs, kbps, below_bounds, [band_18, band_19, above]) in cases {
        let hz = f64::from(fs.hz());
        let tones = |levels: &[(f64, f64, f64)], channel: usize| -> Vec<f64> {
            (0..2 * fs.hz() as usize)
                .map(|t| {
                    (levels.iter())
                        .map(|&(f, l, r)| [l, r][channel] * (TAU * f * t as f64 / hz).sin())
                        .sum()
                })
                .collect()
        };
        let below = |right: f64| {
            [(440.0, 0.2, 0.2)]
                .into_iter()
                .chain(below_bounds.map(|f| (f, 0.1, right)))
                .collect::<Vec<_>>()
        };
        let panned_top = [
            below(0.05).as_slice(),
            &[(band_19, 0.2, 0.1), (above, 0.1, 0.05)],
        ]
        .concat();
        let alike_top = [
            below(0.05).as_slice(),
            &[
                (band_18, 0.05, 0.05),
                (band_19, 0.05, 0.05),
                (above, 0.05, 0.05),
            ],
        ]
        .concat();
        let alike_below = [
            below(0.1).as_slice(),
            &[(band_19, 0.2, 0.1), (above, 0.1, 0.05)],
        ]
        .concat();
        let mut cases = vec![
            (&panned_top, false),
            (&panned_top, true),
            (&alike_top, false),
            (&alike_top, true),
        ];
        if fs == SamplingFrequency::Hz22050 {
            cases.push((&alike_below, true));
        }
        for (levels, ms_stereo) in cases {
            let (left, right) = (tones(levels, 0), tones(levels, 1));
            let mode = Mode::JointStereo {
                intensity_stereo: true,
                ms_stereo,
            };
            let stream = encode_pcm(&config(kbps, fs, mode), &[&left, &right]);
            let intensity = intensity_stereo_frames(&stream);
            let count = intensity.iter().filter(|&&on| on).count();
            assert!(
                count > intensity.len() * 9 / 10,
                "{fs:?}, MS {ms_stereo}: {intensity:?}"
            );
            for Decoder { name, decode } in DECODERS {
                let decoded = decode(&stream);
                assert_eq!(decoded.diagnostics, "", "{name}, {fs:?}, MS {ms_stereo}");
                for &(frequency, left_level, right_level) in levels {
                    for (channel, level) in [(0, left_level), (1, right_level)] {
                        let decoded = amplitude(&decoded.samples, hz, channel, frequency);
                        assert!(
                            (decoded / level - 1.0).abs() < 0.1,
                            "{name}, {fs:?}, MS {ms_stereo}, {frequency} Hz, channel {channel}: \
                             {decoded:.4} for {level:.4}"
                        );
                    }
                }
            }
        }
    }
}

/// Reads `count` bits from bit `at` of `data`, most significant first.
fn bits(data: &[u8], at: usize, count: usize) -> u32 {
    (at..at + count).fold(0, |value, bit| {
        value << 1 | u32::from(data[bit / 8] >> (7 - bit % 8) & 1)
    })
}

/// For each granule of a stereo stream, whether its frame is coded in
/// intensity stereo and both its channels in short blocks: with
/// `window_switching_flag` set and `block_type` '10' (clause 2.4.1.7, and
/// clause 2.4.2.1 of ISO/IEC 13818-3 for the lower sampling frequencies).
fn short_intensity_granules(stream: &[u8]) -> Vec<bool> {
    let mut granules = Vec::new();
    let mut at = 0;
    while at + 4 <= stream.len() {
        let header = &stream[at..];
        assert_eq!(header[0], 0xFF, "no syncword at {at}");
        let mpeg1 = header[1] >> 3 & 1 == 1;
        let crc = if header[1] & 1 == 0 { 2 } else { 0 };
        let side_info = &header[4 + crc..];
        let intensity = header[3] >> 4 & 1 == 1;
        // The bits before the first granule's (main_data_begin, the
        // private bits and, in MPEG-1, scfsi), the bits of each granule of
        // each channel, and those before its window_switching_flag
        // (part2_3_length, big_values, global_gain and scalefac_compress).
        let (start, per_channel, before, count) = if mpeg1 {
            (9 + 3 + 8, 59, 12 + 9 + 8 + 4, 2)
        } else {
            (8 + 2, 63, 12 + 9 + 8 + 9, 1)
        };
        for gr in 0..count {
            let short = (0..2).all(|ch| {
                let flag = start + (2 * gr + ch) * per_channel + before;
                bits(side_info, flag, 3) == 0b110
            });
            granules.push(intensity && short);
        }
        at += frame_len(header);
    }
    granules
}

/// The third difference of `x` at `t`, which passes a sine of frequency `f`
/// at `(2 sin(pi f / fs))^3` of its amplitude: less than a tenth up to
/// 1.6 kHz at 22.05 kHz and 3.2 kHz at 44.1 kHz, and more than its whole
/// amplitude from `fs / 6` up.
fn high_pass(x: impl Fn(usize) -> f64, t: usize) -> f64 {
    x(t + 3) - 3.0 * x(t + 2) + 3.0 * x(t + 1) - x(t)
}

/// Intensity stereo in short blocks. Bursts of noise every 9600 samples,
/// decaying with a time constant of 30 ms, switch both channels to short
/// blocks, over tones at 1.2, 2.8, 3.7 and 5 kHz, with the right channel at
/// half the left throughout.
///
/// At 44.1 kHz and 64 kbit/s, with MS stereo on or off, the frames are
/// short of bits, and those with short blocks are coded in intensity stereo.
/// Amplitudes 2 to 1 take position 4 in every band and window, which
/// decodes the right channel at `1 / tan(60)`, 0.577, of the left.
///
/// At 22.05 kHz and 48 kbit/s the tones lie in the bands just below the
/// bounds of short blocks, bands 4, 7, 8 and 9 (Table B.2 of ISO/IEC
/// 13818-3), so that the right channel need not mark a band in any window,
/// and some short blocks are coded in intensity stereo. Then tones at 1, 2,
/// 3 and 4.5 kHz, in bands 3, 6, 7 and 9: band 4, below the lowest bound,
/// is zero in the right channel, which would need it marked, so the short
/// blocks coded in intensity stereo are coded from a higher bound.
/// Amplitudes 2 to 1 take position 4, which decodes the right channel at
/// `i0^2`, 0.5, of the left.
///
/// In the short granules coded in intensity stereo, the decoded right
/// channel, high-passed, is that share of the decoded left in ffmpeg,
/// mpg123 and LAME, within 2%.
#[test]
fn intensity_stereo_keeps_the_level_of_each_channel_in_short_blocks() {
    let below_bounds = [1_200.0, 2_800.0, 3_700.0, 5_000.0];
    let cases = [
        (
            SamplingFrequency::Hz44100,
            64,
            false,
            below_bounds,
            1.0 / 60f64.to_radians().tan(),
        ),
        (
            SamplingFrequency::Hz44100,
            64,
            true,
            below_bounds,
            1.0 / 60f64.to_radians().tan(),
        ),
        (SamplingFrequency::Hz22050, 48, false, below_bounds, 0.5),
        (
            SamplingFrequency::Hz22050,
            48,
            false,
            [1_000.0, 2_000.0, 3_000.0, 4_500.0],
            0.5,
        ),
    ];
    for (fs, kbps, ms_stereo, frequencies, share) in cases {
        let hz = f64::from(fs.hz());
        let mut seed = 1u64;
        let left: Vec<f64> = (0..(2.0 * hz) as usize)
            .map(|t| {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let noise = (seed >> 11) as f64 / (1u64 << 52) as f64 - 1.0;
                let burst = 0.6 * (-((t % 9600) as f64) / (0.03 * hz)).exp() * noise;
                let tones: f64 = frequencies
                    .iter()
                    .map(|f| 0.1 * (TAU * f * t as f64 / hz).sin())
                    .sum();
                burst + tones
            })
            .collect();
        let right: Vec<f64> = left.iter().map(|x| 0.5 * x).collect();
        let mode = Mode::JointStereo {
            intensity_stereo: true,
            ms_stereo,
        };
        let stream = encode_pcm(&config(kbps, fs, mode), &[&left, &right]);
        let granules = short_intensity_granules(&stream);
        let coded = granules.iter().filter(|&&on| on).count();
        assert!(
            coded >= 4,
            "{fs:?}, MS {ms_stereo}, {frequencies:?}: {granules:?}"
        );
        // The input samples of each short granule coded in intensity
        // stereo, but for the last three.
        let samples: Vec<usize> = (granules.iter().enumerate())
            .filter(|&(_, &on)| on)
            .flat_map(|(granule, _)| granule * GRANULE_LINES..(granule + 1) * GRANULE_LINES)
            .filter(|&t| t + 3 < left.len())
            .collect();
        for Decoder { name, decode } in DECODERS {
            let decoded = decode(&stream);
            assert_eq!(decoded.diagnostics, "", "{name}");
            // The decoded `channel`, high-passed, projected on the left
            // input, high-passed, with the output `delay` samples behind.
            let gain = |channel: usize, delay: usize, samples: &[usize]| {
                let output = |t: usize| {
                    let at = 2 * (t + delay) + channel;
                    decoded.samples.get(at).copied().map_or(0.0, f64::from)
                };
                let (product, energy) = samples.iter().fold((0.0, 0.0), |(p, e), &t| {
                    let input = high_pass(|t| left[t], t);
                    (p + high_pass(output, t) * input, e + input * input)
                });
                product / energy
            };
            let delay = (0..2 * GRANULE_LINES)
                .max_by(|&a, &b| gain(0, a, &samples).total_cmp(&gain(0, b, &samples)))
                .unwrap();
            let decoded_share = gain(1, delay, &samples) / gain(0, delay, &samples);
            assert!(
                (decoded_share / share - 1.0).abs() < 0.02,
                "{name}, {fs:?}, MS {ms_stereo}, {frequencies:?}: {decoded_share:.4} for {share:.4}"
            );
        }
    }
}
