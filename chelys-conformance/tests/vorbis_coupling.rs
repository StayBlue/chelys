// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Stereo streams, whose modes couple their two channels as a magnitude and
//! an angle (clause 4.3.5), decoded by ffmpeg.

use std::f64::consts::TAU;

use chelys::vorbis::internals::setup::Setup;
use chelys::vorbis::{Config, Encoder, Headers, Packet, Rate};
use chelys_conformance::ffmpeg;
use chelys_conformance::ogg::{self, AudioPacket};

/// The rate of every stream: the psychoacoustic model's allowance as it is.
const RATE: Rate = Rate::Variable {
    noise_offset_db: 0.0,
};

const SAMPLE_RATE: u32 = 44_100;

/// The headers and audio packets of `pcm`, interleaved, of `channels`.
fn encode(pcm: &[f32], channels: usize) -> (Headers, Vec<AudioPacket>) {
    let mut encoder = Encoder::new(Config::new(SAMPLE_RATE, channels, RATE)).unwrap();
    let headers = encoder.headers();
    let mut packets = Vec::new();
    let mut out = |packet: Packet<'_>| {
        packets.push(AudioPacket {
            data: packet.data.to_vec(),
            granule_position: packet.granule_position,
        })
    };
    encoder.encode_interleaved(pcm, &mut out).unwrap();
    encoder.finish(&mut out);
    (headers, packets)
}

/// The bytes of `packets`.
fn bytes(packets: &[AudioPacket]) -> usize {
    packets.iter().map(|packet| packet.data.len()).sum()
}

/// `left` and `right` encoded as a stereo stream, which couples them, and
/// decoded by ffmpeg, which is to report nothing and to give back as many
/// samples: each channel's samples, and the stream's packets.
fn round_trip(left: &[f64], right: &[f64]) -> ([Vec<f32>; 2], Vec<AudioPacket>) {
    assert_eq!(left.len(), right.len());
    let pcm: Vec<f32> = (left.iter().zip(right))
        .flat_map(|(&l, &r)| [l as f32, r as f32])
        .collect();
    let (headers, packets) = encode(&pcm, 2);
    // The stream's setup couples its channels, and its residues interleave
    // them (residue type 2).
    let setup = Setup::new(2);
    assert_eq!(headers.setup, setup.header());
    assert!(setup.coupled);
    assert!(
        setup
            .residues
            .iter()
            .all(|residue| residue.residue_type() == 2)
    );
    let decoded = ffmpeg::decode_ogg(&ogg::vorbis(&headers, &packets, 4));
    assert_eq!(decoded.diagnostics, "");
    assert_eq!(decoded.samples.len(), pcm.len());
    let channel = |c: usize| decoded.samples.iter().skip(c).step_by(2).copied().collect();
    ([channel(0), channel(1)], packets)
}

/// The ratio in dB of the power of `reference` to that of what `decoded`
/// differs from it by, over the samples in `range`.
fn snr(reference: &[f64], decoded: &[f32], range: std::ops::Range<usize>) -> f64 {
    let (mut signal, mut noise) = (0.0, 0.0);
    for i in range {
        signal += reference[i].powi(2);
        noise += (reference[i] - f64::from(decoded[i])).powi(2);
    }
    10.0 * (signal / noise).log10()
}

/// The lag, within 16 samples either way, at which `decoded` correlates
/// best with `reference` over the 2048 samples from `from`.
fn lag(reference: &[f64], decoded: &[f32], from: usize) -> isize {
    (-16..=16)
        .max_by(|&a: &isize, &b: &isize| {
            let correlation = |lag: isize| -> f64 {
                (from..from + 2048)
                    .map(|i| reference[i] * f64::from(decoded[i.wrapping_add_signed(lag)]))
                    .sum()
            };
            correlation(a).total_cmp(&correlation(b))
        })
        .unwrap()
}

/// Uniform noise from -0.5 to 0.5, the same each run for each `seed`.
fn noise(len: usize, seed: u32) -> Vec<f64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f64::from(state >> 8) / f64::from(1 << 24) - 0.5
        })
        .collect()
}

/// Music-like: a chord of tones and low-passed noise, with a burst of
/// noise that sets in at each of `onsets` and dies away in about 5 ms,
/// attacks that short blocks code.
fn music(len: usize, seed: u32, onsets: &[usize]) -> Vec<f64> {
    let noise = noise(len, seed);
    let mut low = 0.0;
    let mut pcm: Vec<f64> = (0..len)
        .map(|t| {
            low = 0.9 * low + 0.1 * noise[t];
            let t = t as f64 / f64::from(SAMPLE_RATE);
            let phase = f64::from(seed);
            0.2 * (TAU * 220.0 * t + phase).sin()
                + 0.1 * (TAU * 1320.0 * t).sin()
                + 0.05 * (TAU * 5000.0 * t + phase).sin()
                + 0.4 * low
        })
        .collect();
    let decay = 0.005 * f64::from(SAMPLE_RATE);
    for &onset in onsets.iter().filter(|&&onset| onset < len) {
        for (k, x) in (pcm[onset..].iter_mut().enumerate()).take(10 * decay as usize) {
            *x += 2.0 * noise[(onset + 7 * k) % len] * (-(k as f64) / decay).exp();
        }
    }
    pcm
}

/// Whether an audio packet is of a short block: its mode's first bit, the
/// bit after `packet_type`, is its blockflag in chelys's streams.
fn is_short(packet: &AudioPacket) -> bool {
    (packet.data[0] >> 1) & 1 == 0
}

const LEN: usize = 3 * SAMPLE_RATE as usize;

/// Each channel decodes in place and within the noise that noise masks, 6
/// dB below it, over `range`, where its input is not silent.
fn assert_close(context: &str, input: [&[f64]; 2], decoded: &[Vec<f32>; 2]) {
    for c in 0..2 {
        if input[c].iter().all(|&x| x == 0.0) {
            continue;
        }
        let snr = snr(input[c], &decoded[c], 0..LEN);
        assert!(snr > 6.0, "{context}: channel {c}, {snr:.1} dB");
        let loud = (4096..LEN - 4096)
            .step_by(4096)
            .max_by(|&a, &b| {
                let power = |from: usize| -> f64 {
                    input[c][from..from + 2048].iter().map(|x| x * x).sum()
                };
                power(a).total_cmp(&power(b))
            })
            .unwrap();
        assert_eq!(
            lag(input[c], &decoded[c], loud),
            0,
            "{context}: channel {c}"
        );
    }
}

/// A silent channel beside a coded one, whose floor is unused, decodes to
/// silence, every sample of it, although the mode couples it with the
/// coded channel, whose residue it is then decoded with (clause 4.3.3).
#[test]
fn a_silent_channel_beside_a_coded_one_decodes_to_silence() {
    let sound = music(LEN, 1, &[30_000, 70_000]);
    let silence = vec![0.0; LEN];
    for (context, left, right) in [
        ("right silent", &sound, &silence),
        ("left silent", &silence, &sound),
    ] {
        let (decoded, _) = round_trip(left, right);
        assert_close(context, [left, right], &decoded);
        let silent = if right == &silence { 1 } else { 0 };
        assert!(decoded[silent].iter().all(|&x| x == 0.0), "{context}");
    }
}

/// A channel silent in some blocks only, and coded in others, decodes to
/// silence where it is silent, and close to its input elsewhere.
#[test]
fn a_channel_silent_in_some_blocks_decodes_to_silence_there() {
    let left = music(LEN, 2, &[20_000, 50_000, 90_000]);
    // The right channel sounds for 0.3 s in each 0.6 s, from the 2nd.
    let sounding = |t: usize| (t / 13_230) % 2 == 1;
    let other = music(LEN, 3, &[25_000, 41_000]);
    let right: Vec<f64> = (0..LEN)
        .map(|t| if sounding(t) { other[t] } else { 0.0 })
        .collect();
    for (context, left, right) in [("right", &left, &right), ("left", &right, &left)] {
        let (decoded, _) = round_trip(left, right);
        assert_close(context, [left, right], &decoded);
        let gated = if context == "right" { 1 } else { 0 };
        // Away from where it sets in and stops, by more than a long block,
        // the gated channel decodes to silence.
        let silent: Vec<usize> = (0..LEN)
            .filter(|&t| (t.saturating_sub(2048)..t + 2048).all(|u| !sounding(u)))
            .collect();
        assert!(silent.len() > LEN / 4);
        assert!(
            silent.iter().all(|&t| decoded[gated][t] == 0.0),
            "{context}"
        );
    }
}

/// Identical channels decode to identical channels: every angle is 0, and
/// both channels have the same floor. They take far fewer bytes than twice
/// the one channel alone, as each channel coded on its own would.
#[test]
fn identical_channels_decode_to_identical_channels() {
    let sound = music(LEN, 4, &[30_000, 31_000, 100_000]);
    let (decoded, packets) = round_trip(&sound, &sound);
    assert_close("identical", [&sound, &sound], &decoded);
    assert_eq!(decoded[0], decoded[1]);
    assert!(packets.iter().any(is_short));
    let mono: Vec<f32> = sound.iter().map(|&x| x as f32).collect();
    let (stereo, mono) = (bytes(&packets), bytes(&encode(&mono, 1).1));
    assert!(
        stereo * 10 < 2 * mono * 7,
        "{stereo} bytes against {mono} in mono"
    );
}

/// Channels of opposite phase, whose magnitudes are equal and whose angles
/// are twice them, decode close to their inputs, a loud square wave too,
/// whose angles are the largest the encoder meets.
#[test]
fn inverted_channels_decode_close_to_their_inputs() {
    let sound = music(LEN, 5, &[30_000, 100_000]);
    let inverted: Vec<f64> = sound.iter().map(|x| -x).collect();
    let (decoded, _) = round_trip(&sound, &inverted);
    assert_close("inverted", [&sound, &inverted], &decoded);
    let square: Vec<f64> = (0..LEN)
        .map(|t| if t / 50 % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let inverted: Vec<f64> = square.iter().map(|x| -x).collect();
    let (decoded, _) = round_trip(&square, &inverted);
    for (c, input) in [&square, &inverted].into_iter().enumerate() {
        let snr = snr(input, &decoded[c], 0..LEN);
        assert!(snr > 15.0, "square wave: channel {c}, {snr:.1} dB");
    }
}

/// Channels of different music, a channel far quieter than the other, and
/// attacks in one channel only, which short blocks code in both, decode
/// close to their inputs and in place.
#[test]
fn unlike_channels_decode_close_to_their_inputs() {
    let onsets = [20_000, 60_000, 90_061, 110_000];
    let left = music(LEN, 6, &onsets);
    let right = music(LEN, 7, &[]);
    let quiet: Vec<f64> = right.iter().map(|x| 0.003 * x).collect();
    for (context, left, right) in [
        ("unlike", &left, &right),
        ("quiet right", &left, &quiet),
        ("quiet left", &quiet, &left),
    ] {
        let (decoded, packets) = round_trip(left, right);
        let short = packets.iter().filter(|p| is_short(p)).count();
        assert!(short >= 8 * onsets.len(), "{context}: {short} short blocks");
        assert_close(context, [left, right], &decoded);
    }
}
