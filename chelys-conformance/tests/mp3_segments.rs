// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Streams with independent frames
//! (`Config::independent_frames_every`), cut into segments at them, as
//! ffmpeg, mpg123 and LAME decode each segment on its own.

use std::f64::consts::TAU;
use std::num::NonZeroU32;

use chelys::mp3::{ChannelMode, Config, Encoder, Rate};
use chelys_conformance::DECODERS;

/// A tone over noise that turns loud and quiet every 20000 samples, in
/// `channels` channels, interleaved.
fn input(len: usize, sample_rate: u32, channels: usize) -> Vec<f64> {
    let mut seed = 5u64;
    (0..len)
        .flat_map(|t| {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let noise = (seed >> 11) as f64 / (1u64 << 52) as f64 - 1.0;
            let s = t as f64 / f64::from(sample_rate);
            let level = if (t / 20_000) % 2 == 0 { 0.02 } else { 0.25 };
            let left = 0.1 * (TAU * 440.0 * s).sin() + level * noise;
            [left, 0.7 * left].into_iter().take(channels)
        })
        .collect()
}

fn frames(config: Config, pcm: &[f64]) -> Vec<Vec<u8>> {
    let mut encoder = Encoder::new(config).unwrap();
    let mut frames = Vec::new();
    let mut out = |frame: &[u8]| frames.push(frame.to_vec());
    encoder.encode_interleaved(pcm, &mut out).unwrap();
    encoder.finish(&mut out);
    frames
}

/// The sample of each channel, counted from the start of the decoder's
/// output of the segment from frame `j`, after which it decodes as the
/// whole stream of `frames` does, with which it ends; or `None` if it logs
/// anything for either, or does not give every frame's samples of both,
/// less at most [`DECODER_DELAY`] at the start of each, as many as it
/// trims from the whole stream.
fn decoded_alike_after(
    decode: fn(&[u8]) -> chelys_conformance::Decoded,
    frames: &[Vec<u8>],
    j: usize,
    samples_per_frame: usize,
    channels: usize,
) -> Option<usize> {
    let (whole, part) = (decode(&frames.concat()), decode(&frames[j..].concat()));
    if !whole.diagnostics.is_empty() || !part.diagnostics.is_empty() {
        return None;
    }
    let trimmed = (frames.len() * samples_per_frame * channels)
        .checked_sub(whole.samples.len())
        .filter(|&trimmed| trimmed <= DECODER_DELAY * channels)?;
    if part.samples.len() + trimmed != (frames.len() - j) * samples_per_frame * channels {
        return None;
    }
    let skipped = whole.samples.len() - part.samples.len();
    let last = (part.samples.iter().zip(&whole.samples[skipped..]))
        .rposition(|(a, b)| (a - b).abs() > TOLERANCE)
        .map_or(0, |i| i + 1);
    Some((trimmed + last).div_ceil(channels))
}

/// The samples of each channel by which a decoder delays its output, the
/// most that one trims from the start of its output (LAME does).
const DECODER_DELAY: usize = 529;

/// How far apart two outputs of the same samples may be. LAME writes 16-bit
/// PCM whose last bits vary with where decoding began, long after its
/// filterbank's warm-up, though it decodes each frame alike.
const TOLERANCE: f32 = 4.0 / 32768.0;

/// The samples of each channel that a segment decodes to differently from
/// the whole stream, as the decoder's filterbank starts empty: two
/// granules, one frame in MPEG-1 and two at the lower sampling frequencies.
const WARM_UP: usize = 2 * 576;

/// A segment that starts with a frame that begins its main data in itself
/// decodes as the whole stream does but for its first two granules, in
/// which the decoder's filterbank starts empty: each decoder gives every
/// frame of it, logs nothing, and after [`WARM_UP`] gives the samples it
/// gives in the whole stream. Cut at the same frames, a stream without the
/// option decodes differently further on, as the main data of a segment's
/// first frames begins in the segment before it, and no decoder logs it.
#[test]
fn segments_decode_alone_but_for_their_first_granules() {
    let cases = [
        (
            44_100,
            ChannelMode::Stereo,
            Rate::Constant { kbps: 64 },
            44_100,
        ),
        (
            48_000,
            ChannelMode::JointStereo {
                ms_stereo: true,
                intensity_stereo: false,
            },
            Rate::Quality {
                level: 60,
                min_kbps: None,
                max_kbps: None,
            },
            // 62.5 frames: the later of two frames as near.
            72_000,
        ),
        (
            22_050,
            ChannelMode::Mono,
            Rate::Constant { kbps: 24 },
            22_050,
        ),
    ];
    for (sample_rate, mode, rate, period) in cases {
        let case = format!("{sample_rate} Hz, {rate:?}");
        let channels = mode.channels();
        let pcm = input(6 * sample_rate as usize, sample_rate, channels);
        let config = Config::new(sample_rate, mode, rate);
        let independent = frames(
            config.independent_frames_every(NonZeroU32::new(period)),
            &pcm,
        );
        let plain = frames(config, &pcm);
        let samples_per_frame = Encoder::new(config).unwrap().samples_per_frame();
        let f = samples_per_frame as u64;
        let cuts: Vec<usize> = (1..)
            .map(|n: u64| ((2 * n * u64::from(period) + f) / (2 * f)) as usize)
            // ffmpeg finds no stream in fewer than two frames.
            .take_while(|&j| j + 8 <= independent.len())
            .collect();
        assert!(cuts.len() >= 3, "{case}");
        for decoder in DECODERS {
            let alike_after = |frames: &[Vec<u8>], j: usize| {
                decoded_alike_after(decoder.decode, frames, j, samples_per_frame, channels)
            };
            for &j in &cuts {
                let after = alike_after(&independent, j);
                assert!(
                    after.is_some_and(|after| after <= WARM_UP),
                    "{case}: {} from frame {j}: {after:?}",
                    decoder.name
                );
            }
            assert!(
                (cuts.iter()).any(|&j| alike_after(&plain, j).is_some_and(|after| after > WARM_UP)),
                "{case}: {} decodes the stream without the option alike",
                decoder.name
            );
        }
    }
}
