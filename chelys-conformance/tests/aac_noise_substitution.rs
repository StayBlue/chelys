// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Perceptual noise substitution (ISO/IEC 14496-3, clause 4.6.13) through
//! the AAC encoder, decoded by ffmpeg and FAAD2: the noise each decoder
//! generates in place of the bands substituted, against the input's.

use std::f64::consts::TAU;

use chelys::aac::internals::channel_configuration::ChannelConfiguration;
use chelys::aac::internals::noiseless::decoder::{BitReader, Element, element_of};
use chelys::aac::internals::rate::Control;
use chelys::aac::internals::sampling_frequency::SamplingFrequency;
use chelys::aac::internals::{StreamConfig, StreamEncoder};
use chelys::aac::{Config, Encoder, Framing, Rate};
use chelys_conformance::{Decoded, faad, ffmpeg};

const SAMPLE_RATE: u32 = 44_100;

/// `NOISE_HCB` (ISO/IEC 14496-3, Table 4.169), `INTENSITY_HCB2` and
/// `INTENSITY_HCB`.
const NOISE_HCB: u8 = 13;
const INTENSITY_HCB2: u8 = 14;
const INTENSITY_HCB: u8 = 15;

/// The lines of the transform of [`spectrum`].
const BLOCK: usize = 2048;

/// White noise, uniform within a quarter of full scale.
fn noise(len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = (state.wrapping_mul(6_364_136_223_846_793_005))
                .wrapping_add(1_442_695_040_888_963_407);
            0.25 * ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
        })
        .collect()
}

/// Encodes planar `input` at `bits_per_second` with perceptual noise
/// substitution, in ADTS.
fn encode(input: &[Vec<f64>], bits_per_second: u32) -> Vec<u8> {
    let config = Config::new(SAMPLE_RATE, input.len(), Rate::Constant { bits_per_second })
        .noise_substitution(true);
    let mut encoder = Encoder::new(config).unwrap();
    let mut stream = Vec::new();
    let mut out = |frame: &[u8]| stream.extend_from_slice(frame);
    let channels: Vec<&[f64]> = input.iter().map(Vec::as_slice).collect();
    encoder.encode_planar(&channels, &mut out);
    encoder.finish(&mut out);
    stream
}

/// The element of each frame of `stream`, read as MPEG-4 AAC with the
/// decoder that chelys's tests share.
fn elements(stream: &[u8]) -> Vec<Element> {
    let fs = SamplingFrequency::from_hz(SAMPLE_RATE).unwrap();
    let mut elements = Vec::new();
    let mut frame = stream;
    while frame.len() > 7 {
        assert_eq!((frame[1] >> 3) & 1, 0, "ID 0, MPEG-4");
        let len = (usize::from(frame[3] & 0b11) << 11)
            | (usize::from(frame[4]) << 3)
            | usize::from(frame[5] >> 5);
        elements.push(element_of(&mut BitReader::new(&frame[7..len]), fs, true));
        frame = &frame[len..];
    }
    elements
}

/// The share of the bands of channel `c` of `elements` from `first` up to
/// their `max_sfb`, in frames of long windows past the first two, that are
/// substituted.
fn substituted_share(elements: &[Element], c: usize, first: usize) -> f64 {
    let (mut noise, mut all) = (0, 0);
    for element in &elements[2..elements.len() - 2] {
        let ics = &element.channels[c];
        if ics.info.window_sequence != 0 {
            continue;
        }
        for sfb in first..ics.info.max_sfb {
            all += 1;
            noise += usize::from(ics.codebook(0, sfb) == NOISE_HCB);
        }
    }
    noise as f64 / all as f64
}

/// The Hann-windowed DFT of each block of `BLOCK` samples of `signal`,
/// half overlapped, from its 4th block to its 4th last, in the lines from
/// `low` to `high` Hz.
fn spectrum(signal: &[f64], low: f64, high: f64) -> Vec<Vec<(f64, f64)>> {
    let line_hz = f64::from(SAMPLE_RATE) / BLOCK as f64;
    let lines = (low / line_hz).ceil() as usize..(high / line_hz) as usize;
    let window: Vec<f64> = (0..BLOCK)
        .map(|n| 0.5 - 0.5 * (TAU * n as f64 / BLOCK as f64).cos())
        .collect();
    let starts: Vec<usize> = (0..)
        .map(|b| b * BLOCK / 2)
        .take_while(|&s| s + BLOCK <= signal.len())
        .collect();
    starts[3..starts.len() - 3]
        .iter()
        .map(|&start| {
            let block = &signal[start..start + BLOCK];
            (lines.clone())
                .map(|k| {
                    let w = TAU * k as f64 / BLOCK as f64;
                    block.iter().zip(&window).enumerate().fold(
                        (0.0, 0.0),
                        |(re, im), (n, (x, h))| {
                            let phase = w * n as f64;
                            (re + x * h * phase.cos(), im - x * h * phase.sin())
                        },
                    )
                })
                .collect()
        })
        .collect()
}

/// The power of `spectrum`, summed.
fn power(spectrum: &[Vec<(f64, f64)>]) -> f64 {
    spectrum
        .iter()
        .flatten()
        .map(|(re, im)| re * re + im * im)
        .sum()
}

/// The normalized correlation of two signals' `spectra`.
fn correlation(a: &[Vec<(f64, f64)>], b: &[Vec<(f64, f64)>]) -> f64 {
    let cross: f64 = (a.iter().flatten().zip(b.iter().flatten()))
        .map(|((ar, ai), (br, bi))| ar * br + ai * bi)
        .sum();
    cross / (power(a) * power(b)).sqrt()
}

/// Channel `c` of `decoded`, interleaved `channels`.
fn channel(decoded: &Decoded, channels: usize, c: usize) -> Vec<f64> {
    (decoded.samples.iter().skip(c).step_by(channels))
        .map(|&x| f64::from(x))
        .collect()
}

/// The bands above 4 kHz, the border, from which noise is substituted.
const FIRST_NOISE_BAND: usize = 24;

/// The ranges in Hz over which the decoded noise is measured: within the
/// bands substituted, and below the bandwidth of the input coded.
const RANGES: [(f64, f64); 3] = [(5000.0, 8000.0), (8000.0, 11_000.0), (11_000.0, 14_000.0)];

/// A decoder of streams in ADTS.
type Decoder = fn(&[u8]) -> Decoded;

/// The decoders, by name.
fn decoders() -> [(&'static str, Decoder); 2] {
    [("ffmpeg", ffmpeg::decode_aac), ("FAAD2", faad::decode_aac)]
}

/// White noise coded at 48 kbit/s in mono, its bands above the border
/// substituted, decodes in ffmpeg and FAAD2 at the level of the input in
/// each range above the border, within 1 dB: the energy of each window's
/// band is the one its noise energy gives, `2^(noise_nrg / 2)`.
#[test]
fn ffmpeg_and_faad_decode_noise_at_its_energy() {
    let input = noise(40 * 1024, 3);
    let stream = encode(std::slice::from_ref(&input), 48_000);
    let share = substituted_share(&elements(&stream), 0, FIRST_NOISE_BAND);
    assert!(share > 0.9, "{share:.2} of the bands substituted");
    for (name, decode) in decoders() {
        let decoded = decode(&stream);
        assert_eq!(decoded.diagnostics, "", "{name}");
        let output = channel(&decoded, 1, 0);
        for (low, high) in RANGES {
            let level = 10.0
                * (power(&spectrum(&output, low, high)) / power(&spectrum(&input, low, high)))
                    .log10();
            assert!(
                level.abs() < 1.0,
                "{name}, {low} to {high} Hz: {level:.2} dB"
            );
        }
    }
}

/// The streams of a pair of channels of white noise, the right at half the
/// left's level, and of a pair of unrelated noises, each coded at 96 kbit/s
/// in stereo, the first's noise sent the same in both channels, in
/// intensity stereo or, with `ms_used`, as noise in both.
fn pair_streams(ms_used: bool) -> [(bool, [Vec<f64>; 2], Vec<u8>); 2] {
    let len = 40 * 1024;
    let left = noise(len, 5);
    let half: Vec<f64> = left.iter().map(|x| 0.5 * x).collect();
    [(true, half), (false, noise(len, 6))].map(|(alike, right)| {
        let input = [left.clone(), right];
        let config = StreamConfig {
            sampling_frequency: SamplingFrequency::Hz44100,
            channel_configuration: ChannelConfiguration::Stereo,
            bits_per_second: 96_000,
            control: Control::Constant,
            framing: Framing::Adts,
            flags: Default::default(),
            crc_protected: false,
            noise_substitution: true,
        };
        let mut encoder = StreamEncoder::new(config);
        if ms_used {
            encoder = encoder.noise_shared_by_ms_used();
        }
        let mut stream = Vec::new();
        encoder.encode(&[&input[0], &input[1]], &mut stream);
        encoder.finish(&mut stream);
        (alike, input, stream)
    })
}

/// The share of the bands of noise in the left channel of `elements`,
/// past the first two frames, that the pair sends the same in both
/// channels: in intensity stereo in the right channel, or with `ms_used`
/// where the right channel sends noise too.
fn shared_share(elements: &[Element]) -> f64 {
    let (mut noise, mut shared) = (0, 0);
    for element in &elements[2..elements.len() - 2] {
        let [l, r] = &element.channels[..] else {
            panic!("a pair")
        };
        for sfb in 0..l.info.max_sfb {
            if l.codebook(0, sfb) != NOISE_HCB {
                continue;
            }
            noise += 1;
            let ms_used = element.ms_mask_present == 2
                || element.ms_used.first().is_some_and(|used| used[sfb]);
            shared += usize::from(match r.codebook(0, sfb) {
                NOISE_HCB => ms_used,
                INTENSITY_HCB | INTENSITY_HCB2 => true,
                other => panic!("band {sfb}: codebook {other} beside noise"),
            });
        }
    }
    shared as f64 / noise as f64
}

/// The normalized correlation of the two channels of `decoded`, a pair,
/// in each of [`RANGES`], each channel's level against `input`'s within 1
/// dB.
fn correlations(decoded: &Decoded, input: &[Vec<f64>; 2], context: &str) -> Vec<f64> {
    let output = [channel(decoded, 2, 0), channel(decoded, 2, 1)];
    (RANGES.iter())
        .map(|&(low, high)| {
            let spectra = output.each_ref().map(|x| spectrum(x, low, high));
            for c in 0..2 {
                let level =
                    10.0 * (power(&spectra[c]) / power(&spectrum(&input[c], low, high))).log10();
                assert!(
                    level.abs() < 1.0,
                    "{context}, channel {c}, {low} to {high} Hz: {level:.2} dB"
                );
            }
            correlation(&spectra[0], &spectra[1])
        })
        .collect()
}

/// A pair of channels of white noise, the right at half the left's level,
/// has the bands above the border that are noise in the left channel sent
/// the same in the right, in intensity stereo, so that the decoder takes
/// the left channel's noise into the right after generating it (ISO/IEC
/// 14496-3, clause 4.1.1.2 and Figure 4.2); a pair of unrelated noises has
/// few so. ffmpeg and FAAD2 decode the first pair's channels the same in
/// those bands, and the second's not, each at its level within 1 dB.
#[test]
fn ffmpeg_and_faad_share_a_pairs_noise_through_intensity_stereo() {
    for (alike, input, stream) in pair_streams(false) {
        let elements = elements(&stream);
        for c in 0..2 {
            let share = substituted_share(&elements, c, FIRST_NOISE_BAND);
            // The right channel's shared bands are in intensity stereo.
            let share = if c == 1 && alike { 1.0 - share } else { share };
            assert!(share > 0.9, "alike {alike}, channel {c}: {share:.2}");
        }
        let shared = shared_share(&elements);
        assert!(
            if alike { shared == 1.0 } else { shared < 0.05 },
            "alike {alike}: {shared:.3} shared"
        );
        for (name, decode) in decoders() {
            let decoded = decode(&stream);
            assert_eq!(decoded.diagnostics, "", "{name}");
            let context = format!("{name}, alike {alike}");
            for correlation in correlations(&decoded, &input, &context) {
                let expected = if alike {
                    correlation > 0.95
                } else {
                    correlation.abs() < 0.1
                };
                assert!(expected, "{context}: correlation {correlation:.3}");
            }
        }
    }
}

/// The same pair of alike channels, its shared noise sent instead as noise
/// in both channels with `ms_used` set, which clause 4.6.13.3 decodes from
/// the same random vector in both: FAAD2 decodes the two channels the same,
/// and ffmpeg each with its own noise, uncorrelated, though at its level.
#[test]
fn faad_alone_shares_a_pairs_noise_where_ms_used_is_set() {
    let [(alike, input, stream), _] = pair_streams(true);
    assert!(alike);
    assert_eq!(shared_share(&elements(&stream)), 1.0);
    for (name, decode) in decoders() {
        let decoded = decode(&stream);
        assert_eq!(decoded.diagnostics, "", "{name}");
        for correlation in correlations(&decoded, &input, name) {
            let expected = match name {
                "FAAD2" => correlation > 0.95,
                _ => correlation.abs() < 0.1,
            };
            assert!(expected, "{name}: correlation {correlation:.3}");
        }
    }
}
