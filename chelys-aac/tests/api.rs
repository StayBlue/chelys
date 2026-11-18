// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The public API: what it accepts, and that its ways of giving the same
//! input give the same stream.

use std::panic::{AssertUnwindSafe, catch_unwind};

use chelys_aac::{Config, ConfigError, DELAY, Encoder, Framing, Rate, SAMPLE_RATES};

fn constant(bits_per_second: u32) -> Rate {
    Rate::Constant { bits_per_second }
}

/// A tone in each channel, interleaved.
fn pcm(len: usize, channels: usize) -> Vec<f64> {
    (0..len * channels)
        .map(|i| 0.5 * ((i / channels) as f64 * 0.05 + i as f64).sin())
        .collect()
}

fn encode_whole(config: Config, pcm: &[f64]) -> Vec<u8> {
    let mut encoder = Encoder::new(config).unwrap();
    let mut stream = Vec::new();
    encoder.encode_interleaved(pcm, &mut stream);
    encoder.finish(&mut stream);
    stream
}

fn panics(f: impl FnOnce()) -> bool {
    catch_unwind(AssertUnwindSafe(f)).is_err()
}

#[test]
fn every_sample_rate_and_channel_count_is_taken() {
    for sample_rate in SAMPLE_RATES {
        for channels in [1, 2] {
            let config = Config::new(sample_rate, channels, constant(32_000 * channels as u32));
            assert_eq!(config.check(), Ok(()), "{sample_rate} Hz, {channels}");
            let encoder = Encoder::new(config).unwrap();
            assert_eq!(encoder.channels(), channels);
            assert_eq!(encoder.samples_per_frame(), 1024);
        }
    }
}

#[test]
fn errors_name_what_was_refused() {
    for sample_rate in [0, 7_350, 44_000, 192_000] {
        assert_eq!(
            Encoder::new(Config::new(sample_rate, 1, constant(64_000))).err(),
            Some(ConfigError::UnsupportedSampleRate { sample_rate })
        );
    }
    for channels in [0, 3, 6, usize::MAX] {
        assert_eq!(
            Encoder::new(Config::new(44_100, channels, constant(64_000))).err(),
            Some(ConfigError::UnsupportedChannels { channels })
        );
    }
    for bits_per_second in [0, 100, 1_000_000, u32::MAX] {
        let error = Encoder::new(Config::new(44_100, 1, constant(bits_per_second))).err();
        let Some(ConfigError::UnsupportedBitrate {
            bits_per_second: refused,
            min,
            max,
        }) = error
        else {
            panic!("{bits_per_second} bit/s gave {error:?}");
        };
        assert_eq!(refused, bits_per_second);
        assert!(min < 64_000 && 64_000 < max, "{min} to {max}");
        for edge in [min, max] {
            assert_eq!(Config::new(44_100, 1, constant(edge)).check(), Ok(()));
        }
        assert!(
            error
                .unwrap()
                .to_string()
                .contains(&format!("{min} to {max}"))
        );
    }
}

/// Raw frames lack the headers, so the lowest bitrate is lower.
#[test]
fn the_bitrate_range_depends_on_the_framing() {
    let range = |framing| match Config::new(8_000, 1, constant(0)).framing(framing).check() {
        Err(ConfigError::UnsupportedBitrate { min, max, .. }) => (min, max),
        other => panic!("{other:?}"),
    };
    let (adts, raw) = (range(Framing::Adts), range(Framing::Raw));
    assert!(raw.0 < adts.0);
    assert_eq!(raw.1, adts.1);
}

#[test]
fn finishing_gives_the_counts() {
    let config = Config::new(32_000, 2, constant(96_000));
    for samples in [0, 1, 1023, 1024, 1025, 5000] {
        let mut encoder = Encoder::new(config).unwrap();
        let mut frames = Vec::new();
        let mut out = |frame: &[u8]| frames.push(frame.len());
        encoder.encode_interleaved(&pcm(samples, 2), &mut out);
        let finished = encoder.finish(&mut out);
        let (samples, delay) = (samples as u64, DELAY as u64);
        assert_eq!(finished.samples, samples);
        assert_eq!(finished.frames, (samples + delay).div_ceil(1024));
        assert_eq!(finished.frames, frames.len() as u64);
        assert_eq!(finished.frames * 1024, delay + samples + finished.padding);
        assert!(finished.padding < 1024);
    }
}

/// Interleaved and planar input, of every sample type, give one stream,
/// however the input is divided between calls.
#[test]
fn every_form_of_the_same_input_gives_the_same_stream() {
    for framing in [Framing::Adts, Framing::Raw] {
        let config = Config::new(22_050, 2, constant(64_000)).framing(framing);
        // Values every sample type holds exactly: multiples of 2^-15.
        let input: Vec<f64> = pcm(9_000, 2)
            .into_iter()
            .map(|x| (x * 32_768.0).round() / 32_768.0)
            .collect();
        let whole = encode_whole(config, &input);

        for frames in [1, 377, 1024, 4801] {
            let mut encoder = Encoder::new(config).unwrap();
            let mut stream = Vec::new();
            encoder.encode_interleaved::<f64, _>(&[], &mut stream);
            for (i, part) in input.chunks(2 * frames).enumerate() {
                match i % 3 {
                    0 => encoder.encode_interleaved(part, &mut stream),
                    1 => {
                        let part: Vec<f32> = part.iter().map(|&x| x as f32).collect();
                        encoder.encode_interleaved(&part, &mut stream);
                    }
                    _ => {
                        let part: Vec<i16> = part.iter().map(|&x| (x * 32_768.0) as i16).collect();
                        encoder.encode_interleaved(&part, &mut stream);
                    }
                }
            }
            encoder.finish(&mut stream);
            assert!(stream == whole, "{framing:?}, interleaved by {frames}");
        }

        let left: Vec<f64> = input.iter().copied().step_by(2).collect();
        let right: Vec<f64> = input.iter().copied().skip(1).step_by(2).collect();
        let mut encoder = Encoder::new(config).unwrap();
        let mut stream = Vec::new();
        for (from, to) in [(0, 0), (0, 10), (10, 3000), (3000, 3001), (3001, 9000)] {
            encoder.encode_planar(&[&left[from..to], &right[from..to]], &mut stream);
        }
        encoder.finish(&mut stream);
        assert!(stream == whole, "{framing:?}, planar");
    }
}

/// Samples far beyond full scale are clamped, and not a number, or an
/// infinite one, is refused before any of the call is coded.
#[test]
fn samples_are_finite_and_clamped() {
    let config = Config::new(32_000, 1, constant(64_000));
    let loud: Vec<f64> = pcm(4_000, 1).iter().map(|&x| x * 1e12).collect();
    let clamped: Vec<f64> = loud.iter().map(|&x| x.clamp(-65_536.0, 65_536.0)).collect();
    assert!(encode_whole(config, &loud) == encode_whole(config, &clamped));

    for refused in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut input = pcm(4_000, 1);
        input[3_000] = refused;
        let mut encoder = Encoder::new(config).unwrap();
        let mut stream = Vec::new();
        assert!(panics(|| encoder.encode_interleaved(&input, &mut stream)));
        assert!(stream.is_empty(), "{refused}");
        let mut encoder = Encoder::new(config).unwrap();
        assert!(panics(|| encoder.encode_planar(&[&input[..]], &mut stream)));
        assert!(stream.is_empty(), "{refused}");
    }
}

/// Input that is not whole sample frames, or not a slice of one length per
/// channel, is refused.
#[test]
fn input_must_be_whole_sample_frames() {
    let config = Config::new(48_000, 2, constant(128_000));
    let encoder = || Encoder::new(config).unwrap();
    let mut out = Vec::new();
    assert!(panics(
        || encoder().encode_interleaved(&[0.0f32; 3], &mut out)
    ));
    assert!(panics(
        || encoder().encode_planar(&[&[0.0f32; 4][..]], &mut out)
    ));
    let three: [&[f32]; 3] = [&[0.0; 4], &[0.0; 4], &[0.0; 4]];
    assert!(panics(|| encoder().encode_planar(&three, &mut out)));
    assert!(panics(
        || encoder().encode_planar(&[&[0.0f32; 4][..], &[0.0; 5]], &mut out)
    ));
    assert!(out.is_empty());
}

#[test]
fn every_sample_type_is_taken() {
    let config = Config::new(44_100, 1, constant(64_000));
    let mut encoder = Encoder::new(config).unwrap();
    let mut stream = Vec::new();
    encoder.encode_interleaved(&[i16::MIN, i16::MAX], &mut stream);
    encoder.encode_interleaved(&[f32::MAX, f32::MIN_POSITIVE], &mut stream);
    encoder.encode_planar(&[&[-1e300, 1e-300][..]], &mut stream);
    assert_eq!(encoder.finish(&mut stream).samples, 6);
}

/// One `AudioSpecificConfig()` per sample rate and channel count, with its
/// `samplingFrequencyIndex` and `channelConfiguration`.
#[test]
fn the_audio_specific_config_names_the_stream() {
    for (index, sample_rate) in SAMPLE_RATES.into_iter().enumerate() {
        for channels in [1, 2] {
            let config = Config::new(sample_rate, channels, constant(16_000));
            let asc = Encoder::new(config).unwrap().audio_specific_config();
            let bits = u16::from_be_bytes([asc[0], asc[1]]);
            assert_eq!(asc.len(), 2);
            assert_eq!(bits >> 11, 2, "audioObjectType");
            assert_eq!(usize::from((bits >> 7) & 0xF), index);
            assert_eq!(usize::from((bits >> 3) & 0xF), channels);
            assert_eq!(bits & 0b111, 0);
        }
    }
}

#[test]
fn an_encoder_can_move_between_threads() {
    fn send<T: Send>() {}
    send::<Encoder>();
}

/// No input at the edges of what the encoder takes makes it panic, at
/// any sample rate, channel count or bitrate: square waves at full scale
/// and at the clamp, the highest frequency at the clamp, noise beyond it,
/// full-scale DC and silence, at the lowest and highest bitrates.
#[test]
fn inputs_at_the_edges_encode_at_every_rate() {
    let len = 3 * 1024;
    let square: Vec<f64> = (0..len)
        .map(|t| if (t / 41) % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let mut state = 1u64;
    let noise: Vec<f64> = (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 1e12
        })
        .collect();
    let signals = [
        square.clone(),
        square.iter().map(|x| x * 65_536.0).collect(),
        (0..len)
            .map(|t| if t % 2 == 0 { 1e9 } else { -1e9 })
            .collect(),
        noise,
        vec![1.0; len],
        vec![0.0; len],
    ];
    for sample_rate in SAMPLE_RATES {
        for channels in [1, 2] {
            let Err(ConfigError::UnsupportedBitrate { min, max, .. }) =
                Config::new(sample_rate, channels, constant(0)).check()
            else {
                panic!("0 bit/s is refused");
            };
            for bits_per_second in [min, max] {
                let config = Config::new(sample_rate, channels, constant(bits_per_second));
                for signal in &signals {
                    let interleaved: Vec<f64> = signal
                        .iter()
                        .flat_map(|&x| std::iter::repeat_n(x, channels))
                        .collect();
                    encode_whole(config, &interleaved);
                }
            }
        }
    }
}
