// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The public API: what it accepts, and that its ways of giving the same
//! input give the same stream.

use chelys_aac::{Config, ConfigError, DELAY, Encoder, Framing, Rate, SAMPLE_RATES};

fn constant(bits_per_second: u32) -> Rate {
    Rate::Constant { bits_per_second }
}

/// Some input of `channels` interleaved, with values of every kind.
fn pcm(len: usize, channels: usize) -> Vec<f64> {
    (0..len * channels)
        .map(|i| match i % 7 {
            0 => f64::NAN,
            1 => f64::INFINITY,
            2 => -1e300,
            _ => (i as f64 * 0.01).sin(),
        })
        .collect()
}

fn encode_whole(config: Config, pcm: &[f64]) -> Vec<u8> {
    let mut encoder = Encoder::new(config).unwrap();
    let mut stream = Vec::new();
    encoder.encode_interleaved(pcm, &mut stream);
    encoder.finish(&mut stream);
    stream
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
        assert_eq!(finished.samples, samples);
        assert_eq!(finished.frames, (samples + DELAY).div_ceil(1024));
        assert_eq!(finished.frames, frames.len());
        assert_eq!(finished.frames * 1024, DELAY + samples + finished.padding);
        assert!(finished.padding < 1024);
    }
}

/// However the input is divided, within sample frames too, and in whichever
/// form it comes, the stream is the same.
#[test]
fn the_stream_does_not_depend_on_how_the_input_is_divided() {
    for framing in [Framing::Adts, Framing::Raw] {
        let config = Config::new(22_050, 2, constant(64_000)).framing(framing);
        let input = pcm(9_000, 2);
        let whole = encode_whole(config, &input);

        let mut encoder = Encoder::new(config).unwrap();
        let mut stream = Vec::new();
        let mut rest = input.as_slice();
        for len in [0, 1, 0, 2, 3, 2047, 5, 4096].into_iter().cycle() {
            let (chunk, after) = rest.split_at(len.min(rest.len()));
            encoder.encode_interleaved(chunk, &mut stream);
            rest = after;
            if rest.is_empty() {
                break;
            }
        }
        encoder.finish(&mut stream);
        assert!(stream == whole, "{framing:?}, interleaved");

        let left: Vec<f64> = input.iter().copied().step_by(2).collect();
        let right: Vec<f64> = input.iter().copied().skip(1).step_by(2).collect();
        let mut encoder = Encoder::new(config).unwrap();
        let mut stream = Vec::new();
        // The left channel ahead of the right, then behind it.
        encoder.encode_planar(&[&left[..5000], &right[..10]], &mut stream);
        encoder.encode_planar(&[&left[5000..6000], &right[10..]], &mut stream);
        encoder.encode_planar::<f64, _>(&[], &mut stream);
        encoder.encode_planar(&[&left[6000..]], &mut stream);
        encoder.finish(&mut stream);
        assert!(stream == whole, "{framing:?}, planar");
    }
}

/// A channel given fewer samples than the other is padded with silence.
#[test]
fn uneven_channels_are_padded_with_silence() {
    let config = Config::new(16_000, 2, constant(48_000));
    let mut encoder = Encoder::new(config).unwrap();
    let mut stream = Vec::new();
    encoder.encode_interleaved(&[0.5f32; 3], &mut stream);
    encoder.encode_planar(&[&[0.5f32; 2000][..], &[]], &mut stream);
    let finished = encoder.finish(&mut stream);
    assert_eq!(finished.samples, 2002);
    assert_eq!(finished.frames, (2002 + DELAY).div_ceil(1024));
}

/// Slices past the channels are ignored.
#[test]
fn extra_planar_slices_are_ignored() {
    let config = Config::new(48_000, 1, constant(64_000));
    let input = pcm(3000, 1);
    let mut encoder = Encoder::new(config).unwrap();
    let mut stream = Vec::new();
    encoder.encode_planar(&[&input[..], &input[..10]], &mut stream);
    encoder.finish(&mut stream);
    assert!(stream == encode_whole(config, &input));
}

#[test]
fn every_sample_type_is_taken() {
    let config = Config::new(44_100, 1, constant(64_000));
    let mut encoder = Encoder::new(config).unwrap();
    let mut stream = Vec::new();
    encoder.encode_interleaved(&[i16::MIN, i16::MAX], &mut stream);
    encoder.encode_interleaved(&[f32::NAN, f32::MAX], &mut stream);
    encoder.encode_planar(&[&[f64::NEG_INFINITY, 1e-300][..]], &mut stream);
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
