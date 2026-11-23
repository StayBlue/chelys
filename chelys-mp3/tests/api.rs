// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The public API: what it accepts, and that its ways of giving the same
//! input give the same stream.

use std::f64::consts::TAU;
use std::panic::{AssertUnwindSafe, catch_unwind};

use chelys_mp3::{ChannelMode, Config, ConfigError, DELAY, Encoder, Rate, SAMPLE_RATES, bitrates};

const MODES: [ChannelMode; 4] = [
    ChannelMode::Mono,
    ChannelMode::Stereo,
    ChannelMode::JointStereo {
        ms_stereo: true,
        intensity_stereo: true,
    },
    ChannelMode::DualChannel,
];

/// A tone with a sharp onset halfway, in each channel.
fn tone(sample_rate: u32, len: usize, channels: usize) -> Vec<Vec<f64>> {
    (0..channels)
        .map(|c| {
            (0..len)
                .map(|t| {
                    let s = t as f64 / f64::from(sample_rate);
                    let level = if t < len / 2 { 0.05 } else { 0.8 };
                    level * (TAU * (440.0 + 110.0 * c as f64) * s).sin()
                })
                .collect()
        })
        .collect()
}

fn encode(config: Config, pcm: &[Vec<f64>]) -> Vec<u8> {
    let mut encoder = Encoder::new(config).unwrap();
    let slices: Vec<&[f64]> = pcm.iter().map(Vec::as_slice).collect();
    let mut stream = Vec::new();
    encoder.encode_planar(&slices, &mut stream);
    encoder.finish(&mut stream);
    stream
}

/// Rates to try at `sample_rate`, valid and not.
fn rates(sample_rate: u32) -> Vec<Rate> {
    let table = bitrates(sample_rate).unwrap_or(&[32, 320]);
    let (lowest, highest) = (table[0], table[table.len() - 1]);
    let variable = |min_kbps, max_kbps, noise_offset_db| Rate::Variable {
        min_kbps,
        max_kbps,
        noise_offset_db,
    };
    vec![
        Rate::Constant { kbps: lowest },
        Rate::Constant { kbps: highest },
        Rate::Constant { kbps: lowest + 1 },
        Rate::Mean {
            bits_per_second: lowest * 1000 + 500,
        },
        Rate::Mean {
            bits_per_second: lowest * 1000 - 1,
        },
        Rate::Mean {
            bits_per_second: highest * 1000 + 1,
        },
        Rate::Quality {
            level: 0,
            min_kbps: None,
            max_kbps: None,
        },
        Rate::Quality {
            level: 100,
            min_kbps: Some(highest),
            max_kbps: Some(lowest),
        },
        Rate::Quality {
            level: 101,
            min_kbps: None,
            max_kbps: None,
        },
        variable(None, None, 0.0),
        variable(Some(highest), Some(lowest), 0.0),
        variable(Some(lowest + 1), None, 0.0),
        variable(None, None, f64::NAN),
        Rate::Free {
            bits_per_second: (highest - 1) * 1000,
        },
        Rate::Free {
            bits_per_second: highest * 1000,
        },
        Rate::Free {
            bits_per_second: 1000,
        },
    ]
}

/// No configuration panics: [`Encoder::new`] and [`Config::check`] agree,
/// and every one they take encodes, with the most ancillary data it takes.
#[test]
fn every_configuration_is_taken_or_refused_without_panicking() {
    let sample_rates = SAMPLE_RATES.into_iter().chain([0, 44_000, 96_000]);
    let taken: usize = std::thread::scope(|scope| {
        let threads: Vec<_> = sample_rates
            .map(|sample_rate| scope.spawn(move || sweep(sample_rate)))
            .collect();
        threads.into_iter().map(|t| t.join().unwrap()).sum()
    });
    assert!(taken > 0);
}

/// Tries every configuration of [`rates`] at `sample_rate`, and gives the
/// number taken.
fn sweep(sample_rate: u32) -> usize {
    let mut taken = 0;
    for rate in rates(sample_rate) {
        for mode in MODES {
            for (crc, mixed_blocks) in [(false, false), (true, true)] {
                let config = Config::new(sample_rate, mode, rate)
                    .crc(crc)
                    .mixed_blocks(mixed_blocks)
                    .info_tag(true);
                let case = format!("{config:?}");
                let max = match config.ancillary_bits_per_frame(usize::MAX).check() {
                    Err(ConfigError::AncillaryTooLong { max, .. }) => max,
                    Err(_) => {
                        assert!(config.check().is_err(), "{case}");
                        assert!(Encoder::new(config).is_err(), "{case}");
                        continue;
                    }
                    Ok(()) => panic!("{case}: no limit to ancillary data"),
                };
                assert!(
                    config.ancillary_bits_per_frame(max + 1).check().is_err(),
                    "{case}"
                );
                let config = config.ancillary_bits_per_frame(max);
                assert_eq!(config.check(), Ok(()), "{case}");
                let pcm = tone(sample_rate, 2_000, mode.channels());
                let encoded = catch_unwind(AssertUnwindSafe(|| {
                    let mut encoder = Encoder::new(config).unwrap();
                    encoder.push_ancillary(&[0xA5; 4096]);
                    let slices: Vec<&[f64]> = pcm.iter().map(Vec::as_slice).collect();
                    let mut stream = Vec::new();
                    encoder.encode_planar(&slices, &mut stream);
                    encoder.finish(&mut stream);
                }));
                assert!(encoded.is_ok(), "{case}");
                taken += 1;
            }
        }
    }
    taken
}

#[test]
fn errors_name_what_was_refused() {
    let refused = |config: Config| config.check().unwrap_err();
    let constant = |kbps| Rate::Constant { kbps };
    assert_eq!(
        refused(Config::new(44_000, ChannelMode::Mono, constant(128))),
        ConfigError::UnsupportedSampleRate {
            sample_rate: 44_000
        }
    );
    assert_eq!(
        refused(Config::new(22_050, ChannelMode::Mono, constant(192))),
        ConfigError::UnsupportedBitrate {
            sample_rate: 22_050,
            kbps: 192
        }
    );
    let free = Rate::Free {
        bits_per_second: 128_000,
    };
    assert_eq!(
        refused(Config::new(44_100, ChannelMode::Mono, free)),
        ConfigError::UnsupportedFreeBitrate {
            sample_rate: 44_100,
            bits_per_second: 128_000,
            in_table: true
        }
    );
    let quality = |level| Rate::Quality {
        level,
        min_kbps: None,
        max_kbps: None,
    };
    assert_eq!(
        refused(Config::new(44_100, ChannelMode::Mono, quality(101))),
        ConfigError::QualityLevel { level: 101 }
    );
    let mixed = Config::new(8_000, ChannelMode::Mono, constant(8)).mixed_blocks(true);
    assert_eq!(
        refused(mixed),
        ConfigError::MixedBlocksUnsupported { sample_rate: 8_000 }
    );
    let error: Box<dyn std::error::Error> = Box::new(refused(mixed));
    assert_eq!(error.to_string(), "mixed blocks are not used at 8000 Hz");
}

#[test]
fn the_bitrates_are_those_of_each_sample_rate() {
    for sample_rate in SAMPLE_RATES {
        let table = bitrates(sample_rate).unwrap();
        assert_eq!(table.len(), 14, "{sample_rate}");
        assert!(table.is_sorted(), "{sample_rate}");
        assert_eq!(
            table[table.len() - 1],
            match sample_rate {
                32_000 | 44_100 | 48_000 => 320,
                _ => 160,
            }
        );
    }
    assert_eq!(bitrates(44_000), None);
}

/// Interleaved and planar input, of every sample type, give one stream,
/// however the input is divided between calls.
#[test]
fn every_form_of_the_same_input_gives_the_same_stream() {
    let config = Config::new(
        44_100,
        ChannelMode::JointStereo {
            ms_stereo: true,
            intensity_stereo: false,
        },
        Rate::Constant { kbps: 128 },
    );
    // Values every sample type holds exactly: multiples of 2^-15.
    let pcm: Vec<Vec<f64>> = tone(44_100, 10_000, 2)
        .into_iter()
        .map(|c| {
            c.into_iter()
                .map(|x| (x * 32_768.0).round() / 32_768.0)
                .collect()
        })
        .collect();
    let planar = encode(config, &pcm);

    let interleaved: Vec<f64> = (0..pcm[0].len())
        .flat_map(|t| [pcm[0][t], pcm[1][t]])
        .collect();
    type Feed = dyn Fn(&mut Encoder, &[f64], &mut Vec<u8>);
    let in_chunks = |chunk: usize, samples: &Feed| {
        let mut encoder = Encoder::new(config).unwrap();
        let mut stream = Vec::new();
        for part in interleaved.chunks(chunk) {
            samples(&mut encoder, part, &mut stream);
        }
        encoder.finish(&mut stream);
        stream
    };
    let f64s = |e: &mut Encoder, part: &[f64], out: &mut Vec<u8>| e.encode_interleaved(part, out);
    let f32s = |e: &mut Encoder, part: &[f64], out: &mut Vec<u8>| {
        let part: Vec<f32> = part.iter().map(|&x| x as f32).collect();
        e.encode_interleaved(&part, out);
    };
    let i16s = |e: &mut Encoder, part: &[f64], out: &mut Vec<u8>| {
        let part: Vec<i16> = part.iter().map(|&x| (x * 32_768.0) as i16).collect();
        e.encode_interleaved(&part, out);
    };
    assert!(in_chunks(interleaved.len(), &f64s) == planar);
    assert!(in_chunks(2 * 377, &f64s) == planar);
    assert!(in_chunks(2 * 1_000, &f32s) == planar);
    assert!(in_chunks(2 * 4_801, &i16s) == planar);
}

/// What [`Encoder::finish`] gives agrees with the stream: the tag's frame
/// is as long as its placeholder, and the frames hold the input and
/// [`DELAY`].
#[test]
fn finishing_gives_the_tag_and_the_counts() {
    let config = Config::new(22_050, ChannelMode::Mono, Rate::Constant { kbps: 32 })
        .info_tag(true)
        .ancillary_bits_per_frame(8);
    let mut encoder = Encoder::new(config).unwrap();
    assert_eq!(encoder.samples_per_frame(), 576);
    let tag_len = encoder.info_tag_len().unwrap();
    let pcm = tone(22_050, 5_000, 1);
    let mut stream = Vec::new();
    encoder.push_ancillary(&[0xFF; 1_000]);
    encoder.encode_planar(&[pcm[0].as_slice()], &mut stream);
    let finished = encoder.finish(&mut stream);
    let tag = finished.info_tag.unwrap();
    assert_eq!(tag.len(), tag_len);
    assert_eq!(tag[..4], stream[..4]);
    assert_eq!(finished.samples, 5_000);
    assert_eq!(finished.frames, (5_000 + DELAY as u64).div_ceil(576));
    assert_eq!(
        finished.ancillary_dropped as u64,
        8 * 1_000 - 8 * finished.frames
    );

    let untagged = Config::new(22_050, ChannelMode::Mono, Rate::Constant { kbps: 32 });
    assert_eq!(Encoder::new(untagged).unwrap().info_tag_len(), None);
}

/// Samples far beyond full scale are clamped, and not a number is refused.
#[test]
fn samples_are_finite_and_clamped() {
    let config = Config::new(32_000, ChannelMode::Mono, Rate::Constant { kbps: 64 });
    let shape = tone(32_000, 4_000, 1).remove(0);
    let loud: Vec<f64> = shape
        .iter()
        .map(|&x| (x * 1e12).clamp(-65_536.0, 65_536.0))
        .collect();
    let louder: Vec<f64> = shape.iter().map(|&x| x * 1e12).collect();
    assert!(encode(config, &[louder]) == encode(config, &[loud]));

    let mut nan = shape;
    nan[100] = f64::NAN;
    let refused = catch_unwind(|| encode(config, &[nan]));
    assert!(refused.is_err());
}

#[test]
fn an_encoder_can_move_between_threads() {
    fn send<T: Send>() {}
    send::<Encoder>();
}
