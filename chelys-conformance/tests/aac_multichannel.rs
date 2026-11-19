// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! AAC streams of more than two channels decoded by ffmpeg: where each
//! input channel lands among ffmpeg's output channels, and at what level.

use std::f64::consts::TAU;

use chelys::aac::internals::raw_data_block::decoder::{adts_frame, raw_data_block};
use chelys::aac::internals::sampling_frequency::SamplingFrequency;
use chelys::aac::{Config, DELAY, Encoder, Rate};
use chelys_conformance::ffmpeg::{self, StreamInfo};

/// The frequency in Hz of the tone of each input channel, in the input's
/// order, of a configuration of `channels`: a tone of its own in each
/// main channel, and a low one in a low frequency enhancement channel,
/// last, which codes up to 281 Hz at 48 kHz.
fn frequencies(channels: usize) -> Vec<f64> {
    let lfe = matches!(channels, 6 | 8);
    (0..channels)
        .map(|c| match c {
            c if lfe && c == channels - 1 => 70.0,
            c => [440.0, 700.0, 1100.0, 1700.0, 2300.0, 2900.0, 3700.0][c],
        })
        .collect()
}

/// The amplitude of each input channel's tone, each its own.
fn amplitude(channel: usize) -> f64 {
    0.4 - 0.03 * channel as f64
}

/// The amplitude and phase of the tone of `frequency` Hz in `samples`,
/// over the whole seconds of them: their projections on its sine and
/// cosine, over which tones of other whole frequencies cancel.
fn tone(samples: &[f64], sample_rate: u32, frequency: f64) -> (f64, f64) {
    let len = samples.len() / sample_rate as usize * sample_rate as usize;
    let (mut s, mut c) = (0.0, 0.0);
    for (t, &x) in samples[..len].iter().enumerate() {
        let phase = TAU * frequency * t as f64 / f64::from(sample_rate);
        s += x * phase.sin();
        c += x * phase.cos();
    }
    let (s, c) = (2.0 * s / len as f64, 2.0 * c / len as f64);
    (s.hypot(c), c.atan2(s))
}

/// What ffmpeg gives of a stream of `channels` channels at `sample_rate`
/// and `bits_per_second`, each a tone of [`frequencies`]: for each output
/// channel, the input channel whose tone it carries, and the ratio of the
/// levels of the other tones in it to that tone's, in dB; with the layout
/// ffprobe reports.
struct Mapping {
    layout: (String, String),
    /// For each output channel, the input channel it carries.
    sources: Vec<usize>,
    /// For each output channel, the loudest other tone in it, in dB below
    /// its own.
    crosstalk_db: Vec<f64>,
}

fn decode_tones(sample_rate: u32, channels: usize, bits_per_second: u32) -> Mapping {
    let context = format!("{sample_rate} Hz, {channels} channels, {bits_per_second} bit/s");
    let len = 3 * sample_rate as usize + 500;
    let frequencies = frequencies(channels);
    let input: Vec<Vec<f64>> = (frequencies.iter().enumerate())
        .map(|(c, &f)| {
            (0..len)
                .map(|t| amplitude(c) * (TAU * f * t as f64 / f64::from(sample_rate)).sin())
                .collect()
        })
        .collect();
    let config = Config::new(sample_rate, channels, Rate::Constant { bits_per_second });
    let mut encoder = Encoder::new(config).unwrap();
    let mut stream = Vec::new();
    let planar: Vec<&[f64]> = input.iter().map(Vec::as_slice).collect();
    encoder.encode_planar(&planar, &mut stream);
    let finished = encoder.finish(&mut stream);

    let expected = StreamInfo {
        codec_name: "aac".to_owned(),
        sample_rate,
        channels,
    };
    assert_eq!(
        ffmpeg::probe_aac(&stream),
        (expected, "LC".to_owned()),
        "{context}"
    );
    let decoded = ffmpeg::decode_aac(&stream);
    assert_eq!(decoded.diagnostics, "", "{context}");
    assert_eq!(
        decoded.samples.len(),
        finished.frames as usize * 1024 * channels,
        "{context}"
    );
    let (mut sources, mut crosstalk_db) = (Vec::new(), Vec::new());
    for o in 0..channels {
        // The output from the input's first sample, a second in, for a
        // whole number of seconds.
        let output: Vec<f64> = (decoded.samples.iter().skip(o).step_by(channels))
            .skip(DELAY + sample_rate as usize)
            .take(sample_rate as usize)
            .map(|&x| f64::from(x))
            .collect();
        let tones: Vec<(f64, f64)> = (frequencies.iter())
            .map(|&f| tone(&output, sample_rate, f))
            .collect();
        let source = (0..channels)
            .max_by(|&a, &b| tones[a].0.total_cmp(&tones[b].0))
            .unwrap();
        // Its level and sign: the phase of the input's tone over the same
        // second, which starts on a whole number of its cycles.
        let (level, phase) = tones[source];
        let level_db = 20.0 * (level / amplitude(source)).log10();
        assert!(level_db.abs() < 0.2, "{context}: output {o}, {level_db} dB");
        assert!(phase.abs() < 0.01, "{context}: output {o}, phase {phase}");
        let loudest_other = (0..channels)
            .filter(|&c| c != source)
            .map(|c| tones[c].0)
            .fold(0.0, f64::max);
        sources.push(source);
        crosstalk_db.push(20.0 * (loudest_other / level).log10());
    }
    Mapping {
        layout: ffmpeg::channel_layout_aac(&stream),
        sources,
        crosstalk_db,
    }
}

/// ffmpeg decodes channel configurations 3 to 7 as the layouts it names
/// "3.0", "4.0", "5.0", "5.1" and "7.1", each input channel in exactly one
/// output channel, in place, at its level and sign, with the other tones
/// more than 60 dB below it, at 48 and 44.1 kHz, at 96 kbit/s a channel
/// and at 32, within the bits a sample at which pairs are coded in
/// intensity stereo above 6 kHz.
///
/// Its output channels are those of the layout, in ffmpeg's order: the
/// front left and right first, then the center, then the LFE, then the
/// rest. The input's center, first, lands third; the left and right of the
/// first pair, first and second; the 4 channels' rear surround, a single
/// channel, in ffmpeg's back center, BC; the surround pair of 5.0 and 5.1
/// in ffmpeg's back left and right, BL and BR; the LFE in its LFE.
///
/// In 7.1, ffmpeg gives configuration 7, whose first pair is the left and
/// right center front speakers and second the left and right outside front
/// speakers (Table 42), as "7.1", FL+FR+FC+LFE+BL+BR+SL+SR, whose channels
/// are all of a front pair, center, LFE, back pair and side pair: the
/// first pair lands in FL and FR, the second in SL and SR, and the
/// surround pair in BL and BR.
///
/// This is the evidence for two readings that the library marks as settled
/// by decoding with ffmpeg: the order of the input's channels, those of
/// the configuration's elements in order and of a pair left first
/// (`Config::new`); and the `element_instance_tag`s, each kind of element
/// numbered from 0 (`raw_data_block::element_instance_tags`), which ffmpeg
/// decodes without complaint.
#[test]
fn ffmpeg_maps_each_channel_of_configurations_3_to_7() {
    for (channels, layout, sources) in [
        (3, "3.0", vec![1, 2, 0]),
        (4, "4.0", vec![1, 2, 0, 3]),
        (5, "5.0", vec![1, 2, 0, 3, 4]),
        (6, "5.1", vec![1, 2, 0, 5, 3, 4]),
        (8, "7.1", vec![1, 2, 0, 7, 5, 6, 3, 4]),
    ] {
        let ncc = if channels > 5 { channels - 1 } else { channels } as u32;
        for sample_rate in [48_000, 44_100] {
            // 96 kbit/s a channel, and 32, below the 0.75 bits a sample of
            // intensity stereo.
            for per_channel in [96_000, 32_000] {
                let mapping = decode_tones(sample_rate, channels, per_channel * ncc);
                let context = format!("{sample_rate} Hz, {channels} channels, {per_channel}");
                assert_eq!(mapping.layout.0, layout, "{context}");
                assert_eq!(mapping.sources, sources, "{context}");
                assert!(
                    mapping.crosstalk_db.iter().all(|&db| db < -60.0),
                    "{context}: {:?}",
                    mapping.crosstalk_db
                );
            }
        }
    }
}

/// The layouts' channels as ffmpeg lists them, which the mapping above
/// reads its output by.
#[test]
fn ffmpeg_names_the_channels_of_its_layouts() {
    for (channels, decomposition) in [
        (3, "FL+FR+FC"),
        (4, "FL+FR+FC+BC"),
        (5, "FL+FR+FC+BL+BR"),
        (6, "FL+FR+FC+LFE+BL+BR"),
        (8, "FL+FR+FC+LFE+BL+BR+SL+SR"),
    ] {
        let ncc = if channels > 5 { channels - 1 } else { channels } as u32;
        let config = Config::new(
            48_000,
            channels,
            Rate::Constant {
                bits_per_second: 64_000 * ncc,
            },
        );
        let mut encoder = Encoder::new(config).unwrap();
        let mut stream = Vec::new();
        encoder.encode_interleaved(&vec![0.0f32; 4096 * channels], &mut stream);
        encoder.finish(&mut stream);
        let (_, channels_in_order) = ffmpeg::channel_layout_aac(&stream);
        assert_eq!(channels_in_order, decomposition, "{channels} channels");
    }
}

/// Noise with clicks, a different one in each channel, in 5.1 and 7.1 at
/// the lowest and highest bitrates of 8, 48 and 96 kHz, with the CRC and
/// without, decodes without complaint, every frame whole, and no element
/// longer than its share of the decoder's input buffer, 6144 bits a
/// channel with its `id_syn_ele` (clause 8.2.3.1).
#[test]
fn ffmpeg_decodes_5_1_and_7_1_at_the_ends_of_their_range() {
    use chelys::aac::ConfigError;

    for channels in [6, 8] {
        for sample_rate in [8_000, 48_000, 96_000] {
            for crc in [false, true] {
                let config = |bits_per_second| {
                    Config::new(sample_rate, channels, Rate::Constant { bits_per_second }).crc(crc)
                };
                let Err(ConfigError::UnsupportedBitrate { min, max, .. }) = config(0).check()
                else {
                    panic!("0 bit/s is refused");
                };
                for bits_per_second in [min, max] {
                    let context =
                        format!("{channels} channels, {sample_rate} Hz, {bits_per_second}");
                    let mut state = 1 + channels as u64;
                    let len = 5 * 1024 + 77;
                    let pcm: Vec<f32> = (0..len * channels)
                        .map(|i| {
                            state = state
                                .wrapping_mul(6_364_136_223_846_793_005)
                                .wrapping_add(1_442_695_040_888_963_407);
                            let noise = ((state >> 40) as f32 / (1 << 24) as f32 - 0.5) * 0.6;
                            let (t, c) = (i / channels, i % channels);
                            if t % (700 + 90 * c) == 300 {
                                0.95
                            } else {
                                noise
                            }
                        })
                        .collect();
                    let mut encoder = Encoder::new(config(bits_per_second)).unwrap();
                    let mut stream = Vec::new();
                    encoder.encode_interleaved(&pcm, &mut stream);
                    let finished = encoder.finish(&mut stream);
                    let fs = SamplingFrequency::from_hz(sample_rate).unwrap();
                    let mut rest = stream.as_slice();
                    while !rest.is_empty() {
                        let frame = adts_frame(rest);
                        rest = &rest[frame.aac_frame_length..];
                        for element in raw_data_block(frame.raw_data_block, fs).elements {
                            let bits = 3 + element.bits.len();
                            let channels = element.element.channels.len();
                            let lfe = element.id_syn_ele == 3;
                            assert!(lfe || bits <= 6144 * channels, "{context}: {bits}");
                        }
                    }
                    let decoded = ffmpeg::decode_aac(&stream);
                    assert_eq!(decoded.diagnostics, "", "{context}");
                    assert_eq!(
                        decoded.samples.len(),
                        finished.frames as usize * 1024 * channels,
                        "{context}"
                    );
                }
            }
        }
    }
}
