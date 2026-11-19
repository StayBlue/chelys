// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! AAC streams of more than two channels decoded by ffmpeg and FAAD2:
//! where each input channel lands among each decoder's output channels,
//! and at what level.

use std::f64::consts::TAU;

use chelys::aac::internals::raw_data_block::decoder::{adts_frame, raw_data_block};
use chelys::aac::internals::sampling_frequency::SamplingFrequency;
use chelys::aac::{Config, DELAY, Encoder, Rate};
use chelys_conformance::ffmpeg::{self, StreamInfo};
use chelys_conformance::{Decoded, faad};

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

/// What `decode` gives of a stream of `channels` channels at `sample_rate`
/// and `bits_per_second`, each a tone of [`frequencies`]: for each output
/// channel, the input channel whose tone it carries, and the ratio of the
/// levels of the other tones in it to that tone's, in dB; with the layout
/// ffprobe reports.
struct Mapping {
    layout: (String, String),
    /// For each pair, in order, the frames that code bands as middle and
    /// side, and those that code bands in intensity stereo.
    joint: Vec<(usize, usize)>,
    /// For each output channel, the input channel it carries.
    sources: Vec<usize>,
    /// For each output channel, the loudest other tone in it, in dB below
    /// its own.
    crosstalk_db: Vec<f64>,
}

/// The tones shared: in every channel but a low frequency
/// enhancement channel, their frequencies in Hz and amplitudes.
const SHARED_TONES: [(f64, f64); 2] = [(5_000.0, 0.15), (9_000.0, 0.1)];

/// [`decode_tones`], and with the tones of [`SHARED_TONES`] in every main
/// channel where `shared`, which a pair codes as middle and side below 6
/// kHz and, at a low bitrate, in intensity stereo above, the output
/// checked to within `tolerance_db` of each tone's level.
fn decode_tones(
    decode: fn(&[u8]) -> Decoded,
    sample_rate: u32,
    channels: usize,
    bits_per_second: u32,
) -> Mapping {
    decode_tones_with(decode, sample_rate, channels, bits_per_second, false, 0.2)
}

fn decode_tones_with(
    decode: fn(&[u8]) -> Decoded,
    sample_rate: u32,
    channels: usize,
    bits_per_second: u32,
    shared: bool,
    tolerance_db: f64,
) -> Mapping {
    let context = format!("{sample_rate} Hz, {channels} channels, {bits_per_second} bit/s");
    let len = 3 * sample_rate as usize + 500;
    let frequencies = frequencies(channels);
    let lfe = matches!(channels, 6 | 8);
    let input: Vec<Vec<f64>> = (frequencies.iter().enumerate())
        .map(|(c, &f)| {
            let main = shared && !(lfe && c == channels - 1);
            (0..len)
                .map(|t| {
                    let at = |f: f64| (TAU * f * t as f64 / f64::from(sample_rate)).sin();
                    let others: f64 = (SHARED_TONES.iter())
                        .filter(|_| main)
                        .map(|&(f, a)| a * at(f))
                        .sum();
                    amplitude(c) * at(f) + others
                })
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
    let decoded = decode(&stream);
    assert_eq!(decoded.diagnostics, "", "{context}");
    assert_eq!(
        decoded.samples.len(),
        finished.frames as usize * 1024 * channels,
        "{context}"
    );
    // Which pairs code bands as middle and side, and which in intensity
    // stereo, its codebooks INTENSITY_HCB2 and INTENSITY_HCB, 14 and 15.
    let fs = SamplingFrequency::from_hz(sample_rate).unwrap();
    let mut joint = Vec::new();
    let mut rest = stream.as_slice();
    while !rest.is_empty() {
        let frame = adts_frame(rest);
        rest = &rest[frame.aac_frame_length..];
        let block = raw_data_block(frame.raw_data_block, fs);
        let pairs = block.elements.iter().filter(|e| e.id_syn_ele == 1);
        joint.resize(pairs.clone().count(), (0, 0));
        for (counts, pair) in joint.iter_mut().zip(pairs) {
            let element = &pair.element;
            let ms = (element.ms_used.iter()).any(|group| group.contains(&true));
            let intensity = (element.channels[1].sections.iter())
                .flatten()
                .any(|&(codebook, ..)| matches!(codebook, 14 | 15));
            counts.0 += usize::from(ms);
            counts.1 += usize::from(intensity);
        }
    }
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
        assert!(
            level_db.abs() < tolerance_db,
            "{context}: output {o}, {level_db} dB"
        );
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
        joint,
        sources,
        crosstalk_db,
    }
}

/// ffmpeg decodes channel configurations 3 to 7 as the layouts it names
/// "3.0", "4.0", "5.0", "5.1" and "7.1", each input channel in exactly one
/// output channel, in place, at its level and sign, with the other tones
/// more than 60 dB below it, at 48 and 44.1 kHz, at 96 and 32 kbit/s a
/// channel.
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
                let mapping =
                    decode_tones(ffmpeg::decode_aac, sample_rate, channels, per_channel * ncc);
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

/// FAAD2 decodes channel configurations 3 to 7 as ffmpeg does above, each
/// input channel in exactly one output channel, at its level and sign,
/// with the other tones more than 60 dB below it, at 48 and 44.1 kHz, at
/// 96 and 32 kbit/s a channel; but in an order of its own: those of
/// configurations 3, 4, 5 and 7 in the order the stream sends them, the
/// input's, and those of 5.1 in ffmpeg's, the front pair, the center, the
/// LFE and the surround pair.
#[test]
fn faad_maps_each_channel_of_configurations_3_to_7() {
    for (channels, sources) in [
        (3, vec![0, 1, 2]),
        (4, vec![0, 1, 2, 3]),
        (5, vec![0, 1, 2, 3, 4]),
        (6, vec![1, 2, 0, 5, 3, 4]),
        (8, vec![0, 1, 2, 3, 4, 5, 6, 7]),
    ] {
        let ncc = if channels > 5 { channels - 1 } else { channels } as u32;
        for sample_rate in [48_000, 44_100] {
            for per_channel in [96_000, 32_000] {
                let mapping =
                    decode_tones(faad::decode_aac, sample_rate, channels, per_channel * ncc);
                let context = format!("{sample_rate} Hz, {channels} channels, {per_channel}");
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
    let cut_short = decodes_5_1_and_7_1_at_the_ends_of_their_range(ffmpeg::decode_aac, 1, None);
    assert_eq!(cut_short, 0);
}

/// FAAD2 decodes the streams of 5.1 and 7.1 above as ffmpeg does, at 8
/// kHz at twice the sampling frequency: those of 7.1 at the highest
/// bitrates too, whose frames are kept to [`FAAD_LONGEST`] bytes, the
/// longest it decodes, where the 6144 bits that each of the seven channels
/// but the LFE may take would let them be longer.
#[test]
fn faad_decodes_5_1_and_7_1_at_the_ends_of_their_range() {
    let cut_short =
        decodes_5_1_and_7_1_at_the_ends_of_their_range(faad::decode_aac, 2, Some(FAAD_LONGEST));
    assert_eq!(cut_short, 0);
}

/// The longest `adts_frame()` in bytes that FAAD2 decodes: 768 for each
/// of six channels, though 7.1 has seven besides its LFE (see
/// `faad_decodes_no_frame_longer_than_4608_bytes`).
const FAAD_LONGEST: usize = 4608;

/// `frame`, an `adts_frame()` without a `crc_check`, padded to `len`
/// bytes with fill elements as the encoder writes them: of `EXT_FILL_DATA`,
/// 14 bytes or fewer each, before `ID_END`.
fn padded(frame: &[u8], len: usize) -> Vec<u8> {
    let push = |bits: &mut Vec<bool>, value: u32, n: u32| {
        bits.extend((0..n).rev().map(|i| (value >> i) & 1 == 1));
    };
    assert_eq!(frame[1] & 1, 1, "protection_absent 1");
    let block = &frame[7..];
    let mut bits: Vec<bool> = (0..8 * block.len())
        .map(|i| (block[i / 8] >> (7 - i % 8)) & 1 == 1)
        .collect();
    // The block's last 1 ends its ID_END, which follows the fill.
    let end = bits.iter().rposition(|&bit| bit).unwrap() + 1;
    bits.truncate(end - 3);
    let target = 8 * (len - 7);
    while target - bits.len() - 3 >= 8 {
        let cnt = ((target - bits.len() - 3 - 7) / 8).min(14) as u32;
        push(&mut bits, 0x6, 3); // ID_FIL
        push(&mut bits, cnt, 4);
        if cnt > 0 {
            push(&mut bits, 0b0001, 4); // EXT_FILL_DATA
            push(&mut bits, 0b0000, 4); // fill_nibble
            for _ in 1..cnt {
                push(&mut bits, 0b1010_0101, 8); // fill_byte
            }
        }
    }
    push(&mut bits, 0x7, 3); // ID_END
    bits.resize(target, false);
    let mut header = frame[..7].to_vec();
    header[3] = (header[3] & !0b11) | (len >> 11) as u8;
    header[4] = (len >> 3) as u8;
    header[5] = (header[5] & 0b1_1111) | ((len & 0b111) << 5) as u8;
    let bytes =
        (bits.chunks(8)).map(|byte| byte.iter().fold(0, |b, &bit| (b << 1) | u8::from(bit)));
    header.into_iter().chain(bytes).collect()
}

/// FAAD2 decodes no `adts_frame()` longer than 4608 bytes: a stereo stream
/// whose fourth frame is padded with fill elements to 4608 bytes decodes
/// whole and without complaint, and padded to 4609 bytes, FAAD2 reports
/// an error at it and gives only the three frames before it. Neither frame
/// is one the encoder writes, as a pair's frame holds 6144 bits for each
/// channel at most; they find FAAD2's limit, which the frames of 7.1 would
/// pass at the highest bitrates the standard allows, and within which the
/// encoder keeps every frame in ADTS.
#[test]
fn faad_decodes_no_frame_longer_than_4608_bytes() {
    let sample_rate = 48_000;
    let pcm: Vec<f32> = (0..8 * 1024 * 2)
        .map(|i| 0.3 * (i as f32 * 0.05).sin())
        .collect();
    let config = Config::new(
        sample_rate,
        2,
        Rate::Constant {
            bits_per_second: 64_000,
        },
    );
    let mut encoder = Encoder::new(config).unwrap();
    let mut stream = Vec::new();
    encoder.encode_interleaved(&pcm, &mut stream);
    encoder.finish(&mut stream);
    let mut frames = Vec::new();
    let mut rest = stream.as_slice();
    while !rest.is_empty() {
        let len = adts_frame(rest).aac_frame_length;
        frames.push(&rest[..len]);
        rest = &rest[len..];
    }
    for len in [FAAD_LONGEST, FAAD_LONGEST + 1] {
        let fourth = padded(frames[3], len);
        let fs = SamplingFrequency::from_hz(sample_rate).unwrap();
        let [elements, as_sent] =
            [&fourth, frames[3]].map(|frame| raw_data_block(&frame[7..], fs).elements);
        assert_eq!(elements, as_sent);
        let padded: Vec<u8> = [&frames[..3], &[&fourth[..]], &frames[4..]]
            .concat()
            .concat();
        let decoded = faad::decode_aac(&padded);
        if len == FAAD_LONGEST {
            assert_eq!(decoded.diagnostics, "", "{len} bytes");
            assert_eq!(
                decoded.samples.len(),
                frames.len() * 1024 * 2,
                "{len} bytes"
            );
        } else {
            assert!(
                decoded.diagnostics.starts_with("Error: "),
                "{len} bytes: {:?}",
                decoded.diagnostics
            );
            assert_eq!(decoded.samples.len(), 3 * 1024 * 2, "{len} bytes");
        }
    }
}

/// What the tests above check, with `decode`, which gives at 8 kHz
/// `factor` times the samples of each frame, and decodes no frame longer
/// than `longest` bytes, if given, reporting an error at the first and
/// giving the samples of those before it only; with the streams so cut
/// short.
fn decodes_5_1_and_7_1_at_the_ends_of_their_range(
    decode: fn(&[u8]) -> Decoded,
    factor: usize,
    longest: Option<usize>,
) -> usize {
    use chelys::aac::ConfigError;

    let mut cut_short = 0;
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
                    let mut decodable = None;
                    let mut frames = 0;
                    while !rest.is_empty() {
                        let frame = adts_frame(rest);
                        rest = &rest[frame.aac_frame_length..];
                        if longest.is_some_and(|longest| frame.aac_frame_length > longest) {
                            decodable.get_or_insert(frames);
                        }
                        frames += 1;
                        for element in raw_data_block(frame.raw_data_block, fs).elements {
                            let bits = 3 + element.bits.len();
                            let channels = element.element.channels.len();
                            let lfe = element.id_syn_ele == 3;
                            assert!(lfe || bits <= 6144 * channels, "{context}: {bits}");
                        }
                    }
                    assert_eq!(frames, finished.frames as usize);
                    let decoded = decode(&stream);
                    if decodable.is_some() {
                        cut_short += 1;
                        assert!(
                            decoded.diagnostics.starts_with("Error: "),
                            "{context}: {:?}",
                            decoded.diagnostics
                        );
                    } else {
                        assert_eq!(decoded.diagnostics, "", "{context}");
                    }
                    let factor = if sample_rate == 8_000 { factor } else { 1 };
                    assert_eq!(
                        decoded.samples.len(),
                        decodable.unwrap_or(frames) * 1024 * channels * factor,
                        "{context}"
                    );
                }
            }
        }
    }
    cut_short
}

/// The mapping holds where each pair codes bands as middle and side and in
/// intensity stereo: with tones of 5 and 9 kHz in every main channel
/// beside each channel's own, at 32 kbit/s a channel, within the bits a
/// sample at which pairs are coded in intensity stereo above 6 kHz, each
/// pair codes the first as middle and side and the second in intensity
/// stereo in most frames, and each channel's own tone lands in its place,
/// to within 0.5 dB of its level, with the other channels' more than 40 dB
/// below it.
#[test]
fn ffmpeg_maps_each_channel_of_pairs_coded_jointly() {
    for (channels, sources) in [
        (3, vec![1, 2, 0]),
        (4, vec![1, 2, 0, 3]),
        (5, vec![1, 2, 0, 3, 4]),
        (6, vec![1, 2, 0, 5, 3, 4]),
        (8, vec![1, 2, 0, 7, 5, 6, 3, 4]),
    ] {
        let ncc = if channels > 5 { channels - 1 } else { channels } as u32;
        for sample_rate in [48_000, 44_100] {
            let mapping = decode_tones_with(
                ffmpeg::decode_aac,
                sample_rate,
                channels,
                32_000 * ncc,
                true,
                0.5,
            );
            let context = format!("{sample_rate} Hz, {channels} channels");
            assert_eq!(mapping.sources, sources, "{context}");
            assert!(
                mapping.crosstalk_db.iter().all(|&db| db < -40.0),
                "{context}: {:?}",
                mapping.crosstalk_db
            );
            // Of about 130 frames at 44.1 kHz and 140 at 48.
            assert!(
                mapping
                    .joint
                    .iter()
                    .all(|&(ms, intensity)| ms > 100 && intensity > 100),
                "{context}: {:?}",
                mapping.joint
            );
        }
    }
}

/// FAAD2's mapping holds where each pair codes bands as middle and side
/// and in intensity stereo, as ffmpeg's does above, in FAAD2's order.
#[test]
fn faad_maps_each_channel_of_pairs_coded_jointly() {
    for (channels, sources) in [
        (3, vec![0, 1, 2]),
        (4, vec![0, 1, 2, 3]),
        (5, vec![0, 1, 2, 3, 4]),
        (6, vec![1, 2, 0, 5, 3, 4]),
        (8, vec![0, 1, 2, 3, 4, 5, 6, 7]),
    ] {
        let ncc = if channels > 5 { channels - 1 } else { channels } as u32;
        for sample_rate in [48_000, 44_100] {
            let mapping = decode_tones_with(
                faad::decode_aac,
                sample_rate,
                channels,
                32_000 * ncc,
                true,
                0.5,
            );
            let context = format!("{sample_rate} Hz, {channels} channels");
            assert_eq!(mapping.sources, sources, "{context}");
            assert!(
                mapping.crosstalk_db.iter().all(|&db| db < -40.0),
                "{context}: {:?}",
                mapping.crosstalk_db
            );
        }
    }
}
