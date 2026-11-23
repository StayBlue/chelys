// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

use chelys::aac::{Config, DELAY, Encoder, Rate, SAMPLE_RATES};
use chelys_conformance::faad;
use chelys_conformance::ffmpeg::{self, StreamInfo};

/// Some input that ends within a frame.
const SAMPLES: usize = 10 * 1024 + 300;

fn encode_silence(sample_rate: u32, channels: usize) -> (Vec<u8>, chelys::aac::Finished) {
    // A bitrate at which the bit reservoir fills and frames are padded
    // with fill elements, at every sample rate.
    let bits_per_second = 24 * sample_rate / 10 * channels as u32;
    let config = Config::new(sample_rate, channels, Rate::Constant { bits_per_second });
    let mut encoder = Encoder::new(config).unwrap();
    let mut stream = Vec::new();
    encoder
        .encode_interleaved(&vec![0.0f32; SAMPLES * channels], &mut stream)
        .unwrap();
    let finished = encoder.finish(&mut stream);
    (stream, finished)
}

/// ffmpeg gives every frame's 1024 samples, from the first: [`DELAY`]
/// samples, the input, then the padding.
#[test]
fn ffmpeg_decodes_silence_at_every_sample_rate() {
    for sample_rate in SAMPLE_RATES {
        for channels in [1, 2] {
            let context = format!("{sample_rate} Hz, {channels} channels");
            let (stream, finished) = encode_silence(sample_rate, channels);
            let (samples, delay) = (SAMPLES as u64, DELAY as u64);
            assert_eq!(finished.samples, samples);
            assert_eq!(finished.frames * 1024, delay + samples + finished.padding);

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
            assert!(decoded.samples.iter().all(|&s| s == 0.0), "{context}");
        }
    }
}

/// ffmpeg's `AudioSpecificConfig()` for the stream, which it derives from
/// the ADTS headers, is the encoder's.
#[test]
fn ffmpeg_derives_the_audio_specific_config_from_adts() {
    for sample_rate in SAMPLE_RATES {
        for channels in [1, 2] {
            let (stream, _) = encode_silence(sample_rate, channels);
            let config = Config::new(
                sample_rate,
                channels,
                Rate::Constant {
                    bits_per_second: 24 * sample_rate / 10 * channels as u32,
                },
            );
            assert_eq!(
                ffmpeg::audio_specific_config_from_adts(&stream),
                Encoder::new(config).unwrap().audio_specific_config(),
                "{sample_rate} Hz, {channels} channels"
            );
        }
    }
}

/// A stream of two frames, the fewest but one, which ffprobe would take
/// for FLAC if it detected the format, is read as ADTS.
#[test]
fn ffmpeg_reads_the_shortest_streams() {
    let config = Config::new(
        48_000,
        1,
        Rate::Constant {
            bits_per_second: 64_000,
        },
    );
    let mut encoder = Encoder::new(config).unwrap();
    let mut stream = Vec::new();
    encoder.encode_interleaved(&[0.0f32], &mut stream).unwrap();
    assert_eq!(encoder.finish(&mut stream).frames, 2);
    let (info, profile) = ffmpeg::probe_aac(&stream);
    assert_eq!((info.codec_name.as_str(), profile.as_str()), ("aac", "LC"));
    let decoded = ffmpeg::decode_aac(&stream);
    assert_eq!(decoded.diagnostics, "");
    assert_eq!(decoded.samples.len(), 2 * 1024);
}

/// Guards the oracle itself: if ffmpeg stopped reporting what it cannot
/// decode, a wrong field would pass the test above unnoticed. Here the
/// first frame's `max_sfb` is beyond the bands there are.
#[test]
fn ffmpeg_reports_too_many_scalefactor_bands() {
    let (mut stream, _) = encode_silence(48_000, 1);
    // After the 7 bytes of the header: id_syn_ele, element_instance_tag
    // and global_gain take 15 bits, then ics_reserved_bit, window_sequence
    // and window_shape 4, so the next byte's low 5 bits are max_sfb's high
    // ones.
    stream[7 + 2] |= 0b1_1111;
    let decoded = ffmpeg::decode_aac(&stream);
    assert!(
        decoded.diagnostics.contains("scalefactor bands"),
        "ffmpeg reported: {:?}",
        decoded.diagnostics
    );
}

/// faad gives every frame's 1024 samples, from the first, as ffmpeg does,
/// and trims none; but at 24 kHz and below, twice as many, at twice the
/// sampling frequency, which its WAV header gives.
#[test]
fn faad_decodes_silence_at_every_sample_rate() {
    for sample_rate in SAMPLE_RATES {
        for channels in [1, 2] {
            let context = format!("{sample_rate} Hz, {channels} channels");
            let (stream, finished) = encode_silence(sample_rate, channels);
            let factor = if sample_rate <= 24_000 { 2 } else { 1 };
            assert_eq!(
                faad::sample_rate(&stream),
                factor * sample_rate,
                "{context}"
            );
            let decoded = faad::decode_aac(&stream);
            assert_eq!(decoded.diagnostics, "", "{context}");
            assert_eq!(
                decoded.samples.len(),
                finished.frames as usize * 1024 * channels * factor as usize,
                "{context}"
            );
            assert!(decoded.samples.iter().all(|&s| s == 0.0), "{context}");
        }
    }
}

/// Guards the oracle itself, as for ffmpeg above: faad reports the frame
/// whose `max_sfb` is beyond the bands there are, and decodes nothing of
/// the stream from it.
#[test]
fn faad_reports_too_many_scalefactor_bands() {
    let (mut stream, _) = encode_silence(48_000, 1);
    stream[7 + 2] |= 0b1_1111;
    let decoded = faad::decode_aac(&stream);
    assert!(
        decoded.diagnostics.starts_with("Error: "),
        "faad reported: {:?}",
        decoded.diagnostics
    );
    assert!(decoded.samples.is_empty());
}
