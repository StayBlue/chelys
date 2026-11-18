// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

use chelys::aac::{Config, DELAY, Encoder, Rate, SAMPLE_RATES};
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
    encoder.encode_interleaved(&vec![0.0f32; SAMPLES * channels], &mut stream);
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
            assert_eq!(finished.samples, SAMPLES);
            assert_eq!(finished.frames * 1024, DELAY + SAMPLES + finished.padding);

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
                finished.frames * 1024 * channels,
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
