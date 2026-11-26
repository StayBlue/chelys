// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

use chelys::ogg::internals::{self, Limits};
use chelys::ogg::vorbis::Encoder;
use chelys::vorbis::{Config, Rate};
use chelys_conformance::ffmpeg::{self, StreamInfo};

fn config(sample_rate: u32, channels: usize) -> Config {
    Config::new(sample_rate, channels, Rate::Quality { level: 50 })
}

/// A tone in each channel, of a frequency of its own, with a burst of
/// noise, so that packets are of many sizes.
fn music(sample_rate: u32, channels: usize, samples: usize) -> Vec<f32> {
    let mut seed = 0x1234_5678u32;
    (0..samples * channels)
        .map(|i| {
            let (t, c) = ((i / channels) as f32, (i % channels) as f32);
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (seed >> 8) as f32 / (1 << 24) as f32 - 0.5;
            let tone = (t * (440.0 + 110.0 * c) * std::f32::consts::TAU / sample_rate as f32).sin();
            let burst = if (t as u32 / 4096) % 5 == 0 {
                noise
            } else {
                0.0
            };
            0.3 * tone + 0.3 * burst
        })
        .collect()
}

fn encode(mut encoder: Encoder, pcm: &[f32]) -> Vec<u8> {
    let mut ogg = encoder.header_pages().to_vec();
    for chunk in pcm.chunks(encoder.channels() * 1000) {
        encoder.encode_interleaved(chunk, &mut ogg).unwrap();
    }
    encoder.finish(&mut ogg);
    ogg
}

/// ffmpeg reads the encoder's streams as Vorbis of their sample rate and
/// channels, and decodes as many samples as the input, from a stream of no
/// input on.
#[test]
fn ffmpeg_decodes_the_encoders_streams_to_the_inputs_length() {
    for (sample_rate, channels) in [(44_100, 2), (48_000, 1), (22_050, 6)] {
        for samples in [0, 1, 1023, 4097, 3 * sample_rate as usize + 17] {
            let context = format!("{sample_rate} Hz, {channels} channels, {samples} samples");
            let pcm = music(sample_rate, channels, samples);
            let encoder = Encoder::new(config(sample_rate, channels), 7).unwrap();
            let ogg = encode(encoder, &pcm);
            let expected = StreamInfo {
                codec_name: "vorbis".to_owned(),
                sample_rate,
                channels,
            };
            assert_eq!(ffmpeg::probe_ogg(&ogg), expected, "{context}");
            let decoded = ffmpeg::decode_ogg(&ogg);
            assert_eq!(decoded.diagnostics, "", "{context}");
            assert_eq!(decoded.samples.len(), samples * channels, "{context}");
        }
    }
}

/// ffmpeg decodes a stream whose headers and audio packets continue across
/// pages, and some of whose pages end no packet, to the same samples as
/// the stream of the encoder's own pages: the header pages of granule
/// position 0 and the audio pages of -1 are read as Appendix A.2 and RFC
/// 3533 give them.
#[test]
fn ffmpeg_decodes_packets_continued_across_pages() {
    let pcm = music(44_100, 2, 2 * 44_100 + 500);
    let reference = ffmpeg::decode_ogg(&encode(Encoder::new(config(44_100, 2), 7).unwrap(), &pcm));
    assert_eq!(reference.diagnostics, "");
    for segments in [1, 2, 5] {
        let limits = Limits {
            bytes: 1,
            span: 1,
            segments,
        };
        let encoder = internals::vorbis_encoder(config(44_100, 2), 7, limits).unwrap();
        let ogg = encode(encoder, &pcm);
        let decoded = ffmpeg::decode_ogg(&ogg);
        assert_eq!(decoded.diagnostics, "", "{segments} segments a page");
        assert!(
            decoded.samples == reference.samples,
            "{segments} segments a page: other samples"
        );
    }
}
