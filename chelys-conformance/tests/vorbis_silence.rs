// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

use chelys::vorbis::{Config, Encoder, Finished, Headers, MAX_CHANNELS, Packet, Rate};
use chelys_conformance::ffmpeg::{self, StreamInfo};
use chelys_conformance::ogg::{self, AudioPacket};

/// Some input that ends within a block's hop.
const SAMPLES: usize = 10 * 1024 + 300;

/// Sample rates of the MPEG standards and beyond, as Vorbis has no table
/// of them.
const SAMPLE_RATES: [u32; 9] = [
    8_000, 11_025, 16_000, 22_050, 32_000, 44_100, 48_000, 96_000, 192_000,
];

/// The rate of every stream: the noise offset 0 dB, the psychoacoustic
/// model's allowance, raised with the width of each critical band.
const RATE: Rate = Rate::Variable {
    noise_offset_db: 0.0,
};

fn encode_silence(
    sample_rate: u32,
    channels: usize,
    samples: usize,
) -> (Headers, Vec<AudioPacket>, Finished) {
    let mut encoder = Encoder::new(Config::new(sample_rate, channels, RATE)).unwrap();
    let headers = encoder.headers();
    let mut packets = Vec::new();
    let mut out = |packet: Packet<'_>| {
        packets.push(AudioPacket {
            data: packet.data.to_vec(),
            granule_position: packet.granule_position,
        })
    };
    encoder
        .encode_interleaved(&vec![0.0f32; samples * channels], &mut out)
        .unwrap();
    let finished = encoder.finish(&mut out);
    (headers, packets, finished)
}

/// ffmpeg reads the stream as Vorbis of its sample rate and channels, and
/// decodes the input's length of silence.
#[test]
fn ffmpeg_decodes_silence_at_any_sample_rate_and_channels() {
    for sample_rate in SAMPLE_RATES {
        for channels in 1..=MAX_CHANNELS {
            let context = format!("{sample_rate} Hz, {channels} channels");
            let (headers, packets, finished) = encode_silence(sample_rate, channels, SAMPLES);
            assert_eq!(finished.samples, SAMPLES as u64, "{context}");
            assert_eq!(finished.packets, packets.len() as u64, "{context}");
            let stream = ogg::vorbis(&headers, &packets, 4);
            let expected = StreamInfo {
                codec_name: "vorbis".to_owned(),
                sample_rate,
                channels,
            };
            assert_eq!(ffmpeg::probe_ogg(&stream), expected, "{context}");
            let decoded = ffmpeg::decode_ogg(&stream);
            assert_eq!(decoded.diagnostics, "", "{context}");
            assert_eq!(decoded.samples.len(), SAMPLES * channels, "{context}");
            assert!(decoded.samples.iter().all(|&s| s == 0.0), "{context}");
        }
    }
}

/// ffmpeg decodes as many samples as the last granule position says, the
/// input's, however the packets are put in pages, from a stream of no
/// input on: it trims what the last packet would return past the end.
#[test]
fn ffmpeg_trims_the_stream_to_its_last_granule_position() {
    for samples in [0, 1, 1023, 1024, 1025, 2048, 44_101] {
        for packets_per_page in [1, 3, 100] {
            let context = format!("{samples} samples, {packets_per_page} packets a page");
            let (headers, packets, _) = encode_silence(48_000, 1, samples);
            let decoded = ffmpeg::decode_ogg(&ogg::vorbis(&headers, &packets, packets_per_page));
            assert_eq!(decoded.diagnostics, "", "{context}");
            assert_eq!(decoded.samples.len(), samples, "{context}");
        }
    }
}

/// Where every audio packet is on one page, the first audio page and the
/// last, ffmpeg decodes another length than the page's granule position,
/// the input's.
#[test]
fn ffmpeg_misreads_a_stream_on_one_audio_page() {
    for samples in [1, 1024, 2048, 44_101] {
        let (headers, packets, _) = encode_silence(48_000, 1, samples);
        let decoded = ffmpeg::decode_ogg(&ogg::vorbis_on_one_audio_page(&headers, &packets));
        assert_eq!(decoded.diagnostics, "", "{samples} samples");
        assert_ne!(decoded.samples.len(), samples, "{samples} samples");
    }
}

/// Guards the CRC that `chelys::ogg` computes: ffmpeg reports a
/// page whose CRC is wrong, and so checks the CRC of the pages it reads
/// without error.
#[test]
fn ffmpeg_reports_a_page_whose_crc_is_wrong() {
    let (headers, packets, _) = encode_silence(48_000, 2, SAMPLES);
    let mut stream = ogg::vorbis(&headers, &packets, 1);
    // The first audio page, after the 58 bytes of the first and the
    // header and segment table of the second, of two segments.
    let second = 27 + 2 + headers.comment.len() + headers.setup.len();
    stream[58 + second + 22] ^= 1;
    let decoded = ffmpeg::decode_ogg(&stream);
    assert!(
        decoded.diagnostics.contains("CRC mismatch"),
        "ffmpeg reported: {:?}",
        decoded.diagnostics
    );
}

/// Guards the oracle itself: if ffmpeg stopped reporting a setup header it
/// cannot read, a wrong field would pass the tests above unnoticed. Here
/// the codebook's sync pattern is broken.
#[test]
fn ffmpeg_reports_a_broken_setup_header() {
    let (mut headers, packets, _) = encode_silence(48_000, 1, SAMPLES);
    // After the packet type, "vorbis" and the codebook count.
    headers.setup[8] ^= 1;
    let decoded = ffmpeg::decode_ogg(&ogg::vorbis(&headers, &packets, 1));
    assert!(
        decoded.diagnostics.contains("Codebook setup data corrupt"),
        "ffmpeg reported: {:?}",
        decoded.diagnostics
    );
    assert!(decoded.samples.is_empty());
}

/// Guards the oracle, as above, at the setup header's last bit: ffmpeg
/// reports the header with its framing flag unset.
#[test]
fn ffmpeg_reports_the_setup_headers_framing_flag_unset() {
    let (mut headers, packets, _) = encode_silence(48_000, 1, SAMPLES);
    *headers.setup.last_mut().unwrap() = 0;
    let decoded = ffmpeg::decode_ogg(&ogg::vorbis(&headers, &packets, 1));
    assert!(
        decoded.diagnostics.contains("Invalid Setup header"),
        "ffmpeg reported: {:?}",
        decoded.diagnostics
    );
    assert!(decoded.samples.is_empty());
}
