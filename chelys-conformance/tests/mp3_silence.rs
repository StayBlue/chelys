// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

use chelys::mp3::header::{Bitrate, Mode, SamplingFrequency, Version};
use chelys::mp3::rate::Rate;
use chelys::mp3::{StreamConfig, encode_silence};
use chelys_conformance::ffmpeg::{self, StreamInfo};

#[test]
fn ffmpeg_decodes_silence_in_every_configuration() {
    let modes = [
        Mode::SingleChannel,
        Mode::Stereo,
        Mode::DualChannel,
        Mode::JointStereo {
            intensity_stereo: false,
            ms_stereo: true,
        },
    ];
    let frames = 50;
    for sampling_frequency in SamplingFrequency::STANDARD {
        for bitrate in Bitrate::all(sampling_frequency.version()) {
            for mode in modes {
                for crc_protected in [false, true] {
                    let config = StreamConfig {
                        rate: Rate::Constant(bitrate),
                        sampling_frequency,
                        mode,
                        crc_protected,
                    };
                    let stream = encode_silence(&config, frames);

                    let expected = StreamInfo {
                        codec_name: "mp3".to_owned(),
                        sample_rate: sampling_frequency.hz(),
                        channels: mode.channels(),
                    };
                    assert_eq!(ffmpeg::probe_mp3(&stream), expected, "{config:?}");

                    let decoded = ffmpeg::decode_mp3(&stream);
                    assert_eq!(decoded.diagnostics, "", "{config:?}");
                    assert_eq!(
                        decoded.samples.len(),
                        frames * sampling_frequency.version().samples_per_frame() * mode.channels(),
                        "{config:?}"
                    );
                    assert!(decoded.samples.iter().all(|&s| s == 0.0), "{config:?}");
                }
            }
        }
    }
}

/// Guards the oracle itself: if ffmpeg stopped checking CRCs, a wrong
/// `crc_check` would pass the test above unnoticed. In both versions, and in
/// one and two channels, whose side information the error check covers in
/// different lengths (clause 2.4.3.1, and clause 2.4.2.4 of ISO/IEC
/// 13818-3).
#[test]
fn ffmpeg_reports_a_corrupted_crc() {
    for (version, kbps, sampling_frequency) in [
        (Version::Mpeg1, 128, SamplingFrequency::Hz44100),
        (Version::Mpeg2, 64, SamplingFrequency::Hz22050),
    ] {
        for mode in [Mode::SingleChannel, Mode::Stereo] {
            let config = StreamConfig {
                rate: Rate::Constant(Bitrate::from_kbps(version, kbps).unwrap()),
                sampling_frequency,
                mode,
                crc_protected: true,
            };
            let mut stream = encode_silence(&config, 5);
            stream[5] ^= 1; // low byte of the first frame's crc_check
            let decoded = ffmpeg::decode_mp3(&stream);
            assert!(
                decoded.diagnostics.contains("CRC mismatch"),
                "{config:?}: ffmpeg reported: {:?}",
                decoded.diagnostics
            );
        }
    }
}
