// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The header's `private_bit`, `copyright` and `original/copy`, as mpg123
//! reads them.

use chelys::mp3::internals::header::{Bitrate, Flags, Mode, SamplingFrequency};
use chelys::mp3::internals::rate::Rate;
use chelys::mp3::internals::{StreamConfig, encode_silence};
use chelys_conformance::{DECODERS, Decoder, mpg123};

/// In every combination, in one channel and two and in both versions,
/// mpg123 reads each flag as it was set, and every decoder decodes the
/// stream without complaint.
#[test]
fn mpg123_reads_the_flags() {
    let yes = |set| if set { "Yes" } else { "No" };
    for (fs, mode) in [
        (SamplingFrequency::Hz44100, Mode::Stereo),
        (SamplingFrequency::Hz24000, Mode::SingleChannel),
    ] {
        for bits in 0..8 {
            let flags = Flags {
                private: bits & 1 != 0,
                copyright: bits & 2 != 0,
                original: bits & 4 != 0,
            };
            let config = StreamConfig {
                rate: Rate::Constant(Bitrate::from_kbps(fs.version(), 64).unwrap()),
                sampling_frequency: fs,
                mode,
                crc_protected: bits % 3 == 0,
                flags,
            };
            let stream = encode_silence(&config, 10);
            let described = mpg123::describe_header(&stream);
            let expected = [
                format!("copyright: {}", yes(flags.copyright)),
                format!("original: {}", yes(flags.original)),
                format!("Extension value: {}", u8::from(flags.private)),
            ];
            for part in expected {
                assert!(described.contains(&part), "{config:?}: {described}");
            }
            for Decoder { name, decode, .. } in DECODERS {
                assert_eq!(decode(&stream).diagnostics, "", "{name}, {config:?}");
            }
        }
    }
}
