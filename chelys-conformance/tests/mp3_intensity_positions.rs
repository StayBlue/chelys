// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Intensity stereo positions at the lower sampling frequencies, as ffmpeg,
//! mpg123 and LAME decode them.
//!
//! Each stream codes a single line of the left channel in one long
//! scalefactor band, in every frame, with the right channel all zero, in
//! intensity stereo from a bound, without MS stereo. The right channel's
//! scalefactors send intensity stereo positions from the bound and mark the
//! bands below it as not coded in intensity stereo, with the largest value
//! of their field (clause 2.4.3.2 of ISO/IEC 13818-3). A band read as a
//! position `is_pos` decodes the line times `k_l` in the left channel and
//! `k_r` in the right; a band read as not coded in intensity stereo decodes
//! it in the left channel alone. The level of each channel, against that of
//! the same line coded in left/right stereo, tells the two apart.
//!
//! This is the evidence for sending no marks at the lower sampling
//! frequencies, for keeping positions below 16 and off 7, and for reading a
//! position sent in no bits as 0, which `quantize_frame`, the intensity
//! stereo positions and the scalefactors mark as settled by decoding with
//! ffmpeg, mpg123 and LAME.

use std::ops::Range;

use chelys::mp3::internals::header::{Bitrate, Flags, Mode, SamplingFrequency};
use chelys::mp3::internals::rate::StreamRate;
use chelys::mp3::internals::scalefactor_bands::{self, LONG_BANDS, SHORT_WINDOWS};
use chelys::mp3::internals::scalefactors::ZeroPart;
use chelys::mp3::internals::stereo::intensity;
use chelys::mp3::internals::{
    BlockType, GRANULE_LINES, QuantizedGranule, StreamConfig, encode_granules,
};
use chelys_conformance::{DECODERS, Decoded, Decoder};

const FS: SamplingFrequency = SamplingFrequency::Hz22050;
const FRAMES: usize = 40;

/// `i0` for `intensity_scale` 1.
const I0: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// How a decoder reads a band of the right channel's zero part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reading {
    /// As intensity stereo position `is_pos`.
    Position(u8),
    /// As not coded in intensity stereo: an illegal position.
    Illegal,
}

/// The right channel, all zero; in intensity stereo from `bound` if given,
/// with `positions` from it and every band below it marked.
fn right_channel(
    fs: SamplingFrequency,
    intensity: Option<(usize, [u8; LONG_BANDS])>,
) -> QuantizedGranule {
    let mut right = QuantizedGranule::new(0, [0; GRANULE_LINES], BlockType::Normal, fs);
    if let Some((bound, positions)) = intensity {
        right.scalefactors.values_mut()[bound..].copy_from_slice(&positions[bound..]);
        right.scalefactors.intensity = Some(ZeroPart {
            start: [0; SHORT_WINDOWS],
            bound,
        });
    }
    right
}

/// A stream of the left channel's single line in long scalefactor band
/// `band`, in `mode`, with the right channel of [`right_channel`].
fn stream(
    fs: SamplingFrequency,
    band: usize,
    mode: Mode,
    intensity: Option<(usize, [u8; LONG_BANDS])>,
) -> Vec<u8> {
    let line = scalefactor_bands::long_bands(fs)[band].start + 1;
    let mut values = [0; GRANULE_LINES];
    values[line] = 1;
    let left = QuantizedGranule::new(200, values, BlockType::Normal, fs);
    let right = right_channel(fs, intensity);
    let config = StreamConfig {
        rate: StreamRate::Constant(Bitrate::from_kbps(fs.version(), 160).unwrap()),
        sampling_frequency: fs,
        mode,
        crc_protected: false,
        flags: Flags::default(),
    };
    encode_granules(&config, &vec![vec![left, right]; FRAMES])
}

/// The `scalefac_compress` of the right channel's scalefactors in a stream
/// of [`stream`], with `int_scalefac_compress` below 180, whose `slen` are
/// `[isc / 36, (isc % 36) / 6, (isc % 36) % 6, 0]`.
fn scalefac_compress(slen: [u16; 3]) -> u16 {
    2 * (36 * slen[0] + 6 * slen[1] + slen[2]) + intensity::SCALE
}

/// The root mean square of each channel of `decoded`, over its middle frames.
fn levels(decoded: &Decoded) -> [f64; 2] {
    let frames = decoded.samples.len() / 2;
    let range = frames / 4..frames * 3 / 4;
    [0, 1].map(|channel| {
        let energy: f64 = range
            .clone()
            .map(|t| f64::from(decoded.samples[2 * t + channel]).powi(2))
            .sum();
        (energy / range.len() as f64).sqrt()
    })
}

/// How each decoder reads the value `sent` in the right channel's
/// scalefactor of `band`, in intensity stereo from `bound` with `positions`,
/// at `fs`.
fn readings(
    fs: SamplingFrequency,
    band: usize,
    bound: usize,
    positions: [u8; LONG_BANDS],
    sent: u8,
) -> Vec<Reading> {
    let plain = stream(fs, band, Mode::Stereo, None);
    let joint = Mode::JointStereo {
        intensity_stereo: true,
        ms_stereo: false,
    };
    let intensity = stream(fs, band, joint, Some((bound, positions)));
    let position = Reading::Position(sent);
    let (k_l, k_r) = match sent {
        0 => (1.0, 1.0),
        p if p % 2 == 1 => (I0.powi(i32::from(p.div_ceil(2))), 1.0),
        p => (1.0, I0.powi(i32::from(p / 2))),
    };
    DECODERS
        .iter()
        .map(|Decoder { name, decode, .. }| {
            let [reference, _] = levels(&decode(&plain));
            let decoded = decode(&intensity);
            assert_eq!(decoded.diagnostics, "", "{name}");
            let [left, right] = levels(&decoded).map(|level| level / reference);
            let near = |(l, r): (f64, f64)| (left - l).abs() < 0.02 && (right - r).abs() < 0.02;
            if near((k_l, k_r)) {
                position
            } else if near((1.0, 0.0)) {
                Reading::Illegal
            } else {
                panic!("{name}, {fs:?}, band {band}, value {sent}: {left:.4} and {right:.4}")
            }
        })
        .collect()
}

/// A band below the bound marked as not coded in intensity stereo with the
/// largest value of a field of 1 to 5 bits: 1, 3, 7, 15 or 31. The field is
/// that of the second partition of `int_scalefac_compress` below 180, bands
/// 7 to 13, which the bound, band 10, divides; the positions of bands 10 to
/// 13, 0, 2, 4, 8 or 16, set its width. (A mark of 1 bit takes the second
/// partition from 180, bands 6 to 11, which is cheaper.) The decoders that
/// read a mark as a position show by its level that it was sent.
///
/// The standard reads every mark as an illegal position. ffmpeg reads only
/// 31 so, and the others as positions; mpg123 and LAME read only 7 so, and
/// the others as positions. No mark is read as one by all three.
#[test]
fn marks_are_read_as_illegal_in_different_fields() {
    let (bound, band) = (10, 8);
    for (width, position) in (1..=5u8).zip([0, 2, 4, 8, 16]) {
        let mut positions = [0; LONG_BANDS];
        positions[10..14].fill(position);
        let mark = (1 << width) - 1;
        let illegal_in = |illegal: bool| {
            if illegal {
                Reading::Illegal
            } else {
                Reading::Position(mark)
            }
        };
        let expected = vec![
            illegal_in(mark == 31),
            illegal_in(mark == 7),
            illegal_in(mark == 7),
        ];
        assert_eq!(
            readings(FS, band, bound, positions, mark),
            expected,
            "mark {mark}"
        );
    }
}

/// Positions 0, 5, 7, 13, 14, 15 and 16 in bands 14 to 20, the third
/// partition of `int_scalefac_compress` below 180, from the bound, band 14:
/// each in the narrowest field whose largest value it is not, and 0 in a
/// field of no bits.
///
/// The standard reads them all as positions. ffmpeg reads 16 as illegal,
/// and mpg123 and LAME read 7 as illegal; all three read 0, sent in no
/// bits, 5, 13, 14 and 15 as positions.
#[test]
fn positions_7_and_16_are_read_as_illegal() {
    let (bound, band) = (14, 15);
    let sent = [(0, 0), (5, 3), (7, 4), (13, 4), (14, 4), (15, 5), (16, 5)];
    for (position, width) in sent {
        let mut positions = [0; LONG_BANDS];
        positions[bound..].fill(position);
        let compress = right_channel(FS, Some((bound, positions)))
            .scalefactors
            .compress()
            .unwrap();
        assert_eq!(compress, scalefac_compress([1, 1, width]), "{position}");
        let read = |illegal: bool| {
            if illegal {
                Reading::Illegal
            } else {
                Reading::Position(position)
            }
        };
        let expected = vec![
            read(position == 16),
            read(position == 7),
            read(position == 7),
        ];
        assert_eq!(
            readings(FS, band, bound, positions, position),
            expected,
            "position {position}"
        );
    }
}

/// Every partition of all three ranges of `int_scalefac_compress` (clause
/// 2.4.3.2 of ISO/IEC 13818-3), with every band in intensity stereo, from
/// bound 0, at each of the lower sampling frequencies. In each case the
/// positions of each partition are those of one range, and that range
/// sends them in the fewest bits:
///
/// - below 180, partitions of 7, 7 and 7 bands at positions 4, 2 and 5,
///   `slen` 3, 2 and 3: the other ranges send no bits in bands 18 to 20;
/// - 180 to 243, partitions of 6, 6, 6 and 3 bands at positions 4, 2, 5 and
///   0, `slen` 3, 2, 3 and 0: 48 bits, against 63 below 180, and the range
///   from 244 cannot send 5 in its 2-bit second partition;
/// - 244 to 255, partitions of 8, 8 and 5 bands at positions 4, 2 and 0,
///   `slen` 3, 2 and 0: 40 bits, against 48 from 180 and 56 below 180.
///
/// A band in each partition decodes at its position in ffmpeg, mpg123 and
/// LAME.
#[test]
fn every_range_of_int_scalefac_compress_sends_positions() {
    struct CompressRange {
        /// The bands of each partition, and their position.
        partitions: &'static [(Range<usize>, u8)],
        /// The scalefac_compress expected: 2 * int_scalefac_compress + 1.
        compress: u16,
    }
    let ranges = [
        CompressRange {
            partitions: &[(0..7, 4), (7..14, 2), (14..21, 5)],
            compress: scalefac_compress([3, 2, 3]),
        },
        CompressRange {
            partitions: &[(0..6, 4), (6..12, 2), (12..18, 5), (18..21, 0)],
            compress: 2 * (180 + 16 * 3 + 4 * 2 + 3) + intensity::SCALE,
        },
        CompressRange {
            partitions: &[(0..8, 4), (8..16, 2), (16..21, 0)],
            compress: 2 * (244 + 3 * 3 + 2) + intensity::SCALE,
        },
    ];
    for fs in [
        SamplingFrequency::Hz16000,
        SamplingFrequency::Hz22050,
        SamplingFrequency::Hz24000,
    ] {
        for &CompressRange {
            partitions,
            compress: expected,
        } in &ranges
        {
            let mut positions = [0; LONG_BANDS];
            for (bands, position) in partitions {
                positions[bands.clone()].fill(*position);
            }
            let compress = right_channel(fs, Some((0, positions)))
                .scalefactors
                .compress()
                .unwrap();
            assert_eq!(compress, expected, "{fs:?}, {partitions:?}");
            for (bands, position) in partitions {
                let band = bands.start + 1;
                assert_eq!(
                    readings(fs, band, 0, positions, *position),
                    vec![Reading::Position(*position); DECODERS.len()],
                    "{fs:?}, band {band}"
                );
            }
        }
    }
}
