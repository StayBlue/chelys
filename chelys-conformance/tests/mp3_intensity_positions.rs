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

use chelys::mp3::header::{Bitrate, Mode, SamplingFrequency};
use chelys::mp3::scalefactor_bands::{self, LONG_BANDS};
use chelys::mp3::scalefactors::ZeroPart;
use chelys::mp3::stereo::INTENSITY_SCALE;
use chelys::mp3::{BlockType, GRANULE_LINES, QuantizedGranule, StreamConfig, encode_granules};
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
fn right_channel(intensity: Option<(usize, [u8; LONG_BANDS])>) -> QuantizedGranule {
    let mut right = QuantizedGranule::new(0, [0; GRANULE_LINES], BlockType::Normal, FS);
    if let Some((bound, positions)) = intensity {
        right.scalefactors.values_mut()[bound..].copy_from_slice(&positions[bound..]);
        right.scalefactors.intensity = Some(ZeroPart { start: 0, bound });
    }
    right
}

/// A stream of the left channel's single line in long scalefactor band
/// `band`, in `mode`, with the right channel of [`right_channel`].
fn stream(band: usize, mode: Mode, intensity: Option<(usize, [u8; LONG_BANDS])>) -> Vec<u8> {
    let line = scalefactor_bands::long_bands(FS)[band].start + 1;
    let mut values = [0; GRANULE_LINES];
    values[line] = 1;
    let left = QuantizedGranule::new(200, values, BlockType::Normal, FS);
    let right = right_channel(intensity);
    let config = StreamConfig {
        bitrate: Bitrate::from_kbps(FS.version(), 160).unwrap(),
        sampling_frequency: FS,
        mode,
        crc_protected: false,
    };
    encode_granules(&config, &vec![vec![left, right]; FRAMES])
}

/// The `scalefac_compress` of the right channel's scalefactors in a stream
/// of [`stream`], with `int_scalefac_compress` below 180, whose `slen` are
/// `[isc / 36, (isc % 36) / 6, (isc % 36) % 6, 0]`.
fn scalefac_compress(slen: [u16; 3]) -> u16 {
    2 * (36 * slen[0] + 6 * slen[1] + slen[2]) + INTENSITY_SCALE
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
/// scalefactor of `band`, in intensity stereo from `bound` with `positions`.
fn readings(band: usize, bound: usize, positions: [u8; LONG_BANDS], sent: u8) -> Vec<Reading> {
    let plain = stream(band, Mode::Stereo, None);
    let joint = Mode::JointStereo {
        intensity_stereo: true,
        ms_stereo: false,
    };
    let intensity = stream(band, joint, Some((bound, positions)));
    let position = Reading::Position(sent);
    let (k_l, k_r) = match sent {
        0 => (1.0, 1.0),
        p if p % 2 == 1 => (I0.powi(i32::from(p.div_ceil(2))), 1.0),
        p => (1.0, I0.powi(i32::from(p / 2))),
    };
    DECODERS
        .iter()
        .map(|Decoder { name, decode }| {
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
                panic!("{name}, band {band}, value {sent}: {left:.4} and {right:.4}")
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
            readings(band, bound, positions, mark),
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
        let compress = right_channel(Some((bound, positions)))
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
            readings(band, bound, positions, position),
            expected,
            "position {position}"
        );
    }
}
