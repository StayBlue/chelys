// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Intensity stereo in short blocks, as ffmpeg, mpg123 and LAME decode it.
//!
//! Each stream codes short blocks in every granule of both channels, in
//! every frame, in intensity stereo without MS stereo. The left channel
//! sends a single line; the right channel sends at most a single line too
//! quiet to matter, which ends the nonzero part of its window, and its
//! scalefactors send intensity stereo positions or mark bands as not coded
//! in intensity stereo. A band read as a position `is_pos` decodes the line
//! in each channel at the share the position gives it; a band read as not
//! coded in intensity stereo decodes it in the left channel alone. The level
//! of each channel, against that of the same line coded in left/right
//! stereo, tells the two apart.

use chelys::mp3::internals::header::{Bitrate, Flags, Mode, SamplingFrequency, Version};
use chelys::mp3::internals::rate::StreamRate;
use chelys::mp3::internals::scalefactor_bands::{self, SHORT_BANDS, SHORT_WINDOWS};
use chelys::mp3::internals::scalefactors::ZeroPart;
use chelys::mp3::internals::{
    BlockType, GRANULE_LINES, QuantizedGranule, StreamConfig, encode_granules,
};
use chelys_conformance::{DECODERS, Decoded, Decoder};

const FRAMES: usize = 40;

/// The scalefactors of a short block, `scalefac_s[sfb][window]` at
/// `3 * sfb + window`.
const SCALEFACTORS: usize = SHORT_BANDS * SHORT_WINDOWS;

/// `i0` for `intensity_scale` 1, which the encoder always sends at the lower
/// sampling frequencies.
const I0: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// How a decoder reads a band of the right channel's zero part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reading {
    /// As intensity stereo position `is_pos`.
    Position(u8),
    /// As not coded in intensity stereo.
    NotIntensity,
}

/// The factors by which a decoder multiplies the lines sent in the left
/// channel for a band at `position`, to give the left and the right channel
/// (clause 2.4.3.4.9.3, and clause 2.4.3.2 of ISO/IEC 13818-3).
fn shares(fs: SamplingFrequency, position: u8) -> (f64, f64) {
    match fs.version() {
        Version::Mpeg1 => {
            let ratio = (f64::from(position) * std::f64::consts::PI / 12.0).tan();
            (ratio / (1.0 + ratio), 1.0 / (1.0 + ratio))
        }
        Version::Mpeg2 => match position {
            0 => (1.0, 1.0),
            p if p % 2 == 1 => (I0.powi(i32::from(p.div_ceil(2))), 1.0),
            p => (1.0, I0.powi(i32::from(p / 2))),
        },
    }
}

/// The positions a reading is told apart from: those whose shares differ
/// from those of a band not coded in intensity stereo, the left channel
/// alone, which position 6 gives in ISO/IEC 11172-3.
fn positions(fs: SamplingFrequency) -> std::ops::RangeInclusive<u8> {
    match fs.version() {
        Version::Mpeg1 => 0..=5,
        Version::Mpeg2 => 0..=14,
    }
}

/// The line, in the order sent, at `offset` in band `sfb` of `window`, or
/// in the lines above the highest band if `sfb` is [`SHORT_BANDS`].
fn line(fs: SamplingFrequency, sfb: usize, window: usize, offset: usize) -> usize {
    if sfb == SHORT_BANDS {
        scalefactor_bands::lines_above_bands(BlockType::Short, fs, window).start + offset
    } else {
        scalefactor_bands::coding_bands(BlockType::Short, fs)[SHORT_WINDOWS * sfb + window].start
            + offset
    }
}

/// The right channel's short blocks: zero but for a value of 1 at `quiet`,
/// which `global_gain` 50 leaves some 240 dB below full scale, with
/// `scalefactors` and the zero part `zero_part`.
fn right_channel(
    fs: SamplingFrequency,
    quiet: Option<usize>,
    scalefactors: [u8; SCALEFACTORS],
    zero_part: ZeroPart,
) -> QuantizedGranule {
    let mut values = [0; GRANULE_LINES];
    if let Some(line) = quiet {
        values[line] = 1;
    }
    let mut right = QuantizedGranule::new(50, values, BlockType::Short, fs);
    right
        .scalefactors
        .values_mut()
        .copy_from_slice(&scalefactors);
    right.scalefactors.intensity = Some(zero_part);
    right
}

/// A stream of short blocks, the left channel's single line at `left_line`
/// in the order sent, with the right channel in intensity stereo if given,
/// or else silent in left/right stereo.
fn stream(fs: SamplingFrequency, left_line: usize, right: Option<&QuantizedGranule>) -> Vec<u8> {
    let mut values = [0; GRANULE_LINES];
    values[left_line] = 1;
    let left = QuantizedGranule::new(200, values, BlockType::Short, fs);
    let silent = QuantizedGranule::new(0, [0; GRANULE_LINES], BlockType::Short, fs);
    let mode = match right {
        Some(_) => Mode::JointStereo {
            intensity_stereo: true,
            ms_stereo: false,
        },
        None => Mode::Stereo,
    };
    let config = StreamConfig {
        rate: StreamRate::Constant(Bitrate::from_kbps(fs.version(), 160).unwrap()),
        sampling_frequency: fs,
        mode,
        crc_protected: false,
        flags: Flags::default(),
    };
    let granules = FRAMES * fs.version().granules();
    let granule = vec![left, right.unwrap_or(&silent).clone()];
    encode_granules(&config, &vec![granule; granules])
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

/// How each decoder reads the band of the left channel's line at
/// `left_line`, with the right channel `right`.
fn readings(fs: SamplingFrequency, left_line: usize, right: &QuantizedGranule) -> Vec<Reading> {
    let plain = stream(fs, left_line, None);
    let intensity = stream(fs, left_line, Some(right));
    DECODERS
        .iter()
        .map(|Decoder { name, decode, .. }| {
            let [reference, _] = levels(&decode(&plain));
            let decoded = decode(&intensity);
            assert_eq!(decoded.diagnostics, "", "{name}");
            let [left, right] = levels(&decoded).map(|level| level / reference);
            let near = |(l, r): (f64, f64)| (left - l).abs() < 0.02 && (right - r).abs() < 0.02;
            if near((1.0, 0.0)) {
                return Reading::NotIntensity;
            }
            positions(fs)
                .find(|&position| near(shares(fs, position)))
                .map(Reading::Position)
                .unwrap_or_else(|| {
                    panic!("{name}, {fs:?}, line {left_line}: {left:.4} and {right:.4}")
                })
        })
        .collect()
}

/// The right channel's nonzero part ends in band 6 of its third window, and
/// every scalefactor sends `position`; no band is marked. A line in band 3
/// of the first window lies in that window's zero part, and is decoded in
/// intensity stereo; in band 3 of the third window it lies in that window's
/// nonzero part, and is not. Were the zero part found once for the granule,
/// from the last nonzero value in the order the values are sent, band 3 of
/// the first window would lie before it, and not be decoded in intensity
/// stereo either.
///
/// The standard finds the bound of each window apart at the lower sampling
/// frequencies (clause 2.4.3.2 of ISO/IEC 13818-3), and says that ISO/IEC
/// 11172-3 does so too. ffmpeg, mpg123 and LAME find it so at 44.1 and
/// 22.05 kHz.
///
/// This is the evidence for finding the zero part in each window, which the
/// intensity stereo positions mark as settled by decoding with ffmpeg,
/// mpg123 and LAME.
#[test]
fn short_blocks_find_the_bound_in_each_window() {
    for (fs, position) in [
        (SamplingFrequency::Hz44100, 2),
        (SamplingFrequency::Hz22050, 4),
    ] {
        let zero_part = ZeroPart {
            start: [0, 0, 7],
            bound: 0,
        };
        let quiet = line(fs, 6, 2, 0);
        let right = right_channel(fs, Some(quiet), [position; SCALEFACTORS], zero_part);
        for (window, expected) in [(0, Reading::Position(position)), (2, Reading::NotIntensity)] {
            assert_eq!(
                readings(fs, line(fs, 3, window, 1), &right),
                vec![expected; DECODERS.len()],
                "{fs:?}, window {window}"
            );
        }
    }
}

/// The right channel's nonzero part ends in band 2 of its first window, and
/// the bound is band 8 in every window: bands 3 to 7 of the first window
/// and 0 to 7 of the others are marked with 7, the illegal position of
/// ISO/IEC 11172-3, and bands 8 to 11 send position 2. A line in band 5 of
/// the second window is not decoded in intensity stereo, in band 9 it is,
/// and in band 1 of the first window, in the nonzero part, it is not.
///
/// This is the evidence, with `short_blocks_find_the_bound_in_each_window`,
/// for marking the bands of each window apart, which the intensity stereo
/// positions mark as settled by decoding with ffmpeg, mpg123 and LAME.
#[test]
fn short_blocks_mark_bands_in_each_window() {
    let fs = SamplingFrequency::Hz44100;
    let zero_part = ZeroPart {
        start: [3, 0, 0],
        bound: 8,
    };
    let mut scalefactors = [0; SCALEFACTORS];
    scalefactors[SHORT_WINDOWS * 8..].fill(2);
    let right = right_channel(fs, Some(line(fs, 2, 0, 0)), scalefactors, zero_part);
    let cases = [
        ((5, 1), Reading::NotIntensity),
        ((9, 1), Reading::Position(2)),
        ((1, 0), Reading::NotIntensity),
    ];
    for ((sfb, window), expected) in cases {
        assert_eq!(
            readings(fs, line(fs, sfb, window, 1), &right),
            vec![expected; DECODERS.len()],
            "band {sfb}, window {window}"
        );
    }
}

/// Every band of the right channel lies in its zero part. Band 11, the
/// highest short band, sends a different position in each window, and the
/// others position 0. A line in band 11 of a window, or above it, decodes
/// at the position of band 11 of that window.
///
/// The standard does not say how a decoder treats the lines above the
/// highest band in intensity stereo. ffmpeg, mpg123 and LAME give them the
/// position of that window's highest band, at 44.1 and 22.05 kHz.
///
/// This is the evidence for the position of the lines above the highest
/// band of short blocks, which the intensity stereo positions mark as
/// settled by decoding with ffmpeg, mpg123 and LAME.
#[test]
fn lines_above_the_short_bands_take_the_position_of_the_highest_band() {
    for (fs, highest) in [
        (SamplingFrequency::Hz44100, [1, 2, 5]),
        (SamplingFrequency::Hz22050, [3, 4, 9]),
    ] {
        let zero_part = ZeroPart {
            start: [0; SHORT_WINDOWS],
            bound: 0,
        };
        let mut scalefactors = [0; SCALEFACTORS];
        scalefactors[SHORT_WINDOWS * 11..].copy_from_slice(&highest);
        let right = right_channel(fs, None, scalefactors, zero_part);
        for (window, &position) in highest.iter().enumerate() {
            for sfb in [11, SHORT_BANDS] {
                assert_eq!(
                    readings(fs, line(fs, sfb, window, 1), &right),
                    vec![Reading::Position(position); DECODERS.len()],
                    "{fs:?}, band {sfb}, window {window}"
                );
            }
        }
    }
}

/// Every partition of all three ranges of `int_scalefac_compress` in short
/// blocks (clause 2.4.3.2 of ISO/IEC 13818-3), with every band of every
/// window in intensity stereo, at each of the lower sampling frequencies.
/// The partitions count the scalefactors in the order sent, three to a band.
/// In each case the positions of each partition are those of one range, and
/// that range sends them in the fewest bits:
///
/// - below 180, partitions of 12, 12 and 12 at positions 4, 2 and 5 (bands
///   0 to 3, 4 to 7 and 8 to 11), `slen` 3, 2 and 3: the other ranges send
///   no bits in bands 10 and 11;
/// - 180 to 243, partitions of 12, 9, 9 and 6 at positions 4, 2, 5 and 0
///   (bands 0 to 3, 4 to 6, 7 to 9 and 10 and 11), `slen` 3, 2, 3 and 0: 81
///   bits, against 108 below 180, and the range from 244 cannot send 5 in
///   its 2-bit second partition;
/// - 244 to 255, partitions of 15, 12 and 9 at positions 4, 2 and 0 (bands
///   0 to 4, 5 to 8 and 9 to 11), `slen` 3, 2 and 0: 69 bits, against 81
///   from 180 and 96 below 180.
///
/// A band in each partition, in its second window, decodes at its position
/// in ffmpeg, mpg123 and LAME.
#[test]
fn every_range_of_int_scalefac_compress_sends_positions_in_short_blocks() {
    // The positions of the bands of each partition, and the
    // scalefac_compress expected: 2 * int_scalefac_compress + 1, with
    // intensity_scale 1.
    let ranges: [(&[(std::ops::Range<usize>, u8)], u16); 3] = [
        (
            &[(0..4, 4), (4..8, 2), (8..12, 5)],
            2 * (36 * 3 + 6 * 2 + 3) + 1,
        ),
        (
            &[(0..4, 4), (4..7, 2), (7..10, 5), (10..12, 0)],
            2 * (180 + 16 * 3 + 4 * 2 + 3) + 1,
        ),
        (
            &[(0..5, 4), (5..9, 2), (9..12, 0)],
            2 * (244 + 3 * 3 + 2) + 1,
        ),
    ];
    let zero_part = ZeroPart {
        start: [0; SHORT_WINDOWS],
        bound: 0,
    };
    for fs in [
        SamplingFrequency::Hz16000,
        SamplingFrequency::Hz22050,
        SamplingFrequency::Hz24000,
    ] {
        for &(partitions, expected) in &ranges {
            let mut scalefactors = [0; SCALEFACTORS];
            for (bands, position) in partitions {
                scalefactors[SHORT_WINDOWS * bands.start..SHORT_WINDOWS * bands.end]
                    .fill(*position);
            }
            let right = right_channel(fs, None, scalefactors, zero_part);
            let compress = right.scalefactors.compress().unwrap();
            assert_eq!(compress, expected, "{fs:?}, {partitions:?}");
            for (bands, position) in partitions {
                let sfb = bands.start + 1;
                assert_eq!(
                    readings(fs, line(fs, sfb, 1, 1), &right),
                    vec![Reading::Position(*position); DECODERS.len()],
                    "{fs:?}, band {sfb}"
                );
            }
        }
    }
}
