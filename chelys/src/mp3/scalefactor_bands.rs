// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Layer III scalefactor bands (Table B.8, Table B.2 of ISO/IEC 13818-3 for
//! the lower sampling frequencies, and US patent 6,185,539 B1 for MPEG-2.5).
//!
//! The tables of the standards are transcribed as printed: `(scalefactor
//! band, width of band, index of start, index of end)`, with the end index
//! inclusive. The patent prints widths alone, and they are transcribed so.
//! The consistency of every table is checked at compile time.

use std::ops::Range;

use super::header::SamplingFrequency;
use super::{BlockType, GRANULE_LINES};

/// The frequency lines `start..start + width` of one scalefactor band. Short
/// block indices count the 192 lines of a single short window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Band {
    pub start: usize,
    pub width: usize,
}

impl Band {
    /// One past the last line of the band.
    pub fn end(self) -> usize {
        self.start + self.width
    }
}

pub const LONG_BANDS: usize = 21;
pub const SHORT_BANDS: usize = 12;
/// The short windows of a short block.
pub const SHORT_WINDOWS: usize = 3;
/// The lines of one short window.
const SHORT_WINDOW_LINES: usize = GRANULE_LINES / SHORT_WINDOWS;

/// The bands for long windows (block types 0, 1 and 3).
///
/// Spec gap: US 6,185,539 B1 gives tables for MPEG-2.5 at 8 kHz alone. At
/// 11.025 and 12 kHz, as the patent leaves the bitstream "in all other
/// respects" as ISO/IEC 13818-3 has it (column 3), the bands are those of
/// the frequency of ISO/IEC 13818-3 that has the same `sampling_frequency`
/// code (see `SamplingFrequency::code`), 22.05 and 24 kHz. The same holds
/// for [`short_bands`].
pub fn long_bands(sampling_frequency: SamplingFrequency) -> &'static [Band; LONG_BANDS] {
    match sampling_frequency {
        SamplingFrequency::Hz32000 => &LONG_32000,
        SamplingFrequency::Hz44100 => &LONG_44100,
        SamplingFrequency::Hz48000 => &LONG_48000,
        SamplingFrequency::Hz16000 => &LONG_16000,
        SamplingFrequency::Hz22050 => &LONG_22050,
        SamplingFrequency::Hz24000 => &LONG_24000,
        SamplingFrequency::Hz11025 => &LONG_22050,
        SamplingFrequency::Hz12000 => &LONG_24000,
        SamplingFrequency::Hz8000 => &LONG_8000,
    }
}

/// The bands for short windows (block type 2).
pub fn short_bands(sampling_frequency: SamplingFrequency) -> &'static [Band; SHORT_BANDS] {
    match sampling_frequency {
        SamplingFrequency::Hz32000 => &SHORT_32000,
        SamplingFrequency::Hz44100 => &SHORT_44100,
        SamplingFrequency::Hz48000 => &SHORT_48000,
        SamplingFrequency::Hz16000 => &SHORT_16000,
        SamplingFrequency::Hz22050 => &SHORT_22050,
        SamplingFrequency::Hz24000 => &SHORT_24000,
        SamplingFrequency::Hz11025 => &SHORT_22050,
        SamplingFrequency::Hz12000 => &SHORT_24000,
        SamplingFrequency::Hz8000 => &SHORT_8000,
    }
}

/// The lines of each scalefactor of a granule of `block_type`, in the order
/// in which both the scalefactors and the Huffman coded values are sent
/// (clauses 2.4.1.7 and 2.4.2.7): the long bands for block types 0, 1 and 3,
/// and for short blocks each short band three times, once per window, as
/// `3 * sfb + window`, a band's windows following each other. Lines past the
/// last range have no scalefactor.
pub fn coding_bands(block_type: BlockType, sampling_frequency: SamplingFrequency) -> Vec<Band> {
    if block_type != BlockType::Short {
        return long_bands(sampling_frequency).to_vec();
    }
    short_bands(sampling_frequency)
        .iter()
        .flat_map(|band| {
            (0..SHORT_WINDOWS).map(move |window| Band {
                start: SHORT_WINDOWS * band.start + window * band.width,
                width: band.width,
            })
        })
        .collect()
}

/// The scalefactors, in the order of [`coding_bands`], that the encoder keeps
/// at zero, and so never amplifies: those of the highest of the four
/// partitions of `scalefac_compress` (clause 2.4.3.2 of ISO/IEC 13818-3) at
/// MPEG-2.5's 8 kHz, and none otherwise.
///
/// Spec reading: US 6,185,539 B1 sets "the scale factors of the scale factor
/// bands of the highest region ... to zero without exception" (column 3) at
/// 8 kHz, the region being the last of 6-5-5-5 long or 9-9-9-9 short bands.
/// That is long bands 16 to 20 and short bands 9 to 11 (scalefactors 27 to
/// 35 in the order sent), which its table makes the top 100 lines of a long
/// block and the top 32 lines of each short window.
pub fn fixed_at_zero(block_type: BlockType, sampling_frequency: SamplingFrequency) -> Range<usize> {
    if sampling_frequency != SamplingFrequency::Hz8000 {
        return 0..0;
    }
    match block_type {
        BlockType::Short => 27..SHORT_BANDS * SHORT_WINDOWS,
        _ => 16..LONG_BANDS,
    }
}

/// The windows of a granule of `block_type`: three for short blocks, and one
/// for the others.
pub fn windows(block_type: BlockType) -> usize {
    match block_type {
        BlockType::Short => SHORT_WINDOWS,
        _ => 1,
    }
}

/// The lines of `window` of a granule of `block_type` that lie above its
/// highest scalefactor band, and so have no scalefactor, in the order they
/// are sent: the end of the granule for long blocks, and for short blocks
/// the window's share of the band [`short_coding_order`] sends last.
pub fn lines_above_bands(
    block_type: BlockType,
    sampling_frequency: SamplingFrequency,
    window: usize,
) -> Range<usize> {
    let windows = windows(block_type);
    assert!(window < windows, "window {window} of {windows}");
    let last = match block_type {
        BlockType::Short => short_bands(sampling_frequency)[SHORT_BANDS - 1].end(),
        _ => long_bands(sampling_frequency)[LONG_BANDS - 1].end(),
    };
    let width = GRANULE_LINES / windows - last;
    let start = windows * last + window * width;
    start..start + width
}

/// Reorders the lines of a short block from the subband order of the
/// filterbank (see
/// [`Filterbank::granule`](super::filterbank::Filterbank::granule)) into the
/// order of the bitstream: by scalefactor band, then window, then frequency
/// (clause 2.4.2.7). The decoder undoes this before its IMDCT (clause
/// 2.4.3.4.8).
///
/// Spec gap: Table B.8 ends with short band 11, and the standard does not
/// say how the lines above it, which have no scalefactor, are ordered. They
/// are sent as one more band up to line 192 of each window, window by
/// window. Settled by decoding with ffmpeg (black box): with a tone in those
/// lines, the conformance test
/// `short_blocks_round_trip_at_every_sampling_frequency` decodes at over
/// 45 dB SNR at every sampling frequency, the lower ones of ISO/IEC 13818-3
/// included, and at 2 to 12 dB with the lines interleaved across the windows
/// instead.
pub fn short_coding_order(
    xr: &[f64; GRANULE_LINES],
    sampling_frequency: SamplingFrequency,
) -> [f64; GRANULE_LINES] {
    let bands = short_bands(sampling_frequency);
    let last = bands[SHORT_BANDS - 1].end();
    let top = Band {
        start: last,
        width: SHORT_WINDOW_LINES - last,
    };
    let mut ordered = [0.0; GRANULE_LINES];
    let mut next = 0;
    for band in bands.iter().chain([&top]) {
        for window in 0..SHORT_WINDOWS {
            for line in band.start..band.end() {
                // Line `line` of a window lies in subband `line / 6`, at
                // `18 * subband + 6 * window + line % 6`.
                ordered[next] = xr[18 * (line / 6) + 6 * window + line % 6];
                next += 1;
            }
        }
    }
    ordered
}

/// Builds a band table from printed rows, checking that the bands are
/// numbered in order, start at line 0, follow each other without gaps, and
/// have widths consistent with their start and end indices.
const fn bands<const N: usize>(rows: [(usize, usize, usize, usize); N]) -> [Band; N] {
    let mut bands = [Band { start: 0, width: 0 }; N];
    let mut next_start = 0;
    let mut i = 0;
    while i < N {
        let (sfb, width, start, end) = rows[i];
        assert!(sfb == i, "band out of order");
        assert!(
            start == next_start,
            "band does not start where the last ended"
        );
        assert!(
            end + 1 - start == width,
            "width does not match start and end"
        );
        bands[i] = Band { start, width };
        next_start = end + 1;
        i += 1;
    }
    bands
}

/// Builds a band table, as [`bands`] does, from the printed rows of one of
/// the tables of Table B.2 of ISO/IEC 13818-3. Each ends with one more row,
/// `top`, that carries on to the last of `lines`.
///
/// Table B.2 has 22 long and 13 short bands, but the scalefactor of the last
/// "is set to a fixed value and is not transmitted", so 21 and 12 are sent.
/// The last row is kept apart as the lines with no scalefactor sent, like
/// the lines above Table B.8: it is checked, but not kept as a band.
///
/// Spec gap: the standard does not give the fixed value. The encoder leaves
/// these lines unamplified, as for a scalefactor of 0. Settled by decoding
/// with ffmpeg (black box): with a tone in those lines, the conformance test
/// `short_blocks_round_trip_at_every_sampling_frequency` decodes long and
/// short blocks at the lower sampling frequencies at over 45 dB SNR.
const fn bands_with_top<const N: usize>(
    rows: [(usize, usize, usize, usize); N],
    top: (usize, usize, usize, usize),
    lines: usize,
) -> [Band; N] {
    let bands = bands(rows);
    let (sfb, width, start, end) = top;
    assert!(sfb == N, "top row out of order");
    assert!(
        start == bands[N - 1].start + bands[N - 1].width,
        "top row does not start where the last band ended"
    );
    assert!(
        end + 1 - start == width,
        "width does not match start and end"
    );
    assert!(end + 1 == lines, "top row does not end the lines");
    bands
}

/// Builds a band table, as [`bands_with_top`] does, from the widths of the
/// bands alone, the last of which, `top`, carries on to the last of `lines`.
const fn bands_from_widths<const N: usize>(
    widths: [usize; N],
    top: usize,
    lines: usize,
) -> [Band; N] {
    let mut bands = [Band { start: 0, width: 0 }; N];
    let mut start = 0;
    let mut i = 0;
    while i < N {
        assert!(widths[i] > 0, "empty band");
        bands[i] = Band {
            start,
            width: widths[i],
        };
        start += widths[i];
        i += 1;
    }
    assert!(start + top == lines, "the bands do not end the lines");
    bands
}

// Table B.8a, 32 kHz.

const LONG_32000: [Band; LONG_BANDS] = bands([
    (0, 4, 0, 3),
    (1, 4, 4, 7),
    (2, 4, 8, 11),
    (3, 4, 12, 15),
    (4, 4, 16, 19),
    (5, 4, 20, 23),
    (6, 6, 24, 29),
    (7, 6, 30, 35),
    (8, 8, 36, 43),
    (9, 10, 44, 53),
    (10, 12, 54, 65),
    (11, 16, 66, 81),
    (12, 20, 82, 101),
    (13, 24, 102, 125),
    (14, 30, 126, 155),
    (15, 38, 156, 193),
    (16, 46, 194, 239),
    (17, 56, 240, 295),
    (18, 68, 296, 363),
    (19, 84, 364, 447),
    (20, 102, 448, 549),
]);

const SHORT_32000: [Band; SHORT_BANDS] = bands([
    (0, 4, 0, 3),
    (1, 4, 4, 7),
    (2, 4, 8, 11),
    (3, 4, 12, 15),
    (4, 6, 16, 21),
    (5, 8, 22, 29),
    (6, 12, 30, 41),
    (7, 16, 42, 57),
    (8, 20, 58, 77),
    (9, 26, 78, 103),
    (10, 34, 104, 137),
    (11, 42, 138, 179),
]);

// Table B.8b, 44.1 kHz.

const LONG_44100: [Band; LONG_BANDS] = bands([
    (0, 4, 0, 3),
    (1, 4, 4, 7),
    (2, 4, 8, 11),
    (3, 4, 12, 15),
    (4, 4, 16, 19),
    (5, 4, 20, 23),
    (6, 6, 24, 29),
    (7, 6, 30, 35),
    (8, 8, 36, 43),
    (9, 8, 44, 51),
    (10, 10, 52, 61),
    (11, 12, 62, 73),
    (12, 16, 74, 89),
    (13, 20, 90, 109),
    (14, 24, 110, 133),
    (15, 28, 134, 161),
    (16, 34, 162, 195),
    (17, 42, 196, 237),
    (18, 50, 238, 287),
    (19, 54, 288, 341),
    (20, 76, 342, 417),
]);

const SHORT_44100: [Band; SHORT_BANDS] = bands([
    (0, 4, 0, 3),
    (1, 4, 4, 7),
    (2, 4, 8, 11),
    (3, 4, 12, 15),
    (4, 6, 16, 21),
    (5, 8, 22, 29),
    (6, 10, 30, 39),
    (7, 12, 40, 51),
    (8, 14, 52, 65),
    (9, 18, 66, 83),
    (10, 22, 84, 105),
    (11, 30, 106, 135),
]);

// Table B.8c, 48 kHz.

const LONG_48000: [Band; LONG_BANDS] = bands([
    (0, 4, 0, 3),
    (1, 4, 4, 7),
    (2, 4, 8, 11),
    (3, 4, 12, 15),
    (4, 4, 16, 19),
    (5, 4, 20, 23),
    (6, 6, 24, 29),
    (7, 6, 30, 35),
    (8, 6, 36, 41),
    (9, 8, 42, 49),
    (10, 10, 50, 59),
    (11, 12, 60, 71),
    (12, 16, 72, 87),
    (13, 18, 88, 105),
    (14, 22, 106, 127),
    (15, 28, 128, 155),
    (16, 34, 156, 189),
    (17, 40, 190, 229),
    (18, 46, 230, 275),
    (19, 54, 276, 329),
    (20, 54, 330, 383),
]);

const SHORT_48000: [Band; SHORT_BANDS] = bands([
    (0, 4, 0, 3),
    (1, 4, 4, 7),
    (2, 4, 8, 11),
    (3, 4, 12, 15),
    (4, 6, 16, 21),
    (5, 6, 22, 27),
    (6, 10, 28, 37),
    (7, 12, 38, 49),
    (8, 14, 50, 63),
    (9, 16, 64, 79),
    (10, 20, 80, 99),
    (11, 26, 100, 125),
]);

// Table B.2 of ISO/IEC 13818-3, 16 kHz.

/// As printed, the same as the table for 22.05 kHz.
const LONG_16000: [Band; LONG_BANDS] = bands_with_top(
    [
        (0, 6, 0, 5),
        (1, 6, 6, 11),
        (2, 6, 12, 17),
        (3, 6, 18, 23),
        (4, 6, 24, 29),
        (5, 6, 30, 35),
        (6, 8, 36, 43),
        (7, 10, 44, 53),
        (8, 12, 54, 65),
        (9, 14, 66, 79),
        (10, 16, 80, 95),
        (11, 20, 96, 115),
        (12, 24, 116, 139),
        (13, 28, 140, 167),
        (14, 32, 168, 199),
        (15, 38, 200, 237),
        (16, 46, 238, 283),
        (17, 52, 284, 335),
        (18, 60, 336, 395),
        (19, 68, 396, 463),
        (20, 58, 464, 521),
    ],
    (21, 54, 522, 575),
    GRANULE_LINES,
);

const SHORT_16000: [Band; SHORT_BANDS] = bands_with_top(
    [
        (0, 4, 0, 3),
        (1, 4, 4, 7),
        (2, 4, 8, 11),
        (3, 6, 12, 17),
        (4, 8, 18, 25),
        (5, 10, 26, 35),
        (6, 12, 36, 47),
        (7, 14, 48, 61),
        (8, 18, 62, 79),
        (9, 24, 80, 103),
        (10, 30, 104, 133),
        (11, 40, 134, 173),
    ],
    (12, 18, 174, 191),
    SHORT_WINDOW_LINES,
);

// Table B.2 of ISO/IEC 13818-3, 22.05 kHz.

const LONG_22050: [Band; LONG_BANDS] = bands_with_top(
    [
        (0, 6, 0, 5),
        (1, 6, 6, 11),
        (2, 6, 12, 17),
        (3, 6, 18, 23),
        (4, 6, 24, 29),
        (5, 6, 30, 35),
        (6, 8, 36, 43),
        (7, 10, 44, 53),
        (8, 12, 54, 65),
        (9, 14, 66, 79),
        (10, 16, 80, 95),
        (11, 20, 96, 115),
        (12, 24, 116, 139),
        (13, 28, 140, 167),
        (14, 32, 168, 199),
        (15, 38, 200, 237),
        (16, 46, 238, 283),
        (17, 52, 284, 335),
        (18, 60, 336, 395),
        (19, 68, 396, 463),
        (20, 58, 464, 521),
    ],
    (21, 54, 522, 575),
    GRANULE_LINES,
);

const SHORT_22050: [Band; SHORT_BANDS] = bands_with_top(
    [
        (0, 4, 0, 3),
        (1, 4, 4, 7),
        (2, 4, 8, 11),
        (3, 6, 12, 17),
        (4, 6, 18, 23),
        (5, 8, 24, 31),
        (6, 10, 32, 41),
        (7, 14, 42, 55),
        (8, 18, 56, 73),
        (9, 26, 74, 99),
        (10, 32, 100, 131),
        (11, 42, 132, 173),
    ],
    (12, 18, 174, 191),
    SHORT_WINDOW_LINES,
);

// Table B.2 of ISO/IEC 13818-3, 24 kHz.

const LONG_24000: [Band; LONG_BANDS] = bands_with_top(
    [
        (0, 6, 0, 5),
        (1, 6, 6, 11),
        (2, 6, 12, 17),
        (3, 6, 18, 23),
        (4, 6, 24, 29),
        (5, 6, 30, 35),
        (6, 8, 36, 43),
        (7, 10, 44, 53),
        (8, 12, 54, 65),
        (9, 14, 66, 79),
        (10, 16, 80, 95),
        (11, 18, 96, 113),
        (12, 22, 114, 135),
        (13, 26, 136, 161),
        (14, 32, 162, 193),
        (15, 38, 194, 231),
        (16, 46, 232, 277),
        (17, 54, 278, 331),
        (18, 62, 332, 393),
        (19, 70, 394, 463),
        (20, 76, 464, 539),
    ],
    (21, 36, 540, 575),
    GRANULE_LINES,
);

/// Spec reading: this table alone prints each end index one past the last
/// line of its band, so that it equals the next band's start index (4, 8,
/// 12, ..., 192). Its widths and start indices agree with each other and with
/// 192 lines, so the end indices are read as misprints and transcribed here
/// one lower, inclusive like those of every other table.
const SHORT_24000: [Band; SHORT_BANDS] = bands_with_top(
    [
        (0, 4, 0, 3),
        (1, 4, 4, 7),
        (2, 4, 8, 11),
        (3, 6, 12, 17),
        (4, 8, 18, 25),
        (5, 10, 26, 35),
        (6, 12, 36, 47),
        (7, 14, 48, 61),
        (8, 18, 62, 79),
        (9, 24, 80, 103),
        (10, 32, 104, 135),
        (11, 44, 136, 179),
    ],
    (12, 12, 180, 191),
    SHORT_WINDOW_LINES,
);

// US 6,185,539 B1, column 4: MPEG-2.5, 8 kHz.

const LONG_8000: [Band; LONG_BANDS] = bands_from_widths(
    [
        12, 12, 12, 12, 12, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 76, 90, 2, 2, 2, 2,
    ],
    2,
    GRANULE_LINES,
);

const SHORT_8000: [Band; SHORT_BANDS] = bands_from_widths(
    [8, 8, 8, 12, 16, 20, 24, 28, 36, 2, 2, 2],
    26,
    SHORT_WINDOW_LINES,
);

#[cfg(test)]
mod tests {
    use super::*;

    /// A granule has 576 lines (clause 2.1.70), split three ways between the
    /// short windows of a short block.
    #[test]
    fn bands_fit_within_a_granule() {
        for fs in SamplingFrequency::ALL {
            assert!(long_bands(fs)[LONG_BANDS - 1].end() <= 576, "{fs:?}");
            assert!(short_bands(fs)[SHORT_BANDS - 1].end() <= 192, "{fs:?}");
        }
    }

    /// Reordering a short block moves every line exactly once, and puts
    /// each scalefactor's lines where [`coding_bands`] says.
    #[test]
    fn short_coding_order_is_a_permutation_matching_the_coding_bands() {
        for fs in SamplingFrequency::ALL {
            let xr: [f64; GRANULE_LINES] = std::array::from_fn(|i| i as f64);
            let ordered = short_coding_order(&xr, fs);
            let mut seen = ordered.map(|x| x as usize);
            seen.sort_unstable();
            assert!(seen.iter().enumerate().all(|(i, &x)| i == x), "{fs:?}");

            let ranges = coding_bands(BlockType::Short, fs);
            assert_eq!(ranges.len(), SHORT_BANDS * SHORT_WINDOWS);
            for (index, range) in ranges.iter().enumerate() {
                let (sfb, window) = (index / SHORT_WINDOWS, index % SHORT_WINDOWS);
                let band = short_bands(fs)[sfb];
                for (offset, &x) in ordered[range.start..range.end()].iter().enumerate() {
                    let line = band.start + offset;
                    let expected = 18 * (line / 6) + 6 * window + line % 6;
                    assert_eq!(x as usize, expected, "{fs:?} sfb {sfb} window {window}");
                }
            }
            let last = short_bands(fs)[SHORT_BANDS - 1].end();
            for window in 0..SHORT_WINDOWS {
                let above = lines_above_bands(BlockType::Short, fs, window);
                assert_eq!(above.len(), SHORT_WINDOW_LINES - last, "{fs:?}");
                for (offset, &x) in ordered[above].iter().enumerate() {
                    let line = last + offset;
                    let expected = 18 * (line / 6) + 6 * window + line % 6;
                    assert_eq!(x as usize, expected, "{fs:?} window {window}");
                }
            }
            let long_last = long_bands(fs)[LONG_BANDS - 1].end();
            assert_eq!(
                lines_above_bands(BlockType::Normal, fs, 0),
                long_last..GRANULE_LINES
            );
        }
    }

    #[test]
    fn selects_the_table_for_each_sampling_frequency() {
        let last = |fs| long_bands(fs)[LONG_BANDS - 1];
        assert_eq!(
            last(SamplingFrequency::Hz32000),
            Band {
                start: 448,
                width: 102
            }
        );
        assert_eq!(
            last(SamplingFrequency::Hz44100),
            Band {
                start: 342,
                width: 76
            }
        );
        assert_eq!(
            last(SamplingFrequency::Hz48000),
            Band {
                start: 330,
                width: 54
            }
        );
        assert_eq!(
            short_bands(SamplingFrequency::Hz44100)[SHORT_BANDS - 1].end(),
            136
        );
    }

    /// The bands of the lower sampling frequencies end where the top rows of
    /// Table B.2 of ISO/IEC 13818-3 begin.
    #[test]
    fn lsf_bands_end_below_the_top_rows() {
        let ends = |fs| {
            (
                long_bands(fs)[LONG_BANDS - 1].end(),
                short_bands(fs)[SHORT_BANDS - 1].end(),
            )
        };
        assert_eq!(ends(SamplingFrequency::Hz16000), (522, 174));
        assert_eq!(ends(SamplingFrequency::Hz22050), (522, 174));
        assert_eq!(ends(SamplingFrequency::Hz24000), (540, 180));
        assert_eq!(ends(SamplingFrequency::Hz11025), (522, 174));
        assert_eq!(ends(SamplingFrequency::Hz12000), (540, 180));
        assert_eq!(ends(SamplingFrequency::Hz8000), (574, 166));
    }

    /// The highest region of the patent's 8 kHz table holds the top 100
    /// lines of a long block, the last 2 of them above the bands.
    #[test]
    fn fixed_bands_are_the_top_of_the_8_khz_table() {
        let fs = SamplingFrequency::Hz8000;
        let long = coding_bands(BlockType::Normal, fs);
        let fixed = fixed_at_zero(BlockType::Normal, fs);
        assert_eq!(long[fixed.start].start, 476);
        assert_eq!(GRANULE_LINES - long[fixed.start].start, 100);
        let short = coding_bands(BlockType::Short, fs);
        let fixed = fixed_at_zero(BlockType::Short, fs);
        assert_eq!(short[fixed.start].start, 3 * 160);
        assert!(fixed_at_zero(BlockType::Normal, SamplingFrequency::Hz16000).is_empty());
    }
}
