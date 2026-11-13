// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Layer III scalefactor bands (Table B.8).
//!
//! The tables are transcribed as printed: `(scalefactor band, width of band,
//! index of start, index of end)`, with the end index inclusive. Their
//! consistency is checked at compile time.

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
pub fn long_bands(sampling_frequency: SamplingFrequency) -> &'static [Band; LONG_BANDS] {
    match sampling_frequency {
        SamplingFrequency::Hz32000 => &LONG_32000,
        SamplingFrequency::Hz44100 => &LONG_44100,
        SamplingFrequency::Hz48000 => &LONG_48000,
    }
}

/// The bands for short windows (block type 2).
pub fn short_bands(sampling_frequency: SamplingFrequency) -> &'static [Band; SHORT_BANDS] {
    match sampling_frequency {
        SamplingFrequency::Hz32000 => &SHORT_32000,
        SamplingFrequency::Hz44100 => &SHORT_44100,
        SamplingFrequency::Hz48000 => &SHORT_48000,
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
/// 45 dB SNR at every sampling frequency, and at about 10 dB with the lines
/// interleaved across the windows instead.
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
}
