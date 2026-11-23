// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The threshold calculation partitions of psychoacoustic model 2 for Layer
//! III long and short blocks (Table C.7) and their conversion to scalefactor
//! bands (Table C.8), and their counterparts for the lower sampling
//! frequencies (Tables D.3 and D.4 of ISO/IEC 13818-3).
//!
//! The tables are transcribed as printed, except that the `cbw` column of
//! Tables C.8 and D.4 is left out: the conversion in figure C.6.c does not
//! use it. Their numbering and layout are checked at compile time. The
//! conversion tables are kept for the tests only: the conversion is computed
//! from the frequencies of the partitions and bands at every sampling
//! frequency (see [`band_weights`]).
//!
//! Spec gap: neither standard nor US patent 6,185,539 B1 has tables for
//! MPEG-2.5. Each of its sampling frequencies is half one of ISO/IEC
//! 13818-3: 8 kHz of 16, 11.025 of 22.05 and 12 of 24. Its tables are those
//! of the frequency twice as high, over the frequencies the lower rate has,
//! the same frequencies in Hz keeping the same values:
//!
//! - The partitions below FFT line 256, half the higher rate's Nyquist line,
//!   are kept, each twice as wide in lines, since line `w` lies at the
//!   frequency of line `w / 2` at the higher rate. The last is cut at line
//!   256 and keeps its printed `bval`, which then lies a little above its
//!   median.
//! - `bval` and `minval`, functions of frequency, are kept as printed.
//! - `qthr`, in the tables a per-line threshold in quiet times the width
//!   (see the test `qthr_is_a_per_line_level_times_the_width`), keeps its
//!   per-line level over the new width.
//! - `norm` and the short blocks' SNR are kept as printed. `rnorm` (clause
//!   D.2.4 f)) would follow from the changed set of partitions, but the
//!   printed column lies 1 to 3 dB below it at the higher rate, and the
//!   printed one keeps the threshold of noise closer to the higher rate's
//!   for the same frequencies (see the test
//!   `mpeg25_threshold_follows_the_rate_twice_as_high`).
//! - The conversion to scalefactor bands is computed (see
//!   `weights_by_frequency`), as the bands are not those of the higher
//!   rate (see [`scalefactor_bands::long_bands`]).

use std::sync::LazyLock;

use super::super::header::SamplingFrequency;
use super::super::scalefactor_bands::{self, Band, LONG_BANDS, SHORT_BANDS};

/// One threshold calculation partition: a run of lines of the 1024-point FFT.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Partition {
    /// The first FFT line, line 0 being DC.
    pub start: usize,
    /// The number of FFT lines.
    pub width: usize,
    /// The lower limit of the SNR in dB.
    pub minval: f64,
    /// The threshold in quiet, summed over the partition's lines.
    pub qthr: f64,
    /// The normalising constant that undoes the gain of the spreading
    /// function.
    pub norm: f64,
    /// The median Bark value.
    pub bval: f64,
}

/// One threshold calculation partition for short blocks: a run of lines of
/// the 256-point FFT. Short blocks have a fixed SNR in place of `minval`
/// (clause C.1.5.3.2.1, Tables C.7.d to C.7.f).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShortPartition {
    /// The first FFT line, line 0 being DC.
    pub start: usize,
    /// The number of FFT lines.
    pub width: usize,
    /// The threshold in quiet, summed over the partition's lines.
    pub qthr: f64,
    /// The normalising constant that undoes the gain of the spreading
    /// function, as printed; see [`short_partitions`].
    pub norm: f64,
    /// How far the threshold lies below the spread energy (figure C.6.d).
    pub snr: Snr,
    /// The median Bark value.
    pub bval: f64,
}

/// The SNR of a short partition, as its table prints it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Snr {
    /// In dB, negative, as in Tables C.7.d to C.7.f.
    Db(f64),
    /// As the ratio of the threshold to the spread energy, as in Tables
    /// D.3.d to D.3.f of ISO/IEC 13818-3 (see
    /// `short_partitions_from_ratios`).
    Ratio(f64),
}

impl Snr {
    /// The ratio of the threshold to the spread energy.
    pub fn ratio(self) -> f64 {
        match self {
            Self::Db(db) => 10f64.powf(db / 10.0),
            Self::Ratio(ratio) => ratio,
        }
    }

    #[cfg(any(test, feature = "unstable"))]
    pub fn db(self) -> f64 {
        match self {
            Self::Db(db) => db,
            Self::Ratio(ratio) => 10.0 * ratio.log10(),
        }
    }
}

/// How a scalefactor band collects its energy and threshold from the
/// partitions: partition `bu` weighted by `w1`, those strictly between `bu`
/// and `bo` in full, and partition `bo` weighted by `w2` (figure C.6.c).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandWeights {
    pub bu: usize,
    pub bo: usize,
    pub w1: f64,
    pub w2: f64,
}

pub fn partitions(sampling_frequency: SamplingFrequency) -> &'static [Partition] {
    match sampling_frequency {
        SamplingFrequency::Hz48000 => &PARTITIONS_48000,
        SamplingFrequency::Hz44100 => &PARTITIONS_44100,
        SamplingFrequency::Hz32000 => &PARTITIONS_32000,
        SamplingFrequency::Hz24000 => &PARTITIONS_24000,
        SamplingFrequency::Hz22050 => &PARTITIONS_22050,
        SamplingFrequency::Hz16000 => &PARTITIONS_16000,
        _ => &mpeg25(sampling_frequency).partitions,
    }
}

/// How each long scalefactor band collects the partitions (figure C.6.c):
/// by the frequencies of the partitions' FFT lines (see
/// `weights_by_frequency`).
///
/// Spec reading: Table C.8 (ISO/IEC 11172-3, pp. 88 to 90) and Table D.4 of
/// ISO/IEC 13818-3 (second edition, pp. 105 to 107) are not used, as they
/// gather each band's threshold and energy from partitions above the band's
/// own frequencies. Clause C.1.5.3.2.1 says only that the partitions are
/// converted to the bands with the weights of the tables, and gives no rule
/// for them, but the lowest long bands at 44.1 and 48 kHz agree with the
/// bands' frequencies exactly, an FFT line standing for half a line either
/// side of it. Above them the printed bands lie higher by a number of whole
/// lines (fractions of a line in ISO/IEC 13818-3) that grows with frequency
/// (see the test `printed_tables_collect_above_the_bands`). Long blocks:
/// below the top band, up to 34 FFT lines (1.6 kHz) at 48 kHz, 26 (1.1 kHz)
/// at 44.1 kHz, 36 (1.1 kHz) at 32 kHz, 29.6 (690 Hz) at 24 kHz, 29.1
/// (630 Hz) at 22.05 kHz and 28.8 (450 Hz) at 16 kHz, where every band from
/// the second on starts at least 3 lines high, more than half the width of the
/// lowest bands; the top band takes 66 lines (3.1 kHz) above its upper edge
/// at 48 kHz, 72 (3.1 kHz) at 44.1 kHz and 32 (500 Hz) at 16 kHz. Short
/// blocks: up to 6 to 7 lines (of the 256-point FFT) below the top band, and
/// 16 and 18 lines (3 kHz) above the top band at 48 and 44.1 kHz.
///
/// Where a band collects a tone that lies above its lines, the ratio of
/// threshold to energy it takes from the FFT does not hold for its lines of
/// `xr`, and the band that holds the tone, having lost the tone's energy
/// from its ratio, is allowed noise far above its own energy: Table D.4.c
/// gives a tone of 2 kHz at 16 kHz to the band below its own, and at
/// 44.1 kHz the top band took its ratio from partitions up to 3 kHz above
/// it. Bands at their own frequencies take the FFT's ratio for the lines
/// they code. Quiet pure tones, at -40 dB below full scale, coded in one
/// channel at level 80 of the quality scale, had the band beside them
/// allowed almost no noise, so that at the edges of the subbands, `fs / 8`
/// and `fs / 16`, every frame took the highest bitrate at 16 and 24 kHz,
/// and at `fs / 8` at 44.1 kHz, and so did tones elsewhere, such as one of
/// 10 kHz at 44.1 kHz (85 kbit/s), 11.2 kHz at 32 kHz (132) and 18.8 kHz
/// at 48 kHz (137). At their own frequencies, all of them take 8 to
/// 38 kbit/s but a tone of 2 kHz at 16 kHz, `fs / 8` (see
/// `quiet_allowance` in the model). As ffmpeg decodes them, the tones at
/// the edges keep their level within 0.3 dB, and those above them decode
/// up to 2.2 dB loud, as tones of those frequencies whose bands were not
/// starved did before. Graded with PEAQ (GstPEAQ 0.6.1, black box) on 51 tracks of
/// nine kinds of music at 44.1 kHz, the mean grade rose by 0.025 at
/// 128 kbit/s in two channels and by 0.014 at 64 in one, and held at 192;
/// at levels 50, 80 and 100 of the quality scale it moved by -0.021,
/// +0.005 and +0.003 in two channels, at 2.0 kbit/s less, 0.8 more and
/// 0.1 less, and by -0.013, +0.024 and +0.020 in one, at 1.1 less, 1.0
/// more and 1.3 more.
///
/// The highest band (`sfb 21`), above 16 kHz at 44.1 and 48 kHz, has no
/// allowance, and Annex C leaves its lines to the step the other bands are
/// coded at. Tones there of 16.4 to 18.8 kHz had been kept at a variable
/// bitrate only by the starved band below them, at 68 to 137 kbit/s; they
/// are now lost at level 80, as one of 19.8 kHz already was.
pub fn band_weights(sampling_frequency: SamplingFrequency) -> [BandWeights; LONG_BANDS] {
    weights_by_frequency(
        scalefactor_bands::long_bands(sampling_frequency),
        partitions(sampling_frequency)
            .iter()
            .map(|p| (p.start, p.width)),
        LONG_FFT_LINES_PER_LINE,
    )
}

/// The partitions for short blocks.
///
/// Spec gap: the `norm` column of each of Tables C.7.d to C.7.f repeats,
/// row for row and to within a rounding step, the `norm` column of the
/// long-block table for the same sampling frequency (C.7.a to C.7.c),
/// although the short partitions have other Bark values. It does not match
/// the reciprocal of the spreading function summed over the short
/// partitions, as the long tables' column does (clause D.2.4 f)). Tables
/// D.3.d to D.3.f of ISO/IEC 13818-3 print a column of their own, as they
/// do for long blocks.
pub fn short_partitions(sampling_frequency: SamplingFrequency) -> &'static [ShortPartition] {
    match sampling_frequency {
        SamplingFrequency::Hz48000 => &SHORT_PARTITIONS_48000,
        SamplingFrequency::Hz44100 => &SHORT_PARTITIONS_44100,
        SamplingFrequency::Hz32000 => &SHORT_PARTITIONS_32000,
        SamplingFrequency::Hz24000 => &SHORT_PARTITIONS_24000,
        SamplingFrequency::Hz22050 => &SHORT_PARTITIONS_22050,
        SamplingFrequency::Hz16000 => &SHORT_PARTITIONS_16000,
        _ => &mpeg25(sampling_frequency).short_partitions,
    }
}

/// As [`band_weights`], for the short scalefactor bands.
pub fn short_band_weights(sampling_frequency: SamplingFrequency) -> [BandWeights; SHORT_BANDS] {
    weights_by_frequency(
        scalefactor_bands::short_bands(sampling_frequency),
        short_partitions(sampling_frequency)
            .iter()
            .map(|p| (p.start, p.width)),
        SHORT_FFT_LINES_PER_LINE,
    )
}

/// The lines of the long FFT to a line of `xr` in a long block: 512 to 576,
/// both spanning the frequencies up to half the sampling frequency.
const LONG_FFT_LINES_PER_LINE: f64 = 512.0 / 576.0;

/// The lines of the short FFT to a line of `xr` in a short window: 128 to
/// 192.
const SHORT_FFT_LINES_PER_LINE: f64 = 128.0 / 192.0;

/// The tables of an MPEG-2.5 sampling frequency, derived from those of the
/// frequency of ISO/IEC 13818-3 twice as high (see [`halve`]).
struct Mpeg25Tables {
    partitions: Vec<Partition>,
    short_partitions: Vec<ShortPartition>,
}

static MPEG25_8000: LazyLock<Mpeg25Tables> =
    LazyLock::new(|| Mpeg25Tables::halving(&PARTITIONS_16000, &SHORT_PARTITIONS_16000));
static MPEG25_11025: LazyLock<Mpeg25Tables> =
    LazyLock::new(|| Mpeg25Tables::halving(&PARTITIONS_22050, &SHORT_PARTITIONS_22050));
static MPEG25_12000: LazyLock<Mpeg25Tables> =
    LazyLock::new(|| Mpeg25Tables::halving(&PARTITIONS_24000, &SHORT_PARTITIONS_24000));

fn mpeg25(sampling_frequency: SamplingFrequency) -> &'static Mpeg25Tables {
    match sampling_frequency {
        SamplingFrequency::Hz8000 => &MPEG25_8000,
        SamplingFrequency::Hz11025 => &MPEG25_11025,
        SamplingFrequency::Hz12000 => &MPEG25_12000,
        _ => panic!("{sampling_frequency:?} is not a sampling frequency of MPEG-2.5"),
    }
}

impl Mpeg25Tables {
    /// The partitions of an MPEG-2.5 sampling frequency from those of the
    /// frequency twice as high, `long` and `short`.
    fn halving(long: &[Partition], short: &[ShortPartition]) -> Self {
        let halved = halve(long.iter().map(|p| (p.start, p.width)), 512);
        let partitions: Vec<Partition> = (long.iter().zip(halved))
            .map(|(p, (start, width))| Partition {
                start,
                width,
                qthr: p.qthr / p.width as f64 * width as f64,
                ..*p
            })
            .collect();
        let halved = halve(short.iter().map(|p| (p.start, p.width)), 128);
        let short_partitions: Vec<ShortPartition> = (short.iter().zip(halved))
            .map(|(p, (start, width))| ShortPartition {
                start,
                width,
                qthr: p.qthr / p.width as f64 * width as f64,
                ..*p
            })
            .collect();
        Self {
            partitions,
            short_partitions,
        }
    }
}

/// The partitions `(start, width)` of an FFT whose lines run up to
/// `nyquist`, stretched over the same FFT at half the sampling frequency:
/// those below `nyquist / 2`, the frequencies the lower rate still has,
/// each twice as wide, the last cut at `nyquist / 2`. Line `w` at the lower
/// rate lies at the frequency of line `w / 2` at the higher.
fn halve(
    partitions: impl Iterator<Item = (usize, usize)>,
    nyquist: usize,
) -> impl Iterator<Item = (usize, usize)> {
    partitions
        .take_while(move |&(start, _)| start < nyquist / 2)
        .map(move |(start, width)| (2 * start, 2 * ((start + width).min(nyquist / 2) - start)))
}

/// Converts partitions `(start, width)` of FFT lines to `bands` of lines of
/// `xr`, `fft_per_line` FFT lines to a line of `xr`. Each band takes from
/// each partition the share of the partition's frequencies that lies within
/// the band's: FFT line `j` stands for the frequencies from `j - 1/2` to
/// `j + 1/2` times the FFT's spacing, and the lines of `xr` from `start` to
/// `end` for those from `start` to `end` times their own. DC belongs wholly
/// to the first band. A band within a single partition gets `bo` one past
/// `bu` and a `w2` of zero, so that it takes the ratio of that partition
/// alone, and a band that runs past the last partition collects the
/// partitions after `bu` in full.
fn weights_by_frequency<const N: usize>(
    bands: &[Band; N],
    partitions: impl Iterator<Item = (usize, usize)>,
    fft_per_line: f64,
) -> [BandWeights; N] {
    let partitions: Vec<(f64, f64)> = partitions.map(|(s, w)| (s as f64, w as f64)).collect();
    let containing = |x: f64| partitions.iter().position(|&(s, w)| s <= x && x < s + w);
    // The frequency `line` lines of `xr` up, in the partitions' own
    // measure, where partition `(s, w)` runs from `s` to `s + w`: half a
    // line above the FFT line of that frequency.
    let edge = |line: usize| match line {
        0 => 0.0,
        _ => line as f64 * fft_per_line + 0.5,
    };
    bands.map(|band| {
        let (low, high) = (edge(band.start), edge(band.end()));
        let bu = containing(low).expect("every band starts within the partitions");
        let (start, width) = partitions[bu];
        if high <= start + width {
            return BandWeights {
                bu,
                bo: bu + 1,
                w1: (high - low) / width,
                w2: 0.0,
            };
        }
        let bo = containing(high).unwrap_or(partitions.len());
        let w2 = partitions.get(bo).map_or(0.0, |&(s, w)| (high - s) / w);
        BandWeights {
            bu,
            bo,
            w1: (start + width - low) / width,
            w2,
        }
    })
}

/// Builds a partition table from printed rows `(no., FFT-lines, minval,
/// qthr, norm, bval)`, checking that the rows are numbered in order and that
/// the Bark values rise. The partitions are laid out from line 0 without
/// gaps.
const fn partitions_from<const N: usize>(
    rows: [(usize, usize, f64, f64, f64, f64); N],
) -> [Partition; N] {
    let mut partitions = [Partition {
        start: 0,
        width: 0,
        minval: 0.0,
        qthr: 0.0,
        norm: 0.0,
        bval: 0.0,
    }; N];
    let mut start = 0;
    let mut i = 0;
    while i < N {
        let (no, width, minval, qthr, norm, bval) = rows[i];
        assert!(no == i, "partition out of order");
        assert!(width > 0, "empty partition");
        assert!(i == 0 || bval > rows[i - 1].5, "Bark values do not rise");
        partitions[i] = Partition {
            start,
            width,
            minval,
            qthr,
            norm,
            bval,
        };
        start += width;
        i += 1;
    }
    assert!(start <= 513, "partitions run past the Nyquist line");
    partitions
}

/// Builds a short-block partition table from printed rows `(no.,
/// FFT-lines, qthr, norm, SNR (dB), bval)`, with the checks of
/// [`partitions_from`].
const fn short_partitions_from<const N: usize>(
    rows: [(usize, usize, f64, f64, f64, f64); N],
) -> [ShortPartition; N] {
    short_partitions_with(rows, false)
}

/// Builds a partition table from printed rows as [`short_partitions_from`]
/// does, for tables that print the SNR as a ratio.
///
/// Spec reading: Tables D.3.d to D.3.f of ISO/IEC 13818-3 head the column
/// "SNR (dB)", but print values from 0.15 to 0.4, which as decibels would
/// put the threshold above the spread energy. They are the ratios whose
/// decibels Tables C.7.d to C.7.f print (0.15 is the -8.240 dB those tables
/// start with), and are read as such.
const fn short_partitions_from_ratios<const N: usize>(
    rows: [(usize, usize, f64, f64, f64, f64); N],
) -> [ShortPartition; N] {
    short_partitions_with(rows, true)
}

const fn short_partitions_with<const N: usize>(
    rows: [(usize, usize, f64, f64, f64, f64); N],
    snr_as_ratio: bool,
) -> [ShortPartition; N] {
    let mut partitions = [ShortPartition {
        start: 0,
        width: 0,
        qthr: 0.0,
        norm: 0.0,
        snr: Snr::Db(0.0),
        bval: 0.0,
    }; N];
    let mut start = 0;
    let mut i = 0;
    while i < N {
        let (no, width, qthr, norm, snr, bval) = rows[i];
        assert!(no == i, "partition out of order");
        assert!(width > 0, "empty partition");
        assert!(i == 0 || bval > rows[i - 1].5, "Bark values do not rise");
        partitions[i] = ShortPartition {
            start,
            width,
            qthr,
            norm,
            snr: if snr_as_ratio {
                Snr::Ratio(snr)
            } else {
                Snr::Db(snr)
            },
            bval,
        };
        start += width;
        i += 1;
    }
    assert!(start <= 129, "partitions run past the Nyquist line");
    partitions
}

/// Builds a conversion table from printed rows `(no. sb, bu, bo, w1, w2)`,
/// checking that the bands are numbered in order, start at partition 0, and
/// that each band begins at the partition where the last one ended.
#[cfg(test)]
const fn band_weights_from<const N: usize>(
    rows: [(usize, usize, usize, f64, f64); N],
) -> [BandWeights; N] {
    let mut bands = [BandWeights {
        bu: 0,
        bo: 0,
        w1: 0.0,
        w2: 0.0,
    }; N];
    let mut i = 0;
    while i < N {
        let (sb, bu, bo, w1, w2) = rows[i];
        assert!(sb == i, "band out of order");
        assert!(bu < bo, "band ends before it starts");
        assert!(
            bu == if i == 0 { 0 } else { rows[i - 1].2 },
            "band does not start where the last ended"
        );
        bands[i] = BandWeights { bu, bo, w1, w2 };
        i += 1;
    }
    bands
}

// Table C.7.a, 48 kHz long blocks.

const PARTITIONS_48000: [Partition; 62] = partitions_from([
    (0, 1, 24.5, 4.532, 0.970, 0.000),
    (1, 1, 24.5, 4.532, 0.755, 0.469),
    (2, 1, 24.5, 4.532, 0.738, 0.937),
    (3, 1, 24.5, 0.904, 0.730, 1.406),
    (4, 1, 24.5, 0.904, 0.724, 1.875),
    (5, 1, 20.0, 0.090, 0.723, 2.344),
    (6, 1, 20.0, 0.090, 0.723, 2.812),
    (7, 1, 20.0, 0.029, 0.723, 3.281),
    (8, 1, 20.0, 0.029, 0.718, 3.750),
    (9, 1, 20.0, 0.009, 0.690, 4.199),
    (10, 1, 20.0, 0.009, 0.660, 4.625),
    (11, 1, 18.0, 0.009, 0.641, 5.047),
    (12, 1, 18.0, 0.009, 0.600, 5.437),
    (13, 1, 18.0, 0.009, 0.584, 5.828),
    (14, 1, 12.0, 0.009, 0.531, 6.187),
    (15, 1, 12.0, 0.009, 0.537, 6.522),
    (16, 2, 6.0, 0.018, 0.857, 7.174),
    (17, 2, 6.0, 0.018, 0.858, 7.800),
    (18, 2, 3.0, 0.018, 0.853, 8.402),
    (19, 2, 3.0, 0.018, 0.824, 8.966),
    (20, 2, 3.0, 0.018, 0.778, 9.483),
    (21, 2, 3.0, 0.018, 0.740, 9.966),
    (22, 2, 0.0, 0.018, 0.709, 10.426),
    (23, 2, 0.0, 0.018, 0.676, 10.866),
    (24, 2, 0.0, 0.018, 0.632, 11.279),
    (25, 2, 0.0, 0.018, 0.592, 11.669),
    (26, 2, 0.0, 0.018, 0.553, 12.042),
    (27, 2, 0.0, 0.018, 0.510, 12.386),
    (28, 2, 0.0, 0.018, 0.513, 12.721),
    (29, 3, 0.0, 0.027, 0.608, 13.115),
    (30, 3, 0.0, 0.027, 0.673, 13.561),
    (31, 3, 0.0, 0.027, 0.636, 13.983),
    (32, 3, 0.0, 0.027, 0.586, 14.371),
    (33, 3, 0.0, 0.027, 0.571, 14.741),
    (34, 4, 0.0, 0.036, 0.616, 15.140),
    (35, 4, 0.0, 0.036, 0.640, 15.562),
    (36, 4, 0.0, 0.036, 0.597, 15.962),
    (37, 4, 0.0, 0.036, 0.538, 16.324),
    (38, 4, 0.0, 0.036, 0.512, 16.665),
    (39, 5, 0.0, 0.045, 0.528, 17.020),
    (40, 5, 0.0, 0.045, 0.516, 17.373),
    (41, 5, 0.0, 0.045, 0.493, 17.708),
    (42, 6, 0.0, 0.054, 0.499, 18.045),
    (43, 7, 0.0, 0.063, 0.525, 18.398),
    (44, 7, 0.0, 0.063, 0.541, 18.762),
    (45, 8, 0.0, 0.072, 0.528, 19.120),
    (46, 8, 0.0, 0.072, 0.510, 19.466),
    (47, 8, 0.0, 0.072, 0.506, 19.807),
    (48, 10, 0.0, 0.180, 0.525, 20.159),
    (49, 10, 0.0, 0.180, 0.536, 20.522),
    (50, 10, 0.0, 0.180, 0.518, 20.873),
    (51, 13, 0.0, 0.372, 0.501, 21.214),
    (52, 13, 0.0, 0.372, 0.496, 21.553),
    (53, 14, 0.0, 0.400, 0.497, 21.892),
    (54, 18, 0.0, 1.628, 0.495, 22.231),
    (55, 18, 0.0, 1.628, 0.494, 22.569),
    (56, 20, 0.0, 1.808, 0.497, 22.909),
    (57, 25, 0.0, 22.607, 0.494, 23.248),
    (58, 25, 0.0, 22.607, 0.487, 23.583),
    (59, 35, 0.0, 31.650, 0.483, 23.915),
    (60, 67, 0.0, 605.867, 0.482, 24.246),
    (61, 67, 0.0, 605.867, 0.524, 24.576),
]);

// Table C.7.b, 44.1 kHz long blocks.

const PARTITIONS_44100: [Partition; 63] = partitions_from([
    (0, 1, 24.5, 4.532, 0.951, 0.000),
    (1, 1, 24.5, 4.532, 0.700, 0.431),
    (2, 1, 24.5, 4.532, 0.681, 0.861),
    (3, 1, 24.5, 0.904, 0.675, 1.292),
    (4, 1, 24.5, 0.904, 0.667, 1.723),
    (5, 1, 20.0, 0.090, 0.665, 2.153),
    (6, 1, 20.0, 0.090, 0.664, 2.584),
    (7, 1, 20.0, 0.029, 0.664, 3.015),
    (8, 1, 20.0, 0.029, 0.664, 3.445),
    (9, 1, 20.0, 0.029, 0.655, 3.876),
    (10, 1, 20.0, 0.009, 0.616, 4.279),
    (11, 1, 20.0, 0.009, 0.597, 4.670),
    (12, 1, 18.0, 0.009, 0.578, 5.057),
    (13, 1, 18.0, 0.009, 0.541, 5.415),
    (14, 1, 18.0, 0.009, 0.575, 5.774),
    (15, 2, 12.0, 0.018, 0.856, 6.422),
    (16, 2, 6.0, 0.018, 0.846, 7.026),
    (17, 2, 6.0, 0.018, 0.840, 7.609),
    (18, 2, 3.0, 0.018, 0.822, 8.168),
    (19, 2, 3.0, 0.018, 0.800, 8.710),
    (20, 2, 3.0, 0.018, 0.753, 9.207),
    (21, 2, 3.0, 0.018, 0.704, 9.662),
    (22, 2, 0.0, 0.018, 0.674, 10.099),
    (23, 2, 0.0, 0.018, 0.640, 10.515),
    (24, 2, 0.0, 0.018, 0.609, 10.917),
    (25, 2, 0.0, 0.018, 0.566, 11.293),
    (26, 2, 0.0, 0.018, 0.535, 11.652),
    (27, 2, 0.0, 0.018, 0.531, 11.997),
    (28, 3, 0.0, 0.027, 0.615, 12.394),
    (29, 3, 0.0, 0.027, 0.686, 12.850),
    (30, 3, 0.0, 0.027, 0.650, 13.277),
    (31, 3, 0.0, 0.027, 0.611, 13.681),
    (32, 3, 0.0, 0.027, 0.567, 14.062),
    (33, 3, 0.0, 0.027, 0.520, 14.411),
    (34, 3, 0.0, 0.027, 0.513, 14.751),
    (35, 4, 0.0, 0.036, 0.557, 15.119),
    (36, 4, 0.0, 0.036, 0.584, 15.508),
    (37, 4, 0.0, 0.036, 0.570, 15.883),
    (38, 5, 0.0, 0.045, 0.579, 16.263),
    (39, 5, 0.0, 0.045, 0.585, 16.654),
    (40, 5, 0.0, 0.045, 0.548, 17.020),
    (41, 6, 0.0, 0.054, 0.536, 17.374),
    (42, 6, 0.0, 0.054, 0.550, 17.744),
    (43, 7, 0.0, 0.063, 0.532, 18.104),
    (44, 7, 0.0, 0.063, 0.504, 18.447),
    (45, 7, 0.0, 0.063, 0.496, 18.781),
    (46, 9, 0.0, 0.081, 0.516, 19.130),
    (47, 9, 0.0, 0.081, 0.527, 19.487),
    (48, 9, 0.0, 0.081, 0.516, 19.838),
    (49, 10, 0.0, 0.180, 0.497, 20.179),
    (50, 10, 0.0, 0.180, 0.489, 20.510),
    (51, 11, 0.0, 0.198, 0.502, 20.852),
    (52, 14, 0.0, 0.400, 0.502, 21.196),
    (53, 14, 0.0, 0.400, 0.491, 21.531),
    (54, 15, 0.0, 0.429, 0.497, 21.870),
    (55, 20, 0.0, 1.808, 0.504, 22.214),
    (56, 20, 0.0, 1.808, 0.504, 22.558),
    (57, 21, 0.0, 1.899, 0.495, 22.898),
    (58, 27, 0.0, 24.415, 0.486, 23.232),
    (59, 27, 0.0, 24.415, 0.484, 23.564),
    (60, 36, 0.0, 32.554, 0.483, 23.897),
    (61, 73, 0.0, 660.124, 0.475, 24.229),
    (62, 18, 0.0, 162.770, 0.515, 24.542),
]);

// Table C.7.c, 32 kHz long blocks.

const PARTITIONS_32000: [Partition; 59] = partitions_from([
    (0, 2, 24.5, 9.064, 0.997, 0.312),
    (1, 2, 24.5, 9.064, 0.893, 0.937),
    (2, 2, 24.5, 1.808, 0.881, 1.562),
    (3, 2, 20.0, 0.181, 0.873, 2.187),
    (4, 2, 20.0, 0.181, 0.872, 2.812),
    (5, 2, 20.0, 0.057, 0.871, 3.437),
    (6, 2, 20.0, 0.018, 0.860, 4.045),
    (7, 2, 20.0, 0.018, 0.839, 4.625),
    (8, 2, 18.0, 0.018, 0.812, 5.173),
    (9, 2, 18.0, 0.018, 0.784, 5.698),
    (10, 2, 12.0, 0.018, 0.741, 6.184),
    (11, 2, 12.0, 0.018, 0.697, 6.634),
    (12, 2, 6.0, 0.018, 0.674, 7.070),
    (13, 2, 6.0, 0.018, 0.651, 7.492),
    (14, 2, 6.0, 0.018, 0.633, 7.905),
    (15, 2, 3.0, 0.018, 0.611, 8.305),
    (16, 2, 3.0, 0.018, 0.589, 8.695),
    (17, 2, 3.0, 0.018, 0.575, 9.064),
    (18, 3, 3.0, 0.027, 0.654, 9.483),
    (19, 3, 3.0, 0.027, 0.724, 9.966),
    (20, 3, 0.0, 0.027, 0.701, 10.425),
    (21, 3, 0.0, 0.027, 0.673, 10.866),
    (22, 3, 0.0, 0.027, 0.631, 11.279),
    (23, 3, 0.0, 0.027, 0.592, 11.669),
    (24, 3, 0.0, 0.027, 0.553, 12.042),
    (25, 3, 0.0, 0.027, 0.510, 12.386),
    (26, 3, 0.0, 0.027, 0.505, 12.721),
    (27, 4, 0.0, 0.036, 0.562, 13.091),
    (28, 4, 0.0, 0.036, 0.598, 13.488),
    (29, 4, 0.0, 0.036, 0.589, 13.873),
    (30, 5, 0.0, 0.045, 0.607, 14.268),
    (31, 5, 0.0, 0.045, 0.620, 14.679),
    (32, 5, 0.0, 0.045, 0.580, 15.067),
    (33, 5, 0.0, 0.045, 0.532, 15.424),
    (34, 5, 0.0, 0.045, 0.517, 15.771),
    (35, 6, 0.0, 0.054, 0.517, 16.120),
    (36, 6, 0.0, 0.054, 0.509, 16.466),
    (37, 6, 0.0, 0.054, 0.506, 16.807),
    (38, 8, 0.0, 0.072, 0.522, 17.158),
    (39, 8, 0.0, 0.072, 0.531, 17.518),
    (40, 8, 0.0, 0.072, 0.519, 17.869),
    (41, 10, 0.0, 0.090, 0.512, 18.215),
    (42, 10, 0.0, 0.090, 0.509, 18.562),
    (43, 10, 0.0, 0.090, 0.497, 18.902),
    (44, 12, 0.0, 0.108, 0.494, 19.239),
    (45, 12, 0.0, 0.108, 0.501, 19.579),
    (46, 13, 0.0, 0.117, 0.507, 19.925),
    (47, 14, 0.0, 0.252, 0.502, 20.269),
    (48, 14, 0.0, 0.252, 0.493, 20.606),
    (49, 16, 0.0, 0.289, 0.497, 20.944),
    (50, 20, 0.0, 0.572, 0.506, 21.288),
    (51, 20, 0.0, 0.572, 0.510, 21.635),
    (52, 23, 0.0, 0.658, 0.504, 21.979),
    (53, 27, 0.0, 2.441, 0.496, 22.319),
    (54, 27, 0.0, 2.441, 0.493, 22.656),
    (55, 32, 0.0, 2.894, 0.490, 22.993),
    (56, 37, 0.0, 33.458, 0.483, 23.326),
    (57, 37, 0.0, 33.458, 0.458, 23.656),
    (58, 12, 0.0, 10.851, 0.500, 23.937),
]);

// Table C.7.d, 48 kHz short blocks.

const SHORT_PARTITIONS_48000: [ShortPartition; 38] = short_partitions_from([
    (0, 1, 4.532, 0.970, -8.240, 0.000),
    (1, 1, 0.904, 0.755, -8.240, 1.875),
    (2, 1, 0.029, 0.738, -8.240, 3.750),
    (3, 1, 0.009, 0.730, -8.240, 5.437),
    (4, 1, 0.009, 0.724, -8.240, 6.857),
    (5, 1, 0.009, 0.723, -8.240, 8.109),
    (6, 1, 0.009, 0.723, -8.240, 9.237),
    (7, 1, 0.009, 0.723, -8.240, 10.202),
    (8, 1, 0.009, 0.718, -8.240, 11.083),
    (9, 1, 0.009, 0.690, -8.240, 11.864),
    (10, 1, 0.009, 0.660, -7.447, 12.553),
    (11, 1, 0.009, 0.641, -7.447, 13.195),
    (12, 1, 0.009, 0.600, -7.447, 13.781),
    (13, 1, 0.009, 0.584, -7.447, 14.309),
    (14, 1, 0.009, 0.532, -7.447, 14.803),
    (15, 1, 0.009, 0.537, -7.447, 15.250),
    (16, 1, 0.009, 0.857, -7.447, 15.667),
    (17, 1, 0.009, 0.858, -7.447, 16.068),
    (18, 1, 0.009, 0.853, -7.447, 16.409),
    (19, 2, 0.018, 0.824, -7.447, 17.044),
    (20, 2, 0.018, 0.778, -6.990, 17.607),
    (21, 2, 0.018, 0.740, -6.990, 18.097),
    (22, 2, 0.018, 0.709, -6.990, 18.528),
    (23, 2, 0.018, 0.676, -6.990, 18.930),
    (24, 2, 0.018, 0.632, -6.990, 19.295),
    (25, 2, 0.018, 0.592, -6.990, 19.636),
    (26, 3, 0.054, 0.553, -6.990, 20.038),
    (27, 3, 0.054, 0.510, -6.990, 20.486),
    (28, 3, 0.054, 0.513, -6.990, 20.900),
    (29, 4, 0.114, 0.608, -6.990, 21.305),
    (30, 4, 0.114, 0.673, -6.020, 21.722),
    (31, 5, 0.452, 0.637, -6.020, 22.128),
    (32, 5, 0.452, 0.586, -6.020, 22.512),
    (33, 5, 0.452, 0.571, -6.020, 22.877),
    (34, 7, 6.330, 0.616, -5.229, 23.241),
    (35, 7, 6.330, 0.640, -5.229, 23.616),
    (36, 11, 9.947, 0.597, -5.229, 23.974),
    (37, 17, 153.727, 0.538, -5.229, 24.312),
]);

// Table C.7.e, 44.1 kHz short blocks.

const SHORT_PARTITIONS_44100: [ShortPartition; 39] = short_partitions_from([
    (0, 1, 4.532, 0.952, -8.240, 0.000),
    (1, 1, 0.904, 0.700, -8.240, 1.723),
    (2, 1, 0.029, 0.681, -8.240, 3.445),
    (3, 1, 0.009, 0.675, -8.240, 5.057),
    (4, 1, 0.009, 0.667, -8.240, 6.422),
    (5, 1, 0.009, 0.665, -8.240, 7.609),
    (6, 1, 0.009, 0.664, -8.240, 8.710),
    (7, 1, 0.009, 0.664, -8.240, 9.662),
    (8, 1, 0.009, 0.664, -8.240, 10.515),
    (9, 1, 0.009, 0.655, -8.240, 11.293),
    (10, 1, 0.009, 0.616, -7.447, 12.009),
    (11, 1, 0.009, 0.597, -7.447, 12.625),
    (12, 1, 0.009, 0.578, -7.447, 13.210),
    (13, 1, 0.009, 0.541, -7.447, 13.748),
    (14, 1, 0.009, 0.575, -7.447, 14.241),
    (15, 1, 0.009, 0.856, -7.447, 14.695),
    (16, 1, 0.009, 0.846, -7.447, 15.125),
    (17, 1, 0.009, 0.840, -7.447, 15.508),
    (18, 1, 0.009, 0.822, -7.447, 15.891),
    (19, 2, 0.018, 0.800, -7.447, 16.537),
    (20, 2, 0.018, 0.753, -6.990, 17.112),
    (21, 2, 0.018, 0.704, -6.990, 17.620),
    (22, 2, 0.018, 0.674, -6.990, 18.073),
    (23, 2, 0.018, 0.640, -6.990, 18.470),
    (24, 2, 0.018, 0.609, -6.990, 18.849),
    (25, 3, 0.027, 0.566, -6.990, 19.271),
    (26, 3, 0.027, 0.535, -6.990, 19.741),
    (27, 3, 0.054, 0.531, -6.990, 20.177),
    (28, 3, 0.054, 0.615, -6.990, 20.576),
    (29, 3, 0.054, 0.686, -6.990, 20.950),
    (30, 4, 0.114, 0.650, -6.020, 21.316),
    (31, 4, 0.114, 0.612, -6.020, 21.699),
    (32, 5, 0.452, 0.567, -6.020, 22.078),
    (33, 5, 0.452, 0.520, -6.020, 22.438),
    (34, 5, 0.452, 0.513, -5.229, 22.782),
    (35, 7, 6.330, 0.557, -5.229, 23.133),
    (36, 7, 6.330, 0.584, -5.229, 23.484),
    (37, 7, 6.330, 0.570, -5.229, 23.828),
    (38, 19, 171.813, 0.578, -4.559, 24.173),
]);

// Table C.7.f, 32 kHz short blocks.

const SHORT_PARTITIONS_32000: [ShortPartition; 42] = short_partitions_from([
    (0, 1, 4.532, 0.997, -8.240, 0.000),
    (1, 1, 0.904, 0.893, -8.240, 1.250),
    (2, 1, 0.090, 0.881, -8.240, 2.500),
    (3, 1, 0.029, 0.873, -8.240, 3.750),
    (4, 1, 0.009, 0.872, -8.240, 4.909),
    (5, 1, 0.009, 0.871, -8.240, 5.958),
    (6, 1, 0.009, 0.860, -8.240, 6.857),
    (7, 1, 0.009, 0.839, -8.240, 7.700),
    (8, 1, 0.009, 0.812, -8.240, 8.500),
    (9, 1, 0.009, 0.784, -8.240, 9.237),
    (10, 1, 0.009, 0.741, -7.447, 9.895),
    (11, 1, 0.009, 0.697, -7.447, 10.500),
    (12, 1, 0.009, 0.674, -7.447, 11.083),
    (13, 1, 0.009, 0.651, -7.447, 11.604),
    (14, 1, 0.009, 0.633, -7.447, 12.107),
    (15, 1, 0.009, 0.611, -7.447, 12.554),
    (16, 1, 0.009, 0.589, -7.447, 13.000),
    (17, 1, 0.009, 0.575, -7.447, 13.391),
    (18, 1, 0.009, 0.654, -7.447, 13.781),
    (19, 2, 0.018, 0.724, -7.447, 14.474),
    (20, 2, 0.018, 0.701, -6.990, 15.096),
    (21, 2, 0.018, 0.673, -6.990, 15.667),
    (22, 2, 0.018, 0.631, -6.990, 16.177),
    (23, 2, 0.018, 0.592, -6.990, 16.636),
    (24, 2, 0.018, 0.553, -6.990, 17.057),
    (25, 2, 0.018, 0.510, -6.990, 17.429),
    (26, 2, 0.018, 0.506, -6.990, 17.786),
    (27, 3, 0.027, 0.562, -6.990, 18.177),
    (28, 3, 0.027, 0.598, -6.990, 18.597),
    (29, 3, 0.027, 0.589, -6.990, 18.994),
    (30, 3, 0.027, 0.607, -6.020, 19.352),
    (31, 3, 0.027, 0.620, -6.020, 19.693),
    (32, 4, 0.072, 0.580, -6.020, 20.066),
    (33, 4, 0.072, 0.532, -6.020, 20.461),
    (34, 4, 0.072, 0.517, -5.229, 20.841),
    (35, 5, 0.143, 0.517, -5.229, 21.201),
    (36, 5, 0.143, 0.509, -5.229, 21.549),
    (37, 6, 0.172, 0.506, -5.229, 21.911),
    (38, 7, 0.633, 0.522, -4.559, 22.275),
    (39, 7, 0.633, 0.531, -4.559, 22.625),
    (40, 8, 0.723, 0.519, -3.980, 22.971),
    (41, 10, 9.043, 0.512, -3.980, 23.321),
]);

// Table C.8.a, 48 kHz long blocks.

#[cfg(test)]
const BANDS_48000: [BandWeights; LONG_BANDS] = band_weights_from([
    (0, 0, 4, 1.000, 0.056),
    (1, 4, 7, 0.944, 0.611),
    (2, 7, 11, 0.389, 0.167),
    (3, 11, 14, 0.833, 0.722),
    (4, 14, 17, 0.278, 0.639),
    (5, 17, 19, 0.361, 0.417),
    (6, 19, 22, 0.583, 0.083),
    (7, 22, 24, 0.917, 0.750),
    (8, 24, 27, 0.250, 0.417),
    (9, 27, 30, 0.583, 0.648),
    (10, 30, 33, 0.352, 0.611),
    (11, 33, 36, 0.389, 0.625),
    (12, 36, 40, 0.375, 0.144),
    (13, 40, 43, 0.856, 0.389),
    (14, 43, 46, 0.611, 0.160),
    (15, 46, 49, 0.840, 0.217),
    (16, 49, 52, 0.783, 0.184),
    (17, 52, 54, 0.816, 0.886),
    (18, 54, 57, 0.114, 0.313),
    (19, 57, 59, 0.687, 0.452),
    (20, 59, 60, 0.548, 0.908),
]);

// Table C.8.b, 44.1 kHz long blocks.

#[cfg(test)]
const BANDS_44100: [BandWeights; LONG_BANDS] = band_weights_from([
    (0, 0, 4, 1.000, 0.056),
    (1, 4, 7, 0.944, 0.611),
    (2, 7, 11, 0.389, 0.167),
    (3, 11, 14, 0.833, 0.722),
    (4, 14, 17, 0.278, 0.139),
    (5, 17, 18, 0.861, 0.917),
    (6, 18, 21, 0.083, 0.583),
    (7, 21, 24, 0.417, 0.250),
    (8, 24, 27, 0.750, 0.805),
    (9, 27, 30, 0.194, 0.574),
    (10, 30, 33, 0.426, 0.537),
    (11, 33, 36, 0.463, 0.819),
    (12, 36, 40, 0.180, 0.100),
    (13, 40, 43, 0.900, 0.468),
    (14, 43, 46, 0.532, 0.623),
    (15, 46, 49, 0.376, 0.450),
    (16, 49, 52, 0.550, 0.552),
    (17, 52, 55, 0.448, 0.403),
    (18, 55, 57, 0.597, 0.643),
    (19, 57, 59, 0.357, 0.722),
    (20, 59, 61, 0.278, 0.960),
]);

// Table C.8.c, 32 kHz long blocks.

#[cfg(test)]
const BANDS_32000: [BandWeights; LONG_BANDS] = band_weights_from([
    (0, 0, 2, 1.000, 0.528),
    (1, 2, 4, 0.472, 0.305),
    (2, 4, 6, 0.694, 0.083),
    (3, 6, 7, 0.917, 0.861),
    (4, 7, 9, 0.139, 0.639),
    (5, 9, 11, 0.361, 0.417),
    (6, 11, 14, 0.583, 0.083),
    (7, 14, 16, 0.917, 0.750),
    (8, 16, 19, 0.250, 0.870),
    (9, 19, 22, 0.130, 0.833),
    (10, 22, 26, 0.167, 0.389),
    (11, 26, 30, 0.611, 0.478),
    (12, 30, 34, 0.522, 0.033),
    (13, 34, 37, 0.967, 0.917),
    (14, 37, 41, 0.083, 0.617),
    (15, 41, 44, 0.383, 0.995),
    (16, 44, 48, 0.005, 0.274),
    (17, 48, 51, 0.726, 0.480),
    (18, 51, 54, 0.519, 0.261),
    (19, 54, 56, 0.739, 0.884),
    (20, 56, 58, 0.116, 1.000),
]);

// Table C.8.d, 48 kHz short blocks.

#[cfg(test)]
const SHORT_BANDS_48000: [BandWeights; SHORT_BANDS] = band_weights_from([
    (0, 0, 3, 1.000, 0.167),
    (1, 3, 5, 0.833, 0.833),
    (2, 5, 8, 0.167, 0.500),
    (3, 8, 11, 0.500, 0.167),
    (4, 11, 15, 0.833, 0.167),
    (5, 15, 19, 0.833, 0.583),
    (6, 19, 22, 0.417, 0.917),
    (7, 22, 26, 0.083, 0.944),
    (8, 26, 30, 0.055, 0.042),
    (9, 30, 32, 0.958, 0.567),
    (10, 32, 35, 0.433, 0.167),
    (11, 35, 37, 0.833, 0.618),
]);

// Table C.8.e, 44.1 kHz short blocks.

#[cfg(test)]
const SHORT_BANDS_44100: [BandWeights; SHORT_BANDS] = band_weights_from([
    (0, 0, 3, 1.000, 0.167),
    (1, 3, 5, 0.833, 0.833),
    (2, 5, 8, 0.167, 0.500),
    (3, 8, 11, 0.500, 0.167),
    (4, 11, 15, 0.833, 0.167),
    (5, 15, 20, 0.833, 0.250),
    (6, 20, 23, 0.750, 0.583),
    (7, 23, 27, 0.417, 0.055),
    (8, 27, 30, 0.944, 0.375),
    (9, 30, 33, 0.625, 0.300),
    (10, 33, 36, 0.700, 0.167),
    (11, 36, 38, 0.833, 1.000),
]);

// Table C.8.f, 32 kHz short blocks.

#[cfg(test)]
const SHORT_BANDS_32000: [BandWeights; SHORT_BANDS] = band_weights_from([
    (0, 0, 3, 1.000, 0.167),
    (1, 3, 5, 0.833, 0.833),
    (2, 5, 8, 0.167, 0.500),
    (3, 8, 11, 0.500, 0.167),
    (4, 11, 15, 0.833, 0.167),
    (5, 15, 20, 0.833, 0.250),
    (6, 20, 24, 0.750, 0.250),
    (7, 24, 29, 0.750, 0.055),
    (8, 29, 33, 0.944, 0.375),
    (9, 33, 37, 0.625, 0.472),
    (10, 37, 40, 0.528, 0.937),
    (11, 40, 41, 0.062, 1.000),
]);

// Table D.3.a of ISO/IEC 13818-3, 24 kHz long blocks.

const PARTITIONS_24000: [Partition; 60] = partitions_from([
    (0, 2, 15.0, 17.8250179, 0.697374165, 0.236874461),
    (1, 2, 15.0, 17.8250179, 0.455024809, 0.71016103),
    (2, 2, 15.0, 1.78250182, 0.431440443, 1.18193281),
    (3, 2, 15.0, 1.78250182, 0.42391625, 1.65102732),
    (4, 2, 13.0, 0.178250194, 0.418206781, 2.11632562),
    (5, 2, 13.0, 0.178250194, 0.41158545, 2.57676744),
    (6, 2, 13.0, 0.0563676581, 0.405409157, 3.03136396),
    (7, 2, 13.0, 0.0563676581, 0.399695486, 3.47920918),
    (8, 2, 13.0, 0.0563676581, 0.393753231, 3.91948748),
    (9, 2, 12.0, 0.0178250186, 0.387357473, 4.35147953),
    (10, 2, 12.0, 0.0178250186, 0.38045457, 4.77456427),
    (11, 2, 10.0, 0.0178250186, 0.373053908, 5.18822002),
    (12, 2, 10.0, 0.0178250186, 0.365188122, 5.59202194),
    (13, 2, 10.0, 0.0178250186, 0.356897771, 5.98564005),
    (14, 2, 9.0, 0.0178250186, 0.348700613, 6.36883163),
    (15, 2, 9.0, 0.0178250186, 0.340260029, 6.74143791),
    (16, 2, 6.0, 0.0178250186, 0.332341045, 7.10337448),
    (17, 2, 6.0, 0.0178250186, 0.330462843, 7.45462418),
    (18, 2, 6.0, 0.0178250186, 0.345568359, 7.79523182),
    (19, 3, 3.0, 0.0267375279, 0.377859652, 8.20455742),
    (20, 3, 3.0, 0.0267375279, 0.396689415, 8.67640114),
    (21, 3, 3.0, 0.0267375279, 0.391237885, 9.12561035),
    (22, 3, 3.0, 0.0267375279, 0.37761277, 9.55298138),
    (23, 3, 3.0, 0.0267375279, 0.362836808, 9.95940971),
    (24, 3, 0.0, 0.0267375279, 0.349010617, 10.3458519),
    (25, 3, 0.0, 0.0267375279, 0.339673489, 10.7132998),
    (26, 3, 0.0, 0.0267375279, 0.343845725, 11.0627575),
    (27, 4, 0.0, 0.0356500372, 0.355822682, 11.447506),
    (28, 4, 0.0, 0.0356500372, 0.358104348, 11.8627586),
    (29, 4, 0.0, 0.0356500372, 0.34745428, 12.2520256),
    (30, 4, 0.0, 0.0356500372, 0.334927917, 12.6173973),
    (31, 4, 0.0, 0.0356500372, 0.331643254, 12.9608269),
    (32, 5, 0.0, 0.0445625484, 0.333368897, 13.3219252),
    (33, 5, 0.0, 0.0445625484, 0.332313001, 13.6976833),
    (34, 5, 0.0, 0.0445625484, 0.3314417, 14.047802),
    (35, 6, 0.0, 0.0534750558, 0.330947191, 14.405302),
    (36, 6, 0.0, 0.0534750558, 0.332477689, 14.7684803),
    (37, 7, 0.0, 0.062387567, 0.332647532, 15.1315956),
    (38, 7, 0.0, 0.062387567, 0.330841452, 15.4940481),
    (39, 8, 0.0, 0.0713000745, 0.327769846, 15.8516159),
    (40, 8, 0.0, 0.0713000745, 0.324572712, 16.204628),
    (41, 9, 0.0, 0.0802125856, 0.323825002, 16.5502281),
    (42, 10, 0.0, 0.0891250968, 0.321414798, 16.9067478),
    (43, 10, 0.0, 0.0891250968, 0.318189293, 17.2537231),
    (44, 11, 0.0, 0.0980376005, 0.315934151, 17.5901108),
    (45, 12, 0.0, 0.106950112, 0.315639287, 17.931406),
    (46, 13, 0.0, 0.115862623, 0.316569835, 18.2750721),
    (47, 14, 0.0, 0.124775134, 0.31656, 18.6191597),
    (48, 15, 0.0, 0.133687645, 0.315465957, 18.9621754),
    (49, 16, 0.0, 0.142600149, 0.313576341, 19.3029613),
    (50, 17, 0.0, 0.151512653, 0.311635971, 19.6405869),
    (51, 18, 0.0, 0.160425171, 0.311066717, 19.9742699),
    (52, 20, 0.0, 0.355655879, 0.311465651, 20.3115921),
    (53, 21, 0.0, 0.373438686, 0.311872005, 20.6507797),
    (54, 23, 0.0, 0.409004271, 0.311015964, 20.9890823),
    (55, 24, 0.0, 0.676411927, 0.309207708, 21.3251152),
    (56, 26, 0.0, 0.732779562, 0.3081128, 21.6565971),
    (57, 28, 0.0, 0.789147198, 0.310006589, 21.9881554),
    (58, 31, 0.0, 2.76287794, 0.327113092, 22.3222847),
    (59, 34, 0.0, 3.03025317, 0.416082352, 22.6605186),
]);

// Table D.3.b of ISO/IEC 13818-3, 22.05 kHz long blocks.

const PARTITIONS_22050: [Partition; 60] = partitions_from([
    (0, 2, 15.0, 17.8250179, 0.658683598, 0.217637643),
    (1, 2, 15.0, 17.8250179, 0.432554901, 0.652563453),
    (2, 2, 15.0, 1.78250182, 0.405113578, 1.08633137),
    (3, 2, 15.0, 1.78250182, 0.397231787, 1.51803517),
    (4, 2, 15.0, 1.78250182, 0.392088681, 1.94679713),
    (5, 2, 13.0, 0.178250194, 0.386788279, 2.37177849),
    (6, 2, 13.0, 0.178250194, 0.380574644, 2.79218864),
    (7, 2, 13.0, 0.0563676581, 0.375309765, 3.20729256),
    (8, 2, 13.0, 0.0563676581, 0.370087624, 3.61641645),
    (9, 2, 12.0, 0.0178250186, 0.364568561, 4.01895428),
    (10, 2, 12.0, 0.0178250186, 0.358959526, 4.4143672),
    (11, 2, 12.0, 0.0178250186, 0.352938265, 4.80218887),
    (12, 2, 10.0, 0.0178250186, 0.3465029, 5.18202305),
    (13, 2, 10.0, 0.0178250186, 0.33968094, 5.55354261),
    (14, 2, 10.0, 0.0178250186, 0.332571507, 5.91648674),
    (15, 2, 9.0, 0.0178250186, 0.326015651, 6.27065945),
    (16, 2, 9.0, 0.0178250186, 0.325442046, 6.61592293),
    (17, 2, 9.0, 0.0178250186, 0.341315031, 6.95219517),
    (18, 3, 6.0, 0.0267375279, 0.374984443, 7.3584404),
    (19, 3, 6.0, 0.0267375279, 0.396138102, 7.8290925),
    (20, 3, 3.0, 0.0267375279, 0.39271906, 8.27975655),
    (21, 3, 3.0, 0.0267375279, 0.380755007, 8.71083069),
    (22, 3, 3.0, 0.0267375279, 0.367386311, 9.12284088),
    (23, 3, 3.0, 0.0267375279, 0.354351997, 9.51640987),
    (24, 3, 3.0, 0.0267375279, 0.341508389, 9.89222908),
    (25, 3, 0.0, 0.0267375279, 0.333577901, 10.2510386),
    (26, 3, 0.0, 0.0267375279, 0.338108748, 10.5936022),
    (27, 4, 0.0, 0.0356500372, 0.350744486, 10.9723492),
    (28, 4, 0.0, 0.0356500372, 0.354519457, 11.38272),
    (29, 4, 0.0, 0.0356500372, 0.345274031, 11.7689981),
    (30, 4, 0.0, 0.0356500372, 0.333828837, 12.1329184),
    (31, 4, 0.0, 0.0356500372, 0.331436664, 12.4761295),
    (32, 5, 0.0, 0.0445625484, 0.334172577, 12.8381901),
    (33, 5, 0.0, 0.0445625484, 0.334024847, 13.2160273),
    (34, 5, 0.0, 0.0445625484, 0.33392629, 13.5690479),
    (35, 6, 0.0, 0.0534750558, 0.334218502, 13.9303951),
    (36, 6, 0.0, 0.0534750558, 0.336405039, 14.298193),
    (37, 7, 0.0, 0.062387567, 0.337080389, 14.666563),
    (38, 7, 0.0, 0.062387567, 0.335603535, 15.0346909),
    (39, 8, 0.0, 0.0713000745, 0.332515866, 15.398139),
    (40, 8, 0.0, 0.0713000745, 0.327727586, 15.7570457),
    (41, 9, 0.0, 0.0802125856, 0.322346836, 16.1083431),
    (42, 9, 0.0, 0.0802125856, 0.317575186, 16.4528522),
    (43, 10, 0.0, 0.0891250968, 0.31632933, 16.7886105),
    (44, 11, 0.0, 0.0980376005, 0.317602783, 17.132),
    (45, 12, 0.0, 0.106950112, 0.319945186, 17.4796028),
    (46, 13, 0.0, 0.115862623, 0.320881754, 17.8287659),
    (47, 14, 0.0, 0.124775134, 0.320346534, 18.1774921),
    (48, 15, 0.0, 0.133687645, 0.318628669, 18.5243168),
    (49, 16, 0.0, 0.142600149, 0.316125751, 18.8681736),
    (50, 17, 0.0, 0.151512653, 0.313746184, 19.2082729),
    (51, 18, 0.0, 0.160425171, 0.312971771, 19.5440025),
    (52, 20, 0.0, 0.178250194, 0.313278913, 19.8831882),
    (53, 21, 0.0, 0.373438686, 0.313735574, 20.224247),
    (54, 23, 0.0, 0.409004271, 0.31308493, 20.5646286),
    (55, 24, 0.0, 0.426787049, 0.31156227, 20.903141),
    (56, 26, 0.0, 0.732779562, 0.310435742, 21.2376747),
    (57, 28, 0.0, 0.789147198, 0.31132248, 21.5730591),
    (58, 30, 0.0, 0.845514894, 0.32730341, 21.9066811),
    (59, 33, 0.0, 2.94112802, 0.414659739, 22.2411156),
]);

// Table D.3.c of ISO/IEC 13818-3, 16 kHz long blocks.

const PARTITIONS_16000: [Partition; 56] = partitions_from([
    (0, 3, 15.0, 26.7375278, 0.697374165, 0.236874461),
    (1, 3, 15.0, 26.7375278, 0.455024809, 0.71016103),
    (2, 3, 15.0, 2.67375278, 0.431440443, 1.18193281),
    (3, 3, 15.0, 2.67375278, 0.42391625, 1.65102732),
    (4, 3, 13.0, 0.26737529, 0.418206781, 2.11632562),
    (5, 3, 13.0, 0.26737529, 0.41158545, 2.57676744),
    (6, 3, 13.0, 0.0845514908, 0.405409157, 3.03136396),
    (7, 3, 13.0, 0.0845514908, 0.399695486, 3.47920918),
    (8, 3, 13.0, 0.0845514908, 0.393753231, 3.91948748),
    (9, 3, 12.0, 0.0267375279, 0.387357473, 4.35147953),
    (10, 3, 12.0, 0.0267375279, 0.38045457, 4.77456427),
    (11, 3, 10.0, 0.0267375279, 0.373053908, 5.18822002),
    (12, 3, 10.0, 0.0267375279, 0.365188122, 5.59202194),
    (13, 3, 10.0, 0.0267375279, 0.356897742, 5.98564005),
    (14, 3, 9.0, 0.0267375279, 0.34869957, 6.36883163),
    (15, 3, 9.0, 0.0267375279, 0.340241522, 6.74143791),
    (16, 3, 6.0, 0.0267375279, 0.332089454, 7.10337448),
    (17, 3, 6.0, 0.0267375279, 0.328292668, 7.45462418),
    (18, 3, 6.0, 0.0267375279, 0.336574793, 7.79523182),
    (19, 4, 3.0, 0.0356500372, 0.354600489, 8.17827797),
    (20, 4, 3.0, 0.0356500372, 0.364343345, 8.59994984),
    (21, 4, 3.0, 0.0356500372, 0.359369367, 9.00363636),
    (22, 4, 3.0, 0.0356500372, 0.347775847, 9.38988018),
    (23, 4, 3.0, 0.0356500372, 0.335562587, 9.7592926),
    (24, 4, 0.0, 0.0356500372, 0.326988578, 10.1125278),
    (25, 4, 0.0, 0.0356500372, 0.327966213, 10.4502735),
    (26, 5, 0.0, 0.0445625484, 0.334450752, 10.811614),
    (27, 5, 0.0, 0.0445625484, 0.335228145, 11.1935263),
    (28, 5, 0.0, 0.0445625484, 0.329595625, 11.5549288),
    (29, 5, 0.0, 0.0445625484, 0.326683223, 11.8971443),
    (30, 6, 0.0, 0.0534750558, 0.326986551, 12.2520256),
    (31, 6, 0.0, 0.0534750558, 0.325072199, 12.6173973),
    (32, 6, 0.0, 0.0534750558, 0.323560268, 12.9608269),
    (33, 7, 0.0, 0.062387567, 0.322494298, 13.3093863),
    (34, 7, 0.0, 0.062387567, 0.323403448, 13.6617231),
    (35, 8, 0.0, 0.0713000745, 0.323232353, 14.0134668),
    (36, 8, 0.0, 0.0713000745, 0.322662383, 14.3639784),
    (37, 9, 0.0, 0.0802125856, 0.324054241, 14.7098465),
    (38, 10, 0.0, 0.0891250968, 0.323228806, 15.0686541),
    (39, 10, 0.0, 0.0891250968, 0.320751846, 15.4191036),
    (40, 11, 0.0, 0.0980376005, 0.318823338, 15.7594051),
    (41, 12, 0.0, 0.106950112, 0.318418682, 16.104557),
    (42, 13, 0.0, 0.115862623, 0.318762124, 16.451416),
    (43, 14, 0.0, 0.124775134, 0.317806393, 16.7975388),
    (44, 15, 0.0, 0.133687645, 0.315653771, 17.1411018),
    (45, 16, 0.0, 0.142600149, 0.313369036, 17.4808159),
    (46, 17, 0.0, 0.151512653, 0.312513858, 17.8158207),
    (47, 19, 0.0, 0.169337675, 0.312785119, 18.1543369),
    (48, 20, 0.0, 0.178250194, 0.31343773, 18.4948578),
    (49, 22, 0.0, 0.196075201, 0.313258767, 18.8350143),
    (50, 23, 0.0, 0.20498772, 0.312570423, 19.1740704),
    (51, 25, 0.0, 0.222812727, 0.312572777, 19.5104179),
    (52, 27, 0.0, 0.240637749, 0.313047856, 19.8497677),
    (53, 29, 0.0, 0.515701056, 0.315029174, 20.1900635),
    (54, 31, 0.0, 0.551266611, 0.330613613, 20.5294952),
    (55, 33, 0.0, 0.586832225, 0.41819948, 20.8664398),
]);

// Table D.3.d of ISO/IEC 13818-3, 24 kHz short blocks.

const SHORT_PARTITIONS_24000: [ShortPartition; 45] = short_partitions_from_ratios([
    (0, 1, 8.91250896, 0.971850038, 0.150000006, 0.0),
    (1, 1, 8.91250896, 0.874727964, 0.150000006, 0.946573138),
    (2, 1, 0.891250908, 0.85779953, 0.150000006, 1.88476217),
    (3, 1, 0.0891250968, 0.839743853, 0.150000006, 2.8056457),
    (4, 1, 0.028183829, 0.82260257, 0.150000006, 3.70133615),
    (5, 1, 0.00891250931, 0.80018574, 0.150000006, 4.56532001),
    (6, 1, 0.00891250931, 0.771475196, 0.150000006, 5.39263105),
    (7, 1, 0.00891250931, 0.737389982, 0.150000006, 6.17986727),
    (8, 1, 0.00891250931, 0.701111019, 0.150000006, 6.92507982),
    (9, 1, 0.00891250931, 0.65977633, 0.150000006, 7.62757969),
    (10, 1, 0.00891250931, 0.615037441, 0.150000006, 8.28770351),
    (11, 1, 0.00891250931, 0.568658054, 0.150000006, 8.90657234),
    (12, 1, 0.00891250931, 0.522260666, 0.180000007, 9.48587132),
    (13, 1, 0.00891250931, 0.478903115, 0.180000007, 10.0276566),
    (14, 1, 0.00891250931, 0.43808648, 0.180000007, 10.5341988),
    (15, 1, 0.00891250931, 0.412505627, 0.180000007, 11.0078659),
    (16, 1, 0.00891250931, 0.39070797, 0.180000007, 11.4510288),
    (17, 1, 0.00891250931, 0.371887118, 0.180000007, 11.866004),
    (18, 1, 0.00891250931, 0.367617637, 0.180000007, 12.2550087),
    (19, 1, 0.00891250931, 0.422220588, 0.180000007, 12.6201363),
    (20, 2, 0.0178250186, 0.564990044, 0.180000007, 13.2772083),
    (21, 2, 0.0178250186, 0.519700944, 0.180000007, 13.871047),
    (22, 2, 0.0178250186, 0.455360681, 0.200000003, 14.4024391),
    (23, 2, 0.0178250186, 0.408867925, 0.200000003, 14.8811684),
    (24, 2, 0.0178250186, 0.381538749, 0.200000003, 15.3153324),
    (25, 2, 0.0178250186, 0.362357527, 0.200000003, 15.7116165),
    (26, 2, 0.0178250186, 0.365735918, 0.200000003, 16.0755405),
    (27, 3, 0.0267375279, 0.38064, 0.200000003, 16.4882088),
    (28, 3, 0.0267375279, 0.379183382, 0.200000003, 16.9410992),
    (29, 3, 0.0267375279, 0.360672712, 0.200000003, 17.3513336),
    (30, 3, 0.0267375279, 0.343065977, 0.200000003, 17.7264423),
    (31, 3, 0.0267375279, 0.339290261, 0.200000003, 18.0722466),
    (32, 4, 0.0356500372, 0.342963994, 0.200000003, 18.4426575),
    (33, 4, 0.0356500372, 0.343128443, 0.200000003, 18.8344078),
    (34, 4, 0.0356500372, 0.343988508, 0.25, 19.1955795),
    (35, 5, 0.0445625484, 0.343928397, 0.25, 19.5697021),
    (36, 5, 0.0445625484, 0.339527696, 0.25, 19.9551182),
    (37, 5, 0.0889139697, 0.336541563, 0.280000001, 20.3115921),
    (38, 6, 0.106696762, 0.334955156, 0.280000001, 20.6737747),
    (39, 6, 0.169102982, 0.335601568, 0.300000012, 21.0404968),
    (40, 7, 0.1972868, 0.334716886, 0.300000012, 21.4060211),
    (41, 7, 0.1972868, 0.331676662, 0.300000012, 21.7696877),
    (42, 8, 0.713000774, 0.328550965, 0.400000006, 22.1267223),
    (43, 8, 0.713000774, 0.339241952, 0.400000006, 22.4769249),
    (44, 9, 0.802125871, 0.425207615, 0.400000006, 22.8164864),
]);

// Table D.3.e of ISO/IEC 13818-3, 22.05 kHz short blocks.

const SHORT_PARTITIONS_22050: [ShortPartition; 45] = short_partitions_from_ratios([
    (0, 1, 8.91250896, 0.954045713, 0.150000006, 0.0),
    (1, 1, 8.91250896, 0.833381653, 0.150000006, 0.869851649),
    (2, 1, 0.891250908, 0.815945923, 0.150000006, 1.73325908),
    (3, 1, 0.0891250968, 0.794244766, 0.150000006, 2.58322191),
    (4, 1, 0.028183829, 0.776486695, 0.150000006, 3.4134295),
    (5, 1, 0.00891250931, 0.755260408, 0.150000006, 4.21850443),
    (6, 1, 0.00891250931, 0.731070817, 0.150000006, 4.99414825),
    (7, 1, 0.00891250931, 0.701775849, 0.150000006, 5.73718691),
    (8, 1, 0.00891250931, 0.667876124, 0.150000006, 6.44553185),
    (9, 1, 0.00891250931, 0.630284071, 0.150000006, 7.11807632),
    (10, 1, 0.00891250931, 0.590170324, 0.150000006, 7.75455618),
    (11, 1, 0.00891250931, 0.548788548, 0.150000006, 8.3553915),
    (12, 1, 0.00891250931, 0.507795513, 0.150000006, 8.92152882),
    (13, 1, 0.00891250931, 0.469515711, 0.180000007, 9.45430183),
    (14, 1, 0.00891250931, 0.432291716, 0.180000007, 9.95530319),
    (15, 1, 0.00891250931, 0.411131173, 0.180000007, 10.4262848),
    (16, 1, 0.00891250931, 0.390771538, 0.180000007, 10.8690758),
    (17, 1, 0.00891250931, 0.373318017, 0.180000007, 11.2855215),
    (18, 1, 0.00891250931, 0.36956048, 0.180000007, 11.6774378),
    (19, 1, 0.00891250931, 0.42595759, 0.180000007, 12.0465794),
    (20, 2, 0.0178250186, 0.576900065, 0.180000007, 12.7141209),
    (21, 2, 0.0178250186, 0.533114731, 0.180000007, 13.3197365),
    (22, 2, 0.0178250186, 0.469967514, 0.180000007, 13.8634901),
    (23, 2, 0.0178250186, 0.417268544, 0.200000003, 14.3544445),
    (24, 2, 0.0178250186, 0.389299124, 0.200000003, 14.8002586),
    (25, 2, 0.0178250186, 0.362824857, 0.200000003, 15.2073727),
    (26, 2, 0.0178250186, 0.346801281, 0.200000003, 15.5811834),
    (27, 2, 0.0178250186, 0.349400043, 0.200000003, 15.926218),
    (28, 3, 0.0267375279, 0.364026934, 0.200000003, 16.3194923),
    (29, 3, 0.0267375279, 0.36560446, 0.200000003, 16.752903),
    (30, 3, 0.0267375279, 0.354275256, 0.200000003, 17.1470814),
    (31, 3, 0.0267375279, 0.351219416, 0.200000003, 17.5086212),
    (32, 4, 0.0356500372, 0.354364097, 0.200000003, 17.8938141),
    (33, 4, 0.0356500372, 0.348915905, 0.200000003, 18.2992878),
    (34, 4, 0.0356500372, 0.337649345, 0.200000003, 18.6713982),
    (35, 4, 0.0356500372, 0.332076877, 0.25, 19.015646),
    (36, 5, 0.0445625484, 0.330793113, 0.25, 19.3734016),
    (37, 5, 0.0445625484, 0.327528268, 0.25, 19.7430382),
    (38, 5, 0.0889139697, 0.32551071, 0.280000001, 20.0859604),
    (39, 6, 0.106696762, 0.324436843, 0.280000001, 20.4354992),
    (40, 6, 0.106696762, 0.325835049, 0.280000001, 20.7905579),
    (41, 7, 0.1972868, 0.326221824, 0.300000012, 21.1458054),
    (42, 7, 0.1972868, 0.325960994, 0.300000012, 21.5005951),
    (43, 8, 0.225470632, 0.339019388, 0.300000012, 21.8504524),
    (44, 8, 0.713000774, 0.426850349, 0.400000006, 22.1951065),
]);

// Table D.3.f of ISO/IEC 13818-3, 16 kHz short blocks.

const SHORT_PARTITIONS_16000: [ShortPartition; 46] = short_partitions_from_ratios([
    (0, 1, 8.91250896, 0.834739447, 0.150000006, 0.0),
    (1, 1, 8.91250896, 0.623757005, 0.150000006, 0.631518543),
    (2, 1, 0.891250908, 0.60420388, 0.150000006, 1.2606914),
    (3, 1, 0.891250908, 0.591974258, 0.150000006, 1.88476217),
    (4, 1, 0.0891250968, 0.575301588, 0.150000006, 2.50111985),
    (5, 1, 0.028183829, 0.561547697, 0.150000006, 3.1073606),
    (6, 1, 0.028183829, 0.546665847, 0.150000006, 3.70133615),
    (7, 1, 0.00891250931, 0.52986443, 0.150000006, 4.28118753),
    (8, 1, 0.00891250931, 0.511183441, 0.150000006, 4.84536505),
    (9, 1, 0.00891250931, 0.490902334, 0.150000006, 5.39263105),
    (10, 1, 0.00891250931, 0.46938166, 0.150000006, 5.92205667),
    (11, 1, 0.00891250931, 0.447003782, 0.150000006, 6.43299866),
    (12, 1, 0.00891250931, 0.428170592, 0.150000006, 6.92507982),
    (13, 1, 0.00891250931, 0.414536625, 0.150000006, 7.39815664),
    (14, 1, 0.00891250931, 0.401033074, 0.150000006, 7.85228777),
    (15, 1, 0.00891250931, 0.38779071, 0.150000006, 8.28770351),
    (16, 1, 0.00891250931, 0.374230444, 0.150000006, 8.704772),
    (17, 1, 0.00891250931, 0.360547513, 0.180000007, 9.10397339),
    (18, 1, 0.00891250931, 0.348256677, 0.180000007, 9.48587132),
    (19, 1, 0.00891250931, 0.350327015, 0.180000007, 9.85109234),
    (20, 1, 0.00891250931, 0.406330824, 0.180000007, 10.200304),
    (21, 2, 0.0178250186, 0.554098248, 0.180000007, 10.846529),
    (22, 2, 0.0178250186, 0.528312504, 0.180000007, 11.4447651),
    (23, 2, 0.0178250186, 0.476527005, 0.180000007, 11.9928398),
    (24, 2, 0.0178250186, 0.428205669, 0.180000007, 12.495945),
    (25, 2, 0.0178250186, 0.402271926, 0.180000007, 12.9588718),
    (26, 2, 0.0178250186, 0.378024429, 0.180000007, 13.3859692),
    (27, 2, 0.0178250186, 0.36254698, 0.180000007, 13.7811394),
    (28, 2, 0.0178250186, 0.368058592, 0.200000003, 14.1478529),
    (29, 3, 0.0267375279, 0.385963261, 0.200000003, 14.5674343),
    (30, 3, 0.0267375279, 0.38640517, 0.200000003, 15.0304852),
    (31, 3, 0.0267375279, 0.367834061, 0.200000003, 15.4513416),
    (32, 3, 0.0267375279, 0.349686563, 0.200000003, 15.836277),
    (33, 3, 0.0267375279, 0.345709383, 0.200000003, 16.1904697),
    (34, 4, 0.0356500372, 0.34871915, 0.200000003, 16.5683517),
    (35, 4, 0.0356500372, 0.347054332, 0.200000003, 16.9660263),
    (36, 4, 0.0356500372, 0.346329987, 0.200000003, 17.3304482),
    (37, 5, 0.0445625484, 0.344658494, 0.200000003, 17.7055588),
    (38, 5, 0.0445625484, 0.338779271, 0.200000003, 18.0899811),
    (39, 5, 0.0445625484, 0.334878683, 0.200000003, 18.4440536),
    (40, 6, 0.0534750558, 0.332811534, 0.200000003, 18.8030052),
    (41, 6, 0.0534750558, 0.333717585, 0.25, 19.1665268),
    (42, 7, 0.062387567, 0.333986402, 0.25, 19.5299358),
    (43, 7, 0.062387567, 0.334142625, 0.25, 19.8934898),
    (44, 8, 0.142262354, 0.34677428, 0.280000001, 20.2535706),
    (45, 8, 0.142262354, 0.436254472, 0.280000001, 20.610569),
]);

// Table D.4.a of ISO/IEC 13818-3, 24 kHz long blocks.

#[cfg(test)]
const BANDS_24000: [BandWeights; LONG_BANDS] = band_weights_from([
    (0, 0, 3, 1.0, 0.916666746),
    (1, 3, 6, 0.083333254, 0.583333492),
    (2, 6, 9, 0.416666508, 0.25),
    (3, 9, 11, 0.75, 0.916666985),
    (4, 11, 14, 0.083333015, 0.583333969),
    (5, 14, 17, 0.416666031, 0.25),
    (6, 17, 20, 0.75, 0.537036896),
    (7, 20, 23, 0.462963104, 0.5),
    (8, 23, 27, 0.5, 0.055556000),
    (9, 27, 30, 0.944444001, 0.402778625),
    (10, 30, 33, 0.597221375, 0.766667187),
    (11, 33, 36, 0.233332828, 0.805555999),
    (12, 36, 39, 0.194444016, 0.769841909),
    (13, 39, 42, 0.23015812, 0.611111104),
    (14, 42, 45, 0.388888896, 0.449494779),
    (15, 45, 48, 0.550505221, 0.194444954),
    (16, 48, 50, 0.805555046, 0.913194656),
    (17, 50, 53, 0.086805344, 0.580555737),
    (18, 53, 56, 0.419444263, 0.113426208),
    (19, 56, 58, 0.886573792, 0.533730626),
    (20, 58, 60, 0.466269344, 0.691176474),
]);

// Table D.4.b of ISO/IEC 13818-3, 22.05 kHz long blocks.
//
// Spec reading: `w2` of band 13 is printed "0.09259373ß", with a "ß" where
// its ninth digit belongs. It is read as 0.09259373, at most 1e-8 off,
// which adds up to 1 with `w1` of band 14, as the weights of every other
// partition shared by two bands do.

#[cfg(test)]
const BANDS_22050: [BandWeights; LONG_BANDS] = band_weights_from([
    (0, 0, 3, 1.0, 0.916666746),
    (1, 3, 6, 0.083333254, 0.583333492),
    (2, 6, 9, 0.416666508, 0.25),
    (3, 9, 11, 0.75, 0.916666985),
    (4, 11, 14, 0.083333015, 0.583333969),
    (5, 14, 17, 0.416666031, 0.25),
    (6, 17, 20, 0.75, 0.203703582),
    (7, 20, 23, 0.796296418, 0.166666687),
    (8, 23, 26, 0.833333313, 0.722222686),
    (9, 26, 30, 0.277777344, 0.152778625),
    (10, 30, 33, 0.847221375, 0.566667199),
    (11, 33, 36, 0.433332831, 0.93518573),
    (12, 36, 40, 0.064814247, 0.118056297),
    (13, 40, 43, 0.881943703, 0.09259373),
    (14, 43, 45, 0.907406271, 0.934344172),
    (15, 45, 48, 0.065655798, 0.575398028),
    (16, 48, 51, 0.424601972, 0.232026935),
    (17, 51, 53, 0.767973065, 0.758334339),
    (18, 53, 56, 0.241665646, 0.187501252),
    (19, 56, 58, 0.812498748, 0.533731699),
    (20, 58, 60, 0.466268271, 0.257577598),
]);

// Table D.4.c of ISO/IEC 13818-3, 16 kHz long blocks.

#[cfg(test)]
const BANDS_16000: [BandWeights; LONG_BANDS] = band_weights_from([
    (0, 0, 2, 1.0, 0.944444478),
    (1, 2, 4, 0.055555504, 0.722222328),
    (2, 4, 6, 0.277777672, 0.5),
    (3, 6, 8, 0.5, 0.27777797),
    (4, 8, 10, 0.72222203, 0.055555994),
    (5, 10, 11, 0.944444001, 0.833333313),
    (6, 11, 14, 0.166666672, 0.203703582),
    (7, 14, 17, 0.796296418, 0.166666687),
    (8, 17, 20, 0.833333313, 0.54166698),
    (9, 20, 23, 0.458333015, 0.652778625),
    (10, 23, 27, 0.347221375, 0.166667163),
    (11, 27, 30, 0.833332837, 0.722222924),
    (12, 30, 34, 0.277777106, 0.277778625),
    (13, 34, 37, 0.722221375, 0.604167938),
    (14, 37, 40, 0.395832062, 0.627778649),
    (15, 40, 43, 0.37222138, 0.542736351),
    (16, 43, 46, 0.457263649, 0.371528625),
    (17, 46, 49, 0.628471375, 0.008334339),
    (18, 49, 51, 0.991665661, 0.500001311),
    (19, 51, 53, 0.499998659, 0.886832893),
    (20, 53, 55, 0.113167092, 0.629034221),
]);

// Table D.4.d of ISO/IEC 13818-3, 24 kHz short blocks.

#[cfg(test)]
const SHORT_BANDS_24000: [BandWeights; SHORT_BANDS] = band_weights_from([
    (0, 0, 4, 1.0, 0.166666746),
    (1, 4, 6, 0.833333254, 0.833333492),
    (2, 6, 9, 0.166666508, 0.5),
    (3, 9, 13, 0.5, 0.5),
    (4, 13, 18, 0.5, 0.833333969),
    (5, 18, 23, 0.166666031, 0.25),
    (6, 23, 27, 0.75, 0.25),
    (7, 27, 30, 0.75, 0.611111999),
    (8, 30, 34, 0.388888031, 0.208333969),
    (9, 34, 37, 0.791666031, 0.766667187),
    (10, 37, 41, 0.233332828, 0.45238167),
    (11, 41, 45, 0.54761833, 0.277778625),
]);

// Table D.4.e of ISO/IEC 13818-3, 22.05 kHz short blocks.

#[cfg(test)]
const SHORT_BANDS_22050: [BandWeights; SHORT_BANDS] = band_weights_from([
    (0, 0, 4, 1.0, 0.166666746),
    (1, 4, 6, 0.833333254, 0.833333492),
    (2, 6, 9, 0.166666508, 0.5),
    (3, 9, 13, 0.5, 0.5),
    (4, 13, 17, 0.5, 0.5),
    (5, 17, 21, 0.5, 0.916666985),
    (6, 21, 25, 0.083333015, 0.25),
    (7, 25, 29, 0.75, 0.611111999),
    (8, 29, 33, 0.388888031, 0.458333969),
    (9, 33, 37, 0.541666031, 0.633334339),
    (10, 37, 41, 0.366665661, 0.583334565),
    (11, 41, 45, 0.416665405, 0.437500954),
]);

// Table D.4.f of ISO/IEC 13818-3, 16 kHz short blocks.

#[cfg(test)]
const SHORT_BANDS_16000: [BandWeights; SHORT_BANDS] = band_weights_from([
    (0, 0, 4, 1.0, 0.166666746),
    (1, 4, 6, 0.833333254, 0.833333492),
    (2, 6, 9, 0.166666508, 0.5),
    (3, 9, 13, 0.5, 0.5),
    (4, 13, 18, 0.5, 0.833333969),
    (5, 18, 23, 0.166666031, 0.75),
    (6, 23, 27, 0.25, 0.75),
    (7, 27, 31, 0.25, 0.611111999),
    (8, 31, 35, 0.388888031, 0.458333969),
    (9, 35, 39, 0.541666031, 0.166667163),
    (10, 39, 42, 0.833332837, 0.805555999),
    (11, 42, 46, 0.194444016, 0.4375),
]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::Version;

    fn mpeg1() -> impl Iterator<Item = SamplingFrequency> {
        SamplingFrequency::ALL
            .into_iter()
            .filter(|fs| fs.version() == Version::Mpeg1)
    }
    use crate::psychoacoustic::spreading_function;

    /// Table C.8 or Table D.4 of ISO/IEC 13818-3 for `fs`'s long and short
    /// blocks, as printed.
    fn printed(fs: SamplingFrequency) -> (&'static [BandWeights], &'static [BandWeights]) {
        match fs {
            SamplingFrequency::Hz48000 => (&BANDS_48000, &SHORT_BANDS_48000),
            SamplingFrequency::Hz44100 => (&BANDS_44100, &SHORT_BANDS_44100),
            SamplingFrequency::Hz32000 => (&BANDS_32000, &SHORT_BANDS_32000),
            SamplingFrequency::Hz24000 => (&BANDS_24000, &SHORT_BANDS_24000),
            SamplingFrequency::Hz22050 => (&BANDS_22050, &SHORT_BANDS_22050),
            SamplingFrequency::Hz16000 => (&BANDS_16000, &SHORT_BANDS_16000),
            _ => panic!("{fs:?} has no printed table"),
        }
    }

    /// `norm` is printed without its derivation, but in Tables C.7.a to
    /// C.7.c it matches the normalisation `rnorm` of clause D.2.4 f), the
    /// reciprocal of the spreading function summed over all partitions, to
    /// within the rounding of the printed Bark values. This checks both
    /// columns. (The column of Tables D.3.a to D.3.c of ISO/IEC 13818-3 lies
    /// 1 to 3 dB below it, and is used as printed.)
    #[test]
    fn norm_is_the_reciprocal_of_the_spreading_sum() {
        for fs in mpeg1() {
            let partitions = partitions(fs);
            for (b, p) in partitions.iter().enumerate() {
                let sum: f64 = partitions
                    .iter()
                    .map(|q| spreading_function(q.bval, p.bval))
                    .sum();
                let norm = 1.0 / sum;
                assert!(
                    (norm - p.norm).abs() < 0.002,
                    "{fs:?} partition {b}: {norm} != {}",
                    p.norm
                );
            }
        }
    }

    /// The short tables' `norm` column of ISO/IEC 11172-3 repeats the long
    /// one (see [`short_partitions`]) to within a rounding step. This checks
    /// that reading of the column.
    #[test]
    fn short_norm_repeats_the_long_column() {
        for fs in mpeg1() {
            for (short, long) in short_partitions(fs).iter().zip(partitions(fs)) {
                assert!((short.norm - long.norm).abs() <= 0.0015, "{fs:?}");
            }
        }
    }

    /// Each partition's threshold in quiet is its width times one of a few
    /// per-line levels, to within the rounding of the printed values.
    #[test]
    fn qthr_is_a_per_line_level_times_the_width() {
        const LEVELS: [f64; 13] = [
            // ISO/IEC 11172-3
            0.009, 0.018, 0.0286, 0.0904, 0.904, 4.532, 9.043, // ISO/IEC 13818-3
            0.008913, 0.017783, 0.028184, 0.089125, 0.891251, 8.912509,
        ];
        for fs in SamplingFrequency::ALL {
            let long = partitions(fs).iter().map(|p| (p.width, p.qthr));
            let short = short_partitions(fs).iter().map(|p| (p.width, p.qthr));
            for (b, (width, qthr)) in long.chain(short).enumerate() {
                assert!(
                    LEVELS
                        .iter()
                        .any(|level| (level * width as f64 - qthr).abs() <= 0.0005 + 0.005 * qthr),
                    "{fs:?} partition {b} of the long and short tables: qthr {qthr} over {width} lines"
                );
            }
        }
    }

    /// `minval` falls from low to high frequencies through a few values.
    #[test]
    fn minval_steps_down() {
        for fs in SamplingFrequency::ALL {
            let partitions = partitions(fs);
            for (b, pair) in partitions.windows(2).enumerate() {
                assert!(
                    pair[1].minval <= pair[0].minval,
                    "{fs:?} partition {}",
                    b + 1
                );
            }
            for p in partitions {
                let minvals = match fs.version() {
                    Version::Mpeg1 => [0.0, 3.0, 6.0, 12.0, 18.0, 20.0, 24.5].as_slice(),
                    Version::Mpeg2 => &[0.0, 3.0, 6.0, 9.0, 10.0, 12.0, 13.0, 15.0],
                };
                assert!(minvals.contains(&p.minval), "{fs:?}");
            }
        }
    }

    /// The short blocks' SNR rises from low to high frequencies through a
    /// few values, the same few in both standards but for -5.528 dB, a ratio
    /// of 0.28, in ISO/IEC 13818-3 alone.
    #[test]
    fn short_snr_steps_up() {
        const SNR: [f64; 8] = [
            -8.240, -7.447, -6.990, -6.020, -5.528, -5.229, -4.559, -3.980,
        ];
        for fs in SamplingFrequency::ALL {
            let partitions = short_partitions(fs);
            for (b, pair) in partitions.windows(2).enumerate() {
                assert!(
                    pair[1].snr.db() >= pair[0].snr.db(),
                    "{fs:?} partition {}",
                    b + 1
                );
            }
            assert!(
                (partitions.iter()).all(|p| SNR.iter().any(|db| (db - p.snr.db()).abs() < 0.001)),
                "{fs:?}"
            );
        }
    }

    /// A partition shared by two neighbouring bands is split between them:
    /// the weights it gets in each add up to one, to the printed precision.
    #[test]
    fn shared_partitions_are_split_between_bands() {
        for fs in SamplingFrequency::STANDARD {
            let (long, short) = printed(fs);
            let tables: [(&[BandWeights], usize); 2] = [
                (long, partitions(fs).len()),
                (short, short_partitions(fs).len()),
            ];
            for (bands, partitions) in tables {
                assert_eq!(bands[0].w1, 1.0);
                for (sb, pair) in bands.windows(2).enumerate() {
                    let total = pair[0].w2 + pair[1].w1;
                    assert!(
                        (total - 1.0).abs() <= 0.0015,
                        "{fs:?} bands {sb} and {}: {total}",
                        sb + 1
                    );
                }
                // One past the last partition in most tables of ISO/IEC
                // 13818-3; see `band_ratio`.
                assert!(bands.last().unwrap().bo <= partitions, "{fs:?}");
            }
        }
    }

    #[test]
    fn selects_the_table_for_each_sampling_frequency() {
        let lines = |fs| {
            let last = partitions(fs).last().unwrap();
            last.start + last.width
        };
        assert_eq!(lines(SamplingFrequency::Hz48000), 481);
        assert_eq!(lines(SamplingFrequency::Hz44100), 465);
        assert_eq!(lines(SamplingFrequency::Hz32000), 488);
        assert_eq!(printed(SamplingFrequency::Hz44100).0[5].bo, 18);

        let short_lines = |fs| {
            let last = short_partitions(fs).last().unwrap();
            last.start + last.width
        };
        assert_eq!(short_lines(SamplingFrequency::Hz48000), 107);
        assert_eq!(short_lines(SamplingFrequency::Hz44100), 109);
        assert_eq!(short_lines(SamplingFrequency::Hz32000), 110);
        assert_eq!(printed(SamplingFrequency::Hz32000).1[11].bo, 41);
    }

    /// The computed conversion gives each partition out in shares that add
    /// up to no more than the whole, and to the whole for those lying wholly
    /// below the top band, at every sampling frequency.
    #[test]
    fn computed_weights_share_out_the_partitions() {
        for fs in SamplingFrequency::ALL {
            let tables: [(&[BandWeights], usize); 2] = [
                (&band_weights(fs), partitions(fs).len()),
                (&short_band_weights(fs), short_partitions(fs).len()),
            ];
            for (bands, count) in tables {
                let mut shares = vec![0.0; count + 1];
                for band in bands {
                    assert!(band.bu < band.bo);
                    assert!((0.0..=1.0).contains(&band.w1) && (0.0..=1.0).contains(&band.w2));
                    shares[band.bu] += band.w1;
                    for share in &mut shares[band.bu + 1..band.bo] {
                        *share += 1.0;
                    }
                    shares[band.bo] += band.w2;
                }
                let last = bands.last().unwrap().bu;
                for (b, share) in shares.iter().enumerate() {
                    assert!(*share <= 1.0 + 1e-9, "{fs:?} partition {b}: {share}");
                    if b < last {
                        assert!((share - 1.0).abs() < 1e-9, "{fs:?} partition {b}: {share}");
                    }
                }
            }
        }
    }

    /// The FFT line, counted from `-1/2` as [`weights_by_frequency`] counts
    /// them, at which `band` starts and ends collecting `partitions`.
    fn collected(band: &BandWeights, partitions: &[(usize, usize)]) -> (f64, f64) {
        let (start, width) = partitions[band.bu];
        let low = start as f64 + (1.0 - band.w1) * width as f64;
        let high = match partitions.get(band.bo) {
            Some(&(start, width)) => start as f64 + band.w2 * width as f64,
            None => {
                let (start, width) = partitions[partitions.len() - 1];
                (start + width) as f64
            }
        };
        (low - 0.5, high - 0.5)
    }

    /// How far above the frequencies of their bands the printed tables
    /// collect the partitions, in FFT lines, for the note in
    /// [`band_weights`]: the most by which a band's lower and upper edges
    /// lie above the computed ones, which are its frequencies, short of the
    /// top band, and those of the top band.
    #[test]
    fn printed_tables_collect_above_the_bands() {
        // fs, long: (below the top, top), short: (below the top, top).
        let expected = [
            (SamplingFrequency::Hz48000, (34.0, 66.0), (6.0, 16.0)),
            (SamplingFrequency::Hz44100, (26.0, 72.0), (6.0, 17.8)),
            (SamplingFrequency::Hz32000, (36.0, 36.0), (7.0, 7.0)),
            (SamplingFrequency::Hz24000, (29.6, 29.6), (7.0, 7.0)),
            (SamplingFrequency::Hz22050, (29.1, 29.1), (6.6, 6.6)),
            (SamplingFrequency::Hz16000, (28.8, 32.3), (6.8, 6.8)),
        ];
        for (fs, long, short) in expected {
            let (printed_long, printed_short) = printed(fs);
            let tables = [
                (
                    printed_long,
                    band_weights(fs).to_vec(),
                    partitions(fs)
                        .iter()
                        .map(|p| (p.start, p.width))
                        .collect::<Vec<_>>(),
                    long,
                ),
                (
                    printed_short,
                    short_band_weights(fs).to_vec(),
                    short_partitions(fs)
                        .iter()
                        .map(|p| (p.start, p.width))
                        .collect(),
                    short,
                ),
            ];
            for (printed, computed, partitions, (below_top, top)) in tables {
                let offsets: Vec<(f64, f64)> = (printed.iter().zip(&computed))
                    .map(|(printed, computed)| {
                        let (pl, ph) = collected(printed, &partitions);
                        let (cl, ch) = collected(computed, &partitions);
                        (pl - cl, ph - ch)
                    })
                    .collect();
                // The lowest bands of the long tables of ISO/IEC 11172-3
                // agree with their frequencies to the printed precision.
                assert!(
                    offsets[0].1.abs() < 0.05
                        || fs.version() == Version::Mpeg2
                        || fs == SamplingFrequency::Hz32000,
                    "{fs:?}: {offsets:?}"
                );
                let (last, rest) = offsets.split_last().unwrap();
                let most = rest.iter().map(|&(l, h)| l.max(h)).fold(f64::MIN, f64::max);
                println!(
                    "{fs:?}: {:?} top {last:?}",
                    rest.iter()
                        .map(|&(l, h)| format!("{l:+.1}/{h:+.1}"))
                        .collect::<Vec<_>>()
                );
                assert!(
                    (most - below_top).abs() < 0.1,
                    "{fs:?}: {most:.2}, {offsets:?}"
                );
                assert!((last.0.max(last.1) - top).abs() < 0.1, "{fs:?}: {last:?}");
            }
        }
    }

    /// The tables of ISO/IEC 13818-3 cover the lines of each FFT their
    /// partitions are printed for, and the last scalefactor band of each but
    /// Table D.4.c ends one partition past the last.
    #[test]
    fn lower_sampling_frequency_tables_cover_their_ffts() {
        let lines = |partitions: &[Partition]| {
            let last = partitions.last().unwrap();
            last.start + last.width
        };
        let short_lines = |partitions: &[ShortPartition]| {
            let last = partitions.last().unwrap();
            last.start + last.width
        };
        assert_eq!(lines(&PARTITIONS_24000), 491);
        assert_eq!(lines(&PARTITIONS_22050), 489);
        assert_eq!(lines(&PARTITIONS_16000), 509);
        assert_eq!(short_lines(&SHORT_PARTITIONS_24000), 127);
        assert_eq!(short_lines(&SHORT_PARTITIONS_22050), 121);
        assert_eq!(short_lines(&SHORT_PARTITIONS_16000), 121);
        let last_bo = |bands: &[BandWeights]| bands.last().unwrap().bo;
        assert_eq!(last_bo(&BANDS_24000), PARTITIONS_24000.len());
        assert_eq!(last_bo(&BANDS_22050), PARTITIONS_22050.len());
        assert_eq!(last_bo(&BANDS_16000), PARTITIONS_16000.len() - 1);
        assert_eq!(last_bo(&SHORT_BANDS_24000), SHORT_PARTITIONS_24000.len());
        assert_eq!(last_bo(&SHORT_BANDS_22050), SHORT_PARTITIONS_22050.len());
        assert_eq!(last_bo(&SHORT_BANDS_16000), SHORT_PARTITIONS_16000.len());
    }
}
