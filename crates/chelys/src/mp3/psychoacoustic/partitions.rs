// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The threshold calculation partitions of psychoacoustic model 2 for Layer
//! III long and short blocks (Table C.7) and their conversion to scalefactor
//! bands (Table C.8).
//!
//! The tables are transcribed as printed, except that Table C.8's `cbw`
//! column is left out: the conversion in figure C.6.c does not use it.
//! Their numbering and layout are checked at compile time.

use super::super::header::SamplingFrequency;
use super::super::scalefactor_bands::{LONG_BANDS, SHORT_BANDS};

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
    /// The SNR in dB, negative: the threshold lies this far below the spread
    /// energy (figure C.6.d).
    pub snr: f64,
    /// The median Bark value.
    pub bval: f64,
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
    }
}

pub fn band_weights(sampling_frequency: SamplingFrequency) -> &'static [BandWeights; LONG_BANDS] {
    match sampling_frequency {
        SamplingFrequency::Hz48000 => &BANDS_48000,
        SamplingFrequency::Hz44100 => &BANDS_44100,
        SamplingFrequency::Hz32000 => &BANDS_32000,
    }
}

/// The partitions for short blocks.
///
/// Spec gap: the `norm` column of each of Tables C.7.d to C.7.f repeats,
/// row for row and to within a rounding step, the `norm` column of the
/// long-block table for the same sampling frequency (C.7.a to C.7.c),
/// although the short partitions have other Bark values. It does not match
/// the reciprocal of the spreading function summed over the short
/// partitions, as the long tables' column does (clause D.2.4 f)).
pub fn short_partitions(sampling_frequency: SamplingFrequency) -> &'static [ShortPartition] {
    match sampling_frequency {
        SamplingFrequency::Hz48000 => &SHORT_PARTITIONS_48000,
        SamplingFrequency::Hz44100 => &SHORT_PARTITIONS_44100,
        SamplingFrequency::Hz32000 => &SHORT_PARTITIONS_32000,
    }
}

pub fn short_band_weights(
    sampling_frequency: SamplingFrequency,
) -> &'static [BandWeights; SHORT_BANDS] {
    match sampling_frequency {
        SamplingFrequency::Hz48000 => &SHORT_BANDS_48000,
        SamplingFrequency::Hz44100 => &SHORT_BANDS_44100,
        SamplingFrequency::Hz32000 => &SHORT_BANDS_32000,
    }
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
    let mut partitions = [ShortPartition {
        start: 0,
        width: 0,
        qthr: 0.0,
        norm: 0.0,
        snr: 0.0,
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
            snr,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mp3::psychoacoustic::spreading_function;

    /// `norm` is printed without its derivation, but it matches the
    /// normalisation `rnorm` of clause D.2.4 f), the reciprocal of the
    /// spreading function summed over all partitions, to within the rounding
    /// of the printed Bark values. This checks both columns.
    #[test]
    fn norm_is_the_reciprocal_of_the_spreading_sum() {
        for fs in SamplingFrequency::ALL {
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

    /// The short tables' `norm` column repeats the long one (see
    /// [`short_partitions`]) to within a rounding step. This checks that
    /// reading of the column.
    #[test]
    fn short_norm_repeats_the_long_column() {
        for fs in SamplingFrequency::ALL {
            for (short, long) in short_partitions(fs).iter().zip(partitions(fs)) {
                assert!((short.norm - long.norm).abs() <= 0.0015, "{fs:?}");
            }
        }
    }

    /// Each partition's threshold in quiet is its width times one of a few
    /// per-line levels, to within the rounding of the printed values.
    #[test]
    fn qthr_is_a_per_line_level_times_the_width() {
        const LEVELS: [f64; 7] = [0.009, 0.018, 0.0286, 0.0904, 0.904, 4.532, 9.043];
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
                assert!([0.0, 3.0, 6.0, 12.0, 18.0, 20.0, 24.5].contains(&p.minval));
            }
        }
    }

    /// The short blocks' SNR rises from low to high frequencies through a
    /// few values.
    #[test]
    fn short_snr_steps_up() {
        const SNR: [f64; 7] = [-8.240, -7.447, -6.990, -6.020, -5.229, -4.559, -3.980];
        for fs in SamplingFrequency::ALL {
            let partitions = short_partitions(fs);
            for (b, pair) in partitions.windows(2).enumerate() {
                assert!(pair[1].snr >= pair[0].snr, "{fs:?} partition {}", b + 1);
            }
            assert!(partitions.iter().all(|p| SNR.contains(&p.snr)), "{fs:?}");
        }
    }

    /// A partition shared by two neighbouring bands is split between them:
    /// the weights it gets in each add up to one, to the printed precision.
    #[test]
    fn shared_partitions_are_split_between_bands() {
        for fs in SamplingFrequency::ALL {
            let tables: [(&[BandWeights], usize); 2] = [
                (band_weights(fs), partitions(fs).len()),
                (short_band_weights(fs), short_partitions(fs).len()),
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
                assert!(bands.last().unwrap().bo < partitions, "{fs:?}");
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
        assert_eq!(band_weights(SamplingFrequency::Hz44100)[5].bo, 18);

        let short_lines = |fs| {
            let last = short_partitions(fs).last().unwrap();
            last.start + last.width
        };
        assert_eq!(short_lines(SamplingFrequency::Hz48000), 107);
        assert_eq!(short_lines(SamplingFrequency::Hz44100), 109);
        assert_eq!(short_lines(SamplingFrequency::Hz32000), 110);
        assert_eq!(short_band_weights(SamplingFrequency::Hz32000)[11].bo, 41);
    }
}
