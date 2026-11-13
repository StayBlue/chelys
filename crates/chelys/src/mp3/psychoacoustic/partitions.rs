//! The threshold calculation partitions of psychoacoustic model 2 for Layer
//! III long blocks (Table C.7) and their conversion to scalefactor bands
//! (Table C.8).
//!
//! The tables are transcribed as printed, except that Table C.8's `cbw`
//! column is left out: the conversion in figure C.6.c does not use it.
//! Their numbering and layout are checked at compile time.

use super::super::header::SamplingFrequency;
use super::super::scalefactor_bands::LONG_BANDS;

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

/// Builds a conversion table from printed rows `(no. sb, bu, bo, w1, w2)`,
/// checking that the bands are numbered in order, start at partition 0, and
/// that each band begins at the partition where the last one ended.
const fn band_weights_from(
    rows: [(usize, usize, usize, f64, f64); LONG_BANDS],
) -> [BandWeights; LONG_BANDS] {
    let mut bands = [BandWeights {
        bu: 0,
        bo: 0,
        w1: 0.0,
        w2: 0.0,
    }; LONG_BANDS];
    let mut i = 0;
    while i < LONG_BANDS {
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

    /// Each partition's threshold in quiet is its width times one of a few
    /// per-line levels, to within the rounding of the printed values.
    #[test]
    fn qthr_is_a_per_line_level_times_the_width() {
        const LEVELS: [f64; 7] = [0.009, 0.018, 0.0286, 0.0904, 0.904, 4.532, 9.043];
        for fs in SamplingFrequency::ALL {
            for (b, p) in partitions(fs).iter().enumerate() {
                let width = p.width as f64;
                assert!(
                    LEVELS
                        .iter()
                        .any(|level| (level * width - p.qthr).abs() <= 0.0005 + 0.005 * p.qthr),
                    "{fs:?} partition {b}: qthr {} over {} lines",
                    p.qthr,
                    p.width
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

    /// A partition shared by two neighbouring bands is split between them:
    /// the weights it gets in each add up to one, to the printed precision.
    #[test]
    fn shared_partitions_are_split_between_bands() {
        for fs in SamplingFrequency::ALL {
            let bands = band_weights(fs);
            assert_eq!(bands[0].w1, 1.0);
            for (sb, pair) in bands.windows(2).enumerate() {
                let total = pair[0].w2 + pair[1].w1;
                assert!(
                    (total - 1.0).abs() <= 0.0015,
                    "{fs:?} bands {sb} and {}: {total}",
                    sb + 1
                );
            }
            assert!(bands[LONG_BANDS - 1].bo < partitions(fs).len(), "{fs:?}");
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
    }
}
