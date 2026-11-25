// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! What chelys's psychoacoustic models share from that of ISO/IEC 13818-7
//! (clause C.1): its threshold calculation partitions and its spreading
//! function.

pub mod partitions;

/// The spreading function `sprdngf(i, j)` of clause C.1.3, from `i`, the
/// Bark value of the signal being spread, to `j`, that of the band it is
/// spread into, as printed.
///
/// Spec reading: the function is taken as printed, `tmpx = 3.0 (j - i)`
/// towards higher frequencies and `1.5 (j - i)` towards lower ones. Model
/// 2 of ISO/IEC 11172-3 prints the same rule, which chelys's MP3 encoder
/// reads with the factors swapped, as the `norm` column of that standard's
/// tables bears out; Annex C prints no such column, and computes the
/// normalization from the function (step 6), so nothing here bears out
/// either order. The reading rests on the text alone: chelys's AAC encoder,
/// graded with PEAQ (GstPEAQ 0.6.1, black box), on 8 tracks of pop music at
/// 44.1 kHz in stereo and on 3 tracks of rock music and their 3 windows of
/// 3 s with the most frames coded short, the factors swapped gave means of
/// -3.580, -1.074 and -0.283 at 64, 128 and 192 kbit/s, and -3.573, -0.781
/// and -0.135, against -3.583, -1.087 and -0.282, and -3.579, -0.803 and
/// -0.142, as printed.
pub fn spreading_function(i: f64, j: f64) -> f64 {
    let tmpx = if j >= i { 3.0 } else { 1.5 } * (j - i);
    let tmpz = 8.0 * f64::min((tmpx - 0.5).powi(2) - 2.0 * (tmpx - 0.5), 0.0);
    let tmpy = 15.811389 + 7.5 * (tmpx + 0.474) - 17.5 * (1.0 + (tmpx + 0.474).powi(2)).sqrt();
    if tmpy < -100.0 {
        0.0
    } else {
        10f64.powf((tmpz + tmpy) / 10.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_spreading_function_peaks_where_it_spreads_into_itself() {
        // tmpx 0: tmpz 0 and tmpy 15.811389 + 7.5 * 0.474 - 17.5 *
        // sqrt(1 + 0.474^2), about 0 dB.
        assert!((spreading_function(5.0, 5.0) - 1.0).abs() < 1e-3);
        for d in [0.5, 1.0, 2.0, 4.0] {
            assert!(spreading_function(5.0, 5.0 + d) < spreading_function(5.0, 5.0 + d / 2.0));
            assert!(spreading_function(5.0, 5.0 - d) < spreading_function(5.0, 5.0 - d / 2.0));
        }
        // Far away it is cut to zero.
        assert_eq!(spreading_function(0.0, 20.0), 0.0);
        assert_eq!(spreading_function(20.0, 0.0), 0.0);
    }
}
