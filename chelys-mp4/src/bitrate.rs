// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The bitrates a `DecoderConfigDescriptor` gives (ISO/IEC 14496-1,
//! clause 6.2.6.6.2), measured on a stream's frames, each of which lasts
//! `frame_samples` samples at `sample_rate`.

/// `maxBitrate`: "the maximum bitrate in bits per second of this
/// elementary stream in any time window of one second duration".
///
/// Spec gap: the standard does not say how a frame, which lasts, counts
/// in a window. Each frame counts whole at its decoding time, the start of
/// its samples, so that a window holds the frames whose decoding times
/// fall within it: as many as start within one second, at most
/// `sample_rate / frame_samples` rounded up. That counts each frame that
/// overlaps the window at all, but the last, and so does not understate
/// the rate.
///
/// Spec reading: a stream shorter than a second has all its frames in one
/// window, whose bits over a second understate the rate at which they
/// come, their average over the stream, [`avg_bitrate`]. The maximum is
/// taken as at least that average, so that the descriptor never gives a
/// maximum below its average. This raises the maximum of short streams
/// only, in practice: in a longer stream at a constant rate, some window
/// holds at least the average.
pub fn max_bitrate(sizes: &[u32], frame_samples: u32, sample_rate: u32) -> u32 {
    let window = window_frames(frame_samples, sample_rate) as usize;
    let bytes = |frames: &[u32]| frames.iter().map(|&size| u64::from(size)).sum::<u64>();
    let mut sum = bytes(&sizes[..window.min(sizes.len())]);
    let mut max = sum;
    for (entering, leaving) in sizes.iter().skip(window).zip(sizes) {
        sum = sum + u64::from(*entering) - u64::from(*leaving);
        max = max.max(sum);
    }
    let windowed = u32::try_from(8 * max).unwrap_or(u32::MAX);
    windowed.max(avg_bitrate(sizes, frame_samples, sample_rate))
}

/// The most frames that a window of one second holds, as [`max_bitrate`]
/// counts them: those whose decoding times fall within it.
pub fn window_frames(frame_samples: u32, sample_rate: u32) -> u32 {
    sample_rate.div_ceil(frame_samples)
}

/// `avgBitrate`: "the average bitrate in bits per second of this
/// elementary stream", measured over all its frames and the samples they
/// code, rounded to the nearest bit per second.
pub fn avg_bitrate(sizes: &[u32], frame_samples: u32, sample_rate: u32) -> u32 {
    let samples = sizes.len() as u128 * u128::from(frame_samples);
    if samples == 0 {
        return 0;
    }
    let bits = 8 * sizes.iter().map(|&size| u128::from(size)).sum::<u128>();
    let rate = (2 * bits * u128::from(sample_rate) + samples) / (2 * samples);
    u32::try_from(rate).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_holds_the_frames_that_start_within_a_second() {
        // 48 kHz: 46.875 frames a second, so 47 to a window.
        let mut sizes = vec![100; 200];
        assert_eq!(max_bitrate(&sizes, 1024, 48_000), 47 * 800);
        sizes[120] = 1100;
        assert_eq!(max_bitrate(&sizes, 1024, 48_000), 47 * 800 + 8000);
        // 8 kHz: 7.8125 frames a second, so 8.
        assert_eq!(max_bitrate(&sizes, 1024, 8_000), 8 * 800 + 8000);
    }

    #[test]
    fn a_stream_shorter_than_a_second_is_at_least_its_average() {
        // 3 frames of 60 bytes in 3072 samples: 6891 bit/s on average, and
        // 480 bits in the one window.
        let sizes = [10, 20, 30];
        assert_eq!(avg_bitrate(&sizes, 1024, 44_100), 6891);
        assert_eq!(max_bitrate(&sizes, 1024, 44_100), 6891);
        assert_eq!(max_bitrate(&[], 1024, 44_100), 0);
        // A second or more: the densest window.
        let mut sizes = vec![100; 44];
        sizes[0] = 1000;
        assert_eq!(max_bitrate(&sizes, 1024, 44_100), 8 * (43 * 100 + 1000));
        assert!(avg_bitrate(&sizes, 1024, 44_100) < 8 * (43 * 100 + 1000));
    }

    #[test]
    fn the_maximum_is_never_below_the_average() {
        for frames in 0..200 {
            let sizes: Vec<u32> = (0..frames).map(|i| 300 + (i * 37 % 101)).collect();
            for sample_rate in [8_000, 44_100, 96_000] {
                let max = max_bitrate(&sizes, 1024, sample_rate);
                let avg = avg_bitrate(&sizes, 1024, sample_rate);
                assert!(
                    avg <= max,
                    "{frames} frames at {sample_rate} Hz: {avg} > {max}"
                );
            }
        }
    }

    #[test]
    fn the_average_is_the_bits_over_the_frames_duration() {
        // 43 frames of 373 bytes at 44.1 kHz: 128.6 kbit/s.
        let sizes = vec![373; 43];
        let exact: f64 = 8.0 * 373.0 * 44_100.0 / 1024.0;
        assert_eq!(f64::from(avg_bitrate(&sizes, 1024, 44_100)), exact.round());
        assert_eq!(avg_bitrate(&[], 1024, 44_100), 0);
        // 1 byte in 2 samples at 3 Hz: 12 bit/s.
        assert_eq!(avg_bitrate(&[1], 2, 3), 12);
    }
}
