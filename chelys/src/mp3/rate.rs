// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! How each frame's bitrate is chosen (clause 2.4.2.3): "Layer III supports
//! variable bitrate by switching the bitrate_index. The switching of the
//! bitrate_index can be used either to optimize storage requirements on DSM
//! or to interpolate any mean data rate by switching between nearby values
//! in the bitrate table." [`Rate::Variable`] does the first, and
//! [`Rate::Mean`] the second.

use super::header::{Bitrate, SamplingFrequency, Version};

/// How each frame's bitrate is chosen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rate {
    /// Every frame at one bitrate, padded as
    /// [`PaddingSchedule`](super::header::PaddingSchedule) has it.
    Constant(Bitrate),
    Mean(MeanRate),
    Variable(VariableRate),
}

impl Rate {
    /// The [`Version`] whose bitrates the frames have.
    pub fn version(self) -> Version {
        self.lowest().version()
    }

    /// The lowest bitrate a frame may have.
    pub fn lowest(self) -> Bitrate {
        match self {
            Self::Constant(bitrate) => bitrate,
            Self::Mean(mean) => mean.lower,
            Self::Variable(variable) => variable.lowest,
        }
    }
}

/// A mean bitrate between the lowest and the highest of the table of a
/// [`Version`]: frames switch between the two bitrates of the table nearest
/// to it, or stay at it if it is one of them. See [`MeanRateSchedule`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeanRate {
    bits_per_second: u32,
    /// The highest bitrate of the table at or below the mean.
    lower: Bitrate,
    /// The lowest bitrate of the table at or above the mean.
    upper: Bitrate,
}

impl MeanRate {
    /// `None` if `bits_per_second` is below the lowest bitrate of the table
    /// of `version` or above its highest.
    pub fn new(version: Version, bits_per_second: u32) -> Option<Self> {
        let lower = Bitrate::all(version)
            .take_while(|b| b.bits_per_second() <= bits_per_second)
            .last()?;
        let upper = Bitrate::all(version).find(|b| b.bits_per_second() >= bits_per_second)?;
        Some(Self {
            bits_per_second,
            lower,
            upper,
        })
    }

    pub fn bits_per_second(self) -> u32 {
        self.bits_per_second
    }
}

/// A bitrate for each frame by the noise its granules leave: the lowest of
/// those tried from `lowest` to `highest` at which the quantisation noise of
/// every scalefactor band stays within what the psychoacoustic model allows
/// it, raised by `noise_offset_db`.
///
/// The offset scales the allowed distortion `xmin` of every band by
/// `10^(noise_offset_db / 10)` before the iteration loops see it: above 0
/// it allows more noise, and gives smaller frames, below 0 less. It departs
/// from Annex C, whose loops take `xmin` as the model gives it, and is 0
/// for that.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VariableRate {
    pub lowest: Bitrate,
    pub highest: Bitrate,
    pub noise_offset_db: f64,
}

impl VariableRate {
    /// The bitrates a frame may have, lowest first.
    ///
    /// # Panics
    ///
    /// If `lowest` and `highest` are bitrates of different versions, or
    /// `lowest` is above `highest`.
    pub fn bitrates(self) -> Vec<Bitrate> {
        let version = self.lowest.version();
        assert_eq!(
            version,
            self.highest.version(),
            "the lowest and highest bitrates are of one version"
        );
        let range = self.lowest.bits_per_second()..=self.highest.bits_per_second();
        assert!(!range.is_empty(), "the lowest bitrate is above the highest");
        Bitrate::all(version)
            .filter(|b| range.contains(&b.bits_per_second()))
            .collect()
    }

    /// The factor `noise_offset_db` scales the allowed distortion by.
    pub fn noise_factor(self) -> f64 {
        10f64.powf(self.noise_offset_db / 10.0)
    }
}

/// Chooses each frame's bitrate and padding so that the stream keeps a
/// [`MeanRate`]. After `k` frames the stream is as long as it can be
/// without passing `floor(k * N)` slots, `N` being the slots of a frame at
/// the mean rate by the formula of clause 2.4.3.1; it falls short of that
/// by less than a frame at the upper bitrate. At a bitrate of the table this
/// is what [`PaddingSchedule`](super::header::PaddingSchedule) does.
///
/// Spec gap: clause 2.4.2.3 allows switching "between nearby values in the
/// bitrate table" without saying how. Each frame takes the longest of the
/// lower and upper bitrates, with and without padding, that keeps the
/// stream within its share, and so the frames at the upper bitrate fall
/// where the schedule gets to them, whatever their content.
#[derive(Debug)]
pub struct MeanRateSchedule {
    rate: MeanRate,
    sampling_frequency: SamplingFrequency,
    frames: u64,
    bytes: u64,
}

impl MeanRateSchedule {
    /// # Panics
    ///
    /// If `rate` is not of the version of `sampling_frequency`.
    pub fn new(rate: MeanRate, sampling_frequency: SamplingFrequency) -> Self {
        assert_eq!(
            rate.lower.version(),
            sampling_frequency.version(),
            "{rate:?} is not a rate of {sampling_frequency:?}"
        );
        Self {
            rate,
            sampling_frequency,
            frames: 0,
            bytes: 0,
        }
    }

    /// The next frame's bitrate and `padding_bit`.
    pub fn next_frame(&mut self) -> (Bitrate, bool) {
        let fs = self.sampling_frequency;
        self.frames += 1;
        let slots_per_bit = fs.version().samples_per_frame() as u64 / 8;
        let share =
            self.frames * slots_per_bit * u64::from(self.rate.bits_per_second) / u64::from(fs.hz());
        let room = share - self.bytes;
        let choices = [self.rate.lower, self.rate.upper]
            .into_iter()
            .flat_map(|bitrate| [(bitrate, false), (bitrate, true)]);
        let len = |&(bitrate, padding): &(Bitrate, bool)| bitrate.frame_len(fs, padding) as u64;
        let (bitrate, padding) = choices
            .filter(|choice| len(choice) <= room)
            .max_by_key(len)
            .expect("the lower bitrate's frame is never longer than the mean's share");
        self.bytes += len(&(bitrate, padding));
        (bitrate, padding)
    }
}

/// Searches the bitrates of a [`VariableRate`] for a frame, given their
/// number, the index to start from, and a function that codes the frame at
/// the bitrate of an index and says whether its noise stays within its
/// allowance. Returns the index kept and the frame coded at it.
///
/// From the start, the search goes down while the frame stays within its
/// allowance, and keeps the last bitrate at which it did; if it is not
/// within it at the start, the search goes up until it is, and keeps the
/// highest bitrate if it never is.
///
/// Spec gap: neither standard says how to choose a frame's bitrate. The
/// noise left by the iteration loops does not fall steadily as the bits
/// rise (preemphasis, `scalefac_scale` and scfsi switch at thresholds of
/// their own), so the lowest bitrate tried at which the frame is within its
/// allowance need not be the lowest at which it would be. Starting from the
/// bitrate of the frame before keeps the tries few where the signal is
/// steady, two for most frames.
pub fn search_variable<T>(
    count: usize,
    start: usize,
    mut code: impl FnMut(usize) -> (T, bool),
) -> (usize, T) {
    assert!(start < count, "the start is one of the bitrates");
    let (mut coded, within) = code(start);
    let mut index = start;
    if within {
        while index > 0 {
            let (lower, within) = code(index - 1);
            if !within {
                break;
            }
            (index, coded) = (index - 1, lower);
        }
    } else {
        while index + 1 < count {
            let (higher, within) = code(index + 1);
            (index, coded) = (index + 1, higher);
            if within {
                break;
            }
        }
    }
    (index, coded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mp3::header::PaddingSchedule;

    #[test]
    fn a_mean_rate_lies_between_two_neighbours_of_the_table() {
        let kbps = |b: Bitrate| b.bits_per_second() / 1000;
        let rate = MeanRate::new(Version::Mpeg1, 150_000).unwrap();
        assert_eq!((kbps(rate.lower), kbps(rate.upper)), (128, 160));
        let rate = MeanRate::new(Version::Mpeg1, 128_000).unwrap();
        assert_eq!((kbps(rate.lower), kbps(rate.upper)), (128, 128));
        let rate = MeanRate::new(Version::Mpeg2, 8_500).unwrap();
        assert_eq!((kbps(rate.lower), kbps(rate.upper)), (8, 16));
        assert!(MeanRate::new(Version::Mpeg1, 31_999).is_none());
        assert!(MeanRate::new(Version::Mpeg1, 320_001).is_none());
        assert!(MeanRate::new(Version::Mpeg2, 160_000).is_some());
        assert!(MeanRate::new(Version::Mpeg2, 7_000).is_none());
    }

    /// After every frame the stream is no longer than its share of the mean
    /// rate, and short of it by less than the longest frame.
    #[test]
    fn a_mean_rate_keeps_the_stream_within_its_share() {
        for fs in SamplingFrequency::ALL {
            let version = fs.version();
            let (low, high) = (8_000, 320_000);
            for bps in (low..=high).step_by(3_217) {
                let Some(rate) = MeanRate::new(version, bps) else {
                    continue;
                };
                let mut schedule = MeanRateSchedule::new(rate, fs);
                let longest = rate.upper.frame_len(fs, true) as u64;
                let mut bytes = 0;
                for k in 1..=500u64 {
                    let (bitrate, padding) = schedule.next_frame();
                    assert!(bitrate == rate.lower || bitrate == rate.upper);
                    bytes += bitrate.frame_len(fs, padding) as u64;
                    let share = k * version.samples_per_frame() as u64 / 8 * u64::from(bps)
                        / u64::from(fs.hz());
                    assert!(bytes <= share, "{bps} bit/s at {fs:?}, frame {k}");
                    assert!(share - bytes < longest, "{bps} bit/s at {fs:?}, frame {k}");
                }
            }
        }
    }

    /// At a bitrate of the table, a mean rate pads as a constant one does.
    #[test]
    fn a_mean_rate_of_the_table_is_constant() {
        for fs in SamplingFrequency::ALL {
            for bitrate in Bitrate::all(fs.version()) {
                let rate = MeanRate::new(fs.version(), bitrate.bits_per_second()).unwrap();
                let mut schedule = MeanRateSchedule::new(rate, fs);
                let mut padding = PaddingSchedule::new(bitrate, fs);
                for k in 0..300 {
                    let frame = schedule.next_frame();
                    assert_eq!(frame, (bitrate, padding.next_padding()), "{fs:?}, {k}");
                }
            }
        }
    }

    #[test]
    fn a_variable_rate_offers_the_bitrates_between_its_bounds() {
        let at = |kbps| Bitrate::from_kbps(Version::Mpeg1, kbps).unwrap();
        let rate = VariableRate {
            lowest: at(64),
            highest: at(128),
            noise_offset_db: 0.0,
        };
        let kbps: Vec<u32> = (rate.bitrates().iter())
            .map(|b| b.bits_per_second() / 1000)
            .collect();
        assert_eq!(kbps, [64, 80, 96, 112, 128]);
        assert_eq!(rate.noise_factor(), 1.0);
        let louder = VariableRate {
            noise_offset_db: 10.0,
            ..rate
        };
        assert!((louder.noise_factor() - 10.0).abs() < 1e-12);
    }

    /// For every pattern of bitrates within and outside the allowance, and
    /// every start, the search keeps the bitrate the walk from the start
    /// stops at, and tries only those it passes.
    #[test]
    fn the_search_walks_down_while_within_and_up_until_within() {
        for count in 1..=6usize {
            for pattern in 0..1usize << count {
                let within = |index: usize| pattern >> index & 1 == 1;
                for start in 0..count {
                    let mut tried = Vec::new();
                    let (index, coded) = search_variable(count, start, |index| {
                        tried.push(index);
                        (index * 10, within(index))
                    });
                    assert_eq!(coded, index * 10);
                    let expected: Vec<usize> = if within(start) {
                        let lowest = (0..=start).rev().take_while(|&i| within(i)).last();
                        let lowest = lowest.unwrap();
                        assert_eq!(index, lowest, "{pattern:b} from {start}");
                        (lowest.saturating_sub(1)..=start).rev().collect()
                    } else {
                        let first = (start..count).find(|&i| within(i));
                        assert_eq!(index, first.unwrap_or(count - 1), "{pattern:b}");
                        (start..=index).collect()
                    };
                    assert_eq!(tried, expected, "{pattern:b} from {start}");
                }
            }
        }
    }
}
