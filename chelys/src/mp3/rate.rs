// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! How each frame's bitrate is chosen (clause 2.4.2.3): "Layer III supports
//! variable bitrate by switching the bitrate_index. The switching of the
//! bitrate_index can be used either to optimize storage requirements on DSM
//! or to interpolate any mean data rate by switching between nearby values
//! in the bitrate table." [`Rate::Mean`] does the second.

use super::header::{Bitrate, SamplingFrequency, Version};

/// How each frame's bitrate is chosen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rate {
    /// Every frame at one bitrate, padded as
    /// [`PaddingSchedule`](super::header::PaddingSchedule) has it.
    Constant(Bitrate),
    Mean(MeanRate),
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
}
