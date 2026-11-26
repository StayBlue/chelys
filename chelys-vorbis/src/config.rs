// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! What an [`Encoder`](super::Encoder) is asked for, and its checks.

use std::fmt;

use super::calibration;

/// The most channels the encoder takes: those of the largest layout that
/// clause 4.3.9 orders, 7.1.
pub const MAX_CHANNELS: usize = 8;

/// How much the stream may take.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Rate {
    /// A variable rate: each line of a block is allowed the noise that the
    /// psychoacoustic model allows it, raised with the width of its
    /// critical band, and in long blocks from 11 kHz by up to twice that,
    /// rising to it at 13 kHz, in stereo the less the higher the offset,
    /// scaled by `10^(noise_offset_db / 10)`, and each packet is as long as
    /// that takes. Above 0 it allows more noise and gives shorter packets,
    /// below 0 less. No line is allowed less noise than the rounding of
    /// 16-bit PCM makes, whatever the offset. In stereo, where the two
    /// channels are alike, the floor of one may be lowered so that the two
    /// differ as the channels' levels do; and the two channels' lines may
    /// be made alike, which then cost fewer bits coupled, where that leaves
    /// each channel no more noise than it is allowed, or above 1.5 kHz 1 to
    /// 3 times it, the more the higher the offset, and while no band's
    /// energy of their difference falls against that of their sum by more
    /// than 6 dB at -9 dB and below, or by more as the offset rises. Lines
    /// that lie near halfway between two values may then be rounded to the
    /// other, so that each band decodes with about its energy, which may
    /// add noise to what a band is allowed, below 1.5 kHz too. Where most
    /// of a band's lines would be coded as 0, some are coded as 1 so that
    /// it keeps part of its energy, which above 1.5 kHz adds noise to what
    /// it is allowed. Either may take a band's difference against its sum
    /// below that bound, in about one band in seventy. At 44.1 kHz in stereo, music takes about 67 kbit/s
    /// at 0 dB and 137 at -7.5.
    Variable { noise_offset_db: f64 },
    /// A variable rate at `level`: a level of the [`quality`](crate::quality)
    /// scale that chelys's encoders share, from 0 to
    /// [`MAX_LEVEL`](crate::quality::MAX_LEVEL), higher for better quality.
    ///
    /// It is [`Rate::Variable`] with the noise offset the level maps to,
    /// calibrated to the scale's grades at 44.1 kHz in two channels: music
    /// in stereo averages about 102 kbit/s at level 50, 166 at 80 and 189
    /// at 100. Every other configuration takes the offsets of stereo at
    /// 44.1 kHz, as calibrated with its channels coded each on its own
    /// where they are coded so, so that there a level gives the scale's
    /// grades only roughly.
    Quality { level: u8 },
    /// A mean rate: a variable rate whose noise offset is steered, from
    /// one long block, or the short blocks that take its place, to the
    /// next, toward keeping `bits_per_second` on average over the audio
    /// packets. A packet's length follows its content, as at a variable
    /// rate, and the stream's, over some ten seconds, the rate. The header
    /// packets are not counted, nor what a container adds: the pages of an
    /// Ogg stream of `chelys-ogg` are some 1.1% to 1.4% longer than its
    /// audio packets from 64 to 160 kbit/s, and its headers under 1 KB.
    /// The steering only sets the blocks to come:
    /// blocks of silence, or of a quiet end, which take few bits at any
    /// offset, leave the stream short by what they do not take, and its
    /// first blocks, before the steering has learnt what the content
    /// takes, can run over or under, so that a stream of a few seconds can
    /// miss the mean by much more than a long one. Content that takes few
    /// bits at any offset, such as a few steady tones, falls well short of
    /// any mean, the offset held at the lowest the steering keeps, while
    /// the identification header still gives the mean as nominal: 30 s of
    /// a sine at 440 Hz in stereo took about 31 kbit/s at means of 128 and
    /// 529 kbit/s alike. Each block is coded as at the variable rate of its
    /// offset, the trades of noise for bits that the offset tapers
    /// included.
    ///
    /// On the tuning set of 51 tracks at 44.1 kHz, at means of 96, 128 and
    /// 160 kbit/s, the tracks' audio packets were 0.6% short on average in
    /// stereo and 0.8% to 0.9% in mono, 4.5% at most, and none more than
    /// 0.12% over. Graded with PEAQ (GstPEAQ 0.6.1, black box) in stereo,
    /// they graded as [`Rate::Variable`] at the offset that matches their
    /// total size: -0.845, -0.398 and -0.141, against -0.843, -0.406 and
    /// -0.148; in mono at 64, 96 and 128 kbit/s, -0.511, -0.010 and 0.114,
    /// against -0.501, -0.009 and 0.123.
    ///
    /// The rate is from [`MIN_BITS_PER_SAMPLE`] to [`MAX_BITS_PER_SAMPLE`]
    /// bits for each sample of each channel: at 44.1 kHz, from 6 to 264
    /// kbit/s in mono and from 12 to 529 in stereo. The identification
    /// header gives it as the stream's nominal bitrate.
    Mean { bits_per_second: u32 },
}

/// The fewest bits for each sample of each channel that [`Rate::Mean`]
/// takes: about what music takes at a noise offset of 19 dB, where little
/// but its floors is coded.
pub const MIN_BITS_PER_SAMPLE: f64 = 0.125;

/// The most bits for each sample of each channel that [`Rate::Mean`]
/// takes: about what music takes at a noise offset of -29 dB, near the
/// lowest the steering keeps (see
/// [`OFFSETS_DB`](chelys_core::steering::OFFSETS_DB)).
pub const MAX_BITS_PER_SAMPLE: f64 = 6.0;

/// What to encode: the sample rate, the channels and the rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    sample_rate: u32,
    channels: usize,
    rate: Rate,
}

impl Config {
    /// A configuration of `channels` channels, from 1 to [`MAX_CHANNELS`],
    /// at `sample_rate` Hz, which is any but 0, coded at `rate`.
    /// [`Encoder::new`] checks it.
    ///
    /// The input gives the channels in the order the stream sends them,
    /// which clause 4.3.9 maps to speakers:
    ///
    /// | `channels` | Order |
    /// |---|---|
    /// | 1 | mono |
    /// | 2 | left, right |
    /// | 3 | left, center, right |
    /// | 4 | front left, front right, rear left, rear right |
    /// | 5 | front left, center, front right, rear left, rear right |
    /// | 6 | front left, center, front right, rear left, rear right, LFE |
    /// | 7 | front left, center, front right, side left, side right, rear center, LFE |
    /// | 8 | front left, center, front right, side left, side right, rear left, rear right, LFE |
    ///
    /// Spec reading: above eight channels "channel use and order is defined
    /// by the application"; chelys defines none, and refuses them.
    ///
    /// [`Encoder::new`]: super::Encoder::new
    pub fn new(sample_rate: u32, channels: usize, rate: Rate) -> Self {
        Self {
            sample_rate,
            channels,
            rate,
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn rate(&self) -> Rate {
        self.rate
    }

    /// Whether [`Encoder::new`] takes this configuration.
    ///
    /// # Errors
    ///
    /// As [`Encoder::new`].
    ///
    /// [`Encoder::new`]: super::Encoder::new
    pub fn check(&self) -> Result<(), ConfigError> {
        self.control().map(|_| ())
    }

    /// The range of [`Rate::Mean`] at the sample rate and channels, in bits
    /// per second: from [`MIN_BITS_PER_SAMPLE`] to [`MAX_BITS_PER_SAMPLE`]
    /// for each sample of each channel, rounded inward, and no more than
    /// the identification header's signed field of 32 bits holds. Empty,
    /// the minimum above the maximum, where that field cannot hold the
    /// minimum.
    fn mean_range(&self) -> (u32, u32) {
        let samples = f64::from(self.sample_rate) * self.channels as f64;
        let min = (MIN_BITS_PER_SAMPLE * samples).ceil();
        let max = (MAX_BITS_PER_SAMPLE * samples).floor();
        (
            min.min(f64::from(u32::MAX)) as u32,
            max.min(f64::from(i32::MAX)) as u32,
        )
    }

    /// The bitrate the identification header gives as nominal: that of
    /// [`Rate::Mean`], and 0, none, at the other rates.
    pub(crate) fn nominal_bits_per_second(&self) -> u32 {
        match self.rate {
            Rate::Mean { bits_per_second } => bits_per_second,
            Rate::Variable { .. } | Rate::Quality { .. } => 0,
        }
    }

    /// How each block's noise offset is set, once the configuration is
    /// checked.
    pub(crate) fn control(&self) -> Result<Control, ConfigError> {
        if self.sample_rate == 0 {
            return Err(ConfigError::UnsupportedSampleRate {
                sample_rate: self.sample_rate,
            });
        }
        if !(1..=MAX_CHANNELS).contains(&self.channels) {
            return Err(ConfigError::UnsupportedChannels {
                channels: self.channels,
            });
        }
        match self.rate {
            Rate::Variable { noise_offset_db } if noise_offset_db.is_finite() => {
                Ok(Control::Variable { noise_offset_db })
            }
            Rate::Variable { noise_offset_db } => Err(ConfigError::NoiseOffset { noise_offset_db }),
            Rate::Quality { level } if level <= crate::quality::MAX_LEVEL => {
                Ok(Control::Variable {
                    noise_offset_db: calibration::noise_offset_db(level, self.channels),
                })
            }
            Rate::Quality { level } => Err(ConfigError::QualityLevel { level }),
            Rate::Mean { bits_per_second } => {
                let (min, max) = self.mean_range();
                if (min..=max).contains(&bits_per_second) {
                    Ok(Control::Mean { bits_per_second })
                } else {
                    Err(ConfigError::UnsupportedBitrate {
                        bits_per_second,
                        min,
                        max,
                    })
                }
            }
        }
    }
}

/// How each block's noise offset is set: a [`Rate`], checked.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Control {
    /// At one offset for the whole stream.
    Variable { noise_offset_db: f64 },
    /// Steered toward a mean of `bits_per_second` (see [`Rate::Mean`]).
    Mean { bits_per_second: u32 },
}

/// Why a [`Config`] cannot be encoded.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ConfigError {
    /// The sample rate is 0, which clause 4.2.2 does not allow.
    UnsupportedSampleRate { sample_rate: u32 },
    /// The channels are none, or more than [`MAX_CHANNELS`].
    UnsupportedChannels { channels: usize },
    /// The noise offset of [`Rate::Variable`] is not a finite number.
    NoiseOffset { noise_offset_db: f64 },
    /// The level of [`Rate::Quality`] is above
    /// [`MAX_LEVEL`](crate::quality::MAX_LEVEL).
    QualityLevel { level: u8 },
    /// The bitrate of [`Rate::Mean`] is outside its range at the sample
    /// rate and channels, from `min` to `max`.
    UnsupportedBitrate {
        bits_per_second: u32,
        min: u32,
        max: u32,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::UnsupportedSampleRate { sample_rate } => {
                write!(f, "unsupported sample rate: {sample_rate}")
            }
            Self::UnsupportedChannels { channels } => {
                write!(f, "unsupported number of channels: {channels}")
            }
            Self::NoiseOffset { noise_offset_db } => {
                write!(
                    f,
                    "the noise offset is not a finite number: {noise_offset_db}"
                )
            }
            Self::QualityLevel { level } => write!(
                f,
                "the level of quality is above {}: {level}",
                crate::quality::MAX_LEVEL
            ),
            Self::UnsupportedBitrate {
                bits_per_second,
                min,
                max,
            } => write!(
                f,
                "unsupported bitrate: {bits_per_second} bit/s, not {min} to {max} bit/s"
            ),
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn mean(sample_rate: u32, channels: usize, bits_per_second: u32) -> Config {
        Config::new(sample_rate, channels, Rate::Mean { bits_per_second })
    }

    #[test]
    fn a_mean_rate_is_taken_within_its_range() {
        // 0.125 to 6 bits a sample of each channel.
        for (sample_rate, channels, min, max) in [
            (44_100, 1, 5_513, 264_600),
            (44_100, 2, 11_025, 529_200),
            (48_000, 6, 36_000, 1_728_000),
            (8_000, 1, 1_000, 48_000),
            (1, 1, 1, 6),
        ] {
            for bits_per_second in [min, max] {
                let config = mean(sample_rate, channels, bits_per_second);
                assert_eq!(
                    config.control(),
                    Ok(Control::Mean { bits_per_second }),
                    "{sample_rate} Hz, {channels} channels"
                );
            }
            for bits_per_second in [0, min - 1, max + 1] {
                let config = mean(sample_rate, channels, bits_per_second);
                assert_eq!(
                    config.check(),
                    Err(ConfigError::UnsupportedBitrate {
                        bits_per_second,
                        min,
                        max
                    }),
                    "{sample_rate} Hz, {channels} channels"
                );
            }
        }
    }

    /// The identification header's field is a signed integer of 32 bits.
    #[test]
    fn a_mean_rate_fits_the_identification_header() {
        let config = mean(192_000_000, 8, u32::MAX);
        let max = i32::MAX as u32;
        assert_eq!(
            config.check(),
            Err(ConfigError::UnsupportedBitrate {
                bits_per_second: u32::MAX,
                min: 192_000_000,
                max
            })
        );
        assert_eq!(mean(192_000_000, 8, max).check(), Ok(()));
        // Where the field cannot hold the least, no rate is taken.
        let min = u32::MAX;
        assert_eq!(
            mean(u32::MAX, 8, max).check(),
            Err(ConfigError::UnsupportedBitrate {
                bits_per_second: max,
                min,
                max
            })
        );
    }

    #[test]
    fn the_sample_rate_and_channels_are_checked_first() {
        assert_eq!(
            mean(0, 2, 128_000).check(),
            Err(ConfigError::UnsupportedSampleRate { sample_rate: 0 })
        );
        assert_eq!(
            mean(44_100, 9, 128_000).check(),
            Err(ConfigError::UnsupportedChannels { channels: 9 })
        );
    }
}
