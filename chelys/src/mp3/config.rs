// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! What an [`Encoder`](super::Encoder) is asked for, and its checks: a
//! [`Config`] is lowered to the stream's configuration only if every frame
//! the encoder could write is one the standard allows.

use std::fmt;

use super::header::{Bitrate, Flags, Mode, SamplingFrequency, Version};
use super::rate::{self, MeanRate, VariableRate};
use super::stream::{StreamConfig, max_ancillary_bits};

/// The sample rates the encoder takes, in Hz: those of MPEG-1, those of the
/// lower sampling frequencies of ISO/IEC 13818-3, and those of MPEG-2.5.
pub const SAMPLE_RATES: [u32; 9] = [
    32_000, 44_100, 48_000, 16_000, 22_050, 24_000, 8_000, 11_025, 12_000,
];

/// The bitrates a frame may have at `sample_rate`, in kbit/s, lowest first:
/// those of [`Rate::Constant`] and the bounds of [`Rate::Variable`]. `None`
/// if the sample rate is not one of [`SAMPLE_RATES`].
pub fn bitrates(sample_rate: u32) -> Option<&'static [u32]> {
    sampling_frequency(sample_rate).map(|fs| fs.version().table_kbps())
}

fn sampling_frequency(sample_rate: u32) -> Option<SamplingFrequency> {
    SamplingFrequency::ALL
        .into_iter()
        .find(|fs| fs.hz() == sample_rate)
}

/// How the channels are coded, and with them how many there are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelMode {
    Mono,
    /// Two channels, coded as left and right throughout.
    Stereo,
    /// Two channels, each frame coded as left and right or with the tools
    /// allowed here, as suits the frame (Annex G.2).
    JointStereo {
        /// The encoder may code a frame whose channels are alike as their
        /// sum and difference.
        ms_stereo: bool,
        /// The encoder may code the upper bands of a frame short of bits
        /// by the level of each channel alone, giving up their fine
        /// structure there: a trade for low bitrates.
        intensity_stereo: bool,
    },
    /// Two channels with independent programmes, such as two languages,
    /// coded as left and right throughout (clause 2.1.52).
    DualChannel,
}

impl ChannelMode {
    pub fn channels(self) -> usize {
        match self {
            Self::Mono => 1,
            _ => 2,
        }
    }

    fn mode(self) -> Mode {
        match self {
            Self::Mono => Mode::SingleChannel,
            Self::Stereo => Mode::Stereo,
            Self::JointStereo {
                ms_stereo,
                intensity_stereo,
            } => Mode::JointStereo {
                intensity_stereo,
                ms_stereo,
            },
            Self::DualChannel => Mode::DualChannel,
        }
    }
}

/// How each frame's bitrate is chosen (clause 2.4.2.3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rate {
    /// Every frame at one of the [`bitrates`] of the sample rate.
    Constant { kbps: u32 },
    /// A mean bitrate between the lowest and the highest of the
    /// [`bitrates`], kept by switching between the two nearest to it.
    Mean { bits_per_second: u32 },
    /// Each frame at the lowest bitrate, from `min_kbps` to `max_kbps`, at
    /// which the quantisation noise of every band stays within what the
    /// psychoacoustic model allows it, raised by `noise_offset_db`.
    ///
    /// The offset scales the noise each band is allowed by
    /// `10^(noise_offset_db / 10)`: above 0 it allows more noise and gives
    /// smaller frames, below 0 less. 0 is Annex C's encoding process,
    /// which takes the model's allowance as it is; listeners need less
    /// than that.
    Variable {
        /// One of the [`bitrates`]; `None` for the lowest.
        min_kbps: Option<u32>,
        /// One of the [`bitrates`]; `None` for the highest.
        max_kbps: Option<u32>,
        noise_offset_db: f64,
    },
    /// Every frame at a bitrate of free format, which need not be one of
    /// the [`bitrates`]: for links of a fixed capacity. Not every decoder
    /// decodes it.
    Free { bits_per_second: u32 },
}

/// What to encode, and how: the sample rate, channels and rate, and options
/// set by the methods that take and return it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    sample_rate: u32,
    mode: ChannelMode,
    rate: Rate,
    crc: bool,
    info_tag: bool,
    mixed_blocks: bool,
    flags: Flags,
    ancillary_bits_per_frame: usize,
}

impl Config {
    /// A configuration with every option off. [`Encoder::new`] checks it.
    ///
    /// [`Encoder::new`]: super::Encoder::new
    pub fn new(sample_rate: u32, mode: ChannelMode, rate: Rate) -> Self {
        Self {
            sample_rate,
            mode,
            rate,
            crc: false,
            info_tag: false,
            mixed_blocks: false,
            flags: Flags::default(),
            ancillary_bits_per_frame: 0,
        }
    }

    /// Protects each frame's header and side information with a CRC-16
    /// check word.
    pub fn crc(mut self, on: bool) -> Self {
        self.crc = on;
        self
    }

    /// Starts the stream with a frame for a Xing/Info tag: the stream's
    /// length, a table to seek by, and the samples to trim at its start and
    /// end, which make the decoded output the input exactly for decoders
    /// that apply them.
    ///
    /// The tag is known only once the stream has ended, so its frame is
    /// written first as a placeholder, a silent frame that decoders play as
    /// such, and [`Finished::info_tag`] gives the frame to write over it:
    /// ask for the tag only where the start of the stream can be rewritten.
    /// [`Encoder::info_tag_len`] gives its length. A stream in free format
    /// whose frames are too short for the tag has none.
    ///
    /// [`Finished::info_tag`]: super::Finished::info_tag
    /// [`Encoder::info_tag_len`]: super::Encoder::info_tag_len
    pub fn info_tag(mut self, on: bool) -> Self {
        self.info_tag = on;
        self
    }

    /// Where an attack switches to short blocks, keeps the normal block in
    /// the two lowest subbands, for their frequency resolution (clause
    /// 2.4.2.7). Frames with them are not coded in intensity stereo. Not at
    /// 8 kHz.
    pub fn mixed_blocks(mut self, on: bool) -> Self {
        self.mixed_blocks = on;
        self
    }

    /// Marks the stream as copyright protected.
    pub fn copyright(mut self, on: bool) -> Self {
        self.flags.copyright = on;
        self
    }

    /// Marks the stream as an original rather than a copy.
    pub fn original(mut self, on: bool) -> Self {
        self.flags.original = on;
        self
    }

    /// Sets each frame's bit for private use.
    pub fn private_bit(mut self, on: bool) -> Self {
        self.flags.private = on;
        self
    }

    /// Gives each frame up to `bits` bits of ancillary data, from the bytes
    /// queued with [`Encoder::push_ancillary`], after its audio, which then
    /// has as many bits fewer (clause C.1.5.3.6). A frame with less data
    /// queued keeps the rest for audio. At most as many as the shortest
    /// frame of the stream holds.
    ///
    /// As the standard leaves the bits "user definable" (clause 2.4.2.8),
    /// a reader cannot tell the queued bits from the zeros after them by
    /// the stream alone, and the data needs a framing of its own. Nor is it
    /// kept from imitating a syncword.
    ///
    /// [`Encoder::push_ancillary`]: super::Encoder::push_ancillary
    pub fn ancillary_bits_per_frame(mut self, bits: usize) -> Self {
        self.ancillary_bits_per_frame = bits;
        self
    }

    /// Whether [`Encoder::new`] takes this configuration.
    ///
    /// # Errors
    ///
    /// As [`Encoder::new`].
    ///
    /// [`Encoder::new`]: super::Encoder::new
    pub fn check(&self) -> Result<(), ConfigError> {
        self.lower().map(|_| ())
    }

    pub(super) fn wants_info_tag(&self) -> bool {
        self.info_tag
    }

    pub(super) fn wants_mixed_blocks(&self) -> bool {
        self.mixed_blocks
    }

    pub(super) fn ancillary_bits(&self) -> usize {
        self.ancillary_bits_per_frame
    }

    /// The stream's configuration, if every option is one the stream can
    /// have.
    pub(super) fn lower(&self) -> Result<StreamConfig, ConfigError> {
        let sample_rate = self.sample_rate;
        let fs = sampling_frequency(sample_rate)
            .ok_or(ConfigError::UnsupportedSampleRate { sample_rate })?;
        let version = fs.version();
        let rate = lower_rate(self.rate, sample_rate, version)?;
        if self.mixed_blocks && fs == SamplingFrequency::Hz8000 {
            return Err(ConfigError::MixedBlocksUnsupported { sample_rate });
        }
        let config = StreamConfig {
            rate,
            sampling_frequency: fs,
            mode: self.mode.mode(),
            crc_protected: self.crc,
            flags: self.flags,
        };
        let max = max_ancillary_bits(&config);
        if self.ancillary_bits_per_frame > max {
            return Err(ConfigError::AncillaryTooLong {
                bits_per_frame: self.ancillary_bits_per_frame,
                max,
            });
        }
        Ok(config)
    }
}

fn lower_rate(rate: Rate, sample_rate: u32, version: Version) -> Result<rate::Rate, ConfigError> {
    let table = |kbps: u32| {
        Bitrate::from_kbps(version, kbps)
            .ok_or(ConfigError::UnsupportedBitrate { sample_rate, kbps })
    };
    Ok(match rate {
        Rate::Constant { kbps } => rate::Rate::Constant(table(kbps)?),
        Rate::Mean { bits_per_second } => rate::Rate::Mean(
            MeanRate::new(version, bits_per_second).ok_or(ConfigError::UnsupportedMeanBitrate {
                sample_rate,
                bits_per_second,
            })?,
        ),
        Rate::Variable {
            min_kbps,
            max_kbps,
            noise_offset_db,
        } => {
            let mut bitrates = Bitrate::all(version);
            let lowest = match min_kbps {
                Some(kbps) => table(kbps)?,
                None => bitrates.next().expect("a table of bitrates"),
            };
            let highest = match max_kbps {
                Some(kbps) => table(kbps)?,
                None => bitrates.last().expect("a table of bitrates"),
            };
            if lowest.bits_per_second() > highest.bits_per_second() {
                return Err(ConfigError::BitrateRange {
                    min_kbps: lowest.bits_per_second() / 1000,
                    max_kbps: highest.bits_per_second() / 1000,
                });
            }
            if !noise_offset_db.is_finite() {
                return Err(ConfigError::NoiseOffset { noise_offset_db });
            }
            rate::Rate::Variable(VariableRate {
                lowest,
                highest,
                noise_offset_db,
            })
        }
        Rate::Free { bits_per_second } => {
            rate::Rate::Constant(Bitrate::free(version, bits_per_second).ok_or_else(|| {
                let in_table = bits_per_second.is_multiple_of(1000)
                    && Bitrate::from_kbps(version, bits_per_second / 1000).is_some();
                ConfigError::UnsupportedFreeBitrate {
                    sample_rate,
                    bits_per_second,
                    in_table,
                }
            })?)
        }
    })
}

/// Why a [`Config`] cannot be encoded.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ConfigError {
    /// The sample rate is not one of [`SAMPLE_RATES`].
    UnsupportedSampleRate { sample_rate: u32 },
    /// A bitrate of [`Rate::Constant`] or [`Rate::Variable`] is not one of
    /// the [`bitrates`] of the sample rate.
    UnsupportedBitrate { sample_rate: u32, kbps: u32 },
    /// The mean bitrate is below the lowest of the [`bitrates`] of the
    /// sample rate, or above the highest.
    UnsupportedMeanBitrate {
        sample_rate: u32,
        bits_per_second: u32,
    },
    /// The bitrate of free format is one of the [`bitrates`] (`in_table`),
    /// which free format does not code; or is not below the highest of
    /// them; or is so low that a frame cannot hold its header and side
    /// information.
    UnsupportedFreeBitrate {
        sample_rate: u32,
        bits_per_second: u32,
        in_table: bool,
    },
    /// The lowest bitrate of [`Rate::Variable`] is above its highest.
    BitrateRange { min_kbps: u32, max_kbps: u32 },
    /// The noise offset of [`Rate::Variable`] is not a finite number.
    NoiseOffset { noise_offset_db: f64 },
    /// Mixed blocks were asked for at 8 kHz, where they are not used.
    MixedBlocksUnsupported { sample_rate: u32 },
    /// More bits of ancillary data a frame were asked for than the shortest
    /// frame of the stream holds.
    AncillaryTooLong { bits_per_frame: usize, max: usize },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::UnsupportedSampleRate { sample_rate } => {
                write!(f, "unsupported sample rate: {sample_rate}")
            }
            Self::UnsupportedBitrate { sample_rate, kbps } => {
                write!(f, "unsupported bitrate at {sample_rate} Hz: {kbps} kbit/s")
            }
            Self::UnsupportedMeanBitrate {
                sample_rate,
                bits_per_second,
            } => write!(
                f,
                "unsupported mean bitrate at {sample_rate} Hz: {bits_per_second} bit/s"
            ),
            Self::UnsupportedFreeBitrate {
                sample_rate,
                bits_per_second,
                in_table: true,
            } => write!(
                f,
                "{bits_per_second} bit/s at {sample_rate} Hz is not a bitrate of free format, \
                 but one of the table"
            ),
            Self::UnsupportedFreeBitrate {
                sample_rate,
                bits_per_second,
                in_table: false,
            } => write!(
                f,
                "unsupported free format bitrate at {sample_rate} Hz: {bits_per_second} bit/s"
            ),
            Self::BitrateRange { min_kbps, max_kbps } => write!(
                f,
                "the lowest bitrate, {min_kbps} kbit/s, is above the highest, {max_kbps} kbit/s"
            ),
            Self::NoiseOffset { noise_offset_db } => {
                write!(
                    f,
                    "the noise offset is not a finite number: {noise_offset_db}"
                )
            }
            Self::MixedBlocksUnsupported { sample_rate } => {
                write!(f, "mixed blocks are not used at {sample_rate} Hz")
            }
            Self::AncillaryTooLong {
                bits_per_frame,
                max,
            } => write!(
                f,
                "{bits_per_frame} bits of ancillary data a frame exceed the {max} of the \
                 shortest frame"
            ),
        }
    }
}

impl std::error::Error for ConfigError {}
