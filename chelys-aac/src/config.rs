// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! What an [`Encoder`](super::Encoder) is asked for, and its checks: a
//! [`Config`] is lowered to the stream's configuration only if every frame
//! the encoder could write is one the standard allows.

use std::fmt;

use super::adts::Flags;
use super::calibration;
use super::channel_configuration::ChannelConfiguration;
use super::rate::Control;
use super::sampling_frequency::SamplingFrequency;
use super::stream::StreamConfig;

/// The sample rates the encoder takes, in Hz: those that have a
/// `sampling_frequency_index` (Table 35), highest first.
pub const SAMPLE_RATES: [u32; 12] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
];

/// How the frames' lengths are chosen.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Rate {
    /// A constant rate: frames as long as a channel of `bits_per_second`
    /// carries, give or take what the bit reservoir lends and keeps
    /// (clause 8.2.3.2), and padded with fill where it is full. The rate
    /// counts every byte written, ADTS headers and their CRC included.
    Constant { bits_per_second: u32 },
    /// A variable rate: each frame as long as it takes to keep the
    /// quantization noise of every band within what the psychoacoustic
    /// model allows it, raised by `noise_offset_db`, its channels then coded
    /// by the iteration loops in those bits; but no shorter than a quarter
    /// of a frame at the rate such streams take on average at the offset,
    /// so that quiet passages that the model takes as inaudible are still
    /// coded, and no longer than the decoder's input buffer, or than 4608
    /// bytes in ADTS (see [`ConfigError::UnsupportedBitrate`]).
    ///
    /// The offset scales the noise each band is allowed by
    /// `10^(noise_offset_db / 10)`: above 0 it allows more noise and gives
    /// shorter frames, below 0 less. At 0 each band is allowed what Annex
    /// C's model allows it. At 44.1 kHz in stereo, music takes about 142
    /// kbit/s at 0 dB, 205 at -6 dB and 73 at 6 dB.
    ///
    /// The tools chosen by the rate are chosen for the whole stream by the
    /// offset. Perceptual noise substitution, where it is asked for, is
    /// used as at the constant rate such streams take on average at the
    /// offset. Up to 0 dB the threshold in quiet is lowered by 20 dB, as if
    /// the signal were played louder, so that quiet passages are coded
    /// further, and from 0 to 6 dB by less and less, and above by none. From
    /// 3.1 to 9.1 dB the threshold in quiet goes over from its reading as
    /// noise to its reading as printed, which allows more noise in the bands
    /// it governs, while the offset raises every band's allowance less; and
    /// from 3.8 to 7 dB intensity stereo comes in, a band at a time from the
    /// top. So a stream's length falls with the offset
    /// without a break, but for steps of a few percent as a band's coding
    /// changes; with perceptual noise substitution, it drops where that
    /// comes in.
    ///
    /// In ADTS each header's `adts_buffer_fullness` is 0x7FF, which says
    /// that the stream is of variable rate (clause 8.1.3.2.2).
    Variable { noise_offset_db: f64 },
    /// A mean rate: a variable rate whose noise offset is steered, frame
    /// by frame, toward keeping `bits_per_second` on average, ADTS headers
    /// and their CRC included. A frame's length follows its content, as at
    /// a variable rate, and the stream's, over some ten seconds, the rate.
    /// The steering only sets the frames to come: frames of silence, or of
    /// a quiet end, which take few bits at any offset, leave the stream
    /// short by what they do not take, and its first frames, before the
    /// steering has learnt what the content takes, can run over or under,
    /// so that a stream of a few seconds can miss the mean by much more
    /// than a long one. The tools chosen by the rate are those of the
    /// variable rate whose nominal rate is `bits_per_second`, which takes
    /// the range of a constant rate.
    Mean { bits_per_second: u32 },
    /// A variable rate at `level`: a level of the [`quality`](crate::quality)
    /// scale that chelys's encoders share, from 0 to
    /// [`MAX_LEVEL`](crate::quality::MAX_LEVEL), higher for better quality.
    ///
    /// It is [`Rate::Variable`] with the noise offset the level maps to,
    /// calibrated to the scale's grades at 44.1 kHz in one channel and in
    /// two: music in stereo averages about 139 kbit/s at level 50, 190 at
    /// 80 and 228 at 100. Configurations of more channels take the offsets
    /// of two, and every other sampling frequency those of 44.1 kHz. At 32
    /// and 48 kHz the grades are expected to be near the scale's, but have
    /// not been measured. At the lower sampling frequencies PEAQ grades
    /// every stream far lower, but there too a higher level allows less
    /// noise.
    Quality { level: u8 },
}

impl Rate {
    /// Whether the stream is coded at a constant rate: the frames' lengths
    /// vary about their mean only as the bit reservoir lends and keeps bits.
    /// An MP4 file gives the measured average of such a stream, and none for
    /// a stream of variable rate (see `chelys-mp4`).
    pub fn is_constant(self) -> bool {
        match self {
            Self::Constant { .. } => true,
            Self::Variable { .. } | Self::Mean { .. } | Self::Quality { .. } => false,
        }
    }
}

/// How each frame is given to the [`FrameSink`](super::FrameSink).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Framing {
    /// Each frame in an `adts_frame()`, whose header tells a decoder all it
    /// needs (clause 8.1.3): a stream that plays as it is, as an `.aac`
    /// file does.
    #[default]
    Adts,
    /// Each frame as its `raw_data_block()` alone, for a container that
    /// keeps the frames apart and carries the
    /// [`Encoder::audio_specific_config`](super::Encoder::audio_specific_config),
    /// such as an MPEG-4 file.
    Raw,
}

/// What to encode, and how: the sample rate, channels and rate, and options
/// set by the methods that take and return it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    sample_rate: u32,
    channels: usize,
    rate: Rate,
    framing: Framing,
    flags: Flags,
    crc: bool,
    noise_substitution: bool,
    data_stream_bytes_per_frame: usize,
}

impl Config {
    /// A configuration of `channels` channels, in ADTS: 1 (mono), 2
    /// (stereo), 3, 4, 5, 6 (5.1) or 8 (7.1), each coded as the
    /// `channel_configuration` of as many speakers (Table 42), and 8 as
    /// configuration 7, which ISO/IEC 14496-3 names "7.1 Front". 7
    /// channels have none, and are refused. [`Encoder::new`] checks it.
    ///
    /// The input gives the channels in the order the stream sends them:
    /// that of the configuration's elements, and in a channel pair, left
    /// before right. The low frequency enhancement channel (LFE) comes
    /// last, and only its lowest frequencies are coded: the 12 lowest of
    /// the 1024 lines of its spectrum, up to 281 Hz at 48 kHz (clause
    /// 8.4.1).
    ///
    /// | `channels` | Order |
    /// |---|---|
    /// | 1 | center |
    /// | 2 | left, right |
    /// | 3 | center, left, right |
    /// | 4 | center, left, right, rear surround |
    /// | 5 | center, left, right, left surround, right surround |
    /// | 6 | center, left, right, left surround, right surround, LFE |
    /// | 8 | center, left center, right center, left, right, left surround, right surround, LFE |
    ///
    /// In 7.1, Table 42 names the first pair the "left, right center front
    /// speakers" and the second the "left, right outside front speakers";
    /// Table 1.23 of ISO/IEC 14496-3 names them the "left, right front
    /// center speakers" and the "left, right front speakers". In 4
    /// channels, Table 42 names the pair the "left, right center front
    /// speakers" and the last channel the "rear surround"; Table 1.23 names
    /// them the "left, right front speakers" and the "rear center
    /// speakers".
    ///
    /// Spec reading: Table 42 lists each configuration's elements "in order
    /// received", and the speakers each maps to, those of a
    /// `channel_pair_element()` as "left, right": its first channel is the
    /// left, as clause 1.6.3.5 of ISO/IEC 14496-3 says outright ("the first
    /// channel element that occurs within the channel pair maps to the next
    /// speaker listed"). Settled by decoding with ffmpeg (black box): it
    /// gives each input channel whole in one of its output channels, at its
    /// level and sign, in the layout it names, its channels in its order:
    ///
    /// | `channels` | ffmpeg's layout | Its channels, from the input's |
    /// |---|---|---|
    /// | 3 | 3.0 | FL left, FR right, FC center |
    /// | 4 | 4.0 | FL left, FR right, FC center, BC rear surround |
    /// | 5 | 5.0 | FL left, FR right, FC center, BL left surround, BR right surround |
    /// | 6 | 5.1 | FL left, FR right, FC center, LFE, BL left surround, BR right surround |
    /// | 8 | 7.1 | FL left center, FR right center, FC center, LFE, BL left surround, BR right surround, SL left, SR right |
    ///
    /// ffmpeg names configuration 7 otherwise than both standards: as its
    /// "7.1", whose front left and right, FL and FR, carry the first pair,
    /// the left and right center front speakers of Table 42, and whose side
    /// left and right, SL and SR, the second, its outside front speakers.
    /// The conformance test is
    /// `ffmpeg_maps_each_channel_of_configurations_3_to_7`.
    ///
    /// FAAD2 (black box) also gives each input channel whole in one of its
    /// output channels, at its level and sign, but names none of them in
    /// its raw output, so it does not settle which channel of a pair is the
    /// left: it gives those of 5.1 in ffmpeg's order, and those of the other
    /// configurations in the order the stream sends them, the input's; the
    /// conformance test is
    /// `faad_maps_each_channel_of_configurations_3_to_7`.
    ///
    /// [`Encoder::new`]: super::Encoder::new
    pub fn new(sample_rate: u32, channels: usize, rate: Rate) -> Self {
        Self {
            sample_rate,
            channels,
            rate,
            framing: Framing::default(),
            flags: Flags::default(),
            crc: false,
            noise_substitution: false,
            data_stream_bytes_per_frame: 0,
        }
    }

    /// How the frames' lengths are chosen.
    pub fn rate(&self) -> Rate {
        self.rate
    }

    /// How each frame is given: in ADTS, the default, or raw.
    pub fn framing(mut self, framing: Framing) -> Self {
        self.framing = framing;
        self
    }

    /// Protects each ADTS frame's header and the start of its audio with a
    /// CRC-16 check word, which decoders may check to detect errors: the
    /// first 192 bits of each channel element and the first 128 of the
    /// second channel of a pair, and all of any data stream element (clause
    /// 8.1.3.2.1). The word's 16 bits count towards the rate. ADTS only.
    pub fn crc(mut self, on: bool) -> Self {
        self.crc = on;
        self
    }

    /// Codes noise-like bands by perceptual noise substitution (ISO/IEC
    /// 14496-3, clause 4.6.13) where bits are few: sends the energy of their
    /// noise in place of their lines, for the decoder to generate noise of
    /// that energy in their place, which saves bits for the bands that need
    /// them. Off by default.
    ///
    /// It is used at rates of up to 1.3 bits for each sample of each
    /// channel but the LFE, about 115 kbit/s in stereo at 44.1 kHz, above
    /// which it costs more than it saves; at a variable rate, where the
    /// rate it is taken as is (see [`Rate::Variable`]).
    ///
    /// Perceptual noise substitution is a tool of MPEG-4 AAC, which MPEG-2
    /// AAC does not have: where it is used, each ADTS header says that the
    /// stream is MPEG-4 (`ID` 0), and a decoder of MPEG-2 AAC alone may not
    /// decode it. Raw frames are described by the
    /// [`Encoder::audio_specific_config`](super::Encoder::audio_specific_config),
    /// which is MPEG-4's either way.
    pub fn noise_substitution(mut self, on: bool) -> Self {
        self.noise_substitution = on;
        self
    }

    /// Sets each ADTS header's `private_bit`, which ISO/IEC 11172-3 defines
    /// as a "Bit for private use", which "will not be used in the future by
    /// ISO/IEC" (clause 8.1.3.2.2 refers to it). ADTS only.
    pub fn private_bit(mut self, on: bool) -> Self {
        self.flags.private_bit = on;
        self
    }

    /// Sets each ADTS header's `original_copy`. ADTS only.
    ///
    /// The standard defines it by reference to ISO/IEC 11172-3's
    /// `copyright`, which is set where the stream is copyright protected,
    /// though its name is that of 11172-3's `original/copy` (clauses
    /// 8.1.3.2.2 and 8.1.2.2.2): the bit is set as asked, and what it
    /// marks is left to the reader.
    pub fn original_copy(mut self, on: bool) -> Self {
        self.flags.original_copy = on;
        self
    }

    /// Sets each ADTS header's `home`. ADTS only.
    ///
    /// The standard defines it by reference to ISO/IEC 11172-3's
    /// `original/copy`, which is set where the stream is an original rather
    /// than a copy (clauses 8.1.3.2.2 and 8.1.2.2.2).
    pub fn home(mut self, on: bool) -> Self {
        self.flags.home = on;
        self
    }

    /// Gives each frame up to `bytes` bytes of data, from the bytes queued
    /// with [`Encoder::push_data_stream`], in data stream elements (clause
    /// 8.6) after its audio, which then has as many bits fewer, and some
    /// more for the elements' fields: 16 bits an element, 24 from 255
    /// bytes, and an element for each 510. A frame with less data queued
    /// keeps the rest for audio. At most as many as the rate brings each
    /// frame beyond the shortest: at a mean rate, its mean, and at a
    /// variable rate, the rate it is taken as for the tools it chooses (see
    /// [`Rate::Variable`]).
    ///
    /// The elements carry one data stream, with the `element_instance_tag`
    /// 0, its bytes unaligned (`data_byte_align_flag` 0). A decoder ignores
    /// them, or gives their bytes, "any additional data, e.g. auxiliary
    /// information, that is not part of the audio information itself"
    /// (clause 8.6.2), in order; as each frame says how many it carries,
    /// they need no framing of their own to be told from the audio.
    ///
    /// [`Encoder::push_data_stream`]: super::Encoder::push_data_stream
    pub fn data_stream_bytes_per_frame(mut self, bytes: usize) -> Self {
        self.data_stream_bytes_per_frame = bytes;
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

    pub(super) fn data_stream_bytes(&self) -> usize {
        self.data_stream_bytes_per_frame
    }

    /// The stream's configuration, if it is one the stream can have.
    pub(super) fn lower(&self) -> Result<StreamConfig, ConfigError> {
        let sample_rate = self.sample_rate;
        let sampling_frequency = SamplingFrequency::from_hz(sample_rate)
            .ok_or(ConfigError::UnsupportedSampleRate { sample_rate })?;
        let channel_configuration = ChannelConfiguration::from_channels(self.channels).ok_or(
            ConfigError::UnsupportedChannels {
                channels: self.channels,
            },
        )?;
        if self.framing == Framing::Raw && (self.crc || self.flags != Flags::default()) {
            return Err(ConfigError::AdtsOnly);
        }
        let mut config = StreamConfig {
            sampling_frequency,
            channel_configuration,
            bits_per_second: 0,
            control: Control::Constant,
            framing: self.framing,
            flags: self.flags,
            crc_protected: self.crc,
            noise_substitution: self.noise_substitution,
        };
        let rate = match self.rate {
            Rate::Quality { level } => {
                if level > crate::quality::MAX_LEVEL {
                    return Err(ConfigError::QualityLevel { level });
                }
                Rate::Variable {
                    noise_offset_db: calibration::noise_offset_db(level, self.channels),
                }
            }
            rate => rate,
        };
        let bits_per_second = match rate {
            Rate::Constant { bits_per_second } => bits_per_second,
            Rate::Quality { .. } => unreachable!("lowered to a variable rate"),
            Rate::Variable { noise_offset_db } => {
                if !noise_offset_db.is_finite() {
                    return Err(ConfigError::NoiseOffset { noise_offset_db });
                }
                config.control = Control::Variable { noise_offset_db };
                config.nominal_bits_per_second(noise_offset_db)
            }
            Rate::Mean { bits_per_second } => {
                config.control = Control::Mean;
                bits_per_second
            }
        };
        config.bits_per_second = bits_per_second;
        if !config.is_valid() {
            return Err(ConfigError::UnsupportedBitrate {
                bits_per_second,
                min: config.min_bits_per_second(),
                max: config.max_bits_per_second(),
            });
        }
        let max = config.max_data_stream_bytes();
        if self.data_stream_bytes_per_frame > max {
            return Err(ConfigError::DataStreamTooLong {
                bytes_per_frame: self.data_stream_bytes_per_frame,
                max,
            });
        }
        Ok(config)
    }
}

/// Why a [`Config`] cannot be encoded.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ConfigError {
    /// The sample rate is not one of [`SAMPLE_RATES`].
    UnsupportedSampleRate { sample_rate: u32 },
    /// No channel configuration has as many channels (see [`Config::new`]).
    UnsupportedChannels { channels: usize },
    /// The bitrate is outside the range, from `min` to `max`, of the sample
    /// rate, channels and framing: below the bits of the shortest frame, or
    /// above those of the decoder's input buffer, 6144 a channel, the
    /// low frequency enhancement channel not counted (clauses 8.2.3.1 and
    /// 8.2.3.3), or in ADTS those of a frame of 4608 bytes where they are
    /// fewer, as in 7.1, the longest frame FAAD2 decodes; less a byte for
    /// the bit reservoir.
    UnsupportedBitrate {
        bits_per_second: u32,
        min: u32,
        max: u32,
    },
    /// The CRC or a header flag was asked for in [`Framing::Raw`], whose
    /// frames have no ADTS header to carry them.
    AdtsOnly,
    /// More bytes of data streams a frame were asked for than the rate
    /// brings each frame beyond the shortest, `max`.
    DataStreamTooLong { bytes_per_frame: usize, max: usize },
    /// The noise offset of [`Rate::Variable`] is not a finite number.
    NoiseOffset { noise_offset_db: f64 },
    /// The level of [`Rate::Quality`] is above
    /// [`MAX_LEVEL`](crate::quality::MAX_LEVEL).
    QualityLevel { level: u8 },
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
            Self::UnsupportedBitrate {
                bits_per_second,
                min,
                max,
            } => write!(
                f,
                "unsupported bitrate: {bits_per_second} bit/s, not {min} to {max} bit/s"
            ),
            Self::AdtsOnly => write!(
                f,
                "raw frames have no ADTS header to carry its CRC or flags"
            ),
            Self::DataStreamTooLong {
                bytes_per_frame,
                max,
            } => write!(
                f,
                "{bytes_per_frame} bytes of data streams a frame exceed the {max} the bitrate \
                 brings each frame beyond the shortest"
            ),
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
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Encoder;

    fn config(framing: Framing) -> Config {
        Config::new(
            48_000,
            2,
            Rate::Constant {
                bits_per_second: 96_000,
            },
        )
        .framing(framing)
    }

    #[test]
    fn every_adts_header_carries_the_flags() {
        let config = config(Framing::Adts)
            .private_bit(true)
            .original_copy(false)
            .home(true);
        let mut encoder = Encoder::new(config).unwrap();
        let mut frames = Vec::new();
        let mut out = |frame: &[u8]| frames.push(frame.to_vec());
        encoder.encode_interleaved(&[0.0f32; 2 * 4096], &mut out);
        encoder.finish(&mut out);
        assert_eq!(frames.len(), 5);
        for frame in frames {
            // private_bit is bit 22, original_copy 26 and home 27.
            assert_eq!(frame[2] & 0b10, 0b10);
            assert_eq!((frame[3] >> 4) & 0b11, 0b01);
        }
    }

    #[test]
    fn a_frame_carries_at_most_what_the_rate_brings_beyond_the_shortest() {
        // 256 bytes a frame, of which the header takes 7, or 9 with the
        // CRC, and a pair of silent channels that share their window, with
        // its id_syn_ele, ID_END and the data stream element's fields, 62
        // bits, within 8 bytes with the data.
        let max = config(Framing::Adts)
            .lower()
            .unwrap()
            .max_data_stream_bytes();
        assert_eq!(max, 256 - 7 - 8);
        let with_crc = config(Framing::Adts).crc(true).lower().unwrap();
        assert_eq!(with_crc.max_data_stream_bytes(), 256 - 9 - 8);
        for framing in [Framing::Adts, Framing::Raw] {
            let ok = config(framing).data_stream_bytes_per_frame(max);
            assert_eq!(ok.check(), Ok(()));
        }
        let over = config(Framing::Adts).data_stream_bytes_per_frame(max + 1);
        assert_eq!(
            over.check(),
            Err(ConfigError::DataStreamTooLong {
                bytes_per_frame: max + 1,
                max
            })
        );
    }

    /// The `ID` bit of each ADTS frame of a stereo stream at 44.1 kHz at
    /// `kbps`, with perceptual noise substitution if `noise`: 1 for MPEG-2
    /// AAC, 0 for MPEG-4.
    fn ids(kbps: u32, noise: bool) -> Vec<u8> {
        let config = Config::new(
            44_100,
            2,
            Rate::Constant {
                bits_per_second: kbps * 1000,
            },
        )
        .noise_substitution(noise);
        let mut encoder = Encoder::new(config).unwrap();
        let mut frames = Vec::new();
        let mut out = |frame: &[u8]| frames.push(frame.to_vec());
        encoder.encode_interleaved(&[0.0f32; 2 * 4096], &mut out);
        encoder.finish(&mut out);
        // The bit after the 12 bits of the syncword.
        frames.iter().map(|frame| (frame[1] >> 3) & 1).collect()
    }

    #[test]
    fn noise_substitution_makes_the_stream_mpeg_4_where_it_is_used() {
        assert!(ids(96, false).iter().all(|&id| id == 1));
        assert!(ids(96, true).iter().all(|&id| id == 0));
        // 1.3 bits a sample, 114.66 kbit/s, and no more.
        assert!(ids(114, true).iter().all(|&id| id == 0));
        assert!(ids(115, true).iter().all(|&id| id == 1));
    }

    #[test]
    fn raw_frames_have_no_header_for_the_crc_or_flags() {
        assert_eq!(config(Framing::Raw).check(), Ok(()));
        assert_eq!(config(Framing::Raw).home(false).check(), Ok(()));
        for config in [
            config(Framing::Raw).crc(true),
            config(Framing::Raw).private_bit(true),
            config(Framing::Raw).original_copy(true),
            config(Framing::Raw).home(true),
        ] {
            assert_eq!(config.check(), Err(ConfigError::AdtsOnly));
        }
    }
}
