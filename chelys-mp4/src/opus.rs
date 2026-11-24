// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Opus in MP4 files, from "Encapsulation of Opus in ISO Base Media File
//! Format" version 1.0.0 (xiph/opus, `doc/opus_in_isobmff.html` at commit
//! fa9c1e7), which this module cites as "the Opus mapping", RFC 7845, Ogg
//! Opus, whose identification header the mapping's `OpusSpecificBox`
//! follows, and RFC 6716, Opus, for the duration of a packet. Each packet,
//! whichever encoder made it, is a sample of an `'Opus'` track, which the
//! stream's identification header describes.
//!
//! A decoder's output is at 48 kHz, whatever the input's rate, and starts
//! with the stream's pre-skip, samples that precede the input: the track's
//! timescale is 48 kHz, and its edit leaves the pre-skip out. Each packet
//! decodes by itself, but correctly only after the packets of the 80 ms
//! before it: the track has a pre-roll of them.

use std::fmt;

use crate::boxes::{FourCc, write_box};
use crate::movie::{AudioTrack, Brands, SampleEntry};
use crate::packets::PacketError;

/// The rate of a decoder's output, at which every duration is counted.
pub const SAMPLE_RATE: u32 = 48_000;

/// The pre-roll that every packet needs, in samples at 48 kHz: "at least
/// 80 millisecond pre-roll after each random access" (the Opus mapping,
/// clause 4.3.6.2).
const PRE_ROLL: u32 = 3840;

/// The longest packet, 120 ms: "the audio duration contained within a
/// packet MUST NOT exceed 120 ms" (RFC 6716, clause 3.2.5).
const MAX_PACKET: u32 = 5760;

/// The shortest packet, of one frame of 2.5 ms (RFC 6716, Table 2).
const MIN_PACKET: u32 = 120;

/// What an `'Opus'` track needs of an Opus stream: the fields of its
/// identification header (RFC 7845, clause 5.1), and the duration of its
/// shortest packet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    channels: u8,
    pre_skip: u16,
    input_sample_rate: u32,
    output_gain: i16,
    mapping: Option<ChannelMapping>,
    shortest_packet: u32,
}

/// A channel mapping of a family other than 0, with its table (RFC 7845,
/// clause 5.1.1): the streams that each packet holds, how many of them are
/// coupled, stereo, and for each output channel the decoded channel it
/// takes, or 255 for silence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelMapping {
    /// The channel mapping family: 1, of up to eight channels in Vorbis
    /// order, or 255, of channels of no defined meaning.
    pub family: u8,
    pub stream_count: u8,
    pub coupled_count: u8,
    /// The decoded channel of each output channel, one per channel.
    pub mapping: Vec<u8>,
}

/// Why a configuration was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConfigError {
    /// An identification header that does not start with "OpusHead", or of
    /// a version whose upper four bits are not 0 (RFC 7845, clause 5.1), or
    /// too short for its fields.
    Header,
    /// No channels, or more than the channel mapping family allows: 1 or 2
    /// of family 0, 1 to 8 of family 1 and 1 to 255 of family 255.
    Channels(u8),
    /// A channel mapping family that RFC 7845 does not define: 2 to 254.
    Family(u8),
    /// A channel mapping table that RFC 7845 forbids: no streams, more
    /// coupled streams than streams, more than 255 decoded channels, or a
    /// channel mapped to a decoded channel that does not exist.
    Mapping,
    /// A shortest packet that is not from 2.5 to 120 ms long, 120 to 5760
    /// samples at 48 kHz.
    ShortestPacket(u32),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Header => write!(f, "not an Ogg Opus identification header"),
            Self::Channels(channels) => write!(
                f,
                "{channels} channels, which the channel mapping family does not allow"
            ),
            Self::Family(family) => write!(
                f,
                "channel mapping family {family}, which RFC 7845 does not define"
            ),
            Self::Mapping => write!(f, "a channel mapping table that RFC 7845 forbids"),
            Self::ShortestPacket(samples) => write!(
                f,
                "a shortest packet of {samples} samples, where Opus has 120 to 5760"
            ),
        }
    }
}

impl std::error::Error for ConfigError {}

impl Config {
    /// The configuration of a stream of `channels`, 1 or 2, of channel
    /// mapping family 0, whose decoder's output starts with `pre_skip`
    /// samples at 48 kHz that precede the input, of an input sample rate
    /// that is not given and an output gain of 0, and whose shortest packet
    /// lasts 20 ms.
    ///
    /// # Errors
    ///
    /// [`ConfigError::Channels`] unless `channels` is 1 or 2.
    pub fn new(channels: u8, pre_skip: u16) -> Result<Self, ConfigError> {
        if !(1..=2).contains(&channels) {
            return Err(ConfigError::Channels(channels));
        }
        Ok(Self {
            channels,
            pre_skip,
            input_sample_rate: 0,
            output_gain: 0,
            mapping: None,
            shortest_packet: 960,
        })
    }

    /// The configuration of a stream whose channels `mapping` gives, of
    /// family 1 or 255, as [`new`](Self::new) otherwise.
    ///
    /// Spec reading: the families 2 to 254 are reserved (RFC 7845, clause
    /// 5.1.1.4), and are refused, though "A demuxer implementation
    /// encountering a reserved 'channel mapping family' value SHOULD act as
    /// though the value is 255". A writer that read such a header's table
    /// as family 255's could misread a family defined since, whose table
    /// differs, and the `OpusSpecificBox` would not carry it as its
    /// `ChannelMappingTable` is defined (the Opus mapping, clause 4.3.2).
    ///
    /// # Errors
    ///
    /// [`ConfigError`] if the mapping is not one RFC 7845 allows.
    pub fn with_channel_mapping(
        pre_skip: u16,
        mapping: ChannelMapping,
    ) -> Result<Self, ConfigError> {
        let channels = u8::try_from(mapping.mapping.len()).map_err(|_| ConfigError::Mapping)?;
        let allowed = match mapping.family {
            1 => 8,
            255 => 255,
            0 => return Err(ConfigError::Mapping),
            family => return Err(ConfigError::Family(family)),
        };
        if !(1..=allowed).contains(&channels) {
            return Err(ConfigError::Channels(channels));
        }
        // "This value MUST NOT be zero", "This MUST be no larger than the
        // total number of streams", "(M + N), MUST be no larger than 255",
        // and each index "MUST either be smaller than (M + N) or be the
        // special value 255" (clause 5.1.1).
        let (n, m) = (
            u32::from(mapping.stream_count),
            u32::from(mapping.coupled_count),
        );
        let decoded = n + m;
        let indices = (mapping.mapping.iter()).all(|&i| u32::from(i) < decoded || i == 255);
        if n == 0 || m > n || decoded > 255 || !indices {
            return Err(ConfigError::Mapping);
        }
        Ok(Self {
            channels,
            mapping: Some(mapping),
            ..Self::new(1, pre_skip)?
        })
    }

    /// The configuration of a stream whose identification header, the
    /// "OpusHead" packet of an Ogg Opus stream (RFC 7845, clause 5.1), is
    /// `header`, as [`new`](Self::new) otherwise. Any fields that a later
    /// minor version adds after the channel mapping are not kept, as the
    /// Opus mapping has none.
    ///
    /// # Errors
    ///
    /// [`ConfigError`] if `header` is not an identification header of a
    /// version this one is compatible with, or gives channels or a mapping
    /// that RFC 7845 does not allow.
    pub fn from_id_header(header: &[u8]) -> Result<Self, ConfigError> {
        // "an implementation of this specification SHOULD accept any stream
        // with a version number of '15' or less".
        let fixed = header.get(..19).ok_or(ConfigError::Header)?;
        if &fixed[..8] != b"OpusHead" || fixed[8] > 15 {
            return Err(ConfigError::Header);
        }
        let channels = fixed[9];
        let pre_skip = u16::from_le_bytes([fixed[10], fixed[11]]);
        let input_sample_rate = u32::from_le_bytes([fixed[12], fixed[13], fixed[14], fixed[15]]);
        let output_gain = i16::from_le_bytes([fixed[16], fixed[17]]);
        let config = match fixed[18] {
            0 => Self::new(channels, pre_skip)?,
            family => {
                let table = header.get(19..21 + usize::from(channels));
                let table = table.ok_or(ConfigError::Header)?;
                let mapping = ChannelMapping {
                    family,
                    stream_count: table[0],
                    coupled_count: table[1],
                    mapping: table[2..].to_vec(),
                };
                Self::with_channel_mapping(pre_skip, mapping)?
            }
        };
        Ok(config
            .input_sample_rate(input_sample_rate)
            .output_gain(output_gain))
    }

    /// Gives the sample rate of the input before it was encoded, in Hz,
    /// which is not the rate a decoder outputs (RFC 7845, clause 5.1).
    pub fn input_sample_rate(mut self, hz: u32) -> Self {
        self.input_sample_rate = hz;
        self
    }

    /// Gives a gain to apply when decoding, in dB, in Q7.8 fixed point
    /// (RFC 7845, clause 5.1).
    pub fn output_gain(mut self, q7_8: i16) -> Self {
        self.output_gain = q7_8;
        self
    }

    /// Declares the duration of the stream's shortest packet, in samples at
    /// 48 kHz; by default 20 ms, 960. It sets the pre-roll, the packets
    /// before each that a reader that starts there decodes first: as many
    /// as make 80 ms. Packets that last less are refused.
    ///
    /// # Errors
    ///
    /// [`ConfigError::ShortestPacket`] unless `samples` is from 2.5 to 120
    /// ms long, 120 to 5760.
    pub fn shortest_packet(mut self, samples: u32) -> Result<Self, ConfigError> {
        if !(MIN_PACKET..=MAX_PACKET).contains(&samples) {
            return Err(ConfigError::ShortestPacket(samples));
        }
        self.shortest_packet = samples;
        Ok(self)
    }

    /// The number of output channels.
    pub fn channels(&self) -> u8 {
        self.channels
    }

    /// The samples at 48 kHz that a decoder's output starts with before
    /// the input.
    pub fn pre_skip(&self) -> u16 {
        self.pre_skip
    }

    /// The track of the stream.
    ///
    /// Its timescale is 48 kHz, as the Opus mapping has "typically", for
    /// sample-accurate trimming: the rate a decoder outputs, in which the
    /// pre-skip and every packet's duration are counted (clause 4.4).
    ///
    /// Its one edit leaves out the pre-skip, as clause 4.4 requires: "the
    /// Edit Box and the Edit List Box shall be present", and of its example
    /// edit, "`media_time` = the number of the priming samples".
    ///
    /// Its pre-roll is the packets that make 80 ms, of the shortest
    /// declared: "Pre-roll is indicated by the roll_distance field in
    /// AudioRollRecoveryEntry", "expressed in sample units", "and always
    /// takes negative values" (clause 4.3.6.2). Every packet is mapped to
    /// it, as "at least one Sample Group Description Box and at least one
    /// Sample to Group Box ... shall be present" and in every track
    /// fragment that holds a packet.
    pub(crate) fn track(&self) -> AudioTrack {
        let packets = PRE_ROLL.div_ceil(self.shortest_packet);
        AudioTrack {
            brands: BRANDS,
            sample_entry: self.sample_entry(),
            timescale: SAMPLE_RATE,
            channels: u16::from(self.channels),
            delay: Some(u32::from(self.pre_skip)),
            default_duration: self.shortest_packet,
            roll_distance: Some(-(packets as i16)),
        }
    }

    /// The `OpusSampleEntry` (clause 4.3.1): an `AudioSampleEntry` of
    /// `'Opus'`, its `channelcount` the output channels, its `samplesize`
    /// 16 and its `samplerate` 48 kHz, as "shall be set", with an
    /// `OpusSpecificBox`.
    fn sample_entry(&self) -> SampleEntry {
        const OPUS: FourCc = *b"Opus";
        let mut boxes = Vec::new();
        // The OpusSpecificBox (clause 4.3.2), a Box and not a FullBox, of
        // the OpusDecoderConfigurationRecord: its Version 0, then the
        // fields of the identification header, each "set to the same
        // value", but big-endian, as "all the data are stored as
        // big-endian format".
        //
        // Spec reading: the PreSkip "shall be at least 80 milliseconds'
        // worth of PCM samples even when removing any number of Opus
        // samples which may or may not contain the priming samples", but
        // also "indicates the number of the priming samples", which the
        // edit leaves out. An encoder's pre-skip, such as 312 samples, is
        // less than 80 ms. A PreSkip of 80 ms would give more priming
        // samples than the stream has, and a reader that skipped them would
        // lose the input's start; the field is "informative only", and so
        // gives the pre-skip that the stream has, as its identification
        // header does, as the mapping's example has a PreSkip of 312 with
        // an edit of `media_time` 312 (clause 4.7), and as "a value of at
        // least 3840 samples (80 ms)" is for streams that are cropped (RFC
        // 7845, clause 4.2), which a writer of the encoder's packets does
        // not do.
        write_box(&mut boxes, b"dOps", |out| {
            out.push(0); // Version
            out.push(self.channels); // OutputChannelCount
            out.extend_from_slice(&self.pre_skip.to_be_bytes()); // PreSkip
            out.extend_from_slice(&self.input_sample_rate.to_be_bytes()); // InputSampleRate
            out.extend_from_slice(&self.output_gain.to_be_bytes()); // OutputGain
            match &self.mapping {
                None => out.push(0), // ChannelMappingFamily
                Some(mapping) => {
                    out.push(mapping.family);
                    out.push(mapping.stream_count);
                    out.push(mapping.coupled_count);
                    out.extend_from_slice(&mapping.mapping);
                }
            }
        });
        SampleEntry {
            coding_name: OPUS,
            sample_size: 16,
            sample_rate: SAMPLE_RATE << 16,
            boxes,
        }
    }

    /// The duration of `packet`, as [`packet_duration`], if it is no
    /// shorter than the shortest packet declared.
    pub(crate) fn duration(&self, packet: &[u8]) -> Result<u32, PacketError> {
        let duration = packet_duration(packet)?;
        if duration < self.shortest_packet {
            return Err(PacketError::OpusShorterThanDeclared {
                duration,
                shortest: self.shortest_packet,
            });
        }
        Ok(duration)
    }
}

/// The brands of the `FileTypeBox`: `'Opus'`, which the Opus mapping
/// defines "to declare files are conformant to this specification", as
/// major brand and compatible, and `'iso2'`, as "the minimal support of
/// the encapsulation of Opus bitstreams in ISO Base Media file format
/// requires the 'iso2' brand in the compatible brands list since support
/// of roll groups is required" (clause 4.1), as its example has them
/// (clause 4.7).
const BRANDS: Brands = Brands {
    major: *b"Opus",
    compatible: &[*b"Opus", *b"iso2"],
};

/// The duration of a sample of Opus, in samples at 48 kHz: "the total of
/// frame sizes for a single Opus bitstream" (the Opus mapping, clause
/// 4.3.4), all of whose packets in the sample last as long, as the first.
///
/// A packet's frame size is its configuration's, in the top five bits of
/// its TOC byte (RFC 6716, clause 3.1, Table 2), and its number of frames
/// is coded in the two lowest bits: one, two, two, or for code 3, the low
/// six bits of the next byte, which "MUST NOT be zero" (clause 3.2.5). The
/// self-delimiting framing of the first packets of a sample of several
/// streams adds a length after these bytes, which so are read alike
/// (Appendix B).
///
/// # Errors
///
/// [`PacketError::OpusPacket`] if the packet is empty, which RFC 6716
/// forbids ([R1]), or is of code 3 without its frame count, or with a
/// frame count of 0 or more than 120 ms of audio ([R5]). The rest of the
/// packet, which the duration does not need, is not checked against the
/// other requirements of clause 3.4: the writer does not decode it.
pub(crate) fn packet_duration(packet: &[u8]) -> Result<u32, PacketError> {
    let &toc = packet.first().ok_or(PacketError::OpusPacket)?;
    let config = toc >> 3;
    let frame = match config {
        // SILK-only: 10, 20, 40 and 60 ms.
        0..=11 => [480, 960, 1920, 2880][usize::from(config % 4)],
        // Hybrid: 10 and 20 ms.
        12..=15 => [480, 960][usize::from(config % 2)],
        // CELT-only: 2.5, 5, 10 and 20 ms.
        _ => [120, 240, 480, 960][usize::from(config % 4)],
    };
    let frames = match toc & 0b11 {
        0 => 1,
        1 | 2 => 2,
        _ => u32::from(packet.get(1).ok_or(PacketError::OpusPacket)? & 0x3f),
    };
    let duration = frame * frames;
    if !(1..=MAX_PACKET).contains(&duration) {
        return Err(PacketError::OpusPacket);
    }
    Ok(duration)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An identification header of the fields given, without a channel
    /// mapping table.
    fn id_header(version: u8, channels: u8, pre_skip: u16, family: u8) -> Vec<u8> {
        let mut header = b"OpusHead".to_vec();
        header.extend_from_slice(&[version, channels]);
        header.extend_from_slice(&pre_skip.to_le_bytes());
        header.extend_from_slice(&44_100u32.to_le_bytes());
        header.extend_from_slice(&(-256i16).to_le_bytes());
        header.push(family);
        header
    }

    #[test]
    fn the_identification_header_gives_the_dops_fields_big_endian() {
        let config = Config::from_id_header(&id_header(1, 2, 312, 0)).unwrap();
        assert_eq!((config.channels(), config.pre_skip()), (2, 312));
        let entry = config.track().sample_entry;
        assert_eq!(&entry.coding_name, b"Opus");
        assert_eq!((entry.sample_size, entry.sample_rate), (16, 48_000 << 16));
        let expected = [
            &19u32.to_be_bytes()[..],
            b"dOps",
            &[0, 2],                  // Version, OutputChannelCount
            &[0x01, 0x38],            // PreSkip, 312
            &44_100u32.to_be_bytes(), // InputSampleRate
            &[0xff, 0x00],            // OutputGain, -1 dB
            &[0],                     // ChannelMappingFamily
        ]
        .concat();
        assert_eq!(entry.boxes, expected);
        // The header ffmpeg gives libopus's stream, of mono at 48 kHz.
        let header = [
            0x4f, 0x70, 0x75, 0x73, 0x48, 0x65, 0x61, 0x64, 0x01, 0x01, 0x38, 0x01, 0x80, 0xbb,
            0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let config = Config::from_id_header(&header).unwrap();
        assert_eq!((config.channels(), config.pre_skip()), (1, 312));
        assert_eq!(config.input_sample_rate, 48_000);
    }

    #[test]
    fn a_channel_mapping_table_follows_the_family() {
        let mut header = id_header(1, 6, 312, 1);
        header.extend_from_slice(&[4, 2, 0, 4, 1, 2, 3, 5]);
        let config = Config::from_id_header(&header).unwrap();
        let entry = config.track().sample_entry;
        assert_eq!(&entry.boxes[8 + 10..], [1, 4, 2, 0, 4, 1, 2, 3, 5]);
        assert_eq!(config.track().channels, 6);
    }

    #[test]
    fn headers_and_mappings_that_rfc_7845_forbids_are_refused() {
        assert_eq!(
            Config::from_id_header(b"OpusHead"),
            Err(ConfigError::Header)
        );
        assert_eq!(
            Config::from_id_header(&id_header(16, 2, 0, 0)),
            Err(ConfigError::Header)
        );
        assert!(Config::from_id_header(&id_header(15, 2, 0, 0)).is_ok());
        assert_eq!(
            Config::from_id_header(&id_header(1, 3, 0, 0)),
            Err(ConfigError::Channels(3))
        );
        assert_eq!(
            Config::from_id_header(&id_header(1, 0, 0, 0)),
            Err(ConfigError::Channels(0))
        );
        // A family with no table, or a table cut short.
        assert_eq!(
            Config::from_id_header(&id_header(1, 2, 0, 1)),
            Err(ConfigError::Header)
        );
        assert_eq!(
            Config::from_id_header(&[&id_header(1, 2, 0, 1)[..], &[1, 1, 0]].concat()),
            Err(ConfigError::Header)
        );
        let mapping = |family, stream_count, coupled_count, mapping: &[u8]| ChannelMapping {
            family,
            stream_count,
            coupled_count,
            mapping: mapping.to_vec(),
        };
        let refused = [
            (mapping(2, 1, 0, &[0]), ConfigError::Family(2)),
            (mapping(0, 1, 0, &[0]), ConfigError::Mapping),
            (mapping(1, 1, 0, &[0; 9]), ConfigError::Channels(9)),
            (mapping(255, 1, 0, &[]), ConfigError::Channels(0)),
            (mapping(1, 0, 0, &[255]), ConfigError::Mapping),
            (mapping(1, 1, 2, &[0]), ConfigError::Mapping),
            (mapping(1, 2, 1, &[3]), ConfigError::Mapping),
            (mapping(255, 200, 100, &[0]), ConfigError::Mapping),
        ];
        for (mapping, error) in refused {
            assert_eq!(
                Config::with_channel_mapping(0, mapping.clone()),
                Err(error),
                "{mapping:?}"
            );
        }
        assert!(Config::with_channel_mapping(0, mapping(1, 2, 1, &[2, 255, 0])).is_ok());
        assert!(Config::with_channel_mapping(0, mapping(255, 1, 0, &[0; 255])).is_ok());
    }

    #[test]
    fn the_pre_roll_covers_80_ms_of_the_shortest_packets() {
        let config = Config::new(2, 312).unwrap();
        let track = config.track();
        assert_eq!(track.roll_distance, Some(-4));
        assert_eq!((track.delay, track.timescale), (Some(312), 48_000));
        assert_eq!(track.default_duration, 960);
        for (shortest, roll) in [
            (120, -32),
            (240, -16),
            (480, -8),
            (1920, -2),
            (2880, -2),
            (5760, -1),
        ] {
            let config = config.clone().shortest_packet(shortest).unwrap();
            assert_eq!(config.track().roll_distance, Some(roll), "{shortest}");
        }
        for shortest in [119, 5761, 0] {
            assert_eq!(
                Config::new(1, 0).unwrap().shortest_packet(shortest),
                Err(ConfigError::ShortestPacket(shortest))
            );
        }
    }

    #[test]
    fn a_packet_lasts_its_frames() {
        // Each configuration, of one frame.
        let sizes = [
            480, 960, 1920, 2880, 480, 960, 1920, 2880, 480, 960, 1920, 2880, 480, 960, 480, 960,
            120, 240, 480, 960, 120, 240, 480, 960, 120, 240, 480, 960, 120, 240, 480, 960,
        ];
        for (config, size) in sizes.into_iter().enumerate() {
            let toc = (config as u8) << 3;
            assert_eq!(packet_duration(&[toc]), Ok(size), "{config}");
            // Stereo, and two frames of either kind.
            assert_eq!(packet_duration(&[toc | 0b100]), Ok(size), "{config}");
            assert_eq!(packet_duration(&[toc | 1, 0, 0]), Ok(2 * size), "{config}");
            assert_eq!(packet_duration(&[toc | 2, 0]), Ok(2 * size), "{config}");
        }
        // Code 3: the frame count, its v and p bits aside.
        assert_eq!(packet_duration(&[(31 << 3) | 3, 0b1100_0011]), Ok(3 * 960));
        assert_eq!(packet_duration(&[(16 << 3) | 3, 48]), Ok(48 * 120));
        assert_eq!(packet_duration(&[(3 << 3) | 3, 2]), Ok(5760));
    }

    #[test]
    fn packets_that_rfc_6716_forbids_are_refused() {
        let refused: [&[u8]; 5] = [
            &[],
            &[(31 << 3) | 3],
            &[(31 << 3) | 3, 0],
            &[(31 << 3) | 3, 0b1100_0000],
            &[(16 << 3) | 3, 49],
        ];
        for packet in refused {
            assert_eq!(
                packet_duration(packet),
                Err(PacketError::OpusPacket),
                "{packet:?}"
            );
        }
        assert_eq!(
            packet_duration(&[(3 << 3) | 3, 3]),
            Err(PacketError::OpusPacket)
        );
    }

    #[test]
    fn packets_shorter_than_declared_are_refused() {
        let config = Config::new(2, 312).unwrap();
        assert_eq!(config.duration(&[31 << 3]), Ok(960));
        assert_eq!(config.duration(&[(31 << 3) | 1, 0, 0]), Ok(1920));
        assert_eq!(
            config.duration(&[30 << 3]),
            Err(PacketError::OpusShorterThanDeclared {
                duration: 480,
                shortest: 960
            })
        );
    }
}
