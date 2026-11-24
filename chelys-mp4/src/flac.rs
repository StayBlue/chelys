// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! FLAC in MP4 files, from "Encapsulation of FLAC in ISO Base Media File
//! Format" (xiph/flac, `doc/isoflac.txt` at commit 78d85dd), which this
//! module cites as "isoflac", and RFC 9639, the FLAC format, for the
//! stream's STREAMINFO and its frame headers. Each frame, whichever
//! encoder made it, is a sample of an `'fLaC'` track, which its
//! STREAMINFO describes.
//!
//! Every FLAC frame decodes by itself, from its first sample, and the
//! stream neither starts with a delay nor ends with padding: the track has
//! no edit list and no pre-roll, and presents every sample its frames
//! decode to.

use std::fmt;

use crate::boxes::{FourCc, write_full_box};
use crate::movie::{AudioTrack, Brands, SampleEntry};
use crate::packets::PacketError;

/// The length of a STREAMINFO metadata block's data (RFC 9639, Table 3).
pub const STREAMINFO_LEN: usize = 34;

/// What an `'fLaC'` track needs of a FLAC stream: its STREAMINFO.
///
/// Its `FLACSpecificBox` holds the STREAMINFO block alone, the one block
/// isoflac requires (clause 3.3.2). Other metadata blocks, such as a seek
/// table or Vorbis comments, describe a FLAC file, which chelys does not
/// write, or tag the track, which other tools do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    streaminfo: [u8; STREAMINFO_LEN],
    min_block_size: u16,
    max_block_size: u16,
    sample_rate: u32,
    channels: u8,
    bits_per_sample: u8,
}

/// Why a STREAMINFO block was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConfigError {
    /// The block is not 34 bytes long.
    Length(usize),
    /// The block sizes are not within 16 to 65535, the minimum at most the
    /// maximum (RFC 9639, clause 8.2).
    BlockSize { min: u16, max: u16 },
    /// The sample rate is 0, which is "used when non-audio is represented"
    /// (RFC 9639, clause 8.2), and which an audio track cannot have.
    SampleRate,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length(len) => write!(
                f,
                "a STREAMINFO block of {len} bytes, where it has {STREAMINFO_LEN}"
            ),
            Self::BlockSize { min, max } => write!(
                f,
                "block sizes from {min} to {max}, where FLAC allows 16 to 65535"
            ),
            Self::SampleRate => write!(f, "a sample rate of 0, which is not audio"),
        }
    }
}

impl std::error::Error for ConfigError {}

impl Config {
    /// The configuration of a stream whose STREAMINFO metadata block is
    /// `streaminfo`: the block's data, its 34 bytes (RFC 9639, clause 8.2),
    /// without the metadata block header that precedes it in a FLAC file.
    ///
    /// Its fields are kept as the encoder wrote them, the total number of
    /// samples and the MD5 checksum among them, which an encoder that
    /// writes its STREAMINFO before the stream's end, as for a fragmented
    /// file, gives as 0, "not known".
    ///
    /// # Errors
    ///
    /// [`ConfigError`] if the block is not 34 bytes long, or gives block
    /// sizes that RFC 9639 forbids, or a sample rate of 0.
    pub fn from_streaminfo(streaminfo: &[u8]) -> Result<Self, ConfigError> {
        let streaminfo: [u8; STREAMINFO_LEN] = streaminfo
            .try_into()
            .map_err(|_| ConfigError::Length(streaminfo.len()))?;
        let min_block_size = u16::from_be_bytes([streaminfo[0], streaminfo[1]]);
        let max_block_size = u16::from_be_bytes([streaminfo[2], streaminfo[3]]);
        if min_block_size < 16 || max_block_size < min_block_size {
            return Err(ConfigError::BlockSize {
                min: min_block_size,
                max: max_block_size,
            });
        }
        // After the block sizes, u(16) each, and the frame sizes, u(24)
        // each: the sample rate, u(20), the channels less 1, u(3), and the
        // bits per sample less 1, u(5), from byte 10; the word's low 4 bits
        // start the total number of samples, u(36).
        let packed = u32::from_be_bytes([
            streaminfo[10],
            streaminfo[11],
            streaminfo[12],
            streaminfo[13],
        ]);
        let sample_rate = packed >> 12;
        if sample_rate == 0 {
            return Err(ConfigError::SampleRate);
        }
        let channels = ((packed >> 9) & 0x7) as u8 + 1;
        let bits_per_sample = ((packed >> 4) & 0x1f) as u8 + 1;
        Ok(Self {
            streaminfo,
            min_block_size,
            max_block_size,
            sample_rate,
            channels,
            bits_per_sample,
        })
    }

    /// The sample rate in Hz, which is the track's timescale.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// The number of channels, from 1 to 8.
    pub fn channels(&self) -> u8 {
        self.channels
    }

    /// The bits per sample, from 1 to 32, as STREAMINFO's 5 bits code them,
    /// of which "FLAC supports from 4 to 32" (RFC 9639, Table 3).
    pub fn bits_per_sample(&self) -> u8 {
        self.bits_per_sample
    }

    /// The track of the stream.
    ///
    /// Its timescale is the sample rate, so that every frame's duration is
    /// its block size (isoflac, clause 3.3.4). Spec reading: that clause has
    /// the duration "determined by dividing the decoded block size ... by
    /// the value of the timescale field", which would give it in seconds;
    /// a duration is in the timescale's units (ISO/IEC 14496-12, clause
    /// 8.6.1.2), so that the block size is taken as it is, a timescale of
    /// the sample rate making it exact.
    ///
    /// The `TrackExtendsBox` of a fragmented file gives every frame the
    /// block size where the stream's block size is fixed, its minimum and
    /// maximum the same, as "the minimum block size excludes the last
    /// block" (RFC 9639, clause 8.2); and otherwise none, so that each
    /// fragment gives each frame's.
    pub(crate) fn track(&self) -> AudioTrack {
        let fixed = self.min_block_size == self.max_block_size;
        AudioTrack {
            brands: BRANDS,
            sample_entry: self.sample_entry(),
            timescale: self.sample_rate,
            channels: u16::from(self.channels),
            delay: None,
            default_duration: if fixed {
                u32::from(self.max_block_size)
            } else {
                0
            },
            roll_distance: None,
        }
    }

    /// The `FLACSampleEntry` (isoflac, clause 3.3.1): an `AudioSampleEntry`
    /// of `'fLaC'`, its `channelcount` the stream's channels, its
    /// `samplesize` the bits per sample, and its `samplerate` the sample
    /// rate in 16.16 fixed point "When possible" (see [`samplerate`]), with
    /// a `FLACSpecificBox`.
    fn sample_entry(&self) -> SampleEntry {
        const FLAC: FourCc = *b"fLaC";
        let mut boxes = Vec::new();
        // The FLACSpecificBox (isoflac, clause 3.3.2), of version 0 and flags
        // 0, of one FLACMetadataBlock, a native metadata block: its header,
        // the last-metadata-block flag 1, as no block follows, the block
        // type 0, STREAMINFO, which "must occur first", and its length in 24
        // bits (RFC 9639, clause 8.1); then its data.
        write_full_box(&mut boxes, b"dfLa", 0, 0, |out| {
            out.push(0x80);
            out.extend_from_slice(&(STREAMINFO_LEN as u32).to_be_bytes()[1..]);
            out.extend_from_slice(&self.streaminfo);
        });
        SampleEntry {
            coding_name: FLAC,
            sample_size: u16::from(self.bits_per_sample),
            sample_rate: samplerate(self.sample_rate),
            boxes,
        }
    }
}

/// The brands of the `FileTypeBox`: isoflac defines none, and requires
/// "at least one brand which supports the requirements and the
/// requirements described in this clause", of which "The minimal support
/// ... requires the 'isom' brand" (clause 3.1). They are those of AAC's
/// files, `'mp42'` as major brand and compatible and `'isom'`, which the
/// example of clause 3.5 has too.
const BRANDS: Brands = Brands {
    major: *b"mp42",
    compatible: &[*b"mp42", *b"isom"],
};

/// The `samplerate` field of a stream at `sample_rate` Hz, in 16.16 fixed
/// point: the rate itself, where its 16-bit integer part holds it, and
/// otherwise "the greatest expressible regular division of that rate", or
/// "65535.0 Hz" where there is none (isoflac, clause 3.3.1).
///
/// Spec reading: isoflac follows "the greatest expressible regular
/// division" with "I.e. the samplerate field shall hold 48000.0 for native
/// sample rates of 96 and 192 kHz", which the greatest integer division of
/// 192 kHz within 16 bits, by 3, 64000, would contradict. A regular
/// division is taken as a division by a power of two, which gives the
/// example's 48000 for both: the rate halved until it holds 16 bits, if
/// each halving is exact, as at 88.2, 96, 176.4 and 192 kHz.
fn samplerate(sample_rate: u32) -> u32 {
    let regular = (0..u32::BITS)
        .map(|n| sample_rate >> n)
        .enumerate()
        .find(|&(_, rate)| rate <= 0xffff)
        .filter(|&(n, rate)| rate << n == sample_rate)
        .map(|(_, rate)| rate);
    regular.unwrap_or(0xffff) << 16
}

/// The duration of a FLAC frame: its block size, which its frame header
/// gives (isoflac, clause 3.3.4; RFC 9639, clause 9.1).
///
/// The block size is coded in the 4 bits that follow the 15-bit sync code
/// and the blocking strategy bit (RFC 9639, Table 14); for an uncommon
/// block size, it follows the coded number, a frame or sample number that
/// starts in the fifth byte and is coded in one to seven bytes as UTF-8
/// codes characters (clause 9.1.5, Table 18). Nothing else of the frame is
/// read: that its channels, sample rate and bits per sample "MUST match"
/// the STREAMINFO's (isoflac, clauses 3.3.1 and 3.3.3) is the encoder's to
/// keep, and is not checked.
pub(crate) fn frame_duration(frame: &[u8]) -> Result<u32, PacketError> {
    let header = |at: usize| frame.get(at).copied().ok_or(PacketError::FlacFrame);
    // The sync code, 0b111111111111100, and the blocking strategy bit.
    if header(0)? != 0xff || header(1)? & 0xfe != 0xf8 {
        return Err(PacketError::FlacFrame);
    }
    let bits = header(2)? >> 4;
    match bits {
        0b0001 => Ok(192),
        0b0010..=0b0101 => Ok(144 << bits),
        0b0110 | 0b0111 => {
            // The coded number's length, from the leading 1 bits of its
            // first byte: none for one byte, two to seven for as many.
            let len = match header(4)?.leading_ones() {
                0 => 1,
                n @ 2..=7 => n as usize,
                _ => return Err(PacketError::FlacFrame),
            };
            let at = 4 + len;
            let minus_one = if bits == 0b0110 {
                u32::from(header(at)?)
            } else {
                u32::from(u16::from_be_bytes([header(at)?, header(at + 1)?]))
            };
            // A block size of 65536, coded as 65535, "is forbidden and
            // MUST NOT be used" (clause 9.1.6).
            if minus_one == 0xffff {
                return Err(PacketError::FlacFrame);
            }
            Ok(minus_one + 1)
        }
        0b1000..=0b1111 => Ok(1 << bits),
        // 0b0000 is reserved.
        _ => Err(PacketError::FlacFrame),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A STREAMINFO block of the fields given, its frame sizes, length and
    /// MD5 checksum 0.
    pub fn streaminfo(min: u16, max: u16, sample_rate: u32, channels: u8, bits: u8) -> Vec<u8> {
        let mut block = Vec::new();
        block.extend_from_slice(&min.to_be_bytes());
        block.extend_from_slice(&max.to_be_bytes());
        block.extend_from_slice(&[0; 6]);
        let packed =
            (sample_rate << 12) | (u32::from(channels - 1) << 9) | (u32::from(bits - 1) << 4);
        block.extend_from_slice(&packed.to_be_bytes());
        block.extend_from_slice(&[0; 4 + 16]);
        assert_eq!(block.len(), STREAMINFO_LEN);
        block
    }

    #[test]
    fn the_streaminfo_gives_the_stream_s_properties() {
        let config = Config::from_streaminfo(&streaminfo(4096, 4096, 192_000, 6, 24)).unwrap();
        assert_eq!(config.sample_rate(), 192_000);
        assert_eq!(config.channels(), 6);
        assert_eq!(config.bits_per_sample(), 24);
        let config = Config::from_streaminfo(&streaminfo(16, 65535, 1_048_575, 8, 32)).unwrap();
        assert_eq!(
            (
                config.sample_rate(),
                config.channels(),
                config.bits_per_sample()
            ),
            (1_048_575, 8, 32)
        );
        // The `flac` program's STREAMINFO of 2646 samples of 44.1 kHz
        // stereo of 16 bits, in blocks of 1152.
        let block = [
            0x04, 0x80, 0x04, 0x80, 0x00, 0x00, 0x95, 0x00, 0x01, 0x7b, 0x0a, 0xc4, 0x42, 0xf0,
            0x00, 0x00, 0x0a, 0x56, 0x0a, 0xe0, 0x98, 0x18, 0x2a, 0x22, 0xd3, 0x1f, 0xd8, 0x7d,
            0x8c, 0x22, 0xd5, 0xd3, 0x5e, 0x81,
        ];
        let config = Config::from_streaminfo(&block).unwrap();
        assert_eq!(
            (
                config.sample_rate(),
                config.channels(),
                config.bits_per_sample()
            ),
            (44_100, 2, 16)
        );
    }

    #[test]
    fn a_streaminfo_that_flac_forbids_is_refused() {
        assert_eq!(
            Config::from_streaminfo(&[0; 33]),
            Err(ConfigError::Length(33))
        );
        assert_eq!(
            Config::from_streaminfo(&streaminfo(15, 4096, 44_100, 2, 16)),
            Err(ConfigError::BlockSize { min: 15, max: 4096 })
        );
        assert_eq!(
            Config::from_streaminfo(&streaminfo(4096, 1024, 44_100, 2, 16)),
            Err(ConfigError::BlockSize {
                min: 4096,
                max: 1024
            })
        );
        assert_eq!(
            Config::from_streaminfo(&streaminfo(4096, 4096, 0, 2, 16)),
            Err(ConfigError::SampleRate)
        );
    }

    #[test]
    fn the_sample_rate_is_halved_to_fit_16_bits() {
        assert_eq!(samplerate(44_100), 44_100 << 16);
        assert_eq!(samplerate(65_535), 65_535 << 16);
        for (rate, field) in [
            (96_000, 48_000),
            (192_000, 48_000),
            (88_200, 44_100),
            (176_400, 44_100),
            (384_000, 48_000),
            (65_536, 32_768),
        ] {
            assert_eq!(samplerate(rate), field << 16, "{rate}");
        }
        // No halving holds 16 bits exactly.
        for rate in [65_537, 100_001, 196_607] {
            assert_eq!(samplerate(rate), 0xffff << 16, "{rate}");
        }
    }

    #[test]
    fn the_sample_entry_holds_the_streaminfo() {
        let block = streaminfo(4096, 4096, 96_000, 2, 24);
        let track = Config::from_streaminfo(&block).unwrap().track();
        let entry = &track.sample_entry;
        assert_eq!(&entry.coding_name, b"fLaC");
        assert_eq!(entry.sample_size, 24);
        assert_eq!(entry.sample_rate, 48_000 << 16);
        let expected = [
            &(12 + 4 + 34u32).to_be_bytes()[..],
            b"dfLa",
            &[0, 0, 0, 0],     // version, flags
            &[0x80, 0, 0, 34], // last, STREAMINFO, its length
            &block,
        ]
        .concat();
        assert_eq!(entry.boxes, expected);
        assert_eq!((track.timescale, track.channels), (96_000, 2));
        assert_eq!((track.delay, track.roll_distance), (None, None));
        assert_eq!(track.default_duration, 4096);
        let variable = Config::from_streaminfo(&streaminfo(1024, 4096, 96_000, 2, 24)).unwrap();
        assert_eq!(variable.track().default_duration, 0);
    }

    /// A frame header up to its block size, of the block size bits
    /// `bits`, with `coded` as its coded number and `uncommon` after it.
    fn header(bits: u8, coded: &[u8], uncommon: &[u8]) -> Vec<u8> {
        [&[0xff, 0xf8, (bits << 4) | 0x9, 0x18][..], coded, uncommon].concat()
    }

    #[test]
    fn a_frame_lasts_its_block_size() {
        let common = [
            (0b0001, 192),
            (0b0010, 576),
            (0b0011, 1152),
            (0b0100, 2304),
            (0b0101, 4608),
            (0b1000, 256),
            (0b1001, 512),
            (0b1010, 1024),
            (0b1011, 2048),
            (0b1100, 4096),
            (0b1101, 8192),
            (0b1110, 16384),
            (0b1111, 32768),
        ];
        for (bits, size) in common {
            assert_eq!(
                frame_duration(&header(bits, &[0], &[])),
                Ok(size),
                "{bits:04b}"
            );
        }
        // The blocking strategy bit does not matter.
        let mut variable = header(0b1100, &[0], &[]);
        variable[1] = 0xf9;
        assert_eq!(frame_duration(&variable), Ok(4096));
    }

    #[test]
    fn an_uncommon_block_size_follows_the_coded_number() {
        // Coded numbers of 1, 2, 3 and 7 bytes, as RFC 9639's example of
        // 51 billion is.
        let coded: [&[u8]; 4] = [
            &[0x05],
            &[0xc2, 0x80],
            &[0xe0, 0xa0, 0x80],
            &[0xfe, 0xaf, 0x9f, 0xb5, 0xa3, 0xb8, 0x80],
        ];
        for number in coded {
            assert_eq!(frame_duration(&header(0b0110, number, &[99])), Ok(100));
            assert_eq!(
                frame_duration(&header(0b0111, number, &[0x03, 0xe7])),
                Ok(1000)
            );
            assert_eq!(frame_duration(&header(0b0110, number, &[0])), Ok(1));
            assert_eq!(
                frame_duration(&header(0b0111, number, &[0xff, 0xfe])),
                Ok(65535)
            );
        }
    }

    #[test]
    fn a_frame_that_is_not_flac_is_refused() {
        let refused = [
            vec![],
            vec![0xff, 0xf8],
            vec![0xff, 0xfa, 0xc9, 0x18, 0x00],
            vec![0xfe, 0xf8, 0xc9, 0x18, 0x00],
            header(0b0000, &[0], &[]),
            // The uncommon size cut short, or forbidden.
            header(0b0110, &[0], &[]),
            header(0b0111, &[0], &[0x03]),
            header(0b0111, &[0], &[0xff, 0xff]),
            // A coded number that starts with a continuation byte, or
            // with eight 1 bits.
            header(0b0110, &[0x80], &[1]),
            header(0b0110, &[0xff], &[1]),
        ];
        for frame in refused {
            assert_eq!(
                frame_duration(&frame),
                Err(PacketError::FlacFrame),
                "{frame:02x?}"
            );
        }
    }
}
