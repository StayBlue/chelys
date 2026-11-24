// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! ALAC (Apple Lossless) in MP4 files, from "Apple Lossless Format "Magic
//! Cookie" Description" (macosforge/alac, `ALACMagicCookieDescription.txt`
//! at commit 3814718), which this module cites as "the cookie document",
//! and ISO/IEC 14496-12, to which it refers for the sample entry. Each
//! packet, whichever encoder made it, is a sample of an `'alac'` track,
//! which the encoder's magic cookie describes.
//!
//! The cookie document defines the cookie and where an MP4 file holds it,
//! and nothing of the packets themselves: their bitstream is not read, and
//! a packet's duration is the cookie's frame length unless the caller gives
//! another (see [`Writer::push_with_duration`](crate::Writer::push_with_duration)).
//!
//! Spec gap: the cookie document says nothing of a delay before the input,
//! of padding after it, or of whether a packet decodes without those before
//! it. An ALAC stream is taken to have neither delay nor padding, and every
//! packet to be a sync sample: the track has no edit list and no pre-roll,
//! and presents every sample its packets decode to. Settled by decoding
//! with ffmpeg (black box): it decodes the files to the input exactly, from
//! their start and from where it seeks to. The conformance tests are
//! `ffmpeg_decodes_alac_in_mp4_files_to_the_input_exactly` and
//! `ffmpeg_decodes_alac_in_mp4_files_from_where_it_seeks`.

use std::fmt;

use crate::boxes::{FourCc, write_full_box};
use crate::movie::{AudioTrack, Brands, SampleEntry, put_sample_rate};

/// The length of an `ALACSpecificConfig`, the part of the cookie every
/// stream has.
const SPECIFIC_CONFIG_LEN: usize = 24;

/// The length of an `ALACChannelLayoutInfo`, the part of the cookie that
/// may follow the `ALACSpecificConfig`.
const CHANNEL_LAYOUT_LEN: usize = 24;

/// The length of a cookie with an `ALACChannelLayoutInfo`.
const WITH_CHANNEL_LAYOUT_LEN: usize = SPECIFIC_CONFIG_LEN + CHANNEL_LAYOUT_LEN;

/// What an `'alac'` track needs of an ALAC stream: its magic cookie, as
/// its encoder gave it.
///
/// The cookie "is placed without modification into the audio file", and is
/// kept as it is: only the fields the track needs are read of it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    cookie: Vec<u8>,
    frame_length: u32,
    bit_depth: u8,
    channels: u8,
    sample_rate: u32,
}

/// Why a magic cookie was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConfigError {
    /// The cookie is neither 24 bytes long, an `ALACSpecificConfig`, nor
    /// 48, one followed by an `ALACChannelLayoutInfo`.
    Length(usize),
    /// The `compatibleVersion` is not 0, which it "must be".
    CompatibleVersion(u8),
    /// The `frameLength` is 0: the packets would have no samples.
    FrameLength,
    /// The `bitDepth` is 0 or more than 32, its "maximum value".
    BitDepth(u8),
    /// The `numChannels` is 0.
    Channels,
    /// The `sampleRate` is 0.
    SampleRate,
    /// The 24 bytes after the `ALACSpecificConfig` are not an
    /// `ALACChannelLayoutInfo`: its `channelLayoutInfoSize` is not 24, or
    /// its `channelLayoutInfoID` not `'chan'`.
    ChannelLayout,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length(len) => write!(
                f,
                "a magic cookie of {len} bytes, where it has {SPECIFIC_CONFIG_LEN} or \
                 {WITH_CHANNEL_LAYOUT_LEN}"
            ),
            Self::CompatibleVersion(version) => {
                write!(f, "a compatibleVersion of {version}, where it must be 0")
            }
            Self::FrameLength => write!(f, "a frameLength of 0"),
            Self::BitDepth(bits) => write!(f, "a bitDepth of {bits}, where ALAC allows 1 to 32"),
            Self::Channels => write!(f, "a numChannels of 0"),
            Self::SampleRate => write!(f, "a sampleRate of 0"),
            Self::ChannelLayout => write!(
                f,
                "24 bytes after the ALACSpecificConfig that are not an ALACChannelLayoutInfo"
            ),
        }
    }
}

impl std::error::Error for ConfigError {}

impl Config {
    /// The configuration of a stream whose magic cookie is `cookie`: its
    /// `ALACSpecificConfig`, 24 bytes, alone or followed by its
    /// `ALACChannelLayoutInfo`, 24 more, as the encoder gave it, without the
    /// header of the box that holds it in an MP4 file. The cookie of earlier
    /// encoders, which wraps the `ALACSpecificConfig` in a format atom, the
    /// box and a terminator atom, is not taken: the current cookie is
    /// within it, after its first 24 bytes, "as a contiguous set of bytes".
    ///
    /// # Errors
    ///
    /// [`ConfigError`] if the cookie is of another length, or gives a
    /// `compatibleVersion` other than 0, a `frameLength`, `numChannels` or
    /// `sampleRate` of 0, or a `bitDepth` of 0 or more than 32, or if its
    /// last 24 bytes, of 48, are not an `ALACChannelLayoutInfo`.
    pub fn from_magic_cookie(cookie: &[u8]) -> Result<Self, ConfigError> {
        let u32_at = |at: usize| {
            u32::from_be_bytes([cookie[at], cookie[at + 1], cookie[at + 2], cookie[at + 3]])
        };
        match cookie.len() {
            SPECIFIC_CONFIG_LEN => {}
            WITH_CHANNEL_LAYOUT_LEN => {
                // channelLayoutInfoSize, which "should be set to 24", the
                // block's own length, and channelLayoutInfoID, 'chan'. The
                // rest, the channelLayoutTag among them, is kept unread.
                let at = SPECIFIC_CONFIG_LEN;
                if u32_at(at) != CHANNEL_LAYOUT_LEN as u32 || &cookie[at + 4..at + 8] != b"chan" {
                    return Err(ConfigError::ChannelLayout);
                }
            }
            len => return Err(ConfigError::Length(len)),
        }
        // frameLength u32, compatibleVersion u8, bitDepth u8, pb u8, mb u8,
        // kb u8, numChannels u8, maxRun u16, maxFrameBytes u32, avgBitRate
        // u32, sampleRate u32, all big-endian.
        let frame_length = u32_at(0);
        let compatible_version = cookie[4];
        let bit_depth = cookie[5];
        let channels = cookie[9];
        let sample_rate = u32_at(20);
        if compatible_version != 0 {
            return Err(ConfigError::CompatibleVersion(compatible_version));
        }
        if frame_length == 0 {
            return Err(ConfigError::FrameLength);
        }
        if !(1..=32).contains(&bit_depth) {
            return Err(ConfigError::BitDepth(bit_depth));
        }
        if channels == 0 {
            return Err(ConfigError::Channels);
        }
        if sample_rate == 0 {
            return Err(ConfigError::SampleRate);
        }
        Ok(Self {
            cookie: cookie.to_vec(),
            frame_length,
            bit_depth,
            channels,
            sample_rate,
        })
    }

    /// The sample rate in Hz, which is the track's timescale.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// The number of channels.
    pub fn channels(&self) -> u8 {
        self.channels
    }

    /// The bit depth of the source PCM, from 1 to 32.
    pub fn bit_depth(&self) -> u8 {
        self.bit_depth
    }

    /// The samples of each channel in a packet whose header does not give
    /// its own number: the duration a packet is given unless the caller
    /// gives another.
    pub fn frame_length(&self) -> u32 {
        self.frame_length
    }

    /// The track of the stream, its timescale the sample rate, so that
    /// every packet's duration is its samples of each channel. The
    /// `TrackExtendsBox` of a fragmented file gives every packet the frame
    /// length, which each fragment overrides where a packet's differs.
    pub(crate) fn track(&self) -> AudioTrack {
        AudioTrack {
            brands: BRANDS,
            sample_entry: self.sample_entry(),
            timescale: self.sample_rate,
            channels: u16::from(self.channels),
            delay: None,
            default_duration: self.frame_length,
            roll_distance: None,
        }
    }

    /// The `AudioSampleEntry` of `'alac'`, its `channelcount` the stream's
    /// channels and its `samplesize` "source pcm bitdepth", with the box
    /// that holds the cookie appended: "ALAC Specific Info Size", "ALAC
    /// Specific Info ID", `'alac'`, and "Version Flags", 0, an ISO full box
    /// header, then the cookie.
    ///
    /// Spec reading: the cookie document gives the box's size as "= 36
    /// (12 + sizeof(ALACSpecificConfig))", but has the box hold the cookie,
    /// "(24 or 48 Bytes)", and gives the sample entry's size as
    /// "sizeof(AudioSampleEntry)(36) + sizeof(full ISO box header)(12) +
    /// sizeof(Apple Lossless Magic Cookie)". As a box's size is "the number
    /// of bytes in this box, including all its fields and contained boxes"
    /// (ISO/IEC 14496-12, clause 4.2.2), the box is its header's 12 bytes
    /// and the cookie's: 60 where the cookie has a channel layout.
    ///
    /// Spec reading: the cookie document gives `samplerate` as "sample
    /// rate as a uint_32 value", and refers to ISO/IEC 14496-12 "for a
    /// full description" of the sample entry, where it is the rate in 16.16
    /// fixed point (clause 12.2.3.3). The field is taken as the standard
    /// has it, as for every other codec: the rate shifted left 16 bits, and
    /// above 65535 Hz, which its integer part cannot hold, an integer
    /// division of it, with a `SamplingRateBox` after the cookie's box (see
    /// [`put_sample_rate`]), as for a codec that gives its own rate, as the
    /// cookie does. Settled by decoding with ffmpeg and GStreamer
    /// (`qtdemux`, `avdec_alac`; black box): both decode the files at 96 and
    /// 192 kHz at their rate. The conformance tests are
    /// `ffmpeg_decodes_alac_in_mp4_files_to_the_input_exactly` and
    /// `gstreamer_decodes_alac_in_mp4_files_to_the_input_exactly`.
    fn sample_entry(&self) -> SampleEntry {
        const ALAC: FourCc = *b"alac";
        let mut boxes = Vec::new();
        write_full_box(&mut boxes, &ALAC, 0, 0, |out| {
            out.extend_from_slice(&self.cookie);
        });
        let sample_rate = put_sample_rate(&mut boxes, self.sample_rate);
        SampleEntry {
            coding_name: ALAC,
            sample_size: u16::from(self.bit_depth),
            sample_rate,
            boxes,
        }
    }
}

/// The brands of the `FileTypeBox`. Spec gap: the cookie document names
/// none, only "an MP4/M4A file"; they are those of AAC's and FLAC's files,
/// `'mp42'` as major brand and compatible and `'isom'`.
const BRANDS: Brands = Brands {
    major: *b"mp42",
    compatible: &[*b"mp42", *b"isom"],
};

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::boxes::read::{FourCcs, boxes, u32_at};

    /// An `ALACSpecificConfig` of the fields given, the others as the
    /// cookie document recommends, and its maxFrameBytes and avgBitRate 0,
    /// "unknown".
    pub fn cookie(frame_length: u32, bit_depth: u8, channels: u8, sample_rate: u32) -> Vec<u8> {
        [
            &frame_length.to_be_bytes()[..],
            &[0, bit_depth, 40, 10, 14, channels],
            &255u16.to_be_bytes(),
            &[0; 8],
            &sample_rate.to_be_bytes(),
        ]
        .concat()
    }

    /// An `ALACChannelLayoutInfo` of the layout tag given.
    fn channel_layout(tag: u32) -> Vec<u8> {
        [
            &24u32.to_be_bytes()[..],
            b"chan",
            &[0; 4],
            &tag.to_be_bytes(),
            &[0; 8],
        ]
        .concat()
    }

    #[test]
    fn the_cookie_gives_the_stream_s_properties() {
        let config = Config::from_magic_cookie(&cookie(4096, 24, 6, 96_000)).unwrap();
        assert_eq!(
            (
                config.frame_length(),
                config.bit_depth(),
                config.channels(),
                config.sample_rate()
            ),
            (4096, 24, 6, 96_000)
        );
        // ffmpeg's ALAC encoder's cookie of 6 channels of 24 bits at 96 kHz.
        let ffmpeg = [
            0x00, 0x00, 0x10, 0x00, 0x00, 0x18, 0x28, 0x0a, 0x0e, 0x06, 0x00, 0x00, 0x00, 0x01,
            0x20, 0x04, 0x00, 0xd2, 0xf0, 0x00, 0x00, 0x01, 0x77, 0x00,
        ];
        let config = Config::from_magic_cookie(&ffmpeg).unwrap();
        assert_eq!(
            (
                config.frame_length(),
                config.bit_depth(),
                config.channels(),
                config.sample_rate()
            ),
            (4096, 24, 6, 96_000)
        );
        let with_layout = [cookie(4096, 16, 6, 48_000), channel_layout((124 << 16) | 6)].concat();
        assert_eq!(
            Config::from_magic_cookie(&with_layout).unwrap().channels(),
            6
        );
    }

    #[test]
    fn a_cookie_the_document_forbids_is_refused() {
        let refused = |cookie: &[u8]| Config::from_magic_cookie(cookie).unwrap_err();
        assert_eq!(refused(&[0; 23]), ConfigError::Length(23));
        assert_eq!(refused(&[0; 36]), ConfigError::Length(36));
        let mut versioned = cookie(4096, 16, 2, 44_100);
        versioned[4] = 1;
        assert_eq!(refused(&versioned), ConfigError::CompatibleVersion(1));
        assert_eq!(refused(&cookie(0, 16, 2, 44_100)), ConfigError::FrameLength);
        assert_eq!(
            refused(&cookie(4096, 0, 2, 44_100)),
            ConfigError::BitDepth(0)
        );
        assert_eq!(
            refused(&cookie(4096, 33, 2, 44_100)),
            ConfigError::BitDepth(33)
        );
        assert_eq!(refused(&cookie(4096, 16, 0, 44_100)), ConfigError::Channels);
        assert_eq!(refused(&cookie(4096, 16, 2, 0)), ConfigError::SampleRate);
        let mut layout = channel_layout((124 << 16) | 6);
        layout[4..8].copy_from_slice(b"chnl");
        let not_a_layout = [cookie(4096, 16, 6, 44_100), layout].concat();
        assert_eq!(refused(&not_a_layout), ConfigError::ChannelLayout);
    }

    #[test]
    fn the_sample_entry_holds_the_cookie_unchanged() {
        let config = Config::from_magic_cookie(&cookie(4096, 16, 2, 44_100)).unwrap();
        let track = config.track();
        let entry = &track.sample_entry;
        assert_eq!(&entry.coding_name, b"alac");
        assert_eq!(entry.sample_size, 16);
        assert_eq!(entry.sample_rate, 44_100 << 16);
        let expected = [
            &36u32.to_be_bytes()[..],
            b"alac",
            &[0, 0, 0, 0], // version, flags
            &cookie(4096, 16, 2, 44_100),
        ]
        .concat();
        assert_eq!(entry.boxes, expected);
        assert_eq!((track.timescale, track.channels), (44_100, 2));
        assert_eq!(track.default_duration, 4096);
        assert_eq!((track.delay, track.roll_distance), (None, None));

        // With a channel layout, the box is 60 bytes.
        let cookie = [cookie(4096, 24, 6, 48_000), channel_layout((124 << 16) | 6)].concat();
        let entry = Config::from_magic_cookie(&cookie)
            .unwrap()
            .track()
            .sample_entry;
        let found = boxes(&entry.boxes, 0);
        assert_eq!(found.kinds(), ["alac"]);
        assert_eq!(u32_at(&entry.boxes, 0), 60);
        assert_eq!(&found[0].payload[4..], &cookie[..]);
    }

    #[test]
    fn rates_past_16_bits_are_divided_and_given_in_full_in_srat() {
        let config = Config::from_magic_cookie(&cookie(4096, 24, 2, 192_000)).unwrap();
        let entry = config.track().sample_entry;
        assert_eq!(entry.sample_rate, 64_000 << 16);
        let found = boxes(&entry.boxes, 0);
        assert_eq!(found.kinds(), ["alac", "srat"]);
        assert_eq!(
            found[1].payload,
            [&[0; 4][..], &192_000u32.to_be_bytes()].concat()
        );
    }
}
