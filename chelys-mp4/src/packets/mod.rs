// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! MP4 files of packets that any encoder made, of a codec whose mapping to
//! the file format chelys implements: FLAC ([`flac`](crate::flac)), Opus
//! ([`opus`](crate::opus)), AAC-LC ([`aac`](crate::aac)) and ALAC
//! ([`alac`](crate::alac)). The packets are taken as they are: only what
//! the mapping needs of them is read, each packet's duration, which AAC's
//! and ALAC's configurations give instead, ALAC's unless the caller gives
//! another.

mod fragmented;
mod progressive;

pub use fragmented::{FragmentedWriter, PushError};
pub use progressive::{FinishError, Writer};

use std::fmt;

use crate::fragments::LengthError;
use crate::movie::AudioTrack;
use crate::{aac, alac, flac, opus};

/// The stream that a file of packets holds, as its codec's configuration
/// describes it.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Track {
    Flac(flac::Config),
    Opus(opus::Config),
    Aac(aac::Config),
    Alac(alac::Config),
}

impl From<flac::Config> for Track {
    fn from(config: flac::Config) -> Self {
        Self::Flac(config)
    }
}

impl From<opus::Config> for Track {
    fn from(config: opus::Config) -> Self {
        Self::Opus(config)
    }
}

impl From<aac::Config> for Track {
    fn from(config: aac::Config) -> Self {
        Self::Aac(config)
    }
}

impl From<alac::Config> for Track {
    fn from(config: alac::Config) -> Self {
        Self::Alac(config)
    }
}

/// Why a packet was refused: it is not one of the codec, as far as what
/// the writer reads of it shows, its duration, or for AAC and ALAC, whose
/// duration the configuration gives, that it is not empty; or the duration
/// given with it is not one it can have. The packets are not checked
/// further: an AAC packet that still has its ADTS header is taken, and
/// makes a file that does not decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PacketError {
    /// A FLAC frame that does not start with a frame header's sync code, or
    /// whose header is cut short before its block size, or codes a block
    /// size that RFC 9639 reserves or forbids.
    FlacFrame,
    /// An Opus packet that RFC 6716 forbids: empty, or of code 3 without a
    /// frame count, or with a count of 0 or more than 120 ms of audio.
    OpusPacket,
    /// An Opus packet shorter than the shortest packet declared, which the
    /// pre-roll is reckoned in (see [`opus::Config::shortest_packet`]).
    OpusShorterThanDeclared { duration: u32, shortest: u32 },
    /// An empty AAC packet, which no `raw_data_block()` is: its
    /// `id_syn_ele` alone takes 3 bits.
    AacEmpty,
    /// An empty ALAC packet, which can hold no samples.
    AlacEmpty,
    /// A duration given with a packet other than the one the codec's
    /// mapping reads of it, or its configuration gives it: only an ALAC
    /// packet's can differ.
    Duration { given: u32, read: u32 },
    /// A duration of 0 given with an ALAC packet, which decodes to
    /// samples.
    ZeroDuration,
}

impl fmt::Display for PacketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FlacFrame => write!(f, "not a FLAC frame with a valid block size"),
            Self::OpusPacket => write!(f, "not a valid Opus packet"),
            Self::OpusShorterThanDeclared { duration, shortest } => write!(
                f,
                "an Opus packet of {duration} samples, shorter than the {shortest} declared"
            ),
            Self::AacEmpty => write!(f, "an empty AAC packet"),
            Self::AlacEmpty => write!(f, "an empty ALAC packet"),
            Self::Duration { given, read } => write!(
                f,
                "a packet given a duration of {given}, where it lasts {read}"
            ),
            Self::ZeroDuration => write!(f, "an ALAC packet given a duration of 0"),
        }
    }
}

impl std::error::Error for PacketError {}

/// Why a stream of packets cannot end as its file is to present it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EndError {
    /// The length given or declared is not one the packets present.
    Length(LengthError),
    /// The packets decode to fewer samples than the delay or pre-skip
    /// that the edit list is to leave out: they end before the
    /// presentation would start.
    ShorterThanDelay { samples: u64, delay: u64 },
}

impl fmt::Display for EndError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length(e) => e.fmt(f),
            Self::ShorterThanDelay { samples, delay } => write!(
                f,
                "packets of {samples} samples of each channel, which end before the delay of {delay}"
            ),
        }
    }
}

impl std::error::Error for EndError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Length(e) => Some(e),
            Self::ShorterThanDelay { .. } => None,
        }
    }
}

impl Track {
    /// The track of the stream, of packets of `sizes` where they are known,
    /// and otherwise before any, as in a fragmented file's init segment.
    pub(crate) fn audio_track(&self, sizes: Option<&[u32]>) -> AudioTrack {
        match self {
            Self::Flac(config) => config.track(),
            Self::Opus(config) => config.track(),
            Self::Aac(config) => config.track(sizes),
            Self::Alac(config) => config.track(),
        }
    }

    /// The duration of `packet`, as the codec's mapping has it, or as
    /// `given`, if given. An ALAC packet lasts the cookie's frame length
    /// unless its header gives another number, which only the caller can
    /// give, as the writer does not read its bitstream; a packet of
    /// another codec lasts what its mapping has it last, which `given`, if
    /// given, is to be.
    pub(crate) fn duration(&self, packet: &[u8], given: Option<u32>) -> Result<u32, PacketError> {
        let read = match self {
            Self::Flac(_) => flac::frame_duration(packet)?,
            Self::Opus(config) => config.duration(packet)?,
            Self::Aac(_) if packet.is_empty() => return Err(PacketError::AacEmpty),
            Self::Aac(config) => config.frame_samples(),
            Self::Alac(_) if packet.is_empty() => return Err(PacketError::AlacEmpty),
            // Spec gap: the cookie document gives the frame length as "the
            // frames per packet when no explicit frames per packet setting is
            // present in the packet header", but bounds no explicit setting;
            // any duration is taken but 0, as a packet decodes to samples.
            Self::Alac(_) if given == Some(0) => return Err(PacketError::ZeroDuration),
            Self::Alac(config) => return Ok(given.unwrap_or(config.frame_length())),
        };
        match given {
            Some(given) if given != read => Err(PacketError::Duration { given, read }),
            _ => Ok(read),
        }
    }

    /// The samples that a decoder's output starts with before the input.
    fn delay(&self) -> u64 {
        match self {
            Self::Flac(_) => 0,
            Self::Opus(config) => u64::from(config.pre_skip()),
            Self::Aac(config) => config.delay().map_or(0, u64::from),
            Self::Alac(_) => 0,
        }
    }

    /// Whether the stream ends with padding past the input, which the edit
    /// list of a file written whole leaves out, and the last packet's
    /// duration in a fragmented file: an Opus stream's (the Opus mapping,
    /// clauses 4.3.4 and 4.4), and an AAC stream's whose delay is given, as
    /// chelys's own AAC files have it. A FLAC stream has no padding, and its
    /// last frame lasts its block size, as every frame does; nor has an
    /// ALAC stream, whose last packet lasts what it decodes to, nor an AAC
    /// stream whose delay is not known, whose track has no edit list and
    /// presents every sample.
    fn ends_with_padding(&self) -> bool {
        match self {
            Self::Flac(_) => false,
            Self::Opus(_) => true,
            Self::Aac(config) => config.delay().is_some(),
            Self::Alac(_) => false,
        }
    }

    /// The end of the presentation on the media's timeline, of a stream
    /// whose packets decode to `total` samples, the last `last`, and which
    /// presents `length` samples after its delay, if given, and otherwise
    /// every sample after its delay.
    ///
    /// A stream that ends with padding is to end within its last packet,
    /// which holds the input's last sample, or the delay's: no packet is
    /// past the input's end. A stream without padding is to present every
    /// sample its packets decode to. A stream of no packets presents
    /// nothing.
    ///
    /// # Errors
    ///
    /// [`EndError::ShorterThanDelay`] if there are packets, but they end
    /// before the delay, and [`EndError::Length`] if `length` is not one
    /// the packets can present.
    fn end(&self, length: Option<u64>, total: u64, last: Option<u32>) -> Result<u64, EndError> {
        let delay = self.delay();
        if last.is_some() && total < delay {
            return Err(EndError::ShorterThanDelay {
                samples: total,
                delay,
            });
        }
        let samples = total - delay.min(total);
        let length = length.unwrap_or(samples);
        let error = EndError::Length(LengthError { length, samples });
        let Some(last) = last else {
            return if length == 0 { Ok(0) } else { Err(error) };
        };
        let Some(end) = delay.checked_add(length) else {
            return Err(error);
        };
        let last_start = total - u64::from(last);
        let fits = if self.ends_with_padding() {
            last_start < end && end <= total
        } else {
            end == total
        };
        if fits { Ok(end) } else { Err(error) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stream_with_padding_ends_within_its_last_packet() {
        let opus = Track::from(opus::Config::new(2, 312).unwrap());
        // Three packets of 960 decode to 2880 samples, 2568 after the
        // pre-skip.
        assert_eq!(opus.end(None, 2880, Some(960)), Ok(2880));
        assert_eq!(opus.end(Some(2568), 2880, Some(960)), Ok(2880));
        assert_eq!(opus.end(Some(1609), 2880, Some(960)), Ok(1921));
        let error = |length| {
            EndError::Length(LengthError {
                length,
                samples: 2568,
            })
        };
        assert_eq!(opus.end(Some(1608), 2880, Some(960)), Err(error(1608)));
        assert_eq!(opus.end(Some(2569), 2880, Some(960)), Err(error(2569)));
        // Packets that do not reach past the pre-skip.
        assert_eq!(
            opus.end(None, 240, Some(240)),
            Err(EndError::ShorterThanDelay {
                samples: 240,
                delay: 312
            })
        );
        assert_eq!(opus.end(Some(0), 312, Some(312)), Ok(312));
        // A length past any the stream's samples can count.
        assert_eq!(
            opus.end(Some(u64::MAX), 2880, Some(960)),
            Err(error(u64::MAX))
        );
        assert_eq!(opus.end(Some(0), 960, Some(960)), Ok(312));
        // No packets.
        assert_eq!(opus.end(None, 0, None), Ok(0));
        assert_eq!(
            opus.end(Some(1), 0, None),
            Err(EndError::Length(LengthError {
                length: 1,
                samples: 0
            }))
        );
    }

    #[test]
    fn a_stream_without_padding_presents_every_sample() {
        let block = crate::flac::tests::streaminfo(4096, 4096, 44_100, 2, 16);
        let flac = Track::from(flac::Config::from_streaminfo(&block).unwrap());
        assert_eq!(flac.end(None, 5000, Some(904)), Ok(5000));
        assert_eq!(flac.end(Some(5000), 5000, Some(904)), Ok(5000));
        assert_eq!(
            flac.end(Some(4999), 5000, Some(904)),
            Err(EndError::Length(LengthError {
                length: 4999,
                samples: 5000
            }))
        );
    }
}
