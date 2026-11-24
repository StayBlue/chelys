// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! MP4 files of packets that any encoder made, of a codec whose mapping to
//! the file format chelys implements: FLAC ([`flac`](crate::flac)) and
//! Opus ([`opus`](crate::opus)). The packets are taken as they are: only
//! what the mapping needs of them is read, each packet's duration.

mod fragmented;
mod progressive;

pub use fragmented::{FragmentedWriter, PushError};
pub use progressive::{FinishError, Writer};

use std::fmt;

use crate::fragments::LengthError;
use crate::movie::AudioTrack;
use crate::{flac, opus};

/// The stream that a file of packets holds, as its codec's configuration
/// describes it.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Track {
    Flac(flac::Config),
    Opus(opus::Config),
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

/// Why a packet was refused: it is not one of the codec, as far as its
/// duration shows.
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
        }
    }
}

impl std::error::Error for PacketError {}

impl Track {
    pub(crate) fn audio_track(&self) -> AudioTrack {
        match self {
            Self::Flac(config) => config.track(),
            Self::Opus(config) => config.track(),
        }
    }

    /// The duration of `packet`, as the codec's mapping has it.
    pub(crate) fn duration(&self, packet: &[u8]) -> Result<u32, PacketError> {
        match self {
            Self::Flac(_) => flac::frame_duration(packet),
            Self::Opus(config) => config.duration(packet),
        }
    }

    /// The samples that a decoder's output starts with before the input.
    fn delay(&self) -> u64 {
        match self {
            Self::Flac(_) => 0,
            Self::Opus(config) => u64::from(config.pre_skip()),
        }
    }

    /// Whether the stream ends with padding past the input, which the edit
    /// list of a file written whole leaves out, and the last packet's
    /// duration in a fragmented file: an Opus stream's (the Opus mapping,
    /// clauses 4.3.4 and 4.4). A FLAC stream has no padding, and its last
    /// frame lasts its block size, as every frame does.
    fn ends_with_padding(&self) -> bool {
        matches!(self, Self::Opus(_))
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
    /// [`LengthError`] if `length` is not one the packets can present.
    fn end(&self, length: Option<u64>, total: u64, last: Option<u32>) -> Result<u64, LengthError> {
        let delay = self.delay();
        let samples = total.saturating_sub(delay);
        let length = length.unwrap_or(samples);
        let error = LengthError { length, samples };
        let Some(last) = last else {
            return if length == 0 { Ok(0) } else { Err(error) };
        };
        let end = delay + length;
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
        let error = |length| LengthError {
            length,
            samples: 2568,
        };
        assert_eq!(opus.end(Some(1608), 2880, Some(960)), Err(error(1608)));
        assert_eq!(opus.end(Some(2569), 2880, Some(960)), Err(error(2569)));
        // Packets that do not reach past the pre-skip.
        assert_eq!(
            opus.end(None, 240, Some(240)),
            Err(LengthError {
                length: 0,
                samples: 0
            })
        );
        assert_eq!(opus.end(Some(0), 960, Some(960)), Ok(312));
        // No packets.
        assert_eq!(opus.end(None, 0, None), Ok(0));
        assert_eq!(
            opus.end(Some(1), 0, None),
            Err(LengthError {
                length: 1,
                samples: 0
            })
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
            Err(LengthError {
                length: 4999,
                samples: 5000
            })
        );
    }
}
