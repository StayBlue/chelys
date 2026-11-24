// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! What every fragmented file is to write beyond its stream, and the error
//! of a stream that does not keep to the length it declared.

use std::fmt;
use std::num::NonZeroU32;

/// What a fragmented encoder or writer is to write, beyond the stream.
/// Durations are in samples of each channel at the rate a decoder outputs:
/// the sample rate, for AAC and FLAC, and 48 kHz for Opus.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct FragmentOptions {
    /// The samples of each channel that the input will have, if known: of
    /// chelys's AAC encoder, as `chelys_aac::Finished::samples` counts
    /// them; of Opus packets, after the pre-skip, and of AAC packets of a
    /// known delay, after it, the samples before the padding; and of FLAC
    /// packets, and AAC packets whose delay is not known, every sample they
    /// decode to. `None` by default.
    pub length: Option<u64>,
    /// The duration that each fragment is to have; by default one second.
    pub fragment_samples: Option<NonZeroU32>,
}

impl FragmentOptions {
    /// Declares the samples of each channel that the input will have, so
    /// that the file presents them exactly, as a file written whole does:
    /// the encoder or writer takes no more, and its `finish` fails with
    /// fewer.
    pub fn length(mut self, samples: u64) -> Self {
        self.length = Some(samples);
        self
    }

    /// Has each fragment last about `samples` samples of each channel: as
    /// many whole frames as end nearest each multiple of `samples`, and a
    /// frame at least. A duration in samples is exact at every sample rate,
    /// as the file's times are in samples. It is at most 2^32 - 1 samples,
    /// 12 hours at 96 kHz, which keeps every fragment's tables within their
    /// fields. A fragment's frames are held in memory until it ends: about
    /// the bitrate times its duration.
    pub fn fragment_samples(mut self, samples: NonZeroU32) -> Self {
        self.fragment_samples = Some(samples);
        self
    }
}

/// An input whose length is not the one declared, which the init segment
/// of a fragmented file gives, or which a file of packets is to present.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct LengthError {
    /// The samples of each channel declared.
    pub length: u64,
    /// The samples of each channel the input would have had: of a stream
    /// of packets, those its packets decode to after any delay, the
    /// padding included.
    pub samples: u64,
}

impl fmt::Display for LengthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "an input of {} samples of each channel, where {} were declared",
            self.samples, self.length
        )
    }
}

impl std::error::Error for LengthError {}
