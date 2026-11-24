// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! MP4 files (ISO/IEC 14496-14:2020) of one audio track, implemented from
//! their specifications: packets of FLAC ([`flac`]), Opus ([`opus`]) and
//! AAC-LC ([`aac`]) that any encoder made, which a [`Writer`] or a
//! [`FragmentedWriter`] takes, and AAC-LC coded from PCM by `chelys-aac`
//! (`aac::encode`, with the feature `aac`). The `chelys` crate offers it as
//! `chelys::mp4`.
//!
//! Clause references refer to ISO/IEC 14496-12:2026, the ISO base media
//! file format, or where they name it, to ISO/IEC 14496-14:2020, the MP4
//! file format, ISO/IEC 14496-1:2026, whose descriptors describe the
//! stream, ISO/IEC 14496-34:2025, the syntax description language, or
//! the audio standards. FLAC and Opus follow their mappings to the file
//! format, as each module says.
//!
//! An AAC file is written whole, from its start to its end, so that it can be
//! written where nothing can be sought, as a pipe; one of packets from
//! another encoder has no `edts` where their delay is not given:
//!
//! ```text
//! ftyp                  the brands: mp42, isom
//! moov
//!   mvhd                the movie's timescale, the sample rate
//!   trak
//!     tkhd
//!     edts
//!       elst            one edit: the input's samples, after the delay
//!     mdia
//!       mdhd            the media's timescale, the sample rate
//!       hdlr            soun
//!       minf
//!         smhd
//!         dinf
//!           dref
//!             url       the media data is in this file
//!         stbl
//!           stsd
//!             mp4a      channels and sample rate
//!               esds    the ES_Descriptor: the AudioSpecificConfig()
//!               srat    the sample rate, at 88.2 and 96 kHz only
//!           stts        every frame 1024 samples long
//!           stsc        one chunk of every frame
//!           stsz        each frame's size
//!           stco        where the chunk starts
//!           sbgp        every frame ...
//!           sgpd        ... is to be decoded after the one before it
//! mdat                  the frames, raw_data_block()s, one after another
//! ```
//!
//! A fragmented file (`aac::encode::FragmentedEncoder`) is written as the frames
//! come, as an init segment and then fragments of about a second each:
//!
//! ```text
//! ftyp                  as above
//! moov                  as above, but:
//!   mvhd                its duration 0, that of the moov's samples
//!   trak
//!     tkhd              likewise
//!     edts
//!       elst            the samples after the delay: the input's length,
//!                       if known, or all of them
//!     mdia
//!       mdhd            its duration 0: the moov has no samples
//!       ...
//!         stbl
//!           stsd        as above
//!           stts        ... stco: no entries
//!           sgpd        the pre-roll, which the fragments map frames to
//!   mvex
//!     mehd              the input's length, if known
//!     trex              each frame's duration and flags
//! moof                  then for each fragment:
//!   mfhd                its sequence number, from 1
//!   traf
//!     tfhd
//!     tfdt              the decoding time of its first frame
//!     trun              its frames' sizes, and where they start
//!     sbgp              every frame is to be decoded after the one before
//! mdat                  its frames
//! ```
//!
//! The file has no `ObjectDescriptorBox` (`'iods'`, ISO/IEC 14496-14,
//! clause 6.2), which is optional, nor tags, such as a song's title: a
//! `UserDataBox` for them would come last in the `moov`.
//!
//! ```
//! # #[cfg(feature = "aac")] {
//! use chelys_aac::{Config, Rate};
//! use chelys_mp4::aac::encode::Encoder;
//!
//! let config = Config::new(44_100, 2, Rate::Constant { bits_per_second: 128_000 });
//! let mut encoder = Encoder::new(config)?;
//! encoder.encode_interleaved(&[0.0f32; 2 * 44_100])?;
//! let mut mp4 = Vec::new();
//! let finished = encoder.finish(&mut mp4)?;
//! assert_eq!(finished.samples, 44_100);
//! assert_eq!(&mp4[4..8], b"ftyp");
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod aac;
mod bitrate;
mod boxes;
mod descriptor;
pub mod flac;
mod fragments;
mod movie;
pub mod opus;
mod packets;

pub use fragments::{FragmentOptions, LengthError};
pub use movie::fragmented::{FragmentInfo, FragmentSink};
pub use packets::{EndError, FinishError, FragmentedWriter, PacketError, PushError, Track, Writer};
