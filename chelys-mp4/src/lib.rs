// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! MP4 files (ISO/IEC 14496-14:2020) of one audio track, AAC-LC from
//! `chelys-aac`, implemented from their specifications. The `chelys` crate
//! offers it as `chelys::mp4`.
//!
//! Clause references refer to ISO/IEC 14496-12:2026, the ISO base media
//! file format, or where they name it, to ISO/IEC 14496-14:2020, the MP4
//! file format, ISO/IEC 14496-1:2026, whose descriptors describe the
//! stream, ISO/IEC 14496-34:2025, the syntax description language, or
//! the audio standards.
//!
//! A file is written whole, from its start to its end, so that it can be
//! written where nothing can be sought, as a pipe:
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
//!           sbgp        every frame but the first ...
//!           sgpd        ... is to be decoded after the one before it
//! mdat                  the frames, raw_data_block()s, one after another
//! ```
//!
//! The file has no `ObjectDescriptorBox` (`'iods'`, ISO/IEC 14496-14,
//! clause 6.2), which is optional, nor tags, such as a song's title: a
//! `UserDataBox` for them would come last in the `moov`.
//!
//! ```
//! use chelys_aac::{Config, Rate};
//! use chelys_mp4::aac::Encoder;
//!
//! let config = Config::new(44_100, 2, Rate::Constant { bits_per_second: 128_000 });
//! let mut encoder = Encoder::new(config)?;
//! encoder.encode_interleaved(&[0.0f32; 2 * 44_100]);
//! let mut mp4 = Vec::new();
//! let finished = encoder.finish(&mut mp4)?;
//! assert_eq!(finished.samples, 44_100);
//! assert_eq!(&mp4[4..8], b"ftyp");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod aac;
mod bitrate;
mod boxes;
mod descriptor;
mod movie;
