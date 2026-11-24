// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! AAC-LC in MP4 files (ISO/IEC 14496-14): the raw frames of a stream as
//! the samples of an `'mp4a'` track, which its `AudioSpecificConfig()`
//! describes. The frames are either made by any encoder and given as they
//! are, with their configuration ([`Config`]), to a
//! [`Writer`](crate::Writer) or a [`FragmentedWriter`](crate::FragmentedWriter),
//! or coded from PCM by chelys's own encoder, with the feature `aac`
//! (`encode`).

mod config;
#[cfg(feature = "aac")]
pub mod encode;

pub use config::{Config, ConfigError};

use super::boxes::{FourCc, write_full_box};
use super::descriptor::{AUDIO_ISO_IEC_14496_3, AUDIO_STREAM, DecoderConfig, es_descriptor};
use super::movie::{AudioTrack, Brands, SampleEntry, put_sample_rate};

/// What the track's boxes give of a stream, all known before its first
/// frame but the bitrates.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Stream {
    sample_rate: u32,
    channels: u16,
    frame_samples: u32,
    buffer_size: u32,
    audio_specific_config: Vec<u8>,
}

impl Stream {
    /// The track of the stream, its `ES_Descriptor` of `max_bitrate` and
    /// `avg_bitrate`, whose decoder's output starts with `delay` samples
    /// that its edit leaves out, if given.
    ///
    /// Spec reading: the objectTypeIndication is "Audio ISO/IEC 14496-3",
    /// 0x40, whose DecoderSpecificInfo is the AudioSpecificConfig() (ISO/IEC
    /// 14496-3, clause 1.6.2.1), which names the stream AAC LC: with
    /// perceptual noise substitution it is MPEG-4 AAC, and without it AAC LC
    /// of either standard. That of ISO/IEC 13818-7's Low Complexity profile,
    /// 0x67, would take an adif_header() as its DecoderSpecificInfo instead
    /// (ISO/IEC 14496-1, clause 6.2.6.7.2).
    ///
    /// Spec reading: bufferSizeDB, "the size of the decoding buffer", the
    /// buffer at the decoder's input (ISO/IEC 14496-1, clause 6.1.2.4), is
    /// that of the decoder's input buffer of ISO/IEC 13818-7 (clause
    /// 8.2.3.1), within which the bit reservoir keeps the stream.
    fn track(&self, max_bitrate: u32, avg_bitrate: u32, delay: Option<u32>) -> AudioTrack {
        let es_descriptor = es_descriptor(&DecoderConfig {
            object_type_indication: AUDIO_ISO_IEC_14496_3,
            stream_type: AUDIO_STREAM,
            buffer_size: self.buffer_size,
            max_bitrate,
            avg_bitrate,
            specific_info: &self.audio_specific_config,
        });
        AudioTrack {
            brands: BRANDS,
            sample_entry: sample_entry(self.sample_rate, &es_descriptor),
            timescale: self.sample_rate,
            channels: self.channels,
            delay,
            default_duration: self.frame_samples,
            roll_distance: Some(-1),
        }
    }
}

/// The brands of the `FileTypeBox` (clause 5.2): `'mp42'`, which
/// "identifies this version" of ISO/IEC 14496-14, of which `'mp41'` or
/// `'mp42'` "shall appear in the compatible-brands list" (clause 5), as
/// major brand and compatible, and `'isom'`, the first version of ISO/IEC
/// 14496-12 (clause E.2), which the file keeps to.
///
/// Spec reading: the `SampleToGroupBox` and `SampleGroupDescriptionBox`
/// of the pre-roll (see `write_roll_mapping` in `movie`) are among the
/// boxes that `'avc1'`, and so `'iso2'`, require readers to support
/// (clauses E.3 and E.4), and not among those of `'isom'` (Table E.1); but a
/// reader skips a box it does not recognize (clause 4.2.2), and the file
/// plays without them. So `'isom'` is claimed, which more readers support,
/// and `'iso2'` is not needed. Annex E's brands "should not be used as the
/// major brand" (clause E.1), so `'mp42'` is.
const BRANDS: Brands = Brands {
    major: *b"mp42",
    compatible: &[*b"mp42", *b"isom"],
};

/// The sample entry of a stream at `sample_rate`, whose `ESDBox` holds
/// `es_descriptor`: an `MP4AudioSampleEntry` (ISO/IEC 14496-14, clause
/// 6.7.2), an `AudioSampleEntry` (clause 12.2.3) with an `ESDBox`.
///
/// `samplesize` and `samplerate` are template fields, which "When a file is
/// created as a pure MPEG-4 file ... shall be set to their default values"
/// (ISO/IEC 14496-14, clause 7): `samplesize` 16, and `samplerate` "{
/// default samplerate of media}<<16", the sample rate in 16.16 fixed point
/// (clause 12.2.3.2).
///
/// Spec reading: the two standards conflict above 65535 Hz, as at 88.2 and
/// 96 kHz, where the default `samplerate` cannot be written: its 16-bit
/// integer part cannot hold the rate. ISO/IEC 14496-14 gives no way out;
/// clause 12.2.3.1 does, for a codec that gives its own rate, as the
/// `AudioSpecificConfig()` does, which [`put_sample_rate`] takes: where the
/// rate holds 16 bits, `samplerate` is the default, as clause 7 requires,
/// and above, where no value meets it, an integer division of the rate,
/// with a `SamplingRateBox` after the `ESDBox`.
fn sample_entry(sample_rate: u32, es_descriptor: &[u8]) -> SampleEntry {
    const MP4A: FourCc = *b"mp4a";
    let mut boxes = Vec::new();
    write_full_box(&mut boxes, b"esds", 0, 0, |out| {
        out.extend_from_slice(es_descriptor);
    });
    let sample_rate = put_sample_rate(&mut boxes, sample_rate);
    SampleEntry {
        coding_name: MP4A,
        sample_size: 16,
        sample_rate,
        boxes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boxes::read::{FourCcs, boxes, u32_at};

    #[test]
    fn rates_past_16_bits_are_halved_and_given_in_full_in_srat() {
        for (rate, half) in [(96_000, 48_000), (88_200, 44_100)] {
            let entry = sample_entry(rate, &[0x03, 0x00]);
            assert_eq!(entry.sample_rate, half << 16);
            let found = boxes(&entry.boxes, 0);
            assert_eq!(found.kinds(), ["esds", "srat"]);
            assert_eq!(
                found[1].payload,
                [&[0; 4][..], &rate.to_be_bytes()].concat()
            );
        }
        let entry = sample_entry(64_000, &[0x03, 0x00]);
        assert_eq!(entry.sample_rate, 64_000 << 16);
        let found = boxes(&entry.boxes, 0);
        assert_eq!(found.kinds(), ["esds"], "no srat");
        assert_eq!(u32_at(found[0].payload, 0), 0); // version, flags
        assert_eq!(&found[0].payload[4..], [0x03, 0x00]);
    }
}
