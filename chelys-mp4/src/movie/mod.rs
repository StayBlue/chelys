// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! A movie file (ISO/IEC 14496-12, clause 7) of one audio track, in an MP4
//! file (ISO/IEC 14496-14): its `FileTypeBox` and its `MovieBox`, which
//! describes the track, as every layout of the file writes them. Where the
//! track's samples are, each a frame of the stream, is the layout's: one
//! `MediaDataBox` after the `MovieBox`, in [`progressive`], or movie
//! fragments after it, in [`fragmented`].

pub mod fragmented;
pub mod progressive;

use super::boxes::{FourCc, write_box, write_full_box};

/// What the boxes describe of an audio track, all that is known of it
/// before its first sample: each sample a frame or packet of the stream,
/// which decodes to the presentation's samples after the `delay` samples
/// that its edit leaves out, if it has an edit list.
#[derive(Clone, Debug)]
pub struct AudioTrack {
    /// The brands of the `FileTypeBox`, which are the codec's to choose.
    pub brands: Brands,
    pub sample_entry: SampleEntry,
    /// The timescale of the media, and of the movie: the rate of the
    /// samples that a decoder outputs, in which every duration is counted.
    pub timescale: u32,
    pub channels: u16,
    /// The samples that a decoder's output starts with before the
    /// presentation, which the track's one edit leaves out, or `None` where
    /// the track has no edit list and presents every sample it decodes to.
    pub delay: Option<u32>,
    /// The duration that every sample but the last has, where they all
    /// have one, or 0: the `TrackExtendsBox`'s `default_sample_duration`
    /// in a fragmented file.
    pub default_duration: u32,
    /// The number of samples before each sample that are to be decoded
    /// before it for it to decode correctly, as a negative `roll_distance`
    /// (clause 10.1), or `None` where each decodes correctly alone, and the
    /// track has no pre-roll group.
    pub roll_distance: Option<i16>,
}

/// The brands of the `FileTypeBox` (clause 5.2): its major brand, and its
/// compatible brands.
#[derive(Clone, Copy, Debug)]
pub struct Brands {
    pub major: FourCc,
    pub compatible: &'static [FourCc],
}

/// The fields and boxes of the track's one sample entry, an
/// `AudioSampleEntry` (clause 12.2.3), which are the codec's to give but
/// for `channelcount`, the track's channels.
#[derive(Clone, Debug)]
pub struct SampleEntry {
    /// The sample entry's `format`, its box type, as `'mp4a'`.
    pub coding_name: FourCc,
    /// `samplesize`.
    pub sample_size: u16,
    /// `samplerate`, in 16.16 fixed point.
    pub sample_rate: u32,
    /// The boxes that follow the fields, as an `ESDBox`, one after another.
    pub boxes: Vec<u8>,
}

/// The `samplerate` field of a stream at `sample_rate` Hz, whose codec
/// gives its own rate, appending a `SamplingRateBox` to the sample entry's
/// `boxes` where the field cannot hold the rate.
///
/// `samplerate` is "{ default samplerate of media}<<16", the rate in 16.16
/// fixed point (clause 12.2.3.2), where its 16-bit integer part holds the
/// rate. Above 65535 Hz, as at 88.2 and 96 kHz, it cannot; clause 12.2.3.1
/// gives a way out for a codec that gives its own rate: "a SamplingRateBox
/// is present in the AudioSampleEntryV1 or AudioSampleEntry, and it
/// documents the actual sampling rate", and `samplerate` "should contain a
/// value that matches the media timescale left-shifted 16 bits ... or be
/// an integer division or multiple of it". So the field is the rate
/// divided by the least integer that brings it within 16 bits, 2 at 88.2
/// and 96 kHz, and a `SamplingRateBox` follows the codec's boxes, as "the
/// SamplingRateBox ... should follow, not precede, any boxes defined in or
/// required by derived specifications".
pub fn put_sample_rate(boxes: &mut Vec<u8>, sample_rate: u32) -> u32 {
    let divisor = (1..)
        .find(|&d| sample_rate % d == 0 && sample_rate / d <= 0xffff)
        .expect("the rate over itself, 1, holds 16 bits");
    if divisor > 1 {
        // Spec gap: the SamplingRateBox extends FullBox('srat') without a
        // version or flags; they are 0, as for the other boxes that define
        // none.
        write_full_box(boxes, b"srat", 0, 0, |out| {
            out.extend_from_slice(&sample_rate.to_be_bytes()); // sampling_rate
        });
    }
    (sample_rate / divisor) << 16
}

/// The durations of the `MovieBox`, each in samples, as both timescales
/// are the track's.
#[derive(Clone, Copy, Debug)]
struct Durations {
    /// The `edit_duration` of the one edit, where the track has an edit
    /// list: the samples it presents.
    edit: Option<u64>,
    /// The track's `duration` (clause 8.3.2.3), and the movie's, that of
    /// "the longest track" (clause 8.2.2.3), the one track.
    track: u64,
    /// The media's `duration` (clause 8.4.2.3): that of the samples the
    /// `MovieBox` describes.
    media: u64,
}

/// The `track_ID` of the track (clause 8.3.2.3), from which an
/// `ES_Descriptor` built into a stream takes its `ES_ID` (ISO/IEC 14496-14,
/// clause 4.1.2): the first, as "Zero is not a valid track_ID value".
const TRACK_ID: u32 = 1;

/// The `FileTypeBox` (clause 5.2), which comes first in the file.
fn file_type_box(brands: &Brands) -> Vec<u8> {
    let mut out = Vec::new();
    write_box(&mut out, b"ftyp", |out| {
        out.extend_from_slice(&brands.major);
        out.extend_from_slice(&0u32.to_be_bytes()); // minor_version
        for brand in brands.compatible {
            out.extend_from_slice(brand);
        }
    });
    out
}

/// The version of a box whose unsigned 32-bit fields of version 0 would
/// not hold `values`: 1 if any is `u32::MAX` or beyond (clause 4.2.4:
/// version 1 "should be used only when the 64-bit field sizes they permit,
/// are required"). A duration of all 1s is "indefinite" (clauses 8.2.2.3,
/// 8.3.2.3 and 8.4.2.3), so it takes version 1 too.
fn version(values: &[u64]) -> u8 {
    u8::from(values.iter().any(|&value| value >= u64::from(u32::MAX)))
}

/// Appends a 32-bit field in version 0 and a 64-bit one in version 1.
fn put_sized(out: &mut Vec<u8>, version: u8, value: u64) {
    match version {
        0 => out.extend_from_slice(&(value as u32).to_be_bytes()),
        _ => out.extend_from_slice(&value.to_be_bytes()),
    }
}

/// Appends `entry_count` and `entries`, pairs of 32-bit fields.
fn put_entries(out: &mut Vec<u8>, entries: &[(u32, u32)]) {
    out.extend_from_slice(&(entries.len() as u32).to_be_bytes());
    for &(a, b) in entries {
        out.extend_from_slice(&a.to_be_bytes());
        out.extend_from_slice(&b.to_be_bytes());
    }
}

/// The unity matrix of the movie and track headers (clauses 8.2.2.2 and
/// 8.3.2.2), "the identity matrix" of tracks that do not use one (clause
/// 7.3.2), as audio does not.
fn put_unity_matrix(out: &mut Vec<u8>) {
    for value in [0x0001_0000u32, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000] {
        out.extend_from_slice(&value.to_be_bytes());
    }
}

/// Spec gap: `creation_time` and `modification_time`, in seconds since
/// 1904, are those of the presentation, which a library given only PCM
/// does not know. They are 0, which keeps the file the same for the same
/// input.
const CREATION_TIME: u64 = 0;

/// The `MovieBox` (clause 8.2.1) of `track`, of `durations`: its
/// `SampleTableBox` holds the sample description, then the boxes
/// `sample_tables` appends; `extends` appends any boxes after the
/// `TrackBox`. Each header box comes first in its container (clause
/// 7.3.4).
///
/// Spec reading: it has no `ObjectDescriptorBox` (`'iods'`), whose
/// initial object descriptor would serve an MPEG-4 Systems presentation:
/// ISO/IEC 14496-14 makes the box "Mandatory: No" (clause 6.2.1). An audio
/// file played as it is has no scene for one to start, and the
/// `ES_Descriptor` gives a decoder all it needs.
fn movie_box(
    track: &AudioTrack,
    durations: Durations,
    sample_tables: impl FnOnce(&mut Vec<u8>),
    extends: impl FnOnce(&mut Vec<u8>),
) -> Vec<u8> {
    let mut out = Vec::new();
    write_box(&mut out, b"moov", |out| {
        write_movie_header(out, track, durations);
        write_track(out, track, durations, sample_tables);
        extends(out);
        // A UserDataBox, as for song tags, would come here, last (clause
        // 7.3.4).
    });
    out
}

/// The `MovieHeaderBox` (clause 8.2.2).
///
/// Its `timescale` is the track's, as the media's is (see
/// [`write_media`]), so that the track's edit, whose duration is in this
/// timescale (clause 8.6.6.3), is exact to the sample. Its `duration` is
/// the layout's, that of the one track. `rate`, `volume` and
/// `matrix` keep their template values, as ISO/IEC 14496-14 requires of an
/// MP4 file (clauses 4.4 and 7).
fn write_movie_header(out: &mut Vec<u8>, track: &AudioTrack, durations: Durations) {
    let duration = durations.track;
    let version = version(&[CREATION_TIME, duration]);
    write_full_box(out, b"mvhd", version, 0, |out| {
        put_sized(out, version, CREATION_TIME); // creation_time
        put_sized(out, version, CREATION_TIME); // modification_time
        out.extend_from_slice(&track.timescale.to_be_bytes());
        put_sized(out, version, duration);
        out.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // rate, 1.0
        out.extend_from_slice(&0x0100u16.to_be_bytes()); // volume, 1.0
        out.extend_from_slice(&[0; 2 + 2 * 4]); // reserved
        put_unity_matrix(out);
        out.extend_from_slice(&[0; 6 * 4]); // pre_defined
        out.extend_from_slice(&(TRACK_ID + 1).to_be_bytes()); // next_track_ID
    });
}

/// The `TrackBox` (clause 8.3.1): its header, its edit list if it has
/// one, then its media (clause 7.3.4: the `EditBox` "should precede the
/// MediaBox").
fn write_track(
    out: &mut Vec<u8>,
    track: &AudioTrack,
    durations: Durations,
    sample_tables: impl FnOnce(&mut Vec<u8>),
) {
    write_box(out, b"trak", |out| {
        write_track_header(out, durations);
        if let (Some(delay), Some(duration)) = (track.delay, durations.edit) {
            write_edit(out, delay, duration);
        }
        write_media(out, track, durations, sample_tables);
    });
}

/// `track_enabled`, `track_in_movie` and `track_in_preview` (clause
/// 8.3.2.3): ISO/IEC 14496-14 has the last two "set to the default value
/// of 1 in all files" (clause 6.4).
const TRACK_FLAGS: u32 = 0x00_0007;

/// The `TrackHeaderBox` (clause 8.3.2), its `duration` the layout's;
/// `layer`, `alternate_group`, `volume` and `matrix` keep their template
/// values, `volume` that of an audio track, 1.0; `width` and `height` are
/// 0, as "For non-visual tracks (e.g. audio), they should be set to zero".
fn write_track_header(out: &mut Vec<u8>, durations: Durations) {
    let duration = durations.track;
    let version = version(&[CREATION_TIME, duration]);
    write_full_box(out, b"tkhd", version, TRACK_FLAGS, |out| {
        put_sized(out, version, CREATION_TIME); // creation_time
        put_sized(out, version, CREATION_TIME); // modification_time
        out.extend_from_slice(&TRACK_ID.to_be_bytes());
        out.extend_from_slice(&[0; 4]); // reserved
        put_sized(out, version, duration);
        out.extend_from_slice(&[0; 2 * 4]); // reserved
        out.extend_from_slice(&0u16.to_be_bytes()); // layer
        out.extend_from_slice(&0u16.to_be_bytes()); // alternate_group
        out.extend_from_slice(&0x0100u16.to_be_bytes()); // volume
        out.extend_from_slice(&[0; 2]); // reserved
        put_unity_matrix(out);
        out.extend_from_slice(&0u32.to_be_bytes()); // width
        out.extend_from_slice(&0u32.to_be_bytes()); // height
    });
}

/// The `EditBox` and its `EditListBox` (clauses 8.6.5 and 8.6.6): one
/// edit, which presents `duration` samples from the `delay`'th, at the
/// normal rate, 1.0. A decoder's output starts with
/// `delay` samples that precede the input and ends with samples past it;
/// the edit leaves both out, as Annex A of ISO/IEC 14496-14 describes, so
/// that "the output of the decoder has the same length as the input to
/// the encoder, with the audio in the same temporal position".
///
/// Its `media_time` is in the media's timescale, and its `edit_duration`
/// in the movie's, the same (see [`write_movie_header`]), so that both
/// count samples.
fn write_edit(out: &mut Vec<u8>, delay: u32, duration: u64) {
    let media_time = u64::from(delay);
    // media_time is signed, an int(32) in version 0.
    let version = version(&[duration]).max(u8::from(media_time > i32::MAX as u64));
    write_box(out, b"edts", |out| {
        write_full_box(out, b"elst", version, 0, |out| {
            out.extend_from_slice(&1u32.to_be_bytes()); // entry_count
            put_sized(out, version, duration); // edit_duration
            put_sized(out, version, media_time);
            out.extend_from_slice(&1i16.to_be_bytes()); // media_rate_integer
            out.extend_from_slice(&0i16.to_be_bytes()); // media_rate_fraction
        });
    });
}

/// The `MediaBox` (clause 8.4.1), with its header, handler and information
/// (clause 7.3.4: "the HandlerBox should precede the
/// MediaInformationBox").
///
/// Its `timescale` is the track's, which the codec chooses: "The timescale
/// for an audio track should be chosen to match the sampling rate" (clause
/// 12.2.3.1).
fn write_media(
    out: &mut Vec<u8>,
    track: &AudioTrack,
    durations: Durations,
    sample_tables: impl FnOnce(&mut Vec<u8>),
) {
    let duration = durations.media;
    write_box(out, b"mdia", |out| {
        let version = version(&[CREATION_TIME, duration]);
        write_full_box(out, b"mdhd", version, 0, |out| {
            put_sized(out, version, CREATION_TIME); // creation_time
            put_sized(out, version, CREATION_TIME); // modification_time
            out.extend_from_slice(&track.timescale.to_be_bytes());
            put_sized(out, version, duration);
            out.extend_from_slice(&UNDETERMINED_LANGUAGE.to_be_bytes());
            out.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
        });
        // The HandlerBox (clause 8.4.3) of audio, 'soun' (clause 12.2.1).
        write_full_box(out, b"hdlr", 0, 0, |out| {
            out.extend_from_slice(&0u32.to_be_bytes()); // pre_defined
            out.extend_from_slice(b"soun"); // handler_type
            out.extend_from_slice(&[0; 3 * 4]); // reserved
            out.extend_from_slice(HANDLER_NAME);
        });
        write_box(out, b"minf", |out| {
            // The SoundMediaHeaderBox (clause 12.2.2), its balance at the
            // center, 0, the template value.
            write_full_box(out, b"smhd", 0, 0, |out| {
                out.extend_from_slice(&0i16.to_be_bytes()); // balance
                out.extend_from_slice(&[0; 2]); // reserved
            });
            write_data_information(out);
            write_sample_table(out, track, sample_tables);
        });
    });
}

/// `language`, three letters each less 0x60 in 5 bits (clause 8.4.2.3):
/// `'und'`, which clause 8.4.6.1 offers where there is "no reasonably
/// compatible tag", the PCM coming with no language.
const UNDETERMINED_LANGUAGE: u16 = {
    let [a, b, c] = *b"und";
    ((a as u16 - 0x60) << 10) | ((b as u16 - 0x60) << 5) | (c as u16 - 0x60)
};

/// The `HandlerBox`'s `name`, "a human-readable name for the track type
/// (for debugging and inspection purposes)", as a null-terminated
/// `utf8string` (clause 4.2.1).
const HANDLER_NAME: &[u8] = b"Audio\0";

/// The `DataInformationBox` (clause 8.7.1), with a `DataReferenceBox`
/// (clause 8.7.2) of one entry, a `DataEntryUrlBox` whose flag 1 says the
/// media data is "in the same file", and which so has no `location`.
fn write_data_information(out: &mut Vec<u8>) {
    write_box(out, b"dinf", |out| {
        write_full_box(out, b"dref", 0, 0, |out| {
            out.extend_from_slice(&1u32.to_be_bytes()); // entry_count
            write_full_box(out, b"url ", 0, 0x00_0001, |_| {});
        });
    });
}

/// The `SampleTableBox` (clause 8.5.1): the `SampleDescriptionBox`, first
/// in the order clause 7.3.4 recommends, then the boxes `sample_tables`
/// appends.
///
/// It has no `SyncSampleBox`, so that every sample is a sync sample
/// (clause 8.5.1.1), in a track that is not fragmented, and in one that is,
/// whose samples' flags say so (see [`fragmented`]): each sample of every
/// codec written is decoded from its own bits, which clause 10.1 describes
/// of the audio streams an `AudioRollRecoveryEntry` serves, "in which
/// every sample can be independently decoded", and which the mappings of
/// FLAC and Opus have of theirs ("every FLAC sample is a sync sample",
/// "every Opus sample is a sync sample"), each with its "Sync Sample Box
/// shall not be present".
fn write_sample_table(
    out: &mut Vec<u8>,
    track: &AudioTrack,
    sample_tables: impl FnOnce(&mut Vec<u8>),
) {
    write_box(out, b"stbl", |out| {
        write_full_box(out, b"stsd", 0, 0, |out| {
            out.extend_from_slice(&1u32.to_be_bytes()); // entry_count
            write_sample_entry(out, track);
        });
        sample_tables(out);
    });
}

/// The sample entry, an `AudioSampleEntry` (clause 12.2.3), of the
/// codec's type, fields and boxes. `channelcount` is the number of
/// channels, "the total number of channels in the audio stream".
fn write_sample_entry(out: &mut Vec<u8>, track: &AudioTrack) {
    let entry = &track.sample_entry;
    write_box(out, &entry.coding_name, |out| {
        // SampleEntry (clause 8.5.2.2).
        out.extend_from_slice(&[0; 6]); // reserved
        out.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
        // AudioSampleEntry.
        out.extend_from_slice(&[0; 2 * 4]); // reserved
        out.extend_from_slice(&track.channels.to_be_bytes()); // channelcount
        out.extend_from_slice(&entry.sample_size.to_be_bytes()); // samplesize
        out.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
        out.extend_from_slice(&[0; 2]); // reserved
        out.extend_from_slice(&entry.sample_rate.to_be_bytes()); // samplerate
        out.extend_from_slice(&entry.boxes);
    });
}

/// The number of samples of `sizes`.
fn count(sizes: &[u32]) -> u32 {
    u32::try_from(sizes.len()).expect("fewer than 2^32 samples")
}

/// The sample tables of the samples of `sizes` and `durations`, one chunk
/// of them at `chunk_offset`, after the `SampleDescriptionBox`, in the
/// order clause 7.3.4 recommends: `TimeToSampleBox`, `SampleToChunkBox`,
/// `SampleSizeBox` and `ChunkOffsetBox`. Without samples, each has no
/// entries.
fn write_sample_tables(out: &mut Vec<u8>, sizes: &[u32], durations: &[u32], chunk_offset: u32) {
    debug_assert_eq!(sizes.len(), durations.len());
    let count = count(sizes);
    // The TimeToSampleBox (clause 8.6.1.2): an entry for each run of
    // samples that last as long.
    write_full_box(out, b"stts", 0, 0, |out| {
        put_entries(out, &runs(durations));
    });
    // The SampleToChunkBox (clause 8.7.4): one chunk of every sample,
    // described by the one sample entry.
    //
    // Spec gap: the standard leaves how samples are grouped into chunks
    // to the writer. The track's samples lie one after another in the
    // MediaDataBox, and there is no other track to interleave them
    // with, so they are one chunk, whose offset is all the
    // ChunkOffsetBox gives.
    write_full_box(out, b"stsc", 0, 0, |out| {
        let chunks = u32::from(count > 0);
        out.extend_from_slice(&chunks.to_be_bytes()); // entry_count
        if count > 0 {
            out.extend_from_slice(&1u32.to_be_bytes()); // first_chunk
            out.extend_from_slice(&count.to_be_bytes()); // samples_per_chunk
            out.extend_from_slice(&1u32.to_be_bytes()); // sample_description_index
        }
    });
    // The SampleSizeBox (clause 8.7.3.2), as "the first version is
    // preferred for maximum compatibility" over the CompactSampleSizeBox
    // (clauses 8.7.3.1 and 8.7.3.3); with one sample_size where every
    // sample is as long.
    write_full_box(out, b"stsz", 0, 0, |out| {
        let sample_size = common_size(sizes).unwrap_or(0);
        out.extend_from_slice(&sample_size.to_be_bytes());
        out.extend_from_slice(&count.to_be_bytes()); // sample_count
        if sample_size == 0 {
            for &size in sizes {
                out.extend_from_slice(&size.to_be_bytes()); // entry_size
            }
        }
    });
    // The ChunkOffsetBox (clause 8.7.5): the one chunk's offset from
    // the file's start.
    write_full_box(out, b"stco", 0, 0, |out| {
        let chunks = u32::from(count > 0);
        out.extend_from_slice(&chunks.to_be_bytes()); // entry_count
        if count > 0 {
            out.extend_from_slice(&chunk_offset.to_be_bytes());
        }
    });
}

/// The runs of equal values in `durations`, each its length and the
/// value: the `sample_count` and `sample_delta` of the entries of a
/// `TimeToSampleBox`.
fn runs(durations: &[u32]) -> Vec<(u32, u32)> {
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for &duration in durations {
        match runs.last_mut() {
            Some((count, delta)) if *delta == duration => *count += 1,
            _ => runs.push((1, duration)),
        }
    }
    runs
}

/// The size every sample of `sizes` has, if they all have one, and there
/// is a sample.
fn common_size(sizes: &[u32]) -> Option<u32> {
    let first = *sizes.first()?;
    sizes.iter().all(|&size| size == first).then_some(first)
}

/// The `grouping_type` of the pre-roll (clause 10.1).
const ROLL: &FourCc = b"roll";

/// The pre-roll of every sample, as a sample group of `grouping_type`
/// `'roll'` (clause 10.1): this `SampleToGroupBox` (clause 8.9.2) maps the
/// next `count` samples to the one entry of the `SampleGroupDescriptionBox`
/// (clause 8.9.3) that [`write_roll_description`] writes.
///
/// Of AAC: a frame's samples overlap the previous frame's, and decode correctly
/// only once that frame has been decoded too; Annex A of ISO/IEC 14496-14
/// has "a pre-roll sample group, for example a pre-roll value of ‒1 (minus
/// one)" tell a reader to decode the frame before the first it presents,
/// including at the start of the edit, which falls in the second frame.
///
/// Spec reading: the first frame is a member of the group too, although
/// there is no frame before it and it decodes correctly from the silence a
/// decoder starts with, which the encoder took to precede the input.
/// ISO/IEC 14496-12 neither requires nor forbids that. Apple's QuickTime File
/// Format documentation does require it ("All AAC packets including the
/// encoder delay must be associated with the sample group", in
/// "Representing encoder delay with track structures"; its "Sample-to-group
/// atom" gives one entry "describing all the AAC packets in the track"),
/// and "Using track structures to represent encoder delay explicitly" warns
/// that "An incomplete implementation will result in unspecified
/// interpretation by Apple software and tools". So the `SampleToGroupBox`
/// has the one entry it describes, of every frame.
///
/// The `SampleToGroupBox` is of version 0, as "Version 1 of this box should
/// only be used if a grouping_type_parameter is needed".
fn write_roll_mapping(out: &mut Vec<u8>, count: u32) {
    write_full_box(out, b"sbgp", 0, 0, |out| {
        out.extend_from_slice(ROLL); // grouping_type
        let entries: &[(u32, u32)] = if count > 0 { &[(count, 1)] } else { &[] };
        put_entries(out, entries);
    });
}

/// The `SampleGroupDescriptionBox` of the pre-roll (see
/// [`write_roll_mapping`]), of `flags`, with its one entry, of
/// `roll_distance`.
///
/// Spec reading, of AAC: clause 10.1 offers an `AudioRollRecoveryEntry`
/// (`'roll'`), for "audio streams in which every sample can be
/// independently decoded, but the decoder output is only assured to be
/// correct after pre-rolling", and an `AudioPreRollEntry` (`'prol'`), for
/// streams "in which not every sample can be independently decoded", whose
/// `roll_distance` "shall be a positive value". AAC frames are each
/// decoded from their own bits, and Annex A's value is negative: the entry
/// is a `'roll'`, of `roll_distance` −1, "the number of samples before the
/// sample that is a group member that need to be decoded".
///
/// The box is of version 1, with the entries' `default_length`, as version
/// 0 entries are "deprecated" (clause 8.9.3.1), and not 2, as each frame's
/// group is mapped, none by default.
fn write_roll_description(out: &mut Vec<u8>, roll_distance: i16, flags: u32) {
    write_full_box(out, b"sgpd", 1, flags, |out| {
        out.extend_from_slice(ROLL); // grouping_type
        out.extend_from_slice(&2u32.to_be_bytes()); // default_length
        out.extend_from_slice(&1u32.to_be_bytes()); // entry_count
        out.extend_from_slice(&roll_distance.to_be_bytes());
    });
}

#[cfg(test)]
mod tests {
    use super::progressive::write_file;
    use super::*;
    use crate::boxes::read::{FourCcs, boxes, children, find, u32_at};

    const MP42_ISOM: [FourCc; 2] = [*b"mp42", *b"isom"];

    /// A track of stereo at `sample_rate`, as AAC's are, its sample entry
    /// an `'mp4a'` whose `ESDBox` holds a stub.
    pub fn track(sample_rate: u32) -> AudioTrack {
        let mut boxes = Vec::new();
        write_full_box(&mut boxes, b"esds", 0, 0, |out| {
            out.extend_from_slice(&[0x03, 0x00]);
        });
        AudioTrack {
            brands: Brands {
                major: *b"mp42",
                compatible: &MP42_ISOM,
            },
            sample_entry: SampleEntry {
                coding_name: *b"mp4a",
                sample_size: 16,
                sample_rate: sample_rate << 16,
                boxes,
            },
            timescale: sample_rate,
            channels: 2,
            delay: Some(1024),
            default_duration: 1024,
            roll_distance: Some(-1),
        }
    }

    /// The file of `frames` frames of a byte each, of `length` samples.
    fn write(track: &AudioTrack, frames: usize, length: u64) -> Vec<u8> {
        let mut file = Vec::new();
        let (sizes, durations) = (vec![1; frames], vec![1024; frames]);
        write_file(
            track,
            &sizes,
            &durations,
            length,
            &vec![0; frames],
            &mut file,
        )
        .unwrap();
        file
    }

    #[test]
    fn the_edit_presents_the_input_alone() {
        let file = write(&track(44_100), 4, 3000);
        let elst = find(&file, &[b"moov", b"trak", b"edts", b"elst"]);
        let expected = [
            &[0, 0, 0, 0][..],   // version, flags
            &1u32.to_be_bytes(), // entry_count
            &3000u32.to_be_bytes(),
            &1024u32.to_be_bytes(),
            &[0, 1, 0, 0],
        ]
        .concat();
        assert_eq!(elst.payload, expected);
        let mvhd = find(&file, &[b"moov", b"mvhd"]);
        assert_eq!(u32_at(mvhd.payload, 12), 44_100); // timescale
        assert_eq!(u32_at(mvhd.payload, 16), 3000); // duration
        assert_eq!(u32_at(mvhd.payload, 96), 2); // next_track_ID
        let tkhd = find(&file, &[b"moov", b"trak", b"tkhd"]);
        assert_eq!(u32_at(tkhd.payload, 0), 7); // version, flags
        assert_eq!(u32_at(tkhd.payload, 12), 1); // track_ID
        assert_eq!(u32_at(tkhd.payload, 20), 3000); // duration
        let mdhd = find(&file, &[b"moov", b"trak", b"mdia", b"mdhd"]);
        assert_eq!(u32_at(mdhd.payload, 12), 44_100); // timescale
        assert_eq!(u32_at(mdhd.payload, 16), 4 * 1024); // duration
        assert_eq!(&mdhd.payload[20..], [0x55, 0xc4, 0, 0]); // 'und'
    }

    #[test]
    fn durations_past_32_bits_take_version_1() {
        let length = u64::from(u32::MAX);
        let file = write(&track(44_100), 2, length);
        let mvhd = find(&file, &[b"moov", b"mvhd"]);
        assert_eq!(mvhd.payload[0], 1);
        assert_eq!(u32_at(mvhd.payload, 20), 44_100); // timescale
        assert_eq!(&mvhd.payload[24..32], length.to_be_bytes()); // duration
        assert_eq!(mvhd.payload.len(), 112);
        let tkhd = find(&file, &[b"moov", b"trak", b"tkhd"]);
        assert_eq!(tkhd.payload[0], 1);
        assert_eq!(&tkhd.payload[28..36], length.to_be_bytes());
        assert_eq!(tkhd.payload.len(), 96);
        let elst = find(&file, &[b"moov", b"trak", b"edts", b"elst"]);
        assert_eq!(elst.payload[0], 1);
        assert_eq!(&elst.payload[8..16], length.to_be_bytes());
        assert_eq!(&elst.payload[16..24], 1024u64.to_be_bytes());
        // The media's 2048 samples keep version 0.
        let mdhd = find(&file, &[b"moov", b"trak", b"mdia", b"mdhd"]);
        assert_eq!(mdhd.payload[0], 0);
        let short = write(&track(44_100), 2, length - 1);
        assert_eq!(find(&short, &[b"moov", b"mvhd"]).payload[0], 0);
        // media_time is an int(32) in version 0.
        let mut late = track(44_100);
        late.delay = Some(1 << 31);
        let file = write(&late, 2, 0);
        let elst = find(&file, &[b"moov", b"trak", b"edts", b"elst"]);
        assert_eq!(elst.payload[0], 1);
        assert_eq!(&elst.payload[16..24], (1u64 << 31).to_be_bytes());
        late.delay = Some((1 << 31) - 1);
        let file = write(&late, 2, 0);
        let elst = find(&file, &[b"moov", b"trak", b"edts", b"elst"]);
        assert_eq!(elst.payload[0], 0);
    }

    #[test]
    fn the_sample_entry_holds_the_codec_s_fields_and_boxes() {
        let mut track = track(44_100);
        track.channels = 6;
        track.sample_entry.sample_size = 24;
        let file = write(&track, 1, 0);
        let stsd = find(
            &file,
            &[b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd"],
        );
        let expected = [
            &[0, 0, 0, 0][..],   // version, flags
            &1u32.to_be_bytes(), // entry_count
            &(8 + 28 + 14u32).to_be_bytes(),
            b"mp4a",
            &[0; 6],  // reserved
            &[0, 1],  // data_reference_index
            &[0; 8],  // reserved
            &[0, 6],  // channelcount
            &[0, 24], // samplesize
            &[0; 4],  // pre_defined, reserved
            &(44_100u32 << 16).to_be_bytes(),
            &14u32.to_be_bytes(),
            b"esds",
            &[0, 0, 0, 0], // version, flags
            &[0x03, 0x00], // the ES_Descriptor
        ]
        .concat();
        assert_eq!(stsd.payload, expected);
    }

    #[test]
    fn a_track_without_a_delay_has_no_edit_list() {
        let mut track = track(48_000);
        track.delay = None;
        let file = write(&track, 3, 3 * 1024);
        let kinds = children(&find(&file, &[b"moov", b"trak"])).kinds();
        assert_eq!(kinds, ["tkhd", "mdia"]);
        let tkhd = find(&file, &[b"moov", b"trak", b"tkhd"]);
        assert_eq!(u32_at(tkhd.payload, 20), 3 * 1024); // duration
    }

    #[test]
    fn boxes_follow_the_recommended_order() {
        let file = write(&track(44_100), 2, 100);
        let kinds = |path: &[&[u8; 4]]| children(&find(&file, path)).kinds();
        assert_eq!(kinds(&[b"moov"]), ["mvhd", "trak"]);
        assert_eq!(kinds(&[b"moov", b"trak"]), ["tkhd", "edts", "mdia"]);
        assert_eq!(
            kinds(&[b"moov", b"trak", b"mdia"]),
            ["mdhd", "hdlr", "minf"]
        );
        assert_eq!(
            kinds(&[b"moov", b"trak", b"mdia", b"minf"]),
            ["smhd", "dinf", "stbl"]
        );
        assert_eq!(boxes(&file, 0).kinds()[0], "ftyp");
        let dref = find(
            &file,
            &[b"moov", b"trak", b"mdia", b"minf", b"dinf", b"dref"],
        );
        assert_eq!(
            dref.payload,
            [[0; 4], [0, 0, 0, 1], [0, 0, 0, 12], *b"url ", [0, 0, 0, 1]].concat()
        );
        let hdlr = find(&file, &[b"moov", b"trak", b"mdia", b"hdlr"]);
        assert_eq!(&hdlr.payload[8..12], b"soun");
        assert!(hdlr.payload.ends_with(b"\0"));
    }
}
