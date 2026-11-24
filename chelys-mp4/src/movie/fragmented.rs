// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! A fragmented movie file (clauses 8.8 and A.10): an init segment, its
//! `FileTypeBox` and a `MovieBox` whose track has no samples, and then
//! movie fragments, each a `MovieFragmentBox` and the `MediaDataBox` of its
//! frames, written as the frames come.
//!
//! It is the file of one track whose samples, each a frame or packet of
//! the stream, lie one after another, each fragment's in one run: what an
//! audio encoder writes, and not what a fragmented file can be in general.
//!
//! It has no `MovieFragmentRandomAccessBox` (clause 8.8.9), which comes
//! last and gives each fragment's offset in the file, nor a
//! `SegmentIndexBox` or `SegmentTypeBox` (clauses 8.14.3 and 8.14.2), of
//! segments: each is optional, and needs what is known only once the
//! fragments it indexes are written, or a delivery that divides the file.

use std::num::NonZeroU32;

use super::{
    AudioTrack, Durations, TRACK_ID, common_size, file_type_box, movie_box, put_sized, version,
    write_roll_description, write_roll_mapping, write_sample_tables,
};
use crate::boxes::{self, write_box, write_full_box};

/// Where a fragmented encoder gives the fragments it writes, one call
/// each, in their order, after its init segment.
pub trait FragmentSink {
    /// Takes a fragment: a `MovieFragmentBox` and the `MediaDataBox` of its
    /// frames, which follows the init segment and the fragments before it.
    fn fragment(&mut self, fragment: &[u8], info: FragmentInfo);
}

/// Appends each fragment, as to the file after its init segment.
impl FragmentSink for Vec<u8> {
    fn fragment(&mut self, fragment: &[u8], _: FragmentInfo) {
        self.extend_from_slice(fragment);
    }
}

/// Calls the closure with each fragment, as to send each apart.
impl<F: FnMut(&[u8], FragmentInfo)> FragmentSink for F {
    fn fragment(&mut self, fragment: &[u8], info: FragmentInfo) {
        self(fragment, info);
    }
}

/// What a fragment holds, its durations in the track's timescale: samples
/// of each channel at the rate a decoder outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct FragmentInfo {
    /// The fragment's `sequence_number`: 1 for the first, and one more for
    /// each after it.
    pub sequence_number: u32,
    /// The decoding time of the fragment's first frame, its
    /// `baseMediaDecodeTime`: the samples of the frames before it. It is
    /// not the time at which the fragment is presented: it is on the
    /// media's timeline, which counts any delay that a decoder's output
    /// starts with, the samples before the input that the edit list leaves
    /// out (`chelys_aac::DELAY` for chelys's AAC, the delay given for AAC
    /// packets, the pre-skip for Opus), so that the fragment is presented
    /// from that many samples earlier.
    pub decode_time: u64,
    /// The frames the fragment holds, each a sample of the track: an AAC
    /// frame, a FLAC frame or an Opus packet.
    pub frames: u32,
    /// The duration of those frames in samples; in the last fragment of a
    /// stream that ends with padding, as chelys's AAC, AAC packets of a
    /// known delay and Opus do, its last frame lasts only to the end of the
    /// input, where its length is known, and not to the end of the padding
    /// it decodes to.
    pub duration: u64,
}

/// A fragment written: its bytes and what they hold.
pub struct Fragment {
    pub bytes: Vec<u8>,
    pub info: FragmentInfo,
}

/// The most frames a fragment has, whatever its duration: their sizes and
/// durations in its run, 8 bytes each at most, keep the `MovieFragmentBox`
/// far under the 2 GiB that the run's signed `data_offset` reaches, and
/// they are counted in 32 bits. AAC's fragments, of at most 2^32 - 1 samples in frames of 1024,
/// never reach it.
const MAX_FRAMES: usize = 1 << 24;

/// Divides frames, given one at a time with their durations, into
/// fragments of about `fragment_samples` samples each, as long as a whole
/// number of frames can be, and of [`MAX_FRAMES`] frames at most, and
/// writes each fragment once its last frame has come. It is written for
/// one track.
#[derive(Debug)]
pub struct Fragmenter {
    /// The `TrackExtendsBox`'s `default_sample_duration`, which a fragment
    /// whose frames all last as long need not repeat.
    default_duration: u32,
    /// Whether the frames are mapped to the pre-roll group.
    roll: bool,
    fragment_samples: NonZeroU32,
    /// The frames of the fragment being gathered, one after another.
    data: Vec<u8>,
    sizes: Vec<u32>,
    durations: Vec<u32>,
    /// The duration of the frames written in fragments before it.
    time_written: u64,
    /// The duration of the frames of the fragment being gathered.
    time_gathered: u64,
    /// The `sequence_number` of the fragment being gathered.
    sequence_number: u32,
    /// The number of the next boundary the fragments keep to.
    boundary: u64,
}

impl Fragmenter {
    pub fn new(track: &AudioTrack, fragment_samples: NonZeroU32) -> Self {
        Self {
            default_duration: track.default_duration,
            roll: track.roll_distance.is_some(),
            fragment_samples,
            data: Vec::new(),
            sizes: Vec::new(),
            durations: Vec::new(),
            time_written: 0,
            time_gathered: 0,
            sequence_number: 1,
            boundary: 1,
        }
    }

    /// Whether a fragment that ends at `end`, its last frame lasting
    /// `last`, reaches the `n`'th boundary, at `n` times
    /// `fragment_samples`: whether the boundary is before the middle of a
    /// next frame as long as the last.
    ///
    /// Spec gap: how long a fragment is, is the writer's to choose. The
    /// boundaries are counted from the stream's start, so that fragments
    /// of a duration that is not a whole number of frames vary by a frame,
    /// and their ends keep to the duration's multiples: at 44.1 kHz,
    /// fragments of a second of AAC are 43 frames, and 44 every fifteenth
    /// or so. Where the frames all last as long, a fragment ends with the
    /// frame whose end is nearest the boundary, the later where two are as
    /// near. Where they vary, the next frame's duration is not known when
    /// the fragment is to end, and is taken to be the last's.
    fn reaches(&self, n: u64, end: u64, last: u32) -> bool {
        let boundary = 2 * u128::from(n) * u128::from(self.fragment_samples.get());
        boundary < 2 * u128::from(end) + u128::from(last)
    }

    /// Takes the next frame, which lasts `duration` samples, and gives the
    /// fragment that it ends, if it ends one: if it reaches the next
    /// boundary, or [`MAX_FRAMES`].
    pub fn push_frame(&mut self, frame: &[u8], duration: u32) -> Option<Fragment> {
        self.gather(frame, duration);
        let end = self.time_written + self.time_gathered;
        let full = self.sizes.len() >= MAX_FRAMES;
        (full || self.reaches(self.boundary, end, duration)).then(|| self.write())
    }

    fn gather(&mut self, frame: &[u8], duration: u32) {
        self.data.extend_from_slice(frame);
        self.sizes
            .push(u32::try_from(frame.len()).expect("a frame under 4 GiB"));
        self.durations.push(duration);
        self.time_gathered += u64::from(duration);
    }

    /// Writes the fragment of the frames taken since the last, if there
    /// are any: a fragment always has a frame. The next fragment ends at
    /// the next boundary past them.
    pub fn flush(&mut self) -> Option<Fragment> {
        if self.sizes.is_empty() {
            return None;
        }
        Some(self.write())
    }

    /// Takes the stream's `last` frame, and writes the last fragment, of
    /// the frames taken since the last fragment and that frame, which lasts
    /// `last_duration` samples.
    ///
    /// Spec reading: in a stream that ends with padding, as AAC's and
    /// Opus's do, the last frame's `sample_duration` is shorter than the
    /// samples it decodes to, by the padding that the stream ends with past
    /// the input, so that it ends where the input does. A sample's duration
    /// is the difference between its decoding time and the next's,
    /// "`DT[n+1] = DT[n] + sample_delta[n]`", and "In the absence of
    /// composition offsets, the sum of all sample durations gives the
    /// duration of the media in the track" (clause 8.6.1.2.1), which
    /// TrackRunBox's `sample_duration` gives in a fragment (Table 5). For
    /// the last sample, with no successive sample, `sample_delta` "should be
    /// set equal to duration subtracted by the composition timestamp of the
    /// last sample" (clause 8.6.1.2.3), the duration being the
    /// `MediaHeaderBox`'s, which in the init segment is that of the samples
    /// the `MovieBox` describes, none (see [`init_segment`]); that of the
    /// whole media, which ends with the input, gives the value taken, and
    /// "The decoding of the last ISOBMFF sample of a track can produce
    /// additional audio samples that are not meant to be rendered" (clause
    /// 8.4.2.3). The media then ends at the input's end, the track's delay
    /// and its length into it: where an edit of the input's length from the
    /// delay ends, and with it the duration of the `MovieExtendsHeaderBox`;
    /// so that the implied edit "that inserts the new media material" of
    /// fragments (clause 8.6.6.1, NOTE 1), read as starting at the media's
    /// end, adds no padding past it; and "the accumulated duration of all
    /// samples", which an edit of duration 0 lasts (clause 8.6.6.1), ends at
    /// the input's end too. The duration is at least 1, as the last frame
    /// holds the last sample of the input, or of the delay where there is no
    /// input, and at most the frame's, where there is no padding. The AAC
    /// file without fragments keeps every frame's duration, as its edit list
    /// alone ends its presentation.
    ///
    /// Settled by decoding with ffmpeg and GStreamer (`qtdemux`,
    /// `avdec_aac`, `audiosegmentclip`; black box): GStreamer 1.28.9
    /// presents a fragmented file of a declared length as it presents the
    /// file written whole of the same input, the padding left out, its
    /// `qtdemux` ending the last segment it marks at the input's end, where
    /// the padding is none, a frame's less one, and in between; and ffprobe
    /// n9.0.2 gives the stream the media's duration, the delay and the
    /// input, though ffmpeg presents the padding all the same (see
    /// [`init_segment`]). The conformance tests are
    /// `gstreamer_decodes_fragmented_files_of_a_declared_length_as_the_files_written_whole`
    /// and `ffprobe_reads_fragmented_files_as_long_as_their_media`.
    ///
    /// # Panics
    ///
    /// If `last_duration` is 0.
    pub fn finish(&mut self, last: &[u8], last_duration: u32) -> Fragment {
        assert!(last_duration > 0, "a last frame of 0 samples");
        self.gather(last, last_duration);
        self.write()
    }

    /// Writes the fragment of the frames taken since the last, of which
    /// there is one at least.
    fn write(&mut self) -> Fragment {
        let frames = u32::try_from(self.sizes.len()).expect("MAX_FRAMES frames at most");
        let info = FragmentInfo {
            sequence_number: self.sequence_number,
            decode_time: self.time_written,
            frames,
            duration: self.time_gathered,
        };
        let run = Run {
            sizes: &self.sizes,
            durations: &self.durations,
            default_duration: self.default_duration,
            roll: self.roll,
        };
        let bytes = fragment(&info, &run, &self.data);
        let last = *self.durations.last().expect("a frame at least");
        self.data.clear();
        self.sizes.clear();
        self.durations.clear();
        self.time_written += self.time_gathered;
        self.time_gathered = 0;
        // Spec gap: the sequence_number "usually starts at 1 and increases
        // for each movie fragment in the file" (clause 8.8.5.1), but its 32
        // bits hold no more than 2^32 - 1 fragments: 136 years of fragments
        // of a second, or 2.9 years of fragments of one frame at 48 kHz.
        // The standard says nothing of a file with more. Past that the
        // number wraps to 0 and counts up again, and a reader that checks
        // the order may take the fragment where it wraps as out of order.
        self.sequence_number = self.sequence_number.wrapping_add(1);
        // The first boundary that the frames written do not reach: the
        // least n for which 2 n fragment_samples >= 2 end + last.
        let past = 2 * u128::from(self.time_written) + u128::from(last);
        let fragment_samples = u128::from(self.fragment_samples.get());
        self.boundary = u64::try_from(past.div_ceil(2 * fragment_samples)).unwrap_or(u64::MAX);
        Fragment { bytes, info }
    }
}

/// The frames of a fragment's one run: their sizes and durations, the
/// duration the `TrackExtendsBox` gives each by default, and whether they
/// are mapped to the pre-roll group.
struct Run<'a> {
    sizes: &'a [u32],
    durations: &'a [u32],
    default_duration: u32,
    roll: bool,
}

/// The `default_sample_flags` of every frame (clause 8.8.3.1), as the
/// `TrackExtendsBox` gives them, which no fragment overrides:
///
/// - `is_leading` 2, "this sample is not a leading sample": no frame is
///   presented before the frame before it.
/// - `sample_depends_on` 2, "this sample does not depend on others": each
///   frame is decoded from its own bits, as the pre-roll's
///   `AudioRollRecoveryEntry` takes them to be (see
///   [`write_roll_description`]), and as each codec's are (see
///   `write_sample_table` in `movie`).
/// - `sample_is_depended_on` 0, "the dependency of other samples on this
///   sample is unknown".
/// - `sample_has_redundancy` 2, "there is no redundant coding in this
///   sample".
/// - `sample_padding_value` 0, `sample_is_non_sync_sample` 0 and
///   `sample_degradation_priority` 0.
///
/// Spec reading: no frame depends on another to be decoded, so none is
/// depended on in that sense, which would make `sample_is_depended_on` 2,
/// "disposable"; but an AAC frame or Opus packet is needed to present the
/// next correctly, as its samples overlap the next one's, and a reader
/// that disposed of it would break the pre-roll. A FLAC frame is not, but
/// "unknown" is true of it too. The file without fragments has no
/// `SampleDependencyTypeBox`, in whose absence "the dependency of other
/// samples on this one is unknown" (clause 8.6.4.1), and the fragments say
/// so too.
///
/// Spec reading: every frame is a sync sample, `sample_is_non_sync_sample`
/// 0, as in the file without fragments, which has no `SyncSampleBox` (see
/// [`movie_box`]): "When this value is set to 0 for a sample, it is the same
/// as if the sample were not in a movie fragment and marked with an entry
/// in the SyncSampleBox (or, if all samples are sync samples, the
/// SyncSampleBox were absent)" (clause 8.8.3.1).
const SAMPLE_FLAGS: u32 = (2 << 26) | (2 << 24) | (2 << 20);

/// The init segment of `track`, whose input is `length` samples long, if
/// known: its `FileTypeBox`, as in the file without fragments, and a
/// `MovieBox` of no samples, with a `MovieExtendsBox` that says the
/// fragments follow.
///
/// Spec reading: the track's samples are all in fragments, and the
/// `MovieBox` describes none: "an initial segment (often called an
/// initialization segment) contains a MovieBox in which the track(s)
/// contain no samples" (clause A.10). Clause 8.5.1.1 requires the
/// `SampleDescriptionBox`, a sample size box, the `SampleToChunkBox` and a
/// chunk offset box of a track that "does reference data", and lets a
/// track that "references no data" do without them. The track references
/// data through its fragments, if not through the `MovieBox`, and needs
/// its `SampleDescriptionBox` either way; so it has the sample tables of
/// the file without fragments, each with no entries, which every reading
/// allows.
///
/// Spec reading: the durations. The media's (clause 8.4.2.3) is that of
/// the samples the `MovieBox` describes, none: 0. The track's is 0 too.
/// Clause 8.3.2.3 makes it, where there is an edit list, "the sum of the
/// durations of all of the track's edits", which would be the edit's
/// below; but clause 8.3.2.1 says of the same field that it "does not
/// include the duration of following movie fragments, if any, but only of
/// the media in the enclosing MovieBox", and that "The
/// MovieExtendsHeaderBox may be used to document the duration including
/// movie fragments", as Table 5 has its `fragment_duration` stand for the
/// track's `duration`. The rule for a movie with fragments is taken over
/// the general one; an edit need not end within the track's duration, as
/// "A non-‘empty’ edit may insert a portion of the media timeline that is
/// not present in the initial movie, and is present only in subsequent
/// movie fragments" (clause 8.6.6.1). The movie's, that of "the longest
/// track" (clause 8.2.2.3), is then 0 as well: "If the duration fields in
/// all tracks are 0, and movie fragments are present, the duration in
/// MovieHeaderBox should be set to indefinite or 0" (clause 8.8.2.1).
///
/// A track without an edit list, whose decoder's output starts with no
/// delay, as FLAC's, needs none to present its media from the start and to
/// its end, where its last frame ends; where its length is known, it is the
/// `MovieExtendsHeaderBox`'s duration alone.
///
/// Where the input's length is known, it is the edit's duration and the
/// `MovieExtendsHeaderBox`'s, "the duration of the longest track,
/// including movie fragments" (clause 8.8.2.3), as clause 8.6.6.1 has the
/// last edit "adjusted such that the total duration of all edits
/// corresponds to the duration in the MovieExtendsHeaderBox". Nor does the
/// edit end with an implied edit of the fragments' media (clause 8.6.6.1,
/// NOTE 1): its duration already reaches the end of the presentation that
/// the `MovieExtendsHeaderBox` gives, and of the media, whose last frame
/// lasts only to the end of the input (see [`Fragmenter::finish`]). Where
/// the length is not known, there
/// is no `MovieExtendsHeaderBox`, and the edit's duration is 0, read as
/// the "media duration field set to 0" of the last entry, which "When a
/// movie is fragmented, and does not contain any MovieExtendsHeaderBox"
/// readers "shall interpret ... as being the accumulated duration of all
/// samples defined in the initial movie and any further movie-fragments"
/// (clause 8.6.6.1); and the movie's duration of 0 "should be interpreted
/// as indefinite duration" (clause 8.8.2.1).
///
/// Spec reading: that edit then runs past the media. Its duration, "the
/// accumulated duration of all samples", is that of the media, which ends
/// with the input (see [`Fragmenter::finish`]), but it starts `delay`
/// samples into it, at its `media_time`, so that it runs past the media's
/// end by the delay, where there is nothing to present. No edit list can
/// say both "from `delay` on" and "to the end" without the length. Only
/// decoding shows what a reader presents past the media's end, where the
/// last frame decodes to the padding.
///
/// Settled by decoding with ffmpeg (black box): ffmpeg n9.0.2 presents
/// every sample of the frames from the edit's `media_time` on, the padding
/// included, and nothing past them. It presents the same where the length
/// is known, the padding that the edit then leaves out included; and
/// ffprobe gives the stream the duration of its media, the delay and the
/// input, whether the length is known or not. The conformance tests are
/// `ffmpeg_decodes_fragmented_files_to_the_frames_of_the_file_written_whole`,
/// `ffprobe_reads_fragmented_files_as_long_as_their_media` and
/// `ffmpeg_reads_empty_and_short_fragmented_files`.
///
/// Settled by decoding with GStreamer (`qtdemux`, `avdec_aac`,
/// `audiosegmentclip`; black box): where the length is not known,
/// GStreamer 1.28.9's `qtdemux` marks a segment from the delay with no
/// end, and it presents the padding after the input; where it is known,
/// it presents the input alone (see [`Fragmenter::finish`]). The
/// conformance test is
/// `gstreamer_presents_the_padding_of_fragmented_files_of_no_length`.
pub fn init_segment(track: &AudioTrack, length: Option<u64>) -> Vec<u8> {
    let durations = Durations {
        edit: track.delay.map(|_| length.unwrap_or(0)),
        track: 0,
        media: 0,
    };
    let sample_tables = |out: &mut Vec<u8>| {
        write_sample_tables(out, &[], &[], 0);
        let Some(roll_distance) = track.roll_distance else {
            return;
        };
        // Spec reading: the frames are mapped to the pre-roll in each
        // fragment (see `fragment`), and none are in the MovieBox,
        // so it has no SampleToGroupBox, as "there may also be no
        // SampleToGroupBox with the given grouping_type if no samples are
        // mapped to a description of that grouping_type" (clause 8.9.2.1).
        //
        // The description has static_group_description, flag 0x1, as no
        // fragment describes the pre-roll again: "there are no
        // SampleGroupDescriptionBoxes of this grouping_type in any
        // TrackFragmentBox of this track" (clause 8.9.3.1). It does not
        // have static_mapping, 0x2, as the fragments map their frames.
        write_roll_description(out, roll_distance, 0x00_0001);
    };
    let extends = |out: &mut Vec<u8>| {
        write_box(out, b"mvex", |out| {
            // The MovieExtendsHeaderBox comes with every known length, and
            // with an edit list only with an edit of that length, as clause
            // 8.6.6.1 adjusts the one to the other. GStreamer 1.28.9's qtdemux (black box), given
            // the edit without the box, ends each segment it marks at the
            // end of the fragments it has read, until the last: see the
            // conformance test
            // `gstreamer_ends_a_finite_edit_without_mehd_with_the_fragments_read`.
            if let Some(length) = length {
                let version = version(&[length]);
                write_full_box(out, b"mehd", version, 0, |out| {
                    put_sized(out, version, length); // fragment_duration
                });
            }
            // The TrackExtendsBox (clause 8.8.3), its flags 0.
            write_full_box(out, b"trex", 0, 0, |out| {
                out.extend_from_slice(&TRACK_ID.to_be_bytes());
                out.extend_from_slice(&1u32.to_be_bytes()); // default_sample_description_index
                out.extend_from_slice(&track.default_duration.to_be_bytes()); // default_sample_duration
                out.extend_from_slice(&0u32.to_be_bytes()); // default_sample_size
                out.extend_from_slice(&SAMPLE_FLAGS.to_be_bytes()); // default_sample_flags
            });
        });
    };
    let mut out = file_type_box(&track.brands);
    out.extend(movie_box(track, durations, sample_tables, extends));
    out
}

/// The fragment `info` describes, of the frames of `run`, which
/// `media_data` holds one after another: its `MovieFragmentBox` (clause
/// 8.8.4), whose header gives its `sequence_number` (clause 8.8.5), and
/// whose one `TrackFragmentBox` (clause 8.8.6) holds the track's header, its
/// decoding time, one run of every frame and any pre-roll; then the
/// `MediaDataBox` of the frames.
///
/// Spec reading: the run's `data_offset` is from the first byte of the
/// `MovieFragmentBox`. The `TrackFragmentHeaderBox` has neither
/// base-data-offset-present nor default-base-is-moof, and its track
/// fragment is the first in the `MovieFragmentBox`, whose data reference
/// is a `'url '` and not an `IdentifiedMediaDataBox`'s, so that its
/// `base_data_offset` "is inferred to be equal to 0" and "is relative to
/// the first byte of the MovieFragmentBox containing this box" (Table 6),
/// as with default-base-is-moof; but that flag "shall not be used in brands
/// or compatible brands earlier than 'iso5'" (clause 8.8.7.1), and the
/// files are `'isom'` or `'iso2'` (see each codec's brands). A `base_data_offset`
/// given, from the start of the file, would tie each fragment to where it
/// lies in it.
///
/// Spec reading: the `TrackFragmentBaseMediaDecodeTimeBox` is among the
/// boxes `'iso6'` requires readers to support (Table E.6), not `'isom'`,
/// as the pre-roll's are of `'avc1'`, and is written for the same reasons
/// (see AAC's brands): a reader skips it if it does not
/// recognize it, and finds the time by the frames before, as it would
/// without it. It comes "after the TrackFragmentHeaderBox and before the
/// first TrackRunBox" (clause 8.8.12.1).
fn fragment(info: &FragmentInfo, run: &Run, media_data: &[u8]) -> Vec<u8> {
    let media_data_header = boxes::media_data_header(media_data.len() as u64);
    let common_size = common_size(run.sizes);
    let mut out = Vec::new();
    let mut data_offset_at = 0;
    write_box(&mut out, b"moof", |out| {
        write_full_box(out, b"mfhd", 0, 0, |out| {
            out.extend_from_slice(&info.sequence_number.to_be_bytes());
        });
        write_box(out, b"traf", |out| {
            // The TrackFragmentHeaderBox (clause 8.8.7), with the frames'
            // one size where they have one, default-sample-size-present,
            // as the SampleSizeBox of the file without fragments has. The
            // other defaults are the TrackExtendsBox's.
            let flags = if common_size.is_some() { 0x00_0010 } else { 0 };
            write_full_box(out, b"tfhd", 0, flags, |out| {
                out.extend_from_slice(&TRACK_ID.to_be_bytes());
                if let Some(size) = common_size {
                    out.extend_from_slice(&size.to_be_bytes()); // default_sample_size
                }
            });
            let version = version(&[info.decode_time]);
            write_full_box(out, b"tfdt", version, 0, |out| {
                put_sized(out, version, info.decode_time); // baseMediaDecodeTime
            });
            // The TrackRunBox (clause 8.8.8) of every frame: of version 0,
            // which 'isom' requires readers to support (clause E.2), with
            // data-offset-present, 0x1; sample-duration-present, 0x100,
            // where a frame's duration is not the TrackExtendsBox's
            // default_sample_duration, as the stream's last can be shorter
            // (see Fragmenter::finish); and sample-size-present, 0x200,
            // where the frames' sizes differ.
            let each_duration = (run.durations.iter()).any(|&d| d != run.default_duration);
            let mut flags = 0x00_0001;
            if each_duration {
                flags |= 0x00_0100;
            }
            if common_size.is_none() {
                flags |= 0x00_0200;
            }
            write_full_box(out, b"trun", 0, flags, |out| {
                out.extend_from_slice(&info.frames.to_be_bytes()); // sample_count
                data_offset_at = out.len();
                out.extend_from_slice(&0i32.to_be_bytes()); // data_offset
                for (&size, &duration) in run.sizes.iter().zip(run.durations) {
                    if each_duration {
                        out.extend_from_slice(&duration.to_be_bytes()); // sample_duration
                    }
                    if common_size.is_none() {
                        out.extend_from_slice(&size.to_be_bytes()); // sample_size
                    }
                }
            });
            // Each track fragment maps its own frames: the "at most one
            // instance of this box with a particular grouping_type" (clause
            // 8.9.2.1) is one in each container, as clause 8.9.4 has a
            // SampleToGroupBox in each TrackFragmentBox document "the total
            // number of samples in all the track fragment runs".
            if run.roll {
                write_roll_mapping(out, info.frames);
            }
        });
    });
    let data_offset = i32::try_from(out.len() + media_data_header.len())
        .expect("MAX_FRAMES keeps a MovieFragmentBox under 2 GiB");
    out[data_offset_at..data_offset_at + 4].copy_from_slice(&data_offset.to_be_bytes());
    out.extend(media_data_header);
    out.extend_from_slice(media_data);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boxes::read::{FourCcs, boxes, children, find, u32_at};
    use crate::movie::tests::track;

    fn fragmenter(fragment_samples: u32) -> Fragmenter {
        Fragmenter::new(&track(44_100), NonZeroU32::new(fragment_samples).unwrap())
    }

    #[test]
    fn fragments_of_a_second_keep_to_the_second() {
        let mut fragmenter = fragmenter(44_100);
        let mut ends = Vec::new();
        for i in 0..44 * 60 {
            if let Some(fragment) = fragmenter.push_frame(&[0], 1024) {
                ends.push(i + 1);
                assert_eq!(
                    fragment.info.decode_time + fragment.info.duration,
                    (i + 1) * 1024
                );
            }
        }
        // Each ends at the frame nearest its second.
        for (n, &end) in ends.iter().enumerate() {
            let second = (n as f64 + 1.0) * 44_100.0 / 1024.0;
            assert!((end as f64 - second).abs() <= 0.5, "{n}: {end}");
        }
        let lengths: Vec<u64> = ends.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(lengths.iter().all(|&l| l == 43 || l == 44));
        assert!(lengths.contains(&44));
    }

    #[test]
    fn a_fragment_has_a_frame_at_least() {
        let mut fragmenter = fragmenter(1);
        for i in 0..5 {
            let fragment = fragmenter
                .push_frame(&[i], 1024)
                .expect("a fragment a frame");
            assert_eq!(fragment.info.frames, 1);
            assert_eq!(fragment.info.sequence_number, u32::from(i) + 1);
            assert_eq!(fragment.info.decode_time, 1024 * u64::from(i));
        }
        assert!(fragmenter.flush().is_none());
    }

    /// Frames of a sample each in fragments of the longest duration, as no
    /// AAC stream has them, end their fragment at MAX_FRAMES.
    #[test]
    fn a_fragment_has_max_frames_at_most() {
        let mut one = track(44_100);
        one.default_duration = 1;
        let mut fragmenter = Fragmenter::new(&one, NonZeroU32::MAX);
        for _ in 1..MAX_FRAMES {
            assert!(fragmenter.push_frame(&[], 1).is_none());
        }
        let fragment = fragmenter.push_frame(&[], 1).expect("a full fragment");
        assert_eq!(fragment.info.frames as usize, MAX_FRAMES);
        let trun = find(&fragment.bytes, &[b"moof", b"traf", b"trun"]);
        assert_eq!(u32_at(trun.payload, 4) as usize, MAX_FRAMES);
    }

    #[test]
    fn a_flush_ends_the_fragment_and_the_next_keeps_to_the_boundaries() {
        let mut fragmenter = fragmenter(10 * 1024);
        for _ in 0..3 {
            assert!(fragmenter.push_frame(&[0], 1024).is_none());
        }
        let first = fragmenter.flush().unwrap();
        assert_eq!((first.info.frames, first.info.duration), (3, 3 * 1024));
        assert!(fragmenter.flush().is_none());
        let ends: Vec<u64> = (0..30)
            .filter_map(|_| fragmenter.push_frame(&[0], 1024))
            .map(|f| (f.info.decode_time + f.info.duration) / 1024)
            .collect();
        assert_eq!(ends, [10, 20, 30]);
    }

    /// The frames, of `sizes`, with byte `i` of frame `i`.
    fn frames(sizes: &[u32]) -> Vec<Vec<u8>> {
        (sizes.iter().enumerate())
            .map(|(i, &size)| vec![i as u8; size as usize])
            .collect()
    }

    /// The fragment of the frames of `sizes`, written alone.
    fn one_fragment(sizes: &[u32]) -> Fragment {
        let mut fragmenter = fragmenter(u32::MAX);
        for frame in frames(sizes) {
            assert!(fragmenter.push_frame(&frame, 1024).is_none());
        }
        fragmenter.flush().unwrap()
    }

    #[test]
    fn a_fragment_is_moof_mdat_and_its_run_finds_the_frames() {
        let sizes = [3, 1, 2];
        let fragment = one_fragment(&sizes);
        let bytes = &fragment.bytes;
        assert_eq!(boxes(bytes, 0).kinds(), ["moof", "mdat"]);
        assert_eq!(find(bytes, &[b"mdat"]).payload, frames(&sizes).concat());
        assert_eq!(children(&find(bytes, &[b"moof"])).kinds(), ["mfhd", "traf"]);
        assert_eq!(
            find(bytes, &[b"moof", b"mfhd"]).payload,
            [0, 0, 0, 0, 0, 0, 0, 1]
        );
        assert_eq!(
            children(&find(bytes, &[b"moof", b"traf"])).kinds(),
            ["tfhd", "tfdt", "trun", "sbgp"]
        );
        let tfhd = find(bytes, &[b"moof", b"traf", b"tfhd"]);
        assert_eq!(tfhd.payload, [0, 0, 0, 0, 0, 0, 0, 1]);
        let tfdt = find(bytes, &[b"moof", b"traf", b"tfdt"]);
        assert_eq!(tfdt.payload, [0; 8]);
        let trun = find(bytes, &[b"moof", b"traf", b"trun"]);
        assert_eq!(u32_at(trun.payload, 0), 0x00_0201); // version, flags
        assert_eq!(u32_at(trun.payload, 4), 3); // sample_count
        let offset = u32_at(trun.payload, 8) as usize;
        assert_eq!(&bytes[offset..], frames(&sizes).concat());
        let fields: Vec<u32> = (12..trun.payload.len())
            .step_by(4)
            .map(|at| u32_at(trun.payload, at))
            .collect();
        assert_eq!(fields, sizes);
        let sbgp = find(bytes, &[b"moof", b"traf", b"sbgp"]);
        let expected = [
            &[0; 4][..],
            b"roll",
            &[0, 0, 0, 1],
            &[0, 0, 0, 3],
            &[0, 0, 0, 1],
        ];
        assert_eq!(sbgp.payload, expected.concat());
    }

    #[test]
    fn frames_of_one_size_take_the_default_size() {
        let fragment = one_fragment(&[5; 4]);
        let bytes = &fragment.bytes;
        let tfhd = find(bytes, &[b"moof", b"traf", b"tfhd"]);
        assert_eq!(tfhd.payload, [0, 0, 0, 0x10, 0, 0, 0, 1, 0, 0, 0, 5]);
        let trun = find(bytes, &[b"moof", b"traf", b"trun"]);
        assert_eq!(trun.payload.len(), 12);
        assert_eq!(u32_at(trun.payload, 0), 0x00_0001);
        let offset = u32_at(trun.payload, 8) as usize;
        assert_eq!(&bytes[offset..], frames(&[5; 4]).concat());
    }

    /// The fields of `trun` after its version and flags, as 32-bit words.
    fn words(trun: &[u8]) -> Vec<u32> {
        (4..trun.len())
            .step_by(4)
            .map(|at| u32_at(trun, at))
            .collect()
    }

    #[test]
    fn the_last_frame_lasts_to_the_end_of_the_input() {
        // Frames of sizes 3, 1 and 2, the last lasting 636 samples.
        let mut three = fragmenter(u32::MAX);
        let sizes = [3, 1, 2];
        let frames = frames(&sizes);
        for frame in &frames[..2] {
            assert!(three.push_frame(frame, 1024).is_none());
        }
        let fragment = three.finish(&frames[2], 636);
        assert_eq!(fragment.info.duration, 2 * 1024 + 636);
        let bytes = &fragment.bytes;
        let trun = find(bytes, &[b"moof", b"traf", b"trun"]);
        assert_eq!(u32_at(trun.payload, 0), 0x00_0301);
        let offset = u32_at(trun.payload, 8) as usize;
        assert_eq!(&bytes[offset..], frames.concat());
        // sample_count, data_offset, then each frame's duration and size.
        let rows = &words(trun.payload)[2..];
        assert_eq!(rows, [1024, 3, 1024, 1, 636, 2]);
        // Of one size, the durations alone; a frame alone, its own.
        let mut alike = fragmenter(u32::MAX);
        assert!(alike.push_frame(&[0; 5], 1024).is_none());
        let fragment = alike.finish(&[0; 5], 1);
        let trun = find(&fragment.bytes, &[b"moof", b"traf", b"trun"]);
        assert_eq!(u32_at(trun.payload, 0), 0x00_0101);
        assert_eq!(&words(trun.payload)[2..], [1024, 1]);
        let fragment = alike.finish(&[0; 5], 1024);
        let trun = find(&fragment.bytes, &[b"moof", b"traf", b"trun"]);
        assert_eq!(u32_at(trun.payload, 0), 0x00_0001);
        assert_eq!(fragment.info.duration, 1024);
    }

    #[test]
    #[should_panic(expected = "a last frame of 0 samples")]
    fn a_last_frame_lasts_a_sample_at_least() {
        fragmenter(1024).finish(&[0], 0);
    }

    #[test]
    fn decode_times_past_32_bits_take_version_1() {
        let mut fragmenter = fragmenter(1024);
        fragmenter.time_written = ((1 << 22) - 1) * 1024;
        let before = fragmenter.push_frame(&[0], 1024).unwrap();
        let tfdt = find(&before.bytes, &[b"moof", b"traf", b"tfdt"]);
        assert_eq!(
            tfdt.payload,
            [&[0; 4][..], &(u32::MAX - 1023).to_be_bytes()].concat()
        );
        let after = fragmenter.push_frame(&[0], 1024).unwrap();
        let tfdt = find(&after.bytes, &[b"moof", b"traf", b"tfdt"]);
        assert_eq!(
            tfdt.payload,
            [&[1, 0, 0, 0][..], &(1u64 << 32).to_be_bytes()].concat()
        );
    }

    #[test]
    fn the_init_segment_describes_no_samples_and_extends_the_movie() {
        let init = init_segment(&track(48_000), Some(5000));
        assert_eq!(boxes(&init, 0).kinds(), ["ftyp", "moov"]);
        assert_eq!(
            children(&find(&init, &[b"moov"])).kinds(),
            ["mvhd", "trak", "mvex"]
        );
        let stbl = [b"moov", b"trak", b"mdia", b"minf", b"stbl"];
        assert_eq!(
            children(&find(&init, &stbl)).kinds(),
            ["stsd", "stts", "stsc", "stsz", "stco", "sgpd"]
        );
        let table = |kind: &[u8; 4]| find(&init, &[&stbl[..], &[kind]].concat()).payload;
        assert_eq!(table(b"stts"), [0; 8]);
        assert_eq!(table(b"stsc"), [0; 8]);
        assert_eq!(table(b"stsz"), [0; 12]);
        assert_eq!(table(b"stco"), [0; 8]);
        let sgpd = [
            &[1, 0, 0, 1][..],
            b"roll",
            &[0, 0, 0, 2],
            &[0, 0, 0, 1],
            &[0xff, 0xff],
        ];
        assert_eq!(table(b"sgpd"), sgpd.concat());
        assert_eq!(
            children(&find(&init, &[b"moov", b"mvex"])).kinds(),
            ["mehd", "trex"]
        );
        let mehd = find(&init, &[b"moov", b"mvex", b"mehd"]);
        assert_eq!(mehd.payload, [0, 0, 0, 0, 0, 0, 0x13, 0x88]);
        let trex = find(&init, &[b"moov", b"mvex", b"trex"]);
        let expected = [
            [0; 4],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 4, 0],
            [0; 4],
            [0x0a, 0x20, 0, 0],
        ];
        assert_eq!(trex.payload, expected.concat());
    }

    #[test]
    fn a_known_length_is_the_edit_s_and_the_extended_movie_s_duration() {
        let durations = |init: &[u8]| {
            let mvhd = find(init, &[b"moov", b"mvhd"]);
            let tkhd = find(init, &[b"moov", b"trak", b"tkhd"]);
            let elst = find(init, &[b"moov", b"trak", b"edts", b"elst"]);
            let mdhd = find(init, &[b"moov", b"trak", b"mdia", b"mdhd"]);
            [
                u32_at(mvhd.payload, 16),
                u32_at(tkhd.payload, 20),
                u32_at(elst.payload, 8),
                u32_at(elst.payload, 12), // media_time
                u32_at(mdhd.payload, 16),
            ]
        };
        // The movie's, the track's, the edit's and the media's durations,
        // and the edit's media_time.
        let known = init_segment(&track(44_100), Some(3000));
        assert_eq!(durations(&known), [0, 0, 3000, 1024, 0]);
        let mehd = find(&known, &[b"moov", b"mvex", b"mehd"]);
        assert_eq!(mehd.payload, [0, 0, 0, 0, 0, 0, 0x0b, 0xb8]);
        let unknown = init_segment(&track(44_100), None);
        assert_eq!(durations(&unknown), [0, 0, 0, 1024, 0]);
        assert_eq!(
            children(&find(&unknown, &[b"moov", b"mvex"])).kinds(),
            ["trex"]
        );
        let long = init_segment(&track(44_100), Some(1 << 32));
        let mehd = find(&long, &[b"moov", b"mvex", b"mehd"]);
        assert_eq!(
            mehd.payload,
            [&[1, 0, 0, 0][..], &(1u64 << 32).to_be_bytes()].concat()
        );
    }
}
