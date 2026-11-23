// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The file written whole once the stream has ended: its `FileTypeBox`,
//! its `MovieBox`, whose sample tables describe every frame, and one
//! `MediaDataBox`, which holds the frames one after another.

use std::io::{self, Write};

use super::{
    AudioTrack, Durations, count, file_type_box, movie_box, write_roll_description,
    write_roll_mapping, write_sample_tables,
};
use crate::boxes;

/// Writes the file of `track`, whose frames, of `sizes`, `media_data`
/// holds, one after another, and which decode to `length` samples after
/// the track's delay, to `out`: its `FileTypeBox`, then its `MovieBox`,
/// then its `MediaDataBox`.
///
/// The `MovieBox` comes before the media data, so that a reader that reads
/// the file from its start, as from a network, can play it as it arrives;
/// the file is written once, so it need not be seekable. The box's size
/// then sets where the media data starts, which its `ChunkOffsetBox`
/// gives: the box is built once to find its size, and again with the
/// offset, which a 32-bit `chunk_offset` always holds, as it is within the
/// `MovieBox`'s 32-bit size of the file's start. A later box in the
/// `MovieBox`, such as a `UserDataBox` with song tags, would only move the
/// offset.
///
/// The `MovieBox`'s durations are the input's `length`, which its edit
/// presents, and the media's, "the largest composition timestamp plus the
/// duration of that sample" (clause 8.4.2.3), that of all the frames.
///
/// Spec reading: an input of no samples is coded in one frame, as the
/// stream of an encoder that ends at once, and its edit has an
/// `edit_duration` of 0. The standard is inconsistent on what that means.
/// Clause 8.6.6.1 reads a last edit's duration of 0 as that of all the
/// samples only "When a movie is fragmented, and does not contain any
/// MovieExtendsHeaderBox", which the file is not, so that the edit lasts
/// no time; but the example in clause 8.6.6.4 maps "the media time from 20
/// onwards" with an `edit_duration` of 0 in a movie that it does not say is
/// fragmented. Either way the edit presents nothing here: the one frame's
/// media ends at the edit's `media_time`, as AAC's delay is a frame's
/// length, so that nothing follows it. The alternatives were weighed: no edit would present the
/// frame's every sample; an empty edit alone is not allowed, as "The last
/// edit in a track shall never be an empty edit"; and a track of no
/// samples, which needs no edit, would make empty input the one case in
/// which the file does not hold the frames the encoder reports
/// ([`Finished::frames`](chelys_aac::Finished::frames)). Only decoding
/// shows which reading a reader takes.
///
/// Settled by decoding with ffmpeg (black box): it decodes a file of no
/// samples to none, and files of a sample or so either side of a frame's
/// length to exactly their samples; the conformance test is
/// `ffmpeg_reads_empty_and_short_mp4_files`.
///
/// # Panics
///
/// If the `MovieBox` reaches 4 GiB, which a `ChunkLargeOffsetBox` would
/// not mend, as the box's own compact size could not hold it: about a
/// billion frames, whose sizes take 4 bytes each, months of audio. The
/// panic is left in place of handling a case no music reaches.
///
/// # Errors
///
/// If `out` fails.
pub fn write_file<W: Write + ?Sized>(
    track: &AudioTrack,
    sizes: &[u32],
    length: u64,
    media_data: &[u8],
    out: &mut W,
) -> io::Result<()> {
    let file_type = file_type_box();
    let media_data_header = boxes::media_data_header(media_data.len() as u64);
    let durations = Durations {
        edit: length,
        media: sizes.len() as u64 * u64::from(track.frame_samples),
    };
    let movie = |chunk_offset: u32| {
        let sample_tables = |out: &mut Vec<u8>| {
            write_sample_tables(out, track, sizes, chunk_offset);
            write_roll_mapping(out, count(sizes));
            // Its flags are 0, as the two flags it may have in a
            // SampleTableBox concern movie fragments, which the file has
            // none of.
            write_roll_description(out, track, 0);
        };
        movie_box(track, durations, sample_tables, |_| {})
    };
    let movie_len = movie(0).len();
    let offset = file_type.len() + movie_len + media_data_header.len();
    let movie = movie(u32::try_from(offset).expect("a MovieBox under 4 GiB"));
    debug_assert_eq!(movie.len(), movie_len);
    out.write_all(&file_type)?;
    out.write_all(&movie)?;
    out.write_all(&media_data_header)?;
    out.write_all(media_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boxes::read::{FourCcs, boxes, find, u32_at};
    use crate::movie::tests::track;

    fn write(track: &AudioTrack, sizes: &[u32], length: u64, media_data: &[u8]) -> Vec<u8> {
        let mut file = Vec::new();
        write_file(track, sizes, length, media_data, &mut file).unwrap();
        file
    }

    #[test]
    fn the_file_is_ftyp_moov_mdat() {
        let file = write(&track(44_100), &[3, 1, 2], 2000, &[1, 2, 3, 4, 5, 6]);
        assert_eq!(boxes(&file, 0).kinds(), ["ftyp", "moov", "mdat"]);
        let ftyp = find(&file, &[b"ftyp"]);
        assert_eq!(ftyp.payload, b"mp42\0\0\0\0mp42isom");
        let mdat = find(&file, &[b"mdat"]);
        assert_eq!(mdat.payload, [1, 2, 3, 4, 5, 6]);
        assert_eq!(mdat.start + 8, file.len() - 6);
    }

    #[test]
    fn the_chunk_offset_is_where_the_frames_start() {
        let file = write(&track(44_100), &[3, 1, 2], 2000, &[1, 2, 3, 4, 5, 6]);
        let path = [b"moov", b"trak", b"mdia", b"minf", b"stbl"];
        let stco = find(&file, &[&path[..], &[b"stco"]].concat());
        assert_eq!(u32_at(stco.payload, 4), 1);
        let offset = u32_at(stco.payload, 8) as usize;
        assert_eq!(&file[offset..], [1, 2, 3, 4, 5, 6]);
        let stsc = find(&file, &[&path[..], &[b"stsc"]].concat());
        assert_eq!(
            &stsc.payload[4..],
            [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 3], [0, 0, 0, 1]].concat()
        );
        let stsz = find(&file, &[&path[..], &[b"stsz"]].concat());
        let fields: Vec<u32> = (4..stsz.payload.len())
            .step_by(4)
            .map(|at| u32_at(stsz.payload, at))
            .collect();
        assert_eq!(fields, [0, 3, 3, 1, 2]);
        let stts = find(&file, &[&path[..], &[b"stts"]].concat());
        assert_eq!(
            &stts.payload[4..],
            [[0, 0, 0, 1], [0, 0, 0, 3], [0, 0, 4, 0]].concat()
        );
    }

    #[test]
    fn frames_of_one_size_have_one_sample_size() {
        let file = write(&track(48_000), &[7; 5], 4000, &[0; 35]);
        let path = [b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsz"];
        assert_eq!(
            find(&file, &path).payload,
            [[0; 4], [0, 0, 0, 7], [0, 0, 0, 5]].concat()
        );
    }

    #[test]
    fn every_frame_rolls_back_one() {
        let stbl = [b"moov", b"trak", b"mdia", b"minf", b"stbl"];
        for (frames, expected) in [(1, vec![(1, 1)]), (2, vec![(2, 1)]), (9, vec![(9, 1)])] {
            let sizes = vec![1; frames];
            let file = write(&track(44_100), &sizes, 0, &vec![0; frames]);
            let sbgp = find(&file, &[&stbl[..], &[b"sbgp"]].concat());
            assert_eq!(&sbgp.payload[..8], b"\0\0\0\0roll");
            let entries: Vec<(u32, u32)> = (12..sbgp.payload.len())
                .step_by(8)
                .map(|at| (u32_at(sbgp.payload, at), u32_at(sbgp.payload, at + 4)))
                .collect();
            assert_eq!(u32_at(sbgp.payload, 8) as usize, expected.len());
            assert_eq!(entries, expected);
            let sgpd = find(&file, &[&stbl[..], &[b"sgpd"]].concat());
            let expected = [
                &[1, 0, 0, 0][..],
                b"roll",
                &[0, 0, 0, 2],
                &[0, 0, 0, 1],
                &[0xff, 0xff],
            ]
            .concat();
            assert_eq!(sgpd.payload, expected);
        }
    }

    #[test]
    fn the_sample_table_follows_the_recommended_order() {
        let file = write(&track(44_100), &[1, 2], 100, &[0; 3]);
        let stbl = find(&file, &[b"moov", b"trak", b"mdia", b"minf", b"stbl"]);
        assert_eq!(
            boxes(stbl.payload, 0).kinds(),
            ["stsd", "stts", "stsc", "stsz", "stco", "sbgp", "sgpd"]
        );
    }
}
