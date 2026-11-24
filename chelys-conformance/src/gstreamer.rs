// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! GStreamer, as a black-box reader of MP4 files, run as `gst-launch-1.0`:
//! its own MP4 demuxer, `qtdemux`, then gst-libav's `avdec_aac`, which
//! wraps ffmpeg's AAC decoder, so that what it shows is GStreamer's reading
//! of the file, not an independent decoder's.
//!
//! Observed of GStreamer 1.28.9 as a black box:
//!
//! - `qtdemux` marks the edit list as a segment, from the edit's
//!   `media_time` to its end, which `audiosegmentclip` clips the decoded
//!   samples to, as a player's sink would; without it nothing is clipped.
//! - Its clipped output can differ from the edit list by a sample at
//!   either end: at 44.1 kHz it starts one sample earlier, in files written
//!   whole and fragmented alike, and at 48 kHz the segment of a file
//!   written whole can end a nanosecond short of the input's last sample,
//!   which it then leaves out (see
//!   `gstreamer_decodes_fragmented_files_of_a_declared_length_as_the_files_written_whole`).
//!   So the fragmented files are compared with the files written whole,
//!   both as GStreamer decodes them.
//! - With `-q`, `gst-launch-1.0` writes only errors and warnings, which are
//!   kept as the diagnostics. Its debug log (`GST_DEBUG`) is never used.

use std::process::Command;

use crate::{Decoded, TempFile, run};

/// Decodes the first audio stream of an MP4 file to 32-bit float PCM as a
/// player's sink would take it: through `qtdemux`, `avdec_aac`,
/// `audiosegmentclip`, which clips the samples to the segment `qtdemux`
/// marks, and `audioconvert`.
pub fn decode_mp4(file: &[u8]) -> Decoded {
    let file = TempFile::new("mp4", file);
    let output = TempFile::new("f32", &[]);
    let pipeline = format!(
        "filesrc location={} ! qtdemux ! avdec_aac ! audiosegmentclip ! audioconvert ! \
         audio/x-raw,format=F32LE,layout=interleaved ! filesink location={}",
        file.path().display(),
        output.path().display()
    );
    let result = run(&mut gst_launch(&["-q"], &pipeline));
    let bytes = std::fs::read(output.path()).unwrap();
    Decoded {
        samples: bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        diagnostics: [result.stdout, result.stderr]
            .iter()
            .map(|text| String::from_utf8_lossy(text).into_owned())
            .collect(),
    }
}

/// A segment that `qtdemux` marks on its audio, in nanoseconds of the
/// track's media time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    pub start: u64,
    /// `None` where the segment has no end.
    pub stop: Option<u64>,
}

/// The segments that `qtdemux` marks on the first audio stream of an MP4
/// file, in order, as the segment events that a `fakesink` reports of
/// itself with `gst-launch-1.0 -v`.
pub fn segments_mp4(file: &[u8]) -> Vec<Segment> {
    let file = TempFile::new("mp4", file);
    let pipeline = format!(
        "filesrc location={} ! qtdemux ! fakesink silent=false",
        file.path().display()
    );
    let output = run(&mut gst_launch(&["-v"], &pipeline));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let field = |line: &str, name: &str| -> u64 {
        let key = format!("{name}=(guint64)");
        let at = line
            .find(&key)
            .unwrap_or_else(|| panic!("no {name} in {line}"))
            + key.len();
        let digits: String = line[at..]
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        digits.parse().unwrap()
    };
    stdout
        .lines()
        .filter(|line| line.contains("type: segment"))
        .map(|line| Segment {
            start: field(line, "start"),
            stop: Some(field(line, "stop")).filter(|&stop| stop != u64::MAX),
        })
        .collect()
}

/// `gst-launch-1.0` with `options`, running `pipeline`, its debug log off.
fn gst_launch(options: &[&str], pipeline: &str) -> Command {
    let mut command = Command::new("gst-launch-1.0");
    command
        .env_remove("GST_DEBUG")
        .args(options)
        .args(pipeline.split_whitespace());
    command
}
