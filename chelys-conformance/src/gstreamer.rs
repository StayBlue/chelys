// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! GStreamer, as a black-box reader of MP4 files, run as `gst-launch-1.0`:
//! its own MP4 demuxer, `qtdemux`, then for AAC gst-libav's `avdec_aac`,
//! which wraps ffmpeg's AAC decoder, so that what it shows is GStreamer's
//! reading of the file, not an independent decoder's; for FLAC
//! gst-plugins-good's `flacdec`, which wraps libFLAC; for Opus
//! gst-plugins-base's `opusdec`, which wraps libopus; and for ALAC
//! gst-libav's `avdec_alac`, which wraps ffmpeg's ALAC decoder, as
//! `avdec_aac` does its AAC decoder.
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

/// Decodes the first audio stream of an MP4 file of AAC to 32-bit float
/// PCM as a player's sink would take it: through `qtdemux`, `avdec_aac`,
/// `audiosegmentclip`, which clips the samples to the segment `qtdemux`
/// marks, and `audioconvert`.
pub fn decode_mp4(file: &[u8]) -> Decoded {
    decode_mp4_with(file, "avdec_aac")
}

/// Decodes the first audio stream of an MP4 file of Opus as
/// [`decode_mp4`] does AAC, through gst-plugins-base's `opusdec`, which
/// wraps libopus, the reference decoder.
pub fn decode_opus_mp4(file: &[u8]) -> Decoded {
    decode_mp4_with(file, "opusdec")
}

/// Decodes the first audio stream of an MP4 file of FLAC as
/// [`decode_mp4`] does AAC, through gst-plugins-good's `flacdec`, which
/// wraps libFLAC, the reference decoder.
pub fn decode_flac_mp4(file: &[u8]) -> Decoded {
    decode_mp4_with(file, "flacdec")
}

/// Decodes the first audio stream of an MP4 file of ALAC as
/// [`decode_mp4`] does AAC, through gst-libav's `avdec_alac`, which wraps
/// ffmpeg's ALAC decoder, at `sample_rate`: the samples are taken at that
/// rate only, which `audioconvert` does not change, so that a stream that
/// GStreamer decodes at another fails to negotiate, an error.
pub fn decode_alac_mp4(file: &[u8], sample_rate: u32) -> Decoded {
    decode_mp4_at(file, "avdec_alac", Some(sample_rate))
}

fn decode_mp4_with(file: &[u8], decoder: &str) -> Decoded {
    decode_mp4_at(file, decoder, None)
}

fn decode_mp4_at(file: &[u8], decoder: &str, sample_rate: Option<u32>) -> Decoded {
    let file = TempFile::new("mp4", file);
    let output = TempFile::new("f32", &[]);
    let rate = sample_rate.map_or(String::new(), |rate| format!(",rate={rate}"));
    let pipeline = format!(
        "filesrc location={} ! qtdemux ! {decoder} ! audiosegmentclip ! audioconvert ! \
         audio/x-raw,format=F32LE,layout=interleaved{rate} ! filesink location={}",
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
