// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! PEAQ (ITU-R BS.1387), the basic version, as GstPEAQ computes it, through
//! `gst-launch-1.0`, with `ffmpeg` to prepare its input.

use std::process::Command;

use crate::{TempFile, run};

/// The sampling frequency PEAQ is defined at, and the only one GstPEAQ takes.
const PEAQ_RATE: u32 = 48_000;

/// The spacing in Hz of the lines of PEAQ's FFT, in which it gives
/// bandwidths: 2048 lines at 48 kHz.
pub const LINE_HZ: f64 = PEAQ_RATE as f64 / 2048.0;

/// What GstPEAQ reports of a test signal against its reference.
#[derive(Clone, Copy, Debug)]
pub struct Grade {
    /// The Objective Difference Grade: 0 for an imperceptible difference,
    /// down to -4 for a very annoying one. A test identical to its reference
    /// grades about +0.2.
    pub odg: f64,
    /// The mean bandwidth of the reference and of the test, in lines of
    /// [`LINE_HZ`]. A lower test bandwidth weighs heavily in the grade,
    /// though listeners may not hear content as high as the reference's.
    pub bandwidth_reference: f64,
    pub bandwidth_test: f64,
}

/// Grades `test` against `reference`, both interleaved PCM of `channels`
/// at `sample_rate`. Both are resampled to 48 kHz and given to GstPEAQ as
/// 16-bit PCM, without dither, so that the test clips where the coding
/// noise takes it past full scale, as 16-bit playback would. With float
/// input, GstPEAQ finds no noise floor against which to measure the test's
/// bandwidth, and does not see lost upper bands.
///
/// # Panics
///
/// If `ffmpeg` or `gst-launch-1.0` cannot be run or fails, if GstPEAQ is
/// not installed, or if its output lacks a measure.
pub fn grade(reference: &[f32], test: &[f32], sample_rate: u32, channels: usize) -> Grade {
    let reference = pcm16_at_48k(reference, sample_rate, channels);
    let test = pcm16_at_48k(test, sample_rate, channels);
    let source = |file: &TempFile| {
        [
            "filesrc".to_owned(),
            format!("location={}", file.path().display()),
        ]
        .into_iter()
        .chain(["!", "wavparse", "!", "audioconvert", "!"].map(str::to_owned))
    };
    let mut command = Command::new("gst-launch-1.0");
    command
        .arg("-q")
        .args(source(&reference))
        .args(["peaq", "name=p", "console-output=true"])
        .args(source(&test))
        .arg("p.test");
    let output = run(&mut command);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let measure = |name: &str| -> f64 {
        stdout
            .lines()
            .find_map(|line| line.trim().strip_prefix(name)?.strip_prefix(':'))
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or_else(|| panic!("GstPEAQ did not report {name}: {stdout}"))
    };
    Grade {
        odg: measure("Objective Difference Grade"),
        bandwidth_reference: measure("BandwidthRefB"),
        bandwidth_test: measure("BandwidthTestB"),
    }
}

/// `samples` as a 16-bit WAV file at 48 kHz.
fn pcm16_at_48k(samples: &[f32], sample_rate: u32, channels: usize) -> TempFile {
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    let input = TempFile::new("f32", &bytes);
    let output = TempFile::new("wav", &[]);
    run(Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-y", "-f", "f32le"])
        .args([
            "-ar",
            &sample_rate.to_string(),
            "-ac",
            &channels.to_string(),
        ])
        .arg("-i")
        .arg(input.path())
        .arg("-af")
        .arg(format!("aresample={PEAQ_RATE}:resampler=soxr:precision=28"))
        .args(["-c:a", "pcm_s16le"])
        .arg(output.path()));
    output
}
