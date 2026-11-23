// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Encodes a sweep followed by silence as MP3 at a variable bitrate, as it
//! is generated, with a Xing/Info tag that gives players its length.
//!
//! ```sh
//! cargo run --example mp3_vbr -- sweep.mp3
//! ```

use std::error::Error;
use std::f64::consts::TAU;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};

use chelys::mp3::{ChannelMode, Config, Encoder, Rate};

const SAMPLE_RATE: u32 = 44_100;

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "sweep.mp3".into());

    let rate = Rate::Quality {
        // From 0 to 100, higher for better quality and larger frames.
        level: 80,
        min_kbps: None,
        max_kbps: None,
    };
    // The tag is written over the start of the stream once it has ended, so
    // ask for it only where the output can be rewritten, as a file can.
    let config = Config::new(SAMPLE_RATE, ChannelMode::Mono, rate).info_tag(true);
    let mut encoder = Encoder::new(config)?;
    let mut file = File::create(&path)?;

    // Four seconds sweeping from 100 Hz to 10 kHz, then two of silence,
    // encoded a tenth of a second at a time.
    let sweep = 4 * SAMPLE_RATE as usize;
    let total = 6 * SAMPLE_RATE as usize;
    let mut phase = 0.0;
    let mut chunk = Vec::new();
    let mut mp3 = Vec::new();
    for start in (0..total).step_by(SAMPLE_RATE as usize / 10) {
        chunk.clear();
        for n in start..(start + SAMPLE_RATE as usize / 10).min(total) {
            if n < sweep {
                let frequency = 100.0 * 100f64.powf(n as f64 / sweep as f64);
                phase = (phase + TAU * frequency / f64::from(SAMPLE_RATE)) % TAU;
                chunk.push(0.5 * phase.sin());
            } else {
                chunk.push(0.0);
            }
        }
        // Each call appends the bytes that are final; write them out and
        // reuse the buffer.
        encoder.encode_interleaved(&chunk, &mut mp3)?;
        file.write_all(&mp3)?;
        mp3.clear();
    }

    let finished = encoder.finish(&mut mp3);
    file.write_all(&mp3)?;
    if let Some(tag) = finished.info_tag {
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&tag)?;
    }

    let bytes = file.metadata()?.len();
    let seconds = finished.samples as f64 / f64::from(SAMPLE_RATE);
    println!(
        "wrote {bytes} bytes to {path}: {} frames, {seconds} s, {:.1} kbps on average",
        finished.frames,
        bytes as f64 * 8.0 / seconds / 1000.0,
    );
    Ok(())
}
