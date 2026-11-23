// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Encodes three seconds of a tone, 440 Hz on the left and 660 Hz on the
//! right, as MP3 at a constant 128 kbps.
//!
//! ```sh
//! cargo run --example mp3_tone -- tone.mp3
//! ```

use std::error::Error;
use std::f32::consts::TAU;

use chelys::mp3::{ChannelMode, Config, Encoder, Rate};

const SAMPLE_RATE: u32 = 44_100;

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "tone.mp3".into());

    let config = Config::new(
        SAMPLE_RATE,
        ChannelMode::Stereo,
        Rate::Constant { kbps: 128 },
    );
    let mut encoder = Encoder::new(config)?;

    // Interleaved: a sample of the left channel, then one of the right.
    let pcm: Vec<f32> = (0..3 * SAMPLE_RATE)
        .flat_map(|n| {
            let t = n as f32 / SAMPLE_RATE as f32;
            [0.5 * (TAU * 440.0 * t).sin(), 0.5 * (TAU * 660.0 * t).sin()]
        })
        .collect();

    let mut mp3 = Vec::new();
    encoder.encode_interleaved(&pcm, &mut mp3)?;
    // The stream is complete only once the encoder has finished it.
    encoder.finish(&mut mp3);

    std::fs::write(&path, &mp3)?;
    println!("wrote {} bytes to {path}", mp3.len());
    Ok(())
}
