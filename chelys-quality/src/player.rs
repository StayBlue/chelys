// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Plays PCM through ffplay, one sound at a time.

use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::thread::JoinHandle;

pub struct Player {
    sample_rate: u32,
    channels: usize,
    /// ffplay, and the thread feeding it the sound, while one plays.
    playing: Option<(Child, JoinHandle<()>)>,
}

/// Interleaved samples as 32-bit little-endian floats, ready to be played.
pub type Sound = Arc<[u8]>;

pub fn sound(samples: &[f32]) -> Sound {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

impl Player {
    pub fn new(sample_rate: u32, channels: usize) -> Self {
        Self {
            sample_rate,
            channels,
            playing: None,
        }
    }

    /// Stops what is playing, and plays `sound` from its start.
    pub fn play(&mut self, sound: &Sound) -> Result<(), String> {
        self.stop();
        let layout = if self.channels == 1 { "mono" } else { "stereo" };
        let mut child = Command::new("ffplay")
            .args(["-nodisp", "-autoexit", "-loglevel", "quiet"])
            .args(["-f", "f32le", "-ar", &self.sample_rate.to_string()])
            .args(["-ch_layout", layout, "-i", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("could not run ffplay (is it on PATH?): {e}"))?;
        let mut stdin = child.stdin.take().expect("a piped stdin");
        let sound = Arc::clone(sound);
        // Fed from a thread, as ffplay takes the sound only as fast as it
        // plays it; stopping it ends the write with an error, ignored.
        let feeder = std::thread::spawn(move || {
            let _ = stdin.write_all(&sound);
        });
        self.playing = Some((child, feeder));
        Ok(())
    }

    pub fn stop(&mut self) {
        if let Some((mut child, feeder)) = self.playing.take() {
            let _ = child.kill();
            let _ = child.wait();
            let _ = feeder.join();
        }
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        self.stop();
    }
}
