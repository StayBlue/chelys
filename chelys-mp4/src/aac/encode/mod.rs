// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! chelys's own AAC-LC encoder into an MP4 file: PCM coded by
//! [`chelys_aac::Encoder`] as raw frames, which the file holds as they are
//! coded, its track described as it is of any AAC-LC stream (see
//! [`aac`](super)). An [`Encoder`] writes the file whole, and a
//! [`FragmentedEncoder`] as fragments, as the frames come.

mod fragmented;
mod progressive;

pub use fragmented::{EncodeError, FragmentedEncoder};
pub use progressive::Encoder;

use super::Stream;

/// The encoder of raw frames of the stream `config` asks for.
fn raw_encoder(config: chelys_aac::Config) -> Result<chelys_aac::Encoder, chelys_aac::ConfigError> {
    chelys_aac::Encoder::new(config.framing(chelys_aac::Framing::Raw))
}

impl Stream {
    fn of(encoder: &chelys_aac::Encoder) -> Self {
        Self {
            sample_rate: encoder.sample_rate(),
            channels: encoder.channels() as u16,
            frame_samples: encoder.samples_per_frame() as u32,
            buffer_size: encoder.decoder_buffer_bytes(),
            audio_specific_config: encoder.audio_specific_config(),
        }
    }
}
