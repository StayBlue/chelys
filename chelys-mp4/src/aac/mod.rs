// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! AAC-LC from chelys's encoder in an MP4 file: its raw frames as the
//! samples of an `'mp4a'` track, which its `AudioSpecificConfig()`
//! describes.

mod fragmented;
mod progressive;

pub use fragmented::{EncodeError, FragmentOptions, FragmentedEncoder, LengthError};
pub use progressive::Encoder;

use chelys_aac::{Config, ConfigError, Framing};

use super::descriptor::{AUDIO_ISO_IEC_14496_3, AUDIO_STREAM, DecoderConfig, es_descriptor};
use super::movie::AudioTrack;

/// The encoder of raw frames of the stream `config` asks for.
fn raw_encoder(config: Config) -> Result<chelys_aac::Encoder, ConfigError> {
    chelys_aac::Encoder::new(config.framing(Framing::Raw))
}

/// What the track's boxes give of a stream, all known before its first
/// frame but the bitrates.
struct Stream {
    sample_rate: u32,
    channels: u16,
    frame_samples: u32,
    buffer_size: u32,
    audio_specific_config: Vec<u8>,
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

    /// The track of the stream, its `ES_Descriptor` of `max_bitrate` and
    /// `avg_bitrate`.
    ///
    /// Spec reading: the objectTypeIndication is "Audio ISO/IEC 14496-3",
    /// 0x40, whose DecoderSpecificInfo is the AudioSpecificConfig() (ISO/IEC
    /// 14496-3, clause 1.6.2.1), which names the stream AAC LC: with
    /// perceptual noise substitution it is MPEG-4 AAC, and without it AAC LC
    /// of either standard. That of ISO/IEC 13818-7's Low Complexity profile,
    /// 0x67, would take an adif_header() as its DecoderSpecificInfo instead
    /// (ISO/IEC 14496-1, clause 6.2.6.7.2).
    ///
    /// Spec reading: bufferSizeDB, "the size of the decoding buffer", the
    /// buffer at the decoder's input (ISO/IEC 14496-1, clause 6.1.2.4), is
    /// that of the decoder's input buffer of ISO/IEC 13818-7 (clause
    /// 8.2.3.1), within which the bit reservoir keeps the stream.
    fn track(&self, max_bitrate: u32, avg_bitrate: u32) -> AudioTrack {
        let es_descriptor = es_descriptor(&DecoderConfig {
            object_type_indication: AUDIO_ISO_IEC_14496_3,
            stream_type: AUDIO_STREAM,
            buffer_size: self.buffer_size,
            max_bitrate,
            avg_bitrate,
            specific_info: &self.audio_specific_config,
        });
        AudioTrack {
            coding_name: *b"mp4a",
            es_descriptor,
            sample_rate: self.sample_rate,
            channels: self.channels,
            frame_samples: self.frame_samples,
            delay: chelys_aac::DELAY as u32,
            roll_distance: -1,
        }
    }
}
