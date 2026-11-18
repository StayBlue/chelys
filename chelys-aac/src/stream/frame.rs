// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! A frame of coded channels, as long as the bit reservoir allows, framed
//! as the stream asks.

use chelys_core::bitstream::BitWriter;

use super::{FrameSink, StreamConfig};
use crate::adts;
use crate::channel::CodedChannel;
use crate::config::Framing;
use crate::raw_data_block;
use crate::reservoir::BitReservoir;

/// `more_bits` (clause C.7.3) of every frame.
///
/// Spec gap: `more_bits`, the bits a frame needs beyond the mean, comes
/// from the perceptual entropy that the psychoacoustic model computes
/// (clause C.1.4, step 15), which the encoder does not compute yet. Until
/// it does, every frame asks for none, and is given the mean.
const MORE_BITS: i64 = 0;

/// The bits that each channel's `individual_channel_stream()` may take in
/// the next frame: those available to the frame (clause C.7.3), less its
/// header and the rest of its `raw_data_block()`, shared equally.
///
/// Spec gap: Annex C does not say how the bits of a frame are shared
/// between its channels, whose loops each fit one channel's lines to the
/// bits they are given (clause C.7.1). Each channel is given an equal
/// share, whatever its lines.
pub fn channel_bits(config: &StreamConfig, reservoir: &BitReservoir) -> u32 {
    let available = reservoir.available_bits(MORE_BITS);
    let overhead = 8 * config.header_len() as u32
        + raw_data_block::overhead_bits(config.channel_configuration);
    let channels = config.channel_configuration.channels() as u32;
    available.saturating_sub(overhead) / channels
}

/// Writes the frame of `channels`, spends its bits from `reservoir`, and
/// gives it to `out`: as long as the channels take, or longer, with fill
/// elements, where the reservoir would otherwise overflow (clause
/// 8.8.2.2).
///
/// # Panics
///
/// If the channels take more bytes than the reservoir allows: a caller's
/// error.
pub fn write<O: FrameSink + ?Sized>(
    config: &StreamConfig,
    reservoir: &mut BitReservoir,
    channels: &[CodedChannel],
    out: &mut O,
) {
    let header_len = config.header_len();
    let (shortest, _) = reservoir.frame_len_range();
    let needed = header_len + raw_data_block::min_len(config.channel_configuration, channels);
    let frame_len = shortest.max(needed);
    reservoir.spend(frame_len);
    let block = raw_data_block::write(
        config.channel_configuration,
        channels,
        frame_len - header_len,
    );
    match config.framing {
        Framing::Adts => {
            let header = adts::Header {
                sampling_frequency: config.sampling_frequency,
                channel_configuration: config.channel_configuration,
            };
            let mut w = BitWriter::new();
            header.write(&mut w, frame_len, reservoir.adts_buffer_fullness());
            let mut frame = w.into_bytes();
            frame.extend_from_slice(&block);
            out.frame(&frame);
        }
        Framing::Raw => out.frame(&block),
    }
}
