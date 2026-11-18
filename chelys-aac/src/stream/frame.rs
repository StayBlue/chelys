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
