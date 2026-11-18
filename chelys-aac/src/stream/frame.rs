// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! A frame of coded channels, as long as the bit reservoir allows, framed
//! as the stream asks.

use chelys_core::bitstream::BitWriter;

use super::{FRAME_SAMPLES, FrameSink, StreamConfig};
use crate::adts;
use crate::channel::{self, CodedChannel};
use crate::config::Framing;
use crate::raw_data_block;
use crate::raw_data_block::element::COMMON_WINDOW;
use crate::reservoir::BitReservoir;
use crate::window::Windows;

/// What the channels of the next frame ask of it: each channel's
/// `bit_allocation` (clause C.1.4, step 15) and windows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Demand {
    pub bit_allocation: f64,
    pub windows: Windows,
    /// Whether every line of the channel is zero, so that it is coded
    /// without any (see [`loops::quantize_channel`]).
    pub silent: bool,
}

/// The bits of the frame that its channels' noiseless coding does not
/// take: its header, the rest of its `raw_data_block()`, and each
/// channel's `global_gain`, `ics_info()` and flags.
fn side_info_bits(config: &StreamConfig, demands: &[Demand]) -> u32 {
    let channels: u32 = (demands.iter())
        .map(|demand| channel::bits(demand.windows, COMMON_WINDOW, 0))
        .sum();
    overhead_bits(config) + channels
}

/// The bits of the frame besides its channels' `individual_channel_stream()`s.
fn overhead_bits(config: &StreamConfig) -> u32 {
    8 * config.header_len() as u32 + raw_data_block::overhead_bits(config.channel_configuration)
}

/// `more_bits = bit_allocation - (mean_bits - side_info_bits)` (clause
/// C.1.4, step 15): the bits the frame asks for beyond the mean, which
/// the bit reservoir gives or takes (clause C.7.3).
///
/// Spec gap: step 15 gives `bit_allocation` for a block, which the model
/// computes for each channel, and `more_bits` for the frame, whose bits the
/// reservoir controls; it does not say how the channels' allocations make
/// the frame's. They are summed, each within the limit of 3000 of its own,
/// as each channel's loop codes its own lines (clause C.7.1).
///
/// Spec gap: Annex C does not define `side_info_bits`. They are taken as
/// the bits of the frame that its channels' noiseless coding does not
/// take, which the perceptual entropy does not count: the header, the
/// `raw_data_block()`'s elements and end, and each channel's `global_gain`,
/// `ics_info()` and flags. `mean_bits` is that of the stream as written,
/// headers and all (see [`BitReservoir::new`]), rounded to a whole bit with
/// `more_bits`.
///
/// Spec reading: this deliberately departs from clause C.7.3 where
/// `more_bits` is negative, which is taken as 0. The perceptual entropy as
/// printed is negative in most frames of music, so that `bit_allocation`
/// is 0 (see
/// [`LongThreshold::bit_allocation`](crate::psychoacoustic::LongThreshold::bit_allocation)),
/// and where it is not, it is a few hundred bits; `more_bits` is then
/// about `side_info_bits - mean_bits`. Clause C.7.3 would give such a frame
/// only its side information, until the reservoir is full: the first
/// three frames of a stream at 128 kbit/s in stereo, and more at lower
/// rates, would be coded without lines, and lose their audio. Taken as 0,
/// a frame is given the mean, and the reservoir keeps what the loops leave,
/// for frames that ask for more.
pub fn more_bits(config: &StreamConfig, demands: &[Demand]) -> i64 {
    let allocation: f64 = demands.iter().map(|demand| demand.bit_allocation).sum();
    let mean_bits = f64::from(config.bits_per_second) * FRAME_SAMPLES as f64
        / f64::from(config.sampling_frequency.hz());
    let side_info_bits = f64::from(side_info_bits(config, demands));
    ((allocation - (mean_bits - side_info_bits)).round() as i64).max(0)
}

/// The bits that each channel's `individual_channel_stream()` may take in
/// the next frame: those the reservoir makes available to the frame for
/// its `more_bits` (clause C.7.3), less its header and the rest of its
/// `raw_data_block()`. A silent channel is given what it costs, the bits
/// of a channel without lines, and the others equal shares of the rest.
///
/// Spec gap: Annex C does not say how the bits of a frame are shared
/// between its channels, whose loops each fit one channel's lines to the
/// bits they are given (clause C.7.1). The share does not depend on the
/// order of the channels, so that a channel codes alike whichever of a
/// pair it is. Shares in proportion to each channel's `bit_allocation`
/// were tried, and starved channels: the allocation is 0 in most frames of
/// music, where the perceptual entropy is negative (see
/// [`LongThreshold::bit_allocation`](crate::psychoacoustic::LongThreshold::bit_allocation)),
/// and a channel whose allocation was 0 beside one whose was not was left
/// the bits of a channel without lines.
pub fn bits_per_channel(
    config: &StreamConfig,
    reservoir: &BitReservoir,
    demands: &[Demand],
) -> Vec<u32> {
    let available = reservoir.available_bits(more_bits(config, demands));
    let silent_bits = |demand: &Demand| channel::bits(demand.windows, COMMON_WINDOW, 0);
    let reserved: u32 = (demands.iter())
        .filter(|demand| demand.silent)
        .map(silent_bits)
        .sum();
    let coded = demands.iter().filter(|demand| !demand.silent).count() as u32;
    let share = (available.saturating_sub(overhead_bits(config)))
        .saturating_sub(reserved)
        .checked_div(coded)
        .unwrap_or(0);
    (demands.iter())
        .map(|demand| {
            if demand.silent {
                silent_bits(demand)
            } else {
                share
            }
        })
        .collect()
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
