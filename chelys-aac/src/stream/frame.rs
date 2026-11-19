// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! A frame of coded channels, as long as the bit reservoir allows, framed
//! as the stream asks.

use chelys_core::bitstream::BitWriter;

use super::{FRAME_SAMPLES, FrameSink, StreamConfig};
use crate::adts;
use crate::raw_data_block;
use crate::raw_data_block::element::CodedElement;
use crate::reservoir::BitReservoir;

/// What the frame's element asks of it: the `bit_allocation` of its
/// channels (clause C.1.4, step 15), and the bits it takes besides their
/// noiseless coding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Demand {
    pub bit_allocation: f64,
    /// The element's fields, and each channel's `global_gain`, `ics_info()`
    /// and flags.
    pub side_info_bits: u32,
}

/// The bits of the frame besides its element: its header, the element's
/// `id_syn_ele`, and `ID_END`.
fn overhead_bits(config: &StreamConfig) -> u32 {
    8 * config.header_len() as u32 + raw_data_block::OVERHEAD_BITS
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
/// `raw_data_block()`'s element and end, and each channel's `global_gain`,
/// `ics_info()` and flags. `mean_bits` is that of the stream as written,
/// headers and all (see [`BitReservoir::new`]), rounded to a whole bit with
/// `more_bits`.
///
/// Spec reading: this deliberately departs from clause C.7.3 where
/// `more_bits` is negative, which is taken as 0. The perceptual entropy as
/// printed is negative in most frames of music, and far below 0 for
/// silence (-9535 at 44.1 kHz) and steady tones (about -8980 for one of
/// 1 kHz at half of full scale), so that `bit_allocation` is 0 (see
/// [`Analysis::bit_allocation`](crate::psychoacoustic::Analysis::bit_allocation));
/// where it is not, it is a few hundred bits. `more_bits` is then about
/// `side_info_bits - mean_bits`, and clause C.7.3 would give such a frame
/// only its side information whenever the reservoir has room: the first
/// three frames of a stream at 128 kbit/s in stereo, more at lower rates,
/// and a frame in which audio sets in after the reservoir has given bits
/// away, would be coded without lines, and lose their audio. Taken as 0, a
/// frame is given the mean, and the reservoir keeps what the loops leave,
/// for frames that ask for more.
pub fn more_bits(config: &StreamConfig, demand: &Demand) -> i64 {
    let mean_bits = f64::from(config.bits_per_second) * FRAME_SAMPLES as f64
        / f64::from(config.sampling_frequency.hz());
    let side_info_bits = f64::from(overhead_bits(config) + demand.side_info_bits);
    ((demand.bit_allocation - (mean_bits - side_info_bits)).round() as i64).max(0)
}

/// The bits that the frame's element may take in the next frame: those the
/// reservoir makes available to the frame for its `more_bits` (clause
/// C.7.3), less its header, the element's `id_syn_ele` and `ID_END`.
pub fn element_bits(config: &StreamConfig, reservoir: &BitReservoir, demand: &Demand) -> u32 {
    let available = reservoir.available_bits(more_bits(config, demand));
    available.saturating_sub(overhead_bits(config))
}

/// Writes the frame of `element`, spends its bits from `reservoir`, and
/// gives it to `out`: as long as the element takes, or longer, with fill
/// elements, where the reservoir would otherwise overflow (clause
/// 8.8.2.2).
///
/// # Panics
///
/// If the element takes more bytes than the reservoir allows: a caller's
/// error.
pub fn write<O: FrameSink + ?Sized>(
    config: &StreamConfig,
    reservoir: &mut BitReservoir,
    element: &CodedElement,
    out: &mut O,
) {
    let header_len = config.header_len();
    let (shortest, _) = reservoir.frame_len_range();
    let needed = header_len + raw_data_block::min_len(element);
    let frame_len = shortest.max(needed);
    reservoir.spend(frame_len);
    let block = raw_data_block::write(
        config.channel_configuration,
        element,
        frame_len - header_len,
    );
    match config.adts_header() {
        Some(header) => {
            let mut w = BitWriter::new();
            header.write(&mut w, frame_len, reservoir.adts_buffer_fullness());
            let mut frame = w.into_bytes();
            if header.protected {
                let crc_check = adts::crc_check(&frame, &block);
                frame.extend_from_slice(&crc_check.to_be_bytes());
            }
            frame.extend_from_slice(&block.bytes);
            out.frame(&frame);
        }
        None => out.frame(&block.bytes),
    }
}
