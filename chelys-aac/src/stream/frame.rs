// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! A frame of coded elements and the data streams it carries, as long as
//! the bit reservoir allows, framed as the stream asks.

use chelys_core::bitstream::BitWriter;

use super::{FRAME_SAMPLES, FrameSink, StreamConfig};
use crate::adts;
use crate::raw_data_block;
use crate::raw_data_block::element::CodedElement;
use crate::reservoir::BitReservoir;

/// What an element, or all of the frame's, asks of the frame: the
/// `bit_allocation` of its channels (clause C.1.4, step 15), and the bits
/// it takes besides their noiseless coding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Demand {
    pub bit_allocation: f64,
    /// The element's fields, and each channel's `global_gain`, `ics_info()`
    /// and flags.
    pub side_info_bits: u32,
}

impl std::iter::Sum for Demand {
    fn sum<I: Iterator<Item = Self>>(demands: I) -> Self {
        demands.fold(
            Self {
                bit_allocation: 0.0,
                side_info_bits: 0,
            },
            |sum, demand| Self {
                bit_allocation: sum.bit_allocation + demand.bit_allocation,
                side_info_bits: sum.side_info_bits + demand.side_info_bits,
            },
        )
    }
}

/// The bits of the frame besides its elements, where it carries
/// `data_bits` of data stream elements: its header, the elements'
/// `id_syn_ele`s, the data stream elements, and `ID_END`.
fn overhead_bits(config: &StreamConfig, data_bits: u32) -> u32 {
    let raw_data_block = raw_data_block::overhead_bits(config.channel_configuration);
    8 * config.header_len() as u32 + raw_data_block + data_bits
}

/// `more_bits = bit_allocation - (mean_bits - side_info_bits)` (clause
/// C.1.4, step 15): the bits the frame asks for beyond the mean, which
/// the bit reservoir gives or takes (clause C.7.3).
///
/// Spec gap: step 15 gives `bit_allocation` for a block, which the model
/// computes for each channel, and `more_bits` for the frame, whose bits the
/// reservoir controls; it does not say how the channels' allocations make
/// the frame's. They are summed over the frame's elements, each within the
/// limit of 3000 of its own, as each channel's loop codes its own lines
/// (clause C.7.1).
///
/// Spec gap: Annex C does not define `side_info_bits`. They are taken as
/// the bits of the frame that its channels' noiseless coding does not
/// take, which the perceptual entropy does not count: the header, the
/// `raw_data_block()`'s elements, data stream elements and end, and each
/// channel's `global_gain`, `ics_info()` and flags. `mean_bits` is that of
/// the stream as written,
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
/// for frames that ask for more, up to a point (see [`elements_bits`]).
pub fn more_bits(config: &StreamConfig, demand: &Demand, data_bits: u32) -> i64 {
    let mean_bits = f64::from(config.bits_per_second) * FRAME_SAMPLES as f64
        / f64::from(config.sampling_frequency.hz());
    let side_info_bits = f64::from(overhead_bits(config, data_bits) + demand.side_info_bits);
    ((demand.bit_allocation - (mean_bits - side_info_bits)).round() as i64).max(0)
}

/// The bits that the frame's elements may take in the next frame, which
/// carries `data_bits` of data stream elements: those the reservoir makes
/// available to the frame for its `more_bits` (clause C.7.3), or, where
/// they are fewer, those it offers whatever the frame asks for (see
/// [`BitReservoir::offered_bits`]), less its header, the elements'
/// `id_syn_ele`s, the data stream elements and `ID_END`.
///
/// Spec gap: Annex C does not say what becomes of the bits the loops leave
/// in a frame (figure C.6), beyond what the reservoir may hold. With
/// `more_bits` 0, as in most frames (see [`more_bits`]), clause C.7.3 gives
/// a frame the mean, and the loops, which stop at the finest step that
/// fits, leave some of it; the reservoir keeps what they leave until it is
/// full, and then every frame is filled out to keep the rate with what its
/// loops left (clause 8.8.2.2), which codes nothing. On 8 tracks of pop
/// music and 3 of rock at 44.1 kHz in stereo, those fill bits took 2.4%,
/// 2.9% and 3.0% of the stream at 64, 128 and 192 kbit/s. A frame is
/// therefore also offered what the reservoir holds beyond
/// `max_bit_reservoir - mean_framelength`: the bits a frame leaves come
/// back to the frames after it, and the reservoir stays about a mean frame
/// below its maximum, for frames that ask for more. The fill bits fell to
/// 0.5%, 0.4% and 0.4%. Graded with PEAQ (GstPEAQ 0.6.1, black box), the
/// means of the pop tracks were -3.583, -1.087 and -0.282 at 64, 128 and
/// 192 kbit/s, and of the rock tracks and their 3 windows of 3 s with the
/// most frames coded short -3.579, -0.803 and -0.142, against -3.593,
/// -1.184 and -0.310, and -3.584, -0.843 and -0.160, before. Keeping the
/// reservoir within a quarter of a frame of its maximum graded alike,
/// -1.088 and -0.803 at 128 kbit/s. Spending what the loops leave within
/// the frame instead was tried too, at 128 kbit/s: trying two finer steps
/// after the inner loop, where the bits do not fall as the step grows (see
/// [`inner_loop`](crate::loops::inner::inner_loop)), found one that fits
/// for about 18% of the channels and graded -1.172 and -0.864; amplifying,
/// one at a time, the bands whose distortion lies nearest their allowance,
/// or furthest over it, until four in a row do not fit, used the bits as
/// well as the reservoir does but graded -1.234 and -0.948, and took more
/// than twice as long to encode.
pub fn elements_bits(
    config: &StreamConfig,
    reservoir: &BitReservoir,
    demand: &Demand,
    data_bits: u32,
) -> u32 {
    let asked = reservoir.available_bits(more_bits(config, demand, data_bits));
    let available = asked.max(reservoir.offered_bits());
    available.saturating_sub(overhead_bits(config, data_bits))
}

/// Writes the frame of `elements` and data stream elements of `data`,
/// spends its bits from `reservoir`, and gives it to `out`: as long as they
/// take, or longer, with fill elements, where the reservoir would
/// otherwise overflow (clause 8.8.2.2).
///
/// # Panics
///
/// If the elements and data take more bytes than the reservoir allows: a
/// caller's error.
pub fn write<O: FrameSink + ?Sized>(
    config: &StreamConfig,
    reservoir: &mut BitReservoir,
    elements: &[CodedElement],
    data: &[u8],
    out: &mut O,
) {
    let header_len = config.header_len();
    let configuration = config.channel_configuration;
    let (shortest, _) = reservoir.frame_len_range();
    let needed = header_len + raw_data_block::min_len(configuration, elements, data.len());
    let frame_len = shortest.max(needed);
    reservoir.spend(frame_len);
    let block = raw_data_block::write(configuration, elements, data, frame_len - header_len);
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
