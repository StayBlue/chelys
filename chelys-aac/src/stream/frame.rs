// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! A frame of coded elements and the data streams it carries, as long as
//! the bit reservoir allows, framed as the stream asks.

use chelys_core::bitstream::BitWriter;

use super::{FRAME_SAMPLES, FrameSink, StreamConfig};
use crate::adts;
use crate::psychoacoustic::QuietThreshold;
use crate::rate::{self, Control, MeanSteering};
use crate::raw_data_block;
use crate::raw_data_block::element::CodedElement;
use crate::reservoir::BitReservoir;

/// What bounds the length of each frame of a stream.
#[derive(Clone, Debug)]
pub enum FrameRate {
    /// At a constant rate, the bit reservoir.
    Constant(BitReservoir),
    /// At a variable or mean rate, the longest frame alone (see
    /// [`StreamConfig::max_variable_frame_len`]); at a mean rate, the
    /// offset is steered.
    Variable {
        max_frame_len: usize,
        noise_offset_db: f64,
        steering: Option<MeanSteering>,
        /// How far the offset is lowered before it raises the allowance
        /// (see [`rate::PRINTED_QUIET_SHIFT_DB`]).
        shift_db: f64,
        /// The fewest bits a frame is given (see [`Self::least_bits`]).
        least_bits: u32,
    },
}

/// `adts_buffer_fullness` of a stream of variable rate, in which "buffer
/// fullness is not applicable" (clause 8.1.3.2.2).
pub const VARIABLE_RATE_FULLNESS: u32 = 0x7FF;

impl FrameRate {
    /// The frames' bounds of a stream of `config`, if every frame of it
    /// can be written.
    pub fn new(config: &StreamConfig) -> Option<Self> {
        let reservoir = config.reservoir()?;
        let max_frame_len = config.max_variable_frame_len();
        let mean_bits = u64::from(config.bits_per_second) * FRAME_SAMPLES as u64
            / u64::from(config.sampling_frequency.hz());
        let least_bits = (mean_bits as f64 * rate::LEAST_BITS_OF_THE_MEAN) as u32;
        let shift_db = match config.quiet_threshold() {
            QuietThreshold::Printed => rate::PRINTED_QUIET_SHIFT_DB,
            QuietThreshold::Noise => 0.0,
        };
        Some(match config.control {
            Control::Constant => Self::Constant(reservoir),
            Control::Variable { noise_offset_db } => Self::Variable {
                max_frame_len,
                noise_offset_db,
                steering: None,
                shift_db,
                least_bits,
            },
            Control::Mean => Self::Variable {
                max_frame_len,
                noise_offset_db: 0.0,
                shift_db,
                least_bits,
                steering: Some(MeanSteering::new(
                    config.bits_per_second,
                    config.sampling_frequency.hz(),
                    config.bits_per_sample(),
                )),
            },
        })
    }

    /// The factor by which the next frame's allowed distortion is raised:
    /// 1 at a constant rate.
    pub fn noise_factor(&self) -> f64 {
        match self {
            Self::Constant(_) => 1.0,
            Self::Variable {
                noise_offset_db,
                steering,
                shift_db,
                ..
            } => {
                let offset =
                    (steering.as_ref()).map_or(*noise_offset_db, MeanSteering::noise_offset_db);
                rate::noise_factor(offset - shift_db)
            }
        }
    }

    /// At a variable or mean rate, the fewest bits a frame is given, its
    /// header among them: where its channels within their allowance take
    /// fewer, they are fitted to these by the iteration loops, as at a
    /// constant rate (see [`rate::LEAST_BITS_OF_THE_MEAN`]). `None` at a
    /// constant rate, where the bit reservoir sets them.
    pub fn least_bits(&self) -> Option<u32> {
        match self {
            Self::Constant(_) => None,
            Self::Variable { least_bits, .. } => Some(*least_bits),
        }
    }

    /// The most bits the next frame can be given, whatever it asks for.
    pub fn most_bits(&self) -> u32 {
        match self {
            Self::Constant(reservoir) => reservoir.available_bits(0).max(reservoir.offered_bits()),
            Self::Variable { max_frame_len, .. } => 8 * *max_frame_len as u32,
        }
    }
}

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
pub fn frame_overhead_bits(config: &StreamConfig, data_bits: u32) -> u32 {
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
    let side_info_bits = f64::from(frame_overhead_bits(config, data_bits) + demand.side_info_bits);
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
/// therefore also offered a quarter of what the reservoir holds beyond what
/// it keeps, `max_bit_reservoir` less half of `mean_framelength`, or half
/// of `max_bit_reservoir` where that is more: the bits a frame leaves come
/// back to the frames after it, a little at a time, and the reservoir keeps
/// most of its bits, at least half, for frames that ask for more. The fill
/// bits fell to 0.5%, 0.4% and 0.4%. Graded with PEAQ (GstPEAQ 0.6.1, black
/// box), the means of the pop tracks were -3.584, -1.087 and -0.283 at 64,
/// 128 and 192 kbit/s, and of the rock tracks and their 3 windows of 3 s
/// with the most frames coded short -3.578, -0.801 and -0.134, against
/// -3.593, -1.184 and -0.310, and -3.584, -0.843 and -0.160, before.
///
/// Offering all that the reservoir holds beyond `max_bit_reservoir -
/// mean_framelength`, beyond `max_bit_reservoir` less half of
/// `mean_framelength`, or beyond half of `max_bit_reservoir`, or a quarter
/// of what lies beyond half of it, graded alike, within 0.008. But where
/// the maximum is a mean frame or less, as in mono at 44.1 kHz from 132.3
/// kbit/s, the first leaves the reservoir nothing for an attack after
/// another: with the CRC and data streams that leave the audio a few bits a
/// frame, the frames after the first attack took all it held and coded no
/// lines (the test `an_attack_finds_bits_in_the_reservoir_after_another`).
/// Offering all at once also made the bits frames take jump further from
/// one frame to the next: at 192 kbit/s, by 432 bits or more in one frame
/// of twenty, against 256 before frames were offered any, and 290 offering
/// a quarter. And of the conformance tests that compare encodes with and
/// without temporal noise shaping, whose margins the bits passed between
/// frames move, the first took struck bars before their attacks to 1.1 dB
/// more error with it
/// (`temporal_noise_shaping_holds_back_the_noise_before_attacks`), and a
/// quarter beyond half of the maximum gave a sweep at 256 kbit/s 1.3 dB
/// more SNR without it than with it
/// (`temporal_noise_shaping_costs_a_sweep_little`); as chosen, both keep
/// within their thresholds.
///
/// Spending what the loops leave within the frame instead was tried too, at
/// 128 kbit/s, offering what lies beyond `max_bit_reservoir -
/// mean_framelength` before: trying two finer steps after the inner loop,
/// where the bits do not fall as the step grows (see
/// [`inner_loop`](crate::loops::inner::inner_loop)), found one that fits
/// for about 18% of the channels and graded -1.172 and -0.864, against
/// -1.087 and -0.803 so offered; amplifying, one at a time, the bands whose
/// distortion lies nearest their allowance, or furthest over it, until four
/// in a row do not fit, used the bits as well as the reservoir does but
/// graded -1.234 and -0.948, and took more than twice as long to encode.
///
/// At a variable or mean rate, the elements may take the bits of the
/// longest frame, of which they take what their noise within its allowance
/// takes (see [`ChannelCoding::within_allowance`]).
///
/// [`ChannelCoding::within_allowance`]: crate::loops::ChannelCoding::within_allowance
pub fn elements_bits(
    config: &StreamConfig,
    rate: &FrameRate,
    demand: &Demand,
    data_bits: u32,
) -> u32 {
    let available = match rate {
        FrameRate::Constant(reservoir) => {
            let asked = reservoir.available_bits(more_bits(config, demand, data_bits));
            asked.max(reservoir.offered_bits())
        }
        FrameRate::Variable { .. } => rate.most_bits(),
    };
    available.saturating_sub(frame_overhead_bits(config, data_bits))
}

/// Writes the frame of `elements` and data stream elements of `data`,
/// spends its bits from the reservoir of `rate`, and gives it to `out`: as
/// long as they take, or longer, with fill elements, where the reservoir
/// would otherwise overflow (clause 8.8.2.2). At a variable or mean rate,
/// the frame is as long as they take, and its header says that the stream
/// is of variable rate.
///
/// # Panics
///
/// If the elements and data take more bytes than the reservoir allows, or
/// than the longest frame: a caller's error.
pub fn write<O: FrameSink + ?Sized>(
    config: &StreamConfig,
    rate: &mut FrameRate,
    elements: &[CodedElement],
    data: &[u8],
    out: &mut O,
) {
    let header_len = config.header_len();
    let configuration = config.channel_configuration;
    let needed = header_len + raw_data_block::min_len(configuration, elements, data.len());
    let (frame_len, buffer_fullness) = match rate {
        FrameRate::Constant(reservoir) => {
            let (shortest, _) = reservoir.frame_len_range();
            let frame_len = shortest.max(needed);
            reservoir.spend(frame_len);
            (frame_len, reservoir.adts_buffer_fullness())
        }
        FrameRate::Variable {
            max_frame_len,
            steering,
            ..
        } => {
            assert!(
                needed <= *max_frame_len,
                "a frame of {needed} bytes, not at most {max_frame_len}"
            );
            if let Some(steering) = steering {
                steering.spend(needed);
            }
            (needed, VARIABLE_RATE_FULLNESS)
        }
    };
    let block = raw_data_block::write(configuration, elements, data, frame_len - header_len);
    match config.adts_header() {
        Some(header) => {
            let mut w = BitWriter::new();
            header.write(&mut w, frame_len, buffer_fullness);
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
