// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The granule (clause 2.1.70), the unit every stage of the encoder works
//! on: its frequency lines, the block type that sets its window and
//! transform, and the granule of one channel as it is quantised.

use crate::header::SamplingFrequency;
use crate::huffman::SpectrumCoding;
use crate::scalefactor_bands;
use crate::scalefactors::Scalefactors;
use crate::side_info::{self, Window};

/// Frequency lines in a granule (clause 2.1.70).
pub const GRANULE_LINES: usize = 576;

/// The window and transform of a granule's block (clause C.1.5.3.3, figure
/// C.7): `block_type` 0 to 3 of clause 2.4.2.7, and `block_type` 2 with
/// `mixed_block_flag` set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockType {
    Normal,
    Start,
    Short,
    /// A short block whose two lowest subbands are transformed with the
    /// normal window, and sent with long scalefactor bands (clause 2.4.2.7):
    /// `block_type` 2 with `mixed_block_flag` set. It takes the place of a
    /// short block among the others.
    Mixed,
    Stop,
}

impl BlockType {
    /// Whether the granule is `block_type` 2, with short windows in all its
    /// subbands or all but the two lowest.
    pub fn short_windows(self) -> bool {
        matches!(self, Self::Short | Self::Mixed)
    }
}

/// One granule of one channel, quantised and ready for coding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantizedGranule {
    pub block_type: BlockType,
    /// The quantizer step size; see the requantization formula in clause
    /// 2.4.3.4.7.1.
    pub global_gain: u8,
    pub scalefactors: Scalefactors,
    /// The quantised values in the order they are sent: by frequency, or for
    /// short blocks by scalefactor band, window and frequency (clause
    /// 2.4.2.7), and for mixed blocks so above their two lowest subbands.
    pub values: [i32; GRANULE_LINES],
    pub coding: SpectrumCoding,
}

impl QuantizedGranule {
    /// The main data bits of the granule, `part2_3_length`: its scalefactors
    /// and Huffman code bits.
    ///
    /// # Panics
    ///
    /// If a scalefactor is above its limit.
    pub fn bit_len(&self, sampling_frequency: SamplingFrequency) -> usize {
        let compress = self
            .scalefactors
            .compress()
            .expect("scalefactors within their limits");
        let bands = scalefactor_bands::coding_bands(self.block_type, sampling_frequency);
        self.scalefactors.part2_length(compress) + self.coding.bit_count(&self.values, &bands)
    }

    /// Partitions `values` of a granule of `block_type` and chooses Huffman
    /// tables for them (clause C.1.5.4.4). The scalefactors are all zero.
    pub fn new(
        global_gain: u8,
        values: [i32; GRANULE_LINES],
        block_type: BlockType,
        sampling_frequency: SamplingFrequency,
    ) -> Self {
        let bands = scalefactor_bands::coding_bands(block_type, sampling_frequency);
        let coding = SpectrumCoding::choose(&values, block_type, &bands, sampling_frequency);
        Self::with_coding(global_gain, values, block_type, sampling_frequency, coding)
    }

    /// [`new`](Self::new), with the coding that it would choose already
    /// chosen, as [`SpectrumCoding::count`] gives it.
    pub fn with_coding(
        global_gain: u8,
        values: [i32; GRANULE_LINES],
        block_type: BlockType,
        sampling_frequency: SamplingFrequency,
        coding: SpectrumCoding,
    ) -> Self {
        debug_assert_eq!(
            coding,
            SpectrumCoding::choose(
                &values,
                block_type,
                &scalefactor_bands::coding_bands(block_type, sampling_frequency),
                sampling_frequency,
            )
        );
        Self {
            block_type,
            global_gain,
            scalefactors: Scalefactors::new(block_type, sampling_frequency.version()),
            values,
            coding,
        }
    }

    /// The side information's `window_switching_flag` and the fields that
    /// depend on it.
    ///
    /// Spec gap: Annex C does not say how to choose `subblock_gain`, which
    /// sets a coarser step for one short window. It is always 0: the outer
    /// loop's scalefactors shape the noise of each window instead.
    ///
    pub fn window(&self) -> Window {
        let coding = &self.coding;
        let block_type = match self.block_type {
            BlockType::Normal => {
                return Window::Normal {
                    table_select: coding.table_select,
                    region0_count: coding.region0_count,
                    region1_count: coding.region1_count,
                };
            }
            BlockType::Start => side_info::BlockType::Start,
            BlockType::Short | BlockType::Mixed => side_info::BlockType::Short,
            BlockType::Stop => side_info::BlockType::End,
        };
        Window::Switched {
            block_type,
            mixed_block_flag: self.block_type == BlockType::Mixed,
            table_select: [coding.table_select[0], coding.table_select[1]],
            subblock_gain: [0; 3],
        }
    }
}

/// The largest `part2_3_length`, a 12-bit field (clause 2.4.1.7).
pub const MAX_PART2_3_LENGTH: usize = (1 << 12) - 1;
