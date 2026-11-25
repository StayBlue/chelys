// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Audio packets (clause 4.3).

use super::bitpack::{BitPacker, ilog};
use super::setup::{BLOCKSIZES, Mode, Setup};

/// What a packet codes of a channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel<'a> {
    /// Nothing: the floor unused, so that the channel decodes to silence
    /// and its residue is not coded (clauses 4.3.2 and 4.3.4).
    Unused,
    /// The floor's values, as [`Floor::values`](super::floor::Floor::values)
    /// gives them, and the residue's, one for each line of the block.
    Coded {
        floor: &'a [u32],
        residue: &'a [i32],
    },
}

/// An audio packet in `mode` of `channels`.
///
/// For a long block, `previous_long` and `next_long` are its
/// `previous_window_flag` and `next_window_flag`: whether the blocks it
/// laps with are long (clause 4.3.1). A short block has neither.
pub fn packet(
    setup: &Setup,
    mode: Mode,
    previous_long: bool,
    next_long: bool,
    channels: &[Channel<'_>],
) -> Vec<u8> {
    let mut out = BitPacker::new();
    out.write(0, 1); // packet_type: audio
    out.write(mode as u32, ilog(Mode::ALL.len() as u32 - 1));
    if mode.blockflag() {
        out.write_bool(previous_long);
        out.write_bool(next_long);
    }
    // Every channel is of the mapping's one submap: each floor in channel
    // order, then the residue of every channel together.
    for channel in channels {
        match channel {
            // Floor 1's nonzero flag, unset (clause 7.2.3).
            Channel::Unused => out.write_bool(false),
            Channel::Coded { floor, .. } => setup.floor.write(floor, &mut out),
        }
    }
    let vectors: Vec<Option<&[i32]>> = channels
        .iter()
        .map(|channel| match channel {
            Channel::Unused => None,
            Channel::Coded { residue, .. } => Some(*residue),
        })
        .collect();
    let lines = BLOCKSIZES[usize::from(mode.blockflag())] / 2;
    // With every vector marked "do not decode", residue type 1 reads
    // nothing (clause 8.6.2).
    setup.residue.write(&vectors, lines, &mut out);
    out.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_long_block_of_silence_takes_4_bits_and_one_a_channel() {
        let setup = Setup::new();
        let silence = |mode, previous_long, next_long, channels| {
            packet(
                &setup,
                mode,
                previous_long,
                next_long,
                &vec![Channel::Unused; channels],
            )
        };
        // packet_type 0, mode 1, both window flags, and a channel unused.
        assert_eq!(silence(Mode::Long, true, true, 1), [0b0_1110]);
        assert_eq!(silence(Mode::Long, false, true, 1), [0b0_1010]);
        assert_eq!(silence(Mode::Long, true, true, 4), [0b0000_1110]);
        assert_eq!(silence(Mode::Long, true, true, 5), [0b0000_1110, 0]);
        assert_eq!(silence(Mode::Short, true, true, 2), [0]);
    }
}
