// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Audio packets (clause 4.3).

use super::bitpack::{BitPacker, ilog};
use super::setup::Mode;

/// An audio packet of `channels` channels in `mode` that codes silence:
/// each channel's floor unused, so that no residue is coded (clauses 4.3.2
/// to 4.3.4).
///
/// For a long block, `previous_long` and `next_long` are its
/// `previous_window_flag` and `next_window_flag`: whether the blocks it
/// laps with are long (clause 4.3.1). A short block has neither.
pub fn silence(mode: Mode, previous_long: bool, next_long: bool, channels: usize) -> Vec<u8> {
    let mut out = BitPacker::new();
    out.write(0, 1); // packet_type: audio
    out.write(mode as u32, ilog(Mode::ALL.len() as u32 - 1));
    if mode.blockflag() {
        out.write_bool(previous_long);
        out.write_bool(next_long);
    }
    for _ in 0..channels {
        // Floor 1's nonzero flag, unset: the channel is unused (clause
        // 7.2.3). Its residue vector is then not decoded, and with every
        // vector so, residue type 1 reads nothing (clause 8.6.2).
        out.write_bool(false);
    }
    out.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_long_block_of_silence_takes_4_bits_and_one_a_channel() {
        // packet_type 0, mode 1, both window flags, and a channel unused.
        assert_eq!(silence(Mode::Long, true, true, 1), [0b0_1110]);
        assert_eq!(silence(Mode::Long, false, true, 1), [0b0_1010]);
        assert_eq!(silence(Mode::Long, true, true, 4), [0b0000_1110]);
        assert_eq!(silence(Mode::Long, true, true, 5), [0b0000_1110, 0]);
        assert_eq!(silence(Mode::Short, true, true, 2), [0]);
    }
}
