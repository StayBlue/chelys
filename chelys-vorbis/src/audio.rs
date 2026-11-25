// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Audio packets (clause 4.3).

use super::bitpack::{BitPacker, ilog};
use super::setup::{BLOCKSIZES, Mode, Setup};

/// An audio packet in `mode`, whose channels have the floors `floors`, each
/// `None` where unused, so that the channel decodes to silence (clauses
/// 4.3.2 and 4.3.6), or the floor's values as
/// [`Floor::values`](super::floor::Floor::values) gives them; and the
/// residue vectors `residues`, one value for each line of the block, after
/// the mapping's coupling where the setup couples channels.
///
/// For a long block, `previous_long` and `next_long` are its
/// `previous_window_flag` and `next_window_flag`: whether the blocks it
/// laps with are long (clause 4.3.1). A short block has neither.
///
/// # Panics
///
/// If a vector that a decoder does not decode, and so takes as zeros, is
/// not all zeros: that of an unused floor, unless the mapping couples it
/// with a channel whose floor is used (clause 4.3.3).
pub fn packet(
    setup: &Setup,
    mode: Mode,
    previous_long: bool,
    next_long: bool,
    floors: &[Option<&[u32]>],
    residues: &[&[i32]],
) -> Vec<u8> {
    let mut out = BitPacker::new();
    out.write(0, 1); // packet_type: audio
    out.write(mode as u32, ilog(Mode::ALL.len() as u32 - 1));
    if mode.blockflag() {
        out.write_bool(previous_long);
        out.write_bool(next_long);
    }
    // Every channel is of the mode's mapping's one submap: each floor in
    // channel order, then the residue of every channel together.
    for floor in floors {
        match floor {
            // Floor 1's nonzero flag, unset (clause 7.2.3).
            None => out.write_bool(false),
            Some(floor) => setup.floor(mode).write(floor, &mut out),
        }
    }
    // Clause 4.3.3: a channel coupled with one whose floor is used is
    // decoded too.
    let mut decoded: Vec<bool> = floors.iter().map(Option::is_some).collect();
    if setup.coupled && decoded.contains(&true) {
        decoded.fill(true);
    }
    let vectors: Vec<Option<&[i32]>> = (residues.iter().zip(decoded))
        .map(|(&residue, decoded)| {
            assert!(
                decoded || residue.iter().all(|&v| v == 0),
                "a residue that is not decoded"
            );
            decoded.then_some(residue)
        })
        .collect();
    let lines = BLOCKSIZES[usize::from(mode.blockflag())] / 2;
    setup.residue(mode).write(&vectors, lines, &mut out);
    out.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_long_block_of_silence_takes_4_bits_and_one_a_channel() {
        let zeros = [0; BLOCKSIZES[1] / 2];
        let silence = |mode: Mode, previous_long, next_long, channels| {
            let lines = BLOCKSIZES[usize::from(mode.blockflag())] / 2;
            packet(
                &Setup::new(channels),
                mode,
                previous_long,
                next_long,
                &vec![None; channels],
                &vec![&zeros[..lines]; channels],
            )
        };
        // packet_type 0, mode 1, both window flags, and a channel unused.
        assert_eq!(silence(Mode::Long, true, true, 1), [0b0_1110]);
        assert_eq!(silence(Mode::Long, false, true, 1), [0b0_1010]);
        assert_eq!(silence(Mode::Long, true, true, 4), [0b0000_1110]);
        assert_eq!(silence(Mode::Long, true, true, 5), [0b0000_1110, 0]);
        assert_eq!(silence(Mode::Short, true, true, 2), [0]);
        // In stereo, coupled, with residue type 2, which reads nothing where
        // every vector is not to be decoded (clause 8.6.5).
        assert_eq!(silence(Mode::Long, true, true, 2), [0b0_1110]);
    }

    /// A residue whose floor is unused is coded where the mapping couples
    /// it with a used one (clause 4.3.3), and must be zeros where not.
    #[test]
    fn an_unused_channels_residue_is_coded_only_where_coupled() {
        let short_floor = &Setup::new(1).floors[0];
        let floor = short_floor.values(&vec![60; short_floor.xs().len()]);
        let residue = [1; BLOCKSIZES[0] / 2];
        let packet = |channels| {
            std::panic::catch_unwind(|| {
                let setup = Setup::new(channels);
                let mut floors = vec![None; channels];
                floors[0] = Some(&floor[..]);
                packet(
                    &setup,
                    Mode::Short,
                    true,
                    true,
                    &floors,
                    &vec![&residue[..]; channels],
                )
            })
        };
        assert!(packet(2).is_ok());
        assert!(packet(3).is_err());
    }
}
