// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The setup header (clause 4.2.4): the codebooks, floors, residues,
//! mappings and modes that the audio packets are decoded with.
//!
//! The specification defines only how a decoder reads them; each choice of
//! them here is chelys's own. They are as few as a valid stream of silence
//! needs: one codebook, one floor of type 1 without partitions, one residue
//! of type 1 whose one classification codes no pass, and one mapping,
//! without coupling, for the two modes, of short and long blocks.

use super::bitpack::BitPacker;
use super::codebook::Codebook;
use super::header::{self, HeaderType};

/// The samples of a short block and of a long one, `blocksize_0` and
/// `blocksize_1` (clause 4.2.2), whose spectra have 128 and 1024 lines:
/// chelys's choice, the sizes of its AAC encoder's short and long windows.
pub const BLOCKSIZES: [usize; 2] = [256, 2048];

/// The modes, by `mode_number`: one for each block size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Short = 0,
    Long = 1,
}

impl Mode {
    pub const ALL: [Mode; 2] = [Mode::Short, Mode::Long];

    /// `vorbis_mode_blockflag`: whether the mode's blocks are long.
    pub fn blockflag(self) -> bool {
        self == Mode::Long
    }
}

/// The codebook of the residue's classifications: two entries of one bit,
/// the fewest that make a full tree without the special case of an entry
/// alone. No packet reads it yet, as no residue vector is decoded.
fn classbook() -> Codebook {
    Codebook::new(1, vec![Some(1), Some(1)])
}

/// `rangebits` of the floor: its X values span the 1024 lines of a long
/// block's spectrum, `floor1_X_list` element 1 being 2^10.
const FLOOR_RANGEBITS: u32 = 10;

/// The floor's `floor1_multiplier`, 1: its Y values take the 256 steps of
/// the finest range (clause 7.2.3).
const FLOOR_MULTIPLIER: u32 = 1;

/// The residue's `residue_partition_size`.
const RESIDUE_PARTITION_SIZE: u32 = 32;

/// The setup header.
pub fn setup() -> Vec<u8> {
    let mut out = header::begin(HeaderType::Setup);

    let codebooks = [classbook()];
    out.write(codebooks.len() as u32 - 1, 8);
    for codebook in &codebooks {
        codebook.write(&mut out);
    }

    // vorbis_time_count, of one placeholder, which is 0.
    out.write(0, 6);
    out.write(0, 16);

    out.write(0, 6); // vorbis_floor_count, of one
    out.write(1, 16); // its type
    write_floor1(&mut out);

    out.write(0, 6); // vorbis_residue_count, of one
    out.write(1, 16); // its type
    write_residue(&mut out);

    out.write(0, 6); // vorbis_mapping_count, of one
    out.write(0, 16); // its type
    write_mapping(&mut out);

    out.write(Mode::ALL.len() as u32 - 1, 6); // vorbis_mode_count
    for mode in Mode::ALL {
        out.write_bool(mode.blockflag());
        out.write(0, 16); // vorbis_mode_windowtype
        out.write(0, 16); // vorbis_mode_transformtype
        out.write(0, 8); // vorbis_mode_mapping
    }

    out.write_bool(true); // the framing flag
    out.into_bytes()
}

/// A floor of type 1 without partitions (clause 7.2.2), whose X values are
/// those implied alone, 0 and 2^`rangebits`.
fn write_floor1(out: &mut BitPacker) {
    out.write(0, 5); // floor1_partitions
    out.write(FLOOR_MULTIPLIER - 1, 2);
    out.write(FLOOR_RANGEBITS, 4);
}

/// A residue (clause 8.6.1) over the 1024 lines of a long block, of one
/// classification, coded with codebook 0, that codes no pass.
fn write_residue(out: &mut BitPacker) {
    out.write(0, 24); // residue_begin
    out.write(1 << FLOOR_RANGEBITS, 24); // residue_end
    out.write(RESIDUE_PARTITION_SIZE - 1, 24);
    out.write(0, 6); // residue_classifications, of one
    out.write(0, 8); // residue_classbook
    // The classification's residue_cascade, 0: low_bits 0, and no high
    // bits. With no bit set, it has no books.
    out.write(0, 3);
    out.write_bool(false);
}

/// A mapping of type 0 (clause 4.2.4): one submap, of floor 0 and residue
/// 0, for every channel, and no coupling.
fn write_mapping(out: &mut BitPacker) {
    out.write_bool(false); // vorbis_mapping_submaps, of one
    out.write_bool(false); // no square polar channel mapping
    out.write(0, 2); // reserved
    // With one submap, no vorbis_mapping_mux; its floor and residue:
    out.write(0, 8); // the time configuration placeholder
    out.write(0, 8);
    out.write(0, 8);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_long_mode_is_of_long_blocks() {
        assert_eq!(BLOCKSIZES, [256, 2048]);
        assert!(!Mode::Short.blockflag() && Mode::Long.blockflag());
        assert_eq!(Mode::ALL.map(|mode| mode as usize), [0, 1]);
    }

    #[test]
    fn the_setup_header_holds_its_fields_in_order() {
        let setup = setup();
        assert_eq!(&setup[..7], b"\x05vorbis");
        // After the 56 bits of those: 8 bits of the codebook count, 76 of
        // the codebook (24 + 16 + 24, the ordered flag, its two lengths in
        // a run, 5 + 2 bits, and 4 of the lookup type), 22 of the time
        // placeholder, 22 + 11 of the floor, 22 + 86 + 4 of the residue,
        // 22 + 28 of the mapping, 6 + 2 * 41 of the modes and the framing
        // flag: 446 bits, in 56 bytes.
        assert_eq!(setup.len(), 56);
        // The framing flag is the last bit, bit 5 of the last byte.
        assert_eq!(setup.last(), Some(&0b10_0000));
        // The codebook's sync pattern, after the count, at bit 64.
        assert_eq!(&setup[8..11], [0x42, 0x43, 0x56]);
    }
}
