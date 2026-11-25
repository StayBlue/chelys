// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The setup header (clause 4.2.4): the codebooks, floors, residues,
//! mappings and modes that the audio packets are decoded with.
//!
//! The specification defines only how a decoder reads them; each choice of
//! them here is chelys's own: the codebooks of one residue of type 1 and of
//! one floor of type 1, and one mapping, without coupling, for the two
//! modes, of short and long blocks.

use super::bitpack::BitPacker;
use super::codebook::Codebook;
use super::floor::Floor;
use super::header::{self, HeaderType};
use super::residue::Residue;

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

/// The floor and residue of every packet, and their codebooks.
#[derive(Clone, Debug)]
pub struct Setup {
    pub floor: Floor,
    pub residue: Residue,
}

impl Default for Setup {
    fn default() -> Self {
        Self::new()
    }
}

impl Setup {
    /// The residue's codebooks first, then the floor's.
    pub fn new() -> Self {
        let residue = Residue::new(0);
        let floor = Floor::new(residue.codebooks().len() as u8);
        Self { floor, residue }
    }

    fn codebooks(&self) -> Vec<Codebook> {
        let mut books = self.residue.codebooks();
        books.extend(self.floor.codebooks());
        books
    }

    /// The setup header.
    pub fn header(&self) -> Vec<u8> {
        let mut out = header::begin(HeaderType::Setup);

        let codebooks = self.codebooks();
        out.write(codebooks.len() as u32 - 1, 8);
        for codebook in &codebooks {
            codebook.write(&mut out);
        }

        // vorbis_time_count, of one placeholder, which is 0.
        out.write(0, 6);
        out.write(0, 16);

        out.write(0, 6); // vorbis_floor_count, of one
        out.write(1, 16); // its type
        self.floor.write_setup(&mut out);

        out.write(0, 6); // vorbis_residue_count, of one
        out.write(1, 16); // its type
        self.residue.write_setup(&mut out);

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
    fn the_setup_header_begins_with_its_codebooks() {
        let setup = Setup::new();
        let header = setup.header();
        assert_eq!(&header[..7], b"\x05vorbis");
        assert_eq!(usize::from(header[7]) + 1, setup.codebooks().len());
        // The first codebook's sync pattern, after the count, at bit 64.
        assert_eq!(&header[8..11], [0x42, 0x43, 0x56]);
        // The framing flag is the last bit set.
        assert_ne!(*header.last().unwrap(), 0);
        assert!(setup.codebooks().len() <= 256);
    }
}
