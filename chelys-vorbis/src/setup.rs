// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The setup header (clause 4.2.4): the codebooks, floors, residues,
//! mappings and modes that the audio packets are decoded with.
//!
//! The specification defines only how a decoder reads them; each choice of
//! them here is chelys's own: for each block size, a floor of type 1, a
//! residue of type 1, a mapping without coupling and a mode, the floors
//! sharing one set of codebooks and the residues another.

use super::bitpack::BitPacker;
use super::codebook::Codebook;
use super::floor::Floor;
use super::header::{self, HeaderType};
use super::residue::Residue;

/// The samples of a short block and of a long one, `blocksize_0` and
/// `blocksize_1` (clause 4.2.2), whose spectra have 128 and 1024 lines:
/// chelys's choice, the sizes of its AAC encoder's short and long windows,
/// whose psychoacoustic model's FFTs are of these sizes.
///
/// Short blocks of 512 samples, four in a long block's place, each allowed
/// the least noise of the two short FFTs about its center, graded a little
/// higher with PEAQ (GstPEAQ 0.6.1, black box) on the 51 tracks of the
/// tuning set at 44.1 kHz in stereo, interpolated between noise offsets to
/// the same mean bitrates: -2.826, -1.164 and -0.536 at 96, 128 and 160
/// kbit/s, against -2.840, -1.173 and -0.538 with 256, before the short
/// blocks' `rpelev` and floor were tuned. But PEAQ's basic version models
/// the ear in frames of 2048 samples, and does not see where in them the
/// noise lies: of bursts of noise over a quiet chord, at 44.1 kHz, blocks
/// of 512 left the noise in the 64 to 256 samples before each onset 2.9
/// and 6.7 dB louder than blocks of 256, at two rates, in streams 7 to 8%
/// larger. Pre-echo so near an attack is what short blocks are for, and
/// 256 keeps it nearer.
pub const BLOCKSIZES: [usize; 2] = [256, 2048];

/// The modes, by `mode_number`: one for each block size, whose mapping,
/// floor and residue are those of the same number.
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

/// The floor and residue of each block size, by blockflag, and their
/// codebooks.
#[derive(Clone, Debug)]
pub struct Setup {
    pub floors: [Floor; 2],
    pub residues: [Residue; 2],
}

impl Default for Setup {
    fn default() -> Self {
        Self::new()
    }
}

impl Setup {
    /// The residues' codebooks first, then the floors'.
    pub fn new() -> Self {
        let residues = [false, true].map(|blockflag| Residue::new(blockflag, 0));
        let first_floor_book = residues[1].codebooks().len() as u8;
        Self {
            floors: [
                Floor::short(first_floor_book),
                Floor::long(first_floor_book),
            ],
            residues,
        }
    }

    /// The floor of `mode`'s blocks.
    pub fn floor(&self, mode: Mode) -> &Floor {
        &self.floors[usize::from(mode.blockflag())]
    }

    /// The residue of `mode`'s blocks.
    pub fn residue(&self, mode: Mode) -> &Residue {
        &self.residues[usize::from(mode.blockflag())]
    }

    /// The codebooks, which the residues share, and the floors.
    fn codebooks(&self) -> Vec<Codebook> {
        let mut books = self.residues[1].codebooks();
        books.extend(self.floors[1].codebooks());
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

        out.write(self.floors.len() as u32 - 1, 6); // vorbis_floor_count
        for floor in &self.floors {
            out.write(1, 16); // its type
            floor.write_setup(&mut out);
        }

        out.write(self.residues.len() as u32 - 1, 6); // vorbis_residue_count
        for residue in &self.residues {
            out.write(1, 16); // its type
            residue.write_setup(&mut out);
        }

        out.write(Mode::ALL.len() as u32 - 1, 6); // vorbis_mapping_count
        for mode in Mode::ALL {
            out.write(0, 16); // its type
            write_mapping(mode as u32, &mut out);
        }

        out.write(Mode::ALL.len() as u32 - 1, 6); // vorbis_mode_count
        for mode in Mode::ALL {
            out.write_bool(mode.blockflag());
            out.write(0, 16); // vorbis_mode_windowtype
            out.write(0, 16); // vorbis_mode_transformtype
            out.write(mode as u32, 8); // vorbis_mode_mapping
        }

        out.write_bool(true); // the framing flag
        out.into_bytes()
    }
}

/// A mapping of type 0 (clause 4.2.4): one submap, of floor `number` and
/// residue `number`, for every channel, and no coupling.
fn write_mapping(number: u32, out: &mut BitPacker) {
    out.write_bool(false); // vorbis_mapping_submaps, of one
    out.write_bool(false); // no square polar channel mapping
    out.write(0, 2); // reserved
    // With one submap, no vorbis_mapping_mux; its floor and residue:
    out.write(0, 8); // the time configuration placeholder
    out.write(number, 8);
    out.write(number, 8);
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
