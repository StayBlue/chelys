// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The setup header (clause 4.2.4): the codebooks, floors, residues,
//! mappings and modes that the audio packets are decoded with.
//!
//! The specification defines only how a decoder reads them; each choice of
//! them here is chelys's own: for each block size, a floor of type 1, a
//! residue, a mapping and a mode, the floors sharing one set of codebooks
//! and the residues another. In a stereo stream, each mapping couples the
//! two channels (see [`coupling`](super::coupling)), and the residues are
//! of type 2; in any other, no mapping couples channels, and the residues
//! are of type 1.

use super::bitpack::BitPacker;
use super::codebook::Codebook;
use super::floor::Floor;
use super::header::{self, HeaderType};
use super::residue::{Residue, ResidueKind};

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
///
/// Two modes, whatever the channels: a stereo stream codes every block of
/// either size coupled, rather than having modes of each size coupled and
/// not to choose from. The specification allows up to 64 modes; two is a
/// choice for compatibility with ffmpeg's Vorbis parser, which times the
/// packets of an Ogg stream, and which in ffmpeg n9.0.2 reports streams of
/// more than two modes as not implemented, though its decoder decodes
/// them.
///
/// What it costs: measured with modes of each size coupled and not, each
/// block in whichever took fewer bytes, the tuning set's 51 tracks at 44.1
/// kHz in stereo took 1.4% fewer bytes than coupled throughout, before the
/// angles were narrowed (see `coupling`). Where the
/// channels are unlike, coupling costs more: a channel whose floor is
/// unused is decoded all the same where coupled (clause 4.3.3), and its
/// angles, all 0, are coded interleaved with the other's magnitudes. On 23
/// s of a track of the tuning set at a noise offset of 3 dB, with its right
/// channel silent, the stream took 10% more bytes than before coupling,
/// when each channel was coded on its own; with its channels taking turns,
/// each silent in every other 1024 samples, 20% more.
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
    /// Whether the mappings couple channel 0, as the magnitude, with
    /// channel 1, as the angle: in a stereo stream.
    pub coupled: bool,
}

impl Setup {
    /// The setup of a stream of `channels` channels. The residues'
    /// codebooks come first, then the floors'.
    pub fn new(channels: usize) -> Self {
        let coupled = channels == 2;
        let kind = if coupled {
            ResidueKind::Coupled
        } else {
            ResidueKind::Uncoupled
        };
        let residues = [false, true].map(|blockflag| Residue::new(blockflag, kind, 0));
        let first_floor_book = residues[1].codebooks().len() as u8;
        Self {
            floors: [
                Floor::short(first_floor_book),
                Floor::long(first_floor_book),
            ],
            residues,
            coupled,
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
            out.write(residue.residue_type(), 16);
            residue.write_setup(&mut out);
        }

        out.write(Mode::ALL.len() as u32 - 1, 6); // vorbis_mapping_count
        for mode in Mode::ALL {
            out.write(0, 16); // its type
            self.write_mapping(mode, &mut out);
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

    /// The mapping of `mode`, of type 0 (clause 4.2.4): one submap, of the
    /// floor and residue of the same number, for every channel; and in a
    /// stereo stream, one coupling step, of channel 0 as the magnitude and
    /// channel 1 as the angle.
    fn write_mapping(&self, mode: Mode, out: &mut BitPacker) {
        out.write_bool(false); // vorbis_mapping_submaps, of one
        out.write_bool(self.coupled); // square polar channel mapping
        if self.coupled {
            out.write(0, 8); // vorbis_mapping_coupling_steps, of one
            // Each channel number in ilog(2 - 1) bits.
            out.write(0, 1); // vorbis_mapping_magnitude
            out.write(1, 1); // vorbis_mapping_angle
        }
        out.write(0, 2); // reserved
        // With one submap, no vorbis_mapping_mux; its floor and residue:
        out.write(0, 8); // the time configuration placeholder
        out.write(mode as u32, 8);
        out.write(mode as u32, 8);
    }
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
        for channels in 1..=8 {
            let setup = Setup::new(channels);
            assert_eq!(setup.coupled, channels == 2);
            let types = setup.residues.each_ref().map(Residue::residue_type);
            assert_eq!(types, if channels == 2 { [2, 2] } else { [1, 1] });
        }
        let setup = Setup::new(2);
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
