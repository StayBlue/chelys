// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The bit reservoir (clause C.1.5.4.2.2): main data bits a granule leaves
//! unused are saved for later granules, and handed out again where the
//! psychoacoustic model asks for more than the average.

use crate::mp3::header::Version;

/// The decoder's input buffer, which must hold a frame and the reservoir
/// before it (clause 2.4.3.4.4).
const BUFFER_BITS: usize = 7680;

/// The reservoir's fill, above this share of its maximum, is released.
const RELEASE_ABOVE: f64 = 0.8;

/// `more_bits = 3.1 * PE - mean_bits` (clause C.1.5.4.1); bits are borrowed
/// only when this exceeds 100.
const PE_BITS: f64 = 3.1;
const BORROW_ABOVE: f64 = 100.0;

/// The share of the reservoir a granule asking for more bits may take.
const BORROW_SHARE: f64 = 0.6;

/// Tracks the reservoir across granules and frames.
///
/// The fill is kept in bits, and can dip below zero within a frame for
/// granules quantised without regard to it; a frame must end with a fill
/// of zero or more.
#[derive(Clone, Debug)]
pub struct Reservoir {
    bits: isize,
    max_bytes: usize,
}

impl Reservoir {
    /// A reservoir for frames of `version` of at most `largest_frame_bytes`,
    /// bounded by the decoder's buffer and the largest `main_data_begin`.
    ///
    /// Spec reading: clause 2.4.3.4.4 bounds the reservoir by the 7680-bit
    /// buffer less the frame, and asks for the same buffer at every sampling
    /// frequency. The largest frame, with padding, is used so that every
    /// frame fits.
    ///
    /// Spec gap: ISO/IEC 13818-3 does not restate the buffer for the lower
    /// sampling frequencies, and refers to ISO/IEC 11172-3 for
    /// `main_data_begin`, so the same buffer is assumed there.
    pub fn new(largest_frame_bytes: usize, version: Version) -> Self {
        let max_main_data_begin = (1 << version.main_data_begin_bits()) - 1;
        let max_bytes =
            (BUFFER_BITS.saturating_sub(8 * largest_frame_bytes) / 8).min(max_main_data_begin);
        Self { bits: 0, max_bytes }
    }

    /// A reservoir that never holds anything: every frame's main data starts
    /// right after its side information.
    pub fn disabled() -> Self {
        Self {
            bits: 0,
            max_bytes: 0,
        }
    }

    /// The largest number of bytes the reservoir may hold.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// The reservoir's whole bytes, `main_data_end` in clause C.1.5.4.2.2.
    /// At the start of a frame this is its `main_data_begin`.
    pub fn bytes(&self) -> usize {
        usize::try_from(self.bits).unwrap_or(0) / 8
    }

    /// The bits made available to a granule, `max_bits`: its share of the
    /// frame, `mean_bits`, and what the rules of clause C.1.5.4.2.2 take from
    /// the reservoir for it.
    ///
    /// - When the reservoir holds more than 0.8 of its maximum, the bytes
    ///   above that are released.
    /// - When `more_bits = 3.1 * PE - mean_bits` exceeds 100, the bytes taken
    ///   are the larger of `more_bits / 8` and 0.6 of the reservoir, as
    ///   printed.
    ///
    /// Spec reading: the clause gives both rules "in addition to mean_bits";
    /// they are added, and together limited to what the reservoir holds.
    pub fn granule_bits(&self, mean_bits: usize, perceptual_entropy: f64) -> usize {
        let held = self.bytes();
        let release_above = (RELEASE_ABOVE * self.max_bytes as f64) as usize;
        let mut extra = held.saturating_sub(release_above);
        let more_bits = PE_BITS * perceptual_entropy - mean_bits as f64;
        if more_bits > BORROW_ABOVE {
            extra += f64::max(more_bits / 8.0, BORROW_SHARE * held as f64) as usize;
        }
        mean_bits + 8 * extra.min(held)
    }

    /// Accounts for a granule that used `used` bits of main data, against
    /// its share `mean_bits`: what it left unused goes to the reservoir, what
    /// it took beyond its share comes out of it.
    pub fn granule_done(&mut self, mean_bits: usize, used: usize) {
        self.bits += mean_bits as isize - used as isize;
    }

    /// Ends a frame whose main data slot held `unshared_bits` more than the
    /// granules' shares, and returns the next frame's `main_data_begin`.
    ///
    /// The reservoir keeps whole bytes only, since main data starts on a
    /// byte. Bytes beyond its maximum are left behind as stuffing (clause
    /// C.1.5.4.2.2): the next frame's main data simply starts after them.
    ///
    /// # Panics
    ///
    /// If the frame's granules used more bits than the frame and the
    /// reservoir held.
    pub fn end_frame(&mut self, unshared_bits: usize) -> usize {
        self.bits += unshared_bits as isize;
        assert!(
            self.bits >= 0,
            "a frame's granules overflow it by {} bits",
            -self.bits
        );
        let bytes = self.bytes().min(self.max_bytes);
        self.bits = 8 * bytes as isize;
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At 320 kbit/s and 48 kHz a frame fills the buffer, so the reservoir
    /// is always empty (clause 2.4.3.4.4); at 64 kbit/s it is limited by
    /// `main_data_begin`, which is narrower at the lower sampling
    /// frequencies.
    #[test]
    fn maximum_follows_the_buffer_and_the_field() {
        assert_eq!(Reservoir::new(960, Version::Mpeg1).max_bytes(), 0);
        assert_eq!(Reservoir::new(192, Version::Mpeg1).max_bytes(), 511);
        assert_eq!(Reservoir::new(192, Version::Mpeg2).max_bytes(), 255);
        // 160 kbit/s at 32 or 16 kHz: 720-byte frames leave 240 bytes.
        assert_eq!(Reservoir::new(720, Version::Mpeg1).max_bytes(), 240);
        assert_eq!(Reservoir::new(720, Version::Mpeg2).max_bytes(), 240);
    }

    #[test]
    fn unused_bits_are_saved_and_lent() {
        let mut reservoir = Reservoir::new(400, Version::Mpeg1);
        assert_eq!(reservoir.granule_bits(1000, 0.0), 1000);
        reservoir.granule_done(1000, 600);
        reservoir.granule_done(1000, 1000);
        assert_eq!(reservoir.end_frame(5), 50);
        // A low entropy borrows nothing.
        assert_eq!(reservoir.granule_bits(1000, 300.0), 1000);
        // 3.1 * 380 - 1000 = 178 bits asked for; 0.6 of 50 bytes is more.
        assert_eq!(reservoir.granule_bits(1000, 380.0), 1000 + 8 * 30);
        // A large demand is limited to what the reservoir holds.
        assert_eq!(reservoir.granule_bits(1000, 5000.0), 1000 + 8 * 50);
    }

    #[test]
    fn a_full_reservoir_releases_its_top_and_stuffs_the_excess() {
        let mut reservoir = Reservoir::new(800, Version::Mpeg1);
        assert_eq!(reservoir.max_bytes(), 160);
        reservoir.granule_done(3000, 0);
        assert_eq!(reservoir.end_frame(0), 160);
        // 0.8 * 160 = 128 bytes are kept; 32 are released.
        assert_eq!(reservoir.granule_bits(500, 0.0), 500 + 8 * 32);
    }

    #[test]
    fn a_disabled_reservoir_lends_nothing() {
        let mut reservoir = Reservoir::disabled();
        reservoir.granule_done(100, 40);
        assert_eq!(reservoir.end_frame(0), 0);
        assert_eq!(reservoir.granule_bits(100, 1e6), 100);
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn a_frame_cannot_end_in_debt() {
        let mut reservoir = Reservoir::disabled();
        reservoir.granule_done(100, 101);
        reservoir.end_frame(0);
    }
}
