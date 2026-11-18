// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The bit reservoir (clause C.1.5.4.2.2): main data bits a granule leaves
//! unused are saved for later granules, and handed out again where the
//! psychoacoustic model asks for more than the average.

use crate::header::Version;

/// The decoder's input buffer, which must hold a frame and the reservoir
/// before it (clause 2.4.3.4.4).
pub const BUFFER_BITS: usize = 7680;

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
///
/// The reservoir has two limits. Between frames it holds what the frames
/// the stream may still have could begin in: no more than the limit of the
/// shortest of them. Once a frame's length is chosen,
/// [`for_frame`](Self::for_frame) lowers that to the frame's own limit.
#[derive(Clone, Debug)]
pub struct Reservoir {
    bits: isize,
    /// The limit between frames.
    stream_max_bytes: usize,
    /// The limit of the frame being coded.
    max_bytes: usize,
}

impl Reservoir {
    /// A reservoir for frames of `version` of at most `largest_frame_bytes`,
    /// bounded by the decoder's buffer and the largest `main_data_begin`.
    ///
    /// Spec reading: clause 2.4.3.4.4 bounds the reservoir by the 7680-bit
    /// buffer less the frame ("a maximum deviation (short time buffer) of
    /// 7 680 - 4*768 = 4 608 bits is allowed at 64 kbits/s"), and asks for
    /// the same buffer at every sampling frequency. Where the frame's length
    /// varies with padding, its longest, with padding, is used, so that one
    /// limit holds for every frame at a bitrate.
    ///
    /// Spec reading: the clause has `main_data_begin` "always set to 0" at
    /// `bitrate_index` 14, 320 kbit/s, where at 48 kHz a frame fills the
    /// buffer. Below 48 kHz a frame at the highest bitrates is longer than
    /// the buffer: 1044 or 1045 bytes at 320 kbit/s and 44.1 kHz, and 1440
    /// at 32 kHz, where the buffer holds 960. As the bitrate table offers
    /// those bitrates, the clause is not read as forbidding them; their
    /// frames are coded as at 48 kHz, with all their main data "allocated
    /// between adjacent header words".
    ///
    /// Spec gap: ISO/IEC 13818-3 does not restate the buffer for the lower
    /// sampling frequencies, and refers to ISO/IEC 11172-3 for
    /// `main_data_begin`, so the same buffer is assumed there.
    pub fn new(largest_frame_bytes: usize, version: Version) -> Self {
        let max_main_data_begin = (1 << version.main_data_begin_bits()) - 1;
        let max_bytes = Self::buffer_bytes(largest_frame_bytes).min(max_main_data_begin);
        Self {
            bits: 0,
            stream_max_bytes: max_bytes,
            max_bytes,
        }
    }

    /// A reservoir that never holds anything: every frame's main data starts
    /// right after its side information.
    #[cfg(any(test, feature = "unstable"))]
    pub fn disabled() -> Self {
        Self {
            bits: 0,
            stream_max_bytes: 0,
            max_bytes: 0,
        }
    }

    /// A reservoir with room for whatever a granule leaves and nothing to
    /// lend: a granule given it is limited only by its own share, and need
    /// use none of it.
    pub fn unbounded() -> Self {
        Self {
            bits: 0,
            stream_max_bytes: usize::MAX / 16,
            max_bytes: usize::MAX / 16,
        }
    }

    /// What the buffer leaves of its 7680 bits beside a frame of
    /// `frame_bytes`, in whole bytes.
    fn buffer_bytes(frame_bytes: usize) -> usize {
        BUFFER_BITS.saturating_sub(8 * frame_bytes) / 8
    }

    /// The reservoir as it starts a frame of at most `largest_frame_bytes`:
    /// limited by what the buffer leaves beside the frame (see
    /// [`new`](Self::new)), with the bytes beyond that left behind as
    /// stuffing. Its [`bytes`](Self::bytes) are then the frame's
    /// `main_data_begin`.
    ///
    /// Spec gap: clause 2.4.3.4.4 works out the bound for frames of one
    /// length, as at a constant bitrate, where it is the same for every
    /// frame. Where the bitrate changes from frame to frame, the bound is
    /// taken from each frame's own length, as the buffer holds the frame
    /// being decoded beside the main data it begins in (clause 2.4.3.4.3:
    /// "all main data are resident in the input buffer when the header of
    /// the next frame is arriving"); so the bytes a frame finds are limited
    /// only once its bitrate has been chosen.
    pub fn for_frame(&self, largest_frame_bytes: usize) -> Self {
        let max_bytes = Self::buffer_bytes(largest_frame_bytes).min(self.stream_max_bytes);
        Self {
            bits: self.bits.min(8 * max_bytes as isize),
            stream_max_bytes: self.stream_max_bytes,
            max_bytes,
        }
    }

    /// The largest number of bytes the reservoir may hold in the frame
    /// being coded.
    #[cfg(any(test, feature = "unstable"))]
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

    /// The fewest bits a granule with the share `mean_bits` must use for the
    /// reservoir to hold what it leaves unused: below that, the bits it
    /// leaves would end up as stuffing.
    pub fn least_use(&self, mean_bits: usize) -> usize {
        let room = 8 * self.max_bytes as isize - self.bits;
        usize::try_from(mean_bits as isize - room).unwrap_or(0)
    }

    /// Accounts for a granule that used `used` bits of main data, against
    /// its share `mean_bits`: what it left unused goes to the reservoir, what
    /// it took beyond its share comes out of it.
    pub fn granule_done(&mut self, mean_bits: usize, used: usize) {
        self.bits += mean_bits as isize - used as isize;
    }

    /// Ends a frame whose main data slot held `unshared_bits` more than the
    /// granules' shares, and returns the bytes the reservoir keeps for the
    /// frames after it: the next frame's `main_data_begin`, unless its
    /// length lowers it (see [`for_frame`](Self::for_frame)).
    ///
    /// The reservoir keeps whole bytes only, since main data starts on a
    /// byte. Bytes beyond its limit between frames are left behind as
    /// stuffing (clause C.1.5.4.2.2): the next frame's main data simply
    /// starts after them.
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
        let bytes = self.bytes().min(self.stream_max_bytes);
        self.bits = 8 * bytes as isize;
        self.max_bytes = self.stream_max_bytes;
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
    fn a_granule_may_leave_only_what_the_reservoir_has_room_for() {
        let mut reservoir = Reservoir::new(800, Version::Mpeg1);
        assert_eq!(reservoir.max_bytes(), 160);
        assert_eq!(reservoir.least_use(1000), 0);
        reservoir.granule_done(1000, 600);
        // 400 bits held, 880 of room left.
        assert_eq!(reservoir.least_use(1000), 120);
        assert_eq!(reservoir.least_use(500), 0);
        assert_eq!(Reservoir::disabled().least_use(500), 500);
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

    /// Between frames the reservoir keeps what the shortest frame could
    /// begin in; a longer frame finds only what the buffer leaves beside
    /// it, and the rest becomes stuffing.
    #[test]
    fn a_longer_frame_finds_less_of_the_reservoir() {
        // 32 kbit/s at 48 kHz: 96-byte frames, so main_data_begin's 511.
        let mut reservoir = Reservoir::new(96, Version::Mpeg1);
        reservoir.granule_done(8 * 600, 0);
        assert_eq!(reservoir.end_frame(0), 511);
        // 192 kbit/s: 576 bytes leave 384 of the buffer's 960.
        let frame = reservoir.for_frame(576);
        assert_eq!((frame.bytes(), frame.max_bytes()), (384, 384));
        // 320 kbit/s fills the buffer.
        assert_eq!(reservoir.for_frame(960).bytes(), 0);
        // The frame does not change what the stream keeps.
        let mut frame = reservoir.for_frame(576);
        frame.granule_done(0, 0);
        assert_eq!(frame.end_frame(8 * 400), 511);
        assert_eq!(frame.max_bytes(), 511);
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
