// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The pulse escape (clause 9.1): up to four lines of a long window sent in
//! `spectral_data()` with smaller magnitudes, and the amounts by which the
//! decoder restores them, in `pulse_data()` (Table 21, clause 9.3).

use chelys_core::bitstream::BitWriter;

use crate::scalefactor_bands::{GroupBands, Transform};

/// The most pulses `pulse_data()` sends: `number_pulse` + 1, of 2 bits
/// (Table 21), "from 1 to 4" (clause 9.2.1).
pub const MAX_PULSES: usize = 4;

/// The largest `pulse_offset`, of 5 bits (Table 21).
pub const MAX_OFFSET: usize = (1 << 5) - 1;

/// The largest `pulse_amp`, of 4 bits (Table 21).
pub const MAX_AMP: u32 = (1 << 4) - 1;

/// The bits of `pulse_data()` besides its pulses: `number_pulse` and
/// `pulse_start_sfb` (Table 21).
const HEADER_BITS: u32 = 2 + 6;

/// The bits of each pulse: `pulse_offset` and `pulse_amp` (Table 21).
const PULSE_BITS: u32 = 5 + 4;

/// The pulses of a channel's long window, which its `pulse_data_present`
/// and `pulse_data()` send: none, or 1 to [`MAX_PULSES`], each a line and
/// the magnitude the decoder adds to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pulses {
    count: u8,
    /// `pulse_start_sfb`, and `swb_offset[pulse_start_sfb]`, the line the
    /// first offset counts from.
    start_sfb: u8,
    base: u16,
    /// Each pulse's line, in ascending order, and its `pulse_amp`.
    lines: [u16; MAX_PULSES],
    amps: [u8; MAX_PULSES],
}

impl Pulses {
    /// No pulse: `pulse_data_present` 0.
    pub const NONE: Self = Self {
        count: 0,
        start_sfb: 0,
        base: 0,
        lines: [0; MAX_PULSES],
        amps: [0; MAX_PULSES],
    };

    /// Pulses at `pulses`, each a line of a long window of `bands` and its
    /// `pulse_amp`, in ascending order of their lines, with
    /// `pulse_start_sfb` the band of the first: or `None` where
    /// `pulse_data()` cannot send them. It sends 1 to [`MAX_PULSES`], each
    /// a `pulse_amp` of up to [`MAX_AMP`], the first's line up to
    /// [`MAX_OFFSET`] lines above the start of its band and each other's up
    /// to as many above the one before (clause 9.3), and only in a long
    /// window: "the pulse escape method is illegal for a block whose
    /// window_sequence is EIGHT_SHORT_SEQUENCE" (clause 9.3).
    ///
    /// Spec reading: a `pulse_offset` of 0 after the first would restore
    /// the line before it again. Each line is taken to be restored once,
    /// so that each pulse lies above the one before.
    pub fn new(bands: GroupBands, pulses: &[(usize, u32)]) -> Option<Self> {
        if bands.transform() != Transform::Long || !(1..=MAX_PULSES).contains(&pulses.len()) {
            return None;
        }
        let first = pulses[0].0;
        let start_sfb = (0..bands.num_swb()).find(|&sfb| bands.band(sfb).contains(&first))?;
        let base = bands.offset(start_sfb);
        let mut result = Self {
            count: pulses.len() as u8,
            start_sfb: start_sfb as u8,
            base: base as u16,
            ..Self::NONE
        };
        let mut previous = base;
        for (i, &(line, amp)) in pulses.iter().enumerate() {
            let reach = if i == 0 {
                0..=MAX_OFFSET
            } else {
                1..=MAX_OFFSET
            };
            if !reach.contains(&line.checked_sub(previous)?) || amp > MAX_AMP {
                return None;
            }
            previous = line;
            result.lines[i] = line as u16;
            result.amps[i] = amp as u8;
        }
        Some(result)
    }

    /// `pulse_data_present`.
    pub fn is_present(&self) -> bool {
        self.count > 0
    }

    /// Each pulse's line in the window and its `pulse_amp`, in ascending
    /// order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, u32)> + '_ {
        let count = usize::from(self.count);
        (self.lines[..count].iter())
            .zip(&self.amps[..count])
            .map(|(&line, &amp)| (usize::from(line), u32::from(amp)))
    }

    /// `pulse_start_sfb`.
    #[cfg(test)]
    pub fn start_sfb(&self) -> usize {
        usize::from(self.start_sfb)
    }

    /// The bits of `pulse_data_present` and, where it is 1, `pulse_data()`,
    /// as [`write`](Self::write) writes them.
    pub fn bits(&self) -> u32 {
        if self.is_present() {
            1 + HEADER_BITS + u32::from(self.count) * PULSE_BITS
        } else {
            1
        }
    }

    /// Writes `pulse_data_present` and, where it is 1, `pulse_data()`
    /// (Table 21): `number_pulse`, `pulse_start_sfb`, and each pulse's
    /// `pulse_offset`, from the start of the band or the pulse before, and
    /// `pulse_amp`.
    pub fn write(&self, w: &mut BitWriter) {
        w.write_bool(self.is_present());
        if !self.is_present() {
            return;
        }
        w.write(u32::from(self.count - 1), 2);
        w.write(u32::from(self.start_sfb), 6);
        let mut previous = usize::from(self.base);
        for (line, amp) in self.iter() {
            w.write((line - previous) as u32, 5);
            w.write(amp, 4);
            previous = line;
        }
    }

    /// The lines that `spectral_data()` sends for the quantized `lines` of
    /// a long window: each pulse's line with `pulse_amp` less in magnitude,
    /// keeping its sign, which the decoder restores by adding `pulse_amp`
    /// to a line above 0 and subtracting it from any other (clause 9.3).
    ///
    /// # Panics
    ///
    /// If a pulse's line would not keep its sign, and so be restored: a
    /// caller's error.
    pub fn clip(&self, lines: &mut [i32]) {
        for (line, amp) in self.iter() {
            let q = lines[line];
            let sent = q - q.signum() * amp as i32;
            assert!(
                amp == 0 || (q != 0 && sent.signum() == q.signum()),
                "a pulse of {amp} that leaves line {line}, {q}, without its sign"
            );
            lines[line] = sent;
        }
    }
}

#[cfg(test)]
mod tests {
    use chelys_core::bitstream::binary_digits;

    use super::*;
    use crate::sampling_frequency::SamplingFrequency;

    fn written(pulses: &Pulses) -> String {
        let mut w = BitWriter::new();
        pulses.write(&mut w);
        let len = w.bit_len();
        assert_eq!(len as u32, pulses.bits());
        binary_digits(&w.into_bytes())[..len].to_owned()
    }

    #[test]
    fn no_pulse_is_a_flag_of_0() {
        assert_eq!(written(&Pulses::NONE), "0");
    }

    #[test]
    fn pulses_count_their_offsets_from_the_band_then_from_each_other() {
        // At 44.1 kHz, band 33 spans lines 448 to 480 (Table 45).
        let bands = GroupBands::long(SamplingFrequency::Hz44100);
        assert_eq!(bands.band(33), 448..480);
        let pulses = Pulses::new(bands, &[(450, 1), (481, 15), (490, 7)]).unwrap();
        assert_eq!(pulses.start_sfb(), 33);
        assert_eq!(
            written(&pulses),
            [
                "1", "10", "100001", "00010", "0001", "11111", "1111", "01001", "0111"
            ]
            .concat()
        );
    }

    #[test]
    fn pulses_beyond_the_fields_reach_are_refused() {
        let fs = SamplingFrequency::Hz8000;
        let long = GroupBands::long(fs);
        // At 8 kHz, band 39 spans lines 944 to 1024 (Table 48).
        assert_eq!(long.band(39), 944..1024);
        assert!(Pulses::new(long, &[(975, 1)]).is_some());
        assert_eq!(Pulses::new(long, &[(976, 1)]), None);
        assert_eq!(Pulses::new(long, &[(944, 16)]), None);
        assert_eq!(Pulses::new(long, &[]), None);
        assert_eq!(Pulses::new(long, &[(1, 1); 5]), None);
        // Each above the one before, by 31 at most.
        assert!(Pulses::new(long, &[(0, 1), (31, 1), (62, 1), (93, 1)]).is_some());
        assert_eq!(Pulses::new(long, &[(0, 1), (32, 1)]), None);
        assert_eq!(Pulses::new(long, &[(5, 1), (5, 1)]), None);
        assert_eq!(Pulses::new(long, &[(5, 1), (4, 1)]), None);
        assert_eq!(Pulses::new(long, &[(1024, 1)]), None);
        // Never in short windows.
        assert_eq!(Pulses::new(GroupBands::short(fs, 1), &[(0, 1)]), None);
    }

    #[test]
    fn clipping_keeps_each_lines_sign() {
        let bands = GroupBands::long(SamplingFrequency::Hz48000);
        let pulses = Pulses::new(bands, &[(3, 2), (9, 15)]).unwrap();
        let mut lines = [0; 1024];
        lines[3] = -3;
        lines[9] = 16;
        pulses.clip(&mut lines);
        assert_eq!((lines[3], lines[9]), (-1, 1));
    }
}
