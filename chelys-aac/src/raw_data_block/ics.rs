// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The `individual_channel_stream()` (Table 16) and its `ics_info()`
//! (Table 15), of a channel whose spectrum is all zero.

use chelys_core::bitstream::BitWriter;

/// `window_sequence` (Table 44).
const ONLY_LONG_SEQUENCE: u32 = 0;

/// `window_shape`: 0 for the sine window, 1 for the Kaiser-Bessel derived
/// one (clause 15.3.2).
///
/// Spec gap: the encoder chooses the shape (clause C.3.1.1), and Annex C
/// gives no rule for it. A channel without spectrum decodes alike with
/// either; the sine window is taken until the filterbank chooses.
const SINE_WINDOW: u32 = 0;

/// The `global_gain` of a channel without spectrum.
///
/// Spec gap: `global_gain` is the start value of the scalefactors (clause
/// 11.3.2), of which a channel that transmits no scalefactor band has
/// none, and Annex C's start value of the quantiser step (C.7.2.1) is not
/// defined for a spectrum without a line above zero. Any value decodes
/// alike; 0 is taken. Settled by decoding with ffmpeg (black box): it
/// decodes channels so coded, at every sampling frequency, to digital
/// silence without complaint; the conformance test is
/// `ffmpeg_decodes_silence_at_every_sample_rate`.
const SILENT_GLOBAL_GAIN: u32 = 0;

/// Writes the `ics_info()` (Table 15) of a long window that transmits no
/// scalefactor band.
pub fn write_silent_ics_info(w: &mut BitWriter) {
    w.write(0, 1); // ics_reserved_bit: "Shall be '0'" (clause 8.3.1.1)
    w.write(ONLY_LONG_SEQUENCE, 2); // window_sequence
    w.write(SINE_WINDOW, 1); // window_shape
    // Not EIGHT_SHORT_SEQUENCE:
    w.write(0, 6); // max_sfb
    // predictor_data_present: prediction is not permitted in the Low
    // Complexity profile (clause 7.1.2).
    w.write(0, 1);
}

/// Writes the `individual_channel_stream()` (Table 16) of a channel whose
/// spectrum is all zero: a long window with `max_sfb` 0.
///
/// With `max_sfb` 0 the loops of `section_data()` (Table 17),
/// `scale_factor_data()` (Table 18) and `spectral_data()` (Table 20) run
/// for no scalefactor band and write nothing: "The spectral information for
/// all scalefactor bands at and above max_sfb, for which there is no
/// section data, is zero" (clause 9.3).
pub fn write_silent_individual_channel_stream(w: &mut BitWriter, common_window: bool) {
    w.write(SILENT_GLOBAL_GAIN, 8); // global_gain
    if !common_window {
        write_silent_ics_info(w);
    }
    // section_data(), scale_factor_data(): empty.
    w.write(0, 1); // pulse_data_present
    w.write(0, 1); // tns_data_present
    // gain_control_data_present: gain control is not permitted in the Low
    // Complexity profile (clause 7.1.2).
    w.write(0, 1);
    // spectral_data(): empty.
}

#[cfg(test)]
mod tests {
    use chelys_core::bitstream::binary_digits;

    use super::*;

    fn written(write: impl FnOnce(&mut BitWriter)) -> (String, usize) {
        let mut w = BitWriter::new();
        write(&mut w);
        let len = w.bit_len();
        (binary_digits(&w.into_bytes())[..len].to_owned(), len)
    }

    #[test]
    fn ics_info_is_a_long_window_without_bands() {
        let (bits, len) = written(write_silent_ics_info);
        // ics_reserved_bit, window_sequence, window_shape, max_sfb,
        // predictor_data_present.
        assert_eq!(bits, ["0", "00", "0", "000000", "0"].concat());
        assert_eq!(len, 11);
    }

    #[test]
    fn a_stream_holds_its_ics_info_unless_the_window_is_common() {
        let (own, _) = written(|w| write_silent_individual_channel_stream(w, false));
        // global_gain, ics_info(), pulse, tns and gain control flags.
        assert_eq!(own, ["00000000", "00000000000", "000"].concat());
        let (common, _) = written(|w| write_silent_individual_channel_stream(w, true));
        assert_eq!(common, ["00000000", "000"].concat());
    }
}
