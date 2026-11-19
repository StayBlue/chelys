// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Temporal noise shaping (clause 14): filters across the spectral lines
//! of each window of a channel that shape its quantization noise in time.
//! The encoder filters the lines with the prediction error filter of their
//! linear prediction over frequency (clause C.5, see [`analysis`]), and
//! sends the filters as the channel's `tns_data()` (Table 19), with which
//! the decoder applies their inverse.
//!
//! The decoder applies the filters after M/S stereo and intensity stereo,
//! before the filterbank (clause 4, figure 2): to each channel's spectrum
//! as left and right. So the encoder filters each channel's lines before
//! they are coded jointly, as figure 1 has it, and a pair's middle and side
//! channels, and its intensity signal, are made of the filtered lines.

pub mod analysis;

use chelys_core::bitstream::BitWriter;

use crate::channel::FRAME_LINES;
use crate::sampling_frequency::SamplingFrequency;
use crate::scalefactor_bands::Transform;
use crate::window::{SHORT_WINDOWS, Windows};

/// `TNS_MAX_ORDER` of the Low Complexity profile for long windows, the
/// highest of any window (clause 7.1.6): the most coefficients a filter
/// sends.
pub const MAX_ORDER: usize = 12;

/// The most filters of one window: those that `n_filt` of a long window, 2
/// bits, counts (Table 19).
pub const MAX_FILTERS: usize = 3;

/// `TNS_MAX_ORDER` of the Low Complexity profile (clause 7.1.6): the
/// highest order of a filter, 12 for long windows and 7 for short ones.
pub fn max_order(transform: Transform) -> usize {
    match transform {
        Transform::Long => MAX_ORDER,
        Transform::Short => 7,
    }
}

/// `TNS_MAX_BANDS` of the Low Complexity profile at `sampling_frequency`
/// (Table 33, as printed): the scalefactor bands of a window below which
/// the filters apply.
pub fn max_bands(sampling_frequency: SamplingFrequency, transform: Transform) -> usize {
    use SamplingFrequency::*;
    let (long, short) = match sampling_frequency {
        Hz96000 | Hz88200 => (31, 9),
        Hz64000 => (34, 10),
        Hz48000 => (40, 14),
        Hz44100 => (42, 14),
        Hz32000 => (51, 14),
        Hz24000 | Hz22050 => (46, 14),
        Hz16000 | Hz12000 | Hz11025 => (42, 14),
        Hz8000 => (39, 14),
    };
    match transform {
        Transform::Long => long,
        Transform::Short => short,
    }
}

/// The sizes of the fields of `tns_data()` that depend on the window
/// (clause 14.2.1): those of `n_filt`, `length` and `order`.
fn field_bits(transform: Transform) -> (u32, u32, u32) {
    match transform {
        Transform::Long => (2, 6, 5),
        Transform::Short => (1, 4, 3),
    }
}

/// One filter of a window (Table 19).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Filter {
    /// `length[w][filt]`: the scalefactor bands the filter spans, down from
    /// the top of the one before, or for the first from `num_swb`.
    pub length: u8,
    /// `order[w][filt]`: the filter's order, the coefficients it sends.
    pub order: u8,
    /// `direction[w][filt]`: whether the filter runs downward, from its
    /// highest line.
    pub downward: bool,
    /// `coef_compress[w][filt]`: whether each coefficient is sent in one bit
    /// fewer than `coef_res` gives.
    pub compress: bool,
    /// `coef[w][filt][i]` of the first `order`: each coefficient's index,
    /// signed.
    pub coef: [i8; MAX_ORDER],
}

impl Filter {
    /// A filter of order 0 over `length` bands, which filters nothing.
    pub const fn none(length: u8) -> Self {
        Self {
            length,
            order: 0,
            downward: false,
            compress: false,
            coef: [0; MAX_ORDER],
        }
    }

    /// The bits of one coefficient with `coef_res` bits of resolution.
    fn coef_bits(&self, coef_res: u8) -> u32 {
        u32::from(coef_res) - u32::from(self.compress)
    }
}

/// The filters of one window (Table 19).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowFilters {
    /// The bits of resolution of the coefficients, 3 or 4: `coef_res[w]`
    /// plus 3 (clause 14.3).
    pub coef_res: u8,
    /// `n_filt[w]`.
    pub n_filt: u8,
    pub filters: [Filter; MAX_FILTERS],
}

impl WindowFilters {
    /// A window without filters.
    pub const NONE: Self = Self {
        coef_res: 4,
        n_filt: 0,
        filters: [Filter::none(0); MAX_FILTERS],
    };

    /// The window's filters.
    pub fn filters(&self) -> &[Filter] {
        &self.filters[..usize::from(self.n_filt)]
    }
}

/// How far the decoder's filters raise the quantization noise of the
/// lines of each window that they span: the mean power gain of each
/// window's filter (see [`analysis::Shaping::shape`]), 1 where it has none.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseGains {
    /// Each window's gain, and the bands its filter spans.
    windows: [(f64, u8, u8); SHORT_WINDOWS],
}

impl NoiseGains {
    /// No window filtered.
    pub const NONE: Self = Self {
        windows: [(1.0, 0, 0); SHORT_WINDOWS],
    };

    /// The bands that window `w`'s filter spans: none where it has none.
    pub fn bands(&self, w: usize) -> std::ops::Range<usize> {
        let (_, from, to) = self.windows[w];
        usize::from(from)..usize::from(to)
    }

    /// Window `w`'s filter raises the noise of `bands` by `gain`.
    pub fn set(&mut self, w: usize, gain: f64, bands: std::ops::Range<usize>) {
        self.windows[w] = (gain, bands.start as u8, bands.end as u8);
    }

    /// The gain of each line of a frame of `windows`, in the order they are
    /// sent, into `weights`.
    pub fn weights(
        &self,
        windows: Windows,
        sampling_frequency: SamplingFrequency,
        weights: &mut [f64; FRAME_LINES],
    ) {
        weights.fill(1.0);
        let mut first = 0;
        for (start, bands) in windows.groups_with_start(sampling_frequency) {
            let group = &mut weights[start..start + bands.lines()];
            for w in 0..bands.windows() {
                let (gain, from, to) = self.windows[first + w];
                for sfb in usize::from(from)..usize::from(to).min(bands.num_swb()) {
                    group[bands.window_band(w, sfb)].fill(gain);
                }
            }
            first += bands.windows();
        }
    }
}

/// A channel's temporal noise shaping in a frame: the filters of each of
/// its windows, which its `tns_data_present` and `tns_data()` send.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tns {
    transform: Transform,
    windows: [WindowFilters; SHORT_WINDOWS],
}

impl Tns {
    /// No filter in any window: `tns_data_present` 0.
    pub const NONE: Self = Self {
        transform: Transform::Long,
        windows: [WindowFilters::NONE; SHORT_WINDOWS],
    };

    /// The windows of `transform`, `num_windows` of them, without filters
    /// as yet.
    pub fn new(transform: Transform) -> Self {
        Self {
            transform,
            ..Self::NONE
        }
    }

    /// `num_windows`: one long window or eight short ones.
    fn num_windows(&self) -> usize {
        match self.transform {
            Transform::Long => 1,
            Transform::Short => SHORT_WINDOWS,
        }
    }

    /// The filters of each window.
    pub fn windows(&self) -> &[WindowFilters] {
        &self.windows[..self.num_windows()]
    }

    /// The filters of window `w`.
    pub fn window_mut(&mut self, w: usize) -> &mut WindowFilters {
        let windows = self.num_windows();
        &mut self.windows[..windows][w]
    }

    /// `tns_data_present`: whether any window has a filter.
    pub fn is_present(&self) -> bool {
        self.windows().iter().any(|window| window.n_filt > 0)
    }

    /// The bits of `tns_data_present` and, where it is 1, `tns_data()`, as
    /// [`write`](Self::write) writes them.
    pub fn bits(&self) -> u32 {
        if !self.is_present() {
            return 1;
        }
        let (n_filt, length, order) = field_bits(self.transform);
        let windows: u32 = (self.windows().iter())
            .map(|window| {
                let filters: u32 = (window.filters().iter())
                    .map(|filter| {
                        let coefs = if filter.order > 0 {
                            2 + u32::from(filter.order) * filter.coef_bits(window.coef_res)
                        } else {
                            0
                        };
                        length + order + coefs
                    })
                    .sum();
                n_filt + u32::from(window.n_filt > 0) + filters
            })
            .sum();
        1 + windows
    }

    /// Writes `tns_data_present` and, where it is 1, `tns_data()` (Table
    /// 19).
    ///
    /// # Panics
    ///
    /// If a field does not fit its size, or a coefficient its bits: a
    /// caller's error.
    pub fn write(&self, w: &mut BitWriter) {
        w.write_bool(self.is_present());
        if !self.is_present() {
            return;
        }
        let (n_filt_bits, length_bits, order_bits) = field_bits(self.transform);
        let fits = |value: u8, bits: u32| u32::from(value) < 1 << bits;
        for window in self.windows() {
            assert!(fits(window.n_filt, n_filt_bits), "n_filt {}", window.n_filt);
            w.write(u32::from(window.n_filt), n_filt_bits);
            if window.n_filt == 0 {
                continue;
            }
            assert!(
                matches!(window.coef_res, 3 | 4),
                "coef_res {}",
                window.coef_res
            );
            w.write(u32::from(window.coef_res - 3), 1);
            for filter in window.filters() {
                assert!(fits(filter.length, length_bits), "length {}", filter.length);
                assert!(
                    fits(filter.order, order_bits) && usize::from(filter.order) <= MAX_ORDER,
                    "order {}",
                    filter.order
                );
                w.write(u32::from(filter.length), length_bits);
                w.write(u32::from(filter.order), order_bits);
                if filter.order == 0 {
                    continue;
                }
                w.write_bool(filter.downward);
                w.write_bool(filter.compress);
                let bits = filter.coef_bits(window.coef_res);
                let half = 1i8 << (bits - 1);
                for &coef in &filter.coef[..usize::from(filter.order)] {
                    assert!((-half..half).contains(&coef), "coef {coef} in {bits} bits");
                    w.write(coef as u32 & ((1 << bits) - 1), bits);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use chelys_core::bitstream::binary_digits;

    use super::*;
    use crate::scalefactor_bands::{num_swb, swb_offsets};

    fn written(tns: &Tns) -> String {
        let mut w = BitWriter::new();
        tns.write(&mut w);
        let len = w.bit_len();
        assert_eq!(len as u32, tns.bits());
        binary_digits(&w.into_bytes())[..len].to_owned()
    }

    #[test]
    fn no_filter_sends_the_flag_alone() {
        assert_eq!(written(&Tns::NONE), "0");
        assert_eq!(written(&Tns::new(Transform::Short)), "0");
    }

    #[test]
    fn a_long_window_sends_its_filters_in_the_fields_of_table_19() {
        let mut tns = Tns::new(Transform::Long);
        let mut filter = Filter::none(36);
        filter.order = 3;
        filter.coef[..3].copy_from_slice(&[7, -8, -1]);
        *tns.window_mut(0) = WindowFilters {
            coef_res: 4,
            n_filt: 2,
            filters: [Filter::none(7), filter, Filter::none(0)],
        };
        // tns_data_present, n_filt 2, coef_res 1; a filter of length 7 and
        // order 0; one of length 36 and order 3, upward, uncompressed, and
        // its coefficients in 4 bits, two's complement.
        assert_eq!(
            written(&tns),
            [
                "1", "10", "1", "000111", "00000", "100100", "00011", "0", "0", "0111", "1000",
                "1111"
            ]
            .concat()
        );
    }

    #[test]
    fn short_windows_each_send_theirs() {
        let mut tns = Tns::new(Transform::Short);
        let mut filter = Filter::none(11);
        filter.order = 2;
        filter.downward = true;
        filter.compress = true;
        filter.coef[..2].copy_from_slice(&[1, -2]);
        *tns.window_mut(2) = WindowFilters {
            coef_res: 3,
            n_filt: 1,
            filters: [filter, Filter::none(0), Filter::none(0)],
        };
        // tns_data_present; n_filt 0 for windows 0 and 1; window 2: n_filt
        // 1, coef_res 0, length 11, order 2, downward, compressed, and its
        // coefficients in 2 bits; n_filt 0 for the other five.
        assert_eq!(
            written(&tns),
            [
                "1", "0", "0", "1", "0", "1011", "010", "1", "1", "01", "10", "00000"
            ]
            .concat()
        );
    }

    #[test]
    #[should_panic(expected = "coef 4 in 3 bits")]
    fn a_coefficient_beyond_its_bits_is_refused() {
        let mut tns = Tns::new(Transform::Long);
        let mut filter = Filter::none(10);
        filter.order = 1;
        filter.compress = true;
        filter.coef[0] = 4;
        tns.window_mut(0).n_filt = 1;
        tns.window_mut(0).filters[0] = filter;
        written(&tns);
    }

    #[test]
    fn the_limits_are_those_of_the_low_complexity_profile() {
        use SamplingFrequency::*;
        // Table 33, Low Complexity columns.
        let printed = [
            (Hz96000, 31, 9),
            (Hz88200, 31, 9),
            (Hz64000, 34, 10),
            (Hz48000, 40, 14),
            (Hz44100, 42, 14),
            (Hz32000, 51, 14),
            (Hz24000, 46, 14),
            (Hz22050, 46, 14),
            (Hz16000, 42, 14),
            (Hz12000, 42, 14),
            (Hz11025, 42, 14),
            (Hz8000, 39, 14),
        ];
        for (fs, long, short) in printed {
            assert_eq!(max_bands(fs, Transform::Long), long, "{fs:?}");
            assert_eq!(max_bands(fs, Transform::Short), short, "{fs:?}");
            // Within the bands of each window, and the fields that count
            // them.
            for transform in [Transform::Long, Transform::Short] {
                let bands = max_bands(fs, transform);
                assert!(bands <= num_swb(fs, transform), "{fs:?} {transform:?}");
                let (_, length, _) = field_bits(transform);
                assert!(num_swb(fs, transform) < 1 << length);
                assert!(swb_offsets(fs, transform)[bands] > 0);
            }
        }
        // Clause 7.1.6.
        assert_eq!(max_order(Transform::Long), 12);
        assert_eq!(max_order(Transform::Short), 7);
        let (_, _, order) = field_bits(Transform::Short);
        assert!(max_order(Transform::Short) < 1 << order);
    }
}
