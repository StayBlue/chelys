// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! A decoder of the audio channel elements, written from the syntax of
//! Tables 13 to 20 and from clauses 7.1.6, 8.3.5, 9.3, 10.3, 11.3, 12.1.3,
//! 12.2.3 and 14.3 apart from the encoder, for tests to decode what it
//! writes: the fields of a `single_channel_element()` or
//! `channel_pair_element()`, and the spectra they reconstruct. It shares
//! only the codebooks and the scalefactor band tables.

use super::codebooks::{Codebook, Codeword, SCALEFACTOR_CODEWORDS};
use super::tables;
use crate::sampling_frequency::SamplingFrequency;
use crate::scalefactor_bands::{Transform, swb_offsets};

pub struct BitReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn read(&mut self, bits: u32) -> u32 {
        (0..bits).fold(0, |value, _| {
            let byte = self.bytes[self.position / 8];
            let bit = (byte >> (7 - self.position % 8)) & 1;
            self.position += 1;
            (value << 1) | u32::from(bit)
        })
    }

    /// Reads a codeword of `codewords` bit by bit, and returns its index.
    fn decode(&mut self, codewords: &[Codeword]) -> usize {
        let (mut bits, mut len) = (0, 0);
        loop {
            bits = (bits << 1) | self.read(1);
            len += 1;
            assert!(len <= 19, "no codeword matches");
            if let Some(index) = codewords
                .iter()
                .position(|c| c.len == len && c.bits == bits)
            {
                return index;
            }
        }
    }
}

/// `sect_cb[g][i]`, `sect_start[g][i]` and `sect_end[g][i]` of one group
/// (Table 17).
pub fn section_data(
    r: &mut BitReader,
    transform: Transform,
    max_sfb: usize,
) -> Vec<(u8, usize, usize)> {
    let (bits, sect_esc_val) = match transform {
        Transform::Short => (3, (1 << 3) - 1),
        Transform::Long => (5, (1 << 5) - 1),
    };
    let mut sections = Vec::new();
    let mut k = 0;
    while k < max_sfb {
        let sect_cb = r.read(4) as u8;
        let mut sect_len = 0;
        loop {
            let sect_len_incr = r.read(bits);
            if sect_len_incr != sect_esc_val {
                sect_len += sect_len_incr as usize;
                break;
            }
            sect_len += sect_esc_val as usize;
        }
        assert_ne!(sect_len, 0, "a section of no bands");
        sections.push((sect_cb, k, k + sect_len));
        k += sect_len;
    }
    assert_eq!(k, max_sfb, "the sections end at max_sfb");
    sections
}

/// `ZERO_HCB`, `INTENSITY_HCB2` and `INTENSITY_HCB` (Table 59).
const ZERO_HCB: u8 = 0;
const INTENSITY_HCB2: u8 = 14;
const INTENSITY_HCB: u8 = 15;

/// The scalefactors `sf[g][sfb]` of one group (Table 18, clause 11.3.2),
/// from `last_sf`, which they leave at the last one decoded; and in the
/// bands of an intensity codebook, `is_position[g][sfb]` in their place,
/// from `last_is`, likewise (clause 12.2.3), where the group is of the
/// right channel of a pair that shares its window, which alone may take
/// them, as `intensity` says. A band of `ZERO_HCB` has 0.
pub fn scale_factor_data(
    r: &mut BitReader,
    sections: &[(u8, usize, usize)],
    last_sf: &mut i32,
    last_is: &mut i32,
    intensity: bool,
) -> Vec<i32> {
    let mut sf = Vec::new();
    for &(sect_cb, start, end) in sections {
        let is_intensity = sect_cb == INTENSITY_HCB || sect_cb == INTENSITY_HCB2;
        assert!(!matches!(sect_cb, 12 | 13), "a reserved codebook");
        assert!(
            intensity || !is_intensity,
            "intensity outside a pair's right channel"
        );
        for _ in start..end {
            if sect_cb == ZERO_HCB {
                sf.push(0);
                continue;
            }
            // Table 58: index_offset -60.
            let dpcm = r.decode(&SCALEFACTOR_CODEWORDS) as i32 - 60;
            if is_intensity {
                *last_is += dpcm;
                sf.push(*last_is);
            } else {
                *last_sf += dpcm;
                assert!((0..=255).contains(last_sf));
                sf.push(*last_sf);
            }
        }
    }
    sf
}

/// The quantized lines of one group (Table 20, clause 9.3), `len` of them,
/// zero above the sections.
pub fn spectral_data(
    r: &mut BitReader,
    sections: &[(u8, usize, usize)],
    offsets: &[u16],
    len: usize,
) -> Vec<i32> {
    // Table 59: unsigned_cb, dimension, LAV and codebook.
    let books: [(bool, usize, i32, &[Codeword]); 11] = [
        (false, 4, 1, &tables::SPECTRUM_1),
        (false, 4, 1, &tables::SPECTRUM_2),
        (true, 4, 2, &tables::SPECTRUM_3),
        (true, 4, 2, &tables::SPECTRUM_4),
        (false, 2, 4, &tables::SPECTRUM_5),
        (false, 2, 4, &tables::SPECTRUM_6),
        (true, 2, 7, &tables::SPECTRUM_7),
        (true, 2, 7, &tables::SPECTRUM_8),
        (true, 2, 12, &tables::SPECTRUM_9),
        (true, 2, 12, &tables::SPECTRUM_10),
        (true, 2, 16, &tables::SPECTRUM_11),
    ];
    let mut lines = vec![0; len];
    for &(sect_cb, start, end) in sections {
        if sect_cb == ZERO_HCB || sect_cb > Codebook::ESC.number() {
            continue;
        }
        let (unsigned, dim, lav, codewords) = books[usize::from(sect_cb) - 1];
        let (modulus, off) = if unsigned {
            (lav + 1, 0)
        } else {
            (2 * lav + 1, lav)
        };
        let mut k = usize::from(offsets[start]);
        while k < usize::from(offsets[end]) {
            let mut idx = r.decode(codewords) as i32;
            let mut tuple = [0; 4];
            for (i, line) in tuple[..dim].iter_mut().enumerate() {
                let place = modulus.pow((dim - 1 - i) as u32);
                *line = idx / place - off;
                idx -= (*line + off) * place;
            }
            if unsigned {
                for line in &mut tuple[..dim] {
                    if *line != 0 && r.read(1) == 1 {
                        *line = -*line;
                    }
                }
            }
            if sect_cb == Codebook::ESC.number() {
                for line in &mut tuple[..dim] {
                    if line.abs() == 16 {
                        let mut n = 0;
                        while r.read(1) == 1 {
                            n += 1;
                        }
                        let escape_word = r.read(n + 4) as i32;
                        *line = line.signum() * ((1 << (n + 4)) + escape_word);
                    }
                }
            }
            lines[k..k + dim].copy_from_slice(&tuple[..dim]);
            k += dim;
        }
    }
    lines
}

/// The fields of an `ics_info()` (Table 15) of the Low Complexity profile,
/// with the window groups' lengths that `scale_factor_grouping` gives
/// (clause 8.3.4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IcsInfo {
    pub window_sequence: u8,
    pub window_shape: u8,
    pub max_sfb: usize,
    pub group_lengths: Vec<usize>,
}

impl IcsInfo {
    pub fn transform(&self) -> Transform {
        if self.window_sequence == 2 {
            Transform::Short
        } else {
            Transform::Long
        }
    }

    /// The offsets of the scalefactor bands of a group of `len` windows,
    /// in its lines as they are sent (clause 8.3.5).
    fn offsets(&self, fs: SamplingFrequency, len: usize) -> Vec<u16> {
        (swb_offsets(fs, self.transform()).iter())
            .map(|&o| o * len as u16)
            .collect()
    }
}

/// Reads an `ics_info()` (Table 15).
pub fn ics_info(r: &mut BitReader) -> IcsInfo {
    assert_eq!(r.read(1), 0, "ics_reserved_bit");
    let window_sequence = r.read(2) as u8;
    let window_shape = r.read(1) as u8;
    if window_sequence == 2 {
        let max_sfb = r.read(4) as usize;
        let grouping = r.read(7);
        let mut group_lengths = vec![1];
        for i in 0..7 {
            if grouping & (1 << (6 - i)) == 0 {
                group_lengths.push(1);
            } else {
                *group_lengths.last_mut().expect("a group") += 1;
            }
        }
        IcsInfo {
            window_sequence,
            window_shape,
            max_sfb,
            group_lengths,
        }
    } else {
        let max_sfb = r.read(6) as usize;
        assert_eq!(r.read(1), 0, "predictor_data_present");
        IcsInfo {
            window_sequence,
            window_shape,
            max_sfb,
            group_lengths: vec![1],
        }
    }
}

/// One filter of a window's `tns_data()` (Table 19) as read: `length`,
/// `order`, `direction`, `coef_compress` and the `order` values of `coef`,
/// as sent, unsigned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TnsFilter {
    pub length: usize,
    pub order: usize,
    pub direction: u32,
    pub coef_compress: u32,
    pub coef: Vec<u32>,
}

/// One window's `tns_data()` (Table 19) as read: `coef_res` and its
/// `n_filt` filters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TnsWindow {
    pub coef_res: u32,
    pub filters: Vec<TnsFilter>,
}

/// Reads a `tns_data()` (Table 19) of the windows of `info`, with the
/// fields' sizes of clause 14.2.1.
pub fn tns_data(r: &mut BitReader, info: &IcsInfo) -> Vec<TnsWindow> {
    let short = info.window_sequence == 2;
    let num_windows = if short { 8 } else { 1 };
    let (n_filt_bits, length_bits, order_bits) = if short { (1, 4, 3) } else { (2, 6, 5) };
    (0..num_windows)
        .map(|_| {
            let n_filt = r.read(n_filt_bits);
            let coef_res = if n_filt > 0 { r.read(1) } else { 0 };
            let filters = (0..n_filt)
                .map(|_| {
                    let length = r.read(length_bits) as usize;
                    let order = r.read(order_bits) as usize;
                    let (mut direction, mut coef_compress, mut coef) = (0, 0, Vec::new());
                    if order > 0 {
                        direction = r.read(1);
                        coef_compress = r.read(1);
                        let bits = coef_res + 3 - coef_compress;
                        coef = (0..order).map(|_| r.read(bits)).collect();
                    }
                    TnsFilter {
                        length,
                        order,
                        direction,
                        coef_compress,
                        coef,
                    }
                })
                .collect();
            TnsWindow { coef_res, filters }
        })
        .collect()
}

/// `TNS_MAX_BANDS` (Table 33) and `TNS_MAX_ORDER` (clause 7.1.6) of the
/// Low Complexity profile.
fn tns_max(fs: SamplingFrequency, short: bool) -> (usize, usize) {
    let bands = match (fs.hz(), short) {
        (96_000 | 88_200, false) => 31,
        (96_000 | 88_200, true) => 9,
        (64_000, false) => 34,
        (64_000, true) => 10,
        (48_000, false) => 40,
        (44_100, false) => 42,
        (32_000, false) => 51,
        (24_000 | 22_050, false) => 46,
        (16_000 | 12_000 | 11_025, false) => 42,
        (8_000, false) => 39,
        (_, true) => 14,
        (hz, _) => panic!("{hz} Hz"),
    };
    (bands, if short { 7 } else { 12 })
}

/// `tns_decode_coef()` of clause 14.3, transcribed: the LPC coefficients
/// `a[0..=order]` of a filter's coefficients as sent.
pub fn tns_decode_coef(
    order: usize,
    coef_res_bits: u32,
    coef_compress: u32,
    coef: &[u32],
) -> Vec<f64> {
    let sgn_mask = [0x2, 0x4, 0x8];
    let neg_mask = [!0x3, !0x7, !0xf];
    let coef_res2 = (coef_res_bits - coef_compress) as usize;
    let s_mask = sgn_mask[coef_res2 - 2];
    let n_mask = neg_mask[coef_res2 - 2];
    let tmp: Vec<i32> = (0..order)
        .map(|i| {
            let c = coef[i] as i32;
            if c & s_mask != 0 { c | n_mask } else { c }
        })
        .collect();
    let pi = std::f64::consts::PI;
    let iqfac = ((1 << (coef_res_bits - 1)) as f64 - 0.5) / (pi / 2.0);
    let iqfac_m = ((1 << (coef_res_bits - 1)) as f64 + 0.5) / (pi / 2.0);
    let tmp2: Vec<f64> = (tmp.iter())
        .map(|&t| (f64::from(t) / if t >= 0 { iqfac } else { iqfac_m }).sin())
        .collect();
    let mut a = vec![0.0; order + 1];
    let mut b = vec![0.0; order + 1];
    a[0] = 1.0;
    for m in 1..=order {
        for i in 1..m {
            b[i] = a[i] + tmp2[m - 1] * a[m - i];
        }
        a[1..m].copy_from_slice(&b[1..m]);
        a[m] = tmp2[m - 1];
    }
    a
}

/// `tns_ar_filter()` of clause 14.3: `y(n) = x(n) - lpc[1] y(n-1) - ... -
/// lpc[order] y(n-order)`, in place over `size` lines of `spectrum` from
/// `start` by `inc`, its state 0 to start with.
pub fn tns_ar_filter(spectrum: &mut [f64], start: usize, size: usize, inc: isize, lpc: &[f64]) {
    let order = lpc.len() - 1;
    let mut state = vec![0.0; order];
    let mut n = start as isize;
    for _ in 0..size {
        let i = n as usize;
        let mut y = spectrum[i];
        for (k, s) in state.iter().enumerate() {
            y -= lpc[k + 1] * s;
        }
        state.rotate_right(1);
        if order > 0 {
            state[0] = y;
        }
        spectrum[i] = y;
        n += inc;
    }
}

/// `tns_decode_frame()` of clause 14.3 over one channel's windows,
/// `spec[w][k]`, one window after another in `spec`.
pub fn tns_decode_frame(
    spec: &mut [f64],
    info: &IcsInfo,
    tns: &[TnsWindow],
    fs: SamplingFrequency,
) {
    let short = info.window_sequence == 2;
    let (tns_max_bands, tns_max_order) = tns_max(fs, short);
    let swb_offset = swb_offsets(fs, info.transform());
    let num_swb = swb_offset.len() - 1;
    let window_lines = if short { 128 } else { 1024 };
    for (w, window) in tns.iter().enumerate() {
        let spec = &mut spec[w * window_lines..(w + 1) * window_lines];
        let mut bottom = num_swb;
        for filter in &window.filters {
            let top = bottom;
            bottom = top.saturating_sub(filter.length);
            let tns_order = filter.order.min(tns_max_order);
            if tns_order == 0 {
                continue;
            }
            let lpc = tns_decode_coef(
                tns_order,
                window.coef_res + 3,
                filter.coef_compress,
                &filter.coef,
            );
            let start = usize::from(swb_offset[bottom.min(tns_max_bands).min(info.max_sfb)]);
            let end = usize::from(swb_offset[top.min(tns_max_bands).min(info.max_sfb)]);
            if end <= start {
                continue;
            }
            let size = end - start;
            let (start, inc) = if filter.direction == 1 {
                (end - 1, -1)
            } else {
                (start, 1)
            };
            tns_ar_filter(spec, start, size, inc, &lpc);
        }
    }
}

/// `quant_to_spec()` of clause 8.3.5, transcribed: the lines of a channel,
/// `x_quant` in the order they are sent, window by window in `spec`; and,
/// where `back`, the reverse.
fn quant_to_spec(
    info: &IcsInfo,
    fs: SamplingFrequency,
    x_quant: &mut [f64],
    spec: &mut [f64],
    back: bool,
) {
    let swb_offset = swb_offsets(fs, info.transform());
    let window_lines = if info.window_sequence == 2 { 128 } else { 1024 };
    let mut i = 0;
    let mut k = 0;
    for &len in &info.group_lengths {
        let mut j = 0;
        for sfb in 0..swb_offset.len() - 1 {
            let width = usize::from(swb_offset[sfb + 1] - swb_offset[sfb]);
            for win in 0..len {
                for bin in 0..width {
                    let s = &mut spec[(win + k) * window_lines + bin + j];
                    if back {
                        x_quant[i] = *s;
                    } else {
                        *s = x_quant[i];
                    }
                    i += 1;
                }
            }
            j += width;
        }
        k += len;
    }
}

/// An `individual_channel_stream()` (Table 16) as read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ics {
    pub global_gain: i32,
    pub info: IcsInfo,
    /// Each window's `tns_data()` where `tns_data_present` is 1.
    pub tns: Option<Vec<TnsWindow>>,
    /// Each group's sections, as [`section_data`] gives them.
    pub sections: Vec<Vec<(u8, usize, usize)>>,
    /// Each group's scalefactors, as [`scale_factor_data`] gives them.
    pub scalefactors: Vec<Vec<i32>>,
    /// Each group's quantized lines, in the order they are sent.
    pub lines: Vec<Vec<i32>>,
}

impl Ics {
    /// The codebook of band `sfb` of group `g`.
    pub fn codebook(&self, g: usize, sfb: usize) -> u8 {
        (self.sections[g].iter())
            .find(|&&(_, start, end)| (start..end).contains(&sfb))
            .map_or(ZERO_HCB, |&(cb, _, _)| cb)
    }
}

/// Reads an `individual_channel_stream()` (Table 16) of the Low Complexity
/// profile at `fs`, with its own `ics_info()`, or that of its pair,
/// `common`, whose right channel it is, and so may take intensity
/// positions, if `right`.
pub fn individual_channel_stream(
    r: &mut BitReader,
    fs: SamplingFrequency,
    common: Option<&IcsInfo>,
    right: bool,
) -> Ics {
    let global_gain = r.read(8) as i32;
    let info = match common {
        Some(info) => info.clone(),
        None => ics_info(r),
    };
    let sections: Vec<_> = (info.group_lengths.iter())
        .map(|_| section_data(r, info.transform(), info.max_sfb))
        .collect();
    let (mut last_sf, mut last_is) = (global_gain, 0);
    let scalefactors = (sections.iter())
        .map(|sections| {
            let intensity = right && common.is_some();
            scale_factor_data(r, sections, &mut last_sf, &mut last_is, intensity)
        })
        .collect();
    assert_eq!(r.read(1), 0, "pulse_data_present");
    let tns = (r.read(1) == 1).then(|| tns_data(r, &info));
    assert_eq!(r.read(1), 0, "gain_control_data_present");
    let lines = (sections.iter().zip(&info.group_lengths))
        .map(|(sections, &len)| {
            let offsets = info.offsets(fs, len);
            let lines = usize::from(*offsets.last().expect("bands"));
            spectral_data(r, sections, &offsets, lines)
        })
        .collect();
    Ics {
        global_gain,
        info,
        tns,
        sections,
        scalefactors,
        lines,
    }
}

/// A `single_channel_element()` or `channel_pair_element()` (Tables 13 and
/// 14) as read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Element {
    pub channels: Vec<Ics>,
    pub common_window: bool,
    pub ms_mask_present: u32,
    /// `ms_used[g][sfb]` where `ms_mask_present` is 1.
    pub ms_used: Vec<Vec<bool>>,
}

/// Reads an element, from its `id_syn_ele`: a `single_channel_element()`
/// or a `channel_pair_element()` of the Low Complexity profile at `fs`.
pub fn element(r: &mut BitReader, fs: SamplingFrequency) -> Element {
    let id_syn_ele = r.read(3);
    r.read(4); // element_instance_tag
    match id_syn_ele {
        0 => Element {
            channels: vec![individual_channel_stream(r, fs, None, false)],
            common_window: false,
            ms_mask_present: 0,
            ms_used: Vec::new(),
        },
        1 => {
            let common_window = r.read(1) == 1;
            let mut info = None;
            let mut ms_mask_present = 0;
            let mut ms_used = Vec::new();
            if common_window {
                let common = ics_info(r);
                ms_mask_present = r.read(2);
                assert_ne!(ms_mask_present, 3, "reserved");
                if ms_mask_present == 1 {
                    ms_used = (common.group_lengths.iter())
                        .map(|_| (0..common.max_sfb).map(|_| r.read(1) == 1).collect())
                        .collect();
                }
                info = Some(common);
            }
            let left = individual_channel_stream(r, fs, info.as_ref(), false);
            let right = individual_channel_stream(r, fs, info.as_ref(), true);
            Element {
                channels: vec![left, right],
                common_window,
                ms_mask_present,
                ms_used,
            }
        }
        other => panic!("element {other}"),
    }
}

impl Element {
    /// The spectra the element reconstructs, each channel's lines group by
    /// group in the order they are sent: inversely quantized (clause 10.3),
    /// scaled by their scalefactors (clause 11.3.3), of a pair, M/S (clause
    /// 12.1.3) and intensity stereo decoded (clause 12.2.3), and their
    /// temporal noise shaping decoded (clause 14.3), in that order (clause
    /// 4, figure 2).
    pub fn spectra(&self, fs: SamplingFrequency) -> Vec<Vec<f64>> {
        let mut spectra = self.joint_spectra(fs);
        for (spectrum, ics) in spectra.iter_mut().zip(&self.channels) {
            if let Some(tns) = &ics.tns {
                let mut spec = vec![0.0; spectrum.len()];
                quant_to_spec(&ics.info, fs, spectrum, &mut spec, false);
                tns_decode_frame(&mut spec, &ics.info, tns, fs);
                quant_to_spec(&ics.info, fs, spectrum, &mut spec, true);
            }
        }
        spectra
    }

    /// [`spectra`](Self::spectra) before temporal noise shaping is decoded.
    pub fn joint_spectra(&self, fs: SamplingFrequency) -> Vec<Vec<f64>> {
        let mut spectra: Vec<Vec<f64>> = (self.channels.iter())
            .map(|ics| {
                let mut spectrum = Vec::new();
                for (g, (lines, &len)) in ics.lines.iter().zip(&ics.info.group_lengths).enumerate()
                {
                    let offsets = ics.info.offsets(fs, len);
                    for (i, &q) in lines.iter().enumerate() {
                        let sfb = offsets.iter().rposition(|&o| usize::from(o) <= i);
                        let scaled = match sfb {
                            Some(sfb) if sfb < ics.info.max_sfb => {
                                let cb = ics.codebook(g, sfb);
                                if cb == ZERO_HCB || cb >= INTENSITY_HCB2 {
                                    0.0
                                } else {
                                    let sf = ics.scalefactors[g][sfb];
                                    let gain = (0.25 * f64::from(sf - 100)).exp2();
                                    let x = f64::from(q.unsigned_abs()).powf(4.0 / 3.0);
                                    f64::from(q.signum()) * x * gain
                                }
                            }
                            _ => 0.0,
                        };
                        spectrum.push(scaled);
                    }
                }
                spectrum
            })
            .collect();
        if let ([left, right], true) = (&mut spectra[..], self.common_window) {
            let info = &self.channels[0].info;
            let right_ics = &self.channels[1];
            let mut start = 0;
            for (g, &len) in info.group_lengths.iter().enumerate() {
                let offsets = info.offsets(fs, len);
                for sfb in 0..info.max_sfb {
                    let band =
                        start + usize::from(offsets[sfb])..start + usize::from(offsets[sfb + 1]);
                    let ms_used = self.ms_used.get(g).is_some_and(|used| used[sfb]);
                    let is_intensity = match right_ics.codebook(g, sfb) {
                        INTENSITY_HCB => 1.0,
                        INTENSITY_HCB2 => -1.0,
                        _ => 0.0,
                    };
                    if is_intensity != 0.0 {
                        let invert = if self.ms_mask_present == 1 && ms_used {
                            -1.0
                        } else {
                            1.0
                        };
                        let is_position = right_ics.scalefactors[g][sfb];
                        let scale =
                            is_intensity * invert * 0.5f64.powf(0.25 * f64::from(is_position));
                        for i in band {
                            right[i] = scale * left[i];
                        }
                    } else if ms_used || self.ms_mask_present == 2 {
                        for i in band {
                            let (l, r) = (left[i] + right[i], left[i] - right[i]);
                            (left[i], right[i]) = (l, r);
                        }
                    }
                }
                start += usize::from(*offsets.last().expect("bands"));
            }
        }
        spectra
    }
}

/// A deterministic stream of pseudo-random numbers (xorshift), for tests to
/// make spectra of.
#[cfg(test)]
pub struct Noise(u64);

#[cfg(test)]
impl Noise {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    fn draw(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }

    /// A number in `0..n`.
    pub fn below(&mut self, n: u32) -> u32 {
        self.draw() % n
    }

    /// Lines for `offsets`, whose bands are each of zeros or reach a
    /// largest magnitude drawn from every codebook's range, escapes to
    /// 8191 included, and are sparse at random.
    pub fn spectrum(&mut self, offsets: &[u16], len: usize) -> Vec<i32> {
        const LARGEST: [u32; 12] = [0, 0, 1, 2, 3, 4, 7, 12, 15, 16, 300, 8191];
        let mut lines = vec![0; len];
        for band in offsets.windows(2) {
            let largest = LARGEST[self.below(12) as usize];
            let density = 1 + self.below(4);
            for line in &mut lines[usize::from(band[0])..usize::from(band[1])] {
                if largest > 0 && self.below(4) < density {
                    let magnitude = self.below(largest + 1) as i32;
                    *line = if self.below(2) == 0 {
                        magnitude
                    } else {
                        -magnitude
                    };
                }
            }
        }
        lines
    }
}
