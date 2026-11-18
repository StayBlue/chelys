// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The quantization of a channel's lines (clause C.7.4) and the inner
//! iteration loop (clause C.7.4.5, figure C.8), which sets
//! `common_scalefac` so that the lines fit the bits available.

use std::ops::Range;

use crate::channel::{self, CodedChannel, CodedGroup, FRAME_LINES};
use crate::noiseless::codebooks::MAX_MAGNITUDE;
use crate::noiseless::scalefactors;
use crate::noiseless::sections::{self, IntensityBands, Sections};
use crate::noiseless::spectrum;
use crate::scalefactor_bands::{GroupBands, MAX_BANDS};

use super::ChannelCoding;

/// `SF_OFFSET` (clauses 11.3.3 and C.7.4): `sf_decoder = common_scalefac
/// - scalefactor + SF_OFFSET`.
const SF_OFFSET: i32 = 100;

/// `MAGIC_NUMBER` (clause C.7.4), which the quantizer adds before
/// truncating.
const MAGIC_NUMBER: f64 = 0.4054;

/// `MAX_QUANT` (clause C.7.2.1): "the maximum quantized value which can be
/// encoded in the bitstream, defined as 8191".
const MAX_QUANT: f64 = MAX_MAGNITUDE as f64;

/// `quantizer_change` of the first run of the inner loop (clause
/// C.7.4.5).
const FIRST_CHANGE: i32 = 32;

/// A magnitude that no codebook codes, at which the quantizer stops.
const BEYOND_MAGNITUDE: f64 = 2.0 * MAX_QUANT;

/// The scale of `x_quant = int(mdct_scaled * 2^(-3/16 *
/// common_scalefac) + MAGIC_NUMBER)` (clause C.7.4.2) in a band of the
/// scalefactor `sf_decoder` (see [`Lines::quantize`]), from a table of
/// every scalefactor.
fn step_scale(sf_decoder: u8) -> f64 {
    static TABLE: std::sync::OnceLock<[f64; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        std::array::from_fn(|sf| (-3.0 / 16.0 * f64::from(sf as i32 - SF_OFFSET)).exp2())
    });
    table[usize::from(sf_decoder)]
}

/// `|x|^(3/4)`, as `(|x| * |x|^(1/2))^(1/2)`: two square roots, which the
/// processor computes, rather than a call to the C library's `pow`. It
/// differs from `pow` by about the last bit.
fn power_three_quarters(x: f64) -> f64 {
    let x = x.abs();
    (x * x.sqrt()).sqrt()
}

/// `|x|^(4/3)` of each magnitude the codebooks code: the decoder's
/// inverse quantization (clause 10.3), for the distortion of the outer
/// loop.
pub(super) fn power_four_thirds(magnitude: u32) -> f64 {
    static TABLE: std::sync::OnceLock<Vec<f64>> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        (0..=MAX_MAGNITUDE)
            .map(|q| f64::from(q).powf(4.0 / 3.0))
            .collect()
    });
    table[magnitude as usize]
}

/// `2^(1/4 * (sf_decoder - SF_OFFSET))`: the decoder's gain of a band with
/// the scalefactor `sf_decoder` (clause 11.3.3), from a table of every
/// scalefactor.
pub(super) fn decoder_gain(sf_decoder: u8) -> f64 {
    static TABLE: std::sync::OnceLock<[f64; 256]> = std::sync::OnceLock::new();
    let table = TABLE
        .get_or_init(|| std::array::from_fn(|sf| (0.25 * f64::from(sf as i32 - SF_OFFSET)).exp2()));
    table[usize::from(sf_decoder)]
}

/// A channel's lines with the powers the quantizer takes of them, taken
/// once for every step tried.
pub struct Lines<'a> {
    spectrum: &'a [f64; FRAME_LINES],
    /// `abs(mdct_line)^(3/4)`: `mdct_scaled` of clause C.7.4.1 with every
    /// scalefactor 0.
    scaled: [f64; FRAME_LINES],
    /// One past the last line that is not zero: those above quantize to
    /// zero at every step.
    end: usize,
    /// See [`Self::start_gain`].
    start: u8,
}

impl<'a> Lines<'a> {
    pub fn new(spectrum: &'a [f64; FRAME_LINES]) -> Self {
        let scaled = spectrum.map(power_three_quarters);
        let largest = scaled.iter().fold(0.0f64, |a, &x| a.max(x));
        let start = (16.0 / 3.0 * (largest / MAX_QUANT).log2()).ceil();
        Self {
            spectrum,
            scaled,
            end: spectrum
                .iter()
                .rposition(|&x| x != 0.0)
                .map_or(0, |i| i + 1),
            start: (start + f64::from(SF_OFFSET)).clamp(0.0, 255.0) as u8,
        }
    }

    pub fn spectrum(&self) -> &'a [f64; FRAME_LINES] {
        self.spectrum
    }

    /// `start_common_scalefac = ceiling(16/3 * log2(max_mdct_line^(3/4) /
    /// MAX_QUANT))` (clause C.7.2.1), as the `global_gain` it is sent with,
    /// within its range: the smallest step at which every line can be
    /// coded.
    ///
    /// Spec gap: for lines whose largest is below about 0.005 (at the
    /// decoder's scale), the formula gives a step below the range of
    /// `global_gain`, `common_scalefac` -100, which clause C.7.2.1 does not
    /// provide for. The step is raised to the range's, the finest the
    /// bitstream can send: the lines are coded with fewer levels than
    /// `MAX_QUANT`, as many as that step gives them.
    pub fn start_gain(&self) -> u8 {
        self.start
    }

    /// Quantizes the lines `range` with the step of the scalefactor
    /// `sf_decoder` (clause C.7.4.5, `quantize_spectrum`) into `x_quant`,
    /// carrying the sign of each line.
    ///
    /// `x_quant(i) = int(mdct_scaled(i) * 2^(-3/16 * common_scalefac) +
    /// MAGIC_NUMBER)`, where `int` truncates, and `mdct_scaled(i) =
    /// abs(mdct_line(i))^(3/4) * 2^(3/16 * scalefactor(sb))` (clause
    /// C.7.4.1): with `sf_decoder = common_scalefac - scalefactor +
    /// SF_OFFSET`, `abs(mdct_line(i))^(3/4) * 2^(-3/16 * (sf_decoder -
    /// SF_OFFSET))`. Magnitudes beyond [`MAX_MAGNITUDE`] are left beyond it,
    /// for counting to refuse.
    ///
    /// Spec reading: clause C.7.4 gives the quantizer first as `int((abs(
    /// mdct_line) * 2^(-1/4 * (sf_decoder - SF_OFFSET)))^(3/4) +
    /// MAGIC_NUMBER)`, the inverse of the decoder's, then, with
    /// `sf_decoder = common_scalefac - scalefactor + SF_OFFSET`, as
    /// `2^(-1/4 * (scalefactor - common_scalefac))`, whose sign is the
    /// opposite of what the substitution gives, and of clauses C.7.4.1,
    /// C.7.4.2 and C.7.4.5, and of the text after it ("a positive change
    /// increases the magnitude of x_quant"). The second form is taken as
    /// misprinted, with the sign of the scalefactor reversed, and the first
    /// and clauses C.7.4.1 and C.7.4.2 are followed. The distortion of
    /// clause C.7.4.3 and its attenuation, which take the reversed sign
    /// too, are read alike (see [`outer_loop`](super::outer::outer_loop)).
    pub fn quantize(&self, sf_decoder: u8, range: Range<usize>, x_quant: &mut [i32]) {
        let scale = step_scale(sf_decoder);
        let end = range.end.min(self.end).max(range.start);
        let (coded, above) = x_quant.split_at_mut(end - range.start);
        let lines = (coded.iter_mut())
            .zip(&self.scaled[range.start..end])
            .zip(&self.spectrum[range.start..end]);
        for ((q, &scaled), &x) in lines {
            let magnitude = (scaled * scale + MAGIC_NUMBER).min(BEYOND_MAGNITUDE) as i32;
            *q = if x < 0.0 { -magnitude } else { magnitude };
        }
        above.fill(0);
    }
}

/// The fewest bits that `lines`, one scalefactor band's, take when
/// quantized with the coarsest step whose distortion is within `allowed`:
/// none where their energy is, as they can quantize to zero, and else those
/// of the band's `spectral_data()` in its best codebook (see
/// [`spectrum::least_bits`]).
///
/// The step is searched by successive approximation, as the inner loop
/// searches it, from the smallest that codes every line (see
/// [`Lines::start_gain`]) to the largest `global_gain`, taking the
/// distortion to grow with the step; where even the smallest exceeds
/// `allowed`, it is taken.
pub fn band_bits(lines: &[f64], allowed: f64, scratch: &mut BandScratch) -> u32 {
    let energy: f64 = lines.iter().map(|x| x * x).sum();
    if energy <= allowed {
        return 0;
    }
    let scaled = &mut scratch.scaled[..lines.len()];
    for (scaled, &x) in scaled.iter_mut().zip(lines) {
        *scaled = power_three_quarters(x);
    }
    let scaled = &*scaled;
    let x_quant = &mut scratch.x_quant[..lines.len()];
    let quantize = |sf_decoder: u8, x_quant: &mut [i32]| {
        let scale = step_scale(sf_decoder);
        for ((q, &scaled), &x) in x_quant.iter_mut().zip(scaled).zip(lines) {
            let magnitude = (scaled * scale + MAGIC_NUMBER).min(BEYOND_MAGNITUDE) as i32;
            *q = if x < 0.0 { -magnitude } else { magnitude };
        }
    };
    let distortion = |sf_decoder: u8, x_quant: &mut [i32]| {
        quantize(sf_decoder, x_quant);
        let gain = decoder_gain(sf_decoder);
        (lines.iter().zip(&*x_quant))
            .map(|(&x, &q)| {
                let error = x.abs() - power_four_thirds(q.unsigned_abs().min(MAX_MAGNITUDE)) * gain;
                error * error
            })
            .sum::<f64>()
    };
    let largest = scaled.iter().fold(0.0f64, |a, &x| a.max(x));
    let start = (16.0 / 3.0 * (largest / MAX_QUANT).log2()).ceil() + f64::from(SF_OFFSET);
    let start = start.clamp(0.0, 255.0) as u8;
    // The coarsest step within `allowed`, between `fits`, which is, and
    // `exceeds`, which is not.
    let (mut fits, mut exceeds) = (start, u16::from(u8::MAX) + 1);
    if distortion(start, x_quant) <= allowed {
        while exceeds - u16::from(fits) > 1 {
            let middle = ((u16::from(fits) + exceeds) / 2) as u8;
            if distortion(middle, x_quant) <= allowed {
                fits = middle;
            } else {
                exceeds = u16::from(middle);
            }
        }
    }
    quantize(fits, x_quant);
    spectrum::least_bits(x_quant).unwrap_or(u32::MAX / 8)
}

/// The memory [`band_bits`] works in, for the lines of any band.
pub struct BandScratch {
    scaled: [f64; FRAME_LINES],
    x_quant: [i32; FRAME_LINES],
}

impl Default for BandScratch {
    fn default() -> Self {
        Self {
            scaled: [0.0; FRAME_LINES],
            x_quant: [0; FRAME_LINES],
        }
    }
}

/// The coding of a channel's lines that the loops last counted, with the
/// scalefactors of the outer loop that it was counted with.
pub struct Counting<'a> {
    lines: Lines<'a>,
    coding: ChannelCoding<'a>,
    /// `scalefactor(sb)` of the outer loop (clause C.7.4.1) of each band of
    /// each window group: how many steps of the quantizer each is
    /// amplified by. All 0 to start with (clause C.7.2.1).
    pub scalefactors: Vec<[u8; MAX_BANDS]>,
    /// The lines quantized at the step counted last.
    x_quant: [i32; FRAME_LINES],
    /// Each group's lines counted at the step counted last, kept to reuse
    /// their memory.
    counted: Vec<sections::Counted>,
    /// The sections chosen for each group at the step counted last, if
    /// it fits.
    sections: Vec<Sections>,
    /// The `sf_decoder` of each group's bands below `max_sfb` at the step
    /// counted last, if it fits.
    sf_decoder: Vec<Vec<u8>>,
    /// The step and the bands sent of the count last made, if it fitted,
    /// and the scalefactors it was made with: what the lines, sections and
    /// `sf_decoder` above hold.
    fitted_at: Option<(u8, usize)>,
    fitted_scalefactors: Vec<[u8; MAX_BANDS]>,
}

impl<'a> Counting<'a> {
    /// The coding of `lines` as `coding` asks, with every scalefactor 0,
    /// not yet counted.
    pub fn new(lines: Lines<'a>, coding: ChannelCoding<'a>) -> Self {
        let groups = coding.windows.groups(coding.sampling_frequency).count();
        Self {
            lines,
            coding,
            scalefactors: vec![[0; MAX_BANDS]; groups],
            x_quant: [0; FRAME_LINES],
            counted: Vec::with_capacity(groups),
            sections: Vec::with_capacity(groups),
            sf_decoder: vec![Vec::with_capacity(MAX_BANDS); groups],
            fitted_at: None,
            fitted_scalefactors: vec![[0; MAX_BANDS]; groups],
        }
    }

    pub fn lines(&self) -> &Lines<'a> {
        &self.lines
    }

    /// The window groups' bands, each with its first line in the channel.
    pub fn groups(&self) -> impl Iterator<Item = (usize, GroupBands)> + use<'a> {
        let mut start = 0;
        (self.coding.windows)
            .groups(self.coding.sampling_frequency)
            .map(move |bands| {
                let first = start;
                start += bands.lines();
                (first, bands)
            })
    }

    /// The scalefactor `sf_decoder` of band `sb` of group `g` at the step
    /// `common_gain`, the `global_gain` of a band whose scalefactor is 0:
    /// `common_scalefac - scalefactor + SF_OFFSET` (clause C.7.4), if it is
    /// within the range of 0 to 255 that clause 11.3.2 allows.
    pub fn sf_decoder(&self, common_gain: u8, g: usize, sb: usize) -> Option<u8> {
        u8::try_from(i32::from(common_gain) - i32::from(self.scalefactors[g][sb])).ok()
    }

    /// Quantizes the lines with the step `common_gain`, `common_scalefac +
    /// SF_OFFSET`, and the scalefactors, sending the scalefactor bands below
    /// `band_limit` at most, and counts their bits as the bitstream codes
    /// them (`bit_count()`, clause C.7.4.5): whether they fit the bits
    /// available. A line that no codebook codes does not, nor does a
    /// scalefactor outside its range or a difference of scalefactors that
    /// the scalefactor codebook does not code.
    ///
    /// Spec gap: the loops give `common_scalefac` and each band's
    /// scalefactor (clause C.7.1), not `global_gain`, which is "typically
    /// the value of the first active scalefactor" (clause 8.3.2.5). It is
    /// taken as the scalefactor of the first band that sends one and has a
    /// line that is not zero, so that the first difference sent is 0; a
    /// band of zeros before it repeats it (see
    /// [`scalefactors`](crate::noiseless::scalefactors)). The bits counted
    /// are those the channel is written in.
    pub fn count_at(&mut self, common_gain: u8, band_limit: usize) -> bool {
        self.fitted_at = None;
        let fits = self.count(common_gain, band_limit);
        if fits {
            self.fitted_at = Some((common_gain, band_limit));
            self.fitted_scalefactors.copy_from_slice(&self.scalefactors);
        }
        fits
    }

    /// [`Self::count_at`], but for keeping what it fitted.
    fn count(&mut self, common_gain: u8, band_limit: usize) -> bool {
        let mut start = 0;
        let groups = self.coding.windows.groups(self.coding.sampling_frequency);
        for (g, bands) in groups.clone().enumerate() {
            let sent = band_limit.min(bands.num_swb());
            for sb in 0..sent {
                let Some(sf_decoder) = self.sf_decoder(common_gain, g, sb) else {
                    return false;
                };
                let range = start + bands.offset(sb)..start + bands.offset(sb + 1);
                self.lines
                    .quantize(sf_decoder, range.clone(), &mut self.x_quant[range]);
            }
            self.x_quant[start + bands.offset(sent)..start + bands.lines()].fill(0);
            start += bands.lines();
        }
        let ChannelCoding {
            windows,
            available_bits,
            common_window,
            least_max_sfb,
            ..
        } = self.coding;
        let intensity = self.coding.intensity;
        let intensity_end = intensity.map_or(0, |intensity| {
            (intensity.bands.iter())
                .map(|bands| bands.end())
                .max()
                .unwrap_or(0)
        });
        let max_sfb = self
            .group_lines()
            .map(|(bands, lines)| sections::max_sfb(lines, bands))
            .max()
            .unwrap_or(0)
            .max(least_max_sfb)
            .max(intensity_end);
        let element_bits = self.coding.element_bits[max_sfb];
        let mut least_bits = channel::bits(windows, common_window, 0) + element_bits;
        self.sections.clear();
        self.counted.clear();
        let mut start = 0;
        for (g, bands) in groups.enumerate() {
            let lines = &self.x_quant[start..start + bands.lines()];
            start += bands.lines();
            let intensity = intensity.map_or(IntensityBands::NONE, |i| i.bands[g]);
            let Some(group) = Sections::count_with_intensity(lines, bands, max_sfb, intensity)
            else {
                return false;
            };
            least_bits += group.least_bits();
            if least_bits > available_bits {
                return false;
            }
            self.counted.push(group);
        }
        (self.sections).extend(self.counted.drain(..).map(sections::Counted::choose));
        for g in 0..self.sf_decoder.len() {
            let mut sf_decoder = std::mem::take(&mut self.sf_decoder[g]);
            sf_decoder.clear();
            for sb in 0..max_sfb {
                let Some(value) = self.sf_decoder(common_gain, g, sb) else {
                    return false;
                };
                sf_decoder.push(value);
            }
            self.sf_decoder[g] = sf_decoder;
        }
        let positions = |g: usize| intensity.map_or(&[][..], |i| &i.positions[g][..]);
        let groups = (self.sections.iter().zip(&self.sf_decoder).enumerate())
            .map(|(g, (sections, sf_decoder))| (sections, sf_decoder.as_slice(), positions(g)));
        let Some(scalefactor_bits) = scalefactors::bits(self.global_gain(common_gain), groups)
        else {
            return false;
        };
        let noiseless: u32 = (self.sections.iter())
            .map(|s| s.section_data_bits() + s.spectral_data_bits())
            .sum();
        channel::bits(windows, common_window, noiseless + scalefactor_bits) + element_bits
            <= available_bits
    }

    /// `global_gain` for the sections and scalefactors counted last: the
    /// scalefactor of the first band that sends one and has a line that is
    /// not zero, or, where no band has, `common_gain`, which no band then
    /// uses.
    fn global_gain(&self, common_gain: u8) -> u8 {
        (self.sections.iter().zip(&self.sf_decoder))
            .find_map(|(sections, sf_decoder)| {
                (sections.bands_with_scalefactors())
                    .find(|&sb| !sections.is_zero_band(sb))
                    .map(|sb| sf_decoder[sb])
            })
            .unwrap_or(common_gain)
    }

    /// Each group's bands with its quantized lines.
    pub fn group_lines(&self) -> impl Iterator<Item = (GroupBands, &[i32])> {
        self.groups()
            .map(move |(start, bands)| (bands, &self.x_quant[start..start + bands.lines()]))
    }

    /// The channel of the lines quantized at the step `common_gain` with
    /// the scalefactors, sending the bands below `band_limit` at most, which
    /// fit: the coding last counted, if it was made so, or else counted
    /// again.
    ///
    /// # Panics
    ///
    /// If that coding does not fit: a caller's error.
    pub fn into_channel(mut self, common_gain: u8, band_limit: usize) -> CodedChannel {
        let counted = self.fitted_at == Some((common_gain, band_limit))
            && self.fitted_scalefactors == self.scalefactors;
        if !counted {
            assert!(self.count_at(common_gain, band_limit), "a coding that fits");
        }
        let global_gain = self.global_gain(common_gain);
        let groups = (self.sections.into_iter())
            .zip(self.sf_decoder)
            .enumerate()
            .map(|(g, (sections, scalefactors))| CodedGroup {
                positions: (self.coding.intensity).map_or(Vec::new(), |i| {
                    i.positions[g][..sections.max_sfb()].to_vec()
                }),
                sections,
                scalefactors,
            })
            .collect();
        CodedChannel::new(
            self.coding.windows,
            self.coding.sampling_frequency,
            global_gain,
            self.x_quant,
            groups,
        )
        .expect("the scalefactors were counted")
    }
}

/// The inner iteration loop (clause C.7.4.5): quantizes the channel's
/// lines with the scalefactors of `counting`, and searches
/// `common_scalefac` by successive approximation from `start`, as
/// `common_gain`, `common_scalefac + SF_OFFSET`, with a first
/// `quantizer_change` of `change`, for the smallest step whose coding fits
/// in the channel's available bits, of which `counting` keeps the coding.
/// Gives the step, or `None` if none fits up to the largest.
///
/// The first run of the inner loop starts at `start_common_scalefac`
/// (clause C.7.2.1) with a change of 32, the later runs of the outer loop
/// at the step before with a change of 1 (clause C.7.4.5).
///
/// Spec reading: the pseudo code of clause C.7.4.5 changes
/// `common_scalefac` after each count, also after the last, which fits,
/// and so ends with it a step below the quantization it counted last, at
/// a step it has not counted, which with bits that fall as the step grows
/// is mostly one that does not fit. Figure C.8 adds `quantizer_change` to
/// `common_scalefac` only where it is not 0, and so ends at the step
/// counted last. The figure is followed: the quantization counted last is
/// the one coded, with the `common_scalefac` it was made with, so that the
/// bits coded are those counted. The pseudo code is followed for the rest,
/// as the figure does not give the values of `quantizer_change` or the
/// rule that keeps it 1 while the bits do not fit. With the steps of 32,
/// 16, 8, 4 and 2 that come before, and bits that fall as the step grows,
/// the step coded is the smallest that fits or the one above it.
///
/// Spec reading: successive approximation assumes that the bits fall as
/// the step grows, and they need not: the noiseless coding of the lines
/// can cost more at a coarser step. Where they do not fall, the search can
/// stop at a step above a finer one that fits, and it is followed as
/// written. For example, at 48 kHz in mono at 6375 bit/s, the third frame
/// of `0.8 * cos(pi * 73 * n / 1024 + 0.9)` costs 70 bits at `global_gain`
/// 167 and 168, 71 at 169 to 172 and 69 at 173; with about 70 bits
/// available it is coded at 173, though 167 and 168 fit.
///
/// Spec reading: `common_scalefac` "must not become less than
/// start_common_scalefac" (clause C.7.2.1), so a change that would take it
/// lower takes it there instead. Where the first count, at the start,
/// fits, every later step is the start again, so the loop ends there.
/// Likewise a change that would take `global_gain` past 255, the largest
/// scalefactor (clause 11.3.2), takes it there instead.
pub fn inner_loop(counting: &mut Counting, start: u8, change: i32) -> Option<u8> {
    let floor = counting.lines.start_gain().min(start);
    let mut gain = i32::from(start);
    let mut change = change;
    loop {
        let counted = gain as u8;
        let fits = counting.count_at(counted, MAX_BANDS);
        if fits && counted == floor {
            return Some(counted);
        }
        if !fits && counted == u8::MAX {
            return None;
        }
        gain = if fits { gain - change } else { gain + change };
        gain = gain.clamp(i32::from(floor), i32::from(u8::MAX));
        change /= 2;
        if change == 0 && !fits {
            change = 1;
        }
        if change == 0 {
            return Some(counted);
        }
    }
}

/// The step of the first run of the inner loop, from
/// `start_common_scalefac` (clause C.7.2.1), at which `counting` keeps its
/// coding, and the bands it sends, all of them unless none fits.
///
/// Spec gap: Annex C does not say what to do with lines whose coding does
/// not fit even at the largest step that `global_gain` allows, as from
/// samples far beyond full scale at the lowest bitrates. At that step,
/// the highest scalefactor bands are left unsent, one at a time, until the
/// rest fit; the channel without any fits, as the lowest bitrate is that
/// of frames of channels without lines (see
/// [`StreamConfig::min_bits_per_second`](crate::stream::StreamConfig::min_bits_per_second)).
///
/// # Panics
///
/// If not even a channel without lines fits in the available bits: a
/// caller's error.
pub fn first_inner_loop(counting: &mut Counting) -> (u8, usize) {
    let start = counting.lines.start_gain();
    if let Some(gain) = inner_loop(counting, start, FIRST_CHANGE) {
        return (gain, MAX_BANDS);
    }
    for band_limit in (0..MAX_BANDS).rev() {
        if counting.count_at(u8::MAX, band_limit) {
            return (u8::MAX, band_limit);
        }
    }
    panic!(
        "not even a channel without lines fits in {} bits",
        counting.coding.available_bits
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling_frequency::SamplingFrequency;
    use crate::window::Windows;

    fn spectrum(f: impl FnMut(usize) -> f64) -> [f64; FRAME_LINES] {
        std::array::from_fn(f)
    }

    #[test]
    fn the_quantizer_inverts_the_decoders_requantization() {
        // At sf_decoder = global_gain, a line decodes to x_quant^(4/3) *
        // 2^(0.25 * (global_gain - 100)) (clauses 10.3 and 11.3.3).
        let lines = spectrum(|i| (i as f64 * 37.5).powf(1.3) * if i % 2 == 0 { 1.0 } else { -1.0 });
        let quantizer = Lines::new(&lines);
        let mut x_quant = [0i32; FRAME_LINES];
        // From the start step, at which the largest line is codable, up.
        assert_eq!(quantizer.start_gain(), 110);
        for gain in [110, 140, 200] {
            quantizer.quantize(gain, 0..FRAME_LINES, &mut x_quant);
            let step = (0.25 * f64::from(i32::from(gain) - 100)).exp2();
            for (&x, &q) in lines.iter().zip(&x_quant) {
                let decoded = f64::from(q).abs().powf(4.0 / 3.0) * step * f64::from(q.signum());
                // Within the quantizer's step at that magnitude.
                let next = (f64::from(q.abs()) + 1.0).powf(4.0 / 3.0) * step;
                assert!(
                    (x - decoded).abs() <= next - decoded.abs(),
                    "{gain}: {x} as {q}"
                );
                assert!(q == 0 || q.signum() == x.signum() as i32);
            }
        }
    }

    #[test]
    fn the_quantizer_rounds_up_from_0_5946() {
        // int(x + 0.4054): 0.5946 is the least fraction that rounds up.
        let lines = spectrum(|i| match i {
            0 => 0.5945f64.powf(4.0 / 3.0),
            1 => 0.5947f64.powf(4.0 / 3.0),
            2 => 1.5947f64.powf(4.0 / 3.0),
            _ => 0.0,
        });
        let mut x_quant = [0i32; FRAME_LINES];
        // global_gain 100: common_scalefac 0, a step of 1.
        Lines::new(&lines).quantize(100, 0..FRAME_LINES, &mut x_quant);
        assert_eq!(x_quant[..4], [0, 1, 2, 0]);
    }

    #[test]
    fn the_start_step_codes_the_largest_line() {
        for largest in [1e-3, 1.0, 1234.5, 3.2e7, 6.0e12] {
            let lines = spectrum(|i| if i == 7 { -largest } else { largest / 3.0 });
            let quantizer = Lines::new(&lines);
            let start = quantizer.start_gain();
            let mut x_quant = [0i32; FRAME_LINES];
            quantizer.quantize(start, 0..FRAME_LINES, &mut x_quant);
            assert!(x_quant[7].unsigned_abs() <= MAX_MAGNITUDE, "{largest}");
            if start > 0 {
                // One step finer, the largest line is beyond MAX_QUANT
                // before the quantizer rounds it.
                quantizer.quantize(start - 1, 0..FRAME_LINES, &mut x_quant);
                assert!(x_quant[7].unsigned_abs() >= MAX_MAGNITUDE, "{largest}");
            }
        }
    }

    /// At the clamp of the encoder's input, 65536 times full scale, no line
    /// is above `2 * 32768 * 65536 * sum_n w(n)` (clause C.3.1.2), with
    /// the sum of the sine window `2048 * 2 / pi`; the first step for such
    /// a line leaves the loop most of the range of `global_gain`.
    #[test]
    fn the_clamp_leaves_the_start_step_within_global_gain() {
        let bound = 2.0 * 32_768.0 * 65_536.0 * 2048.0 * 2.0 / std::f64::consts::PI;
        let lines = spectrum(|i| if i == 0 { bound } else { 0.0 });
        assert_eq!(Lines::new(&lines).start_gain(), 201);
    }

    fn noise(seed: u64) -> [f64; FRAME_LINES] {
        let mut state = seed;
        spectrum(|i| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let uniform = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            uniform * 1e6 / (1.0 + i as f64 / 50.0)
        })
    }

    /// The first run of the inner loop with every scalefactor 0.
    fn flat(lines: &[f64; FRAME_LINES], fs: SamplingFrequency, available: u32) -> CodedChannel {
        let mut counting = Counting::new(
            Lines::new(lines),
            ChannelCoding::single(Windows::LONG_SINE, fs, available),
        );
        let (gain, band_limit) = first_inner_loop(&mut counting);
        counting.into_channel(gain, band_limit)
    }

    #[test]
    fn the_loop_finds_the_smallest_step_that_fits_or_the_next() {
        let fs = SamplingFrequency::Hz44100;
        for (seed, available) in [(1, 100), (2, 400), (3, 1500), (4, 4000), (5, 12_000)] {
            let lines = noise(seed);
            let channel = flat(&lines, fs, available);
            assert!(channel.bits(false) <= available, "{available}");
            let mut counting = Counting::new(
                Lines::new(&lines),
                ChannelCoding::single(Windows::LONG_SINE, fs, available),
            );
            let start = counting.lines.start_gain();
            let smallest = (start..=u8::MAX)
                .find(|&gain| counting.count_at(gain, MAX_BANDS))
                .unwrap();
            let gain = channel.global_gain();
            assert!(
                (smallest..=smallest + 1).contains(&gain),
                "{available}: {gain}, {smallest}"
            );
            assert!(gain > start + 1, "{available}: the loop ran");
        }
    }

    #[test]
    fn plenty_of_bits_leave_the_start_step() {
        let fs = SamplingFrequency::Hz48000;
        let lines = noise(2);
        let channel = flat(&lines, fs, 1 << 20);
        assert_eq!(channel.global_gain(), Lines::new(&lines).start_gain());
    }

    #[test]
    fn lines_too_loud_for_any_step_lose_their_highest_bands() {
        let fs = SamplingFrequency::Hz8000;
        let lines = spectrum(|i| if i % 2 == 0 { 1e14 } else { -1e14 });
        let silent = CodedChannel::silent(Windows::LONG_SINE, fs).bits(false);
        for available in [silent, silent + 30, 300] {
            let channel = flat(&lines, fs, available);
            assert!(channel.bits(false) <= available);
            assert!(channel.max_sfb() < 40);
            if channel.max_sfb() > 0 {
                assert_eq!(channel.global_gain(), u8::MAX);
            }
        }
    }

    #[test]
    fn amplified_bands_are_sent_with_lower_scalefactors() {
        let fs = SamplingFrequency::Hz44100;
        let lines = noise(9);
        let mut counting = Counting::new(
            Lines::new(&lines),
            ChannelCoding::single(Windows::LONG_SINE, fs, 1 << 20),
        );
        counting.scalefactors[0][3] = 5;
        counting.scalefactors[0][4] = 2;
        let gain = 140;
        assert!(counting.count_at(gain, MAX_BANDS));
        let channel = counting.into_channel(gain, MAX_BANDS);
        let (group, quantized) = channel.groups().next().unwrap();
        assert_eq!(group.scalefactors[2..6], [140, 135, 138, 140]);
        // global_gain is the first band's scalefactor, the first difference
        // 0.
        assert_eq!(channel.global_gain(), 140);
        // The lines of band 3 are quantized with the finer step.
        let bands = GroupBands::long(fs);
        let mut alone = [0i32; FRAME_LINES];
        Lines::new(&lines).quantize(135, 0..FRAME_LINES, &mut alone);
        let band = bands.offset(3)..bands.offset(4);
        assert_eq!(quantized[band.clone()], alone[band]);
    }

    #[test]
    fn a_scalefactor_below_zero_does_not_fit() {
        let fs = SamplingFrequency::Hz44100;
        let lines = noise(9);
        let mut counting = Counting::new(
            Lines::new(&lines),
            ChannelCoding::single(Windows::LONG_SINE, fs, 1 << 20),
        );
        counting.scalefactors[0][3] = 41;
        assert_eq!(counting.sf_decoder(41, 0, 3), Some(0));
        assert_eq!(counting.sf_decoder(40, 0, 3), None);
        assert!(!counting.count_at(40, MAX_BANDS));
    }

    #[test]
    fn a_channel_is_built_from_the_scalefactors_it_is_asked_for() {
        let fs = SamplingFrequency::Hz44100;
        let lines = noise(9);
        let mut counting = Counting::new(
            Lines::new(&lines),
            ChannelCoding::single(Windows::LONG_SINE, fs, 1 << 20),
        );
        assert!(counting.count_at(140, MAX_BANDS));
        // Changed after the count, as the outer loop restores them: the
        // channel is counted again with them.
        counting.scalefactors[0][3] = 2;
        let channel = counting.into_channel(140, MAX_BANDS);
        let (group, _) = channel.groups().next().unwrap();
        assert_eq!(group.scalefactors[2..5], [140, 138, 140]);
    }
}
