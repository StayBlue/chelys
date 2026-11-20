// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Perceptual noise substitution (ISO/IEC 14496-3, clause 4.6.13), which
//! MPEG-4 adds to AAC-LC: a scalefactor band whose signal is noise-like is
//! sent as no lines but the energy of the noise the decoder puts in their
//! place, chosen as Annex 4.B.11 of ISO/IEC 14496-3 describes.
//!
//! Annex 4.B.11 gives the procedure in outline: above "a lower border
//! frequency (e.g. 4 kHz)", a band is noise-like "if the corresponding
//! signal is neither tonal nor contains strong changes in energy over
//! time", judged from the psychoacoustic model's tonality and FFT
//! energies; a noise-like band's lines are set to zero before quantization,
//! and its energy sent in place of its scalefactor, in steps of 1.5 dB.

use crate::channel::FRAME_LINES;
use crate::noiseless::codebooks::Codebook;
use crate::noiseless::sections::PseudoBands;
use crate::psychoacoustic::Analysis;
use crate::sampling_frequency::SamplingFrequency;
use crate::scalefactor_bands::{GroupBands, MAX_BANDS, Transform, first_band_above, swb_offsets};
use crate::stereo::intensity::{Intensity, MAX_POSITION};
use crate::window::{SHORT_WINDOWS, Windows};

/// The lower border frequency above which bands may be substituted: Annex
/// 4.B.11's example, "e.g. 4 kHz".
///
/// Spec gap: the annex gives the border as an example only. With a
/// [`TONALITY_LIMIT`] of 0.35 and no [`FLATNESS_LIMIT`], borders of 3, 4
/// and 6 kHz graded alike: -3.121, -3.117 and -3.137 at 64 kbit/s, and
/// -2.091, -2.012 and -1.997 at 96 kbit/s (see [`TONALITY_LIMIT`] for how).
pub const BORDER_HZ: f64 = 4000.0;

/// The tonality `tb(b)` (clause C.1.4, step 7, of ISO/IEC 13818-7) below
/// which a band may be noise-like.
///
/// Spec gap: Annex 4.B.11 does not say how tonal a band may be and still
/// be noise-like.
///
/// The grades below are of PEAQ (GstPEAQ 0.6.1, black box), means over 30
/// s of each of 6 tracks of pop music and 4 of rock at 44.1 kHz in stereo,
/// with the encoder's bandwidth cut alike with and without perceptual noise
/// substitution, at 10 kHz at 64 kbit/s and 13 kHz at 96 kbit/s, so that
/// PEAQ's measures of bandwidth weigh alike: they favour noise in place of
/// treble that would otherwise be lost, which outweighs the noise it adds
/// in PEAQ's grade (see
/// [`NOISE_BITS_PER_SAMPLE`](crate::stream::NOISE_BITS_PER_SAMPLE), which
/// says too how the shared noise of a pair was sent as they were taken).
/// Without perceptual noise substitution, the means were -3.221 and
/// -2.409.
///
/// With no [`FLATNESS_LIMIT`], limits of 0.15, 0.25, 0.35, 0.5 and 0.7
/// gave -3.273, -3.279, -3.117, -2.978 and -2.938 at 64 kbit/s, and
/// -2.395, -2.236, -2.012, -1.934 and -1.936 at 96 kbit/s. The tonality of
/// the long block above its sixth line is that of the short FFTs (step 4),
/// which do not resolve the harmonics of a chord: steady harmonics read as
/// noise, and [`FLATNESS_LIMIT`] keeps them.
pub const TONALITY_LIMIT: f64 = 0.5;

/// The spectral flatness of the model's FFT in a band below which the
/// band is tonal, whatever its tonality `tb(b)`.
///
/// Spec reading: Annex 4.B.11 says the tonality "can be estimated by using
/// the tonality values calculated in the psychoacoustic model"; its
/// tonality alone was not enough, and the flatness of the model's FFT
/// energies in the band is taken with it. A chord of three notes, from 220
/// Hz, with every harmonic up to 15 kHz and a slight vibrato, graded -0.345
/// and -0.084 at 96 and 112 kbit/s without perceptual noise substitution
/// (and its bandwidth uncut), and with it -2.718 and -2.620 without a
/// flatness limit, -0.593 and -0.225 with a limit of 0.2, -0.431 and
/// -0.114 with 0.3, and -0.426 and -0.128 with 0.4; with 0.3, and its
/// shared noise sent in intensity stereo, -0.491 and -0.157. Noise has a
/// flatness of
/// about 0.56, and the bands of the music above the border a median of
/// about 0.53, against 0.10 for the chord's. With a [`TONALITY_LIMIT`] of
/// 0.5, the music graded -2.978, -2.981, -3.017 and -3.111 at 64 kbit/s,
/// and -1.934, -1.924, -1.944 and -2.012 at 96 kbit/s, with no limit and
/// limits of 0.2, 0.3 and 0.4 (see [`TONALITY_LIMIT`] for how).
pub const FLATNESS_LIMIT: f64 = 0.3;

/// The ratio of the most to the least energy of a band over the short
/// blocks of its frame, or the windows of its group, above which its energy
/// changes strongly over time.
///
/// Spec gap: Annex 4.B.11 does not say what change of energy is strong.
/// With a [`TONALITY_LIMIT`] of 0.25 and no [`FLATNESS_LIMIT`], ratios of
/// 4, 10 and 30 gave -3.306, -3.279 and -3.271 at 64 kbit/s, and -2.370,
/// -2.236 and -2.253 at 96 kbit/s (see [`TONALITY_LIMIT`] for how).
pub const ENERGY_CHANGE_LIMIT: f64 = 10.0;

/// The normalized correlation of a pair's lines in a band, either way,
/// above which both channels' noise is to be the same (see [`shared`]).
///
/// Spec gap: Annex 4.B.11 does not say when a pair's noise is to be
/// correlated. The decoder's noise is either the same in both channels,
/// correlated fully, in phase or out of phase, or independent, not
/// correlated at all; a band takes the nearest of the three.
pub const CORRELATION_LIMIT: f64 = 0.5;

/// The largest difference of two noise energies sent one after the other:
/// the scalefactor codebook codes differences of -60 to 60 (Table A.1 of
/// ISO/IEC 13818-7).
const MAX_DIFFERENCE: i32 = 60;

/// The bands of a channel's window groups coded by perceptual noise
/// substitution, each with its noise energy, `noise_nrg[g][sfb]`: the
/// energy of the noise the decoder generates in each of the group's
/// windows' band, `2^(noise_nrg / 2)` (clause 4.6.13.3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Noise {
    pub bands: [PseudoBands; SHORT_WINDOWS],
    pub energies: [[i16; MAX_BANDS]; SHORT_WINDOWS],
}

impl Noise {
    /// No band coded by perceptual noise substitution.
    pub const NONE: Self = Self {
        bands: [PseudoBands::NONE; SHORT_WINDOWS],
        energies: [[0; MAX_BANDS]; SHORT_WINDOWS],
    };

    /// Whether any band is.
    pub fn is_present(&self) -> bool {
        self.end() > 0
    }

    /// One past the highest band of any group, or 0.
    pub fn end(&self) -> usize {
        self.bands
            .iter()
            .map(|bands| bands.end())
            .max()
            .unwrap_or(0)
    }

    /// Whether band `sfb` of group `g` is coded by perceptual noise
    /// substitution.
    pub fn is_noise(&self, g: usize, sfb: usize) -> bool {
        self.bands[g].is_noise(sfb)
    }
}

/// A set of bands of each window group: bit `sfb` of `[g]` for band `sfb`
/// of group `g`.
pub type BandSet = [u64; SHORT_WINDOWS];

/// The bands of each window group of a channel coded with `windows`, whose
/// lines in the order they are sent are `spectrum`, that are noise-like as
/// Annex 4.B.11 describes, from `border` Hz up: those that are neither
/// tonal, by `analysis`, the model's analysis of the frame, nor change
/// their energy strongly over time.
///
/// A band is tonal where any of the partitions its lines fall in, in any of
/// its group's windows, has a tonality `tb(b)` (clause C.1.4, step 7, of
/// ISO/IEC 13818-7) of [`TONALITY_LIMIT`] or more, or where the energies of
/// the model's FFT lines in the band, in any of its group's windows, have
/// a spectral flatness below [`FLATNESS_LIMIT`].
///
/// Spec gap: Annex 4.B.11 evaluates changes in energy over time "using the
/// FFT energies calculated in the psychoacoustic model", without saying
/// which. A band of a long window takes those of the model's eight short
/// FFTs, which span the frame's window, in the short bands that overlap
/// its lines; a band of a group of short windows takes the energy of its
/// lines in each of the group's windows. Its energy changes strongly where
/// the most of them exceeds the least by more than [`ENERGY_CHANGE_LIMIT`].
pub fn noise_like(
    windows: Windows,
    sampling_frequency: SamplingFrequency,
    spectrum: &[f64; FRAME_LINES],
    analysis: &Analysis,
    border: f64,
) -> BandSet {
    let transform = windows.sequence.transform();
    let first = first_band_above(sampling_frequency, transform, border);
    let energy = |lines: &[f64]| -> f64 { lines.iter().map(|x| x * x).sum() };
    let mut set = [0; SHORT_WINDOWS];
    let mut first_window = 0;
    for (g, (start, bands)) in windows.groups_with_start(sampling_frequency).enumerate() {
        let lines = &spectrum[start..start + bands.lines()];
        for sfb in first..bands.num_swb() {
            if energy(&lines[bands.band(sfb)]) == 0.0 {
                continue;
            }
            let (tonality, change) = match transform {
                Transform::Long => (
                    analysis.long.tonality[sfb],
                    long_energy_change(sampling_frequency, analysis, sfb),
                ),
                Transform::Short => {
                    let thresholds = &analysis.short[first_window..first_window + bands.windows()];
                    let tonality = (thresholds.iter())
                        .map(|threshold| threshold.tonality[sfb])
                        .fold(0.0, f64::max);
                    let energies =
                        (0..bands.windows()).map(|w| energy(&lines[bands.window_band(w, sfb)]));
                    (tonality, change(energies))
                }
            };
            let flat = match transform {
                Transform::Long => analysis.long.flatness[sfb],
                Transform::Short => {
                    let thresholds = &analysis.short[first_window..first_window + bands.windows()];
                    thresholds
                        .iter()
                        .map(|t| t.flatness[sfb])
                        .fold(1.0, f64::min)
                }
            };
            if tonality < TONALITY_LIMIT && flat >= FLATNESS_LIMIT && change <= ENERGY_CHANGE_LIMIT
            {
                set[g] |= 1 << sfb;
            }
        }
        first_window += bands.windows();
    }
    set
}

/// The ratio of the most to the least energy of band `sfb` of a long
/// window over the eight short FFTs of `analysis`, in the short bands
/// that overlap its lines (see [`noise_like`]).
fn long_energy_change(
    sampling_frequency: SamplingFrequency,
    analysis: &Analysis,
    sfb: usize,
) -> f64 {
    let long = swb_offsets(sampling_frequency, Transform::Long);
    let short = swb_offsets(sampling_frequency, Transform::Short);
    // The band's lines on the scale of a short window's, eight times as
    // wide.
    let scale = (Transform::Long.lines() / Transform::Short.lines()) as u16;
    let (low, high) = (long[sfb] / scale, long[sfb + 1].div_ceil(scale));
    let overlapping: Vec<usize> = (0..short.len() - 1)
        .filter(|&n| short[n] < high && short[n + 1] > low)
        .collect();
    change(
        (analysis.short.iter())
            .map(|threshold| overlapping.iter().map(|&n| threshold.epart[n]).sum()),
    )
}

/// The ratio of the most of `energies` to the least: infinite where the
/// least is 0 and the most is not, and 1 where all are 0.
fn change(energies: impl Iterator<Item = f64>) -> f64 {
    let (least, most) = energies.fold((f64::INFINITY, 0.0f64), |(least, most), e| {
        (least.min(e), most.max(e))
    });
    if most == 0.0 { 1.0 } else { most / least }
}

/// `noise_nrg` of a band whose lines' energy in a window is `energy`, above
/// 0: the decoder scales its noise to the energy `2^(noise_nrg / 2)`
/// (clause 4.6.13.3), so `noise_nrg = 2 log2(energy)`, to the nearest
/// step of 1.5 dB.
///
/// Spec reading: the pseudo code of clause 4.6.13.3 scales the random
/// vector by `scale *= 2.0^(0.25*noise_nrg[g][sfb]) / sqrt_nrg`, with
/// `scale` set nowhere; it is read as `scale =`, which gives each window's
/// band the energy that its noise energy states, `2^(0.5 noise_nrg)`, in
/// steps of 1.5 dB, as the clause says. Settled by decoding with ffmpeg
/// and FAAD2 (black box): white noise so coded decodes at its level within
/// 1 dB in each range above the border; the conformance test is
/// `ffmpeg_and_faad_decode_noise_at_its_energy`.
pub fn energy_index(energy: f64) -> i16 {
    (2.0 * energy.log2()).round() as i16
}

/// The noise energy of band `sfb` of a group of `bands`, whose lines in
/// the order they are sent are `lines`, from the mean energy of its lines
/// in each of the group's windows; `None` where it has none.
fn band_energy(bands: GroupBands, lines: &[f64], sfb: usize) -> Option<i16> {
    let energy: f64 = lines[bands.band(sfb)].iter().map(|x| x * x).sum();
    (energy > 0.0).then(|| energy_index(energy / bands.windows() as f64))
}

/// The bands of `set` of a channel coded with `windows`, its lines in the
/// order they are sent in `spectrum`, that [`substitute`] substitutes:
/// those with energy, each within reach of the one before it.
///
/// Spec gap: the noise energies are sent as differences that the
/// scalefactor codebook codes from -60 to 60, 90 dB either way, which do
/// not reach every energy. A band whose energy lies beyond the reach of the
/// band substituted before it is not substituted, and keeps its lines.
pub fn reachable(
    windows: Windows,
    sampling_frequency: SamplingFrequency,
    set: &BandSet,
    spectrum: &[f64; FRAME_LINES],
) -> BandSet {
    let mut reachable = [0; SHORT_WINDOWS];
    let mut last: Option<i32> = None;
    for (g, (start, bands)) in windows.groups_with_start(sampling_frequency).enumerate() {
        let lines = &spectrum[start..start + bands.lines()];
        for sfb in (0..bands.num_swb()).filter(|&sfb| (set[g] >> sfb) & 1 == 1) {
            let Some(index) = band_energy(bands, lines, sfb).map(i32::from) else {
                continue;
            };
            if last.is_some_and(|last| (index - last).abs() > MAX_DIFFERENCE) {
                continue;
            }
            last = Some(index);
            reachable[g] |= 1 << sfb;
        }
    }
    reachable
}

/// Codes those of the bands `set` of a channel coded with `windows` that
/// are [`reachable`] by perceptual noise substitution, its lines in the
/// order they are sent in `spectrum`: gives each band's noise energy, from
/// the mean energy of its lines in each of its group's windows, and sets
/// its lines to zero (Annex 4.B.11).
pub fn substitute(
    windows: Windows,
    sampling_frequency: SamplingFrequency,
    set: &BandSet,
    spectrum: &mut [f64; FRAME_LINES],
) -> Noise {
    let set = reachable(windows, sampling_frequency, set, spectrum);
    let mut noise = Noise::NONE;
    for (g, (start, bands)) in windows.groups_with_start(sampling_frequency).enumerate() {
        let lines = &mut spectrum[start..start + bands.lines()];
        for sfb in (0..bands.num_swb()).filter(|&sfb| (set[g] >> sfb) & 1 == 1) {
            noise.bands[g].set(sfb, Some(Codebook::NOISE));
            noise.energies[g][sfb] = band_energy(bands, lines, sfb).expect("reachable");
            lines[bands.band(sfb)].fill(0.0);
        }
    }
    noise
}

/// Measures the noise energies of the bands of `noise` again, from the
/// lines `spectrum` of a channel coded with `windows`, in the order they
/// are sent: those the filters of temporal noise shaping did not shape,
/// where the channel drops them.
///
/// Spec gap: an energy beyond the reach of the one before it is brought
/// within it, the nearest it can be sent; a band without energy is given
/// that of the band before it, or, the first, the least within reach of
/// the next.
pub fn measure(
    windows: Windows,
    sampling_frequency: SamplingFrequency,
    noise: &mut Noise,
    spectrum: &[f64; FRAME_LINES],
) {
    let mut measured = Vec::new();
    for (g, (start, bands)) in windows.groups_with_start(sampling_frequency).enumerate() {
        let lines = &spectrum[start..start + bands.lines()];
        for sfb in (0..bands.num_swb()).filter(|&sfb| noise.is_noise(g, sfb)) {
            measured.push((g, sfb, band_energy(bands, lines, sfb).map(i32::from)));
        }
    }
    let first = measured.iter().find_map(|&(_, _, index)| index);
    let mut last = first.map(|first| first - MAX_DIFFERENCE).unwrap_or(0);
    for (g, sfb, index) in measured {
        let index = index.unwrap_or(last);
        let index = index.clamp(last - MAX_DIFFERENCE, last + MAX_DIFFERENCE);
        noise.energies[g][sfb] = index as i16;
        last = index;
    }
}

/// The bands of noise of a pair whose noise is to be the same in both
/// channels, which the right channel takes from the left in intensity
/// stereo (see [`shared`]): the bands, and their intensity codebooks and
/// positions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shared {
    pub set: BandSet,
    pub intensity: Intensity,
}

impl Shared {
    /// Keeps only the bands of `set`.
    pub fn keep(&mut self, set: &BandSet) {
        for ((kept, bands), set) in self.set.iter_mut().zip(&mut self.intensity.bands).zip(set) {
            *kept &= set;
            for sfb in (0..MAX_BANDS).filter(|&sfb| (*kept >> sfb) & 1 == 0) {
                bands.set(sfb, None);
            }
        }
    }
}

/// The bands of `set` of a pair coded with `windows`, its channels' lines
/// in the order they are sent `left` and `right`, whose noise is to be the
/// same in both channels: those whose lines' normalized correlation
/// exceeds [`CORRELATION_LIMIT`] either way, each coded in intensity
/// stereo in the right channel, in phase or out of phase as they
/// correlate, with the position of the channels' energies, `is_position =
/// NINT(2 log2(El / Er))` (clause C.6.2 of ISO/IEC 13818-7), within
/// [`MAX_POSITION`] of 0 so that any two are within reach of each other.
pub fn shared(
    windows: Windows,
    sampling_frequency: SamplingFrequency,
    set: &BandSet,
    [left, right]: [&[f64; FRAME_LINES]; 2],
) -> Shared {
    let mut shared = Shared {
        set: [0; SHORT_WINDOWS],
        intensity: Intensity::NONE,
    };
    let energy = |x: &[f64]| x.iter().map(|x| x * x).sum::<f64>();
    for (g, (start, bands)) in windows.groups_with_start(sampling_frequency).enumerate() {
        for sfb in (0..bands.num_swb()).filter(|&sfb| (set[g] >> sfb) & 1 == 1) {
            let band = bands.band(sfb);
            let range = start + band.start..start + band.end;
            let (l, r) = (&left[range.clone()], &right[range]);
            let (el, er) = (energy(l), energy(r));
            if el == 0.0 || er == 0.0 {
                continue;
            }
            let product: f64 = l.iter().zip(r).map(|(l, r)| l * r).sum();
            let correlation = product / (el * er).sqrt();
            let position = (2.0 * (el / er).log2()).round();
            if correlation.abs() <= CORRELATION_LIMIT || position.abs() > f64::from(MAX_POSITION) {
                continue;
            }
            let codebook = if correlation > 0.0 {
                Codebook::INTENSITY
            } else {
                Codebook::INTENSITY_OUT_OF_PHASE
            };
            shared.set[g] |= 1 << sfb;
            shared.intensity.bands[g].set(sfb, Some(codebook));
            shared.intensity.positions[g][sfb] = position as i8;
        }
    }
    shared
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noiseless::decoder::Noise as Random;

    #[test]
    fn energy_indexes_step_by_one_and_a_half_db() {
        assert_eq!(energy_index(1.0), 0);
        assert_eq!(energy_index(2.0), 2);
        assert_eq!(energy_index(0.5), -2);
        // 2^(0.5 * 20) = 1024.
        assert_eq!(energy_index(1024.0), 20);
        let step = 10.0 * 2f64.powf(0.5).log10();
        assert!((step - 1.505).abs() < 0.001);
    }

    #[test]
    fn substituted_bands_are_zeroed_and_keep_their_energy() {
        let fs = SamplingFrequency::Hz44100;
        let windows = Windows::LONG_SINE;
        let mut random = Random::new(3);
        let mut spectrum = [0.0; FRAME_LINES];
        for x in spectrum.iter_mut() {
            *x = f64::from(random.below(2001) as i32 - 1000);
        }
        let original = spectrum;
        let mut set = [0; SHORT_WINDOWS];
        set[0] = (1 << 40) | (1 << 41);
        let noise = substitute(windows, fs, &set, &mut spectrum);
        let offsets = swb_offsets(fs, Transform::Long);
        for sfb in 0..offsets.len() - 1 {
            let band = usize::from(offsets[sfb])..usize::from(offsets[sfb + 1]);
            if sfb == 40 || sfb == 41 {
                assert!(noise.is_noise(0, sfb));
                assert!(spectrum[band.clone()].iter().all(|&x| x == 0.0));
                let energy: f64 = original[band].iter().map(|x| x * x).sum();
                let decoded = 2f64.powf(0.5 * f64::from(noise.energies[0][sfb]));
                // Within half a step of 1.5 dB.
                assert!((10.0 * (decoded / energy).log10()).abs() <= 0.76);
            } else {
                assert!(!noise.is_noise(0, sfb));
                assert_eq!(spectrum[band.clone()], original[band]);
            }
        }
        assert_eq!(noise.end(), 42);
    }

    #[test]
    fn a_band_beyond_reach_of_the_last_keeps_its_lines() {
        let fs = SamplingFrequency::Hz48000;
        let windows = Windows::LONG_SINE;
        let offsets = swb_offsets(fs, Transform::Long);
        let mut spectrum = [0.0; FRAME_LINES];
        // Band 40 at 1, band 41 some 100 dB louder: 2^(0.5 * 66) is
        // 10^9.9.
        let band = |sfb: usize| usize::from(offsets[sfb])..usize::from(offsets[sfb + 1]);
        spectrum[band(40)].fill(1.0);
        spectrum[band(41)].fill(1.0e5);
        spectrum[band(42)].fill(2.0);
        let mut set = [0; SHORT_WINDOWS];
        set[0] = (1 << 40) | (1 << 41) | (1 << 42);
        let noise = substitute(windows, fs, &set, &mut spectrum);
        assert!(noise.is_noise(0, 40) && !noise.is_noise(0, 41) && noise.is_noise(0, 42));
        assert!(spectrum[band(41)].iter().all(|&x| x == 1.0e5));
    }

    #[test]
    fn like_channels_share_their_noise_and_unlike_ones_do_not() {
        let fs = SamplingFrequency::Hz48000;
        let windows = Windows::LONG_SINE;
        let offsets = swb_offsets(fs, Transform::Long);
        let band = |sfb: usize| usize::from(offsets[sfb])..usize::from(offsets[sfb + 1]);
        let mut random = Random::new(9);
        let mut draw = || f64::from(random.below(2001) as i32 - 1000);
        let (mut left, mut right) = ([0.0; FRAME_LINES], [0.0; FRAME_LINES]);
        for i in band(40) {
            left[i] = draw();
            right[i] = 0.5 * left[i];
        }
        for i in band(41) {
            left[i] = draw();
            right[i] = draw();
        }
        for i in band(42) {
            left[i] = draw();
            right[i] = -left[i];
        }
        // Alike, but 50 dB apart: beyond the reach of a position.
        for i in band(43) {
            left[i] = draw();
            right[i] = 0.003 * left[i];
        }
        let mut set = [0; SHORT_WINDOWS];
        set[0] = 0b1111 << 40;
        let mut shared = shared(windows, fs, &set, [&left, &right]);
        assert_eq!(shared.set[0], (1 << 40) | (1 << 42));
        let bands = shared.intensity.bands[0];
        assert_eq!(bands.codebook(40), Some(Codebook::INTENSITY));
        assert_eq!(bands.codebook(42), Some(Codebook::INTENSITY_OUT_OF_PHASE));
        // 2 log2(4) and 2 log2(1).
        assert_eq!(shared.intensity.positions[0][40], 4);
        assert_eq!(shared.intensity.positions[0][42], 0);
        let mut kept = [0; SHORT_WINDOWS];
        kept[0] = 1 << 42;
        shared.keep(&kept);
        assert_eq!(shared.set[0], 1 << 42);
        assert_eq!(shared.intensity.bands[0].codebook(40), None);
    }

    #[test]
    fn the_least_and_most_energy_give_the_change() {
        assert_eq!(change([1.0, 4.0, 2.0].into_iter()), 4.0);
        assert_eq!(change([0.0, 0.0].into_iter()), 1.0);
        assert_eq!(change([0.0, 1.0].into_iter()), f64::INFINITY);
    }
}
