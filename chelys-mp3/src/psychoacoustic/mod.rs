// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Psychoacoustic model 2 (clause D.2) as adapted for Layer III in clause
//! C.1.5.3.2.1, figures C.6.a to C.6.d.
//!
//! For each granule the model gives the ratio of allowed noise to signal
//! energy in every scalefactor band, from which the allowed distortion
//! `xmin` of the outer iteration loop follows, and the perceptual entropy.
//! [`PsychoacousticModel::granule`] gives the threshold for long blocks;
//! [`switching::BlockSwitching`] also chooses the block type and gives the
//! threshold for short blocks where it switches to them.

mod fft;
pub mod partitions;
pub mod switching;

use std::f64::consts::PI;

use super::granule::{BlockType, GRANULE_LINES};
use super::header::{SamplingFrequency, Version};
use super::scalefactor_bands::{self, LONG_BANDS, SHORT_BANDS, SHORT_WINDOWS};
use super::stereo;
use fft::RealFft;
use partitions::{BandWeights, Partition, ShortPartition};

const LONG_FFT: usize = 1024;
/// The FFT of the model run with a shift length of 192 samples (clause
/// C.1.5.3.2.1, "General considerations").
const SHORT_FFT: usize = 256;
const SHORT_SHIFT: usize = 192;

/// The unpredictability `cw` comes from the long FFT below this line, from
/// the short FFT below [`SHORT_CW_LINES`], and is 0.4 above (clause
/// C.1.5.3.2.1).
const LONG_CW_LINES: usize = 6;
const SHORT_CW_LINES: usize = 206;
const HIGH_CW: f64 = 0.4;

/// The tonality conversion `tb = conv1 + conv2 * ln(cb)` (clause
/// C.1.5.3.2.1, figure C.6.b).
const CONV1: f64 = -0.299;
const CONV2: f64 = -0.43;
/// Tone masking noise and noise masking tone, in dB, for every partition
/// (clause C.1.5.3.2.1).
const TMN: f64 = 29.0;
const NMT: f64 = 6.0;
/// The pre-echo control constants (clause C.1.5.3.2.1).
const RPELEV: f64 = 2.0;
const RPELEV2: f64 = 16.0;

/// Samples are scaled to 16-bit integers before the FFT, which is
/// normalised by `1 / N`, so that the energies are on the scale of the
/// threshold in quiet `qthr` of Table C.7.
///
/// Spec gap: the standard does not say what scale `qthr` is on. Clause
/// D.2.4 l) puts the absolute thresholds of Table D.4 relative to a sine of
/// +-1/2 LSB and asks for them to be converted "after considering the FFT
/// normalization actually used", and Table D.4 sets 0 dB at 96 dB below the
/// energy of a sine of amplitude 32 760. With 16-bit samples and a `1 / N`
/// FFT, such a sine peaks at an energy of `(32 760 / 4)^2` in the
/// Hann-windowed spectrum, putting 0 dB at about 0.017. Table C.7's lowest
/// per-line threshold, about 0.009, then lies near the bottom of Table D.4
/// (-5 dB), as a threshold in quiet should. An unnormalised FFT would put it
/// 60 dB lower.
const PCM_SCALE: f64 = 32_768.0;

/// The samples the model keeps: the current granule and the two before it.
const HISTORY: usize = 3 * GRANULE_LINES;

/// Where in the history the filterbank's current block is centred, so that
/// the analysis is centred on it as clause D.2.1 b) asks.
///
/// Spec gap: clause D.2.1 b) asks for the centring but leaves the delay to the
/// implementation. It is derived here from this encoder's filterbank. The
/// filterbank returns the block formed by the subband samples of the previous
/// granule and the current one. Subband sample `t` of a granule is computed
/// with input sample `32 t + 31` of that granule newest, and the analysis
/// window (Table C.1) is symmetric about the 256th sample back, so it is
/// centred on sample `32 t - 225`. The block runs from sample 0 of the
/// previous granule, centred on sample `-576 - 225 = -801` of the current one,
/// to sample 17 of the current granule, centred on `32 * 17 - 225 = 319`, so
/// its middle is sample -241. The Hann windows are centred half a sample
/// earlier, which does not matter at this resolution.
const BLOCK_CENTRE: usize = 2 * GRANULE_LINES - 241;

/// The spreading function of clause D.2.3 as changed for Layer III in clause
/// C.1.5.3.2.1: the spread from Bark value `i` into Bark value `j`.
///
/// Spec reading: clause C.1.5.3.2.1 prints the change as "If j >= i tmpy = 3,0
/// (j-i), else tmpy = 1,5 (j-i)". Read literally, as a replacement of `tmpy`,
/// the spread would grow without bound towards high frequencies, so it is read
/// as a change to the scaling of `tmpx`, which `tmpx = 1,05 (j-i)` sets in
/// clause D.2.3. The two factors are also swapped here: the `norm` column of
/// Table C.7 matches the reciprocal of the spread summed over the partitions,
/// as in clause D.2.4 f), to within the rounding of the table only with a
/// factor of 1.5 towards higher frequencies and 3.0 towards lower ones. With
/// the factors as printed it is off by up to 0.37.
///
/// Values of 10^-6 and below are set to zero (clause C.1.5.3.2.1).
pub(crate) fn spreading_function(i: f64, j: f64) -> f64 {
    let tmpx = if j >= i { 1.5 } else { 3.0 } * (j - i);
    let x = 8.0 * f64::min((tmpx - 0.5).powi(2) - 2.0 * (tmpx - 0.5), 0.0);
    let tmpy = 15.811389 + 7.5 * (tmpx + 0.474) - 17.5 * (1.0 + (tmpx + 0.474).powi(2)).sqrt();
    if tmpy < -100.0 {
        return 0.0;
    }
    let spread = 10f64.powf((x + tmpy) / 10.0);
    if spread > 1e-6 { spread } else { 0.0 }
}

/// The model's output for one granule of one channel.
#[derive(Clone, Debug, PartialEq)]
pub struct Masking {
    /// The ratio of the allowed noise to the signal energy in each long
    /// scalefactor band, `ratio(sb) = thm(sb) / en(sb)` (figure C.6.c).
    ///
    /// Spec gap: infinite for a band in which the spectrum has no energy,
    /// which the standard does not provide for.
    pub ratio: [f64; LONG_BANDS],
    /// The perceptual entropy `pe` (clause C.1.5.3.2.1).
    pub perceptual_entropy: f64,
    /// The energy `r(w)^2` of each line of the long FFT below line 512, for
    /// the MS stereo decision (Annex G.2 a)).
    pub spectrum: Vec<f64>,
    /// The threshold `thr(b)` of each threshold calculation partition of
    /// [`partitions::partitions`] (figure C.6.b), on the scale of
    /// [`spectrum`](Self::spectrum).
    pub threshold: Vec<f64>,
}

impl Masking {
    /// The allowed distortion of each scalefactor band,
    /// `xmin(sb) = ratio(sb) * en(sb) / bw(sb)` (clause C.1.5.4.1).
    ///
    /// Spec reading: `en(sb)` is taken as the energy of the band's lines in
    /// `xr` and `bw(sb)` as their number, so that `xmin` is a mean energy per
    /// line on the scale of the distortion `xfsf` it is compared with (clause
    /// C.1.5.4.3.3). Spec gap: a band whose lines are all zero allows none,
    /// where its ratio would make the product undefined.
    pub fn allowed_distortion(
        &self,
        xr: &[f64; GRANULE_LINES],
        sampling_frequency: SamplingFrequency,
    ) -> [f64; LONG_BANDS] {
        let bands = scalefactor_bands::long_bands(sampling_frequency);
        std::array::from_fn(|sb| {
            let band = bands[sb];
            let energy: f64 = xr[band.start..band.end()].iter().map(|x| x * x).sum();
            if energy == 0.0 {
                0.0
            } else {
                self.ratio[sb] * energy / band.width as f64
            }
        })
    }
}

/// The model's output for one granule of one channel coded with short
/// blocks.
#[derive(Clone, Debug, PartialEq)]
pub struct ShortMasking {
    /// The ratio of the allowed noise to the signal energy in each short
    /// scalefactor band of each short window, indexed `[window][sb]`
    /// (figure C.6.d). Infinite where the band has no energy, as in
    /// [`Masking::ratio`].
    pub ratio: [[f64; SHORT_BANDS]; SHORT_WINDOWS],
    /// The perceptual entropy `pe`, which figure C.6.a computes from the
    /// long-block threshold whatever the block type.
    pub perceptual_entropy: f64,
    /// As [`Masking::spectrum`]: the long FFT is computed whatever the block
    /// type.
    pub spectrum: Vec<f64>,
}

impl ShortMasking {
    /// The allowed distortion of each short scalefactor band of each window,
    /// `xmin = ratio * en / bw` as for long blocks (see
    /// [`Masking::allowed_distortion`]), in the order the scalefactors are
    /// sent: `3 * sfb + window`. `xr` holds the lines in the order they are
    /// sent too (see
    /// [`short_coding_order`](scalefactor_bands::short_coding_order)).
    pub fn allowed_distortion(
        &self,
        xr: &[f64; GRANULE_LINES],
        sampling_frequency: SamplingFrequency,
    ) -> [f64; SHORT_BANDS * SHORT_WINDOWS] {
        let bands = scalefactor_bands::coding_bands(BlockType::Short, sampling_frequency);
        std::array::from_fn(|index| {
            let band = bands[index];
            let energy: f64 = xr[band.start..band.end()].iter().map(|x| x * x).sum();
            let ratio = self.ratio[index % SHORT_WINDOWS][index / SHORT_WINDOWS];
            if energy == 0.0 {
                0.0
            } else {
                ratio * energy / band.width as f64
            }
        })
    }
}

/// What [`PsychoacousticModel::analyse`] keeps of a granule for its
/// threshold for short blocks: the energies of the lines of the three short
/// blocks' FFTs, and the rest of [`ShortMasking`].
struct ShortSpectra {
    energy: [Vec<f64>; SHORT_WINDOWS],
    perceptual_entropy: f64,
    spectrum: Vec<f64>,
}

/// The magnitude `r` and phase `f` of one FFT line (clause D.2.4 b)), the
/// phase kept as its cosine and sine (see [`Analysis::next_block`]).
#[derive(Clone, Copy, Debug)]
struct Polar {
    r: f64,
    cos: f64,
    sin: f64,
}

impl Default for Polar {
    fn default() -> Self {
        Self {
            r: 0.0,
            cos: 1.0,
            sin: 0.0,
        }
    }
}

/// One FFT of the model with its Hann window, and the spectra of the two
/// blocks before, from which it predicts the next (clause D.2.4 c)).
struct Analysis {
    fft: RealFft,
    window: Vec<f64>,
    /// The windowed samples, kept from one block to the next for their
    /// memory.
    windowed: Vec<f64>,
    /// The lines whose unpredictability the model uses, from DC up; above
    /// them it is not computed, and the phases not kept.
    predicted: usize,
    last: Vec<Polar>,
    before_last: Vec<Polar>,
}

impl Analysis {
    fn new(size: usize, predicted: usize) -> Self {
        // Clause D.2.4 b): sw(i) = s(i) * (0.5 - 0.5 cos(2 pi (i - 0.5) / N))
        // for i = 1..N.
        let window = (0..size)
            .map(|n| 0.5 - 0.5 * (2.0 * PI * (n as f64 + 0.5) / size as f64).cos())
            .collect();
        // Clause D.2.1 asks for the stored spectra to start at zero.
        assert!(predicted <= size / 2 + 1, "lines to predict");
        Self {
            fft: RealFft::new(size),
            windowed: Vec::with_capacity(size),
            window,
            predicted,
            last: vec![Polar::default(); predicted],
            before_last: vec![Polar::default(); predicted],
        }
    }

    /// Transforms the windowed `samples` and returns the energies of the
    /// lines from DC to Nyquist and the unpredictabilities `c` of the lines
    /// predicted (clause D.2.4 b) to d)), remembering their spectrum for the
    /// next block.
    fn next_block(&mut self, samples: &[f64]) -> (Vec<f64>, Vec<f64>) {
        // Exact: the size is a power of two, so multiplying by its
        // reciprocal divides by it, without a division for each sample.
        let reciprocal = 1.0 / self.window.len() as f64;
        self.windowed.clear();
        self.windowed.extend(
            (samples.iter().zip(&self.window)).map(|(s, w)| s * PCM_SCALE * w * reciprocal),
        );
        let x = self.fft.transform(&self.windowed);

        // Spec reading: clause D.2.4 b) takes the magnitude `r` of every
        // line, of which clause D.2.4 e) and figures C.6.b and C.6.d take
        // only the square, the energy; it is taken as the sum of the squares
        // of the real and imaginary parts, without the square root, for
        // speed. Only the predicted lines need it (clause D.2.4 c) and d)).
        // The energies are the same but for rounding.
        let energy: Vec<f64> = x.iter().map(|x| x.re * x.re + x.im * x.im).collect();
        let mut c = Vec::with_capacity(self.predicted);
        for (w, x) in x[..self.predicted].iter().enumerate() {
            // Spec reading: clause D.2.4 c) predicts the phase as
            // `2 f(t-1) - f(t-2)`, of which clause D.2.4 d) takes only the
            // cosine and sine. Those are had here from the cosines and sines
            // of the two phases, by the formulas for a double angle and a
            // difference of angles, and those of a line's phase from its real
            // and imaginary parts divided by its magnitude, rather than
            // through the angles themselves, for speed. The values are the
            // same but for rounding. A line without magnitude is taken to
            // have the phase 0, as `atan2(0, 0)` gives and the stored spectra
            // start with.
            let polar = match energy[w].sqrt() {
                0.0 => Polar::default(),
                r => Polar {
                    r,
                    cos: x.re / r,
                    sin: x.im / r,
                },
            };
            let (last, before_last) = (self.last[w], self.before_last[w]);
            let predicted_r = 2.0 * last.r - before_last.r;
            // cos and sin of 2 f(t-1), then of 2 f(t-1) - f(t-2).
            let (cos_double, sin_double) = (
                last.cos * last.cos - last.sin * last.sin,
                2.0 * last.cos * last.sin,
            );
            let (predicted_cos, predicted_sin) = (
                cos_double * before_last.cos + sin_double * before_last.sin,
                sin_double * before_last.cos - cos_double * before_last.sin,
            );
            let (re, im) = (
                x.re - predicted_r * predicted_cos,
                x.im - predicted_r * predicted_sin,
            );
            let distance = (re * re + im * im).sqrt();
            let scale = polar.r + predicted_r.abs();
            // Spec gap: silence that was predicted to stay silent divides zero
            // by zero. The line has no energy, so its unpredictability carries
            // no weight and any value does.
            c.push(if scale == 0.0 { 0.0 } else { distance / scale });
            // The spectrum before last is not needed again, and takes this
            // block's, line by line, to become the last.
            self.before_last[w] = polar;
        }
        std::mem::swap(&mut self.last, &mut self.before_last);
        (energy, c)
    }
}

/// Psychoacoustic model 2 for one channel, fed the same PCM granules as the
/// channel's filterbank.
pub struct PsychoacousticModel {
    partitions: &'static [Partition],
    band_weights: &'static [BandWeights; LONG_BANDS],
    /// For each partition `b`, the partitions `bb` that spread into it, with
    /// `sprdngf(bval(bb), bval(b))`, in order: those with a spread of zero
    /// are left out, as adding nothing to the sums they are in changes none.
    spread_into: Vec<Vec<(usize, f64)>>,
    short_partitions: &'static [ShortPartition],
    short_band_weights: &'static [BandWeights; SHORT_BANDS],
    /// As `spread_into`, between the short partitions.
    short_spread_into: Vec<Vec<(usize, f64)>>,
    /// The normalising constant of each short partition.
    short_norm: Vec<f64>,
    /// The ratio of the threshold to the spread energy of each short
    /// partition.
    short_snr: Vec<f64>,
    /// The threshold in quiet that each partition allows noise, and each
    /// short partition: see [`noise_in_quiet`].
    noise_in_quiet: Vec<f64>,
    short_noise_in_quiet: Vec<f64>,
    history: [f64; HISTORY],
    long: Analysis,
    short: Analysis,
    /// The thresholds `nb` of the last two granules, for pre-echo control.
    nb_last: Vec<f64>,
    nb_before_last: Vec<f64>,
}

impl PsychoacousticModel {
    pub fn new(sampling_frequency: SamplingFrequency) -> Self {
        let partitions = partitions::partitions(sampling_frequency);
        let short_partitions = partitions::short_partitions(sampling_frequency);
        let short_spreading = spreading_matrix(short_partitions.iter().map(|p| p.bval));
        // Spec reading: with the tables of ISO/IEC 11172-3, the normalising
        // constant is computed from the short partitions' Bark values as
        // `rnorm` in clause D.2.4 f), the reciprocal of the spreading
        // function summed over the partitions, which the long tables' `norm`
        // column matches. The short tables' own column is a copy of the long
        // one (see `partitions::short_partitions`) and differs from this by
        // up to 1.9 dB. The tables of ISO/IEC 13818-3 print a column of their
        // own for short blocks, which is used as printed, as their long
        // tables' column is.
        let short_norm = match sampling_frequency.version() {
            Version::Mpeg1 => (0..short_partitions.len())
                .map(|b| 1.0 / short_spreading.iter().map(|from| from[b]).sum::<f64>())
                .collect(),
            Version::Mpeg2 => short_partitions.iter().map(|p| p.norm).collect(),
        };
        Self {
            partitions,
            band_weights: partitions::band_weights(sampling_frequency),
            spread_into: spread_into(&spreading_matrix(partitions.iter().map(|p| p.bval))),
            short_partitions,
            short_band_weights: partitions::short_band_weights(sampling_frequency),
            short_spread_into: spread_into(&short_spreading),
            short_norm,
            short_snr: short_partitions.iter().map(|p| p.snr.ratio()).collect(),
            noise_in_quiet: noise_in_quiet(partitions.iter().map(|p| (p.qthr, p.width, p.bval))),
            short_noise_in_quiet: noise_in_quiet(
                short_partitions.iter().map(|p| (p.qthr, p.width, p.bval)),
            ),
            history: [0.0; HISTORY],
            long: Analysis::new(LONG_FFT, LONG_CW_LINES),
            // The lines of the short FFT that stand for the long FFT's
            // below SHORT_CW_LINES (see `analyse`).
            short: Analysis::new(SHORT_FFT, (SHORT_CW_LINES - 1 + 2) / 4 + 1),
            nb_last: vec![0.0; partitions.len()],
            nb_before_last: vec![0.0; partitions.len()],
        }
    }

    /// Analyses the next granule of PCM, nominally in -1.0 to 1.0, centred on
    /// the block the filterbank returns for the same granule, for long
    /// blocks.
    #[cfg(any(test, feature = "unstable"))]
    pub fn granule(&mut self, pcm: &[f64; GRANULE_LINES]) -> Masking {
        self.analyse(pcm).0
    }

    /// Analyses the next granule like [`Self::granule`], giving the
    /// threshold for long blocks, and the spectra that
    /// [`Self::short_masking`] takes the threshold for short blocks from.
    fn analyse(&mut self, pcm: &[f64; GRANULE_LINES]) -> (Masking, ShortSpectra) {
        self.history.copy_within(GRANULE_LINES.., 0);
        self.history[HISTORY - GRANULE_LINES..].copy_from_slice(pcm);

        let (energy_short, cw_short) = self.short_blocks();
        let long_start = BLOCK_CENTRE - LONG_FFT / 2;
        let (energy, cw_long) = self
            .long
            .next_block(&self.history[long_start..][..LONG_FFT]);
        let cw = |w: usize| match w {
            ..LONG_CW_LINES => cw_long[w],
            LONG_CW_LINES..SHORT_CW_LINES => cw_short[(w + 2) / 4],
            _ => HIGH_CW,
        };

        // Figure C.6.b: energy and unpredictability in each partition...
        let weighted: Vec<f64> = (energy.iter().enumerate())
            .map(|(w, &energy)| cw(w) * energy)
            .collect();
        let (eb, cb): (Vec<f64>, Vec<f64>) = self
            .partitions
            .iter()
            .map(|p| {
                let lines = p.start..p.start + p.width;
                (energy[lines.clone()].iter().zip(&weighted[lines]))
                    .fold((0.0, 0.0), |(e, c), (de, dc)| (e + de, c + dc))
            })
            .unzip();

        // ...convolved with the spreading function, to give the tonality,
        // the required SNR and the threshold...
        let mut nb = Vec::with_capacity(self.partitions.len());
        for (b, p) in self.partitions.iter().enumerate() {
            let (mut ecb, mut ct) = (0.0, 0.0);
            for &(bb, spreading) in &self.spread_into[b] {
                ecb += eb[bb] * spreading;
                ct += cb[bb] * spreading;
            }
            // Spec gap: with no energy `ct / ecb` is undefined. Taking it as 0
            // makes ln(0) give a fully tonal partition, whose threshold is
            // zero anyway.
            let cbb = if ecb == 0.0 { 0.0 } else { ct / ecb };
            let tb = (CONV1 + CONV2 * cbb.ln()).clamp(0.0, 1.0);
            let snr = f64::max(p.minval, TMN * tb + NMT * (1.0 - tb));
            // Spec reading: figure C.6.b prints the exponent as SNR(b)/10;
            // clause D.2.4 i) has the sign that puts the threshold below the
            // energy. The power of ten, 10^(-SNR(b)/10), is taken as
            // e^(-SNR(b)/10 ln 10), for speed: the same but for rounding.
            nb.push(ecb * p.norm * (-snr / 10.0 * std::f64::consts::LN_10).exp());
        }

        // ...with pre-echo control: the threshold may rise at most by RPELEV
        // over the last granule's and RPELEV2 over the one before, and never
        // falls below the threshold in quiet.
        //
        // Spec reading: this deliberately departs from figure C.6.b, which
        // takes the maximum of all four terms (clause C.1.5.3.2.1 gives only
        // the constants). As printed, the step could only raise the threshold,
        // never lower it, so it could not control pre-echo, its stated
        // purpose. For any steady signal it would also give 16 nb, 12 dB above
        // the SNR that TMN and NMT ask for. With the minimum over `nb` and its
        // history, steady signals keep their threshold, and a granule with an
        // attack after quieter ones gets a lower threshold, keeping the
        // quantisation noise that spreads before the attack small. RPELEV <
        // RPELEV2 also fits limits on growth over one and two granules, rather
        // than a memory in which the older granule weighs more.
        //
        // The perceptual entropy, computed from this threshold, bears it out
        // against the spec's own window switching threshold of 1800 (clause
        // C.1.5.3.2). For decaying noise bursts after silence or over a quiet
        // tone, the minimum puts the entropy of a granule at every attack
        // above 1800, at all three sampling frequencies, with either the
        // natural or the decimal logarithm, and with the energy scale moved by
        // +6, -6 or -20 dB. The printed maximum misses up to all of them.
        // Neither exceeds 1800 on steady tones, noise, a crescendo or a
        // tremolo.
        let limited: Vec<f64> = (0..nb.len())
            .map(|b| {
                nb[b]
                    .min(RPELEV * self.nb_last[b])
                    .min(RPELEV2 * self.nb_before_last[b])
            })
            .collect();
        self.nb_before_last = std::mem::replace(&mut self.nb_last, nb);
        let thr: Vec<f64> = (self.partitions.iter().zip(&limited))
            .map(|(p, limited)| p.qthr.max(*limited))
            .collect();
        // The allowed distortion takes the threshold in quiet as noise has
        // it (see `noise_in_quiet`); the perceptual entropy keeps the printed
        // one, against which the window switching threshold of 1800 (clause
        // C.1.5.3.2) and the entropy of silence above are set.
        let allowed: Vec<f64> = (self.noise_in_quiet.iter().zip(&limited))
            .map(|(quiet, limited)| quiet.max(*limited))
            .collect();

        // Spec gap: clause C.1.5.3.2.1 does not give the base of `log`. The
        // natural logarithm matches the tonality conversion beside it, and the
        // formula is written with the C double literal `1.`, for which `log`
        // would be the natural logarithm, though neither settles it. Against
        // the spec's switching threshold of 1800, the natural logarithm lets
        // the entropy separate attacks from steady signals; with the decimal
        // logarithm, impulses and quiet bursts stay below it. A change of
        // energy scale cannot stand in for the base: it moves the entropy of
        // attacks much more than that of steady signals.
        //
        // Spec reading: the formula is taken as printed, with no floor. A
        // partition whose threshold exceeds its energy plus one subtracts from
        // the sum, so silence and quiet passages, whose upper partitions sit
        // below the large thresholds in quiet there, get a negative entropy:
        // -348 for silence at 44.1 kHz. Flooring each partition at zero would
        // give silence +557 instead, as partitions whose threshold is below
        // one count even with no energy, and that is enough for the bit
        // reservoir to lend silence bits at 64 and 128 kbit/s.
        let perceptual_entropy = -self
            .partitions
            .iter()
            .zip(thr.iter().zip(&eb))
            .map(|(p, (thr, eb))| p.width as f64 * (thr / (eb + 1.0)).ln())
            .sum::<f64>();

        // Figure C.6.c: the partitions converted to scalefactor bands.
        let spectrum: Vec<f64> = energy[..stereo::CRITERION_LINES].to_vec();
        let long = Masking {
            ratio: band_ratio(self.band_weights, &allowed, &eb),
            perceptual_entropy,
            spectrum: spectrum.clone(),
            threshold: thr,
        };
        let short = ShortSpectra {
            energy: energy_short,
            perceptual_entropy,
            spectrum,
        };
        (long, short)
    }

    /// The threshold for short blocks of a granule, from its `spectra`, which
    /// [`Self::analyse`] gives; it depends on nothing else, so it is worked
    /// out only for the granules that need it.
    fn short_masking(&self, spectra: ShortSpectra) -> ShortMasking {
        ShortMasking {
            ratio: spectra.energy.map(|energy| self.short_ratio(&energy)),
            perceptual_entropy: spectra.perceptual_entropy,
            spectrum: spectra.spectrum,
        }
    }

    /// Figure C.6.d: the ratio of each short scalefactor band in one short
    /// block, from the energies of the lines of its FFT.
    ///
    /// Spec reading: the figure says to compare the threshold with the last
    /// one as well as the threshold in quiet, but its formula,
    /// `thr(b) = maximum(qthr(b), nbb(b))`, has no last threshold, and none
    /// of the pre-echo constants. The formula is taken as printed, with
    /// `qthr` taken as noise has it, as for long blocks (see
    /// [`noise_in_quiet`]). So is the exponent `SNR(b)/10`: the short
    /// tables' SNR is negative, so, unlike the long blocks' (figure C.6.b),
    /// it puts the threshold below the spread energy. With a fixed SNR, the unpredictability that the
    /// figure convolves alongside the energy has no use, and is left out.
    fn short_ratio(&self, energy: &[f64]) -> [f64; SHORT_BANDS] {
        let eb: Vec<f64> = self
            .short_partitions
            .iter()
            .map(|p| energy[p.start..p.start + p.width].iter().sum())
            .collect();
        let thr: Vec<f64> = self
            .short_partitions
            .iter()
            .enumerate()
            .map(|(b, _)| {
                let ecb: f64 = (self.short_spread_into[b].iter())
                    .map(|&(bb, spreading)| eb[bb] * spreading)
                    .sum();
                let nbb = ecb * self.short_norm[b] * self.short_snr[b];
                self.short_noise_in_quiet[b].max(nbb)
            })
            .collect();
        band_ratio(self.short_band_weights, &thr, &eb)
    }

    /// The energies of the lines of the granule's three short blocks, and the
    /// unpredictability of the lines of the second (clause C.1.5.3.2.1). The
    /// short blocks are 192 samples apart and centred on the long block, as
    /// the three short windows are in the filterbank's block (clause
    /// C.1.5.3.3), and each is predicted from the two short blocks before
    /// it.
    fn short_blocks(&mut self) -> ([Vec<f64>; SHORT_WINDOWS], Vec<f64>) {
        let mut middle = Vec::new();
        let r = std::array::from_fn(|block| {
            let centre = BLOCK_CENTRE + block * SHORT_SHIFT - SHORT_SHIFT;
            let samples = &self.history[centre - SHORT_FFT / 2..][..SHORT_FFT];
            let (energy, c) = self.short.next_block(samples);
            if block == 1 {
                middle = c;
            }
            energy
        });
        (r, middle)
    }
}

/// The threshold in quiet that each partition allows noise, from the
/// `(qthr, width, bval)` of the partitions in order: the share of a tone's
/// threshold in quiet that falls into the partition, `qthr / width` times
/// its width in Bark. A partition's width in Bark is taken as half the
/// distance between the `bval` of its neighbours, or the distance to its one
/// neighbour at either end.
///
/// Spec reading: this deliberately departs from figures C.6.b and C.6.d for
/// the allowed distortion. Clause D.2.4 l) applies the threshold in quiet to
/// each FFT line, with the values of Table D.4 relative to the level that a
/// sine wave of +-1/2 LSB has in the FFT, and the `qthr` columns of Table
/// C.7, and of Table D.3 of ISO/IEC 13818-3, are such a level per line times
/// the partition's width (see the test
/// `qthr_is_a_per_line_level_times_the_width` in [`partitions`]). Taken as
/// printed, a band of noise at the threshold in every line is allowed as
/// many times a tone's energy at threshold as it has lines. Yet within a
/// critical band, noise is heard at about the energy a tone is, a property
/// of hearing that the standard does not state: over a Bark, the printed
/// threshold allows noise 2.3 times a tone's energy at the lowest
/// frequencies and 81 times at 12 to 14.5 kHz, at 44.1 kHz in the long FFT.
/// Music has little energy above 12 kHz, and from there to 16 kHz the
/// threshold in quiet set the threshold of 25 to 37% of the granules in
/// three tracks of pop music, and of 80 to 84% in two tracks of rock, where
/// it allowed noise 7 to 8 dB above their energy (median); PEAQ found the
/// largest share of chelys's coding noise there. Taken as noise has it,
/// graded with PEAQ (GstPEAQ 0.6.1, black box), 22 tracks of pop music at
/// 44.1 kHz in stereo rose from a mean of -1.79 to -1.70 at 96 kbit/s, from
/// -0.66 to -0.59 at 128, from -0.17 to -0.13 at 192, and from 0.03 to 0.04
/// at 320, and 14 tracks of rock from -0.82 to -0.76 at 128 kbit/s, with the
/// total noise-to-mask ratio 0.6 to 0.7 dB lower at 128 and 192 kbit/s.
/// Lowering every `qthr` by 10 or 20 dB instead graded lower at both.
fn noise_in_quiet(partitions: impl Iterator<Item = (f64, usize, f64)>) -> Vec<f64> {
    let partitions: Vec<(f64, usize, f64)> = partitions.collect();
    let bval = |b: usize| partitions[b].2;
    let last = partitions.len() - 1;
    (partitions.iter().enumerate())
        .map(|(b, &(qthr, width, _))| {
            let bark = match b {
                0 => bval(1) - bval(0),
                _ if b == last => bval(last) - bval(last - 1),
                _ => (bval(b + 1) - bval(b - 1)) / 2.0,
            };
            qthr / width as f64 * bark
        })
        .collect()
}

/// `sprdngf(bval(bb), bval(b))` between all partitions, indexed `[bb][b]`.
/// For each partition `b`, the partitions `bb` with a spread
/// `spreading[bb][b]` other than zero, and their spread, in order.
fn spread_into(spreading: &[Vec<f64>]) -> Vec<Vec<(usize, f64)>> {
    (0..spreading.len())
        .map(|b| {
            (spreading.iter().enumerate())
                .map(|(bb, from)| (bb, from[b]))
                .filter(|&(_, spread)| spread != 0.0)
                .collect()
        })
        .collect()
}

fn spreading_matrix(bval: impl Iterator<Item = f64> + Clone) -> Vec<Vec<f64>> {
    bval.clone()
        .map(|from| {
            bval.clone()
                .map(|into| spreading_function(from, into))
                .collect()
        })
        .collect()
}

/// The ratio `thr / eb` of each scalefactor band, with the partitions
/// converted to scalefactor bands (figure C.6.c).
///
/// Spec gap: in Tables D.4.a, D.4.b and D.4.d to D.4.f of ISO/IEC 13818-3,
/// `bo` of the last band is one past the last partition of the matching
/// table of Table D.3. A partition that does not exist adds nothing; the
/// band collects the partitions after `bu` in full.
fn band_ratio<const N: usize>(bands: &[BandWeights; N], thr: &[f64], eb: &[f64]) -> [f64; N] {
    bands.map(|band| {
        let collect = |values: &[f64]| {
            band.w1 * values[band.bu]
                + values[band.bu + 1..band.bo.min(values.len())]
                    .iter()
                    .sum::<f64>()
                + values.get(band.bo).map_or(0.0, |v| band.w2 * v)
        };
        collect(thr) / collect(eb)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filterbank::Filterbank;

    const FS: SamplingFrequency = SamplingFrequency::Hz44100;

    fn sine(frequency: f64, amplitude: f64, granule: usize) -> [f64; GRANULE_LINES] {
        std::array::from_fn(|t| {
            let n = (granule * GRANULE_LINES + t) as f64;
            amplitude * (2.0 * PI * frequency / FS.hz() as f64 * n).sin()
        })
    }

    /// A noise source that needs no dependencies: a linear congruential
    /// generator mapped to -1.0 to 1.0.
    fn noise(seed: &mut u64, amplitude: f64) -> [f64; GRANULE_LINES] {
        std::array::from_fn(|_| {
            *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            amplitude * ((*seed >> 11) as f64 / (1u64 << 52) as f64 - 1.0)
        })
    }

    fn steady_state(mut next: impl FnMut(usize) -> [f64; GRANULE_LINES]) -> Masking {
        let mut model = PsychoacousticModel::new(FS);
        (0..6).map(|g| model.granule(&next(g))).last().unwrap()
    }

    fn band_of(frequency: f64) -> usize {
        // MDCT line `i` is centred on (i + 0.5) * fs / 1152.
        let line = (frequency / FS.hz() as f64 * 2.0 * GRANULE_LINES as f64) as usize;
        scalefactor_bands::long_bands(FS)
            .iter()
            .position(|b| (b.start..b.end()).contains(&line))
            .unwrap()
    }

    #[test]
    fn spreading_function_peaks_at_zero_distance() {
        assert!((spreading_function(10.0, 10.0) - 1.0).abs() < 0.01);
        // Shallower towards higher frequencies than towards lower ones.
        let up = spreading_function(10.0, 11.0);
        let down = spreading_function(10.0, 9.0);
        assert!(up > 10.0 * down, "{up} {down}");
        assert_eq!(spreading_function(10.0, 0.0), 0.0);
    }

    /// The tone's own band is masked by its energy: a tone needs the full
    /// tone masking noise of 29 dB, so its band's ratio is near or below
    /// -29 dB and far below that of noise, which needs 6 dB.
    #[test]
    fn a_tone_needs_a_higher_snr_than_noise() {
        let frequency = 1000.0;
        let tone = steady_state(|g| sine(frequency, 0.5, g));
        let mut seed = 1;
        let noisy = steady_state(|_| noise(&mut seed, 0.5));
        let sb = band_of(frequency);
        let tone_db = 10.0 * tone.ratio[sb].log10();
        let noise_db = 10.0 * noisy.ratio[sb].log10();
        assert!(tone_db < -25.0, "tone: {tone_db} dB");
        assert!(noise_db > -12.0, "noise: {noise_db} dB");
    }

    /// Pre-echo control: when noise turns 40 dB louder, the threshold may
    /// rise only gradually, so the first loud granules allow far less noise
    /// relative to their energy than the steady state does.
    #[test]
    fn an_attack_lowers_the_allowed_noise() {
        let sb = band_of(2000.0);
        let mut model = PsychoacousticModel::new(FS);
        let mut seed = 3;
        let ratios: Vec<f64> = (0..12)
            .map(|g| {
                let amplitude = if g < 6 { 0.005 } else { 0.5 };
                model.granule(&noise(&mut seed, amplitude)).ratio[sb]
            })
            .collect();
        let db = |r: f64| 10.0 * r.log10();
        let steady = db(ratios[11]);
        let attack = ratios[5..8]
            .iter()
            .copied()
            .map(db)
            .fold(f64::MAX, f64::min);
        assert!(
            attack < steady - 15.0,
            "attack {attack} dB, steady {steady} dB"
        );
        // Quiet and loud steady states are the same noise, so the same ratio.
        assert!((db(ratios[4]) - steady).abs() < 3.0, "{ratios:?}");
    }

    /// Bands far from a loud tone see only the threshold in quiet and the
    /// tone's spread, both far below the tone's own band.
    #[test]
    fn a_tone_masks_its_own_band_most() {
        let frequency = 1000.0;
        let masking = steady_state(|g| sine(frequency, 0.5, g));
        let sb = band_of(frequency);
        // Ratios elsewhere are high: little energy against a floor.
        for other in [0, LONG_BANDS - 1] {
            assert!(masking.ratio[other] > masking.ratio[sb], "band {other}");
        }
    }

    /// With no input the threshold in quiet is all there is, and against the
    /// zero energy the ratio is infinite.
    #[test]
    fn silence_is_below_the_threshold_in_quiet() {
        let masking = steady_state(|_| [0.0; GRANULE_LINES]);
        assert!(masking.ratio.iter().all(|r| r.is_infinite()));
        // thr is qthr and eb is 0, so pe = -sum(width * ln(qthr)). That is not
        // zero, and with the large qthr of the wide top partitions it is
        // negative: the formula has no floor.
        let partitions = partitions::partitions(FS);
        let expected: f64 = -partitions
            .iter()
            .map(|p| p.width as f64 * p.qthr.ln())
            .sum::<f64>();
        assert!((masking.perceptual_entropy - expected).abs() < 1e-9);
        let xmin = masking.allowed_distortion(&[0.0; GRANULE_LINES], FS);
        assert_eq!(xmin, [0.0; LONG_BANDS]);
    }

    /// Noise allowed by the threshold in quiet totals a tone's threshold over
    /// a Bark (see [`noise_in_quiet`]): white noise at the threshold in quiet
    /// in every line around 14 kHz is allowed coding noise only as far below
    /// it as its spread energy asks (9 dB), where the printed `qthr` would
    /// allow coding noise 9 dB louder than the signal. Noise 30 dB below
    /// that level needs no bits.
    #[test]
    fn noise_at_the_threshold_in_quiet_per_line_is_audible() {
        let partitions = partitions::partitions(FS);
        let line = (14_000.0 / FS.hz() as f64 * LONG_FFT as f64) as usize;
        let p = partitions
            .iter()
            .find(|p| (p.start..p.start + p.width).contains(&line))
            .unwrap();
        let per_line = |masking: &Masking| {
            masking.spectrum[p.start..p.start + p.width]
                .iter()
                .sum::<f64>()
                / p.width as f64
        };
        let ratio_db = |below_db: f64| {
            let mut seed = 5;
            let measured = per_line(&steady_state(|_| noise(&mut seed, 0.01)));
            let target = p.qthr / p.width as f64 * 10f64.powf(-below_db / 10.0);
            let amplitude = 0.01 * (target / measured).sqrt();
            let mut seed = 5;
            let masking = steady_state(|_| noise(&mut seed, amplitude));
            10.0 * masking.ratio[band_of(14_500.0)].log10()
        };
        let at_threshold = ratio_db(0.0);
        assert!(at_threshold < -5.0, "{at_threshold} dB");
        let below = ratio_db(30.0);
        assert!(below > 0.0, "{below} dB");
    }

    /// Louder and noisier signals have more perceptual entropy.
    #[test]
    fn noise_has_more_entropy_than_a_tone() {
        let tone = steady_state(|g| sine(1000.0, 0.5, g));
        let mut seed = 7;
        let noisy = steady_state(|_| noise(&mut seed, 0.5));
        let mut seed = 7;
        let quiet = steady_state(|_| noise(&mut seed, 0.005));
        assert!(noisy.perceptual_entropy > tone.perceptual_entropy);
        assert!(noisy.perceptual_entropy > quiet.perceptual_entropy);
    }

    /// [`BLOCK_CENTRE`] is where the filterbank's block is centred: of all
    /// the places a click could be, one there gives the filterbank's output
    /// the most energy.
    #[test]
    fn block_centre_is_where_the_filterbank_responds_most() {
        const GRANULES: usize = 4;
        let energy = |click: usize| {
            let mut filterbank = Filterbank::new();
            let mut xr = [0.0; GRANULE_LINES];
            for g in 0..GRANULES {
                let pcm = std::array::from_fn(|t| {
                    if g * GRANULE_LINES + t == click {
                        1.0
                    } else {
                        0.0
                    }
                });
                xr = filterbank.granule(&pcm, BlockType::Normal, BlockType::Normal);
            }
            xr.iter().map(|x| x * x).sum::<f64>()
        };
        // The history ends with the last granule passed in.
        let centre = (GRANULES * GRANULE_LINES + BLOCK_CENTRE) - HISTORY;
        let loudest = (centre - 400..centre + 400)
            .step_by(4)
            .max_by(|&a, &b| energy(a).total_cmp(&energy(b)))
            .unwrap();
        assert!(loudest.abs_diff(centre) <= 16, "{loudest} vs {centre}");
    }

    /// With a fixed SNR for short blocks, and the normalising constant
    /// undoing the spread, the ratio of every short band of loud noise lies
    /// near the SNR of the partitions it collects, offset by however far the
    /// constant used falls short of undoing the spread (the printed one of
    /// ISO/IEC 13818-3 does by up to about 2 dB).
    #[test]
    fn short_ratio_of_noise_follows_the_snr_column() {
        for fs in SamplingFrequency::ALL {
            let mut model = PsychoacousticModel::new(fs);
            let mut seed = 1;
            let mut short = None;
            for _ in 0..6 {
                let spectra = model.analyse(&noise(&mut seed, 0.5)).1;
                short = Some(model.short_masking(spectra));
            }
            let short = short.unwrap();
            let partitions = partitions::short_partitions(fs);
            let offset = |b: usize| {
                let spread: f64 = model.short_spread_into[b].iter().map(|&(_, s)| s).sum();
                10.0 * (model.short_norm[b] * spread).log10()
            };
            for (w, ratio) in short.ratio.iter().enumerate() {
                for (sb, band) in partitions::short_band_weights(fs).iter().enumerate() {
                    let last = band.bo.min(partitions.len() - 1);
                    let snr = (band.bu..=last).map(|b| partitions[b].snr.db() + offset(b));
                    let (low, high) =
                        snr.fold((f64::MAX, f64::MIN), |(l, h), s| (l.min(s), h.max(s)));
                    let db = 10.0 * ratio[sb].log10();
                    assert!(
                        (low - 1.5..=high + 1.5).contains(&db),
                        "{fs:?} window {w} band {sb}: {db} dB, SNR {low} to {high}"
                    );
                }
            }
        }
    }

    /// The three short windows follow each other in time: a click that only
    /// the last short block sees leaves the first two silent.
    #[test]
    fn short_windows_follow_time() {
        let mut model = PsychoacousticModel::new(FS);
        // In the history of the second granule, the click lands 192 samples
        // after the block centre, the centre of the last short block, and
        // more than 128 samples after the end of the second.
        let click = BLOCK_CENTRE + SHORT_SHIFT - (HISTORY - 2 * GRANULE_LINES);
        let mut pcm = [0.0; GRANULE_LINES];
        pcm[click] = 0.5;
        model.analyse(&pcm);
        let (_, spectra) = model.analyse(&[0.0; GRANULE_LINES]);
        let short = model.short_masking(spectra);
        assert!(short.ratio[0].iter().all(|r| r.is_infinite()));
        assert!(short.ratio[1].iter().all(|r| r.is_infinite()));
        assert!(
            short.ratio[2].iter().all(|r| r.is_finite()),
            "{:?}",
            short.ratio[2]
        );
    }

    /// One granule of steady noise below 3.2 kHz, the same at every sampling
    /// frequency: 2000 sinusoids of random phase about 1.5 Hz apart, too
    /// close for the FFT to resolve at any of the frequencies compared, so
    /// that the model finds it as unpredictable at each.
    fn noise_below_3_2_khz(fs: SamplingFrequency, granule: usize) -> [f64; GRANULE_LINES] {
        let mut seed = 11u64;
        let components: Vec<(f64, f64)> = (0..2000)
            .map(|i| {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let phase = (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 * PI;
                (100.0 + 3100.0 * (i as f64 + 0.5) / 2000.0, phase)
            })
            .collect();
        std::array::from_fn(|t| {
            let time = (granule * GRANULE_LINES + t) as f64 / f64::from(fs.hz());
            (components.iter())
                .map(|(f, phase)| 0.005 * (2.0 * PI * f * time + phase).sin())
                .sum()
        })
    }

    /// How far, in dB, the threshold relative to the signal at `half` lies
    /// from that at `double`, over each whole Bark below 3.2 kHz of
    /// `double`'s partitions and the same frequencies at `half`: the mean and
    /// the largest distance. Each partition's threshold is spread evenly over
    /// its lines, as clause D.2.4 k) spreads it. A Bark at a time, rather than
    /// a partition, so that the leakage of the window, which spreads a tone
    /// over twice as many partitions at the higher rate, evens out.
    fn threshold_distance(
        signal: fn(SamplingFrequency, usize) -> [f64; GRANULE_LINES],
        half: SamplingFrequency,
        double: SamplingFrequency,
    ) -> (f64, f64) {
        let lines = |fs: SamplingFrequency| {
            let mut model = PsychoacousticModel::new(fs);
            let masking = (0..8)
                .map(|g| model.granule(&signal(fs, g)))
                .last()
                .unwrap();
            let mut threshold = vec![0.0; LONG_FFT / 2];
            for (p, thr) in partitions::partitions(fs).iter().zip(&masking.threshold) {
                for line in &mut threshold[p.start..p.start + p.width] {
                    *line = thr / p.width as f64;
                }
            }
            (threshold, masking.spectrum)
        };
        let (half_thr, half_energy) = lines(half);
        let (double_thr, double_energy) = lines(double);
        let cutoff = (3200.0 / f64::from(double.hz()) * LONG_FFT as f64) as usize;
        let partitions = partitions::partitions(double);
        let distances: Vec<f64> = (1..25)
            .filter_map(|bark| {
                let in_bark: Vec<&Partition> = (partitions.iter())
                    .filter(|p| p.bval.floor() as usize == bark && p.start + p.width <= cutoff)
                    .collect();
                let (first, last) = (in_bark.first()?, in_bark.last()?);
                let end = last.start + last.width;
                let ratio = |thr: &[f64], energy: &[f64], range: std::ops::Range<usize>| {
                    thr[range.clone()].iter().sum::<f64>() / energy[range].iter().sum::<f64>()
                };
                let double = ratio(&double_thr, &double_energy, first.start..end);
                let half = ratio(&half_thr, &half_energy, 2 * first.start..2 * end);
                Some((10.0 * (half / double).log10()).abs())
            })
            .collect();
        let mean = distances.iter().sum::<f64>() / distances.len() as f64;
        (mean, distances.iter().copied().fold(0.0, f64::max))
    }

    /// The model hears the same noise at the same frequencies alike at an
    /// MPEG-2.5 sampling frequency and at the frequency twice as high, whose
    /// tables its own are derived from (see [`partitions`]): within 2.1 dB
    /// on average over each Bark from 100 Hz to 3.2 kHz. The tables of
    /// 16 kHz, which MPEG-2.5 used before, put every frequency 1.33 to 2
    /// times too high and come to 2.1 to 2.4 dB; `rnorm` in place of the
    /// printed `norm` to 2.1 to 3.4 dB, and the printed `norm` to 1.1, 1.6
    /// and 2.0 dB at 8, 11.025 and 12 kHz.
    ///
    /// Tones do not show the tables' effect as well: the model's FFT, fixed
    /// in samples, resolves them twice as finely in frequency at the lower
    /// rate and spans twice the time, which changes both their spread over
    /// the partitions and how predictable they are.
    #[test]
    fn mpeg25_threshold_follows_the_rate_twice_as_high() {
        use SamplingFrequency::*;
        for (half, double) in [(Hz8000, Hz16000), (Hz11025, Hz22050), (Hz12000, Hz24000)] {
            let (mean, max) = threshold_distance(noise_below_3_2_khz, half, double);
            assert!(
                mean < 2.1,
                "{half:?}: {mean:.2} dB on average, up to {max:.2} dB"
            );
        }
    }

    /// `xmin` is the band's mean energy per line scaled by its ratio.
    #[test]
    fn allowed_distortion_scales_the_mean_energy_per_line() {
        let masking = Masking {
            ratio: [0.01; LONG_BANDS],
            perceptual_entropy: 0.0,
            spectrum: Vec::new(),
            threshold: Vec::new(),
        };
        let mut xr = [0.0; GRANULE_LINES];
        let band = scalefactor_bands::long_bands(FS)[5];
        xr[band.start] = 3.0;
        xr[band.start + 1] = -4.0;
        let xmin = masking.allowed_distortion(&xr, FS);
        assert!((xmin[5] - 0.01 * 25.0 / band.width as f64).abs() < 1e-12);
        assert_eq!(xmin[4], 0.0);
    }
}
