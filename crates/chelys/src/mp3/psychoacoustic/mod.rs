//! Psychoacoustic model 2 (clause D.2) as adapted for Layer III in clause
//! C.1.5.3.2.1, figures C.6.a to C.6.c. Long blocks only: the model computes
//! the threshold for long blocks and leaves out the switch to short blocks.
//!
//! For each granule the model gives the ratio of allowed noise to signal
//! energy in every scalefactor band, from which the allowed distortion
//! `xmin` of the outer iteration loop follows, and the perceptual entropy.

mod fft;
pub mod partitions;

use std::f64::consts::PI;

use super::GRANULE_LINES;
use super::header::SamplingFrequency;
use super::scalefactor_bands::{self, LONG_BANDS};
use fft::{Complex, Fft};
use partitions::{BandWeights, Partition};

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
/// The standard does not say what scale `qthr` is on. Clause D.2.4 l) puts
/// the absolute thresholds of Table D.4 relative to a sine of +-1/2 LSB and
/// asks for them to be converted "after considering the FFT normalization
/// actually used", and Table D.4 sets 0 dB at 96 dB below the energy of a
/// sine of amplitude 32 760. With 16-bit samples and a `1 / N` FFT, such a
/// sine peaks at an energy of `(32 760 / 4)^2` in the Hann-windowed spectrum,
/// putting 0 dB at about 0.017. Table C.7's lowest per-line threshold, about
/// 0.009, then lies near the bottom of Table D.4 (-5 dB), as a threshold in
/// quiet should. An unnormalised FFT would put it 60 dB lower.
const PCM_SCALE: f64 = 32_768.0;

/// The samples the model keeps: the current granule and the two before it.
const HISTORY: usize = 3 * GRANULE_LINES;

/// Where in the history the filterbank's current block is centred, so that
/// the analysis is centred on it as clause D.2.1 b) asks.
///
/// The filterbank returns the block formed by the subband samples of the
/// previous granule and the current one. Subband sample `t` of a granule is
/// computed with input sample `32 t + 31` of that granule newest, and the
/// analysis window (Table C.1) is symmetric about the 256th sample back, so
/// it is centred on sample `32 t - 225`. The block runs from sample 0 of the
/// previous granule, centred on sample `-576 - 225 = -801` of the current
/// one, to sample 17 of the current granule, centred on `32 * 17 - 225 =
/// 319`, so its middle is sample -241. The Hann windows are centred half a
/// sample earlier, which does not matter at this resolution.
const BLOCK_CENTRE: usize = 2 * GRANULE_LINES - 241;

/// The spreading function of clause D.2.3 as changed for Layer III in clause
/// C.1.5.3.2.1: the spread from Bark value `i` into Bark value `j`.
///
/// Clause C.1.5.3.2.1 prints the change as "If j >= i tmpy = 3,0 (j-i), else
/// tmpy = 1,5 (j-i)". Read literally, as a replacement of `tmpy`, the spread
/// would grow without bound towards high frequencies, so it is read as a
/// change to the scaling of `tmpx`, which `tmpx = 1,05 (j-i)` sets in clause
/// D.2.3. The two factors are also swapped here: the `norm` column of Table
/// C.7 matches the reciprocal of the spread summed over the partitions, as
/// in clause D.2.4 f), to within the rounding of the table only with a
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
    /// Infinite for a band in which the spectrum has no energy.
    pub ratio: [f64; LONG_BANDS],
    /// The perceptual entropy `pe` (clause C.1.5.3.2.1).
    pub perceptual_entropy: f64,
}

impl Masking {
    /// The allowed distortion of each scalefactor band,
    /// `xmin(sb) = ratio(sb) * en(sb) / bw(sb)` (clause C.1.5.4.1), where `en(sb)` is the energy of
    /// the band's lines in `xr` and `bw(sb)` their number. This is a mean
    /// energy per line, as is the distortion `xfsf` it is compared with
    /// (clause C.1.5.4.3.3). A band whose lines are all zero allows none.
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

/// The magnitude `r` and phase `f` of one FFT line (clause D.2.4 b)).
#[derive(Clone, Copy, Debug, Default)]
struct Polar {
    r: f64,
    f: f64,
}

/// One FFT of the model with its Hann window, and the spectra of the two
/// blocks before, from which it predicts the next (clause D.2.4 c)).
struct Analysis {
    fft: Fft,
    window: Vec<f64>,
    last: Vec<Polar>,
    before_last: Vec<Polar>,
}

impl Analysis {
    fn new(size: usize) -> Self {
        // Clause D.2.4 b): sw(i) = s(i) * (0.5 - 0.5 cos(2 pi (i - 0.5) / N))
        // for i = 1..N.
        let window = (0..size)
            .map(|n| 0.5 - 0.5 * (2.0 * PI * (n as f64 + 0.5) / size as f64).cos())
            .collect();
        // Clause D.2.1 asks for the stored spectra to start at zero.
        let lines = size / 2 + 1;
        Self {
            fft: Fft::new(size),
            window,
            last: vec![Polar::default(); lines],
            before_last: vec![Polar::default(); lines],
        }
    }

    /// Transforms the windowed `samples` and returns the magnitudes and
    /// unpredictabilities `c` of the lines from DC to Nyquist (clause D.2.4
    /// b) to d)), remembering the spectrum for the next block.
    fn next_block(&mut self, samples: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let size = self.window.len();
        let mut x: Vec<Complex> = samples
            .iter()
            .zip(&self.window)
            .map(|(s, w)| Complex {
                re: s * PCM_SCALE * w / size as f64,
                im: 0.0,
            })
            .collect();
        self.fft.transform(&mut x);

        let lines = size / 2 + 1;
        let mut r = Vec::with_capacity(lines);
        let mut c = Vec::with_capacity(lines);
        let mut current = Vec::with_capacity(lines);
        for (w, x) in x[..lines].iter().enumerate() {
            let polar = Polar {
                r: x.re.hypot(x.im),
                f: x.im.atan2(x.re),
            };
            let (last, before_last) = (self.last[w], self.before_last[w]);
            let predicted_r = 2.0 * last.r - before_last.r;
            let predicted_f = 2.0 * last.f - before_last.f;
            let distance = (x.re - predicted_r * predicted_f.cos())
                .hypot(x.im - predicted_r * predicted_f.sin());
            let scale = polar.r + predicted_r.abs();
            // Silence that was predicted to stay silent: the line has no
            // energy, so its unpredictability carries no weight.
            c.push(if scale == 0.0 { 0.0 } else { distance / scale });
            r.push(polar.r);
            current.push(polar);
        }
        self.before_last = std::mem::replace(&mut self.last, current);
        (r, c)
    }
}

/// Psychoacoustic model 2 for one channel, fed the same PCM granules as the
/// channel's filterbank.
pub struct PsychoacousticModel {
    partitions: &'static [Partition],
    band_weights: &'static [BandWeights; LONG_BANDS],
    /// `sprdngf(bval(bb), bval(b))`, indexed `[bb][b]`.
    spreading: Vec<Vec<f64>>,
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
        let spreading = partitions
            .iter()
            .map(|from| {
                partitions
                    .iter()
                    .map(|into| spreading_function(from.bval, into.bval))
                    .collect()
            })
            .collect();
        Self {
            partitions,
            band_weights: partitions::band_weights(sampling_frequency),
            spreading,
            history: [0.0; HISTORY],
            long: Analysis::new(LONG_FFT),
            short: Analysis::new(SHORT_FFT),
            nb_last: vec![0.0; partitions.len()],
            nb_before_last: vec![0.0; partitions.len()],
        }
    }

    /// Analyses the next granule of PCM, nominally in -1.0 to 1.0, centred on
    /// the block the filterbank returns for the same granule.
    pub fn granule(&mut self, pcm: &[f64; GRANULE_LINES]) -> Masking {
        self.history.copy_within(GRANULE_LINES.., 0);
        self.history[HISTORY - GRANULE_LINES..].copy_from_slice(pcm);

        let cw_short = self.unpredictability();
        let long_start = BLOCK_CENTRE - LONG_FFT / 2;
        let (r, cw_long) = self
            .long
            .next_block(&self.history[long_start..][..LONG_FFT]);
        let cw = |w: usize| match w {
            ..LONG_CW_LINES => cw_long[w],
            LONG_CW_LINES..SHORT_CW_LINES => cw_short[(w + 2) / 4],
            _ => HIGH_CW,
        };

        // Figure C.6.b: energy and unpredictability in each partition...
        let (eb, cb): (Vec<f64>, Vec<f64>) = self
            .partitions
            .iter()
            .map(|p| {
                (p.start..p.start + p.width)
                    .map(|w| (r[w] * r[w], cw(w) * r[w] * r[w]))
                    .fold((0.0, 0.0), |(e, c), (de, dc)| (e + de, c + dc))
            })
            .unzip();

        // ...convolved with the spreading function, to give the tonality,
        // the required SNR and the threshold...
        let mut nb = Vec::with_capacity(self.partitions.len());
        for (b, p) in self.partitions.iter().enumerate() {
            let (mut ecb, mut ct) = (0.0, 0.0);
            for (bb, spreading) in self.spreading.iter().enumerate() {
                ecb += eb[bb] * spreading[b];
                ct += cb[bb] * spreading[b];
            }
            // With no energy, ln(0) makes the partition fully tonal.
            let cbb = if ecb == 0.0 { 0.0 } else { ct / ecb };
            let tb = (CONV1 + CONV2 * cbb.ln()).clamp(0.0, 1.0);
            let snr = f64::max(p.minval, TMN * tb + NMT * (1.0 - tb));
            // Figure C.6.b prints the exponent as SNR(b)/10; clause D.2.4 i)
            // has the sign that puts the threshold below the energy.
            nb.push(ecb * p.norm * 10f64.powf(-snr / 10.0));
        }

        // ...with pre-echo control: the threshold may rise at most by RPELEV
        // over the last granule's and RPELEV2 over the one before, and never
        // falls below the threshold in quiet.
        //
        // This deliberately departs from figure C.6.b, which takes the maximum
        // of all four terms (clause C.1.5.3.2.1 gives only the constants). As
        // printed, the step could only raise the threshold, never lower it, so
        // it could not control pre-echo, its stated purpose. For any steady
        // signal it would also give 16 nb, 12 dB above the SNR that TMN and
        // NMT ask for. With the minimum over `nb` and its history, steady
        // signals keep their threshold, and a granule with an attack after
        // quieter ones gets a lower threshold, keeping the quantisation noise
        // that spreads before the attack small. RPELEV < RPELEV2 also fits
        // limits on growth over one and two granules, rather than a memory in
        // which the older granule weighs more.
        let thr: Vec<f64> = self
            .partitions
            .iter()
            .enumerate()
            .map(|(b, p)| {
                let limited = nb[b]
                    .min(RPELEV * self.nb_last[b])
                    .min(RPELEV2 * self.nb_before_last[b]);
                p.qthr.max(limited)
            })
            .collect();
        self.nb_before_last = std::mem::replace(&mut self.nb_last, nb);

        // Clause C.1.5.3.2.1 does not give the base of `log`. The natural
        // logarithm matches the tonality conversion beside it, and the formula
        // is written with the C double literal `1.`, for which `log` would be
        // the natural logarithm, though neither settles it.
        let perceptual_entropy = -self
            .partitions
            .iter()
            .zip(thr.iter().zip(&eb))
            .map(|(p, (thr, eb))| p.width as f64 * (thr / (eb + 1.0)).ln())
            .sum::<f64>();

        // Figure C.6.c: the partitions converted to scalefactor bands.
        let ratio = self.band_weights.map(|band| {
            let collect = |values: &[f64]| {
                band.w1 * values[band.bu]
                    + values[band.bu + 1..band.bo].iter().sum::<f64>()
                    + band.w2 * values[band.bo]
            };
            collect(&thr) / collect(&eb)
        });
        Masking {
            ratio,
            perceptual_entropy,
        }
    }

    /// The unpredictability of the short FFT's lines in the second of the
    /// granule's three short blocks (clause C.1.5.3.2.1). The short blocks
    /// are 192 samples apart and centred on the long block, and each is
    /// predicted from the two short blocks before it.
    fn unpredictability(&mut self) -> Vec<f64> {
        let mut middle = Vec::new();
        for block in 0..3 {
            let centre = BLOCK_CENTRE + block * SHORT_SHIFT - SHORT_SHIFT;
            let samples = &self.history[centre - SHORT_FFT / 2..][..SHORT_FFT];
            let (_, c) = self.short.next_block(samples);
            if block == 1 {
                middle = c;
            }
        }
        middle
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mp3::filterbank::Filterbank;

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
                xr = filterbank.granule(&pcm);
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

    /// `xmin` is the band's mean energy per line scaled by its ratio.
    #[test]
    fn allowed_distortion_scales_the_mean_energy_per_line() {
        let masking = Masking {
            ratio: [0.01; LONG_BANDS],
            perceptual_entropy: 0.0,
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
