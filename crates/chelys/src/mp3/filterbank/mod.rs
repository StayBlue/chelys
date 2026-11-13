//! The analysis half of the Layer III hybrid filterbank: the polyphase
//! subband filter (clause C.1.3, figure C.4) followed by the MDCT and the
//! encoder's aliasing butterflies (clause C.1.5.3.3). Long blocks only.

mod analysis_window;

use std::f64::consts::PI;

use crate::mp3::GRANULE_LINES;
use analysis_window::ANALYSIS_WINDOW;

const SUBBANDS: usize = 32;
/// Subband samples per granule, and MDCT lines per subband.
const SUBBAND_SAMPLES: usize = GRANULE_LINES / SUBBANDS;
/// Samples in a long MDCT block: the subband samples of two granules.
const LONG_BLOCK: usize = 2 * SUBBAND_SAMPLES;

/// The decoder's long-block IMDCT (clause 2.4.3.4.10.2) with windowing and
/// overlap-add returns the encoder's MDCT input (clause C.1.5.3.3) scaled by
/// `n / 4 = 9`, since neither transform is normalised, so the MDCT output is
/// divided by 9. Clause C.1.5.4.1 leaves this scaling to the implementation
/// as part of its "system constant". With the polyphase analysis and
/// synthesis filters also at unity gain, which the standard does not state
/// but the decoded round trip confirms, the result is on the scale of the
/// decoder's requantised values `xr` (clause 2.4.3.4.7.1).
const LONG_BLOCK_GAIN: f64 = LONG_BLOCK as f64 / 4.0;

/// Table B.9: the coefficients `c_i` of the aliasing butterflies.
const ALIAS_COEFFICIENTS: [f64; 8] = [
    -0.6, -0.535, -0.33, -0.185, -0.095, -0.041, -0.0142, -0.0037,
];

/// Filters one channel of PCM into granules of 576 frequency lines.
pub struct Filterbank {
    /// The input sample vector `X` of figure C.4, most recent sample first.
    history: [f64; 512],
    /// `M_ik` of clause C.1.3, indexed `[i][k]`.
    matrix: Box<[[f64; 64]; SUBBANDS]>,
    /// The MDCT kernel of clause C.1.5.3.3 for `n = 36`, indexed `[i][k]`,
    /// with the normal window `sin(pi / 36 * (k + 1/2))` folded in.
    mdct: Box<[[f64; LONG_BLOCK]; SUBBAND_SAMPLES]>,
    /// The previous granule's subband samples, indexed `[subband][time]`.
    previous: [[f64; SUBBAND_SAMPLES]; SUBBANDS],
    /// `cs_i` and `ca_i` of Table B.9.
    butterflies: [(f64, f64); 8],
}

impl Default for Filterbank {
    fn default() -> Self {
        Self::new()
    }
}

impl Filterbank {
    pub fn new() -> Self {
        let mut matrix = Box::new([[0.0; 64]; SUBBANDS]);
        for (i, row) in matrix.iter_mut().enumerate() {
            for (k, m) in row.iter_mut().enumerate() {
                *m = ((2 * i + 1) as f64 * (k as f64 - 16.0) * PI / 64.0).cos();
            }
        }
        let n = LONG_BLOCK as f64;
        let mut mdct = Box::new([[0.0; LONG_BLOCK]; SUBBAND_SAMPLES]);
        for (i, row) in mdct.iter_mut().enumerate() {
            for (k, m) in row.iter_mut().enumerate() {
                let window = (PI / 36.0 * (k as f64 + 0.5)).sin();
                let kernel =
                    (PI / (2.0 * n) * (2.0 * k as f64 + 1.0 + n / 2.0) * (2.0 * i as f64 + 1.0))
                        .cos();
                *m = window * kernel;
            }
        }
        let butterflies = ALIAS_COEFFICIENTS.map(|c| {
            let norm = (1.0 + c * c).sqrt();
            (1.0 / norm, c / norm)
        });
        Self {
            history: [0.0; 512],
            matrix,
            mdct,
            previous: [[0.0; SUBBAND_SAMPLES]; SUBBANDS],
            butterflies,
        }
    }

    /// Filters the next 576 PCM samples, nominally in -1.0 to 1.0, into the
    /// granule's frequency lines `xr`, line `18 * subband + k` holding MDCT
    /// output `k` of `subband`. The lines describe the block formed with the
    /// previous granule, so they lag the input by one granule.
    pub fn granule(&mut self, pcm: &[f64; GRANULE_LINES]) -> [f64; GRANULE_LINES] {
        let mut current = [[0.0; SUBBAND_SAMPLES]; SUBBANDS];
        for (time, samples) in pcm.chunks_exact(SUBBANDS).enumerate() {
            let subbands = self.subband_samples(samples.try_into().unwrap());
            for (subband, &sample) in subbands.iter().enumerate() {
                // The decoder negates every odd time sample of every odd
                // subband to compensate for the frequency inversion of the
                // polyphase filterbank (clause 2.4.3.4.10.5). Annex C does not
                // mention it, but the encoder has to apply the same inversion
                // for the decoder to restore the input.
                let invert = subband % 2 == 1 && time % 2 == 1;
                current[subband][time] = if invert { -sample } else { sample };
            }
        }

        let mut xr = [0.0; GRANULE_LINES];
        for subband in 0..SUBBANDS {
            let block = self.previous[subband].iter().chain(&current[subband]);
            let lines = &mut xr[SUBBAND_SAMPLES * subband..][..SUBBAND_SAMPLES];
            for (line, kernel) in lines.iter_mut().zip(self.mdct.iter()) {
                *line = kernel
                    .iter()
                    .zip(block.clone())
                    .map(|(m, x)| m * x)
                    .sum::<f64>()
                    / LONG_BLOCK_GAIN;
            }
        }
        self.previous = current;
        self.reduce_aliasing(&mut xr);
        xr
    }

    /// One pass of figure C.4: shifts in 32 samples, oldest first, and returns
    /// the 32 subband samples.
    fn subband_samples(&mut self, samples: &[f64; SUBBANDS]) -> [f64; SUBBANDS] {
        self.history.copy_within(0..512 - SUBBANDS, SUBBANDS);
        for (i, &sample) in samples.iter().enumerate() {
            // "for i=31 down to 0 do X[i]=next_input_audio_sample"
            self.history[SUBBANDS - 1 - i] = sample;
        }
        let mut y = [0.0; 64];
        for (i, y) in y.iter_mut().enumerate() {
            *y = (0..8)
                .map(|j| ANALYSIS_WINDOW[i + 64 * j] * self.history[i + 64 * j])
                .sum();
        }
        self.matrix
            .map(|row| row.iter().zip(&y).map(|(m, y)| m * y).sum())
    }

    /// The encoder's aliasing butterflies (figure C.8) between each pair of
    /// adjacent subbands, applied to the eight lines on either side of the
    /// boundary as in figure A.5.
    fn reduce_aliasing(&self, xr: &mut [f64; GRANULE_LINES]) {
        for subband in 1..SUBBANDS {
            let boundary = SUBBAND_SAMPLES * subband;
            for (i, &(cs, ca)) in self.butterflies.iter().enumerate() {
                let (lower, upper) = (boundary - 1 - i, boundary + i);
                let (a, b) = (xr[lower], xr[upper]);
                xr[lower] = a * cs + b * ca;
                xr[upper] = b * cs - a * ca;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The printed coefficients are all multiples of 2^-21 rounded to nine
    /// decimals. A misread digit almost always breaks this.
    #[test]
    fn analysis_window_lies_on_a_2_pow_21_grid() {
        for (i, c) in ANALYSIS_WINDOW.iter().enumerate() {
            let scaled = c * f64::from(1 << 21);
            assert!((scaled - scaled.round()).abs() < 0.0011, "C[{i}] = {c}");
        }
    }

    /// The two printed halves mirror each other: `C[512 - i] = -C[i]`, except
    /// at multiples of 64, where the two are equal. Checking one half against
    /// the other catches misread digits and signs.
    #[test]
    fn analysis_window_halves_mirror_each_other() {
        for i in 1..256 {
            let (a, b) = (ANALYSIS_WINDOW[i], ANALYSIS_WINDOW[512 - i]);
            let expected = if i % 64 == 0 { a } else { -a };
            assert_eq!(b, expected, "C[{i}] and C[{}]", 512 - i);
        }
    }

    /// The decoder's long-block IMDCT, windowing and overlap-add (clauses
    /// 2.4.3.4.10.2 to 2.4.3.4.10.4) for one subband, written out from the
    /// formulas to check the encoder's MDCT against.
    fn decoder_imdct(
        lines: &[f64; SUBBAND_SAMPLES],
        overlap: &mut [f64; SUBBAND_SAMPLES],
    ) -> [f64; SUBBAND_SAMPLES] {
        let n = LONG_BLOCK as f64;
        let mut z = [0.0; LONG_BLOCK];
        for (i, z) in z.iter_mut().enumerate() {
            let x: f64 = lines
                .iter()
                .enumerate()
                .map(|(k, &x)| {
                    x * (PI / (2.0 * n) * (2.0 * i as f64 + 1.0 + n / 2.0) * (2.0 * k as f64 + 1.0))
                        .cos()
                })
                .sum();
            *z = x * (PI / 36.0 * (i as f64 + 0.5)).sin();
        }
        let mut out = [0.0; SUBBAND_SAMPLES];
        for i in 0..SUBBAND_SAMPLES {
            out[i] = z[i] + overlap[i];
            overlap[i] = z[i + SUBBAND_SAMPLES];
        }
        out
    }

    #[test]
    fn mdct_inverts_through_the_decoder_imdct() {
        let filterbank = Filterbank::new();
        let signal: Vec<f64> = (0..SUBBAND_SAMPLES * 6)
            .map(|t| ((t * 7919) % 101) as f64 / 50.0 - 1.0)
            .collect();
        let mut overlap = [0.0; SUBBAND_SAMPLES];
        let mut previous = [0.0; SUBBAND_SAMPLES];
        for (g, current) in signal.chunks_exact(SUBBAND_SAMPLES).enumerate() {
            let block = previous.iter().chain(current);
            let mut lines = [0.0; SUBBAND_SAMPLES];
            for (line, kernel) in lines.iter_mut().zip(filterbank.mdct.iter()) {
                *line = kernel
                    .iter()
                    .zip(block.clone())
                    .map(|(m, x)| m * x)
                    .sum::<f64>()
                    / LONG_BLOCK_GAIN;
            }
            let out = decoder_imdct(&lines, &mut overlap);
            // Output g restores input g - 1, the first half of block g.
            if g > 0 {
                let expected = &signal[SUBBAND_SAMPLES * (g - 1)..][..SUBBAND_SAMPLES];
                for (o, e) in out.iter().zip(expected) {
                    assert!((o - e).abs() < 1e-12, "granule {g}: {o} != {e}");
                }
            }
            previous.copy_from_slice(current);
        }
    }

    /// The decoder's butterflies (clause 2.4.3.4.10.1) undo the encoder's.
    #[test]
    fn decoder_alias_reduction_undoes_the_encoder_butterflies() {
        let filterbank = Filterbank::new();
        let original: [f64; GRANULE_LINES] = std::array::from_fn(|i| ((i * 31) % 17) as f64 - 8.0);
        let mut xr = original;
        filterbank.reduce_aliasing(&mut xr);
        assert_ne!(xr, original);
        for sb in 1..SUBBANDS {
            for (i, &(cs, ca)) in filterbank.butterflies.iter().enumerate() {
                let (lower, upper) = (18 * sb - 1 - i, 18 * sb + i);
                let (a, b) = (xr[lower], xr[upper]);
                xr[lower] = a * cs - b * ca;
                xr[upper] = b * cs + a * ca;
            }
        }
        for (a, b) in xr.iter().zip(&original) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    /// A sinusoid at the centre of subband `sb`, `(2 sb + 1) * fs / 128`,
    /// lands in that subband's lines.
    #[test]
    fn a_tone_lands_in_its_subband() {
        for subband in [0, 5, 6, 31] {
            let mut filterbank = Filterbank::new();
            let frequency = (2 * subband + 1) as f64 / 128.0;
            let mut energy = [0.0; SUBBANDS];
            for g in 0..6 {
                let pcm = std::array::from_fn(|t| {
                    (2.0 * PI * frequency * (GRANULE_LINES * g + t) as f64).sin() * 0.5
                });
                let xr = filterbank.granule(&pcm);
                if g >= 3 {
                    for (sb, lines) in xr.chunks_exact(SUBBAND_SAMPLES).enumerate() {
                        energy[sb] += lines.iter().map(|x| x * x).sum::<f64>();
                    }
                }
            }
            let total: f64 = energy.iter().sum();
            assert!(
                energy[subband] > 0.99 * total,
                "subband {subband}: {energy:?}"
            );
        }
    }
}
