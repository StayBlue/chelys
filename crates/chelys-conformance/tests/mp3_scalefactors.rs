//! Scalefactors decoded by ffmpeg.
//!
//! The decoder divides a band by `2^(scalefac_multiplier * (scalefac_l +
//! preflag * pretab))` (clause 2.4.3.4.7.1). Since `8^(4/3) = 2^4`, a band
//! whose values are multiplied by 8 and whose scalefactors divide it by 2^4
//! decodes to the same output as the band as it was, and an amplification
//! of all bands by `2^(g/4)` is undone by raising `global_gain` by `g`. Each
//! check encodes one spectrum several such ways and requires the same
//! decoded output. A
//! scalefactor sent with the wrong width, in the wrong band, or with the
//! wrong multiplier or preemphasis decodes at least a factor of 2^(1/2) off.

use chelys::mp3::header::{Bitrate, Mode, SamplingFrequency};
use chelys::mp3::scalefactor_bands::{self, LONG_BANDS};
use chelys::mp3::scalefactors::{PRETAB, Scalefactors};
use chelys::mp3::{GRANULE_LINES, QuantizedGranule, StreamConfig, encode_granules};
use chelys_conformance::ffmpeg;

const GLOBAL_GAIN: u8 = 150;

fn config(sampling_frequency: SamplingFrequency) -> StreamConfig {
    StreamConfig {
        bitrate: Bitrate::from_kbps(320).unwrap(),
        sampling_frequency,
        mode: Mode::SingleChannel,
        crc_protected: false,
    }
}

/// A deterministic sequence (xorshift32).
struct Rng(u32);

impl Rng {
    fn next(&mut self, below: u32) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0 % below
    }
}

/// Values of -1, 0 or 1 in every line of bands 0 to 20; band 21, which has
/// no scalefactor, stays empty.
fn spectrum(rng: &mut Rng, sampling_frequency: SamplingFrequency) -> [i32; GRANULE_LINES] {
    let end = scalefactor_bands::long_bands(sampling_frequency)[LONG_BANDS - 1].end();
    std::array::from_fn(|i| if i < end { rng.next(3) as i32 - 1 } else { 0 })
}

/// `values` with each band whose amplification is 2^4 beyond
/// `2^(gain_steps / 4)` multiplied by 8.
fn compensate(
    values: &[i32; GRANULE_LINES],
    scalefactors: &Scalefactors,
    gain_steps: u8,
    sampling_frequency: SamplingFrequency,
) -> [i32; GRANULE_LINES] {
    let mut values = *values;
    let common = (f64::from(gain_steps) / 4.0).exp2();
    for (sb, band) in scalefactor_bands::long_bands(sampling_frequency)
        .iter()
        .enumerate()
    {
        let amplification = scalefactors.amplification(sb) / common;
        assert!(amplification == 1.0 || amplification == 16.0, "band {sb}");
        if amplification == 16.0 {
            for v in &mut values[band.start..band.end()] {
                *v *= 8;
            }
        }
    }
    values
}

/// Encodes three frames of the two spectra with `scalefactors`, compensated
/// by the values and by `gain_steps` more `global_gain`, and decodes them.
fn decode(
    sampling_frequency: SamplingFrequency,
    spectra: &[[i32; GRANULE_LINES]; 2],
    scalefactors: &Scalefactors,
    gain_steps: u8,
) -> Vec<f32> {
    let granules: Vec<_> = spectra
        .iter()
        .map(|values| {
            let values = compensate(values, scalefactors, gain_steps, sampling_frequency);
            let global_gain = GLOBAL_GAIN + gain_steps;
            let mut granule = QuantizedGranule::new(global_gain, values, sampling_frequency);
            granule.scalefactors = scalefactors.clone();
            vec![granule]
        })
        .cycle()
        .take(6)
        .collect();
    let decoded = ffmpeg::decode_mp3(&encode_granules(&config(sampling_frequency), &granules));
    assert_eq!(decoded.diagnostics, "");
    assert!(decoded.samples.iter().any(|&s| s != 0.0), "decoded silence");
    decoded.samples
}

fn assert_close(label: &str, samples: &[f32], expected: &[f32]) {
    assert_eq!(samples.len(), expected.len(), "{label}");
    let peak = expected.iter().fold(0f32, |m, s| m.max(s.abs()));
    let worst = samples
        .iter()
        .zip(expected)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        worst <= 1e-4 * peak,
        "{label}: off by {worst} against a peak of {peak}"
    );
}

#[test]
fn scalefactors_divide_their_bands() {
    for fs in SamplingFrequency::ALL {
        let mut rng = Rng(0x5CA1_EFAC);
        let spectra = [spectrum(&mut rng, fs), spectrum(&mut rng, fs)];
        let expected = decode(fs, &spectra, &Scalefactors::default(), 0);

        let mut variants = Vec::new();
        // scalefac_scale 0: 8 steps of 2^(1/2) in random bands of 0 to 10,
        // which slen1 can hold.
        for _ in 0..3 {
            let mut scalefactors = Scalefactors::default();
            for scalefac in &mut scalefactors.scalefac_l[..11] {
                *scalefac = 8 * rng.next(2) as u8;
            }
            variants.push(("scalefac_scale 0", scalefactors, 0));
        }
        // scalefac_scale 1: 4 steps of 2 in random bands of 0 to 20.
        for _ in 0..3 {
            let mut scalefactors = Scalefactors {
                scalefac_scale: true,
                ..Scalefactors::default()
            };
            for scalefac in &mut scalefactors.scalefac_l {
                *scalefac = 4 * rng.next(2) as u8;
            }
            variants.push(("scalefac_scale 1", scalefactors, 0));
        }
        // Preemphasis: scalefactors that make up the rest of 4 steps of 2
        // over pretab in every band.
        let mut scalefactors = Scalefactors {
            scalefac_scale: true,
            preflag: true,
            ..Scalefactors::default()
        };
        for (scalefac, pretab) in scalefactors.scalefac_l.iter_mut().zip(PRETAB) {
            *scalefac = 4 - pretab;
        }
        variants.push(("preflag", scalefactors, 0));
        // The same scalefactor in every band, undone by global_gain: 1, 2 and
        // 3 steps for the narrower field widths.
        for (scalefac, scalefac_scale) in [(1, true), (2, false), (3, false), (2, true)] {
            let scalefactors = Scalefactors {
                scalefac_l: [scalefac; LONG_BANDS],
                scalefac_scale,
                preflag: false,
            };
            let gain_steps = if scalefac_scale { 4 } else { 2 } * scalefac;
            variants.push(("uniform", scalefactors, gain_steps));
        }
        for (label, scalefactors, gain_steps) in &variants {
            let samples = decode(fs, &spectra, scalefactors, *gain_steps);
            assert_close(
                &format!("{fs:?} {label} {scalefactors:?}"),
                &samples,
                &expected,
            );
        }
    }
}
