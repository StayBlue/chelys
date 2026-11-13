// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

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

use chelys::mp3::header::{Bitrate, Mode, SamplingFrequency, Version};
use chelys::mp3::scalefactor_bands::{self, LONG_BANDS};
use chelys::mp3::scalefactors::{PRETAB, Scalefac, Scalefactors};
use chelys::mp3::{BlockType, GRANULE_LINES, QuantizedGranule, StreamConfig, encode_granules};
use chelys_conformance::ffmpeg;

const GLOBAL_GAIN: u8 = 150;

fn config(sampling_frequency: SamplingFrequency) -> StreamConfig {
    StreamConfig {
        // The highest: 320 kbit/s, or 160 at the lower sampling frequencies.
        bitrate: Bitrate::all(sampling_frequency.version()).last().unwrap(),
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

/// Values of -1, 0 or 1 in every line that has a scalefactor in a granule of
/// `block_type`, in the order they are sent; the lines above, which have
/// none, stay empty.
fn spectrum(
    rng: &mut Rng,
    block_type: BlockType,
    sampling_frequency: SamplingFrequency,
) -> [i32; GRANULE_LINES] {
    let bands = scalefactor_bands::coding_bands(block_type, sampling_frequency);
    let end = bands.iter().map(|band| band.end()).max().unwrap();
    std::array::from_fn(|i| if i < end { rng.next(3) as i32 - 1 } else { 0 })
}

/// `values` with each band whose amplification is 2^4 beyond
/// `2^(gain_steps / 4)` multiplied by 8.
fn compensate(
    values: &[i32; GRANULE_LINES],
    scalefactors: &Scalefactors,
    block_type: BlockType,
    gain_steps: u8,
    sampling_frequency: SamplingFrequency,
) -> [i32; GRANULE_LINES] {
    let mut values = *values;
    let common = (f64::from(gain_steps) / 4.0).exp2();
    for (sb, band) in scalefactor_bands::coding_bands(block_type, sampling_frequency)
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
    decode_with_scfsi(
        sampling_frequency,
        spectra,
        scalefactors,
        gain_steps,
        [false; 4],
    )
}

/// [`decode`], with the second granule of each frame taking over the first
/// granule's scalefactors in the scfsi bands of `scfsi`.
fn decode_with_scfsi(
    sampling_frequency: SamplingFrequency,
    spectra: &[[i32; GRANULE_LINES]; 2],
    scalefactors: &Scalefactors,
    gain_steps: u8,
    scfsi: [bool; 4],
) -> Vec<f32> {
    let block_type = match scalefactors.scalefac {
        Scalefac::Long(_) => BlockType::Normal,
        Scalefac::Short(_) => BlockType::Short,
    };
    let granules: Vec<_> = spectra
        .iter()
        .map(|values| {
            let values = compensate(
                values,
                scalefactors,
                block_type,
                gain_steps,
                sampling_frequency,
            );
            let global_gain = GLOBAL_GAIN + gain_steps;
            let mut granule =
                QuantizedGranule::new(global_gain, values, block_type, sampling_frequency);
            granule.scalefactors = scalefactors.clone();
            granule
        })
        .zip([[false; 4], scfsi])
        .map(|(mut granule, scfsi)| {
            granule.scalefactors.scfsi = scfsi;
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
        let spectra = [
            spectrum(&mut rng, BlockType::Normal, fs),
            spectrum(&mut rng, BlockType::Normal, fs),
        ];
        let expected = decode(
            fs,
            &spectra,
            &Scalefactors::new(BlockType::Normal, fs.version()),
            0,
        );

        let mut variants = Vec::new();
        // scalefac_scale 0: 8 steps of 2^(1/2) in random bands of 0 to 10,
        // which slen1 can hold.
        for _ in 0..3 {
            let mut scalefactors = Scalefactors::new(BlockType::Normal, fs.version());
            for scalefac in &mut scalefactors.values_mut()[..11] {
                *scalefac = 8 * rng.next(2) as u8;
            }
            variants.push(("scalefac_scale 0", scalefactors, 0));
        }
        // scalefac_scale 1: 4 steps of 2 in random bands of 0 to 20.
        for _ in 0..3 {
            let mut scalefactors = Scalefactors {
                scalefac_scale: true,
                ..Scalefactors::new(BlockType::Normal, fs.version())
            };
            for scalefac in scalefactors.values_mut() {
                *scalefac = 4 * rng.next(2) as u8;
            }
            variants.push(("scalefac_scale 1", scalefactors, 0));
        }
        // Preemphasis: scalefactors that make up the rest of 4 steps of 2
        // over pretab in every band.
        let mut scalefactors = Scalefactors {
            scalefac_scale: true,
            preflag: true,
            ..Scalefactors::new(BlockType::Normal, fs.version())
        };
        for (scalefac, pretab) in scalefactors.values_mut().iter_mut().zip(PRETAB) {
            *scalefac = 4 - pretab;
        }
        variants.push(("preflag", scalefactors, 0));
        // 4 steps of 2 in random bands of 0 to 17 and none above: at the
        // lower sampling frequencies, scalefac_compress from 400 sends these
        // in partitions of 6, 5, 7 and 3 bands, the last in no bits.
        let mut scalefactors = Scalefactors {
            scalefac_scale: true,
            ..Scalefactors::new(BlockType::Normal, fs.version())
        };
        for scalefac in &mut scalefactors.values_mut()[..18] {
            *scalefac = 4 * rng.next(2) as u8;
        }
        if fs.version() == Version::Mpeg2 {
            let compress = scalefactors.compress().unwrap();
            assert!((400..500).contains(&compress), "{fs:?}: {compress}");
        }
        variants.push(("none above band 17", scalefactors, 0));
        // The same scalefactor in every band, undone by global_gain: 1, 2 and
        // 3 steps for the narrower field widths.
        for (scalefac, scalefac_scale) in [(1, true), (2, false), (3, false), (2, true)] {
            let scalefactors = Scalefactors {
                scalefac: Scalefac::Long([scalefac; LONG_BANDS]),
                scalefac_scale,
                ..Scalefactors::new(BlockType::Normal, fs.version())
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

/// Short blocks send `scalefac_s[sfb][window]` for short bands 0 to 11, the
/// windows of a band together, bands 0 to 5 with `slen1` (clause 2.4.2.7).
/// Each must divide its own band in its own window, which also needs the
/// values to be sent by band, window and frequency.
#[test]
fn short_scalefactors_divide_their_band_and_window() {
    for fs in SamplingFrequency::ALL {
        let mut rng = Rng(0x5401_7B1C);
        let spectra = [
            spectrum(&mut rng, BlockType::Short, fs),
            spectrum(&mut rng, BlockType::Short, fs),
        ];
        let short = Scalefactors::new(BlockType::Short, fs.version());
        let expected = decode(fs, &spectra, &short, 0);

        let mut variants = Vec::new();
        // scalefac_scale 0: 8 steps in random bands and windows sent with
        // slen1, which can hold them.
        for _ in 0..3 {
            let mut scalefactors = short.clone();
            for scalefac in &mut scalefactors.values_mut()[..18] {
                *scalefac = 8 * rng.next(2) as u8;
            }
            variants.push(("scalefac_scale 0", scalefactors, 0));
        }
        // scalefac_scale 1: 4 steps in any band and window.
        for _ in 0..3 {
            let mut scalefactors = Scalefactors {
                scalefac_scale: true,
                ..short.clone()
            };
            for scalefac in scalefactors.values_mut() {
                *scalefac = 4 * rng.next(2) as u8;
            }
            variants.push(("scalefac_scale 1", scalefactors, 0));
        }
        // The same scalefactor everywhere, undone by global_gain.
        for (scalefac, scalefac_scale) in [(1, true), (2, false), (3, false)] {
            let mut scalefactors = Scalefactors {
                scalefac_scale,
                ..short.clone()
            };
            scalefactors.values_mut().fill(scalefac);
            let gain_steps = if scalefac_scale { 4 } else { 2 } * scalefac;
            variants.push(("uniform", scalefactors, gain_steps));
        }
        for (label, scalefactors, gain_steps) in &variants {
            let samples = decode(fs, &spectra, scalefactors, *gain_steps);
            assert_close(
                &format!("{fs:?} short {label} {scalefactors:?}"),
                &samples,
                &expected,
            );
        }
    }
}

/// With `scfsi` set, the second granule of a frame does not send the
/// scalefactors of that scfsi band but takes over the first granule's
/// (clauses 2.4.1.7 and 2.4.2.7). Encoded either way, the same scalefactors
/// decode alike. The lower sampling frequencies have no scfsi.
#[test]
fn scfsi_takes_over_the_first_granules_scalefactors() {
    let mpeg1 = SamplingFrequency::ALL
        .into_iter()
        .filter(|fs| fs.version() == Version::Mpeg1);
    for fs in mpeg1 {
        let mut rng = Rng(0x5CF5_1000);
        let spectra = [
            spectrum(&mut rng, BlockType::Normal, fs),
            spectrum(&mut rng, BlockType::Normal, fs),
        ];
        let mut scalefactors = Scalefactors {
            scalefac_scale: true,
            ..Scalefactors::default()
        };
        for scalefac in scalefactors.values_mut() {
            *scalefac = 4 * rng.next(2) as u8;
        }
        let expected = decode(fs, &spectra, &scalefactors, 0);
        for scfsi in [
            [true, false, false, false],
            [false, true, true, false],
            [false, false, false, true],
            [true; 4],
        ] {
            let samples = decode_with_scfsi(fs, &spectra, &scalefactors, 0, scfsi);
            assert_close(&format!("{fs:?} scfsi {scfsi:?}"), &samples, &expected);
        }
    }
}
