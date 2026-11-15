// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Huffman-coded spectra decoded by ffmpeg.
//!
//! Most checks encode the same quantised values in different but equally
//! valid ways (other tables, partitions or region splits) and require
//! bit-identical decoded output. A wrongly transcribed code word, a swapped
//! pair of code words, or a region boundary that disagrees with the decoder
//! makes one encoding decode differently from the rest.

use chelys::mp3::header::{Bitrate, Mode, SamplingFrequency};
use chelys::mp3::huffman::{MAX_MAGNITUDE, QuadTable, SpectrumCoding, pair_table};
use chelys::mp3::rate::Rate;
use chelys::mp3::{BlockType, GRANULE_LINES, QuantizedGranule, StreamConfig, encode_granules};
use chelys_conformance::ffmpeg;

type Spectrum = [i32; GRANULE_LINES];

fn config(sampling_frequency: SamplingFrequency) -> StreamConfig {
    StreamConfig {
        // The highest: 320 kbit/s, or 160 at the lower sampling frequencies.
        rate: Rate::Constant(Bitrate::all(sampling_frequency.version()).last().unwrap()),
        sampling_frequency,
        mode: Mode::SingleChannel,
        crc_protected: false,
    }
}

/// Encodes three frames, each holding `spectra` as its two granules, with
/// `adjust` applied to every granule's coding, and decodes the result.
fn decode(
    sampling_frequency: SamplingFrequency,
    global_gain: u8,
    spectra: [&Spectrum; 2],
    adjust: impl Fn(&mut SpectrumCoding, &Spectrum),
) -> Vec<f32> {
    decode_blocks(
        sampling_frequency,
        global_gain,
        spectra,
        BlockType::Normal,
        adjust,
    )
}

/// [`decode`] with every granule of `block_type`.
fn decode_blocks(
    sampling_frequency: SamplingFrequency,
    global_gain: u8,
    spectra: [&Spectrum; 2],
    block_type: BlockType,
    adjust: impl Fn(&mut SpectrumCoding, &Spectrum),
) -> Vec<f32> {
    let granules: Vec<_> = spectra
        .map(|values| {
            let mut granule =
                QuantizedGranule::new(global_gain, *values, block_type, sampling_frequency);
            adjust(&mut granule.coding, values);
            vec![granule]
        })
        .into_iter()
        .cycle()
        .take(6)
        .collect();
    let stream = encode_granules(&config(sampling_frequency), &granules);
    let decoded = ffmpeg::decode_mp3(&stream);
    assert_eq!(decoded.diagnostics, "");
    assert!(decoded.samples.iter().any(|&s| s != 0.0), "decoded silence");
    decoded.samples
}

/// Moves every value into the `big_values` partition, coded with `table`.
fn all_pairs(coding: &mut SpectrumCoding, values: &Spectrum, table: u8) {
    let last = values.iter().rposition(|&v| v != 0).map_or(0, |i| i + 1);
    coding.big_values = last.div_ceil(2);
    coding.count1 = 0;
    coding.table_select = [table; 3];
}

/// Every table that can code magnitudes up to `max`.
fn tables_for(max: u32) -> Vec<u8> {
    (0..32)
        .filter(|&t| {
            pair_table(t).is_some_and(|table| {
                if table.linbits == 0 {
                    (max as usize) < table.size
                } else {
                    max < 15 + (1 << table.linbits)
                }
            })
        })
        .collect()
}

/// A deterministic sequence for test spectra (xorshift32).
struct Rng(u32);

impl Rng {
    fn next(&mut self, below: u32) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0 % below
    }

    fn signed(&mut self, max: u32) -> i32 {
        let magnitude = self.next(max + 1) as i32;
        if self.next(2) == 0 {
            magnitude
        } else {
            -magnitude
        }
    }
}

/// Splits pairs of values over two granules' spectra, in order.
fn spread(pairs: &[(i32, i32)]) -> [Spectrum; 2] {
    let mut spectra = [[0; GRANULE_LINES]; 2];
    let half = pairs.len().div_ceil(2);
    for (i, &(x, y)) in pairs.iter().enumerate() {
        let (spectrum, at) = (i / half, 2 * (i % half));
        spectra[spectrum][at] = x;
        spectra[spectrum][at + 1] = y;
    }
    spectra
}

fn assert_all_equal(results: &[(String, Vec<f32>)]) {
    let (reference, expected) = &results[0];
    for (label, samples) in &results[1..] {
        assert!(
            samples == expected,
            "{label} decodes differently from {reference}"
        );
    }
}

#[test]
fn every_table_codes_every_pair_alike() {
    let fs = SamplingFrequency::Hz32000;
    for size in [2, 3, 4, 6, 8, 16] {
        let mut rng = Rng(size as u32);
        let mut pairs = Vec::new();
        for x in 0..size {
            for y in 0..size {
                let sign = |rng: &mut Rng| if rng.next(2) == 0 { 1 } else { -1 };
                pairs.push((x * sign(&mut rng), y * sign(&mut rng)));
            }
        }
        let spectra = spread(&pairs);
        let results: Vec<_> = tables_for(size as u32 - 1)
            .into_iter()
            .map(|table| {
                let samples = decode(fs, 150, [&spectra[0], &spectra[1]], |coding, values| {
                    all_pairs(coding, values, table)
                });
                (format!("table {table} on {size}x{size} pairs"), samples)
            })
            .collect();
        assert!(results.len() > 1);
        assert_all_equal(&results);
    }
}

#[test]
fn every_escape_table_codes_escapes_alike() {
    let fs = SamplingFrequency::Hz32000;
    for linbits in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13] {
        let max = (14 + (1u32 << linbits)).min(MAX_MAGNITUDE);
        let mut rng = Rng(linbits);
        // Escaped magnitudes from 15 to the largest this linbits allows,
        // paired with small ones and each other.
        let escapes: Vec<i32> = (0..40).map(|i| (15 + (max - 15) * i / 39) as i32).collect();
        let pairs: Vec<_> = escapes
            .iter()
            .enumerate()
            .map(|(i, &e)| {
                let sign = if rng.next(2) == 0 { 1 } else { -1 };
                match i % 3 {
                    0 => (e * sign, rng.signed(14)),
                    1 => (rng.signed(14), -e * sign),
                    _ => (e * sign, escapes[39 - i] * -sign),
                }
            })
            .collect();
        let spectra = spread(&pairs);
        let results: Vec<_> = tables_for(max)
            .into_iter()
            .map(|table| {
                let samples = decode(fs, 100, [&spectra[0], &spectra[1]], |coding, _| {
                    coding.table_select = [table; 3];
                });
                (format!("table {table} with escapes up to {max}"), samples)
            })
            .collect();
        assert!(results.len() > 1, "linbits {linbits}");
        assert_all_equal(&results);
    }
}

#[test]
fn quadruples_decode_like_pairs() {
    let fs = SamplingFrequency::Hz32000;
    let mut rng = Rng(7);
    let mut values = [0; GRANULE_LINES];
    for v in &mut values[..400] {
        *v = rng.signed(1);
    }
    values[399] = 1;
    let both = [&values, &values];
    let results = vec![
        (
            "count1 table A".to_owned(),
            decode(fs, 180, both, |c, _| c.count1_table = QuadTable::A),
        ),
        (
            "count1 table B".to_owned(),
            decode(fs, 180, both, |c, _| c.count1_table = QuadTable::B),
        ),
        (
            "pairs with table 1".to_owned(),
            decode(fs, 180, both, |c, v| all_pairs(c, v, 1)),
        ),
    ];
    assert_all_equal(&results);
}

#[test]
fn region_boundaries_match_the_decoder_at_every_sampling_frequency() {
    let mut rng = Rng(11);
    let mut values = [0; GRANULE_LINES];
    for v in &mut values[..560] {
        *v = rng.signed(1);
    }
    values[559] = 1;
    for fs in SamplingFrequency::STANDARD {
        let mut results = vec![(
            "one table".to_owned(),
            decode(fs, 180, [&values, &values], |c, v| all_pairs(c, v, 1)),
        )];
        // Different tables per region: if either side placed a boundary
        // differently, some pairs would be decoded with the wrong table.
        for (region0_count, region1_count) in [(0, 0), (2, 3), (5, 7), (9, 7), (12, 6)] {
            let samples = decode(fs, 180, [&values, &values], |coding, values| {
                all_pairs(coding, values, 1);
                coding.table_select = [2, 5, 3];
                coding.region0_count = region0_count;
                coding.region1_count = region1_count;
            });
            results.push((
                format!("regions {region0_count}/{region1_count} at {fs:?}"),
                samples,
            ));
        }
        assert_all_equal(&results);
    }
}

/// With window switching the region boundaries are not sent: region 0 holds
/// 8 long scalefactor bands, or for short blocks 9 short ones counted once
/// per window, and region 1 the rest of `big_values` (clause 2.4.2.7). Two
/// tables, one per region, decode like one if the decoder agrees.
#[test]
fn switched_window_regions_match_the_decoder_at_every_sampling_frequency() {
    let mut rng = Rng(13);
    let mut values = [0; GRANULE_LINES];
    for v in &mut values[..560] {
        *v = rng.signed(1);
    }
    values[559] = 1;
    for fs in SamplingFrequency::STANDARD {
        for block_type in [BlockType::Start, BlockType::Short, BlockType::Stop] {
            let both = [&values, &values];
            let results = vec![
                (
                    format!("one table, {block_type:?} at {fs:?}"),
                    decode_blocks(fs, 180, both, block_type, |c, v| all_pairs(c, v, 1)),
                ),
                (
                    format!("two tables, {block_type:?} at {fs:?}"),
                    decode_blocks(fs, 180, both, block_type, |coding, values| {
                        all_pairs(coding, values, 1);
                        coding.table_select = [3, 1, 0];
                    }),
                ),
            ];
            assert_all_equal(&results);
        }
    }
}

#[test]
fn chosen_coding_decodes_like_a_single_wide_table() {
    for (seed, fs) in SamplingFrequency::STANDARD.into_iter().enumerate() {
        let mut rng = Rng(100 + seed as u32);
        let mut values = [0; GRANULE_LINES];
        for (i, v) in values.iter_mut().enumerate() {
            // Loud low lines, quieter middle lines, a sparse top.
            *v = match i {
                0..40 => rng.signed(60),
                40..200 => rng.signed(6),
                200..420 => rng.signed(1),
                _ => 0,
            };
        }
        values[419] = -1;
        let chosen = decode(fs, 150, [&values, &values], |_, _| {});
        let wide = decode(fs, 150, [&values, &values], |c, _| c.table_select = [31; 3]);
        assert!(chosen == wide, "{fs:?}: chosen tables decode differently");
    }
}

/// The power of `samples` at `frequency`, by the Goertzel recurrence over a
/// Hann window.
fn power(samples: &[f32], frequency: f64, sampling_frequency: f64) -> f64 {
    let n = samples.len() as f64;
    let coefficient = 2.0 * (std::f64::consts::TAU * frequency / sampling_frequency).cos();
    let (mut s1, mut s2) = (0.0, 0.0);
    for (i, &x) in samples.iter().enumerate() {
        let window = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n).cos();
        let s0 = f64::from(x) * window + coefficient * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    s1 * s1 + s2 * s2 - coefficient * s1 * s2
}

/// The frequency, in units of the line spacing `fs / 1152`, at which
/// `samples` peak between lines `from` and `to`.
fn peak_line(samples: &[f32], from: usize, to: usize, sampling_frequency: f64) -> f64 {
    let spacing = sampling_frequency / 1152.0;
    (from * 100..=to * 100)
        .map(|centi| centi as f64 / 100.0)
        .max_by(|&a, &b| {
            power(samples, a * spacing, sampling_frequency).total_cmp(&power(
                samples,
                b * spacing,
                sampling_frequency,
            ))
        })
        .unwrap()
}

/// A single nonzero line `k` decodes to a tone near `(k + 0.5) * fs / 1152`:
/// the long-block IMDCT (clause 2.4.3.4.10.2) puts line `j` of subband `sb` at
/// `(j + 0.5) * fs / 1152` above the subband's lower edge at `sb * fs / 64`.
/// Repeating one coefficient in every granule does not make a pure tone at
/// exactly that frequency, since the transform's phase advances from block to
/// block, but it stays within half a line of it. Lines 8 and 9 of a subband
/// are untouched by alias reduction (clause 2.4.3.4.10.1), so the tone is
/// clean. Placing a value in the wrong line, for example by swapping `x` and
/// `y`, moves the tone by at least a line.
#[test]
fn a_single_line_decodes_to_a_tone_at_its_frequency() {
    let fs = SamplingFrequency::Hz44100;
    for k in [
        3 * 18 + 8,
        3 * 18 + 9,
        7 * 18 + 8,
        7 * 18 + 9,
        20 * 18 + 8,
        20 * 18 + 9,
    ] {
        let mut values = [0; GRANULE_LINES];
        values[k] = 3;
        let samples = decode(fs, 200, [&values, &values], |_, _| {});
        // Skip the decoder's start-up.
        let steady = &samples[2 * 1152..];
        let peak = peak_line(steady, k - 3, k + 4, f64::from(fs.hz()));
        assert!(
            (peak - (k as f64 + 0.5)).abs() <= 0.55,
            "line {k} peaks at line {peak}"
        );
    }
}
