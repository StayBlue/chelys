// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The outer iteration loop, or distortion control loop (clause C.7.4.1,
//! figure C.7), which amplifies the scalefactor bands whose quantization
//! noise exceeds what the psychoacoustic model allows them.

use super::inner::{Counting, decoder_gain, first_inner_loop, inner_loop, power_four_thirds};
use crate::channel::CodedChannel;
use crate::scalefactor_bands::MAX_BANDS;

/// The most that the outer loop amplifies a band by, in steps of the
/// quantizer: the largest difference of two scalefactors that the
/// scalefactor codebook codes (Table A.1), as every band starts at 0 (see
/// [`outer_loop`]).
pub const MAX_AMPLIFICATION: u8 = 60;

/// The outer iteration loop (figure C.7): quantizes the lines of
/// `counting` in its available bits with the inner loop, then amplifies
/// each scalefactor band whose distortion exceeds its allowed distortion
/// `xmin`, the energy of noise the psychoacoustic model allows it (clause
/// C.1.4, step 14), and runs the inner loop again, until no band exceeds
/// it, or every band that could has been amplified, or the scalefactors
/// would pass their limits (clause C.7.4.4). The channel is then coded as
/// the iteration whose distortion exceeds the allowance least quantized it.
///
/// Spec reading: this deliberately departs from figure C.7, which on
/// stopping restores the scalefactors saved before the last amplification,
/// and so codes the last iteration. Each iteration fits the bits, so any is
/// a valid coding, and the one kept is that of the least excess: the sum,
/// over the bands whose distortion exceeds their `xmin`, of the natural
/// logarithm of the ratio, as chelys's MP3 encoder keeps. Short of bits,
/// amplifying the bands over their allowance takes bits for their finer
/// steps and coarsens every other band, and the last iteration can be the
/// worst, with more bands over their allowance, and further, than the
/// first.
/// Graded with PEAQ (GstPEAQ 0.6.1, black box), on 8 tracks of pop music
/// at 44.1 kHz in stereo, with `xmin` raised by 0, 3 and 6 dB (see
/// [`XMIN_OFFSET_DB`](super::XMIN_OFFSET_DB)), restoring the last
/// iteration gave means of -2.54, -2.24 and -2.19 at 128 kbit/s, and
/// keeping the least excess -2.35, -2.17 and -2.15; at 192 kbit/s, keeping
/// it gave -0.650, -0.664 and -0.682 (restoring the last, -0.690 at 6 dB),
/// and at 64 kbit/s -3.756, -3.749 and -3.748 (-3.791).
///
/// `xmin` gives each window group's bands, in the order of
/// [`Counting::groups`].
///
/// Spec reading: clause C.7.4.3 computes the distortion with the lines
/// requantized by `2^(1/4 * (scalefactor(sb) - common_scalefac))`, and
/// attenuates a band whose distortion exceeds `xmin` by lowering its
/// `scalefactor(sb)` by 1. Both take the scalefactor with the sign
/// reversed, as the second form of the quantizer in clause C.7.4 does,
/// which is taken as misprinted (see
/// [`Lines::quantize`](super::inner::Lines::quantize)): consistent among
/// themselves, they disagree with clauses C.7.4.1 and C.7.4.2, with
/// `sf_decoder = common_scalefac - scalefactor + SF_OFFSET` and with
/// figure C.7. With the sign of
/// clauses C.7.4.1 and C.7.4.2, by which "a positive change increases the
/// magnitude of x_quant, and so decreases the distortion", the lines are
/// requantized by the decoder's gain, `2^(1/4 * (common_scalefac -
/// scalefactor(sb)))`, `2^(1/4 * (sf_decoder - SF_OFFSET))`, and a band is
/// amplified by raising its `scalefactor(sb)` by 1, as figure C.7 has it
/// ("Amplify scalefactor bands with more than the allowed distortion").
///
/// Spec reading: figure C.7 stops when "All scalefactor bands amplified",
/// which clause C.7.4.4 gives as "All scalefactor bands with an energy
/// exceeding xmin(sb) are already attenuated". A band is taken as
/// amplified when its scalefactor is above 0, and only the bands whose
/// MDCT energy exceeds their `xmin` are asked about: the others could be
/// quantized to zero within their allowance, and are never amplified. As
/// figure C.7 asks after amplifying, a frame in whose first iteration
/// every such band exceeds its allowance stops there.
///
/// Spec reading: figure C.7 stops unless the "Amplification of all bands
/// below upper limit" holds, which clause C.7.4.4 gives as "The difference
/// between two consecutive scalefactors is greater than 60", the most the
/// scalefactor codebook codes. Which scalefactors are consecutive in
/// `scale_factor_data()` depends on which bands the next quantization
/// leaves with lines, so the limit is kept for any two: every band starts
/// at 0, and none is amplified past [`MAX_AMPLIFICATION`]. The upper limit
/// of `sf_decoder` is kept too: no band is amplified past the step of the
/// last quantization, which would take its `sf_decoder` below 0 (clause
/// 11.3.2); the later runs of the inner loop only raise the step. So no
/// zero band's scalefactor, which scales nothing and is sent as a
/// difference of 0 (see [`scalefactors`](crate::noiseless::scalefactors)),
/// need bridge a difference beyond the codebook's.
///
/// Spec reading: figure C.7 asks whether there is "At least one band with
/// more than the allowed distortion" after the amplification; that is read
/// as the distortion measured before it, as the amplification changes no
/// distortion. Where no band had more, nothing was amplified, and the
/// iteration measured last, within every allowance, is the one kept.
///
/// Spec gap: the later runs of the inner loop raise the step from the one
/// before until the lines fit (clause C.7.4.5), and Annex C does not say
/// what to do where they do not fit even at the largest step. The loop
/// stops there, as at its limits.
///
/// Spec gap: where the lines do not fit even at the largest step without
/// scalefactors, the first run of the inner loop leaves the highest bands
/// unsent (see [`first_inner_loop`]); the channel is coded so, without
/// running the outer loop on what is left.
///
/// The offset to every `xmin` of clause C.7.4.4 is applied before the loop
/// (see [`quantize_channel`](super::quantize_channel)).
pub fn outer_loop(mut counting: Counting, xmin: &[[f64; MAX_BANDS]]) -> CodedChannel {
    let groups: Vec<_> = counting.groups().collect();
    assert_eq!(xmin.len(), groups.len(), "an xmin for each window group");
    let (mut gain, band_limit) = first_inner_loop(&mut counting);
    if band_limit < MAX_BANDS {
        return counting.into_channel(gain, band_limit);
    }
    // Whether each band's energy exceeds its xmin: those the loop is to
    // amplify, if it can.
    let mut audible = vec![[false; MAX_BANDS]; groups.len()];
    let spectrum = counting.lines().spectrum();
    for (g, &(start, bands)) in groups.iter().enumerate() {
        for (sb, audible) in audible[g][..bands.num_swb()].iter_mut().enumerate() {
            let lines = &spectrum[start + bands.offset(sb)..start + bands.offset(sb + 1)];
            let energy: f64 = lines.iter().map(|x| x * x).sum();
            *audible = energy > xmin[g][sb];
        }
    }
    let mut distortion = vec![[0.0; MAX_BANDS]; groups.len()];
    // The iteration of the least excess: its scalefactors, and its excess
    // and step.
    let mut best_scalefactors = counting.scalefactors.clone();
    let mut best: Option<(f64, u8)> = None;
    loop {
        // Calculate the distortion for each scalefactor band.
        quantization_distortion(&counting, gain, &mut distortion);
        let excess = excess(&groups, &distortion, xmin);
        if best.is_none_or(|(least, _)| excess < least) {
            best = Some((excess, gain));
            best_scalefactors.copy_from_slice(&counting.scalefactors);
        }
        // Amplify the scalefactor bands with more than the allowed
        // distortion.
        let mut violated = false;
        for (g, &(_, bands)) in groups.iter().enumerate() {
            for sb in 0..bands.num_swb() {
                if distortion[g][sb] > xmin[g][sb] {
                    counting.scalefactors[g][sb] += 1;
                    violated = true;
                }
            }
        }
        let all_amplified = (groups.iter().enumerate()).all(|(g, &(_, bands))| {
            (0..bands.num_swb()).all(|sb| !audible[g][sb] || counting.scalefactors[g][sb] > 0)
        });
        let below_limit = (counting.scalefactors.iter())
            .flatten()
            .all(|&scalefactor| scalefactor <= MAX_AMPLIFICATION.min(gain));
        if all_amplified || !below_limit || !violated {
            break;
        }
        match inner_loop(&mut counting, gain, 1) {
            Some(step) => gain = step,
            None => break,
        }
    }
    let (_, gain) = best.expect("an iteration was measured");
    counting.scalefactors.copy_from_slice(&best_scalefactors);
    counting.into_channel(gain, MAX_BANDS)
}

/// How far the distortion of the bands exceeds their allowance: the sum,
/// over the bands whose distortion exceeds their `xmin`, of the natural
/// logarithm of the ratio.
fn excess(
    groups: &[(usize, crate::scalefactor_bands::GroupBands)],
    distortion: &[[f64; MAX_BANDS]],
    xmin: &[[f64; MAX_BANDS]],
) -> f64 {
    (groups.iter().enumerate())
        .flat_map(|(g, &(_, bands))| (0..bands.num_swb()).map(move |sb| (g, sb)))
        .filter(|&(g, sb)| distortion[g][sb] > xmin[g][sb])
        .map(|(g, sb)| (distortion[g][sb] / xmin[g][sb]).ln())
        .sum()
}

/// The distortion `error_energy(sb)` of each band of each window group
/// of the quantization `counting` counted last, at the step `common_gain`
/// (clause C.7.4.3; see [`outer_loop`] for its sign): the energy of the
/// difference between the lines and the lines the decoder reconstructs.
fn quantization_distortion(
    counting: &Counting,
    common_gain: u8,
    distortion: &mut [[f64; MAX_BANDS]],
) {
    let spectrum = counting.lines().spectrum();
    for (g, ((start, bands), (_, quantized))) in
        counting.groups().zip(counting.group_lines()).enumerate()
    {
        for sb in 0..bands.num_swb() {
            let range = bands.offset(sb)..bands.offset(sb + 1);
            let lines = &spectrum[start + range.start..start + range.end];
            let sf_decoder = counting
                .sf_decoder(common_gain, g, sb)
                .expect("the scalefactors quantized with are in range");
            let gain = decoder_gain(sf_decoder);
            distortion[g][sb] = (lines.iter().zip(&quantized[range]))
                .map(|(&x, &q)| {
                    let error = x.abs() - power_four_thirds(q.unsigned_abs()) * gain;
                    error * error
                })
                .sum();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::inner::Lines;
    use super::*;
    use crate::channel::FRAME_LINES;
    use crate::raw_data_block::element::COMMON_WINDOW;
    use crate::sampling_frequency::SamplingFrequency;
    use crate::scalefactor_bands::GroupBands;
    use crate::window::Windows;

    fn noise(seed: u64) -> [f64; FRAME_LINES] {
        let mut state = seed;
        std::array::from_fn(|i| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let uniform = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            uniform * 1e5 / (1.0 + i as f64 / 50.0)
        })
    }

    fn band_energy(spectrum: &[f64; FRAME_LINES], bands: GroupBands) -> [f64; MAX_BANDS] {
        let mut energy = [0.0; MAX_BANDS];
        for (sb, energy) in energy[..bands.num_swb()].iter_mut().enumerate() {
            *energy = spectrum[bands.offset(sb)..bands.offset(sb + 1)]
                .iter()
                .map(|x| x * x)
                .sum();
        }
        energy
    }

    fn outcome(
        spectrum: &[f64; FRAME_LINES],
        available: u32,
        xmin: [f64; MAX_BANDS],
    ) -> (CodedChannel, [[f64; MAX_BANDS]; 1]) {
        let fs = SamplingFrequency::Hz44100;
        let counting = Counting::new(Lines::new(spectrum), Windows::LONG_SINE, fs, available);
        let channel = outer_loop(counting, &[xmin]);
        // Measure the coded channel's distortion as a decoder would.
        let bands = GroupBands::long(fs);
        let (group, lines) = channel.groups().next().unwrap();
        let mut distortion = [[0.0; MAX_BANDS]];
        for sb in 0..bands.num_swb() {
            let gain = if sb < group.sections.max_sfb() && !group.sections.is_zero_band(sb) {
                decoder_gain(group.scalefactors[sb])
            } else {
                0.0
            };
            let range = bands.offset(sb)..bands.offset(sb + 1);
            distortion[0][sb] = (spectrum[range.clone()].iter().zip(&lines[range]))
                .map(|(&x, &q)| {
                    let decoded =
                        power_four_thirds(q.unsigned_abs()) * gain * f64::from(q.signum());
                    (x - decoded).powi(2)
                })
                .sum();
        }
        (channel, distortion)
    }

    #[test]
    fn plenty_of_bits_meet_every_allowance() {
        let spectrum = noise(3);
        let bands = GroupBands::long(SamplingFrequency::Hz44100);
        // Ask for noise 30 dB below the energy of the low bands and 10 dB
        // below that of the high ones.
        let energy = band_energy(&spectrum, bands);
        let xmin = std::array::from_fn(|sb| energy[sb] * if sb < 20 { 1e-4 } else { 1e-1 });
        let (channel, distortion) = outcome(&spectrum, 7000, xmin);
        assert!(channel.bits(COMMON_WINDOW) <= 7000);
        for (sb, xmin) in xmin[..bands.num_swb()].iter().enumerate() {
            assert!(distortion[0][sb] <= *xmin, "band {sb}");
        }
        // Low bands were amplified over the high ones: given finer steps,
        // lower scalefactors.
        let (group, _) = channel.groups().next().unwrap();
        let low = group.scalefactors[..20].iter().min().unwrap();
        assert!(*low < group.scalefactors[30], "{:?}", group.scalefactors);
    }

    #[test]
    fn short_of_bits_the_noise_moves_out_of_the_tightest_bands() {
        let spectrum = noise(4);
        let fs = SamplingFrequency::Hz44100;
        let bands = GroupBands::long(fs);
        let energy = band_energy(&spectrum, bands);
        let xmin = std::array::from_fn(|sb| energy[sb] * if sb < 20 { 1e-4 } else { 1e-1 });
        let (channel, distortion) = outcome(&spectrum, 3000, xmin);
        assert!(channel.bits(COMMON_WINDOW) <= 3000);
        // With every scalefactor 0, as the first run of the inner loop
        // quantizes.
        let mut counting = Counting::new(Lines::new(&spectrum), Windows::LONG_SINE, fs, 3000);
        let (gain, _) = first_inner_loop(&mut counting);
        let mut flat = [[0.0; MAX_BANDS]];
        quantization_distortion(&counting, gain, &mut flat);
        let excess = |distortion: &[f64; MAX_BANDS]| -> f64 {
            (0..20).map(|sb| (distortion[sb] / xmin[sb]).ln()).sum()
        };
        assert!(
            excess(&distortion[0]) < excess(&flat[0]) - 1.0,
            "{} against {}",
            excess(&distortion[0]),
            excess(&flat[0])
        );
    }

    #[test]
    fn scalefactors_stay_within_the_codebook_and_their_range() {
        // An allowance no band can meet: the loop runs to its limits.
        for (seed, available) in [(5, 800), (6, 4000), (7, 30_000)] {
            let spectrum = noise(seed);
            let (channel, _) = outcome(&spectrum, available, [1e-9; MAX_BANDS]);
            assert!(channel.bits(COMMON_WINDOW) <= available);
            let (group, _) = channel.groups().next().unwrap();
            let sent: Vec<u8> = (group.sections.bands_with_scalefactors())
                .filter(|&sb| !group.sections.is_zero_band(sb))
                .map(|sb| group.scalefactors[sb])
                .collect();
            let range = sent.iter().max().unwrap() - sent.iter().min().unwrap();
            assert!(range <= MAX_AMPLIFICATION, "{seed}: {sent:?}");
        }
    }

    #[test]
    fn the_coding_kept_exceeds_the_allowance_no_more_than_the_first() {
        let fs = SamplingFrequency::Hz44100;
        let bands = GroupBands::long(fs);
        let mut improved = false;
        for (seed, available) in [(11, 600), (12, 1200), (13, 2500), (14, 4000)] {
            let spectrum = noise(seed);
            let energy = band_energy(&spectrum, bands);
            let xmin = std::array::from_fn(|sb| energy[sb] * if sb < 20 { 1e-4 } else { 0.3 });
            let (_, kept) = outcome(&spectrum, available, xmin);
            let mut counting =
                Counting::new(Lines::new(&spectrum), Windows::LONG_SINE, fs, available);
            let (gain, _) = first_inner_loop(&mut counting);
            let mut first = [[0.0; MAX_BANDS]];
            quantization_distortion(&counting, gain, &mut first);
            let groups = [(0, bands)];
            let (kept, first) = (
                excess(&groups, &kept, &[xmin]),
                excess(&groups, &first, &[xmin]),
            );
            assert!(kept <= first + 1e-9, "{seed}: {kept} against {first}");
            improved |= kept < first;
        }
        assert!(improved, "the loop found a coding better than the first");
    }
}
