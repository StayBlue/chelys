// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The bits of a channel at a variable rate: those it takes with each
//! scalefactor band at the coarsest step whose distortion stays within its
//! allowed distortion.

use super::inner::Counting;
use super::outer::MAX_AMPLIFICATION;
use crate::scalefactor_bands::MAX_BANDS;

/// Sets the scalefactors of `counting` so that each scalefactor band is
/// quantized at the coarsest step at which its distortion is within its
/// allowed distortion `xmin`, and those that can quantize to zero within it
/// to zero, and gives the common step to quantize them from, if the channel
/// then fits the bits `counting` makes available, which
/// [`Counting::counted_bits`] then gives; or leaves `counting` as it was,
/// and gives `None`, if it does not.
///
/// Spec reading: this deliberately departs from Annex C, whose iteration
/// loops fit each channel to the bits the bit reservoir makes available
/// (clause C.7.3) and shape its noise within them (clause C.7.4); it has
/// no variable rate. Clause 8.2.4 allows one, in which "each
/// raw_data_block() can have the minimum length (rate) such that the
/// desired audio quality is achieved". The outer loop amplifies, one step
/// at a time, each band whose distortion exceeds `xmin` (figure C.7), and
/// with bits enough it ends with every band within its allowance; each
/// band is given here the step it would end at, searched for each band
/// alone, as its distortion depends on its own step only. The bits that
/// takes are the bits the loops then fit the channel to (see
/// [`quantize_channel`](super::quantize_channel)). The quantizer, the
/// distortion and `xmin` are the loops' (see
/// [`outer_loop`](super::outer::outer_loop)), and `xmin` is the model's,
/// raised as a variable rate asks (see [`Variable`](crate::Rate::Variable)).
///
/// Spec gap: coded as this quantizes them, with every band's noise at its
/// allowance, channels graded lower than the loops code them in the same
/// bits. Graded with PEAQ (GstPEAQ 0.6.1, black box), on 51 tracks of 9
/// kinds of music at 44.1 kHz in stereo at a mean of 128.2 kbit/s, with
/// noise offsets of -0.47 and -0.53 dB, the mean was -1.303 so, and -1.047
/// fitted by the loops, better on every track; at a constant 128 kbit/s it
/// was -1.072. Limiting the steps' span to 20 rather than 60, so that the
/// noise is shaped less closely to the allowance, gave -1.133.
///
/// Spec reading: the scalefactors a channel sends differ by 60 at most
/// (Table A.1), as in the outer loop (see [`MAX_AMPLIFICATION`]). Where
/// the steps its bands need span more, the coarsest are made finer, to
/// 60 above the finest: their noise falls further below their allowance,
/// and none exceeds it. The common step, from which the bands are
/// amplified, is the coarsest of those steps, or of the steps at which the
/// bands that need no lines quantize to zero where that is coarser, within
/// the same 60, so that those bands send none.
///
/// [`Counting::counted_bits`]: super::inner::Counting::counted_bits
pub fn within_allowance(counting: &mut Counting<'_>, xmin: &[[f64; MAX_BANDS]]) -> Option<u8> {
    let groups: Vec<_> = counting.groups().collect();
    assert_eq!(xmin.len(), groups.len(), "an xmin for each window group");
    // Each band's step: the coarsest within its allowance, or where it
    // needs no lines, the finest at which it quantizes to zero.
    let mut steps = vec![[(0u8, false); MAX_BANDS]; groups.len()];
    for (g, &(start, bands)) in groups.iter().enumerate() {
        for (sb, step) in steps[g][..bands.num_swb()].iter_mut().enumerate() {
            let band = bands.band(sb);
            let range = start + band.start..start + band.end;
            *step = match counting.lines().coarsest_step(range.clone(), xmin[g][sb]) {
                Some(step) => (step, true),
                None => (counting.lines().zero_step(range), false),
            };
        }
    }
    let all = || {
        (groups.iter().enumerate())
            .flat_map(|(g, &(_, bands))| (0..bands.num_swb()).map(move |sb| (g, sb)))
            .map(|(g, sb)| steps[g][sb])
    };
    let coarsest = all().map(|(step, _)| step).max().unwrap_or(0);
    let finest_needed = all()
        .filter(|&(_, needed)| needed)
        .map(|(step, _)| step)
        .min();
    let gain = finest_needed.map_or(coarsest, |finest| {
        coarsest.min(finest.saturating_add(MAX_AMPLIFICATION))
    });
    for (g, &(_, bands)) in groups.iter().enumerate() {
        for sb in 0..bands.num_swb() {
            let (step, needed) = steps[g][sb];
            counting.scalefactors[g][sb] = if needed { gain - step.min(gain) } else { 0 };
        }
    }
    if counting.count_at(gain, MAX_BANDS) {
        Some(gain)
    } else {
        for scalefactors in &mut counting.scalefactors {
            scalefactors.fill(0);
        }
        None
    }
}

/// Spends the bits of `counting` that its coding at `common_gain` leaves:
/// makes its bands' steps finer by one, a band at a time from the lowest,
/// every band with a line that is not zero in turn and again, for as many
/// steps as fit the bits it makes available, each band no finer than its
/// lines can be coded at, nor than 60 below the coarsest (see
/// [`MAX_AMPLIFICATION`]); and gives the common step to quantize them from.
/// `counting` holds the coding at that step, which fits.
///
/// Spec gap: at a variable rate the loops are given the bits that the
/// channel takes within its allowance, or the frame's least bits, and the
/// outer loop keeps the iteration of the least excess (see
/// [`outer_loop`](super::outer::outer_loop)), which can take far fewer:
/// the first, where every band is within its allowance, the last, or one
/// between, as the bands within their allowance change. At a constant rate
/// the bit reservoir keeps what a frame leaves; at a variable rate it would
/// be lost, and the frame's length would jump as the offset changes.
/// Given the same bits as its least, a channel of quiet noise was so coded
/// in 240 bits at one noise offset and in 387 at an offset 0.001 dB
/// higher. Spent a band at a time, the bits rise in steps of about a
/// line's, and a frame takes about the bits its allowance or its least
/// length gives it.
pub fn spend(counting: &mut Counting<'_>, common_gain: u8) -> u8 {
    let groups: Vec<_> = counting.groups().collect();
    // Each band with a line that is not zero: where, its step, and the
    // finest its lines can be coded at.
    let mut bands = Vec::new();
    for (g, &(start, group)) in groups.iter().enumerate() {
        for sb in 0..group.num_swb() {
            let band = group.band(sb);
            let range = start + band.start..start + band.end;
            let lines = counting.lines();
            if lines.spectrum()[range.clone()].iter().all(|&x| x == 0.0) {
                continue;
            }
            let step = common_gain - counting.scalefactors[g][sb];
            let finest = lines.finest_step(range).min(step);
            bands.push((g, sb, step, finest));
        }
    }
    if bands.is_empty() {
        return common_gain;
    }
    let original = counting.scalefactors.clone();
    let count = bands.len();
    // The coding with the steps made finer `n` times, a band at a time.
    let code = |counting: &mut Counting<'_>, n: usize| {
        let (rounds, first) = (n / count, n % count);
        let steps: Vec<u8> = (bands.iter().enumerate())
            .map(|(k, &(_, _, step, finest))| {
                let finer = rounds + usize::from(k < first);
                u8::try_from(usize::from(step).saturating_sub(finer))
                    .unwrap_or(0)
                    .max(finest)
            })
            .collect();
        let coarsest = *steps.iter().max().expect("a band");
        let finest = *steps.iter().min().expect("a band");
        let gain = coarsest.min(finest.saturating_add(MAX_AMPLIFICATION));
        for (&(g, sb, _, _), &step) in bands.iter().zip(&steps) {
            counting.scalefactors[g][sb] = gain - step.min(gain);
        }
        counting.count_at(gain, MAX_BANDS).then_some(gain)
    };
    // The most steps that fit, the bits taken to grow with them: doubling
    // the steps, then halving the distance.
    let limit = count * usize::from(u8::MAX);
    let (mut fitting, mut over, mut gain) = (0, None, None);
    let mut next = 1;
    while next <= limit {
        match code(counting, next) {
            Some(g) => (fitting, gain) = (next, Some(g)),
            None => {
                over = Some(next);
                break;
            }
        }
        next *= 2;
    }
    if let Some(mut over) = over {
        while over - fitting > 1 {
            let middle = fitting + (over - fitting) / 2;
            match code(counting, middle) {
                Some(g) => (fitting, gain) = (middle, Some(g)),
                None => over = middle,
            }
        }
    }
    match gain {
        Some(gain) if code(counting, fitting) == Some(gain) => gain,
        _ => {
            counting.scalefactors.copy_from_slice(&original);
            assert!(
                counting.count_at(common_gain, MAX_BANDS),
                "a coding that fitted"
            );
            common_gain
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::ChannelCoding;
    use super::super::inner::{Lines, decoder_gain, powers_four_thirds};
    use super::*;
    use crate::channel::{CodedChannel, FRAME_LINES};
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

    /// Each band's energy, and its distortion as a decoder restores the
    /// lines of `channel`.
    fn measure(spectrum: &[f64; FRAME_LINES], channel: &CodedChannel) -> Vec<(f64, f64)> {
        let bands = GroupBands::long(SamplingFrequency::Hz44100);
        let (group, lines) = channel.groups().next().unwrap();
        (0..bands.num_swb())
            .map(|sb| {
                let sent = sb < group.sections.max_sfb() && !group.sections.is_zero_band(sb);
                let gain = if sent {
                    decoder_gain(group.scalefactors[sb])
                } else {
                    0.0
                };
                let range = bands.offset(sb)..bands.offset(sb + 1);
                let energy = spectrum[range.clone()].iter().map(|x| x * x).sum();
                let distortion = (spectrum[range.clone()].iter().zip(&lines[range]))
                    .map(|(&x, &q)| {
                        let decoded = powers_four_thirds()[q.unsigned_abs() as usize]
                            * gain
                            * f64::from(q.signum());
                        (x - decoded).powi(2)
                    })
                    .sum();
                (energy, distortion)
            })
            .collect()
    }

    fn band_energy(spectrum: &[f64; FRAME_LINES]) -> [f64; MAX_BANDS] {
        let bands = GroupBands::long(SamplingFrequency::Hz44100);
        std::array::from_fn(|sb| {
            (sb < bands.num_swb())
                .then(|| bands.band(sb).map(|i| spectrum[i] * spectrum[i]).sum())
                .unwrap_or(0.0)
        })
    }

    fn quantize(
        spectrum: &[f64; FRAME_LINES],
        available_bits: u32,
        xmin: [f64; MAX_BANDS],
    ) -> Result<CodedChannel, ()> {
        let fs = SamplingFrequency::Hz44100;
        let coding = ChannelCoding::single(Windows::LONG_SINE, fs, available_bits);
        let mut counting = Counting::new(Lines::new(spectrum), coding);
        let gain = within_allowance(&mut counting, &[xmin]).ok_or(())?;
        Ok(counting.into_channel(gain, MAX_BANDS))
    }

    /// Every band's noise is within its allowance, and the bands whose
    /// energy is send no lines; the allowance, not the bits, sets the
    /// length: a larger allowance takes fewer bits.
    #[test]
    fn every_band_is_within_its_allowance() {
        let spectrum = noise(3);
        let bands = GroupBands::long(SamplingFrequency::Hz44100);
        let energy = band_energy(&spectrum);
        let mut bits = Vec::new();
        for factor in [1e-4, 1e-2] {
            let xmin = std::array::from_fn(|sb| match sb {
                // Bands quiet enough to send nothing.
                40 | 41 => energy[sb] * 2.0,
                sb if sb < bands.num_swb() => energy[sb] * factor,
                _ => 0.0,
            });
            let channel = quantize(&spectrum, 20_000, xmin).unwrap();
            for (sb, (energy, distortion)) in measure(&spectrum, &channel).into_iter().enumerate() {
                assert!(
                    distortion <= xmin[sb],
                    "band {sb}: {distortion} > {}",
                    xmin[sb]
                );
                if sb == 40 || sb == 41 {
                    assert_eq!(distortion, energy, "band {sb} sends no lines");
                }
            }
            bits.push(channel.bits(false));
        }
        assert!(bits[0] > bits[1], "{bits:?}");
    }

    /// Where the bands need steps more than 60 apart, which their
    /// scalefactors cannot span, the coarsest are made finer: every band
    /// stays within its allowance, the scalefactors within 60.
    #[test]
    fn steps_too_far_apart_are_made_finer() {
        let spectrum = noise(4);
        let bands = GroupBands::long(SamplingFrequency::Hz44100);
        let energy = band_energy(&spectrum);
        // Some 95 dB between the noise the lowest bands are allowed and
        // that of the highest, and more between their steps.
        let xmin = std::array::from_fn(|sb| match sb {
            sb if sb < 10 => energy[sb] * 1e-7,
            sb if sb < bands.num_swb() => energy[sb] * 0.9,
            _ => 0.0,
        });
        let channel = quantize(&spectrum, 30_000, xmin).unwrap();
        for (sb, (_, distortion)) in measure(&spectrum, &channel).into_iter().enumerate() {
            assert!(distortion <= xmin[sb], "band {sb}");
        }
        let (group, _) = channel.groups().next().unwrap();
        let sent: Vec<u8> = (group.sections.bands_with_scalefactors())
            .filter(|&sb| !group.sections.is_zero_band(sb))
            .map(|sb| group.scalefactors[sb])
            .collect();
        let range = sent.iter().max().unwrap() - sent.iter().min().unwrap();
        assert_eq!(range, MAX_AMPLIFICATION, "{sent:?}");
    }

    /// The bits the outer loop leaves are spent: given ever more bits, a
    /// channel takes nearly all of them, and never fewer than the loops
    /// alone spend, nor more than it is given.
    #[test]
    fn the_bits_the_loops_leave_are_spent() {
        let fs = SamplingFrequency::Hz44100;
        let spectrum = noise(6).map(|x| x * 1e-3);
        let energy = band_energy(&spectrum);
        let xmin = [std::array::from_fn(|sb| energy[sb] * 0.05)];
        let weights = [1.0; FRAME_LINES];
        let mut last = 0;
        for available_bits in (300..3000).step_by(100) {
            let coded = |spend_all| {
                let coding = ChannelCoding {
                    spend_all,
                    ..ChannelCoding::single(Windows::LONG_SINE, fs, available_bits)
                };
                crate::loops::quantize_channel(&spectrum, &weights, &xmin, coding).bits(false)
            };
            let (spent, left) = (coded(true), coded(false));
            assert!(
                left <= spent && spent <= available_bits,
                "{available_bits}: {left}, {spent}"
            );
            assert!(spent + 40 >= available_bits, "{available_bits}: {spent}");
            assert!(spent >= last, "{available_bits}: {last} to {spent}");
            last = spent;
        }
    }

    /// Where the channel does not fit the bits within its allowance, the
    /// counting is given back, for the loops to fit it to them.
    #[test]
    fn a_channel_that_does_not_fit_is_given_back() {
        let spectrum = noise(5);
        let xmin = [1e-6; MAX_BANDS];
        assert!(quantize(&spectrum, 1000, xmin).is_err());
        assert!(quantize(&spectrum, 100_000, xmin).is_ok());
    }
}
