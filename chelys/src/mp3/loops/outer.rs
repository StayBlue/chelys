// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The outer iteration loop, or distortion control loop (clause C.1.5.4.3,
//! figure C.9.b).

use super::inner::{Lines, Step, inner_loop_step, step_size};
use crate::mp3::header::SamplingFrequency;
use crate::mp3::huffman::{Counted, MAX_MAGNITUDE, SpectrumCoding};
use crate::mp3::scalefactor_bands::{self, Band, LONG_BANDS};
use crate::mp3::scalefactors::{PRETAB, Scalefactors, ZeroPart};
use crate::mp3::side_info::SCFSI_BANDS;
use crate::mp3::stereo::IntensityPositions;
use crate::mp3::{BlockType, GRANULE_LINES, QuantizedGranule};

/// The scalefactors of a frame's first granule, and the scfsi bands in which
/// the second granule takes them over (clause C.1.5.4.2.3).
#[derive(Clone, Copy, Debug)]
pub struct Reuse<'a> {
    pub first: &'a Scalefactors,
    pub scfsi: [bool; SCFSI_BANDS],
}

/// The bits a granule may use.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bits {
    /// The most it may use with its scalefactors, `max_bits` (clause
    /// C.1.5.4.2.2).
    pub available: usize,
    /// If given, a normal block within its allowance may leave bits unused
    /// (see [`outer_loop`]). If not, it uses all it can, as Annex C has it.
    pub saving: Option<Saving>,
}

impl Bits {
    /// `available` bits, all of which a granule uses if it can.
    pub fn all(available: usize) -> Self {
        Self {
            available,
            saving: None,
        }
    }
}

/// How a normal block within its allowance may save bits (see
/// [`outer_loop`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Saving {
    /// The fewest bits it must still use.
    pub least: usize,
    /// How much more noise than its allowance, in dB, it may be given.
    pub noise_db: f64,
}

/// A granule quantised by the iteration loops.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    pub granule: QuantizedGranule,
    /// Whether the distortion of every scalefactor band sent is within its
    /// allowance: whether the granule had the bits it needed.
    pub within_allowance: bool,
}

/// The upper scalefactor bands whose distortion decides preemphasis.
const PREEMPHASIS_BANDS: std::ops::Range<usize> = LONG_BANDS - 4..LONG_BANDS;

/// How much more noise than its allowance, in dB, a normal block within its
/// allowance may be given to save bits at a constant or mean bitrate (see
/// [`outer_loop`]).
pub const SAVING_NOISE_DB: f64 = 4.0;

/// Quantises the nonzero lines `xr` of a granule of `block_type` in at most
/// the [`available`](Bits::available) `bits`, amplifying the scalefactor
/// bands whose quantisation noise exceeds the allowed distortion `xmin`.
/// The lines and `xmin` are in the order they are sent; for short blocks each scalefactor band counts
/// once per window (see
/// [`coding_bands`](crate::mp3::scalefactor_bands::coding_bands)).
///
/// Each iteration amplifies the lines by the current scalefactors, runs the
/// inner loop with the bits left after the scalefactors, and measures the
/// distortion `xfsf` of every band (clause C.1.5.4.3.3). It saves that
/// result, then amplifies each band whose distortion exceeds its `xmin`, and
/// raises its `xmin` with it (clause C.1.5.4.3.5). It stops when no band
/// exceeds its `xmin`, when every band has been amplified, or when a
/// scalefactor would pass its limit (clause C.1.5.4.3.6), and returns, of
/// the results it measured, the one whose noise exceeds the allowances least.
///
/// Spec reading: when the loop stops with bands still over their allowance,
/// clause C.1.5.4.3.6 gives as its output the scalefactors saved last, those
/// of the last iteration (clause C.1.5.4.3.7). This deliberately departs from
/// that, and keeps the iteration whose excess is least: the sum, over the
/// bands whose distortion keeps the loop going, of the logarithm of how far
/// each band's distortion exceeds its allowance. Short of bits, amplifying
/// the bands over their allowance takes bits for scalefactors and a coarser
/// step for every band, and the last iteration can be the worst. Every
/// iteration fits the bits, so any of them is a valid granule. Graded with
/// PEAQ (GstPEAQ 0.6.1, black box), 22 tracks of pop music at 44.1 kHz in
/// stereo rose from a mean of -1.59 to -1.21 at 128 kbit/s (from -1.99 to
/// -1.61 for the worst track), from -2.84 to -2.60 at 96 kbit/s, and from
/// -0.25 to -0.23 at 192 kbit/s; `--vbr`, whose frames end within their
/// allowance, did not change. The largest excess, in place of the sum, and
/// the sum weighted by the bands' widths both graded lower.
///
/// Spec reading: figure C.9.b asks whether any band has "more than the allowed
/// distortion" after the amplification. That is read as a question about the
/// distortion measured in the iteration against the allowance it was measured
/// with. Comparing it with the allowances just raised would end the loop as
/// soon as raising them sufficed, and restoring the saved scalefactors would
/// then undo the amplification it was meant to keep.
///
/// Spec gap: clause C.1.5.4.3.4 leaves the choice of preemphasis to the
/// implementation and suggests the rule used here: `preflag` is set if, after
/// the first inner loop, all of the upper four bands exceed their `xmin`.
/// Short blocks never use it (clause 2.4.2.7), nor does the right channel of
/// a frame coded in intensity stereo at the lower sampling frequencies
/// (clause 2.4.3.2 of ISO/IEC 13818-3).
///
/// With `reuse` in at least one scfsi band, the second granule of a frame
/// takes over the first granule's `preflag` and `scalefac_scale`
/// (`ifqstep`), and its scalefactors in those bands, which are then never
/// amplified (clauses C.1.5.4.3.4 and C.1.5.4.3.5). Spec reading: as those
/// bands cannot be amplified, their distortion does not keep the loop going;
/// and as `ifqstep` must stay that of the first granule, the loop stops,
/// rather than switching `scalefac_scale`, when a scalefactor would pass its
/// limit. Should the lines not quantise at all with the scalefactors taken
/// over, the granule is coded without `scfsi`.
///
/// With `intensity`, the granule is the right channel of a frame coded in
/// intensity stereo, whose zero part sends intensity stereo positions in
/// place of scalefactors (see [`IntensityPositions::mark`]). Those bands'
/// lines quantise to zero whatever their scalefactors, so the loop works
/// with scalefactors as usual, and each result is marked before its bits are
/// counted. If the marks take more bits than the scalefactors they replace,
/// the inner loop runs again with that many fewer bits. At the lower
/// sampling frequencies, where no marks are sent, a result with nothing in
/// the band just below the bound is first given a value there (see
/// [`IntensityPositions::fill_below_bound`]); if that takes more bits than
/// there are, the inner loop runs again with that many fewer, and failing
/// that the result is kept marked, which the frame cannot use.
///
/// `scalefac_scale` starts at 0. Clause C.1.5.4.3 switches it to 1 when the
/// scalefactors would otherwise pass their limits (figure C.9.b leaves this
/// out). Spec gap: the clause says only that the scalefactors are then
/// "corrected accordingly"; here they are converted to the coarser step
/// rounding up, so that no band is amplified less than before. Only when they
/// would pass the limits again does the loop stop.
///
/// At MPEG-2.5's 8 kHz, the scalefactors of the highest region stay zero
/// (see [`fixed_at_zero`](scalefactor_bands::fixed_at_zero)), and, like
/// bands taken over with scfsi, their distortion does not keep the loop
/// going. Spec reading: preemphasis is not used there, as `pretab` would
/// amplify those bands, where US 6,185,539 B1 codes them with "a
/// multiplication factor of 1" (claim 1).
///
/// Given a [`Saving`], a normal block that ends within its allowance is not
/// left with the finest step that fits, but given the coarsest step at
/// which no band's distortion exceeds its allowance raised by the saving's
/// [`noise_db`](Saving::noise_db), as long as it still uses at least its
/// [`least`](Saving::least) bits; the bits it leaves are the bit
/// reservoir's, for granules short of bits.
/// Spec reading: clause C.1.5.4.4 has the inner loop take the finest step
/// that fits whatever the distortion, so that a granule spends all the bits
/// it is given, and the reservoir gains only what a granule cannot spend
/// (clause C.1.5.4.2.2). This deliberately departs from that. In a track at
/// 128 kbit/s, three granules in four met their allowance from the first
/// iteration and spent their bits lowering noise the model had already
/// judged inaudible, while the rest went short. The least is what the
/// reservoir has room for: bits saved beyond that would be lost as
/// stuffing. With the noise raised by [`SAVING_NOISE_DB`], graded with PEAQ
/// (GstPEAQ 0.6.1, black box), 22 tracks of pop music at 44.1 kHz in
/// stereo rose from a mean of -1.21 to -0.69 at 128 kbit/s (from -1.61 to
/// -0.98 for the worst track), from -2.60 to -1.88 at 96 kbit/s, and from
/// -0.23 to -0.18 at 192 kbit/s; every one of its measures improved at
/// 128 kbit/s, the frames with audible distortion falling from 34% to 18%.
/// At 128 kbit/s, saving down to the allowance itself graded -0.86, and 3,
/// 5 and 6 dB above it -0.72, -0.70 and -0.73. Without the limit of the
/// reservoir, 192 kbit/s fell to -0.37 (measured with every block type
/// saving bits).
///
/// Start, short and stop blocks keep the finest step that fits. The model
/// allows a granule around an attack noise by the attack's energy, which
/// spreads before it; with them saving bits too, the noise ahead of an
/// attack rose by up to 33 dB (see the test
/// `attacks_switch_to_short_blocks_and_shorten_pre_echo`) for a mean grade
/// of -0.67 at 128 kbit/s.
///
/// The amplified lines are recomputed from `xr` each iteration rather than
/// multiplied incrementally as clause C.1.5.4.3 describes; the result is
/// the same.
///
/// # Panics
///
/// If `xr` cannot be quantised even without scalefactors, which takes lines
/// far beyond the decoder's output range, or the marks of `intensity` take
/// more than the available bits, which takes fewer than its
/// [`silent_bits`](IntensityPositions::silent_bits).
pub fn outer_loop(
    xr: &[f64; GRANULE_LINES],
    xmin: &[f64],
    block_type: BlockType,
    bits: Bits,
    sampling_frequency: SamplingFrequency,
    reuse: Option<Reuse>,
    intensity: Option<&IntensityPositions>,
) -> Outcome {
    let Bits {
        available: available_bits,
        saving,
    } = bits;
    let bands = scalefactor_bands::coding_bands(block_type, sampling_frequency);
    assert_eq!(xmin.len(), bands.len(), "one xmin per scalefactor");
    let fixed = scalefactor_bands::fixed_at_zero(block_type, sampling_frequency);
    let reuse = reuse.filter(|reuse| reuse.scfsi.contains(&true));
    assert!(
        reuse.is_none() || intensity.is_none(),
        "intensity stereo positions are not taken over with scfsi"
    );
    let mut scalefactors = Scalefactors::new(block_type, sampling_frequency.version());
    scalefactors.intensity = intensity.map(|_| ZeroPart::UNMARKED);
    if let Some(reuse) = reuse {
        assert!(!block_type.short_windows(), "short blocks have no scfsi");
        scalefactors.preflag = reuse.first.preflag;
        scalefactors.scalefac_scale = reuse.first.scalefac_scale;
        scalefactors.scfsi = reuse.scfsi;
        for (sb, scalefac) in scalefactors.values_mut().iter_mut().enumerate() {
            if reuse.scfsi[crate::mp3::scalefactors::scfsi_band(sb)] {
                *scalefac = reuse.first.values()[sb];
            }
        }
    }
    // The least excess measured, and the result that had it, whether it was
    // within its allowance, and its scalefactors before any intensity
    // stereo marks.
    let mut best: Option<(f64, (Fitted, bool), Scalefactors)> = None;
    let mut start_gain = None;
    // The lines of the last iteration, whose powers the next takes over.
    let mut lines: Option<Lines> = None;
    let lines_for = |lines: &Option<Lines>, amplified: &[f64; GRANULE_LINES]| match lines {
        Some(lines) => lines.updated(amplified),
        None => Lines::new(amplified),
    };

    loop {
        let amplified = amplify(xr, &scalefactors, &bands);
        let current = lines_for(&lines, &amplified);
        let granule = fit(
            &current,
            &scalefactors,
            block_type,
            available_bits,
            sampling_frequency,
            start_gain,
            intensity,
        );
        lines = Some(current);
        let Some(granule) = granule else {
            if best.is_none() && reuse.is_some() {
                return outer_loop(
                    xr,
                    xmin,
                    block_type,
                    bits,
                    sampling_frequency,
                    None,
                    intensity,
                );
            }
            break;
        };
        start_gain = Some(granule.global_gain());

        let distortion =
            quantisation_distortion(&amplified, granule.values(), granule.global_gain(), &bands);
        let allowed_for = |scalefactors: &Scalefactors| -> Vec<f64> {
            (xmin.iter().enumerate())
                .map(|(sb, xmin)| xmin * scalefactors.amplification(sb).powi(2))
                .collect()
        };
        let measured_against = allowed_for(&scalefactors);
        let excess: f64 = (0..bands.len())
            .filter(|&sb| scalefactors.sent(sb) && !fixed.contains(&sb))
            .map(
                |sb| match (distortion[sb] / measured_against[sb]).max(1.0) {
                    // ln(1) is 0 exactly, which most bands, within their
                    // allowance, need no logarithm for.
                    1.0 => 0.0,
                    ratio => ratio.ln(),
                },
            )
            .sum();
        let violated = excess > 0.0;
        let first = best.is_none();
        if best.as_ref().is_none_or(|(least, ..)| excess < *least) {
            best = Some((excess, (granule, !violated), scalefactors.clone()));
        }

        // C.1.5.4.3.4: preemphasis, which raises the allowed distortion of the
        // preemphasised bands along with their lines.
        if first
            && scalefactors.can_preemphasise()
            && reuse.is_none()
            && fixed.is_empty()
            && PREEMPHASIS_BANDS
                .clone()
                .all(|sb| distortion[sb] > measured_against[sb])
        {
            scalefactors.preflag = true;
        }

        // C.1.5.4.3.5: amplify the bands with more than the allowed distortion.
        let allowed = allowed_for(&scalefactors);
        let sent: Vec<bool> = (0..bands.len()).map(|sb| scalefactors.sent(sb)).collect();
        for (sb, scalefac) in scalefactors.values_mut().iter_mut().enumerate() {
            if sent[sb] && !fixed.contains(&sb) && allowed[sb] - distortion[sb] < 0.0 {
                *scalefac += 1;
            }
        }

        // C.1.5.4.3.6: the conditions for stopping, in the order of figure C.9.b.
        let mut free = (scalefactors.values().iter().enumerate())
            .filter(|(sb, _)| !fixed.contains(sb))
            .map(|(_, &s)| s);
        if free.all(|s| s > 0) {
            break;
        }
        if exceeds_limits(&scalefactors) {
            if scalefactors.scalefac_scale || reuse.is_some() {
                break;
            }
            scalefactors = coarser(&scalefactors);
        }
        if !violated {
            break;
        }
    }
    let (_, (fitted, within_allowance), scalefactors) =
        best.expect("lines without scalefactors always quantise");
    match saving {
        Some(saving) if within_allowance && block_type == BlockType::Normal => {
            let amplified = amplify(xr, &scalefactors, &bands);
            let to_save = ToSave {
                lines: lines_for(&lines, &amplified),
                xmin,
                scalefactors: &scalefactors,
                bands: &bands,
                fixed,
                sampling_frequency,
                intensity,
            };
            let granule = to_save.coarsest(fitted, available_bits, saving);
            Outcome {
                granule,
                within_allowance: true,
            }
        }
        _ => Outcome {
            granule: fitted.granule(block_type, sampling_frequency),
            within_allowance,
        },
    }
}

/// A granule that [`fit`] gives: the inner loop's step, whose Huffman
/// coding is chosen only for the granule that the outer loop keeps, with
/// its scalefactors, or with intensity stereo, whose bits are counted
/// with the marks, the granule made at once.
enum Fitted {
    Step(Step, Scalefactors),
    Granule(QuantizedGranule),
}

impl Fitted {
    fn global_gain(&self) -> u8 {
        match self {
            Self::Step(step, _) => step.global_gain,
            Self::Granule(granule) => granule.global_gain,
        }
    }

    fn values(&self) -> &[i32; GRANULE_LINES] {
        match self {
            Self::Step(step, _) => &step.values,
            Self::Granule(granule) => &granule.values,
        }
    }

    fn scalefactors(&self) -> &Scalefactors {
        match self {
            Self::Step(_, scalefactors) => scalefactors,
            Self::Granule(granule) => &granule.scalefactors,
        }
    }

    fn granule(
        self,
        block_type: BlockType,
        sampling_frequency: SamplingFrequency,
    ) -> QuantizedGranule {
        match self {
            Self::Step(step, scalefactors) => {
                let mut granule = step.granule(block_type, sampling_frequency);
                granule.scalefactors = scalefactors;
                granule
            }
            Self::Granule(granule) => granule,
        }
    }
}

/// A step of the saving search that passes, with its values, its
/// scalefactors marked for intensity stereo, and their counts.
struct Kept {
    global_gain: u8,
    values: [i32; GRANULE_LINES],
    scalefactors: Scalefactors,
    counted: Counted,
}

/// A granule that met its allowance, to quantise with a coarser step.
struct ToSave<'a> {
    /// Its lines amplified by its scalefactors.
    lines: Lines,
    xmin: &'a [f64],
    /// Its scalefactors before any intensity stereo marks.
    scalefactors: &'a Scalefactors,
    bands: &'a [Band],
    fixed: std::ops::Range<usize>,
    sampling_frequency: SamplingFrequency,
    intensity: Option<&'a IntensityPositions>,
}

impl ToSave<'_> {
    /// The granule with the coarsest step, from that of `granule` up, at
    /// which no band's distortion exceeds its allowance raised as `saving`
    /// says, and which takes between its least bits and `available_bits`.
    /// Every band counts, those taken over with scfsi
    /// too, but for those fixed at zero. In the right channel of a frame
    /// coded in intensity stereo, a step that would mark a band where the
    /// granule had none fails too; at the lower sampling frequencies each
    /// step is first given a value in the band just below the bound where
    /// it has none, as [`fit`] gives it. The search steps up from the
    /// granule's step, doubling each stride while the steps pass, then
    /// bisects between the last that passed and the first that failed; it
    /// counts the bits of a step only where it has to. As distortion and
    /// bits follow the step only roughly, a finer step that it skipped may
    /// fail.
    ///
    /// Against a search that stepped up one step at a time and stopped at the
    /// first that failed, graded with PEAQ (GstPEAQ 0.6.1, black box), 22
    /// tracks of pop music at 44.1 kHz in stereo rose from a mean of -1.697
    /// to -1.671 at 96 kbit/s and from -0.585 to -0.577 at 128 kbit/s, and 14
    /// of rock from -1.903 to -1.885 and from -0.756 to -0.751; at 192 kbit/s
    /// both stayed within 0.001. At `--vbr` the streams came out 0.6% smaller
    /// and graded 0.006 lower for pop and 0.005 for rock, and as before once
    /// matched in size. With `--intensity-stereo` at 96 kbit/s, pop held, but
    /// rock fell from a median of -1.345 to -1.361 and its worst track from
    /// -2.514 to -2.547, a loss taken for the rest. Encoding took 3 to 10%
    /// less time.
    fn coarsest(&self, granule: Fitted, available_bits: usize, saving: Saving) -> QuantizedGranule {
        let fs = self.sampling_frequency;
        let block_type = BlockType::Normal;
        let amplified = self.lines.xr();
        let raised = 10f64.powf(saving.noise_db / 10.0);
        let allowed: Vec<f64> = (self.xmin.iter().enumerate())
            .map(|(sb, xmin)| raised * xmin * self.scalefactors.amplification(sb).powi(2))
            .collect();
        // The scalefactors, and so their bits, change only where intensity
        // stereo marks them.
        let unmarked_part2 = part2_length(self.scalefactors);
        // A step's values and scalefactors, marked for intensity stereo, if
        // no band's distortion exceeds its allowance and it marks a band
        // only if `marked`, that is if the last step that passed did.
        let quantised = |global_gain: u8, marked: bool| {
            let mut values = self.lines.quantize(global_gain)?;
            if let Some(intensity) = self.intensity {
                intensity.fill_below_bound(&mut values, amplified);
            }
            // The distortion first, as it needs no Huffman coding, band by
            // band until one exceeds its allowance.
            let distortion = band_distortion(amplified, &values, global_gain);
            if (self.bands.iter().enumerate())
                .any(|(sb, band)| !self.fixed.contains(&sb) && distortion(band) > allowed[sb])
            {
                return None;
            }
            let mut scalefactors = self.scalefactors.clone();
            if let Some(intensity) = self.intensity {
                intensity.mark(&mut scalefactors, &values);
                if scalefactors.marks_any() && !marked {
                    return None;
                }
            }
            Some((values, scalefactors))
        };
        // The step with those values, if their bits are between the least
        // and the available bits.
        let counted =
            |global_gain: u8, (values, scalefactors): ([i32; GRANULE_LINES], Scalefactors)| {
                let part2 = match self.intensity {
                    Some(_) => part2_length(&scalefactors),
                    None => unmarked_part2,
                };
                let most = available_bits.checked_sub(part2)?;
                let least = saving.least.saturating_sub(part2);
                let counted = SpectrumCoding::count(&values, block_type, self.bands, fs);
                counted.bits_between(least, most).then_some(Kept {
                    global_gain,
                    values,
                    scalefactors,
                    counted,
                })
            };
        // Coarser steps, each stride twice the last while their distortion
        // and marks pass, then bisecting between the last that passed and the
        // first that failed; then the bits of the step found, and if they
        // fail, a bisection between it and the granule's step for the last
        // whose distortion and bits pass, rather than counting the bits of
        // every step, for speed. This assumes that coarser steps never need
        // more bits, which the Huffman tables do not quite guarantee; where
        // they do, the search may keep a step coarser than one whose bits
        // fail. With intensity stereo, the bits of the scalefactors change
        // with the bands marked, which need not follow the step, and the bits
        // of every step whose distortion and marks pass are counted too.
        // `check` says whether a step passes, given whether the last step
        // that passed marked a band; with intensity stereo it keeps the
        // step, its bits counted.
        let mut marked = granule.scalefactors().marks_any();
        let mut kept = None;
        let check = |global_gain: u8, marked: &mut bool, kept: &mut Option<Kept>| {
            let Some(step) = quantised(global_gain, *marked) else {
                return false;
            };
            let now_marked = step.1.marks_any();
            if self.intensity.is_some() {
                let Some(step) = counted(global_gain, step) else {
                    return false;
                };
                *kept = Some(step);
            }
            *marked = now_marked;
            true
        };
        let start = granule.global_gain();
        let (mut passed, mut distance, mut failed) = (start, 1u8, None);
        while passed < u8::MAX {
            let next = passed.saturating_add(distance);
            if check(next, &mut marked, &mut kept) {
                passed = next;
                distance = distance.saturating_mul(2);
            } else {
                failed = Some(next);
                break;
            }
        }
        while let Some(over) = failed.filter(|&over| over - passed > 1) {
            let middle = passed + (over - passed) / 2;
            if check(middle, &mut marked, &mut kept) {
                passed = middle;
            } else {
                failed = Some(middle);
            }
        }
        let last = (self.intensity.is_none() && passed > start).then_some(passed);
        let kept = kept.or_else(|| {
            let coarsest = last?;
            let step = quantised(coarsest, marked).expect("the step passed");
            if let Some(kept) = counted(coarsest, step) {
                return Some(kept);
            }
            // The steps between were not all tried, so each is quantised
            // and measured again.
            let (mut passed, mut failed, mut kept) = (start, coarsest, None);
            while failed - passed > 1 {
                let middle = passed + (failed - passed) / 2;
                match quantised(middle, marked).and_then(|step| counted(middle, step)) {
                    Some(step) => {
                        passed = middle;
                        kept = Some(step);
                    }
                    None => failed = middle,
                }
            }
            kept
        });
        let Some(Kept {
            global_gain,
            values,
            scalefactors,
            counted,
        }) = kept
        else {
            return granule.granule(block_type, fs);
        };
        let (coding, _) = counted.choose();
        let mut coarsest =
            QuantizedGranule::with_coding(global_gain, values, block_type, fs, coding);
        coarsest.scalefactors = scalefactors;
        debug_assert!((saving.least..=available_bits).contains(&coarsest.bit_len(fs)));
        coarsest
    }
}

/// The bits of `scalefactors`, `part2_length`, with the `scalefac_compress`
/// that sends them in the fewest.
fn part2_length(scalefactors: &Scalefactors) -> usize {
    let compress = (scalefactors.compress()).expect("scalefactors are kept within their limits");
    scalefactors.part2_length(compress)
}

/// The inner loop for the lines `amplified` by `scalefactors`, in at most
/// `available_bits` with the scalefactors, marked for intensity stereo if
/// `intensity` is given. `None` if they do not fit.
fn fit(
    lines: &Lines,
    scalefactors: &Scalefactors,
    block_type: BlockType,
    available_bits: usize,
    sampling_frequency: SamplingFrequency,
    start_gain: Option<u8>,
    intensity: Option<&IntensityPositions>,
) -> Option<Fitted> {
    let compress = scalefactors
        .compress()
        .expect("scalefactors are kept within their limits");
    let amplified = lines.xr();
    let mut huffman_bits = available_bits.checked_sub(scalefactors.part2_length(compress))?;
    loop {
        let step = inner_loop_step(
            lines,
            huffman_bits,
            block_type,
            sampling_frequency,
            start_gain,
        )?;
        let Some(intensity) = intensity else {
            return Some(Fitted::Step(step, scalefactors.clone()));
        };
        let mut granule = step.granule(block_type, sampling_frequency);
        let mut values = granule.values;
        if intensity.fill_below_bound(&mut values, amplified) {
            let mut filled =
                QuantizedGranule::new(granule.global_gain, values, block_type, sampling_frequency);
            filled.scalefactors = scalefactors.clone();
            intensity.clear_empty_bands(&mut filled.scalefactors, &filled.values);
            intensity.mark(&mut filled.scalefactors, &filled.values);
            let bits = filled.bit_len(sampling_frequency);
            if bits <= available_bits {
                return Some(Fitted::Granule(filled));
            }
            // Fewer bits for the inner loop if that leaves room for the
            // value; failing that, the granule as it is, marked.
            if let Some(fewer) = huffman_bits.checked_sub(bits - available_bits) {
                huffman_bits = fewer;
                continue;
            }
        }
        granule.scalefactors = scalefactors.clone();
        intensity.mark(&mut granule.scalefactors, &granule.values);
        let bits = granule.bit_len(sampling_frequency);
        if bits <= available_bits {
            return Some(Fitted::Granule(granule));
        }
        huffman_bits = huffman_bits.checked_sub(bits - available_bits)?;
    }
}

/// How far a quantised granule's noise goes beyond its allowance: the
/// largest ratio, over its scalefactor bands, of the distortion `xfsf` to
/// the allowed distortion `xmin`, both as the outer loop measures them, of
/// the lines `xr` the granule was quantised from. At most 1 if every band is
/// within its allowance.
///
/// Unlike the outer loop's own test, every band counts: those whose
/// scalefactors are taken over with scfsi, or fixed at zero, are noise as
/// much as the rest. A band whose lines are all zero allows no distortion
/// and has none. Spec gap: the lines above the last scalefactor band have
/// no allowed distortion (clause C.1.5.4.1 gives `xmin` for the bands
/// only), and are not measured.
pub fn excess(
    xr: &[f64; GRANULE_LINES],
    xmin: &[f64],
    granule: &QuantizedGranule,
    sampling_frequency: SamplingFrequency,
) -> f64 {
    let bands = scalefactor_bands::coding_bands(granule.block_type, sampling_frequency);
    assert_eq!(xmin.len(), bands.len(), "one xmin per scalefactor");
    let amplified = amplify(xr, &granule.scalefactors, &bands);
    let distortion = distortion(&amplified, granule, &bands);
    (distortion.iter().zip(xmin).enumerate())
        .map(|(sb, (&distortion, &xmin))| {
            let allowed = xmin * granule.scalefactors.amplification(sb).powi(2);
            match (distortion, allowed) {
                (0.0, _) => 0.0,
                (_, 0.0) => f64::INFINITY,
                _ => distortion / allowed,
            }
        })
        .fold(0.0, f64::max)
}

/// `xr` with each scalefactor band multiplied by its amplification. Lines
/// above the last band have no scalefactor and stay as they are.
fn amplify(
    xr: &[f64; GRANULE_LINES],
    scalefactors: &Scalefactors,
    bands: &[Band],
) -> [f64; GRANULE_LINES] {
    let mut amplified = *xr;
    for (sb, band) in bands.iter().enumerate() {
        let factor = scalefactors.amplification(sb);
        for x in &mut amplified[band.start..band.end()] {
            *x *= factor;
        }
    }
    amplified
}

/// `xfsf(sb)` of clause C.1.5.4.3.3: the mean squared difference between
/// the amplified lines and their requantised values in each band.
fn distortion(
    amplified: &[f64; GRANULE_LINES],
    granule: &QuantizedGranule,
    bands: &[Band],
) -> Vec<f64> {
    quantisation_distortion(amplified, &granule.values, granule.global_gain, bands)
}

/// [`distortion`] for the quantised `values` at `global_gain`.
fn quantisation_distortion(
    amplified: &[f64; GRANULE_LINES],
    values: &[i32; GRANULE_LINES],
    global_gain: u8,
    bands: &[Band],
) -> Vec<f64> {
    bands
        .iter()
        .map(band_distortion(amplified, values, global_gain))
        .collect()
}

/// [`quantisation_distortion`] of one band at a time.
fn band_distortion<'a>(
    amplified: &'a [f64; GRANULE_LINES],
    values: &'a [i32; GRANULE_LINES],
    global_gain: u8,
) -> impl Fn(&Band) -> f64 + 'a {
    let step = step_size(global_gain);
    let powers = four_thirds_powers();
    move |band| {
        let lines = band.start..band.end();
        let sum: f64 = (amplified[lines.clone()].iter().zip(&values[lines]))
            .map(|(x, value)| {
                let requantised = powers[value.unsigned_abs() as usize] * step;
                (x.abs() - requantised).powi(2)
            })
            .sum();
        sum / band.width as f64
    }
}

/// `magnitude^(4/3)` for every magnitude the Huffman tables can code, as
/// the distortion of each band takes it for every line.
fn four_thirds_powers() -> &'static [f64] {
    static TABLE: std::sync::OnceLock<Vec<f64>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        (0..=MAX_MAGNITUDE)
            .map(|magnitude| f64::from(magnitude).powf(4.0 / 3.0))
            .collect()
    })
}

fn exceeds_limits(scalefactors: &Scalefactors) -> bool {
    (scalefactors.values().iter().enumerate())
        .any(|(sb, &s)| scalefactors.sent(sb) && s > scalefactors.limit(sb))
}

/// The scalefactors with `scalefac_scale` set: each band gets the smallest
/// scalefactor that amplifies it at least as much as before. A band
/// amplified by `2^((s + p) / 2)` with `scalefac_scale` 0, where `p` is its
/// preemphasis, needs `ceil((s - p) / 2)` steps of 2, and none if
/// preemphasis alone already amplifies it enough.
fn coarser(scalefactors: &Scalefactors) -> Scalefactors {
    let mut coarser = scalefactors.clone();
    coarser.scalefac_scale = true;
    for (sb, scalefac) in coarser.values_mut().iter_mut().enumerate() {
        let pretab = if scalefactors.preflag { PRETAB[sb] } else { 0 };
        *scalefac = scalefac.saturating_sub(pretab).div_ceil(2);
    }
    coarser
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mp3::loops::{inner_loop, quantize};

    const FS: SamplingFrequency = SamplingFrequency::Hz44100;

    /// A spectrum falling from low to high frequencies, with every line
    /// nonzero.
    fn spectrum() -> [f64; GRANULE_LINES] {
        std::array::from_fn(|i| {
            let level = 0.05 * (-(i as f64) / 150.0).exp();
            if i % 2 == 0 { level } else { -0.7 * level }
        })
    }

    fn total_bits(granule: &QuantizedGranule) -> usize {
        granule.bit_len(FS)
    }

    /// With a generous allowance nothing needs amplifying, and the result is
    /// the inner loop's alone.
    #[test]
    fn stops_at_once_when_every_band_is_within_its_allowance() {
        let xr = spectrum();
        let granule = outer_loop(
            &xr,
            &[f64::INFINITY; LONG_BANDS],
            BlockType::Normal,
            Bits::all(1500),
            FS,
            None,
            None,
        )
        .granule;
        assert_eq!(granule.scalefactors, Scalefactors::default());
        let inner = inner_loop(&xr, 1500, BlockType::Normal, FS, None).unwrap();
        assert_eq!(granule.global_gain, inner.global_gain);
    }

    /// Bands asked for less noise than an even spread gives are amplified,
    /// and the result still fits.
    #[test]
    fn amplifies_the_bands_that_need_less_noise() {
        let xr = spectrum();
        let bands = scalefactor_bands::long_bands(FS);
        let plain = inner_loop(&xr, 1500, BlockType::Normal, FS, None).unwrap();
        let noise = distortion(&xr, &plain, bands);
        // Bands 4 to 7 must be 10 dB quieter; the rest may be as they are.
        let xmin: [f64; LONG_BANDS] = std::array::from_fn(|sb| {
            if (4..8).contains(&sb) {
                noise[sb] / 10.0
            } else {
                f64::INFINITY
            }
        });
        let granule = outer_loop(
            &xr,
            &xmin,
            BlockType::Normal,
            Bits::all(1500),
            FS,
            None,
            None,
        )
        .granule;
        assert!(total_bits(&granule) <= 1500);
        for sb in 4..8 {
            assert!(granule.scalefactors.values()[sb] > 0, "band {sb}");
        }
        for sb in (0..4).chain(8..LONG_BANDS) {
            assert_eq!(granule.scalefactors.values()[sb], 0, "band {sb}");
        }
        assert!(!granule.scalefactors.preflag);
    }

    /// When no allowance can be met, the loop still ends within the limits of
    /// the scalefactors and the bits.
    #[test]
    fn ends_within_the_limits_when_nothing_can_be_met() {
        let xr = spectrum();
        for bits in [60, 300, 1500, 4000] {
            let granule = outer_loop(
                &xr,
                &[0.0; LONG_BANDS],
                BlockType::Normal,
                Bits::all(bits),
                FS,
                None,
                None,
            )
            .granule;
            assert!(total_bits(&granule) <= bits, "{bits}");
            assert!(granule.scalefactors.compress().is_some(), "{bits}");
        }
    }

    /// Preemphasis is switched on when all of the upper four bands are too
    /// noisy after the first inner loop.
    #[test]
    fn preemphasis_follows_the_upper_bands() {
        let xr = spectrum();
        let bands = scalefactor_bands::long_bands(FS);
        let plain = inner_loop(&xr, 1500, BlockType::Normal, FS, None).unwrap();
        let noise = distortion(&xr, &plain, bands);
        let xmin: [f64; LONG_BANDS] = std::array::from_fn(|sb| {
            if PREEMPHASIS_BANDS.contains(&sb) {
                noise[sb] / 4.0
            } else {
                f64::INFINITY
            }
        });
        assert!(
            outer_loop(
                &xr,
                &xmin,
                BlockType::Normal,
                Bits::all(1500),
                FS,
                None,
                None
            )
            .granule
            .scalefactors
            .preflag
        );
        let mut xmin = xmin;
        xmin[LONG_BANDS - 1] = f64::INFINITY;
        assert!(
            !outer_loop(
                &xr,
                &xmin,
                BlockType::Normal,
                Bits::all(1500),
                FS,
                None,
                None
            )
            .granule
            .scalefactors
            .preflag
        );
    }

    /// The second granule takes over the first granule's scalefactors in
    /// the scfsi bands and its `preflag` and `scalefac_scale`, and never
    /// amplifies those bands, however far their distortion exceeds `xmin`.
    /// The loop still ends, within its bits.
    #[test]
    fn scfsi_bands_keep_the_first_granules_scalefactors() {
        let xr = spectrum();
        let mut first = Scalefactors {
            preflag: true,
            ..Scalefactors::default()
        };
        for (sb, scalefac) in first.values_mut().iter_mut().enumerate() {
            *scalefac = (sb % 4) as u8;
        }
        let scfsi = [true, false, true, false];
        for bits in [300, 1500] {
            let reuse = Reuse {
                first: &first,
                scfsi,
            };
            let granule = outer_loop(
                &xr,
                &[0.0; LONG_BANDS],
                BlockType::Normal,
                Bits::all(bits),
                FS,
                Some(reuse),
                None,
            )
            .granule;
            let scalefactors = &granule.scalefactors;
            assert_eq!(scalefactors.scfsi, scfsi);
            assert!(scalefactors.preflag && !scalefactors.scalefac_scale);
            for sb in (0..6).chain(11..16) {
                assert_eq!(scalefactors.values()[sb], first.values()[sb], "band {sb}");
            }
            assert!((6..11).any(|sb| scalefactors.values()[sb] != first.values()[sb]));
            assert!(total_bits(&granule) <= bits, "{bits}");
        }
    }

    /// Without any scfsi band, reuse changes nothing.
    #[test]
    fn reuse_without_scfsi_is_ignored() {
        let xr = spectrum();
        let first = Scalefactors {
            preflag: true,
            scalefac_scale: true,
            ..Scalefactors::default()
        };
        let reuse = Reuse {
            first: &first,
            scfsi: [false; SCFSI_BANDS],
        };
        let xmin = [1e-9; LONG_BANDS];
        assert_eq!(
            outer_loop(
                &xr,
                &xmin,
                BlockType::Normal,
                Bits::all(1500),
                FS,
                Some(reuse),
                None
            ),
            outer_loop(
                &xr,
                &xmin,
                BlockType::Normal,
                Bits::all(1500),
                FS,
                None,
                None
            )
        );
    }

    /// The right channel of a frame in intensity stereo sends its marks in
    /// place of the scalefactors of its zero part, within its bits, even
    /// when the marks need wider fields than the scalefactors did.
    #[test]
    fn intensity_marks_fit_in_the_bits() {
        let bands = scalefactor_bands::long_bands(FS);
        let mut xr = spectrum();
        xr[bands[12].start..].fill(0.0);
        let positions = IntensityPositions::new(&spectrum(), &xr, BlockType::Normal, 12, FS);
        for bits in [80, 150, 300, 1500] {
            let outcome = outer_loop(
                &xr,
                &[0.0; LONG_BANDS],
                BlockType::Normal,
                Bits::all(bits),
                FS,
                None,
                Some(&positions),
            );
            let granule = &outcome.granule;
            assert!(total_bits(granule) <= bits, "{bits}");
            assert!(!outcome.within_allowance, "{bits}");
            let mut marked = granule.scalefactors.clone();
            positions.mark(&mut marked, &granule.values);
            assert_eq!(marked, granule.scalefactors, "{bits}");
            assert_ne!(granule.scalefactors.values()[20], 0, "{bits}");
        }
        let outcome = outer_loop(
            &xr,
            &[f64::INFINITY; LONG_BANDS],
            BlockType::Normal,
            Bits::all(1500),
            FS,
            None,
            Some(&positions),
        );
        assert!(outcome.within_allowance);
    }

    /// A normal block within its allowance, given a [`Saving`], takes a
    /// coarser step and fewer bits, with no band's distortion beyond its
    /// allowance raised by the saving's noise; but no fewer bits than its
    /// least, and a short block keeps the finest step that fits.
    #[test]
    fn a_granule_within_its_allowance_saves_bits() {
        let xr = spectrum();
        let bands = scalefactor_bands::long_bands(FS);
        let plain = inner_loop(&xr, 1500, BlockType::Normal, FS, None).unwrap();
        // The allowance admits the noise of the finest step that fits, and
        // raised, that of four steps coarser, and of those between.
        let raised = 10f64.powf(SAVING_NOISE_DB / 10.0);
        let mut xmin = distortion(&xr, &plain, bands);
        for gain in plain.global_gain + 1..=plain.global_gain + 4 {
            let values = quantize(&xr, gain).unwrap();
            let coarser = QuantizedGranule::new(gain, values, BlockType::Normal, FS);
            for (xmin, noise) in xmin.iter_mut().zip(distortion(&xr, &coarser, bands)) {
                *xmin = f64::max(*xmin, 1.000_001 * noise / raised);
            }
        }
        let saving = |least| {
            Some(Saving {
                least,
                noise_db: SAVING_NOISE_DB,
            })
        };
        let quantize = |block_type, saving| {
            let bits = Bits {
                available: 1500,
                saving,
            };
            outer_loop(&xr, &xmin, block_type, bits, FS, None, None)
        };
        let spent = quantize(BlockType::Normal, None);
        assert!(spent.within_allowance);
        assert_eq!(spent.granule.global_gain, plain.global_gain);

        let saved = quantize(BlockType::Normal, saving(0));
        assert!(saved.within_allowance);
        assert!(saved.granule.global_gain >= plain.global_gain + 4);
        assert!(total_bits(&saved.granule) < total_bits(&spent.granule));
        let noise = distortion(&xr, &saved.granule, bands);
        for sb in 0..LONG_BANDS {
            assert!(noise[sb] <= raised * xmin[sb], "band {sb}");
        }

        // Without raising the noise, the allowance itself is kept.
        let unraised = Saving {
            least: 0,
            noise_db: 0.0,
        };
        let within = quantize(BlockType::Normal, Some(unraised)).granule;
        assert!(within.global_gain < saved.granule.global_gain);
        assert!(total_bits(&within) <= total_bits(&spent.granule));
        let noise = distortion(&xr, &within, bands);
        for sb in 0..LONG_BANDS {
            assert!(noise[sb] <= xmin[sb], "band {sb}");
        }

        let least = total_bits(&spent.granule) - 1;
        let limited = quantize(BlockType::Normal, saving(least)).granule;
        assert!(total_bits(&limited) >= least);

        let short_xmin =
            vec![f64::INFINITY; scalefactor_bands::coding_bands(BlockType::Short, FS).len()];
        let short = |saving| {
            let bits = Bits {
                available: 1500,
                saving,
            };
            outer_loop(&xr, &short_xmin, BlockType::Short, bits, FS, None, None).granule
        };
        assert_eq!(short(saving(0)), short(None));
    }

    /// The conversion to `scalefac_scale` never amplifies a band less.
    #[test]
    fn coarser_scalefactors_amplify_at_least_as_much() {
        for preflag in [false, true] {
            let mut scalefactors = Scalefactors {
                preflag,
                ..Scalefactors::default()
            };
            for sb in 0..LONG_BANDS {
                scalefactors.values_mut()[sb] = (sb % 17) as u8;
            }
            let coarser = coarser(&scalefactors);
            for sb in 0..LONG_BANDS {
                let (before, after) = (scalefactors.amplification(sb), coarser.amplification(sb));
                assert!(after >= before, "band {sb}");
                // Rounding up costs at most one step of the finer scale.
                let finer_step = 2f64.sqrt() * (1.0 + 1e-12);
                assert!(
                    after <= before * finer_step || coarser.values()[sb] == 0,
                    "band {sb}"
                );
            }
        }
    }

    #[test]
    fn distortion_is_the_mean_squared_requantisation_error() {
        let bands = scalefactor_bands::long_bands(FS);
        let mut xr = [0.0; GRANULE_LINES];
        xr[0] = 1.5;
        let mut values = [0; GRANULE_LINES];
        values[0] = 1;
        let granule = QuantizedGranule::new(210, values, BlockType::Normal, FS);
        let noise = distortion(&xr, &granule, bands);
        assert!((noise[0] - 0.25 / bands[0].width as f64).abs() < 1e-12);
    }
}
