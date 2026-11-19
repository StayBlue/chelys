// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The block switching decision (clause C.1.4, step 13), the delay of the
//! model's output by one frame that lets it be made (clause C.1.1), and the
//! rest of each frame's windows: the grouping of its short windows and its
//! window shape.

use super::middle_side::MidSideModel;
use super::{Analysis, Model};
use crate::sampling_frequency::SamplingFrequency;
use crate::stream::FRAME_SAMPLES;
use crate::window::{Grouping, SHORT_WINDOWS, WindowSequence, WindowShape};

/// `switch_pe`: a frame has an attack, and is coded with short blocks, when
/// the perceptual entropy of one of its short blocks rises by more than
/// this above the most of the eight short blocks before it (step 13, as
/// departed from below).
///
/// Spec gap: step 13 calls `switch_pe` "a implementation dependend
/// constant", which it compares with the perceptual entropy of the long
/// block. This deliberately departs from that comparison, and compares it
/// with the largest rise of the entropy of the short blocks (step 12), each
/// above the most of the eight before it, those of the frame before among
/// them (see [`Analysis::entropy_rise`]); step 15's entropy is left as
/// printed.
///
/// The entropy is taken as printed (see [`Model::analyze`]), and its
/// partitions below their thresholds in quiet subtract from it, by an
/// amount that depends on the signal and the sampling frequency far more
/// than an attack adds: silence has -9535 at 44.1 kHz and pop music a
/// median of about -1100, but the frame that holds a one-sample impulse of
/// 0.3 over a tone of 0.002 has -868. No constant for the long block's
/// entropy tells attacks from steady music, and a rise from one short
/// block to the next, 128 samples later, does, as both lie over the same
/// background. The criteria were compared by the onsets they catch: of
/// one-sample impulses of 0.3 and 0.9, clicks and bursts of noise and drum
/// hits over a quiet tone, and bursts over a louder tone and noise, each at
/// 32 places in the frame, at every sampling frequency, an onset caught
/// where a frame whose short windows span it is coded short; and by the
/// share of frames coded short in two tracks each of pop and rock music,
/// resampled to every sampling frequency.
///
/// - The long block's entropy, with a constant for each sampling frequency
///   set so that 2% of the frames of pop music are short: 51 to 104
///   onsets of 192 caught at 8 to 24 kHz, 134 at 32 kHz and 154 at
///   44.1 kHz, no impulse of 0.3 at any of them; 188 to 192 at 48 to
///   96 kHz, but with 17 to 21% of the frames of rock music short.
/// - The long block's entropy over its partitions below 20 kHz alone, so
///   set: 51 to 151, no impulse of 0.3 at any sampling frequency.
/// - The rise of each short block's entropy from the one before it, above
///   200 at every sampling frequency: every onset over a quiet tone
///   caught, at every place and sampling frequency, and 0 to 16 of the 32
///   bursts over the louder background; 0 to 3.2% of the frames of pop
///   music short, and 0 to 1.5% of rock. Above 150, 16 to 24 of those
///   bursts are caught, but up to 11% of the frames of pop music at 8 to
///   16 kHz are short.
///
/// The rise from the one block before missed attacks over a loud
/// background, which temporal noise shaping then shaped in long windows:
/// with it off in long windows, 8 tracks of pop music at 44.1 kHz in
/// stereo at 128 kbit/s graded -1.367 with PEAQ (GstPEAQ 0.6.1, black
/// box), against -1.087, the noise in the 1024 samples before the onsets
/// of the frames whose filters' prediction gain exceeded 3 being 1.4 dB
/// louder. Nor could it be lowered: the entropy of the short blocks of a
/// steady tone of 80 Hz at half of full scale rises by up to 119 from one
/// block to the next, as the tone's phase moves through the block, and
/// that of a sweep from 20 Hz to 20 kHz in 3 s by up to 161, though
/// neither has an attack. Above the most of the eight blocks before,
/// which span a period of either, the entropy of tones of 20 Hz to 16 kHz
/// rises by at most 6, and the sweep's by 47. With bursts of noise that
/// die away in 5 ms over a tone of 0.1 and noise, at 31 to 37 places, at
/// about 6, 11 and 16 dB above its level, a rise above 200 caught 0, 0 and
/// 0 to 14 of the bursts at 8, 16, 44.1 and 48 kHz, above 100 2 to 26, 18
/// to 34 and 24 to 37, and above 70 19 to 35, 25 to 37 and 28 to 37; none
/// of the three switched a frame of steady noise, or of a tone in noise.
///
/// The constant is set from the noise of the frames that each value codes
/// short and the next value up does not, in the 8 tracks of pop music: it
/// is to fall more in the 384 samples before their short windows than it
/// rises within them, at every bitrate. Above 100, 3.2% of the frames are
/// coded short that the rise from the one block before, above 200, coded
/// long, and their noise, at 64, 128 and 192 kbit/s, is 2.3, 3.4 and 3.7
/// dB less before their short windows, and 0.6 less, 0.9 more and 0.8
/// less within them. Above 70, another 2.8% are,
/// whose noise is 1.1, 1.5 and 1.7 dB less before, and 0.1 less, 1.8 more
/// and alike within, as of a gradual rise of the signal rather than an
/// attack. Above 100, the noise in the 1024 samples before the onsets of
/// the frames shaped as above is 0.9, 0.6 and 1.1 dB less; 0.5 to 6.7% of
/// the frames of the pop music are short, 8.1 to 24.6% at 8 kHz and 2.1 to
/// 24.5% at 16 kHz.
///
/// Graded with PEAQ, the 8 tracks gave means of -3.543, -1.002 and -0.237
/// at 64, 128 and 192 kbit/s, against -3.584, -1.087 and -0.283 with the
/// rise from the one block before above 200; 3 tracks of rock music and
/// their 3 windows of 3 s with the most frames coded short, -3.563, -0.770
/// and -0.110, against -3.578, -0.801 and -0.134; and 30 s of each pop
/// track, resampled and graded against the excerpt resampled alike, -3.795
/// against -3.826 at 8 kHz and 16 kbit/s, -2.552 against -2.644 at 16 kHz
/// and 32 kbit/s, and -1.019 against -1.063 at 48 kHz and 128 kbit/s. Of
/// the rise of the pop music's grades, the measures of noise give all at
/// 192 kbit/s, four fifths at 128 kbit/s, where the bandwidth measures
/// count the treble kept above 16 kHz, and more than all at 64 kbit/s,
/// where they count treble lost.
const SWITCH_PE: f64 = 100.0;

/// Two short windows of a frame share a group while the energy of each,
/// in its short FFT, is within this factor of every other's in the group.
const GROUPING_RATIO: f64 = 10.0;

/// Whether the frame analyzed as `analysis` has an attack (step 13).
pub fn attack(analysis: &Analysis) -> bool {
    analysis.entropy_rise > SWITCH_PE
}

/// The model of one channel, with its output delayed by a frame, so that a
/// frame's window sequence is chosen knowing whether the frame after it
/// has an attack: "the long block before the short blocks has to be
/// 'patched' to a start block type" (clause C.1.1).
#[derive(Debug)]
pub struct BlockSwitching {
    model: Model,
    /// The analysis of the frame whose windows are chosen next, once
    /// the model has analyzed the frame after it.
    current: Box<Analysis>,
    /// The analysis of the frame the model analyzed last.
    ahead: Box<Analysis>,
    /// Whether the model has analyzed a frame.
    primed: bool,
    /// The window sequence of the frame before the current one, `None`
    /// before the first.
    last: Option<WindowSequence>,
}

/// Whether a frame and the one after it have attacks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Attacks {
    pub this: bool,
    pub next: bool,
}

impl BlockSwitching {
    pub fn new(sampling_frequency: SamplingFrequency) -> Self {
        Self {
            model: Model::new(sampling_frequency),
            current: Box::new(Analysis::new(sampling_frequency)),
            ahead: Box::new(Analysis::new(sampling_frequency)),
            primed: false,
            last: None,
        }
    }

    /// Analyzes the next block of the channel, at full scale 1.0, and gives
    /// whether the frame before it, whose windows are to be chosen next,
    /// and this one have attacks; or `None` for the first block, before
    /// which there is no frame to choose windows for.
    pub fn look_ahead(&mut self, block: &[f64; FRAME_SAMPLES]) -> Option<Attacks> {
        self.begin(block);
        self.end()
    }

    /// [`look_ahead`](Self::look_ahead) for the two channels of a pair,
    /// `[left, right]`, with the model of its middle and side channels,
    /// which adjusts their thresholds (see [`MidSideModel::look_ahead`]).
    pub fn look_ahead_pair(
        [left, right]: [&mut Self; 2],
        mid_side: &mut MidSideModel,
        [left_block, right_block]: [&[f64; FRAME_SAMPLES]; 2],
    ) -> Option<[Attacks; 2]> {
        left.begin(left_block);
        right.begin(right_block);
        mid_side.look_ahead(&mut left.model, &mut right.model);
        let left = left.end();
        let right = right.end();
        left.zip(right).map(|(left, right)| [left, right])
    }

    /// Takes the next block through steps 1 to 10 of the model.
    fn begin(&mut self, block: &[f64; FRAME_SAMPLES]) {
        if self.primed {
            std::mem::swap(&mut self.current, &mut self.ahead);
        }
        self.model.blocks(block);
    }

    /// Finishes the analysis of the block [`begin`](Self::begin) took.
    fn end(&mut self) -> Option<Attacks> {
        self.model.finish(&mut self.ahead);
        let primed = std::mem::replace(&mut self.primed, true);
        primed.then(|| Attacks {
            this: attack(&self.current),
            next: attack(&self.ahead),
        })
    }

    /// Whether the model has analyzed a block, so that the next one
    /// analyzed gives the attacks of the frame before it.
    pub fn is_primed(&self) -> bool {
        self.primed
    }

    /// The analysis of the frame whose windows are chosen next.
    pub fn current(&self) -> &Analysis {
        &self.current
    }

    /// Chooses the window sequence of the current frame, given `attacks`,
    /// whether it and the frame after it have attacks, its short windows,
    /// if it has any, each a group of its own, for [`grouping`] to group;
    /// or, where the frame cannot `afford_short` windows, the long sequence
    /// that may take their place: a LONG_STOP_SEQUENCE after a sequence
    /// that ends short, and an ONLY_LONG_SEQUENCE else, so that the
    /// transitions stay meaningful.
    pub fn decide(&mut self, attacks: Attacks, afford_short: bool) -> WindowSequence {
        let sequence = match next_sequence(self.last, attacks) {
            WindowSequence::EightShort { .. } if !afford_short => {
                if ends_short(self.last) {
                    WindowSequence::LongStop
                } else {
                    WindowSequence::OnlyLong
                }
            }
            sequence => sequence,
        };
        self.last = Some(sequence);
        sequence
    }
}

/// Whether `sequence`'s window ends with short windows' slope, so that the
/// next must start with one (see [`WindowSequence::follows`]).
fn ends_short(sequence: Option<WindowSequence>) -> bool {
    matches!(
        sequence,
        Some(WindowSequence::LongStart | WindowSequence::EightShort { .. })
    )
}

/// The window sequence of a frame, given that of the frame before it, if
/// any, and whether it and the frame after it have attacks: short where the
/// frame has an attack, with a LONG_START_SEQUENCE before and a
/// LONG_STOP_SEQUENCE after, so that every transition is one of the
/// meaningful ones of clause 15.3.1 (see [`WindowSequence::follows`]). The
/// short windows are not grouped.
///
/// Spec reading: step 13 reads, after choosing the frame's block type from
/// its entropy, "if (coding_block_type == short_block_type) and
/// (last_coding_block_type == long_type) then last coding block type =
/// start_type else last_coding_block_type = short_type", and says that
/// these lines "are necessary since there is no combined stop/start block
/// type in AAC". Taken literally, the `else` would make the frame before
/// short whenever the frame is long, or follows short ones; that is taken
/// as misprinted, and the rule is read from what it is said to do: a long
/// frame before a short one is patched to a start one, as clause C.1.1
/// says too. Step 13 gives no rule for the stop type, which clause 15.3.1
/// requires after short blocks before a long one: a frame without an
/// attack after a short one is a stop frame. A frame without an attack
/// before one with an attack, after a short frame, stays short: there is no
/// combined stop and start type to take.
///
/// Spec gap: neither step 13 nor clause 15.3.1 says what the first frame
/// of a stream, which has no frame before it, may be. It is short if it has
/// an attack, without a start frame before it: the block before it is
/// silent (see [`Filterbank::new`](crate::filterbank::Filterbank::new)),
/// and so is its aliasing. Settled by decoding with ffmpeg (black box): a
/// burst of noise at the first sample of the input, which makes the first
/// frame short, decodes in place, at every sampling frequency, in mono and
/// stereo; the conformance test is
/// `attacks_are_coded_with_short_windows_at_every_sample_rate`.
fn next_sequence(last: Option<WindowSequence>, attacks: Attacks) -> WindowSequence {
    let after_short = ends_short(last);
    match (attacks.this, attacks.next) {
        (true, _) => WindowSequence::EIGHT_SHORT,
        (false, true) if after_short => WindowSequence::EIGHT_SHORT,
        (false, true) => WindowSequence::LongStart,
        (false, false) if after_short => WindowSequence::LongStop,
        (false, false) => WindowSequence::OnlyLong,
    }
}

/// Makes the channels of one element agree on their attacks, where they
/// are to share their windows: an attack in either is taken as one in
/// both, so that their window sequences, chosen alike from the same state,
/// are of the same kind. Their groupings, chosen from each channel's own
/// short blocks (see [`grouping`]), are not yet made to agree.
pub fn agree(attacks: &mut [Attacks], common_window: bool) {
    if common_window {
        let this = attacks.iter().any(|a| a.this);
        let next = attacks.iter().any(|a| a.next);
        attacks.fill(Attacks { this, next });
    }
}

/// The grouping of a frame's short windows (clause 8.3.4) in the channels
/// `analyses`, which share it: consecutive windows share a group while the
/// energy of each, in its short FFT, is within a factor of 10 of every
/// other's in the group, in every channel.
///
/// Spec gap: Annex C says only that "the coefficients associated with
/// contiguous short windows can be grouped such that they share
/// scalefactors amongst all scalefactor bands within the group" (clause
/// C.8.4), and gives no rule for choosing the groups. A group's windows
/// share each band's scalefactor, and so the step their lines are
/// quantized with, and about the same noise: a quiet window grouped with
/// a loud one, as before an attack, would take the loud one's noise. Those
/// of like energy are grouped, to save the scalefactors and section data
/// of a group for each window where their noise can be alike.
/// Graded with PEAQ (GstPEAQ 0.6.1, black box) at 128 kbit/s, on 3 tracks
/// of rock music and their 3 windows of 3 s with the most frames coded
/// short, with the sine window for every frame: each window a group of its
/// own gave a mean of -1.216, a factor of 4 -1.070, of 10 -1.058, of 30
/// -1.058, and every window in one group -1.057.
pub fn grouping(analyses: &[&Analysis]) -> Grouping {
    let energy = |analysis: &Analysis, w: usize| analysis.short[w].epart.iter().sum::<f64>();
    let mut starts = [false; SHORT_WINDOWS];
    let mut first = 0;
    for (w, starts) in starts.iter_mut().enumerate().skip(1) {
        *starts = analyses.iter().any(|analysis| {
            let group = (first..=w).map(|w| energy(analysis, w));
            let least = group.clone().fold(f64::INFINITY, f64::min);
            let most = group.fold(0.0, f64::max);
            most > GROUPING_RATIO * least
        });
        if *starts {
            first = w;
        }
    }
    Grouping::from_starts(starts)
}

/// The window shape of a frame whose window sequence is `sequence`: the
/// Kaiser-Bessel derived window where the right half of its window is
/// long, in an ONLY_LONG_SEQUENCE or a LONG_STOP_SEQUENCE, and the sine
/// window where it is short, in a LONG_START_SEQUENCE or an
/// EIGHT_SHORT_SEQUENCE.
///
/// Spec gap: "Window shape decisions are made by the encoder on a
/// frame-by-frame-basis" (clause C.3.1.1), and Annex C gives no rule for
/// them; Figure C.3 shows the KBD window in steady conditions. Graded with PEAQ (GstPEAQ 0.6.1, black box), on 8 tracks of
/// pop music at 44.1 kHz in stereo, the sine window for every frame gave
/// means of -3.719, -1.975 and -0.569 at 64, 128 and 192 kbit/s, and this
/// rule -3.680, -1.894 and -0.494; on 3 tracks of rock music and their 3
/// windows of 3 s with the most frames coded short, -3.617, -1.058 and
/// -0.187 against -3.625, -1.077 and -0.172. At 128 kbit/s, the KBD window
/// for every frame gave -1.901 and -1.100, and for short windows alone
/// -1.982 and -1.077.
pub fn shape(sequence: WindowSequence) -> WindowShape {
    match sequence {
        WindowSequence::OnlyLong | WindowSequence::LongStop => WindowShape::Kbd,
        WindowSequence::LongStart | WindowSequence::EightShort { .. } => WindowShape::Sine,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use WindowSequence::*;

    fn attacks(this: bool, next: bool) -> Attacks {
        Attacks { this, next }
    }

    #[test]
    fn sequences_follow_step_13() {
        let short = WindowSequence::EIGHT_SHORT;
        assert_eq!(
            next_sequence(Some(OnlyLong), attacks(false, false)),
            OnlyLong
        );
        assert_eq!(
            next_sequence(Some(OnlyLong), attacks(false, true)),
            LongStart
        );
        assert_eq!(next_sequence(Some(LongStart), attacks(true, false)), short);
        assert_eq!(next_sequence(Some(short), attacks(true, true)), short);
        assert_eq!(next_sequence(Some(short), attacks(false, true)), short);
        assert_eq!(next_sequence(Some(short), attacks(false, false)), LongStop);
        assert_eq!(
            next_sequence(Some(LongStop), attacks(false, false)),
            OnlyLong
        );
        assert_eq!(
            next_sequence(Some(LongStop), attacks(false, true)),
            LongStart
        );
        assert_eq!(next_sequence(None, attacks(true, false)), short);
        assert_eq!(next_sequence(None, attacks(false, true)), LongStart);
    }

    /// Every run of attacks gives sequences whose transitions are all
    /// meaningful, every frame with an attack short.
    #[test]
    fn sequences_fit_together_for_any_attacks() {
        let mut seed = 5u64;
        for _ in 0..300 {
            let marks: Vec<bool> = (0..40)
                .map(|_| {
                    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    seed >> 62 == 0
                })
                .collect();
            let mut last = None;
            for (i, &this) in marks.iter().enumerate() {
                let next = marks.get(i + 1).copied().unwrap_or(false);
                let sequence = next_sequence(last, attacks(this, next));
                assert!(
                    !this || sequence.transform() == crate::scalefactor_bands::Transform::Short
                );
                if let Some(last) = last {
                    assert!(
                        sequence.follows(last),
                        "{last:?} then {sequence:?} in {marks:?}"
                    );
                }
                last = Some(sequence);
            }
        }
    }

    #[test]
    fn a_common_window_shares_its_attacks() {
        let mut pair = [attacks(true, false), attacks(false, true)];
        agree(&mut pair, false);
        assert_eq!(pair, [attacks(true, false), attacks(false, true)]);
        agree(&mut pair, true);
        assert_eq!(pair, [attacks(true, true); 2]);
    }

    /// Noise after a quiet tone: the windows before it share a group, and
    /// so do those it fills.
    #[test]
    fn windows_of_like_energy_are_grouped() {
        let fs = SamplingFrequency::Hz44100;
        let mut model = Model::new(fs);
        let mut analysis = Analysis::new(fs);
        let mut state = 1u64;
        for block in 0..3 {
            let samples: [f64; FRAME_SAMPLES] = std::array::from_fn(|t| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let noise = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                let tone = 0.01 * ((block * FRAME_SAMPLES + t) as f64 * 0.1).sin();
                // Noise from sample 320 of the last block, which windows 6
                // and 7 of its frame span: window j spans samples 128 j -
                // 576 to 128 j - 321 of the frame's block.
                if block == 2 && t >= 320 {
                    tone + 0.5 * noise
                } else {
                    tone
                }
            });
            model.analyze(&samples, &mut analysis);
        }
        let grouping = grouping(&[&analysis]);
        assert_eq!(grouping.lengths().collect::<Vec<_>>(), [6, 2]);
    }

    /// Neither a loud steady tone of 80 Hz, whose phase moves through each
    /// short block, nor a sweep from 20 Hz to 20 kHz in 3 s, which moves
    /// through the partitions, has an attack after its onset: the rise of
    /// their short blocks' entropy from the block before reaches 119 and
    /// 161.
    #[test]
    fn steady_tones_and_sweeps_have_no_attack() {
        use std::f64::consts::TAU;
        let fs = SamplingFrequency::Hz44100;
        let rate = f64::from(fs.hz());
        let frames = 3 * fs.hz() as usize / FRAME_SAMPLES;
        let k = (20_000.0f64 / 20.0).ln();
        let duration = (frames * FRAME_SAMPLES) as f64 / rate;
        let signal = |name: &str, t: f64| match name {
            "tone" => 0.5 * (TAU * 80.0 * t).sin(),
            _ => 0.5 * (TAU * 20.0 * duration / k * ((k * t / duration).exp() - 1.0)).sin(),
        };
        for name in ["tone", "sweep"] {
            let mut switching = BlockSwitching::new(fs);
            for block in 0..frames {
                let samples: [f64; FRAME_SAMPLES] = std::array::from_fn(|t| {
                    signal(name, (block * FRAME_SAMPLES + t) as f64 / rate)
                });
                if let Some(attacks) = switching.look_ahead(&samples) {
                    assert!(block < 2 || !attacks.this, "{name}, frame {}", block - 1);
                }
            }
        }
    }

    /// A burst of noise in a quiet tone is coded with short windows in the
    /// frame it sets in, with a start frame before it and a stop frame
    /// after, and the tone again with long ones.
    #[test]
    fn an_attack_switches_to_short_windows() {
        let fs = SamplingFrequency::Hz44100;
        let mut switching = BlockSwitching::new(fs);
        let mut state = 3u64;
        let mut sequences = Vec::new();
        for block in 0..16 {
            let samples: [f64; FRAME_SAMPLES] = std::array::from_fn(|t| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let noise = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                let tone = 0.01 * ((block * FRAME_SAMPLES + t) as f64 * 0.1).sin();
                let since = (block * FRAME_SAMPLES + t) as f64 - (8 * FRAME_SAMPLES + 100) as f64;
                if since >= 0.0 {
                    tone + 0.5 * noise * (-since / 200.0).exp()
                } else {
                    tone
                }
            });
            if let Some(attacks) = switching.look_ahead(&samples) {
                sequences.push(switching.decide(attacks, true).number());
            }
        }
        // The first frame holds the tone's onset after the silence before
        // the stream, and the short windows of frame 8, which span samples
        // -576 to 575 of its block, the burst's.
        assert_eq!(
            sequences,
            [2, 3, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0],
            "ONLY_LONG 0, LONG_START 1, EIGHT_SHORT 2, LONG_STOP 3"
        );
    }
}
