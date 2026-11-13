//! The window switching decision (clause C.1.5.3.2, figure C.7) and the
//! delay of the model's output by one granule that lets it be made (figure
//! C.6.a).

use super::super::header::SamplingFrequency;
use super::super::{BlockType, GRANULE_LINES};
use super::{Masking, PsychoacousticModel, ShortMasking};

/// A granule has an attack, and is coded with short blocks, when its
/// perceptual entropy exceeds this (clause C.1.5.3.2, `switch_pe` in figure
/// C.6.a).
const SWITCH_PE: f64 = 1800.0;

/// The model's output for one granule of one channel, with its block type.
#[derive(Clone, Debug, PartialEq)]
pub enum SwitchedMasking {
    Normal(Masking),
    Start(Masking),
    Short(ShortMasking),
    Stop(Masking),
}

impl SwitchedMasking {
    pub fn block_type(&self) -> BlockType {
        match self {
            Self::Normal(_) => BlockType::Normal,
            Self::Start(_) => BlockType::Start,
            Self::Short(_) => BlockType::Short,
            Self::Stop(_) => BlockType::Stop,
        }
    }

    pub fn perceptual_entropy(&self) -> f64 {
        match self {
            Self::Normal(masking) | Self::Start(masking) | Self::Stop(masking) => {
                masking.perceptual_entropy
            }
            Self::Short(masking) => masking.perceptual_entropy,
        }
    }

    /// The line energies of the long FFT, [`Masking::spectrum`].
    pub fn spectrum(&self) -> &[f64] {
        match self {
            Self::Normal(masking) | Self::Start(masking) | Self::Stop(masking) => &masking.spectrum,
            Self::Short(masking) => &masking.spectrum,
        }
    }

    /// The allowed distortion of each scalefactor of the granule, in the
    /// order they are sent, from its lines `xr` in the order they are sent.
    pub fn allowed_distortion(
        &self,
        xr: &[f64; GRANULE_LINES],
        sampling_frequency: SamplingFrequency,
    ) -> Vec<f64> {
        match self {
            Self::Normal(masking) | Self::Start(masking) | Self::Stop(masking) => {
                masking.allowed_distortion(xr, sampling_frequency).to_vec()
            }
            Self::Short(masking) => masking.allowed_distortion(xr, sampling_frequency).to_vec(),
        }
    }
}

/// Psychoacoustic model 2 with window switching for one channel. Its output
/// lags its input by one granule, as a granule's block type depends on
/// whether the next one has an attack.
pub struct BlockSwitching {
    model: PsychoacousticModel,
    /// The threshold of the granule analysed last, for long and for short
    /// blocks, until its block type is known.
    pending: Option<(Masking, ShortMasking)>,
    /// The block type of the granule before the pending one.
    last: BlockType,
}

impl BlockSwitching {
    /// Spec gap: the standard does not give the state at the start of the
    /// stream. It starts as normal, so an attack in the first granule makes
    /// it short with no start block before it; no block precedes it for its
    /// window to overlap.
    pub fn new(sampling_frequency: SamplingFrequency) -> Self {
        Self {
            model: PsychoacousticModel::new(sampling_frequency),
            pending: None,
            last: BlockType::Normal,
        }
    }

    /// Analyses the next granule of PCM, as
    /// [`PsychoacousticModel::granule`] does, and gives the threshold and
    /// block type of the granule before it, or `None` for the first granule.
    ///
    /// Spec reading: figure C.6.a computes the threshold for short blocks
    /// only for a granule whose entropy exceeds `switch_pe`. This model
    /// computes it for every granule, and the one for long blocks too, and
    /// keeps the one its block type needs: a granule without an attack is
    /// coded with short blocks as well when it falls between two that are.
    pub fn granule(&mut self, pcm: &[f64; GRANULE_LINES]) -> Option<SwitchedMasking> {
        let next = self.model.analyse(pcm);
        let attack_next = next.0.perceptual_entropy > SWITCH_PE;
        let (long, short) = self.pending.replace(next)?;
        let attack = long.perceptual_entropy > SWITCH_PE;
        self.last = block_type(self.last, attack, attack_next);
        Some(match self.last {
            BlockType::Normal => SwitchedMasking::Normal(long),
            BlockType::Start => SwitchedMasking::Start(long),
            BlockType::Short => SwitchedMasking::Short(short),
            BlockType::Stop => SwitchedMasking::Stop(long),
        })
    }
}

/// The block type of a granule, given that of the granule before it and
/// whether it and the granule after it have an attack.
///
/// Spec reading: figure C.7 does not say which granule's attack drives each
/// transition. Figure C.6.a has an attack make its own granule short, and a
/// normal granule before it a start block, so an attack in the next granule
/// is what takes normal to start, and likewise stop to start. Short stays
/// short on an attack in either its own granule or the next: a stop block
/// cannot come before a short one, as a stop window ends long and a short
/// one starts short (clause C.1.5.3.3).
fn block_type(last: BlockType, attack: bool, attack_next: bool) -> BlockType {
    match (last, attack, attack_next) {
        (_, true, _) | (BlockType::Short, false, true) => BlockType::Short,
        (_, false, true) => BlockType::Start,
        (BlockType::Short, false, false) => BlockType::Stop,
        _ => BlockType::Normal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every sequence of attacks gives a sequence of block types whose
    /// windows fit together (clause C.1.5.3.3): a start block is followed by
    /// a short one, a short block by a short or stop one, and a short block
    /// follows a start or short one, except at the start of the stream. Every
    /// granule with an attack is short.
    #[test]
    fn block_types_fit_together_for_any_attacks() {
        use BlockType::*;
        let mut seed = 5u64;
        for _ in 0..200 {
            let attacks: Vec<bool> = (0..40)
                .map(|_| {
                    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    seed >> 62 == 0
                })
                .collect();
            let mut last = Normal;
            let types: Vec<BlockType> = (0..attacks.len())
                .map(|g| {
                    let next = attacks.get(g + 1).copied().unwrap_or(false);
                    last = block_type(last, attacks[g], next);
                    last
                })
                .collect();
            for (g, &t) in types.iter().enumerate() {
                assert!(!attacks[g] || t == Short, "{attacks:?} {types:?}");
            }
            for pair in types.windows(2) {
                let fits = match pair {
                    [Start, next] => *next == Short,
                    [Short, next] => matches!(next, Short | Stop),
                    [_, Short] => false,
                    [_, Stop] => false,
                    _ => true,
                };
                assert!(fits, "{pair:?} in {types:?} for {attacks:?}");
            }
        }
    }

    #[test]
    fn block_types_follow_figure_c7() {
        use BlockType::*;
        assert_eq!(block_type(Normal, false, false), Normal);
        assert_eq!(block_type(Normal, false, true), Start);
        assert_eq!(block_type(Start, true, false), Short);
        assert_eq!(block_type(Short, true, true), Short);
        assert_eq!(block_type(Short, false, false), Stop);
        assert_eq!(block_type(Stop, false, false), Normal);
        assert_eq!(block_type(Stop, false, true), Start);
    }

    /// A burst of noise after silence switches to short blocks for the
    /// granule it starts in, with a start block before it and a stop block
    /// after the attack has passed. The output lags by one granule.
    #[test]
    fn an_attack_switches_to_short_blocks() {
        const ATTACK: usize = 8;
        let mut switching = BlockSwitching::new(SamplingFrequency::Hz44100);
        let mut seed = 11u64;
        let mut outputs = Vec::new();
        for g in 0..24 {
            let pcm: [f64; GRANULE_LINES] = std::array::from_fn(|t| {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let noise = (seed >> 11) as f64 / (1u64 << 52) as f64 - 1.0;
                // The burst starts three quarters into the granule.
                if g * GRANULE_LINES + t >= ATTACK * GRANULE_LINES + 432 {
                    0.5 * noise
                } else {
                    0.0
                }
            });
            outputs.extend(switching.granule(&pcm));
        }
        assert_eq!(outputs.len(), 23);
        let types: Vec<BlockType> = outputs.iter().map(SwitchedMasking::block_type).collect();

        // The model's block is centred 241 samples before the end of the
        // granule before the one fed in last (see `BLOCK_CENTRE`), so the
        // burst first shows in the granule after ATTACK.
        let short = types.iter().position(|&t| t == BlockType::Short).unwrap();
        assert!((ATTACK..=ATTACK + 1).contains(&short), "{types:?}");
        assert_eq!(types[short - 1], BlockType::Start, "{types:?}");
        assert!(types[..short - 1].iter().all(|&t| t == BlockType::Normal));
        let stop = types.iter().position(|&t| t == BlockType::Stop).unwrap();
        assert!(types[short..stop].iter().all(|&t| t == BlockType::Short));
        // Steady noise needs no short blocks.
        assert!(
            types[stop + 1..].iter().all(|&t| t == BlockType::Normal),
            "{types:?}"
        );
        assert!(matches!(outputs[short], SwitchedMasking::Short(_)));
    }
}
