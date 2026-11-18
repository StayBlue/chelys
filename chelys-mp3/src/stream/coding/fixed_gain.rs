// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Quantisation of every granule with one `global_gain`, in place of the
//! encoding process, for chelys's own tests: it gives streams whose
//! quantised values follow from the input alone.

use super::{CodedFrame, Coding, Filtered};
use crate::granule::{BlockType, GRANULE_LINES, QuantizedGranule};
use crate::header::SamplingFrequency;
use crate::reservoir::Reservoir;
use crate::stream::StreamConfig;
use crate::stream::frame_encoder::FrameEncoder;
use crate::{loops, stereo};

/// Encodes PCM like [`encode_pcm`](crate::stream::encode_pcm), but with
/// long blocks only, quantising every granule with the same `global_gain`
/// instead of fitting it to the bitrate, and without a bit reservoir.
///
/// # Panics
///
/// If the channel count does not match the configured mode, or a granule
/// cannot be quantised with `global_gain` or does not fit in its frame.
pub fn encode_pcm_with_global_gain(
    config: &StreamConfig,
    global_gain: u8,
    channels: &[&[f64]],
) -> Vec<u8> {
    encode_with_block_types(config, global_gain, channels, |_, _| BlockType::Normal)
}

/// Encodes PCM like [`encode_pcm_with_global_gain`], with the block type of
/// each granule given by `block_type(channel, granule)`. The block types must
/// follow each other as figure C.7 allows, with mixed blocks in place of
/// short ones if any, and not at 8 kHz. If the mode allows MS stereo,
/// every frame is coded in it, with no limitation of the side channel, and
/// both channels must have the same block types.
///
/// # Panics
///
/// As [`encode_pcm_with_global_gain`], and if the mode allows intensity
/// stereo.
pub fn encode_with_block_types(
    config: &StreamConfig,
    global_gain: u8,
    channels: &[&[f64]],
    block_type: impl FnMut(usize, usize) -> BlockType,
) -> Vec<u8> {
    let (ms_stereo, intensity_stereo) = config.joint_stereo();
    assert!(!intensity_stereo, "intensity stereo needs encode_pcm");
    let coding = FixedGainCoding {
        global_gain,
        block_type,
        next: vec![0; config.mode.channels()],
        ms_stereo,
        sampling_frequency: config.sampling_frequency,
    };
    let mut encoder = FrameEncoder::new(*config, coding, Reservoir::disabled());
    let mut out = Vec::new();
    encoder.encode(channels, &mut out);
    encoder.finish(&mut out);
    out
}

/// Quantisation of every granule with the same `global_gain`, with the block
/// type of each given by `block_type(channel, granule)`, for
/// [`encode_with_block_types`].
struct FixedGainCoding<F> {
    global_gain: u8,
    block_type: F,
    /// The index of the granule each channel's look-ahead is given next.
    next: Vec<usize>,
    ms_stereo: bool,
    sampling_frequency: SamplingFrequency,
}

impl<F: FnMut(usize, usize) -> BlockType> Coding for FixedGainCoding<F> {
    type Analysis = ();

    fn look_ahead(&mut self, channel: usize, _: &[f64; GRANULE_LINES]) -> Option<(BlockType, ())> {
        // Called once ahead of the first granule, whose type it gives on the
        // second call.
        let granule = self.next[channel].checked_sub(1);
        self.next[channel] += 1;
        granule.map(|g| ((self.block_type)(channel, g), ()))
    }

    fn quantize(
        &mut self,
        frame: &[Vec<Filtered<()>>],
        mean_bits: usize,
        reservoir: &mut Reservoir,
        _intensity: bool,
        _only_within: bool,
    ) -> Option<CodedFrame> {
        let (global_gain, fs) = (self.global_gain, self.sampling_frequency);
        let mut granules = Vec::with_capacity(frame.len());
        for granule in frame {
            let mut lines: Vec<_> = granule.iter().map(|filtered| filtered.xr).collect();
            if self.ms_stereo {
                let [left, right] = &granule[..] else {
                    unreachable!("MS stereo has two channels")
                };
                assert_eq!(
                    left.block_type, right.block_type,
                    "MS stereo needs one block type"
                );
                let (middle, side) = stereo::middle_side(&left.xr, &right.xr);
                lines = vec![middle, side];
            }
            let quantized: Vec<_> = (granule.iter().zip(&lines))
                .map(|(filtered, xr)| {
                    let values = loops::quantize(xr, global_gain).unwrap_or_else(|| {
                        panic!("a granule is too loud for global_gain {global_gain}")
                    });
                    let granule =
                        QuantizedGranule::new(global_gain, values, filtered.block_type, fs);
                    reservoir.granule_done(mean_bits, granule.bit_len(fs));
                    granule
                })
                .collect();
            granules.push(quantized);
        }
        Some(CodedFrame {
            granules,
            ms_stereo: self.ms_stereo,
            intensity_stereo: false,
            excess: 0.0,
        })
    }
}
