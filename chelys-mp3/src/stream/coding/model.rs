// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The encoding process of Annex C: psychoacoustic model 2 with block
//! switching, the iteration loops and the bit reservoir, with joint stereo
//! as Annex G describes.

use std::cell::OnceCell;

use super::{CodedFrame, Coding, Filtered};
use crate::granule::{BlockType, GRANULE_LINES, MAX_PART2_3_LENGTH, QuantizedGranule};
use crate::header::{SamplingFrequency, Version};
use crate::loops::scfsi::{self, GranuleSummary};
use crate::psychoacoustic::switching::{BlockSwitching, SwitchedMasking};
use crate::rate::StreamRate;
use crate::reservoir::Reservoir;
use crate::scalefactors::Scalefactors;
use crate::stream::StreamConfig;
use crate::{loops, scalefactor_bands, side_info, stereo};

/// The encoding process of Annex C: psychoacoustic model 2 with block
/// switching, the iteration loops and the bit reservoir, with joint stereo as
/// Annex G describes.
pub(in crate::stream) struct ModelCoding {
    /// One for each channel.
    models: Vec<BlockSwitching>,
    ms_allowed: bool,
    intensity_allowed: bool,
    /// The factor the allowed distortion of every band is scaled by: 1 but
    /// at a variable bitrate (see [`rate::VariableRate`](crate::rate::VariableRate)).
    noise_factor: f64,
    /// How much more noise than its allowance, in dB, a normal block within
    /// its allowance may be given to leave bits to the reservoir (see
    /// [`loops::outer_loop`]): [`loops::SAVING_NOISE_DB`] but at a variable
    /// bitrate, where it is 0. There the allowance is already raised by
    /// [`noise_factor`](Self::noise_factor), and a frame is kept at a
    /// bitrate only if its noise stays within it; a granule given more
    /// noise than that would push its frame to the highest bitrate. Graded
    /// with PEAQ (GstPEAQ 0.6.1, black box), 22 tracks of pop music at
    /// 44.1 kHz in stereo, with `noise_offset_db` 4, averaged 184 kbit/s and
    /// a grade of -0.35 without saving bits, 298 kbit/s and -0.06 saving
    /// them with the noise raised 4 dB, and 140 kbit/s and -0.50 saving them
    /// down to the allowance.
    saving_noise_db: f64,
    /// Whether a normal block that leaves bits to the reservoir keeps the
    /// bands fixed at zero within their allowance too (see
    /// [`loops::Saving::fixed_bands_within`]): at a variable bitrate, but
    /// not at a constant or mean one.
    ///
    /// Spec gap: the bands fixed at zero, at MPEG-2.5's 8 kHz alone, cannot
    /// be amplified, and the outer loop leaves them out of its test (see
    /// [`loops::outer_loop`]); their noise follows the step the other bands
    /// are coded at. At a variable bitrate a frame is kept at a bitrate
    /// only if they too are within their allowance (see
    /// [`loops::excess`]). Left out of the saving search as well, a granule
    /// within its allowance was coarsened past theirs at every bitrate whose
    /// frame left the reservoir room for the bits saved, and so the frame was
    /// kept only where it did not: over one minute of each of nine tracks,
    /// one of each kind of music, the mean bitrate in two channels at levels
    /// 0, 50, 80 and 100 fell from 22.2 kbit/s to 18.7, 40.7 to 36.0, 60.7
    /// to 55.6 and 78.9 to 74.7, and in one channel from 14.3 to 13.3, 20.8
    /// to 20.0, 25.5 to 24.8 and 28.6 to 28.0, with the fixed bands of as
    /// many frames beyond their allowance as before. At a constant or mean
    /// bitrate the frame's bits are given, and the bands fixed at zero are
    /// left out as before.
    fixed_bands_within: bool,
    /// Whether a granule first aims for the noise offset the reservoir
    /// steers (see [`quantize_frame`]): at a constant or mean bitrate, but
    /// not at a variable one, which has an offset of its own.
    steered: bool,
    sampling_frequency: SamplingFrequency,
    /// Whether the frame being coded is coded in MS stereo, and its
    /// granules as every try at coding it starts from, both worked out
    /// once by [`Coding::prepare`].
    ms_stereo: bool,
    prepared: Vec<PreparedGranule>,
}

impl ModelCoding {
    /// The coding of a stream of `config`.
    pub(in crate::stream) fn new(config: &StreamConfig) -> Self {
        let fs = config.sampling_frequency;
        let (ms_allowed, intensity_allowed) = config.joint_stereo();
        let noise_factor = match config.rate {
            StreamRate::Variable(variable) => variable.noise_factor(),
            _ => 1.0,
        };
        Self {
            models: (0..config.mode.channels())
                .map(|_| BlockSwitching::new(fs))
                .collect(),
            ms_allowed,
            intensity_allowed,
            noise_factor,
            saving_noise_db: match config.rate {
                StreamRate::Variable(_) => 0.0,
                _ => loops::SAVING_NOISE_DB,
            },
            fixed_bands_within: matches!(config.rate, StreamRate::Variable(_)),
            steered: !matches!(config.rate, StreamRate::Variable(_)),
            sampling_frequency: fs,
            ms_stereo: false,
            prepared: Vec::new(),
        }
    }

    /// Has every channel's model code mixed blocks in place of short ones
    /// (see [`BlockSwitching::with_mixed_blocks`]).
    pub(in crate::stream) fn use_mixed_blocks(&mut self, sampling_frequency: SamplingFrequency) {
        let models = std::mem::take(&mut self.models);
        self.models = (models.into_iter())
            .map(|model| model.with_mixed_blocks(sampling_frequency))
            .collect();
    }
}

/// A granule of a frame as every try at coding the frame starts from: each
/// channel's lines and allowed distortion, in MS stereo if the frame is
/// coded in it, and what scfsi takes of them, unless intensity stereo
/// changes the lines.
struct PreparedGranule {
    lines: Vec<([f64; GRANULE_LINES], Vec<f64>)>,
    summaries: Vec<GranuleSummary>,
    /// Each channel's lines as the outer loop starts from them, once the
    /// first try works them out (see [`loops::outer_loop_with`]).
    unscaled: Vec<OnceCell<loops::Unscaled>>,
}

impl PreparedGranule {
    fn new(
        granule: &[Filtered<Analysis>],
        ms_stereo: bool,
        sampling_frequency: SamplingFrequency,
    ) -> Self {
        let fs = sampling_frequency;
        let mut lines: Vec<_> = granule
            .iter()
            .map(|f| (f.xr, f.analysis.1.clone()))
            .collect();
        if ms_stereo {
            let [(left, left_xmin), (right, right_xmin)] = &mut lines[..] else {
                unreachable!("joint stereo has two channels")
            };
            (*left, *right) = stereo::middle_side::matrix(left, right);
            stereo::middle_side::limit_side(right, granule[0].block_type, fs);
            let joint = stereo::middle_side::distortion(left_xmin, right_xmin);
            left_xmin.clone_from(&joint);
            *right_xmin = joint;
        }
        let summaries = (granule.iter().zip(&lines))
            .map(|(f, (xr, xmin))| GranuleSummary::new(xr, xmin, f.block_type, fs))
            .collect();
        let unscaled = lines.iter().map(|_| OnceCell::new()).collect();
        Self {
            lines,
            summaries,
            unscaled,
        }
    }
}

impl Coding for ModelCoding {
    type Analysis = Analysis;

    fn look_ahead(
        &mut self,
        channel: usize,
        pcm: &[f64; GRANULE_LINES],
    ) -> Option<(BlockType, Analysis)> {
        let masking = self.models[channel].granule(pcm)?;
        Some((masking.block_type(), (masking, Vec::new())))
    }

    fn prepare(&mut self, frame: &mut [Vec<Filtered<Analysis>>]) {
        let fs = self.sampling_frequency;
        for granule in frame.iter_mut() {
            for filtered in granule.iter_mut() {
                let (masking, xmin) = &mut filtered.analysis;
                *xmin = masking.allowed_distortion(&filtered.xr, fs, self.noise_factor);
            }
        }
        self.ms_stereo = self.ms_allowed && middle_side_frame(frame);
        self.prepared = (frame.iter())
            .map(|granule| PreparedGranule::new(granule, self.ms_stereo, fs))
            .collect();
    }

    fn quantize(
        &mut self,
        frame: &[Vec<Filtered<Analysis>>],
        mean_bits: usize,
        reservoir: &mut Reservoir,
        intensity: bool,
        only_within: bool,
    ) -> Option<CodedFrame> {
        let fs = self.sampling_frequency;
        let ms_stereo = self.ms_stereo;
        let steps = if intensity && self.intensity_allowed && intensity_frame(frame) {
            0..stereo::intensity::STEPS
        } else {
            0..0
        };
        let to_quantize = FrameToQuantize {
            frame,
            prepared: &self.prepared,
            mean_bits,
            saving_noise_db: self.saving_noise_db,
            fixed_bands_within: self.fixed_bands_within,
            steered: self.steered,
            ms_stereo,
            sampling_frequency: fs,
        };
        let if_outside = match only_within {
            true => IfOutsideAllowance::GiveUpBeyond,
            false => IfOutsideAllowance::Finish,
        };
        let plain = quantize_frame(&to_quantize, None, reservoir, if_outside)?;
        let tries: Vec<_> = if plain.within_allowance {
            Vec::new()
        } else {
            steps
                .filter_map(|step| IntensityTry::new(frame, &plain.granules, step, fs))
                .collect()
        };
        let kept = search_intensity(&tries, |intensity, if_outside| {
            quantize_frame(&to_quantize, Some(intensity), reservoir, if_outside)
        });
        let intensity_stereo = kept.is_some();
        let FrameTry {
            granules,
            excess,
            reservoir: kept_reservoir,
            ..
        } = kept.unwrap_or(plain);
        *reservoir = kept_reservoir;
        Some(CodedFrame {
            granules,
            ms_stereo,
            intensity_stereo,
            excess,
        })
    }
}

/// Annex G.1: if the frame needs more bits than it has, more and more bands
/// are coded in intensity stereo, down to the lowest bound, which is kept
/// even if the frame still needs more. Bounds that would not lower the bits
/// are not tried (see [`IntensityTry::new`]).
///
/// Given the tries worth making, from the fewest bands to the most, and a
/// function that quantises the frame for a try, returns the try kept: `None`
/// if there are no tries, or if the function gives up on every one, as
/// [`quantize_frame`] can whatever the distortion. A try before the last is
/// kept only if the whole frame is within its allowed distortion, so it is
/// given up at the first granule that is not; the frame then comes out as if
/// it had been finished and passed over.
///
/// Spec gap: G.1 does not say what to do when the frame cannot be coded
/// from the lowest bound at all, as when at the lower sampling frequencies
/// the right channel would need a band marked (see [`quantize_frame`]). The
/// tries are then finished from the lowest bound up, and the first that can
/// be coded is kept: the most bands that can be coded in intensity stereo,
/// as G.1 would have from the lowest bound. Tries given up for their
/// distortion are quantised again to the end.
fn search_intensity<Try, Frame>(
    tries: &[Try],
    mut quantize: impl FnMut(&Try, IfOutsideAllowance) -> Option<Frame>,
) -> Option<Frame> {
    let (_, earlier) = tries.split_last()?;
    earlier
        .iter()
        .find_map(|intensity| quantize(intensity, IfOutsideAllowance::GiveUp))
        .or_else(|| {
            (tries.iter().rev())
                .find_map(|intensity| quantize(intensity, IfOutsideAllowance::Finish))
        })
}

/// A frame to quantise, the same for each try at coding it: its granules,
/// indexed `[gr][ch]`, whether it is in MS stereo, and the share of its main
/// data of each granule of each channel.
struct FrameToQuantize<'a> {
    frame: &'a [Vec<Filtered<Analysis>>],
    /// Its granules as [`Coding::prepare`] readied them.
    prepared: &'a [PreparedGranule],
    mean_bits: usize,
    /// How much more noise than its allowance, in dB, a normal block within
    /// its allowance may be given to leave bits to the reservoir (see
    /// [`loops::outer_loop`]).
    saving_noise_db: f64,
    /// Whether a normal block that leaves bits to the reservoir keeps the
    /// bands fixed at zero within their allowance too.
    fixed_bands_within: bool,
    /// Whether a granule first aims for the noise offset the reservoir
    /// steers (see [`quantize_frame`]).
    steered: bool,
    ms_stereo: bool,
    sampling_frequency: SamplingFrequency,
}

/// What [`quantize_frame`] does when a granule comes out outside its allowed
/// distortion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IfOutsideAllowance {
    /// Quantise the rest of the frame.
    Finish,
    /// Stop, and return no frame.
    GiveUp,
    /// Stop, and return no frame, once a granule's noise exceeds its
    /// allowance as [`loops::excess`] measures it, which the frame's
    /// [`CodedFrame::excess`] then does too.
    GiveUpBeyond,
}

/// A frame quantised one way, and the bit reservoir after it, which is taken
/// over only if the frame is kept.
struct FrameTry {
    /// The frame's granules, with one entry per channel each.
    granules: Vec<Vec<QuantizedGranule>>,
    /// Whether every granule was within its allowed distortion, as the
    /// outer loop judges it.
    within_allowance: bool,
    /// The largest [`loops::excess`] of its granules.
    excess: f64,
    reservoir: Reservoir,
}

/// A granule's psychoacoustic analysis and the allowed distortion of each
/// of its scalefactors, in the order they are sent.
type Analysis = (SwitchedMasking, Vec<f64>);

/// Whether to code a stereo frame in MS stereo (Annex G.2 a)).
///
/// The normative decoding needs both channels of a granule to have the same
/// block type in MS stereo (clause 2.4.3.4.9). Spec gap: Annex G does not
/// say what to do when they do not; such frames are coded in left/right
/// stereo.
fn middle_side_frame(frame: &[Vec<Filtered<Analysis>>]) -> bool {
    let same_types = frame
        .iter()
        .all(|granule| granule[0].block_type == granule[1].block_type);
    let spectra: Vec<_> = frame
        .iter()
        .map(|granule| {
            (
                granule[0].analysis.0.spectrum(),
                granule[1].analysis.0.spectrum(),
            )
        })
        .collect();
    same_types && stereo::middle_side::switches_on(&spectra)
}

/// Whether a stereo frame may be coded in intensity stereo: if each of its
/// granules has the same block type in both channels, and none is a mixed
/// block. Short blocks find the bound in each window (see
/// [`stereo::intensity::Positions::mark`]).
///
/// Spec gap: the standard asks for the same block type in both channels
/// only for MS stereo (clause 2.4.3.4.9). As intensity stereo moves the left
/// channel's lines into the right channel's synthesis, whose window should
/// then be the same, and short blocks send their lines in another order
/// (clause 2.4.2.7), frames are coded in it only if their channels have the
/// same block types.
///
/// Spec gap: the bound of intensity stereo is found "individually for each
/// window in the case of short blocks (block_type == '10')" (clause 2.4.3.2
/// of ISO/IEC 13818-3), which mixed blocks are too, but neither standard
/// says how a mixed block's long bands, which have one window, and its
/// short bands share a bound. Frames with mixed blocks are not coded in
/// intensity stereo.
fn intensity_frame(frame: &[Vec<Filtered<Analysis>>]) -> bool {
    frame.iter().all(|granule| {
        granule[0].block_type == granule[1].block_type && granule[0].block_type != BlockType::Mixed
    })
}

/// A try at coding a frame in intensity stereo: the positions of each of
/// its granules, and the bits the right channel of each keeps.
///
/// Spec gap: Annex G does not say how to share the bits between the
/// channels in intensity stereo. In Layers I and II a combined subband is
/// allotted bits once, the higher of its two channels' allocations (Annex
/// G.1), but the Layer III iteration loops use as many of the bits each
/// channel is given as they can (clauses C.1.5.4.1 and C.1.5.4.2.1): the
/// right
/// channel would spend the bits of the bands it no longer codes on those it
/// still does, while the left channel codes the sum of both on the bits it
/// had for itself alone. So the right channel keeps only the bits it spent
/// below the bound when the frame was coded without intensity stereo, and
/// those its scalefactors then take, marked with its intensity stereo
/// positions, or the most the marks can take if that is more. The left
/// channel gets what that leaves of the right channel's share of the
/// frame.
struct IntensityTry {
    positions: Vec<stereo::intensity::Positions>,
    right_bits: Vec<usize>,
}

impl IntensityTry {
    /// The try at coding `frame` in intensity stereo from the bounds of
    /// `step` of [`stereo::intensity::bounds`], long or short as each
    /// granule's block type has them, given its granules as coded without
    /// it, `plain`.
    ///
    /// Spec reading: Annex G.1 codes bands in intensity stereo to lower the
    /// bits a frame needs. When the right channel's bits from the bound up
    /// in `plain` are no more than its intensity stereo positions would take,
    /// the bound would not lower them, and there is no try: `None`. This is
    /// so at low bitrates, where the right channel's upper bands often
    /// quantise to zero without intensity stereo.
    fn new(
        frame: &[Vec<Filtered<Analysis>>],
        plain: &[Vec<QuantizedGranule>],
        step: usize,
        sampling_frequency: SamplingFrequency,
    ) -> Option<Self> {
        let fs = sampling_frequency;
        let positions: Vec<_> = frame
            .iter()
            .map(|granule| {
                let block_type = granule[0].block_type;
                let bound = stereo::intensity::bounds(block_type, fs)[step];
                let (left, right) = (&granule[0].xr, &granule[1].xr);
                stereo::intensity::Positions::new(left, right, block_type, bound, fs)
            })
            .collect();
        let right_bits: Vec<usize> = (plain.iter().zip(&positions))
            .map(|(granule, positions)| {
                let right = &granule[1];
                let bands = scalefactor_bands::coding_bands(right.block_type, fs);
                // The lines from the bound up follow those below it in every
                // window.
                let start = positions.first_line();
                let below = right.coding.bit_count_below(&right.values, &bands, start);
                // Its scalefactors as they would be sent with those values:
                // all of them, as it will not use scfsi, and marked.
                let mut values = right.values;
                values[start..].fill(0);
                // At the lower sampling frequencies the right channel's
                // branch of scalefac_compress has no preflag, and sends the
                // scalefactors within their limits without it.
                let mut scalefactors = right.scalefactors.clone();
                scalefactors.scfsi = [false; side_info::SCFSI_BANDS];
                if fs.version() == Version::Mpeg2 {
                    scalefactors.preflag = false;
                }
                positions.mark(&mut scalefactors, &values);
                let compress = scalefactors
                    .compress()
                    .expect("scalefactors within their limits");
                // Enough for the marks if fewer bits leave more bands zero.
                (scalefactors.part2_length(compress) + below).max(positions.silent_bits())
            })
            .collect();
        let plain_bits: usize = plain.iter().map(|granule| granule[1].bit_len(fs)).sum();
        (right_bits.iter().sum::<usize>() < plain_bits).then_some(Self {
            positions,
            right_bits,
        })
    }
}

/// Quantises a frame's granules, from them as [`Coding::prepare`] readied
/// them, in MS stereo if it chose that, and in intensity stereo as
/// `intensity` gives, accounting for their bits in a
/// copy of the reservoir, so that a try leaves the encoder's state as it
/// was. `None` if `if_outside` gives up on a granule outside its allowed
/// distortion, or if the reservoir gives the right channel of a granule
/// in intensity stereo fewer bits than its marks take with all its lines
/// zero (see [`stereo::intensity::Positions::silent_bits`]), which can happen
/// at the lowest bitrates of the lower sampling frequencies. `None` too at
/// the lower sampling frequencies if the right channel of a granule in
/// intensity stereo would mark a band.
///
/// Spec reading: at the lower sampling frequencies the illegal intensity
/// position that marks a band of the right channel's zero part as not coded
/// in intensity stereo is the largest value of its field (clause 2.4.3.2 of
/// ISO/IEC 13818-3), but decoders do not agree on it, and no mark is read as
/// one by all of them. So the encoder sends no marks there: the right
/// channel is given a value in the band just below the bound where it has
/// none, in any window of a short block (see
/// [`stereo::intensity::Positions::fill_below_bound`]), and a frame whose
/// right channel would still need a mark is coded from another bound (see
/// [`search_intensity`]) or without intensity stereo.
///
/// Settled by decoding with ffmpeg, mpg123 and LAME (black box): of the
/// marks of fields of 1 to 5 bits, 1, 3, 7, 15 and 31, ffmpeg reads only 31
/// as illegal and the others as positions, and mpg123 and LAME read only 7
/// as illegal and the others as positions; a marked band is then decoded in
/// intensity stereo. The conformance test is
/// `marks_are_read_as_illegal_in_different_fields`. In MS stereo at 16,
/// 22.05 and 24 kHz, with the side channel all but zero below every bound,
/// frames are coded in intensity stereo with the side channel given a value
/// in the band just below the bound, and all three decode every tone at its
/// level; the conformance test is
/// `intensity_stereo_keeps_the_level_of_each_channel_at_the_lower_sampling_frequencies`.
///
/// Spec reading: at a constant or mean bitrate, a granule is first
/// quantised against its allowed distortion raised by the offset the
/// reservoir's fill gives ([`Reservoir::noise_offset_db`]): 4 dB when the
/// reservoir is empty, falling to 0 as it fills, but in each band to no
/// more than half the band's energy per line (see [`raised_allowance`]).
/// If it meets that allowance, a normal block is given the coarsest step
/// at which it still does (see [`loops::outer_loop_with`]), and the bits
/// it leaves go to the reservoir. If it does not, it is quantised again
/// as it would be without the offset: against the model's own allowance,
/// saving bits above that by [`loops::SAVING_NOISE_DB`] where it meets
/// it. This deliberately departs from Annex C, whose loops take the
/// allowance from the model (clauses C.1.5.4.1 and C.1.5.4.3) and never
/// raise it by a choice of their own, and use as many of the bits they
/// are given as they can, the reservoir gaining only what a granule
/// cannot spend (clauses C.1.5.4.1, C.1.5.4.2.1 and C.1.5.4.2.2).
/// The noise then follows the model's allowance at a level that moves only
/// as fast as the reservoir fills or drains. Without the offset, it moved
/// with whether each granule met its allowance and with what the
/// reservoir lent it. At 64 kbit/s in mono, the decoded noise of a track
/// of 1970s rock changed by 2.0 dB from one granule's 576 samples to the
/// next on average, against 1.8 dB for the signal, and PEAQ's modulation
/// difference (AvgModDiff1B) was the measure that weighed most in the
/// grade (by the basic version's network, ITU-R BS.1387-2, Tables 13 to
/// 16). With the offset, the noise changed by 1.5 dB.
///
/// The side channel of a frame in MS stereo, and the right channel of one
/// in intensity stereo, skip the first try. Where the reservoir filled,
/// the side channel, which takes few bits, was given the finer noise of a
/// low offset, and the middle channel was then short of them. A granule
/// kept from the first try counts as within its allowance, so a frame
/// whose granules all meet the raised one is not tried in intensity stereo
/// (see [`search_intensity`]).
///
/// The limit to half a band's energy keeps the bandwidth. Raised alike in
/// every band, the allowance passed the energy of quiet upper bands, which
/// then met it with all their lines zero: the global step came out finer,
/// but at 64 kbit/s in mono the last line coded in a track of orchestral
/// music fell from 352 to 328 on average. PEAQ's test bandwidth
/// (BandwidthTestB) fell from 683 to 677 FFT lines in mono, and with
/// intensity stereo at 96 kbit/s from 698 to 680 for analog recordings,
/// whose grade fell by 0.1; with the limit it is 686 and 703. In mono,
/// limits of 1/4, 0.35 and 0.7 of the energy graded 0.11, 0.03 and 0.03
/// lower than 1/2, and no limit 0.11 lower.
///
/// Graded with PEAQ (GstPEAQ 0.6.1, black box) on 51 tracks of nine kinds
/// of music, the mean grade changed:
/// - at 64 kbit/s in mono at 44.1 kHz, it rose from -1.435 to -1.161,
///   every track higher, with AvgModDiff1B falling from 16.2 to 13.7;
/// - in stereo at 44.1 kHz, it rose from -2.066 to -2.052 at 96 kbit/s,
///   from -0.861 to -0.822 at 128 kbit/s and from -0.194 to -0.173 at
///   192 kbit/s;
/// - at 64 kbit/s in stereo at 22.05 kHz, it moved from -3.226 to -3.230;
/// - with intensity stereo, it rose from -1.828 to -1.702 at 96 kbit/s and
///   from -0.997 to -0.906 at 128 kbit/s, where piano music fell by 0.036.
///
/// Other choices graded lower. They were measured before the limit to
/// half a band's energy was added, and in mono on another downmix of the
/// same tracks, 3 dB louder and in part beyond full scale, on which the
/// change, without the limit, gave +0.187:
/// - A constant offset of 3.5 dB, without the second try, raised
///   64 kbit/s in mono by only 0.13.
/// - Offsets falling to -4 dB as the reservoir filled did as well in mono
///   (+0.198), but less well in stereo: +0.004 at 96 kbit/s and +0.025
///   at 128.
/// - Steering the side channel too lost 0.083 at 96 kbit/s.
/// - Keeping the first try where it missed the raised allowance, without
///   quantising again, gave +0.167 in mono and +0.011 at 128 kbit/s,
///   though +0.042 at 96.
///
/// The perceptual entropy each channel's bits are reckoned from stays that
/// of the left and right channels; in intensity stereo the two then share
/// them as [`IntensityTry`] describes.
///
/// Spec gap: Annex G does not say how to find the allowed distortion of
/// the channels coded in MS or intensity stereo; see
/// [`stereo::middle_side::distortion`] and
/// [`stereo::intensity::Positions::sum_distortion`].
///
/// Spec gap: scfsi would have the second granule take over the first
/// granule's intensity stereo positions, which follow the signal of each
/// granule (Annex G.2 c)). In intensity stereo the right channel does not
/// use scfsi.
fn quantize_frame(
    to_quantize: &FrameToQuantize,
    intensity: Option<&IntensityTry>,
    reservoir: &Reservoir,
    if_outside: IfOutsideAllowance,
) -> Option<FrameTry> {
    let &FrameToQuantize {
        frame,
        prepared,
        mean_bits,
        saving_noise_db,
        fixed_bands_within,
        steered,
        ms_stereo,
        sampling_frequency: fs,
    } = to_quantize;
    let mut reservoir = reservoir.clone();
    // Each channel's first granule, for scfsi in the second.
    let mut first: Vec<Option<(GranuleSummary, Scalefactors)>> = vec![None; frame[0].len()];
    let mut within_allowance = true;
    let mut excess: f64 = 0.0;
    let mut granules = Vec::with_capacity(frame.len());
    for (gr, granule) in frame.iter().enumerate() {
        let mut lines = prepared[gr].lines.clone();
        let positions = intensity.map(|intensity| &intensity.positions[gr]);
        if let Some(positions) = positions {
            let [(left, left_xmin), (right, _)] = &mut lines[..] else {
                unreachable!("joint stereo has two channels")
            };
            positions.code(&granule[0].xr, &granule[1].xr, [left, right]);
            let (left_alone, right_alone) = (&granule[0].analysis.1, &granule[1].analysis.1);
            let coded = positions.first_scalefactor();
            for (index, xmin) in left_xmin.iter_mut().enumerate().skip(coded) {
                *xmin = positions.sum_distortion(index, left_alone[index], right_alone[index]);
            }
        }
        // The right channel's bits are reckoned before the left channel takes
        // any, and need not all be there after; it gets no more than the
        // reservoir then gives it. The left channel gets only what the right
        // leaves of its share of the frame, which the right then returns.
        let right_bits = intensity.map(|intensity| intensity.right_bits[gr]);
        let mut quantized = Vec::with_capacity(granule.len());
        for (channel, (filtered, (xr, xmin))) in granule.iter().zip(&lines).enumerate() {
            let block_type = filtered.block_type;
            let mut available_bits =
                reservoir.granule_bits(mean_bits, filtered.analysis.0.perceptual_entropy());
            if let Some(right_bits) = right_bits {
                available_bits = match channel {
                    0 => available_bits + mean_bits.saturating_sub(right_bits),
                    _ => available_bits.min(right_bits),
                };
            }
            let available_bits = available_bits.min(MAX_PART2_3_LENGTH);
            let intensity = positions.filter(|_| channel == 1);
            if intensity.is_some_and(|positions| available_bits < positions.silent_bits()) {
                return None;
            }
            let summary = match positions {
                None => prepared[gr].summaries[channel].clone(),
                Some(_) => GranuleSummary::new(xr, xmin, block_type, fs),
            };
            let reuse = match (gr, &first[channel]) {
                (1, Some((first_summary, first_scalefactors))) if intensity.is_none() => {
                    Some(loops::Reuse {
                        first: first_scalefactors,
                        scfsi: scfsi::select(first_summary, &summary),
                    })
                }
                _ => None,
            };
            let bits = |noise_db| loops::Bits {
                available: available_bits,
                saving: Some(loops::Saving {
                    least: reservoir.least_use(mean_bits),
                    noise_db,
                    fixed_bands_within,
                }),
            };
            // Intensity stereo changes the lines from those prepared.
            let unscaled = positions.is_none().then(|| &prepared[gr].unscaled[channel]);
            // The first try against the allowance raised by the reservoir's
            // offset, kept if it meets it; else the model's own allowance.
            let side = ms_stereo && channel == 1;
            let raised = (steered && !side && intensity.is_none())
                .then(|| raised_allowance(xr, xmin, block_type, fs, reservoir.noise_offset_db()));
            let quantize = |xmin: &[f64], bits, intensity| {
                loops::quantize_granule(xr, xmin, block_type, bits, fs, reuse, intensity, unscaled)
            };
            let outcome = (raised.as_deref())
                .map(|raised| quantize(raised, bits(0.0), None))
                .filter(|outcome| outcome.within_allowance)
                .unwrap_or_else(|| quantize(xmin, bits(saving_noise_db), intensity));
            if gr == 0 {
                first[channel] = Some((summary, outcome.granule.scalefactors.clone()));
            }
            if fs.version() == Version::Mpeg2 && outcome.granule.scalefactors.marks_any() {
                return None;
            }
            reservoir.granule_done(mean_bits, outcome.granule.bit_len(fs));
            if !outcome.within_allowance && if_outside == IfOutsideAllowance::GiveUp {
                return None;
            }
            within_allowance &= outcome.within_allowance;
            excess = excess.max(loops::excess(xr, xmin, &outcome.granule, fs));
            if excess > 1.0 && if_outside == IfOutsideAllowance::GiveUpBeyond {
                return None;
            }
            quantized.push(outcome.granule);
        }
        granules.push(quantized);
    }
    Some(FrameTry {
        granules,
        within_allowance,
        excess,
        reservoir,
    })
}

/// The share of a band's energy per line that the reservoir's offset raises
/// the band's allowed distortion to at most (see [`raised_allowance`]):
/// 3 dB under the energy.
const RAISED_UNDER_ENERGY: f64 = 0.5;

/// The allowed distortion `xmin` of each scalefactor band of the lines `xr`,
/// in the order they are sent for `block_type`, raised by `offset_db`, but
/// to no more than [`RAISED_UNDER_ENERGY`] of the band's energy per line,
/// the measure of `xmin` and of the distortion. A band the model allows less
/// noise than its energy then cannot meet the raised allowance with all its
/// lines zero; the offset never lowers an allowance.
fn raised_allowance(
    xr: &[f64; GRANULE_LINES],
    xmin: &[f64],
    block_type: BlockType,
    sampling_frequency: SamplingFrequency,
    offset_db: f64,
) -> Vec<f64> {
    let factor = 10f64.powf(offset_db / 10.0);
    let bands = scalefactor_bands::coding_bands(block_type, sampling_frequency);
    (xmin.iter().zip(&bands))
        .map(|(&xmin, band)| {
            let lines = &xr[band.start..band.end()];
            let energy = lines.iter().map(|x| x * x).sum::<f64>() / band.width as f64;
            (factor * xmin).min(RAISED_UNDER_ENERGY * energy).max(xmin)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The offset raises each band's allowance, but not past half its
    /// energy per line, and never lowers it.
    #[test]
    fn the_raised_allowance_stays_under_each_bands_energy() {
        let fs = SamplingFrequency::ALL[0];
        let bands = scalefactor_bands::coding_bands(BlockType::Normal, fs);
        let mut xr = [0.0; GRANULE_LINES];
        for (sb, band) in bands.iter().enumerate() {
            xr[band.start..band.end()].fill(f64::from(sb as u32 + 1));
        }
        let energy = |sb: usize| f64::from(sb as u32 + 1).powi(2);
        // Far under the energy, near it, and above it.
        let xmin: Vec<f64> = (0..bands.len())
            .map(|sb| energy(sb) * [0.01, 0.4, 2.0][sb % 3])
            .collect();
        let raised = raised_allowance(&xr, &xmin, BlockType::Normal, fs, 10.0);
        for sb in 0..bands.len() {
            let expected = match sb % 3 {
                0 => 10.0 * xmin[sb],
                1 => 0.5 * energy(sb),
                _ => xmin[sb],
            };
            assert!(
                (raised[sb] - expected).abs() <= 1e-12 * expected,
                "band {sb}"
            );
        }
    }

    /// How a try at intensity stereo comes out: within its allowed
    /// distortion, outside it, or not coded at all.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum TryOutcome {
        Within,
        Outside,
        Uncodable,
    }

    /// For every pattern of outcomes of tries, up to the most a frame can
    /// have, the search keeps the first try before the last within its
    /// allowed distortion, or else the one with the most bands that can be
    /// coded. It gives up only on tries before the last, finishes tries only
    /// once it has given up on all of those, from the last back, and
    /// quantises none after the one it keeps.
    #[test]
    fn intensity_search_keeps_the_first_try_that_fits_or_else_the_lowest_that_can_be_coded() {
        use TryOutcome::*;
        for count in 0..=stereo::intensity::STEPS {
            for pattern in 0..3usize.pow(count as u32) {
                let outcomes: Vec<_> = (0..count)
                    .map(|index| {
                        [Within, Outside, Uncodable][pattern / 3usize.pow(index as u32) % 3]
                    })
                    .collect();
                let tries: Vec<usize> = (0..count).collect();
                let mut calls = Vec::new();
                let kept = search_intensity(&tries, |&index, if_outside| {
                    calls.push((index, if_outside));
                    match (outcomes[index], if_outside) {
                        (Within, _) | (Outside, IfOutsideAllowance::Finish) => Some(index),
                        _ => None,
                    }
                });

                let earlier = count.saturating_sub(1);
                let within = (0..earlier).find(|&index| outcomes[index] == Within);
                let expected =
                    within.or_else(|| (0..count).rev().find(|&i| outcomes[i] != Uncodable));
                assert_eq!(kept, expected, "{outcomes:?}");
                let gave_up = (0..within.map_or(earlier, |index| index + 1))
                    .map(|index| (index, IfOutsideAllowance::GiveUp));
                let finished = match within {
                    Some(_) => 0..0,
                    None => expected.unwrap_or(0)..count,
                };
                let finished = (finished.rev()).map(|index| (index, IfOutsideAllowance::Finish));
                let expected_calls: Vec<_> = gave_up.chain(finished).collect();
                assert_eq!(calls, expected_calls, "{outcomes:?}");
            }
        }
    }
}
