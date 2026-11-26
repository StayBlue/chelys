// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The encoder's psychoacoustic model: how much noise each line of a long
//! block, and of each of the short blocks that may take its place, may
//! take, from which the floor is set; and whether the short blocks are to
//! take its place, where the signal has an attack.
//!
//! The specification describes no encoder (clause 1.1.5), and this model is
//! chelys's own. It follows the psychoacoustic model of ISO/IEC 13818-7,
//! Annex C (clause C.1), which chelys's AAC encoder implements; the clauses
//! C.1.1 to C.1.4, and the steps of clause C.1.4, cited here are those of
//! ISO/IEC 13818-7, not of the Vorbis I specification. It differs from that
//! model in these:
//!
//! - Its partitions are those of [`partitions`], at any sample rate.
//! - Its long FFT is that of a long block, the samples of the block's
//!   MDCT, and its eight short FFTs those of the eight short blocks that
//!   would take its place, centered on 128 samples apart from the 576th
//!   of the block: each of 256 samples from sample `448 + 128 k`, as in
//!   AAC, and each the samples of its short block's MDCT. Long blocks are
//!   1024 samples apart, as AAC's frames are.
//! - The threshold in quiet of each partition is the level of one of its
//!   lines times its width in Bark, as chelys's AAC encoder reads it at
//!   higher bitrates (`QuietThreshold::Noise` there): half the distance
//!   between the `bval(b)` of its neighbours, or the distance to its one
//!   neighbour at either end. The partitions at the highest `bval(b)` of
//!   the tables, 24, whose width the tables do not give, take
//!   [`TOP_WIDTH_BARK`], and the partition below them its width from its
//!   lower neighbour alone. All are lowered by [`QUIET_LOWERED_DB`].
//! - Step 4 takes the long FFT's unpredictability for the lines below
//!   [`LONG_PREDICTED_HZ`], rather than for its first six lines alone.
//! - Step 8 takes `TMN(b)` as [`TMN`], 15 dB, rather than 18.
//! - Each partition's threshold, scaled by the rate's noise offset, is
//!   raised further with the width of its critical band (see
//!   [`BAND_EXPONENT`]), and above 11 kHz in long blocks further still
//!   (see [`TREBLE_HZ`]).
//! - Nothing past step 13 is taken: the threshold of each partition is
//!   shared evenly among its lines, and becomes the noise each line of the
//!   block's spectrum may take, as noise at the scale of the spectrum
//!   that the decoder takes (see [`Model::analyze`]).
//! - Step 13 finds an attack where the perceptual entropy of the short
//!   blocks rises, as chelys's AAC encoder finds one, rather than from the
//!   long block's (see [`SWITCH_PE`]).
//!
//! The floor is the quantizer's step, set at each post so that rounding to
//! it makes no more noise than the model allows the lines about the post
//! (see [`floor_targets`]).

pub mod partitions;

use std::f64::consts::PI;

use chelys_core::fft::{Complex, RealFft};
use chelys_core::psychoacoustic::spreading_function;

use super::floor::{Floor, INVERSE_DB_TABLE, MULTIPLIER, RANGE};
use super::setup::BLOCKSIZES;
use partitions::{Fft, Partition};

/// The samples of a long block, and of the long FFT.
const BLOCK: usize = BLOCKSIZES[1];

/// The lines of a long block.
pub const LINES: usize = BLOCK / 2;

/// The samples of a short block, and of a short FFT.
const SHORT_BLOCK: usize = BLOCKSIZES[0];

/// The lines of a short block.
pub const SHORT_LINES: usize = SHORT_BLOCK / 2;

/// The short blocks that take the place of a long one: as many as keep
/// the long blocks before and after them where they would be, 1024 samples
/// from it (clause 4.3.8: from one block's center to the next, a quarter
/// of each).
pub const SHORT_BLOCKS: usize = BLOCK / SHORT_BLOCK;

/// The first sample of the first short block in the long block whose place
/// the short blocks take: a short block after a long one is centered
/// `2048 / 4 + 256 / 4` samples after the long one's center (clause 4.3.8),
/// 576 samples into the long block it replaces, which begins 1024 samples
/// before the long one's center.
pub const FIRST_SHORT: usize = BLOCK / 4 + SHORT_BLOCK / 4 - SHORT_BLOCK / 2;

/// The lines of the long FFT whose unpredictability the long FFT gives in
/// ISO/IEC 13818-7, clause C.1.4, step 4; the short FFTs give it above.
/// The model takes the long FFT's for more lines, up to
/// [`LONG_PREDICTED_HZ`], and for these at any sample rate.
const LONG_CW_LINES: usize = 6;

/// The frequency below which the long FFT gives each line's
/// unpredictability, where step 4 takes the short FFTs' above its sixth
/// line.
///
/// chelys's own reading of step 4, for music whose partials the short FFTs
/// do not resolve. A line of a short FFT is eight times as wide as one of
/// the long FFT, 172 Hz at 44.1 kHz, wider than a critical band below about
/// 1.5 kHz and than the spacing of the partials of most pitched sounds.
/// Partials that share a line of it beat, and its phase, unlike theirs, is
/// not predictable: steady partials are found as unpredictable as noise,
/// and allowed noise's noise (step 7), 12 dB more than a tone's, where they
/// are most heard. The long FFT's lines resolve them.
///
/// Graded with PEAQ (GstPEAQ 0.6.1, black box) on the 51 tracks of the
/// tuning set at 44.1 kHz in stereo, at noise offsets matched to 128.4
/// kbit/s, the streams graded -0.537 with the short FFTs' above the sixth
/// line, and -0.514 with the long FFT's below 2 kHz and -0.522 below 4
/// kHz; PEAQ's `EHSB`, which measures the error's harmonic structure, fell
/// from 0.295 to 0.272 below 2 kHz. Taking the lesser of the two for every
/// line, with the noise tilted by 2 dB per octave rather than weighted as
/// [`BAND_EXPONENT`] weighs it, graded -0.532 against -0.542.
const LONG_PREDICTED_HZ: f64 = 2000.0;

/// `NMT(b)` in dB, for all `b` (step 8).
const NMT: f64 = 6.0;

/// `TMN(b)` in dB, for all `b`: 15, where step 8 gives 18.
///
/// chelys's own choice, a deliberate departure from step 8: at the rates
/// chelys is asked for, the noise lies about the threshold, and how far
/// below a tonal partition's energy the threshold lies against a noisy
/// one's is a choice of the encoder, as how far it lies above it in one
/// band against another is (see [`BAND_EXPONENT`]). Graded with PEAQ
/// (GstPEAQ 0.6.1, black box) on the 51 tracks of the tuning set at 44.1
/// kHz in stereo, at noise offsets matched to 96.5, 128.4 and 159.2
/// kbit/s, once the energy of each band was balanced (see the encoder's
/// `energy::balance`), the streams graded -0.892, -0.454 and -0.242 with
/// 15, -0.884, -0.456 and -0.243 with 12, and -0.905, -0.463 and -0.250
/// with 18; and with 24, 0.04 to 0.09 lower than with 18. With 15, PEAQ's
/// `AvgModDiff1B` fell from 19.2, 14.6 and 11.4 to 18.7, 14.2 and 11.3.
const TMN: f64 = 15.0;

/// `rpelev` of long blocks (step 11).
const RPELEV: f64 = 2.0;

/// `rpelev` of short blocks: 2, as of long ones, where step 11 gives
/// short blocks 1 (as chelys's AAC encoder takes it).
///
/// Graded with PEAQ (GstPEAQ 0.6.1, black box) on the 51 tracks of the
/// tuning set at 44.1 kHz in stereo, interpolated between noise offsets
/// to the same mean bitrate of 128 kbit/s, the streams graded -1.159
/// with 2, against -1.164 with 1; and -1.061 against -1.069 on the 12
/// tracks with the most short blocks (see [`SWITCH_PE`]), from 3.1 to
/// 6.1% of the long blocks' places. With 4, they graded -1.161 and
/// -1.061.
const RPELEV_SHORT: f64 = 2.0;

/// How much more noise, in power, a line of a short block may take than
/// the long block whose place it takes allows the lines of its frequencies
/// over as much bandwidth: no line of a short block takes more than 4
/// times, 6 dB, the least noise of the 8 lines of the long block at its
/// frequency, times the 8 for its bandwidth (see [`noise_scale`]).
///
/// The short blocks' FFTs resolve frequency eight times as coarsely as the
/// long block's, and so find the partials of tonal music noise-like, which
/// allows them the noise of noise (step 7); the long block's threshold
/// tells them apart, and is held near the threshold of the block before
/// it at an attack (step 11). Without this limit, the streams graded
/// -1.211 at 128 kbit/s, measured as for [`RPELEV_SHORT`] but with
/// `rpelev` 1 and 15 posts in a short block's floor, against -1.185 with
/// long blocks alone, losing 0.33 to 0.57 on 5 tracks, on one of which
/// PEAQ's `RmsNoiseLoudB` rose from 0.46 to 2.14; with it at 0, 3, 6, 9
/// and 12 dB above,
/// -1.183, -1.175, -1.173, -1.175 and -1.181. Taken as the mean of the 8
/// lines rather than the least, -1.207.
const SHORT_NOISE_ABOVE_LONG: f64 = 4.0;

/// `switch_pe`: a long block's place is taken by short blocks, where the
/// perceptual entropy of one of them rises by more than this above the
/// most of the eight short blocks before it, or by more than
/// [`SWITCH_PE_STEP`] above the one before it (step 13, as chelys's AAC
/// encoder departs from it).
///
/// The criterion is that of chelys's AAC encoder, whose documentation
/// shows why step 13's comparison with the long block's entropy does not
/// tell attacks from steady music, and these two rises do. The entropy is
/// that of step 12, from each short block's threshold before the rate's
/// noise offset scales it, with the threshold in quiet lowered as for
/// coding; its energies are taken at the scale of 16-bit PCM, to which
/// step 12 adds 1 (see [`ENTROPY_UNIT`]).
///
/// Half that of chelys's AAC encoder, whose [`SWITCH_PE_STEP`] is kept.
/// Measured as for [`RPELEV_SHORT`],
/// the streams graded -1.163, -1.158, -1.159 and -1.167 with this at 50,
/// 70, 100 and 140, and [`SWITCH_PE_STEP`] at twice it; and -1.072,
/// -1.062, -1.061 and -1.072 on the 12 tracks with the most short blocks.
/// Long blocks alone graded -1.185, and -1.159 on the 12. Once the encoder
/// coupled the channels and balanced each band's energy (see the encoder's
/// `energy::balance`), at noise offsets matched to 96.5, 128.4 and 159.2
/// kbit/s, the streams graded -0.904, -0.465 and -0.253 with 100, and
/// -0.905, -0.463 and -0.250 with 50, higher on 34 and 50 of the 51 tracks
/// at the two higher rates; PEAQ's `ADBB` fell from 0.31 to 0.29 at 159.2.
/// They graded alike with [`SWITCH_PE_STEP`] at 100 or 200. In one channel
/// at 65.2 and 81.0 kbit/s, -0.552 and -0.230 with 100, and -0.551 and
/// -0.227 with 50 (the step at 100). With 25, and the step at 50, the
/// streams graded lower at every rate.
///
/// Below [`SWITCH_PE_LOW_RATE`], the rise is [`SWITCH_PE_LOW`] instead.
const SWITCH_PE: f64 = 50.0;

/// The sample rate, in Hz, below which [`SWITCH_PE_LOW`] takes the place of
/// [`SWITCH_PE`].
const SWITCH_PE_LOW_RATE: u32 = 32_000;

/// `switch_pe` below [`SWITCH_PE_LOW_RATE`]: that of chelys's AAC encoder.
///
/// A short block lasts longer at a lower sample rate, 256 samples being
/// more than 8 ms below 32 kHz, and its FFT's lines are narrower: a loud
/// partial that wavers in pitch, as with vibrato, moves across them from
/// one short block to the next, and the entropy rises by more than
/// [`SWITCH_PE`] as at an attack. An 880 Hz tone at 0.9 of full scale, its
/// pitch swinging by 2% 5.5 times a second, coded in one channel at a
/// noise offset of -6.5 dB, took short blocks in 399 of 506 packets at 8
/// kHz with [`SWITCH_PE`], and its bitrate rose from 4.0 to 17.9 kbit/s;
/// at 16 kHz, from 8.0 to 15.9; at 32 kHz and above, it took none. Graded
/// with PEAQ (GstPEAQ 0.6.1, black box) on the tuning set converted to
/// 22.05 and 11.025 kHz in stereo, the streams graded 0.04 and 0.05 higher
/// with [`SWITCH_PE`] than with this, but with three times the short
/// blocks, and up to 26% of them on one track; with this, the switching
/// below 32 kHz is that of chelys's earlier streams.
const SWITCH_PE_LOW: f64 = 100.0;

/// The rise of a short block's perceptual entropy above the one before it
/// that is an attack too (see [`SWITCH_PE`]): that of chelys's AAC encoder,
/// four times [`SWITCH_PE`]. At twice it, 100, a steady tone of 80 Hz
/// setting in after silence is found an attack once it has set in.
const SWITCH_PE_STEP: f64 = 200.0;

/// What step 12 adds to each partition's energy, 1 at the scale of 16-bit
/// PCM, at the model's scale of full scale 1.0.
const ENTROPY_UNIT: f64 = 1.0 / (32768.0 * 32768.0);

/// The highest `bval(b)` of the tables, at which those of the highest
/// partitions stop rising.
const TOP_BARK: f64 = 24.0;

/// The width in Bark of each partition at [`TOP_BARK`], from about 15 kHz
/// up, by which the level of one of its lines is multiplied for its
/// threshold in quiet: one critical band. Those partitions span 1 to 3 kHz
/// at 44.1 and 48 kHz, no wider than the critical band below them, of
/// about 3.5 kHz.
///
/// chelys's AAC encoder takes their threshold as printed, the level of a
/// line times their lines, 17 to 19 dB more over their 50 to 80 lines.
/// That takes everything above about 16 kHz out of most music: graded
/// with PEAQ (GstPEAQ 0.6.1, black box), on 30 s of each of 8 tracks of
/// the tuning set, one of each kind of music but live recordings, with
/// the threshold not lowered, the streams' mean grade was -1.496 as
/// printed and -1.387 at one Bark, for 1% more bits.
const TOP_WIDTH_BARK: f64 = 1.0;

/// The width of the narrowest critical bands, in Hz, below about 500 Hz,
/// against which [`BAND_EXPONENT`] weighs each partition's: about one Bark
/// of the tables' lowest partitions.
const NARROWEST_BAND_HZ: f64 = 100.0;

/// How the noise each partition is allowed grows with the width in Hz of
/// its critical band (see [`band_weights`]): as that width, against
/// [`NARROWEST_BAND_HZ`], to this power. At 44.1 kHz the weight is about
/// 1.5 dB at 1 kHz, 6 dB at 4 kHz and 10 dB at 16 kHz, where
/// [`TREBLE_WEIGHT`] adds up to 3 dB more.
///
/// chelys's own design, a deliberate departure from the model's threshold
/// scaled alike in every band. The rate's offset sets how far the noise
/// lies above or below the threshold, and at the rates chelys is asked for
/// it mostly lies about it, where how far it lies above the threshold in
/// one band against another is a choice of the encoder. Each critical
/// band counts alike to how loud the noise is heard, and to PEAQ's measures,
/// which average over bands evenly spaced in Bark; but noise lowered by a
/// dB in a band costs bits in proportion to its lines, and the bands above
/// 500 Hz widen with frequency, to some 35 times the narrowest at 16 kHz.
/// Bits spent where the bands are narrow lower the noise of more bands than
/// the same bits spent where they are wide. Spread so as to make the sum
/// over the bands of their ratios of noise to threshold the least for the
/// bits, each band's noise would grow with its width in lines, to the
/// power 1; the power is lower, as the grades find, as the ratios of the
/// bands where the noise lies furthest above the threshold count for more
/// than their sum.
///
/// Graded with PEAQ (GstPEAQ 0.6.1, black box) on the 51 tracks of the
/// tuning set at 44.1 kHz in stereo, at noise offsets matched to 128.4
/// kbit/s, the streams graded -0.654 unweighted, and -0.538 and -0.537 with
/// powers of 1/2 and 2/3. The gain is the in-band noise's: PEAQ's
/// `TotalNMRB` fell from -9.73 to -10.7 dB, its `AvgModDiff1B` from 20.5 to
/// 14.9 and its `RmsNoiseLoudB` from 0.484 to 0.296, while the bandwidth it
/// measured fell from 853 to 838 lines. With each band's width taken as
/// its lines for each Bark of its own width, the streams graded -0.538,
/// -0.536, -0.543 and -0.574 with powers of 1/2, 2/3, 0.8 and 1; and
/// weighted instead by a tilt of 1.5, 2 and 3 dB per octave above 250 Hz,
/// -0.542, -0.542 and -0.575.
///
/// Once the energy of each band was balanced (see the encoder's
/// `energy::balance`), with [`TMN`] at 15 dB, at noise offsets matched to
/// 96.5, 128.4 and 159.2 kbit/s, the streams graded -0.892, -0.454 and
/// -0.242 with 2/3, -0.900, -0.452 and -0.237 with 3/4, and -0.912, -0.452
/// and -0.239 with 0.8. Each step trades the bandwidth PEAQ measures for
/// lower noise in the band below: with 3/4, its `AvgModDiff1B` fell from
/// 18.7, 14.2 and 11.3 to 18.1, 13.7 and 10.9, and its `RmsNoiseLoudB` from
/// 0.390, 0.279 and 0.216 to 0.372, 0.266 and 0.207, while the bandwidth
/// it measured fell from 817, 839 and 849 lines to 813, 835 and 846. With
/// the energy's balance aimed lower where bits are scarce (see the
/// encoder's `energy::balanced_share`), [`SWITCH_PE`] at 50 and the
/// narrowing where bits are plentiful within the allowance itself, the
/// streams graded -0.890, -0.451 and -0.230 with 2/3, and -0.896, -0.445
/// and -0.222 with 3/4, lower on 38 of the 51 tracks at 96.5 kbit/s: the
/// power stays 2/3.
const BAND_EXPONENT: f64 = 2.0 / 3.0;

/// The frequency above which each partition of a long block is allowed
/// more noise still than [`BAND_EXPONENT`] allows it, rising over
/// [`TREBLE_RAMP_HZ`] to [`treble_weight`] times as much, as the
/// partition's center rises.
///
/// chelys's own design, a deliberate departure from the model's threshold,
/// as [`BAND_EXPONENT`] is: noise there is heard least, and the bits it
/// saves go to the bands below. PEAQ measures little of the noise there
/// but the bandwidth it finds: on 18 tracks of the tuning set coded at 128
/// kbit/s, the coding error above 12 kHz alone, added to the reference,
/// made an `AvgModDiff1B` of 1.0 and a `RmsNoiseLoudB` of 0.02, against
/// 14.0 and 0.28 for all of it, while the lines above 12 kHz took a quarter
/// of the residue's bits. Graded with PEAQ (GstPEAQ 0.6.1, black box) on
/// the 51 tracks of the tuning set at 44.1 kHz in one channel, at noise
/// offsets matched to 65.2 and 81.0 kbit/s, the streams graded -0.536 and
/// -0.209 without it, and -0.493 and -0.156 with [`TREBLE_WEIGHT`], then
/// in short blocks too; PEAQ's `AvgModDiff1B` fell from 13.6 and 10.5 to
/// 12.6 and 9.6.
///
/// In a coupled pair, the narrowing of angles takes a multiple of each
/// channel's allowance as its room for setting the two channels' values
/// alike (see the encoder's `coupling`), and much of the noise allowed
/// here goes into that, which narrows the stereo image (see the coupling's
/// `SIDE_LOSS_DB`), and where bits are scarce costs more than it saves. In
/// stereo, at noise offsets matched to 96.5, 128.4 and 159.2 kbit/s, with
/// the floors matched and the energy kept as the encoder keeps them, the
/// streams graded -0.848, -0.428 and -0.203 without it, -0.882, -0.407 and
/// -0.159 with [`TREBLE_WEIGHT`] at every rate, -0.870, -0.416 and -0.193
/// with it where bits are scarce falling to none where they are plentiful,
/// and -0.847, -0.410 and -0.161 as [`treble_weight`] gives it, falling
/// from [`TREBLE_WEIGHT`] where bits are plentiful to none where they are
/// scarce. With the narrowing's room taken from the allowance without it,
/// the image stayed as without it, and the streams graded -0.897, -0.444
/// and -0.195.
///
/// Short blocks are allowed no more noise there than [`BAND_EXPONENT`]
/// allows them: they code attacks, and more noise in them lies before the
/// attack. With it in short blocks too, white noise setting in after
/// silence, coded in one channel at the rate of the conformance tests,
/// decoded with -49.4 dBFS of noise in the 1024 samples before it, past
/// the -50 that `ffmpeg_decodes_a_loud_onset_after_silence` allows.
const TREBLE_HZ: f64 = 11_000.0;

/// The span of frequency over which the noise allowed above [`TREBLE_HZ`]
/// rises to [`TREBLE_WEIGHT`] times what [`BAND_EXPONENT`] allows.
const TREBLE_RAMP_HZ: f64 = 2_000.0;

/// How many times the noise that [`BAND_EXPONENT`] allows each partition
/// is allowed above [`TREBLE_HZ`] and [`TREBLE_RAMP_HZ`], at most (see
/// [`treble_weight`]).
const TREBLE_WEIGHT: f64 = 2.0;

/// How many times the noise that [`BAND_EXPONENT`] allows each partition
/// of a long block is allowed above [`TREBLE_HZ`] and [`TREBLE_RAMP_HZ`]:
/// [`TREBLE_WEIGHT`] where the channels are coded each on its own; in a
/// coupled pair, as `coupled`, where bits are as scarce as `scarcity`, from
/// 0 where they are plentiful to 1 where they are scarce, from
/// [`TREBLE_WEIGHT`] to 1, none more, in a straight line in dB.
pub fn treble_weight(coupled: bool, scarcity: f64) -> f64 {
    if coupled {
        TREBLE_WEIGHT.powf(1.0 - scarcity.clamp(0.0, 1.0))
    } else {
        TREBLE_WEIGHT
    }
}

/// How far, in dB, the threshold in quiet is lowered below the tables'.
///
/// The tables put it relative to an LSB of 16-bit PCM (ISO/IEC 13818-7,
/// clause C.1.4, step 11), whatever the level the signal is played at.
/// The floor follows the threshold wherever the music lies below it, as
/// in quiet passages, which are then coded with noise just below it, and
/// with no allowance for a louder playback.
///
/// Lowered by 20 dB, as chelys's AAC encoder lowers it at variable rates
/// for the same reason, the 8 clips of [`TOP_WIDTH_BARK`]
/// graded -0.629, against -1.387 unlowered, for 20% more bits; and
/// against -0.850 with the encoder's earlier rule, noise a fixed 9.5 dB
/// below the power about each post of the floor, at the same mean
/// bitrate. On the whole of the tuning set's 51 tracks, at 44.1 kHz in
/// stereo, the model graded -0.578 at 189 kbit/s, and the earlier rule
/// -0.712 at the same size, 9.65 dB below; the model graded higher on 30
/// tracks, by 0.43 to 0.59 on average on electronic music, hip-hop and
/// pop, and lower by 0.03 to 0.15 on the other kinds but rock.
///
/// Lowered further, quiet passages take more bits, and are worth them
/// where the rest of the music is coarsely coded: on the tuning set, at
/// 44.1 kHz in stereo, with codebooks counted as in [`frequencies`] and the
/// encoder's dead zone, interpolated between noise offsets to the same
/// mean bitrate, the streams graded -1.070 at 140 kbit/s lowered by 20 dB,
/// -0.952 by 30, -0.866 by 40, -0.825 by 50 and -0.822 by 60, where no
/// line is allowed less noise than [`absolute_noise`] in any case; and
/// -0.384 at 200 kbit/s lowered by 20 dB, -0.348 by 35 and -0.318 by 50.
/// Where it is finely coded, they are not: about -0.22 at 230 kbit/s by
/// any of them, and -0.004 at 280 kbit/s lowered by 20 dB, -0.060 by 35
/// and -0.077 by 50. It is lowered alike at every offset all the same, as
/// the offset scales the threshold in quiet with the rest: lowered less at
/// lower offsets, by more than the offset falls, quiet passages would be
/// coded with more noise at a higher level of quality.
///
/// [`frequencies`]: super::frequencies
const QUIET_LOWERED_DB: f64 = 50.0;

/// The energy, in the FFT of `n` samples at full scale 1.0, of the line
/// that a sine wave of +-1/2 LSB of 16-bit PCM is centered on, to which
/// the tables' thresholds in quiet are relative (step 11): `a / 2` times
/// the sum of the Hann window, `n / 2`, for `a = 1 / 65536`, squared.
fn quiet_reference(n: usize) -> f64 {
    (n as f64 / 4.0 / 65536.0).powi(2)
}

/// The power of noise in a line of the spectrum of a block of `n` samples
/// that the decoder takes, for each unit of the energy of the same noise in
/// a line of the FFT of the block's samples: white noise of power `P` has
/// the energy `P` times the sum of the squared Hann window, `3 n / 8`, in
/// each line of the FFT, and the power `4 P / n` in each line of the
/// spectrum (see [`absolute_noise`]'s scale).
fn noise_scale(n: usize) -> f64 {
    4.0 / n as f64 / (3.0 * n as f64 / 8.0)
}

/// The least noise power a line of a block of `n` samples is allowed, at
/// the decoder's scale of the spectrum: that of the rounding of 16-bit PCM,
/// white noise of power `1 / (12 * 32768^2)`, some 101 dB below full scale,
/// whose power in each line of the block is `4 / n` times its own.
fn absolute_noise(n: usize) -> f64 {
    4.0 / n as f64 / (12.0 * 32768.0 * 32768.0)
}

/// What the model computes once for each sample rate.
#[derive(Debug)]
pub struct Tables {
    long: FftTables,
    short: FftTables,
    /// The lines of the long FFT whose unpredictability it gives itself
    /// (see [`LONG_PREDICTED_HZ`]).
    long_predicted: usize,
    /// `switch_pe` at the sample rate (see [`SWITCH_PE_LOW`]).
    switch_pe: f64,
    /// The weight of the noise long blocks are allowed in the treble (see
    /// [`treble_weight`]).
    #[cfg_attr(not(test), allow(dead_code))]
    treble_weight: f64,
}

impl Tables {
    /// The tables of `sample_rate`, whose long blocks are allowed
    /// `treble_weight` times the noise above [`TREBLE_HZ`] (see
    /// [`treble_weight`]).
    pub fn new(sample_rate: u32, treble_weight: f64) -> Self {
        let below = (LONG_PREDICTED_HZ * BLOCK as f64 / f64::from(sample_rate)).round();
        Self {
            long: FftTables::new(sample_rate, Fft::Long, treble_weight),
            short: FftTables::new(sample_rate, Fft::Short, 1.0),
            treble_weight,
            long_predicted: (below as usize).clamp(LONG_CW_LINES, LINES),
            switch_pe: if sample_rate < SWITCH_PE_LOW_RATE {
                SWITCH_PE_LOW
            } else {
                SWITCH_PE
            },
        }
    }

    /// The weight of the noise long blocks are allowed above
    /// [`TREBLE_HZ`], as [`new`](Self::new) took it.
    #[cfg(test)]
    pub fn treble_weight(&self) -> f64 {
        self.treble_weight
    }

    /// The partition of each line of a long block, if `long`, or of a short
    /// one: the bands whose lines the model allows the same noise each.
    pub fn partition_of_lines(&self, long: bool) -> Vec<usize> {
        let tables = if long { &self.long } else { &self.short };
        (tables.partitions.iter().enumerate())
            .flat_map(|(b, partition)| partition.lines.clone().map(move |_| b))
            .collect()
    }
}

/// What the model computes once for each sample rate, for the long or the
/// short FFT.
#[derive(Debug)]
struct FftTables {
    partitions: Vec<Partition>,
    /// `sprdngf(bval(bb), bval(b))` at `bb * n + b`, for `n` partitions.
    spreading: Vec<f64>,
    /// `rnorm(b)` (step 6).
    rnorm: Vec<f64>,
    /// The threshold in quiet of each partition as an energy of the FFT.
    quiet: Vec<f64>,
    /// The factor by which each partition's allowance is raised for the
    /// width of its critical band (see [`band_weights`]).
    weights: Vec<f64>,
    /// The samples of the FFT, and of its block.
    samples: usize,
}

impl FftTables {
    fn new(sample_rate: u32, fft: Fft, treble_weight: f64) -> Self {
        let partitions = partitions::partitions(sample_rate, fft);
        let n = partitions.len();
        let mut spreading = vec![0.0; n * n];
        for (bb, from) in partitions.iter().enumerate() {
            for (b, to) in partitions.iter().enumerate() {
                spreading[bb * n + b] = spreading_function(from.bval, to.bval);
            }
        }
        let rnorm = (0..n)
            .map(|b| 1.0 / (0..n).map(|bb| spreading[bb * n + b]).sum::<f64>())
            .collect();
        let samples = 2 * fft.lines();
        Self {
            quiet: quiet(&partitions, quiet_reference(samples)),
            weights: band_weights(
                &partitions,
                f64::from(sample_rate) / samples as f64,
                treble_weight,
            ),
            partitions,
            spreading,
            rnorm,
            samples,
        }
    }

    /// Each partition's values convolved with the spreading function (step
    /// 6).
    fn spread(&self, values: &[f64], out: &mut [f64]) {
        let n = self.partitions.len();
        out.fill(0.0);
        for (bb, &value) in values.iter().enumerate() {
            if value != 0.0 {
                let row = &self.spreading[bb * n..(bb + 1) * n];
                for (out, &sprdngf) in out.iter_mut().zip(row) {
                    *out += value * sprdngf;
                }
            }
        }
    }
}

/// The threshold in quiet of each of `partitions` as an energy of the FFT
/// whose line a sine wave of +-1/2 LSB is centered on has the energy
/// `reference`, as the [module](self) describes.
fn quiet(partitions: &[Partition], reference: f64) -> Vec<f64> {
    (partitions.iter().enumerate())
        .map(|(b, p)| {
            let line = reference * 10f64.powf((p.quiet_db - QUIET_LOWERED_DB) / 10.0);
            line * width_bark(partitions, b).unwrap_or(TOP_WIDTH_BARK)
        })
        .collect()
}

/// The last of `partitions` below [`TOP_BARK`], or the first if none is.
fn last_below_top(partitions: &[Partition]) -> usize {
    partitions
        .iter()
        .position(|p| p.bval >= TOP_BARK)
        .unwrap_or(partitions.len())
        .saturating_sub(1)
}

/// The width in Bark of partition `b` of `partitions`, as the
/// [module](self) describes, or `None` for those at [`TOP_BARK`] and above,
/// whose width the tables do not give.
fn width_bark(partitions: &[Partition], b: usize) -> Option<f64> {
    let bval = |b: usize| partitions[b].bval;
    let last = last_below_top(partitions);
    if b > last || last == 0 {
        return None;
    }
    Some(match b {
        0 => bval(1) - bval(0),
        _ if b == last => bval(last) - bval(last - 1),
        _ => (bval(b + 1) - bval(b - 1)) / 2.0,
    })
}

/// The factor by which each of `partitions`, whose lines are `line_hz` Hz
/// wide, is allowed more noise than the model's threshold scaled by the
/// rate's offset: the width in Hz of the critical band about it, against
/// [`NARROWEST_BAND_HZ`], raised to [`BAND_EXPONENT`], and no less than 1;
/// and above [`TREBLE_HZ`] up to `treble_weight` times that, rising in dB
/// in a straight line over [`TREBLE_RAMP_HZ`] with the partition's center.
/// That width is the Hz between the centers of its neighbours for each
/// Bark between their `bval(b)`, or between its own and its one neighbour's
/// at either end. The partitions at [`TOP_BARK`] and above, at which the
/// tables' Bark values stop rising, take that of the last partition below
/// them.
fn band_weights(partitions: &[Partition], line_hz: f64, treble_weight: f64) -> Vec<f64> {
    let last = last_below_top(partitions);
    let center = |b: usize| {
        let lines = &partitions[b].lines;
        (lines.start + lines.end) as f64 / 2.0 * line_hz
    };
    let band_hz = |b: usize| {
        let (low, high) = (b.saturating_sub(1), (b + 1).min(last));
        let bark = partitions[high].bval - partitions[low].bval;
        (bark > 0.0).then(|| (center(high) - center(low)) / bark)
    };
    (0..partitions.len())
        .map(|b| {
            let treble = ((center(b) - TREBLE_HZ) / TREBLE_RAMP_HZ).clamp(0.0, 1.0);
            let width = band_hz(b.min(last)).map_or(1.0, |hz| {
                (hz / NARROWEST_BAND_HZ).max(1.0).powf(BAND_EXPONENT)
            });
            width * treble_weight.powf(treble)
        })
        .collect()
}

/// The Hann window of 2 `iblen` samples (step 2): `0.5 - 0.5 cos(pi (i +
/// 0.5) / iblen)`.
fn hann(samples: usize) -> Vec<f64> {
    let iblen = (samples / 2) as f64;
    (0..samples)
        .map(|i| 0.5 - 0.5 * (PI * (i as f64 + 0.5) / iblen).cos())
        .collect()
}

/// Step 7: the tonality `tb(b)` of a partition of normalized
/// unpredictability `cb(b)`, within 0 and 1.
fn tonality(cb: f64) -> f64 {
    (-0.299 - 0.43 * cb.ln()).clamp(0.0, 1.0)
}

/// Steps 8 and 9: the power ratio `bc(b)` of a partition of tonality
/// `tb(b)`.
fn power_ratio(tb: f64) -> f64 {
    let snr = tb * TMN + (1.0 - tb) * NMT;
    10f64.powf(-snr / 10.0)
}

/// The magnitude and phase of each line of an FFT's last two blocks, from
/// which steps 3 and 4 predict the next.
#[derive(Debug)]
struct Predictor {
    /// Of each line, the magnitude and `e^(i f)` of the last block and of
    /// the block before it.
    last: Vec<(f64, Complex)>,
    before_last: Vec<(f64, Complex)>,
}

/// A line of the silence the model starts from: no magnitude, and the
/// phase 0 (ISO/IEC 13818-7, clause C.1.1).
const SILENT_LINE: (f64, Complex) = (0.0, Complex { re: 1.0, im: 0.0 });

impl Predictor {
    fn new(lines: usize) -> Self {
        Self {
            last: vec![SILENT_LINE; lines],
            before_last: vec![SILENT_LINE; lines],
        }
    }

    /// The unpredictability `c(w)` of the lines `x` of the next block, into
    /// `c`, and the block kept as the last.
    ///
    /// `c(w) = |r e^(i f) - r_pred e^(i f_pred)| / (r + |r_pred|)`, with
    /// `r_pred = 2 r(t-1) - r(t-2)` and `f_pred = 2 f(t-1) - f(t-2)`. A
    /// line without magnitude takes the phase 0, and one without magnitude
    /// and none predicted is wholly unpredictable, 1, as in chelys's AAC
    /// encoder.
    fn unpredictability(&mut self, x: &[Complex], c: &mut [f64]) {
        for (w, (x, c)) in x.iter().zip(c).enumerate() {
            let ((r1, f1), (r2, f2)) = (self.last[w], self.before_last[w]);
            let r_pred = 2.0 * r1 - r2;
            let conj = Complex {
                re: f2.re,
                im: -f2.im,
            };
            let predicted = f1 * f1 * conj;
            let r = (x.re * x.re + x.im * x.im).sqrt();
            let dre = x.re - r_pred * predicted.re;
            let dim = x.im - r_pred * predicted.im;
            let scale = r + r_pred.abs();
            *c = if scale == 0.0 {
                1.0
            } else {
                (dre * dre + dim * dim).sqrt() / scale
            };
            let phase = if r == 0.0 {
                SILENT_LINE.1
            } else {
                Complex {
                    re: x.re / r,
                    im: x.im / r,
                }
            };
            self.before_last[w] = self.last[w];
            self.last[w] = (r, phase);
        }
    }
}

/// What the model gives for a long block and the short blocks that may take
/// its place: the noise power each of their lines may take, at the scale of
/// the spectrum that the decoder takes, and whether the short blocks are to
/// take its place.
#[derive(Clone, Debug)]
pub struct Analysis {
    /// The noise of each line of the long block.
    pub long: Vec<f64>,
    /// The noise of each line of each short block, in their order.
    pub short: Vec<Vec<f64>>,
    /// The largest rise of a short block's perceptual entropy above the
    /// most of the eight before it, and above the one before it (see
    /// [`SWITCH_PE`]).
    pub entropy_rise: f64,
    pub entropy_step: f64,
    /// The rise of `entropy_rise` that is an attack at the sample rate (see
    /// [`SWITCH_PE_LOW`]).
    pub switch_pe: f64,
}

impl Default for Analysis {
    fn default() -> Self {
        Self {
            long: vec![0.0; LINES],
            short: vec![vec![0.0; SHORT_LINES]; SHORT_BLOCKS],
            entropy_rise: f64::NEG_INFINITY,
            entropy_step: f64::NEG_INFINITY,
            switch_pe: SWITCH_PE,
        }
    }
}

impl Analysis {
    /// Whether the block has an attack, which short blocks are to code
    /// (step 13).
    pub fn attack(&self) -> bool {
        self.entropy_rise > self.switch_pe || self.entropy_step > SWITCH_PE_STEP
    }
}

/// The partitions' values of a block that steps 5 to 11 compute, in
/// buffers kept from one block to the next for their memory.
#[derive(Debug)]
struct Partitioned {
    /// `e(b)`.
    e: Vec<f64>,
    /// The weighted unpredictability `c(b)` (step 5).
    weighted: Vec<f64>,
    ecb: Vec<f64>,
    ct: Vec<f64>,
    /// `nb(b)` of the last block of the FFT, before pre-echo control (step
    /// 11).
    nb_last: Vec<f64>,
}

impl Partitioned {
    fn new(tables: &FftTables) -> Self {
        let n = tables.partitions.len();
        Self {
            e: vec![0.0; n],
            weighted: vec![0.0; n],
            ecb: vec![0.0; n],
            ct: vec![0.0; n],
            nb_last: vec![0.0; n],
        }
    }

    /// Steps 5 to 12 for a block of the FFT of `tables`, from the energy
    /// `r(w)^2` and the unpredictability `c(w)` of each of its lines, with
    /// pre-echo control by `rpelev`: the noise each line of the block may
    /// take, into `noise`, its threshold scaled by `allowance` and shared
    /// as [`Model::analyze`] describes; and the block's perceptual entropy,
    /// from its threshold before the allowance scales it.
    fn threshold(
        &mut self,
        tables: &FftTables,
        energy: &[f64],
        c: &[f64],
        rpelev: f64,
        allowance: f64,
        noise: &mut [f64],
    ) -> f64 {
        // Step 5: each partition's energy and weighted unpredictability.
        for (b, p) in tables.partitions.iter().enumerate() {
            let lines = p.lines.clone();
            self.e[b] = energy[lines.clone()].iter().sum();
            self.weighted[b] = (energy[lines.clone()].iter())
                .zip(&c[lines])
                .map(|(&e, &c)| e * c)
                .sum();
        }

        // Step 6: spread.
        tables.spread(&self.e, &mut self.ecb);
        tables.spread(&self.weighted, &mut self.ct);

        // Steps 7 to 12, each partition's threshold, shared among its
        // lines.
        let scale = noise_scale(tables.samples);
        let least = absolute_noise(tables.samples);
        let mut entropy = 0.0;
        for (b, p) in tables.partitions.iter().enumerate() {
            let ecb = self.ecb[b];
            // Without energy, `cb(b)` is undefined, and the threshold 0
            // whatever the tonality.
            let cb = if ecb == 0.0 { 0.0 } else { self.ct[b] / ecb };
            let nb = ecb * tables.rnorm[b] * power_ratio(tonality(cb));
            let limited = nb.min(self.nb_last[b] * rpelev);
            self.nb_last[b] = nb;
            let thr = tables.quiet[b].max(limited);
            // Step 12, each partition weighed by one less than its lines,
            // as chelys's AAC encoder reads it.
            entropy -= (p.lines.len() - 1) as f64 * (thr / (self.e[b] + ENTROPY_UNIT)).log10();
            let line =
                (thr * allowance * tables.weights[b] / p.lines.len() as f64 * scale).max(least);
            noise[p.lines.clone()].fill(line);
        }
        entropy
    }
}

/// The model of one channel, with what it keeps from one block to the
/// next.
#[derive(Debug)]
pub struct Model {
    /// The factor by which each partition's threshold is scaled, from the
    /// rate's noise offset.
    allowance: f64,
    long_fft: RealFft,
    short_fft: RealFft,
    long_hann: Vec<f64>,
    short_hann: Vec<f64>,
    long_predictor: Predictor,
    short_predictor: Predictor,
    long: Partitioned,
    short: Partitioned,
    /// The perceptual entropy of the last eight short blocks, `None`
    /// before the first.
    recent_entropies: Option<[f64; SHORT_BLOCKS]>,
    // Buffers kept from one block to the next for their memory.
    windowed: Vec<f64>,
    short_windowed: Vec<f64>,
    energy: Vec<f64>,
    short_energy: Vec<Vec<f64>>,
    c: Vec<f64>,
    short_c: Vec<Vec<f64>>,
    short_c_min: Vec<f64>,
}

impl Model {
    /// The model of a channel before its first block, which follows
    /// silence, which allows each partition its threshold scaled by
    /// `10^(noise_offset_db / 10)`.
    pub fn new(tables: &Tables, noise_offset_db: f64) -> Self {
        Self {
            allowance: 10f64.powf(noise_offset_db / 10.0),
            long_fft: RealFft::new(BLOCK),
            short_fft: RealFft::new(SHORT_BLOCK),
            long_hann: hann(BLOCK),
            short_hann: hann(SHORT_BLOCK),
            long_predictor: Predictor::new(tables.long_predicted),
            short_predictor: Predictor::new(SHORT_LINES + 1),
            long: Partitioned::new(&tables.long),
            short: Partitioned::new(&tables.short),
            recent_entropies: None,
            windowed: vec![0.0; BLOCK],
            short_windowed: vec![0.0; SHORT_BLOCK],
            energy: vec![0.0; LINES],
            short_energy: vec![vec![0.0; SHORT_LINES]; SHORT_BLOCKS],
            c: vec![0.0; LINES],
            short_c: vec![vec![0.0; SHORT_LINES + 1]; SHORT_BLOCKS],
            short_c_min: vec![0.0; SHORT_LINES + 1],
        }
    }

    /// Analyzes the long block of `samples`, at full scale 1.0, and the
    /// short blocks that may take its place, into `out`; each call takes the
    /// block 1024 samples after the last.
    ///
    /// Each partition's threshold, an energy of its FFT, is scaled by the
    /// model's allowance, shared evenly among its lines, and taken as
    /// noise: each line's share times [`noise_scale`] of the block. Line
    /// `k` of the spectrum, centered at `k + 1/2` lines of the FFT, takes
    /// the share of line `k` of the FFT. No line takes less than
    /// [`absolute_noise`].
    ///
    /// The thresholds of the long and of the short blocks are computed for
    /// every block, as the two processes of ISO/IEC 13818-7, clause C.1.1,
    /// each with its shift length, run over the whole signal, as
    /// chelys's AAC encoder computes them: each short block's threshold is
    /// limited by the last short block's (step 11), and each short block's
    /// unpredictability predicted from the two before it, whatever the
    /// blocks the encoder codes.
    pub fn analyze(&mut self, tables: &Tables, samples: &[f64], out: &mut Analysis) {
        assert_eq!(samples.len(), BLOCK);

        // The short blocks, steps 2 to 4 for each, and for the long block
        // the least unpredictability of each of their lines over the eight
        // (step 4).
        self.short_c_min.fill(f64::INFINITY);
        for k in 0..SHORT_BLOCKS {
            let start = FIRST_SHORT + k * SHORT_BLOCK / 2;
            let short = &samples[start..start + SHORT_BLOCK];
            for ((s, &x), &w) in self
                .short_windowed
                .iter_mut()
                .zip(short)
                .zip(&self.short_hann)
            {
                *s = x * w;
            }
            let x = self.short_fft.transform(&self.short_windowed);
            for (energy, x) in self.short_energy[k].iter_mut().zip(x) {
                *energy = x.re * x.re + x.im * x.im;
            }
            self.short_predictor
                .unpredictability(x, &mut self.short_c[k]);
            for (min, &c) in self.short_c_min.iter_mut().zip(&self.short_c[k]) {
                *min = min.min(c);
            }
        }

        // Steps 2 to 4: the long FFT, the energy of its lines, and their
        // unpredictability, each line from `LONG_PREDICTED_HZ` on taking
        // that of the short FFTs' line nearest it in frequency, as in
        // chelys's AAC encoder.
        for ((s, &x), &w) in self.windowed.iter_mut().zip(samples).zip(&self.long_hann) {
            *s = x * w;
        }
        let x = self.long_fft.transform(&self.windowed);
        for (energy, x) in self.energy.iter_mut().zip(x) {
            *energy = x.re * x.re + x.im * x.im;
        }
        let predicted = tables.long_predicted;
        self.long_predictor
            .unpredictability(&x[..predicted], &mut self.c[..predicted]);
        for (w, c) in self.c.iter_mut().enumerate().skip(predicted) {
            *c = self.short_c_min[((w + SHORT_BLOCKS / 2) / SHORT_BLOCKS).min(SHORT_LINES)];
        }

        // Steps 5 to 11.
        self.long.threshold(
            &tables.long,
            &self.energy,
            &self.c,
            RPELEV,
            self.allowance,
            &mut out.long,
        );

        // Steps 5 to 12 for each short block.
        let mut entropies = [0.0; SHORT_BLOCKS];
        for (k, (noise, entropy)) in out.short.iter_mut().zip(&mut entropies).enumerate() {
            *entropy = self.short.threshold(
                &tables.short,
                &self.short_energy[k],
                &self.short_c[k],
                RPELEV_SHORT,
                self.allowance,
                noise,
            );
            // No line takes more noise than the long block allows the
            // lines of its frequencies (see `SHORT_NOISE_ABOVE_LONG`).
            for (j, noise) in noise.iter_mut().enumerate() {
                let lines = &out.long[j * SHORT_BLOCKS..(j + 1) * SHORT_BLOCKS];
                let least = lines.iter().copied().fold(f64::INFINITY, f64::min);
                *noise = noise.min(least * SHORT_BLOCKS as f64 * SHORT_NOISE_ABOVE_LONG);
            }
        }

        // Step 13: each short block rises from the most of the eight before
        // it and from the one before it, the first of the stream from
        // itself.
        let mut recent = self
            .recent_entropies
            .unwrap_or([entropies[0]; SHORT_BLOCKS]);
        let mut last = recent[SHORT_BLOCKS - 1];
        out.switch_pe = tables.switch_pe;
        out.entropy_rise = f64::NEG_INFINITY;
        out.entropy_step = f64::NEG_INFINITY;
        for (k, &entropy) in entropies.iter().enumerate() {
            let most = recent.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            out.entropy_rise = out.entropy_rise.max(entropy - most);
            out.entropy_step = out.entropy_step.max(entropy - last);
            recent[k] = entropy;
            last = entropy;
        }
        self.recent_entropies = Some(recent);
    }
}

/// The Y value of each post of `floor`, in its list order, whose floor
/// rounds each line about the post to no more than the noise `noise` gives
/// the line: rounding to a step `s` makes noise of power `s^2 / 12`. The
/// lines about a post are those nearer it than any other post, and it is
/// set for the least noise any of them may take, as the floor between two
/// posts follows a straight line in dB from one to the other.
pub fn floor_targets(noise: &[f64], floor: &Floor) -> Vec<i32> {
    floor
        .post_lines(noise.len())
        .into_iter()
        .map(|lines| {
            let least = noise[lines].iter().copied().fold(f64::INFINITY, f64::min);
            y_value((12.0 * least).sqrt())
        })
        .collect()
}

/// The Y value whose floor is nearest `amplitude`, in dB.
fn y_value(amplitude: f64) -> i32 {
    let lowest = f64::from(INVERSE_DB_TABLE[0]);
    let step = (f64::from(INVERSE_DB_TABLE[255]) / lowest).ln() / 255.0;
    let index = (amplitude / lowest).ln() / step / f64::from(MULTIPLIER);
    (index.round() as i32).clamp(0, RANGE - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn y_values_follow_the_table() {
        for y in 0..RANGE {
            let amplitude = f64::from(INVERSE_DB_TABLE[(y * MULTIPLIER as i32) as usize]);
            assert_eq!(y_value(amplitude), y);
        }
        assert_eq!(y_value(1e-12), 0);
        assert_eq!(y_value(1e6), RANGE - 1);
    }

    #[test]
    fn each_line_has_a_partition_in_order() {
        for sample_rate in [8_000, 44_100, 48_000, 96_000, 192_000] {
            let tables = Tables::new(sample_rate, 1.0);
            for (long, lines) in [(true, LINES), (false, SHORT_LINES)] {
                let partitions = tables.partition_of_lines(long);
                assert_eq!(partitions.len(), lines, "{sample_rate} Hz");
                assert_eq!(partitions[0], 0);
                assert!(
                    partitions
                        .windows(2)
                        .all(|w| w[1] == w[0] || w[1] == w[0] + 1)
                );
            }
        }
    }

    /// The weights are about 1 where the critical bands are narrowest, rise
    /// with frequency, and are alike in a long and a short block at the
    /// same frequency.
    #[test]
    fn band_weights_rise_with_the_width_of_the_critical_bands() {
        for sample_rate in [8_000, 22_050, 44_100, 48_000, 96_000] {
            let tables = Tables::new(sample_rate, 1.0);
            for fft in [&tables.long, &tables.short] {
                assert!(fft.weights[0] < 1.1, "{sample_rate} Hz");
                // The tables' Bark values are rounded, and some partitions,
                // as the last of a table, are narrower than the one below.
                let mut most: f64 = 1.0;
                for &weight in &fft.weights {
                    assert!(weight >= most / 2.0, "{sample_rate} Hz");
                    most = most.max(weight);
                }
            }
        }
        let tables = Tables::new(44_100, TREBLE_WEIGHT);
        let db_at = |fft: &FftTables, hz: f64| {
            let line = (hz * fft.samples as f64 / 44_100.0) as usize;
            let b = (fft.partitions.iter())
                .position(|p| p.lines.contains(&line))
                .unwrap();
            10.0 * fft.weights[b].log10()
        };
        assert!(db_at(&tables.long, 200.0) < 0.5);
        // Above 13 kHz, the treble's weight too, in long blocks.
        let treble = 10.0 * TREBLE_WEIGHT.log10();
        for (hz, long_db, short_db) in [(4_000.0, 6.0, 6.0), (16_000.0, 10.0 + treble, 10.0)] {
            for (fft, db) in [(&tables.long, long_db), (&tables.short, short_db)] {
                let weight = db_at(fft, hz);
                assert!((weight - db).abs() < 1.5, "{hz} Hz: {weight:.1} dB");
            }
        }
    }

    #[test]
    fn a_coupled_pair_takes_less_noise_in_the_treble_as_bits_grow_scarce() {
        for scarcity in [-1.0, 0.0, 0.5, 1.0] {
            assert_eq!(treble_weight(false, scarcity), TREBLE_WEIGHT);
        }
        assert_eq!(treble_weight(true, 0.0), TREBLE_WEIGHT);
        assert_eq!(treble_weight(true, -1.0), TREBLE_WEIGHT);
        assert_eq!(treble_weight(true, 1.0), 1.0);
        assert!((treble_weight(true, 0.5) - TREBLE_WEIGHT.sqrt()).abs() < 1e-12);
        // Where bits are scarce, a pair's treble is weighed as below it.
        let (plain, coupled) = (
            Tables::new(44_100, 1.0),
            Tables::new(44_100, treble_weight(true, 1.0)),
        );
        assert_eq!(plain.long.weights, coupled.long.weights);
    }

    #[test]
    fn the_short_blocks_lie_where_they_would_take_a_long_blocks_place() {
        // Centered from 576 to 1472: 576 after the center of the long block
        // before, at 0, and 576 before that of the long block after, at
        // 2048.
        assert_eq!(FIRST_SHORT + SHORT_BLOCK / 2, 576);
        let last = FIRST_SHORT + (SHORT_BLOCKS - 1) * SHORT_BLOCK / 2;
        assert_eq!(last + SHORT_BLOCK / 2 + 576, BLOCK);
    }

    /// The analyses of the first `blocks` blocks of `signal`.
    fn analyses(sample_rate: u32, signal: impl Fn(usize) -> f64, blocks: usize) -> Vec<Analysis> {
        let tables = Tables::new(sample_rate, 1.0);
        let mut model = Model::new(&tables, 0.0);
        (0..blocks)
            .map(|k| {
                let samples: Vec<f64> = (0..BLOCK).map(|t| signal(k * LINES + t)).collect();
                let mut out = Analysis::default();
                model.analyze(&tables, &samples, &mut out);
                out
            })
            .collect()
    }

    fn analyze(sample_rate: u32, signal: impl Fn(usize) -> f64, blocks: usize) -> Analysis {
        analyses(sample_rate, signal, blocks).pop().unwrap()
    }

    #[test]
    fn silence_gets_the_threshold_in_quiet() {
        let analysis = analyze(44_100, |_| 0.0, 3);
        let tables = Tables::new(44_100, 1.0);
        for (tables, noise) in std::iter::once((&tables.long, &analysis.long))
            .chain(analysis.short.iter().map(|noise| (&tables.short, noise)))
        {
            let scale = noise_scale(tables.samples);
            let least = absolute_noise(tables.samples);
            for (b, p) in tables.partitions.iter().enumerate() {
                let expected =
                    (tables.quiet[b] * tables.weights[b] / p.lines.len() as f64 * scale).max(least);
                assert!(noise[p.lines.clone()].iter().all(|&n| n == expected));
            }
        }
        assert!(!analysis.attack());
    }

    /// The threshold of the partition holding `line`, against its energy
    /// in the long FFT, in dB, in the last of `blocks` blocks of `signal`.
    fn margin_db(signal: impl Fn(usize) -> f64, blocks: usize, line: usize) -> f64 {
        let tables = Tables::new(44_100, 1.0);
        let mut model = Model::new(&tables, 0.0);
        let mut out = Analysis::default();
        for k in 0..blocks {
            let samples: Vec<f64> = (0..BLOCK).map(|t| signal(k * LINES + t)).collect();
            model.analyze(&tables, &samples, &mut out);
        }
        let partitions = &tables.long.partitions;
        let b = partitions
            .iter()
            .position(|p| p.lines.contains(&line))
            .unwrap();
        let thr = out.long[line] * partitions[b].lines.len() as f64
            / noise_scale(BLOCK)
            / tables.long.weights[b];
        10.0 * (thr / model.long.e[b]).log10()
    }

    #[test]
    fn a_steady_tone_is_masked_less_than_noise() {
        let white = white(8 * BLOCK);
        let line = 93;
        let noise_db = margin_db(|t| white[t], 6, line);
        assert!((-9.0..-3.0).contains(&noise_db), "{noise_db:.1} dB");
        let f = (line as f64 + 0.25) / BLOCK as f64;
        let tone_db = margin_db(|t| 0.5 * (2.0 * PI * f * t as f64).sin(), 6, line);
        assert!(
            tone_db < noise_db - 8.0,
            "{tone_db:.1} against {noise_db:.1} dB"
        );
    }

    /// Two steady partials closer than a line of the short FFTs, below
    /// `LONG_PREDICTED_HZ`, are found tonal, as one is.
    #[test]
    fn partials_that_share_a_short_line_are_tonal() {
        let white = white(8 * BLOCK);
        let line = 23;
        let noise_db = margin_db(|t| white[t], 6, line);
        let partial = |line: f64, t: usize| (2.0 * PI * line / BLOCK as f64 * t as f64).sin();
        let pair_db = margin_db(|t| 0.3 * (partial(23.25, t) + partial(27.25, t)), 6, line);
        assert!(
            pair_db < noise_db - 8.0,
            "{pair_db:.1} against {noise_db:.1} dB"
        );
    }

    #[test]
    fn a_loud_block_after_silence_is_held_near_the_silence() {
        let onset = 3 * LINES + LINES / 2;
        let loud = |t: usize| {
            if t >= onset {
                0.5 * (t as f64 * 0.3).sin()
            } else {
                0.0
            }
        };
        let held = analyze(44_100, loud, 3);
        let steady = analyze(44_100, |t| 0.5 * (t as f64 * 0.3).sin(), 6);
        let line = (0.3 / (2.0 * PI) * BLOCK as f64) as usize;
        assert!(held.long[line] < steady.long[line] / 100.0);
        // The first short block in which it sets in is held near the one
        // before.
        let short = (0.3 / (2.0 * PI) * SHORT_BLOCK as f64) as usize;
        let first = (onset - 2 * LINES + 1 - (FIRST_SHORT + SHORT_BLOCK)).div_ceil(SHORT_BLOCK / 2);
        assert_eq!(first, 7);
        assert!(held.short[first][short] < steady.short[first][short] / 100.0);
    }

    /// Uniform noise from -0.5 to 0.5, the same each run.
    fn white(len: usize) -> Vec<f64> {
        let mut state = 1u32;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                f64::from(state >> 8) / f64::from(1 << 24) - 0.5
            })
            .collect()
    }

    #[test]
    fn steady_tones_and_noise_have_no_attack() {
        let noise = white(20 * LINES);
        for sample_rate in [8_000, 11_025, 16_000, 22_050, 44_100, 48_000] {
            let fs = f64::from(sample_rate);
            let tone = |f: f64, t: usize| (2.0 * PI * f * t as f64 / fs).sin();
            // 880 Hz, its pitch swinging by 2% 5.5 times a second.
            let vibrato = |t: usize| {
                let t = t as f64 / fs;
                let swing = 880.0 * 0.02 / (2.0 * PI * 5.5);
                (2.0 * PI * (880.0 * t - swing * (2.0 * PI * 5.5 * t).cos())).sin()
            };
            let signal = |name: &str, t: usize| match name {
                "80 Hz" => 0.5 * tone(80.0, t),
                "1 kHz" => 0.5 * tone(1000.0, t),
                "noise" => 0.5 * noise[t],
                "vibrato" => 0.9 * vibrato(t),
                _ => 0.3 * tone(440.0, t) + 0.05 * noise[t],
            };
            for name in ["80 Hz", "1 kHz", "noise", "tone in noise", "vibrato"] {
                // The first block has none, after the silence before it.
                let signal = |t| signal(name, t);
                for (k, analysis) in analyses(sample_rate, signal, 18).iter().enumerate().skip(1) {
                    assert!(
                        !analysis.attack(),
                        "{name} at {sample_rate} Hz, block {k}: {} and {}",
                        analysis.entropy_rise,
                        analysis.entropy_step
                    );
                }
            }
        }
    }

    /// A click over a quiet tone is an attack of the block whose short
    /// blocks span it, wherever it lies among them, and of no other.
    #[test]
    fn a_click_is_an_attack_of_the_block_whose_short_blocks_span_it() {
        let blocks = 8;
        // The short blocks of block 5 are centered from its 576th sample,
        // 5 * 1024 into the signal, to its 1472nd; those of the blocks
        // beside it span up to the first, and from the last.
        let first = FIRST_SHORT + SHORT_BLOCK / 2;
        let span = 5 * LINES + first..5 * LINES + BLOCK - first;
        for click in span.step_by(61) {
            let signal = |t: usize| {
                let tone = 0.01 * (2.0 * PI * 440.0 * t as f64 / 44_100.0).sin();
                if t == click { tone + 0.5 } else { tone }
            };
            let attacks: Vec<bool> = analyses(44_100, signal, blocks)
                .iter()
                .map(Analysis::attack)
                .collect();
            let mut expected = vec![false; blocks];
            expected[5] = true;
            assert_eq!(attacks, expected, "a click at {click}");
        }
    }

    #[test]
    fn each_post_is_set_for_the_least_noise_about_it() {
        let floor = Floor::long(0);
        let noise: Vec<f64> = (0..LINES).map(|k| 1e-6 * (1.0 + k as f64)).collect();
        let targets = floor_targets(&noise, &floor);
        let mut sorted: Vec<u32> = floor.xs().to_vec();
        sorted.sort_unstable();
        for (&x, &y) in floor.xs().iter().zip(&targets) {
            let i = sorted.binary_search(&x).unwrap();
            let low = if i == 0 {
                0
            } else {
                (sorted[i - 1] + x).div_ceil(2) as usize
            };
            assert_eq!(
                y,
                y_value((12.0 * noise[low.min(LINES - 1)]).sqrt()),
                "post {x}"
            );
        }
    }
}
