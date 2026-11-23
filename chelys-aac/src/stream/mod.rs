// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The frames of a stream: one per block of 1024 samples of each channel,
//! each as long as the bit reservoir lets it be, or at a variable rate as
//! its noise within its allowance takes, framed as asked.

mod element;
mod frame;

use super::DELAY;
use super::adts::{self, Flags};
use super::audio_specific_config::audio_specific_config;
use super::channel::CodedChannel;
use super::channel_configuration::{ChannelConfiguration, ElementKind};
use super::config::Framing;
use super::noise_substitution;
use super::psychoacoustic::{Quiet, QuietThreshold};
use super::rate::{self, Control};
use super::raw_data_block;
use super::raw_data_block::data_stream::{self, DataStreamQueue};
use super::raw_data_block::element::CodedElement;
use super::reservoir::{self, BitReservoir};
use super::sampling_frequency::SamplingFrequency;
use super::stereo::intensity;
use super::stereo::middle_side::MsMask;
use super::window::{WindowSequence, Windows};
use element::ElementEncoder;
use frame::FrameRate;

/// The samples of each channel a frame codes (clause 8.2.1.1).
pub const FRAME_SAMPLES: usize = 1024;

/// The most bits for each sample of each channel at which a pair is coded
/// in intensity stereo (see [`intensity::code`]).
///
/// Spec gap: clause C.6.2 codes the bands above `f0` in intensity stereo,
/// "one possible implementation" among several, and does not say when.
/// Intensity stereo gives up the right channel's waveform in those bands
/// for the bits it saves, which pays only where bits are few. Graded with
/// PEAQ (GstPEAQ 0.6.1, black box), on 8 tracks of pop music at 44.1 kHz
/// in stereo and on 3 tracks of rock music and their 3 windows of 3 s
/// with the most frames coded short, with M/S stereo, intensity stereo
/// from 6 kHz gave means of -3.711 and -3.611 at 48 kbit/s against -3.754
/// and -3.636 without, and -3.648 and -3.596 at 64 kbit/s against -3.703
/// and -3.579. Before bands where a channel is silent and frames of short
/// windows were left out of it, which changed no grade at 64 kbit/s by
/// more than 0.001, it gave -3.163 and -2.497 at 96 kbit/s against -2.992
/// and -2.166, -1.965 and -1.403 at 128 kbit/s against -1.485 and -0.938,
/// and -0.622 and -0.355 at 192 kbit/s against -0.411 and -0.197; and at
/// 64 kbit/s from 9 and 12 kHz, -3.657 and -3.591, and -3.673 and -3.591.
/// The limit lies between 64 and 96 kbit/s, 0.73 and 1.09 bits a sample;
/// it is set per sample, untested at other sampling frequencies, so as to
/// hold where a band's lines are as few for its bits. Below about 16 kHz
/// sampling, the border of 6 kHz leaves few bands above it or none, so the
/// limit matters little there. At 88.2 and 96 kHz, where a frame's bits
/// are spread over a band twice as wide, the limit is generous, and
/// untested.
pub const INTENSITY_BITS_PER_SAMPLE: f64 = 0.75;

/// The most bits for each sample of each channel at which channels are
/// coded by perceptual noise substitution, where the stream codes them so
/// (see [`noise_substitution::noise_like`]).
///
/// Spec gap: Annex 4.B.11 of ISO/IEC 14496-3 does not say when. Perceptual
/// noise substitution gives up a band's waveform for the bits it saves,
/// which pays only where bits are few. Graded as for the limits of
/// [`noise_substitution::noise_like`] (PEAQ, black box, with the encoder's
/// bandwidth cut alike with and without it, at 10, 13, 14 and 15 kHz at
/// 64, 96, 112 and 128 kbit/s), the 10 excerpts on which they were chosen
/// gained 0.226 at 112 kbit/s, 9 of them, and lost 0.129 at 128 kbit/s,
/// all but one; 10 others, of other tracks of the same albums, gained
/// 0.300, 0.477 and 0.253 at 64, 96 and 112 kbit/s, every one at each.
/// The limit lies between 112 and 128 kbit/s, 1.27 and 1.45 bits a
/// sample, set per sample as for [`INTENSITY_BITS_PER_SAMPLE`], untested
/// at other sampling frequencies.
///
/// With the bandwidth uncut, PEAQ grades it far higher, as it fills with
/// noise the treble that is otherwise lost, which PEAQ's measures of
/// bandwidth weigh heavily: the 10 other excerpts graded -2.117, -0.960 and
/// -0.805 at 64, 96 and 112 kbit/s, against -3.564, -2.451 and -1.615
/// without it, their bandwidth 20.7 to 20.9 kHz against 12.3, 15.0 and
/// 15.9 kHz. That gain is not counted here: on the excerpts whose measures
/// were examined, PEAQ's measures of noise grew with it.
///
/// Of 5 synthetic signals of 15 s, with the bandwidth uncut, its tonal
/// ones lost 0.15 at most at 64, 96 and 112 kbit/s: the chord of
/// [`noise_substitution::FLATNESS_LIMIT`], a harpsichord of plucked
/// harmonic notes and castanets of clicks; its noise-like ones gained,
/// cymbals of decaying noise by 0.82, 2.78 and 2.84, and sibilants of
/// noise bursts over a tone by 0.61, 0.95 and 0.31.
///
/// The grade at 128 kbit/s, and those that chose the limits of
/// [`noise_substitution::noise_like`], were taken while a pair's noise
/// that is to be the same in both channels was sent with `ms_used`, which
/// ffmpeg, which decoded them, takes as noise of each channel's own (see
/// `ElementEncoder::noise_substitution`); sent in intensity stereo, as it
/// is, the music's grades moved by 0.02 at most.
pub const NOISE_BITS_PER_SAMPLE: f64 = 1.3;

/// The most bits for each sample of each channel at which the model takes
/// the threshold in quiet as printed; above, it takes it as noise has it
/// (see [`StreamConfig::quiet_threshold`]).
pub const PRINTED_QUIET_BITS_PER_SAMPLE: f64 = 0.8;

/// What the stream is: what its headers say, its rate and its framing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreamConfig {
    pub sampling_frequency: SamplingFrequency,
    pub channel_configuration: ChannelConfiguration,
    /// The stream's bitrate, ADTS headers included: at a constant rate the
    /// rate, at a mean rate the mean, and at a variable rate the nominal
    /// rate of its offset (see [`rate::nominal_bits_per_sample`]). It
    /// keys the tools that pay only where bits are few.
    pub bits_per_second: u32,
    /// How the frames' lengths are chosen.
    pub control: Control,
    pub framing: Framing,
    /// The flags of each ADTS header; raw frames have none.
    pub flags: Flags,
    /// Whether each ADTS frame has an `adts_error_check()`; raw frames
    /// have none.
    pub crc_protected: bool,
    /// Whether channels are coded by perceptual noise substitution, at
    /// rates of up to [`NOISE_BITS_PER_SAMPLE`], which makes the stream
    /// MPEG-4 AAC (see [`noise_substitution::noise_like`]).
    pub noise_substitution: bool,
}

impl StreamConfig {
    /// The header of each frame in ADTS.
    pub fn adts_header(&self) -> Option<adts::Header> {
        (self.framing == Framing::Adts).then_some(adts::Header {
            sampling_frequency: self.sampling_frequency,
            channel_configuration: self.channel_configuration,
            flags: self.flags,
            protected: self.crc_protected,
            mpeg4: self.noise_border().is_some(),
        })
    }

    /// The bytes each frame adds before its `raw_data_block()`: its header
    /// and `adts_error_check()` in ADTS.
    pub fn header_len(&self) -> usize {
        self.adts_header().map_or(0, |header| header.bytes())
    }

    /// The shortest frame of channels with `windows` that carries
    /// `data_bytes` of data streams, as the encoder writes it: its channels
    /// silent, without fill, a pair sharing its `ics_info()` where its
    /// channels share their attacks, and so their windows, in every frame
    /// (see [`element::SHARED_ATTACKS`]), and each with its own otherwise,
    /// and a low frequency enhancement channel with long windows, its only
    /// ones (see [`CodedElement::lfe`]).
    fn shortest_frame_len(&self, windows: Windows, data_bytes: usize) -> usize {
        let fs = self.sampling_frequency;
        let channel = CodedChannel::silent(windows, fs);
        let configuration = self.channel_configuration;
        let elements: Vec<CodedElement> = (configuration.elements().iter())
            .map(|kind| match kind {
                ElementKind::Single => CodedElement::single(channel.clone()),
                ElementKind::Pair if element::SHARED_ATTACKS => {
                    CodedElement::common_pair(channel.clone(), channel.clone(), MsMask::NONE)
                }
                ElementKind::Pair => CodedElement::pair(channel.clone(), channel.clone()),
                ElementKind::Lfe => CodedElement::lfe(CodedChannel::silent(Windows::LONG_SINE, fs)),
            })
            .collect();
        self.header_len() + raw_data_block::min_len(configuration, &elements, data_bytes)
    }

    /// The lowest bitrate that brings each frame the bits of the shortest,
    /// that of long windows, so that the reservoir never runs below empty.
    /// A frame that cannot afford the longer `ics_info()` of eight short
    /// windows is coded with long ones.
    pub fn min_bits_per_second(&self) -> u32 {
        let bits = 8 * self.shortest_frame_len(Windows::LONG_SINE, 0) as u64;
        let sample_rate = u64::from(self.sampling_frequency.hz());
        (bits * sample_rate).div_ceil(FRAME_SAMPLES as u64) as u32
    }

    /// The most bytes of data streams that each frame can carry: as many as
    /// the shortest frame that carries them, that of long windows, keeps
    /// within the whole bytes the rate brings each frame, so that the
    /// reservoir never runs below empty (see [`Self::min_bits_per_second`]).
    /// 0 if the stream is not [valid](Self::is_valid).
    pub fn max_data_stream_bytes(&self) -> usize {
        let mean = u64::from(self.bits_per_second) * FRAME_SAMPLES as u64
            / (8 * u64::from(self.sampling_frequency.hz()));
        let fits = |bytes| self.shortest_frame_len(Windows::LONG_SINE, bytes) as u64 <= mean;
        // The frame grows with the bytes.
        let (mut fitting, mut over) = (0, mean as usize + 1);
        if !fits(fitting) {
            return 0;
        }
        while over - fitting > 1 {
            let mid = fitting + (over - fitting) / 2;
            if fits(mid) {
                fitting = mid;
            } else {
                over = mid;
            }
        }
        fitting
    }

    /// The bits of the shortest frame of eight short windows, those of a
    /// low frequency enhancement channel long, without data streams, which
    /// a frame must be able to take with its data streams to be coded with
    /// them (see [`StreamEncoder::affords_short`]).
    fn shortest_short_frame_bits(&self) -> u32 {
        let windows = Windows {
            sequence: WindowSequence::EIGHT_SHORT,
            ..Windows::LONG_SINE
        };
        8 * self.shortest_frame_len(windows, 0) as u32
    }

    /// The longest frame in bytes: [`reservoir::MAX_ADTS_FRAME_LEN`] in
    /// ADTS, and none but the buffer's for raw frames.
    fn max_frame_len(&self) -> Option<usize> {
        (self.framing == Framing::Adts).then_some(reservoir::MAX_ADTS_FRAME_LEN)
    }

    /// The highest bitrate, at which the reservoir holds a byte, and a
    /// mean frame is a byte shorter than the longest at least (see
    /// [`BitReservoir::new`]).
    pub fn max_bits_per_second(&self) -> u32 {
        BitReservoir::max_bits_per_second(
            self.sampling_frequency,
            self.channel_configuration,
            self.max_frame_len(),
        )
    }

    /// The bitrate of [`Control::Variable`]'s offset: the nominal bits a
    /// sample of [`rate::nominal_bits_per_sample`] for each channel of the
    /// NCC, within the range of bitrates.
    pub fn nominal_bits_per_second(&self, noise_offset_db: f64) -> u32 {
        let ncc = self.channel_configuration.ncc() as f64;
        let fs = f64::from(self.sampling_frequency.hz());
        let nominal = rate::nominal_bits_per_sample(noise_offset_db) * ncc * fs;
        let (min, max) = (self.min_bits_per_second(), self.max_bits_per_second());
        (nominal.round().clamp(0.0, f64::from(u32::MAX)) as u32).clamp(min, max.max(min))
    }

    /// The longest frame in bytes at a variable or mean rate: as long as
    /// the decoder's input buffer, or [`reservoir::MAX_ADTS_FRAME_LEN`] in
    /// ADTS where that is shorter.
    ///
    /// Spec reading: "For variable bitrate channels the encoder must
    /// operate in a way that the input buffer requirements do not exceed
    /// the minimum decoder input buffer" (clause 8.2.3.2), and in a variable
    /// rate, demand read channel "there is no minimum input data
    /// requirement prior to the start of decoding" (clause 8.2.4): a decoder
    /// that reads each frame as it decodes it holds one frame, its headers
    /// among its bits (clause 8.2.3.1). So each frame is no longer than the
    /// buffer, and each element no longer than its share of it (see
    /// [`reservoir::element_max_bits`]), as at a constant rate.
    ///
    /// Settled by decoding with ffmpeg and FAAD2 (black box): both decode
    /// streams at variable and mean rates, each header's
    /// `adts_buffer_fullness` 0x7FF, without complaint, among them frames as
    /// long as the buffer, 768 bytes, in mono at 8 kHz; above 24 kHz,
    /// FAAD2 decodes them to within 134 dB of what ffmpeg does. The
    /// conformance test is `ffmpeg_and_faad_decode_variable_and_mean_rates`.
    pub fn max_variable_frame_len(&self) -> usize {
        let buffer = reservoir::buffer_bits(self.channel_configuration) as usize / 8;
        self.max_frame_len().map_or(buffer, |len| buffer.min(len))
    }

    pub(super) fn reservoir(&self) -> Option<BitReservoir> {
        if self.bits_per_second < self.min_bits_per_second() {
            return None;
        }
        BitReservoir::new(
            self.sampling_frequency,
            self.channel_configuration,
            self.bits_per_second,
            self.max_frame_len(),
        )
    }

    /// The lower border frequency above which a pair is coded in intensity
    /// stereo, if it is: Annex C's (see [`intensity::BORDER_HZ`]), where
    /// the stream has at most [`INTENSITY_BITS_PER_SAMPLE`] bits for each
    /// sample of each channel of the NCC.
    ///
    /// At a variable or mean rate, the border falls from half the sampling
    /// frequency to Annex C's as the noise offset rises over
    /// [`rate::INTENSITY_DB`], or at a mean rate the offset of its nominal
    /// rate, evenly in the logarithm of the frequency, and stays at Annex
    /// C's above.
    ///
    /// Spec gap: within their allowance, as at a variable rate, a pair's
    /// bands in intensity stereo can take more bits than coded apart, as
    /// their signal is allowed the less of the two channels' noise (see
    /// [`intensity::code`]): switched on at one nominal rate, intensity
    /// stereo made one second of noise, partly correlated between the
    /// channels, 19% longer from 6.80 to 6.85 dB, against the fall of the
    /// length with the offset. Brought in a band at a time as the offset
    /// rises, it lengthens a stream by a band's bits at most at each step
    /// (see `the_length_falls_with_the_offset_without_a_break`). Of music it
    /// shortens streams, and pays: graded with PEAQ (GstPEAQ 0.6.1, black
    /// box), on the tuning set at 64 kbit/s, the mean was -3.018 with it
    /// and -3.084 without, 44 of 51 tracks better with it.
    pub fn intensity_border(&self) -> Option<f64> {
        let Some(noise_offset_db) = self.variable_offset_db() else {
            return (self.bits_per_sample() <= INTENSITY_BITS_PER_SAMPLE)
                .then_some(intensity::BORDER_HZ);
        };
        let (low, high) = rate::INTENSITY_DB.into_inner();
        let top = f64::from(self.sampling_frequency.hz()) / 2.0;
        match (noise_offset_db - low) / (high - low) {
            ..=0.0 => None,
            1.0.. => Some(intensity::BORDER_HZ),
            share => Some(top * (intensity::BORDER_HZ / top).powf(share)),
        }
    }

    /// The noise offset of a variable rate, or that of a mean rate's
    /// nominal rate (see [`rate::nominal_noise_offset_db`]), by which the
    /// tools that pay only where bits are few are chosen; `None` at a
    /// constant rate.
    fn variable_offset_db(&self) -> Option<f64> {
        match self.control {
            Control::Constant => None,
            Control::Variable { noise_offset_db } => Some(noise_offset_db),
            Control::Mean => Some(rate::nominal_noise_offset_db(self.bits_per_sample())),
        }
    }

    /// The lower border frequency above which channels are coded by
    /// perceptual noise substitution, if they are: Annex 4.B.11's (see
    /// [`noise_substitution::BORDER_HZ`]), where the stream codes them so
    /// and has at most [`NOISE_BITS_PER_SAMPLE`] bits for each sample of
    /// each channel of the NCC.
    fn noise_border(&self) -> Option<f64> {
        (self.noise_substitution && self.bits_per_sample() <= NOISE_BITS_PER_SAMPLE)
            .then_some(noise_substitution::BORDER_HZ)
    }

    /// How the model reads the threshold in quiet: as noise has it (see
    /// [`QuietThreshold::Noise`]) where the stream has more than
    /// [`PRINTED_QUIET_BITS_PER_SAMPLE`] bits for each sample of each
    /// channel of the NCC, and as printed where it has that many or fewer,
    /// as 5.1 at 48 kHz and 192 kbit/s has exactly. With it, the
    /// loops raise `xmin` by the offset of clause C.7.4.4 only where it is
    /// read as printed (see [`raise_xmin`](crate::loops::raise_xmin)).
    ///
    /// Spec gap: Annex C reads the threshold in quiet one way at every rate;
    /// the reading as noise departs from it (see
    /// [`QuietThreshold::Noise`]), and where bits are few it does not pay.
    /// Graded with PEAQ (GstPEAQ 0.6.1, black box), on 51 tracks of 9 kinds
    /// of music (rock, pop, hip-hop, electronic, acoustic, piano, orchestral,
    /// analog recordings from 1966 to 1978, and live) at 44.1 kHz in stereo,
    /// the means were, with the threshold as printed and `xmin` raised by
    /// [`XMIN_OFFSET_DB`](crate::loops::XMIN_OFFSET_DB), 3 dB; as noise with
    /// that offset; as noise without it; and as noise with `xmin` lowered by
    /// 3 dB:
    ///
    /// | kbit/s | printed, 3 dB | noise, 3 dB | noise, 0 dB | noise, -3 dB |
    /// |-------:|--------------:|------------:|------------:|-------------:|
    /// |     64 |        -3.357 |      -3.389 |      -3.391 |       -3.393 |
    /// |    128 |        -1.149 |      -1.101 |      -1.067 |       -1.088 |
    /// |    192 |        -0.222 |      -0.212 |      -0.211 |       -0.234 |
    ///
    /// Those grades were taken with the partition below the cap of `bval(b)`
    /// as wide as the distance to the capped value above it, and before a
    /// silent channel left with lines by bands coded as middle and side was
    /// coded silent (see `ElementEncoder::find_silence`). As coded now, the
    /// means are -3.357, -1.074 and -0.213 at 64, 128 and 192 kbit/s. At
    /// 128 kbit/s, every kind of music gained, analog recordings the most,
    /// 0.187, and their worst track rose from -2.472 to -1.681. At 96 and
    /// 80 kbit/s, the means are -2.332 and -3.071, against -2.500 and
    /// -3.082 as printed, 47 and 25 tracks of the 51 better. At 64 kbit/s,
    /// 0.73 bits a sample, the reading as noise loses, and the loss lies in
    /// PEAQ's bandwidth of the coding, which rises with the reading: near
    /// the bottom of its scale, PEAQ's network of BS.1387 grades a wider
    /// band of coarse noise lower. The limit lies between 64 and 80 kbit/s,
    /// 0.73 and 0.91 bits a sample; it is set per sample, as for
    /// [`INTENSITY_BITS_PER_SAMPLE`], untested at other sampling
    /// frequencies.
    ///
    /// The partitions at 24 Bark keep the threshold as printed (see
    /// [`QuietThreshold::Noise`]). Without any threshold in quiet there,
    /// with the offset of 3 dB, the means were -3.465, -0.832 and -0.254 at
    /// 64, 128 and 192 kbit/s, the bandwidth PEAQ measured rising from 16.2
    /// to 19.1 kHz at 128 kbit/s: content above 15 kHz that the threshold
    /// in quiet takes as inaudible, coded for PEAQ's measures of bandwidth.
    ///
    /// At a variable or mean rate, the reading goes from the one to the
    /// other as the noise offset rises over [`rate::BLENDED_QUIET_DB`], or
    /// at a mean rate the offset of its nominal rate (see
    /// [`rate::nominal_noise_offset_db`]), between the two in dB (see
    /// [`QuietThreshold::Between`]); the loops raise `xmin` by as much of
    /// the offset of clause C.7.4.4, and the noise offset is lowered by as
    /// much of [`rate::PRINTED_QUIET_SHIFT_DB`].
    ///
    /// Spec gap: the threshold in quiet as printed, with the offset to
    /// `xmin`, allows far more noise in the bands that the threshold
    /// governs than the threshold as noise has it, and in every band 3 dB
    /// more: at 6 dB, the tuning set took 38 kbit/s with it and 68 kbit/s
    /// without. At a variable rate, the reading switched at one nominal
    /// rate, as at a constant rate, would make a stream's length jump where
    /// the noise offset passes the one whose nominal rate it is, and leave
    /// the rates between unreachable; lowering the offset as it switched,
    /// to keep the mean of the tuning set's rates level, still made the
    /// length of one second of noise jump by 55% from 6.10 to 6.11 dB.
    /// Taken between the two as the offset rises, and the offset lowered
    /// alike, the allowance of every band rises with the offset without a
    /// break, and the length falls with it; the test
    /// `the_length_falls_with_the_offset_without_a_break` sweeps it.
    ///
    /// The reading as printed is kept where the nominal rate is low, as at
    /// a constant rate. Graded with PEAQ (GstPEAQ 0.6.1, black box), on the
    /// tuning set, with each frame quantized within its allowance (before
    /// the loops fitted each channel to the bits that takes), at 64.3
    /// kbit/s as printed the mean was -3.154, at 64.9 kbit/s as noise has
    /// it -3.269, and at a constant 64 kbit/s -3.356. Taken between the
    /// two, with intensity stereo brought in alike (see
    /// [`Self::intensity_border`]), the means at 64.1 and 80.3 kbit/s were
    /// -3.020 and -2.552, against -3.012 and -2.546 at 64.1 and 81.0
    /// kbit/s switched at 0.8 bits a sample with the offset lowered.
    pub fn quiet_threshold(&self) -> QuietThreshold {
        if let Some(noise_offset_db) = self.variable_offset_db() {
            let (low, high) = rate::BLENDED_QUIET_DB.into_inner();
            return match (noise_offset_db - low) / (high - low) {
                ..=0.0 => QuietThreshold::Noise,
                1.0.. => QuietThreshold::Printed,
                share => QuietThreshold::Between(share),
            };
        }
        if self.bits_per_sample() > PRINTED_QUIET_BITS_PER_SAMPLE {
            QuietThreshold::Noise
        } else {
            QuietThreshold::Printed
        }
    }

    /// How far, in dB, the model lowers the threshold in quiet below the cap
    /// of `bval(b)` (see [`Quiet`](crate::psychoacoustic::Quiet)): at a
    /// variable or mean rate, by [`rate::QUIET_LOWERING_DB`] where the noise
    /// offset, or at a mean rate the offset of its nominal rate, is at the
    /// lower end of [`rate::LOWERED_QUIET_DB`] or below, by none at its upper
    /// end or above, and in dB evenly between; at a constant rate, by none.
    ///
    /// Spec gap: Annex C puts the threshold in quiet at a level relative to
    /// an LSB of 16-bit PCM (step 11), and does not say at what level the
    /// signal is played. At a constant rate, the bits the rate brings code
    /// what lies below it where the music leaves them spare; at a variable
    /// rate, a frame takes only the bits its allowance needs, and a quiet or
    /// sparse passage, whose bands the threshold in quiet governs, takes
    /// few. PEAQ (GstPEAQ 0.6.1, black box), which takes full scale to be
    /// played at 92 dB, louder than the threshold assumes, heard what those
    /// frames left out: on the tuning set of 51 tracks of 9 kinds of music
    /// at 44.1 kHz in stereo, the variable rate whose streams averaged 192
    /// kbit/s graded -0.272, against -0.211 at a constant 192 kbit/s, and a
    /// quiet piano track -1.781 at 73 kbit/s, its bandwidth as PEAQ
    /// measured it 6.9 kHz.
    ///
    /// Lowered by 20 dB at offsets up to 0 dB, and by none from 6 dB, the
    /// variable rate graded -0.110 at 192.1 kbit/s and -0.958 at 128.2
    /// kbit/s, against -0.272 and -1.053 at 192.2 and 127.7 kbit/s
    /// unlowered, and better than at constant 192 and 128 kbit/s on 44 and
    /// 38 tracks of the 51; and the piano track -0.197. At 96 and 80 kbit/s it graded -1.962 and -2.534,
    /// against -1.969 and -2.552 unlowered, and -2.328 and -3.069 at
    /// constant rates; at 64 kbit/s, about 7.2 dB, it is not lowered, and
    /// graded -3.030 at 63.7 kbit/s, against -3.356 at a constant rate. A
    /// mean rate of 128 and 192 kbit/s graded -1.020 and -0.142, against
    /// -1.094 and -0.228 unlowered. Lowered by 10 dB, the variable rate
    /// graded -0.149 and -0.972 at 193.0 and 128.5 kbit/s; by 30 dB, -0.107
    /// and -0.926 at 192.3 and 129.6 kbit/s; by 20 dB from -4 dB to none
    /// from 4 dB, -0.105 at 192.7 kbit/s and -0.973 at 128.6 kbit/s, the
    /// less lowered at 128 kbit/s the lower. Lowered by 20 dB at every
    /// offset, an earlier variable rate lost 0.18 at 64 kbit/s, where bits
    /// are few, as the reading of the threshold as noise loses at constant
    /// rates there (see [`Self::quiet_threshold`]). Those grades of 80 and
    /// 96 kbit/s, and of other lowerings, were taken before the nominal
    /// rates were measured again with the threshold lowered (see
    /// [`rate::nominal_bits_per_sample`]), which sets each frame's least
    /// bits, and before the bits the loops leave were spent (see
    /// [`spend`](crate::loops::allowance::spend)), which raised the mean
    /// grades swept from -15 to 12 dB by 0.010 at most, and lowered none.
    ///
    /// The partitions from the cap of `bval(b)` up, from about 15 kHz, keep
    /// their threshold: lowered too, the variable rate graded -0.062 at
    /// 192.4 kbit/s and -0.919 at 128.8 kbit/s, the bandwidth PEAQ measured
    /// at 192 kbit/s rising from 16.0 to 17.0 kHz, content above 15 kHz
    /// that the threshold takes as inaudible coded for PEAQ's measures of
    /// bandwidth, as without the threshold there (see
    /// [`Self::quiet_threshold`]).
    ///
    /// The lowering falls with the offset, so that the allowance of a band
    /// that the threshold in quiet governs rises with the offset, by 4.3 dB
    /// for each dB over [`rate::LOWERED_QUIET_DB`], and a stream's length
    /// falls with it without a break; as it falls fastest where the stream
    /// is quiet, the test `the_length_falls_with_the_offset_without_a_break`
    /// sweeps quiet noise in steps of 0.1 dB.
    pub fn quiet_lowered_db(&self) -> f64 {
        self.variable_offset_db()
            .map_or(0.0, rate::quiet_lowered_db)
    }

    /// The bits for each sample of each channel of the NCC.
    pub fn bits_per_sample(&self) -> f64 {
        let ncc = self.channel_configuration.ncc() as f64;
        f64::from(self.bits_per_second) / (ncc * f64::from(self.sampling_frequency.hz()))
    }

    /// Whether every frame of a stream so configured can be written.
    pub fn is_valid(&self) -> bool {
        self.reservoir().is_some()
    }
}

/// Where an [`Encoder`](crate::Encoder) gives the frames it codes, one call each, in their
/// order.
pub trait FrameSink {
    /// Takes a frame: an `adts_frame()` or a `raw_data_block()`, as
    /// [`Config::framing`](crate::Config::framing) asked.
    fn frame(&mut self, frame: &[u8]);
}

/// Appends each frame, as to a stream in ADTS.
impl FrameSink for Vec<u8> {
    fn frame(&mut self, frame: &[u8]) {
        self.extend_from_slice(frame);
    }
}

/// Calls the closure with each frame, as to keep raw frames apart:
///
/// ```
/// # use chelys_aac::{Config, Encoder, Framing, Rate};
/// let config = Config::new(48_000, 1, Rate::Constant { bits_per_second: 64_000 })
///     .framing(Framing::Raw);
/// let mut encoder = Encoder::new(config).unwrap();
/// let mut frames = Vec::new();
/// let mut out = |frame: &[u8]| frames.push(frame.to_vec());
/// encoder.encode_interleaved(&[0i16; 4096], &mut out).unwrap();
/// encoder.finish(&mut out);
/// assert_eq!(frames.len(), 5);
/// ```
impl<F: FnMut(&[u8])> FrameSink for F {
    fn frame(&mut self, frame: &[u8]) {
        self(frame);
    }
}

/// Codes a stream from PCM as it arrives, in any lengths, a frame for each
/// block of [`FRAME_SAMPLES`] of each channel, once the block after it has
/// arrived, from which the frame's windows are chosen.
#[derive(Debug)]
pub struct StreamEncoder {
    config: StreamConfig,
    rate: FrameRate,
    /// The elements of each frame, which `channel_configuration` gives, in
    /// order.
    elements: Vec<ElementEncoder>,
    /// Whether frames may be coded with short windows.
    block_switching: bool,
    /// [`StreamConfig::shortest_short_frame_bits`].
    shortest_short_frame_bits: u32,
    /// The bytes queued for the frames' data streams.
    data: DataStreamQueue,
    /// The samples each channel was given.
    samples: u64,
    frames: u64,
}

/// What a [`StreamEncoder`] coded, once it has finished.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamEnd {
    pub frames: u64,
    /// The samples each channel was given.
    pub samples: u64,
    /// The bytes of data streams that the last frames had no room for,
    /// dropped.
    pub data_stream_dropped: usize,
}

impl StreamEncoder {
    /// # Panics
    ///
    /// If the configuration is not [valid](StreamConfig::is_valid).
    pub fn new(config: StreamConfig) -> Self {
        let quiet = config.quiet_threshold();
        Self::with_quiet(config, quiet.lowered(config.quiet_lowered_db()))
    }

    /// [`new`](Self::new), the models taking the threshold in quiet as
    /// `quiet` asks.
    fn with_quiet(config: StreamConfig, quiet: Quiet) -> Self {
        let rate = FrameRate::new(&config)
            .unwrap_or_else(|| panic!("a stream that cannot be written: {config:?}"));
        let elements = (config.channel_configuration.elements().iter())
            .map(|&kind| {
                ElementEncoder::new(
                    config.sampling_frequency,
                    kind,
                    config.intensity_border(),
                    config.noise_border(),
                    quiet,
                    config.control.is_variable(),
                )
            })
            .collect();
        Self {
            config,
            rate,
            elements,
            block_switching: true,
            shortest_short_frame_bits: config.shortest_short_frame_bits(),
            data: DataStreamQueue::default(),
            samples: 0,
            frames: 0,
        }
    }

    /// The encoder, with each frame carrying up to `bytes_per_frame` bytes
    /// of those [pushed](Self::push_data_stream) in data stream elements,
    /// which take bits from its audio.
    ///
    /// # Panics
    ///
    /// If `bytes_per_frame` is above
    /// [`StreamConfig::max_data_stream_bytes`]: a caller's error.
    pub fn with_data_stream(mut self, bytes_per_frame: usize) -> Self {
        let max = self.config.max_data_stream_bytes();
        assert!(
            bytes_per_frame <= max,
            "{bytes_per_frame} bytes of data streams a frame, not at most {max}"
        );
        self.data = DataStreamQueue::new(bytes_per_frame);
        self
    }

    /// Queues `bytes` for the data streams of the frames coded from now on.
    pub fn push_data_stream(&mut self, bytes: &[u8]) {
        self.data.push(bytes);
    }

    /// The bytes queued that no frame has carried yet.
    pub fn data_stream_pending(&self) -> usize {
        self.data.pending()
    }

    /// The encoder, but coding every frame with long windows, as if no
    /// frame had an attack, for comparing with block switching in tests.
    #[cfg(any(test, feature = "unstable"))]
    pub fn without_block_switching(mut self) -> Self {
        self.block_switching = false;
        self
    }

    /// The encoder, but without temporal noise shaping, for comparing with
    /// it in tests.
    #[cfg(any(test, feature = "unstable"))]
    pub fn without_tns(mut self) -> Self {
        for element in &mut self.elements {
            element.without_tns();
        }
        self
    }

    /// The encoder, but with temporal noise shaping in frames of short
    /// windows and the start and stop windows around them too, as clause
    /// C.5 shapes each window, for tests (see `ElementEncoder`'s temporal
    /// noise shaping).
    #[cfg(any(test, feature = "unstable"))]
    pub fn with_tns_around_attacks(mut self) -> Self {
        for element in &mut self.elements {
            element.with_tns_around_attacks();
        }
        self
    }

    /// The encoder, but with spectrum clipping on or off, for comparing
    /// the two in tests.
    #[cfg(any(test, feature = "unstable"))]
    pub fn spectrum_clipping(mut self, on: bool) -> Self {
        for element in &mut self.elements {
            element.spectrum_clipping(on);
        }
        self
    }

    /// The encoder, but with the noise of a pair's bands that is to be the
    /// same in both channels sent as noise in both, with `ms_used` set, as
    /// clause 4.6.13.3 of ISO/IEC 14496-3 has it, rather than in intensity
    /// stereo in the right channel, for comparing the two in tests (see
    /// `ElementEncoder::noise_substitution`).
    #[cfg(feature = "unstable")]
    pub fn noise_shared_by_ms_used(mut self) -> Self {
        for element in &mut self.elements {
            element.noise_shared_by_ms_used();
        }
        self
    }

    /// Whether the next frame's bits, those the reservoir makes available
    /// to it when it asks for none beyond the mean, or offers it whatever it
    /// asks for (see [`BitReservoir::offered_bits`]), pay for the shortest
    /// frame of eight short windows in every channel but a low frequency
    /// enhancement channel, with `data_bits` of data stream elements, and
    /// so for any windows: a frame is given no fewer when it asks for more.
    /// Near the lowest bitrate they may not, as it is set by the frames of
    /// long windows (see [`StreamConfig::min_bits_per_second`]).
    fn affords_short(&self, data_bits: u32) -> bool {
        self.shortest_short_frame_bits + data_bits <= self.rate.most_bits()
    }

    /// The `AudioSpecificConfig()` of ISO/IEC 14496-3 that describes the
    /// stream.
    pub fn audio_specific_config(&self) -> Vec<u8> {
        audio_specific_config(
            self.config.sampling_frequency,
            self.config.channel_configuration,
        )
    }

    /// The sampling frequency in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.config.sampling_frequency.hz()
    }

    /// The bits of the decoder's input buffer (clause 8.2.3.1), within
    /// which the bit reservoir keeps the stream.
    pub fn decoder_buffer_bits(&self) -> u32 {
        reservoir::buffer_bits(self.config.channel_configuration)
    }

    /// Takes `channels`, the next samples of each channel at full scale
    /// 1.0, in the order of the elements that carry them (see
    /// [`ChannelConfiguration::elements`]), and gives `out` the frames they
    /// complete, each once the block after it has arrived.
    ///
    /// The samples are to be finite and within
    /// [`MAX_MAGNITUDE`](chelys_core::sample::MAX_MAGNITUDE), as
    /// [`Encoder`](crate::Encoder) converts them; they are not checked.
    ///
    /// # Panics
    ///
    /// If there is not one slice for each channel, all of one length: a
    /// caller's error.
    pub fn encode<O: FrameSink + ?Sized>(&mut self, channels: &[&[f64]], out: &mut O) {
        assert_eq!(
            channels.len(),
            self.config.channel_configuration.channels(),
            "one slice per channel"
        );
        let len = channels[0].len();
        assert!(
            channels.iter().all(|c| c.len() == len),
            "every channel has the same number of samples"
        );
        let mut rest = channels;
        for element in &mut self.elements {
            let (own, after) = rest.split_at(element.channels());
            element.push(own);
            rest = after;
        }
        self.samples += len as u64;
        self.encode_buffered(out);
    }

    /// The first element, whose buffers and models are in step with the
    /// others'.
    fn first(&self) -> &ElementEncoder {
        &self.elements[0]
    }

    /// Codes the frames whose blocks, and the block after each, are
    /// buffered, and drops their samples.
    fn encode_buffered<O: FrameSink + ?Sized>(&mut self, out: &mut O) {
        if !self.first().is_primed() && self.first().buffered() >= FRAME_SAMPLES {
            for element in &mut self.elements {
                element.prime();
            }
        }
        let mut start = 0;
        while self.first().is_primed() && self.first().buffered() - start >= 2 * FRAME_SAMPLES {
            self.encode_frame(start, out);
            start += FRAME_SAMPLES;
        }
        for element in &mut self.elements {
            element.drain(start);
        }
    }

    /// Ends the stream: pads the input with silence to whole frames, at
    /// least [`DELAY`] samples past it, so that the decoded output holds
    /// all of the input, and gives `out` the frames left. The block after
    /// the last frame, from which its windows are chosen, is silence too.
    pub fn finish<O: FrameSink + ?Sized>(mut self, out: &mut O) -> StreamEnd {
        let frames = (self.samples + DELAY as u64).div_ceil(FRAME_SAMPLES as u64);
        // Few: those of the samples held and of DELAY, and the block after
        // the last.
        let remaining = (frames - self.frames) as usize;
        for element in &mut self.elements {
            element.pad((remaining + 1) * FRAME_SAMPLES);
        }
        self.encode_buffered(out);
        debug_assert_eq!(self.frames, frames);
        StreamEnd {
            frames: self.frames,
            samples: self.samples,
            data_stream_dropped: self.data.pending(),
        }
    }

    /// Codes the frame whose block starts at `start` in the samples
    /// buffered, the block after it following, and gives it to `out`, in
    /// four steps: has the elements' psychoacoustic models analyze the
    /// block after it, for the attacks that choose the frame's windows;
    /// prepares the elements' channels for quantization (see
    /// [`ElementEncoder::prepare`]); has the bit reservoir make bits
    /// available to the frame for the bits the models ask for; and
    /// quantizes the elements in them, less those of the data the frame
    /// carries (see [`Self::quantize`]). At a variable or mean rate, the
    /// elements take the bits that their noise within its allowance, raised
    /// by the offset, takes, up to the longest frame, and the frame is
    /// quantized again in its least bits where it takes fewer (see
    /// [`FrameRate::least_bits`]).
    fn encode_frame<O: FrameSink + ?Sized>(&mut self, start: usize, out: &mut O) {
        for element in &mut self.elements {
            element.look_ahead(start + FRAME_SAMPLES, self.block_switching);
        }
        let data = self.data.take_next_frame();
        let data_bits = data_stream::bits(data.len());
        let afford_short = self.affords_short(data_bits);
        let noise_factor = self.rate.noise_factor();
        for element in &mut self.elements {
            element.set_noise_factor(noise_factor);
            element.prepare(start, afford_short);
        }
        let demand = self.elements.iter().map(ElementEncoder::demand).sum();
        let available_bits = frame::elements_bits(&self.config, &self.rate, &demand, data_bits);
        let mut elements = self.quantize(available_bits);
        if let Some(least) = self.rate.least_bits() {
            let overhead = frame::frame_overhead_bits(&self.config, data_bits);
            if bits(&elements) + overhead < least {
                for element in &mut self.elements {
                    element.set_within_allowance(false);
                }
                elements = self.quantize(least - overhead);
                for element in &mut self.elements {
                    element.set_within_allowance(true);
                }
            }
        }
        frame::write(&self.config, &mut self.rate, &elements, &data, out);
        self.frames += 1;
    }

    /// Quantizes the frame's elements, prepared, in `available_bits`.
    ///
    /// No element is given more than its share of the decoder's input
    /// buffer (see [`reservoir::element_max_bits`]); what it cannot take
    /// goes to those quantized after it.
    ///
    /// Spec gap: Annex C does not say how the bits of a frame are shared
    /// between its elements, any more than between its channels (see
    /// [`ElementEncoder::quantize`]). Each element is given the fewest bits
    /// it can be coded in (see [`ElementEncoder::least_bits`]), and the
    /// bits beyond them are shared by the elements in proportion to the
    /// channels each codes with lines, so that each channel has about an
    /// equal share, as within an element. The elements are quantized one
    /// after another, each in its share of the bits the elements before it
    /// left, so that what one leaves goes to those quantized after it: a
    /// low frequency enhancement channel first, which sends a few lines
    /// and leaves most of its share, then the others in order of their
    /// perceptual entropy (see [`ElementEncoder::perceptual_entropy`]), the
    /// least first, as the channels of an element are.
    ///
    /// Carried forward alone, the bits an element leaves can come too late:
    /// an element quantized early in a share too small for any line is
    /// coded without lines, while one after it codes its own in the bits
    /// the first left, and the frame ends with bits unspent. In 4 channels
    /// at 48 kHz and 11 kbit/s with the CRC, the center channel carrying
    /// the same tone of 79 Hz as the rear surround so decoded to silence
    /// throughout, the rear surround at its level. An element with lines
    /// to code that its share left without any is therefore quantized
    /// again, once, after all the others, in its bits and those the frame
    /// has left, within its share of the buffer. With one element, the
    /// frame's bits are its share, and the element is not quantized again.
    fn quantize(&mut self, available_bits: u32) -> Vec<CodedElement> {
        let mut order: Vec<usize> = (0..self.elements.len()).collect();
        order.sort_by(|&a, &b| {
            let key = |e: usize| {
                let element = &self.elements[e];
                (
                    element.kind() != ElementKind::Lfe,
                    element.perceptual_entropy(),
                )
            };
            let (a, b) = (key(a), key(b));
            a.0.cmp(&b.0).then(a.1.total_cmp(&b.1))
        });
        let mut coded: Vec<Option<CodedElement>> = vec![None; self.elements.len()];
        let mut budgets = vec![0; self.elements.len()];
        let mut remaining = available_bits;
        for (k, &e) in order.iter().enumerate() {
            let rest = &order[k..];
            let element = &self.elements[e];
            let budget = if rest.len() == 1 {
                remaining
            } else {
                let least: u32 = rest.iter().map(|&i| self.elements[i].least_bits()).sum();
                let channels: u32 = (rest.iter())
                    .map(|&i| self.elements[i].coded_channels())
                    .sum();
                let beyond = u64::from(remaining.saturating_sub(least));
                let share = (beyond * u64::from(element.coded_channels()))
                    .checked_div(u64::from(channels))
                    .unwrap_or(0);
                element.least_bits() + share as u32
            };
            let budget =
                reservoir::element_max_bits(element.kind()).map_or(budget, |max| budget.min(max));
            let element = self.elements[e].quantize(budget);
            remaining = remaining.saturating_sub(element.bits());
            budgets[e] = budget;
            coded[e] = Some(element);
        }
        // An element with lines to code that its share left without any is
        // quantized again, once, with the bits the frame has left.
        for &e in &order {
            let element = coded[e].as_ref().expect("quantized");
            let silenced = self.elements[e].coded_channels() > 0
                && (element.channels().iter()).all(|c| c.lines().iter().all(|&q| q == 0));
            let budget = element.bits() + remaining;
            let budget =
                reservoir::element_max_bits(element.kind()).map_or(budget, |max| budget.min(max));
            // Within its allowance, as at a variable rate, it takes no more
            // than it did.
            if silenced && budget > budgets[e] && !self.config.control.is_variable() {
                remaining += element.bits();
                let element = self.elements[e].quantize(budget);
                remaining = remaining.saturating_sub(element.bits());
                coded[e] = Some(element);
            }
        }
        coded.into_iter().map(|e| e.expect("quantized")).collect()
    }
}

/// The bits of a frame's elements.
fn bits(elements: &[CodedElement]) -> u32 {
    elements.iter().map(CodedElement::bits).sum()
}

/// A stream of `frames` silent frames, each apart, for tests: the
/// frames of as many blocks of silence, and of the block after them.
#[cfg(any(test, feature = "unstable"))]
pub fn encode_silence(config: StreamConfig, frames: usize) -> Vec<Vec<u8>> {
    let silence = vec![0.0; (frames + 1) * FRAME_SAMPLES];
    let channels = vec![silence.as_slice(); config.channel_configuration.channels()];
    let mut encoder = StreamEncoder::new(config);
    let mut out = Vec::new();
    encoder.encode(&channels, &mut |frame: &[u8]| out.push(frame.to_vec()));
    assert_eq!(out.len(), frames);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(framing: Framing, bits_per_second: u32) -> StreamConfig {
        StreamConfig {
            sampling_frequency: SamplingFrequency::Hz44100,
            channel_configuration: ChannelConfiguration::Stereo,
            bits_per_second,
            control: Control::Constant,
            framing,
            flags: Flags::default(),
            crc_protected: false,
            noise_substitution: false,
        }
    }

    #[test]
    fn frames_keep_the_rate_once_the_reservoir_is_full() {
        for framing in [Framing::Adts, Framing::Raw] {
            let config = config(framing, 128_000);
            let frames = encode_silence(config, 200);
            let total: usize = frames.iter().map(Vec::len).sum();
            // What the rate brings, less what the reservoir holds at most.
            let mean_total = 200 * 128_000 * 1024 / 44_100 / 8;
            let max_bit_reservoir = 2 * 6144 - 128_000 * 1024 / 44_100;
            assert!(total <= mean_total, "{framing:?}");
            assert!(
                total + max_bit_reservoir / 8 + 1 >= mean_total,
                "{framing:?}"
            );
            // The first frames are the shortest, filling the reservoir: a
            // pair of silent channels that share their window.
            let silent = CodedChannel::silent(Windows::LONG_SINE, config.sampling_frequency);
            let element = CodedElement::common_pair(silent.clone(), silent, MsMask::NONE);
            let shortest = config.header_len()
                + raw_data_block::min_len(config.channel_configuration, &[element], 0);
            assert_eq!(frames[0].len(), shortest, "{framing:?}");
            assert_eq!(shortest, config.shortest_frame_len(Windows::LONG_SINE, 0));
        }
    }

    #[test]
    fn the_crc_raises_the_lowest_bitrate_by_its_bits() {
        // The shortest frame of a pair that shares its window, 13 bytes of
        // which 7 are the header, and with the CRC 15, at 44.1 kHz,
        // rounded up.
        let mut config = config(Framing::Adts, 0);
        assert_eq!(config.min_bits_per_second(), 4479);
        config.crc_protected = true;
        assert_eq!(config.min_bits_per_second(), 5168);
    }

    #[test]
    fn adts_frames_give_their_length_and_the_reservoir() {
        let mut config = config(Framing::Adts, 128_000);
        config.crc_protected = true;
        let frames = encode_silence(config, 200);
        let mut reservoir = config.reservoir().unwrap();
        for frame in &frames {
            reservoir.spend(frame.len());
            let length = (usize::from(frame[3] & 0b11) << 11)
                | (usize::from(frame[4]) << 3)
                | usize::from(frame[5] >> 5);
            let fullness = (u32::from(frame[5] & 0b1_1111) << 6) | u32::from(frame[6] >> 2);
            assert_eq!(length, frame.len());
            assert_eq!(fullness, reservoir.adts_buffer_fullness());
        }
        assert!(frames.last().unwrap().len() > config.shortest_frame_len(Windows::LONG_SINE, 0));
    }

    #[test]
    fn the_bitrate_bounds_admit_every_frame() {
        for fs in SamplingFrequency::ALL {
            for channel_configuration in [ChannelConfiguration::Mono, ChannelConfiguration::Stereo]
            {
                for (framing, crc_protected) in [
                    (Framing::Adts, false),
                    (Framing::Adts, true),
                    (Framing::Raw, false),
                ] {
                    let mut config = StreamConfig {
                        sampling_frequency: fs,
                        channel_configuration,
                        bits_per_second: 0,
                        control: Control::Constant,
                        framing,
                        flags: Flags::default(),
                        crc_protected,
                        noise_substitution: false,
                    };
                    let (min, max) = (config.min_bits_per_second(), config.max_bits_per_second());
                    for bits_per_second in [min - 1, min, max, max + 1] {
                        config.bits_per_second = bits_per_second;
                        let valid = (min..=max).contains(&bits_per_second);
                        assert_eq!(config.is_valid(), valid, "{config:?}");
                        if valid {
                            encode_silence(config, 50);
                        }
                    }
                }
            }
        }
    }

    /// A click after silence is coded with short windows, but for an
    /// encoder without block switching.
    #[test]
    fn block_switching_can_be_turned_off() {
        let config = config(Framing::Raw, 64_000);
        let mut input = vec![0.0; 8 * FRAME_SAMPLES];
        input[5 * FRAME_SAMPLES + 300] = 0.9;
        for switching in [true, false] {
            let mut encoder = StreamEncoder::new(config);
            if !switching {
                encoder = encoder.without_block_switching();
            }
            let mut sequences = Vec::new();
            encoder.encode(&[&input, &input], &mut |frame: &[u8]| {
                // id_syn_ele, element_instance_tag, common_window 1,
                // ics_reserved_bit, then window_sequence.
                assert_eq!(frame[0] & 1, 1, "common_window");
                sequences.push((frame[1] >> 5) & 0b11);
            });
            assert_eq!(sequences.len(), 7);
            assert_eq!(sequences.contains(&2), switching, "{sequences:?}");
        }
    }

    /// At the lowest bitrate, which is set by frames of long windows, a
    /// frame cannot afford the side information of eight short windows,
    /// and an attack is coded with long ones; a little above it, it is
    /// coded short.
    #[test]
    fn tns_can_be_turned_off() {
        use crate::noiseless::decoder::{BitReader, element};
        let config = config(Framing::Raw, 192_000);
        // A click in each of the frames, coded with long windows, in bits
        // enough to code some of its lines.
        let mut input = vec![0.0; 8 * FRAME_SAMPLES];
        for frame in 0..8 {
            input[frame * FRAME_SAMPLES + 300] = 0.9;
        }
        for tns in [true, false] {
            let mut encoder = StreamEncoder::new(config).without_block_switching();
            if !tns {
                encoder = encoder.without_tns();
            }
            let mut shaped = 0;
            encoder.encode(&[&input, &input], &mut |frame: &[u8]| {
                let mut r = BitReader::new(frame);
                let element = element(&mut r, SamplingFrequency::Hz44100);
                shaped += element.channels.iter().filter(|c| c.tns.is_some()).count();
            });
            assert_eq!(shaped > 0, tns, "{shaped}");
        }
    }

    #[test]
    fn spectrum_clipping_is_off_unless_asked_for() {
        use crate::noiseless::decoder::{BitReader, element};
        let config = config(Framing::Raw, 128_000);
        // Noise falling by 6 dB an octave, whose highest lines quantize to
        // few above 1.
        let (mut state, mut y) = (1u64, 0.0);
        let input: Vec<f64> = (0..24 * FRAME_SAMPLES)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                y = 0.98 * y + ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5);
                0.2 * y
            })
            .collect();
        let right: Vec<f64> = input.iter().map(|x| -0.7 * x).collect();
        for on in [None, Some(false), Some(true)] {
            let mut encoder = StreamEncoder::new(config);
            if let Some(on) = on {
                encoder = encoder.spectrum_clipping(on);
            }
            let mut pulses = 0;
            encoder.encode(&[&input, &right], &mut |frame: &[u8]| {
                let mut r = BitReader::new(frame);
                let element = element(&mut r, SamplingFrequency::Hz44100);
                pulses += element
                    .channels
                    .iter()
                    .filter(|c| c.pulses.is_some())
                    .count();
            });
            assert_eq!(pulses > 0, on == Some(true), "{on:?}: {pulses}");
        }
    }

    #[test]
    fn the_blocks_past_the_input_are_not_shaped() {
        use crate::noiseless::decoder::{BitReader, element};
        let config = config(Framing::Raw, 192_000);
        // A click in each block, the last block half filled.
        let len = 5 * FRAME_SAMPLES + FRAME_SAMPLES / 2;
        let mut input = vec![0.0; len];
        for at in (300..len).step_by(FRAME_SAMPLES) {
            input[at] = 0.9;
        }
        let mut encoder = StreamEncoder::new(config).without_block_switching();
        let mut shaped = Vec::new();
        let mut sink = |frame: &[u8]| {
            let mut r = BitReader::new(frame);
            let element = element(&mut r, SamplingFrequency::Hz44100);
            shaped.push(element.channels.iter().any(|c| c.tns.is_some()));
        };
        encoder.encode(&[&input, &input], &mut sink);
        let end = encoder.finish(&mut sink);
        assert_eq!(end.frames, 7);
        // Blocks 5, the input's last, and 6, after it, hold padding.
        assert!(shaped[1..5].iter().any(|&shaped| shaped), "{shaped:?}");
        assert_eq!(shaped[5..], [false, false]);
    }

    #[test]
    fn a_frame_without_bits_for_short_windows_stays_long() {
        let mut config = StreamConfig {
            sampling_frequency: SamplingFrequency::Hz44100,
            channel_configuration: ChannelConfiguration::Mono,
            bits_per_second: 0,
            control: Control::Constant,
            framing: Framing::Adts,
            flags: Flags::default(),
            crc_protected: false,
            noise_substitution: false,
        };
        let min = config.min_bits_per_second();
        let mut input = vec![0.0; 12 * FRAME_SAMPLES];
        for click in [3 * FRAME_SAMPLES + 300, 8 * FRAME_SAMPLES + 900] {
            input[click] = 0.9;
        }
        for (bits_per_second, short) in [(min, false), (min + 1000, true)] {
            config.bits_per_second = bits_per_second;
            let mut sequences = Vec::new();
            let mut encoder = StreamEncoder::new(config);
            encoder.encode(&[&input], &mut |frame: &[u8]| {
                // After the header, id_syn_ele, element_instance_tag,
                // global_gain and ics_reserved_bit.
                sequences.push((frame[9] >> 6) & 0b11);
            });
            assert_eq!(
                sequences.contains(&2),
                short,
                "{bits_per_second}: {sequences:?}"
            );
            assert!(sequences.contains(&1), "{bits_per_second}: {sequences:?}");
        }
    }

    #[test]
    fn noise_after_silence_borrows_from_the_reservoir() {
        let config = StreamConfig {
            sampling_frequency: SamplingFrequency::Hz44100,
            channel_configuration: ChannelConfiguration::Mono,
            bits_per_second: 8_000,
            control: Control::Constant,
            framing: Framing::Adts,
            flags: Flags::default(),
            crc_protected: false,
            noise_substitution: false,
        };
        // Silence, which takes the shortest frames and fills the
        // reservoir, then white noise, whose perceptual entropy asks for
        // more than the mean.
        let mut input = vec![0.0; 40 * FRAME_SAMPLES];
        let mut state = 3u64;
        input.extend((0..10 * FRAME_SAMPLES).map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            0.25 * ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
        }));
        let mut encoder = StreamEncoder::new(config);
        let mut frames = Vec::new();
        encoder.encode(&[&input], &mut |frame: &[u8]| frames.push(frame.len()));
        let mean = 8_000.0 * FRAME_SAMPLES as f64 / 44_100.0 / 8.0;
        let longest = frames[40..].iter().copied().max().unwrap();
        assert!(
            longest as f64 > 1.5 * mean,
            "{longest} bytes against {mean}"
        );
    }

    #[test]
    fn the_bits_frames_leave_come_back_to_their_audio() {
        use crate::raw_data_block::decoder::{adts_frame, raw_data_block};

        let config = config(Framing::Adts, 128_000);
        let reservoir = config.reservoir().unwrap();
        // Noise falling with frequency, whose perceptual entropy asks for no
        // bits beyond the mean, and which takes all it is given, but for
        // what the loops leave: quiet enough, at some -46 dB below full
        // scale, that the threshold in quiet, taken as noise has it, keeps
        // its perceptual entropy low.
        let red = |seed| {
            let mut y = 0.0;
            (noise(301 * FRAME_SAMPLES, seed).into_iter())
                .map(|x| {
                    y = 0.99 * y + 0.01 * x;
                    y
                })
                .collect::<Vec<f64>>()
        };
        let (left, right) = (red(5), red(6));
        let mut encoder = StreamEncoder::new(config);
        let mut stream = Vec::new();
        encoder.encode(&[&left, &right], &mut stream);
        let (mut frame_bits, mut element_bits, mut fullness) = (0, 0, Vec::new());
        let mut rest = stream.as_slice();
        for k in 0..300 {
            let frame = adts_frame(rest);
            rest = &rest[frame.aac_frame_length..];
            if k >= 100 {
                let block = raw_data_block(frame.raw_data_block, config.sampling_frequency);
                frame_bits += 8 * frame.aac_frame_length;
                element_bits += block.elements[0].bits.len();
                fullness.push(frame.adts_buffer_fullness);
            }
        }
        // The audio takes all but the header, the element's id_syn_ele and
        // ID_END, 62 of the 2972 bits of a mean frame, and a few the loops
        // leave; filled out to the mean, it took 92.2%.
        let share = element_bits as f64 / frame_bits as f64;
        assert!(share > 0.975, "{share}");
        // The reservoir is never full, which would have frames filled out,
        // and keeps what lies within half a mean frame of its maximum, for
        // frames that ask for more.
        let mean = 128_000 * FRAME_SAMPLES as u64 / 44_100;
        let full = (reservoir.max_bits() / 2 / 32) as u32;
        let kept = ((reservoir.max_bits() - mean / 2) / 2 / 32) as u32;
        assert!(
            fullness.iter().all(|&f| (kept..full).contains(&f)),
            "{fullness:?}"
        );
    }

    /// A mono stream at 44.1 kHz whose reservoir is as large as a mean
    /// frame, 3072 bits, with the CRC and data streams that leave its audio
    /// a few bits a frame: 38 blocks of silence fill the reservoir, then
    /// white noise sets in, with a second attack after a gap in block 44.
    /// Neither the first attack nor the frames after it take all that the
    /// reservoir holds, which leaves the second bits to borrow: every frame
    /// codes lines, the second attack's in eight short windows, in more
    /// than the 436 bits it took before frames were offered any.
    #[test]
    fn an_attack_finds_bits_in_the_reservoir_after_another() {
        use crate::raw_data_block::decoder::{adts_frame, raw_data_block};

        let fs = SamplingFrequency::Hz44100;
        let config = StreamConfig {
            sampling_frequency: fs,
            channel_configuration: ChannelConfiguration::Mono,
            bits_per_second: 132_300,
            control: Control::Constant,
            framing: Framing::Adts,
            flags: Flags::default(),
            crc_protected: true,
            noise_substitution: false,
        };
        let mut input = vec![0.0; 38 * FRAME_SAMPLES];
        let gap = 6 * FRAME_SAMPLES..6 * FRAME_SAMPLES + 600;
        let noise = noise(12 * FRAME_SAMPLES, 3);
        input.extend(
            (noise.iter().enumerate()).map(|(t, &x)| if gap.contains(&t) { 0.0 } else { x }),
        );
        let mut encoder = StreamEncoder::new(config).with_data_stream(368);
        let mut stream = Vec::new();
        encoder.encode(&[&input[..38 * FRAME_SAMPLES]], &mut stream);
        encoder.push_data_stream(&[0x33; 368 * 60]);
        encoder.encode(&[&input[38 * FRAME_SAMPLES..]], &mut stream);
        let mut rest = stream.as_slice();
        // The frames from the first attack's on.
        let frames: Vec<_> = (0..49)
            .map(|_| {
                let frame = adts_frame(rest);
                rest = &rest[frame.aac_frame_length..];
                raw_data_block(frame.raw_data_block, fs).elements[0].clone()
            })
            .skip(38)
            .collect();
        for (k, element) in frames.iter().enumerate() {
            let lines = &element.element.channels[0].lines;
            assert!(lines.iter().flatten().any(|&q| q != 0), "frame {}", 38 + k);
        }
        let second = &frames[45 - 38];
        assert_eq!(second.element.channels[0].info.window_sequence, 2);
        assert!(second.bits.len() > 436, "{}", second.bits.len());
    }

    /// A mono stream at 44.1 kHz and 48 kbit/s, with the CRC and 124 bytes
    /// of data streams a frame, whose mean frame does not pay for eight
    /// short windows, codes an attack with them once the reservoir offers
    /// the bits.
    #[test]
    fn an_attack_is_coded_short_in_the_bits_the_reservoir_offers() {
        use crate::raw_data_block::decoder::{adts_frame, raw_data_block};

        let fs = SamplingFrequency::Hz44100;
        let config = StreamConfig {
            sampling_frequency: fs,
            channel_configuration: ChannelConfiguration::Mono,
            bits_per_second: 48_000,
            control: Control::Constant,
            framing: Framing::Adts,
            flags: Flags::default(),
            crc_protected: true,
            noise_substitution: false,
        };
        let mut encoder = StreamEncoder::new(config).with_data_stream(124);
        let data_bits = data_stream::bits(124);
        assert!(!encoder.affords_short(data_bits));
        // Silence, without data, fills the reservoir; then noise sets in
        // within block 40, with data in every frame.
        let mut input = vec![0.0; 40 * FRAME_SAMPLES + 512];
        input.extend(noise(4 * FRAME_SAMPLES, 7));
        let mut stream = Vec::new();
        encoder.encode(&[&input[..36 * FRAME_SAMPLES]], &mut stream);
        encoder.push_data_stream(&[0x5A; 124 * 50]);
        encoder.encode(&[&input[36 * FRAME_SAMPLES..]], &mut stream);
        let mut rest = stream.as_slice();
        let sequences: Vec<u32> = (0..43)
            .map(|k| {
                let frame = adts_frame(rest);
                rest = &rest[frame.aac_frame_length..];
                let block = raw_data_block(frame.raw_data_block, fs);
                let carried: usize = block.data_streams.iter().map(|d| d.bytes.len()).sum();
                assert!(k < 36 || carried == 124, "{k}: {carried}");
                u32::from(block.elements[0].element.channels[0].info.window_sequence)
            })
            .collect();
        // Block 40's frame is coded short, between a start and a stop.
        assert_eq!(sequences[39..42], [1, 2, 3], "{sequences:?}");
        assert!(sequences[..39].iter().all(|&s| s == 0), "{sequences:?}");
    }

    fn noise(len: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                0.25 * ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
            })
            .collect()
    }

    /// Frames carry the bytes pushed in order, each up to its share, in
    /// raw frames and in ADTS, those pushed after the frames were coded
    /// among them, and the rest are dropped at the end; the stream keeps
    /// its rate.
    #[test]
    fn data_streams_carry_the_bytes_pushed_in_order() {
        use crate::raw_data_block::decoder::{adts_frame, raw_data_block};

        let fs = SamplingFrequency::Hz44100;
        let input = noise(12 * FRAME_SAMPLES, 5);
        let data: Vec<u8> = (0..20_000u32).map(|i| (i * 7 + i / 255) as u8).collect();
        // At 320 kbit/s, two elements a frame.
        for (framing, bits_per_second, per_frame) in
            [(Framing::Raw, 128_000, 100), (Framing::Adts, 320_000, 600)]
        {
            let config = config(framing, bits_per_second);
            let mut encoder = StreamEncoder::new(config).with_data_stream(per_frame);
            let mut frames = Vec::new();
            let mut sink = |frame: &[u8]| frames.push(frame.to_vec());
            let (first, second) = data.split_at(300);
            encoder.push_data_stream(first);
            encoder.encode(&[&input[..6 * FRAME_SAMPLES]; 2], &mut sink);
            encoder.push_data_stream(second);
            encoder.encode(&[&input[6 * FRAME_SAMPLES..]; 2], &mut sink);
            let end = encoder.finish(&mut sink);
            // Each frame's bytes, from its data stream elements.
            let carried: Vec<Vec<u8>> = (frames.iter())
                .map(|frame| {
                    let block = match framing {
                        Framing::Adts => adts_frame(frame).raw_data_block,
                        Framing::Raw => frame,
                    };
                    (raw_data_block(block, fs).data_streams.iter())
                        .flat_map(|stream| stream.bytes.clone())
                        .collect()
                })
                .collect();
            // The first five frames, coded before the second push, carry
            // the first push; the rest carry their share.
            let lens: Vec<usize> = carried.iter().map(Vec::len).collect();
            let first_lens: Vec<usize> = (0..5)
                .map(|k| 300usize.saturating_sub(k * per_frame).min(per_frame))
                .collect();
            assert_eq!(lens[..5], first_lens, "{framing:?}");
            assert!(lens[5..].iter().all(|&len| len == per_frame), "{lens:?}");
            let carried = carried.concat();
            assert_eq!(carried, data[..carried.len()], "{framing:?}");
            assert_eq!(carried.len() + end.data_stream_dropped, data.len());
            let lens: Vec<usize> = frames.iter().map(Vec::len).collect();
            let total: usize = lens.iter().sum();
            let mean_total = lens.len() * bits_per_second as usize * 1024 / 44_100 / 8;
            assert!(total <= mean_total, "{framing:?}");
        }
    }

    /// At the lowest bitrates and above, a frame of noise or silence
    /// carries the most bytes of data streams allowed, and one more is not.
    #[test]
    fn the_most_data_fits_every_frame() {
        let input = noise(10 * FRAME_SAMPLES, 9);
        for fs in [SamplingFrequency::Hz8000, SamplingFrequency::Hz48000] {
            for channel_configuration in [ChannelConfiguration::Mono, ChannelConfiguration::Stereo]
            {
                let mut config = StreamConfig {
                    sampling_frequency: fs,
                    channel_configuration,
                    bits_per_second: 0,
                    control: Control::Constant,
                    framing: Framing::Adts,
                    flags: Flags::default(),
                    crc_protected: true,
                    noise_substitution: false,
                };
                let min = config.min_bits_per_second();
                assert_eq!(config.max_data_stream_bytes(), 0);
                for bits_per_second in [min, min + 200, min + 5_000, 4 * min + 30_000] {
                    config.bits_per_second = bits_per_second;
                    let max = config.max_data_stream_bytes();
                    // The shortest frame with them is within the mean, and
                    // with one more byte beyond it.
                    let mean = bits_per_second as usize * FRAME_SAMPLES / (8 * fs.hz() as usize);
                    assert!(config.shortest_frame_len(Windows::LONG_SINE, max) <= mean);
                    assert!(config.shortest_frame_len(Windows::LONG_SINE, max + 1) > mean);
                    let mut encoder = StreamEncoder::new(config).with_data_stream(max);
                    encoder.push_data_stream(&vec![0xFF; 20 * max]);
                    let channels = vec![input.as_slice(); channel_configuration.channels()];
                    let mut stream = Vec::new();
                    encoder.encode(&channels, &mut stream);
                    let end = encoder.finish(&mut stream);
                    assert_eq!(end.data_stream_dropped, (20 - end.frames as usize) * max);
                }
            }
        }
    }

    /// A pair at 48 kHz and 96 kbit/s, with the CRC, carries 239 bytes in
    /// every frame of noise: as many as its shortest frame, of a common
    /// window, leaves of the 256 a frame the rate brings.
    #[test]
    fn a_pair_carries_what_its_common_window_leaves() {
        use crate::raw_data_block::decoder::{adts_frame, raw_data_block};

        let mut config = config(Framing::Adts, 96_000);
        config.sampling_frequency = SamplingFrequency::Hz48000;
        config.crc_protected = true;
        assert_eq!(config.max_data_stream_bytes(), 239);
        let input = noise(31 * FRAME_SAMPLES, 11);
        let mut encoder = StreamEncoder::new(config).with_data_stream(239);
        encoder.push_data_stream(&[0x5A; 32 * 239]);
        let mut stream = Vec::new();
        encoder.encode(&[&input, &input], &mut stream);
        let end = encoder.finish(&mut stream);
        assert_eq!((end.frames, end.data_stream_dropped), (32, 0));
        let mut rest = stream.as_slice();
        while !rest.is_empty() {
            let frame = adts_frame(rest);
            rest = &rest[frame.aac_frame_length..];
            let block = raw_data_block(frame.raw_data_block, SamplingFrequency::Hz48000);
            assert!(block.elements[0].element.common_window);
            let carried: usize = block.data_streams.iter().map(|d| d.bytes.len()).sum();
            assert_eq!(carried, 239);
        }
    }

    /// The channel configurations of more than two channels.
    const MULTICHANNEL: [ChannelConfiguration; 5] = [
        ChannelConfiguration::Three,
        ChannelConfiguration::Four,
        ChannelConfiguration::Five,
        ChannelConfiguration::FiveOne,
        ChannelConfiguration::SevenOne,
    ];

    /// Each frame of a configuration holds its elements in order, each kind
    /// tagged from 0, and each channel's tone in its own place: the input's
    /// channels in the order of the elements that send them.
    #[test]
    fn each_configuration_sends_its_channels_in_its_elements_in_order() {
        use crate::raw_data_block::decoder::raw_data_block;
        use crate::raw_data_block::element_instance_tags;

        let fs = SamplingFrequency::Hz48000;
        for channel_configuration in MULTICHANNEL {
            let channels = channel_configuration.channels();
            let config = StreamConfig {
                sampling_frequency: fs,
                channel_configuration,
                bits_per_second: 96_000 * channel_configuration.ncc() as u32,
                control: Control::Constant,
                framing: Framing::Raw,
                flags: Flags::default(),
                crc_protected: false,
                noise_substitution: false,
            };
            // Channel c a steady tone at the centre of line 40 + 37c, and a
            // low frequency enhancement channel, last, at line 5, among the
            // 12 it sends.
            let lfe = channel_configuration.elements().last() == Some(&ElementKind::Lfe);
            let line = |c: usize| match c {
                c if lfe && c == channels - 1 => 5,
                c => 40 + 37 * c,
            };
            let input: Vec<Vec<f64>> = (0..channels)
                .map(|c| {
                    let k = line(c) as f64 + 0.5;
                    (0..8 * FRAME_SAMPLES)
                        .map(|t| 0.3 * (std::f64::consts::PI * k * t as f64 / 1024.0).sin())
                        .collect()
                })
                .collect();
            let input: Vec<&[f64]> = input.iter().map(Vec::as_slice).collect();
            let mut frames = Vec::new();
            StreamEncoder::new(config).encode(&input, &mut |frame: &[u8]| {
                frames.push(frame.to_vec());
            });
            assert_eq!(frames.len(), 7);
            let kinds = channel_configuration.elements();
            for frame in &frames[2..] {
                let block = raw_data_block(frame, fs);
                let mut ids: Vec<u32> = kinds.iter().map(|k| k.id_syn_ele()).collect();
                ids.push(7);
                let sent: Vec<u32> = block.ids.iter().copied().filter(|&id| id != 6).collect();
                assert_eq!(sent, ids, "{channel_configuration:?}");
                let tags: Vec<u32> = (block.elements.iter())
                    .map(|e| e.element.element_instance_tag)
                    .collect();
                let expected: Vec<u32> = element_instance_tags(channel_configuration).collect();
                assert_eq!(tags, expected, "{channel_configuration:?}");
                let spectra: Vec<Vec<f64>> = (block.elements.iter())
                    .flat_map(|e| e.element.spectra(fs))
                    .collect();
                assert_eq!(spectra.len(), channels);
                // A tone at the centre of a line spreads to the lines beside
                // it with its phase.
                for (c, spectrum) in spectra.iter().enumerate() {
                    let energy = |lines: &[f64]| lines.iter().map(|x| x * x).sum::<f64>();
                    let near = energy(&spectrum[line(c) - 2..=line(c) + 2]);
                    assert!(
                        near > 0.99 * energy(spectrum),
                        "{channel_configuration:?}, channel {c}"
                    );
                }
            }
        }
    }

    /// A low frequency enhancement channel keeps to its restrictions
    /// (clause 8.4.1) whatever it is given: clicks over loud noise, which
    /// the other channels code with short windows and temporal noise
    /// shaping, it codes with long windows of the sine shape, unshaped, in
    /// its 12 lowest lines, which carry its tone.
    #[test]
    fn a_low_frequency_enhancement_channel_keeps_to_its_restrictions() {
        use crate::raw_data_block::decoder::raw_data_block;
        use crate::raw_data_block::element::LFE_LINES;

        let fs = SamplingFrequency::Hz48000;
        for (channel_configuration, clipping) in [
            (ChannelConfiguration::FiveOne, false),
            (ChannelConfiguration::SevenOne, true),
        ] {
            let channels = channel_configuration.channels();
            let config = StreamConfig {
                sampling_frequency: fs,
                channel_configuration,
                bits_per_second: 64_000 * channel_configuration.ncc() as u32,
                control: Control::Constant,
                framing: Framing::Raw,
                flags: Flags::default(),
                crc_protected: false,
                noise_substitution: false,
            };
            let mut input = noise(10 * FRAME_SAMPLES, 13);
            for (t, x) in input.iter_mut().enumerate() {
                *x += 0.5 * (std::f64::consts::PI * 5.5 * t as f64 / 1024.0).sin();
                if t % 1500 == 700 {
                    *x += 0.9;
                }
            }
            // The other channels' every window shaped, as clause C.5 shapes
            // it, as the clicks leave them few frames of long windows alone.
            let mut encoder = StreamEncoder::new(config)
                .spectrum_clipping(clipping)
                .with_tns_around_attacks();
            let (mut short, mut shaped, mut lfe_frames) = (false, false, 0);
            encoder.encode(&vec![input.as_slice(); channels], &mut |frame: &[u8]| {
                let block = raw_data_block(frame, fs);
                for element in &block.elements {
                    for ics in &element.element.channels {
                        if element.id_syn_ele != 3 {
                            short |= ics.info.window_sequence == 2;
                            shaped |= ics.tns.is_some();
                            continue;
                        }
                        lfe_frames += 1;
                        assert_eq!((ics.info.window_sequence, ics.info.window_shape), (0, 0));
                        assert!(ics.tns.is_none());
                        assert!(ics.lines[0][LFE_LINES..].iter().all(|&q| q == 0));
                        assert!(ics.lines[0][4..8].iter().any(|&q| q != 0));
                    }
                }
            });
            assert_eq!(lfe_frames, 9, "{channel_configuration:?}");
            assert!(short && shaped, "{channel_configuration:?}");
        }
    }

    /// No element is longer than its share of the decoder's input buffer,
    /// 6144 bits a channel with its `id_syn_ele`, however many bits the
    /// frame has and however few of its elements ask for them: noise in
    /// one element of 5.1 and 7.1 beside silence, up to the highest
    /// bitrate.
    #[test]
    fn no_element_outgrows_its_buffer() {
        use crate::raw_data_block::decoder::raw_data_block;

        for (fs, channel_configuration, loud, bits_per_second) in [
            (
                SamplingFrequency::Hz48000,
                ChannelConfiguration::FiveOne,
                0,
                320_000,
            ),
            (
                SamplingFrequency::Hz48000,
                ChannelConfiguration::FiveOne,
                1,
                640_000,
            ),
            (
                SamplingFrequency::Hz96000,
                ChannelConfiguration::SevenOne,
                1,
                0,
            ),
        ] {
            let mut config = StreamConfig {
                sampling_frequency: fs,
                channel_configuration,
                bits_per_second,
                control: Control::Constant,
                framing: Framing::Raw,
                flags: Flags::default(),
                crc_protected: false,
                noise_substitution: false,
            };
            if bits_per_second == 0 {
                config.bits_per_second = config.max_bits_per_second();
            }
            // Loud noise in the channels of element `loud`, silence in the
            // rest.
            let noise = noise(12 * FRAME_SAMPLES, 21)
                .iter()
                .map(|x| 3.0 * x)
                .collect::<Vec<_>>();
            let silence = vec![0.0; noise.len()];
            let mut channels = Vec::new();
            for (e, kind) in channel_configuration.elements().iter().enumerate() {
                let samples = if e == loud { &noise } else { &silence };
                channels.extend(std::iter::repeat_n(samples.as_slice(), kind.channels()));
            }
            let mut longest = 0;
            StreamEncoder::new(config).encode(&channels, &mut |frame: &[u8]| {
                for element in raw_data_block(frame, fs).elements {
                    let channels = element.element.channels.len();
                    let bits = 3 + element.bits.len();
                    if element.id_syn_ele != 3 {
                        assert!(bits <= 6144 * channels, "{channel_configuration:?}: {bits}");
                    }
                    longest = longest.max(bits);
                }
            });
            // The loud element takes all its buffer allows.
            let kind = channel_configuration.elements()[loud];
            assert!(
                longest > 6144 * kind.channels() - 100,
                "{channel_configuration:?}: {longest}"
            );
        }
    }

    /// An element that its share leaves without lines is quantized again
    /// with the bits the frame has left: in 4 channels at 48 kHz and 11
    /// kbit/s with the CRC, the center and rear surround carrying the same
    /// tone of 79 Hz, and the front pair one of 1.3 kHz, the center sends
    /// lines in every frame of the tone as the rear surround does, where
    /// it once sent none.
    #[test]
    fn an_element_left_without_lines_takes_the_bits_left() {
        use crate::raw_data_block::decoder::{adts_frame, raw_data_block};

        let fs = SamplingFrequency::Hz48000;
        let config = StreamConfig {
            sampling_frequency: fs,
            channel_configuration: ChannelConfiguration::Four,
            bits_per_second: 11_000,
            control: Control::Constant,
            framing: Framing::Adts,
            flags: Flags::default(),
            crc_protected: true,
            noise_substitution: false,
        };
        let tone = |frequency: f64, amplitude: f64| -> Vec<f64> {
            (0..3 * 48_000)
                .map(|t| {
                    amplitude * (std::f64::consts::TAU * frequency * t as f64 / 48_000.0).sin()
                })
                .collect()
        };
        let (low, mid) = (tone(79.0, 0.8), tone(1300.0, 0.5));
        let mut stream = Vec::new();
        StreamEncoder::new(config).encode(&[&low, &mid, &mid, &low], &mut stream);
        let mut sending = Vec::new();
        let mut rest = stream.as_slice();
        while !rest.is_empty() {
            let frame = adts_frame(rest);
            rest = &rest[frame.aac_frame_length..];
            let block = raw_data_block(frame.raw_data_block, fs);
            let sends = |e: usize| {
                let lines = &block.elements[e].element.channels[0].lines;
                lines.iter().flatten().any(|&q| q != 0)
            };
            sending.push((sends(0), sends(2)));
        }
        assert!(sending.len() > 100);
        // From the third frame, the first of the tone throughout.
        assert!(
            sending[2..].iter().all(|&s| s == (true, true)),
            "{sending:?}"
        );
    }

    /// The bitrate of a configuration ranges from that of its shortest
    /// frame to that of its NCC's input buffer, or of 4608 bytes where that
    /// is less (see [`reservoir::MAX_ADTS_FRAME_LEN`]), and every frame of
    /// silence and noise fits it at both ends, with the CRC and without.
    #[test]
    fn every_configuration_codes_at_both_ends_of_its_range() {
        let fs = SamplingFrequency::Hz44100;
        for channel_configuration in MULTICHANNEL {
            let input = noise(4 * FRAME_SAMPLES, 3);
            for crc_protected in [false, true] {
                let mut config = StreamConfig {
                    sampling_frequency: fs,
                    channel_configuration,
                    bits_per_second: 0,
                    control: Control::Constant,
                    framing: Framing::Adts,
                    flags: Flags::default(),
                    crc_protected,
                    noise_substitution: false,
                };
                let max = config.max_bits_per_second();
                let ncc = channel_configuration.ncc() as u64;
                assert_eq!(max as u64, ((6144 * ncc).min(8 * 4608) - 8) * 44_100 / 1024);
                for bits_per_second in [config.min_bits_per_second(), max] {
                    config.bits_per_second = bits_per_second;
                    assert!(config.is_valid());
                    let channels = vec![input.as_slice(); channel_configuration.channels()];
                    let mut encoder = StreamEncoder::new(config);
                    let mut stream = Vec::new();
                    encoder.encode(&channels, &mut stream);
                    encoder.finish(&mut stream);
                }
            }
        }
    }

    /// No frame of 7.1 in ADTS is longer than 4608 bytes (see
    /// [`reservoir::MAX_ADTS_FRAME_LEN`]): of loud noise at the highest
    /// bitrate at 8, 48 and 96 kHz, nor of loud noise after the silence
    /// that fills the reservoir at 1700 kbit/s at 48 kHz, below the
    /// highest, 1727, whose raw frames, which keep the standard's limits,
    /// are longer. At lower bitrates, frames borrow too little from the
    /// reservoir to reach 4608 bytes.
    #[test]
    fn adts_frames_of_7_1_are_no_longer_than_4608_bytes() {
        use crate::raw_data_block::decoder::adts_frame;

        let configuration = ChannelConfiguration::SevenOne;
        let loud = |len| noise(len, 5).iter().map(|x| 3.0 * x).collect::<Vec<_>>();
        let mut silence_then_noise = vec![0.0; 8 * FRAME_SAMPLES];
        silence_then_noise.extend(loud(6 * FRAME_SAMPLES));
        let at_the_highest = loud(6 * FRAME_SAMPLES);
        for (fs, bits_per_second, input) in [
            (SamplingFrequency::Hz8000, 0, &at_the_highest),
            (SamplingFrequency::Hz48000, 0, &at_the_highest),
            (SamplingFrequency::Hz96000, 0, &at_the_highest),
            (SamplingFrequency::Hz48000, 1_700_000, &silence_then_noise),
        ] {
            let channels = vec![input.as_slice(); configuration.channels()];
            let mut longest = [0; 2];
            for (i, framing) in [Framing::Adts, Framing::Raw].into_iter().enumerate() {
                let mut config = StreamConfig {
                    sampling_frequency: fs,
                    channel_configuration: configuration,
                    bits_per_second,
                    control: Control::Constant,
                    framing,
                    flags: Flags::default(),
                    crc_protected: false,
                    noise_substitution: false,
                };
                if bits_per_second == 0 {
                    config.bits_per_second = StreamConfig {
                        framing: Framing::Adts,
                        ..config
                    }
                    .max_bits_per_second();
                }
                let mut encoder = StreamEncoder::new(config);
                let mut sink = |frame: &[u8]| {
                    let len = match framing {
                        Framing::Adts => adts_frame(frame).aac_frame_length,
                        Framing::Raw => frame.len(),
                    };
                    longest[i] = longest[i].max(len);
                };
                encoder.encode(&channels, &mut sink);
                encoder.finish(&mut sink);
            }
            let [adts, raw] = longest;
            assert!(adts <= 4608, "{fs:?} {bits_per_second}: {adts}");
            if bits_per_second != 0 {
                assert!(raw > 4608, "{fs:?} {bits_per_second}: {raw}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "bytes of data streams a frame")]
    fn a_share_beyond_the_most_is_refused() {
        let config = config(Framing::Raw, 64_000);
        StreamEncoder::new(config).with_data_stream(config.max_data_stream_bytes() + 1);
    }

    fn variable(framing: Framing, noise_offset_db: f64) -> StreamConfig {
        let mut config = StreamConfig {
            control: Control::Variable { noise_offset_db },
            ..config(framing, 0)
        };
        config.bits_per_second = config.nominal_bits_per_second(noise_offset_db);
        config
    }

    /// The frames of `config` coding `channels`, each apart.
    fn frames_of(config: StreamConfig, channels: &[&[f64]]) -> Vec<Vec<u8>> {
        let mut frames = Vec::new();
        let mut sink = |frame: &[u8]| frames.push(frame.to_vec());
        let mut encoder = StreamEncoder::new(config);
        encoder.encode(channels, &mut sink);
        encoder.finish(&mut sink);
        frames
    }

    /// At a variable rate, silence takes the shortest frames, with no fill
    /// to keep a rate, and noise longer ones; every header's
    /// `adts_buffer_fullness` is 0x7FF.
    #[test]
    fn variable_frames_take_what_they_need_and_say_so() {
        use crate::raw_data_block::decoder::{adts_frame, raw_data_block};

        let config = variable(Framing::Adts, 0.0);
        let mut input = vec![0.0; 8 * FRAME_SAMPLES];
        input.extend(noise(8 * FRAME_SAMPLES, 3));
        let frames = frames_of(config, &[&input, &input]);
        let shortest = config.shortest_frame_len(Windows::LONG_SINE, 0);
        assert!(frames[1..6].iter().all(|f| f.len() == shortest));
        assert!(frames[10..15].iter().all(|f| f.len() > 4 * shortest));
        for frame in &frames {
            let frame = adts_frame(frame);
            assert_eq!(frame.adts_buffer_fullness, 0x7FF);
            let block = raw_data_block(frame.raw_data_block, config.sampling_frequency);
            assert!(!block.ids.contains(&6), "ID_FIL: {:?}", block.ids);
        }
    }

    /// A larger noise offset allows more noise, and gives shorter frames.
    #[test]
    fn the_noise_offset_trades_length_for_noise() {
        let input = noise(12 * FRAME_SAMPLES, 4);
        let right: Vec<f64> = input.iter().map(|x| 0.5 * x).collect();
        let lens: Vec<usize> = [-6.0, 0.0, 6.0]
            .into_iter()
            .map(|offset| {
                let frames = frames_of(variable(Framing::Raw, offset), &[&input, &right]);
                frames.iter().map(Vec::len).sum()
            })
            .collect();
        assert!(lens[0] > lens[1] && lens[1] > lens[2], "{lens:?}");
    }

    /// Where the noise allowed asks for more than the decoder's input
    /// buffer holds, each frame is as long as it may be, at most, and
    /// coded as at a constant rate within it; in mono at 8 kHz, whose
    /// lines are few, and in raw frames and ADTS, with the CRC.
    #[test]
    fn variable_frames_are_no_longer_than_the_buffer() {
        use crate::noiseless::decoder::{BitReader, element};

        let input = noise(6 * FRAME_SAMPLES, 5);
        for (framing, crc_protected) in [(Framing::Raw, false), (Framing::Adts, true)] {
            let mut config = StreamConfig {
                sampling_frequency: SamplingFrequency::Hz8000,
                channel_configuration: ChannelConfiguration::Mono,
                crc_protected,
                ..variable(framing, -60.0)
            };
            config.bits_per_second = config.nominal_bits_per_second(-60.0);
            assert_eq!(config.bits_per_second, config.max_bits_per_second());
            let max = config.max_variable_frame_len();
            assert_eq!(max, 6144 / 8);
            let frames = frames_of(config, &[&input]);
            assert!(frames.iter().all(|f| f.len() <= max));
            assert!(
                frames[2..6].iter().any(|f| f.len() > max - 16),
                "{framing:?}"
            );
            for frame in &frames {
                let mut r = BitReader::new(&frame[config.header_len()..]);
                element(&mut r, config.sampling_frequency);
            }
        }
    }

    /// The tools that pay only where bits are few are chosen by the
    /// offset, for the whole stream: perceptual noise substitution, which
    /// makes every ADTS header MPEG-4's, by its nominal rate; intensity
    /// stereo from a border that falls with the offset to Annex C's; and
    /// the threshold in quiet, read as noise has it, then between the two
    /// readings, then as printed, and lowered less and less, then not at
    /// all, as at a constant rate.
    #[test]
    fn the_offset_chooses_the_tools() {
        let border = |offset| variable(Framing::Adts, offset).intensity_border();
        assert_eq!(border(3.8), None);
        let borders = [4.0, 5.0, 6.0, 6.9].map(|o| border(o).unwrap());
        assert!(borders.windows(2).all(|w| w[0] > w[1]), "{borders:?}");
        assert_eq!(border(7.0), Some(intensity::BORDER_HZ));
        assert_eq!(border(12.0), Some(intensity::BORDER_HZ));
        let quiet = |offset| variable(Framing::Adts, offset).quiet_threshold();
        assert_eq!(quiet(3.1), QuietThreshold::Noise);
        let share = quiet(6.1).printed_share();
        assert!((share - 0.5).abs() < 1e-9, "{share}");
        assert!(matches!(quiet(6.1), QuietThreshold::Between(_)));
        assert_eq!(quiet(9.1), QuietThreshold::Printed);
        let lowered = |offset| variable(Framing::Adts, offset).quiet_lowered_db();
        assert_eq!(lowered(-3.0), rate::QUIET_LOWERING_DB);
        assert_eq!(lowered(0.0), rate::QUIET_LOWERING_DB);
        assert!((lowered(3.0) - rate::QUIET_LOWERING_DB / 2.0).abs() < 1e-9);
        assert_eq!(lowered(6.0), 0.0);
        assert_eq!(config(Framing::Adts, 128_000).quiet_lowered_db(), 0.0);
        for (offset, few) in [(-3.0, false), (12.0, true)] {
            let config = StreamConfig {
                noise_substitution: true,
                ..variable(Framing::Adts, offset)
            };
            assert_eq!(config.noise_border().is_some(), few);
            assert_eq!(config.adts_header().unwrap().mpeg4, few);
        }
    }

    /// As the noise offset rises across the offsets over which the
    /// threshold in quiet is lowered less and less (see
    /// [`StreamConfig::quiet_lowered_db`]) and its reading goes from the one
    /// to the other (see [`StreamConfig::quiet_threshold`]), the length of a
    /// stream falls without a break: by at most 20% from one offset to the
    /// next, 0.1 dB above it, and it rises by at most 6%, as a band's coding
    /// changes, or as the loops fit frames to their least bits in coarse
    /// steps; of partly correlated loud noise, of tones and of noise 50 dB
    /// below full scale; and of noise some 75 dB below full scale in one
    /// channel at 48 kHz, repeated, which also rises by at most 2% in steps
    /// of 0.001 dB from 0.5 to 0.6 dB, where it grew by 7% before the bits
    /// the loops leave were spent (see
    /// [`spend`](crate::loops::allowance::spend)). Over the 11 dB swept,
    /// each of the first three falls by a quarter at least. Where the
    /// threshold in quiet is lowered less and read more as printed at once,
    /// from 3.1 to 6 dB, the quiet noise falls by up to 11% a step, and fell
    /// by 21% in a step of 0.2 dB before the loops' bits were spent and it
    /// was swept finer. Switched at a nominal rate of 0.8 bits a sample,
    /// with the offset lowered by 6.2 dB, the reading made a second of such
    /// loud noise 55% longer from 6.10 to 6.11 dB.
    #[test]
    fn the_length_falls_with_the_offset_without_a_break() {
        let len = 6 * FRAME_SAMPLES;
        let (a, b) = (noise(len, 8), noise(len, 9));
        let tone = |f: f64, gain: f64| -> Vec<f64> {
            (0..len)
                .map(|t| gain * (std::f64::consts::TAU * f * t as f64 / 44_100.0).sin())
                .collect()
        };
        let add =
            |x: &[f64], y: &[f64]| -> Vec<f64> { x.iter().zip(y).map(|(x, y)| x + y).collect() };
        let scale = |x: &[f64], gain: f64| -> Vec<f64> { x.iter().map(|x| gain * x).collect() };
        let inputs: [[Vec<f64>; 2]; 3] = [
            [scale(&a, 3.0), add(&scale(&a, 1.5), &scale(&b, 1.5))],
            [
                add(&tone(440.0, 0.1), &tone(3300.0, 0.05)),
                add(&tone(440.0, 0.08), &tone(5500.0, 0.03)),
            ],
            [scale(&a, 0.012), scale(&a, 0.005)],
        ];
        let low = rate::BLENDED_QUIET_DB
            .start()
            .min(*rate::LOWERED_QUIET_DB.start());
        let high = rate::BLENDED_QUIET_DB
            .end()
            .max(*rate::LOWERED_QUIET_DB.end());
        // Quiet noise in one channel at 48 kHz, 8192 samples repeated: the
        // loops, given the frames' least bits, made it 7% longer from 0.559
        // to 0.560 dB before the bits they leave were spent.
        let mut period = noise(8192, 14);
        period.iter_mut().for_each(|x| *x *= 0.0012);
        let quiet = period.repeat(3);
        let mono = |offset| {
            let mut config = StreamConfig {
                sampling_frequency: SamplingFrequency::Hz48000,
                channel_configuration: ChannelConfiguration::Mono,
                ..variable(Framing::Adts, offset)
            };
            config.bits_per_second = config.nominal_bits_per_second(offset);
            config
        };
        let cases = inputs
            .iter()
            .map(|[left, right]| {
                (
                    variable as fn(Framing, f64) -> StreamConfig,
                    vec![&left[..], &right[..]],
                )
            })
            .collect::<Vec<_>>();
        let lengths = |config: StreamConfig, channels: &[&[f64]]| -> usize {
            frames_of(config, channels).iter().map(Vec::len).sum()
        };
        for (k, (config, channels)) in cases.iter().enumerate() {
            let (mut first, mut last): (Option<usize>, Option<usize>) = (None, None);
            let mut offset = low - 1.0;
            while offset <= high + 1.0 {
                let len = lengths(config(Framing::Adts, offset), channels);
                if let Some(last) = last {
                    let ratio = len as f64 / last as f64;
                    assert!(
                        (0.8..=1.06).contains(&ratio),
                        "input {k}, {offset:.1} dB: {last} to {len} bytes"
                    );
                }
                first = first.or(Some(len));
                last = Some(len);
                offset += 0.1;
            }
            let (first, last) = (first.unwrap(), last.unwrap());
            assert!(4 * last <= 3 * first, "input {k}: {first} to {last} bytes");
        }
        let mut last = None;
        for step in 0..=(10.0 * (high - low + 2.0)) as usize {
            let offset = low - 1.0 + 0.1 * step as f64;
            let len = lengths(mono(offset), &[&quiet]);
            if let Some(last) = last {
                let ratio = len as f64 / last as f64;
                assert!(
                    (0.8..=1.06).contains(&ratio),
                    "quiet mono, {offset:.1} dB: {last} to {len} bytes"
                );
            }
            last = Some(len);
        }
        let mut last = None;
        for step in 0..=100 {
            let offset = 0.5 + 0.001 * f64::from(step);
            let len = lengths(mono(offset), &[&quiet]);
            if let Some(last) = last {
                assert!(
                    len as f64 <= 1.02 * last as f64,
                    "quiet mono, {offset:.3} dB: {last} to {len} bytes"
                );
            }
            last = Some(len);
        }
    }

    /// At a variable rate at a low offset, noise some 60 dB below full
    /// scale, whose bands the threshold in quiet governs, is coded with the
    /// threshold lowered, in frames more than twice as long as without (2177
    /// bytes against 780); at a constant rate nothing is lowered.
    #[test]
    fn quiet_noise_is_coded_further_where_the_threshold_is_lowered() {
        let len = 6 * FRAME_SAMPLES;
        let [left, right] = [noise(len, 12), noise(len, 13)]
            .map(|x| x.iter().map(|x| 0.003 * x).collect::<Vec<_>>());
        let bytes = |config: StreamConfig, lowered: bool| {
            let quiet = config.quiet_threshold();
            let quiet = if lowered {
                quiet.lowered(config.quiet_lowered_db())
            } else {
                quiet.into()
            };
            let mut stream = Vec::new();
            let mut encoder = StreamEncoder::with_quiet(config, quiet);
            encoder.encode(&[&left, &right], &mut stream);
            encoder.finish(&mut stream);
            stream
        };
        let variable = variable(Framing::Adts, -3.0);
        assert_eq!(variable.quiet_lowered_db(), rate::QUIET_LOWERING_DB);
        let (lowered, unlowered) = (bytes(variable, true), bytes(variable, false));
        assert_eq!(frames_of(variable, &[&left, &right]).concat(), lowered);
        assert!(
            lowered.len() > 2 * unlowered.len(),
            "{} against {} bytes",
            lowered.len(),
            unlowered.len()
        );
        let constant = config(Framing::Adts, 128_000);
        assert_eq!(bytes(constant, true), bytes(constant, false));
    }

    /// At a mean rate, a second of loud noise, which takes far more than
    /// the nominal rate's offset gives, is steered to the mean within a few
    /// frames: the stream runs over by a tenth at most.
    #[test]
    fn a_short_stream_learns_what_its_content_takes() {
        let config = StreamConfig {
            control: Control::Mean,
            ..config(Framing::Adts, 128_000)
        };
        let len = 43 * FRAME_SAMPLES;
        let [left, right] = [noise(len, 10), noise(len, 11)]
            .map(|x| x.iter().map(|x| 3.0 * x).collect::<Vec<f64>>());
        let frames = frames_of(config, &[&left, &right]);
        let bytes: usize = frames.iter().map(Vec::len).sum();
        let mean = 128_000.0 * FRAME_SAMPLES as f64 / 44_100.0 / 8.0;
        let ratio = bytes as f64 / (frames.len() as f64 * mean);
        assert!((1.0..1.1).contains(&ratio), "{ratio}");
    }
}
