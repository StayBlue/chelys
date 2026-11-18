// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Encoding a stream of frames from PCM: [`StreamEncoder`], which runs
//! each channel through the filterbank and the encoding process of Annex C
//! to frames at the configured rate, and the frames' layout, which also
//! encodes streams from quantised granules.

#[cfg(test)]
mod check;
mod coding;
mod frame_encoder;
mod layout;
#[cfg(test)]
mod random_streams;

#[cfg(feature = "unstable")]
pub use coding::{encode_pcm_with_global_gain, encode_with_block_types};
pub use layout::max_ancillary_bits;
#[cfg(feature = "unstable")]
pub use layout::{encode_granules, encode_silence, write_frame};

use crate::header::{Bitrate, Emphasis, Flags, Header, Mode, SamplingFrequency};
use crate::info_tag;
use crate::rate::StreamRate;
use crate::reservoir::Reservoir;
use coding::ModelCoding;
use frame_encoder::FrameEncoder;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreamConfig {
    pub rate: StreamRate,
    pub sampling_frequency: SamplingFrequency,
    pub mode: Mode,
    pub crc_protected: bool,
    /// The `private_bit`, `copyright` and `original/copy` of every frame.
    pub flags: Flags,
}

impl StreamConfig {
    /// The bitrate of every frame.
    ///
    /// # Panics
    ///
    /// If the rate is not [`StreamRate::Constant`].
    #[cfg(any(test, feature = "unstable"))]
    fn constant_bitrate(&self) -> Bitrate {
        match self.rate {
            StreamRate::Constant(bitrate) => bitrate,
            rate => panic!("{rate:?} is not a constant bitrate"),
        }
    }

    /// Whether the configured mode allows MS stereo and intensity stereo:
    /// joint stereo with `ms_stereo` or `intensity_stereo` set, in that order.
    fn joint_stereo(&self) -> (bool, bool) {
        match self.mode {
            Mode::JointStereo {
                intensity_stereo,
                ms_stereo,
            } => (ms_stereo, intensity_stereo),
            _ => (false, false),
        }
    }

    /// The header of a frame at `bitrate`, with `mode_extension` switching
    /// MS stereo and intensity stereo on for this frame alone if `ms_stereo`
    /// and `intensity_stereo` are set.
    fn frame_header(
        &self,
        bitrate: Bitrate,
        padding: bool,
        ms_stereo: bool,
        intensity_stereo: bool,
    ) -> Header {
        let mode = match self.mode {
            Mode::JointStereo { .. } => Mode::JointStereo {
                intensity_stereo,
                ms_stereo,
            },
            mode => {
                assert!(
                    !ms_stereo && !intensity_stereo,
                    "MS and intensity stereo need the joint stereo mode"
                );
                mode
            }
        };
        Header {
            crc_protected: self.crc_protected,
            bitrate,
            sampling_frequency: self.sampling_frequency,
            padding,
            mode,
            flags: self.flags,
            emphasis: Emphasis::None,
        }
    }
}

/// Encodes PCM, one slice of samples (nominally -1.0 to 1.0) per channel, in
/// one go, with a [`StreamEncoder`], without a Xing/Info tag.
///
/// # Panics
///
/// If the channel count does not match the configured mode, or the channels
/// have different numbers of samples.
#[cfg(any(test, feature = "unstable"))]
pub fn encode_pcm(config: &StreamConfig, channels: &[&[f64]]) -> Vec<u8> {
    let mut encoder = StreamEncoder::new(*config);
    let mut out = Vec::new();
    encoder.encode(channels, &mut out);
    encoder.finish(&mut out);
    out
}

/// Encodes PCM like [`encode_pcm`], with a Xing/Info tag at the start if
/// the stream has one (see [`StreamEncoder::with_info_tag`]).
///
/// # Panics
///
/// As [`encode_pcm`].
#[cfg(any(test, feature = "unstable"))]
pub fn encode_pcm_with_info_tag(config: &StreamConfig, channels: &[&[f64]]) -> Vec<u8> {
    let mut encoder = StreamEncoder::with_info_tag(*config);
    let mut out = Vec::new();
    encoder.encode(channels, &mut out);
    if let Some(tag) = encoder.finish(&mut out).info_tag {
        out[..tag.len()].copy_from_slice(&tag);
    }
    out
}

/// Encodes PCM at the configured rate as it arrives.
///
/// Psychoacoustic model 2 chooses each granule's block type, switching to
/// short blocks for attacks, and sets the distortion each scalefactor band
/// allows and the perceptual entropy of each granule. Each granule of each
/// channel gets an equal share of its frame's main data, the `mean_bits` of
/// clause C.1.5.3.4, plus what the bit reservoir lends it, and the iteration
/// loops shape the quantisation noise to the allowed distortion within those
/// bits.
///
/// In joint stereo, frames whose channels are alike are coded in MS stereo
/// if the mode allows it, and frames short of bits are coded in intensity
/// stereo if the mode allows that (Annex G.2).
///
/// At a [`StreamRate::Variable`] each frame is kept at a bitrate at which every
/// band's noise stays within its allowance (see [`loops::excess`](crate::loops::excess)). In
/// MPEG-1, a frame of normal blocks is quantised once, each granule left as
/// coarse as its allowance lets it, and kept at the lowest bitrate whose
/// frame, with the bit reservoir, holds it. Other frames, and those that
/// no frame holds, are coded at bitrates from the one below that of the
/// frame before it, as
/// [`rate::search_variable`](crate::rate::search_variable) tries them, and
/// kept at the lowest of those at which their noise stays within its
/// allowance.
/// Frames are not padded. Intensity stereo, which gives up the stereo image
/// above its bound for bits, is tried only at the highest bitrate, where a
/// larger frame is no longer to be had.
///
/// Spec reading: a padded frame is one slot longer to keep a mean bitrate
/// (clause 2.4.2.3: "the frame contains an additional slot to adjust the
/// mean bitrate to the sampling frequency"); at a variable bitrate there is
/// no mean to keep, and the bit reservoir carries what a frame does not
/// use. Settled by decoding with ffmpeg, mpg123 and LAME (black box): at
/// 44.1, 22.05 and 11.025 kHz, where a constant bitrate pads, all three
/// decode streams of unpadded frames at a variable bitrate without
/// complaint, and find each frame and its main data; the conformance test
/// is `every_decoder_follows_the_bitrate_from_frame_to_frame`.
///
/// The stream does not depend on how the input is divided between calls to
/// [`encode`](Self::encode).
pub struct StreamEncoder {
    frames: FrameEncoder<ModelCoding>,
    tag: Option<TagState>,
}

/// What [`StreamEncoder::finish`] gives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamEnd {
    /// The Xing/Info tag's frame, for an encoder made
    /// [`with_info_tag`](StreamEncoder::with_info_tag) whose stream has one.
    pub info_tag: Option<Vec<u8>>,
    /// The frames of audio, after the tag's frame if there is one.
    pub frames: u64,
    /// The samples each channel was given.
    pub samples: u64,
    /// The bits of ancillary data that the last frames had no room for.
    pub ancillary_dropped: usize,
}

/// The Xing/Info tag of a [`StreamEncoder`] that writes one.
struct TagState {
    header: Header,
    /// Whether the tag's frame has been written, as a placeholder.
    written: bool,
    /// The CRC of the stream after the tag's frame so far.
    music_crc: u16,
}

impl StreamEncoder {
    /// # Panics
    ///
    /// If the rate's bitrates are not of the version of the sampling
    /// frequency, or a [`StreamRate::Variable`] has its lowest bitrate above its
    /// highest.
    pub fn new(config: StreamConfig) -> Self {
        let fs = config.sampling_frequency;
        let coding = ModelCoding::new(&config);
        let shortest = config.rate.lowest().frame_len(fs, true);
        let reservoir = Reservoir::new(shortest, fs.version());
        Self {
            frames: FrameEncoder::new(config, coding, reservoir),
            tag: None,
        }
    }

    /// An encoder whose stream starts with a frame for a Xing/Info tag (see
    /// `info_tag`): the number of frames and bytes, a table to seek by
    /// (but at a constant bitrate whose frames are too short for it), and
    /// the samples added before and after the input, which make the
    /// decoded output the input exactly for decoders that apply them. As
    /// those are known only at the end, the frame is written as a
    /// placeholder, silent and with no tag, and [`finish`](Self::finish)
    /// returns it with the tag, to be written over it. A stream in free
    /// format whose frames are too short for the tag has none, as its
    /// bitrate cannot change for the tag's frame ([`info_tag::header`]).
    ///
    /// # Panics
    ///
    /// As [`new`](Self::new).
    pub fn with_info_tag(config: StreamConfig) -> Self {
        Self {
            tag: info_tag::header(&config).map(|header| TagState {
                header,
                written: false,
                music_crc: info_tag::MUSIC_CRC_START,
            }),
            ..Self::new(config)
        }
    }

    /// Gives each frame up to `bits_per_frame` bits of ancillary data, from
    /// the bytes queued with [`push_ancillary`](Self::push_ancillary), after
    /// its audio. Its audio then has as many bits fewer, as ancillary data
    /// "will reduce the number of bits available for audio" (clause
    /// C.1.5.3.6); a frame with less data queued keeps the rest for audio.
    ///
    /// The ancillary data of a frame are all its bits from the end of its
    /// audio to where the next frame's main data begins (clause 2.4.2.8):
    /// the queued bits come first, and any after them are zero. As the
    /// standard leaves the bits "user definable", a reader cannot tell the
    /// queued bits from those zeros by the stream alone, and the data needs
    /// a framing of its own. Nor is it kept from imitating a syncword, which
    /// "may hamper synchronization" (clause C.1.5.3.6).
    ///
    /// # Panics
    ///
    /// If `bits_per_frame` is more than the main data slot of the stream's
    /// shortest frame holds.
    pub fn with_ancillary(mut self, bits_per_frame: usize) -> Self {
        let slot = max_ancillary_bits(&self.frames.config);
        assert!(
            bits_per_frame <= slot,
            "{bits_per_frame} bits of ancillary data exceed the {slot} of the shortest frame"
        );
        self.frames.ancillary.set_bits_per_frame(bits_per_frame);
        self
    }

    /// The samples of each channel a frame codes.
    pub fn samples_per_frame(&self) -> usize {
        self.frames.samples_per_frame()
    }

    /// The length of the Xing/Info tag's frame, which [`finish`](Self::finish)
    /// returns, if the stream has one.
    pub fn info_tag_len(&self) -> Option<usize> {
        self.tag.as_ref().map(|tag| tag.header.frame_len())
    }

    /// Codes mixed blocks in place of short ones: the two lowest subbands of
    /// a granule with an attack keep the frequency resolution of the normal
    /// window (clause 2.4.2.7). See
    /// [`BlockSwitching::with_mixed_blocks`](crate::psychoacoustic::switching::BlockSwitching::with_mixed_blocks).
    ///
    /// Frames with mixed blocks are not coded in intensity stereo (see
    /// `intensity_frame`).
    ///
    /// # Panics
    ///
    /// At MPEG-2.5's 8 kHz, where mixed blocks are not used (see
    /// [`scalefactor_bands::mixed_long_bands`](crate::scalefactor_bands::mixed_long_bands)).
    pub fn with_mixed_blocks(mut self) -> Self {
        let fs = self.frames.config.sampling_frequency;
        self.frames.coding.use_mixed_blocks(fs);
        self
    }

    /// Queues `bytes` of ancillary data for the frames coded from now on,
    /// as many bits a frame as [`with_ancillary`](Self::with_ancillary)
    /// allows, each byte's most significant bit first.
    pub fn push_ancillary(&mut self, bytes: &[u8]) {
        self.frames.ancillary.push(bytes);
    }

    /// The bits of ancillary data queued that no frame has carried yet. A
    /// frame is coded once the granule after it has arrived;
    /// [`finish`](Self::finish) drops what its last frames have no room
    /// for.
    pub fn ancillary_pending(&self) -> usize {
        self.frames.ancillary.pending()
    }

    /// Encodes `channels`, one slice of samples (nominally -1.0 to 1.0) per
    /// channel, and appends to `out` the bytes of the stream that are final.
    ///
    /// A frame is coded once the granule after it has arrived, as its block
    /// type depends on whether that granule has an attack, and its bytes are
    /// final once no later frame's main data can begin in them: the bytes
    /// the bit reservoir holds, at most 511 of main data (255 at the lower
    /// sampling frequencies), and the frames
    /// after them stay behind until later calls.
    ///
    /// # Panics
    ///
    /// If the channel count does not match the configured mode, or the
    /// channels have different numbers of samples.
    pub fn encode(&mut self, channels: &[&[f64]], out: &mut Vec<u8>) {
        let start = self.write_placeholder(out);
        self.frames.encode(channels, out);
        self.add_to_crc(&out[start..]);
    }

    /// Ends the stream: pads the input with silence to whole frames, at
    /// least [`DELAY`](crate::DELAY) samples past the input, and appends
    /// the rest of the stream to `out`. For an encoder made
    /// [`with_info_tag`](Self::with_info_tag), gives the tag's frame, if the stream has one,
    /// to be written over the first bytes of the stream.
    pub fn finish(mut self, out: &mut Vec<u8>) -> StreamEnd {
        let start = self.write_placeholder(out);
        let config = self.frames.config;
        let ended = self.frames.finish(out);
        let info_tag = self.tag.map(|mut tag| {
            tag.music_crc = info_tag::music_crc(tag.music_crc, &out[start..]);
            let stream = info_tag::Stream {
                config: &config,
                frame_lens: &ended.frame_lens,
                samples: ended.samples,
                music_crc: tag.music_crc,
            };
            info_tag::frame(&tag.header, &stream)
        });
        StreamEnd {
            info_tag,
            frames: ended.frame_lens.len() as u64,
            samples: ended.samples,
            ancillary_dropped: ended.ancillary_dropped,
        }
    }

    /// Writes the tag's placeholder to `out` if it is the first thing
    /// written, and returns where the frames after it start.
    fn write_placeholder(&mut self, out: &mut Vec<u8>) -> usize {
        if let Some(tag) = self.tag.as_mut().filter(|tag| !tag.written) {
            out.extend(info_tag::placeholder(&tag.header));
            tag.written = true;
        }
        out.len()
    }

    fn add_to_crc(&mut self, bytes: &[u8]) {
        if let Some(tag) = &mut self.tag {
            tag.music_crc = info_tag::music_crc(tag.music_crc, bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::check;
    use super::layout::main_data_bits;
    use super::*;
    use crate::DELAY;
    use crate::granule::GRANULE_LINES;
    use crate::header::{PaddingSchedule, Version};
    use crate::rate::{self, MeanRateSchedule};

    /// A deterministic sequence of samples from -1.0 to 1.0 (an LCG).
    fn noise(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (*seed >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    /// However the input is divided between calls, the encoder writes the
    /// same stream, of whole frames for the input padded with silence to at
    /// least [`DELAY`] samples past it, and holds back no more than the next frame's main data can
    /// begin in: its `main_data_begin` bytes, and the header and side
    /// information of each frame after the first of them.
    #[test]
    fn streaming_does_not_depend_on_how_the_input_is_divided() {
        let config = StreamConfig {
            rate: StreamRate::Constant(Bitrate::from_kbps(Version::Mpeg1, 32).unwrap()),
            sampling_frequency: SamplingFrequency::Hz44100,
            mode: Mode::JointStereo {
                intensity_stereo: false,
                ms_stereo: true,
            },
            crc_protected: false,
            flags: Flags::default(),
        };
        // Silence fills the reservoir to its limit, past which the rest is
        // stuffing, and the noise after it reaches back into it.
        let mut seed = 1;
        let samples = 12 * Version::Mpeg1.samples_per_frame() + 300;
        let pcm: Vec<[f64; 2]> = (0..samples)
            .map(|n| {
                let level = if n < 9 * Version::Mpeg1.samples_per_frame() {
                    0.0
                } else {
                    0.5
                };
                [level * noise(&mut seed), level * noise(&mut seed)]
            })
            .collect();
        let (left, right): (Vec<f64>, Vec<f64>) = pcm.iter().map(|&[l, r]| (l, r)).unzip();
        let whole = encode_pcm(&config, &[&left, &right]);
        check::frames(&whole);
        let bitrate = config.constant_bitrate();
        let fs = config.sampling_frequency;
        let mut padding = PaddingSchedule::new(bitrate, fs);
        let frames = (samples + DELAY).div_ceil(Version::Mpeg1.samples_per_frame());
        let frame_lens = (0..frames).map(|_| bitrate.frame_len(fs, padding.next_padding()));
        assert_eq!(whole.len(), frame_lens.sum::<usize>());

        // Frames without padding have the smallest slots, and so span the
        // most frames for the same bytes.
        let header = config.frame_header(bitrate, false, false, false);
        let slot = main_data_bits(&header) / 8;
        let overhead = header.frame_len() - slot;
        let streams = std::thread::scope(|scope| {
            [
                1,
                100,
                GRANULE_LINES,
                Version::Mpeg1.samples_per_frame() - 1,
                Version::Mpeg1.samples_per_frame() + GRANULE_LINES + 1,
            ]
            .map(|chunk| {
                let (left, right) = (&left, &right);
                scope.spawn(move || {
                    let mut encoder = StreamEncoder::new(config);
                    let mut out = Vec::new();
                    let mut furthest = 0;
                    for (left, right) in left.chunks(chunk).zip(right.chunks(chunk)) {
                        encoder.encode(&[left, right], &mut out);
                        let begin = encoder.frames.reservoir_bytes;
                        furthest = furthest.max(begin);
                        let held = encoder.frames.writer.out.len();
                        let most = begin + begin.div_ceil(slot) * overhead;
                        assert!(held <= most, "{held} bytes held in chunks of {chunk}");
                    }
                    encoder.finish(&mut out);
                    (chunk, out, furthest)
                })
            })
            .map(|stream| stream.join().expect("the encoding passes"))
        });
        for (chunk, stream, furthest) in streams {
            assert!(stream == whole, "chunks of {chunk}");
            assert!(
                furthest > slot,
                "main data reaches back {furthest} bytes, within the frame before"
            );
        }
    }

    /// Silence, full-scale noise, a quiet tone, bursts of noise with sharp
    /// onsets, and silence again, over `seconds` seconds at `fs`, in two
    /// channels that differ in the noise.
    fn varied(fs: SamplingFrequency, seconds: f64) -> [Vec<f64>; 2] {
        let hz = f64::from(fs.hz());
        let mut seed = 7;
        let (mut left, mut right) = (Vec::new(), Vec::new());
        for t in 0..(seconds * hz) as usize {
            let s = t as f64 / hz / seconds;
            let (common, own) = (noise(&mut seed), noise(&mut seed));
            let tone = (std::f64::consts::TAU * 440.0 * t as f64 / hz).sin();
            let (l, r) = match s {
                s if s < 0.15 => (0.0, 0.0),
                s if s < 0.4 => (0.9 * common, 0.9 * (0.5 * common + 0.5 * own)),
                s if s < 0.6 => (0.01 * tone, 0.01 * tone),
                s if s < 0.9 => {
                    let burst = 0.7 * (-(s - 0.6) % 0.1 * seconds / 0.02).exp();
                    (burst * common + 0.05 * tone, burst * own)
                }
                _ => (0.0, 0.0),
            };
            left.push(l);
            right.push(r);
        }
        [left, right]
    }

    const MS: Mode = Mode::JointStereo {
        intensity_stereo: false,
        ms_stereo: true,
    };

    fn variable(fs: SamplingFrequency, kbps: [u32; 2], noise_offset_db: f64) -> StreamRate {
        let at = |kbps| Bitrate::from_kbps(fs.version(), kbps).unwrap();
        StreamRate::Variable(rate::VariableRate {
            lowest: at(kbps[0]),
            highest: at(kbps[1]),
            noise_offset_db,
        })
    }

    fn encode(config: StreamConfig, [left, right]: &[Vec<f64>; 2]) -> Vec<u8> {
        match config.mode {
            Mode::SingleChannel => encode_pcm(&config, &[left]),
            _ => encode_pcm(&config, &[left, right]),
        }
    }

    /// A stream in dual channel is the stream in stereo but for the `mode`
    /// of each header, and the `crc_check` that covers it: "the coding
    /// process is the same as for the stereo mode" (clause 2.1.52).
    #[test]
    fn dual_channel_codes_as_stereo() {
        use SamplingFrequency::*;
        let mean = |fs: SamplingFrequency, kbps: u32| {
            StreamRate::Mean(rate::MeanRate::new(fs.version(), kbps * 1000).unwrap())
        };
        let at = |fs: SamplingFrequency, kbps| {
            StreamRate::Constant(Bitrate::from_kbps(fs.version(), kbps).unwrap())
        };
        let cases = [
            (Hz44100, at(Hz44100, 128), false),
            (Hz48000, mean(Hz48000, 100), true),
            (Hz32000, variable(Hz32000, [32, 320], 0.0), false),
            (Hz22050, at(Hz22050, 64), true),
            (Hz11025, variable(Hz11025, [8, 64], 4.0), false),
        ];
        for (fs, rate, crc_protected) in cases {
            let pcm = varied(fs, 1.0);
            let config = |mode| StreamConfig {
                rate,
                sampling_frequency: fs,
                mode,
                crc_protected,
                flags: Flags::default(),
            };
            let stereo = encode(config(Mode::Stereo), &pcm);
            let mut dual = encode(config(Mode::DualChannel), &pcm);
            assert_eq!(dual.len(), stereo.len(), "{fs:?}");
            let mut at = 0;
            for frame in check::frames(&stereo) {
                assert_eq!(stereo[at + 3] >> 6, 0b00, "{fs:?}: stereo at {at}");
                assert_eq!(dual[at + 3] >> 6, 0b10, "{fs:?}: dual channel at {at}");
                dual[at + 3] &= 0b0011_1111;
                if crc_protected {
                    dual[at + 4..at + 6].copy_from_slice(&stereo[at + 4..at + 6]);
                }
                at += frame.len;
            }
            assert!(dual == stereo, "{fs:?}");
        }
    }

    /// Each frame's ancillary data, from the end of its Huffman code bits to
    /// where the next frame's main data begins (clause 2.4.2.8), starts with
    /// the next of the queued bits, as many as it may carry, and is zero
    /// after them, at constant, mean and variable bitrates, in both
    /// versions, with a share of a whole number of bytes or not, and with
    /// data that imitates headers. Once the queue runs dry the frames carry
    /// none, and without ancillary data the bits there are all zero.
    #[test]
    fn frames_carry_the_queued_ancillary_data() {
        use SamplingFrequency::*;
        let at = |fs: SamplingFrequency, kbps| {
            StreamRate::Constant(Bitrate::from_kbps(fs.version(), kbps).unwrap())
        };
        let mean = StreamRate::Mean(rate::MeanRate::new(Version::Mpeg1, 100_000).unwrap());
        let cases = [
            (Hz44100, at(Hz44100, 128), MS, false, 1000),
            (Hz48000, mean, Mode::Stereo, true, 2000),
            (
                Hz32000,
                variable(Hz32000, [32, 320], 0.0),
                Mode::SingleChannel,
                false,
                300,
            ),
            (Hz22050, at(Hz22050, 64), Mode::Stereo, true, 77),
            (Hz8000, at(Hz8000, 16), Mode::SingleChannel, false, 8),
        ];
        for (fs, rate, mode, crc_protected, per_frame) in cases {
            let case = format!("{fs:?}, {per_frame} bits a frame");
            let config = StreamConfig {
                rate,
                sampling_frequency: fs,
                mode,
                crc_protected,
                flags: Flags::default(),
            };
            let pcm = varied(fs, 1.0);
            let channels: [&[f64]; 2] = [&pcm[0], &pcm[1]];
            let channels = &channels[..mode.channels()];
            let frames = (pcm[0].len() + DELAY).div_ceil(fs.version().samples_per_frame());
            // Enough for two thirds of the frames, headers among it.
            let mut seed = 3;
            let mut data: Vec<u8> = (0..frames * per_frame * 2 / 3 / 8)
                .map(|_| (noise(&mut seed) * 128.0 + 128.0) as u8)
                .collect();
            data[..8].copy_from_slice(&[0xFF, 0xFB, 0x90, 0x64, 0xFF, 0xF3, 0x48, 0xC4]);
            let bits: Vec<bool> = (0..8 * data.len())
                .map(|i| (data[i / 8] >> (7 - i % 8)) & 1 == 1)
                .collect();

            let mut encoder = StreamEncoder::new(config).with_ancillary(per_frame);
            encoder.push_ancillary(&data);
            let mut stream = Vec::new();
            encoder.encode(channels, &mut stream);
            assert!(encoder.ancillary_pending() < 8 * data.len(), "{case}");
            encoder.finish(&mut stream);

            let ancillary = check::ancillary(&stream);
            assert_eq!(ancillary.len(), frames, "{case}");
            let mut sent = 0;
            for (k, frame) in ancillary.iter().enumerate() {
                let carried = per_frame.min(bits.len() - sent);
                assert!(frame.len() >= carried, "{case}, frame {k}");
                assert_eq!(
                    frame[..carried],
                    bits[sent..sent + carried],
                    "{case}, frame {k}"
                );
                assert!(!frame[carried..].contains(&true), "{case}, frame {k}");
                sent += carried;
            }
            assert_eq!(sent, bits.len(), "{case}");

            let plain = check::ancillary(&encode_pcm(&config, channels));
            assert!(plain.iter().flatten().all(|&bit| !bit), "{case}");
        }
    }

    /// Ancillary data that fills the shortest frame's main data slot is
    /// allowed, and leaves its audio no bits of its own; any more is not.
    #[test]
    fn ancillary_data_fits_the_shortest_frame() {
        let config = StreamConfig {
            rate: StreamRate::Constant(Bitrate::from_kbps(Version::Mpeg1, 32).unwrap()),
            sampling_frequency: SamplingFrequency::Hz48000,
            mode: Mode::SingleChannel,
            crc_protected: false,
            flags: Flags::default(),
        };
        // 96 bytes, less the header and 17 of side information.
        let slot = 8 * (96 - 4 - 17);
        let pcm = &varied(SamplingFrequency::Hz48000, 0.5)[0];
        let mut encoder = StreamEncoder::new(config).with_ancillary(slot);
        encoder.push_ancillary(&vec![0x55; 200 * slot / 8]);
        let mut stream = Vec::new();
        encoder.encode(&[pcm], &mut stream);
        encoder.finish(&mut stream);
        for frame in check::ancillary(&stream) {
            assert!(frame.len() >= slot);
        }
        let result =
            std::panic::catch_unwind(|| StreamEncoder::new(config).with_ancillary(slot + 1));
        assert!(result.is_err());
    }

    /// The flags mark every frame, the Xing/Info tag's too, and change
    /// nothing else but the `crc_check` that covers them: `private_bit` is
    /// the last bit of the header's third byte, and `copyright` and
    /// `original/copy` the fifth and sixth of its fourth (clause 2.4.1.3).
    #[test]
    fn flags_mark_every_frame_and_change_nothing_else() {
        use SamplingFrequency::*;
        let at = |fs: SamplingFrequency, kbps| {
            StreamRate::Constant(Bitrate::from_kbps(fs.version(), kbps).unwrap())
        };
        let cases = [
            (Hz44100, at(Hz44100, 128), MS, true),
            (
                Hz32000,
                variable(Hz32000, [32, 320], 0.0),
                Mode::SingleChannel,
                false,
            ),
            (Hz22050, at(Hz22050, 64), Mode::Stereo, true),
        ];
        let flags = Flags {
            private: true,
            copyright: true,
            original: true,
        };
        for (fs, rate, mode, crc_protected) in cases {
            let pcm = varied(fs, 1.0);
            let config = |flags| StreamConfig {
                rate,
                sampling_frequency: fs,
                mode,
                crc_protected,
                flags,
            };
            let plain = encode(config(Flags::default()), &pcm);
            let mut marked = encode(config(flags), &pcm);
            assert_eq!(marked.len(), plain.len(), "{fs:?}");
            let mut at = 0;
            for frame in check::frames(&plain) {
                assert_eq!(plain[at + 2] & 1, 0, "{fs:?} at {at}");
                assert_eq!((plain[at + 3] >> 2) & 0b11, 0b00, "{fs:?} at {at}");
                assert_eq!(marked[at + 2] & 1, 1, "{fs:?} at {at}");
                assert_eq!((marked[at + 3] >> 2) & 0b11, 0b11, "{fs:?} at {at}");
                marked[at + 2] &= !1;
                marked[at + 3] &= !0b1100;
                if crc_protected {
                    marked[at + 4..at + 6].copy_from_slice(&plain[at + 4..at + 6]);
                }
                at += frame.len;
            }
            assert!(marked == plain, "{fs:?}");

            let channels: [&[f64]; 2] = [&pcm[0], &pcm[1]];
            let tagged = encode_pcm_with_info_tag(&config(flags), &channels[..mode.channels()]);
            assert_eq!(tagged[2] & 1, 1, "{fs:?}");
            assert_eq!((tagged[3] >> 2) & 0b11, 0b11, "{fs:?}");
        }
    }

    /// At a variable bitrate, frames switch among the bitrates allowed,
    /// silence takes the lowest and loud passages more, no frame is padded, and every frame's main data keeps to the rules of the bit
    /// reservoir: where it may begin, and what the buffer leaves beside the
    /// frame, down to none for frames longer than the buffer at 320 kbit/s
    /// and 32 kHz.
    #[test]
    fn variable_bitrate_streams_keep_the_rules_of_the_reservoir() {
        use SamplingFrequency::*;
        let cases = [
            (Hz44100, [32, 320], MS, false),
            (Hz32000, [96, 320], Mode::Stereo, true),
            (Hz48000, [32, 192], Mode::SingleChannel, false),
            (
                Hz22050,
                [8, 160],
                Mode::JointStereo {
                    intensity_stereo: true,
                    ms_stereo: true,
                },
                false,
            ),
            (Hz24000, [32, 64], Mode::Stereo, true),
            (Hz8000, [8, 64], Mode::SingleChannel, false),
        ];
        std::thread::scope(|scope| {
            for (fs, kbps, mode, crc_protected) in cases {
                scope.spawn(move || {
                    let config = StreamConfig {
                        rate: variable(fs, kbps, 0.0),
                        sampling_frequency: fs,
                        mode,
                        crc_protected,
                        flags: Flags::default(),
                    };
                    let input = varied(fs, 1.0);
                    let frames = check::frames(&encode(config, &input));
                    let samples = fs.version().samples_per_frame();
                    let case = format!("{fs:?} from {} to {} kbit/s", kbps[0], kbps[1]);
                    let expected = (input[0].len() + DELAY).div_ceil(samples);
                    assert_eq!(frames.len(), expected, "{case}");
                    assert!(frames.iter().all(|f| !f.padding), "{case}");
                    let kbps_of = |f: &check::Frame| f.bitrate.bits_per_second() / 1000;
                    let used: std::collections::BTreeSet<u32> =
                        frames.iter().map(kbps_of).collect();
                    assert_eq!(used.first(), Some(&kbps[0]), "{case}: {used:?}");
                    assert!(used.len() >= 2, "{case}: {used:?}");
                    assert!(used.last() <= Some(&kbps[1]), "{case}: {used:?}");
                    // The frames of the silence at the start, short of the
                    // one whose block the noise after it reaches into.
                    let silent = (0.15 * f64::from(fs.hz())) as usize / samples - 1;
                    assert!(
                        frames[..silent].iter().all(|f| kbps_of(f) == kbps[0]),
                        "{case}"
                    );
                });
            }
        });
    }

    /// In MPEG-1 at a variable bitrate, a frame of normal blocks is kept at
    /// the lowest bitrate whose frame, with the reservoir it finds, holds
    /// its main data: at the bitrate below, its slot and what the reservoir
    /// could give a frame of that length fall short of it.
    #[test]
    fn frames_of_normal_blocks_take_the_lowest_bitrate_that_holds_them() {
        use std::f64::consts::TAU;
        let fs = SamplingFrequency::Hz44100;
        let config = StreamConfig {
            rate: variable(fs, [32, 320], 0.0),
            sampling_frequency: fs,
            mode: MS,
            crc_protected: false,
            flags: Flags::default(),
        };
        // Noise that swells and fades slowly over a quiet tone, with no
        // attack to switch to short blocks.
        let hz = f64::from(fs.hz());
        let mut state = 0x9E37_79B9u32;
        let mut noise = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            f64::from(state) / f64::from(u32::MAX) * 2.0 - 1.0
        };
        let (left, right): (Vec<f64>, Vec<f64>) = (0..(2.0 * hz) as usize)
            .map(|t| {
                let s = t as f64 / hz;
                let swell = 0.25 * (1.0 - (TAU * s / 2.0).cos());
                let tone = 0.05 * (TAU * 440.0 * s).sin();
                (tone + swell * noise(), tone + swell * noise())
            })
            .unzip();
        let frames = check::frames(&encode(config, &[left, right]));
        let bitrates: Vec<Bitrate> = Bitrate::all(fs.version()).collect();
        let reservoir = |bitrate: Bitrate| {
            Reservoir::new(bitrate.frame_len(fs, true), fs.version()).max_bytes()
        };
        // What the reservoir keeps between frames.
        let kept = reservoir(bitrates[0]);
        let mut above_lowest = 0;
        for (k, frame) in frames.iter().enumerate() {
            let index = bitrates.iter().position(|&b| b == frame.bitrate).unwrap();
            let Some(below) = index.checked_sub(1).map(|index| bitrates[index]) else {
                continue;
            };
            let held = frame.free_before.min(kept).min(reservoir(below));
            let slot = main_data_bits(&config.frame_header(below, false, false, false));
            assert!(
                frame.audio_bits > 8 * held + slot,
                "frame {k} at {:?} fits at {below:?}",
                frame.bitrate
            );
            above_lowest += 1;
        }
        assert!(
            above_lowest > frames.len() / 2,
            "{above_lowest} of {}",
            frames.len()
        );
    }

    /// In free format every frame has `bitrate_index` 0 and `N` or `N + 1`
    /// slots, padded as at a bitrate of the table, keeps to the rules of the
    /// bit reservoir, from frames shorter than any of the table to frames
    /// longer than the decoder's buffer. Where its frames hold the
    /// Xing/Info tag, with its table or without, the tag's frame is in free
    /// format too; where they do not, the stream has none.
    #[test]
    fn free_format_streams_keep_their_bitrate() {
        use SamplingFrequency::*;
        let cases = [
            (Hz44100, 100_000, Mode::Stereo, false, true),
            (Hz48000, 12_667, Mode::SingleChannel, true, false),
            (Hz32000, 319_999, MS, false, true),
            (Hz22050, 50_000, MS, true, true),
            (Hz24000, 8_001, Mode::SingleChannel, false, false),
            (Hz16000, 159_999, Mode::Stereo, true, true),
            (Hz11025, 33_333, MS, false, true),
            (Hz8000, 20_000, Mode::SingleChannel, true, true),
        ];
        std::thread::scope(|scope| {
            for (fs, bits_per_second, mode, crc_protected, tagged) in cases {
                scope.spawn(move || {
                    let bitrate = Bitrate::free(fs.version(), bits_per_second).unwrap();
                    let config = StreamConfig {
                        rate: StreamRate::Constant(bitrate),
                        sampling_frequency: fs,
                        mode,
                        crc_protected,
                        flags: Flags::default(),
                    };
                    let case = format!("{fs:?} at {bits_per_second} bit/s");
                    let input = varied(fs, 1.0);
                    let mut encoder = StreamEncoder::with_info_tag(config);
                    let mut stream = Vec::new();
                    match mode {
                        Mode::SingleChannel => encoder.encode(&[&input[0]], &mut stream),
                        _ => encoder.encode(&[&input[0], &input[1]], &mut stream),
                    }
                    let tag = encoder.finish(&mut stream).info_tag;
                    assert_eq!(tag.is_some(), tagged, "{case}");
                    if let Some(tag) = tag {
                        stream[..tag.len()].copy_from_slice(&tag);
                    }
                    let frames = check::free_frames(&stream, bitrate);
                    let samples = fs.version().samples_per_frame();
                    let expected = (input[0].len() + DELAY).div_ceil(samples);
                    assert_eq!(frames.len(), expected + usize::from(tagged), "{case}");
                    // After k frames the stream holds floor(k * N) slots,
                    // with N as exact as 144 * bit_rate / fs (72 at the lower
                    // sampling frequencies) can be; the tag's frame, which
                    // is not padded, takes the place of the first.
                    let numerator = samples as u64 / 8 * u64::from(bits_per_second);
                    let hz = u64::from(fs.hz());
                    let mut total = 0;
                    for (k, frame) in frames.iter().enumerate() {
                        total += frame.len as u64;
                        let k = (k + 1 - usize::from(tagged)) as u64;
                        let floor = usize::from(tagged) as u64 * (numerator / hz);
                        assert_eq!(total, floor + k * numerator / hz, "{case}, frame {k}");
                    }
                    if bitrate.frame_len(fs, true) * 8 > 7680 {
                        assert!(frames.iter().all(|f| f.main_data_begin == 0), "{case}");
                    } else {
                        assert!(frames.iter().any(|f| f.main_data_begin > 0), "{case}");
                    }
                });
            }
        });
    }

    /// Allowing more noise gives smaller streams, and less noise larger.
    #[test]
    fn the_noise_offset_trades_size_for_noise() {
        let fs = SamplingFrequency::Hz44100;
        let input = varied(fs, 0.6);
        let lens: Vec<usize> = [-6.0, 0.0, 6.0]
            .map(|offset| {
                let config = StreamConfig {
                    rate: variable(fs, [32, 320], offset),
                    sampling_frequency: fs,
                    mode: Mode::SingleChannel,
                    crc_protected: false,
                    flags: Flags::default(),
                };
                encode(config, &input).len()
            })
            .to_vec();
        assert!(lens.windows(2).all(|w| w[0] > w[1]), "{lens:?}");
    }

    /// A mean bitrate between two of the table switches between them, and
    /// the stream is as long as its schedule makes it.
    #[test]
    fn a_mean_bitrate_switches_between_its_neighbours() {
        for (fs, kbps, neighbours) in [
            (SamplingFrequency::Hz44100, 150, [128, 160]),
            (SamplingFrequency::Hz22050, 100, [96, 112]),
        ] {
            let mean = rate::MeanRate::new(fs.version(), kbps * 1000).unwrap();
            let config = StreamConfig {
                rate: StreamRate::Mean(mean),
                sampling_frequency: fs,
                mode: MS,
                crc_protected: false,
                flags: Flags::default(),
            };
            let frames = check::frames(&encode(config, &varied(fs, 0.5)));
            let mut schedule = MeanRateSchedule::new(mean, fs);
            for (k, frame) in frames.iter().enumerate() {
                let (bitrate, padding) = schedule.next_frame();
                assert_eq!(
                    (frame.bitrate, frame.padding),
                    (bitrate, padding),
                    "frame {k}"
                );
            }
            let used: std::collections::BTreeSet<u32> = (frames.iter())
                .map(|f| f.bitrate.bits_per_second() / 1000)
                .collect();
            assert_eq!(used.into_iter().collect::<Vec<_>>(), neighbours, "{fs:?}");
        }
    }

    /// At a variable bitrate too, the stream does not depend on how the
    /// input is divided between calls.
    #[test]
    fn variable_streaming_does_not_depend_on_how_the_input_is_divided() {
        let fs = SamplingFrequency::Hz44100;
        let config = StreamConfig {
            rate: variable(fs, [32, 320], 0.0),
            sampling_frequency: fs,
            mode: MS,
            crc_protected: false,
            flags: Flags::default(),
        };
        let [left, right] = varied(fs, 0.5);
        let whole = encode_pcm(&config, &[&left, &right]);
        for chunk in [1000, Version::Mpeg1.samples_per_frame() + 1] {
            let mut encoder = StreamEncoder::new(config);
            let mut out = Vec::new();
            for (left, right) in left.chunks(chunk).zip(right.chunks(chunk)) {
                encoder.encode(&[left, right], &mut out);
            }
            encoder.finish(&mut out);
            assert!(out == whole, "chunks of {chunk}");
        }
    }
}
