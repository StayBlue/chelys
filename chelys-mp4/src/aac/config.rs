// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The configuration of AAC-LC packets that any encoder made: their
//! `AudioSpecificConfig()` (ISO/IEC 14496-3, clause 1.6.2.1), which the
//! track's `ES_Descriptor` carries as it is given, and of which the track
//! needs the sample rate, the channels, the frame's length and the
//! decoder's input buffer.

use std::fmt;

use super::Stream;
use crate::bitrate::{max_bitrate, window_frames};
use crate::descriptor::MAX_SPECIFIC_INFO;
use crate::movie::AudioTrack;

/// The audio object type of AAC LC (ISO/IEC 14496-3, Table 1.1).
const AAC_LC: u32 = 2;

/// The sampling frequencies of `samplingFrequencyIndex` 0 to 12 (ISO/IEC
/// 14496-3, Table 1.22); 13 and 14 are reserved, and 15 is the escape to an
/// explicit frequency.
const SAMPLING_FREQUENCIES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];

/// The bits of the decoder's input buffer per channel of the NCC, the
/// number of considered channels: "6144 bits per SCE or independently
/// switched CCE, plus 12288 bits per CPE" (ISO/IEC 13818-7, clause
/// 8.2.3.1).
const BUFFER_BITS_PER_CHANNEL: u32 = 6144;

/// What an `'mp4a'` track needs of a stream of AAC-LC packets that any
/// encoder made, which a [`Writer`](crate::Writer) or a
/// [`FragmentedWriter`](crate::FragmentedWriter) takes: its
/// `AudioSpecificConfig()`, and the samples that a decoder's output starts
/// with before the input, the encoder's delay, if known.
///
/// Each packet is a `raw_data_block()`, an access unit (ISO/IEC 14496-3,
/// clause 1.6.2.2.2.1), without an ADTS header, and decodes to one frame,
/// 1024 samples, or 960 where the configuration says so. Each decodes
/// correctly only after the one before it: the track has a pre-roll of one
/// packet, as every AAC track chelys writes.
///
/// ```
/// use chelys_mp4::{Writer, aac};
///
/// // AAC LC at 44.1 kHz in stereo, of an encoder's delay of 1024 samples,
/// // and its packets, raw_data_block()s without ADTS headers, as a
/// // demuxer gives them.
/// let config = aac::Config::from_audio_specific_config(&[0x12, 0x10], Some(1024))?;
/// let mut writer = Writer::new(config);
/// for packet in [[0x21, 0x10, 0x04, 0x60, 0x8c, 0x1c]; 3] {
///     writer.push(&packet)?;
/// }
/// // The input was 2000 samples long: the edit list leaves out the delay
/// // and the padding after the input.
/// let mut mp4 = Vec::new();
/// writer.finish(Some(2000), &mut mp4)?;
/// assert_eq!(&mp4[4..12], b"ftypmp42");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    stream: Stream,
    delay: Option<u32>,
}

/// Why an `AudioSpecificConfig()` was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConfigError {
    /// It is cut short, or too long, or breaks its syntax: a reserved
    /// `samplingFrequencyIndex`, a sampling frequency of 0, or an
    /// `extensionFlag` of 1, which "Shall be '0'" for AAC LC (ISO/IEC
    /// 14496-3, clause 4.5.1.1).
    Syntax,
    /// An audio object type other than AAC LC, 2: among others, HE-AAC's
    /// SBR (5) and PS (29), signalled hierarchically.
    ObjectType(u32),
    /// An extension tool that the configuration signals as present after
    /// AAC LC's own, which an AAC-LC decoder would not decode: SBR or PS,
    /// signalled in a backward compatible way, or MPEG Surround or SAOC.
    Extension,
    /// A reserved `channelConfiguration`: 8 to 10, or 15.
    ChannelConfiguration(u8),
    /// A `program_config_element()` of no channels.
    NoChannels,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax => write!(f, "not a valid AudioSpecificConfig()"),
            Self::ObjectType(aot) => {
                write!(f, "audio object type {aot}, where AAC LC is 2")
            }
            Self::Extension => write!(
                f,
                "an AudioSpecificConfig() that signals an extension tool, such as SBR"
            ),
            Self::ChannelConfiguration(value) => {
                write!(f, "channelConfiguration {value}, which is reserved")
            }
            Self::NoChannels => write!(f, "a program_config_element() of no channels"),
        }
    }
}

impl std::error::Error for ConfigError {}

impl Config {
    /// The configuration of the stream whose `AudioSpecificConfig()` is
    /// `audio_specific_config`, of AAC LC, whose decoder's output starts
    /// with `delay` samples, at the sample rate, that precede the input,
    /// where the delay is known.
    ///
    /// With a delay, the track's edit list leaves it out, and the length
    /// given to the writer is that of the input, so that the edit leaves
    /// out the padding after it too. A delay of 0 is a delay: the file then
    /// has an edit list, which can leave out the padding. Without a delay,
    /// `None`, the track has no edit list and presents every sample the
    /// packets decode to, the delay and the padding included, and a length
    /// given to the writer is to be every one of them.
    ///
    /// The delay is the encoder's to give, as the packets do not, and it
    /// differs from encoder to encoder; where a demuxer reports 0 for a
    /// delay it does not know, that is `None`. A delay that is not the
    /// stream's shifts the presentation, and may cut the input's start or
    /// end, or present samples of the delay or the padding.
    ///
    /// The configuration is kept as it is given, and read as far as the
    /// track needs and to make sure it is AAC LC alone. Its length, which
    /// the file conveys, is its end: an extension after AAC LC's own
    /// configuration is read, where present, to refuse a tool signalled
    /// there.
    ///
    /// Spec gap: SBR may also be signalled implicitly, by its data in the
    /// packets alone (ISO/IEC 14496-3, clause 1.6.5), which a configuration
    /// cannot show: such a stream is taken as the AAC LC that it is to an
    /// AAC-LC decoder, at the sample rate of its configuration.
    ///
    /// # Errors
    ///
    /// [`ConfigError`] if it is not a valid configuration of AAC LC alone,
    /// or is longer than an `ES_Descriptor` can hold, 256 MiB.
    pub fn from_audio_specific_config(
        audio_specific_config: &[u8],
        delay: Option<u32>,
    ) -> Result<Self, ConfigError> {
        if audio_specific_config.len() > MAX_SPECIFIC_INFO {
            return Err(ConfigError::Syntax);
        }
        let parsed = parse(audio_specific_config)?;
        Ok(Self {
            stream: Stream {
                sample_rate: parsed.sample_rate,
                channels: parsed.channels,
                frame_samples: parsed.frame_samples,
                buffer_size: parsed.ncc * BUFFER_BITS_PER_CHANNEL / 8,
                audio_specific_config: audio_specific_config.to_vec(),
            },
            delay,
        })
    }

    /// The samples that a decoder's output starts with before the input,
    /// which the edit list leaves out, if known.
    pub fn delay(&self) -> Option<u32> {
        self.delay
    }

    /// The sampling frequency in Hz, the track's timescale.
    pub fn sample_rate(&self) -> u32 {
        self.stream.sample_rate
    }

    /// The number of channels, the low frequency enhancement channels
    /// among them.
    pub fn channels(&self) -> u8 {
        u8::try_from(self.stream.channels).expect("at most 93 channels")
    }

    /// The samples of each channel that a packet decodes to: 1024, or 960.
    pub fn frame_samples(&self) -> u32 {
        self.stream.frame_samples
    }

    /// The track of the stream, of packets of `sizes` where they are known,
    /// as in a file written whole, and otherwise, as in a fragmented file's
    /// init segment, before any.
    ///
    /// Spec reading: `avgBitrate` "shall be set to zero" "For streams with
    /// variable bitrate" (ISO/IEC 14496-1, clause 6.2.6.6.2). Whether a
    /// stream of another encoder is at a constant rate cannot be told from
    /// its packets, whose sizes vary about the mean through the bit
    /// reservoir (ISO/IEC 13818-7, clause 8.2.3.2) as at a variable rate:
    /// it is 0.
    ///
    /// Spec reading: `maxBitrate` is measured on the packets where they are
    /// known (see `bitrate::max_bitrate`). Before them, it is a bound, as
    /// "not known" has no value: no frame is longer than the decoder's
    /// input buffer (ISO/IEC 13818-7, clause 8.2.3.1), and a window of a
    /// second holds at most `w` frames, as `bitrate::max_bitrate` counts
    /// them, `w` buffers, as chelys's encoder bounds a stream at a variable
    /// rate.
    pub(crate) fn track(&self, sizes: Option<&[u32]>) -> AudioTrack {
        let Stream {
            sample_rate,
            frame_samples,
            buffer_size,
            ..
        } = self.stream;
        let max = match sizes {
            Some(sizes) => max_bitrate(sizes, frame_samples, sample_rate),
            None => window_frames(frame_samples, sample_rate).saturating_mul(8 * buffer_size),
        };
        self.stream.track(max, 0, self.delay)
    }
}

/// What the track needs of an `AudioSpecificConfig()`.
#[derive(Debug, PartialEq, Eq)]
struct Parsed {
    sample_rate: u32,
    channels: u16,
    frame_samples: u32,
    /// The number of considered channels (ISO/IEC 13818-7, clause 3.58).
    ncc: u32,
}

/// Reads an `AudioSpecificConfig()` of AAC LC (ISO/IEC 14496-3, Table
/// 1.19), with its `GASpecificConfig()` (Table 4.1).
fn parse(bytes: &[u8]) -> Result<Parsed, ConfigError> {
    let mut r = BitReader::new(bytes);
    let aot = audio_object_type(&mut r)?;
    let index = r.read(4)?;
    let sample_rate = match index {
        0xf => r.read(24)?,
        0xd | 0xe => return Err(ConfigError::Syntax),
        index => SAMPLING_FREQUENCIES[index as usize],
    };
    // Spec reading: a frequency of 0, which the standard does not forbid,
    // is no rate a track's timescale can be.
    if sample_rate == 0 {
        return Err(ConfigError::Syntax);
    }
    let channel_configuration = r.read(4)? as u8;
    // Of the types whose configuration is GASpecificConfig(), AAC LC alone:
    // 5 and 29, SBR and PS signalled hierarchically, are HE-AAC.
    if aot != AAC_LC {
        return Err(ConfigError::ObjectType(aot));
    }
    // GASpecificConfig()
    let frame_samples = if r.read(1)? == 1 { 960 } else { 1024 }; // frameLengthFlag
    if r.read(1)? == 1 {
        // dependsOnCoreCoder
        r.read(14)?; // coreCoderDelay
    }
    // extensionFlag: "Shall be '0' for audio object types 1, 2, 3, 4, 6,
    // 7" (clause 4.5.1.1).
    if r.read(1)? == 1 {
        return Err(ConfigError::Syntax);
    }
    let (channels, ncc) = match channel_configuration {
        0 => program_config_element(&mut r)?,
        configuration => channels_of(configuration)?,
    };
    // AAC LC has no layerNr, of types 6 and 20, and no epConfig, of the
    // error resilient types.
    extensions(&mut r)?;
    Ok(Parsed {
        sample_rate,
        channels,
        frame_samples,
        ncc,
    })
}

/// `GetAudioObjectType()` (ISO/IEC 14496-3, Table 1.20).
fn audio_object_type(r: &mut BitReader) -> Result<u32, ConfigError> {
    let aot = r.read(5)?;
    Ok(if aot == 31 { 32 + r.read(6)? } else { aot })
}

/// The channels of a `channelConfiguration` other than 0, the low
/// frequency enhancement channels among them, and its NCC, the channels of
/// its SCEs and CPEs, from the elements of ISO/IEC 14496-3, Table 1.23.
///
/// Spec reading: 11 to 14, which ISO/IEC 14496-3 defines and ISO/IEC
/// 13818-7 does not, are taken as configurations of AAC LC, as they are
/// what MPEG-4 adds to the channel configurations, of any audio object
/// type whose configuration has one.
fn channels_of(configuration: u8) -> Result<(u16, u32), ConfigError> {
    Ok(match configuration {
        1 => (1, 1),
        2 => (2, 2),
        3 => (3, 3),
        4 => (4, 4),
        5 => (5, 5),
        // 5.1: an SCE, two CPEs and an LFE.
        6 => (6, 5),
        // 7.1 front: an SCE, three CPEs and an LFE.
        7 => (8, 7),
        // 6.1: an SCE, two CPEs, an SCE and an LFE.
        11 => (7, 6),
        // 7.1 back and 7.1 top: an SCE, three CPEs and an LFE.
        12 | 14 => (8, 7),
        // 22.2: six SCEs, eight CPEs and two LFEs.
        13 => (24, 22),
        reserved => return Err(ConfigError::ChannelConfiguration(reserved)),
    })
}

/// Reads a `program_config_element()` (ISO/IEC 14496-3, Table 4.2), and
/// gives its channels, the low frequency enhancement channels among them,
/// and its NCC: once each SCE and independently switched CCE, twice each
/// CPE (ISO/IEC 13818-7, clause 3.58). A CCE's channel is coupled into the
/// others, and is not output.
fn program_config_element(r: &mut BitReader) -> Result<(u16, u32), ConfigError> {
    r.read(4)?; // element_instance_tag
    r.read(2)?; // object_type
    r.read(4)?; // sampling_frequency_index
    let front = r.read(4)?; // num_front_channel_elements
    let side = r.read(4)?; // num_side_channel_elements
    let back = r.read(4)?; // num_back_channel_elements
    let lfe = r.read(2)?; // num_lfe_channel_elements
    let assoc_data = r.read(3)?; // num_assoc_data_elements
    let valid_cc = r.read(4)?; // num_valid_cc_elements
    if r.read(1)? == 1 {
        // mono_mixdown_present
        r.read(4)?; // mono_mixdown_element_number
    }
    if r.read(1)? == 1 {
        // stereo_mixdown_present
        r.read(4)?; // stereo_mixdown_element_number
    }
    if r.read(1)? == 1 {
        // matrix_mixdown_idx_present
        r.read(2)?; // matrix_mixdown_idx
        r.read(1)?; // pseudo_surround_enable
    }
    let mut channels = 0;
    for _ in 0..front + side + back {
        // {front,side,back}_element_is_cpe, then _tag_select
        channels += 1 + r.read(1)?;
        r.read(4)?;
    }
    for _ in 0..lfe + assoc_data {
        r.read(4)?; // lfe_element_tag_select, assoc_data_element_tag_select
    }
    let mut independent_cces = 0;
    for _ in 0..valid_cc {
        independent_cces += r.read(1)?; // cc_element_is_ind_sw
        r.read(4)?; // valid_cc_element_tag_select
    }
    // byte_alignment(), "relative to the start of the
    // AudioSpecificConfig()" (Table 4.2, Note 1).
    r.align();
    let comment_field_bytes = r.read(8)?;
    for _ in 0..comment_field_bytes {
        r.read(8)?; // comment_field_data
    }
    let channels = channels + lfe;
    if channels == 0 {
        return Err(ConfigError::NoChannels);
    }
    let ncc = channels - lfe + independent_cces;
    Ok((channels as u16, ncc))
}

/// Reads the extensions that may follow AAC LC's configuration in an
/// `AudioSpecificConfig()` (ISO/IEC 14496-3, Table 1.19), each after its
/// `extensionIdentifier`, and refuses any that signals a tool present.
///
/// The syntax reads an identifier only where at least 11 bits are left,
/// `bits_to_decode()`, the bits of the configuration not yet read, whose
/// length the file conveys; bits that follow without an identifier it
/// knows are not read, as by the syntax.
fn extensions(r: &mut BitReader) -> Result<(), ConfigError> {
    let mut identifier = None;
    if r.left() >= 11 {
        identifier = Some(r.read(11)?);
    }
    if identifier == Some(0x2b7) {
        identifier = None;
        // extensionAudioObjectType is 0 here, as audioObjectType was
        // neither 5 nor 29.
        if r.left() >= 5 {
            match audio_object_type(r)? {
                5 => {
                    // sbrPresentFlag: of 1, the stream is HE-AAC.
                    if r.read(1)? == 1 {
                        return Err(ConfigError::Extension);
                    }
                    if r.left() >= 12 {
                        identifier = Some(r.read(11)?);
                        if identifier == Some(0x548) {
                            identifier = None;
                            if r.read(1)? == 1 {
                                // psPresentFlag
                                return Err(ConfigError::Extension);
                            }
                        }
                    }
                }
                // Spec reading: the syntax reads on, an sbrPresentFlag and
                // an extensionChannelConfiguration, whatever the flag; but
                // the extension is ER BSAC's, of another object type, and
                // is refused without them.
                22 => return Err(ConfigError::Extension),
                _ => {}
            }
        }
    }
    // MPEG Surround (0x76a), SAOC (0x7cb), LD MPEG Surround (0x7cc) and
    // SAOC-DE (0x7cd), each signalled by a flag of 1 after its identifier.
    for sync in [0x76a, 0x7cb, 0x7cc, 0x7cd] {
        if identifier.is_none() && r.left() >= 11 {
            identifier = Some(r.read(11)?);
        }
        if identifier == Some(sync) {
            identifier = None;
            if r.left() >= 1 && r.read(1)? == 1 {
                return Err(ConfigError::Extension);
            }
        }
    }
    Ok(())
}

/// Reads bits most significant first, as the syntax's fields are.
struct BitReader<'a> {
    bytes: &'a [u8],
    /// The bits read.
    at: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    /// The bits not yet read.
    fn left(&self) -> usize {
        8 * self.bytes.len() - self.at
    }

    /// Reads the next `bits`, at most 24.
    fn read(&mut self, bits: usize) -> Result<u32, ConfigError> {
        debug_assert!(bits <= 24);
        if bits > self.left() {
            return Err(ConfigError::Syntax);
        }
        let mut value = 0;
        for _ in 0..bits {
            let bit = (self.bytes[self.at / 8] >> (7 - self.at % 8)) & 1;
            value = (value << 1) | u32::from(bit);
            self.at += 1;
        }
        Ok(value)
    }

    /// Skips to the next byte's start, if not at one.
    fn align(&mut self) {
        self.at = self.at.next_multiple_of(8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes fields of the syntax, most significant bit first.
    #[derive(Default)]
    struct Bits {
        bits: Vec<bool>,
    }

    impl Bits {
        fn put(mut self, value: u32, bits: usize) -> Self {
            self.bits
                .extend((0..bits).rev().map(|i| (value >> i) & 1 == 1));
            self
        }

        fn align(mut self) -> Self {
            while self.bits.len() % 8 != 0 {
                self.bits.push(false);
            }
            self
        }

        fn bytes(self) -> Vec<u8> {
            let this = self.align();
            this.bits
                .chunks(8)
                .map(|byte| byte.iter().fold(0, |b, &bit| (b << 1) | u8::from(bit)))
                .collect()
        }
    }

    /// The `AudioSpecificConfig()` of AAC LC of `index` and `configuration`,
    /// up to its `GASpecificConfig()`'s `extensionFlag`.
    fn lc(index: u32, configuration: u32) -> Bits {
        Bits::default()
            .put(2, 5)
            .put(index, 4)
            .put(configuration, 4)
            .put(0, 3)
    }

    #[test]
    fn reads_the_rate_channels_and_buffer() {
        // chelys's own: 44.1 kHz stereo, 0x12 0x10.
        let config = Config::from_audio_specific_config(&[0x12, 0x10], None).unwrap();
        assert_eq!(config.sample_rate(), 44_100);
        assert_eq!(config.channels(), 2);
        assert_eq!(config.frame_samples(), 1024);
        assert_eq!(config.stream.buffer_size, 2 * 768);
        for (configuration, channels, ncc) in [
            (1, 1, 1),
            (6, 6, 5),
            (7, 8, 7),
            (11, 7, 6),
            (12, 8, 7),
            (13, 24, 22),
            (14, 8, 7),
        ] {
            let asc = lc(3, configuration).bytes();
            let config = Config::from_audio_specific_config(&asc, None).unwrap();
            assert_eq!(config.sample_rate(), 48_000);
            assert_eq!(config.channels(), channels, "{configuration}");
            assert_eq!(config.stream.buffer_size, ncc * 768, "{configuration}");
        }
        // An explicit frequency, and frames of 960 after a core coder's
        // delay.
        let asc = Bits::default()
            .put(2, 5)
            .put(0xf, 4)
            .put(37_800, 24)
            .put(1, 4)
            .put(1, 1)
            .put(1, 1)
            .put(123, 14)
            .put(0, 1)
            .bytes();
        let config = Config::from_audio_specific_config(&asc, None).unwrap();
        assert_eq!(config.sample_rate(), 37_800);
        assert_eq!(config.frame_samples(), 960);
        // An escaped audio object type is read whole: 32 + 6 bits.
        let asc = Bits::default()
            .put(31, 5)
            .put(0, 6)
            .put(3, 4)
            .put(2, 4)
            .bytes();
        assert_eq!(
            Config::from_audio_specific_config(&asc, None),
            Err(ConfigError::ObjectType(32))
        );
    }

    #[test]
    fn reads_a_program_config_element() {
        // Front: an SCE and a CPE; side: a CPE; back: none; an LFE, a DSE,
        // two CCEs, one independently switched; a matrix mixdown; and a
        // comment of two bytes.
        let pce = |comment: &[u8]| {
            let mut bits = lc(4, 0)
                .put(0, 4) // element_instance_tag
                .put(1, 2) // object_type
                .put(4, 4) // sampling_frequency_index
                .put(2, 4)
                .put(1, 4)
                .put(0, 4)
                .put(1, 2)
                .put(1, 3)
                .put(2, 4)
                .put(0, 1) // mono_mixdown_present
                .put(0, 1) // stereo_mixdown_present
                .put(1, 1) // matrix_mixdown_idx_present
                .put(0b101, 3)
                .put(0, 5) // the front SCE
                .put(0b1_0001, 5) // the front CPE
                .put(0b1_0010, 5) // the side CPE
                .put(0, 4) // the LFE
                .put(0, 4) // the DSE
                .put(0b1_0000, 5) // the independently switched CCE
                .put(0b0_0001, 5)
                .align()
                .put(comment.len() as u32, 8);
            for &byte in comment {
                bits = bits.put(u32::from(byte), 8);
            }
            bits
        };
        let config = Config::from_audio_specific_config(&pce(b"hi").bytes(), None).unwrap();
        assert_eq!(config.sample_rate(), 44_100);
        assert_eq!(config.channels(), 6);
        // Five channels of SCEs and CPEs, and the CCE.
        assert_eq!(config.stream.buffer_size, 6 * 768);
        // Cut within the comment.
        let mut cut = pce(b"hi").bytes();
        cut.pop();
        assert_eq!(
            Config::from_audio_specific_config(&cut, None),
            Err(ConfigError::Syntax)
        );
        // No channels.
        // The tag, object type and index, the counts of elements, all 0,
        // and no mixdowns.
        let empty = lc(4, 0)
            .put(0, 4 + 2 + 4)
            .put(0, 4 + 4 + 4 + 2 + 3 + 4)
            .put(0, 3);
        let empty = empty.align().put(0, 8);
        assert_eq!(
            Config::from_audio_specific_config(&empty.bytes(), None),
            Err(ConfigError::NoChannels)
        );
    }

    #[test]
    fn refuses_what_is_not_aac_lc_alone() {
        let refused =
            |bits: Bits| Config::from_audio_specific_config(&bits.bytes(), None).unwrap_err();
        // HE-AAC, signalled hierarchically: SBR (5) then AAC LC.
        let he = Bits::default()
            .put(5, 5)
            .put(6, 4)
            .put(2, 4)
            .put(3, 4)
            .put(2, 5);
        assert_eq!(refused(he), ConfigError::ObjectType(5));
        // Backward compatible SBR: 0x2b7, type 5 and sbrPresentFlag 1.
        let sbr = lc(6, 2).put(0x2b7, 11).put(5, 5).put(1, 1).put(3, 4);
        assert_eq!(refused(sbr), ConfigError::Extension);
        // And PS after SBR signalled absent.
        let ps = lc(3, 1)
            .put(0x2b7, 11)
            .put(5, 5)
            .put(0, 1)
            .put(0x548, 11)
            .put(1, 1);
        assert_eq!(refused(ps), ConfigError::Extension);
        // MPEG Surround.
        let mps = lc(3, 2).put(0x76a, 11).put(1, 1).put(0, 8);
        assert_eq!(refused(mps), ConfigError::Extension);
        // A reserved index, channel configuration, or a frequency of 0.
        assert_eq!(refused(lc(0xd, 2)), ConfigError::Syntax);
        assert_eq!(refused(lc(3, 8)), ConfigError::ChannelConfiguration(8));
        assert_eq!(refused(lc(3, 15)), ConfigError::ChannelConfiguration(15));
        let zero = Bits::default()
            .put(2, 5)
            .put(0xf, 4)
            .put(0, 24)
            .put(2, 4)
            .put(0, 3);
        assert_eq!(refused(zero), ConfigError::Syntax);
        // An extensionFlag of 1.
        let extension_flag = Bits::default().put(2, 5).put(3, 4).put(2, 4).put(0b001, 3);
        assert_eq!(refused(extension_flag), ConfigError::Syntax);
        // Cut short.
        assert_eq!(
            Config::from_audio_specific_config(&[0x12], None),
            Err(ConfigError::Syntax)
        );
        assert_eq!(
            Config::from_audio_specific_config(&[], None),
            Err(ConfigError::Syntax)
        );
    }

    #[test]
    fn takes_extensions_that_signal_no_tool() {
        // SBR signalled absent, then MPEG Surround absent.
        let asc = lc(4, 2)
            .put(0x2b7, 11)
            .put(5, 5)
            .put(0, 1)
            .put(0x76a, 11)
            .put(0, 1)
            .bytes();
        assert!(Config::from_audio_specific_config(&asc, None).is_ok());
        // Another object type's extension, which carries no flag.
        let asc = lc(4, 2).put(0x2b7, 11).put(3, 5).bytes();
        assert!(Config::from_audio_specific_config(&asc, None).is_ok());
        // Bits after the configuration that are no identifier.
        let asc = lc(4, 2).put(0x123, 11).put(0x7f, 7).bytes();
        assert!(Config::from_audio_specific_config(&asc, None).is_ok());
    }
}
