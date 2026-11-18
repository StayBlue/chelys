// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The 32-bit frame header (clauses 2.4.1.3 and 2.4.2.3, and clause 2.4.2.3
//! of ISO/IEC 13818-3 for the lower sampling frequencies, and US patent
//! 6,185,539 B1 for MPEG-2.5) and the frame length it implies (clause
//! 2.4.3.1).

use crate::bitstream::BitWriter;
use crate::granule::GRANULE_LINES;
use crate::side_info::SideInfo;

const SYNCWORD: u32 = 0xFFF;
/// The sync word that marks MPEG-2.5, "eleven ones and a zero" (US
/// 6,185,539 B1, column 3).
const MPEG25_SYNCWORD: u32 = 0xFFE;
const LAYER_III: u32 = 0b01;

/// The `ID` bit: the standard that codes the stream, and with it the
/// sampling frequencies, bitrates and frame layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Version {
    /// ISO/IEC 11172-3, `ID` 1.
    Mpeg1,
    /// The extension of ISO/IEC 13818-3 to lower sampling frequencies, `ID`
    /// 0 (clause 2.4 of ISO/IEC 13818-3), and MPEG-2.5, which codes its
    /// stream in the same way (see [`SamplingFrequency::mpeg25`]).
    Mpeg2,
}

impl Version {
    fn id(self) -> u32 {
        match self {
            Self::Mpeg1 => 1,
            Self::Mpeg2 => 0,
        }
    }

    /// The granules of a frame: two, or one at the lower sampling
    /// frequencies (clause 2.4.3.2 of ISO/IEC 13818-3).
    pub fn granules(self) -> usize {
        match self {
            Self::Mpeg1 => 2,
            Self::Mpeg2 => 1,
        }
    }

    /// The samples of each channel that a frame codes.
    pub fn samples_per_frame(self) -> usize {
        self.granules() * GRANULE_LINES
    }

    /// The width of `main_data_begin`: 9 bits, or 8 at the lower sampling
    /// frequencies (clause 2.4.1.7 of ISO/IEC 13818-3).
    pub fn main_data_begin_bits(self) -> u32 {
        match self {
            Self::Mpeg1 => 9,
            Self::Mpeg2 => 8,
        }
    }

    /// The Layer III bitrates of the table in kbit/s, lowest first, without
    /// free format.
    pub fn table_kbps(self) -> &'static [u32] {
        &self.kbps()[1..]
    }

    /// The Layer III bitrates in kbit/s, indexed by `bitrate_index`; index 0
    /// is free format (clause 2.4.2.3, and clause 2.4.2.3 of ISO/IEC
    /// 13818-3).
    fn kbps(self) -> &'static [u32; 15] {
        match self {
            Self::Mpeg1 => &[
                0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
            ],
            Self::Mpeg2 => &[0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
        }
    }
}

/// A Layer III bitrate: one of the `bitrate_index` table of a [`Version`],
/// or a free format bitrate, sent as `bitrate_index` 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bitrate {
    version: Version,
    /// The `bitrate_index`, 0 in free format.
    index: u8,
    bits_per_second: u32,
}

impl Bitrate {
    pub fn from_kbps(version: Version, kbps: u32) -> Option<Self> {
        let index = version.kbps()[1..].iter().position(|&k| k == kbps)? + 1;
        Some(Self::from_index(version, index as u8))
    }

    fn from_index(version: Version, index: u8) -> Self {
        Self {
            version,
            index,
            bits_per_second: version.kbps()[usize::from(index)] * 1000,
        }
    }

    pub fn all(version: Version) -> impl Iterator<Item = Self> {
        (1..version.kbps().len() as u8).map(move |index| Self::from_index(version, index))
    }

    /// A free format bitrate (clause 2.1.67, and clause 2.1.118 of ISO/IEC
    /// 13818-3): "any bitrate other than the defined bitrates that is less
    /// than the maximum valid bitrate", here in whole bits per second, and
    /// none where its frames are too short for the header, the `crc_check`
    /// word and the side information of two channels at the version's
    /// highest sampling frequency, whose frames are the shortest. A frame
    /// at a free format bitrate holds `N` or `N + 1` slots, as at one of
    /// the table (clause 2.4.2.3).
    ///
    /// Spec reading: clause 2.4.2.3 adds that a decoder "is also not
    /// required to support bitrates higher than [...] 320 kbits/s" in free
    /// format, or
    /// 160 kbit/s at the lower sampling frequencies (clause 2.4.2.3 of
    /// ISO/IEC 13818-3), as though there were such bitrates; the definition
    /// excludes them, and is followed.
    ///
    /// Spec gap: neither standard bounds a free format bitrate from below.
    /// The bound here keeps every frame within the syntax of clause
    /// 2.4.1.2 in any mode.
    ///
    /// Spec gap: US 6,185,539 B1 does not mention free format. MPEG-2.5,
    /// which extends ISO/IEC 13818-3, has it as that standard does, with
    /// the same bounds. Settled by decoding with mpg123 and LAME (black
    /// box): both decode every frame of it at 8, 11.025 and 12 kHz; the
    /// conformance test is `mpg123_and_lame_decode_free_format`. ffmpeg
    /// decodes no free format.
    pub fn free(version: Version, bits_per_second: u32) -> Option<Self> {
        let table = version.kbps();
        let highest = table[table.len() - 1] * 1000;
        let defined = table[1..]
            .iter()
            .any(|&kbps| kbps * 1000 == bits_per_second);
        let fullest = Header::LEN_BYTES + 2 + SideInfo::len_bytes(version, 2);
        let shortest = SamplingFrequency::ALL
            .into_iter()
            .filter(|fs| fs.version() == version)
            .map(|fs| slots_per_frame(version, bits_per_second, fs).0)
            .min()
            .expect("sampling frequencies of every version");
        (!defined && bits_per_second < highest && shortest >= fullest).then_some(Self {
            version,
            index: 0,
            bits_per_second,
        })
    }

    /// Whether this is a free format bitrate.
    pub fn is_free(self) -> bool {
        self.index == 0
    }

    pub fn version(self) -> Version {
        self.version
    }

    pub fn bits_per_second(self) -> u32 {
        self.bits_per_second
    }

    /// The length in bytes of a frame at this bitrate, header included, with
    /// or without `padding`.
    ///
    /// # Panics
    ///
    /// If the bitrate is not one of those of the sampling frequency's
    /// [`Version`].
    pub fn frame_len(self, sampling_frequency: SamplingFrequency, padding: bool) -> usize {
        assert_eq!(
            self.version,
            sampling_frequency.version(),
            "{self:?} is not a bitrate of {sampling_frequency:?}"
        );
        let (slots, _) = slots_per_frame(self.version, self.bits_per_second, sampling_frequency);
        slots + usize::from(padding)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingFrequency {
    Hz44100,
    Hz48000,
    Hz32000,
    Hz22050,
    Hz24000,
    Hz16000,
    Hz11025,
    Hz12000,
    Hz8000,
}

impl SamplingFrequency {
    /// The sampling frequencies of ISO/IEC 11172-3 and 13818-3.
    #[cfg(any(test, feature = "unstable"))]
    pub const STANDARD: [Self; 6] = [
        Self::Hz44100,
        Self::Hz48000,
        Self::Hz32000,
        Self::Hz22050,
        Self::Hz24000,
        Self::Hz16000,
    ];

    /// The sampling frequencies of MPEG-2.5.
    pub const MPEG25: [Self; 3] = [Self::Hz11025, Self::Hz12000, Self::Hz8000];

    pub const ALL: [Self; 9] = [
        Self::Hz44100,
        Self::Hz48000,
        Self::Hz32000,
        Self::Hz22050,
        Self::Hz24000,
        Self::Hz16000,
        Self::Hz11025,
        Self::Hz12000,
        Self::Hz8000,
    ];

    pub fn hz(self) -> u32 {
        match self {
            Self::Hz44100 => 44_100,
            Self::Hz48000 => 48_000,
            Self::Hz32000 => 32_000,
            Self::Hz22050 => 22_050,
            Self::Hz24000 => 24_000,
            Self::Hz16000 => 16_000,
            Self::Hz11025 => 11_025,
            Self::Hz12000 => 12_000,
            Self::Hz8000 => 8_000,
        }
    }

    /// Spec reading: US 6,185,539 B1 changes the sync word of MPEG-2.5 and
    /// says that "in all other respects the structure of the bit stream is
    /// unchanged with respect to the Standard ISO/IEC 13818-3" (column 3).
    /// MPEG-2.5 is therefore coded as ISO/IEC 13818-3 codes its sampling
    /// frequencies: `ID` 0, one granule a frame, and its bitrates and side
    /// information.
    pub fn version(self) -> Version {
        match self {
            Self::Hz44100 | Self::Hz48000 | Self::Hz32000 => Version::Mpeg1,
            Self::Hz22050 | Self::Hz24000 | Self::Hz16000 => Version::Mpeg2,
            Self::Hz11025 | Self::Hz12000 | Self::Hz8000 => Version::Mpeg2,
        }
    }

    /// Whether this is a sampling frequency of MPEG-2.5, which US patent
    /// 6,185,539 B1 adds to those of ISO/IEC 13818-3, and whose frames begin
    /// with another sync word.
    pub fn mpeg25(self) -> bool {
        Self::MPEG25.contains(&self)
    }

    /// The `sampling_frequency` field, whose meaning depends on `ID` and,
    /// for MPEG-2.5, on the sync word.
    ///
    /// Spec gap: US 6,185,539 B1 does not give the codes of MPEG-2.5. It
    /// samples at "preferably half the sampling rate" of ISO/IEC 13818-3,
    /// and compares its table of scalefactor bands for 8 kHz with the one
    /// for 16 kHz (column 4), so each frequency takes the code of the one of
    /// ISO/IEC 13818-3 that is twice as high.
    ///
    /// Settled by decoding with ffmpeg, mpg123 and LAME (black box): all
    /// three read these codes, after the sync word "FFE" and with `ID` 0, as
    /// 11.025, 12 and 8 kHz. The conformance test is
    /// `decoders_read_the_mpeg25_sampling_frequencies`.
    fn code(self) -> u32 {
        match self {
            Self::Hz44100 | Self::Hz22050 | Self::Hz11025 => 0b00,
            Self::Hz48000 | Self::Hz24000 | Self::Hz12000 => 0b01,
            Self::Hz32000 | Self::Hz16000 | Self::Hz8000 => 0b10,
        }
    }

    fn syncword(self) -> u32 {
        if self.mpeg25() {
            MPEG25_SYNCWORD
        } else {
            SYNCWORD
        }
    }
}

/// The `mode` field, with the Layer III meaning of `mode_extension` folded
/// into the joint stereo variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Stereo,
    JointStereo {
        intensity_stereo: bool,
        ms_stereo: bool,
    },
    /// Two channels with independent programmes, such as two languages,
    /// coded as in [`Stereo`](Self::Stereo) (clause 2.1.52).
    DualChannel,
    SingleChannel,
}

impl Mode {
    pub fn channels(self) -> usize {
        match self {
            Self::SingleChannel => 1,
            _ => 2,
        }
    }

    fn code(self) -> u32 {
        match self {
            Self::Stereo => 0b00,
            Self::JointStereo { .. } => 0b01,
            Self::DualChannel => 0b10,
            Self::SingleChannel => 0b11,
        }
    }

    fn extension_code(self) -> u32 {
        match self {
            Self::JointStereo {
                intensity_stereo,
                ms_stereo,
            } => u32::from(ms_stereo) << 1 | u32::from(intensity_stereo),
            _ => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    not(feature = "unstable"),
    expect(
        dead_code,
        reason = "the encoder writes no emphasis; the conformance tests do"
    )
)]
pub enum Emphasis {
    None,
    Microseconds50_15,
    CcittJ17,
}

impl Emphasis {
    fn code(self) -> u32 {
        match self {
            Self::None => 0b00,
            Self::Microseconds50_15 => 0b01,
            Self::CcittJ17 => 0b11,
        }
    }
}

/// The bits of the header that describe the stream and leave its decoding
/// alone (clause 2.4.2.3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Flags {
    /// `private_bit`, a "bit for private use".
    pub private: bool,
    /// `copyright`: whether the stream is copyright protected.
    pub copyright: bool,
    /// `original/copy`: whether the stream is an original rather than a
    /// copy.
    pub original: bool,
}

/// A Layer III frame header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    /// Whether a `crc_check` word follows the header. Written inverted, as
    /// `protection_bit`.
    pub crc_protected: bool,
    pub bitrate: Bitrate,
    pub sampling_frequency: SamplingFrequency,
    pub padding: bool,
    pub mode: Mode,
    pub flags: Flags,
    pub emphasis: Emphasis,
}

impl Header {
    pub const LEN_BYTES: usize = 4;

    /// # Panics
    ///
    /// If the bitrate is not one of those of the sampling frequency's
    /// [`Version`].
    pub fn write(&self, w: &mut BitWriter) {
        let version = self.sampling_frequency.version();
        assert_eq!(
            self.bitrate.version, version,
            "{:?} is not a bitrate of {:?}",
            self.bitrate, self.sampling_frequency
        );
        w.write(self.sampling_frequency.syncword(), 12);
        w.write(version.id(), 1);
        w.write(LAYER_III, 2);
        w.write_bool(!self.crc_protected); // protection_bit
        w.write(u32::from(self.bitrate.index), 4);
        w.write(self.sampling_frequency.code(), 2);
        w.write_bool(self.padding);
        w.write_bool(self.flags.private);
        w.write(self.mode.code(), 2);
        w.write(self.mode.extension_code(), 2);
        w.write_bool(self.flags.copyright);
        w.write_bool(self.flags.original);
        w.write(self.emphasis.code(), 2);
    }

    /// Frame length in bytes (one slot is one byte in Layer III), header
    /// included.
    pub fn frame_len(&self) -> usize {
        self.bitrate
            .frame_len(self.sampling_frequency, self.padding)
    }
}

/// `N = 144 * bit_rate / sampling_frequency` (clause 2.4.3.1), with 72 in
/// place of 144 at the lower sampling frequencies (clause 2.4.3.2 of ISO/IEC
/// 13818-3), returned as the truncated slot count and the remainder of the
/// division. Both constants are a frame's samples over the 8 bits of a slot.
fn slots_per_frame(
    version: Version,
    bits_per_second: u32,
    sampling_frequency: SamplingFrequency,
) -> (usize, u32) {
    let samples = version.samples_per_frame() as u32;
    let numerator = samples / 8 * bits_per_second;
    let fs = sampling_frequency.hz();
    ((numerator / fs) as usize, numerator % fs)
}

/// Decides `padding_bit` frame by frame so that the mean frame length matches
/// the bitrate exactly: after `k` frames the total length is `floor(k * N)`
/// slots, `N` as in [`slots_per_frame`], which meets the "(+0, -1 slot)"
/// bound of clause 2.4.2.3.
///
/// Spec gap: that clause's sample procedure ("can be used") is optional; it
/// pads at different frames and is not followed here.
#[derive(Debug)]
pub struct PaddingSchedule {
    remainder_step: u32,
    sampling_frequency_hz: u32,
    remainder: u32,
}

impl PaddingSchedule {
    pub fn new(bitrate: Bitrate, sampling_frequency: SamplingFrequency) -> Self {
        let (_, remainder_step) =
            slots_per_frame(bitrate.version, bitrate.bits_per_second, sampling_frequency);
        Self {
            remainder_step,
            sampling_frequency_hz: sampling_frequency.hz(),
            remainder: 0,
        }
    }

    pub fn next_padding(&mut self) -> bool {
        self.remainder += self.remainder_step;
        let padding = self.remainder >= self.sampling_frequency_hz;
        if padding {
            self.remainder -= self.sampling_frequency_hz;
        }
        padding
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(kbps: u32, sampling_frequency: SamplingFrequency, mode: Mode) -> Header {
        Header {
            crc_protected: false,
            bitrate: Bitrate::from_kbps(sampling_frequency.version(), kbps).unwrap(),
            sampling_frequency,
            padding: false,
            mode,
            flags: Flags::default(),
            emphasis: Emphasis::None,
        }
    }

    fn header_bytes(h: &Header) -> Vec<u8> {
        let mut w = BitWriter::new();
        h.write(&mut w);
        assert_eq!(w.bit_len(), Header::LEN_BYTES * 8);
        w.into_bytes()
    }

    #[test]
    fn bitrate_lookup_rejects_values_outside_the_table() {
        let mpeg1 = |kbps| Bitrate::from_kbps(Version::Mpeg1, kbps);
        let mpeg2 = |kbps| Bitrate::from_kbps(Version::Mpeg2, kbps);
        assert_eq!(mpeg1(128).unwrap().bits_per_second(), 128_000);
        assert_eq!(mpeg2(8).unwrap().bits_per_second(), 8_000);
        assert!(mpeg1(0).is_none());
        assert!(mpeg1(384).is_none());
        assert!(mpeg1(8).is_none());
        assert!(mpeg2(192).is_none());
        assert_eq!(Bitrate::all(Version::Mpeg1).count(), 14);
        assert_eq!(Bitrate::all(Version::Mpeg2).count(), 14);
    }

    #[test]
    fn encodes_joint_stereo_header_fields() {
        let mut h = header(
            128,
            SamplingFrequency::Hz44100,
            Mode::JointStereo {
                intensity_stereo: false,
                ms_stereo: true,
            },
        );
        assert_eq!(header_bytes(&h), [0xFF, 0xFB, 0b1001_0000, 0b0110_0000]);

        h.padding = true;
        h.flags.original = true;
        h.emphasis = Emphasis::CcittJ17;
        assert_eq!(header_bytes(&h), [0xFF, 0xFB, 0b1001_0010, 0b0110_0111]);

        h.flags = Flags {
            private: true,
            copyright: true,
            original: false,
        };
        assert_eq!(header_bytes(&h), [0xFF, 0xFB, 0b1001_0011, 0b0110_1011]);
    }

    #[test]
    fn encodes_single_channel_at_48khz() {
        let h = header(320, SamplingFrequency::Hz48000, Mode::SingleChannel);
        assert_eq!(header_bytes(&h), [0xFF, 0xFB, 0b1110_0100, 0b1100_0000]);
    }

    /// `ID` 0, and the `bitrate_index` and `sampling_frequency` codes of
    /// ISO/IEC 13818-3.
    #[test]
    fn encodes_lower_sampling_frequencies() {
        let h = header(64, SamplingFrequency::Hz22050, Mode::Stereo);
        assert_eq!(header_bytes(&h), [0xFF, 0xF3, 0b1000_0000, 0b0000_0000]);
        let h = header(8, SamplingFrequency::Hz24000, Mode::SingleChannel);
        assert_eq!(header_bytes(&h), [0xFF, 0xF3, 0b0001_0100, 0b1100_0000]);
        let h = header(160, SamplingFrequency::Hz16000, Mode::SingleChannel);
        assert_eq!(header_bytes(&h), [0xFF, 0xF3, 0b1110_1000, 0b1100_0000]);
    }

    /// The sync word "FFE", then the fields of ISO/IEC 13818-3, with the
    /// `sampling_frequency` code of the frequency twice as high.
    #[test]
    fn encodes_mpeg25() {
        let h = header(64, SamplingFrequency::Hz11025, Mode::Stereo);
        assert_eq!(header_bytes(&h), [0xFF, 0xE3, 0b1000_0000, 0b0000_0000]);
        let h = header(8, SamplingFrequency::Hz12000, Mode::SingleChannel);
        assert_eq!(header_bytes(&h), [0xFF, 0xE3, 0b0001_0100, 0b1100_0000]);
        let h = header(160, SamplingFrequency::Hz8000, Mode::SingleChannel);
        assert_eq!(header_bytes(&h), [0xFF, 0xE3, 0b1110_1000, 0b1100_0000]);
    }

    #[test]
    #[should_panic(expected = "is not a bitrate of")]
    fn rejects_a_bitrate_of_the_other_version() {
        let mut h = header(128, SamplingFrequency::Hz44100, Mode::Stereo);
        h.sampling_frequency = SamplingFrequency::Hz22050;
        header_bytes(&h);
    }

    #[test]
    fn crc_protection_clears_protection_bit() {
        let mut h = header(320, SamplingFrequency::Hz48000, Mode::SingleChannel);
        h.crc_protected = true;
        assert_eq!(header_bytes(&h), [0xFF, 0xFA, 0b1110_0100, 0b1100_0000]);
    }

    #[test]
    fn frame_len_follows_slot_formula() {
        let mut h = header(128, SamplingFrequency::Hz44100, Mode::Stereo);
        assert_eq!(h.frame_len(), 417);
        h.padding = true;
        assert_eq!(h.frame_len(), 418);
        assert_eq!(
            header(128, SamplingFrequency::Hz48000, Mode::Stereo).frame_len(),
            384
        );
        assert_eq!(
            header(32, SamplingFrequency::Hz32000, Mode::Stereo).frame_len(),
            144
        );
        // Half as many slots for frames of one granule.
        assert_eq!(
            header(64, SamplingFrequency::Hz24000, Mode::Stereo).frame_len(),
            192
        );
        assert_eq!(
            header(8, SamplingFrequency::Hz16000, Mode::Stereo).frame_len(),
            36
        );
        assert_eq!(
            header(64, SamplingFrequency::Hz22050, Mode::Stereo).frame_len(),
            208
        );
        assert_eq!(
            header(8, SamplingFrequency::Hz8000, Mode::Stereo).frame_len(),
            72
        );
        assert_eq!(
            header(160, SamplingFrequency::Hz8000, Mode::Stereo).frame_len(),
            1440
        );
        assert_eq!(
            header(32, SamplingFrequency::Hz11025, Mode::Stereo).frame_len(),
            208
        );
    }

    #[test]
    fn padding_keeps_cumulative_length_exact() {
        for fs in SamplingFrequency::ALL {
            let slots_per_kbit = fs.version().samples_per_frame() as u64 / 8;
            for bitrate in Bitrate::all(fs.version()) {
                let mut schedule = PaddingSchedule::new(bitrate, fs);
                let mut h = header(bitrate.bits_per_second() / 1000, fs, Mode::Stereo);
                let mut total = 0u64;
                for k in 1..=2000u64 {
                    h.padding = schedule.next_padding();
                    total += h.frame_len() as u64;
                    let bits = u64::from(bitrate.bits_per_second());
                    let exact = k * slots_per_kbit * bits / u64::from(fs.hz());
                    assert_eq!(total, exact, "{bitrate:?} at {fs:?}, frame {k}");
                }
            }
        }
    }

    #[test]
    fn padding_is_never_needed_but_at_44_1_22_05_and_11_025_khz() {
        for fs in [
            SamplingFrequency::Hz48000,
            SamplingFrequency::Hz32000,
            SamplingFrequency::Hz24000,
            SamplingFrequency::Hz16000,
            SamplingFrequency::Hz12000,
            SamplingFrequency::Hz8000,
        ] {
            for bitrate in Bitrate::all(fs.version()) {
                let mut schedule = PaddingSchedule::new(bitrate, fs);
                assert!(
                    (0..100).all(|_| !schedule.next_padding()),
                    "{bitrate:?} at {fs:?}"
                );
            }
        }
    }

    /// Free format bitrates lie below the highest of the table and leave
    /// out its own, and start where a frame at the version's highest
    /// sampling frequency holds the header, `crc_check` and the side
    /// information of two channels: 38 bytes at 48 kHz from 12 667 bit/s,
    /// and 23 at 24 kHz from 7 667, where 8 kbit/s is in the table.
    #[test]
    fn free_format_bitrates() {
        use Version::*;
        let free = |version, bps| Bitrate::free(version, bps).map(|b| b.bits_per_second());
        assert_eq!(free(Mpeg1, 12_666), None);
        assert_eq!(free(Mpeg1, 12_667), Some(12_667));
        assert_eq!(free(Mpeg1, 100_000), Some(100_000));
        assert_eq!(free(Mpeg1, 128_000), None);
        assert_eq!(free(Mpeg1, 319_999), Some(319_999));
        assert_eq!(free(Mpeg1, 320_000), None);
        assert_eq!(free(Mpeg1, 640_000), None);
        assert_eq!(free(Mpeg2, 7_666), None);
        assert_eq!(free(Mpeg2, 8_000), None);
        assert_eq!(free(Mpeg2, 8_001), Some(8_001));
        assert_eq!(free(Mpeg2, 159_999), Some(159_999));
        assert_eq!(free(Mpeg2, 160_000), None);

        let shortest = Bitrate::free(Mpeg1, 12_667).unwrap();
        assert_eq!(shortest.frame_len(SamplingFrequency::Hz48000, false), 38);
        assert_eq!(shortest.frame_len(SamplingFrequency::Hz32000, false), 57);

        // bitrate_index 0, and N or N + 1 slots: 144 * 100 000 / 44 100 is
        // 326.5, so that padding alternates.
        let mut h = header(128, SamplingFrequency::Hz44100, Mode::Stereo);
        h.bitrate = Bitrate::free(Mpeg1, 100_000).unwrap();
        let mut w = BitWriter::new();
        h.write(&mut w);
        assert_eq!(w.into_bytes()[2] >> 4, 0);
        let mut padding = PaddingSchedule::new(h.bitrate, SamplingFrequency::Hz44100);
        let lens: Vec<_> = (0..4)
            .map(|_| {
                h.bitrate
                    .frame_len(SamplingFrequency::Hz44100, padding.next_padding())
            })
            .collect();
        assert_eq!(lens, [326, 327, 326, 327]);
    }
}
