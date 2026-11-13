//! The 32-bit frame header (clauses 2.4.1.3 and 2.4.2.3) and the frame length
//! it implies (clause 2.4.3.1).

use crate::bitstream::BitWriter;

const SYNCWORD: u32 = 0xFFF;
const ID_MPEG_AUDIO: u32 = 1;
const LAYER_III: u32 = 0b01;

/// Indexed by `bitrate_index`; index 0 is free format.
const LAYER_III_KBPS: [u32; 15] = [
    0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
];

/// A Layer III bitrate from the `bitrate_index` table. Free format is not
/// supported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bitrate {
    index: u8,
}

impl Bitrate {
    pub fn from_kbps(kbps: u32) -> Option<Self> {
        let index = LAYER_III_KBPS[1..].iter().position(|&k| k == kbps)? + 1;
        Some(Self { index: index as u8 })
    }

    pub fn all() -> impl Iterator<Item = Self> {
        (1..LAYER_III_KBPS.len() as u8).map(|index| Self { index })
    }

    pub fn bits_per_second(self) -> u32 {
        LAYER_III_KBPS[usize::from(self.index)] * 1000
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingFrequency {
    Hz44100,
    Hz48000,
    Hz32000,
}

impl SamplingFrequency {
    pub const ALL: [Self; 3] = [Self::Hz44100, Self::Hz48000, Self::Hz32000];

    pub fn hz(self) -> u32 {
        match self {
            Self::Hz44100 => 44_100,
            Self::Hz48000 => 48_000,
            Self::Hz32000 => 32_000,
        }
    }

    fn code(self) -> u32 {
        match self {
            Self::Hz44100 => 0b00,
            Self::Hz48000 => 0b01,
            Self::Hz32000 => 0b10,
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

/// A Layer III frame header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    /// Whether a `crc_check` word follows the header. Written inverted, as
    /// `protection_bit`.
    pub crc_protected: bool,
    pub bitrate: Bitrate,
    pub sampling_frequency: SamplingFrequency,
    pub padding: bool,
    pub private: bool,
    pub mode: Mode,
    pub copyright: bool,
    pub original: bool,
    pub emphasis: Emphasis,
}

impl Header {
    pub const LEN_BYTES: usize = 4;

    pub fn write(&self, w: &mut BitWriter) {
        w.write(SYNCWORD, 12);
        w.write(ID_MPEG_AUDIO, 1);
        w.write(LAYER_III, 2);
        w.write_bool(!self.crc_protected); // protection_bit
        w.write(u32::from(self.bitrate.index), 4);
        w.write(self.sampling_frequency.code(), 2);
        w.write_bool(self.padding);
        w.write_bool(self.private);
        w.write(self.mode.code(), 2);
        w.write(self.mode.extension_code(), 2);
        w.write_bool(self.copyright);
        w.write_bool(self.original);
        w.write(self.emphasis.code(), 2);
    }

    /// Frame length in bytes (one slot is one byte in Layer III), header
    /// included.
    pub fn frame_len(&self) -> usize {
        let (slots, _) = slots_per_frame(self.bitrate, self.sampling_frequency);
        slots + usize::from(self.padding)
    }
}

/// `N = 144 * bit_rate / sampling_frequency` (clause 2.4.3.1), returned as the
/// truncated slot count and the remainder of the division.
fn slots_per_frame(bitrate: Bitrate, sampling_frequency: SamplingFrequency) -> (usize, u32) {
    let numerator = 144 * bitrate.bits_per_second();
    let fs = sampling_frequency.hz();
    ((numerator / fs) as usize, numerator % fs)
}

/// Decides `padding_bit` frame by frame so that the mean frame length matches
/// the bitrate exactly: after `k` frames the total length is `floor(k * 144 *
/// bit_rate / sampling_frequency)` slots, which meets the "(+0, -1 slot)"
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
        let (_, remainder_step) = slots_per_frame(bitrate, sampling_frequency);
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
            bitrate: Bitrate::from_kbps(kbps).unwrap(),
            sampling_frequency,
            padding: false,
            private: false,
            mode,
            copyright: false,
            original: false,
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
        assert_eq!(Bitrate::from_kbps(128).unwrap().bits_per_second(), 128_000);
        assert!(Bitrate::from_kbps(0).is_none());
        assert!(Bitrate::from_kbps(384).is_none());
        assert_eq!(Bitrate::all().count(), 14);
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
        h.original = true;
        h.emphasis = Emphasis::CcittJ17;
        assert_eq!(header_bytes(&h), [0xFF, 0xFB, 0b1001_0010, 0b0110_0111]);
    }

    #[test]
    fn encodes_single_channel_at_48khz() {
        let h = header(320, SamplingFrequency::Hz48000, Mode::SingleChannel);
        assert_eq!(header_bytes(&h), [0xFF, 0xFB, 0b1110_0100, 0b1100_0000]);
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
    }

    #[test]
    fn padding_keeps_cumulative_length_exact() {
        for fs in SamplingFrequency::ALL {
            for bitrate in Bitrate::all() {
                let mut schedule = PaddingSchedule::new(bitrate, fs);
                let mut h = header(bitrate.bits_per_second() / 1000, fs, Mode::Stereo);
                let mut total = 0u64;
                for k in 1..=2000u64 {
                    h.padding = schedule.next_padding();
                    total += h.frame_len() as u64;
                    let exact = k * 144 * u64::from(bitrate.bits_per_second()) / u64::from(fs.hz());
                    assert_eq!(total, exact, "{bitrate:?} at {fs:?}, frame {k}");
                }
            }
        }
    }

    #[test]
    fn padding_is_never_needed_at_48_and_32_khz() {
        for fs in [SamplingFrequency::Hz48000, SamplingFrequency::Hz32000] {
            for bitrate in Bitrate::all() {
                let mut schedule = PaddingSchedule::new(bitrate, fs);
                assert!(
                    (0..100).all(|_| !schedule.next_padding()),
                    "{bitrate:?} at {fs:?}"
                );
            }
        }
    }
}
