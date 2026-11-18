// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The sampling frequencies AAC codes, by their `sampling_frequency_index`
//! (Table 35; Table 1.22 of ISO/IEC 14496-3 gives the same).

/// A sampling frequency that has a `sampling_frequency_index` of its own.
///
/// Spec reading: ISO/IEC 14496-3 adds 7350 Hz at index 0xc (Table 1.22),
/// which Table 35 of ISO/IEC 13818-7 reserves, and with it the
/// `sampling_frequency_index` of the ADTS this encoder writes; it is left
/// out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingFrequency {
    Hz96000,
    Hz88200,
    Hz64000,
    Hz48000,
    Hz44100,
    Hz32000,
    Hz24000,
    Hz22050,
    Hz16000,
    Hz12000,
    Hz11025,
    Hz8000,
}

impl SamplingFrequency {
    /// In the order of their index.
    pub const ALL: [Self; 12] = [
        Self::Hz96000,
        Self::Hz88200,
        Self::Hz64000,
        Self::Hz48000,
        Self::Hz44100,
        Self::Hz32000,
        Self::Hz24000,
        Self::Hz22050,
        Self::Hz16000,
        Self::Hz12000,
        Self::Hz11025,
        Self::Hz8000,
    ];

    pub fn from_hz(hz: u32) -> Option<Self> {
        Self::ALL.into_iter().find(|fs| fs.hz() == hz)
    }

    pub fn hz(self) -> u32 {
        match self {
            Self::Hz96000 => 96_000,
            Self::Hz88200 => 88_200,
            Self::Hz64000 => 64_000,
            Self::Hz48000 => 48_000,
            Self::Hz44100 => 44_100,
            Self::Hz32000 => 32_000,
            Self::Hz24000 => 24_000,
            Self::Hz22050 => 22_050,
            Self::Hz16000 => 16_000,
            Self::Hz12000 => 12_000,
            Self::Hz11025 => 11_025,
            Self::Hz8000 => 8_000,
        }
    }

    /// The `sampling_frequency_index` of ADTS (Table 8) and the
    /// `samplingFrequencyIndex` of the `AudioSpecificConfig()`.
    pub fn index(self) -> u32 {
        self as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexes_follow_table_35() {
        let table = [
            (0x0, 96_000),
            (0x1, 88_200),
            (0x2, 64_000),
            (0x3, 48_000),
            (0x4, 44_100),
            (0x5, 32_000),
            (0x6, 24_000),
            (0x7, 22_050),
            (0x8, 16_000),
            (0x9, 12_000),
            (0xa, 11_025),
            (0xb, 8_000),
        ];
        for (fs, (index, hz)) in SamplingFrequency::ALL.into_iter().zip(table) {
            assert_eq!((fs.index(), fs.hz()), (index, hz));
            assert_eq!(SamplingFrequency::from_hz(hz), Some(fs));
        }
        assert_eq!(SamplingFrequency::from_hz(7_350), None);
    }
}
