// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The `AudioSpecificConfig()` of ISO/IEC 14496-3 (Table 1.19), with its
//! `GASpecificConfig()` (Table 4.1): what a decoder of raw frames, as
//! from an MPEG-4 file, needs to be told of the stream.

use chelys_core::bitstream::BitWriter;

use super::channel_configuration::ChannelConfiguration;
use super::sampling_frequency::SamplingFrequency;

/// `audioObjectType` of AAC LC (Table 1.21).
const AAC_LC: u32 = 2;

/// The `AudioSpecificConfig()` of an AAC LC stream.
///
/// It ends after `GASpecificConfig()`: the optional fields after it,
/// which signal SBR and the like where `bits_to_decode()` leaves room,
/// are not written. Its 16 bits are whole bytes.
pub fn audio_specific_config(
    sampling_frequency: SamplingFrequency,
    channel_configuration: ChannelConfiguration,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    // GetAudioObjectType() (Table 1.20): below 31, no escape.
    w.write(AAC_LC, 5); // audioObjectType
    // samplingFrequencyIndex: not the escape 0xf, so no samplingFrequency.
    w.write(sampling_frequency.index(), 4);
    w.write(channel_configuration.value(), 4); // channelConfiguration
    // audioObjectType is neither 5 nor 29: no SBR fields; 2 is among those
    // of GASpecificConfig().
    // GASpecificConfig()
    w.write(0, 1); // frameLengthFlag: 1024 lines, not 960
    w.write(0, 1); // dependsOnCoreCoder
    // extensionFlag: "Shall be '0' for audio object types 1, 2, 3, 4, 6, 7"
    // (clause 4.5.1.1).
    w.write(0, 1);
    // channelConfiguration is not 0, so no program_config_element(); 2 is
    // neither 6 nor 20, so no layerNr. No epConfig, which only the error
    // resilient types have.
    debug_assert_eq!(w.bit_len() % 8, 0);
    w.into_bytes()
}

#[cfg(test)]
mod tests {
    use chelys_core::bitstream::binary_digits;

    use super::*;

    #[test]
    fn spells_its_fields() {
        let config =
            audio_specific_config(SamplingFrequency::Hz48000, ChannelConfiguration::Stereo);
        let fields = [
            "00010", // audioObjectType
            "0011",  // samplingFrequencyIndex
            "0010",  // channelConfiguration
            "0",     // frameLengthFlag
            "0",     // dependsOnCoreCoder
            "0",     // extensionFlag
        ];
        assert_eq!(binary_digits(&config), fields.concat());
        assert_eq!(config, [0x11, 0x90]);
    }
}
