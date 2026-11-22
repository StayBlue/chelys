// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The descriptors of ISO/IEC 14496-1:2026 that an MP4 file stores in a
//! track's sample description (ISO/IEC 14496-14, clause 4.1.2): the
//! `ES_Descriptor` (clause 6.2.6.5), with its `DecoderConfigDescriptor`
//! (clause 6.2.6.6), `DecoderSpecificInfo` (clause 6.2.6.7) and
//! `SLConfigDescriptor` (clause 6.3.2.3).

/// The class tags of the descriptors (Table 1).
const ES_DESCR_TAG: u8 = 0x03;
const DECODER_CONFIG_DESCR_TAG: u8 = 0x04;
const DEC_SPECIFIC_INFO_TAG: u8 = 0x05;
const SL_CONFIG_DESCR_TAG: u8 = 0x06;

/// `objectTypeIndication` "Audio ISO/IEC 14496-3" (Table 5).
pub const AUDIO_ISO_IEC_14496_3: u8 = 0x40;

/// `streamType` "AudioStream" (Table 6).
pub const AUDIO_STREAM: u8 = 0x05;

/// The `predefined` value of an `SLConfigDescriptor` "Reserved for use in
/// MP4 files" (Table 13), which ISO/IEC 14496-14 requires of a stream in
/// the file (clause 4.1.1). Its `durationFlag` is 0 and its
/// `useTimeStampsFlag` 1 (Table 14), so that no field follows it.
const SL_PREDEFINED_MP4: u8 = 0x02;

/// The largest `sizeOfInstance` of a descriptor: `BaseDescriptor` is
/// `expandable(2^28-1)` (clause 6.2.2.2).
const MAX_SIZE_OF_INSTANCE: usize = (1 << 28) - 1;

/// The encoding of `sizeOfInstance`, which precedes every descriptor's
/// fields after its tag, as `BaseDescriptor` is an expandable class: 7 bits
/// a byte, most significant first, each byte but the last with its top
/// bit, `nextByte`, set.
///
/// Spec reading: ISO/IEC 14496-1:2026 refers to "5.3 in ISO/IEC
/// 14496-34:2025" for the encoding, but clause 5.3 there is "Case
/// sensitivity"; the encoding is in clause 7.5, "Expandable classes"
/// (printed pages 33 and 34), which this follows.
///
/// Spec gap: the syntax lets a size be written in more bytes than it
/// needs, with leading `sizeByte`s of 0. The fewest bytes are written.
fn size_of_instance(size: usize) -> Vec<u8> {
    assert!(size <= MAX_SIZE_OF_INSTANCE, "a descriptor of {size} bytes");
    let groups = (1..4).take_while(|&i| size >> (7 * i) != 0).count() + 1;
    (0..groups)
        .rev()
        .map(|i| {
            let next_byte = if i > 0 { 0x80 } else { 0 };
            next_byte | ((size >> (7 * i)) & 0x7f) as u8
        })
        .collect()
}

/// A descriptor: its tag, `sizeOfInstance`, and `fields`, as many bytes as
/// that counts.
fn descriptor(tag: u8, fields: &[u8]) -> Vec<u8> {
    let mut out = vec![tag];
    out.extend(size_of_instance(fields.len()));
    out.extend_from_slice(fields);
    out
}

/// What a `DecoderConfigDescriptor` tells a decoder of an elementary stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecoderConfig<'a> {
    pub object_type_indication: u8,
    pub stream_type: u8,
    /// `bufferSizeDB`: the size of the decoding buffer in bytes, of 24
    /// bits.
    pub buffer_size: u32,
    /// `maxBitrate` in bits per second.
    pub max_bitrate: u32,
    /// `avgBitrate` in bits per second.
    pub avg_bitrate: u32,
    /// The `DecoderSpecificInfo`'s bytes after its tag and size, as the
    /// `AudioSpecificConfig()` of ISO/IEC 14496-3 for an object type of
    /// [`AUDIO_ISO_IEC_14496_3`] (clause 6.2.6.7.2).
    pub specific_info: &'a [u8],
}

/// The `ES_Descriptor` of a stream within the file, as ISO/IEC 14496-14
/// restricts it (clause 4.1.2): its `ES_ID` 0, `streamDependenceFlag`,
/// `URL_Flag` and `OCRstreamFlag` 0, and the `SLConfigDescriptor`
/// predefined as 2.
///
/// Spec gap: `streamPriority`'s values "are not normatively defined", and
/// ISO/IEC 14496-14 does not restrict it. A file of one stream has no
/// other to rank it against: it is 0.
///
/// No descriptor follows the `SLConfigDescriptor`: the IPI, IP
/// identification, IPMP, language, QoS, registration and extension
/// descriptors are each optional, `[0 .. 1]` or `[0 .. 255]`.
pub fn es_descriptor(config: &DecoderConfig) -> Vec<u8> {
    assert!(config.buffer_size < 1 << 24, "bufferSizeDB is of 24 bits");
    let mut decoder_config = vec![config.object_type_indication];
    // streamType, upStream 0 and the reserved bit, 1.
    decoder_config.push((config.stream_type << 2) | 1);
    decoder_config.extend_from_slice(&config.buffer_size.to_be_bytes()[1..]);
    decoder_config.extend_from_slice(&config.max_bitrate.to_be_bytes());
    decoder_config.extend_from_slice(&config.avg_bitrate.to_be_bytes());
    decoder_config.extend(descriptor(DEC_SPECIFIC_INFO_TAG, config.specific_info));
    // No profileLevelIndicationIndexDescriptor, of which there are 0 to
    // 255.

    let mut fields = Vec::new();
    fields.extend_from_slice(&0u16.to_be_bytes()); // ES_ID
    // streamDependenceFlag, URL_Flag and OCRstreamFlag, then
    // streamPriority.
    fields.push(0);
    fields.extend(descriptor(DECODER_CONFIG_DESCR_TAG, &decoder_config));
    fields.extend(descriptor(SL_CONFIG_DESCR_TAG, &[SL_PREDEFINED_MP4]));
    descriptor(ES_DESCR_TAG, &fields)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_take_7_bits_a_byte_most_significant_first() {
        assert_eq!(size_of_instance(0), [0x00]);
        assert_eq!(size_of_instance(2), [0x02]);
        assert_eq!(size_of_instance(127), [0x7f]);
        assert_eq!(size_of_instance(128), [0x81, 0x00]);
        assert_eq!(size_of_instance(300), [0x82, 0x2c]);
        assert_eq!(size_of_instance(16_383), [0xff, 0x7f]);
        assert_eq!(size_of_instance(16_384), [0x81, 0x80, 0x00]);
        assert_eq!(size_of_instance((1 << 21) - 1), [0xff, 0xff, 0x7f]);
        assert_eq!(size_of_instance(1 << 21), [0x81, 0x80, 0x80, 0x00]);
        assert_eq!(
            size_of_instance(MAX_SIZE_OF_INSTANCE),
            [0xff, 0xff, 0xff, 0x7f]
        );
    }

    /// Reads a size as clause 7.5 of ISO/IEC 14496-34 does.
    fn read_size(bytes: &[u8]) -> (usize, usize) {
        let mut size = 0;
        for (i, &byte) in bytes.iter().enumerate() {
            size = (size << 7) | usize::from(byte & 0x7f);
            if byte & 0x80 == 0 {
                return (size, i + 1);
            }
        }
        panic!("the size does not end");
    }

    #[test]
    fn every_size_reads_back() {
        for size in (0..70_000).chain([(1 << 21) - 1, 1 << 21, MAX_SIZE_OF_INSTANCE]) {
            let bytes = size_of_instance(size);
            assert_eq!(read_size(&bytes), (size, bytes.len()), "{size}");
        }
    }

    #[test]
    #[should_panic(expected = "a descriptor of 268435456 bytes")]
    fn refuses_a_size_past_the_largest() {
        size_of_instance(MAX_SIZE_OF_INSTANCE + 1);
    }

    #[test]
    fn spells_the_es_descriptor() {
        let config = DecoderConfig {
            object_type_indication: AUDIO_ISO_IEC_14496_3,
            stream_type: AUDIO_STREAM,
            buffer_size: 1536,
            max_bitrate: 140_000,
            avg_bitrate: 128_000,
            specific_info: &[0x12, 0x10],
        };
        let expected = [
            &[0x03, 25][..],     // ES_DescrTag, sizeOfInstance
            &[0x00, 0x00],       // ES_ID
            &[0x00],             // flags, streamPriority
            &[0x04, 17],         // DecoderConfigDescrTag, sizeOfInstance
            &[0x40],             // objectTypeIndication
            &[0b0001_0101],      // streamType 5, upStream 0, reserved 1
            &[0x00, 0x06, 0x00], // bufferSizeDB
            &140_000u32.to_be_bytes(),
            &128_000u32.to_be_bytes(),
            &[0x05, 2, 0x12, 0x10], // DecSpecificInfoTag, size, the bytes
            &[0x06, 1, 0x02],       // SLConfigDescrTag, size, predefined
        ]
        .concat();
        assert_eq!(es_descriptor(&config), expected);
    }
}
