// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The frames' layout (clause 2.4.1.2): the header, `crc_check` word, side
//! information and main data of each frame, and main data that begins in
//! earlier frames (clause 2.4.3.4.3); and streams encoded from quantised
//! granules alone.

use std::collections::VecDeque;
use std::ops::Range;

use super::StreamConfig;
use crate::ancillary::AncillaryQueue;
use crate::bitstream::BitWriter;
use crate::crc;
use crate::granule::{MAX_PART2_3_LENGTH, QuantizedGranule};
use crate::header::{Header, Mode};
use crate::huffman::QuadTable;
use crate::scalefactor_bands;
use crate::side_info::{GranuleChannel, SideInfo};
#[cfg(any(test, feature = "unstable"))]
use crate::{
    granule::{BlockType, GRANULE_LINES},
    header::PaddingSchedule,
};

/// Appends one frame (clause 2.4.1.2): the header, the `crc_check` word if
/// the header asks for one, the side information, and then `main_data`
/// zero-filled to the frame length. The main data starts in this frame, so
/// `side_info.main_data_begin` should be 0.
///
/// # Panics
///
/// If `main_data` does not fit in the frame.
#[cfg(any(test, feature = "unstable"))]
pub fn write_frame(header: &Header, side_info: &SideInfo, main_data: &[u8], out: &mut Vec<u8>) {
    let slot = write_frame_start(header, side_info, out);
    assert!(
        main_data.len() <= slot.len(),
        "{} bytes of main data exceed the {} available in the frame",
        main_data.len(),
        slot.len(),
    );
    out[slot.start..slot.start + main_data.len()].copy_from_slice(main_data);
}

/// Appends a frame's header, `crc_check` word and side information, and
/// zeros for the rest of the frame, whose range in `out` it returns: the
/// frame's main data slot.
fn write_frame_start(header: &Header, side_info: &SideInfo, out: &mut Vec<u8>) -> Range<usize> {
    let start = out.len();
    let mut w = BitWriter::new();
    header.write(&mut w);
    side_info.write(header.mode.channels(), &mut w);
    let bytes = w.into_bytes();
    let (header_bytes, side_info_bytes) = bytes.split_at(Header::LEN_BYTES);

    out.extend_from_slice(header_bytes);
    if header.crc_protected {
        // Table B.5: bits 16..31 of the header and all of the side information.
        let protected = header_bytes[2..].iter().chain(side_info_bytes).copied();
        out.extend_from_slice(&crc::crc16(protected).to_be_bytes());
    }
    out.extend_from_slice(side_info_bytes);

    let slot = out.len()..start + header.frame_len();
    out.resize(slot.end, 0);
    slot
}

/// Writes frames whose main data may start in earlier frames (clause
/// 2.4.3.4.3, figure A.7). The main data of all frames forms one stream of
/// bytes, laid out over the frames' main data slots in order, and each
/// frame's `main_data_begin` counts back into the slot bytes before its
/// header that earlier frames left unused.
#[derive(Debug, Default)]
pub(super) struct FrameWriter {
    pub(super) out: Vec<u8>,
    /// The positions in `out` of slot bytes no main data has used, in stream
    /// order.
    unused: VecDeque<usize>,
}

impl FrameWriter {
    /// # Panics
    ///
    /// If fewer unused bytes than `main_data_begin` precede the frame, or
    /// `main_data` does not fit in them and the frame's slot.
    pub(super) fn write(&mut self, header: &Header, side_info: &SideInfo, main_data: &[u8]) {
        let begin = usize::from(side_info.main_data_begin);
        assert!(
            begin <= self.unused.len(),
            "main_data_begin {begin} reaches past the {} unused bytes before the frame",
            self.unused.len()
        );
        // Unused bytes further back stay zero: stuffing (clause C.1.5.4.2.2).
        self.unused.drain(..self.unused.len() - begin);
        let slot = write_frame_start(header, side_info, &mut self.out);
        self.unused.extend(slot);
        assert!(
            main_data.len() <= self.unused.len(),
            "{} bytes of main data exceed the {} available to the frame",
            main_data.len(),
            self.unused.len()
        );
        for &byte in main_data {
            let at = self.unused.pop_front().expect("checked above");
            self.out[at] = byte;
        }
    }

    /// Keeps only the last `main_data_begin` unused bytes, those the next
    /// frame's main data begins in; those before them stay zero.
    pub(super) fn keep_unused(&mut self, main_data_begin: usize) {
        let keep = main_data_begin.min(self.unused.len());
        self.unused.drain(..self.unused.len() - keep);
    }

    /// Moves to `out` the bytes before the first that later main data may
    /// still fill.
    pub(super) fn take_final(&mut self, out: &mut Vec<u8>) {
        let end = self.unused.front().copied().unwrap_or(self.out.len());
        out.extend(self.out.drain(..end));
        for at in &mut self.unused {
            *at -= end;
        }
    }

    /// Moves the rest of the stream to `out`, with no frame to follow.
    pub(super) fn finish(self, out: &mut Vec<u8>) {
        out.extend(self.out);
    }
}

/// The most bits of ancillary data a frame of `config` can carry: as many
/// as the main data slot of its shortest frame holds.
pub fn max_ancillary_bits(config: &StreamConfig) -> usize {
    let shortest = config.frame_header(config.rate.lowest(), false, false, false);
    main_data_bits(&shortest)
}

/// The bits a frame has for main data: all of it but the header, the
/// `crc_check` word and the side information.
pub(super) fn main_data_bits(header: &Header) -> usize {
    let crc = if header.crc_protected { 2 } else { 0 };
    let side_info =
        SideInfo::len_bytes(header.sampling_frequency.version(), header.mode.channels());
    8 * (header.frame_len() - Header::LEN_BYTES - crc - side_info)
}

/// The side information and main data of one frame of granules, one entry
/// per channel each, under `header`.
///
/// # Panics
///
/// If the right channel's scalefactors give a zero part
/// ([`Scalefactors::intensity`](crate::scalefactors::Scalefactors::intensity)) and
/// `header` does not have intensity stereo on, or the other way round: the
/// decoder would read its zero part, and at the lower sampling frequencies
/// its `scalefac_compress`, otherwise than they were coded.
///
/// With `ancillary`, the main data ends with that many bits taken from the
/// queue, as the frame's ancillary data.
pub(super) fn frame_contents(
    header: &Header,
    frame: &[Vec<QuantizedGranule>],
    main_data_begin: usize,
    ancillary: Option<(&mut AncillaryQueue, usize)>,
) -> (SideInfo, Vec<u8>) {
    let sampling_frequency = header.sampling_frequency;
    let version = sampling_frequency.version();
    assert!(
        main_data_begin < 1 << version.main_data_begin_bits(),
        "main_data_begin {main_data_begin} overflows its field"
    );
    let intensity_stereo = matches!(
        header.mode,
        Mode::JointStereo {
            intensity_stereo: true,
            ..
        }
    );
    for granule in frame {
        for (ch, channel) in granule.iter().enumerate() {
            assert_eq!(
                channel.scalefactors.intensity.is_some(),
                intensity_stereo && ch == 1,
                "only the right channel of a frame in intensity stereo has a zero part"
            );
        }
    }
    let mut side_info = SideInfo {
        main_data_begin: main_data_begin as u16,
        ..SideInfo::new(version)
    };
    let mut main_data = BitWriter::new();
    for (gr, granule) in frame.iter().enumerate() {
        for (ch, channel) in granule.iter().enumerate() {
            let start = main_data.bit_len();
            let scalefactors = &channel.scalefactors;
            // scfsi is sent once per frame and applies to its second granule.
            if gr == 0 {
                assert!(
                    !scalefactors.scfsi.contains(&true),
                    "the first granule of a frame sends all its scalefactors"
                );
            } else {
                side_info.scfsi[ch] = scalefactors.scfsi;
            }
            let scalefac_compress = scalefactors
                .compress()
                .expect("scalefactors within their limits");
            scalefactors.write(scalefac_compress, &mut main_data);
            let bands = scalefactor_bands::coding_bands(channel.block_type, sampling_frequency);
            channel
                .coding
                .write(&channel.values, &bands, &mut main_data);
            let part2_3_length = main_data.bit_len() - start;
            assert!(
                part2_3_length <= MAX_PART2_3_LENGTH,
                "{part2_3_length} bits overflow the 12-bit part2_3_length"
            );
            let coding = &channel.coding;
            side_info.granules[gr][ch] = GranuleChannel {
                part2_3_length: part2_3_length as u16,
                big_values: coding.big_values as u16,
                global_gain: channel.global_gain,
                scalefac_compress,
                window: channel.window(),
                preflag: scalefactors.preflag,
                scalefac_scale: scalefactors.scalefac_scale,
                count1table_select: coding.count1_table == QuadTable::B,
            };
        }
    }
    if let Some((queue, bits)) = ancillary {
        queue.write(bits, &mut main_data);
    }
    (side_info, main_data.into_bytes())
}

/// Encodes quantised granules, given in order with one entry per channel,
/// into frames of two granules each, or one at the lower sampling
/// frequencies, at a constant bitrate.
///
/// Main data is not shared between frames: `main_data_begin` is always 0. In
/// joint stereo every frame has MS stereo and intensity stereo on as the
/// configured mode sets them; in intensity stereo the right channel's
/// scalefactors must give its zero part
/// ([`Scalefactors::intensity`](crate::scalefactors::Scalefactors::intensity)).
///
/// # Panics
///
/// If the rate is not [`Rate::Constant`](crate::rate::Rate::Constant), the
/// granules do not divide into frames for the configured channels, or a
/// frame's granules do not fit in it.
#[cfg(any(test, feature = "unstable"))]
pub fn encode_granules(config: &StreamConfig, granules: &[Vec<QuantizedGranule>]) -> Vec<u8> {
    let channels = config.mode.channels();
    let per_frame = config.sampling_frequency.version().granules();
    assert!(
        granules.len().is_multiple_of(per_frame),
        "a frame holds {per_frame} granules"
    );
    assert!(
        granules.iter().all(|g| g.len() == channels),
        "each granule needs one entry per channel"
    );
    let (ms_stereo, intensity_stereo) = config.joint_stereo();
    let bitrate = config.constant_bitrate();
    let mut padding = PaddingSchedule::new(bitrate, config.sampling_frequency);
    let mut writer = FrameWriter::default();
    for frame in granules.chunks_exact(per_frame) {
        let header =
            config.frame_header(bitrate, padding.next_padding(), ms_stereo, intensity_stereo);
        let (side_info, main_data) = frame_contents(&header, frame, 0, None);
        writer.write(&header, &side_info, &main_data);
    }
    writer.out
}

/// Encodes `frames` frames of digital silence. Every granule codes to
/// nothing, so each frame is zero-filled after its side information.
#[cfg(any(test, feature = "unstable"))]
pub fn encode_silence(config: &StreamConfig, frames: usize) -> Vec<u8> {
    let silent = QuantizedGranule::new(
        0,
        [0; GRANULE_LINES],
        BlockType::Normal,
        config.sampling_frequency,
    );
    let granule = vec![silent; config.mode.channels()];
    let granules = frames * config.sampling_frequency.version().granules();
    encode_granules(config, &vec![granule; granules])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::{Bitrate, Emphasis, Flags, SamplingFrequency, Version};
    use crate::rate::Rate;

    fn config(crc_protected: bool) -> StreamConfig {
        StreamConfig {
            rate: Rate::Constant(Bitrate::from_kbps(Version::Mpeg1, 128).unwrap()),
            sampling_frequency: SamplingFrequency::Hz44100,
            mode: Mode::Stereo,
            crc_protected,
            flags: Flags::default(),
        }
    }

    #[test]
    fn silence_frames_are_back_to_back() {
        let bytes = encode_silence(&config(false), 3);
        // 128 kbit/s at 44.1 kHz is 417.96 slots per frame.
        assert_eq!(bytes.len(), 417 + 418 + 418);
        for (offset, padded) in [(0, false), (417, true), (835, true)] {
            assert_eq!(bytes[offset..offset + 2], [0xFF, 0xFB]);
            assert_eq!(bytes[offset + 2] & 0b10 != 0, padded);
        }
    }

    #[test]
    fn crc_word_sits_between_header_and_side_info() {
        let plain = encode_silence(&config(false), 1);
        let protected = encode_silence(&config(true), 1);
        assert_eq!(plain.len(), protected.len());
        assert_eq!(protected[..2], [0xFF, 0xFA]);
        assert_eq!(protected[2..4], plain[2..4]);

        let side_info = &plain[4..4 + SideInfo::len_bytes(Version::Mpeg1, 2)];
        assert_eq!(protected[6..6 + side_info.len()], *side_info);
        let expected = crc::crc16(plain[2..4].iter().chain(side_info).copied());
        assert_eq!(protected[4..6], expected.to_be_bytes());
    }

    fn small_header() -> Header {
        Header {
            crc_protected: false,
            bitrate: Bitrate::from_kbps(Version::Mpeg1, 32).unwrap(),
            sampling_frequency: SamplingFrequency::Hz32000,
            padding: false,
            mode: Mode::SingleChannel,
            flags: Flags::default(),
            emphasis: Emphasis::None,
        }
    }

    /// Figure A.7: a frame's main data starts `main_data_begin` bytes back
    /// in the slot bytes the frames before it left unused, and runs on into
    /// its own slot.
    #[test]
    fn main_data_continues_from_earlier_slots() {
        let header = small_header();
        // 144 slots minus the 4-byte header and 17 bytes of side information.
        let slot = 144 - 4 - 17;
        let mut writer = FrameWriter::default();
        writer.write(&header, &SideInfo::new(Version::Mpeg1), &[1; 100]);
        let side_info = SideInfo {
            main_data_begin: 20,
            ..SideInfo::new(Version::Mpeg1)
        };
        writer.write(&header, &side_info, &[2; 30]);
        let first = &writer.out[21..144];
        assert!(first[..100].iter().all(|&b| b == 1));
        // The last 20 of the 23 bytes the first frame left unused hold the
        // start of the second frame's main data; the 3 before them are
        // stuffing.
        assert_eq!(first[100..103], [0; 3]);
        assert!(first[103..].iter().all(|&b| b == 2));
        let second = &writer.out[144 + 21..];
        assert_eq!(second.len(), slot);
        assert!(second[..10].iter().all(|&b| b == 2));
        assert!(second[10..].iter().all(|&b| b == 0));
        // main_data_begin is the first 9 bits of the side information.
        assert_eq!(writer.out[144 + 4], 20 >> 1);
    }

    #[test]
    #[should_panic(expected = "reaches past")]
    fn main_data_cannot_begin_before_the_stream() {
        let side_info = SideInfo {
            main_data_begin: 1,
            ..SideInfo::new(Version::Mpeg1)
        };
        FrameWriter::default().write(&small_header(), &side_info, &[]);
    }

    #[test]
    #[should_panic(expected = "exceed")]
    fn rejects_main_data_larger_than_the_frame() {
        let header = Header {
            crc_protected: false,
            bitrate: Bitrate::from_kbps(Version::Mpeg1, 32).unwrap(),
            sampling_frequency: SamplingFrequency::Hz32000,
            padding: false,
            mode: Mode::SingleChannel,
            flags: Flags::default(),
            emphasis: Emphasis::None,
        };
        // 144 slots minus the 4-byte header and 17 bytes of side information.
        let main_data = [0; 144 - 4 - 17 + 1];
        write_frame(
            &header,
            &SideInfo::new(Version::Mpeg1),
            &main_data,
            &mut Vec::new(),
        );
    }
}
