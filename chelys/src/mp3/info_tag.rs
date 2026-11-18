// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The Xing/Info tag, from "Mp3 Info Tag rev 1 specifications - draft 0"
//! (here "the document"), which no ISO standard defines: a frame at the
//! start of the stream, silent to a decoder that does not know it, that
//! gives the number of frames and bytes, a table to seek by, and the samples
//! the encoder added before and after the input.
//!
//! The document lays the tag out by a hex dump of a frame at 44.1 kHz in two
//! channels, with "Xing" at 0x24, right after the side information, and the
//! fields of its extension at offsets of that frame up to 0xBF. Spec gap: it
//! has "Xing" "mostly at 0x24", and does not say where it goes in frames
//! whose side information is shorter. The tag starts after the side
//! information in every frame here, and its fields keep their places
//! relative to it. Settled by decoding with ffmpeg, mpg123 and LAME (black
//! box): all three read the frame count, and mpg123 and LAME the encoder
//! delay and padding, in frames of one and two channels at every sampling
//! frequency; the conformance test is
//! `decoders_read_the_tag_at_every_sampling_frequency`.

use super::header::{Bitrate, Emphasis, Header, Mode, SamplingFrequency};
use super::rate::Rate;
use super::side_info::SideInfo;
use super::{DELAY, StreamConfig, crc};
use crate::bitstream::BitWriter;

/// The bytes of the tag: 120 of the Xing fields, 36 of the extension.
const TAG_LEN: usize = 156;

/// The bytes of the table of contents, which a tag leaves out where its
/// frame is too short for [`TAG_LEN`] (see [`header`]).
const TABLE_LEN: usize = 100;

/// Spec reading: the document's dump has the flags `0x0000000F`, and after
/// them the frame count, the byte count, the table of contents and the
/// quality, in that order. The comment beside the dump lists the table first;
/// the dump is followed, and all four fields are sent. Settled by decoding
/// with ffmpeg, mpg123 and LAME (black box): read in this order, ffmpeg and
/// mpg123 take the frame count as sent; the conformance test is
/// `decoders_read_the_tag_at_every_sampling_frequency`.
const FLAGS: u32 = 0xF;

/// The bit of [`FLAGS`] that marks the table of contents as sent: the third
/// field of the dump, the third bit.
const TABLE_FLAG: u32 = 0x4;

/// The samples by which a decoder delays its output, that the encoder delay
/// of the extension leaves out.
///
/// Spec gap: the document has the encoder delay be "how many samples were
/// added at start", so that the frames' samples less the delay and the
/// padding are "the exact number of samples in original wav", but not
/// whether the decoder's own delay is among them. Settled by decoding with
/// mpg123 and LAME (black box): both skip 529 samples more than the delay
/// the tag gives, and so give the input exactly when it is [`DELAY`] less
/// 529; the conformance test is
/// `decoders_read_the_tag_at_every_sampling_frequency`.
const DECODER_DELAY: usize = 529;

/// The frame that holds the tag, at a bitrate whose frames hold it: the
/// stream's own if it is constant, the one of the table nearest a mean rate,
/// and otherwise, or where those do not hold it, the lowest that does. At a
/// constant bitrate whose frames are too short for the whole tag, the tag
/// leaves out its table of contents, and its frame keeps the stream's
/// bitrate if it holds the rest. It
/// has no `crc_check` word, no padding, and in joint stereo neither MS nor
/// intensity stereo, the stream's [`Flags`](super::header::Flags) (so that a stream marked as
/// copyright protected has no frame unmarked), and its side information
/// is all zero: a decoder that
/// does not know the tag decodes it as a granule of silence in each
/// channel.
///
/// Spec gap: the document suggests the frame be "housed in a CBR file using
/// the same framesize as the mp3 itself", and that a stream that is not at
/// a constant bitrate could have one "as close as possible to the average
/// bitrate of the file", for decoders "guessing file length based on size
/// first frame"; it says nothing of the rest of its header. A variable
/// bitrate's average is known only once the stream has ended, after the
/// frame has been written, so its tag takes the lowest bitrate that holds
/// it.
///
/// Spec gap: the document lays the tag out with all four Xing fields, and
/// the extension at fixed offsets after them, but does not say what the
/// bits of the flags mean, or where the fields after one left out go. At a
/// constant bitrate whose frames are too short for the whole tag, the tag
/// leaves out the table, with its bit of the flags cleared, and the extension
/// follows the quality, so that the frame keeps the stream's bitrate as the
/// document suggests; at a constant bitrate a decoder can find where to seek
/// from the bitrate alone. A mean or variable rate keeps the table, whose
/// offsets a decoder cannot work out, and its frame is at a higher bitrate
/// if need be. A constant bitrate whose frames do not hold even the tag
/// without its table keeps it whole at a higher bitrate too, as the bitrate
/// may change from frame to frame in Layer III (clause 2.4.2.3 of ISO/IEC
/// 11172-3). Settled by decoding with ffmpeg, mpg123 and LAME (black box):
/// mpg123 and LAME give the input exactly, and follow a changed encoder
/// delay where the extension follows the quality; ffmpeg skips the tag's
/// frame, gives the duration of the frames after it, and applies the delays
/// given "LAME" for the encoder there; the conformance test is
/// `decoders_read_the_tag_without_its_table`.
///
/// Spec gap: the document does not mention free format, whose bitrate does
/// not change ("in free format, fixed bitrate is required", clause 2.4.2.3
/// of ISO/IEC 11172-3), and whose frames' length a decoder learns from the
/// distance between their headers. The tag's frame is at the stream's
/// bitrate, and a stream whose frames do not hold the tag, even without its
/// table, has none: `None`.
/// Settled by decoding with mpg123 and LAME (black box): both find the
/// frames after the tag's, and read the tag, giving the input exactly; the
/// conformance test is `mpg123_and_lame_decode_free_format`. ffmpeg
/// decodes no free format.
pub fn header(config: &StreamConfig) -> Option<Header> {
    let fs = config.sampling_frequency;
    let channels = config.mode.channels();
    let side_info = SideInfo::len_bytes(fs.version(), channels);
    let room = |bitrate: &Bitrate| bitrate.frame_len(fs, false).saturating_sub(4 + side_info);
    let holds = |bitrate: &Bitrate| room(bitrate) >= TAG_LEN;
    let preferred = match config.rate {
        Rate::Constant(bitrate) => Some(bitrate).filter(|b| room(b) >= TAG_LEN - TABLE_LEN),
        Rate::Mean(mean) => Bitrate::all(fs.version())
            .min_by_key(|b| b.bits_per_second().abs_diff(mean.bits_per_second()))
            .filter(holds),
        Rate::Variable(_) => None,
    };
    let bitrate = match preferred {
        Some(bitrate) => bitrate,
        None if config.rate.lowest().is_free() => return None,
        None => Bitrate::all(fs.version())
            .find(holds)
            .expect("the highest bitrate's frames hold the tag"),
    };
    let mode = match config.mode {
        Mode::JointStereo { .. } => Mode::JointStereo {
            intensity_stereo: false,
            ms_stereo: false,
        },
        mode => mode,
    };
    Some(Header {
        crc_protected: false,
        bitrate,
        sampling_frequency: fs,
        padding: false,
        mode,
        flags: config.flags,
        emphasis: Emphasis::None,
    })
}

/// The tag's frame with nothing in it but its header: what the stream
/// starts with until the tag is known.
pub fn placeholder(header: &Header) -> Vec<u8> {
    let mut w = BitWriter::new();
    header.write(&mut w);
    let mut frame = w.into_bytes();
    frame.resize(header.frame_len(), 0);
    frame
}

/// What the tag gives of the stream after it.
pub struct Stream<'a> {
    pub config: &'a StreamConfig,
    /// The length in bytes of each frame of the stream after the tag's.
    pub frame_lens: &'a [usize],
    /// The input samples of each channel.
    pub samples: usize,
    /// The CRC of the bytes of the stream after the tag's frame (see
    /// [`music_crc`]).
    pub music_crc: u16,
}

/// The CRC of the extension's "MusicCRC" and "CRC-16 of Info Tag" fields,
/// of `bytes` from the state `register`, starting with [`MUSIC_CRC_START`].
///
/// Spec gap: the document says "CRC-16" with "CRCInitValue := $0000", and
/// that "CRC-16 routines are present in both mp3 encoder and decoders", but
/// gives no generator polynomial. The one an MP3 encoder has is that of the
/// `crc_check` word (clause 2.4.3.1), so that is used, from the all-zero
/// state. No decoder settled it: ffmpeg, mpg123 and LAME decoded the same
/// with either CRC wrong.
pub fn music_crc(register: u16, bytes: &[u8]) -> u16 {
    crc::update(register, bytes.iter().copied())
}

/// The state [`music_crc`] starts from.
pub const MUSIC_CRC_START: u16 = 0;

/// The tag's frame, under `header` (see [`header`]), for `stream`.
///
/// # Panics
///
/// If the stream ends too soon after the input for the delay and padding
/// to be what [`DELAY`] makes them.
pub fn frame(header: &Header, stream: &Stream) -> Vec<u8> {
    let config = stream.config;
    let fs = config.sampling_frequency;
    let mut frame = placeholder(header);
    let tag_len = frame.len();
    let frames = stream.frame_lens.len();
    let bytes = tag_len + stream.frame_lens.iter().sum::<usize>();
    let start = 4 + SideInfo::len_bytes(fs.version(), config.mode.channels());
    // Only at a constant bitrate is the frame too short for the table (see
    // `header`).
    let with_table = tag_len - start >= TAG_LEN;
    debug_assert!(with_table || matches!(config.rate, Rate::Constant(_)));

    let mut tag = Vec::with_capacity(TAG_LEN);
    // Spec reading: "Xing" for streams whose bitrate varies and "Info" for
    // those at one bitrate, as the document has LAME do ("LAME VBR & ABR:
    // "Xing"", "LAME CBR: "Info"").
    tag.extend_from_slice(match config.rate {
        Rate::Constant(_) => b"Info",
        Rate::Mean(_) | Rate::Variable(_) => b"Xing",
    });
    let flags = match with_table {
        true => FLAGS,
        false => FLAGS & !TABLE_FLAG,
    };
    tag.extend_from_slice(&flags.to_be_bytes());
    // Spec reading: the frames after the tag's, and the bytes of the
    // stream with it. The document's dump has the fields big-endian.
    // Settled by decoding with ffmpeg and mpg123 (black box): with the
    // tag's frame left out of the count, ffmpeg gives the duration of the
    // frames after it and mpg123 the input's samples exactly; the
    // conformance test is `decoders_read_the_tag_at_every_sampling_frequency`.
    tag.extend_from_slice(&(frames as u32).to_be_bytes());
    tag.extend_from_slice(&length(bytes).to_be_bytes());
    if with_table {
        tag.extend(table_of_contents(tag_len, stream.frame_lens, bytes));
    }
    // Spec gap: the document gives the quality as "0=best 100=worst" beside
    // the dump, and as LAME's own scale, with 100 the best, in the
    // extension, where it is one byte, "byte $9B", the last of the four the
    // dump has. The encoder has no such scale, and sends 0, which neither
    // scale reads as unknown.
    tag.extend_from_slice(&0u32.to_be_bytes());

    // The extension, from 0x9C of the dump, or 100 bytes before without the
    // table.
    tag.extend_from_slice(&encoder_version());
    // Revision 1, and the "VBR method": constant bitrate, "restricted VBR
    // targetting a given average bitrate (ABR)", or "full VBR method1".
    // Spec reading: a mean rate targets an average bitrate, and the variable
    // rate is the encoder's only full VBR method.
    let method = match config.rate {
        Rate::Constant(_) => 1,
        Rate::Mean(_) => 2,
        Rate::Variable(_) => 3,
    };
    tag.push(0x10 | method);
    // The lowpass filter ("00h => unknown"); the peak amplitude and the
    // two ReplayGain fields, not set; the encoding flags and ATH type, which
    // are LAME's own.
    tag.extend_from_slice(&[0; 1 + 4 + 2 + 2 + 1]);
    // The bitrate: the target of a mean rate, the lowest of a variable one,
    // or the constant one, in kbit/s, "255 if bigger".
    let kbps = match config.rate {
        Rate::Constant(bitrate) => bitrate.bits_per_second() / 1000,
        Rate::Mean(mean) => mean.bits_per_second().div_ceil(1000),
        Rate::Variable(variable) => variable.lowest.bits_per_second() / 1000,
    };
    tag.push(kbps.min(255) as u8);
    tag.extend_from_slice(&delays(stream));
    tag.push(misc(config));
    // The MP3 gain, "set to (00)h by default", and the preset and surround
    // information, unknown.
    tag.extend_from_slice(&[0; 1 + 2]);
    // "MusicLength": "The first byte it counts is the first byte of this
    // LAME Tag and the last byte it counts is the last byte of the last mp3
    // frame containing music."
    tag.extend_from_slice(&length(bytes).to_be_bytes());
    tag.extend_from_slice(&stream.music_crc.to_be_bytes());

    frame[start..start + tag.len()].copy_from_slice(&tag);
    // Spec reading: the CRC of the tag covers "the first 190 bytes
    // ($00-$BD) of the Info header frame", everything before the CRC in the
    // dump's frame; so it covers the frame from its header up to the CRC.
    let end = start + tag.len();
    let tag_crc = music_crc(MUSIC_CRC_START, &frame[..end]);
    frame[end..end + 2].copy_from_slice(&tag_crc.to_be_bytes());
    frame
}

/// A length in bytes as the tag sends it: "Musiclength not set / unknown /
/// larger than 4G" is 0. Spec gap: the document says so of the extension's
/// length alone, and nothing of the Xing fields' byte count; it is sent the
/// same way.
fn length(bytes: usize) -> u32 {
    u32::try_from(bytes).unwrap_or(0)
}

/// The table of contents: for each percent of the stream's duration, the
/// share of its bytes before the frame that starts it, in 256ths.
///
/// Spec gap: the document names the table but does not say what it holds.
/// Each entry is the byte offset, from the start of the tag's frame, of the
/// frame after it at which that percent of the frames begins, over the
/// stream's bytes with the tag's frame. Settled by decoding with ffmpeg
/// (black box): seeking by the table (its `usetoc` option), ffmpeg lands
/// within a frame of where it was asked to, and with the offsets counted
/// from the first frame after the tag's, some 2000 samples early in a short
/// stream at 8 kHz; the conformance test is `ffmpeg_seeks_by_the_table`.
/// (The document's dump, whose tag's frame is 417 of 12481 bytes, has 0 for
/// the first entry, where this reading has 8.)
fn table_of_contents(tag_len: usize, frame_lens: &[usize], bytes: usize) -> Vec<u8> {
    let mut offsets = Vec::with_capacity(frame_lens.len());
    let mut offset = tag_len;
    for len in frame_lens {
        offsets.push(offset);
        offset += len;
    }
    (0..100)
        .map(|percent| {
            let frame = (percent * frame_lens.len() / 100).min(frame_lens.len().saturating_sub(1));
            let offset = offsets.get(frame).copied().unwrap_or(tag_len);
            (offset * 256 / bytes).min(255) as u8
        })
        .collect()
}

/// The encoder's name and version, in the 9 bytes the extension has for
/// them ("LAME3.90a" in the document's example), cut short or padded with
/// spaces.
///
/// Spec gap: the document leaves the string to the encoder, and says
/// nothing of a shorter one. (Its heading for the field, "bytes $9A-$A4",
/// is 11 bytes; its "9 characters" and its dump put it at 0x9C to 0xA4.)
/// The encoder gives its own name. Decoding with ffmpeg, mpg123 and LAME
/// (black box) showed that ffmpeg applies the encoder delay and padding
/// only when the string starts with "LAME" or "Lav", and so not to these
/// streams, which it decodes with the delay and the padding in, while
/// mpg123 and LAME apply them whatever the string; the conformance tests
/// are `ffmpeg_applies_the_delays_for_some_encoders_only` and
/// `decoders_read_the_tag_at_every_sampling_frequency`.
fn encoder_version() -> [u8; 9] {
    let mut version = [b' '; 9];
    let name = concat!("chelys", env!("CARGO_PKG_VERSION"));
    for (byte, &b) in version.iter_mut().zip(name.as_bytes()) {
        *byte = b;
    }
    version
}

/// The encoder delay and the padding at the end, 12 bits each. The padding
/// is every sample after the input, more than "to complete the last frame"
/// as the document has it, as the stream ends at least [`DELAY`] past the
/// input (see [`DECODER_DELAY`]).
fn delays(stream: &Stream) -> [u8; 3] {
    let fs = stream.config.sampling_frequency;
    let decoded = stream.frame_lens.len() * fs.version().samples_per_frame();
    let delay = DELAY - DECODER_DELAY;
    let padding = decoded
        .checked_sub(stream.samples + delay)
        .filter(|padding| (DECODER_DELAY..1 << 12).contains(padding))
        .expect("the stream covers the input and the decoder's delay after it");
    let both = (delay << 12 | padding) as u32;
    let [_, high, middle, low] = both.to_be_bytes();
    [high, middle, low]
}

/// The extension's "Misc" byte: no noise shaping, the stereo mode, no
/// "unwise settings", and the source's sampling frequency, which is the
/// stream's.
fn misc(config: &StreamConfig) -> u8 {
    let stereo = match config.mode {
        Mode::SingleChannel => 0b000,
        Mode::Stereo => 0b001,
        Mode::DualChannel => 0b010,
        Mode::JointStereo { .. } => 0b011,
    };
    let source = match config.sampling_frequency {
        SamplingFrequency::Hz44100 => 0b01,
        SamplingFrequency::Hz48000 => 0b10,
        _ => 0b00,
    };
    source << 6 | stereo << 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mp3::header::Flags;
    use crate::mp3::rate::VariableRate;

    fn config(fs: SamplingFrequency, rate: Rate, mode: Mode) -> StreamConfig {
        StreamConfig {
            rate,
            sampling_frequency: fs,
            mode,
            crc_protected: true,
            flags: Flags::default(),
        }
    }

    fn kbps(fs: SamplingFrequency, kbps: u32) -> Bitrate {
        Bitrate::from_kbps(fs.version(), kbps).unwrap()
    }

    /// At 44.1 kHz in two channels the tag's fields fall where the
    /// document's dump has them: "Xing" at 0x24, the table at 0x34, the
    /// extension at 0x9C, the delays at 0xB1 and the tag's CRC at 0xBE.
    #[test]
    fn fields_fall_where_the_dump_has_them() {
        let fs = SamplingFrequency::Hz44100;
        let rate = Rate::Variable(VariableRate {
            lowest: kbps(fs, 32),
            highest: kbps(fs, 320),
            noise_offset_db: 0.0,
        });
        let config = config(fs, rate, Mode::Stereo);
        let header = header(&config).unwrap();
        // 192 bytes of tag need 64 kbit/s: 208-byte frames.
        assert_eq!(header.bitrate, kbps(fs, 64));
        assert!(!header.crc_protected);
        let frame_lens = vec![417; 40];
        let samples = 40 * 1152 - 528 - 1000;
        let stream = Stream {
            config: &config,
            frame_lens: &frame_lens,
            samples,
            music_crc: 0x1234,
        };
        let frame = frame(&header, &stream);
        assert_eq!(frame.len(), 208);
        assert_eq!(frame[..4], [0xFF, 0xFB, 0x50, 0x00]);
        assert!(frame[4..0x24].iter().all(|&b| b == 0));
        assert_eq!(&frame[0x24..0x28], b"Xing");
        assert_eq!(frame[0x28..0x2C], [0, 0, 0, 0x0F]);
        assert_eq!(frame[0x2C..0x30], 40u32.to_be_bytes());
        assert_eq!(frame[0x30..0x34], (208 + 40 * 417u32).to_be_bytes());
        // The table starts at the tag's frame and rises by a 40th of the
        // rest each 2.5 percent.
        assert_eq!(frame[0x34], (208 * 256 / (208 + 40 * 417)) as u8);
        assert!(frame[0x34..0x98].windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(frame[0x98..0x9C], [0; 4]);
        assert_eq!(&frame[0x9C..0xA2], b"chelys");
        assert_eq!(frame[0xA5], 0x13);
        assert_eq!(frame[0xB0], 32);
        // 528 and 1000, 12 bits each.
        assert_eq!(frame[0xB1..0xB4], [0x21, 0x03, 0xE8]);
        assert_eq!(frame[0xB4], 0b0100_0100);
        assert_eq!(frame[0xB8..0xBC], (208 + 40 * 417u32).to_be_bytes());
        assert_eq!(frame[0xBC..0xBE], [0x12, 0x34]);
        let tag_crc = music_crc(MUSIC_CRC_START, &frame[..0xBE]);
        assert_eq!(frame[0xBE..0xC0], tag_crc.to_be_bytes());
        assert!(frame[0xC0..].iter().all(|&b| b == 0));
    }

    /// A constant bitrate keeps its frames for the tag where they hold it,
    /// and says "Info"; so does a mean rate's nearest bitrate.
    #[test]
    fn a_constant_bitrate_keeps_its_frame_size_where_it_can() {
        let fs = SamplingFrequency::Hz48000;
        let mono = config(fs, Rate::Constant(kbps(fs, 64)), Mode::SingleChannel);
        // 4 + 17 + 156 = 177 bytes fit in 192.
        assert_eq!(header(&mono).unwrap().bitrate, kbps(fs, 64));
        let stream = Stream {
            config: &mono,
            frame_lens: &[192; 10],
            samples: 10 * 1152 - 528 - 600,
            music_crc: 0,
        };
        let frame = frame(&header(&mono).unwrap(), &stream);
        assert_eq!(&frame[4 + 17..4 + 21], b"Info");
        assert_eq!(frame[4 + 21..4 + 25], [0, 0, 0, 0x0F]);
        assert_eq!(frame[4 + 17 + 129], 0x11);
        let mean = crate::mp3::rate::MeanRate::new(fs.version(), 150_000).unwrap();
        let near = config(fs, Rate::Mean(mean), Mode::Stereo);
        assert_eq!(header(&near).unwrap().bitrate, kbps(fs, 160));
    }

    /// A constant bitrate whose frames are too short for the whole tag
    /// keeps them for the tag without its table, where they hold that, and
    /// the extension follows the quality. A mean rate keeps the table, at a
    /// higher bitrate.
    #[test]
    fn a_constant_bitrate_too_short_for_the_table_leaves_it_out() {
        let fs = SamplingFrequency::Hz48000;
        // 192 bytes do not fit in 168; 4 + 32 + 56 = 92 do.
        let stereo = config(fs, Rate::Constant(kbps(fs, 56)), Mode::Stereo);
        let tag_header = header(&stereo).unwrap();
        assert_eq!(tag_header.bitrate, kbps(fs, 56));
        let frame_lens = [168; 10];
        let stream = Stream {
            config: &stereo,
            frame_lens: &frame_lens,
            samples: 10 * 1152 - 528 - 600,
            music_crc: 0x1234,
        };
        let frame = frame(&tag_header, &stream);
        assert_eq!(frame.len(), 168);
        assert_eq!(&frame[0x24..0x28], b"Info");
        assert_eq!(frame[0x28..0x2C], [0, 0, 0, 0x0B]);
        assert_eq!(frame[0x2C..0x30], 10u32.to_be_bytes());
        assert_eq!(frame[0x30..0x34], (11 * 168u32).to_be_bytes());
        assert_eq!(frame[0x34..0x38], [0; 4]);
        assert_eq!(&frame[0x38..0x3E], b"chelys");
        assert_eq!(frame[0x41], 0x11);
        assert_eq!(frame[0x4C], 56);
        assert_eq!(frame[0x4D..0x50], [0x21, 0x02, 0x58]);
        assert_eq!(frame[0x54..0x58], (11 * 168u32).to_be_bytes());
        assert_eq!(frame[0x58..0x5A], [0x12, 0x34]);
        let tag_crc = music_crc(MUSIC_CRC_START, &frame[..0x5A]);
        assert_eq!(frame[0x5A..0x5C], tag_crc.to_be_bytes());
        assert!(frame[0x5C..].iter().all(|&b| b == 0));

        let mean = crate::mp3::rate::MeanRate::new(fs.version(), 50_000).unwrap();
        let mean = config(fs, Rate::Mean(mean), Mode::Stereo);
        assert_eq!(header(&mean).unwrap().bitrate, kbps(fs, 64));

        // At 8 kHz the tag without its table takes 4 + 9 + 56 = 69 bytes in
        // one channel, which fit in 72, and 77 in two, which do not: the
        // whole tag's 177 bytes need 24 kbit/s's 216.
        let fs = SamplingFrequency::Hz8000;
        let mono = config(fs, Rate::Constant(kbps(fs, 8)), Mode::SingleChannel);
        assert_eq!(header(&mono).unwrap().bitrate, kbps(fs, 8));
        let stereo = config(fs, Rate::Constant(kbps(fs, 8)), Mode::Stereo);
        assert_eq!(header(&stereo).unwrap().bitrate, kbps(fs, 24));

        // In free format the frame keeps the stream's bitrate or has no tag.
        let fs = SamplingFrequency::Hz48000;
        let free = |bps| Rate::Constant(Bitrate::free(fs.version(), bps).unwrap());
        let short = config(fs, free(50_000), Mode::Stereo);
        assert_eq!(header(&short).unwrap().bitrate.bits_per_second(), 50_000);
        assert!(header(&config(fs, free(20_000), Mode::Stereo)).is_none());
    }
}
