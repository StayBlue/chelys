// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The MP4 file of packets, written whole once the stream has ended.

use std::fmt;
use std::io::{self, Write};

use super::{EndError, PacketError, Track};
use crate::movie::progressive::write_file;

/// Writes packets of FLAC or Opus, which any encoder made, as an MP4 file,
/// which [`finish`](Self::finish) writes whole.
///
/// Each packet is a sample of the track: a FLAC frame, or an Opus packet,
/// one per stream for a stream of several. The packets are kept in memory
/// until the file is written, its sample tables before them, as they
/// describe each packet. A [`FragmentedWriter`](super::FragmentedWriter)
/// writes a file as the packets come instead.
///
/// ```
/// use chelys_mp4::{Writer, opus};
///
/// // Mono, of a pre-skip of 312 samples, and two packets of 20 ms of
/// // CELT, 960 samples each, as an encoder would give them.
/// let config = opus::Config::new(1, 312)?;
/// let mut writer = Writer::new(config);
/// writer.push(&[0xf8, 0xff, 0xfe])?;
/// writer.push(&[0xf8, 0xff, 0xfe])?;
/// let mut mp4 = Vec::new();
/// writer.finish(Some(1000), &mut mp4)?;
/// assert_eq!(&mp4[4..12], b"ftypOpus");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Writer {
    track: Track,
    data: Vec<u8>,
    sizes: Vec<u32>,
    durations: Vec<u32>,
    /// The samples that the packets decode to.
    samples: u64,
}

/// Why a [`Writer`] could not write its file.
#[derive(Debug)]
#[non_exhaustive]
pub enum FinishError {
    /// The stream cannot end as asked: the length given is not one the
    /// packets present, or they end before their delay. See
    /// [`Writer::finish`]. Nothing was written.
    End(EndError),
    /// The output failed.
    Io(io::Error),
}

impl fmt::Display for FinishError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::End(e) => e.fmt(f),
            Self::Io(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for FinishError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::End(e) => Some(e),
            Self::Io(e) => Some(e),
        }
    }
}

impl From<io::Error> for FinishError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl Writer {
    /// A writer of the stream `track` describes: a [`flac::Config`] or an
    /// [`opus::Config`].
    ///
    /// [`flac::Config`]: crate::flac::Config
    /// [`opus::Config`]: crate::opus::Config
    pub fn new(track: impl Into<Track>) -> Self {
        Self {
            track: track.into(),
            data: Vec::new(),
            sizes: Vec::new(),
            durations: Vec::new(),
            samples: 0,
        }
    }

    /// Takes the next packet, whose duration it reads from the packet: a
    /// FLAC frame's block size, or an Opus packet's frames.
    ///
    /// # Errors
    ///
    /// [`PacketError`] if the packet is not one of the codec, as far as its
    /// duration shows. It is not taken.
    ///
    /// # Panics
    ///
    /// If the packet reaches 4 GiB.
    pub fn push(&mut self, packet: &[u8]) -> Result<(), PacketError> {
        let duration = self.track.duration(packet)?;
        self.data.extend_from_slice(packet);
        self.sizes
            .push(u32::try_from(packet.len()).expect("a packet under 4 GiB"));
        self.durations.push(duration);
        self.samples += u64::from(duration);
        Ok(())
    }

    /// The samples of each channel that the packets taken decode to, any
    /// delay included, in the track's timescale: at the sample rate for
    /// FLAC, and at 48 kHz for Opus.
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Writes the file to `out`, which need not be seekable, as the file is
    /// written from its start to its end. It presents `length` samples of
    /// each channel, in the track's timescale, if given, and otherwise every
    /// sample that the packets decode to.
    ///
    /// For Opus, the samples presented follow the pre-skip, which the file's
    /// edit list leaves out, and `length` is that of the input, the samples
    /// before the padding that the stream ends with, where the edit ends.
    /// For FLAC, which has neither,
    /// `length`, if given, is to be every sample the frames decode to, as
    /// a check of the stream: the file has no edit list.
    ///
    /// # Errors
    ///
    /// [`FinishError::End`] if the packets end before their delay, or
    /// `length` is given and is more than the
    /// packets present, or would end the presentation before the last
    /// packet, or for FLAC before the last frame's end; nothing is written.
    /// [`FinishError::Io`] if `out` fails; `out` may then hold the start of
    /// the file, which is to be discarded.
    pub fn finish<W: Write + ?Sized>(
        self,
        length: Option<u64>,
        out: &mut W,
    ) -> Result<(), FinishError> {
        let Self {
            track,
            data,
            sizes,
            durations,
            samples,
        } = self;
        let last = durations.last().copied();
        let end = track.end(length, samples, last).map_err(FinishError::End)?;
        let mut audio = track.audio_track();
        let presented = match last {
            // Spec reading: the Opus mapping has the last sample last only
            // to the input's end, "the duration of the last Opus sample of an
            // Opus bitstream is given by multiplying the number of the valid
            // samples by the value produced by dividing the value of the
            // timescale field in the Media Header Box by 48000" (clause
            // 4.3.4); but its example keeps the last sample's whole duration,
            // 18 samples of 1920 and a media of 34560, and ends the
            // presentation with the edit alone, of `segment_duration` 33600
            // from `media_time` 312 (clause 4.7). The file written whole
            // follows the example, as AAC's does: its edit gives the end. The
            // last sample's duration is what clause 4.4 offers where the edit
            // cannot give the end, "in the case producing movie fragments on
            // the fly", which the fragmented file does (see
            // `FragmentedWriter::finish`).
            //
            // Settled by decoding with ffmpeg and GStreamer (`qtdemux`,
            // `opusdec`, `audiosegmentclip`; black box): both present
            // the files of the example's form from the pre-skip to the
            // input's end, to the sample, as ffmpeg presents the Ogg Opus
            // files the packets came in, though ffmpeg n9.0.2 presents
            // nothing of an input no longer than the pre-skip; ffmpeg
            // decodes them to exactly the samples of the Ogg Opus files,
            // and GStreamer, whose libopus decodes otherwise, to within
            // 2^-11 of them. The
            // conformance tests are
            // `ffmpeg_decodes_opus_in_mp4_files_as_the_ogg_opus_files` and
            // `gstreamer_decodes_opus_in_mp4_files_as_the_ogg_opus_files`.
            Some(_) => end - track.delay(),
            // Spec reading: a stream of no packets is a track of no samples,
            // which needs no edit to present nothing; the Opus mapping's
            // edit list, which "shall be present", leaves out priming
            // samples that such a track does not have.
            None => {
                audio.delay = None;
                0
            }
        };
        write_file(&audio, &sizes, &durations, presented, &data, out)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LengthError;
    use crate::boxes::read::{FourCcs, boxes, children, find, u32_at};
    use crate::flac::tests::streaminfo;
    use crate::{flac, opus};

    /// An Opus packet of one frame of 20 ms of CELT, of `size` bytes.
    fn opus_packet(size: usize) -> Vec<u8> {
        let mut packet = vec![0xa5; size];
        packet[0] = 31 << 3;
        packet
    }

    /// A FLAC frame of `block_size` samples, uncommon or not, of `size`
    /// bytes.
    fn flac_frame(block_size: u16, size: usize) -> Vec<u8> {
        let mut frame = vec![0xff, 0xf8, 0x79, 0x18, 0x00];
        frame.extend_from_slice(&(block_size - 1).to_be_bytes());
        frame.resize(size, 0x5a);
        frame
    }

    fn stbl(file: &[u8], kind: &[u8; 4]) -> Vec<u32> {
        let path = [b"moov", b"trak", b"mdia", b"minf", b"stbl", kind];
        let payload = find(file, &path).payload;
        (4..payload.len())
            .step_by(4)
            .map(|at| u32_at(payload, at))
            .collect()
    }

    #[test]
    fn an_opus_file_leaves_out_the_pre_skip_and_the_padding() {
        let mut writer = Writer::new(opus::Config::new(2, 312).unwrap());
        for size in [10, 20, 30] {
            writer.push(&opus_packet(size)).unwrap();
        }
        assert_eq!(writer.samples(), 2880);
        let mut file = Vec::new();
        writer.finish(Some(2000), &mut file).unwrap();
        assert_eq!(boxes(&file, 0).kinds(), ["ftyp", "moov", "mdat"]);
        assert_eq!(find(&file, &[b"ftyp"]).payload, b"Opus\0\0\0\0Opusiso2");
        let mdat = find(&file, &[b"mdat"]).payload;
        assert_eq!(
            mdat,
            [opus_packet(10), opus_packet(20), opus_packet(30)].concat()
        );
        // Every packet keeps its duration: the edit ends at the input's
        // end, 2312.
        assert_eq!(stbl(&file, b"stts"), [1, 3, 960]);
        let elst = find(&file, &[b"moov", b"trak", b"edts", b"elst"]);
        assert_eq!(
            &elst.payload[4..16],
            [[0, 0, 0, 1], 2000u32.to_be_bytes(), 312u32.to_be_bytes()].concat()
        );
        let mdhd = find(&file, &[b"moov", b"trak", b"mdia", b"mdhd"]);
        assert_eq!(u32_at(mdhd.payload, 12), 48_000);
        assert_eq!(u32_at(mdhd.payload, 16), 2880);
        let sgpd = find(
            &file,
            &[b"moov", b"trak", b"mdia", b"minf", b"stbl", b"sgpd"],
        );
        assert_eq!(
            sgpd.payload,
            [
                &[1, 0, 0, 0][..],
                b"roll",
                &[0, 0, 0, 2],
                &[0, 0, 0, 1],
                &[0xff, 0xfc]
            ]
            .concat()
        );
        let stbl_box = find(&file, &[b"moov", b"trak", b"mdia", b"minf", b"stbl"]);
        assert_eq!(
            children(&stbl_box).kinds(),
            ["stsd", "stts", "stsc", "stsz", "stco", "sbgp", "sgpd"]
        );
        // Without a length, every sample after the pre-skip.
        let mut writer = Writer::new(opus::Config::new(2, 312).unwrap());
        writer.push(&opus_packet(10)).unwrap();
        let mut file = Vec::new();
        writer.finish(None, &mut file).unwrap();
        assert_eq!(stbl(&file, b"stts"), [1, 1, 960]);
        let elst = find(&file, &[b"moov", b"trak", b"edts", b"elst"]);
        assert_eq!(u32_at(elst.payload, 8), 648);
    }

    #[test]
    fn a_flac_file_presents_every_frame() {
        let config = flac::Config::from_streaminfo(&streaminfo(1000, 1000, 44_100, 2, 16)).unwrap();
        let mut writer = Writer::new(config);
        let frames = [
            flac_frame(1000, 40),
            flac_frame(1000, 40),
            flac_frame(321, 30),
        ];
        for frame in &frames {
            writer.push(frame).unwrap();
        }
        let mut file = Vec::new();
        writer.finish(Some(2321), &mut file).unwrap();
        assert_eq!(find(&file, &[b"ftyp"]).payload, b"mp42\0\0\0\0mp42isom");
        assert_eq!(find(&file, &[b"mdat"]).payload, frames.concat());
        assert_eq!(
            children(&find(&file, &[b"moov", b"trak"])).kinds(),
            ["tkhd", "mdia"]
        );
        assert_eq!(stbl(&file, b"stts"), [2, 2, 1000, 1, 321]);
        let tkhd = find(&file, &[b"moov", b"trak", b"tkhd"]);
        assert_eq!(u32_at(tkhd.payload, 20), 2321);
        let stbl_box = find(&file, &[b"moov", b"trak", b"mdia", b"minf", b"stbl"]);
        assert_eq!(
            children(&stbl_box).kinds(),
            ["stsd", "stts", "stsc", "stsz", "stco"]
        );
    }

    #[test]
    fn a_length_the_packets_do_not_present_is_refused() {
        let config = flac::Config::from_streaminfo(&streaminfo(1000, 1000, 44_100, 2, 16)).unwrap();
        let mut writer = Writer::new(config);
        writer.push(&flac_frame(1000, 40)).unwrap();
        let mut file = Vec::new();
        let error = writer.finish(Some(999), &mut file).unwrap_err();
        assert!(
            matches!(
                error,
                FinishError::End(EndError::Length(LengthError {
                    length: 999,
                    samples: 1000
                }))
            ),
            "{error}"
        );
        assert!(file.is_empty());
        let mut writer = Writer::new(opus::Config::new(1, 312).unwrap());
        writer.push(&opus_packet(10)).unwrap();
        writer.push(&opus_packet(10)).unwrap();
        // Within the first packet, so that the second is past the end.
        let error = writer.finish(Some(600), &mut file).unwrap_err();
        assert!(matches!(error, FinishError::End(_)), "{error}");
        assert!(file.is_empty());
    }

    #[test]
    fn packets_of_another_codec_are_refused() {
        let mut writer = Writer::new(opus::Config::new(1, 312).unwrap());
        assert_eq!(writer.push(&[]), Err(PacketError::OpusPacket));
        let config = flac::Config::from_streaminfo(&streaminfo(1000, 1000, 44_100, 2, 16)).unwrap();
        let mut writer = Writer::new(config);
        assert_eq!(writer.push(&opus_packet(10)), Err(PacketError::FlacFrame));
        assert_eq!(writer.samples(), 0);
    }

    #[test]
    fn no_packets_make_a_track_of_no_samples() {
        let mut file = Vec::new();
        Writer::new(opus::Config::new(1, 312).unwrap())
            .finish(None, &mut file)
            .unwrap();
        assert_eq!(
            children(&find(&file, &[b"moov", b"trak"])).kinds(),
            ["tkhd", "mdia"]
        );
        assert_eq!(stbl(&file, b"stts"), [0]);
    }
}
