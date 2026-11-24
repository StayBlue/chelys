// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The fragmented MP4 file of packets, written as the packets come.

use std::fmt;
use std::num::NonZeroU32;

use super::{EndError, PacketError, Track};
use crate::fragments::{FragmentOptions, LengthError};
use crate::movie::fragmented::{FragmentSink, Fragmenter, init_segment};

/// Writes packets of FLAC, Opus, AAC-LC or ALAC, which any encoder made, as a
/// fragmented MP4 file (ISO/IEC 14496-12, clause 8.8), written as the
/// packets come: its [`init_segment`](Self::init_segment), then fragments
/// of about a second each, which a [`FragmentSink`] takes, so that the
/// file can be sent or played as the stream is encoded, as over HLS.
///
/// Where the input's length is declared in advance
/// ([`FragmentOptions::length`]), the file gives a player its exact length,
/// as a [`Writer`](super::Writer)'s does. Otherwise an Opus file's edit
/// list leaves out the pre-skip, or an AAC file's the delay given, and its
/// media ends with the packets, but it does not give where the
/// presentation ends, so that a player may present the padding that the
/// last packet decodes to. A FLAC or ALAC stream has no padding, nor has
/// an AAC stream whose delay is not given, and needs no length to end where
/// it does.
///
/// ffmpeg n9.0.2 (black box) presents the padding of a fragmented Opus file
/// whether its length is declared or not, as it does of AAC's. GStreamer
/// 1.28.9 (`qtdemux`, `opusdec`, `audiosegmentclip`; black box)
/// presents the input alone where the length is declared, and the padding
/// too where it is not. See the conformance test
/// `fragmented_opus_in_mp4_files_decode_as_the_files_written_whole`.
///
/// Of a stream that ends with padding, Opus, or AAC whose delay is given,
/// the latest packet is held until the next comes, or the stream ends, so
/// that the last can last only to the end of the input: each fragment is
/// written a packet later than its last packet is pushed.
///
/// ```
/// use chelys_mp4::{FragmentOptions, FragmentedWriter, flac};
///
/// let streaminfo = [
///     0x10, 0x00, 0x10, 0x00, 0, 0, 0, 0, 0, 0, 0x0a, 0xc4, 0x42, 0xf0,
///     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
/// ];
/// let config = flac::Config::from_streaminfo(&streaminfo)?;
/// let mut writer = FragmentedWriter::new(config, FragmentOptions::default());
/// let mut mp4 = writer.init_segment().to_vec();
/// // A frame of 4096 samples, its header and the rest of its bytes.
/// let frame = [&[0xff, 0xf8, 0xc9, 0x18, 0x00, 0x00][..], &[0; 100]].concat();
/// for _ in 0..20 {
///     writer.push(&frame, &mut mp4)?;
/// }
/// writer.finish(&mut mp4)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct FragmentedWriter {
    track: Track,
    fragmenter: Fragmenter,
    init_segment: Vec<u8>,
    length: Option<u64>,
    /// The latest packet and its duration, held until the next, of a stream
    /// that ends with padding.
    held: Option<(Vec<u8>, u32)>,
    /// The duration of the last packet taken.
    last: Option<u32>,
    /// The samples that the packets taken decode to.
    samples: u64,
}

/// Why a [`FragmentedWriter`] refused a packet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PushError {
    /// The packet is not one of the codec.
    Packet(PacketError),
    /// The packet would take the stream past the length declared: for
    /// FLAC, ALAC, and AAC whose delay is not given, its samples; for Opus, and
    /// AAC whose delay is given, it would start after the input's end.
    Length(LengthError),
}

impl fmt::Display for PushError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Packet(e) => e.fmt(f),
            Self::Length(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for PushError {}

impl From<PacketError> for PushError {
    fn from(e: PacketError) -> Self {
        Self::Packet(e)
    }
}

impl FragmentedWriter {
    /// A writer of the stream `track` describes, a [`flac::Config`], an
    /// [`opus::Config`], an [`aac::Config`] or an [`alac::Config`], into
    /// the file `options` asks for. The length and the fragments' duration
    /// are in the track's timescale: at the sample rate for FLAC, AAC and
    /// ALAC, and at 48 kHz for Opus, the length after the pre-skip or the
    /// delay.
    ///
    /// [`flac::Config`]: crate::flac::Config
    /// [`opus::Config`]: crate::opus::Config
    /// [`aac::Config`]: crate::aac::Config
    /// [`alac::Config`]: crate::alac::Config
    pub fn new(track: impl Into<Track>, options: FragmentOptions) -> Self {
        let track = track.into();
        let audio = track.audio_track(None);
        let one_second = NonZeroU32::new(audio.timescale).expect("a timescale");
        let fragment_samples = options.fragment_samples.unwrap_or(one_second);
        Self {
            fragmenter: Fragmenter::new(&audio, fragment_samples),
            init_segment: init_segment(&audio, options.length),
            track,
            length: options.length,
            held: None,
            last: None,
            samples: 0,
        }
    }

    /// The init segment: the file's `FileTypeBox` and `MovieBox`, which
    /// come before every fragment. It is fixed when the writer is made, and
    /// can be given again, as to each reader that starts at a later
    /// fragment.
    pub fn init_segment(&self) -> &[u8] {
        &self.init_segment
    }

    /// The samples of each channel that the packets taken decode to, any
    /// delay included, in the track's timescale.
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Takes the next packet, as [`Writer::push`](super::Writer::push)
    /// does, and gives `out` the fragment it completes, if it completes
    /// one.
    ///
    /// # Errors
    ///
    /// [`PushError`] if the packet is not one of the codec, or would take
    /// the stream past the length declared. It is not taken, and `out` is
    /// given nothing.
    pub fn push<O: FragmentSink + ?Sized>(
        &mut self,
        packet: &[u8],
        out: &mut O,
    ) -> Result<(), PushError> {
        self.take(packet, None, out)
    }

    /// Takes the next packet of `duration` samples, as
    /// [`Writer::push_with_duration`](super::Writer::push_with_duration)
    /// does, and gives `out` the fragment it completes, if it completes
    /// one.
    ///
    /// # Errors
    ///
    /// [`PushError`] as for [`push`](Self::push), and if `duration` is not
    /// one the packet can have (see
    /// [`Writer::push_with_duration`](super::Writer::push_with_duration)).
    pub fn push_with_duration<O: FragmentSink + ?Sized>(
        &mut self,
        packet: &[u8],
        duration: u32,
        out: &mut O,
    ) -> Result<(), PushError> {
        self.take(packet, Some(duration), out)
    }

    fn take<O: FragmentSink + ?Sized>(
        &mut self,
        packet: &[u8],
        duration: Option<u32>,
        out: &mut O,
    ) -> Result<(), PushError> {
        let duration = self.track.duration(packet, duration)?;
        if let Some(length) = self.length {
            let delay = self.track.delay();
            let end = delay.saturating_add(length);
            let past = if self.track.ends_with_padding() {
                self.samples >= end
            } else {
                self.samples + u64::from(duration) > end
            };
            if past {
                let samples = (self.samples + u64::from(duration)).saturating_sub(delay);
                return Err(PushError::Length(LengthError { length, samples }));
            }
        }
        let ready = if self.track.ends_with_padding() {
            self.held.replace((packet.to_vec(), duration))
        } else {
            Some((packet.to_vec(), duration))
        };
        if let Some((packet, duration)) = ready {
            if let Some(fragment) = self.fragmenter.push_frame(&packet, duration) {
                out.fragment(&fragment.bytes, fragment.info);
            }
        }
        self.samples += u64::from(duration);
        self.last = Some(duration);
        Ok(())
    }

    /// Ends the fragment being gathered now, and gives it to `out`, if it
    /// has a packet: a fragment always has one, so that this does nothing
    /// where no packet has been written since the last. The next fragment
    /// ends where it would have, at the next multiple of the fragments'
    /// duration past this one's end. Of a stream that ends with padding,
    /// the latest packet, which is held, is not in the fragment.
    pub fn end_fragment<O: FragmentSink + ?Sized>(&mut self, out: &mut O) {
        if let Some(fragment) = self.fragmenter.flush() {
            out.fragment(&fragment.bytes, fragment.info);
        }
    }

    /// Ends the stream, and gives `out` the fragments left, the last ending
    /// with the last packet, which lasts only to the end of the input where
    /// the length was declared.
    ///
    /// # Errors
    ///
    /// [`EndError::Length`] if the length declared, which the init segment
    /// already gives, is not the one the packets present, and
    /// [`EndError::ShorterThanDelay`] if the packets end before the delay
    /// that the init segment's edit leaves out: the file would not present
    /// what it says, and is to be discarded. `out` is then given nothing.
    pub fn finish<O: FragmentSink + ?Sized>(self, out: &mut O) -> Result<(), EndError> {
        let Self {
            track,
            mut fragmenter,
            length,
            held,
            last,
            samples,
            ..
        } = self;
        let end = track.end(length, samples, last)?;
        if let Some((packet, duration)) = held {
            // The last packet lasts to the input's end, as the Opus mapping
            // has it (clause 4.3.4), and offers "in the case producing movie
            // fragments on the fly", where the edit list cannot give the end
            // (clause 4.4); see `Writer::finish`. An AAC stream's last
            // packet lasts as long as the last frame of chelys's own
            // fragmented AAC files. Without a length, the end is that of the
            // last packet, and its duration is its own.
            let trimmed = end - (samples - u64::from(duration));
            let trimmed = u32::try_from(trimmed).expect("within the last packet");
            let fragment = fragmenter.finish(&packet, trimmed);
            out.fragment(&fragment.bytes, fragment.info);
        } else if let Some(fragment) = fragmenter.flush() {
            out.fragment(&fragment.bytes, fragment.info);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boxes::read::{FourCcs, boxes, children, find, u32_at};
    use crate::flac::tests::streaminfo;
    use crate::{FragmentInfo, alac, flac, opus};

    /// An Opus packet of one frame of 20 ms of CELT, of `size` bytes.
    fn opus_packet(size: usize) -> Vec<u8> {
        let mut packet = vec![0xa5; size];
        packet[0] = 31 << 3;
        packet
    }

    /// A FLAC frame of `block_size` samples, of `size` bytes.
    fn flac_frame(block_size: u16, size: usize) -> Vec<u8> {
        let mut frame = vec![0xff, 0xf8, 0x79, 0x18, 0x00];
        frame.extend_from_slice(&(block_size - 1).to_be_bytes());
        frame.resize(size, 0x5a);
        frame
    }

    /// The file and the fragments' infos of `packets` given in turn.
    fn write(writer: FragmentedWriter, packets: &[Vec<u8>]) -> (Vec<u8>, Vec<FragmentInfo>) {
        let mut writer = writer;
        let mut file = writer.init_segment().to_vec();
        let mut infos = Vec::new();
        let mut out = |bytes: &[u8], info: FragmentInfo| {
            file.extend_from_slice(bytes);
            infos.push(info);
        };
        for packet in packets {
            writer.push(packet, &mut out).unwrap();
        }
        writer.finish(&mut out).unwrap();
        (file, infos)
    }

    fn media_data(file: &[u8]) -> Vec<u8> {
        (boxes(file, 0).iter())
            .filter(|b| &b.kind == b"mdat")
            .flat_map(|b| b.payload.to_vec())
            .collect()
    }

    #[test]
    fn an_opus_file_s_last_packet_lasts_to_the_input_s_end() {
        let config = opus::Config::new(2, 312).unwrap();
        let options = FragmentOptions::default().length(100 * 960 - 500);
        let packets: Vec<Vec<u8>> = (0..100).map(|i| opus_packet(10 + i % 7)).collect();
        let (file, infos) = write(FragmentedWriter::new(config, options), &packets);
        assert_eq!(media_data(&file), packets.concat());
        assert_eq!(find(&file, &[b"ftyp"]).payload, b"Opus\0\0\0\0Opusiso2");
        // Fragments of a second, 50 packets of 20 ms.
        let frames: Vec<u32> = infos.iter().map(|info| info.frames).collect();
        assert_eq!(frames, [50, 50]);
        assert_eq!(infos[1].duration, 50 * 960 - 500 + 312);
        let elst = find(&file, &[b"moov", b"trak", b"edts", b"elst"]);
        assert_eq!(u32_at(elst.payload, 8), 100 * 960 - 500);
        assert_eq!(u32_at(elst.payload, 12), 312);
        let trex = find(&file, &[b"moov", b"mvex", b"trex"]);
        assert_eq!(u32_at(trex.payload, 12), 960);
        let init_stbl = find(&file, &[b"moov", b"trak", b"mdia", b"minf", b"stbl"]);
        assert_eq!(
            children(&init_stbl).kinds(),
            ["stsd", "stts", "stsc", "stsz", "stco", "sgpd"]
        );
    }

    #[test]
    fn a_flac_file_has_no_edit_and_no_pre_roll() {
        let config = flac::Config::from_streaminfo(&streaminfo(1000, 1000, 8_000, 1, 16)).unwrap();
        let mut packets: Vec<Vec<u8>> = (0..20).map(|_| flac_frame(1000, 30)).collect();
        packets.push(flac_frame(17, 10));
        let options = FragmentOptions::default().length(20_017);
        let (file, infos) = write(FragmentedWriter::new(config, options), &packets);
        assert_eq!(media_data(&file), packets.concat());
        assert_eq!(
            children(&find(&file, &[b"moov", b"trak"])).kinds(),
            ["tkhd", "mdia"]
        );
        assert_eq!(
            children(&find(&file, &[b"moov", b"mvex"])).kinds(),
            ["mehd", "trex"]
        );
        let frames: Vec<u32> = infos.iter().map(|info| info.frames).collect();
        assert_eq!(frames, [8, 8, 5]);
        assert_eq!(infos[2].duration, 4017);
        let moofs: Vec<_> = boxes(&file, 0)
            .into_iter()
            .filter(|b| &b.kind == b"moof")
            .collect();
        let traf = |i: usize| children(&children(&moofs[i])[1]);
        assert_eq!(traf(0).kinds(), ["tfhd", "tfdt", "trun"]);
        // The last fragment gives each frame's duration.
        let trun = &traf(2)[2];
        assert_eq!(u32_at(trun.payload, 0), 0x00_0301);
    }

    #[test]
    fn packets_past_the_declared_length_are_refused() {
        let config = flac::Config::from_streaminfo(&streaminfo(1000, 1000, 8_000, 1, 16)).unwrap();
        let options = FragmentOptions::default().length(1500);
        let mut writer = FragmentedWriter::new(config, options);
        let mut file = Vec::new();
        writer.push(&flac_frame(1000, 30), &mut file).unwrap();
        assert_eq!(
            writer.push(&flac_frame(1000, 30), &mut file),
            Err(PushError::Length(LengthError {
                length: 1500,
                samples: 2000
            }))
        );
        assert_eq!(
            writer.finish(&mut file),
            Err(EndError::Length(LengthError {
                length: 1500,
                samples: 1000
            }))
        );
        let options = FragmentOptions::default().length(1000);
        let mut writer = FragmentedWriter::new(opus::Config::new(1, 312).unwrap(), options);
        writer.push(&opus_packet(10), &mut file).unwrap();
        writer.push(&opus_packet(10), &mut file).unwrap();
        // The third would start at 1920, past 1312.
        assert!(matches!(
            writer.push(&opus_packet(10), &mut file),
            Err(PushError::Length(_))
        ));
        assert_eq!(writer.samples(), 1920);
        assert!(writer.finish(&mut file).is_ok());
    }

    /// chelys's own frames, given as packets with their delay, make the
    /// file that its fragmented encoder makes at a variable rate, whether
    /// the length is declared or not.
    #[test]
    #[cfg(feature = "aac")]
    fn aac_packets_make_the_file_of_chelys_s_own_encoder() {
        use crate::packets::progressive::tests::{chelys_aac, tone};
        use chelys_aac::{Config, Rate};
        for len in [0, 1, 5000, 3 * 44_100 + 17] {
            for declared in [false, true] {
                let pcm = tone(len);
                let options = if declared {
                    FragmentOptions::default().length(len as u64)
                } else {
                    FragmentOptions::default()
                };
                let config = Config::new(
                    44_100,
                    2,
                    Rate::Variable {
                        noise_offset_db: 0.0,
                    },
                );
                let mut encoder =
                    crate::aac::encode::FragmentedEncoder::new(config, options).unwrap();
                let mut expected = encoder.init_segment().to_vec();
                let mut expected_infos = Vec::new();
                let mut out = |bytes: &[u8], info: FragmentInfo| {
                    expected.extend_from_slice(bytes);
                    expected_infos.push(info);
                };
                encoder.encode_interleaved(&pcm, &mut out).unwrap();
                encoder.finish(&mut out).unwrap();

                let (_, frames, asc) = chelys_aac(&pcm);
                let config = crate::aac::Config::from_audio_specific_config(
                    &asc,
                    Some(chelys_aac::DELAY as u32),
                )
                .unwrap();
                let (file, infos) = write(FragmentedWriter::new(config, options), &frames);
                if declared {
                    assert!(file == expected, "{len}");
                    assert_eq!(infos, expected_infos, "{len}");
                } else {
                    // chelys's encoder knows the input's end, and the last
                    // frame lasts to it; the writer, not told, keeps the
                    // last packet's duration.
                    assert!(media_data(&file) == media_data(&expected), "{len}");
                    let frames = |infos: &[FragmentInfo]| -> Vec<u32> {
                        infos.iter().map(|info| info.frames).collect()
                    };
                    assert_eq!(frames(&infos), frames(&expected_infos), "{len}");
                    let total: u64 = infos.iter().map(|info| info.duration).sum();
                    assert_eq!(total, 1024 * frames(&infos).iter().sum::<u32>() as u64);
                }
            }
        }
    }

    #[test]
    fn an_alac_file_s_fragments_give_the_durations_that_differ() {
        let cookie = crate::alac::tests::cookie(4096, 16, 1, 8_000);
        let config = alac::Config::from_magic_cookie(&cookie).unwrap();
        let options = FragmentOptions::default().length(4 * 4096 + 100);
        let mut writer = FragmentedWriter::new(config, options);
        let mut file = writer.init_segment().to_vec();
        let mut infos = Vec::new();
        let mut out = |bytes: &[u8], info: FragmentInfo| {
            file.extend_from_slice(bytes);
            infos.push(info);
        };
        let packets: Vec<Vec<u8>> = (0..5).map(|i| vec![0x20; 40 + i]).collect();
        for packet in &packets[..4] {
            writer.push(packet, &mut out).unwrap();
        }
        assert!(matches!(
            writer.push_with_duration(&packets[4], 101, &mut out),
            Err(PushError::Length(_))
        ));
        writer
            .push_with_duration(&packets[4], 100, &mut out)
            .unwrap();
        writer.finish(&mut out).unwrap();
        assert_eq!(media_data(&file), packets.concat());
        assert_eq!(
            children(&find(&file, &[b"moov", b"trak"])).kinds(),
            ["tkhd", "mdia"]
        );
        let trex = find(&file, &[b"moov", b"mvex", b"trex"]);
        assert_eq!(u32_at(trex.payload, 12), 4096);
        // Fragments of a second, two packets each, the last of the last
        // packet alone, which gives its duration.
        let frames: Vec<u32> = infos.iter().map(|info| info.frames).collect();
        assert_eq!(frames, [2, 2, 1]);
        assert_eq!(infos[2].duration, 100);
        let moofs: Vec<_> = boxes(&file, 0)
            .into_iter()
            .filter(|b| &b.kind == b"moof")
            .collect();
        let traf = |i: usize| children(&children(&moofs[i])[1]);
        assert_eq!(traf(0).kinds(), ["tfhd", "tfdt", "trun"]);
        assert_eq!(u32_at(traf(0)[2].payload, 0) & 0x100, 0);
        assert_eq!(u32_at(traf(2)[2].payload, 0) & 0x100, 0x100);
    }

    #[test]
    fn an_opus_file_of_no_length_presents_every_packet() {
        let config = opus::Config::new(1, 312).unwrap();
        let packets: Vec<Vec<u8>> = (0..3).map(|_| opus_packet(9)).collect();
        let (file, infos) = write(
            FragmentedWriter::new(config, FragmentOptions::default()),
            &packets,
        );
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].duration, 3 * 960);
        let elst = find(&file, &[b"moov", b"trak", b"edts", b"elst"]);
        assert_eq!(u32_at(elst.payload, 8), 0);
        assert_eq!(
            children(&find(&file, &[b"moov", b"mvex"])).kinds(),
            ["trex"]
        );
    }
}
