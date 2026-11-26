// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! chelys's Vorbis encoder into an Ogg stream (Appendix A.2 of the Vorbis I
//! specification).

use chelys_vorbis::{Config, ConfigError, Finished, NonFiniteSample, Packet, PacketSink, Sample};

use crate::PageSink;
use crate::page::{Limits, MAX_SEGMENTS, NO_PACKET_ENDS, Pager};

/// The bytes of packets at which a page of audio takes no further packet.
///
/// Spec gap: neither RFC 3533 nor Appendix A.2 says how many packets a page
/// holds; RFC 3533 says pages are "usually 4-8 kB" (clause 6). chelys's
/// own choice: a page takes further packets until it holds 4096 bytes, or
/// its packets span a second of audio ([`Encoder::new`]), so that a reader
/// that seeks or starts within the stream finds a page near where it
/// starts. The packet that reaches either stays on the page, which may so
/// pass 4096 bytes, or a second, by a packet: at sample rates below 1024
/// Hz, a short block alone spans more than a second. A page holds at most
/// 255 segments, and a packet that does not fit continues on the next.
const PAGE_BYTES: usize = 4096;

/// Encodes PCM as Vorbis I, as [`chelys_vorbis::Encoder`] does, into an Ogg
/// stream of one logical stream, written as the packets come: its
/// [`header_pages`](Self::header_pages), then pages of audio, which a
/// [`PageSink`] takes, so that the stream can be sent or played as it is
/// encoded.
///
/// The stream gives a player the input's exact length: the granule
/// position of its last page is the input's length, which tells the
/// decoder to discard what the last packet returns past it (Appendix A.2).
///
/// The pages are laid out as Appendix A.2 asks: the identification header
/// alone on the first page, marked as the beginning of the stream; the
/// comment and setup headers on the pages after it, the last of which
/// they end; then the audio packets, the first on a fresh page, and the
/// last page marked as the end of the stream. A packet that does not fit
/// on a page continues on the next.
///
/// ```
/// use chelys_ogg::vorbis::Encoder;
/// use chelys_vorbis::{Config, Rate};
///
/// let config = Config::new(44_100, 2, Rate::Quality { level: 50 });
/// let mut encoder = Encoder::new(config, 0x6368_656c)?;
/// let mut ogg = encoder.header_pages().to_vec();
/// encoder.encode_interleaved(&[0.0f32; 2 * 44_100], &mut ogg)?;
/// let finished = encoder.finish(&mut ogg);
/// assert_eq!(finished.samples, 44_100);
/// assert_eq!(&ogg[..4], b"OggS");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Encoder {
    encoder: chelys_vorbis::Encoder,
    pager: Pager,
    header_pages: Vec<u8>,
    /// Whether the first audio packet has been given.
    started: bool,
}

/// Gives each packet to the pager, and the pages it writes to the sink, as
/// the encoder codes them.
struct Pages<'a, O: ?Sized> {
    pager: &'a mut Pager,
    started: &'a mut bool,
    out: &'a mut O,
}

impl<O: PageSink + ?Sized> PacketSink for Pages<'_, O> {
    fn packet(&mut self, packet: Packet<'_>) {
        self.pager
            .packet(packet.data, packet.granule_position, self.out);
        // Spec gap: Appendix A.2 reads the granule position of the first
        // audio page against the samples its packets return, as the start
        // of the stream, and that of the last page as its end; it does not
        // say how a page that is both is read. The first audio packet,
        // which returns no samples, is put alone on the first audio page,
        // whose granule position, 0, then gives the start of the stream
        // alone, and the last page gives its end.
        if !*self.started {
            *self.started = true;
            self.pager.close();
        }
    }
}

impl Encoder {
    /// An encoder of the stream `config` asks for, as
    /// [`chelys_vorbis::Encoder::new`], into a logical stream of serial
    /// number `serial_number`.
    ///
    /// Spec reading: RFC 3533 says a logical stream's serial number "is
    /// created randomly" (clause 4), and requires only that each logical
    /// stream of a physical stream has its own. It is taken from the
    /// caller, who knows what else the physical stream holds, such as
    /// streams chained before or after this one, so that the same input
    /// gives the same stream; a caller that does not know may draw it at
    /// random.
    ///
    /// # Errors
    ///
    /// As [`chelys_vorbis::Encoder::new`].
    pub fn new(config: Config, serial_number: u32) -> Result<Self, ConfigError> {
        let limits = Limits {
            bytes: PAGE_BYTES,
            span: u64::from(config.sample_rate()),
            segments: MAX_SEGMENTS,
        };
        Self::with_limits(config, serial_number, limits)
    }

    /// As [`new`](Self::new), with the pages of audio ended at `limits`.
    pub(crate) fn with_limits(
        config: Config,
        serial_number: u32,
        limits: Limits,
    ) -> Result<Self, ConfigError> {
        let encoder = chelys_vorbis::Encoder::new(config)?;
        let headers = encoder.headers();
        // Spec reading: Appendix A.2 gives the pages "containing only
        // headers" the granule position 0, where RFC 3533 gives a page on
        // which no packet ends -1 (clause 6): a setup header that spans
        // pages leaves such pages. Appendix A.2, the mapping, is followed.
        let mut pager = Pager::new(serial_number, limits, 0);
        let mut header_pages = Vec::new();
        pager.packet(&headers.identification, 0, &mut header_pages);
        pager.flush(&mut header_pages);
        pager.packet(&headers.comment, 0, &mut header_pages);
        pager.packet(&headers.setup, 0, &mut header_pages);
        pager.flush(&mut header_pages);
        pager.set_no_packet_ends(NO_PACKET_ENDS);
        Ok(Self {
            encoder,
            pager,
            header_pages,
            started: false,
        })
    }

    /// The pages of the three headers, which begin the stream, before any
    /// page that [`encode_interleaved`](Self::encode_interleaved),
    /// [`encode_planar`](Self::encode_planar) or [`finish`](Self::finish)
    /// gives. They are fixed when the encoder is made.
    pub fn header_pages(&self) -> &[u8] {
        &self.header_pages
    }

    /// The number of channels the input has.
    pub fn channels(&self) -> usize {
        self.encoder.channels()
    }

    /// The sample rate in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.encoder.sample_rate()
    }

    /// Encodes interleaved samples, as
    /// [`chelys_vorbis::Encoder::encode_interleaved`], and gives `out` the
    /// pages they complete.
    ///
    /// # Errors
    ///
    /// As [`chelys_vorbis::Encoder::encode_interleaved`], if a sample is NaN
    /// or infinite: none of the call is taken, `out` is given nothing, and
    /// the encoder is as it was before the call.
    ///
    /// # Panics
    ///
    /// As [`chelys_vorbis::Encoder::encode_interleaved`].
    pub fn encode_interleaved<S: Sample, O: PageSink + ?Sized>(
        &mut self,
        samples: &[S],
        out: &mut O,
    ) -> Result<(), NonFiniteSample> {
        let mut pages = Pages {
            pager: &mut self.pager,
            started: &mut self.started,
            out,
        };
        self.encoder.encode_interleaved(samples, &mut pages)
    }

    /// Encodes one slice of samples per channel, as
    /// [`chelys_vorbis::Encoder::encode_planar`], and gives `out` the pages
    /// they complete.
    ///
    /// # Errors
    ///
    /// As [`encode_interleaved`](Self::encode_interleaved).
    ///
    /// # Panics
    ///
    /// As [`chelys_vorbis::Encoder::encode_planar`].
    pub fn encode_planar<S: Sample, O: PageSink + ?Sized>(
        &mut self,
        channels: &[&[S]],
        out: &mut O,
    ) -> Result<(), NonFiniteSample> {
        let mut pages = Pages {
            pager: &mut self.pager,
            started: &mut self.started,
            out,
        };
        self.encoder.encode_planar(channels, &mut pages)
    }

    /// Ends the stream, as [`chelys_vorbis::Encoder::finish`], and gives
    /// `out` the pages left, the last marked as the end of the stream.
    pub fn finish<O: PageSink + ?Sized>(self, out: &mut O) -> Finished {
        let Self {
            encoder,
            mut pager,
            mut started,
            ..
        } = self;
        let finished = encoder.finish(&mut Pages {
            pager: &mut pager,
            started: &mut started,
            out,
        });
        pager.finish(out);
        finished
    }
}

#[cfg(test)]
mod tests {
    use chelys_vorbis::{Headers, Rate};

    use super::*;
    use crate::page::tests::{packets, read};

    const BEGINNING_OF_STREAM: u8 = 0x02;
    const END_OF_STREAM: u8 = 0x04;

    fn config(channels: usize) -> Config {
        Config::new(48_000, channels, Rate::Quality { level: 50 })
    }

    fn tone(channels: usize, samples: usize) -> Vec<f32> {
        (0..samples * channels)
            .map(|i| ((i / channels) as f32 * 0.05).sin() * 0.4)
            .collect()
    }

    /// The headers and audio packets that chelys-vorbis gives for `pcm`.
    fn vorbis_packets(config: Config, pcm: &[f32]) -> (Headers, Vec<(Vec<u8>, u64)>) {
        let mut encoder = chelys_vorbis::Encoder::new(config).unwrap();
        let mut packets = Vec::new();
        let mut out = |p: Packet<'_>| packets.push((p.data.to_vec(), p.granule_position));
        encoder.encode_interleaved(pcm, &mut out).unwrap();
        encoder.finish(&mut out);
        (encoder_headers(config), packets)
    }

    fn encoder_headers(config: Config) -> Headers {
        chelys_vorbis::Encoder::new(config).unwrap().headers()
    }

    fn encode(encoder: Encoder, pcm: &[f32]) -> Vec<u8> {
        let mut encoder = encoder;
        let mut ogg = encoder.header_pages().to_vec();
        // In uneven calls, as the input may come.
        for chunk in pcm.chunks(encoder.channels() * 777) {
            encoder.encode_interleaved(chunk, &mut ogg).unwrap();
        }
        encoder.finish(&mut ogg);
        ogg
    }

    #[test]
    fn the_stream_holds_the_headers_and_packets_as_appendix_a_lays_them_out() {
        let pcm = tone(2, 3 * 48_000 + 123);
        let (headers, expected) = vorbis_packets(config(2), &pcm);
        let ogg = encode(Encoder::new(config(2), 99).unwrap(), &pcm);
        let pages = read(&ogg);
        // The identification header alone, on a page of 58 bytes.
        assert_eq!(pages[0].len, 58);
        assert_eq!(pages[0].parts, [(headers.identification.clone(), true)]);
        assert_eq!(pages[0].granule_position, 0);
        // The comment and setup headers, ending their page.
        assert_eq!(
            pages[1].parts,
            [
                (headers.comment.clone(), true),
                (headers.setup.clone(), true)
            ]
        );
        assert_eq!(pages[1].granule_position, 0);
        // The first audio packet alone.
        assert_eq!(pages[2].parts, [(expected[0].0.clone(), true)]);
        assert_eq!(pages[2].granule_position, 0);
        for (i, page) in pages.iter().enumerate() {
            assert_eq!(page.serial_number, 99);
            assert_eq!(page.sequence_number, i as u32);
            let flags = (if i == 0 { BEGINNING_OF_STREAM } else { 0 })
                | (if i == pages.len() - 1 {
                    END_OF_STREAM
                } else {
                    0
                });
            assert_eq!(page.flags, flags, "page {i}");
        }
        let audio = packets(&pages[2..]);
        assert_eq!(audio.len(), expected.len());
        for ((data, _), (expected, _)) in audio.iter().zip(&expected) {
            assert_eq!(data, expected);
        }
        // Each page's granule position is that of the last packet ended on
        // it, or -1 where none is.
        let mut ended = 0;
        for (i, page) in pages.iter().enumerate().skip(2) {
            let ends = page.parts.iter().filter(|(_, ends)| *ends).count();
            ended += ends;
            let position = if ends > 0 {
                expected[ended - 1].1
            } else {
                NO_PACKET_ENDS
            };
            assert_eq!(page.granule_position, position, "page {i}");
        }
        assert_eq!(pages.last().unwrap().granule_position, pcm.len() as u64 / 2);
        // No page of audio took a packet once it held 4096 bytes.
        assert!(pages.len() > 5);
        for page in &pages[3..] {
            let bytes: usize = page.parts.iter().map(|(d, _)| d.len()).sum();
            let last = page.parts.last().unwrap().0.len();
            assert!(bytes - last < PAGE_BYTES, "{bytes} bytes");
        }
    }

    #[test]
    fn the_stream_does_not_depend_on_how_the_input_is_divided() {
        let pcm = tone(1, 2 * 48_000);
        let whole = {
            let mut encoder = Encoder::new(config(1), 1).unwrap();
            let mut ogg = encoder.header_pages().to_vec();
            encoder.encode_interleaved(&pcm, &mut ogg).unwrap();
            encoder.finish(&mut ogg);
            ogg
        };
        assert_eq!(encode(Encoder::new(config(1), 1).unwrap(), &pcm), whole);
    }

    #[test]
    fn small_pages_split_headers_and_packets_across_pages() {
        let limits = Limits {
            bytes: 1,
            span: 1,
            segments: 2,
        };
        let pcm = tone(2, 48_000);
        let (headers, expected) = vorbis_packets(config(2), &pcm);
        let ogg = encode(Encoder::with_limits(config(2), 5, limits).unwrap(), &pcm);
        let pages = read(&ogg);
        assert_eq!(pages[0].parts, [(headers.identification.clone(), true)]);
        // The headers' pages: 0 where no packet ends on them.
        let header_pages =
            1 + (headers.comment.len() / 255 + 1 + headers.setup.len() / 255 + 1).div_ceil(2);
        let all = packets(&pages);
        assert_eq!(all[1].0, headers.comment);
        assert_eq!(all[2].0, headers.setup);
        assert!(
            (pages[..header_pages].iter()).all(|p| p.granule_position == 0),
            "header pages"
        );
        assert!(pages[header_pages - 1].parts.last().unwrap().1);
        // The audio, from a fresh page, some pages ending no packet.
        assert!(!pages[header_pages].parts.is_empty());
        assert_eq!(pages[header_pages].flags & 0x01, 0);
        let audio: Vec<_> = all[3..].iter().map(|(d, _)| d.clone()).collect();
        let expected: Vec<_> = expected.into_iter().map(|(d, _)| d).collect();
        assert_eq!(audio, expected);
        assert!(pages.iter().any(|p| p.granule_position == NO_PACKET_ENDS));
    }
}
