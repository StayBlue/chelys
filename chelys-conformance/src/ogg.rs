// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Ogg streams of chelys's Vorbis packets laid out as a test asks, for
//! decoders to read, written with `chelys-ogg`'s page writer: those of a
//! [`chelys::ogg::vorbis::Encoder`] are laid out as it chooses, and these
//! as the tests' streams of silence and of broken headers need.

use chelys::ogg::internals::{Limits, MAX_SEGMENTS, NO_PACKET_ENDS, Pager};
use chelys::vorbis::Headers;

/// The `bitstream_serial_number` of every stream written.
const SERIAL: u32 = 0x6368_656c;

/// Pages ended only where the layout asks.
const UNBOUNDED: Limits = Limits {
    bytes: usize::MAX,
    span: u64::MAX,
    segments: MAX_SEGMENTS,
};

/// An audio packet, with its granule position.
pub struct AudioPacket {
    pub data: Vec<u8>,
    pub granule_position: u64,
}

/// A physical Ogg stream of one logical stream of Vorbis: the headers, the
/// identification header alone on the first page and the other two on the
/// second, then the first audio packet alone, and then `packets_per_page`
/// audio packets on each page, the last page marked as the end of the
/// stream (Appendix A.2). Each page's granule position is that of the last
/// packet ended on it, and 0 on the pages of headers. A packet that does
/// not fit on a page continues on the next, as `chelys::ogg` lays out (see
/// [`chelys::ogg::vorbis::Encoder`] on the first audio page).
///
/// # Panics
///
/// If there are no audio packets, or `packets_per_page` is 0.
pub fn vorbis(headers: &Headers, packets: &[AudioPacket], packets_per_page: usize) -> Vec<u8> {
    assert!(packets_per_page > 0, "pages of no packets");
    let (first, rest) = packets.split_first().expect("a stream of no audio packets");
    let pages = std::iter::once(std::slice::from_ref(first)).chain(rest.chunks(packets_per_page));
    write(headers, &pages.collect::<Vec<_>>())
}

/// A physical Ogg stream as [`vorbis`] writes, but with every audio packet
/// on one page, which is both the first audio page and the last: of a
/// stream whose granule positions Appendix A.2 does not say how to read.
pub fn vorbis_on_one_audio_page(headers: &Headers, packets: &[AudioPacket]) -> Vec<u8> {
    assert!(!packets.is_empty(), "a stream of no audio packets");
    write(headers, &[packets])
}

/// The pages of the headers, then a page of each group of audio packets
/// that `pages` gives, the last marked as the end of the stream.
fn write(headers: &Headers, pages: &[&[AudioPacket]]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut pager = Pager::new(SERIAL, UNBOUNDED, 0);
    pager.packet(&headers.identification, 0, &mut out);
    pager.flush(&mut out);
    pager.packet(&headers.comment, 0, &mut out);
    pager.packet(&headers.setup, 0, &mut out);
    pager.flush(&mut out);
    pager.set_no_packet_ends(NO_PACKET_ENDS);
    let last = pages.len() - 1;
    for (i, page) in pages.iter().enumerate() {
        for packet in *page {
            pager.packet(&packet.data, packet.granule_position, &mut out);
        }
        if i < last {
            pager.flush(&mut out);
        }
    }
    pager.finish(&mut out);
    out
}
