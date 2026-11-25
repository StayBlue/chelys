// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Ogg streams of chelys's Vorbis packets, for decoders to read: the page
//! format of RFC 3533 (clause 6, with its verified errata), and the
//! encapsulation of Vorbis in it of Appendix A.2 of the Vorbis I
//! specification. chelys gives Vorbis packets, not Ogg; this writer is for
//! the tests alone.

use chelys::vorbis::Headers;

/// The `header_type_flag` bits (RFC 3533, clause 6).
const CONTINUED: u8 = 0x01;
const BEGINNING_OF_STREAM: u8 = 0x02;
const END_OF_STREAM: u8 = 0x04;

/// The `bitstream_serial_number` of every stream written.
const SERIAL: u32 = 0x6368_656c;

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
/// packet on it, and 0 on the pages of headers.
///
/// Spec gap: Appendix A.2 reads the granule position of the first audio
/// page against the samples its packets return, as the start of the
/// stream, and that of the last page as its end; it does not say how a
/// page that is both is read. The first audio packet, which returns no
/// samples, is put alone on the first audio page, whose granule position, 0,
/// then gives the start of the stream alone, and the last page gives its
/// end.
///
/// # Panics
///
/// If there are no audio packets, or `packets_per_page` is 0, or a page
/// would need more than 255 segments: this writer splits no packet between
/// pages.
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
    let mut writer = Writer::default();
    writer.page(&[&headers.identification], 0, BEGINNING_OF_STREAM);
    writer.page(&[&headers.comment, &headers.setup], 0, 0);
    let last = pages.len() - 1;
    for (i, page) in pages.iter().enumerate() {
        let data: Vec<&[u8]> = page.iter().map(|packet| packet.data.as_slice()).collect();
        let granule_position = page.last().unwrap().granule_position;
        let flags = if i == last { END_OF_STREAM } else { 0 };
        writer.page(&data, granule_position, flags);
    }
    writer.out
}

#[derive(Default)]
struct Writer {
    out: Vec<u8>,
    page_sequence_number: u32,
}

impl Writer {
    /// Writes a page of whole packets (RFC 3533, clause 6), none continued
    /// from the page before.
    fn page(&mut self, packets: &[&[u8]], granule_position: u64, flags: u8) {
        assert_eq!(flags & CONTINUED, 0, "this writer splits no packet");
        // Each packet's lacing values: 255 for each 255 bytes, then what is
        // left, 0 to 254, which ends it (clause 5).
        let segment_table: Vec<u8> = packets
            .iter()
            .flat_map(|packet| {
                std::iter::repeat_n(255, packet.len() / 255).chain([(packet.len() % 255) as u8])
            })
            .collect();
        assert!(segment_table.len() <= 255, "a page of too many segments");
        let start = self.out.len();
        self.out.extend(b"OggS");
        self.out.push(0); // stream_structure_version
        self.out.push(flags);
        self.out.extend(granule_position.to_le_bytes());
        self.out.extend(SERIAL.to_le_bytes());
        self.out.extend(self.page_sequence_number.to_le_bytes());
        self.out.extend([0; 4]); // the CRC_checksum, zeroed to compute it
        self.out.push(segment_table.len() as u8);
        self.out.extend(&segment_table);
        for packet in packets {
            self.out.extend(*packet);
        }
        let crc = crc(&self.out[start..]);
        self.out[start + 22..start + 26].copy_from_slice(&crc.to_le_bytes());
        self.page_sequence_number += 1;
    }
}

/// The 32-bit CRC of a page, with the generator polynomial 0x04c11db7
/// (RFC 3533, clause 6, as verified erratum 8825 words it): over the header,
/// its CRC field zeroed, and the page's content.
///
/// Spec gap: RFC 3533 gives the polynomial alone, not the CRC's initial
/// value, final XOR or bit order. It is read literally: the remainder of
/// the bytes, each most significant bit first, times x^32, divided by the
/// polynomial; that is, from an initial value of 0, with no final XOR and
/// unreflected. (Reported erratum 8824 proposes the "unreflected" CRC, but
/// is not verified, and says it was taken from another implementation; it
/// is not a source.)
pub fn crc(bytes: &[u8]) -> u32 {
    let mut register = 0u32;
    for &byte in bytes {
        register ^= u32::from(byte) << 24;
        for _ in 0..8 {
            register = if register & 0x8000_0000 != 0 {
                (register << 1) ^ 0x04c1_1db7
            } else {
                register << 1
            };
        }
    }
    register
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_crc_is_the_remainder_of_the_polynomial() {
        // A message of the polynomial's own 33 bits, x^32 + ...: its
        // remainder is 0, as is any multiple's.
        assert_eq!(crc(&[]), 0);
        assert_eq!(crc(&[0; 7]), 0);
        // x^32 mod p is p's low 32 bits: the message 0x01, times x^32,
        // is x^32.
        assert_eq!(crc(&[1]), 0x04c1_1db7);
    }
}
