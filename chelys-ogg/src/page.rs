// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The pages of one logical stream (RFC 3533, clauses 5 and 6), as its
//! packets come.

use crate::PageSink;

/// The `header_type_flag` bits (clause 6).
const CONTINUED: u8 = 0x01;
const BEGINNING_OF_STREAM: u8 = 0x02;
const END_OF_STREAM: u8 = 0x04;

/// The granule position -1, in two's complement, of a page on which no
/// packet ends (clause 6).
pub const NO_PACKET_ENDS: u64 = u64::MAX;

/// The most lacing values a page's segment table holds: its count,
/// `number_page_segments`, is one byte (clause 6).
pub const MAX_SEGMENTS: usize = 255;

/// When a page is ended. `bytes` and `span` are thresholds checked as each
/// packet comes, before it joins the page: the packet that reaches one
/// stays on the page, which may so pass it by up to a packet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// A page that holds at least these bytes of packets takes no further
    /// packet.
    pub bytes: usize,
    /// A page whose packets span at least this many granule positions,
    /// from the last packet ended on a page before to the last ended on
    /// it, takes no further packet.
    pub span: u64,
    /// The most lacing values a page holds, from 1 to [`MAX_SEGMENTS`]. A
    /// packet that does not fit continues on the next page.
    pub segments: usize,
}

/// The pages of one logical stream, each written to a [`PageSink`] once it
/// is ended: when it is full, when a packet comes that it does not take,
/// or when the stream ends.
pub struct Pager {
    serial_number: u32,
    limits: Limits,
    /// The number of the next page written (clause 6).
    sequence_number: u32,
    /// Whether no page has been written yet.
    first: bool,
    /// The granule position of a page on which no packet ends.
    no_packet_ends: u64,
    /// The granule position of the last packet ended on a page written.
    written_granule_position: Option<u64>,
    /// The page being filled: its lacing values ...
    segments: Vec<u8>,
    /// ... the bytes of its segments ...
    body: Vec<u8>,
    /// ... whether its first segment continues a packet of the page
    /// before ...
    continued: bool,
    /// ... the granule position of the last packet ended on it, if any ...
    granule_position: Option<u64>,
    /// ... and whether it takes no further packet.
    closed: bool,
    /// The page as it is written.
    page: Vec<u8>,
}

impl Pager {
    /// The pages of the logical stream of `serial_number`, ended at
    /// `limits`, where no packet ends on a page with granule position
    /// `no_packet_ends` until [`set_no_packet_ends`](Self::set_no_packet_ends)
    /// says otherwise.
    ///
    /// # Panics
    ///
    /// If `limits.segments` is 0 or more than [`MAX_SEGMENTS`].
    pub fn new(serial_number: u32, limits: Limits, no_packet_ends: u64) -> Self {
        assert!(
            (1..=MAX_SEGMENTS).contains(&limits.segments),
            "a page holds 1 to {MAX_SEGMENTS} lacing values, not {}",
            limits.segments,
        );
        Self {
            serial_number,
            limits,
            sequence_number: 0,
            first: true,
            no_packet_ends,
            written_granule_position: None,
            segments: Vec::new(),
            body: Vec::new(),
            continued: false,
            granule_position: None,
            closed: false,
            page: Vec::new(),
        }
    }

    /// Gives pages on which no packet ends the granule position
    /// `no_packet_ends` from the page being filled on.
    pub fn set_no_packet_ends(&mut self, no_packet_ends: u64) {
        self.no_packet_ends = no_packet_ends;
    }

    /// Adds a packet whose granule position is `granule_position`, after
    /// writing the page being filled to `out` if it takes no further
    /// packet, and writes each page the packet fills.
    pub fn packet<O: PageSink + ?Sized>(
        &mut self,
        data: &[u8],
        granule_position: u64,
        out: &mut O,
    ) {
        if !self.segments.is_empty() && self.full() {
            self.write(false, out);
        }
        self.closed = false;
        // Each 255 bytes of the packet are a segment of lacing value 255,
        // and what is left, 0 to 254 bytes, the segment that ends it
        // (clause 5).
        let mut rest = data;
        loop {
            if self.segments.len() == self.limits.segments {
                self.write(false, out);
                self.continued = true;
            }
            let len = rest.len().min(255);
            self.segments.push(len as u8);
            self.body.extend_from_slice(&rest[..len]);
            rest = &rest[len..];
            if len < 255 {
                break;
            }
        }
        self.granule_position = Some(granule_position);
    }

    /// Whether the page being filled takes no further packet.
    fn full(&self) -> bool {
        let start = self.written_granule_position.unwrap_or(0);
        let span = (self.granule_position).map_or(0, |end| end.saturating_sub(start));
        self.closed
            || self.segments.len() == self.limits.segments
            || self.body.len() >= self.limits.bytes
            || span >= self.limits.span
    }

    /// Ends the page being filled: the next packet begins a fresh page.
    pub fn close(&mut self) {
        self.closed = true;
    }

    /// Writes the page being filled to `out` now, if it holds a segment.
    pub fn flush<O: PageSink + ?Sized>(&mut self, out: &mut O) {
        if !self.segments.is_empty() {
            self.write(false, out);
        }
    }

    /// Ends the stream: writes the page being filled to `out`, marked as
    /// its last, or if it holds no segment, a page of none, with the
    /// granule position of the last packet ended.
    pub fn finish<O: PageSink + ?Sized>(mut self, out: &mut O) {
        // Spec reading: an end-of-stream page may contain "no content but
        // simply a page header with position information" (clause 4),
        // where -1 says that "no packets finish on this page" (clause 6).
        // A page of no content is read as one whose position is that of
        // the stream's end, which the last packet ended on a page before
        // gives, so that a reader that takes the end from the last page
        // finds it there.
        if self.segments.is_empty() {
            self.granule_position = self.written_granule_position;
        }
        self.write(true, out);
    }

    /// Writes the page being filled to `out` (clause 6), and begins the
    /// next.
    fn write<O: PageSink + ?Sized>(&mut self, last: bool, out: &mut O) {
        let mut flags = 0;
        if self.continued {
            flags |= CONTINUED;
        }
        if self.first {
            flags |= BEGINNING_OF_STREAM;
        }
        if last {
            flags |= END_OF_STREAM;
        }
        let granule_position = self.granule_position.unwrap_or(self.no_packet_ends);
        let page = &mut self.page;
        page.clear();
        page.extend(b"OggS");
        page.push(0); // stream_structure_version
        page.push(flags);
        page.extend(granule_position.to_le_bytes());
        page.extend(self.serial_number.to_le_bytes());
        page.extend(self.sequence_number.to_le_bytes());
        page.extend([0; 4]); // the CRC_checksum, zeroed to compute it
        page.push(self.segments.len() as u8);
        page.extend(&self.segments);
        page.extend(&self.body);
        let crc = crc(page);
        page[22..26].copy_from_slice(&crc.to_le_bytes());
        out.page(page);
        // Spec gap: the page_sequence_number is four bytes and "increasing"
        // (clause 6), but RFC 3533 does not say what follows 2^32 - 1
        // pages, some 16 TB of pages of 4 kB. It wraps to 0, which leaves
        // a reader that checks for lost pages one difference of 1 apart,
        // modulo 2^32, as everywhere else.
        self.sequence_number = self.sequence_number.wrapping_add(1);
        self.first = false;
        if self.granule_position.is_some() {
            self.written_granule_position = self.granule_position;
        }
        self.segments.clear();
        self.body.clear();
        self.continued = false;
        self.granule_position = None;
        self.closed = false;
    }
}

/// The 32-bit CRC of a page, with the generator polynomial 0x04c11db7
/// (clause 6, as verified erratum 8825 words it): over the header, its CRC
/// field zeroed, and the page's content.
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
pub(crate) mod tests {
    use super::*;

    /// A page as read back: its fields, and the packets or parts of
    /// packets its lacing values divide its content into, each with
    /// whether it ends on the page.
    #[derive(Debug)]
    pub struct Page {
        pub flags: u8,
        pub granule_position: u64,
        pub serial_number: u32,
        pub sequence_number: u32,
        pub parts: Vec<(Vec<u8>, bool)>,
        pub len: usize,
    }

    /// Reads the pages of `stream`, checking each one's capture pattern,
    /// version and CRC.
    pub fn read(mut stream: &[u8]) -> Vec<Page> {
        let mut pages = Vec::new();
        while !stream.is_empty() {
            assert_eq!(&stream[..4], b"OggS");
            assert_eq!(stream[4], 0);
            let count = stream[26] as usize;
            let lacing = &stream[27..27 + count];
            let len = 27 + count + lacing.iter().map(|&l| l as usize).sum::<usize>();
            let mut zeroed = stream[..len].to_vec();
            zeroed[22..26].fill(0);
            let field = u32::from_le_bytes(stream[22..26].try_into().unwrap());
            assert_eq!(crc(&zeroed), field, "page {}", pages.len());
            let mut parts = Vec::new();
            let (mut at, mut part) = (27 + count, Vec::new());
            for &l in lacing {
                part.extend_from_slice(&stream[at..at + l as usize]);
                at += l as usize;
                if l < 255 {
                    parts.push((std::mem::take(&mut part), true));
                }
            }
            if count > 0 && lacing[count - 1] == 255 {
                parts.push((part, false));
            }
            pages.push(Page {
                flags: stream[5],
                granule_position: u64::from_le_bytes(stream[6..14].try_into().unwrap()),
                serial_number: u32::from_le_bytes(stream[14..18].try_into().unwrap()),
                sequence_number: u32::from_le_bytes(stream[18..22].try_into().unwrap()),
                parts,
                len,
            });
            stream = &stream[len..];
        }
        pages
    }

    /// The packets of `pages`, joined across pages, each with the granule
    /// position of the page it ends on.
    pub fn packets(pages: &[Page]) -> Vec<(Vec<u8>, u64)> {
        let mut packets = Vec::new();
        let mut part = Vec::new();
        for page in pages {
            assert_eq!(page.flags & CONTINUED != 0, !part.is_empty());
            for (data, ends) in &page.parts {
                part.extend_from_slice(data);
                if *ends {
                    packets.push((std::mem::take(&mut part), page.granule_position));
                }
            }
        }
        assert!(part.is_empty(), "a packet left unended");
        packets
    }

    const WIDE: Limits = Limits {
        bytes: usize::MAX,
        span: u64::MAX,
        segments: MAX_SEGMENTS,
    };

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

    #[test]
    fn a_packet_is_laced_in_segments_of_255_bytes_and_one_shorter() {
        for len in [0, 1, 254, 255, 256, 509, 510, 511, 1000] {
            let mut pager = Pager::new(7, WIDE, NO_PACKET_ENDS);
            let data: Vec<u8> = (0..len).map(|i| i as u8).collect();
            pager.packet(&data, 3, &mut |_: &[u8]| unreachable!());
            let mut stream = Vec::new();
            pager.finish(&mut stream);
            let count = stream[26] as usize;
            assert_eq!(count, len / 255 + 1, "{len} bytes");
            assert!(stream[27..27 + count - 1].iter().all(|&l| l == 255));
            assert_eq!(stream[27 + count - 1] as usize, len % 255);
            let pages = read(&stream);
            assert_eq!(pages.len(), 1);
            assert_eq!(pages[0].flags, BEGINNING_OF_STREAM | END_OF_STREAM);
            assert_eq!(pages[0].serial_number, 7);
            assert_eq!(packets(&pages), [(data, 3)]);
        }
    }

    #[test]
    fn a_packet_that_does_not_fit_continues_on_the_next_page() {
        let limits = Limits {
            segments: 3,
            ..WIDE
        };
        let mut pager = Pager::new(1, limits, NO_PACKET_ENDS);
        let mut stream = Vec::new();
        // 2 segments, then 5: one fills the first page, three the second,
        // on which no packet ends, and the last begins the third, which the
        // next packet joins.
        let (a, b, c) = (vec![1; 300], vec![2; 1100], vec![3; 10]);
        pager.packet(&a, 100, &mut stream);
        pager.packet(&b, 200, &mut stream);
        pager.packet(&c, 300, &mut stream);
        pager.finish(&mut stream);
        let pages = read(&stream);
        let fields: Vec<_> = (pages.iter())
            .map(|p| (p.flags, p.granule_position, p.sequence_number))
            .collect();
        assert_eq!(
            fields,
            [
                (BEGINNING_OF_STREAM, 100, 0),
                (CONTINUED, NO_PACKET_ENDS, 1),
                (CONTINUED | END_OF_STREAM, 300, 2),
            ]
        );
        assert_eq!(packets(&pages), [(a, 100), (b, 300), (c, 300)]);
    }

    #[test]
    fn a_page_ends_at_its_bounds_before_the_packet_that_passes_them() {
        let limits = Limits {
            bytes: 100,
            span: 1000,
            segments: MAX_SEGMENTS,
        };
        let mut pager = Pager::new(1, limits, NO_PACKET_ENDS);
        let mut stream = Vec::new();
        // 60 bytes, then 120, past 100; then spans from 300 of 400, then
        // 1100, past 1000.
        for (len, position) in [(60, 100), (60, 300), (10, 700), (10, 1400), (10, 1500)] {
            pager.packet(&vec![0; len], position, &mut stream);
        }
        pager.finish(&mut stream);
        let pages = read(&stream);
        let shape: Vec<_> = (pages.iter())
            .map(|p| (p.parts.len(), p.granule_position))
            .collect();
        assert_eq!(shape, [(2, 300), (2, 1400), (1, 1500)]);
    }

    #[test]
    fn a_packet_of_255_full_segments_ends_with_an_empty_one_on_the_next_page() {
        let mut pager = Pager::new(1, WIDE, NO_PACKET_ENDS);
        let mut stream = Vec::new();
        let data = vec![9; 255 * MAX_SEGMENTS];
        pager.packet(&data, 77, &mut stream);
        pager.finish(&mut stream);
        let pages = read(&stream);
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[0].len, 27 + 255 + 255 * 255);
        assert_eq!(pages[0].granule_position, NO_PACKET_ENDS);
        assert_eq!(
            (pages[1].flags, pages[1].len),
            (CONTINUED | END_OF_STREAM, 28)
        );
        assert_eq!(packets(&pages), [(data, 77)]);
    }

    #[test]
    fn the_sequence_number_wraps_and_only_the_first_page_begins_the_stream() {
        let mut pager = Pager::new(1, WIDE, NO_PACKET_ENDS);
        pager.sequence_number = u32::MAX;
        let mut stream = Vec::new();
        pager.packet(&[1], 1, &mut stream);
        pager.flush(&mut stream);
        pager.packet(&[2], 2, &mut stream);
        pager.finish(&mut stream);
        let fields: Vec<_> = (read(&stream).iter())
            .map(|p| (p.flags, p.sequence_number))
            .collect();
        assert_eq!(
            fields,
            [(BEGINNING_OF_STREAM, u32::MAX), (END_OF_STREAM, 0)]
        );
    }

    #[test]
    fn a_closed_page_takes_no_further_packet() {
        let mut pager = Pager::new(1, WIDE, NO_PACKET_ENDS);
        let mut stream = Vec::new();
        pager.packet(&[1], 0, &mut stream);
        pager.close();
        assert!(
            stream.is_empty(),
            "a closed page is written with the next packet"
        );
        pager.packet(&[2], 5, &mut stream);
        pager.packet(&[3], 9, &mut stream);
        pager.finish(&mut stream);
        let shape: Vec<_> = (read(&stream).iter())
            .map(|p| (p.parts.len(), p.granule_position))
            .collect();
        assert_eq!(shape, [(1, 0), (2, 9)]);
    }

    #[test]
    fn a_stream_ended_after_a_page_is_written_ends_with_a_page_of_no_segment() {
        let mut pager = Pager::new(1, WIDE, NO_PACKET_ENDS);
        let mut stream = Vec::new();
        pager.packet(&[1], 42, &mut stream);
        pager.flush(&mut stream);
        pager.finish(&mut stream);
        let pages = read(&stream);
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[1].flags, END_OF_STREAM);
        assert_eq!(pages[1].granule_position, 42);
        assert_eq!((pages[1].parts.len(), pages[1].len), (0, 27));
    }

    #[test]
    fn pages_on_which_no_packet_ends_take_the_granule_position_given() {
        let limits = Limits {
            segments: 1,
            ..WIDE
        };
        let mut pager = Pager::new(1, limits, 0);
        let mut stream = Vec::new();
        pager.packet(&[0; 300], 0, &mut stream);
        pager.flush(&mut stream);
        pager.set_no_packet_ends(NO_PACKET_ENDS);
        pager.packet(&[0; 300], 64, &mut stream);
        pager.finish(&mut stream);
        let positions: Vec<_> = read(&stream).iter().map(|p| p.granule_position).collect();
        assert_eq!(positions, [0, 0, NO_PACKET_ENDS, 64]);
    }
}
