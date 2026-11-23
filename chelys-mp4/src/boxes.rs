// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Boxes (ISO/IEC 14496-12, clause 4.2.2): each a header that gives its
//! size and type, then its payload. Fields are big-endian (clause 4.2.1),
//! which every `to_be_bytes` here writes.

/// A four-character code, a box's type or a brand (clause 4.2.2).
pub type FourCc = [u8; 4];

/// Appends a box of type `kind`, whose payload `payload` appends, with the
/// compact header: a 32-bit size, which counts the header, and the type.
///
/// # Panics
///
/// If the box reaches 4 GiB, which only the media data's can: see
/// [`media_data_header`].
pub fn write_box(out: &mut Vec<u8>, kind: &FourCc, payload: impl FnOnce(&mut Vec<u8>)) {
    let start = out.len();
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(kind);
    payload(out);
    let size = u32::try_from(out.len() - start).expect("a box of structure-data under 4 GiB");
    out[start..start + 4].copy_from_slice(&size.to_be_bytes());
}

/// Appends a full box (`FullBox`, clause 4.2.2): a box whose payload
/// starts with a `version` of 8 bits and `flags` of 24.
pub fn write_full_box(
    out: &mut Vec<u8>,
    kind: &FourCc,
    version: u8,
    flags: u32,
    payload: impl FnOnce(&mut Vec<u8>),
) {
    debug_assert!(flags < 1 << 24);
    write_box(out, kind, |out| {
        out.push(version);
        out.extend_from_slice(&flags.to_be_bytes()[1..]);
        payload(out);
    });
}

/// The header of a `MediaDataBox` (`'mdat'`, clause 8.1.1) of
/// `payload_len` bytes: with the compact 32-bit size where it holds the
/// box's, and otherwise `size` 1 and the 64-bit `largesize` (clause
/// 4.2.2).
pub fn media_data_header(payload_len: u64) -> Vec<u8> {
    let mut header = Vec::with_capacity(16);
    match u32::try_from(payload_len + 8) {
        Ok(size) => header.extend_from_slice(&size.to_be_bytes()),
        Err(_) => header.extend_from_slice(&1u32.to_be_bytes()),
    }
    header.extend_from_slice(b"mdat");
    if header[..4] == 1u32.to_be_bytes() {
        header.extend_from_slice(&(payload_len + 16).to_be_bytes());
    }
    header
}

/// Reading boxes back, for the tests.
#[cfg(test)]
pub mod read {
    use super::FourCc;

    /// A box found in a file: its type, payload, and where it starts.
    #[derive(Debug)]
    pub struct Found<'a> {
        pub kind: FourCc,
        pub payload: &'a [u8],
        pub start: usize,
    }

    /// The types of boxes found, as text.
    pub trait FourCcs {
        fn kinds(&self) -> Vec<String>;
    }

    impl FourCcs for [Found<'_>] {
        fn kinds(&self) -> Vec<String> {
            self.iter()
                .map(|b| String::from_utf8_lossy(&b.kind).into_owned())
                .collect()
        }
    }

    /// The boxes one after another in `bytes`, which start `base` bytes
    /// into the file, by their compact sizes.
    pub fn boxes(bytes: &[u8], base: usize) -> Vec<Found<'_>> {
        let mut found = Vec::new();
        let mut at = 0;
        while at < bytes.len() {
            let size = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
            assert!(
                size >= 8 && at + size <= bytes.len(),
                "a box past its container"
            );
            found.push(Found {
                kind: bytes[at + 4..at + 8].try_into().unwrap(),
                payload: &bytes[at + 8..at + size],
                start: base + at,
            });
            at += size;
        }
        found
    }

    /// The boxes within `container`: after the fields that precede them
    /// in a `SampleDescriptionBox` and an `'mp4a'` sample entry.
    pub fn children<'a>(container: &Found<'a>) -> Vec<Found<'a>> {
        // The sample entry's boxes follow its 28 bytes of fields, and
        // stsd's its version, flags and entry_count.
        let skip = match &container.kind {
            b"stsd" => 8,
            b"mp4a" => 28,
            _ => 0,
        };
        boxes(&container.payload[skip..], container.start + 8 + skip)
    }

    /// The box of `path` in `file`, each type within the one before: the
    /// first of its type at each step.
    pub fn find<'a>(file: &'a [u8], path: &[&[u8; 4]]) -> Found<'a> {
        let mut found = boxes(file, 0);
        for (i, kind) in path.iter().enumerate() {
            let at = (found.iter().position(|b| &b.kind == *kind))
                .unwrap_or_else(|| panic!("no {}", String::from_utf8_lossy(*kind)));
            let this = found.swap_remove(at);
            if i + 1 == path.len() {
                return this;
            }
            found = children(&this);
        }
        unreachable!()
    }

    pub fn u32_at(bytes: &[u8], at: usize) -> u32 {
        u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_box_counts_its_header_and_payload() {
        let mut out = vec![0xaa];
        write_box(&mut out, b"free", |out| out.extend_from_slice(&[1, 2, 3]));
        assert_eq!(out, [0xaa, 0, 0, 0, 11, b'f', b'r', b'e', b'e', 1, 2, 3]);
    }

    #[test]
    fn a_full_box_starts_with_its_version_and_flags() {
        let mut out = Vec::new();
        write_full_box(&mut out, b"tkhd", 1, 0x00_0007, |_| {});
        assert_eq!(out, [0, 0, 0, 12, b't', b'k', b'h', b'd', 1, 0, 0, 7]);
    }

    #[test]
    fn media_data_past_4_gib_takes_the_64_bit_size() {
        let compact = u64::from(u32::MAX) - 8;
        assert_eq!(media_data_header(0), [0, 0, 0, 8, b'm', b'd', b'a', b't']);
        assert_eq!(
            media_data_header(compact),
            [0xff, 0xff, 0xff, 0xff, b'm', b'd', b'a', b't']
        );
        let large = compact + 1;
        let mut expected = vec![0, 0, 0, 1, b'm', b'd', b'a', b't'];
        expected.extend_from_slice(&(large + 16).to_be_bytes());
        assert_eq!(media_data_header(large), expected);
    }
}
