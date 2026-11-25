// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Codebooks as the setup header carries them (clause 3.2.1): each entry's
//! codeword length, from which a decoder builds the Huffman tree.

use super::bitpack::{BitPacker, ilog};

/// The codebook's sync pattern, 0x564342: "BCV" in its three bytes.
const SYNC: u32 = 0x56_43_42;

/// A codebook without a value mapping (lookup type 0): one whose entries
/// are read in a scalar context only (clause 3.3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Codebook {
    dimensions: u16,
    /// Each entry's codeword length in bits, from 1 to 32, or `None` for an
    /// entry left unused, which has no codeword.
    lengths: Vec<Option<u8>>,
}

/// How a codebook's codeword lengths are written: the `ordered` and
/// `sparse` flags (clause 3.2.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LengthCoding {
    /// Each entry's length in 5 bits.
    Unordered,
    /// Each entry's flag of use, and the length of each used entry in 5
    /// bits.
    Sparse,
    /// The number of entries of each length, from the first entry's length
    /// up: only where every entry is used, and none is shorter than the one
    /// before it.
    Ordered,
}

impl Codebook {
    /// A codebook of `lengths.len()` entries, each a vector of `dimensions`
    /// scalars.
    ///
    /// # Panics
    ///
    /// If the codebook cannot be written, or a decoder could not build its
    /// tree: if it has no entries or more than 2^24 - 1, or `dimensions`
    /// is 0; if a length is not from 1 to 32; or if the used entries'
    /// lengths do not make a tree whose every node has two branches, as
    /// clause 3.2.1 requires, but where one entry alone is used, of length
    /// 1, as the errata of 2015-02-26 allow.
    pub fn new(dimensions: u16, lengths: Vec<Option<u8>>) -> Self {
        assert!(dimensions > 0, "a codebook's vectors have a dimension");
        assert!(
            !lengths.is_empty() && lengths.len() < 1 << 24,
            "{} entries do not fit a codebook",
            lengths.len()
        );
        let used: Vec<u8> = lengths.iter().flatten().copied().collect();
        assert!(
            used.iter().all(|length| (1..=32).contains(length)),
            "codeword lengths are from 1 to 32 bits"
        );
        // The tree is full where the codewords' shares of the code space,
        // 2^-length each, add up to all of it (Kraft's equality).
        let space: u64 = used.iter().map(|&length| 1 << (32 - length)).sum();
        assert!(
            space == 1 << 32 || used == [1],
            "the codeword lengths {used:?} do not make a full tree"
        );
        Self {
            dimensions,
            lengths,
        }
    }

    pub fn entries(&self) -> usize {
        self.lengths.len()
    }

    /// The coding of the lengths that takes the fewest bits, of those that
    /// can carry them; of those that tie, the first that `LengthCoding`
    /// lists.
    pub fn length_coding(&self) -> LengthCoding {
        [
            LengthCoding::Unordered,
            LengthCoding::Sparse,
            LengthCoding::Ordered,
        ]
        .into_iter()
        .filter_map(|coding| Some((self.length_bits(coding)?, coding)))
        .min_by_key(|&(bits, _)| bits)
        .map(|(_, coding)| coding)
        .expect("a sparse coding carries any lengths")
    }

    /// The bits that `coding` writes the lengths in, after the `ordered`
    /// flag, or `None` if it cannot carry them.
    fn length_bits(&self, coding: LengthCoding) -> Option<usize> {
        match coding {
            LengthCoding::Unordered => self
                .lengths
                .iter()
                .all(Option::is_some)
                .then(|| 1 + 5 * self.entries()),
            LengthCoding::Sparse => Some(
                1 + self
                    .lengths
                    .iter()
                    .map(|length| if length.is_some() { 6 } else { 1 })
                    .sum::<usize>(),
            ),
            LengthCoding::Ordered => {
                let runs = self.ordered_runs()?;
                Some(
                    5 + runs
                        .iter()
                        .map(|&(start, _)| ilog((self.entries() - start) as u32) as usize)
                        .sum::<usize>(),
                )
            }
        }
    }

    /// The runs that the ordered coding writes, as the entry each starts at
    /// and its number of entries: one for each length from the first
    /// entry's to the last's, of the entries of that length; or `None`
    /// if an entry is unused or shorter than the one before it.
    fn ordered_runs(&self) -> Option<Vec<(usize, usize)>> {
        let lengths: Vec<u8> = self.lengths.iter().copied().collect::<Option<_>>()?;
        if lengths.windows(2).any(|pair| pair[1] < pair[0]) {
            return None;
        }
        let mut runs = Vec::new();
        let mut start = 0;
        let mut length = lengths[0];
        while start < lengths.len() {
            let number = lengths[start..]
                .iter()
                .take_while(|&&l| l == length)
                .count();
            runs.push((start, number));
            start += number;
            length += 1;
        }
        Some(runs)
    }

    /// Writes the codebook as clause 3.2.1 reads it.
    pub fn write(&self, out: &mut BitPacker) {
        out.write(SYNC, 24);
        out.write(u32::from(self.dimensions), 16);
        out.write(self.entries() as u32, 24);
        let coding = self.length_coding();
        out.write_bool(coding == LengthCoding::Ordered);
        match coding {
            LengthCoding::Unordered | LengthCoding::Sparse => {
                let sparse = coding == LengthCoding::Sparse;
                out.write_bool(sparse);
                for length in &self.lengths {
                    if sparse {
                        out.write_bool(length.is_some());
                    }
                    if let Some(length) = length {
                        out.write(u32::from(length - 1), 5);
                    }
                }
            }
            LengthCoding::Ordered => {
                let first = self.lengths[0].expect("an ordered codebook's entries are used");
                out.write(u32::from(first - 1), 5);
                for (start, number) in self.ordered_runs().expect("the lengths are ordered") {
                    out.write(number as u32, ilog((self.entries() - start) as u32));
                }
            }
        }
        // codebook_lookup_type 0: no value mapping.
        out.write(0, 4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fields of a codebook written by [`Codebook::write`], read back
    /// as clause 3.2.1 reads them.
    struct Read {
        dimensions: u16,
        lengths: Vec<Option<u8>>,
        lookup_type: u32,
        bits: usize,
    }

    /// Reads fields least significant bit first, as clause 2.1.4 packs
    /// them.
    struct Reader<'a> {
        bytes: &'a [u8],
        at: usize,
    }

    impl Reader<'_> {
        fn read(&mut self, bits: u32) -> u32 {
            let mut value = 0;
            for i in 0..bits {
                let bit = (self.bytes[self.at / 8] >> (self.at % 8)) & 1;
                value |= u32::from(bit) << i;
                self.at += 1;
            }
            value
        }
    }

    fn write_and_read(codebook: &Codebook) -> Read {
        let mut out = BitPacker::new();
        codebook.write(&mut out);
        let bits = out.bit_len();
        let bytes = out.into_bytes();
        let mut r = Reader {
            bytes: &bytes,
            at: 0,
        };
        assert_eq!(r.read(24), 0x564342);
        let dimensions = r.read(16) as u16;
        let entries = r.read(24) as usize;
        let ordered = r.read(1) == 1;
        let mut lengths = Vec::new();
        if ordered {
            let mut length = r.read(5) + 1;
            while lengths.len() < entries {
                let number = r.read(ilog((entries - lengths.len()) as u32));
                lengths.extend((0..number).map(|_| Some(length as u8)));
                length += 1;
            }
            assert_eq!(lengths.len(), entries);
        } else {
            let sparse = r.read(1) == 1;
            for _ in 0..entries {
                let used = !sparse || r.read(1) == 1;
                lengths.push(used.then(|| r.read(5) as u8 + 1));
            }
        }
        let lookup_type = r.read(4);
        assert_eq!(r.at, bits);
        Read {
            dimensions,
            lengths,
            lookup_type,
            bits,
        }
    }

    fn round_trips(codebook: &Codebook, coding: LengthCoding) {
        assert_eq!(codebook.length_coding(), coding);
        let read = write_and_read(codebook);
        assert_eq!(read.dimensions, codebook.dimensions);
        assert_eq!(read.lengths, codebook.lengths);
        assert_eq!(read.lookup_type, 0);
        assert_eq!(
            read.bits,
            24 + 16 + 24 + 1 + codebook.length_bits(coding).unwrap() + 4
        );
    }

    #[test]
    fn writes_the_example_tree_of_clause_3_2_1() {
        let lengths = [2, 4, 4, 4, 4, 2, 3, 3].map(Some).to_vec();
        round_trips(&Codebook::new(1, lengths), LengthCoding::Unordered);
    }

    #[test]
    fn writes_ordered_lengths_in_runs() {
        // 1, then two of 3, then four of 4: no entry of length 2, whose run
        // is written as 0.
        let lengths = [1, 3, 3, 4, 4, 4, 4].map(Some).to_vec();
        let codebook = Codebook::new(2, lengths);
        assert_eq!(
            codebook.ordered_runs(),
            Some(vec![(0, 1), (1, 0), (1, 2), (3, 4)])
        );
        round_trips(&codebook, LengthCoding::Ordered);
    }

    #[test]
    fn writes_unused_entries_sparse() {
        let lengths = vec![Some(1), None, None, Some(2), None, Some(2)];
        round_trips(&Codebook::new(1, lengths), LengthCoding::Sparse);
    }

    #[test]
    fn takes_one_entry_alone_of_length_1() {
        round_trips(&Codebook::new(1, vec![Some(1)]), LengthCoding::Unordered);
        let sparse = vec![None, None, None, None, None, None, None, Some(1)];
        round_trips(&Codebook::new(1, sparse), LengthCoding::Sparse);
    }

    #[test]
    #[should_panic(expected = "do not make a full tree")]
    fn refuses_a_tree_with_a_branch_missing() {
        Codebook::new(1, [2, 4, 4, 4, 4, 2, 3].map(Some).to_vec());
    }

    #[test]
    #[should_panic(expected = "do not make a full tree")]
    fn refuses_a_tree_with_too_many_codewords() {
        Codebook::new(1, [1, 1, 1].map(Some).to_vec());
    }

    #[test]
    #[should_panic(expected = "do not make a full tree")]
    fn refuses_one_entry_alone_of_another_length() {
        Codebook::new(1, vec![Some(2)]);
    }
}
