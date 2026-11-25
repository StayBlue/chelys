// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Codebooks as the setup header carries them (clause 3.2.1): each entry's
//! codeword length, from which a decoder builds the Huffman tree, and the
//! value mapping that gives each entry its vector (clause 3.2.1, "VQ lookup
//! table vector representation"); and the codewords that the audio packets
//! write the entries with.

use super::bitpack::{BitPacker, ilog};

/// The codebook's sync pattern, 0x564342: "BCV" in its three bytes.
const SYNC: u32 = 0x56_43_42;

/// A codebook: its entries' codewords, and, if they are read as vectors
/// (clause 3.3), their value mapping.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Codebook {
    dimensions: u16,
    /// Each entry's codeword length in bits, from 1 to 32, or `None` for an
    /// entry left unused, which has no codeword.
    lengths: Vec<Option<u8>>,
    /// Each used entry's codeword, its first bit the most significant of
    /// its length: as clause 3.2.1 assigns them.
    codewords: Vec<Option<u32>>,
    lattice: Option<Lattice>,
}

/// A value mapping of lookup type 1, a lattice (clause 3.2.1): each of an
/// entry's scalars one of `values` integers, `minimum`, `minimum + delta`
/// and so on, its `codebook_multiplicands` being 0 to `values - 1` in order,
/// and `codebook_sequence_p` unset. The first scalar of entry `e` is then
/// the value of index `e mod values`, the second that of `(e / values) mod
/// values`, and so on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lattice {
    pub minimum: i32,
    pub delta: u32,
    pub values: u32,
}

impl Lattice {
    /// The scalars `minimum + i * delta` for `i` below `values`.
    #[cfg(test)]
    pub fn scalar(&self, index: u32) -> i32 {
        self.minimum + (index * self.delta) as i32
    }

    /// The index of `scalar` among the lattice's values, if it is one.
    pub fn index(&self, scalar: i32) -> Option<u32> {
        let offset = i64::from(scalar) - i64::from(self.minimum);
        let delta = i64::from(self.delta);
        (offset >= 0 && offset % delta == 0 && offset / delta < i64::from(self.values))
            .then(|| (offset / delta) as u32)
    }
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
        Self::build(dimensions, lengths, None)
    }

    /// A codebook as [`new`](Self::new) makes, whose entries are read as
    /// vectors of `dimensions` scalars of `lattice`: one entry for each
    /// vector, all used.
    ///
    /// # Panics
    ///
    /// As [`new`](Self::new), and if `lengths` are not one for each vector,
    /// or a scalar or the delta cannot be written as clause 9.2.2 unpacks
    /// them.
    pub fn with_lattice(dimensions: u16, lengths: Vec<u8>, lattice: Lattice) -> Self {
        assert!(lattice.values >= 1 && lattice.delta >= 1, "{lattice:?}");
        assert_eq!(
            Some(lengths.len()),
            (lattice.values as usize).checked_pow(u32::from(dimensions)),
            "one entry for each vector of the lattice"
        );
        // Each is checked to be written exactly.
        float32_pack(f64::from(lattice.minimum));
        float32_pack(f64::from(lattice.delta));
        Self::build(
            dimensions,
            lengths.into_iter().map(Some).collect(),
            Some(lattice),
        )
    }

    fn build(dimensions: u16, lengths: Vec<Option<u8>>, lattice: Option<Lattice>) -> Self {
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
            codewords: assign_codewords(&lengths),
            lengths,
            lattice,
        }
    }

    pub fn entries(&self) -> usize {
        self.lengths.len()
    }

    pub fn dimensions(&self) -> usize {
        usize::from(self.dimensions)
    }

    /// The entry whose vector is `vector`, of a codebook with a lattice, if
    /// there is one.
    pub fn entry_of(&self, vector: &[i32]) -> Option<usize> {
        let lattice = self.lattice.expect("a codebook with a value mapping");
        assert_eq!(vector.len(), self.dimensions(), "a vector of the codebook");
        // The first scalar is the least significant digit (clause 3.2.1,
        // "Vector value decode: Lookup type 1").
        vector.iter().rev().try_fold(0usize, |entry, &scalar| {
            Some(entry * lattice.values as usize + lattice.index(scalar)? as usize)
        })
    }

    /// The vector of `entry`, as a decoder unpacks it.
    #[cfg(test)]
    pub fn vector(&self, entry: usize) -> Vec<i32> {
        let lattice = self.lattice.expect("a codebook with a value mapping");
        let mut divisor = 1usize;
        (0..self.dimensions())
            .map(|_| {
                let index = (entry / divisor) % lattice.values as usize;
                divisor *= lattice.values as usize;
                lattice.scalar(index as u32)
            })
            .collect()
    }

    /// The bits that `entry`'s codeword takes.
    pub fn length(&self, entry: usize) -> u32 {
        u32::from(self.lengths[entry].expect("a used entry"))
    }

    /// Writes `entry`'s codeword, its bits in the order a decoder reads
    /// them, one at a time, walking the tree (clause 3.3).
    ///
    /// # Panics
    ///
    /// If the entry is unused.
    pub fn write_entry(&self, out: &mut BitPacker, entry: usize) {
        let codeword = self.codewords[entry].expect("a used entry");
        let length = self.length(entry);
        for bit in (0..length).rev() {
            out.write((codeword >> bit) & 1, 1);
        }
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
        match self.lattice {
            // codebook_lookup_type 0: no value mapping.
            None => out.write(0, 4),
            Some(lattice) => {
                out.write(1, 4);
                out.write(float32_pack(f64::from(lattice.minimum)), 32);
                out.write(float32_pack(f64::from(lattice.delta)), 32);
                let value_bits = ilog(lattice.values - 1).max(1);
                out.write(value_bits - 1, 4);
                out.write_bool(false); // codebook_sequence_p
                debug_assert_eq!(
                    lookup1_values(self.entries(), self.dimensions()),
                    lattice.values as usize
                );
                for multiplicand in 0..lattice.values {
                    out.write(multiplicand, value_bits);
                }
            }
        }
    }
}

/// Each used entry's codeword, in order the lowest valued that is unused
/// and of the entry's length, its leftmost bit the most significant
/// (clause 3.2.1, "Huffman decision tree representation").
///
/// A codeword of length `l` and value `c` is the span of the code space
/// from `c * 2^(32 - l)` to `(c + 1) * 2^(32 - l)`, and is unused where
/// that span meets no codeword's before it: where neither is a prefix of
/// the other. The free spans are kept in order, so that the lowest that
/// fits is the first.
fn assign_codewords(lengths: &[Option<u8>]) -> Vec<Option<u32>> {
    let mut free: Vec<(u64, u64)> = vec![(0, 1 << 32)];
    lengths
        .iter()
        .map(|length| {
            let length = u32::from((*length)?);
            let size = 1u64 << (32 - length);
            let (i, start) = free
                .iter()
                .enumerate()
                .find_map(|(i, &(lo, hi))| {
                    let start = lo.div_ceil(size) * size;
                    (start + size <= hi).then_some((i, start))
                })
                .expect("the lengths make a tree");
            let (lo, hi) = free[i];
            let mut left = Vec::with_capacity(2);
            if lo < start {
                left.push((lo, start));
            }
            if start + size < hi {
                left.push((start + size, hi));
            }
            free.splice(i..=i, left);
            Some((start / size) as u32)
        })
        .collect()
}

/// `float32_pack`, the inverse of `float32_unpack` (clause 9.2.2): `value`
/// as a mantissa of 21 bits and its sign, and an exponent of 10 bits from
/// which 788 is taken.
///
/// # Panics
///
/// If `value` cannot be written so exactly.
pub fn float32_pack(value: f64) -> u32 {
    if value == 0.0 {
        return 0;
    }
    // The odd mantissa and its power of two, from which to take the
    // exponent that keeps the mantissa within 21 bits.
    let bits = value.abs().to_bits();
    let (mut mantissa, mut exponent) = (
        (bits & ((1 << 52) - 1)) | (1 << 52),
        ((bits >> 52) & 0x7ff) as i64 - 1075,
    );
    assert!(bits >> 52 != 0 && value.is_finite(), "{value}");
    let zeros = mantissa.trailing_zeros();
    mantissa >>= zeros;
    exponent += i64::from(zeros);
    assert!(mantissa < 1 << 21, "{value} needs more than 21 bits");
    let biased = exponent + 788;
    assert!((0..1024).contains(&biased), "{value} is out of range");
    let sign = if value < 0.0 { 0x8000_0000 } else { 0 };
    sign | ((biased as u32) << 21) | mantissa as u32
}

/// `float32_unpack` (clause 9.2.2).
#[cfg(test)]
pub fn float32_unpack(x: u32) -> f64 {
    let mantissa = f64::from(x & 0x1f_ffff);
    let exponent = ((x & 0x7fe0_0000) >> 21) as i32;
    let signed = if x & 0x8000_0000 != 0 {
        -mantissa
    } else {
        mantissa
    };
    signed * 2f64.powi(exponent - 788)
}

/// `lookup1_values` (clause 9.2.3): the greatest integer whose power of
/// `dimensions` is at most `entries`.
pub fn lookup1_values(entries: usize, dimensions: usize) -> usize {
    let fits = |r: usize| {
        u32::try_from(dimensions)
            .ok()
            .and_then(|d| r.checked_pow(d))
            .is_some_and(|power| power <= entries)
    };
    let mut r = 1;
    while fits(r + 1) {
        r += 1;
    }
    r
}

/// The codeword lengths of a Huffman code for entries of the given
/// weights, each in proportion to the entry's probability: no length above
/// 32, nor below 1, and their tree full, as [`Codebook::new`] requires.
///
/// Where the code would need longer codewords, the weights are flattened,
/// each to its square root, until it does not.
///
/// # Panics
///
/// If there are no weights, or one is not positive and finite.
pub fn huffman_lengths(weights: &[f64]) -> Vec<u8> {
    assert!(!weights.is_empty(), "a code of no entries");
    assert!(
        weights.iter().all(|w| w.is_finite() && *w > 0.0),
        "weights {weights:?}"
    );
    if weights.len() == 1 {
        return vec![1];
    }
    let mut weights = weights.to_vec();
    loop {
        let lengths = huffman_depths(&weights);
        if lengths.iter().all(|&length| length <= 32) {
            return lengths.into_iter().map(|length| length as u8).collect();
        }
        for weight in &mut weights {
            *weight = weight.sqrt();
        }
    }
}

/// The depths of the leaves of the Huffman tree of `weights`, of at least
/// two: the two lightest nodes joined, again and again, ties going to the
/// node made first.
fn huffman_depths(weights: &[f64]) -> Vec<u32> {
    use std::cmp::{Ordering, Reverse};
    use std::collections::BinaryHeap;

    #[derive(PartialEq)]
    struct Node(f64, usize);
    impl Eq for Node {}
    impl PartialOrd for Node {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Node {
        fn cmp(&self, other: &Self) -> Ordering {
            self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
        }
    }

    // Leaves are nodes 0 to n - 1, and joined nodes follow; each node's
    // parent, the root's none.
    let n = weights.len();
    let mut parent = vec![usize::MAX; 2 * n - 1];
    let mut heap: BinaryHeap<Reverse<Node>> = (weights.iter().enumerate())
        .map(|(i, &w)| Reverse(Node(w, i)))
        .collect();
    let mut next = n;
    while let (Some(Reverse(a)), Some(Reverse(b))) = (heap.pop(), heap.pop()) {
        parent[a.1] = next;
        parent[b.1] = next;
        heap.push(Reverse(Node(a.0 + b.0, next)));
        next += 1;
    }
    (0..n)
        .map(|leaf| {
            let mut depth = 0;
            let mut node = leaf;
            while parent[node] != usize::MAX {
                node = parent[node];
                depth += 1;
            }
            depth
        })
        .collect()
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
        /// `codebook_minimum_value`, `codebook_delta_value`,
        /// `codebook_sequence_p` and `codebook_multiplicands`.
        lookup: Option<(f64, f64, bool, Vec<u32>)>,
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
        let lookup = (lookup_type == 1).then(|| {
            let minimum = float32_unpack(r.read(32));
            let delta = float32_unpack(r.read(32));
            let value_bits = r.read(4) + 1;
            let sequence_p = r.read(1) == 1;
            let values = lookup1_values(entries, usize::from(dimensions));
            let multiplicands = (0..values).map(|_| r.read(value_bits)).collect();
            (minimum, delta, sequence_p, multiplicands)
        });
        assert_eq!(r.at, bits);
        Read {
            dimensions,
            lengths,
            lookup_type,
            lookup,
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
    fn assigns_the_codewords_of_clause_3_2_1() {
        let lengths = [2, 4, 4, 4, 4, 2, 3, 3].map(Some).to_vec();
        let codebook = Codebook::new(1, lengths);
        let expected = [0b00, 0b0100, 0b0101, 0b0110, 0b0111, 0b10, 0b110, 0b111];
        assert_eq!(codebook.codewords, expected.map(Some));
        // Written leftmost bit first: entry 1, 0100, then entry 6, 110.
        let mut out = BitPacker::new();
        codebook.write_entry(&mut out, 1);
        codebook.write_entry(&mut out, 6);
        assert_eq!(out.into_bytes(), [0b0011_0010]);
    }

    #[test]
    fn assigns_codewords_to_lengths_out_of_order() {
        // Entry 0 takes 000; entry 1, of one bit, the lowest free, 1; the
        // rest fill what entry 0 left, in order.
        let lengths = vec![Some(3), Some(1), None, Some(3), Some(2)];
        let codebook = Codebook::new(1, lengths);
        assert_eq!(
            codebook.codewords,
            [Some(0b000), Some(0b1), None, Some(0b001), Some(0b01)]
        );
    }

    #[test]
    fn writes_a_lattice_and_unpacks_its_vectors() {
        let lattice = Lattice {
            minimum: -4,
            delta: 2,
            values: 5,
        };
        let codebook = Codebook::with_lattice(2, huffman_lengths(&[1.0; 25]), lattice);
        let read = write_and_read(&codebook);
        assert_eq!(read.lookup_type, 1);
        let (minimum, delta, sequence_p, multiplicands) = read.lookup.unwrap();
        assert_eq!((minimum, delta, sequence_p), (-4.0, 2.0, false));
        assert_eq!(multiplicands, [0, 1, 2, 3, 4]);
        // Entry 7 is 2 + 1 * 5: first scalar -4 + 2 * 2, second -4 + 2.
        assert_eq!(codebook.vector(7), [0, -2]);
        for entry in 0..25 {
            assert_eq!(codebook.entry_of(&codebook.vector(entry)), Some(entry));
        }
        assert_eq!(codebook.entry_of(&[1, 0]), None);
        assert_eq!(codebook.entry_of(&[6, 0]), None);
    }

    #[test]
    fn floats_pack_and_unpack_exactly() {
        for value in [
            0.0,
            1.0,
            -1.0,
            17.0,
            -4335.0,
            0.5,
            0.375,
            2097151.0,
            2f64.powi(-700),
        ] {
            assert_eq!(float32_unpack(float32_pack(value)), value, "{value}");
        }
        // 1 is the mantissa 1 at the exponent 788.
        assert_eq!(float32_pack(1.0), (788 << 21) | 1);
        assert_eq!(float32_pack(-3.0), 0x8000_0000 | (788 << 21) | 3);
    }

    #[test]
    #[should_panic(expected = "needs more than 21 bits")]
    fn refuses_a_float_of_too_many_bits() {
        float32_pack(f64::from((1 << 21) + 1));
    }

    #[test]
    fn lookup1_values_is_the_root_rounded_down() {
        for (entries, dimensions, expected) in [
            (81, 4, 3),
            (80, 4, 2),
            (25, 2, 5),
            (31, 1, 31),
            (1, 8, 1),
            (289, 2, 17),
        ] {
            assert_eq!(lookup1_values(entries, dimensions), expected);
        }
    }

    #[test]
    fn huffman_lengths_make_a_full_tree() {
        assert_eq!(huffman_lengths(&[1.0]), [1]);
        assert_eq!(huffman_lengths(&[1.0, 1.0]), [1, 1]);
        assert_eq!(huffman_lengths(&[4.0, 2.0, 1.0, 1.0]), [1, 2, 3, 3]);
        // Weights falling by half each time would need lengths up to 39.
        let weights: Vec<f64> = (0..40).map(|i| 0.5f64.powi(i)).collect();
        let lengths = huffman_lengths(&weights);
        assert!(lengths.iter().all(|&l| (1..=32).contains(&l)));
        Codebook::new(1, lengths.into_iter().map(Some).collect());
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
