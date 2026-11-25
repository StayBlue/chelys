// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The residue (clause 8): the spectrum divided by the floor, as integers,
//! in partitions of lines that each take the books of their class, in up
//! to three passes.
//!
//! The specification defines only the decoding; the residue's type,
//! partitions, classes and books here are chelys's own. It is of type 1
//! (clause 8.4): each channel's vector coded on its own, its scalars in
//! order.

use super::bitpack::BitPacker;
use super::codebook::{Codebook, Lattice, huffman_lengths};
use super::floor::RANGEBITS;

/// `residue_partition_size`.
pub const PARTITION_SIZE: usize = 16;

/// `residue_end`: the residue spans every line of a long block, from
/// `residue_begin`, 0.
const END: usize = 1 << RANGEBITS;

/// The largest magnitude a residue value may have: the largest that the
/// passes of the last class reach.
pub const MAX_VALUE: i32 = 15 * 289 + 8 * 17 + 8;

/// The value books, each a lattice of integers coded as vectors of its
/// dimension: its dimension, its lattice, and the scale of the magnitudes
/// of its values, in steps of its delta, from which the codeword lengths
/// are made (see [`Residue::new`]).
const VALUE_BOOKS: [(u16, Lattice, f64); 6] = [
    (4, lattice(1, 1), 0.9),
    (2, lattice(2, 1), 1.0),
    (2, lattice(4, 1), 1.8),
    (2, lattice(8, 1), 3.0),
    (1, lattice(15, 17), 4.0),
    (1, lattice(15, 289), 4.0),
];

/// The lattice of the integers from `-reach` to `reach` times `delta`.
const fn lattice(reach: i32, delta: u32) -> Lattice {
    Lattice {
        minimum: -reach * delta as i32,
        delta,
        values: 2 * reach as u32 + 1,
    }
}

/// The classes, by the largest magnitude of a partition's values that each
/// codes, and the value books of its passes, as indices into
/// [`VALUE_BOOKS`]. A partition takes the first class that codes it.
///
/// Each pass but the last codes the values rounded to the delta of its
/// book; the last, of delta 1, codes what is left. Class 0 codes a
/// partition of zeros, in no pass.
const CLASSES: [(i32, &[usize]); 7] = [
    (0, &[]),
    (1, &[0]),
    (2, &[1]),
    (4, &[2]),
    (8, &[3]),
    (15 * 17 + 8, &[4, 3]),
    (MAX_VALUE, &[5, 4, 3]),
];

/// The most passes of any class.
const PASSES: usize = 3;

/// The weights of the classes, from which the codeword lengths of the
/// classification book are made: chelys's guess at how often each is
/// chosen.
const CLASS_WEIGHTS: [f64; CLASSES.len()] = [0.3, 0.2, 0.15, 0.13, 0.12, 0.08, 0.02];

/// The classifications coded by each codeword of the classification book:
/// its dimension.
const CLASSWORDS_PER_CODEWORD: usize = 2;

/// The residue's configuration and books.
#[derive(Clone, Debug)]
pub struct Residue {
    /// The number of the classification book among the setup header's
    /// codebooks; the value books follow it.
    first_book: u8,
    classbook: Codebook,
    value_books: Vec<Codebook>,
}

impl Residue {
    /// The residue, whose books are numbered from `first_book` among the
    /// setup header's codebooks, in the order of
    /// [`codebooks`](Self::codebooks).
    ///
    /// The codeword lengths of each value book are those of a Huffman code
    /// for a model of its values: each scalar's magnitude, in steps of the
    /// book's delta, falling off exponentially at the book's scale, and the
    /// scalars of a vector independent.
    pub fn new(first_book: u8) -> Self {
        let classifications = CLASSES.len();
        let class_weights: Vec<f64> = (0..classifications.pow(CLASSWORDS_PER_CODEWORD as u32))
            .map(|entry| {
                (0..CLASSWORDS_PER_CODEWORD)
                    .map(|i| CLASS_WEIGHTS[entry / classifications.pow(i as u32) % classifications])
                    .product()
            })
            .collect();
        let classbook = Codebook::new(
            CLASSWORDS_PER_CODEWORD as u16,
            huffman_lengths(&class_weights)
                .into_iter()
                .map(Some)
                .collect(),
        );
        let value_books = VALUE_BOOKS
            .iter()
            .map(|&(dimensions, lattice, scale)| {
                let scalar = |index: u32| {
                    let steps = f64::from(lattice.scalar(index).unsigned_abs() / lattice.delta);
                    (-steps / scale).exp()
                };
                let entries = (lattice.values as usize).pow(u32::from(dimensions));
                let weights: Vec<f64> = (0..entries)
                    .map(|entry| {
                        let mut divisor = 1;
                        (0..dimensions)
                            .map(|_| {
                                let index = entry / divisor % lattice.values as usize;
                                divisor *= lattice.values as usize;
                                scalar(index as u32)
                            })
                            .product()
                    })
                    .collect();
                Codebook::with_lattice(dimensions, huffman_lengths(&weights), lattice)
            })
            .collect();
        Self {
            first_book,
            classbook,
            value_books,
        }
    }

    /// The residue's codebooks, the classification book and then the value
    /// books, in order.
    pub fn codebooks(&self) -> Vec<Codebook> {
        let mut books = vec![self.classbook.clone()];
        books.extend(self.value_books.iter().cloned());
        books
    }

    /// Writes the residue's configuration (clause 8.6.1), after its type.
    pub fn write_setup(&self, out: &mut BitPacker) {
        out.write(0, 24); // residue_begin
        out.write(END as u32, 24);
        out.write(PARTITION_SIZE as u32 - 1, 24);
        out.write(CLASSES.len() as u32 - 1, 6);
        out.write(u32::from(self.first_book), 8); // residue_classbook
        for (_, passes) in CLASSES {
            // residue_cascade: a bit for each pass, all in low_bits.
            out.write((1 << passes.len()) - 1, 3);
            out.write_bool(false);
        }
        for (_, passes) in CLASSES {
            for &book in passes {
                out.write(u32::from(self.first_book) + 1 + book as u32, 8);
            }
        }
    }

    /// Writes the residue of a packet whose vectors, each of `lines` lines,
    /// are `vectors`, in their channels' order: `None` for a vector marked
    /// "do not decode" (clause 8.6.2).
    ///
    /// # Panics
    ///
    /// If a vector is not of `lines` lines, or a value is beyond
    /// [`MAX_VALUE`].
    pub fn write(&self, vectors: &[Option<&[i32]>], lines: usize, out: &mut BitPacker) {
        let partitions = END.min(lines) / PARTITION_SIZE;
        // Each coded vector's classes, and the values of each pass.
        let coded: Vec<(Vec<usize>, Vec<Vec<i32>>)> = vectors
            .iter()
            .flatten()
            .map(|vector| {
                assert_eq!(vector.len(), lines, "a vector of the block's lines");
                classify(&vector[..partitions * PARTITION_SIZE])
            })
            .collect();
        for pass in 0..PASSES {
            let mut partition = 0;
            while partition < partitions {
                if pass == 0 {
                    for (classes, _) in &coded {
                        // The first classification the most significant
                        // digit; those past the last partition, 0.
                        let entry = (0..CLASSWORDS_PER_CODEWORD).fold(0, |entry, i| {
                            entry * CLASSES.len() + classes.get(partition + i).copied().unwrap_or(0)
                        });
                        self.classbook.write_entry(out, entry);
                    }
                }
                for _ in 0..CLASSWORDS_PER_CODEWORD {
                    if partition >= partitions {
                        break;
                    }
                    for (classes, by_pass) in &coded {
                        if let Some(&book) = CLASSES[classes[partition]].1.get(pass) {
                            let book = &self.value_books[book];
                            let start = partition * PARTITION_SIZE;
                            for vector in by_pass[pass][start..start + PARTITION_SIZE]
                                .chunks(book.dimensions())
                            {
                                let entry = book.entry_of(vector).expect("a value of the book");
                                book.write_entry(out, entry);
                            }
                        }
                    }
                    partition += 1;
                }
            }
        }
    }
}

/// Each partition's class, and the values each pass codes.
fn classify(values: &[i32]) -> (Vec<usize>, Vec<Vec<i32>>) {
    let mut by_pass = vec![vec![0; values.len()]; PASSES];
    let classes = values
        .chunks(PARTITION_SIZE)
        .enumerate()
        .map(|(p, partition)| {
            let largest = partition.iter().map(|v| v.abs()).max().unwrap_or(0);
            let class = CLASSES
                .iter()
                .position(|&(limit, _)| largest <= limit)
                .unwrap_or_else(|| panic!("a residue value of {largest}"));
            for (k, &value) in partition.iter().enumerate() {
                let mut left = value;
                let passes = CLASSES[class].1;
                for (pass, &book) in passes.iter().enumerate() {
                    let delta = VALUE_BOOKS[book].1.delta as i32;
                    // The last pass's delta is 1, and takes all that is
                    // left. The deltas are odd, so none is halfway.
                    let coded = delta * (f64::from(left) / f64::from(delta)).round() as i32;
                    by_pass[pass][p * PARTITION_SIZE + k] = coded;
                    left -= coded;
                }
                debug_assert_eq!(left, 0);
            }
            class
        })
        .collect();
    (classes, by_pass)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_class_reaches_its_limit_with_its_passes() {
        for (class, &(limit, passes)) in CLASSES.iter().enumerate().skip(1) {
            let reach: i32 = passes
                .iter()
                .map(|&book| {
                    let lattice = VALUE_BOOKS[book].1;
                    lattice.scalar(lattice.values - 1)
                })
                .sum();
            assert!(reach >= limit, "class {class}");
            assert_eq!(VALUE_BOOKS[*passes.last().unwrap()].1.delta, 1);
        }
    }

    #[test]
    fn the_passes_add_up_to_the_values() {
        let values: Vec<i32> = (-MAX_VALUE..=MAX_VALUE).step_by(7).collect();
        let len = values.len() / PARTITION_SIZE * PARTITION_SIZE;
        let (classes, by_pass) = classify(&values[..len]);
        for k in 0..len {
            let class = classes[k / PARTITION_SIZE];
            let passes = CLASSES[class].1;
            let sum: i32 = (0..PASSES).map(|pass| by_pass[pass][k]).sum();
            assert_eq!(sum, values[k]);
            for (pass, &book) in passes.iter().enumerate() {
                assert!(VALUE_BOOKS[book].1.index(by_pass[pass][k]).is_some());
            }
        }
    }

    #[test]
    fn partitions_take_the_first_class_that_codes_them() {
        let mut values = vec![0; 4 * PARTITION_SIZE];
        values[PARTITION_SIZE] = -1;
        values[2 * PARTITION_SIZE + 3] = 9;
        values[3 * PARTITION_SIZE] = -264;
        assert_eq!(classify(&values).0, [0, 1, 5, 6]);
    }

    #[test]
    fn the_classification_book_has_an_entry_for_each_pair() {
        let residue = Residue::new(0);
        assert_eq!(residue.classbook.entries(), CLASSES.len().pow(2));
        assert_eq!(residue.classbook.dimensions(), CLASSWORDS_PER_CODEWORD);
        for book in &residue.value_books {
            assert_eq!(PARTITION_SIZE % book.dimensions(), 0);
        }
    }
}
