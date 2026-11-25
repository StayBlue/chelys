// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Floor 1 (clause 7): the spectral envelope, as a piecewise straight line
//! in dB over the posts of `floor1_X_list`, whose Y values a packet codes
//! (clause 7.2.3) and a decoder draws (clause 7.2.4).
//!
//! The specification defines only the decoding; the floor's posts,
//! partitions, classes and books here are chelys's own. The encoder uses
//! the floor as its quantizer's step: each line's residue is the line
//! divided by the floor and rounded (see [`residue`](super::residue)).

use super::bitpack::{BitPacker, ilog};
use super::codebook::{Codebook, huffman_lengths};
use super::frequencies;
use super::setup::BLOCKSIZES;

/// `floor1_multiplier`: each Y value is two steps of the table of clause
/// 10.1, about 1.1 dB.
pub const MULTIPLIER: u32 = 2;

/// The Y values' range for [`MULTIPLIER`] (clause 7.2.3): 0 to 127.
pub const RANGE: i32 = [256, 128, 86, 64][MULTIPLIER as usize - 1];

/// The Y values coded together by each partition's class.
const CLASS_DIMENSIONS: usize = 3;

/// The partitions of the floor of a long block, all of class 0: 45 posts
/// besides the first two.
const LONG_PARTITIONS: usize = 15;

/// The partitions of the floor of a short block, all of class 0: 9 posts
/// besides the first two, a fifth as many as a long block's floor has
/// over eight times as many lines.
///
/// Short blocks come eight at a time, and their floors' bits count.
/// Graded with PEAQ (GstPEAQ 0.6.1, black box) on the 51 tracks of the
/// tuning set at 44.1 kHz in stereo, interpolated between noise offsets to
/// the same mean bitrate of 128 kbit/s, with short blocks' `rpelev` 1, the
/// streams graded -1.165 with 2 partitions, -1.164 with 3, -1.173 with 5
/// and -1.177 with 7.
const SHORT_PARTITIONS: usize = 3;

/// `floor1_class_subclasses` of class 0: two bits of the master book's
/// entry choose each Y value's book.
const SUBCLASS_BITS: u32 = 2;

/// The largest value of each subclass's book, the smallest values going
/// to the first that codes them. Subclass 0 has no book, and gives 0.
const SUBCLASS_LIMITS: [u32; 4] = [0, 3, 15, RANGE as u32 - 1];

/// By how much a Y value may differ from the one the floor predicts there
/// and still be coded as that prediction, as 0, which takes no bits of its
/// own: chelys's choice, the envelope being smooth to within a Y step.
const TOLERANCE: i32 = 1;

/// A floor's posts and books.
///
/// The setup header has one floor for each block size, whose X values
/// span the block's lines, 2^`rangebits`: the floor of a short block draws
/// no line past its lines, and spends no Y values there. Both take the
/// same books, which the setup header writes once.
#[derive(Clone, Debug)]
pub struct Floor {
    /// `rangebits`: the X values span 2^`rangebits` lines.
    rangebits: u32,
    /// The partitions, of [`CLASS_DIMENSIONS`] posts each.
    partitions: usize,
    /// `floor1_X_list`, in its order.
    xs: Vec<u32>,
    /// `low_neighbor` and `high_neighbor` of each post from the third
    /// (clauses 9.2.4 and 9.2.5).
    neighbors: Vec<(usize, usize)>,
    /// The posts in ascending order of X.
    sorted: Vec<usize>,
    /// The number of the master book among the setup header's codebooks;
    /// the books of subclasses 1 to 3 follow it.
    first_book: u8,
    master: Codebook,
    /// The books of subclasses 1 to 3.
    subclass_books: [Codebook; 3],
}

impl Floor {
    /// The floor of a long block, whose books are numbered from
    /// `first_book` among the setup header's codebooks, in the order of
    /// [`codebooks`](Self::codebooks).
    pub fn long(first_book: u8) -> Self {
        Self::new(first_book, BLOCKSIZES[1] / 2, LONG_PARTITIONS)
    }

    /// The floor of a short block, as [`long`](Self::long).
    pub fn short(first_book: u8) -> Self {
        Self::new(first_book, BLOCKSIZES[0] / 2, SHORT_PARTITIONS)
    }

    /// The floor of a block of `lines` lines, a power of two, of
    /// `partitions` partitions.
    fn new(first_book: u8, lines: usize, partitions: usize) -> Self {
        let rangebits = lines.ilog2();
        let xs = list_order(&posts(rangebits, partitions * CLASS_DIMENSIONS), rangebits);
        let neighbors = (0..xs.len())
            .map(|i| {
                if i < 2 {
                    (0, 0)
                } else {
                    (low_neighbor(&xs, i), high_neighbor(&xs, i))
                }
            })
            .collect();
        let mut sorted: Vec<usize> = (0..xs.len()).collect();
        sorted.sort_by_key(|&i| xs[i]);
        // The codeword lengths of each book are those of a Huffman code for
        // how often each of its entries was coded (see `frequencies`).
        let master = Codebook::new(
            1,
            huffman_lengths(&frequencies::weights(&frequencies::FLOOR_CLASSES))
                .into_iter()
                .map(Some)
                .collect(),
        );
        // Each subclass's book codes values above the limit of the one
        // before; its entries for the values below are unused.
        let counts: [&[u32]; 3] = [
            &frequencies::FLOOR_SUBCLASS_1,
            &frequencies::FLOOR_SUBCLASS_2,
            &frequencies::FLOOR_SUBCLASS_3,
        ];
        let subclass_books = [1, 2, 3].map(|s| {
            let mut lengths = vec![None; SUBCLASS_LIMITS[s - 1] as usize + 1];
            lengths.extend(
                huffman_lengths(&frequencies::weights(counts[s - 1]))
                    .into_iter()
                    .map(Some),
            );
            Codebook::new(1, lengths)
        });
        Self {
            rangebits,
            partitions,
            xs,
            neighbors,
            sorted,
            first_book,
            master,
            subclass_books,
        }
    }

    /// The floor's codebooks, the master book and those of subclasses 1 to
    /// 3, in order.
    pub fn codebooks(&self) -> Vec<Codebook> {
        let mut books = vec![self.master.clone()];
        books.extend(self.subclass_books.iter().cloned());
        books
    }

    /// `floor1_X_list`, in its order.
    pub fn xs(&self) -> &[u32] {
        &self.xs
    }

    /// Writes the floor's configuration (clause 7.2.2), after its type.
    pub fn write_setup(&self, out: &mut BitPacker) {
        out.write(self.partitions as u32, 5);
        for _ in 0..self.partitions {
            out.write(0, 4); // floor1_partition_class_list: class 0
        }
        out.write(CLASS_DIMENSIONS as u32 - 1, 3);
        out.write(SUBCLASS_BITS, 2);
        out.write(u32::from(self.first_book), 8); // floor1_class_masterbooks
        for s in 0..1 << SUBCLASS_BITS {
            // floor1_subclass_books, plus one: 0 for subclass 0's none.
            let book = if s == 0 {
                0
            } else {
                u32::from(self.first_book) + s + 1
            };
            out.write(book, 8);
        }
        out.write(MULTIPLIER - 1, 2);
        out.write(self.rangebits, 4);
        for &x in &self.xs[2..] {
            out.write(x, self.rangebits);
        }
    }

    /// The values a packet codes for the floor nearest `targets`, the Y
    /// value wanted at each post in list order, each from 0 to
    /// [`RANGE`] - 1: `floor1_Y`, as clause 7.2.4 reads it.
    ///
    /// The first two posts take their targets. Each other takes the Y value
    /// that the posts before it in the list predict there, coded as 0,
    /// where its target is within `TOLERANCE` of it, and else its target.
    pub fn values(&self, targets: &[i32]) -> Vec<u32> {
        assert_eq!(targets.len(), self.xs.len(), "a target for each post");
        assert!(
            targets.iter().all(|t| (0..RANGE).contains(t)),
            "targets {targets:?}"
        );
        let mut final_y = vec![0; self.xs.len()];
        let mut values = vec![0; self.xs.len()];
        for i in 0..self.xs.len() {
            let target = targets[i];
            if i < 2 {
                final_y[i] = target;
                values[i] = target as u32;
                continue;
            }
            let predicted = self.predict(&final_y, i);
            if (target - predicted).abs() <= TOLERANCE {
                final_y[i] = predicted;
            } else {
                final_y[i] = target;
                values[i] = wrap(target, predicted);
            }
        }
        debug_assert_eq!(self.amplitudes(&values).0, final_y);
        values
    }

    /// [`values`](Self::values) for `targets`, raised where the floor they
    /// draw is too low for `spectrum`, the lines of a block, so that no line
    /// is more than `limit` steps of the floor, the most a residue value
    /// may be; and the floor's curve, as [`curve`](Self::curve) gives it.
    ///
    /// Where a line is, the two posts about it are raised to the Y value
    /// whose floor is a `limit`th of the line, or by one where they are at
    /// it already and the floor drawn is still too low, and the floor drawn
    /// again, until no line is. A line beyond `limit` steps of the highest
    /// floor stays so.
    pub fn values_within(
        &self,
        targets: &[i32],
        spectrum: &[f64],
        limit: f64,
    ) -> (Vec<u32>, Vec<f32>) {
        let mut targets = targets.to_vec();
        loop {
            let values = self.values(&targets);
            let curve = self.curve(&values, spectrum.len());
            // The Y value each post is to be raised to, where any.
            let mut wanted: Vec<Option<i32>> = vec![None; targets.len()];
            for (k, (&x, &f)) in spectrum.iter().zip(&curve).enumerate() {
                if x.abs() > limit * f64::from(f) {
                    let need = (0..RANGE)
                        .find(|&y| {
                            limit * f64::from(INVERSE_DB_TABLE[(y * MULTIPLIER as i32) as usize])
                                >= x.abs()
                        })
                        .unwrap_or(RANGE - 1);
                    // The posts about the line, below or at it and above it.
                    let above = self
                        .sorted
                        .iter()
                        .position(|&i| self.xs[i] as usize > k)
                        .expect("the last post is past every line");
                    for i in [self.sorted[above - 1], self.sorted[above]] {
                        wanted[i] = Some(wanted[i].map_or(need, |w| w.max(need)));
                    }
                }
            }
            let mut raised = false;
            for (target, wanted) in targets.iter_mut().zip(wanted) {
                if let Some(need) = wanted {
                    let new = if *target < need { need } else { *target + 1 };
                    if new < RANGE {
                        raised |= new != *target;
                        *target = new;
                    }
                }
            }
            if !raised {
                return (values, curve);
            }
        }
    }

    /// `render_point` at post `i` from its neighbors' final Y values.
    fn predict(&self, final_y: &[i32], i: usize) -> i32 {
        let (low, high) = self.neighbors[i];
        render_point(
            self.xs[low] as i32,
            final_y[low],
            self.xs[high] as i32,
            final_y[high],
            self.xs[i] as i32,
        )
    }

    /// Writes the floor of `values`, as [`values`](Self::values) gives
    /// them (clause 7.2.3).
    pub fn write(&self, values: &[u32], out: &mut BitPacker) {
        out.write_bool(true); // nonzero
        let y_bits = ilog(RANGE as u32 - 1);
        out.write(values[0], y_bits);
        out.write(values[1], y_bits);
        for partition in values[2..].chunks(CLASS_DIMENSIONS) {
            let subclasses: Vec<usize> = partition
                .iter()
                .map(|&v| {
                    SUBCLASS_LIMITS
                        .iter()
                        .position(|&limit| v <= limit)
                        .expect("a value within the range")
                })
                .collect();
            let cval = (subclasses.iter().enumerate())
                .map(|(j, &s)| s << (SUBCLASS_BITS as usize * j))
                .sum();
            self.master.write_entry(out, cval);
            for (&v, &s) in partition.iter().zip(&subclasses) {
                if s > 0 {
                    self.subclass_books[s - 1].write_entry(out, v as usize);
                }
            }
        }
    }

    /// Step 1 of clause 7.2.4: `floor1_final_Y` and `floor1_step2_flag`
    /// from `floor1_Y`.
    fn amplitudes(&self, values: &[u32]) -> (Vec<i32>, Vec<bool>) {
        let mut final_y = vec![0; self.xs.len()];
        let mut step2 = vec![false; self.xs.len()];
        final_y[0] = values[0] as i32;
        final_y[1] = values[1] as i32;
        step2[0] = true;
        step2[1] = true;
        for i in 2..self.xs.len() {
            let (low, high) = self.neighbors[i];
            let predicted = self.predict(&final_y, i);
            let val = values[i] as i32;
            let highroom = RANGE - predicted;
            let lowroom = predicted;
            let room = if highroom < lowroom {
                highroom * 2
            } else {
                lowroom * 2
            };
            if val != 0 {
                step2[low] = true;
                step2[high] = true;
                step2[i] = true;
                final_y[i] = if val >= room {
                    if highroom > lowroom {
                        val - lowroom + predicted
                    } else {
                        predicted - val + highroom - 1
                    }
                } else if val % 2 == 1 {
                    predicted - (val + 1) / 2
                } else {
                    predicted + val / 2
                };
            } else {
                step2[i] = false;
                final_y[i] = predicted;
            }
        }
        (final_y, step2)
    }

    /// The floor curve of `values` over `lines` lines, as a decoder
    /// computes it (clause 7.2.4).
    pub fn curve(&self, values: &[u32], lines: usize) -> Vec<f32> {
        let (final_y, step2) = self.amplitudes(values);
        let multiplier = MULTIPLIER as i32;
        // Room for a line drawn to the last post, past `lines`.
        let mut floor = vec![0; lines.max(1 << self.rangebits)];
        let (mut hx, mut hy) = (0, 0);
        let mut lx = 0;
        let mut ly = final_y[self.sorted[0]] * multiplier;
        for &i in &self.sorted[1..] {
            if step2[i] {
                hy = final_y[i] * multiplier;
                hx = self.xs[i] as i32;
                render_line(lx, ly, hx, hy, &mut floor);
                lx = hx;
                ly = hy;
            }
        }
        if (hx as usize) < lines {
            render_line(hx, hy, lines as i32, hy, &mut floor);
        }
        floor.truncate(lines);
        floor
            .into_iter()
            .map(|y| INVERSE_DB_TABLE[y as usize])
            .collect()
    }
}

/// The value that codes `target` where `predicted` is predicted: the
/// inverse of step 1 of clause 7.2.4.
fn wrap(target: i32, predicted: i32) -> u32 {
    let highroom = RANGE - predicted;
    let lowroom = predicted;
    let room = 2 * highroom.min(lowroom);
    let difference = target - predicted;
    // Within the room, a rise is even and a fall odd.
    let val = if difference > 0 {
        2 * difference
    } else {
        -2 * difference - 1
    };
    let val = if val < room {
        val
    } else if highroom > lowroom {
        // Beyond the room on the side of the larger, the only side left.
        difference + lowroom
    } else {
        highroom - 1 - difference
    };
    debug_assert!((1..RANGE).contains(&val), "{target} from {predicted}");
    val as u32
}

/// The `count` posts besides 0 and 2^`rangebits`, in ascending order:
/// spaced evenly on a scale of frequency that is linear below a 64th of
/// the lines, 16 lines of a long block, and logarithmic above, so that
/// they are dense where hearing resolves frequency finely, and at least a
/// line apart.
fn posts(rangebits: u32, count: usize) -> Vec<u32> {
    let top = f64::from(1u32 << rangebits);
    let knee = top / 64.0;
    let span = (1.0 + top / knee).log2();
    let mut posts = Vec::with_capacity(count);
    let mut last = 0;
    for i in 1..=count {
        let t = i as f64 / (count + 1) as f64;
        let x = (knee * ((t * span).exp2() - 1.0)).round() as u32;
        last = x.max(last + 1);
        posts.push(last);
    }
    assert!(last < 1 << rangebits);
    posts
}

/// `floor1_X_list`: 0 and 2^`rangebits`, then `posts` in the order of
/// halving: the middle post, then the middles of the halves on each side
/// of it, and so on, breadth first, so that each post is predicted from
/// the two posts around it that came before it.
fn list_order(posts: &[u32], rangebits: u32) -> Vec<u32> {
    let mut xs = vec![0, 1 << rangebits];
    let mut ranges = std::collections::VecDeque::from([(0, posts.len())]);
    while let Some((lo, hi)) = ranges.pop_front() {
        if lo < hi {
            let mid = (lo + hi) / 2;
            xs.push(posts[mid]);
            ranges.push_back((lo, mid));
            ranges.push_back((mid + 1, hi));
        }
    }
    xs
}

/// `low_neighbor(v, x)` (clause 9.2.4): the position before `x` of the
/// greatest value less than `v[x]`.
fn low_neighbor(v: &[u32], x: usize) -> usize {
    (0..x)
        .filter(|&n| v[n] < v[x])
        .max_by_key(|&n| v[n])
        .expect("post 0 is below every other")
}

/// `high_neighbor(v, x)` (clause 9.2.5): the position before `x` of the
/// lowest value greater than `v[x]`.
fn high_neighbor(v: &[u32], x: usize) -> usize {
    (0..x)
        .filter(|&n| v[n] > v[x])
        .min_by_key(|&n| v[n])
        .expect("post 1 is above every other")
}

/// `render_point(x0, y0, x1, y1, X)` (clause 9.2.6).
fn render_point(x0: i32, y0: i32, x1: i32, y1: i32, x: i32) -> i32 {
    let dy = y1 - y0;
    let adx = x1 - x0;
    let ady = dy.abs();
    let err = ady * (x - x0);
    let off = err / adx;
    if dy < 0 { y0 - off } else { y0 + off }
}

/// `render_line(x0, y0, x1, y1, v)` (clause 9.2.7), its integer division
/// rounding toward zero, as Rust's does.
fn render_line(x0: i32, y0: i32, x1: i32, y1: i32, v: &mut [i32]) {
    let dy = y1 - y0;
    let adx = x1 - x0;
    let mut ady = dy.abs();
    let base = dy / adx;
    let mut y = y0;
    let mut err = 0;
    let sy = if dy < 0 { base - 1 } else { base + 1 };
    ady -= base.abs() * adx;
    v[x0 as usize] = y;
    for x in x0 + 1..x1 {
        err += ady;
        if err >= adx {
            err -= adx;
            y += sy;
        } else {
            y += base;
        }
        v[x as usize] = y;
    }
}

/// `floor1_inverse_dB_table` (clause 10.1), as the clause prints it.
#[rustfmt::skip]
#[allow(clippy::excessive_precision)]
pub const INVERSE_DB_TABLE: [f32; 256] = [
    1.0649863e-07, 1.1341951e-07, 1.2079015e-07, 1.2863978e-07,
    1.3699951e-07, 1.4590251e-07, 1.5538408e-07, 1.6548181e-07,
    1.7623575e-07, 1.8768855e-07, 1.9988561e-07, 2.1287530e-07,
    2.2670913e-07, 2.4144197e-07, 2.5713223e-07, 2.7384213e-07,
    2.9163793e-07, 3.1059021e-07, 3.3077411e-07, 3.5226968e-07,
    3.7516214e-07, 3.9954229e-07, 4.2550680e-07, 4.5315863e-07,
    4.8260743e-07, 5.1396998e-07, 5.4737065e-07, 5.8294187e-07,
    6.2082472e-07, 6.6116941e-07, 7.0413592e-07, 7.4989464e-07,
    7.9862701e-07, 8.5052630e-07, 9.0579828e-07, 9.6466216e-07,
    1.0273513e-06, 1.0941144e-06, 1.1652161e-06, 1.2409384e-06,
    1.3215816e-06, 1.4074654e-06, 1.4989305e-06, 1.5963394e-06,
    1.7000785e-06, 1.8105592e-06, 1.9282195e-06, 2.0535261e-06,
    2.1869758e-06, 2.3290978e-06, 2.4804557e-06, 2.6416497e-06,
    2.8133190e-06, 2.9961443e-06, 3.1908506e-06, 3.3982101e-06,
    3.6190449e-06, 3.8542308e-06, 4.1047004e-06, 4.3714470e-06,
    4.6555282e-06, 4.9580707e-06, 5.2802740e-06, 5.6234160e-06,
    5.9888572e-06, 6.3780469e-06, 6.7925283e-06, 7.2339451e-06,
    7.7040476e-06, 8.2047000e-06, 8.7378876e-06, 9.3057248e-06,
    9.9104632e-06, 1.0554501e-05, 1.1240392e-05, 1.1970856e-05,
    1.2748789e-05, 1.3577278e-05, 1.4459606e-05, 1.5399272e-05,
    1.6400004e-05, 1.7465768e-05, 1.8600792e-05, 1.9809576e-05,
    2.1096914e-05, 2.2467911e-05, 2.3928002e-05, 2.5482978e-05,
    2.7139006e-05, 2.8902651e-05, 3.0780908e-05, 3.2781225e-05,
    3.4911534e-05, 3.7180282e-05, 3.9596466e-05, 4.2169667e-05,
    4.4910090e-05, 4.7828601e-05, 5.0936773e-05, 5.4246931e-05,
    5.7772202e-05, 6.1526565e-05, 6.5524908e-05, 6.9783085e-05,
    7.4317983e-05, 7.9147585e-05, 8.4291040e-05, 8.9768747e-05,
    9.5602426e-05, 0.00010181521, 0.00010843174, 0.00011547824,
    0.00012298267, 0.00013097477, 0.00013948625, 0.00014855085,
    0.00015820453, 0.00016848555, 0.00017943469, 0.00019109536,
    0.00020351382, 0.00021673929, 0.00023082423, 0.00024582449,
    0.00026179955, 0.00027881276, 0.00029693158, 0.00031622787,
    0.00033677814, 0.00035866388, 0.00038197188, 0.00040679456,
    0.00043323036, 0.00046138411, 0.00049136745, 0.00052329927,
    0.00055730621, 0.00059352311, 0.00063209358, 0.00067317058,
    0.00071691700, 0.00076350630, 0.00081312324, 0.00086596457,
    0.00092223983, 0.00098217216, 0.0010459992, 0.0011139742,
    0.0011863665, 0.0012634633, 0.0013455702, 0.0014330129,
    0.0015261382, 0.0016253153, 0.0017309374, 0.0018434235,
    0.0019632195, 0.0020908006, 0.0022266726, 0.0023713743,
    0.0025254795, 0.0026895994, 0.0028643847, 0.0030505286,
    0.0032487691, 0.0034598925, 0.0036847358, 0.0039241906,
    0.0041792066, 0.0044507950, 0.0047400328, 0.0050480668,
    0.0053761186, 0.0057254891, 0.0060975636, 0.0064938176,
    0.0069158225, 0.0073652516, 0.0078438871, 0.0083536271,
    0.0088964928, 0.009474637, 0.010090352, 0.010746080,
    0.011444421, 0.012188144, 0.012980198, 0.013823725,
    0.014722068, 0.015678791, 0.016697687, 0.017782797,
    0.018938423, 0.020169149, 0.021479854, 0.022875735,
    0.024362330, 0.025945531, 0.027631618, 0.029427276,
    0.031339626, 0.033376252, 0.035545228, 0.037855157,
    0.040315199, 0.042935108, 0.045725273, 0.048696758,
    0.051861348, 0.055231591, 0.058820850, 0.062643361,
    0.066714279, 0.071049749, 0.075666962, 0.080584227,
    0.085821044, 0.091398179, 0.097337747, 0.10366330,
    0.11039993, 0.11757434, 0.12521498, 0.13335215,
    0.14201813, 0.15124727, 0.16107617, 0.17154380,
    0.18269168, 0.19456402, 0.20720788, 0.22067342,
    0.23501402, 0.25028656, 0.26655159, 0.28387361,
    0.30232132, 0.32196786, 0.34289114, 0.36517414,
    0.38890521, 0.41417847, 0.44109412, 0.46975890,
    0.50028648, 0.53279791, 0.56742212, 0.60429640,
    0.64356699, 0.68538959, 0.72993007, 0.77736504,
    0.82788260, 0.88168307, 0.9389798, 1.0,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_subclass_has_a_frequency_for_each_value_it_codes() {
        let counts: [&[u32]; 3] = [
            &frequencies::FLOOR_SUBCLASS_1,
            &frequencies::FLOOR_SUBCLASS_2,
            &frequencies::FLOOR_SUBCLASS_3,
        ];
        for (s, counts) in counts.iter().enumerate() {
            let range = SUBCLASS_LIMITS[s + 1] - SUBCLASS_LIMITS[s];
            assert_eq!(counts.len(), range as usize, "subclass {}", s + 1);
        }
        assert_eq!(
            frequencies::FLOOR_CLASSES.len(),
            1 << (SUBCLASS_BITS * CLASS_DIMENSIONS as u32)
        );
    }

    #[test]
    fn the_posts_are_unique_and_within_the_range() {
        for (floor, lines, partitions) in [
            (Floor::long(0), 1024, LONG_PARTITIONS),
            (Floor::short(0), 128, SHORT_PARTITIONS),
        ] {
            let xs = floor.xs();
            assert_eq!(xs.len(), 2 + partitions * CLASS_DIMENSIONS);
            assert!(xs.len() <= 65, "clause 7.2.2 allows 65");
            let mut sorted = xs.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), xs.len());
            assert_eq!((sorted[0], *sorted.last().unwrap()), (0, lines));
            assert_eq!(
                floor.sorted.iter().map(|&i| xs[i]).collect::<Vec<_>>(),
                sorted
            );
        }
    }

    #[test]
    fn neighbors_are_those_of_clause_9_2() {
        // The example list order of clause 7.2.1.
        let v = [0, 128, 64, 32, 96, 16, 48, 80, 112];
        assert_eq!((low_neighbor(&v, 2), high_neighbor(&v, 2)), (0, 1));
        assert_eq!((low_neighbor(&v, 4), high_neighbor(&v, 4)), (2, 1));
        assert_eq!((low_neighbor(&v, 6), high_neighbor(&v, 6)), (3, 2));
        assert_eq!((low_neighbor(&v, 8), high_neighbor(&v, 8)), (4, 1));
    }

    #[test]
    fn render_point_and_line_agree_on_the_line() {
        for (x0, y0, x1, y1) in [
            (0, 10, 7, 100),
            (3, 90, 40, 2),
            (5, 7, 6, 7),
            (0, 0, 1024, 254),
        ] {
            let mut v = vec![0; 1100];
            render_line(x0, y0, x1, y1, &mut v);
            for x in x0..x1 {
                assert_eq!(v[x as usize], render_point(x0, y0, x1, y1, x), "{x}");
            }
        }
    }

    #[test]
    fn values_code_their_targets_to_within_the_tolerance() {
        let floor = Floor::long(0);
        let n = floor.xs().len();
        for seed in 0..50u32 {
            let targets: Vec<i32> = (0..n)
                .map(|i| ((i as u32).wrapping_mul(2_654_435_761) ^ (seed * 40_503)) % RANGE as u32)
                .map(|y| y as i32)
                .collect();
            let values = floor.values(&targets);
            let (final_y, _) = floor.amplitudes(&values);
            for (i, (&f, &t)) in final_y.iter().zip(&targets).enumerate() {
                assert!((f - t).abs() <= TOLERANCE, "post {i}: {f} for {t}");
            }
            assert!(values.iter().all(|&v| v < RANGE as u32));
        }
    }

    #[test]
    fn values_wrap_back_through_step_1() {
        let floor = Floor::long(0);
        for predicted in 0..RANGE {
            for target in 0..RANGE {
                if target != predicted {
                    // Post 2 is predicted from posts 0 and 1, of the same Y.
                    let mut values = vec![0; floor.xs().len()];
                    values[0] = predicted as u32;
                    values[1] = predicted as u32;
                    values[2] = wrap(target, predicted);
                    assert_eq!(
                        floor.amplitudes(&values).0[2],
                        target,
                        "{target} from {predicted}"
                    );
                }
            }
        }
    }

    #[test]
    fn values_within_raise_the_floor_under_loud_lines() {
        for (floor, lines) in [(Floor::long(0), 1024), (Floor::short(0), 128)] {
            let targets = vec![0; floor.xs().len()];
            let mut spectrum = vec![1e-7; lines];
            for (k, line) in spectrum.iter_mut().enumerate().step_by(37) {
                *line = if k % 2 == 0 { 0.9 } else { -0.02 };
            }
            let limit = 4479.0;
            let (values, curve) = floor.values_within(&targets, &spectrum, limit);
            assert_eq!(curve, floor.curve(&values, lines));
            for (k, (&x, &f)) in spectrum.iter().zip(&curve).enumerate() {
                assert!(x.abs() <= limit * f64::from(f), "line {k}: {x} over {f}");
            }
            // Where no line is too loud, the floor is the targets'.
            let quiet = vec![1e-9; lines];
            let (values, _) = floor.values_within(&targets, &quiet, limit);
            assert_eq!(values, floor.values(&targets));
        }
    }

    #[test]
    fn a_flat_floor_is_flat() {
        let floor = Floor::long(0);
        let mut values = vec![0; floor.xs().len()];
        values[0] = 100;
        values[1] = 100;
        let curve = floor.curve(&values, 1024);
        assert_eq!(curve.len(), 1024);
        assert!(curve.iter().all(|&y| y == INVERSE_DB_TABLE[200]));
        let short = Floor::short(0);
        let mut values = vec![0; short.xs().len()];
        values[0] = 100;
        values[1] = 100;
        let curve = short.curve(&values, 128);
        assert_eq!(curve.len(), 128);
        assert!(curve.iter().all(|&y| y == INVERSE_DB_TABLE[200]));
    }

    #[test]
    fn the_table_rises_by_a_constant_ratio_to_one() {
        assert_eq!(INVERSE_DB_TABLE[255], 1.0);
        for pair in INVERSE_DB_TABLE.windows(2) {
            let ratio = pair[1] / pair[0];
            assert!((ratio - 1.064_986).abs() < 1e-5, "{pair:?}");
        }
    }
}
