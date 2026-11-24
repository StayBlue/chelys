// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Where an encoder gives the stream it codes.

/// Where an encoder gives the stream it codes, in pieces, one call each,
/// in their order. The pieces put together are the stream; each codec says
/// what a piece of its stream is, and whether one can be taken on its own.
pub trait FrameSink {
    /// Takes the next piece of the stream.
    fn frame(&mut self, frame: &[u8]);
}

/// Appends each piece, so that the vector holds the stream.
impl FrameSink for Vec<u8> {
    fn frame(&mut self, frame: &[u8]) {
        self.extend_from_slice(frame);
    }
}

/// Calls the closure with each piece, as to keep them apart:
///
/// ```
/// # use chelys_core::FrameSink;
/// let mut pieces = Vec::new();
/// let mut out = |piece: &[u8]| pieces.push(piece.to_vec());
/// out.frame(&[1, 2]);
/// out.frame(&[3]);
/// assert_eq!(pieces, [vec![1, 2], vec![3]]);
/// ```
impl<F: FnMut(&[u8])> FrameSink for F {
    fn frame(&mut self, frame: &[u8]) {
        self(frame);
    }
}
