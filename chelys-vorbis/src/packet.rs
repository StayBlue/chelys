// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Where an encoder gives the audio packets it codes.

/// An audio packet, as an [`Encoder`](super::Encoder) gives it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Packet<'a> {
    /// The packet's bytes.
    pub data: &'a [u8],
    /// The samples of each channel that a decoder has returned once it has
    /// decoded this packet and those before it: the position, counted from
    /// the first sample of the input, that a container such as Ogg gives
    /// the end of the packet (Appendix A.2). The first packet returns none,
    /// and the last returns up to the end of the input and no further, so
    /// that the last packet's is the length of the input.
    pub granule_position: u64,
}

/// Where an encoder gives the audio packets it codes, one call each, in
/// their order.
pub trait PacketSink {
    /// Takes the next packet.
    fn packet(&mut self, packet: Packet<'_>);
}

/// Calls the closure with each packet, as to keep them:
///
/// ```
/// # use chelys_vorbis::{Config, Encoder, Packet};
/// let mut encoder = Encoder::new(Config::new(48_000, 1)).unwrap();
/// let mut packets = Vec::new();
/// let mut out = |packet: Packet<'_>| packets.push((packet.data.to_vec(), packet.granule_position));
/// encoder.encode_interleaved(&[0.0f32; 4096], &mut out).unwrap();
/// encoder.finish(&mut out);
/// let positions: Vec<u64> = packets.iter().map(|(_, position)| *position).collect();
/// assert_eq!(positions, [0, 1024, 2048, 3072, 4096]);
/// ```
impl<F: FnMut(Packet<'_>)> PacketSink for F {
    fn packet(&mut self, packet: Packet<'_>) {
        self(packet);
    }
}
