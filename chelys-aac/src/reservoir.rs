// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The bit reservoir of a stream at a constant rate (clause 8.2.3.2): how
//! far each frame may fall short of the mean length, or run past it, and
//! the state ADTS reports as `adts_buffer_fullness`.

use super::channel_configuration::{ChannelConfiguration, ElementKind};
use super::sampling_frequency::SamplingFrequency;
use super::stream::FRAME_SAMPLES;

/// The bits of the decoder's input buffer per channel of the NCC (clause
/// 8.2.3.1).
pub const BUFFER_BITS_PER_CHANNEL: u64 = 6144;

/// The bits of the decoder's input buffer of a stream of
/// `channel_configuration`: [`BUFFER_BITS_PER_CHANNEL`] for each channel
/// of its NCC.
pub fn buffer_bits(channel_configuration: ChannelConfiguration) -> u32 {
    (BUFFER_BITS_PER_CHANNEL * channel_configuration.ncc() as u64) as u32
}

/// The longest `adts_frame()` written, in bytes, its header and
/// `adts_error_check()` included, as the reservoir counts them (see
/// [`BitReservoir::new`]).
///
/// Spec reading: clauses 8.2.3.1 and 8.2.3.3 let a frame be as long as
/// the decoder's input buffer, 6144 bits for each channel of the NCC,
/// which is longer than 4608 bytes only in 7.1 (configuration 7, NCC 7),
/// at 5376 bytes. FAAD2 2.11.4 (black box) decodes no ADTS frame longer
/// than 4608 bytes, where ffmpeg does: it reports an error at the first
/// and decodes nothing from it on. So a stream in ADTS is kept to them, a
/// stricter subset of what the standard allows, as `ms_mask_present` 1 is
/// sent for ffmpeg where 2 would do (see
/// [`crate::stereo::middle_side::choose`]). Only the longest frame, and
/// with it the highest bitrate, are lowered: the buffer,
/// `max_bit_reservoir`, `adts_buffer_fullness` and each element's share of
/// the buffer (see [`element_max_bits`]) stay the standard's. Raw frames
/// keep the standard's limits. The conformance tests are
/// `faad_decodes_no_frame_longer_than_4608_bytes`, which finds FAAD2's
/// limit, and `faad_decodes_5_1_and_7_1_at_the_ends_of_their_range`, in
/// which FAAD2 decodes every stream of 7.1 at its highest bitrates.
pub const MAX_ADTS_FRAME_LEN: usize = 4608;

/// The most bits an element of `kind` may take in a frame, after its
/// `id_syn_ele`: those of its own share of the decoder's input buffer,
/// 6144 for a single channel and 12288 for a pair, less the 3 of its
/// `id_syn_ele`; `None` for a low frequency enhancement channel, whose bits
/// come from the whole buffer's.
///
/// Spec reading: clause 8.2.3.1 limits "the individual buffer sizes" of
/// the elements as well as the whole buffer's, "permitting the decoder to
/// break a multichannel bitstream into separate mono and stereo bitstreams
/// which are decoded by separate mono and stereo decoders"; "All bits for
/// LFE's ... must be supplied from the total buffer requirements". It does
/// not say at what rate such a decoder's buffer fills, so the limit is
/// taken frame by frame: an element is never longer than its buffer. Nor
/// does it say whether the element's `id_syn_ele`, which a decoder that
/// breaks the stream apart reads to do so, is counted: it is, which is the
/// stricter reading. With one element, the frame, its header among its
/// bits, is no longer than the whole buffer, so that the limit holds of
/// itself.
pub fn element_max_bits(kind: ElementKind) -> Option<u32> {
    match kind {
        ElementKind::Lfe => None,
        kind => Some(BUFFER_BITS_PER_CHANNEL as u32 * kind.channels() as u32 - 3),
    }
}

/// The state of the bit reservoir, kept exactly: in bits times the
/// sampling frequency, as the mean length of a frame is a fraction of a
/// bit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitReservoir {
    sample_rate: u64,
    /// The NCC (see [`ChannelConfiguration::ncc`]).
    ncc: u64,
    /// `mean_framelength`, times the sampling frequency.
    mean: u64,
    /// `max_bit_reservoir`, times the sampling frequency.
    max: u64,
    /// `bit_reservoir_state`, times the sampling frequency.
    state: u64,
    /// The longest frame in bytes: [`MAX_ADTS_FRAME_LEN`] in ADTS, and
    /// `u64::MAX` for raw frames, which only the buffer limits.
    max_frame_len: u64,
}

impl BitReservoir {
    /// The reservoir of a stream of `channel_configuration` at
    /// `bits_per_second`, its buffer sized by the NCC, and its frames no
    /// longer than `max_frame_len` bytes if given, if a frame can be given
    /// its mean length within a byte: if `max_bit_reservoir` is 8 bits at
    /// least, and the mean is a byte shorter than `max_frame_len` at least.
    ///
    /// Spec reading: the maximum bitrate (clause 8.2.3.3) is that of a
    /// frame as long as the decoder's input buffer, where
    /// `max_bit_reservoir` is 0 and every frame must be exactly the mean
    /// length, which is a fraction of a bit where the rate does not divide
    /// evenly. Frames are whole bytes, so the reservoir is kept a byte at
    /// least, and the highest bitrate is below the maximum by a byte a
    /// frame. The same byte is kept below `max_frame_len`, which lowers
    /// the highest bitrate of 7.1 in ADTS (see [`MAX_ADTS_FRAME_LEN`]).
    ///
    /// The mean is that of the stream as it is written: a stream in ADTS
    /// counts its headers, as the decoder's input buffer holds them too
    /// (clause 8.2.3.1).
    ///
    /// Spec gap: the standard does not give the state before the first
    /// frame. It is taken as empty, so that the first frame is no longer
    /// than the mean, and no frame needs more bits than the rate has
    /// brought by its end.
    pub fn new(
        sampling_frequency: SamplingFrequency,
        channel_configuration: ChannelConfiguration,
        bits_per_second: u32,
        max_frame_len: Option<usize>,
    ) -> Option<Self> {
        let sample_rate = u64::from(sampling_frequency.hz());
        let ncc = channel_configuration.ncc() as u64;
        let mean = u64::from(bits_per_second) * FRAME_SAMPLES as u64;
        let buffer = BUFFER_BITS_PER_CHANNEL * ncc * sample_rate;
        // max_bit_reservoir = INT(buffer - mean_framelength), as in the
        // example of clause 8.2.3.2.
        let max_bits = buffer.checked_sub(mean)? / sample_rate;
        let fits =
            max_frame_len.is_none_or(|len| mean + 8 * sample_rate <= 8 * len as u64 * sample_rate);
        let max_frame_len = max_frame_len.map_or(u64::MAX, |len| len as u64);
        (max_bits >= 8 && fits).then_some(Self {
            sample_rate,
            ncc,
            mean,
            max: max_bits * sample_rate,
            state: 0,
            max_frame_len,
        })
    }

    /// The highest bitrate that [`Self::new`] takes: that of the buffer,
    /// or of frames of `max_frame_len` bytes where they are shorter, less
    /// a byte.
    pub fn max_bits_per_second(
        sampling_frequency: SamplingFrequency,
        channel_configuration: ChannelConfiguration,
        max_frame_len: Option<usize>,
    ) -> u32 {
        let buffer = BUFFER_BITS_PER_CHANNEL * channel_configuration.ncc() as u64;
        let longest = max_frame_len.map_or(buffer, |len| buffer.min(8 * len as u64));
        let sample_rate = u64::from(sampling_frequency.hz());
        ((longest - 8) * sample_rate / FRAME_SAMPLES as u64) as u32
    }

    /// The lengths in bytes the next frame may have, from the shortest
    /// that keeps the reservoir within `max_bit_reservoir` to the longest
    /// that keeps it from running below empty (clause 8.2.3.2), and is no
    /// longer than the longest frame given to [`Self::new`]:
    ///
    /// `0 ≤ bit_reservoir_state[frame] ≤ max_bit_reservoir`, where
    /// `bit_reservoir_state[frame] = bit_reservoir_state[frame - 1] +
    /// mean_framelength - framelength[frame]`.
    ///
    /// The shortest is never longer than the longest: as the reservoir is
    /// within `max_bit_reservoir`, the shortest is a mean frame at most,
    /// rounded up to a byte, which [`Self::new`] keeps a byte shorter than
    /// the longest frame given.
    pub fn frame_len_range(&self) -> (usize, usize) {
        let available = self.state + self.mean;
        let byte = 8 * self.sample_rate;
        let shortest = available.saturating_sub(self.max).div_ceil(byte);
        let longest = (available / byte).min(self.max_frame_len);
        (shortest as usize, longest as usize)
    }

    /// `available_bits` (clause C.7.3): the bits the next frame may take,
    /// given `more_bits`, the bits beyond the mean that the psychoacoustic
    /// model asks for it:
    ///
    /// `available_bits = mean_bits + min(more_bits,
    /// bit_reservoir_state[frame])` if `more_bits > 0`, and `mean_bits +
    /// max(more_bits, bit_reservoir_state[frame] - max_bit_reservoir)` if
    /// `more_bits < 0`;
    ///
    /// in whole bits, and no more than the longest frame that
    /// [`Self::frame_len_range`] allows, in its whole bytes.
    ///
    /// Spec reading: clause C.7.3 leaves out `more_bits = 0`, for which
    /// both of its formulas give `mean_bits`, as the state is within 0 and
    /// `max_bit_reservoir`; that is taken.
    ///
    /// Spec reading: `bit_reservoir_state[frame]` is the state after the
    /// frame (clause 8.2.3.2), which is not known until the frame is coded
    /// with the bits made available to it. The state before it, after the
    /// frame before, is taken: the bits the reservoir holds for it. The
    /// bits are floored to whole bits, and to the frame's whole bytes, so
    /// that a frame that takes them is one [`Self::spend`] takes.
    pub fn available_bits(&self, more_bits: i64) -> u32 {
        let scale = self.sample_rate as i128;
        let more = i128::from(more_bits) * scale;
        let (state, max) = (self.state as i128, self.max as i128);
        let extra = match more_bits {
            1.. => more.min(state),
            0 => 0,
            ..0 => more.max(state - max),
        };
        let bits = (self.mean as i128 + extra).div_euclid(scale);
        let (_, longest) = self.frame_len_range();
        bits.clamp(0, 8 * longest as i128) as u32
    }

    /// The bits the next frame is offered whatever it asks for: the mean,
    /// and a quarter of what the reservoir holds beyond `max_bit_reservoir`
    /// less half of `mean_framelength`, or beyond half of
    /// `max_bit_reservoir` where that is more, in whole bits, and no more
    /// than the longest frame that [`Self::frame_len_range`] allows, in its
    /// whole bytes. Frames that take them leave the reservoir within half a
    /// mean frame of its maximum, and at least half full.
    pub fn offered_bits(&self) -> u32 {
        let keep = (self.max / 2).max(self.max.saturating_sub(self.mean / 2));
        let beyond = self.state.saturating_sub(keep);
        let bits = (self.mean + beyond / 4) / self.sample_rate;
        let (_, longest) = self.frame_len_range();
        bits.min(8 * longest as u64) as u32
    }

    /// Takes a frame of `len` bytes, within [`Self::frame_len_range`].
    pub fn spend(&mut self, len: usize) {
        let (shortest, longest) = self.frame_len_range();
        assert!(
            (shortest..=longest).contains(&len),
            "a frame of {len} bytes, not {shortest} to {longest}"
        );
        self.state = self.state + self.mean - 8 * len as u64 * self.sample_rate;
    }

    /// The state after the last frame spent, in whole bits.
    #[cfg(any(test, feature = "unstable"))]
    pub fn state_bits(&self) -> u64 {
        self.state / self.sample_rate
    }

    /// `max_bit_reservoir`.
    #[cfg(any(test, feature = "unstable"))]
    pub fn max_bits(&self) -> u64 {
        self.max / self.sample_rate
    }

    /// `adts_buffer_fullness`: the state "divided by NCC divided by 32 and
    /// truncated to an integer value" (clause 8.1.3.2.2).
    ///
    /// Spec reading: the state is that after the frame the header is in,
    /// its bits spent: "the state of the bit reservoir in the course of
    /// encoding the ADTS frame, up to and including the first
    /// raw_data_block()", as `adif_buffer_fullness` is "the state of the
    /// bit reservoir after encoding the first raw_data_block()".
    pub fn adts_buffer_fullness(&self) -> u32 {
        (self.state / self.sample_rate / self.ncc / 32) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ChannelConfiguration::{FiveOne, Mono, SevenOne, Stereo};
    use SamplingFrequency::{Hz8000, Hz44100, Hz48000, Hz96000};

    #[test]
    fn the_maximum_follows_the_example_of_the_standard() {
        // 96 kbit/s, stereo, 44.1 kHz: mean_framelength 2229.1156...,
        // max_bit_reservoir 10058 (clause 8.2.3.2).
        let reservoir = BitReservoir::new(Hz44100, Stereo, 96_000, None).unwrap();
        assert_eq!(reservoir.max_bits(), 10_058);
        assert_eq!(reservoir.frame_len_range(), (0, 278));
    }

    #[test]
    fn short_frames_fill_the_reservoir_to_its_maximum() {
        let mut reservoir = BitReservoir::new(Hz48000, Mono, 64_000, None).unwrap();
        // mean_framelength 1365.33... bits.
        let mut total = 0;
        for _ in 0..100 {
            let (shortest, _) = reservoir.frame_len_range();
            let len = shortest.max(10);
            reservoir.spend(len);
            total += 8 * len as u64;
            assert!(reservoir.state_bits() <= reservoir.max_bits());
        }
        let mean_total = 100 * 64_000 * 1024 / 48_000;
        assert_eq!(total + reservoir.state_bits(), mean_total);
        assert!(reservoir.max_bits() - reservoir.state_bits() <= 8);
        assert_eq!(
            reservoir.adts_buffer_fullness(),
            (reservoir.state_bits() / 32) as u32
        );
    }

    #[test]
    fn available_bits_follow_more_bits_within_the_reservoir() {
        // mean_framelength 2229.1156... bits, max_bit_reservoir 10058.
        let mut reservoir = BitReservoir::new(Hz44100, Stereo, 96_000, None).unwrap();
        // Empty, the frame may be at most 278 bytes long: 2224 bits.
        assert_eq!(reservoir.available_bits(0), 2224);
        assert_eq!(reservoir.available_bits(5000), 2224);
        assert_eq!(reservoir.available_bits(-100), 2129);
        for _ in 0..5 {
            reservoir.spend(100);
        }
        // 5 * (2229.1156... - 800) bits.
        assert_eq!(reservoir.state_bits(), 7145);
        assert_eq!(reservoir.available_bits(0), 2229);
        assert_eq!(reservoir.available_bits(1000), 3229);
        // All of the reservoir: 9374.69... bits, in whole bytes.
        assert_eq!(reservoir.available_bits(1_000_000), 8 * 1171);
        assert_eq!(reservoir.available_bits(-1000), 1229);
        // Negative more_bits lower it only as far as the reservoir can
        // take what the frame leaves.
        while reservoir.max_bits() - reservoir.state_bits() > 300 {
            let (shortest, _) = reservoir.frame_len_range();
            reservoir.spend(shortest.max(10));
        }
        let room = reservoir.max_bits() as i64 - reservoir.state_bits() as i64;
        let available = i64::from(reservoir.available_bits(-1000));
        assert!(
            (available - (2229 - 1000 + 1000 - room)).abs() <= 1,
            "{available}"
        );
    }

    #[test]
    fn the_bits_offered_are_a_quarter_of_those_beyond_what_is_kept() {
        // mean_framelength 2229.1156... bits, max_bit_reservoir 10058: what
        // lies beyond 8943.44... bits.
        let mut reservoir = BitReservoir::new(Hz44100, Stereo, 96_000, None).unwrap();
        // Empty, the frame is offered the most it may take: 278 bytes.
        assert_eq!(reservoir.offered_bits(), 2224);
        for _ in 0..6 {
            reservoir.spend(100);
        }
        // 8574.69... bits: the mean.
        assert_eq!(reservoir.offered_bits(), 2229);
        // Full, the mean and a quarter of what lies beyond; frames that take
        // them, in whole bytes, bring the reservoir down to what is kept,
        // and no further.
        while reservoir.max_bits() - reservoir.state_bits() > 8 {
            let (shortest, _) = reservoir.frame_len_range();
            reservoir.spend(shortest.max(10));
        }
        let beyond = reservoir.state_bits() as f64 - 8943.44;
        let offered = f64::from(reservoir.offered_bits());
        assert!(
            (offered - (2229.1156 + beyond / 4.0)).abs() <= 1.0,
            "{offered}"
        );
        for _ in 0..100 {
            reservoir.spend(reservoir.offered_bits() as usize / 8);
            assert!(reservoir.state_bits() >= 8943, "{}", reservoir.state_bits());
        }
        assert!(
            reservoir.state_bits() <= 8943 + 40,
            "{}",
            reservoir.state_bits()
        );
        // Where the maximum less half a mean frame is less than half of
        // the maximum, half of it is kept: in mono at 192 kbit/s, whose
        // maximum of 1685 bits is less than half a mean frame, 842.
        let mut reservoir = BitReservoir::new(Hz44100, Mono, 192_000, None).unwrap();
        assert_eq!(reservoir.max_bits(), 1685);
        while reservoir.max_bits() - reservoir.state_bits() > 8 {
            let (shortest, _) = reservoir.frame_len_range();
            reservoir.spend(shortest);
        }
        for _ in 0..100 {
            reservoir.spend(reservoir.offered_bits() as usize / 8);
            assert!(reservoir.state_bits() >= 842, "{}", reservoir.state_bits());
        }
    }

    #[test]
    fn bitrates_up_to_the_buffer_less_a_byte() {
        for (fs, channels) in [(Hz8000, Mono), (Hz44100, Stereo), (Hz96000, Stereo)] {
            let max = BitReservoir::max_bits_per_second(fs, channels, None);
            let reservoir = BitReservoir::new(fs, channels, max, None).unwrap();
            assert!(reservoir.max_bits() >= 8);
            assert!(BitReservoir::new(fs, channels, max + 1, None).is_none());
            let (shortest, longest) = reservoir.frame_len_range();
            assert!(shortest <= longest);
        }
        assert!(BitReservoir::new(Hz8000, Mono, u32::MAX, None).is_none());
    }

    /// Spends the shortest frames the reservoir allows, filling it, and
    /// gives the range of each frame's lengths.
    fn fill(reservoir: &mut BitReservoir) -> Vec<(usize, usize)> {
        (0..40)
            .map(|_| {
                let range = reservoir.frame_len_range();
                reservoir.spend(range.0);
                range
            })
            .collect()
    }

    #[test]
    fn adts_frames_of_7_1_are_no_longer_than_4608_bytes() {
        let longest = Some(MAX_ADTS_FRAME_LEN);
        for fs in [Hz8000, Hz48000, Hz96000] {
            let hz = u64::from(fs.hz());
            let max = BitReservoir::max_bits_per_second(fs, SevenOne, longest);
            assert_eq!(u64::from(max), (8 * 4608 - 8) * hz / 1024);
            assert!(BitReservoir::new(fs, SevenOne, max + 1, longest).is_none());
            // At the highest bitrate and at one where the buffer, 5376
            // bytes, would let a frame be longer once the reservoir fills.
            for bits_per_second in [max, max / 3] {
                let mut reservoir =
                    BitReservoir::new(fs, SevenOne, bits_per_second, longest).unwrap();
                // The standard's reservoir, which the cap leaves as it is.
                let buffer = 6144 * 7 - (u64::from(bits_per_second) * 1024).div_ceil(hz);
                assert_eq!(reservoir.max_bits(), buffer);
                for (shortest, longest) in fill(&mut reservoir) {
                    assert!(
                        shortest <= longest && longest <= 4608,
                        "{shortest} {longest}"
                    );
                }
                assert!(reservoir.max_bits() - reservoir.state_bits() <= 8);
                assert_eq!(reservoir.frame_len_range().1, 4608);
                assert!(reservoir.available_bits(1_000_000) <= 36_864);
                assert!(reservoir.offered_bits() <= 36_864);
                let mut raw = BitReservoir::new(fs, SevenOne, bits_per_second, None).unwrap();
                fill(&mut raw);
                assert!(raw.frame_len_range().1 > 4608);
            }
        }
    }

    #[test]
    fn the_cap_leaves_raw_7_1_and_adts_5_1_as_they_were() {
        for fs in [Hz8000, Hz44100, Hz48000, Hz96000] {
            let hz = u64::from(fs.hz());
            let raw = BitReservoir::max_bits_per_second(fs, SevenOne, None);
            assert_eq!(u64::from(raw), (6144 * 7 - 8) * hz / 1024);
            assert!(BitReservoir::new(fs, SevenOne, raw, None).is_some());
            let adts = BitReservoir::max_bits_per_second(fs, FiveOne, Some(MAX_ADTS_FRAME_LEN));
            assert_eq!(adts, BitReservoir::max_bits_per_second(fs, FiveOne, None));
            assert_eq!(u64::from(adts), (6144 * 5 - 8) * hz / 1024);
            for bits_per_second in [adts, adts / 3] {
                let [mut capped, mut uncapped] = [Some(MAX_ADTS_FRAME_LEN), None].map(|longest| {
                    BitReservoir::new(fs, FiveOne, bits_per_second, longest).unwrap()
                });
                assert_eq!(fill(&mut capped), fill(&mut uncapped));
                assert_eq!(
                    capped.available_bits(1_000_000),
                    uncapped.available_bits(1_000_000)
                );
                assert_eq!(capped.offered_bits(), uncapped.offered_bits());
            }
        }
    }
}
