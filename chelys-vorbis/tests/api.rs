// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! The public API: what it accepts, and that its ways of giving the same
//! input give the same stream.

use std::panic::{AssertUnwindSafe, catch_unwind};

use chelys_vorbis::{Config, ConfigError, Encoder, MAX_CHANNELS, NonFiniteSample, Packet, Rate};

/// The rate of every stream: the psychoacoustic model's allowance as it is.
const RATE: Rate = Rate::Variable {
    noise_offset_db: 0.0,
};

/// A tone in each channel, interleaved.
fn pcm(len: usize, channels: usize) -> Vec<f32> {
    (0..len * channels)
        .map(|i| 0.5 * ((i / channels) as f32 * 0.05 + i as f32).sin())
        .collect()
}

/// Each packet given, with its granule position.
type Packets = Vec<(Vec<u8>, u64)>;

fn keep(packets: &mut Packets) -> impl FnMut(Packet<'_>) + '_ {
    |packet| packets.push((packet.data.to_vec(), packet.granule_position))
}

fn encode_whole(config: Config, pcm: &[f32]) -> Packets {
    let mut encoder = Encoder::new(config).unwrap();
    let mut packets = Vec::new();
    encoder
        .encode_interleaved(pcm, &mut keep(&mut packets))
        .unwrap();
    encoder.finish(&mut keep(&mut packets));
    packets
}

fn panics(f: impl FnOnce()) -> bool {
    catch_unwind(AssertUnwindSafe(f)).is_err()
}

#[test]
fn every_channel_count_up_to_eight_is_taken() {
    for channels in 1..=MAX_CHANNELS {
        for sample_rate in [1, 8_000, 44_100, 192_000, u32::MAX] {
            let config = Config::new(sample_rate, channels, RATE);
            assert_eq!(config.check(), Ok(()), "{sample_rate} Hz, {channels}");
            let encoder = Encoder::new(config).unwrap();
            assert_eq!(encoder.channels(), channels);
            assert_eq!(encoder.sample_rate(), sample_rate);
        }
    }
}

#[test]
fn errors_name_what_was_refused() {
    assert_eq!(
        Encoder::new(Config::new(0, 1, RATE)).err(),
        Some(ConfigError::UnsupportedSampleRate { sample_rate: 0 })
    );
    for channels in [0, 9, 255, usize::MAX] {
        assert_eq!(
            Encoder::new(Config::new(44_100, channels, RATE)).err(),
            Some(ConfigError::UnsupportedChannels { channels })
        );
    }
    for noise_offset_db in [f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            Encoder::new(Config::new(44_100, 2, Rate::Variable { noise_offset_db })).err(),
            Some(ConfigError::NoiseOffset { noise_offset_db })
        );
    }
    let nan = Rate::Variable {
        noise_offset_db: f64::NAN,
    };
    assert!(matches!(
        Encoder::new(Config::new(44_100, 2, nan)).err(),
        Some(ConfigError::NoiseOffset { noise_offset_db }) if noise_offset_db.is_nan()
    ));
    for level in [101, u8::MAX] {
        assert_eq!(
            Encoder::new(Config::new(44_100, 2, Rate::Quality { level })).err(),
            Some(ConfigError::QualityLevel { level })
        );
    }
}

/// A higher level allows less noise, and so never gives a shorter stream.
#[test]
fn a_higher_level_gives_no_shorter_stream() {
    let pcm = pcm(20_000, 2);
    let mut last = 0;
    for level in [0, 10, 25, 50, 75, 90, 100] {
        let packets = encode_whole(Config::new(44_100, 2, Rate::Quality { level }), &pcm);
        let bytes: usize = packets.iter().map(|(data, _)| data.len()).sum();
        assert!(bytes >= last, "level {level}: {bytes} bytes, below {last}");
        last = bytes;
    }
}

#[test]
fn the_headers_describe_the_stream() {
    let headers = Encoder::new(Config::new(32_000, 6, RATE))
        .unwrap()
        .headers();
    assert_eq!(&headers.identification[..7], b"\x01vorbis");
    assert_eq!(headers.identification[11], 6);
    assert_eq!(&headers.identification[12..16], 32_000u32.to_le_bytes());
    assert_eq!(&headers.comment[..7], b"\x03vorbis");
    assert_eq!(&headers.setup[..7], b"\x05vorbis");
}

/// Of a steady signal, every block is long.
#[test]
fn granule_positions_advance_a_hop_a_packet_and_end_at_the_input() {
    for (samples, expected) in [
        (0, vec![0]),
        (1, vec![0, 1]),
        (1024, vec![0, 1024]),
        (1025, vec![0, 1024, 1025]),
        (3000, vec![0, 1024, 2048, 3000]),
    ] {
        let packets = encode_whole(Config::new(48_000, 2, RATE), &vec![0.0; 2 * samples]);
        let positions: Vec<u64> = packets.iter().map(|&(_, position)| position).collect();
        assert_eq!(positions, expected, "{samples} samples");
    }
}

/// The mode of an audio packet, short or long (clause 4.3.1), and for a
/// long block its window flags.
fn mode(packet: &[u8]) -> (bool, Option<(bool, bool)>) {
    let bit = |i: u32| (packet[0] >> i) & 1 == 1;
    let long = bit(1);
    (long, long.then(|| (bit(2), bit(3))))
}

#[test]
fn a_packet_is_given_once_its_block_and_the_next_hop_have_arrived() {
    let mut encoder = Encoder::new(Config::new(48_000, 1, RATE)).unwrap();
    let mut packets = Vec::new();
    encoder
        .encode_interleaved(&[0.0f32; 2047], &mut keep(&mut packets))
        .unwrap();
    assert!(packets.is_empty());
    // The first block spans the 1024 samples before the input and its
    // first 1024, and the encoder looks a hop further, to whether the next
    // block has an attack.
    encoder
        .encode_interleaved(&[0.0f32], &mut keep(&mut packets))
        .unwrap();
    assert_eq!(packets.len(), 1);
    let finished = encoder.finish(&mut keep(&mut packets));
    assert_eq!((finished.packets, finished.samples), (3, 2048));
}

/// Eight short blocks take the place of the long block whose short blocks
/// would span a click, with the long blocks before and after them where
/// they would be, their windows shaped for them.
#[test]
fn short_blocks_take_a_long_blocks_place_at_a_click() {
    let mut input = vec![0.0f32; 10_000];
    // In the short blocks of the long block centered at 5120, centered from
    // 4672 to 5568.
    input[5000] = 0.5;
    let packets = encode_whole(Config::new(44_100, 1, RATE), &input);
    let positions: Vec<u64> = packets.iter().map(|&(_, position)| position).collect();
    let mut expected: Vec<u64> = (0..5).map(|k| 1024 * k).collect();
    expected.extend((0..8).map(|k| 5120 - 448 + 128 * k));
    expected.extend([6144, 7168, 8192, 9216, 10_000]);
    assert_eq!(positions, expected);
    let modes: Vec<_> = packets.iter().map(|(data, _)| mode(data)).collect();
    assert_eq!(modes[3], (true, Some((true, true))));
    assert_eq!(modes[4], (true, Some((true, false))));
    assert!(modes[5..13].iter().all(|&mode| mode == (false, None)));
    assert_eq!(modes[13], (true, Some((false, true))));
    assert_eq!(modes[14], (true, Some((true, true))));
}

/// The stream ends at the first block centered at or past the end of the
/// input, though it be among short blocks.
#[test]
fn the_stream_ends_among_short_blocks_at_the_end_of_the_input() {
    let mut input = vec![0.0f32; 5000];
    input[4990] = 0.5;
    let packets = encode_whole(Config::new(44_100, 1, RATE), &input);
    let positions: Vec<u64> = packets.iter().map(|&(_, position)| position).collect();
    assert_eq!(
        positions,
        [0, 1024, 2048, 3072, 4096, 4672, 4800, 4928, 5000]
    );
    assert!(
        packets[5..]
            .iter()
            .all(|(data, _)| mode(data) == (false, None))
    );
}

#[test]
fn the_stream_does_not_depend_on_how_the_input_is_given() {
    let channels = 3;
    let input = pcm(5000, channels);
    let config = Config::new(44_100, channels, RATE);
    let whole = encode_whole(config, &input);

    let mut encoder = Encoder::new(config).unwrap();
    let mut chunked = Vec::new();
    for chunk in input.chunks(7 * channels) {
        encoder
            .encode_interleaved(chunk, &mut keep(&mut chunked))
            .unwrap();
    }
    encoder.finish(&mut keep(&mut chunked));
    assert_eq!(chunked, whole);

    let planar: Vec<Vec<f32>> = (0..channels)
        .map(|c| input.iter().skip(c).step_by(channels).copied().collect())
        .collect();
    let slices: Vec<&[f32]> = planar.iter().map(Vec::as_slice).collect();
    let mut encoder = Encoder::new(config).unwrap();
    let mut from_planar = Vec::new();
    encoder
        .encode_planar(&slices, &mut keep(&mut from_planar))
        .unwrap();
    encoder.finish(&mut keep(&mut from_planar));
    assert_eq!(from_planar, whole);
}

#[test]
fn a_non_finite_sample_refuses_the_whole_call() {
    let config = Config::new(48_000, 2, RATE);
    let mut input = pcm(3000, 2);
    input[2 * 2500 + 1] = f32::NAN;
    let mut encoder = Encoder::new(config).unwrap();
    let mut packets = Vec::new();
    let error: NonFiniteSample = encoder
        .encode_interleaved(&input, &mut keep(&mut packets))
        .unwrap_err();
    assert_eq!((error.channel, error.sample), (1, 2500));
    assert!(packets.is_empty());
    // Nothing of the call was taken: the stream is that of the rest.
    let rest = pcm(3000, 2);
    encoder
        .encode_interleaved(&rest, &mut keep(&mut packets))
        .unwrap();
    encoder.finish(&mut keep(&mut packets));
    assert_eq!(packets, encode_whole(config, &rest));
}

#[test]
fn misshapen_input_panics() {
    let mut encoder = Encoder::new(Config::new(48_000, 2, RATE)).unwrap();
    let mut packets = Vec::new();
    assert!(panics(|| {
        let _ = encoder.encode_interleaved(&[0.0f32; 3], &mut keep(&mut packets));
    }));
    assert!(panics(|| {
        let _ = encoder.encode_planar(&[&[0.0f32; 3][..]], &mut keep(&mut packets));
    }));
    assert!(panics(|| {
        let _ = encoder.encode_planar(&[&[0.0f32; 3][..], &[0.0; 2]], &mut keep(&mut packets));
    }));
}
