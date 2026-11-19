// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! Streams of random configurations, from input that reaches the edges of
//! what the encoder takes: silence, subnormal and clamped samples, steady
//! and alternating full scale, impulses, tones up to the Nyquist frequency
//! and noise, in channels alike, opposite or apart, given in random
//! divisions. Each stream is checked as a decoder would read it.
//!
//! The cases follow from their seeds. `CHELYS_RANDOM_CASES` sets how many
//! are run, and `CHELYS_RANDOM_SEED` the first seed, to search further or
//! to run one that failed alone; with optimisations, the overflow checks
//! and debug assertions are kept by
//!
//! ```text
//! CHELYS_RANDOM_CASES=100000 cargo test --release -p chelys-mp3 random_streams \
//!     --config profile.release.overflow-checks=true \
//!     --config profile.release.debug-assertions=true
//! ```

use std::f64::consts::TAU;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use super::check::{self, Frame};
use super::{StreamConfig, max_ancillary_bits};
use crate::header::{Bitrate, Mode};
use crate::{ChannelMode, Config, DELAY, Encoder, Finished, Rate, SAMPLE_RATES, Sample, bitrates};
use chelys_core::bitstream::BitWriter;
use chelys_core::crc::crc16;

/// The cases a plain `cargo test` runs.
const CASES: u64 = 48;

/// SplitMix64: a stream of random numbers that follows from its seed.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// From 0 up to, not including, `n`.
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// From 0.0 up to, not including, 1.0.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn between(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.unit()
    }

    fn chance(&mut self, p: f64) -> bool {
        self.unit() < p
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }
}

/// What one case encodes, and how.
struct Case {
    config: Config,
    stream_config: StreamConfig,
    rate: Rate,
    ancillary_bits_per_frame: usize,
    pcm: Vec<Vec<f64>>,
    /// The samples of each channel in each call, in turn, as long as there
    /// are samples left.
    chunks: Vec<usize>,
    ancillary: Vec<u8>,
    sample_type: SampleType,
}

#[derive(Clone, Copy, Debug)]
enum SampleType {
    F64,
    F32,
    I16,
}

impl Case {
    fn new(seed: u64) -> Self {
        let mut random = Random(seed);
        let (config, rate) = config(&mut random);
        let stream_config = config.lower().expect("a configuration the encoder takes");
        let max = max_ancillary_bits(&stream_config);
        let ancillary_bits_per_frame = match random.below(3) {
            0 => 0,
            1 => random.below(max + 1),
            _ => max,
        };
        let config = config.ancillary_bits_per_frame(ancillary_bits_per_frame);
        let samples_per_frame = stream_config
            .sampling_frequency
            .version()
            .samples_per_frame();
        let pcm = signal(
            &mut random,
            stream_config.sampling_frequency.hz(),
            samples_per_frame,
            stream_config.mode.channels(),
        );
        let mut chunks = Vec::new();
        let mut covered = 0;
        while covered < pcm[0].len() {
            let chunk = match random.below(4) {
                0 => 0,
                1 => 1 + random.below(16),
                2 => random.below(samples_per_frame + 2),
                _ => random.below(4 * samples_per_frame),
            };
            chunks.push(chunk);
            covered += chunk;
        }
        let ancillary_len = match random.below(3) {
            0 => 0,
            _ => random.below(2_000),
        };
        let ancillary = (0..ancillary_len).map(|_| random.next() as u8).collect();
        let sample_type = random.pick(&[SampleType::F64, SampleType::F32, SampleType::I16]);
        Self {
            config,
            stream_config,
            rate,
            ancillary_bits_per_frame,
            pcm,
            chunks,
            ancillary,
            sample_type,
        }
    }

    fn describe(&self, seed: u64) -> String {
        format!(
            "seed {seed}: {:?}, {} samples of {:?} in {} calls, {} bytes of ancillary data",
            self.config,
            self.pcm[0].len(),
            self.sample_type,
            self.chunks.len(),
            self.ancillary.len()
        )
    }

    /// Encodes the input in one call of planar samples and in the case's
    /// calls of interleaved samples, and checks that they give one stream,
    /// and that the stream keeps to the standard.
    fn run(&self) {
        let (stream, finished, tag_len) = match self.sample_type {
            SampleType::F64 => self.encode_both(|x| x),
            SampleType::F32 => self.encode_both(|x| x as f32),
            SampleType::I16 => {
                self.encode_both(|x| (x * 32_768.0).round().clamp(-32_768.0, 32_767.0) as i16)
            }
        };
        self.check(stream, &finished, tag_len);
    }

    fn encode_both<S: Sample>(&self, convert: fn(f64) -> S) -> (Vec<u8>, Finished, Option<usize>) {
        let pcm: Vec<Vec<S>> = self
            .pcm
            .iter()
            .map(|channel| channel.iter().map(|&x| convert(x)).collect())
            .collect();
        let new = || {
            let mut encoder = Encoder::new(self.config).expect("a configuration it takes");
            encoder.push_ancillary(&self.ancillary);
            encoder
        };

        let mut encoder = new();
        let tag_len = encoder.info_tag_len();
        let slices: Vec<&[S]> = pcm.iter().map(Vec::as_slice).collect();
        let mut stream = Vec::new();
        encoder.encode_planar(&slices, &mut stream);
        let finished = encoder.finish(&mut stream);

        let interleaved: Vec<S> = (0..pcm[0].len())
            .flat_map(|t| pcm.iter().map(move |channel| channel[t]))
            .collect();
        let channels = pcm.len();
        let mut encoder = new();
        let mut parts = Vec::new();
        let mut rest = interleaved.as_slice();
        for &chunk in &self.chunks {
            let (part, after) = rest.split_at((chunk * channels).min(rest.len()));
            encoder.encode_interleaved(part, &mut parts);
            rest = after;
        }
        let parts_finished = encoder.finish(&mut parts);
        assert!(parts == stream, "the calls change the stream");
        assert_eq!(
            parts_finished, finished,
            "the calls change what finishing gives"
        );
        (stream, finished, tag_len)
    }

    /// Checks `stream`: its Xing/Info tag, its frames and the rules of the
    /// bit reservoir, their headers, error checks and side information, and
    /// the ancillary data they carry.
    fn check(&self, mut stream: Vec<u8>, finished: &Finished, tag_len: Option<usize>) {
        let free = match self.rate {
            Rate::Free { bits_per_second } => {
                let version = self.stream_config.sampling_frequency.version();
                Some(Bitrate::free(version, bits_per_second).expect("a free bitrate"))
            }
            _ => None,
        };
        let frames_of = |stream: &[u8]| match free {
            Some(bitrate) => check::free_frames(stream, bitrate),
            None => check::frames(stream),
        };

        let start = match (&finished.info_tag, tag_len) {
            (Some(tag), Some(len)) => {
                assert_eq!(
                    tag.len(),
                    len,
                    "the tag's frame is not its placeholder's length"
                );
                stream[..len].copy_from_slice(tag);
                let frames = frames_of(&stream[..len]);
                assert_eq!(frames.len(), 1, "the tag is not one frame");
                self.check_frame(&stream[..len], &frames[0], true, &mut [None; 2]);
                len
            }
            (None, None) => 0,
            (tag, len) => panic!(
                "a tag {:?} for a placeholder of {len:?} bytes",
                tag.as_ref().map(Vec::len)
            ),
        };
        let audio = &stream[start..];
        let frames = frames_of(audio);

        let samples = self.pcm[0].len();
        let samples_per_frame = self
            .stream_config
            .sampling_frequency
            .version()
            .samples_per_frame();
        assert_eq!(finished.samples, samples as u64);
        assert_eq!(finished.frames, frames.len() as u64);
        assert_eq!(frames.len(), (samples + DELAY).div_ceil(samples_per_frame));

        let mut at = 0;
        let mut block_types = [None; 2];
        for (k, frame) in frames.iter().enumerate() {
            let bytes = &audio[at..at + frame.len];
            let result = catch_unwind(AssertUnwindSafe(|| {
                self.check_frame(bytes, frame, false, &mut block_types);
            }));
            if let Err(panic) = result {
                eprintln!("in frame {k} of {}", frames.len());
                std::panic::resume_unwind(panic);
            }
            at += frame.len;
        }

        let ancillary = match free {
            Some(bitrate) => check::free_ancillary(audio, bitrate),
            None => check::ancillary(audio),
        };
        let bits: Vec<bool> = (0..8 * self.ancillary.len())
            .map(|i| (self.ancillary[i / 8] >> (7 - i % 8)) & 1 == 1)
            .collect();
        let mut sent = 0;
        for (k, frame) in ancillary.iter().enumerate() {
            let carried = self.ancillary_bits_per_frame.min(bits.len() - sent);
            assert!(
                frame.len() >= carried,
                "frame {k} has no room for its ancillary data"
            );
            assert!(
                frame[..carried] == bits[sent..sent + carried],
                "frame {k}'s ancillary data"
            );
            assert!(!frame[carried..].contains(&true), "frame {k}'s stuffing");
            sent += carried;
        }
        assert_eq!(
            sent + finished.ancillary_dropped,
            bits.len(),
            "ancillary bits lost"
        );
    }

    /// Checks a frame's header against the configuration and the bitrates
    /// the rate allows, its error check, and the fields of its side
    /// information, `block_types` holding the last granule's of each
    /// channel.
    fn check_frame(
        &self,
        bytes: &[u8],
        frame: &Frame,
        tag_frame: bool,
        block_types: &mut [Option<usize>; 2],
    ) {
        let config = &self.stream_config;
        let extension = (bytes[3] >> 4) & 0b11;
        // mode_extension: MS stereo the higher bit, intensity stereo the
        // lower (clause 2.4.2.3).
        let (ms, intensity) = (extension & 0b10 != 0, extension & 0b01 != 0);
        match config.mode {
            Mode::JointStereo {
                intensity_stereo,
                ms_stereo,
            } => {
                assert!(!ms || ms_stereo, "MS stereo, not allowed");
                assert!(
                    !intensity || intensity_stereo,
                    "intensity stereo, not allowed"
                );
            }
            _ => assert_eq!(extension, 0, "a mode extension outside joint stereo"),
        }
        // The Xing/Info tag's frame has no error check (see `info_tag`).
        let crc_protected = config.crc_protected && !tag_frame;
        let mut header = config.frame_header(frame.bitrate, frame.padding, ms, intensity);
        header.crc_protected = crc_protected;
        let mut expected = BitWriter::new();
        header.write(&mut expected);
        assert_eq!(bytes[..4], expected.into_bytes(), "the header");

        if !tag_frame && !frame.bitrate.is_free() {
            let kbps = frame.bitrate.bits_per_second() / 1000;
            let (lowest, highest) = self.bitrate_range();
            assert!((lowest..=highest).contains(&kbps), "{kbps} kbit/s");
        }

        let version = config.sampling_frequency.version();
        let channels = config.mode.channels();
        let crc = usize::from(crc_protected) * 2;
        let side_info_len = crate::side_info::SideInfo::len_bytes(version, channels);
        let side_info = &bytes[4 + crc..4 + crc + side_info_len];
        if crc_protected {
            let covered = bytes[2..4].iter().chain(side_info).copied();
            assert_eq!(
                u16::from_be_bytes([bytes[4], bytes[5]]),
                crc16(covered),
                "the error check"
            );
        }

        // The fields of each granule of each channel, after
        // main_data_begin, private_bits and scfsi (clause 2.4.1.7, and
        // clause 2.4.1.7 of ISO/IEC 13818-3).
        let mpeg1 = version == crate::header::Version::Mpeg1;
        let (before, per_channel, compress_bits) = match (mpeg1, channels) {
            (true, 1) => (9 + 5 + 4, 59, 4),
            (true, _) => (9 + 3 + 8, 59, 4),
            (false, 1) => (8 + 1, 63, 9),
            (false, _) => (8 + 2, 63, 9),
        };
        let mut at = before;
        let mut read = |count: usize| {
            let value = (at..at + count).fold(0, |value, bit| {
                (value << 1) | usize::from((side_info[bit / 8] >> (7 - bit % 8)) & 1)
            });
            at += count;
            value
        };
        for _ in 0..version.granules() {
            for block_type_before in block_types.iter_mut().take(channels) {
                let _part2_3_length = read(12);
                let big_values = read(9);
                let _global_gain = read(8);
                let _scalefac_compress = read(compress_bits);
                let window_switching = read(1) == 1;
                let (block_type, mixed, tables) = if window_switching {
                    let block_type = read(2);
                    let mixed = read(1) == 1;
                    let tables = [read(5), read(5)];
                    let _subblock_gain = [read(3), read(3), read(3)];
                    assert_ne!(block_type, 0, "a switched window of the normal block type");
                    (block_type, mixed, tables.to_vec())
                } else {
                    let tables = [read(5), read(5), read(5)];
                    let (region0_count, region1_count) = (read(4), read(3));
                    // region2 begins at the band after both regions' counts,
                    // at most the end of the 22 long bands.
                    assert!(
                        region0_count + region1_count + 2 <= 22,
                        "regions of {region0_count} and {region1_count} past the bands"
                    );
                    (0, false, tables.to_vec())
                };
                let _flags = read(if mpeg1 { 3 } else { 2 });
                assert!(big_values <= 288, "big_values {big_values}");
                // Tables 4 and 14 are not used (table B.7).
                assert!(
                    !tables.iter().any(|t| matches!(t, 4 | 14)),
                    "table_select {tables:?}"
                );
                if mixed {
                    assert_eq!(block_type, 2, "a mixed block not of short blocks");
                    assert!(
                        self.config.wants_mixed_blocks(),
                        "a mixed block not asked for"
                    );
                    assert!(!intensity, "a mixed block in intensity stereo");
                }
                // A granule's window begins as the window before it ends:
                // long after normal and stop blocks, short after start and
                // short blocks (Annex C).
                if !tag_frame {
                    if let Some(before) = *block_type_before {
                        let ends_short = matches!(before, 1 | 2);
                        let begins_short = matches!(block_type, 2 | 3);
                        assert_eq!(
                            ends_short, begins_short,
                            "block type {block_type} after {before}"
                        );
                    }
                    *block_type_before = Some(block_type);
                }
            }
        }
        debug_assert_eq!(at, before + version.granules() * channels * per_channel);
    }

    /// The lowest and the highest bitrate, in kbit/s, of the frames of a
    /// rate of the table.
    fn bitrate_range(&self) -> (u32, u32) {
        let table = bitrates(self.stream_config.sampling_frequency.hz()).expect("a sample rate");
        let (lowest, highest) = (table[0], table[table.len() - 1]);
        match self.rate {
            Rate::Constant { kbps } => (kbps, kbps),
            Rate::Mean { bits_per_second } => (
                table
                    .iter()
                    .copied()
                    .filter(|&k| k * 1000 <= bits_per_second)
                    .max()
                    .unwrap(),
                table
                    .iter()
                    .copied()
                    .filter(|&k| k * 1000 >= bits_per_second)
                    .min()
                    .unwrap(),
            ),
            Rate::Variable {
                min_kbps, max_kbps, ..
            } => (min_kbps.unwrap_or(lowest), max_kbps.unwrap_or(highest)),
            Rate::Free { .. } => unreachable!("free format has no bitrate of the table"),
        }
    }
}

/// A configuration the encoder takes, every option drawn at random, but
/// with no ancillary data.
fn config(random: &mut Random) -> (Config, Rate) {
    loop {
        let sample_rate = random.pick(&SAMPLE_RATES);
        let table = bitrates(sample_rate).expect("a sample rate");
        let (lowest, highest) = (table[0], table[table.len() - 1]);
        let mode = match random.below(4) {
            0 => ChannelMode::Mono,
            1 => ChannelMode::Stereo,
            2 => ChannelMode::JointStereo {
                ms_stereo: random.chance(0.7),
                intensity_stereo: random.chance(0.5),
            },
            _ => ChannelMode::DualChannel,
        };
        let rate = match random.below(4) {
            0 => Rate::Constant {
                kbps: random.pick(table),
            },
            1 => Rate::Mean {
                bits_per_second: random.between(f64::from(lowest), f64::from(highest)) as u32
                    * 1000
                    + random.below(1000) as u32,
            },
            2 => {
                let (a, b) = (random.pick(table), random.pick(table));
                Rate::Variable {
                    min_kbps: random.chance(0.7).then_some(a.min(b)),
                    max_kbps: random.chance(0.7).then_some(a.max(b)),
                    noise_offset_db: random.between(-12.0, 24.0),
                }
            }
            _ => Rate::Free {
                bits_per_second: random.between(8_000.0, f64::from(highest) * 1000.0) as u32,
            },
        };
        let config = Config::new(sample_rate, mode, rate)
            .crc(random.chance(0.5))
            .info_tag(random.chance(0.5))
            .mixed_blocks(random.chance(0.4))
            .copyright(random.chance(0.5))
            .original(random.chance(0.5))
            .private_bit(random.chance(0.5));
        if config.check().is_ok() {
            return (config, rate);
        }
    }
}

/// Up to six frames of input, in stretches of random length and content.
fn signal(
    random: &mut Random,
    sample_rate: u32,
    samples_per_frame: usize,
    channels: usize,
) -> Vec<Vec<f64>> {
    let len = match random.below(8) {
        0 => 0,
        1 => random.below(samples_per_frame),
        _ => random.below(6 * samples_per_frame),
    };
    let mut pcm = vec![Vec::with_capacity(len); channels];
    while pcm[0].len() < len {
        let left = len - pcm[0].len();
        let n = match random.chance(0.2) {
            true => 1 + random.below(16),
            false => 1 + random.below(2 * samples_per_frame),
        }
        .min(left);
        let first = stretch(random, n, sample_rate);
        let second = match random.below(5) {
            0 => first.clone(),
            1 => first.iter().map(|x| -x).collect(),
            2 => vec![0.0; n],
            3 => first
                .iter()
                .map(|x| x + 1e-3 * random.between(-1.0, 1.0))
                .collect(),
            _ => stretch(random, n, sample_rate),
        };
        pcm[0].extend(first);
        if channels == 2 {
            pcm[1].extend(second);
        }
    }
    pcm
}

/// `len` samples of one kind of signal, at a level from subnormal to far
/// past the clamp.
fn stretch(random: &mut Random, len: usize, sample_rate: u32) -> Vec<f64> {
    let level = match random.below(7) {
        0 => 1.0,
        1 => random.unit(),
        2 => 10f64.powf(random.between(-12.0, 0.0)),
        3 => 10f64.powf(random.between(0.0, 4.9)),
        4 => 10f64.powf(random.between(4.9, 12.0)),
        5 => 1e-310,
        _ => 65_536.0,
    };
    let fs = f64::from(sample_rate);
    let sign = if random.chance(0.5) { level } else { -level };
    match random.below(9) {
        0 => vec![0.0; len],
        1 => vec![sign; len],
        2 => (0..len)
            .map(|_| level * random.between(-1.0, 1.0))
            .collect(),
        3 => {
            let hz = random.between(0.0, fs / 2.0);
            let phase = random.between(0.0, TAU);
            (0..len)
                .map(|t| level * (TAU * hz * t as f64 / fs + phase).sin())
                .collect()
        }
        4 => {
            let period = 1 + random.below(2_000);
            (0..len)
                .map(|t| if t % period == 0 { sign } else { 0.0 })
                .collect()
        }
        5 => {
            let period = 2 + random.below(400);
            (0..len)
                .map(|t| {
                    if t % period < period / 2 {
                        level
                    } else {
                        -level
                    }
                })
                .collect()
        }
        6 => (0..len)
            .map(|t| {
                let s = t as f64 / fs;
                let sweep = fs / 2.0 / (len as f64 / fs);
                level * (TAU * 0.5 * sweep * s * s).sin()
            })
            .collect(),
        7 => (0..len)
            .map(|t| if t % 2 == 0 { level } else { -level })
            .collect(),
        _ => {
            let decay = 1.0 + random.below(500) as f64;
            (0..len)
                .map(|t| level * (-(t as f64) / decay).exp() * random.between(-1.0, 1.0))
                .collect()
        }
    }
}

fn from_env(name: &str) -> Option<u64> {
    std::env::var(name).ok().map(|value| {
        value
            .parse()
            .unwrap_or_else(|_| panic!("{name} is not a number: {value}"))
    })
}

/// Random configurations and input give streams that do not depend on how
/// the input is divided between calls, and that keep to the standard.
#[test]
fn random_streams_keep_to_the_standard() {
    let cases = from_env("CHELYS_RANDOM_CASES").unwrap_or(CASES);
    let first = from_env("CHELYS_RANDOM_SEED").unwrap_or(0);
    let next = AtomicU64::new(first);
    let failed = Mutex::new(Vec::new());
    let workers = std::thread::available_parallelism().map_or(4, usize::from);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let seed = next.fetch_add(1, Ordering::Relaxed);
                    if seed >= first + cases {
                        break;
                    }
                    let case = Case::new(seed);
                    if catch_unwind(AssertUnwindSafe(|| case.run())).is_err() {
                        eprintln!("failed: {}", case.describe(seed));
                        failed.lock().unwrap().push(seed);
                    }
                }
            });
        }
    });
    let mut failed = failed.into_inner().unwrap();
    failed.sort_unstable();
    assert!(
        failed.is_empty(),
        "{} of {cases} cases failed, seeds {failed:?}; run one alone with \
         CHELYS_RANDOM_SEED=<seed> CHELYS_RANDOM_CASES=1",
        failed.len()
    );
}
