// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

//! AAC streams whose channels send some of their lines with the pulse
//! escape (`pulse_data()`), as ffmpeg, and in places FAAD2, decode them.

use chelys::aac::internals::channel::{CodedChannel, CodedGroup, FRAME_LINES};
use chelys::aac::internals::channel_configuration::ChannelConfiguration;
use chelys::aac::internals::noiseless::clipping;
use chelys::aac::internals::noiseless::decoder::{BitReader, element};
use chelys::aac::internals::noiseless::pulses::Pulses;
use chelys::aac::internals::noiseless::sections::{self, Sections};
use chelys::aac::internals::rate::Control;
use chelys::aac::internals::raw_data_block::{self, element::CodedElement};
use chelys::aac::internals::sampling_frequency::SamplingFrequency;
use chelys::aac::internals::scalefactor_bands::GroupBands;
use chelys::aac::internals::tns::Tns;
use chelys::aac::internals::window::Windows;
use chelys::aac::internals::{StreamConfig, StreamEncoder};
use chelys::aac::{DELAY, Framing};
use chelys_conformance::{Decoded, faad, ffmpeg};

/// A deterministic stream of pseudo-random numbers (xorshift).
struct Noise(u64);

impl Noise {
    fn below(&mut self, n: u32) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 32) as u32) % n
    }

    /// `magnitude` with a sign drawn at random.
    fn signed(&mut self, magnitude: i32) -> i32 {
        if self.below(2) == 0 {
            magnitude
        } else {
            -magnitude
        }
    }
}

/// A mono channel of a long window that sends `lines`, with those of
/// `pulses` clipped, every band's scalefactor 200.
fn channel(fs: SamplingFrequency, pulses: Pulses, lines: [i32; FRAME_LINES]) -> CodedChannel {
    let bands = GroupBands::long(fs);
    let mut sent = lines;
    pulses.clip(&mut sent);
    let max_sfb = sections::max_sfb(&sent, bands);
    let group = CodedGroup {
        sections: Sections::choose(&sent, bands, max_sfb).expect("codable lines"),
        scalefactors: vec![200; max_sfb],
        positions: Vec::new(),
        noise_energies: Vec::new(),
    };
    CodedChannel::new(
        Windows::LONG_SINE,
        fs,
        200,
        Tns::NONE,
        pulses,
        sent,
        vec![group],
    )
    .expect("scalefactors of one value")
}

/// An ADTS stream of a single channel element for each of `channels`, its
/// headers as Tables 6 and 7 give them: `ID` 1, Low Complexity,
/// without CRC, `adts_buffer_fullness` 0x7FF.
fn stream(fs: SamplingFrequency, channels: Vec<CodedChannel>) -> Vec<u8> {
    let mut stream = Vec::new();
    for channel in channels {
        let element = CodedElement::single(channel);
        let block = raw_data_block::write(
            ChannelConfiguration::Mono,
            std::slice::from_ref(&element),
            &[],
            0,
        )
        .bytes;
        let len = 7 + block.len();
        let fields: [(u64, u32); 15] = [
            (0xFFF, 12),
            (1, 1),
            (0, 2),
            (1, 1),
            (1, 2),
            (u64::from(fs.index()), 4),
            (0, 1),
            (1, 3),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (len as u64, 13),
            (0x7FF, 11),
            (0, 2),
        ];
        let header = fields
            .iter()
            .fold(0u64, |h, &(value, bits)| (h << bits) | value);
        stream.extend_from_slice(&header.to_be_bytes()[1..]);
        stream.extend(block);
    }
    stream
}

/// Lines of ones and zeros below `end`, from which `peaks` stand out.
fn lines(noise: &mut Noise, end: usize, peaks: &[(usize, i32)]) -> [i32; FRAME_LINES] {
    let mut lines = [0; FRAME_LINES];
    for line in &mut lines[..end] {
        *line = noise.below(3) as i32 - 1;
    }
    for &(line, q) in peaks {
        lines[line] = q;
    }
    lines
}

/// ffmpeg restores the lines that the pulse escape clips as clause 9.3
/// decodes them: streams whose channels send pulses, chosen by spectrum
/// clipping (clause C.8.2) or placed at the limits of `pulse_data()`,
/// decode without complaint as the same lines sent unclipped, to within
/// the rounding of the decoder's arithmetic, and far from the lines sent
/// clipped without their pulses.
///
/// This is the evidence for the reading of the standard that the clipping
/// procedure (`noiseless::clipping`) marks as settled by decoding with
/// ffmpeg: each `pulse_offset` after the first counts from the line of
/// the pulse before (clause 9.3), not from the start of the band (clause
/// C.8.2). Counted from the band, the second pulse of these streams would
/// fall on another line.
#[test]
fn ffmpeg_restores_clipped_lines_as_sent_unclipped() {
    restores_clipped_lines_as_sent_unclipped(ffmpeg::decode_aac, 1);
}

/// FAAD2 restores the clipped lines as ffmpeg does above, counting each
/// `pulse_offset` after the first from the pulse before; at 8 kHz, at
/// twice the sampling frequency.
#[test]
fn faad_restores_clipped_lines_as_sent_unclipped() {
    restores_clipped_lines_as_sent_unclipped(faad::decode_aac, 2);
}

/// What the tests above check, with `decode`, which gives at 8 kHz
/// `factor` times the samples of each frame.
fn restores_clipped_lines_as_sent_unclipped(decode: fn(&[u8]) -> Decoded, factor: usize) {
    let mut noise = Noise(0x9e37_79b9_7f4a_7c15);
    for fs in [
        SamplingFrequency::Hz8000,
        SamplingFrequency::Hz44100,
        SamplingFrequency::Hz96000,
    ] {
        let bands = GroupBands::long(fs);
        let (mut clipped, mut unclipped, mut dropped) = (Vec::new(), Vec::new(), Vec::new());
        let mut sent_pulses = 0;
        for frame in 0..24 {
            let end = 256 + noise.below(768) as usize;
            let pulses_and_lines = if frame % 2 == 0 {
                // As spectrum clipping chooses: up to four peaks among
                // the highest lines.
                let peaks: Vec<_> = (0..1 + noise.below(4))
                    .map(|_| {
                        let line = end - 1 - noise.below(64) as usize;
                        let magnitude = 2 + noise.below(15) as i32;
                        (line, noise.signed(magnitude))
                    })
                    .collect();
                let lines = lines(&mut noise, end, &peaks);
                let max_sfb = sections::max_sfb(&lines, bands);
                let counted = Sections::count(&lines, bands, max_sfb).unwrap();
                let mut found = Vec::new();
                clipping::clippings(&lines, bands, &counted, &mut found);
                let (pulses, _) = clipping::choose(&counted, &mut found);
                (pulses, lines)
            } else {
                // Four pulses at the limits of their fields: the first up
                // to 31 lines into its band, among the highest, each 31
                // lines above the one before, with pulse_amp 15 or 1,
                // positive and negative.
                let highest = (0..bands.num_swb())
                    .rfind(|&sfb| bands.offset(sfb) + 4 * 31 < FRAME_LINES)
                    .expect("a band low enough");
                let first = bands.offset(highest - noise.below(6) as usize) + 31;
                let at: Vec<usize> = (0..4).map(|j| first + 31 * j).collect();
                let peaks: Vec<_> = (at.iter().enumerate())
                    .map(|(j, &line)| {
                        let magnitude = if j % 2 == 0 { 16 } else { 2 };
                        (line, noise.signed(magnitude))
                    })
                    .collect();
                let lines = lines(&mut noise, at[3] + 1, &peaks);
                let amps: Vec<_> = (peaks.iter())
                    .map(|&(line, q)| (line, q.unsigned_abs() - 1))
                    .collect();
                (
                    Pulses::new(bands, &amps).expect("pulses within reach"),
                    lines,
                )
            };
            let (pulses, lines) = pulses_and_lines;
            sent_pulses += pulses.iter().count();
            let mut sent = lines;
            pulses.clip(&mut sent);
            clipped.push(channel(fs, pulses, lines));
            unclipped.push(channel(fs, Pulses::NONE, lines));
            dropped.push(channel(fs, Pulses::NONE, sent));
        }
        assert!(sent_pulses > 24, "{fs:?}: {sent_pulses} pulses");
        let [clipped, unclipped, dropped] =
            [clipped, unclipped, dropped].map(|channels| stream(fs, channels));
        assert_ne!(clipped, unclipped);
        let [clipped, unclipped, dropped] =
            [clipped, unclipped, dropped].map(|stream| decode(&stream));
        let factor = if fs == SamplingFrequency::Hz8000 {
            factor
        } else {
            1
        };
        for decoded in [&clipped, &unclipped, &dropped] {
            assert_eq!(decoded.diagnostics, "", "{fs:?}");
            assert_eq!(decoded.samples.len(), 24 * FRAME_LINES * factor, "{fs:?}");
        }
        let error = |decoded: &Decoded| {
            (decoded.samples.iter())
                .zip(&unclipped.samples)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        let peak = unclipped.samples.iter().fold(0.0f32, |a, x| a.max(x.abs()));
        assert!(peak > 0.1, "{fs:?}: {peak}");
        // Within the rounding of the decoder's arithmetic, where the lines
        // without their pulses decode far from it.
        let (restored, without) = (error(&clipped), error(&dropped));
        assert!(restored < 1e-6 * peak, "{fs:?}: {restored} of {peak}");
        assert!(without > 1e-2 * peak, "{fs:?}: {without} of {peak}");
    }
}

/// Streams the encoder codes with spectrum clipping decode without
/// complaint, with pulses in some frames, and with their input as near as
/// coded without it.
#[test]
fn ffmpeg_decodes_streams_coded_with_spectrum_clipping() {
    // Noise falling by 6 dB an octave, whose highest lines quantize to few
    // above 1.
    let (mut noise, mut y) = (Noise(7), 0.0);
    let left: Vec<f64> = (0..48 * FRAME_LINES)
        .map(|_| {
            y = 0.98 * y + (f64::from(noise.below(1 << 20)) / f64::from(1 << 20) - 0.5);
            0.2 * y
        })
        .collect();
    let right: Vec<f64> = left.iter().map(|x| -0.7 * x).collect();
    for bits_per_second in [64_000, 128_000] {
        let config = StreamConfig {
            sampling_frequency: SamplingFrequency::Hz44100,
            channel_configuration: ChannelConfiguration::Stereo,
            bits_per_second,
            control: Control::Constant,
            framing: Framing::Adts,
            flags: Default::default(),
            crc_protected: false,
            noise_substitution: false,
        };
        let encode = |on: bool| {
            let mut encoder = StreamEncoder::new(config).spectrum_clipping(on);
            let mut stream = Vec::new();
            encoder.encode(&[&left, &right], &mut stream);
            encoder.finish(&mut stream);
            stream
        };
        let snr = |stream: &[u8]| {
            let decoded = ffmpeg::decode_aac(stream);
            assert_eq!(decoded.diagnostics, "", "{bits_per_second}");
            let (mut signal, mut error) = (0.0, 0.0);
            for (t, &x) in left.iter().enumerate() {
                let y = f64::from(decoded.samples[2 * (t + DELAY)]);
                signal += x * x;
                error += (x - y) * (x - y);
            }
            10.0 * (signal / error).log10()
        };
        let clipped = encode(true);
        let pulses: usize = frames(&clipped)
            .map(|frame| {
                let mut r = BitReader::new(&frame[7..]);
                let element = element(&mut r, SamplingFrequency::Hz44100);
                element
                    .channels
                    .iter()
                    .filter(|c| c.pulses.is_some())
                    .count()
            })
            .sum();
        assert!(pulses > 0, "{bits_per_second}");
        let (with, without) = (snr(&clipped), snr(&encode(false)));
        assert!(
            with > without - 0.5,
            "{bits_per_second}: {with:.2} dB, {without:.2} dB without"
        );
    }
}

/// Each `adts_frame()` of `stream`, by its `aac_frame_length`.
fn frames(stream: &[u8]) -> impl Iterator<Item = &[u8]> {
    let mut rest = stream;
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        let len = (usize::from(rest[3] & 0b11) << 11)
            | (usize::from(rest[4]) << 3)
            | usize::from(rest[5] >> 5);
        let (frame, after) = rest.split_at(len);
        rest = after;
        Some(frame)
    })
}
