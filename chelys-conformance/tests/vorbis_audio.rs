// Copyright (c) 2026 StayBlue
// SPDX-License-Identifier: MIT

use std::f64::consts::TAU;

use chelys::vorbis::{Config, Encoder, Headers, MAX_CHANNELS, Packet, Rate};
use chelys_conformance::ffmpeg;
use chelys_conformance::ogg::{self, AudioPacket};

/// Sample rates of the MPEG standards and beyond, as Vorbis has no table
/// of them.
const SAMPLE_RATES: [u32; 9] = [
    8_000, 11_025, 16_000, 22_050, 32_000, 44_100, 48_000, 96_000, 192_000,
];

/// The rate of every stream but where a test says otherwise: the noise
/// offset 0 dB, the psychoacoustic model's allowance, raised with the width
/// of each critical band, where the encoder finds bits scarce.
const RATE: Rate = Rate::Variable {
    noise_offset_db: 0.0,
};

/// The headers and audio packets of `pcm`, interleaved.
fn encode_packets(pcm: &[f32], sample_rate: u32, channels: usize) -> (Headers, Vec<AudioPacket>) {
    encode_packets_at(pcm, sample_rate, channels, RATE)
}

/// The headers and audio packets of `pcm`, interleaved, coded at `rate`.
fn encode_packets_at(
    pcm: &[f32],
    sample_rate: u32,
    channels: usize,
    rate: Rate,
) -> (Headers, Vec<AudioPacket>) {
    let mut encoder = Encoder::new(Config::new(sample_rate, channels, rate)).unwrap();
    let headers = encoder.headers();
    let mut packets = Vec::new();
    let mut out = |packet: Packet<'_>| {
        packets.push(AudioPacket {
            data: packet.data.to_vec(),
            granule_position: packet.granule_position,
        })
    };
    encoder.encode_interleaved(pcm, &mut out).unwrap();
    encoder.finish(&mut out);
    (headers, packets)
}

/// `pcm` encoded and decoded by ffmpeg, which is to report nothing and to
/// give back as many samples.
fn round_trip(pcm: &[f32], sample_rate: u32, channels: usize) -> Vec<f32> {
    round_trip_at(pcm, sample_rate, channels, RATE)
}

/// `pcm` encoded at `rate` and decoded by ffmpeg, as [`round_trip`].
fn round_trip_at(pcm: &[f32], sample_rate: u32, channels: usize, rate: Rate) -> Vec<f32> {
    let (headers, packets) = encode_packets_at(pcm, sample_rate, channels, rate);
    let decoded = ffmpeg::decode_ogg(&ogg::vorbis(&headers, &packets, 4));
    assert_eq!(decoded.diagnostics, "");
    assert_eq!(decoded.samples.len(), pcm.len());
    decoded.samples
}

/// The ratio in dB of the power of `reference` to that of what `decoded`
/// differs from it by, over every `step`th sample from `start`.
fn snr(reference: &[f32], decoded: &[f32], start: usize, step: usize) -> f64 {
    let (mut signal, mut noise) = (0.0, 0.0);
    for (&r, &d) in reference.iter().zip(decoded).skip(start).step_by(step) {
        signal += f64::from(r).powi(2);
        noise += (f64::from(r) - f64::from(d)).powi(2);
    }
    10.0 * (signal / noise).log10()
}

/// Uniform noise from -0.5 to 0.5, the same each run.
fn noise(len: usize) -> Vec<f64> {
    let mut state = 1u32;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f64::from(state >> 8) / f64::from(1 << 24) - 0.5
        })
        .collect()
}

/// ffmpeg decodes a tone at its level and in phase: at the encoder's input
/// scale, and neither delayed, advanced nor inverted. So it takes the
/// spectrum at the scale, phase and sign that the encoder gives it.
#[test]
fn ffmpeg_decodes_a_tone_at_its_level_and_phase() {
    let sample_rate = 48_000;
    let pcm: Vec<f32> = (0..sample_rate as usize)
        .map(|t| (0.5 * (TAU * 1000.0 * t as f64 / f64::from(sample_rate)).sin()) as f32)
        .collect();
    let decoded = round_trip(&pcm, sample_rate, 1);
    let middle = 4096..pcm.len() - 4096;
    let correlation = |lag: isize| -> f64 {
        middle
            .clone()
            .map(|i| f64::from(pcm[i]) * f64::from(decoded[i.wrapping_add_signed(lag)]))
            .sum()
    };
    let power: f64 = middle.clone().map(|i| f64::from(pcm[i]).powi(2)).sum();
    let gain_db = 20.0 * (correlation(0) / power).log10();
    assert!(gain_db.abs() < 0.5, "a gain of {gain_db:.2} dB");
    for lag in [-2, -1, 1, 2] {
        assert!(correlation(lag) < correlation(0), "lag {lag}");
    }
    assert!(snr(&pcm, &decoded, middle.start, 1) > 15.0);
}

/// ffmpeg decodes each channel, at any sample rate, close to its own
/// input: a tone of its own in each. ffmpeg gives the channels in its own
/// order of speakers, not that of clause 4.3.9 above two channels, so each
/// channel it gives is matched to the input it is nearest. The tones are
/// too short for the psychoacoustic model to find them tonal from the
/// start, and it may take them as noise, which masks noise 6 dB below it.
#[test]
fn ffmpeg_decodes_each_channel_at_any_sample_rate() {
    let samples = 5 * 1024 + 300;
    for sample_rate in SAMPLE_RATES {
        for channels in 1..=MAX_CHANNELS {
            let context = format!("{sample_rate} Hz, {channels} channels");
            let pcm: Vec<f32> = (0..samples * channels)
                .map(|i| {
                    let (t, c) = (i / channels, i % channels);
                    let cycles = (c + 1) as f64 * 0.017;
                    (0.4 * (TAU * cycles * t as f64).sin()) as f32
                })
                .collect();
            let decoded = round_trip(&pcm, sample_rate, channels);
            let channel = |pcm: &[f32], c: usize| -> Vec<f32> {
                pcm.iter().skip(c).step_by(channels).copied().collect()
            };
            let mut matched = Vec::new();
            for out in 0..channels {
                let decoded = channel(&decoded, out);
                let (input, snr) = (0..channels)
                    .map(|c| (c, snr(&channel(&pcm, c), &decoded, 0, 1)))
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .unwrap();
                assert!(snr > 6.0, "{context}: channel {out}, {snr:.1} dB");
                matched.push(input);
            }
            matched.sort_unstable();
            matched.dedup();
            assert_eq!(matched.len(), channels, "{context}");
        }
    }
}

/// ffmpeg decodes noise, quiet or loud, and a mix of tones and noise, with
/// no more noise than the psychoacoustic model allows: noise masks noise 6
/// dB below it, and tones noise 15 dB below them. The streams are coded at
/// an offset of -10.5 dB, which takes off at least as much as the encoder
/// raises the allowance of any band for its width at their 44.1 kHz: at
/// most 10.1 dB.
#[test]
fn ffmpeg_decodes_noise_and_tones_within_what_the_model_allows() {
    let sample_rate = 44_100;
    let rate = Rate::Variable {
        noise_offset_db: -10.5,
    };
    let round_trip =
        |pcm: &[f32], sample_rate, channels| round_trip_at(pcm, sample_rate, channels, rate);
    let noise = noise(2 * sample_rate as usize);
    for level in [0.01, 0.3, 1.5] {
        let pcm: Vec<f32> = noise.iter().map(|&x| (level * x) as f32).collect();
        let snr = snr(&pcm, &round_trip(&pcm, sample_rate, 1), 0, 1);
        assert!(snr > 6.0, "noise at {level}: {snr:.1} dB");
    }
    let mut low = 0.0;
    let pcm: Vec<f32> = (noise.iter().enumerate())
        .map(|(t, &x)| {
            low = 0.95 * low + 0.05 * x;
            let t = t as f64 / f64::from(sample_rate);
            (0.3 * (TAU * 220.0 * t).sin() + 0.1 * (TAU * 3520.0 * t).sin() + 0.5 * low) as f32
        })
        .collect();
    let snr = snr(&pcm, &round_trip(&pcm, sample_rate, 1), 0, 1);
    assert!(snr > 15.0, "the mix: {snr:.1} dB");
}

/// ffmpeg decodes a square wave at full scale, whose lines are the largest
/// the encoder meets in music, without clipping their residue.
#[test]
fn ffmpeg_decodes_a_square_wave_at_full_scale() {
    let pcm: Vec<f32> = (0..88_200)
        .map(|t| if t / 50 % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let snr = snr(&pcm, &round_trip(&pcm, 44_100, 1), 0, 1);
    assert!(snr > 15.0, "{snr:.1} dB");
}

/// Whether an audio packet is of a short block: its mode, the bit after
/// `packet_type`, is chelys's mode 0, of short blocks (clause 4.3.1).
fn is_short(packet: &AudioPacket) -> bool {
    (packet.data[0] >> 1) & 1 == 0
}

/// A quiet chord, with a burst of noise that sets in at each of `onsets`
/// and dies away in about 5 ms: attacks, which short blocks code.
fn bursts(len: usize, sample_rate: u32, onsets: &[usize]) -> Vec<f64> {
    let noise = noise(len);
    let mut pcm: Vec<f64> = (0..len)
        .map(|t| {
            let t = t as f64 / f64::from(sample_rate);
            0.05 * (TAU * 220.0 * t).sin() + 0.03 * (TAU * 330.0 * t).sin()
        })
        .collect();
    let decay = 0.005 * f64::from(sample_rate);
    for &onset in onsets.iter().filter(|&&onset| onset < len) {
        for (k, x) in pcm[onset..]
            .iter_mut()
            .enumerate()
            .take(10 * decay as usize)
        {
            *x += noise[onset + k] * (-(k as f64) / decay).exp();
        }
    }
    pcm
}

/// The lag, within 16 samples either way, at which `decoded` correlates
/// best with `pcm` over the 256 samples from `onset`.
fn lag(pcm: &[f32], decoded: &[f32], onset: usize) -> isize {
    (-16..=16)
        .max_by(|&a: &isize, &b: &isize| {
            let correlation = |lag: isize| -> f64 {
                (onset..onset + 256)
                    .map(|i| f64::from(pcm[i]) * f64::from(decoded[i.wrapping_add_signed(lag)]))
                    .sum()
            };
            correlation(a).total_cmp(&correlation(b))
        })
        .unwrap()
}

/// ffmpeg decodes a stream of short blocks among long ones, at any sample
/// rate and in one or two channels, without a word and to the sample: each
/// attack, wherever it lies among the short blocks that take a long
/// block's place, decodes where it was, and the whole within the noise
/// that noise masks, 6 dB below it.
#[test]
fn ffmpeg_decodes_short_blocks_in_place() {
    // The chord sets in at the first sample, in the first long block's
    // place, which is coded long; an attack in the second's, whose place
    // short blocks take, and then one at every 61st sample of a hop, from
    // the first of those long blocks whose places short blocks would take.
    let mut onsets = vec![1300];
    onsets.extend((0..17).map(|k| 3 * 1024 + 600 + k * (3 * 1024 + 61)));
    let len = onsets.last().unwrap() + 4 * 1024;
    for sample_rate in [16_000, 44_100, 48_000] {
        for channels in [1, 2] {
            let context = format!("{sample_rate} Hz, {channels} channels");
            let mono = bursts(len, sample_rate, &onsets);
            let pcm: Vec<f32> = (mono.iter())
                .flat_map(|&x| (0..channels).map(move |c| (x * (1.0 - 0.3 * c as f64)) as f32))
                .collect();
            let (headers, packets) = encode_packets(&pcm, sample_rate, channels);
            let short = packets.iter().filter(|packet| is_short(packet)).count();
            assert!(short >= 8 * onsets.len(), "{context}: {short} short blocks");
            assert!(!is_short(&packets[0]) && is_short(&packets[1]), "{context}");
            let decoded = ffmpeg::decode_ogg(&ogg::vorbis(&headers, &packets, 3));
            assert_eq!(decoded.diagnostics, "", "{context}");
            assert_eq!(decoded.samples.len(), pcm.len(), "{context}");
            for c in 0..channels {
                let channel = |pcm: &[f32]| -> Vec<f32> {
                    pcm.iter().skip(c).step_by(channels).copied().collect()
                };
                let (input, output) = (channel(&pcm), channel(&decoded.samples));
                for &onset in &onsets {
                    assert_eq!(lag(&input, &output, onset), 0, "{context}: {onset}");
                }
                let snr = snr(&input, &output, 0, 1);
                assert!(snr > 6.0, "{context}: channel {c}, {snr:.1} dB");
            }
        }
    }
}

/// The coding noise of an attack does not spread to the samples before it,
/// where nothing masks it: in the 256 samples that end 256 before each
/// onset of a burst of noise over a quiet chord, the noise is within 1.5
/// dB of that of the chord far before the onset, where long blocks alone
/// make it 2.9 dB louder.
#[test]
fn short_blocks_keep_the_noise_of_an_attack_from_before_it() {
    let sample_rate = 44_100;
    let onsets: Vec<usize> = (0..12).map(|k| 4000 + k * 11_111).collect();
    let pcm: Vec<f32> = (bursts(onsets.last().unwrap() + 8192, sample_rate, &onsets).iter())
        .map(|&x| x as f32)
        .collect();
    let decoded = round_trip(&pcm, sample_rate, 1);
    // The power of the coding noise over every onset, from `from` samples
    // after it, of 256 samples.
    let coding_noise = |from: isize| -> f64 {
        (onsets.iter())
            .flat_map(|&onset| (0..256).map(move |k| onset.wrapping_add_signed(from) + k))
            .map(|i| (f64::from(decoded[i]) - f64::from(pcm[i])).powi(2))
            .sum()
    };
    let (before, chord) = (coding_noise(-512), coding_noise(-4000));
    let db = 10.0 * (before / chord).log10();
    assert!(db < 1.5, "{db:.1} dB");
}

/// ffmpeg ends a stream where the input ends, among short blocks too: its
/// last packet may be any of the short blocks that take a long block's
/// place, and ffmpeg trims what that packet would return past the end, as
/// for a long one, however the packets are put in pages.
#[test]
fn ffmpeg_trims_a_stream_that_ends_among_short_blocks() {
    let sample_rate = 44_100;
    // An attack in the short blocks of the long block centered at 5120,
    // centered from 4672 to 5568; the input ends before, among and after
    // their centers, and after the attack, which then decodes in place.
    let onset = 4900;
    let mut ended_short = 0;
    for len in [
        4700, 4800, 4801, 5000, 5100, 5200, 5300, 5567, 5568, 5569, 6000,
    ] {
        let pcm: Vec<f32> = (bursts(len, sample_rate, &[onset]).iter())
            .map(|&x| x as f32)
            .collect();
        let (headers, packets) = encode_packets(&pcm, sample_rate, 1);
        assert_eq!(packets.last().unwrap().granule_position, len as u64);
        ended_short += usize::from(is_short(packets.last().unwrap()));
        for packets_per_page in [1, 3, 100] {
            let context = format!("{len} samples, {packets_per_page} packets a page");
            let decoded = ffmpeg::decode_ogg(&ogg::vorbis(&headers, &packets, packets_per_page));
            assert_eq!(decoded.diagnostics, "", "{context}");
            assert_eq!(decoded.samples.len(), len, "{context}");
            if len >= onset + 300 {
                assert_eq!(lag(&pcm, &decoded.samples, onset), 0, "{context}");
            }
        }
    }
    assert!(ended_short >= 5, "{ended_short} streams end short");
}

/// ffmpeg times each packet from the granule position of its page as the
/// span of samples that clause 4.3.8 returns for it, from the center of the
/// block before to the center of its own, a long block before a short one
/// too: with each packet on a page of its own, ffprobe gives each packet
/// after the first two the span from the granule position of the one
/// before it to its own.
#[test]
fn ffmpeg_times_each_packet_from_its_granule_position() {
    let sample_rate = 44_100;
    let onsets = [1300, 5000, 9000, 9500, 15_000];
    let pcm: Vec<f32> = (bursts(20_000, sample_rate, &onsets).iter())
        .map(|&x| x as f32)
        .collect();
    let (headers, packets) = encode_packets(&pcm, sample_rate, 1);
    let times = ffmpeg::packet_times_ogg(&ogg::vorbis(&headers, &packets, 1));
    assert_eq!(times.len(), packets.len());
    let mut long_before_short = 0;
    for k in 2..packets.len() {
        let (pts, duration) = times[k];
        let start = packets[k - 1].granule_position as i64;
        let end = packets[k].granule_position as i64;
        assert_eq!((pts, pts + duration as i64), (start, end), "packet {k}");
        long_before_short += usize::from(!is_short(&packets[k - 1]) && is_short(&packets[k]));
    }
    assert!(long_before_short >= 4, "{long_before_short}");
}

/// ffmpeg decodes a loud onset after digital silence close to the input
/// and with no noise before it, as it does a short loud input: pre-echo
/// control holds the threshold of the blocks at the onset near the
/// threshold in quiet, and the floor is raised where it would put a line
/// past the residue's reach.
///
/// The noise before each onset is measured in streams coded at `RATE`, and
/// the signal to noise after it in streams coded at -10.5 dB, as in
/// `ffmpeg_decodes_noise_and_tones_within_what_the_model_allows`, and for
/// the same reason: the samples measured after an onset lie in blocks
/// coded at the allowance, which noise, masked by itself 6 dB below it,
/// takes more of in the widest bands, where the encoder raises it for
/// their width (see `RATE`). At 0 dB, the signal to noise of loud white
/// noise there is that allowance's, and not bounded by 6 dB: it ranges
/// from about 4.5 to 6.5 dB with the onset, the noise's seed and its level.
#[test]
fn ffmpeg_decodes_a_loud_onset_after_silence() {
    let sample_rate = 44_100;
    let rate = Rate::Variable {
        noise_offset_db: -10.5,
    };
    let tone = |t: usize| 0.9 * (TAU * 1000.0 * t as f64 / f64::from(sample_rate)).sin();
    let noise = noise(40_000);
    for onset in [5000, 5420, 9000] {
        for (name, signal) in [
            ("tone", &tone as &dyn Fn(usize) -> f64),
            ("noise", &|t| 1.6 * noise[t]),
        ] {
            let context = format!("{name} from {onset}");
            let pcm: Vec<f32> = (0..40_000)
                .map(|t| {
                    if (onset..onset + 8000).contains(&t) {
                        signal(t) as f32
                    } else {
                        0.0
                    }
                })
                .collect();
            let decoded = round_trip_at(&pcm, sample_rate, 1, rate);
            let snr = snr(&pcm, &decoded[..onset + 512], onset, 1);
            assert!(snr > 6.0, "{context}: {snr:.1} dB");
            let decoded = round_trip(&pcm, sample_rate, 1);
            let before: f64 = (onset - 1024..onset)
                .map(|i| f64::from(decoded[i]).powi(2))
                .sum::<f64>()
                / 1024.0;
            let before_db = 10.0 * before.log10();
            assert!(
                before_db < -50.0,
                "{context}: {before_db:.1} dBFS before the onset"
            );
        }
    }
    for len in [100, 300, 577] {
        let pcm: Vec<f32> = (0..len).map(|t| tone(t) as f32).collect();
        let snr = snr(&pcm, &round_trip(&pcm, sample_rate, 1), 0, 1);
        assert!(snr > 6.0, "{len} samples: {snr:.1} dB");
    }
}

/// ffmpeg decodes a tone in short blocks at its level and in phase, as in
/// long ones (see `ffmpeg_decodes_a_tone_at_its_level_and_phase`): over
/// the samples between the centers of two short blocks, which they alone
/// give, after the tone steps up tenfold, which short blocks code.
#[test]
fn ffmpeg_decodes_a_tone_in_short_blocks_at_its_level_and_phase() {
    let sample_rate = 48_000;
    let onsets: Vec<usize> = (0..10).map(|k| 3000 + k * 6000 + k * 37).collect();
    let loud = |t: usize| {
        onsets
            .iter()
            .any(|&onset| (onset..onset + 3000).contains(&t))
    };
    let pcm: Vec<f32> = (0..66_000)
        .map(|t| {
            let level = if loud(t) { 0.5 } else { 0.05 };
            (level * (TAU * 1000.0 * t as f64 / f64::from(sample_rate)).sin()) as f32
        })
        .collect();
    let (headers, packets) = encode_packets(&pcm, sample_rate, 1);
    let decoded = ffmpeg::decode_ogg(&ogg::vorbis(&headers, &packets, 4));
    assert_eq!(decoded.diagnostics, "");
    assert_eq!(decoded.samples.len(), pcm.len());
    // The samples between two short blocks' centers, 64 or more past an
    // onset, of the loud tone.
    let span: Vec<usize> = (packets.windows(2))
        .filter(|pair| is_short(&pair[0]) && is_short(&pair[1]))
        .flat_map(|pair| pair[0].granule_position as usize..pair[1].granule_position as usize)
        .filter(|&t| loud(t) && loud(t.saturating_sub(64)) && t + 2 < pcm.len())
        .collect();
    assert!(span.len() > 2000, "{} samples", span.len());
    let correlation = |lag: isize| -> f64 {
        (span.iter())
            .map(|&i| f64::from(pcm[i]) * f64::from(decoded.samples[i.wrapping_add_signed(lag)]))
            .sum()
    };
    let power: f64 = span.iter().map(|&i| f64::from(pcm[i]).powi(2)).sum();
    let gain_db = 20.0 * (correlation(0) / power).log10();
    assert!(gain_db.abs() < 0.5, "a gain of {gain_db:.2} dB");
    for lag in [-2, -1, 1, 2] {
        assert!(correlation(lag) < correlation(0), "lag {lag}");
    }
}
