# chelys

> [!WARNING]
> chelys is experimental. It is not meant for daily use: its output has not
> been compared with that of established encoders, and its API may change at
> any time.

chelys encodes PCM audio as MP3 and as AAC, written from the standards
without reference to the source code of other encoders or decoders:

- MP3: MPEG-1 Audio Layer III, at the lower sampling frequencies of 16,
  22.05 and 24 kHz as MPEG-2 Audio Layer III, and at 8, 11.025 and 12 kHz as
  the unofficial "MPEG-2.5" extension; from ISO/IEC 11172-3 and
  ISO/IEC 13818-3, and for MPEG-2.5 from US patent 6,185,539 B1.
- AAC: MPEG-2 AAC in the Low Complexity profile (AAC-LC), from mono to 7.1,
  at sampling frequencies from 8 to 96 kHz, in ADTS or as raw frames; from
  ISO/IEC 13818-7, and for what MPEG-4 adds, ISO/IEC 14496-3.

## Usage

The command-line encoder reads raw interleaved 32-bit float PCM from stdin and
writes MP3 or AAC-LC (in ADTS) to stdout. Other tools convert from other
formats, for example:

```sh
ffmpeg -i song.flac -f f32le -ar 44100 -ac 2 - \
  | cargo run --release -p chelys-cli -- mp3 --sample-rate 44100 --channels 2 > song.mp3
ffmpeg -i song.flac -f f32le -ar 44100 -ac 2 - \
  | cargo run --release -p chelys-cli -- aac --sample-rate 44100 --channels 2 > song.aac
```

The codec comes first; run it with `--help` for the codecs, with `mp3 --help`
for the bitrates, sampling frequencies and stereo modes of MP3, and with
`aac --help` for the sampling frequencies, channels and bitrates of AAC.
