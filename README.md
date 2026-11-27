# chelys

> [!WARNING]
> chelys is experimental. It is not meant for daily use: its quality still
> trails established encoders at some bitrates, and its API may change at any
> time.

chelys encodes PCM audio as MP3, AAC and Vorbis, and writes MP4 files, each
written from its specifications without reference to the source code of other
encoders or decoders:

- MP3: MPEG-1 Audio Layer III, at the lower sampling frequencies of 16,
  22.05 and 24 kHz as MPEG-2 Audio Layer III, and at 8, 11.025 and 12 kHz as
  the unofficial "MPEG-2.5" extension; from ISO/IEC 11172-3 and
  ISO/IEC 13818-3, and for MPEG-2.5 from US patent 6,185,539 B1.
- AAC: MPEG-2 AAC in the Low Complexity profile (AAC-LC), from mono to 7.1,
  at sampling frequencies from 8 to 96 kHz, in ADTS, as raw frames, or in MP4
  files; from ISO/IEC 13818-7, ISO/IEC 14496-3 for what MPEG-4 adds, and
  ISO/IEC 14496-12, 14496-14, 14496-1 and 14496-34 for MP4.
- Vorbis: Vorbis I, from mono to 8 channels, at a variable rate set by a
  quality level or steered toward a mean bitrate, or at a constant bitrate
  kept within a buffer, as packets with their granule positions or in Ogg
  streams; from the Xiph.Org Foundation's "Vorbis I specification", and
  RFC 3533 for Ogg.
- MP4 from other encoders: the library also writes FLAC, Opus, AAC-LC and
  ALAC packets that any encoder made in MP4 files, whole or fragmented for
  streaming: FLAC and Opus from Xiph's mappings of each to the ISO base media
  file format, RFC 9639, and RFC 7845 and RFC 6716, AAC-LC as chelys's own,
  and ALAC from Apple's description of its magic cookie.

## Usage

The command-line encoder reads raw interleaved 32-bit float PCM from stdin and
writes MP3, AAC-LC (in ADTS, or an MP4 file with `aac --mp4`) or Vorbis (in an
Ogg stream) to stdout.
Other tools convert from other formats, for example:

```sh
ffmpeg -i song.flac -f f32le -ar 44100 -ac 2 - \
  | cargo run --release -p chelys-cli -- mp3 --sample-rate 44100 --channels 2 > song.mp3
ffmpeg -i song.flac -f f32le -ar 44100 -ac 2 - \
  | cargo run --release -p chelys-cli -- aac --sample-rate 44100 --channels 2 > song.aac
ffmpeg -i song.flac -f f32le -ar 44100 -ac 2 - \
  | cargo run --release -p chelys-cli -- aac --mp4 --sample-rate 44100 --channels 2 > song.mp4
ffmpeg -i song.flac -f f32le -ar 44100 -ac 2 - \
  | cargo run --release -p chelys-cli -- vorbis --sample-rate 44100 --channels 2 > song.ogg
```

The codec comes first; run it with `--help` for the codecs, with `mp3 --help`
for the bitrates, sampling frequencies and stereo modes of MP3, with
`aac --help` for the sampling frequencies, channels and bitrates of AAC, and
with `vorbis --help` for the channels, quality levels, and constant and mean
bitrates of Vorbis.
