# chelys

> [!WARNING]
> chelys is experimental. It is not meant for daily use: its output has not
> been compared with that of established encoders, and its API may change at
> any time.

chelys is an MP3 encoder: it encodes PCM audio as MPEG-1 Audio Layer III, at
the lower sampling frequencies of 16, 22.05 and 24 kHz as MPEG-2 Audio Layer
III, and at 8, 11.025 and 12 kHz as the unofficial "MPEG-2.5" extension. It is
written from the standards, ISO/IEC 11172-3 and ISO/IEC 13818-3, and for
MPEG-2.5 from US patent 6,185,539 B1, without reference to the source code of
other encoders or decoders.

## Usage

The command-line encoder reads raw interleaved 32-bit float PCM from stdin and
writes MP3 to stdout. Other tools convert from other formats, for example:

```sh
ffmpeg -i song.flac -f f32le -ar 44100 -ac 2 - \
  | cargo run --release -p chelys-cli -- --sample-rate 44100 --channels 2 > song.mp3
```

Run it with `--help` for the bitrates, sampling frequencies and stereo modes it
supports.
