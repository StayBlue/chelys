# Instructions for agents working on chelys

Each entry below is an explicit decision by the project's owner, with when and
how it was made.

## Changing this file

Additions and changes to this file require an explicit decision by the user.
Do not add or change an entry because it seems to follow from the user's
intent, from a choice made for one case, or from how earlier work was done. An
agent may propose an entry, but must ask and wait for the user to decide.

*User, 2026-11-12, when this file was created.*

## Changing the README

Additions and changes to `README.md` require an explicit decision by the user,
as for this file. Do not add or change anything in it because it seems to
follow from the user's intent, from a choice made for one case, or from how
earlier work was done. An agent may propose a change, but must ask and wait for
the user to decide.

*User, 2026-11-13, when the README was written: the same rule as for this
file.*

## Clean room

chelys is implemented from ISO/IEC 11172-3, ISO/IEC 13818-3 (its extension to
lower sampling frequencies only), and their official corrigenda only.

- Do not read, or reproduce from memory, the source code or internals of other
  encoders and decoders (for example LAME, the ISO reference code (dist10),
  ffmpeg, minimp3, symphonia or Shine).
- Other decoders may be used only as black boxes: run them as separate programs
  and use only their observable output.
- Follow the informative annexes too, in particular Annex C (the encoding
  process) and Annex D (the psychoacoustic models), as written. Any deviation
  must be deliberate, must still produce a valid bitstream, and must be
  documented in the code where it is made.

*User, 2026-11-12, at the start of the project; "strictly the spec" for the
annexes.* *User, 2026-11-13: ISO/IEC 13818-3 added for the lower sampling
frequencies; the wording was proposed by the agent and approved by the user.*

## MPEG-2.5

The sampling frequencies 8, 11.025 and 12 kHz ("MPEG-2.5") are not part of
ISO/IEC 11172-3 or 13818-3. They are implemented from US patent 6,185,539 B1
only, as an extension of ISO/IEC 13818-3, under the same clean-room rules as
the standards; the patent is the one source added to those named under "Clean
room". Where it leaves something open, black-box decoding with ffmpeg, mpg123
and LAME may settle it, documented as for the standards.

*User, 2026-11-13, when the patent was given: the wording was proposed by the
agent and approved by the user.*

## Scope: PCM only

The `chelys` library accepts PCM samples and nothing else. Reading WAV or any
other container or file format is out of scope. The command-line tool reads
raw interleaved 32-bit float PCM, and format conversion is left to other tools.
Propose any new input path outside the library, and ask before adding it.

*User, 2026-11-12.*

## Documenting gaps and readings of the standard

Where the standard leaves something open, or the code reads it other than
literally, say so in a comment at the place where the decision is made,
starting with one of these markers so that a search finds them all:

- `Spec gap:` — the standard does not settle it, and the code makes a choice.
- `Spec reading:` — the code interprets the printed text, departs from it, or
  takes it literally where that needs saying. Give the reason.

Where decoding with ffmpeg as a black box settled such a gap, add
"Settled by decoding with ffmpeg (black box)" and name the test that provides
the evidence. Record the ffmpeg version that evidence was recorded with in the
`chelys-conformance` crate's documentation (currently n9.0.2), and update it
when the evidence is recorded again with another version.

*User, 2026-11-12: document where ffmpeg filled gaps left by the standard. The
marker phrases were proposed by the agent and approved by the user.*

## mpg123 and LAME as sources of evidence

mpg123 and LAME may also be used as black-box decoders for evidence, as
ffmpeg is. They are optional: evidence need not be checked against them, but
may be, for example where decoders could disagree. Evidence from them is
documented in the same way: say "Settled by decoding with" and name every
decoder the evidence was taken from, for example "Settled by decoding with
ffmpeg, mpg123 and LAME (black box)", and name the test that provides it.
Record the versions of mpg123 and LAME that the evidence was recorded with
in the `chelys-conformance` crate's documentation, next to ffmpeg's
(currently mpg123 1.33.7 and LAME 4.0), and update them when the evidence is
recorded again with other versions.

*User, 2026-11-13: mpg123 and LAME evidence added as approved sources, when
they disagreed with ffmpeg on intensity stereo at the lower sampling
frequencies. The wording was written by the agent.*
*User, 2026-11-13: they may be used, but do not have to be.*

## Commits for ffmpeg evidence

Commit "Settled by decoding with ffmpeg" evidence, and the note on the test
that provides it, separately from the feature that the evidence is for, as a
`docs:` commit after it. `Spec gap:` and `Spec reading:` markers stay in the
feature's commit.

*User, 2026-11-12, when committing short blocks: two commits, one for the
ffmpeg evidence, as with the earlier evidence. The entry was proposed by the
agent and approved by the user.*

## Xing/Info header

The Xing/Info header, which gives a stream's length, seek points and encoder
delay and padding, and which no ISO standard defines, is implemented from
"Mp3 Info Tag rev 1 specifications - draft 0" only, under the same clean-room
rules as the standards; it is the one source added for it. Where it leaves
something open or contradicts itself, black-box decoding with ffmpeg, mpg123
and LAME may settle it, documented as for the standards.

*User, 2026-11-15, when the document was given: the wording was proposed by
the agent and approved by the user.*
*User, 2026-11-15: the document's title corrected to its heading.*

## PEAQ as evidence of quality

The quality of chelys's output may be judged with PEAQ (ITU-R BS.1387) as
computed by GstPEAQ, run as a black box like the decoders: only its grades and
the measures it reports are used. Its grades guide tuning, such as the defaults
of rate control; they do not settle gaps in the standards. Both signals are
given to it at 48 kHz as 16-bit PCM. Record the GstPEAQ version the evidence
was recorded with (currently 0.6.1) in the `chelys-conformance` crate's
documentation, next to the decoders', and update it when the evidence is
recorded again with another version.

The text of ITU-R BS.1387 may be read to interpret PEAQ's measures and grades,
for example to attribute a grade to its measures or to find where they judge
chelys's output. It is not a source for the encoder, and its model is not
implemented in chelys.

*User, 2026-11-15, when listening tests proved too laborious and GstPEAQ was
installed: the wording was proposed by the agent and approved by the user.*
*User, 2026-11-16: reading ITU-R BS.1387 to interpret PEAQ added, when PEAQ's
grades were to be attributed to its measures; the wording was proposed by the
agent and approved by the user.*

## LAME's encoder as a benchmark

LAME's encoder may be run as a black box to benchmark the quality of chelys's
output. Only the PEAQ grades and measures of its decoded output are used. They
are taken from the same input, at the same bitrate (at a variable bitrate, at
the same average size), and through the same decoding and grading as chelys's
output. Its bitstreams are not inspected beyond decoding them, and its choices
are not copied: its grades show how far chelys's tuning has to go, not how to
get there. Record the LAME version the benchmark was run with in the
`chelys-conformance` crate's documentation, next to the decoders'.

*User, 2026-11-16, when the gap to an established encoder was to be measured:
the wording was proposed by the agent and approved by the user.*

## AAC

chelys's AAC encoder is implemented from ISO/IEC 13818-7:2006 and ISO/IEC
14496-3:2019, and their official corrigenda, only, under the same clean-room
rules as the standards named under "Clean room"; they are the sources added
for AAC. It encodes AAC-LC only.

- 13818-7 is followed for the AAC-LC bitstream, ADTS and the encoding process.
  Its Annex C, the encoder including its psychoacoustic model, is followed as
  written, as the annexes of ISO/IEC 11172-3 are; 14496-3 refers to it for
  these tools.
- 14496-3 is followed only for what MPEG-4 adds to AAC-LC, such as the
  `AudioSpecificConfig`.
- Other AAC encoders and decoders, for example FDK AAC, libxaac, FAAC, FAAD2
  and the reference software of ISO/IEC 13818-5 and 14496-5, fall under
  "Clean room" as the codecs named there do.
- Ask before adding any tool outside AAC-LC, or an MPEG-4 tool such as
  perceptual noise substitution (PNS).

Where the standards leave something open, black-box decoding with ffmpeg may
settle it, documented as for the standards.

*User, 2026-11-18, when the standards were given: the wording was proposed by
the agent and approved by the user.*

## FAAD2 as a source of evidence for AAC

FAAD2's decoder (`faad`) may be used as a black-box decoder for AAC evidence,
as ffmpeg is: run as a separate program, with only its observable output used.
Its source code stays under "Clean room", as the AAC entry says. It is
optional: evidence need not be checked against it, but may be, for example
where it could disagree with ffmpeg, or where ffmpeg does not check something,
such as the ADTS CRC. Evidence from it is documented as under "mpg123 and LAME
as sources of evidence": say "Settled by decoding with", name every decoder the
evidence was taken from, and name the test. Record the FAAD2 version the
evidence was recorded with in the `chelys-conformance` crate's documentation,
next to ffmpeg's.

*User, 2026-11-19, when ffmpeg contradicted clause 12.2.3 on intensity bands
under `ms_mask_present` 2 and was found not to check the ADTS CRC: the wording
was proposed by the agent and approved by the user.*
