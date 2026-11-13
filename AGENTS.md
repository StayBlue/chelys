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

chelys is implemented from ISO/IEC 11172-3 and its official corrigenda only.

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
annexes.*

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

## Commits for ffmpeg evidence

Commit "Settled by decoding with ffmpeg" evidence, and the note on the test
that provides it, separately from the feature that the evidence is for, as a
`docs:` commit after it. `Spec gap:` and `Spec reading:` markers stay in the
feature's commit.

*User, 2026-11-12, when committing short blocks: two commits, one for the
ffmpeg evidence, as with the earlier evidence. The entry was proposed by the
agent and approved by the user.*
