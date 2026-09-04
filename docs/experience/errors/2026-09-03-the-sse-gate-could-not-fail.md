# The SSE gate passed on the arm it was written to reject

**Date:** 2026-09-03
**Target:** cpu (pure Python; no GPU needed to find or fix)

## Context

Incremental SSE shipped earlier the same day with a wins entry, a measured
number (streamed 32.4 tok/s against non-streamed 32.4 — no cost, which is real
and stands) and a green suite. Rebasing the branch onto current `main` meant
reading the committed `_stream` loop line by line to resolve a conflict against
upstream's `strip_think`. The code in the tree was not the code the entry
described: the two fixes made while debugging were never committed.

## Root cause

Two defects in one loop, each masking the other.

1. `if len(live) > sent` — `live` is a **token** list, `sent` a **character**
   count. Two different units compared as one.
2. `text.find(FFFD)` then `text = text[:cut]` — cuts at the **first**
   replacement char rather than the trailing run. `tiny()` has random weights,
   so its bytes are mostly not valid UTF-8 and most prefixes *start* with one.
   The cut therefore returned `""` at nearly every poll.

With (2) blanking the text, `sent` never advancing under (1) had no visible
effect. Fixing either alone looks like it changes nothing.

Measured both arms over four model seeds, 24 tokens, temperature 0:

| seed | broken: whole reply in tail | fixed: whole reply in tail |
|------|------|------|
| 7  | yes | no |
| 21 | yes | no |
| 42 | **no** | no |
| 99 | yes | no |

`joined == expected` was true in **all 8 runs**. That is the whole problem: the
suite asserted delta count and joined text, and both hold when the poll loop
emits nothing and the final tail chunk carries everything.

Two further findings, each worse than the bug:

- The module `client` fixture builds its engine at **seed 42** — the one seed
  of four measured where the defect does not surface. The test could not have
  failed on either version.
- The `no U+FFFD in a mid-stream delta` assertion is **unsatisfiable** for any
  loop that actually streams: the non-streamed reply carries interior
  replacement chars on 4 of 4 seeds, so every split point emits one. It passed
  only because nothing was emitted before the end.

## Fix

- `seen` counts tokens, `sent` counts characters of the stripped reply — one
  unit each, named for what it holds.
- `rstrip(FFFD)` holds only the trailing replacement run, which is the only
  part that can be an incomplete character.
- The test builds its own engine at seed 7 and says why in the docstring.
- The unsatisfiable containment assertion is replaced by two that hold and
  discriminate: no incremental delta may **end** on U+FFFD, and the last delta
  may not be the entire reply.
- Negative control run against the broken loop: fails with
  `only 1 content delta(s): the stream is not incremental`.

## Rule

A gate must be run against the arm it exists to reject, on the input where that
arm fails — not against a synthetic case, and not on whatever seed the fixture
happened to carry. Two corollaries measured here: an assertion that cannot fail
is worse than no assertion, because it reads as coverage; and when a defect is
seed-dependent, the fixture's seed is part of the gate and belongs in the
docstring with its reason.

And the process failure behind both: entry written, number measured on the card,
code never re-read. The number was true and upstream of the bug.
