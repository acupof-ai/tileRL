# The NCCL floor a design verdict rests on has no instrument — 2026-09-06

> Status: open. `scripts/nccl_probe.py` is the instrument, exists, and is cited by
> nothing; no run of it is recorded anywhere.

## Context

After finding the `~60 µs` eager launch floor asserted in five places with no
measurement ([a number with no instrument](2026-09-06-a-number-with-no-instrument.md)),
the same criterion was applied to every quantity cited three or more times across the
tree. One other number fails it, and it fails worse.

## Root Cause

**`21.5 µs` NCCL all-reduce floor: ten citation sites, three files, no instrument.**

| where | what it offers |
|---|---|
| `docs/design-parallel.md:16` | cites "CHANGELOG 2026-08-30", in a table headed *shipped* |
| `:135` | builds a ring-latency table from it — 344 / 1032 / **2408 µs** |
| `:146` | "the measured 21.5 µs NCCL floor (flat from 20 KB to 1.3 MB, CHANGELOG 08-30)" |
| `:159-161` | three rows of a compute-vs-hop table |
| `:165` | derives a block-size crossover |
| `:235` | a stated input to a whole-step ms model |
| `docs/roadmap.md:173` | "NCCL's ~15 µs floor (21.5 µs measured, CHANGELOG 2026-08-30)" |
| `CHANGELOG.md:357` | the terminus |

The terminus does not produce it. Verbatim: "128 all-reduces per tick cost **~2.8 ms
at a 21.5 µs floor**." That line reports a real TP=4 A/B (10.9 → 15.7 tok/s, 57.9 →
92.6) and then **consumes** the floor as a given to derive 2.8 ms. `design-parallel.md:16`
cites that CHANGELOG entry as the floor's source, so the ring closes: the design page
points at the changelog, the changelog assumes the number.

**The instrument exists and nothing cites it.** `scripts/nccl_probe.py` times
`dist.all_reduce` at four message sizes and prints `us/allreduce`. `grep -rn nccl_probe`
across the tree returns exactly one line — the usage comment in its own docstring. No
`21.5` in any log, and no wins/errors entry for the TP=4 bench at all.

**Three aggravating factors, all absent from the 60 µs case.**

*A second claim rides the same absence.* "flat 20 KB → 1.3 MB" is independently
untested, and `design-parallel.md:148` leans on it directly — "the chunks here are
inside the flat region, so the floor *is* the cost." Same shape as "regardless of
shape" in the 60 µs entry: one absent measurement carrying two claims.

*A design decision rests on it.* The ring-vs-all-gather verdict at
`design-parallel.md:191` — "**all-gather for the RL path, ring only past ~58K**" — is
7x arithmetic whose only latency input is 21.5. The page labels it "on floor
arithmetic, pending an exposed-cost measurement", and
[cp-attention-gather](../wins/2026-09-05-cp-attention-gather.md) at `:80` states
plainly that "the exposed-cost measurement the ring decision rests on still does not
exist." So the pending measurement is flagged, and the *input to the arithmetic in the
meantime* is not.

*The roadmap carries two floors in one sentence.* `roadmap.md:173` quotes ~15 µs and
21.5 µs together, the first with no source at all. A denominator that moves that easily
is being remembered, which is the tell named in
[roofline is the streamed subset](2026-09-02-roofline-is-the-streamed-subset.md).

## Fix

Run `torchrun --nproc_per_node=8 scripts/nccl_probe.py` on the H20 and replace the
figure, or mark all ten sites as unmeasured. The probe takes `min` over 7 windows at
four sizes, so it settles the flatness claim in the same run. Until then the ring-vs-
all-gather crossover is arithmetic over an unmeasured operand, and the ~58K number
inherits that.

Not run here: the H20 needs 8 cards for `--nproc_per_node=8`, and this session holds
one job at a time.

## A gate for this was tried and does not work

The obvious automation — flag a number whose line carries no instrument word
(`measured`, `scripts/*.py`, `profiler`, `ncu`, `sweep`, `A/B`, run log) — was tested
against the known-bad 60 µs case first. **3 of its 10 occurrences pass**, all false
positives. The clearest is
[npartition is not the m32 lever](2026-09-02-npartition-is-not-the-m32-lever.md) at
`:114`: "the microbench predicts 4558 ms … against **3406 measured** in-graph, 0.75x.
Same order, so it is a usable **A/B** harness at M=32 — unlike M=1, where its **~60 µs**
eager launch floor dominates." Both instrument words modify the 3406 ms control, not the
floor. Line-level co-occurrence cannot tell what a word modifies, so a gate on it grades
the known defect as partly compliant and cannot go red for the reason under test. No gate
ships.

## Rule

A number consumed to derive a second number does not thereby become measured, and the
file that derives is not a source for what it consumed. When tracing provenance, the
question is not "where is this stated" but "which line reports a run" — and a citation
naming a *file* rather than a run is a pointer to the next hop, not an answer.

Second: a verdict can flag one missing measurement while resting on another. Here
"pending an exposed-cost measurement" is written on the page, so the gap looks
accounted for, and the unmeasured latency floor underneath it does not.

## Results

No runtime change.

| date | commit | machine | measurement | value |
|---|---|---|---|---|
| 2026-09-06 | (this) | H20 | NCCL all-reduce floor | **unmeasured — instrument uncited, needs 8 cards** |
| 2026-09-06 | (this) | — | instrument-word gate on the known-bad case | **3/10 false positives, gate rejected** |
