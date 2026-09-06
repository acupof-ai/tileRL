# A number with no instrument, cited five times — 2026-09-06

> Status: **Closed 2026-09-06.** Measured on an idle-issuing V100: **10.1 µs amortized,
> 21.2 µs per-call-with-sync, 25.6 µs isolated**, flat across four decades of element
> count. So "regardless of shape" was right and the magnitude was 6x high, and the ~60
> came from no timer convention this card produces. All five citations now carry the
> measurement. [the launch floor is ten
> microseconds](../wins/2026-09-06-the-launch-floor-is-ten-microseconds.md)

## Context

Task #21 arrives each tick with four figures. Three were already withdrawn. Tracing
the fourth — "the microbench has a **~60 µs eager launch floor** regardless of shape"
— found no measurement behind it anywhere in the tree.

## Root Cause

**Five citations, zero instruments.** Every occurrence, and what each offers as
support:

| where | what it says about provenance |
|---|---|
| `scripts/ab_scale_f16.py:17` | asserts it as a property of the harness |
| `scripts/prof_prefill_split.py:11` | cites "docs/experience: the ~60us eager launch floor" — no file, no line |
| `errors/2026-09-02-npartition-is-not-the-m32-lever.md:114` | uses it as a known quantity |
| `errors/2026-09-02-per-shape-gap-was-a-wrong-shape-table.md:17` | states it as the reason to re-measure in-graph |
| `errors/2026-09-05-task-21-numbers-do-not-survive-a-grep.md:11` | states it as the reason the instruction exists |

No A/B, no null-kernel sweep, no run log, no `~60` in any output. The `~` and
"regardless of shape" are the only characterisation given, and "regardless of shape"
is a second claim that nothing tests separately from the magnitude.

**The audit is one of the five.** `2026-09-05-task-21-numbers-do-not-survive-a-grep.md`
exists to check task #21's figures against their instruments. It caught the 5%, the
32% and the 144 — and at `:11` it uses the floor as the premise for why those needed
checking. An audit citing an unaudited number as its own justification is the shape
that let this survive five citations.

**It is not a restatement of the H20 figure.** `errors/2026-08-27-fp4-gemv-issue-bound-ncu.md:26`
measures **~40 µs** on H20 with supporting detail (o_proj 40.9 → 41-42 µs across an 8x
block-count sweep, and the rule "never A/B a <40 µs kernel with the eager harness").
That number has an instrument. The sm70 60 does not cite it, and no file says whether
60 is a re-measurement, an adjustment, or a guess — the string `60` does not appear in
that entry at all.

## Fix

`scripts/probe_launch_floor.py` sweeps work downward through `benchkit.timeit` — the
same CUDA-events-around-a-Python-loop harness the claim is about — so the floor is read
as the point where per-call time stops falling. A no-op launch, a 32-element add, a
7-decade element sweep, and one `1×5120×6144` matmul, which makes "regardless of shape"
falsifiable independently of the magnitude.

It refuses a card holding more than 1 GiB. A launch floor is a CPU-side and issue-path
quantity, so a card serving traffic contends for exactly what is under measurement, and
this pod has already produced that error: an idle-card probe read 11.55 ms for a 144 MiB
pinned copy against **161.9 ms** in the live path, 14.0x
([a two-variable condition read as a dead end](2026-09-05-a-two-variable-condition-read-as-a-dead-end.md)).
The refusal is exercised rather than assumed — against the live V100 it printed
`REFUSING: 28354 MiB is already resident` and exited 1.

The floor stays **unmeasured**. The endpoint holds 28.0 of 32.8 GB, and a contended
reading is worse than none because it looks like a measurement.

## Rule

**N files agreeing is not N measurements.** It can be one absence cited N times, and
repetition is what stops a number being checked: each new citation looks like
corroboration to the next reader, so the fifth author has more apparent support than the
first and less reason to look. Before using a number as a premise, find the run that
produced it — not the file that states it.

Corollary, from how this one lasted: **an audit is not exempt from its own criterion.**
The entry that checked three figures against their instruments used a fourth as its
premise. Whatever standard a check applies, apply it to the check's own inputs first.

## Results

No runtime change. `scripts/probe_launch_floor.py` is dev tooling; its own output on a
busy card is a refusal and an exit code, which is the correct result rather than a
measurement.

| date | commit | machine | target | measurement | value |
|---|---|---|---|---|---|
| 2026-09-06 | (this) | V100 32GB | cuda sm70 | eager launch floor | **unmeasured — card busy, probe refused** |
| 2026-08-27 | — | H20 | cuda sm90 | eager launch floor (for contrast, measured) | ~40 µs |
