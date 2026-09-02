# The roofline is what a token streams, not what the checkpoint weighs — 2026-09-02

## Context

Three different weight-bandwidth ceilings have been published for this V100 in
two days, each one used to judge how much headroom was left:

| cited | source | dense 35.3 tok/s reads as |
|---|---|---|
| 14 GB → 64 tok/s | remembered | 55% of roofline |
| 20.35 GB → 44.2 tok/s | `du` on the shards | 80% of roofline |
| **16.04 GB → 56.1 tok/s** | per-tensor, by role | **63% of roofline** |

Only the third is the decode roofline. The other two are a guess and a
different quantity that happens to be measurable.

## Root cause

`du` measures the checkpoint. A decode token streams the weights it multiplies
by, which is a strict subset:

| GB | bucket | streamed per decode token? |
|---:|---|---|
| 15.24 | trunk layers (64) | yes |
| 0.80 | lm_head | yes |
| 2.54 | `embed_tokens` | **no** — one row gathered, not the plane |
| 0.92 | visual tower | **no** — text-only decode never runs it |
| 0.85 | mtp draft head | only on a speculative tick |

16.04 GB is trunk + lm_head. Counting `embed_tokens` because it is big and in
the file is the same class of mistake as citing 14 from memory: the number was
never derived from what the kernel reads.

Also wrong in the 20.35 breakdown: "f32 block scales 3.22" was right by
accident. The scale block is **32, not 16** — one f32 per 32 weights, which is
why the plane is a quarter of the nibbles rather than half.

## Fix

`scripts/check_scale_f16.py` sums per-tensor bytes bucketed by role and by kind
(nibble / scale / dense), so the streamed subset is a column and not a
subtraction done in prose. Every roofline claim now cites 16.04 GB → 56.1 tok/s
for dense decode, and says which bucket it excludes.

## Rule

A roofline is a property of the *inner loop*, not of the artifact on disk. Before
dividing bandwidth by a byte count, name the loop and ask which bytes it
actually touches per iteration: a gathered embedding row, a tower that never
runs, and a draft head on a dense tick are all in the file and none of them are
in the loop.

The tell that this was never checked: the same number was quoted with three
different values in 48 hours and each time it was used to conclude something
about headroom. If a denominator moves that easily, it is being remembered, not
measured.
