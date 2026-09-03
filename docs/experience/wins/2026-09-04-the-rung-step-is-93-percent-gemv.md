# The verify rung step is 93% fp4 GEMV, at a flat launch count — V100, 2026-09-04

> Status: attribution, no code change. It closes the last open piece of the
> block-parallel reject: the rung step is one kernel's per-row cost, and the
> 1.34x speedup that would flip the verdict is larger than any measured fix.

## Context

The block-parallel reject (Task #22) now rests on a single comparison: rung 8's
verify alone (80.31 ms) exceeds our entire k=3 tick (62.74 ms), so the width a
block head exists to buy is unaffordable before its own forward is priced
(`wins/2026-09-04-a-difference-amplifies-its-operands-noise.md`). That makes the
rung step load-bearing, and nothing had attributed it.

The step is large and the wrong shape for bandwidth. A verify at rung W streams the
**same weights** as W=1 — only X grows — so a bandwidth-bound GEMV should be nearly
flat in M. Measured end-to-end it is not: **5.68 / 8.06 / 8.71 ms per extra row**
across rungs 1→2→4→8. X's own traffic cannot explain it: at M=8, X is 27.03 MB =
**0.030 ms** at 900 GB/s, so 8.71 ms/row is **2320x the byte cost**.

## What Worked

`scripts/prof_decode_budget.py --depths 1,2,3,4` at ctx=1024, one process, wikitext.
Per-class ms/forward, with the GEMV launch count printed beside it as the control:

| class | d1 (rung 2) | d2 (rung 4) | d3 (rung 4) | d4 (rung 8) |
|---|---:|---:|---:|---:|
| **fp4 GEMV** | **22.62** | **37.66** | **38.70** | **69.28** |
| elementwise | 3.55 | 4.29 | 3.79 | 4.44 |
| GDN | 1.94 | 2.36 | 2.91 | 3.16 |
| attention | 1.89 | 2.66 | 3.38 | 4.03 |
| rmsnorm | 1.25 | 1.31 | 1.36 | 1.40 |
| GPU total | 31.78 | 49.66 | 50.96 | 83.88 |
| GEMV calls/fwd | 313.4 | 321.5 | 329.8 | 338.0 |

**The step is one class.** At the 4→8 crossing the GPU forward grows 32.92 ms and the
GEMV carries **30.58 of it — 92.9%**. At 2→4 it is 15.04 of 17.88 = **84.1%**. Every
other class moves under 0.8 ms. Attention grows too (it has more query rows) but at
2.0% of the step, so it is not where a fix goes.

**The launch count is flat, which is what makes this an M cost.** 313.4 → 338.0
calls/forward, **+2.6% across four depths**, against a GEMV that grows 3.06x. Same
kernels, same number of times, same weight bytes; only M moves.

**The within-rung control.** d2 and d3 are *both* rung 4. Their GEMV differs by
+1.04 ms (+2.8%) against a launch count that differs by +2.6% — so inside one rung
the GEMV tracks the launch count and not the depth. That growth is the extra draft
forward, not a dearer verify, which is the rung thesis stated as a control rather
than as an assumption.

**Two instruments agree on the per-row cost.** The profiler's GEMV gives **7.52 and
7.64 ms/row** for the two crossings; the end-to-end tick gave **8.06 and 8.71**.
Ratios 1.072x and 1.140x, and the residual is the non-GEMV classes the same table
prices at 7.1% and 4.4% of the two steps — same direction, same size.

## What this decides

**The mechanism was already located, and this says it is the whole story.** The
kernel's grid is `T.ceildiv(N, n_partition)` with **no M term**
(`kernels_linear.py:936`); M appears only as `T.unroll(M)` over the accumulator, so
extra rows are serial work in the same threads. The `X_REUSE` ablation reads 0.97x at
M=1, **3.05x at M=8**, 8.75x at M=32 — an M-dependence only a per-row cost produces
(`wins/2026-09-03-x-dominates-the-gemv-at-m32.md`).

**No known fix reaches the bar.** To flip the reject, rung 8's verify must fall from
80.31 to under 62.74 ms — 17.57 ms off a GEMV that is 69.28 of the 83.88 ms forward,
i.e. the kernel must get **1.34x faster at M=8**. Measured at M=8: SMEM staging
**0.72x**, PIPELINE **0.99x**, occupancy/`min_blocks` **1.00x**, NO_SCALE 0.98x,
NO_DECODE 0.98x. None is a speedup, let alone 1.34x.

**And the obvious kernel is unavailable.** `make_linear_fp4_mma8` is written for
exactly this range — "Marlin-style decode GEMM for M <= 8" — but it is not in the
sm70 cell and cannot be: it issues `mma.m16n8k16`, which is Ampere and later. On
Volta the rung ladder is the only path.

So the reject stands, and it stands on a measured kernel property rather than on
arithmetic over a model. What would reopen it is a new M≤8 sm70 GEMV, not a
re-derivation.

## Rule

**Attribute a step to a class before pricing a fix for it, and print the launch
count next to it.** The GEMV growing 3.06x while its launch count grows 2.6% is what
makes "this is a per-row cost" a measurement instead of an inference — the same
number that caught the wrong shape table in
`errors/2026-09-02-per-shape-gap-was-a-wrong-shape-table.md`.

Second: **a within-rung pair is the control a rung claim needs.** d2 and d3 share
rung 4, so their delta measures everything *except* M. Without it, "the step is the
rung" is asserted; with it, the 2.8% residual is shown to be the draft's extra
launches.

Third, a dead end worth recording: **`record_shapes=True` does not separate a
TileLang kernel's shapes.** It groups by the ATen input shapes of the launching op,
and a TileLang kernel arrives as a raw CUDA kernel name with no ATen op above it —
all 330 launches came back under one empty shape row. The guard printed
`!! one shape row only`, which is why this cost one run and not a conclusion. The
depth sweep is the instrument that works, because it varies M through the engine
rather than trying to read M off the profile.

## Results

No runtime change; no rate row. `scripts/prof_decode_budget.py` gains `--depths` and
the per-class rung table.

The run's script was md5 `808d7db`, which is `0d2d168` **plus** the per-shape GEMV
table that turned out not to work. The commit removes that table and nothing else —
it printed one empty row and the `!! one shape row only` guard — so every number
below is from the same measurement code. Recorded rather than glossed: a pod tree is
not at any commit unless it is checked
(`errors/2026-09-04-file-push-sync-is-not-a-checkout.md`).

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-04 | 0d2d168 | V100 | cuda sm70 | 27B ctx1024 | rung 4→8 step carried by GEMV | **92.9%** |
| 2026-09-04 | 0d2d168 | V100 | cuda sm70 | 27B ctx1024 | rung 2→4 step carried by GEMV | 84.1% |
| 2026-09-04 | 0d2d168 | V100 | cuda sm70 | 27B ctx1024 | GEMV calls/fwd, d1→d4 | 313.4 → 338.0 (+2.6%) |
| 2026-09-04 | 0d2d168 | V100 | cuda sm70 | 27B ctx1024 | GEMV ms/row, rung 2→4 / 4→8 | 7.52 / 7.64 |
| 2026-09-04 | 0d2d168 | V100 | cuda sm70 | 27B ctx1024 | speedup needed at M=8 to flip #22 | **1.34x** |

Source: `$HOME/tilerl-logs/bud7.log` on the V100.
