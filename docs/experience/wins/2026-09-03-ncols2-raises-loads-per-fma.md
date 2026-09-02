# ncols=2 is 1.82× — raising loads-per-FMA is the lever, V100 sm70, 2026-09-03

> Status: **ACCEPTED at the microbench level**, 1.82× at M=32 against a ≥1.25×
> threshold committed before the run, numerically correct, zero spills. Not yet
> wired into the shipped dispatch — that needs a padding gate and a full-model
> bench, tracked separately. Fourth attempt in this family and the first that pays.

## Context

Eight mechanisms for the M=32 gap were excluded by measurement, and the last one
(SMEM staging, 0.67×) revealed why they all failed: they preserved a ratio. Per row
per tile the kernel issues **2 loads and 8 HFMA2**, and reordering them (PIPELINE
0.99×), moving them to shared memory (SMEM 0.67×), changing block shape, splitting
the accumulator chain and halving the register budget all left **1 load : 4 FMA**
untouched. The only variant that ever beat the kernel was X_REUSE (8.79×), which
breaks the ratio by deleting the loads — not a fix.

`ncols=2` raises it: one thread computes **two** output columns from one X load, so
the same `xw[8]` feeds two decoded-weight sets and 16 FMAs. X traffic halves; W
traffic is unchanged (each column has its own `n` and its own bytes, and half as many
blocks run).

This was set aside two entries ago as "contraindicated at 255 registers". What made
it affordable is what the SMEM rejection *established*: 127 registers with no spills
is reachable in this kernel, so ~+41 for a second weight set and accumulator fits.

## Results

`scripts/ab_gemv_ablate.py`, `abl=6`. No relerr warning fired.

| shape | M | X_REUSE | PIPELINE | SMEM | **NCOLS2** |
|---|---:|---:|---:|---:|---:|
| gate_up | 32 | 9.06× | 0.99× | 0.68× | **1.79×** |
| down | 32 | 10.78× | 1.00× | 0.65× | **1.88×** |
| qkvz | 32 | 8.70× | 0.99× | 0.69× | 1.80× |
| gdn out | 32 | 5.31× | 1.00× | 0.65× | 1.83× |
| qkv | 32 | 8.48× | 0.99× | 0.69× | 1.77× |
| attn o | 32 | 5.66× | 1.00× | 0.65× | 1.84× |

Per-pass totals: M=1 **1.05×**, M=8 **1.72×**, M=32 **1.82×**. The M=1 row sits
inside the ±4% noise floor established for it across three runs, so it is "no
regression", not a gain.

## The mechanism, confirmed from the cubin before trusting the timing

| | LDG | HFMA2 | **HFMA2 per LDG** | registers | spills |
|---|---:|---:|---:|---:|---:|
| base | 363 | 1280 | **3.53** | 255 | 24 B |
| NCOLS2 | 338 | **2048** | **6.06** | 254 | **0** |

The ratio moved **1.72×**, and the measured speedup is 1.72× at M=8 and 1.82× at
M=32. **The speedup tracks the ratio**, which is what makes this the claimed
mechanism rather than a coincidence — the check SMEM taught me to run, having done
exactly what it promised (86% of loads removed) and lost anyway.

Note the register story is better than predicted: 254 with **zero spills**, against
the shipped kernel's 255 *with* 24 B of spill stores. Doubling the arithmetic per
thread made it cheaper per register, not dearer.

## What it is worth, and what is not yet done

Prefill is 8.92 ms/token at 4096 and ~85% of it is this kernel, so 1.82× gives

    8.92 → 5.50 ms/token,  TTFT 36.5 → 22.5 s

**A microbench win is not a model win**, and this is not wired into the dispatch
yet. Two things gate that:

1. **Padding.** The kernel derives `half = N // 2` from its own `N`, which is the
   *padded* `Np` the backend hands it. Every shipped shape is even and unpadded
   (34816 / 5120 / 6144 / 17408), so pairing is safe today — but a padded `Np` would
   pair a real column with a pad column and write garbage into `Y[:, N/2:]`, which
   the `[:Mr, :N]` slice **keeps**. That needs an explicit guard, not a coincidence.
2. **A full-model bench** on the 27B prefill path, not six shapes in isolation.

Until both are done the flag stays `abl=6`, off by default, and the shipped path is
byte-for-byte unchanged.

## Rule

**When every candidate fails, look for what they all preserved.** Eight mechanisms
were ranked by which quantity they reduced; the question was which quantity *binds*.
The answer was visible in any two cubins side by side — HFMA2 per load — and once
named, the fix followed in one attempt after six ticks of guessing.

Second: **a failed experiment's byproducts are evidence.** SMEM staging was rejected
at 0.67×, and its 127-register/zero-spill cubin is what made `ncols` affordable. The
rejection is what unblocked the acceptance.

Third: **confirm the mechanism moved before believing the timing.** 3.53 → 6.06
HFMA2/LDG, and a speedup that matches. Had the ratio not moved, 1.82× would have
been some other effect wearing this fix's name.

## Gate

`abl=6` is numerically correct (harness checks relerr on every shape) and defaults
to 0; 182 tests pass. `ncols` requires even N — enforced when it ships, not now.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | e8e7c95 | V100 | cuda sm70 | GEMV M=32 | ncols=2 (abl=6) | **1.82× — accept** |
| 2026-09-03 | e8e7c95 | V100 | cuda sm70 | GEMV M=8 | ncols=2 | 1.72× |
| 2026-09-03 | e8e7c95 | V100 | cuda sm70 | GEMV M=1 | ncols=2 | 1.05× (noise floor ±4%) |
| 2026-09-03 | e8e7c95 | V100 | cuda sm70 | GEMV M=32 | HFMA2 per LDG | 3.53 → **6.06** |
| 2026-09-03 | e8e7c95 | V100 | cuda sm70 | GEMV M=32 | registers / spills | 254 / **0** (was 255 / 24 B) |
| 2026-09-03 | pending | V100 | cuda sm70 | qwen38-27b | prefill ms/token | pending-remote |
