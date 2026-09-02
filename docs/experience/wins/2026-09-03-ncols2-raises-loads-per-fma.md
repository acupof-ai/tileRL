# ncols=2 is 1.82× — raising loads-per-FMA is the lever, V100 sm70, 2026-09-03

> Status: **SHIPPED.** 1.82× on the microbench at M=32 against a ≥1.25× threshold
> committed before the run, and **1.59× on the 27B's prefill at 4096 (8.66 → 5.47
> ms/token, TTFT 35.5 → 22.4 s)**. Greedy text is identical to `ncols=1` on real
> prompts at both M=1 and M>8. Fourth attempt in this family and the first that pays.

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

`scripts/ab_gemv_ablate.py`, the `NCOLS2` column. No relerr warning fired.

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

## End to end on the 27B

`scripts/ab_prefill_ncols.py` — one process, one engine, one set of weights,
flipping `backend._NCOLS` between measurements. Arms run **nc2, nc1, nc2** so drift
shows up as the two nc2 readings disagreeing instead of as a gain; they agreed to
1.000× at every context.

| ctx | nc1 ms/token | **nc2** | gain |
|---:|---:|---:|---:|
| 512 | 7.81 | **4.61** | 1.69× |
| 2048 | 8.17 | **4.97** | 1.64× |
| 4096 | 8.66 | **5.47** | **1.59×** |

TTFT at 4096: **35.5 → 22.4 s**. The nc1 arm reproduces the recorded 7.89/8.33/8.92
baseline to within 1-3%, so it is the control it claims to be — this is the check
the first attempt failed (below).

1.59× end to end against 1.82× in the kernel is Amdahl, and inverting it prices the
rest of prefill without a profiler: a 1.82× kernel produces gain
`1 / (s/1.82 + (1-s))`, so

| ctx | measured | implied GEMV share |
|---:|---:|---:|
| 512 | 1.694× | 0.91 |
| 2048 | 1.644× | 0.87 |
| 4096 | 1.585× | 0.82 |

consistent with the ~85% used to price this, and the trend is the interesting part:
the GEMV's share **falls** with context, so what grows is attention — the one part
of prefill that is quadratic. At 4096 the non-GEMV 18% is now the ceiling on any
further GEMV work.

**Text parity, `scripts/parity_ncols.py`**: greedy continuations are identical
between arms for two real prompts, at 10-17 tokens (M=1) and 600 tokens (prefill
above the 32-row rung). This is the check that matters for this kernel, since
pairing column j with j + N/2 fails by corrupting *half* the output columns.

## The two gates, both closed

1. **Padding.** The kernel derives `half = N // 2` from its own `N`, which is the
   *padded* `Np` the backend hands it. Every shipped shape is even and unpadded
   (34816 / 5120 / 6144 / 17408), so pairing is safe today — but a padded `Np` would
   pair a real column with a pad column and write garbage into `Y[:, N/2:]`, which
   the `[:Mr, :N]` slice **keeps**. The dispatch gates on
   `nc = _NCOLS if Np == N and N % 2 == 0 else 1`, and
   `tests/test_ncols_contract.py` checks the guard's own source (both negative
   controls verified).
2. **A full-model bench**, above. The first attempt was invalid in a way worth
   knowing: `ncols` was passed positionally, landed on `abl`, and both arms ran
   ablation kernels that return wrong numbers by design —
   [`errors/2026-09-03-the-ab-measured-abl-not-ncols.md`](../errors/2026-09-03-the-ab-measured-abl-not-ncols.md).

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

Fourth, from the failed first A/B: **the arm that is supposed to be current
behavior is a control — price it against its recorded number before reading the
delta.** Here nc1 read 7.81/8.17/8.66 against a recorded 7.89/8.33/8.92, so the
1.59× is a delta between a verified baseline and a new kernel, not between two
unknowns.

## Gate

Greedy text identical to `ncols=1` on real prompts at M=1 and M>8
(`scripts/parity_ncols.py`); microbench relerr zero on every shape; 186 tests pass.
`ncols`, `abl` and `min_blocks` are keyword-only on the factory, and the dispatch
gates `ncols=2` on `Np == N and N % 2 == 0`.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | e8e7c95 | V100 | cuda sm70 | GEMV M=32 | ncols=2 | **1.82× — accept** |
| 2026-09-03 | e8e7c95 | V100 | cuda sm70 | GEMV M=8 | ncols=2 | 1.72× |
| 2026-09-03 | e8e7c95 | V100 | cuda sm70 | GEMV M=1 | ncols=2 | 1.05× (noise floor ±4%) |
| 2026-09-03 | e8e7c95 | V100 | cuda sm70 | GEMV M=32 | HFMA2 per LDG | 3.53 → **6.06** |
| 2026-09-03 | e8e7c95 | V100 | cuda sm70 | GEMV M=32 | registers / spills | 254 / **0** (was 255 / 24 B) |
| 2026-09-03 | 01fa731 | V100 | cuda sm70 | qwen38-27b | prefill ms/token @4096, HEAD control | 8.91 (recorded 8.92, 0.1%) |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | prefill ms/token @4096 | 8.66 → **5.47 (1.59×)** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | prefill ms/token @2048 | 8.17 → **4.97 (1.64×)** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | prefill ms/token @512 | 7.81 → **4.61 (1.69×)** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | TTFT @4096 | 35.5 → **22.4 s** |
| 2026-09-03 | (this) | V100 | cuda sm70 | qwen38-27b | greedy text vs ncols=1 | identical, M=1 and M>8 |
