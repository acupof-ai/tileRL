# SMEM staging is 0.67× — the binding quantity is loads per FMA, V100, 2026-09-03

> Status: **rejected at 0.67×** (33% SLOWER), threshold ≥1.30× committed before the
> run, output numerically correct. It removes 86% of the global loads and loses
> anyway, which is the result that finally explains the whole search.

## Context

`PIPELINE` (reorder the loads) measured 0.99×, so the cost is throughput, not
latency, and the remaining move was to *remove* loads. The kernel has an obvious
redundancy for that: `X[0, base]` carries no `ni`, so all four `n_partition` column
groups in a block load the **same** X addresses — 4× redundant global traffic that
one shared-memory stage collapses.

Priced before writing, from the LDG arithmetic: X is 98.5% of tile-body loads at
M=32, staging keeps one group of four, so at most 73.8% of loads go away → **2.92×
ceiling** if loads are 89% of the time and an LDS read were free. Threshold set at
1.30× precisely because LDS is not free.

Two implementation facts fixed before measuring, not after:

- M=32 × `GROUP`×`block_K` halves is **128 KB** against Volta's 96 KB/SM, so the `g`
  loop moved out to tilelang and the extern runs `G=1` on a 32 KB slice.
- The first version staged with `if ni == 0`, which cuts traffic to 1/4 but issues
  **32 serial LDG per lane** instead of 8 — and at 255 registers there is one block
  per SM with nothing to overlap that with. Rows were spread across the four groups
  instead. **A candidate's own implementation flaw impersonates "this direction does
  not work"**, so the pod run in flight was killed (by fd-verified pid) rather than
  allowed to measure my bug.

## Results

| shape | M | X_REUSE | PIPELINE | **SMEM** |
|---|---:|---:|---:|---:|
| gate_up | 32 | 8.97× | 0.99× | **0.68×** |
| down | 32 | 10.59× | 1.00× | **0.65×** |
| qkvz | 32 | 8.77× | 0.99× | 0.68× |
| gdn out | 32 | 5.63× | 1.00× | 0.65× |
| qkv | 32 | 8.60× | 0.99× | 0.68× |
| attn o | 32 | 5.41× | 1.01× | 0.66× |

Per-pass totals: M=1 0.95×, M=8 **0.72×**, M=32 **0.67×**. No relerr warning, so the
staged kernel is numerically correct — this is a real slowdown, not a broken variant.

## Why, from the cubin

Counts read back off the compiled kernel (the check I skipped once and promised to
run):

| | LDG | LDS | STS | BAR | HFMA2 | registers |
|---|---:|---:|---:|---:|---:|---:|
| base (G=4) | 363 | 0 | 0 | 0 | 1280 | 255 |
| SMEM (G=1) | **51** | 64 | 16 | 2 | 256 | 127 |

Staging did exactly what it promised — **86% of the global loads are gone**, spills
are zero, registers halved to 127. And it is 33% slower.

The reason is a ratio that none of the numbers above changed. Per row, per tile:

| variant | loads | HFMA2 | loads : FMA | measured |
|---|---:|---:|---:|---:|
| base | 2 × LDG.128 | 8 | 1 : 4 | 1.00× (ref) |
| PIPELINE | 2 × LDG.128 | 8 | 1 : 4 | 0.99× |
| **SMEM** | 2 × **LDS.128** | 8 | **1 : 4** | 0.67× |
| X_REUSE | ~0 (hoisted) | 8 | — | 8.77× |

ptxas vectorized the eight staged words into two `LDS.128`, so staging **swapped LDG
for LDS one-for-one** and left the ratio at 1:4, then added the stage itself (8 LDG +
16 STS + 2 barriers) on top. The only variant that beat the kernel is the one that
broke the ratio — by deleting the loads entirely, which is not a fix.

**The binding quantity is loads per FMA.** Every failed candidate preserved it:
reordering (PIPELINE), moving to another memory space (SMEM), changing block shape
(`n_partition`), splitting the accumulator chain, halving registers (`min_blocks`).
Eight mechanisms, one invariant.

The two barriers are worth naming separately: `T.sync_threads()` twice per tile ×
10 tiles per column, with 4 warps resident, is a full-block stall each time nothing
else can fill.

## What this decides

**Every fix that keeps 1 load : 4 FMA is dead**, which retires the whole family the
last six ticks explored. Raising the ratio requires one row's X to feed *more* FMAs,
i.e. each thread holding X for several output columns — the `ncols` shape, whose
register cost lands on the decoded weights.

That is newly affordable and the cubin says why: **the staged kernel runs at 127
registers with zero spills**, half the 255 the shipped kernel takes. The `ncols`
direction was set aside as "contraindicated at 255 registers"; from a 127-register
kernel there is room. Whether it pays is a measurement — the SMEM stage's own
overhead (barriers, STS) would still be there, and 1 : 8 needs `ncols=2` to also
avoid re-reading W.

Prefill is 8.92 ms/token at 4096 (TTFT 36.5 s), ~85% this kernel, so the prize is
real. But this is the third rejection in a row from the same family, and the next
attempt should be the ratio, or nothing.

## Rule

**Read the ratio, not the count.** "Removes 86% of global loads" is true and
irrelevant; the kernel's cost tracks loads *per FMA*, and swapping memory spaces
preserves it. Six candidates were ranked by which quantity they reduced when the
question was which quantity binds.

Second: **check a candidate's own implementation before letting it vote.** The
`ni == 0` staging would have measured ~4× worse for a reason that has nothing to do
with staging, and I would have recorded "SMEM staging rejected" either way — same
verdict, wrong evidence.

Third: **a rejection that halves the register count is not worthless.** 255 → 127
with no spills reopens `ncols`, which was excluded on register grounds. Record what a
failed experiment *established*, not only what it refuted.

## Gate

`abl=5` is numerically correct and the harness asserts it; it stays a factory flag
defaulting to 0, so the shipped path is untouched. 182 tests pass.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | d9bc18e | V100 | cuda sm70 | GEMV M=32 | SMEM staging (abl=5) | **0.67× — reject** |
| 2026-09-03 | d9bc18e | V100 | cuda sm70 | GEMV M=8 | SMEM staging | 0.72× |
| 2026-09-03 | d9bc18e | V100 | cuda sm70 | GEMV M=32 | LDG, base vs staged | 363 → **51** |
| 2026-09-03 | d9bc18e | V100 | cuda sm70 | GEMV M=32 | LDS / STS / BAR added | 64 / 16 / 2 |
| 2026-09-03 | d9bc18e | V100 | cuda sm70 | GEMV M=32 | registers, base vs staged | 255 → **127**, 0 spills |
