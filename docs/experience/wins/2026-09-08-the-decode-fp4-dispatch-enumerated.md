# The decode fp4 dispatch, enumerated — 2026-09-08

**Not a defect.** A read, recorded because two people inferred the dispatch from the plan table
and the table does not describe what runs.

## Context

The decode tick measures 29.67 ms against a 6.47 ms bandwidth lower bound (weights 94.4%, KV
0.5% — GQA has 4 kv heads), so there is 4.59x inside the kernels. The target is the fp4 weight
stream on the decode forward, and #240 had just found the *backward* fp4 kernel running Ampere
`mma.sync` on a Hopper card, worth 2.765x when fixed. The question: does the forward have it too?

## What runs

**No.** At B=8 the forward reaches `linear_fp4_mma8` (`backend.py:767`), whose inline PTX is
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (`kernels_linear.py:297`) — so it *is*
`mma.sync`, and that is a **measured verdict, not an unreachable `wgmma`**. CHANGELOG 2026-08-28:
mma8 "replaces the padded WGMMA decode paths", B=8 aggregate d512 219 → 286.7 tok/s (+31%). The
reason is shape: the compiled WGMMA M tiles start at 16 (see the table below) and 8 rows fill half
of one, while `m16n8k16` reaches the tensor cores without a warpgroup.

Three levers encode a three-way ordering — GEMV loses to WGMMA, WGMMA loses to mma8 at these M:

```
M = 1        scalar GEMV        (mma8 is 2.2x slower here: 39.9 vs 87 tok/s)
2 <= M <= 3  GEMV, M=1 packed   (_MGEMV = 3, backend.py:108)
4 <= M <= 8  linear_fp4_mma8    (_MX = 8, backend.py:105)
M >= 9       linear_fp4_fp8_*   (wgmma, via _CUDA_PLAN)
```

## The table does not describe the dispatch

`_CUDA_PLAN[("linear_fp4", "decode")]` names `linear_fp4_fp8_decode` (`backend.py:136`), but the
mma8 branch sits **after** the plan lookup and **returns**, so at 2 ≤ M ≤ 8 the plan's kernel is
never called. #240's entry states "prefill and decode fp4 both dispatch to `linear_fp4_fp8`,
already 52/60 `wgmma`" — true for prefill, false for a decode batch of 8, and derived from the
table. A comment now sits on that branch saying it precedes the table.

## Fill rate against B, computed from `_plan` and `_snap_mma_tile`

`_snap_mma_tile(m, 128)` returns the first of **16/32/64/128** that is ≥ m — the "WGMMA Square
policy: 16/32/64/128 compile, 48/80/96/112 do not" (`backend.py:48`). So the WGMMA M granularity
in this tree is **16, not 64**:

| B | kernel | bM | Mp | rows used |
|---:|---|---:|---:|---:|
| 1 | scalar GEMV | 1 | 1 | 100% |
| 2–3 | GEMV, M=1 packed | 16 | 16 | 12–19% |
| 4–8 | `linear_fp4_mma8` | 16 | 16 | 25–**50%** |
| 9–12 | `linear_fp4_fp8_decode` | 16 | 16 | 56–75% |
| **16** | `linear_fp4_fp8_decode` | 16 | 16 | **100%** |
| 17–32 | `linear_fp4_fp8_prefill` | 32 | 32 | 53–100% |
| 33–64 | `linear_fp4_fp8_prefill` | 64 | 64 | 52–100% |

At B=8 the tensor-core ceiling is **50% by shape**: `m16n8k16` takes 16 rows and 8 are padding,
visible in the asm as `{xa[k].x, zero, xa[k].y, zero}` — half the operands are literal zeros.

**B=8 → 16 is monotone: 50% → 100%, and it moves to `wgmma`.** No valley to cross; 9–15 all sit
above 50%. The only valley is B=2–3, which is *below* the current batch and bounded by
`_MGEMV = 3`. The next cliff is **B=17**, which crosses into the prefill bucket where bM jumps to
32 and the fill falls back to 53% — so the sweet spot is B=16, and B=32's 100% is on a kernel
tuned for long sequences, which is a measurement, not an inference.

This is a static read; nothing here is a wall-clock claim. Whether B=16's KV fits is a card
question — the training path hand-computes `num_blocks` and never calls `_fit_blocks`, which does
not model gradients or optimizer state anyway.

## Rule

**A dispatch table is a claim about intent; the branch that returns first is the dispatch.** Read
for `return` between the lookup and the call before quoting a table — the same inference was
published once already in a verified, carefully-measured entry.

And a lever's verdict names its loser. "small-M GEMV is 2.18x slower than WGMMA" and "mma8 beats
padded WGMMA at M ≤ 8" are both true and point opposite ways; using the first alone to judge the
mma8 path inverts the conclusion. Read which alternative a verdict was measured against.
