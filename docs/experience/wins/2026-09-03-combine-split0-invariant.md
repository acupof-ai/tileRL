---
question: Can the split-KV combine's all-empty row happen, and does the guard against it do anything?
source: H20 sm90 card 7, tilelang 0.1.13, torch 2.11.0+cu129, scripts/verify_combine_guard.py
---

# The combine's all-empty row is unreachable, and the reason was written down nowhere

`paged_attention_combine` merges the split-KV partials as
`Out = Σ_s w_s·PO_s / Σ_s w_s·PL_s` with `w_s = exp2(PM_s − max_s PM_s)`. An
empty split carries the sentinel `PM = -inf, PL = 0`. If *every* split for a row
carried it, `max_s PM_s` is `-inf` and both halves are NaN: `exp2(-inf − -inf)`
and `0/0`.

Two separate NaN-freedoms rest on that row not occurring, and neither the code
nor the docstring said why it does not.

## Why it cannot occur

Split 0 always runs at least one tile. For `n ≥ 1`,
`tiles = ceildiv(n, 64) ≥ 1` ⇒ `per = ceildiv(tiles, KVSPLIT) ≥ 1`, so `t0 = 0`
and `t1 = min(tiles, per) ≥ 1`. Tile 0 holds key 0, and every chain row may
attend it: row `i`'s causal bound is `hist + i%W + 1` with
`hist = SeqLens − SeqQLens ≥ 0`, so `0 < hist + i%W + 1` unconditionally.

Measured, KVSPLIT=16 — splits running zero tiles:

| n | 1 | 8 | 63 | 64 | 65 | 100 | 512 | 1024 | 2048 | 65536 |
|---|---|---|---|---|---|---|---|---|---|---|
| empty splits | 15 | 15 | 15 | 15 | 14 | 14 | 8 | 0 | 0 | 0 |

15 of 16, never 16. So `m[0]` is always finite, each empty split contributes
`exp2(-inf − finite) = 0`, and `l[0] > 0` because the non-empty split's `PL` is
positive.

`n = 0` is the one input that would reach the row. Whether a row with
`SeqLens = 0` can arrive here is not evident from the source.

## Why a guard and not an assert

The host cannot check it. `seq_lens` is a device tensor, and reading it would
sync — which this path must not do: the split count is chosen host-static
precisely so the tick stays graph-capturable (`backend.py:548-552`, *"split count
from the pool's reach (host-static, graph-safe)"*). So the check goes where the
value already lives, in the kernel, at two `T.if_then_else`s on a
32-thread-per-row kernel.

## What the measurement adds over "no NaN observed"

Nothing on the live path can distinguish the guard from its absence — the row is
unreachable, so both spellings print no NaN. The check that means something is
the one that fails without it. Hand-built partials with `PM = -inf` and `PL = 0`
on **every** split, fed to both kernels:

| combine | NaN | max \|out\| |
|---|---|---|
| guarded (this PR) | **0 of 256** | 0.0 |
| the pre-guard arithmetic, rebuilt | **256 of 256** | — |

Blast radius on the live path, guarded combine against the dense
`paged_attention` kernel, every geometry a verify tick reaches:

| W | n=64 | n=100 | n=4096 |
|---|---|---|---|
| 1 | 1.7e-03 | 2.6e-03 | 2.9e-03 |
| 2 | 2.0e-03 | 2.5e-03 | 2.8e-03 |
| 4 | 1.9e-03 | 2.1e-03 | 2.9e-03 |
| 8 | 2.8e-03 | 2.3e-03 | 2.9e-03 |

All finite, all inside the 2e-2 bf16 tolerance, `KVSPLIT` 16 at n≤100 and 64 at
n=4096, `block_m` 16/16/32/64. A row with any non-empty split takes the
arithmetic it took before — the guards fire only on the row that cannot occur.

## Rule

An unreachable-by-construction branch needs the construction written down, or
the next reader re-derives it as a live bug. Two of them here rested on split 0
being non-empty and nothing stated it — a kernel-layer reader has no way to see
the argument.

A guard whose triggering state the live path cannot produce is unproven by any
number of clean runs. Prove it by running the arithmetic it replaced on the
state it guards against: 0 NaN means something only next to 256.
