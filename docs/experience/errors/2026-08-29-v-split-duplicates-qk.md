# The GDN V split is slower, because splitting V duplicates the q/k half

## Context

`scripts/occ_gdn.py` measured the prefill kernel as SM-limited: 48 blocks on 78
SMs, and 2x the blocks in one launch costs 1.20x the time. I read that as "a
2-way V split is worth 1.67x on the kernel, +13.4% on prefill" and built it.

## The Measurement

VS=2, against VS=1 which the refactor reproduces by construction:

| | VS=1 | VS=2 |
|---|---:|---:|
| us/step | 3.05 | **4.22** |
| prefill/len512 | 2237.8 | 1873.6 (0.837x) |
| accuracy | 81.0% | 23.5% |
| full-scale parity | passes | out 165%, state 100%, window 88% |

Wrong *and* slower.

## Root Cause

The kernel's block owns one (value head, batch) and its 128 threads index BOTH
dimensions: V for the state columns it carries, and K for the q/k conv + L2
norm. q/k belong to the KEY head, so every block computes all K of them.

Splitting V halves the state work per block and **doubles the q/k work in
total** — the head's two blocks each redo the full q/k conv, SiLU and norm.
That is a real cost, not bookkeeping, and it swamps the occupancy gain.

The occupancy probe was not wrong; my inference from it was. It varied the
batch, so 2x the blocks were doing 2x the TOTAL work, and it showed the machine
can absorb them. It said nothing about splitting one block's work into two when
half of that work is per-block rather than per-column.

## What Would Actually Work

Hoist q/k out. They are already computed redundantly — 48 value heads share 16
key heads, so every q/k column is recomputed 3 times today, and a V split would
make it 6. A separate prologue kernel producing normalized q/k per key head
would remove the 3x redundancy AND make the V split free of duplication. That
is a different, larger change; nothing here argues for it beyond the arithmetic
above.

Reverted; VS=1 was identical to the shipped kernel by construction, so the knob
carries no information and goes with it.

## Rule

A measured upper bound bounds the thing it measured. "More blocks are absorbed"
and "this split is free" are different claims: check whether the split
duplicates work that the original did once, before spending the bound on it.
