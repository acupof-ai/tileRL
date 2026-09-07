# load_kv writes two copies, not two per block — 2026-09-07

> **Status: pending-remote.** The launch count and the byte volume are measured
> here; the wall clock is not. The card number is the restart-hit bench against
> `wins/2026-09-05-the-ssd-tiers-read-path-did-not-exist.md`'s **1.738–1.821x**,
> and it fills this entry in the next card window. No CHANGELOG line until then.
>
> Row 50 PR A. Approach: `docs/design-ssd-read-path.md`.

## Context

Scoping the async fetch (row 50 PR B) rather than measuring anything. `load_kv`
faults a prefix off the SSD tier into fresh blocks, and it did so one block at a
time:

```python
for i, b in enumerate(blocks):
    pool.k_pool[:, b].copy_(blob["k"][i].to(pool.device))
    pool.v_pool[:, b].copy_(blob["v"][i].to(pool.device))
```

At 30,000 tokens that is 1,875 blocks — **3,750 copies of 512 KiB each**, for
1.83 GiB of KV. The reason it matters is not the loop's own cost: PR B puts this
read on a side stream so it overlaps other sessions' compute, and 3,750 launches
contend with the prefill stream rather than overlapping it.

## What worked

Two `index_copy_` calls. The spill blob is `[nblocks, planes, …]` and the pool is
`[planes, nblocks, …]`, so `index_copy_` reads the permuted view — non-contiguous
on the host side, which it accepts.

**No on-disk format change.** Making the spill plane-major would remove the
permute, and it was the first thing I tried. `_recover` adopts entries by **file
size alone** (`kv_cache.py:504-512`, deliberately: reading tensors there would
load a 20 GiB directory to answer what the filename answers), so an
already-written blob in the old layout would be adopted and misread silently. The
fingerprint covers every config field but not the tier's own serialization
layout. Keeping the format fixed makes that unreachable instead of guarded.

## Measured, and what is not

| quantity | before | after |
|---|---:|---:|
| launches, 100 blocks | 200 `copy_` | **0 `copy_` + 2 `index_copy_`** |
| bytes moved, 200 blocks | 400.0 MiB | **400.0 MiB** |
| wall clock on a card | — | **not measured** |

**The claim is launch count and nothing else.** The same volume moves; a counter
is not seconds. A CPU host-to-host timing reads 27.8 → 22.0 ms at 200 blocks
(1.26x), but that is memcpy on the test target and says nothing about a device
launch cost — it is recorded here so nobody re-derives it as a speedup.

A peer caught this before it left as "200 → 2, a 100x": volume is not time, and
if the win is launch overhead the claim has to say launches.

## The gate, and the luck it removes

`test_load_kv_writes_the_same_bytes_in_two_calls_not_two_per_block` asserts byte
parity against the per-block loop, on **out-of-order non-contiguous blocks filled
with noise**. That matters: the neighbouring restart test fills block *i* with the
constant *i+1* in ascending order, so there a sorted index or a transposed view
writes the right bytes to the right place by luck, and every assertion passes.

Three mutations, two reds:

| mutation | result |
|---|---|
| `sorted(blocks)` for the index | **red** — "block 3 holds another block's KV" |
| write `v_pool` from `blob["k"]` | **red** — "v differs" |
| `permute(1,0,2,3,4)` → `transpose(0,1)` | green, **correctly** — on a 5-D tensor these are the same operation, so it was never a mutation |

The third is worth keeping in the record: writing a mutation is not the same as
making a change, and a green from one proves nothing about the gate.

The test also asserts **zero per-block `copy_` calls**, so a revert to the loop
fails on the count rather than passing on the bytes — the bytes would still be
right.

## Rule

A launch count and a byte volume are different claims, and the loop-to-batch
rewrite changes only the first. State which one the win is before quoting a
ratio; "N times fewer calls" reads as "N times faster" to every later reader,
including the one who wrote it.
