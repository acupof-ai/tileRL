# sm70 split-KV decode attention: 2K ctx 5.3 → 20.5 tok/s — 2026-08-31

> Status: Shipped

## Context

B=1 decode fell off a cliff with context: 20.3 tok/s at 31 tokens, 8.7 at 1K,
5.3 at 2K. Short context was fine — V100 900 GB/s vs H20 4000 GB/s is 4.44×,
and 87.5 → 20.3 tok/s is 4.31×, so 97% of the bandwidth ratio. The scaling was
the anomaly, not the absolute number.

## What Worked

`kernels.py` generic `paged_attention` is sm70's only decode path, and it has
two defects that only bite at B=1:

```python
with T.Kernel(B, H, threads=threads) as (bb, hh):   # B=1, H=24 -> 24 blocks on 80 SMs
    for pos in T.serial(upper):
        for d in T.serial(D):                       # serial fragment reduction
            s[0] += Q[bb, t, hh, d] * KCache[blk, hkv, off, d]
```

24 blocks on an 80-SM card, and the dot product is a `T.serial` fragment
reduction — so each block runs **one** active thread while the other 63 idle.

Two-point slope cancels the ctx-independent terms (GEMM + GDN):

(190 − 114) ms ÷ (203.7 − 103.4)M FMA = **0.76 ns/FMA** ≈ 1.16 clocks
@1.53 GHz — the theoretical rate of a single-threaded serial scalar loop.
Direct proof, not inference. The same KV is 134 MB = 0.15 ms at bandwidth.

sm90 never exposed it: it dispatches to `paged_attention_decode` (split-KV
flash-decoding). Same class as `gdn_chunk_fused` registered only in
`_SM90_KERNELS`.

**Fix**: split the position loop across the grid — `T.Kernel(KVSPLIT, H, B)`,
each block owning a contiguous history slice, then combine the partials in the
log domain (`make_paged_attention_split` + `_split_combine`, KVSPLIT=16).

The sm90 kernel cannot be reused: it needs `T.gemm` + bf16 and sm70 has
neither. And `T.serial(D)` cannot become `T.Parallel(D)` — `kernels.py:14-18`
records that Metal cannot cross-thread-reduce a fragment scalar, so a parallel
reduction silently computes per-thread partials. The grid split sidesteps both:
the parallelism is the **grid**, not a cross-thread reduction, so the serial
dot survives untouched.

| ctx | before | after | speedup |
|---:|---:|---:|---:|
| 31 | 20.3 | 25.8 | 1.27× |
| 1052 | 8.7 | 23.1 | 2.65× |
| 2072 | 5.3 | 20.5 | 3.87× |

Before: 20.3 → 5.3 (3.8× falloff). After: 25.8 → 20.5 (20% falloff).

## Blast radius

Registered in the **sm70 cell only**. The kernel source is target-neutral, but
the win is filling 80 SMs, so it is a loss where `T.Kernel` lowers to a serial
loop (cpu/rocm — 16 slices plus combine overhead over one pass) and unproven on
metal. First draft put it in `_CPU_KERNELS`, which cpu/metal/rocm/sm70 all
inherit — that would have changed the decode path on four targets to chase one.
Dispatch is arch-gated (`self.arch == "sm70" and s == 1`) so registry presence
alone cannot re-enable it elsewhere.

Verified: `paged_attention_split` present in `fp4/sm70` only; absent from
bf16/fp4 cpu, metal, rocm, sm90.

## Correctness

`scripts/check_split_attn_parity.py` builds both kernels from source (not the
registry, so it gates on any target) and compares against the generic kernel at
n = 1, 15, 16, 17, 37, 100, 129 — straddling BLOCK=16 and KVSPLIT=16 so empty,
ragged, and sub-page slices are all covered.

- CPU: max |split − generic| = 2.4e-07, all OK
- V100 sm70: max 1.8e-07, all OK
- Full suite: 146 passed, 4 skipped

## Results

| date | commit | machine | target | model | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|
| 2026-08-31 | pre-fix | V100 sm70 | cuda | Qwen3.8-27B-NVFP4 | 190 @2K | 5.3 |
| 2026-08-31 | post-fix | V100 sm70 | cuda | Qwen3.8-27B-NVFP4 | 49 @2K | 20.5 |

## Rule

A `T.Kernel(B, H)` grid is a decode bug waiting to happen: at B=1 the block
count collapses to H, and if the inner reduction is serial each block is
single-threaded. Any decode kernel gets its parallelism from splitting the
**history**, not from the batch/head dims.

Second rule, on measurement: the first post-fix number was 289 tok/s, which is
**above the weight-bandwidth roofline** (16.04 GB streamed / 900 GB/s = 17.8
ms/tok = 56.1 tok/s; this entry originally cited a remembered 14 GB / 64 tok/s
— `errors/2026-09-02-roofline-is-the-streamed-subset.md`) — JIT compile had
landed inside the `lo` timing point and the slope subtracted it out.
`bench_b1_decode.py` now warms up first. Always check a throughput claim against
the roofline before believing it.
