# B is a shape axis too, and bucketing it is rejected: the padding costs 2-8x the compile it saves

**Date:** 2026-09-04
**Arch:** sm70 (Tesla V100-SXM2-32GB), 27B NVFP4 + 456M draft
**Task:** #74
**Instrument:** `scripts/probe_b_axis.py` (CPU, negative result) + the pod's tilelang
cache read by `(rank, dtype)` signature and mtime
**Verdict:** REJECTED — do not bucket B. The measurement stands; the fix does not.

## Where this came from

#73 fixed two unbucketed shape axes in `spec.py` because `kernels_mma.py:22` reads

```python
B, S, H, D = T.const("B, S, H, D")
```

and `params.pkl` showed the whole tuple is what varies per compile. **B is in that same
tuple**, and nothing rounds it: `engine.py:666` sizes rows by `len(reqs)`, `spec.py:412`
by `len(plan)`, and `spec.py:484`'s chain loop by `len(live)`, which *shrinks* mid-chain as
rows hit block boundaries. 20 kernels across the four kernel files bake B this way.

## Measured: B costs 41 of 76 compiles

From the pod's cache, `write_tokens_f32` entries only (matched on the 7-param
`(4,4,4,4,2,1,1)` rank+dtype signature, not on a guessed name):

| | count |
|---|---:|
| distinct (B, S) pairs | **76** |
| distinct S values | 35 |
| distinct B values | **7** (1,2,3,4,5,6,7) |
| pairs / distinct S | **2.17** (1.0 would mean B never varied) |
| compiles if B were one value | 35 — **41 fewer** |

So B is not a theoretical axis. It is 54% of this kernel's compile count.

## And bucketing it is still wrong

Padding B is not like padding S. A padded prefill row is thrown away inside the kernel
(`kernels_mma.py:71` gates on `SeqQLens`), which is why #73's width fix was free. A padded
*batch* row is a real row through every kernel of every tick — #41 measured one at **3.3x**
the cost of a useful row at B=4.

The compile is paid once per (B, S). The padding is paid on every tick of that request's
life. `/health` on the live server reads `decode_forwards 124` against `prefill_forwards 5`,
so a request runs ~124 ticks. Break-even, at 1108 ms/compile (#73's measured figure) and a
35.56 ms B=1 tick:

| B | bucket | pad rows | added ms/tick | ticks to break even | verdict |
|---:|---:|---:|---:|---:|---|
| 1, 2, 4 | — | 0 | — | never pads | free |
| 3 | 4 | 1 | 39.1 | **28.3** | loss |
| 5 | 8 | 3 | 70.4 | **15.7** | loss |
| 6 | 8 | 2 | 39.1 | **28.3** | loss |
| 7 | 8 | 1 | 16.8 | **66.1** | loss |

Every B that would need padding breaks even in 16-66 ticks against the ~124 a request
actually runs. Bucketing B pays **2-8x the compile it saves**. Rejected.

The asymmetry is the point: S and Mb were free to bucket because the kernel discards the
padding, and B is not, because nothing discards a batch row.

## What this does NOT explain

I opened #74 predicting this was the mechanism behind the 269-compile B=8 run that
`errors/2026-09-04-the-recompiles-reproduce-and-are-not-a-shape-set.md:165` leaves
unexplained. **It is not, and the cache says so:** the B values present are 1-7, and that
run was B=8 at depth 1, where `spec.py:484`'s chain loop never executes. 7 B values across
35 S values cannot produce 269 compiles of one kernel. That run stays open.

## Two instrument corrections

**1. The CPU target cannot answer this.** `scripts/probe_b_axis.py` recorded nothing and
its assert caught it: `backend.py:889` falls back to the pool's torch loop when
`write_tokens` is absent from the arch's registry, and the CPU path dispatches
`paged_attention`, never `paged_attention_split`. The two kernels that bake B on the served
path do not run on the twin. The script is kept for its negative result.

**2. `params.pkl` has no names, so my first two labels were wrong.** I mapped rank tuples
to kernels by guessing: `(4,2,1,4)` and `(5,4,4,4)`. Dumping the full param lists showed
`(5,4,4,4)` with dtypes f16/f32/f32/f32 is `paged_attention_split_**combine**`
(PO[B,S,H,KVSPLIT,D]), and `(4,2,1,4)` is neither kernel. The B×S table I printed first was
for two kernels I had not intended to measure. Matching on rank **and dtype** against the
declaration fixed it. `paged_attention_split` itself matched **zero** entries — its sm70
partials went f16 in #44, so the signature I built from `kernels.py:31` is stale, and that
kernel's axis count is **unmeasured** rather than zero.

## Why the cache's unbucketed S values are not a #73 regression

The cache holds S ∈ {13,15,16,23,24,26,37,...} — not multiples of 64 — with mtimes as late
as 14:40 today, which would contradict #73's "a first visit compiles 0". It does not: the
live server (pid 1829415) started at **14:49**, its own log has **0** occurrences of
"compil", and the pod checkout is at `6c6f6df` with both fixes at `spec.py:407` and `:413`.
Those entries are pre-fix history from the day's bench and probe runs. The cache is
cumulative, so every claim read from it needs an mtime window.

## Rule

An axis being real and an axis being worth removing are two measurements. B is provably a
shape axis worth 41 compiles, and removing it is still a 2-8x loss, because the padding a
bucket introduces is discarded on one axis and executed on the other. Check what the kernel
does with the padding before pricing the bucket.

And when the artifact carries no names, the signature must be matched against the
declaration, not recognized. Two of my three labels were wrong on rank alone; adding dtype
settled all three and revealed the third kernel was not in the cache at all.
