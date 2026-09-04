# B is a shape axis too, and bucketing it is rejected: the padding costs 2-8x the compile it saves

**Date:** 2026-09-04
**Arch:** sm70 (Tesla V100-SXM2-32GB), 27B NVFP4 + 456M draft
**Task:** #74
**Instrument:** `scripts/read_kernel_cache.py` (the pod's tilelang cache, read by
`(rank, dtype)` signature with an mtime window) + `scripts/probe_b_axis.py` (CPU,
negative result)
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

## Measured: B costs 34 of the served model's 59 compiles, per kernel

From the pod's cache, matched on the full `(rank, dtype)` signature of every param
(`params.pkl` stores inputs, the scalar, **and** the `T.empty` outputs — see the
corrections below) **and filtered to the served geometry**, since the same cache holds
tiny-model compiles from CPU-parity runs:

| | `write_tokens_f32` | `paged_attention_split` |
|---|---:|---:|
| distinct (B, S) pairs | **59** | **59** |
| distinct S values | 25 | 25 |
| distinct B values | **7** (1-7) | **7** (1-7) |
| pairs / distinct S | **2.36** | **2.36** |
| compiles if B were one value | 25 — **34 fewer** | 25 — **34 fewer** |

Both kernels give identical counts over the same 558 entries, which is the expected
result: they are called once each per attention layer with the same batch and width.
B is ~58% of each kernel's compile count for the served model.

The unfiltered figures are **76 and 77 pairs, 41 and 42 attributable to B** — inflated by
the tiny config's 36-37 pairs. Either number supports the same verdict, but the served one
is the one that describes serving.

**Both figures are pre-fix.** The cache's last write is 14:40 and `6c6f6df` was
committed 14:48, so **zero** entries for either kernel postdate the fix; the live
server (started 14:49) has compiled nothing. The B-axis count is therefore a
property of the dispatch, which `029b27c`/`6c6f6df` did not touch, but the post-fix
cache is empty and cannot confirm it independently.

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
25 served S values cannot produce 269 compiles of one kernel. That run stays open.

## Four instrument corrections

**1. The CPU target cannot answer this.** `scripts/probe_b_axis.py` recorded nothing and
its assert caught it: `backend.py:889` falls back to the pool's torch loop when
`write_tokens` is absent from the arch's registry, and the CPU path dispatches
`paged_attention`, never `paged_attention_split`. The two kernels that bake B on the served
path do not run on the twin. The script is kept for its negative result.

**2. `params.pkl` stores the whole kernel signature, not the declared inputs.** I built
signatures from the `T.Tensor` lines and matched on rank tuples, which was wrong three
times before it was right:

- `(5,4,4,4)` f16/f32/f32/f32 is `paged_attention_split_**combine**` (PO[B,S,H,KVSPLIT,D]),
  not `_split`. `(4,2,1,4)` is neither kernel.
- `paged_attention_split` is **10** params, not the 6 it declares: `params.pkl` includes the
  scalar `scale` and the three `T.empty` outputs PO/PM/PL.
- Two arity-10 groups exist with the same rank tuple, 634 entries and 39. They differ only
  in **PO's dtype**: f16 is the live kernel, f32 is pre-#44 history. My first correct-arity
  guess still matched the dead one, reporting "B costs 0 of 7 compiles" for a kernel that
  had not run since #44.

Only dumping the concrete shapes of one entry from each group settled it. `KVSPLIT` appears
as 16 and 32 in the live group and is a factory arg (`backend.py:789` passes per-width ks),
so it is not a shape axis.

**3. Mb reads as 78 distinct values, and that is pre-fix history, not a missed call site.**
The live split kernel's BlockTable width takes 78 values across the cache — which would
mean `6c6f6df` failed. Splitting on the commit's own timestamp (`git log --format=%ct`,
14:48) against file mtime: **78 before, 0 after**. Every entry predates the fix.

**4. One cache serves three models, and I published a number over all of them.** The first
figures (41 of 76, 42 of 77) counted the tiny config's compiles alongside the served
model's — ~20% inflation. Filtering on `(H, D, Hkv) = (24, 256, 4)` gives **34 of 59** for
both kernels. This surfaced only because I had asserted the extra geometries were "trunk vs
draft" and then checked: `spec.py:312` builds the draft as `replace(trunk.cfg, ...)`, so it
shares H/D/Hkv and cannot be a distinct geometry at all. Every cache-derived count needs a
geometry filter as well as an mtime window.

## B is not the only unbucketed axis, and the per-dimension view is what shows it

The per-kernel scripts I wrote first counted (B, S) pairs, so they could not see that the
live split kernel has **98** distinct Q shapes against only **77** (B, S) pairs. Reading
every dimension separately (`read_kernel_cache.py` prints `=N` for a dim that never varied):

```
4f32,4f32,4f32,2i32,1i32,1i32,0f32,5f16,4f32,4f32     paged_attention_split, 634 entries
    p0 (Q):          [7, 35, 3, 2]      B=7  S=35  H=3  D=2
    p1 (KCache):     [24, 3, =16, 2]    NB=24  Hkv=3  block=16  D=2
    p3 (BlockTable): [7, 78]            B=7  Mb=78
```

H taking 3 values and D taking 2 is a *third* and *fourth* axis, and reading the values
shows what they are — three whole model geometries, not three head counts:

| H | D | Hkv | entries | what |
|---:|---:|---:|---:|---|
| 24 | 256 | 4 | 558 | `qwen38_27b()` (config.py:123-125) — the served model |
| 4 | 16 | 2 | 74 | `tiny()` (config.py:165-167) — the CPU test config |
| 8 | 16 | 1 | 2 | matches no config in the tree; a microbench's own literals |

**The draft head is not among them.** `spec.py:312` builds it as
`replace(trunk.cfg, num_layers=num_layers, ...)`, so it shares H, D and Hkv with the trunk
and cannot be a distinct geometry — my first reading of this table asserted "trunk-vs-draft
head geometry" and that is wrong. What actually inflates 77 (B, S) pairs to 98 Q shapes is
the pod's cache holding *tiny-model* compiles from CPU-parity runs alongside the served
model's, plus two stray microbench shapes.

That makes the extra 21 compiles irrelevant to serving rather than benign-for-a-good-reason:
they belong to other configs entirely. A per-axis printout is what distinguishes those cases;
the pair count attributes them to nothing.

## The window between the two #73 fixes is direct evidence they worked

`--since 029b27c` (the width fix) selects the 6 entries written before `6c6f6df` (the block
table). They read:

```
    p0: [=1, =64, =24, =256]      S pinned at 64
    p3: [=1, 6]                   Mb still taking 6 values
```

S is `=64` — the width fix is in effect — while Mb is the only remaining axis, which is
exactly the intermediate state, and `--since 6c6f6df` returns **no entries at all**. The
pair of windows is a positive and a negative control on the same instrument: an empty result
alone would equally well mean the mtime filter was broken.

## Rule

An axis being real and an axis being worth removing are two measurements. B is provably a
shape axis worth 41-42 compiles per kernel, and removing it is still a 2-8x loss, because
the padding a bucket introduces is discarded on one axis and executed on the other. Check
what the kernel does with the padding before pricing the bucket.

And a cache entry is identified by its full signature, not recognized by its shape. Four
readings of the same 634 files gave four different answers — wrong kernel, wrong arity,
wrong dtype, then right — and each intermediate one looked like a result: "B costs 0 of 7
compiles" was a complete, plausible sentence about a kernel that had not run since #44.
Where `params.pkl` carries no names, the only check is dumping the concrete shapes of one
entry per candidate group and reading which declaration they match.
