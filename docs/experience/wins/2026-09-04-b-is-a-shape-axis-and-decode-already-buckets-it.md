# B is a shape axis worth 34 compiles; my REJECT of bucketing it was wrong, and decode already does it

**Date:** 2026-09-04
**Arch:** sm70 (Tesla V100-SXM2-32GB), 27B NVFP4 + 456M draft
**Task:** #74
**Instrument:** `scripts/read_kernel_cache.py` (the pod's tilelang cache, read by
`(rank, dtype)` signature with an mtime window and a geometry filter) +
`scripts/probe_b_axis.py` (CPU, negative result)
**Verdict:** B costs **34 of the served model's 59 compiles** per kernel. Bucketing it is
**free** on the rung cost model — my first REJECT used a cost model I invented instead of
the one this repo already measured. Decode already buckets B (`engine.py:817`); the draft
and the eager verify path do not, which is the real open scope.

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

## And bucketing it: my first verdict was REJECT on a cost model I invented

**The REJECT below is withdrawn.** I priced each padding row at `TICK_MS / B × 3.3` —
39-70 ms — and both inputs were wrong:

- **3.3× is not a multiplier on a tick share.** `#41` fitted
  `tick_ms = 25.6 + 7.53·launched + 2.29·useful`; 3.3× is the ratio of the *marginal
  launched row* (7.53 ms) to the *marginal useful row* (2.29 ms).
- **A tick costs its rung, not its rows.** `wins/2026-09-04-rung-cost-not-useful-rows.md`
  measured rung 8 with 3 of 8 rows idle at **82.15 ms** against **83.40 ms** fully packed —
  60% more useful rows for 1.5% more time, and `engine.py:870` carries that comment.

So padding B costs nothing *unless it pushes M = B·W onto a higher rung*. At the served
config (depth 1, W=2), for every B the cache actually shows:

| B | M=B·W | rung | B→bucket | M₂ | rung₂ | rung step | cost |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2 | 2 | 1 | 2 | 2 | 0 | free |
| 2 | 4 | 4 | 2 | 4 | 4 | 0 | free |
| 3 | 6 | 8 | 4 | 8 | 8 | 0 | free |
| 4 | 8 | 8 | 4 | 8 | 8 | 0 | free |
| 5 | 10 | 32 | 8 | 16 | 32 | 0 | free |
| 6 | 12 | 32 | 8 | 16 | 32 | 0 | free |
| 7 | 14 | 32 | 8 | 16 | 32 | 0 | free |

**Every one lands on the rung it already occupied.** 0 of 7 lose.

And the result is not specific to the served width — sweeping every W the engine can reach
(`scripts/price_b_bucket.py`):

| W | B values whose bucket crosses a rung |
|---:|---|
| 1, 2, 3, 4 | none — free for every B |
| **5** | **B=5, 6** |
| **6** | **B=5** |
| 7, 8 | none — free for every B |

Only W=5 and W=6 cost anything, and W=5 is depth 4 — the width `engine.py:849` **warns**
about rather than clamps, so it is reachable. So bucketing B is free at every width the
ladder endorses and costs only where the ladder already says the width is wrong. The 34
compiles are a real saving.

I nearly published "free at W=2" instead: my first negative control was W=4, which is also
free, so it did not discriminate. Sweeping the whole range is what turned a config-specific
claim into a property of the ladder.

## The scope was also wrong: decode already does this, the draft never can

`engine.py:817` `_graph_bucket` rounds rows to `_GRAPH_BUCKETS = (1,2,4,8,16,24,32,48,64,96,128)`
and `_run_decode_graph` parks the padding rows on a spare slot and block
(`engine.py:921-926`). **B-bucketing is already implemented on the decode graph path.**

`spec.py` contains the string "graph" **zero** times: the draft always runs eager, so its B
is never bucketed — and neither is the eager verify/prefill path the compiles were measured
on. That is the actual scope of the finding, and it is the same file #73 fixed for S and Mb.

What this leaves is a narrower, cheaper question than the one I opened: the draft's own B.
Not priced here, because doing it properly needs the draft's rung behaviour measured rather
than borrowed from the trunk's, and the card is unavailable.

## What this does NOT explain

I opened #74 predicting this was the mechanism behind the 269-compile B=8 run that
`errors/2026-09-04-the-recompiles-reproduce-and-are-not-a-shape-set.md:165` leaves
unexplained. **It is not, and the cache says so:** the B values present are 1-7, and that
run was B=8 at depth 1, where `spec.py:484`'s chain loop never executes. 7 B values across
25 served S values cannot produce 269 compiles of one kernel. That run stays open.

## Five instrument corrections

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

**5. The geometry filter itself matched a coincidence.** Written as "the last dim of the
leading tensor", `--geom 256` also kept 103 entries of an `[M,256] x [M,1]` reduction, where
256 is an N dim that merely equals the served head_dim — 16 signatures survived the filter
where 6 should. Only a rank-4 `[B,S,H,D]` leading tensor carries head_dim, so the predicate
now requires rank 4. The two attention kernels read 558 either way, so #74's number is
unaffected; a count over any other kernel would not have been. `--selftest` asserts both
directions against `_is_geom` itself, and reverting the rank check fails it.

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

**Use the cost model the repo already measured; do not invent one from a ratio.** The 3.3×
figure was sitting in an entry that also carried the fit it came from
(`tick_ms = 25.6 + 7.53·launched + 2.29·useful`) and a second entry measured the padding
directly at 1.5%. I turned that into `TICK_MS / B × 3.3` and got 39-70 ms per padding row,
16-66 tick break-evens, and a REJECT — from numbers whose own source says the padding is
free at this rung. A borrowed constant without its model is not a measurement.

**And check whether the fix already exists before pricing it.** `engine.py:817` has bucketed
B on the decode graph path the whole time. Two ticks of pricing preceded reading it.

An axis being real and an axis being worth removing are two measurements. B is a shape axis
worth 34 of the served model's 59 compiles per kernel, and at W=2 the rung ladder is coarse
enough that bucketing it costs nothing.

A cache entry is identified by its full signature, not recognized by its shape. Four
readings of the same 634 files gave four different answers — wrong kernel, wrong arity,
wrong dtype, then right — and each intermediate one looked like a result: "B costs 0 of 7
compiles" was a complete, plausible sentence about a kernel that had not run since #44.
Where `params.pkl` carries no names, the only check is dumping the concrete shapes of one
entry per candidate group and reading which declaration they match.
