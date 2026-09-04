# The context-capped verify width is REJECTED, and its premise had the sign backwards

**Date:** 2026-09-04
**Arch:** sm70 (Tesla V100-SXM2-32GB), 27B NVFP4 + draft, B=1, wikitext x3, `--tokens 128`
**Task:** #60

## The rule, registered before the data

#60 asked whether capping verify width by context is worth it, and named the
instrument trap itself: the profiler's 82.0 ms/fwd carries overhead the 61.6 ms
bench tick does not, so the ratio must come from one process, not from dividing
those two. Before the ctx=2048 rows landed:

> A cap has a lever ONLY IF at ctx=2048 either **(A)** frac4 at depth 3 drops
> materially below 98.8%, or **(B)** the rung-4 tick dearens faster than rung 2,
> widening the 16.11 ms step. REJECT if depth 3 stays ~99% rung 4 and both rungs
> scale together — then a cap cannot move ticks out of rung 4 without lowering the
> WIDTH, which is depth selection (#61), not a context cap.

## The premise was wrong twice

**First: a mean is not a rung.** The task reads "the mean chain widens 2.28 → 3.70,
crossing rung 2 → 4". At B=1 the ladder gives `rung(W) = 1/2/4/4/8` for W=1..5, so
`ceil(2.28) = 3 → rung 4` and `ceil(3.70) = 4 → rung 4` — the *same* rung. A mean
of a quantity that lives on a staircase is not a point on the staircase.

I then over-corrected: I told the peer both figures were *mixtures* of rung 2 and 4,
quoting `ab_draft_depth.py:164`'s recorded 2.56 = 72/28 and 2.88 = 56/44. Measured
here, each depth is nearly **pure** — 100% / 98.3% / 98.8% / 89.8% rung. Those
recorded mixes were a different run, and I generalised one to this config without
checking. "A mean is not a rung" was right; "therefore it is a mixture" was invented.

**Second, and this is the finding: the cost is 6.3x smaller than stated and the
sign of the verdict flips.** #60 says rung 4 at ctx 2048 costs **+10.0 ms** for
**+0.16 tok/fwd**. Measured on the clean depth-2 arm, the *whole tick* grows
**+1.59 ms** (56.18 → 57.77) for that same **+0.16 tok/fwd** (2.16 → 2.32):

| | ctx 1024 | ctx 2048 |
|---|---:|---:|
| ms/tick | 56.18 | 57.77 |
| tok/fwd | 2.16 | 2.32 |
| **tok/s** | **38.4** | **40.2** |

**1.0445x — the wider chain at longer context is a win, not a 10 ms tax.** The
10.0 ms came from dividing the profiler's 82.0 by the bench's 61.6, the exact
division the task warned against.

## Verdict: REJECTED

| pre-registered rule | measured | |
|---|---|---|
| (A) frac4 below 98.8%? | 98.3% → **97.0%**, moved 1.3 points | NO |
| (B) rung 4 dearens faster? | rung 4 **1.034x**, rung 2 **1.029x** | NO |

The mix does not move and the 16.11 ms step does not widen, so a cap cannot take a
tick out of rung 4 without lowering the width — and that is depth selection, a
different lever with a different owner. Nothing to ship.

## Two of four rows were compile-contaminated; the cause is still open

`write_tokens_f32` and `paged_attention_split` each recompiled **51 times** in one
run, 3-5 s apiece. That is the measured fact. The mechanism is not.

**A first attribution was wrong and is recorded here rather than deleted**, because
it was one `grep` from being checked. I claimed `NB = T.const("NB")`
(`kernels_mma.py:59`) bakes the block-table width into a compile-time constant, so
every KV-pool growth recompiles. Two errors: `NB` is the pool's block **count**
(`Mb` is the table width), and **the pool never grows** — `PagedKvPool` sets
`num_blocks` once at construction (`kv_cache.py:64`) and `alloc_block` *raises* on
exhaustion (`:81`). There is no grow path in the tree, so that mechanism does not
exist. The check was `grep -n num_blocks kv_cache.py`.

What genuinely varies per call, of `B, S, H, D, NB, Mb`, is **`S` — the query width
of the tick**: 1 for a draft step, W for a verify of width W. `H`/`D` are model
constants and `B` is 1 throughout. But `S ∈ {1,2,3,4,5}` across this whole run, so
five shapes cannot account for 51 compiles: something is missing the **disk** cache
across repeats, and that is not established.

The count itself was also first reported as 55/54 — that was 102 log lines read as
distinct events when they are 51 matched begin/complete pairs.

**The contamination is still identifiable without the cause**, by two independent
routes rather than by picking rows:

- **Monotonicity.** depth 3 reads 114.17 ms on 100% rung 4; depth 4 reads 95.53 ms
  on 88% rung 8. A dearer rung plus an extra draft forward cannot be 1.20x
  *faster*, so the ordering is impossible and depth 3 carries compile time.
- **The two-term model** `verify(rung) + (W-1)·draft`: **5.46x / 1.01x / 1.82x /
  0.92x**. It flags the same pair, and depth 4 sitting *below* the model at 0.92x
  is consistent with carrying no compile at all.

The harness's own decomposition named the location: `verify: r2:30.58 ms` against an
independently measured 29.38 is clean, while `draft: 168.24 ms/forward` against 5.75
is **29x** — the compile lands inside the draft's timing window, so the subtraction
charges all of it to the draft.

`p0: 36.07 ms / 52.5 tok/s` is the tightest control: byte-for-byte the ctx=1024
figure, measured *in* the ctx=2048 arm, before p1 and p2 collapsed to 6.4.

Why this did not touch the sm90 throughput cells, and does not reach production:
the graph path **forces the JIT to finish before capture** — `engine.py:223`,
"Warmup on a side stream: tilelang JIT (host work) must finish before capture",
running two warmup forwards per `(B, W)` graph. So a served tick replays a graph
whose compiles were paid at capture. `B` is bucketed too (`engine.py:823` pads rows
to `_GRAPH_BUCKETS`), so at the shipped `--max-batch 8` the reachable buckets are
`{1,2,4,8}` and the whole `(B, S)` space is **20 pairs**, each compiled once.

The trap is therefore **bench-only, and specifically eager-path**: a harness that
sweeps a per-tick shape *inside a timed window* with no capture to absorb the
compile. `ab_draft_depth.py` warms one `(B, width)` per depth before timing, which
is why depths 2 and 4 came out clean — the contaminated depths met a shape that
warmup did not cover.



## Repeatability, and one instrument confirming another

Two independent processes at ctx=1024, fresh weight load each:

| depth | run1 | run2 | spread | tok/fwd | histogram |
|---|---:|---:|---:|---|---|
| 1 | 36.12 | 36.06 | 0.17% | 1.75 / 1.75 | identical |
| 2 | 56.55 | 56.18 | 0.66% | 2.16 / 2.16 | identical |
| 3 | 61.40 | 60.95 | 0.74% | 2.38 / 2.38 | identical |
| 4 | 99.46 | 99.00 | 0.46% | 2.45 / 2.45 | identical |

tok/fwd identical to 3 s.f. at every depth and the rung histograms **byte-identical
bucket for bucket**, so any fraction difference is signal. This harness's own
spread is **0.17–0.74%**, which agrees with `bench_ctx_decode`'s 0.3–0.9% — a
second instrument confirming the first, measured here rather than borrowed across
harnesses.

## Open, not decided

At ctx=2048 depth 4 reads **30.5 tok/s** against **24.7** at ctx=1024, and
acceptance rises monotonically with context (1.78 / 2.32 / 2.73 / 2.91), in the
same direction as #67's p=0.786 at 8192. **Longer context makes deeper chains
pay**, which is the opposite of the short-context depth default #61 settled.
Not flipped: two of these rows carry compile time, and a default flip needs one
clean rerun with the compile pre-warmed. The recompile's *cause* is open (see above),
so "pre-warm" means visiting every `S` the sweep will use before the timed window,
not a fix.

## Rule

Register the decision rule before the data lands. Both conditions here were
written down while the run was in flight, so "both fail → reject" is a reading
rather than a fit — and the same discipline is what let the contaminated rows be
identified by a criterion (monotonicity, model ratio) instead of by which rows
disagreed with the answer I wanted.

A mean of a staircase-valued quantity is not a point on the staircase. Neither
`ceil` of it, nor — as I found out one message later — a mixture inferred from
someone else's recorded mix.

And an attribution is not a measurement. The `NB` mechanism above survived long
enough to be written into an entry and offered to a peer for promotion, on nothing
but its own plausibility; one `grep -n num_blocks kv_cache.py` ended it. When a
number is measured and its cause is inferred, say which is which — the number here
(51 compiles, 3-5 s each) stands, and every mechanism proposed for it so far has
been wrong.

