---
question: Of the sweep's surviving perf findings, which does a B=1 decode tick reach, and what is each worth?
status: measured
source: tileRL, 2026-09-03; engine spy on cpu, arm thresholds and recorded kernel timings read off origin/main
---

# Two of the three surviving tick-shape findings are worth nothing at B=1

A 12-agent sweep proposed 20 optimizations; 13 survived adversarial
verification. Three of those concern the shape of a decode tick, and ckl's
target is B=1. This classifies each by whether a B=1 tick reaches it, with the
arithmetic attached instead of a severity label.

The sweep's readers worked against a session-pinned tree **117 commits behind
`origin/main`**, which is why one of the three is already fixed and another's
magnitude is wrong by more than an order of magnitude.

## 1. Mixed prefill+decode padding — B=1: unreachable. B=8: up to 7.97x

`engine.py:673` sizes a mixed tick from the widest row, so a decode row needing
`seq_q=1` is computed at the prefill's bucketed width through all 64 layers.

Measured with a spy on `_build_plan`, staggered arrivals: **0 mixed ticks at
`max_batch=1`, 3 at `max_batch=8`** on the same schedule — 128 rows computed
where 43 were needed (2.98x), 192 where 44 were needed (4.36x). At serving
shapes: 7 decodes plus a 512-token prefill computes 4096 rows against 519
needed, **7.89x**; at a 2048 chunk, 16384 against 2055, **7.97x**.

`engine.py:528` breaks admission on `len(decodes) + len(prefills) >= max_batch`,
so a B=1 tick holds one row and there is no second row to pad. Details in
[the mixed-tick entry](2026-09-03-mixed-tick-padding-needs-batch-above-one.md).

## 2. `_GRAPH_BUCKETS` moving rows off their kernel — the claim was 17-23 rows; it is one

The sweep said the bucket ladder "silently moves 17-23 rows off the ks8 decode
kernel". Enumerating every row count 1..39 against `_GRAPH_BUCKETS = (1, 2, 4, 8,
16, 24, 32, 48, 64, 96, 128)` and the real dispatch in `Backend.linear_fp4`:

**exactly one row count changes arm, n=3.** `M=3` satisfies `2 <= M <= _MGEMV`
(backend.py:372) and takes the M-row GEMV; `B=4` falls through to `linear_fp4_mma8`
(backend.py:389). Every other n in 1..39 lands on the same arm before and after
bucketing, because `mma8` already pads `x2` to `_MX = 8` rows, so 5→8, 9→16 and
so on are shape-identical.

Cost of the one real case, from the recorded table in
[m-row-gemv](../wins/2026-08-29-m-row-gemv.md): at M=3 the GEMV is **27.06 ms
against mma8's 29.68 ms**, so bucketing 3→4 forfeits **1.10x, 2.62 ms** on that
tick's linears. Not reachable at B=1 (a 1-row tick buckets to 1) and not
reachable at B=8 (8 is a bucket). It needs a tick with exactly 3 rows.

## 3. Per-row sampler restriction — already fixed on `origin/main`

The sweep reported `engine.py` restricting logits one row at a time: "5 aten
launches per row over a `[1,248320]` slice, where one batched top_k over `[B,V]`
is 4 total". That is true of the pinned tree and false of main. `a123676`
("perf(engine): the sampler stops paying for scores nobody asked for") added the
batched path, and `engine.py:814-817` now reads:

```python
if all((p.allowed_ids, p.top_k) == (cut.allowed_ids, cut.top_k) for p in params):
    logits = _restrict(logits, cut)  # one topk and one id upload, not N
```

with the per-row `torch.stack` kept only as the fallback for a batch whose rows
disagree. Nothing to do.

## Rule

Before costing a tick-shape finding, get the tick shape from a spy on the
planner and the arm from the dispatch code, then price it against a recorded
kernel timing. Two of these three collapsed under that: one because `mma8` pads
to 8 rows so most bucket jumps are shape-identical, one because it had already
been fixed 117 commits ago. A finding worth 7.97x at B=8 is worth 1.00x at B=1,
and B=1 is the target — reachability is part of the magnitude, not a caveat
after it.
