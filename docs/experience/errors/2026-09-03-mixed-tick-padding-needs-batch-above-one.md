---
question: Which of the sweep's surviving findings can a B=1 decode tick actually reach?
status: measured
source: tileRL, tiny model on cpu, measured 2026-09-03 with an engine spy on _build_plan
---

# The mixed prefill+decode padding is real and a B=1 tick never reaches it

A 12-agent sweep proposed 20 optimizations and 13 survived adversarial
verification. ckl asked for them split by whether a B=1 tick touches them, with
tick arithmetic rather than a category label. This is that measurement for the
one whose reachability was in doubt.

## The finding

`engine.py:673` sizes a mixed tick's rectangle from the widest row:

```python
width = -(-max(seq_q) // _PREFILL_BUCKET) * _PREFILL_BUCKET if chunk > 1 else max(seq_q)
```

A decode row needs `seq_q = 1`. A prefill row in the same tick needs its chunk
rounded up to `_PREFILL_BUCKET = 64`. Every row in the batch is then computed at
that width, through all 64 layers' linears.

## Measured, not derived

An engine spy on `_build_plan` counting ticks where `decodes` and `prefills` are
both non-empty, tiny model, requests arriving mid-decode:

| max_batch | mixed ticks | shape | rows computed | rows needed | waste |
|---|---:|---|---:|---:|---:|
| **1** | **0** | — | — | — | — |
| 8 | 3 | 1 decode + 1 prefill(42) | 128 | 43 | **2.98x** |
| 8 | | 2 decode + 1 prefill(42) | 192 | 44 | **4.36x** |

Arithmetic at serving shapes, from the same formula: 7 decodes + one 512-token
prefill computes 4096 rows where 519 are needed, **7.89x**; at a 2048 chunk it is
16384 against 2055, **7.97x**. The ceiling is `max_batch`, approached as the
prefill fills its bucket.

**Why B=1 cannot reach it**, read off the planner rather than inferred:
`engine.py:528` breaks admission on `len(decodes) + len(prefills) >=
limits.max_batch`, so a tick at `max_batch=1` holds exactly one row and there is
no second row to pad. Confirmed by the spy: 0 mixed ticks in 30 ticks with three
requests arriving mid-decode, against 3 at `max_batch=8` on the same schedule.

A first attempt at this measurement saw 0 mixed ticks at **both** batch sizes,
because all four requests were submitted together and so prefilled together.
Staggered arrival is what produces the mix; a probe that submits everything up
front cannot observe this finding at all.

## Rule

Classify an optimization by whether the shipped tick shape reaches it, and get
that from a spy on the planner, not from reading the width formula. `B=1` and
`B=8` are different code paths through `_build_plan`, and a finding worth 7.97x
at B=8 is worth exactly nothing at B=1 — which is ckl's stated target. When
probing a batching effect, stagger the arrivals: simultaneous submission
collapses the schedule the effect lives in.
