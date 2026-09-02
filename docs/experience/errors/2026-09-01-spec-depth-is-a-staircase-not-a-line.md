# Speculation depth is a staircase, not a line — the M ladder sets it — 2026-09-01

## Context

Fixed `--depth 3` speculation measured a net LOSS against dense on two of four
realistic workloads. I modelled the tick as `bias + row*(1+d)`, found a missing
`_draft_step` term (6.0 ms/step), added a survival cut so depth would shrink on
high-entropy work, and raised the cap to 5 to let the cut choose.

Every workload got SLOWER — including counting, which the cut can never fire on:

| workload | dense | fixed d3 | "adaptive" d<=5 |
|---|---:|---:|---:|
| counting | 32.7 | 52.8 | 47.3 |
| coding | 32.6 | 43.9 | 37.4 |
| dialogue | 31.9 | 30.1 | 27.5 |

## Root Cause

`tok/fwd` went UP everywhere (counting 4.00 -> 5.33) while tok/s went down. More
tokens per forward and less throughput means depth INCREASED — the cut almost
never fired, because the draft's own softmax is near 1.0 exactly when it is
confident, and the cap change from 3 to 5 was the whole effect.

The reason deeper lost is a staircase I built myself this morning. The verify
width is `W = 1 + depth`, and the sm70 GEMV ladder serves only 1/2/4/8 rows,
rounding up. Sweeping depth exposes it:

| depth | W | rung | counting | coding |
|---:|---:|---:|---:|---:|
| 1 | 2 | 2 | 41.1 | 38.2 |
| 2 | 3 | **4** | 42.8 | 38.2 |
| **3** | **4** | 4 | **53.0** | **43.8** |
| 4 | 5 | **8** | 41.0 | **31.5** |
| 5 | 6 | **8** | 47.3 | 33.3 |
| 7 | 8 | 8 | 54.0 | 30.2 |

Depth 4 buys an 8-row launch to verify 5 rows, and on coding lands at 31.5 —
**below the 32.6 it gets with no speculation at all**. The cliff is between
depth 3 and 4, exactly where W crosses from rung 4 to rung 8.

Fitting a line through a staircase is what produced the nonsense intermediate
result: least squares over d3/d5 gave "18.5 ms per depth step" and an implied
fixed cost of 1.9 ms — i.e. a tick with no fixed component, which contradicts
the premise speculation rests on. That number was the staircase being dragged
into a slope.

Same-rung depths are NOT equal (d2 38.2 vs d3 43.8, both rung 4), so the cost
is two superposed terms: the draft forward grows linearly in depth, the verify
steps on the ladder. Depth 3 wins because it is the DEEPEST depth that still
fills rung 4 — every verify row paid for is a row used.

## Fix

Defaults moved onto the ladder; no new policy code.

- `serve/train --depth` 2 -> 3 (W=3 was rounding up to rung 4, wasting a row)
- `build_engine(spec_depth=)` 4 -> 3 (W=5 sat on the cliff)
- `spec.LADDER_WIDTHS = (1, 2, 4, 8)` and a construction-time warning on sm70 naming the
  two nearest good depths. Warn, not clamp: the ladder is one arch's shape.

The survival cut was reverted. Verified at the new default:

| workload | dense | spec d3 | |
|---|---:|---:|---:|
| counting (control) | 32.7 | 52.7 | 1.61x |
| coding | 32.6 | 43.4 | 1.33x |
| dialogue | 31.9 | 32.0 | 1.00x |
| thinking | 32.0 | 30.2 | 0.94x |

## Rule

When a cost model's residual is stable, suspect a term; when its FIT is absurd
(a fixed cost near zero), suspect the functional form. A quantized kernel
dispatch makes cost a step function of width, so any policy that treats depth
as continuous will pick points that pay for capacity they cannot use — and the
penalty for overshooting a rung is larger than the entire benefit of
speculating.

Second: an experiment that changes two variables answers neither. Raising the
cap and adding the cut in one run made a staircase look like a failed policy;
the depth sweep that isolated one variable settled it in a single job.
