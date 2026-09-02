# A synchronize() inside a captured graph measures the probe — 2026-09-02

## Context

The question was what fraction of a speculative tick is drafting, because that
number is the entire ceiling on block-parallel drafting (one draft forward
instead of D). `scripts/prof_spec_tick.py` answers it directly: wrap
`_run_forward`, `_draft_step` and `_verify`, `cuda.synchronize()` around each,
bucket the wall time.

It reported:

```
generated 59 tokens in 158702 ms = 0.4 tok/s (ctx=512, decode ticks only)
draft_step  21  104430.2  4972.87  65.8%
verify      21      24.7     1.18   0.0%
```

The same configuration measures **48.4 tok/s** end-to-end. The probe was 120×
slow, and it put 4972 ms/tick into a draft step that is a few tens of ms.

## Root cause

The decode path is CUDA-graph captured. `torch.cuda.synchronize()` inside the
captured region breaks replay, so every tick fell back to eager launch and
re-capture — the harness was timing graph construction, once per tick, and
attributing it to whatever bucket it had just wrapped.

The tell was in the output, not in the code: 0.4 tok/s against a known 48.4 is a
120× contradiction, and `run_forward` and `decode_graph` both read 100.0% of wall
— two nested buckets each accounting for the whole clock is a harness artifact,
not a measurement. The docstring even records the previous version of this same
failure ("prof_draft_step.py timed the draft with W=1 and got 4.98 ms/step, which
predicted 41.6 tok/s; serving measured 2.7"), and the fix at the time was to
instrument the engine — which introduced this one.

## Fix

Measure the slope, not the parts. Depth D costs D draft forwards plus one verify,
so `ms_tick(D) = verify + D * draft` is affine: sweep depth 1..4 with NO
instrumentation, regress, and the slope is one draft forward while the intercept
is everything else. `scripts/ab_draft_depth.py`. It cannot perturb what it
measures because it only reads the engine's own counters and a wall clock outside
the tick.

A fresh engine per depth, and `del` + `empty_cache` after each — `shutdown()`
only joins the daemon thread, so four 1024-block KV pools plus their captured
graphs would otherwise coexist.

## Rule

Anything captured in a CUDA graph can only be measured from outside the graph or
by `torch.profiler` (which reads the replayed kernels). A `synchronize()` in the
middle does not slow the thing down a little — it removes the graph, so the
number describes a different program.

When a subsystem cannot be probed without disturbing it, vary a parameter it is
linear in and take the slope. One free variable often beats any amount of
instrumentation, and it degrades gracefully: a bad fit is visible, whereas a
plausible-looking bucket total is not.
