# A decode tick with no pad row captured a graph inside the request — cpu, 2026-09-06

> Status: **Fixed.** `precapture` exists so serving never pays a capture; one
> fallback in `_run_decode_graph` could still key on a graph precapture never
> built, and capture it on the request. Reachability is narrow and it was never
> observed in a run — shipped as a correctness hole, not a perf win. No card: the
> change removes a capture, so the cost it avoids is `precapture`'s measured
> per-graph figure, not a new measurement.

## Context

`precapture` builds `graph_keys()`, which enumerates **buckets only**:
`{_graph_bucket(rows) for rows in 1..max_batch}` — at `max_batch=4`, `{1, 2, 4}`.
The gate `test_graph_keys_covers_what_a_decode_tick_keys_on` already asserts that
every bucket a tick can key on is in that set.

Cost of a capture on the request path, measured on the V100 in
`wins/2026-09-02-precapture-the-decode-graphs.md`: **8 graphs in 19 s** warm,
**208 s cold**, and the two captures a generate-and-hope warmup missed cost
**14.0 s and 11.7 s** on the requests that hit them. That entry is why
`precapture` exists.

## Root Cause

`_run_decode_graph` computed its key by a second route the gate does not cover.
An under-full tick (`n < B`) needs a pad row to park the unused graph rows on;
that row is reserved in `__init__`, and if the pools were sized without the spare
the tick retried the allocation and, on failure, fell back to the exact size:

```python
except RuntimeError:
    B = n  # no spare capacity to park padding rows on: exact size
```

`n=3` at `max_batch=4` gives `B=3`, and `3 ∉ {1, 2, 4}`. `_graph_for` captures a
missing key on first use, so the tick captured **inside a live request** — under
exactly the pool pressure that had removed the pad row in the first place.

The existing gate could not see it: it checks `_graph_bucket(rows)`, and this
path deliberately bypasses `_graph_bucket`. **Two routes to the same key, one of
them tested.**

Reachability is narrow. The pad row is normally reserved in `__init__`
(`engine.py:369`), so this needs pools sized without the spare —
`num_slots == max_batch` — **and** the runtime retry to fail too. Not the default
configuration, and never seen in a run.

## Fix

Return False and let the caller run eager:

```python
except RuntimeError:
    # no pad row: an exact-size graph is off the graph_keys grid and would
    # capture mid-request
    return False
```

One eager tick (the `~10x` of `engine.py`'s mixed-tick comment) against a full
capture (seconds). Not permanent: `_pad_slot` stays None, so the next tick
retries the allocation and picks the graph back up as soon as a request frees a
slot. `_decode_graph_on` is untouched — only a real capture *failure* turns
graphs off.

**The gate spies on `_graph_for` rather than capturing**, so it runs on cpu: what
was wrong is which key the tick asks for, not what the capture does with it. Two
arms in one test, because the branch only means something against its control —
the same 3-row tick *with* the pad row must still key on `(4, 1)`.

Negative control, with the one line reverted:

```
>       assert asked == [], f"an unpadded tick asked for an off-grid graph: {asked}"
E       AssertionError: an unpadded tick asked for an off-grid graph: [(3, 1)]
```

It names the exact off-grid key, not just a boolean.

## Rule

**A lazily-captured graph may only be keyed on something the warmup enumerated.**
When a fallback computes the key by a different route than the one the warmup
walks, it has to run eager instead of capturing — otherwise precapture's report
is a claim about a different set of graphs than the ones the ticks ask for. A
gate written against the primary route does not cover the fallback; the fallback
needs its own arm.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-06 | (this) | Mac M-series | cpu | tiny | — | — | — (no timing: a removed capture) |

Gate: `tests/test_decode_graph.py::test_a_tick_with_no_pad_row_runs_eager_instead_of_capturing_mid_request`.
Suite at this commit: **408 passed, 14 skipped** (`TILERL_TARGET=cpu uv run pytest`).
