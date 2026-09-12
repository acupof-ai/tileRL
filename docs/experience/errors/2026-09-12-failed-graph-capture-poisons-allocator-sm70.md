# A failed sm70 decode-graph capture poisons torch's caching allocator for the process — 2026-09-12

> Status: **fixed by prevention (auto capture off on sm70), shipped with #545
> (main 6b714563).** The end-to-end V100 point ran 2026-09-12: the auto arm is
> verified (graph off, warning, both empty_cache calls survive). The explicit
> opt-in arm did NOT reproduce the poison in a short probe — capture succeeded
> there — so the opt-in failure stays an observed, not-on-demand-reproduced,
> property (details in the measured section). The CPU gate still covers the
> guard.

## Context

65's fidelity harness builds a dense engine then a sparse engine in one process
and calls `torch.cuda.empty_cache()` between arms. On the V100 (sm70, torch
2.5.1+cu121, CUDA 12.4) the dense engine's lazy decode-graph capture fails
mid-kernel on its first decode tick; the engine catches at `_graph_for`, warns
"decode graph capture failed … eager fallback", and runs eager fine. The
second arm's `empty_cache` then dies:

```
RuntimeError: captures_underway.empty() INTERNAL ASSERT FAILED
  at "../c10/cuda/CUDACachingAllocator.cpp":2967
```

The failure surfaces far from its cause, and the documented workaround
(`decode_graph=False`) had to be carried in the harness.

## Root cause

torch 2.5.1's `torch/cuda/graphs.py` `graph.__exit__` is:

```python
self.cuda_graph.capture_end()
self.stream_ctx.__exit__(*args)
```

with no try/finally. In C++, `at::cuda::CUDAGraph::capture_end()` calls
`cudaStreamEndCapture` and only then
`CUDACachingAllocator::notifyCaptureEnd`, which pops the capture the
allocator began at `capture_begin`. On a poisoned capture `cudaStreamEndCapture`
raises ("operation failed due to a previous error during capture"), so the
allocator's `captures_underway` set is never popped and the stream context is
never restored. The Python exception is catchable; the allocator state behind
it is not — torch exposes no binding to clear `captures_underway`, so no
in-process recovery exists: every later allocator operation that asserts the
set is empty (notably `empty_cache`) fails for the rest of the process.

The failing capture is a DENSE decode capture incompatibility on sm70
(`sparse_k=0`), separate from the sparse `write_tokens` page_base bug.

## Fix

Prevention at the single choke point `_graph_on`: the auto path (the
`decode_graph=None` default) returns False on `arch == "sm70"` with a
one-time warning naming the consequence and the opt-in. Explicit
`decode_graph=True` is still honoured — the user accepted capture
debugging. All three sizing callers (Engine init, `build_engine` pad row,
the CLI slot fit) share the one answer, so the pool is sized consistently
with no capture.

## Measured V100 point (2026-09-12, main 75d1788e)

Two separate processes, each a 27B dense engine then `empty_cache` (the
auto arm adds a sparse engine and a second `empty_cache`), 2048 prompt tokens
+ 8 greedy on Tesla V100-SXM2-32GB, torch 2.5.1+cu121. Driver
`scripts/probe_sm70_graph_poison.py`.

- **Auto (`decode_graph=None`): verified.** The one-time warning fired once,
  `_decode_graph_on=False`, dense generated, `EMPTY_CACHE_AFTER_DENSE OK`,
  sparse generated the same greedy prefix, `EMPTY_CACHE_AFTER_SPARSE OK`
  (`AUTO_ARM_PASS`). This is the exact dense→`empty_cache`→sparse shape that
  asserted pre-#545; the guard prevents it.
- **Explicit opt-in (`decode_graph=True`): poison NOT reproduced in this
  probe.** `_decode_graph_on=True` stayed on, decode ran, and the post-run
  `empty_cache` survived. Capture evidently did not fail at this small
  standalone context (2048 tokens, fresh process, no prior arm in the same
  process). The original poison came from 65's full fidelity harness — a
  32k flow after the harness's own allocator history — so the failing
  conditions are broader than this probe sets up. The opt-in path remains
  "honour the flag, user accepts capture debugging"; the non-reproduction
  does not exonerate it, it only says the failure is not a short-context
  deterministic one. A repro at 32k inside the fidelity harness is still
  owed before anyone relies on `decode_graph=True` on sm70.

## Rule

A failure inside a non-reentrant global resource (a CUDA stream capture, a
process-wide allocator mode) is not local to its try/except: find what state
the exit path leaves behind on the exception branch before relying on eager
fallback. When a C++ resource exposes no Python reset, the only fix is to
not enter the failing state.

Operational: a "poison on failure" claim needs the failure to actually fire.
The fix here is verified on the prevention arm; the opt-in failure only
reproduces under the production harness, so record it as observed-but-
not-reproduced-on-demand rather than padding a short probe to a confirmation.
