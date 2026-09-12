# A failed sm70 decode-graph capture poisons torch's caching allocator for the process — 2026-09-12

> Status: **fixed on `fix/sm70-decode-graph-allocator`** by prevention (auto
> capture off on sm70). The CPU gate covers the guard; the end-to-end V100
> point (dense→sparse two-arm harness, empty_cache between arms) is
> pending-remote, cc runs it after the write_tokens PR.

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

## Rule

A failure inside a non-reentrant global resource (a CUDA stream capture, a
process-wide allocator mode) is not local to its try/except: find what state
the exit path leaves behind on the exception branch before relying on eager
fallback. When a C++ resource exposes no Python reset, the only fix is to
not enter the failing state.
