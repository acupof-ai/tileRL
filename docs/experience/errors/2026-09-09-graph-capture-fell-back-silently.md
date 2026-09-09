# Graph capture fell back to eager silently — 6x wall, one warning, nothing on /health

**Date:** 2026-09-09
**Arch:** H20 (sm90) card 6, 27B NVFP4, B=1, 50 GSM8K questions

## Context

A prefill profiler wrapped `Model.forward` with `torch.cuda.synchronize()` on every
call. During decode-graph capture a synchronize is illegal
(`cudaErrorStreamCaptureInvalidated`); the engine's `_graph_for` caught the exception,
warned once, flipped `_decode_graph_on = False`, and ran eager decode for the rest of
the process. The same workload then took **1269.4s wall against 211.8s — 6.0x** —
while producing correct output. The warning fired once into a log nobody was watching,
and `/health` reported nothing: `stats()` did not include the runtime flag, so a
degraded engine answered identically to a healthy one.

## Root Cause

The fallback itself is correct — a capture failure must not kill serving. The defect
is visibility: a 6x degradation announced itself once as a `UserWarning` and was
otherwise indistinguishable from a healthy engine. `config()` carried the runtime
state, but `/health` reads `stats()`, which did not. Nobody queries a system that is
still working, so the warning had no audience.

## Fix

`stats()` now reports `decode_graph` (the runtime state, not the build setting), so
`/health` shows `decode_graph: false` after a fallback. The warning already existed
and stays — the fallback stays, it must not raise. Test:
`test_stats_reports_the_eager_fallback_after_a_capture_failure` (CPU, where capture
always fails, asserts the flag flips into `stats()`).

The test's first version asserted the fallback with `max_new_tokens=1` and passed
while exercising nothing: a one-token request completes inside the prefill step
(the prefill's last-position logit samples the token directly), so no decode tick
ever ran and the capture path was never reached. The test was green and the
fallback warning was unverified. `max_new_tokens=2` forces a decode tick and the
assertion then fails when the wiring is broken. A test that does not execute the
path it guards is the vacuous-green shape; confirm it can go red before trusting it.

## Rule

A fallback that keeps correctness and throws away 6x of performance must be louder
than an error, because nobody goes looking inside a system that still works.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-09 | f80e894 | H20 card 6 | cuda/sm90 eager fallback | Qwen3.8-27B-NVFP4 | — | — | 50q in 1269.4s wall (vs 211.8s graph, 6.0x) |

Raw artifacts: `/work/accpf.log` (the broken run) and `/work/accpf3.log` (the clean
run) on the pod.
