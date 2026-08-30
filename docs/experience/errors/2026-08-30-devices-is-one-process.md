# `--devices` replicates inside one process, and one replica's CUDA fault kills them all — 2026-08-30

## Context

Serving Qwen3.8-27B (NVFP4) as a scoring backend for a sibling project, two cards.
Started it as `serve --devices 0,1` under `CUDA_VISIBLE_DEVICES=4,5`, smoke-tested one
request, reported it healthy, and handed the endpoint over.

The first real caller got `CUDA error: operation failed due to a previous error during
capture` (`cudaErrorStreamCaptureInvalidated`) on every request, and `engine.py:717`
`_make_kv` threw `AcceleratorError` in a loop. `/v1/models` still answered 200 the
whole time.

## Root Cause

`--devices` builds a `DataParallelEngine` (`cli.py:56-81`): N replicas **inside one
process**, not N independent servers. The help text said "replicate across these CUDA
indices", which reads like the latter.

A CUDA error is sticky per process. When one replica's graph capture was invalidated,
every later CUDA call in that process inherited the error — GPU4 dropped 26.7 -> 11.2 GB
while GPU5 held 26.8, one PID across both, and neither could serve. uvicorn was
unaffected, so the health endpoint kept reporting success.

Two failures stacked:
- The deployment shape coupled two cards that had no reason to be coupled.
- The readiness check tested the HTTP layer, not the engine. It passed for reasons
  unrelated to what it was supposed to establish.

What invalidated the capture is **not established**. Concurrent capture between the two
replicas fits the evidence (the single-request smoke test only exercised one replica),
but it was not proven — and the single-card split removes the failure mode regardless.

## Fix

One process per card, `--devices` deliberately not passed, each pinned by
`CUDA_VISIBLE_DEVICES` so it takes the plain `build_engine` path. Readiness is a real
1-token completion, never `/v1/models`. The `--devices` help text now says the replicas
share a process and what that costs.

## Rule

**`--devices` is for throughput on one workload, never for independent endpoints.** Two
consumers that must not take each other down get two processes.

**A readiness probe must exercise the thing it claims is ready.** An HTTP liveness check
in front of a GPU engine establishes nothing about the engine.

**A smoke test that touches one replica has not tested a replicated deployment.** Size
the check to the shape of the thing, not to the shape of the happy path.
