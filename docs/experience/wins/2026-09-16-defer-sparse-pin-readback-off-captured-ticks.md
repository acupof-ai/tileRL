# deferring sparse pin-set host readback out of captured decode ticks — H20 sm90, 2026-09-16

> Status: Shipped (device-measured; CPU parity green)

## Context

CUDA graph capture gave sm90 sparse decode no speedup over eager (32k 21 vs
20 tok/s). The sibling errors entry
([2026-09-16-sm90-sparse-graph-no-speedup-finalize-d2h](../errors/2026-09-16-sm90-sparse-graph-no-speedup-finalize-d2h.md))
localized it with nsys: post-replay residency `finalize` round-trips the
device-chosen pages to a host Python set every tick
(`selected_pages().tolist()`), so each captured forward paid ~109 D2H copies
and ~78 `cudaStreamSynchronize`s and the GPU sat ~10% busy. The wall-clock
metric is steady single-row decode ms/forward at 32k context.

## What Worked

Skip the host pin-set reconciliation on a captured CUDA decode tick. The
decode forward already runs entirely on device page tensors, and resident
frames self-prune through `resolve()` → `evict_victim()` when a new own page
needs a frame. The every-`SPARSE_REFRESH_TICKS` (1-in-8) eager tick keeps the
host path unchanged — it builds `_chosen` on the host, reconciles the pin set,
demotes, and publishes drops, so the deferred bookkeeping lands once per eight
ticks instead of blocking every replayed forward. Eager, refresh and prefill
paths are untouched (they use the host `selected_pages()` branch); CPU parity
is unchanged. One 4-line guard in `sparse_runtime.py finalize`, gated to
`device_select and device.type == "cuda"`; no default flips.

Same process/driver/card, a pure-captured window (64 warm forwards, then 7
steady forwards with no eager refresh inside; `scripts/perf2_nsys_ticks.py`
under a `cudaProfilerApi` nsys range), base tree vs fixed tree:

| 32k sparse graph, per forward | base | fixed | delta |
|---|---:|---:|---:|
| wall ms/forward | 33.43 | 27.55 | **−17.6%** |
| D2H copies | 108.9 | 6.3 | **−94%** |
| cudaStreamSynchronize | 77.9 | 26.1 | **−67%** |
| cudaGraphLaunch | 7 | 7 | same (forward still captured) |

nsys reports (H20 container `/work`, 2026-09-16):
`nsys_pin32k_pure7.nsys-rep` (fixed), `nsys_base32k_pure7.nsys-rep` (base).
CPU: `pytest tests/test_sparse_engine.py tests/test_sparse_runtime_facade.py`
67 passed.

## Next lever, not in this change

Residual per-forward cost after the guard: ~6 D2H + ~26 `cudaStreamSynchronize`
and ~2.4 ms of GPU kernels inside a ~27.5 ms tick — the stall is now the
out-of-graph **verify / sample / draft_step** segment (`toks.tolist()` and the
spec verify chain construction at `spec.py:556-558,591-593`), not pin
readback. Closing it touches `spec.py` and collides with the draft W-window
work (`spec.py:373`, queued after #672), so it is deliberately deferred; dense
graph at this context runs ~25 ms near GPU-bound and is the direction.

## Rule

A `.tolist()` page-id readback kept around for host bookkeeping does not have
to run on the tick that computes it: when a periodic host reconciliation
already exists (the eager refresh), defer the readback to it and let device
self-pruning hold residency in between. Measure D2H/sync counts per forward,
not just wall ms — the win here is a −94% copy count that the 18% wall delta
understates because a separate stall remains.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/forward | throughput |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-16 | stacked on 646e223f (#674) | H20 sm90 | cuda graph | qwen38-27b NVFP4 (32k) | — | 33.43 → 27.55 | ~21 → ~25 tok/s |

Raw artifacts: `nsys_pin32k_pure7.nsys-rep`, `nsys_base32k_pure7.nsys-rep`,
driver `scripts/perf2_nsys_ticks.py`, logs `nsys_pin_pure7.log` /
`nsys_base_pure7.log` under `/work` in the sglang-test container.
