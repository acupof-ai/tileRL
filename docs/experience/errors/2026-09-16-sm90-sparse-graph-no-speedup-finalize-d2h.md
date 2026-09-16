# sm90 sparse graph capture gives no speedup — out-of-graph finalize reads selected pages back to host every tick — 2026-09-16

**Status:** fixed (stacked on the diagnostic PR #674) — captured CUDA decode
ticks no longer reconcile the host pin set; they skip the
`selected_pages().tolist()` readback in `finalize`, residency self-prunes via
`evict_victim`, and the 1-in-8 eager refresh tick reconciles and publishes.
32k: −17.6% ms/forward, D2H −94%, stream sync −67%. See the win
[2026-09-16-defer-sparse-pin-readback-off-captured-ticks.md](../wins/2026-09-16-defer-sparse-pin-readback-off-captured-ticks.md).
The residual out-of-graph verify/sample/draft stall stays open as a separate
lever. Full device-resident page tables were not needed for this gain.
**Arch:** H20 sm90, 27B NVFP4, sparse-k 128 / sparse-min-tokens 8192, draft depth 1.
**Discovered:** perf investigation of the V100 sparse-decode slowdown (6-9 tok/s,
GPU util 21-37%) carried to sm90 to test whether CUDA graph capture removes the
eager per-tick launch gaps. It does not, and this entry records why.

## Claim corrected

Earlier framing said sparse decode was launch-gap bound (600 aten ops / 72 H↔D
copies per eager tick on sm70) and expected sm90 graph capture to eat those
gaps. That is wrong. On sm90 the decode forward **is** captured correctly; the
wall time sits in an out-of-graph host readback that both the captured and the
eager path run. Graph-on vs eager is ~1.0x for that reason.

## Observed

Served single-request (B=1, slots/max-batch 4, max-ctx 131072), model
qwen38-27b at `/work/tilerl-ckpt/Qwen3.8-27B-NVFP4`, steady-state decode:

| ctx (tokens) | graph-on tok/s | eager tok/s |
|---|---|---|
| ~8.7k sparse | 24.1-24.4 | 24.1-25.2 |
| 32k sparse  | 21.2-22.3 | 19.5-20.2 |
| dense warm (control) | 77 | — |

Graph-on vs eager ≈ 1.0x (8.7k identical, 32k ~1.05x — noise). No large jump
from capture. Both arms pinned the SM clock at 1980 MHz; sampled GPU util was
low in steady decode (graph mean 11%, eager 30%) with power only 100-285 W of
the 700 W cap — the card is not compute bound.

`TILERL_STEP_TIMING=1 TILERL_STEP_TIMING_SLOW_MS=0` on the 32k graph arm
splits one ~80 ms wall tick as: `model=70-79 ms` (the enveloppe around
`g.run()` graph replay, 88%), and everything outside only ~10 ms
(`sparse_select 1`, `sparse_finalize 2-3`, `sample 1-3`, `draft_step 5` with
the draft CUDA event itself only 4.9 ms, `offers_pub 0`). No single large
host sync is visible in the wall brackets — the stall is *inside* the model
segment, so it needs the GPU timeline, not the wall timers.

## nsys root cause

In-process driver `scripts/perf2_nsys_ticks.py` (builds one 27B engine,
prefills to 32k, burns 64 warm decode ticks, then runs 12 steady decode
forwards inside a `cudaProfilerApi` capture range so the long prefill is
excluded). 32k, 12 ticks, three arms:

`nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi`
reports on the H20 container (`/work/nsys_*.nsys-rep`):

**sparse32k_graph** (514 ms / 12 forwards = **42.9 ms/forward**):

- GPU kernel time **5.33 ms/forward → GPU busy only 12.4%**; ~37.6 ms is the
  stream idle, waiting on the host.
- `cudaGraphLaunch` = 11 over 12 forwards (one forward per 8 is a refresh tick
  that runs eager — `SPARSE_REFRESH_TICKS = 8`). The forward is genuinely a
  replayed graph, not an eager shell: graph coverage is not the defect.
- **D2H memcpy 1512 = 126/forward, 745 MB** (`cuda_gpu_mem_time_sum`).
- **`cudaStreamSynchronize` 1420 = 118/forward, 240 ms total = 79.8% of host
  API wall time** (max 18.8 ms one sync). `cudaLaunchKernel` 4530 and
  `cudaMemcpyAsync` 3773 are the surrounding out-of-graph work.
- biggest captured kernels: `paged_attention_kernel` 1.58 ms × 12 (29.7%),
  `linear_fp8_gemv` 157 calls, `linear_fp4_gemv` 160 calls — small GEMVs, but
  their total is only the 5.33 ms above.

**dense32k_graph** (305 ms / 11 forwards = **25.4 ms/forward**, control):

- GPU kernel time 2.93 ms/forward.
- **D2H only 25 = 2.3/forward** (vs sparse 126): the dense path has no
  per-tick host page readback and almost never stalls.

So sparse GPU work is only 2.4 ms more than dense (the packed/sparse
paged-attention gather), but the sparse tick is ~18 ms longer — the delta is
the D2H/stream-sync wait, not compute, not launch count, not graph coverage.

## Mechanism and code sites

The device selection path is already sync-free: `SparseForward._select_device`
and `_attention_args_device` score, top-k-union and build the packed table as
pure device ops, so the captured forward issues no host readback. The readback
happens **after** replay, in residency finalize, which still maintains a host
Python set of pinned pages:

- `src/tilerl/sparse_runtime.py:552` `kept = sf.selected_pages(bi)` — one call
  per row per tick.
- `src/tilerl/sparse_engine.py:754-756` `selected_pages()` loops every source
  group and does `ch[bi, : int(ns[bi])].tolist()` on the device-chosen page
  tensor for each (row × group); each `.tolist()` forces a D2H +
  `cudaStreamSynchronize`.
- `src/tilerl/sparse_engine.py:816` `selected()` has the same `.tolist()`.
- the host set then drives `demote_page` / `map_evict` in finalize
  (`sparse_runtime.py:553-567`) and `process_offers` publishes cold pages with
  `.cpu()` clones (`sparse_runtime.py:437-443`).

Every out-of-graph `.tolist()` blocks until the replayed graph drains, so the
next `cudaGraphLaunch` cannot be enqueued until the host round-trips the chosen
page ids. Eager stalls on launches; the graph stalls on this same readback —
which is why capture cannot help until finalize stops reading back.

## Fix direction (for the owner)

Keep pin/evict residency on the device: maintain the resident set and the
demote mask as device tensors (physical ids already live in `l2p_t` and the
device page-pick tensors), move the host-set bookkeeping to GPU ops, or defer
the logical-id readback to the 1-in-8 refresh tick as one batched async copy.
Expected to remove the ~18 ms/tick readback wait on sm90 (32k 43 → ~25 ms,
toward the ~5 ms GPU-bound floor). sm70 eager finalize has the same host
readback and is likely the same root, so the win should carry to V100.

## Reproduce

```bash
# H20 container sglang-test, shared env /work/tl013 (torch 2.11+cu129; do NOT
# use the per-tree .venv, it is torch cu130 and incompatible with driver 535).
scripts/pod_run.sh --lend-ref '<grant>' <name> 1 -- bash /work/perf2_nsys.sh
# or the served arms directly; nsys reports:
nsys stats --report cuda_api_sum       nsys_sparse32k_graph.nsys-rep
nsys stats --report cuda_gpu_kern_sum  nsys_sparse32k_graph.nsys-rep
nsys stats --report cuda_gpu_mem_time_sum nsys_sparse32k_graph.nsys-rep
```

Driver: `scripts/perf2_nsys_ticks.py`; wrapper and reports were under `/work`
in the `sglang-test` container on 2026-09-16 (`nsys_sparse32k_graph`,
`nsys_sparse32k_eager`, `nsys_dense32k_graph`). Code at main `a9868742`.

## Rule

A captured graph that replays the forward does not make the tick graph-bound.
Profile the GPU timeline before crediting capture: here the wall timer put the
stall inside `model` and stopped, but the kernels only filled 12% of it — the
rest was a post-replay `.tolist()` readback the wall bracket could not see.

CI note: GitHub Actions dropped the pull_request events for two prior commits
on this branch (no ci workflow run was created, only the security check suite),
recorded while retriggering checks for review.
