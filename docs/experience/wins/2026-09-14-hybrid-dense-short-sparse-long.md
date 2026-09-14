# Hybrid serve: short prompts dense on the captured graph, long prompts sparse — V100, 2026-09-14

> Status: device-verified on sm70 (ops-0b, n37-002-027). CPU gates in
> `tests/test_sparse_engine.py -k hybrid` (memoize, hot-headroom reserve,
> fairness semantics, finish-at-N). Probe:
> `scripts/probe_hybrid_schedule_trace.py`.

## Context

Sparse k=128 is accuracy-equivalent but slower on short prompts; the dense
captured graph is fast for them but cannot hold a long context. The hybrid
engine (`--sparse-min-tokens N`) fixes the mode per request at submit: prompts
up to N tokens run dense on the precaptured decode graph and pin their whole
context in the device KV pool (no sparse sharing); longer ones run sparse.
Ticks never mix the two BatchKv geometries. Scheduling is **rolling-window
wall-time fairness**: while both modes are runnable, sparse owns a tick only
when dense has accrued more wall time in the window since the last sparse
tick; a tie serves dense, and ticks that run with the other mode absent never
enter the window. A sparse prefill tick is capped to ~1 s
(`--sparse-prefill-tokens`, default 192: V100 sparse prefill measured
191 tok/s, so 192 ≈ 1 s and is a whole 3x64-token bucket above the forced
8-page window). A dense admit reserves every live sparse row's headroom to its
per-slot hot ceiling, so a dense pin cannot make a sparse row raise
"hot pool undersized" inside a live tick. The dense graph is precaptured
before traffic and sparse ticks run **eager** deliberately: eager sparse is
token-exact on sm70, so hybrid does not depend on the sparse-graph capture or
its warmup-frame fix (#585).

The hybrid memory ledger is the dense whole-pool view and is memoized like
the plain dense engine (#581) — without that, an all-dense hybrid tick
re-walked `memory.plan` over every 27B param tensor twice per step.

## ponytail — concurrent short latency during a long prefill is fill-bound

A short dense request served WHILE a long sparse prefill runs drops to
**~9.9 tok/s (TTFT 2.5 s)**, about **20% of its solo 52.6 tok/s** — even with
correct wall-time scheduling, free slots, warm graphs and no admission
blocking. The cause is device-level: a dense decode tick syncs behind the
in-flight sparse prefill kernel on the device. This is the measured number,
not a derived 50% wall share. Upgrade path is chunk scheduling or kernel/queue
work to decouple dense decode from the sparse fill — deliberately not built
here. A smaller prefill cap (`--sparse-prefill-tokens 96`) was measured and
rejected: it cuts the worst TTFT but adds 124% to the 32k fill
(195.8 → 438.0 s), so 192 stays the default.

## Rule

A hybrid of a fast bounded path and a slow unbounded one needs the mode fixed
before admission (a row cannot re-pin mid-request), a capacity route for
prompts the bounded path cannot hold, and admission that reserves the
unbounded path's future pages. Wall-time fairness across ticks is necessary
but not sufficient for tail latency when the two paths share one device queue.

## Results (V100 sm70, 27B NVFP4, MTP d1, --decode-graph, k=128 N=8192)

| check | result |
|---|---|
| all-dense hybrid vs main (same flags) | within **0.9%** (53.2 → ~52.6-equivalent after the ledger memoize) |
| 120k sparse fill with spill | answers (116,912 + 32 tok, TTFT 1270 s); **RSS peak 15.1 GiB**, GPU 32,448/32,768 MiB, spill file 8.24 GiB |
| short dense DURING a 32k fill (real server) | delivered; **TTFT 2.5 s, 64 tokens at 9.9 tok/s** vs 52.6 solo (~20%) |
| cap 96 vs 192 | 32k fill 195.8 → 438.0 s (+124%); rejected, 192 stays default; greedy output token-identical |
| dense precapture | dense (B,W) graphs precaptured before traffic; sparse graph never created |

Notes: two single-tick dense stalls of 7.6/20.2 s appeared in one trace and
were **not reproduced** in a second warmed run (max ≤977 ms) — recorded as
not-reproduced-in-1-of-2, not as fixed.

Raw artifacts: ops-0b V100 `~/trace192v2.log`, `~/trace192.log`, `~/trace96.log`,
dumps `/tmp/long192.json` / `/tmp/long96.json`.
