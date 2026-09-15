# sm70 paged_attention launch hangs forever after SSE churn, wedging the engine — 2026-09-15

**Status:** open — unresolved. Trigger and exact blocking point not reproduced
in isolation; do not record as root-caused.
**Arch:** V100 sm70, hybrid 27B serve (`--sparse-k 128 --draft … --decode-graph`),
served sha ad0d3a1a.
**Discovered:** P0 during ops late-frame SSE disconnect verification (≈16
mid-stream cancellations). Evidence bundle on the card:
`~/wedge_evidence_2026-09-15/`.

## Observed

After a burst of late-frame SSE disconnects, the engine loop thread held
`engine._lock` and never returned:

- thread state `R`, 92–107% userspace CPU, `wchan=0` (a Python/C busy or
  blocked launch, not an IO wait or a normal lock wait);
- **GPU utilization pinned 0%**, no `dmesg` Xid/ECC/NVRM error;
- `step` / `decode_forwards` frozen for 8s+ and never advanced; new `submit`
  timed out on `engine._lock`; three cancel workers queued behind the lock;
- zero traceback; rows neither progressed nor cancelled.
- `/health` kept answering **200** the whole time: `stats()` returns the last
  `_stats_snapshot`, so a wedged step looks healthy (the liveness blind spot
  tracked separately).

py-spy parked the main thread at
`model.forward → _full_attn → Backend.paged_attention → tilelang
paged_attention_split`, JIT key `(16,64)/(3,24,512,4,2214,256,2214)` —
B=3, S=512 (wide KVSPLIT=16), num_blocks 2214. The launch did not return.

## Ruled out (measured, not inferred)

`scripts/repro_sm70_split_b3_hang.py` runs the **real**
`paged_attention_split` end to end on the V100, ragged seq_q 512/384/256,
three launches per shape (first pays lazy JIT), with a per-launch wall
watchdog:

- **B dimension is not it**: B∈{1,2,3,4}, S=512 all return; B=3 repeat 2+ is
  18–19 ms (first B=3 JIT ~90 ms; other B first JIT 3.4–5.8 s is ordinary
  nvcc compile).
- **num_blocks 2214 is not it**: rerun at the incident's NB=2214, every B
  returns, B=3 repeat 2+ 18.6–19 ms.
- Therefore not a B/NB-specialized device grid dead-loop and not an un-warmed
  shape. The kernel is correct in isolation.

(The repro needs `PATH=/usr/local/cuda-12.4/bin` first — the default
`/usr/bin/nvcc` rejects `-std=c++20`. A standalone process with free memory is
NOT a faithful repro of the allocator state, see below.)

## Leading hypothesis, not reproduced

At the incident the device had only **~114 MiB free**; the split kernel's `PO`
output is a fixed `[B,S,H,KVSPLIT,D]` f16 tensor = 288 MiB at B=3/S=512,
allocated inside the kernel via `T.empty`. Leading suspicion: after a long
serving run, when VRAM is at the edge, the PyTorch caching allocator's
reserve / `cudaMalloc` (or an implicit sync around it) on the host side blocks
indefinitely rather than returning OOM, on the engine loop thread, holding the
GIL and `engine._lock`.

This is NOT independently reproducible: a bare process that pushes free memory
below the allocation size gets an immediate OOM (different allocator state —
no long-run reserve pool, no churned blocks). The incident allocator state is
specific to the long-running server under churn, which is why enumeration on
the card was stopped.

The SSE fix #649 (close the sync generator body off the event loop) is
correct and unrelated to the hang; by making cancel earlier/more frequent it
may only have raised the probability that admitted-then-cancelled prefill
rows coexist in one batched tick.

## Needed / planned

Resilience (decoupled from the trigger — any single kernel/allocator stall
must not sink the server):

1. **Observability (first, zero-risk):** `/health` carries a step-loop
   last-progress timestamp; `now - last_progress` over a threshold reports
   DEGRADED/503 with `stuck_secs`. Turns the false-200 into something LB/ops
   can detect.
2. **Fail-fast boundary:** a step/forward wall watchdog that, on timeout, does
   NOT try to kill the C-extension thread (unsafe — would corrupt the CUDA
   context) but forces a controlled process exit for the supervisor to restart.
   Needs confirmation the supervisor auto-restarts an unhealthy child (this
   incident was recovered by a manual kill -9).
3. **Prevention (parameter):** keep a larger VRAM safety margin for KV-pool fit
   / sparse admission (the ~114 MiB headroom is too tight; reserve on the order
   of one PO / a few hundred MiB) to reduce the chance of reaching the edge.

Related: [2026-09-15-sm70-long-step-tick-holds-engine-lock.md](2026-09-15-sm70-long-step-tick-holds-engine-lock.md)
(the self-limiting 1–5.6 s sibling tick — that one returns, this one does not),
[2026-09-15-sse-midstream-disconnect-never-cancelled-the-row.md](2026-09-15-sse-midstream-disconnect-never-cancelled-the-row.md).

## Addendum 2026-09-15 (second trigger, post-#650) — ops device run

A second re-run of the 20-rep SSE-disconnect churn on the V100 at **29e767de**
(#650 `/health` step-progress watchdog deployed) re-triggered the wedge at
**disconnect rep 14** (the first incident was ~rep 16). Data points added by
ops; conclusions above unchanged:

- The stuck leaf this time was **`Model._mlp` (model.py:566) → `forward`
  (model.py:646)** — NOT `paged_attention`. GPU 0 %, and **free VRAM was only
  28 MiB** (tighter than the first incident's ~114 MiB). The blocking point
  moving from `paged_attention_split` to `_mlp` as free memory shrank is the
  strongest evidence yet that this is an **op-agnostic memory-edge CUDA launch
  / caching-allocator host-side stall**, not one kernel.
- #650 behaved as designed: while wedged `/health` returned **503
  `{"status":"unhealthy","stuck_secs":82.0}` in ~54 ms** instead of the old
  false 200.
- SSE reps 0–13 all released rows (0.11–1.65 s) and the non-stream arm passed
  with in-flight `/health` under 0.5 s, so the #649 cancel path is clean; the
  failure is only the underlying wedge.
- Supervisor recovery was verified end to end on the EXISTING (pre-fast-503)
  liveness: 3 consecutive chat failures over ~3 min → liveness exit 10 →
  wedged child killed (rc=137) → GPU back to 0 → launcher boot 1 → 29e767de
  health 200 after warmup, zero tracebacks. The in-flight #2 change shortens
  this to two 503s; this run is the slow-path baseline.

Evidence on the V100: `~/wedge_evidence_2026-09-15/verify649b.log` and
`health_649b.log` (independent-subprocess `/health` samples and the rep-by-rep
release trace for this second trigger).

