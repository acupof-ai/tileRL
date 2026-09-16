# sm70 serve wedges under a simultaneous-SSE-hangup cancel storm (GIL spin), not a CUDA launch — 2026-09-15/16

**Status:** CLOSED 2026-09-16 — root-caused, fixed in #658 (0cc82a36), and
confirmed on the V100 (device evidence below). The title's "paged_attention
launch hang" was a misread: the engine thread is the *victim*, parked waiting
for the GIL. The main event-loop thread busy-spins inside `stream_or_cancel`'s
SSE final drain during a burst of simultaneous hangups.
**Arch:** V100 sm70, hybrid 27B serve (`--sparse-k 128 --draft … --decode-graph`),
served shas ad0d3a1a → ff3e08e9 (wedged), fixed 0cc82a36.
**Discovered:** P0 during ops late-frame SSE disconnect verification (≈16
mid-stream cancellations). Evidence bundles on the card:
`~/wedge_evidence_2026-09-15/` and `~/wedge_evidence_2026-09-16/`
(`pyspy_*.txt`, `gdb_bt.txt`, `probe.log`, `free.log`); confirmation gate logs
in `~/gate658_081919/`.

## Device confirmation (2026-09-16, ops-cb, V100 0cc82a36 env-off boot 0)

The decisive gate passed end to end, closing the P0:

- serve child pid **346578 unchanged for 1h06m**, boot stayed **0**, **zero**
  exit-10/exit-11, restarts or fuse trips;
- **20/20** SSE and **3/3** non-stream disconnects cancelled and released clean
  (release 0.053–1.748 s);
- the fixed shape — an 8-socket simultaneous-hangup storm
  (`Pool(cap=8).shutdown()`) — drained with no leaked rows and no loop spin; a
  subprocess sampler got **146/146 HTTP 200** through the storm, worst in-flight
  answer **0.003 s**;
- **causal vs memory discriminator:** during the storm physical free bottomed
  at **48 MiB — below the 98 MiB free at which the pre-fix server wedged** — and
  it did not wedge, confirming the GIL-spin cause over the refuted VRAM/allocator
  hypothesis;
- a **128k cold sparse** request (`prefix_hits=0`) streamed 200/`finish=stop` in
  **1037 s (~118 prefill tok/s)** with a **333 MiB** SSD spill.

A non-blocking limit found by the same gate is tracked separately
(OPEN): a *non-stream* 128k request 504s on the fixed 30-min completion timeout;
streaming (used above) has no such deadline.


## Actual root cause (2026-09-16)

`stream_or_cancel`'s `finally` drained the in-flight `asyncio.to_thread(next,
body, …)` worker from **inside the disconnecting SSE task's own cancellation**:

```python
while not worker.done():
    try:
        await asyncio.shield(worker)
    except asyncio.CancelledError:
        continue          # <- spins under a latched anyio cancel scope
    except BaseException:
        break
```

Starlette 1.6 + ASGI 2.3 (httptools) wraps each `StreamingResponse` in an anyio
collapsing task group; a client hang-up cancels that scope **once**, and the
scope stays cancelled, so **every checkpoint** the SSE task hits inside it
re-raises `CancelledError`. With the `to_thread(next)` worker in flight,
`await asyncio.shield(worker)` (itself a checkpoint in the cancelled scope)
throws immediately; `continue` re-checks `done()` (still false) and re-awaits —
no real waiter, a tight Python loop. Eight SSE tasks doing this on one loop turn
pin the main thread on the GIL; the engine `_loop` thread, re-entering Python
through tvm_ffi, blocks in `PyEval_RestoreThread` (gdb: no cuLaunch/cudaMalloc
in the frame). `/health` never gets scheduled. That is why the apparent "stuck
kernel leaf" moved between op shapes (paged_attention, then `_mlp` model.py:566):
the engine is wherever it last crossed into Python. Trigger is op-independent.

A bare `task.cancel()` / `async for` aclose does NOT reproduce it (single
cancel, one re-raise, then a normal wait; bare async-gen aclose throws
GeneratorExit at the yield). It needs the **latched anyio CancelScope** plus N
simultaneous in-flight hangups. A unit reproduction of the inner construct:
inside a cancelled anyio scope a `shield(executor_future); continue` loop spins
~9k iterations/150 ms and starves a peer coroutine.

**Fix:** finalize the sync generator in a task **detached from the SSE cancel
scope** — a module-level strong-ref set holds `_drain_body`, which cancels the
row (idempotent backstop), awaits the in-flight worker (bounded by a timeout)
and then runs `body.close()` on a worker thread, so #649's off-loop
GeneratorExit/cancel constraint still holds. The SSE task returns at once and
cannot spin in its own cancellation. The prompt-path `engine.cancel(rid)` slot
releases stay where they are (awaited in frame, not detached).

**Second defect found by the reproduction gate (same day):** the first detached
drain awaited the worker *before* cancelling. In an 8-way hangup storm, 2 of 8
teardowns (measured) delivered the throw to the `yield` point as a plain
`GeneratorExit`, which `stream_or_cancel`'s `except CancelledError` disconnect
branch never sees — so no in-scope cancel ran, the in-flight `to_thread(next)`
was parked until a cancel that never came, and the drain dead-waited its full
timeout and abandoned `body.close()`: a leaked slot, with no loop spin. The
deterministic structural gate pins the drain's ordering (cancel before
wait-for-worker); the real-uvicorn 8-socket gate asserts all rows release.
Both are red on pre-fix code and green after, and the socket gate also required
a widened default executor in the harness: N parked workers plus N cancel jobs
exceed the stock executor's `min(32, cpu+4)` (=8 on 4-vCPU CI), a harness-only
constraint since the real sync body polls in short slices rather than parking
solid.

Review added two more requirements to the detached drain, both about its
failure/exit edges:

- a drain that times out or whose worker raises no longer skips `body.close()`
  silently — it logs a warning with the rid, timeout-vs-exception, and
  "close skipped"; the slot is already free from the first cancel, but an
  unclosed generator on the wedge box must be visible;
- graceful shutdown (SIGTERM) joins in-flight drains via the app lifespan with
  the same bounded `asyncio.wait(_draining, timeout=_DRAIN_WAIT_S)` (it does
  NOT cancel them — each is already self-bounded — and logs pending count).
  The supervisor's SIGKILL-on-wedge path needs no join; ordinary restarts do,
  since an unclosed sync generator on that path is a real leak. A gate starts a
  drain with a blocking stub `close()` and asserts the join stays pending until
  close returns.

## The VRAM/allocator hypothesis — refuted as the cause

Two candidate preventions were built and measured before the GIL root was found;
both are useful defense-in-depth but neither is causal:

- a build-time KV-pool trim (#654, peak-live reserve) could not hold free:
  `TILERL_DEVICE_RESERVE_MIB=768` cut 0 blocks, post-warmup idle free 602 MiB,
  because the caching allocator re-reserves ~4.1 GiB of segments after the build
  snapshot;
- a held process memory fraction (#655, `set_per_process_memory_fraction`)
  deployed and bounded the process correctly, but the 2026-09-16 churn wedged
  again with **zero** fence hits (no exit-11/OOM; min physical free 98 MiB was
  correlational), and the engine thread was off-CUDA waiting on the GIL.

#655 stays (default off): an over-fence allocator OOM is independently made
fatal — classified by concrete `torch.cuda.OutOfMemoryError` in the forward and
build paths → `FatalDeviceError` → `os._exit(11)` + marker → supervisor restart,
instead of being swallowed by the daemon loop's log-and-continue.

## Earlier observations (2026-09-15, consistent with the GIL root)

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

### 2026-09-16: a build-time peak-live reserve does NOT hold — allocator re-reserves

#654 first implemented the margin as a build-time KV-pool trim: read free after
weights/state, size the pool down once so projected idle free met the floor.
The V100 env experiment (`TILERL_DEVICE_RESERVE_MIB=768` on c52fbbdc) showed
this is **not a held reservation** and excluded that hypothesis:

- the reserve recorded (`device_reserve_bytes=768 MiB`) but **cut 0 blocks** —
  the sparse pool stayed 2213, post-warmup idle free only **602 MiB < 768**;
- decomposition from `/health`'s ledger: live tensors ~28035 MiB (live-only free
  ~4733 MiB) but `mem_get_info` free 602 MiB — the missing ~4.1 GiB is the
  caching allocator's **reserved-but-unused segments**, which appear after the
  build snapshot (warmup/lazy JIT) and which a build-time projection cannot see.
  Trimming the pool lowers the peak of live tensors; it does not pin free
  against the allocator reclaiming it.

The held mechanism is a process memory fraction
(`torch.cuda.set_per_process_memory_fraction((total−reserve)/total)`) set before
weights load: it caps `mem_get_info` and turns an over-fence `cudaMalloc` into
an explicit, **catchable** `torch.cuda.OutOfMemoryError` (the supervisor restarts
on that) instead of a host-side blocking launch. Precondition verified on sm70
(torch 2.5.1+cu121) by the one-off on-box probe `~/probe_memory_fraction.py`
(kept on the V100 host, not in `scripts/` — a closure-gate dead script; rerun if
a torch/CUDA change is suspected to alter OOM behavior): fence raised catchable
OOM and a small alloc+fill+sync after it succeeded, so the context survives.
Result: `PROBE_OK`.

That catchable OOM is **fatal, not swallowed**: a first implementation let
`step()`'s `except Exception` drain the rows and re-raise into `_loop`'s
log-and-continue, so the process stayed alive, drained to idle, answered
`/health` 200, and kept accepting rows — the #650 watchdog could not catch it.
The fix classifies `torch.cuda.OutOfMemoryError` (forward AND build allocation
paths) as a `FatalDeviceError`: rows are NOT drained, the engine records fatal,
`liveness()` reports dead, `submit()` refuses new rows, and a single seam
(`fatal_device_exit`, os._exit(11) with a greppable marker) terminates so the
#652 supervisor restarts. There is no in-process recovery — the fraction is
fixed for the process, so freeing the rows in flight cannot make room. Ordinary
per-request `RuntimeError`s still go through `_finish` and keep serving. The
GDN state ledger row was also found to undercount ~3x
(776 vs 2273 MiB) — `memory_rows` passed `spec_steps=0` and dropped both spec
step planes; fixed alongside the fraction work.

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

