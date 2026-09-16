# Project wrap-up closure — 2026-09-16

**Status: sealed.** ckl decided 2026-09-13 to stop growing tileRL's own framework and
put the weight on the surrounding ecosystem ("不自己搞了，生态更重要"). This entry freezes
the final state of the three closing work lines — the 11-step layered-architecture
refactor, the 35-finding quality audit, and the online robustness chain — at
origin/main `178af92b` (the SSE cancel-storm P0 closed at this point). Every PR
number below was read from `git log`/`gh`, not recalled. What remains is
device-deferred work and one fixed-deadline serving limit, not new framework work.

## 1. Final state

### 1.1 Layered architecture refactor — 11/11 merged

The full plan is [`docs/design-architecture.md`](../../design-architecture.md). The
numbering has no step 3. All steps are on main:

| step | what landed | PR |
|---|---|---|
| 0 | target architecture doc | fa2ae898 |
| 1 | `tests/test_layering.py` direction gate, shrink-only allowlist | #592 |
| 2 | stale design docs archived/reconciled | #593 |
| 4 | dead cli/route code removed (`pretrain`, `rollout`) | #594 |
| 5 | scripts/ cleanup closed: 7-set reachability, 0 DEAD, MANUAL_KEEP gate | #647 |
| 6 | five cross-layer cycles broken; `dflash2` merged into `spec` | #601 |
| 7a/7b | `build.py` single assembler **and** the ~120 caller sites in one PR (no shim) | #599 |
| 8a | training orchestration → `train.py`, finalization → `ledger.py` | #606 |
| 8 | bench commands → `bench.py` | #646 |
| 8b | one non-stream request path in `prompt.py` (SSE/ws excluded by name) | #614 |
| 9 | host/SSD tiers + boot store → `kv_tiers.py` | #612 |
| 10a | captured decode graphs → `decode_graph.py` | #616 |
| 10b | pure ledger memory builders → `memory.py` | #618 |
| 11 | `SparseRuntime` seam (12 sparse-tick methods, frozen SparseCtx) | #622 |
| — | phase-exit write-up | #623 |

Imports point downward only (L0–L6); the layering gate's allowlist is empty. Two
production correctness bugs were found and fixed during the move (#605: sparse
no-draft full-prefix hang + phantom final-tick block). The follow-on packaging
fix that made `benchrec` importable from a wheel is #638.

### 1.2 Quality audit — 32/32 executed + F6 device arm closed; F21/F22 deferred

The 2026-09-14 audit ([`docs/quality-audit-2026-09-14.md`](../../quality-audit-2026-09-14.md))
confirmed **35 findings, 11 refuted**. Of the 35, one is sm90/H20-only (F22) and
waits on a card; the other **32 executable findings are 32/32 closed**, and both sm90
verification arms **F6 and F21 have now run green on an H20** (2026-09-16; F21 verdict
[here](2026-09-16-kvfp8-27b-sm90-device-verdict.md)) — only **F22** remains
device-deferred. Every executable finding is fixed and on main:

| finding | fix | PR |
|---|---|---|
| 1, 2 sparse no-draft hang + phantom block | engine fix + gates | #605 |
| 3 timeout/error path did not cancel the row | off-loop cancel on all routes | #631 / #637 |
| 4 SSE mid-stream close never cancelled | real-uvicorn cancel watcher | #631 |
| 5 disconnect harness fed cancel, not `http.disconnect` | #598 real-disconnect gates | #598 |
| 7 /health wall-clock lock gates | event-synced liveness gates | #642 |
| 8 `*_world*` negative controls never ran | hermetic CI negative controls | #635 |
| 9 layering gate missed absolute imports | absolute+relative resolution | #592 |
| 10 /ws hang-up was source-grep only | behavioral ws disconnect gate | #631 |
| 11, 12 tool transcript / `input_text` parts | route render + gates | #624 |
| 13, 15, 16 tool_choice none / streamed tool_calls / hosted tools | one PR | #625 |
| 14 reasoning-effort caps only on chat | caps on all three routes | #627 |
| 17 unbounded submit admission | `EngineOverloaded` 503 backpressure | #630 (+#633/#634 envelopes) |
| 18 calibration `_row` hardcoded sm90 | writes its target | #611 |
| 19 corrupt KvBoot entry 500'd and leaked blocks | admits as a miss, frees blocks | #615 |
| 20 `select_pages` bypassed the Backend seam | routed through Backend | #648 |
| 23 framework reached backend privates | exposed on the Backend seam | #643 |
| 24 `benchrec` loaded by filesystem path | packaged into the wheel | #638 |
| 25 dead `_evict_to_ssd` | deleted | #636 |
| 26, 27 archive/open-defect counts drifted | corrected + anti-drift gate | #645 |
| 28, 29, 30 stale design/serving docs | reconciled | #607 |
| 31–34 pod selftests / audit scanner / gil bench rewrites / `--source` | one sweep (selftests later gated in #645) | #603 |
| 35 duplicated `0.02` poll literal | shared `POLL_INTERVAL_S` | #644 |

Device arms:

- **F6 — closed, including sm90 (2026-09-16).** The gate defect is fixed in #663:
  `tests/test_attn_prelude_oracle.py` now builds both preludes into a real
  `PagedKvPool` and asserts the fused `attn_prep` is strictly closer to the f64 oracle
  than the discrete chain by mean error (`ef < ed`, `ef ≤ 0.6 ed`), with non-vacuity
  guards. The sm90 arm ran **unskipped on H20 card 0** (tree `8428babd` + #663,
  `/work/tl013` torch 2.11.0+cu129): 3/3 PASSED, and on 3579 differing elements the
  discrete mean was 1.815e-03 vs fused 9.297e-04 — discrete/fused **1.9527** (fused
  ≈ 0.512× discrete), matching the 27B record of 2.0007x in
  [`errors/2026-09-03-unfused-prelude-double-rounds.md`](../errors/2026-09-03-unfused-prelude-double-rounds.md).
  The run used the pod's maintained `/work/tl013` (torch 2.11.0+cu129) via
  `python -m pytest`, not `uv run` — the fresh `.venv` carries a cu130 torch the 12.9
  driver cannot load; runbook:
  [`errors/2026-09-16-h20-pod-uses-tl013-cu129-not-uv-run-cu130.md`](../errors/2026-09-16-h20-pod-uses-tl013-cu129-not-uv-run-cu130.md).
- **F21 — CLOSED 2026-09-16 (sm90 device arm).** KV fp8 verified on the 27B/H20:
  next-token agreement 24/24 (1.0, first divergence null), e4m3 per-token K/V
  round-trip error 3.57% over row amax, 1.969x resident KV capacity (45338→89281
  blocks @32k, 22→32 of B=32 resident). Per-tick decode is still 0.83–0.95x at
  B≤8 (+~20% gather dequant, 24.4 GB weights dominate; KV ≤41% of a tick), so the
  path is correct/usable but stays **default off** — a capacity lever, not a
  speed-up. Full numbers:
  [`2026-09-16-kvfp8-27b-sm90-device-verdict.md`](2026-09-16-kvfp8-27b-sm90-device-verdict.md).
- **F22** Quest `page_bounds` cells are sm70-only with no target-neutral CPU twins and
  the Backend methods are unused by production.

So: 32 CPU/route-executable findings **32/32 closed**; F6 and F21 sm90 verification
arms both ran green on H20; only the one sm90/H20 device arm **F22** waits on a named card.

## 2. Online capability (V100 sm70)

A dense 4-slot / 8k-context serve endpoint is up and stable
(`n37-002-027:8000`, verified 2026-09-13), alongside the hybrid sparse long-context
serve. Around it, a robustness chain closes detection → fast restart → fatal-OOM →
observability:

- **#650** `/health` reports **503 when the step loop stops progressing** — a wedged
  engine no longer answers "ok" (detection).
- **#651** liveness is stamped on the **submit idle→active edge**, so an idle-then-busy
  transition cannot hide a stalled loop from #650.
- **#652** a hybrid-V100 **supervisor** does a fast restart on lost liveness, with a
  restart fuse so a crash loop does not hot-loop (fast restart).
- **#655** device reserve is held via a memory fraction and the GDN state ledger is
  fixed, closing the fatal-OOM class (fatal-OOM).
- **#656** `/health` exposes in-process device **free/limit bytes** (observability).

The cancel path these protect was itself hardened on the device (#631/#637) and the
worker-exception hole after it closed (#641); worst in-flight `/health` while a cancel
parks on the engine lock is under 0.5 s, versus a pre-fix multi-second event-loop
freeze.

## 3. P0 — SSE cancel storm that wedged the engine (CLOSED)

**Status: closed 2026-09-16** — root-caused, fixed in #658 (`0cc82a36`), and
verified end to end on the V100; documented in #661. Full evidence:
[`errors/2026-09-15-sm70-paged-attention-launch-hang-wedges-engine.md`](../errors/2026-09-15-sm70-paged-attention-launch-hang-wedges-engine.md).

Symptom on the unfixed server: after ~16 late-SSE disconnects at ~114 MiB free
on sm70, the engine wedged — `/health` falsely 200, submit/cancel blocked, GPU 0%,
no Xid. The apparent stuck op moved between shapes (`paged_attention_split`
B=3/S=512/NB=2214, then `_mlp`), which is why neither the kernel nor the
allocator was the cause.

Actual root cause: `stream_or_cancel`'s `finally` drained the in-flight
`to_thread(next, body)` worker from *inside the disconnecting SSE task's own
cancellation*. Under Starlette 1.6 + ASGI 2.3 the anyio CancelScope latches
cancelled, so every checkpoint re-raises `CancelledError`; an
`await asyncio.shield(worker); continue` loop then spins with no real waiter.
Eight SSE tasks doing it in one loop turn pinned the main thread on the GIL, and
the engine thread re-entering Python via tvm_ffi blocked in
`PyEval_RestoreThread` — op-independent, hence the moving "stuck leaf". Fix:
finalize the sync generator in a task **detached from the SSE cancel scope**
(a module-level strong-ref set cancels the row, awaits the worker bounded, and
runs `body.close()` on a worker thread); a second storm-only drain-ordering defect
found by the same gate was fixed in the same change.

Device confirmation (ops-cb, V100 `0cc82a36`, env-off, boot 0):

- serve child **pid 346578 unchanged for 1h06m**, boot stayed 0, zero
  exit-10/exit-11, restart, or fuse trip;
- **20/20 SSE** and **3/3 non-stream** disconnects cancelled and released (0.053–1.748 s);
- the fixed shape — an 8-socket simultaneous-hangup storm — drained with no leaked
  row and no loop spin; a sampler got **146/146 HTTP 200** through the storm, worst
  in-flight answer **0.003 s**;
- **causal discriminator:** physical free bottomed at **48 MiB — below the 98 MiB
  free at which the pre-fix server wedged** — and it did not wedge, confirming the
  GIL spin over the refuted VRAM/allocator hypothesis;
- a 128k cold-sparse request streamed 200/`finish=stop` in **1037 s (~118 prefill
  tok/s)** with a **333 MiB** SSD spill.

The route-layer hardening beneath this (#631/#637/#641/#649) and the resilience
pieces (#650 stuck-503, #652 supervisor restart, #655 reserve) all hold; this was
the one P0, and it is closed rather than worked around.

## 4. Carry-forward ledger

- **Non-stream 128k hits the fixed 30-min completion timeout (504).** The #658 gate
  also exposed a real serving limit distinct from the now-closed wedge: a **non-stream**
  128k cold-sparse request reaches the fixed `_COMPLETION_TIMEOUT_S = 1800.0` and 504s,
  while the **streaming** arm has no such deadline and completes (the 1037 s / 118
  tok/s fill in §3). The loop is healthy and the row progresses — only the fixed await
  deadline fires. Tracked as OPEN:
  [`errors/2026-09-16-nonstream-128k-hits-fixed-completion-timeout-504.md`](../errors/2026-09-16-nonstream-128k-hits-fixed-completion-timeout-504.md);
  fix is a configurable per-request/long-context timeout or pointing long-context
  callers at `stream=true`.
- **One intermittent gate flake** is open, not blocking:
  [`#659`](https://github.com/acupof-ai/tileRL/issues/659)
  (`test_a_nonstream_client_disconnect_cancels_its_request` occasionally reports
  "request never reached the engine").
- **256k long-context is paused**, not abandoned, on device-memory grounds — see
  [`errors/2026-09-12-sparse-256k-spill-host-rss-oom.md`](../errors/2026-09-12-sparse-256k-spill-host-rss-oom.md),
  [`errors/2026-09-13-v100-256k-sparse-prefill-host-oom.md`](../errors/2026-09-13-v100-256k-sparse-prefill-host-oom.md),
  and the partial 5-of-8 NLL result. The sparse cold tier's mmap spill still lives
  behind `--cold-ssd-path`; the dense SSD KvTier was removed (1.65x worse at 12
  sessions).
- **One H20/sm90 device arm** (**F22**) waits on a named H20 window; the F6 sm90 arm
  closed green on card 0 (§1.2) and **F21 closed 2026-09-16** on card 2 — KV fp8
  correct (24/24 agreement), 1.969x capacity, default off (decode 0.83–0.95x at
  B≤8); see [`2026-09-16-kvfp8-27b-sm90-device-verdict.md`](2026-09-16-kvfp8-27b-sm90-device-verdict.md).
- **Slow periodic forward pair (unlogged observation).** On the device the forward
  time shows a recurring slow pair at roughly **~980 / ~1100 ms every ~50 ticks**
  against a normal ~10 ms baseline. This is an on-box observation only — it is not yet
  pinned to an errors entry or a run record, the trigger is unlocalized (candidate: a
  periodic allocator or housekeeping sweep), and no number here is quoted from a file.
  Recording it so the anomaly is not lost; it needs a segment-timed capture before it
  becomes a measured defect. (Instrumentation for that capture landed in #639, the
  env-gated per-segment step timing.)

The standing OPEN list is [`docs/experience/OPEN.md`](../OPEN.md), eleven live rows:
the non-stream 128k 504 above; the sm70 1–5.6 s lock-holding step tick; sm70
sparse-decode-graph + MTP d1 multi-token corruption; the 5.59 tok/s
short-request-during-sparse-prefill; the irreproducible sm90 B=8 cold spec wave; the
recorded 2.6x training/serving rollout gap (no gap on sm70, needs sm90); four
sparse-publish cost rows (self-reinforcing miss, entries-per-row vs snapshot budget,
the 1.2–2.5 s depth-independent hit cost, and the tier converting byte pressure into
block pressure); and the unbounded CUDA GDN chunk-rounding path. The former P0 wedge
(§3) was closed and replaced on the list by the non-stream 504, so the count holds at
eleven. These are measured and owned; they are deferred device work or a fixed-deadline
limit, not gaps in the shipped architecture.

## 5. What this deliberately is not

No new framework feature ships after this. Remaining effort is device verification
(H20 windows), the non-stream 128k completion-timeout limit (§4), and
ecosystem-facing work; the cancel-storm P0 is closed (§3). The CPU/route
surface is sealed by CI: the layering gate, the scripts reachability gate, the
frozen-route shapes, the real-uvicorn disconnect gates, and the docs count gates all
fail on regression rather than relying on a reviewer remembering.
