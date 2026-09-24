# v2 lag-1 async carry rejected at W1024/R32 — V100 sm70, 2026-09-24

> Status: **REJECTED 2026-09-25 (ckl decision).** The v2 async-carry code stays
> on branch `probe/805-v2` and is not merged. This entry also records two
> contradictory on-device readings of the #827 prefix-hit fix; that question is
> not settled (see the final section).

## What v2 was

The sparse decode graph replays a fixed resident page set; every `R=32` decode
ticks the window advances and a refresh tick re-selects pages and promotes cold
ones, eagerly, on the main stream. At W1024/R32 on a 37.6k-token prompt that
eager tick costs ~230 ms against a ~40 ms graph tick. v2 (`LagController`,
`src/tilerl/sparse_lag.py`, probe-only) moved the refresh's snapshot, page
selection and H2D promotion onto a CUDA side stream plus a worker thread one
tick early (lag 1): tick 7 prepares, tick 8 commits the promoted frames and
replays the graph. `TILERL_SPARSE_V2=inline|async|off`;
`TILERL_SPARSE_V2_RESERVE` frames (default 200) held aside for promotions.

The hypothesis being tested: hide the eager refresh cost entirely. Impl's
offline upper bound was ~44.9 tok/s; the A reference was 39.

## Two carry-failure defects found on the real card, both fixed on the branch

The CPU gates cannot see either: both are parity gaps between the lag job's
promotion classifier and the production resolver
(`SparseRuntime.resolve`), and both show up only with the card's real cold-tier
and shared-tracker state.

1. **Shared-prefix pages were never promoted.** A selected page can be resident
   in another request's published prefix (`tracker.shared` page→content-key);
   `resolve()` takes it through `share_take`. The lag job only knew two classes
   — resident and private-cold — so a shared pick found no blob, the carry threw,
   and every carry fell back to eager. Fixed (`fcd4859d`): `prepare()` snapshots
   the per-request shared maps, `_run_job()` adds a `shared_need` class that
   `share_take`s the content key, pins it into the request's pinned set, H2Ds
   into a reserve frame, and registers per-request `request_pins` on commit.
   Gate `scripts/probe_v2_shared_carry_gate.py` (real two-request sparse+spec
   prefix engine; negative control with the branch deleted reads red).

2. **Pages resident nowhere got no frame at all — the `resolve()` else parity
   gap.** The first full-prefill device run armed 0/17 carries; the shared
   hypothesis above was real but not the active cause. The classification diag
   (`bd17b816`) showed selected pages that were in the candidate snapshot with
   resident value `-1`, with `shared_n=0` and non-empty private cold sets: these
   are logical pages never written on this path, which production `resolve()`
   handles in its final `else` by allocating a fresh block and moving no bytes.
   The lag job had no third class. Fixed (`366f5c7c`): a `fresh_need` class pops
   a reserve frame, sets `pool.refcount[blk]=1`, appends it with no blob, so no
   H2D, exactly mirroring `resolve()`. Rollback returns the frame without
   re-homing anything.

After both fixes carries arm: **9 armed / 8 fallback over 547 warm ticks** (B
arm, two prompts). The probe also gained a B>1 guard (v2 only supports B==1;
fallback before any snapshot) and classified fallback reason counters; a
separate device-only blocker, the frozen sparse context binding its build-time
`verify` so the graph tick teacher-forced the wrong object, was fixed in the
probe path. The CPU gates (clean / B=2 guard / teacher-forced / shared-carry)
all pass.

## The remaining failures are capacity, not logic — and the successful carries are too slow anyway

All 8 B-arm fallbacks read identically:

```
reserve 200 < picks 212 (cold 212 shared 0 fresh 0)
```

At this geometry one refresh selects ~212 private cold pages. The reserve pool
is 200 frames, so the whole carry is refused and the tick refreshes eagerly.
Raising the reserve above 212 removes the refusal but does not remove the
transfer: a successful carry still had to H2D ~212 pages on the side stream.

The decisive number is the latency of the **successful** carries
(`ab_B.json`, CUDA-event ms over the run):

| quantity | A: v2 off | B: v2 async |
|---|---:|---:|
| warm effective tok/s | **37.80** | **23.18** |
| B/A | | **0.613 (39% slower)** |
| graph tick p50 | 40.57 ms | 40.90 ms |
| eager tick p50 | 230.27 ms | 270.01 ms |
| eager tick fraction | 0.0317 | 0.0146 |
| carry ms p50 / p90 | 226.7 / 254.7 (eager refreshes) | **257.0 / 1055.9** |
| accept rate | 0.7898 | 0.7625 |
| tokens/tick | 1.795 | 1.766 |

The graph tick is ~40 ms. A successful async carry takes p50 257 ms and p90
1056 ms — it cannot fit inside the tick it is meant to hide behind, so the side
stream's work serialises against the main stream at commit (frames are needed
before graph replay) and adds an average 55.6 ms "plain tick" mean vs 40.9. The
eager path v2 replaces is intrinsically rare at R32: 3.2% of warm ticks, and
when it fires its 230 ms is already paid only every 32 steps. There is no
reserve size or scheduling change that makes ~212 pages × per-page H2D fit in
40 ms on sm70; the lever would have to shrink the per-refresh promotion batch,
which is the cold-tier/refresh-cadence design rather than v2. `B ≥ 40` was the
pre-registered quality-followup trigger; B measured 23.18, so no teacher-forced
quality run was warranted — the speed verdict alone rejects.

Instrument checks all green: lag enabled strictly before graph capture
(0 violations), B==1 every tick (0 violations), residency structural check on
9 carried cycles not violated (`max_resident_frames` 65, pin ceiling 609, no
monotonic free drain), tree `366f5c7c`.

Setup: V100 sm70, Qwen3.8-27B-NVFP4 + MTP d1, W1024/R32,
`TILERL_DRAFT_TRUE_Q_WIDTH=1`, sparse_k 128, full prefill
(`USE_SNAPSHOT=0`; prefill snapshots reproduce the draft prefix-hit path whose
#827 status is unresolved, see below, so they could not be used for this
read), the first two serve805
prompts, 512 generated tokens each, warm window ticks [16,end), CUDA-event ms,
one engine per subprocess A then B. Earlier wall-clock readings (31 vs impl's
39) were a caliber error: wall includes ~7 ms/tick CPU launch gaps; the
GPU-event caliber matches impl and is what the JSONs record.

## #827 prefix-hit readings conflict between the served path and the test harness — not resolved

The #827 fix (`b78f7571`, merged in `5a0c54cc`) holds the entry's stored
boundary hidden at admission and restores it in `_finish_prefills`, so after
the fix the draft reads that stored hidden instead of the hidden recomputed by
the last-page re-forward (the pre-fix bug state; the commit title phrases the
bug, not the fix). Two device readings on
the **same tree `5a0c54cc`** contradict each other:

| path | prompt | hit decode | first differing token | output |
|---|---|---:|---:|---|
| production service, streaming client (this window) | serve805 p0, 37,600 tok | **38.78 tok/s** | — | **byte-identical** to the miss run |
| fixmisc test harness `hitref` | same prompt p0 | **18.16 tok/s** | **1** | diverges, same as pre-fix |

Production-path detail: service restarted first to empty the prefix cache, the
prompt issued twice, 512 tokens requested, temp 0, streamed. Run 1 (miss) paid
75 prefill forwards / 242.19 s TTFT; run 2 (full-page hit) paid **1 prefill
forward / 3.10 s TTFT**, accept 0.4923 in both runs, both outputs the same 304
characters. The hit's 38.78 tok/s matches this geometry's non-hit band (the A
arm above measured 37.8 warm tok/s). The miss run's client-side 5.08 decode
tok/s is not quoted: its SSE first-chunk timestamp lands after the 242 s
prefill and distorts the decode window start. Artifact: `hit_check.json` in
this directory.

The test harness reproduces the pre-fix defect on the identical tree, so this
is not a stale-tree artifact; the two paths exercise different builders,
request shapes and draft-pool limits (the harness's `max_total_tokens` sets a
smaller pool than the service), and fixmisc is discriminating which of those
routes around the fix. Until that lands, the only statement supported is:
**the served path measures a fixed hit (38.78 tok/s, identical output); the
test-harness path measures the opposite on the same tree; #827 is not
confirmed fixed.**

## Artifacts

Committed in `2026-09-24-v2-async-carry-rejected-w1024-r32/` next to this
entry, copied from the V100 run directory `~/v2ab4_out/`:

- `ab_verdict.json` — A/B summary, ratio 0.6134, instrument flags
- `ab_A.json` / `ab_B.json` — full per-arm records including per-prompt
  tok/s, carry/fallback counters, fallback reason string, structural checks
- `hit_check.json` — the two-run #827 miss/hit client record (counter deltas,
  timings, full output text)

Driver: `scripts/run_v2_stacked_ab_v100.sh` + `scripts/probe_v2_window.py` on
`probe/805-v2`. Production on the card was restored to latest main and healthy
after the window; the branch and its probe scripts are retained for the
cold-tier batching work that could change the ~212-page number.
