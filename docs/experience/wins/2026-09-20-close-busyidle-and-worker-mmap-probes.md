# Close busy/idle attribution, worker-mmap split, cap reclaim sampler — 2026-09-20

> Status: landed env-gated, default OFF. Device-only instrumentation for the
> final #743 follow-up window; full CPU suite 1030 green. No behavior change
> unless `TILERL_CLOSE_BUSYIDLE=1` (which requires `TILERL_STEP_TIMING=1`);
> the worker/step spill split is always correct but only printed under the
> busy/idle gate.

## Context

The #743 device reconciliation (wins/2026-09-20-background-close-publish.md)
left one question UNDECIDED: with the background publisher queue large enough
to avoid inline fallback, `release_close_request` still charged ~1.0–6.5 s and
the ColdSsdFile self-measured `ssd_mmap` landed in that tick. Two causes were
indistinguishable from the existing timer:

1. the step thread genuinely blocked on the device stream or on the cold
   tier's `_tlock` (a real stall the planned 3PR lock split would remove);
2. the background worker's spill-file disk time was being **attributed across
   threads** into the step tick's `ssd_mmap` drain — an accounting artifact
   overlaid on whatever real contention exists.

`fwd_gpu` could not separate them: it is a no-`synchronize` CUDA-event span
that inflates under any mid-forward blocking wait even with the device idle.
This PR adds the two instruments that split the two, plus the reclaim sampler
that was missing from the #740 window.

## What Worked

Three pieces, one opt-in surface:

- **Close busy/idle bracket** (`src/tilerl/engine.py`,
  `TILERL_CLOSE_BUSYIDLE=1`). `_StepTiming.close_bracket_start/end` bracket the
  sparse `release_close_request` block with a host `perf_counter` anchor and a
  reused pair of async CUDA events. At tick end `_close_busyidle_fields`
  resolves them with non-blocking `query()`/`elapsed_time` only — never a
  synchronize — and appends `close_wall=<w> close_dev=<d> close_host=<w-d>` to
  the slow-tick line. `close_dev` is stream work completed inside the bracket;
  `close_host` is the off-stream wait (a lock or a host mmap). If the end event
  has not drained it prints `close_dev=pending`, honest rather than forced.
- **Worker vs step spill accounting** (`src/tilerl/kv_tiers.py`).
  `ColdSsdFile._charge` now routes IO done on the thread named
  `tilerl-cold-publish` to a separate `ssd_ms_worker` accumulator;
  `drain_ssd_ms()` stays step-thread-only and a new
  `drain_worker_ssd_ms()` returns the worker share. The engine drains the
  worker bucket every tick (so it cannot accumulate) and, under the busy/idle
  gate, prints it as `ssd_mmap_worker` — never charged into a blocking segment.
  Thread name is the only discriminator, so the split is exact without touching
  the worker's call sites.
- **Cap reclaim sampler** (`scripts/probe_headroom_coldtail.py
  reclaim-sample`). A passive subcommand samples one spill file's apparent size
  on a fixed interval with timestamps and writes a GiB (2³⁰) summary —
  `peak_apparent_gib`, `last_apparent_gib`, `reclaimed_off_peak_gib`,
  `shrank_after_peak` — built from a SINGLE operand (apparent bytes), so it
  cannot repeat the #742/#744 mixed-unit/operand errors. It drives no requests:
  run it alongside an arm whose publish refs release mid-window, keep the spill
  until sampling ends, then stop the serve to delete. A plateau honestly
  reports zero reclaim.

## Gates (CPU, hermetic)

- `test_close_busyidle_bracket_is_off_by_default_and_pending_on_cpu`: gate
  unset → bracket no-ops and emits nothing; on, the CPU wheel opens/closes and
  resolves to `close_dev=pending` (no fabricated device number).
- `test_close_busyidle_splits_host_and_device_with_a_fake_event_pair`: a fake
  200 ms event span inside a 500 ms wall prints `close_dev=200
  close_host=300`, asserting the busy/host split reads the events, not a block.
- `test_spill_io_on_publish_thread_is_billed_separately`: real write/read on a
  thread renamed `tilerl-cold-publish` accrues only to `ssd_ms_worker` and
  leaves the step `ssd_ms` byte-for-byte unchanged; main-thread IO still bills
  to `ssd_ms`.
- The pure reclaim sampler core (`_reclaim_rows`/`_reclaim_summary`) is
  self-checked with a grow-then-truncate series (peak 8 → last 5 GiB, reclaim
  3), a plateau (reclaim 0, `shrank_after_peak=False`), and an all-zero series.
- The existing AST gate still holds: the new bracket adds no `synchronize`,
  uses `query`/`elapsed_time`, and the new `perf_counter` reads live inside
  `_StepTiming` (the engine call sites use the bracket methods, not raw reads).
- Full suite 1030 passed / 22 skipped / 1 xfailed; `ruff check` +
  `ruff format` clean.

## Device use (next window, pending-remote)

Boot with `TILERL_STEP_TIMING=1 TILERL_STEP_TIMING_SLOW_MS=0
TILERL_CLOSE_BUSYIDLE=1` plus the close flags under test, then:

- read `close_wall/close_dev/close_host` on the release ticks to call the
  residual device-busy vs host-blocked;
- compare `ssd_mmap` (step) against `ssd_mmap_worker` to subtract cross-thread
  disk accounting;
- start `reclaim-sample --spill-path …/.prefix.bin --interval-s 20 --samples N`
  before the arm and let it run past the publish refs' release for the #740
  trailing-truncation evidence, spilling retained until the sample completes.

## Rule

A wall timer that brackets one thread cannot, by itself, say whether that
thread waited on the device or on a lock — bracket it with non-blocking device
events, and never let another thread's self-timed IO drain into the measured
thread's bucket. Report unresolved events as pending rather than synchronizing.

## Results

| date | commit | machine | target | result |
|---|---|---|---|---|
| 2026-09-20 | pending PR | CPU (hermetic) | close busy/idle + worker mmap split + reclaim sampler | device-busy/host-blocked split and worker/step spill split unit-pinned; reclaim sampler single-operand GiB; device attribution pending-remote |
