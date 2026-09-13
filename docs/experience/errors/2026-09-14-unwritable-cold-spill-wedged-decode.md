# An unwritable cold-spill path wedged a 128k sparse decode instead of failing the request — 2026-09-14

## Context

V100 (sm70), sparse k=128, eager, draft d1, slots 4, 8 GiB f16 cold cap,
`--cold-ssd-path /data00/sparse_cold_128k.bin`. A 128k prefill finished
(RSS 20.9 GiB, `kv_cold_bytes` peak 7.21 GiB, SSD 0); decode then ran exactly
128 forwards and froze: counters stopped, `/health` stayed 200, one slot
leaked, and the client hung with no error. 16/32/64/96k answered normally.

`/data00` root is owned by `tiger`; the runnable user can only write under
`/data00/home/chenkailun.c`.

## Root cause

Shared-prefix pages past the host budget spill to a **sibling** of
`--cold-ssd-path`, `sparse_cold_128k.prefix.bin` (`kv_cache.py`
`_shared_ssd_path`), created lazily on the FIRST shared eviction. That
sibling was never validated:

- The private path never got that far — selected own pages stay pinned and
  the shared prefix entries age to the LRU head first, so `kv_cold_ssd`
  stayed 0 while the `.prefix.bin` open raised `PermissionError`.
- It needed ~8 GiB of shared RAM before the first eviction, so short
  contexts never touched the unwritable open; 128k did.

The call stack was
`_sparse_finalize` → `pool.demotions().__exit__` → `cold.hold` →
`_enforce_budget` → `_shared_evict_ram` → `_write_shared_ssd` →
`ColdSsdFile(open("wb"))` → `PermissionError`.

### Why it wedged instead of erroring

The exception escaped `step`; the loop's `catch-and-continue`
(`engine.py` `_loop`) printed a traceback and retried. The throw point is
after the trunk forward but BEFORE `_sample_commit`, so:

- no token committed → the request never advanced, and the next tick threw
  the same error (infinite retry, frozen counters);
- `_release` only runs from `_finish`/cancel, never on that exception →
  leaked slot and blocks;
- the request was never marked `_failed` → the client polled forever rather
  than receiving an error;
- `/health` is independent → stayed 200.

A 120k run failed differently (client got an error at 727 s, slot released):
it hit a capacity refusal path that does go through `_finish`, which is what
made the 128k shape look like a hang.

## Fix

Two changes, one PR.

1. **Fail fast at build.** `HostKvPages` now opens (creates/append) both the
   private spill and its `.prefix.bin` sibling at construction
   (`assert_spill_writable`); an unwritable path refuses engine build with
   the path and errno. A spill capacity that cannot be written is a lie at
   admission (#550 already counts it toward capacity).

2. **No wedge at runtime.** Spill writes are split by who needs the page:
   - a SHARED prefix spill failure is a cache miss: disable shared SSD spill
     for the process (log once), keep the page in RAM, fail no request —
     tokens are unchanged, memory stays bounded by the other tier;
   - a PRIVATE cold spill failure (a page a live row still needs) raises
     `SpillWriteError`; `step` finishes every row in that tick's sparse set
     with `reason="cold_spill_failed"` (the batched `demotions()` exit cannot
     attribute the page to one row), frees their slots/blocks, makes the
     error visible to the client via `take`/`poll`, and keeps serving other
     requests. The batched-exit cleanup frees only the frames not already
     returned, so the pool neither leaks nor double-frees; failed requests
     skip the prefix-end snapshot in `_release` (their cold blobs are gone).

## Gates (CPU, red on main)

- `test_an_unwritable_spill_path_refuses_to_construct` — a read-only
  directory makes `HostKvPages(...)` raise `SpillWriteError`.
- `test_a_shared_spill_failure_stays_in_ram_and_disables_spill` — shared
  write OSError keeps the page served from RAM and disables shared spill; a
  private write still raises.
- `test_a_shared_spill_failure_lets_requests_finish_token_exact` — an engine
  whose shared spill always raises completes two requests token-identically
  to a healthy engine.
- `test_a_private_spill_failure_fails_the_request_and_frees_its_slot` — a
  private spill OSError finishes the owner with `cold_spill_failed`, the
  client sees `RequestFailed`, the slot frees, and a second request completes.

## Rule

A lazily-created output file is an untested precondition: validate every
path the engine will open, including derived siblings, before serving. And a
catch-and-continue loop that does not finish the requests a tick raised on
turns every post-forward/pre-commit exception into a silent slot leak plus a
hung client — the error path must fail the affected rows and release them.
