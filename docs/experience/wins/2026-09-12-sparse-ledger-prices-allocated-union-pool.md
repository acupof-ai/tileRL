# Sparse ledger kv_hot prices the allocated cross-group pool — CPU dry-run, 2026-09-12

> Status: **CPU dry-run** (ledger is derived bytes; no device allocation). The
> `memory.plan` kv_hot row now equals the pool `build_engine` allocates.

## Context

Reviewing #544 (the cross-group `union_cap` design) surfaced a ledger/model
mismatch present since the sparse rows landed: `memory.sparse_rows` priced the
hot set as `k_pages + WINDOW_PAGES` per row, but `build_engine` allocates the
worst-case cross-group pool

```
num_blocks = slots * (n_groups*k + WINDOW_PAGES + chunk_pages) + 1
            chunk_pages = max_num_batched_tokens // 16 + 1
```

On the 27B `n_groups = 4`. At `--sparse-k 128` with the 512-token default
chunk budget the engine allocates `1*(4*128 + 8 + 33) + 1 = 554` blocks per
slot, while the dry-run ledger priced `128 + 8 = 136`. The device row
under-stated the real hot pool ~4.07x (69.06 MiB vs 281.33 MiB at fp8); at 8
slots the gap is the same per-slot factor. The plan is the card-fit and
`--device-free` gate, so an under-priced row plans a pool that does not fit.

## What worked

- `memory.sparse_hot_pages_per_slot(cfg, k, max_batched)` holds the one
  canonical expression (`n_groups*k + W + chunk`), reusing the existing
  `sparse_source_count(cfg)` which already equals `build_engine`'s
  `len(group_map(cfg)[0])` on both cells. `sparse_pool_num_blocks` adds the
  per-slot multiply and the single `+1` spare. `build_engine` reuses the
  helper after the sparse stack hold lifts (this PR is ledger-only per the
  stack freeze — no engine.py/kv_cache.py/sparse_engine.py change).
- `sparse_rows` now takes `num_slots` and `max_num_batched_tokens`; kv_hot is
  `pool_blocks * block` with a note naming the slot ceiling, i.e. allocated
  CAPACITY (worst-case disjoint groups), not per-context resident pages.
- kv_cold stays per concurrent row over the pages outside the
  selected+window set; the transient chunk pages are write headroom and not
  part of the context cold set, so they do not enter kv_cold.
- `cli._sparse_spec` passes `args.slots` and the same chunk budget
  `build_engine` sees (0 → engine default 512).

## Gates

`test_sparse_rows_match_real_tensor_storage_on_27b` now asserts kv_hot equals
the allocated `554 * block` (it asserted `136 * block` before — it passed
while under-pricing by ~4x, so the gate now would have failed on the old
formula). 19 passed / 1 skipped in test_memory_ledger; the checkpoint dry-run
three-row gate and the sparse engine/tier suites green; ruff clean.

## Rule

A ledger row that names an allocation must be computed from the same
expression as the allocator. Factor the pool-size expression into one helper
both price and allocate from; a factor present in one (`n_groups`, the chunk
headroom, the `+1` spare) that is absent in the other is an under-count the
review can only catch by reading both.

## Results

| date | machine | target | result |
|---|---|---|---|
| 2026-09-12 | Mac CPU (27B config, derived) | cpu dry-run | kv_hot k=128: 136→554 blocks, 69.06→281.33 MiB fp8 (4.07x), matches build_engine's per-slot num_blocks; ledger gates green |
