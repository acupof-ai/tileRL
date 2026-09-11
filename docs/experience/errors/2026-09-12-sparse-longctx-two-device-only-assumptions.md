# Sparse long-context could never run: three device-only assumptions past a short-prompt test — 2026-09-12

> Status: **fixed on `fix/sparse-longctx-capacity`** (capacity guard + stable
> cold keys). CPU gates green; the 27B H20/V100 points are the remote proof.

## Context

After #518 merged, the first real sparse requests (H20 128k, V100 32k) both
died at `submit`: "request exceeds KV pool capacity", pool 339 device blocks.
A tiny-model probe past that exposed a second crash two chunks later. Neither
showed in CI.

## Root cause — two independent defects

1. **Admission counted only device blocks.** Under `sparse_k`, `build_engine`
   sizes the KV pool as the per-slot hot set
   `num_slots*(k+window+chunk_pages)+1` (339 on the 27B) and `_admit` allocates
   blocks lazily (`alloc_blocks = 0 if sparse`), demoting older pages to the
   host cold tier. But the submit guard and `room_for` compared the FULL request
   against `usable_blocks` (device only), and `_admit` itself required
   `free_blocks >= total_blocks`. Every request longer than the hot pool (~5.4k
   tokens on the 27B) was refused or queued forever before sparse allocation
   ran. Fix: one `_logical_capacity_blocks` = device `usable_blocks` +
   `cold.budget_bytes // cold_page_nbytes`, used by submit and `room_for`;
   sparse `_admit` needs a slot, not the whole context.

2. **The host cold tier keyed pages by the recycled physical frame id.**
   `HostKvPages.hold(block_id, …)` used the device block id, which `demote_page`
   frees straight back to a pool far smaller than the cold set (13 blocks in the
   tiny repro, 339 on the 27B; cold holds thousands). The next chunk's fresh
   own-page `alloc_block()` reissued a live id and `hold()` dropped the colliding
   page ("host tier dropped block 7"). Fix (58's shape): cold blobs are keyed
   `(req_id, logical_page)` via `demote_page(block, key=…)` / `promote_keyed`;
   the engine's automatic path stores bare logical pages in `cold_pages`. The
   #500 `sparse_retier` demote-all/promote-all seam keeps physical keys and is
   untouched.

3. **The hot pool sized one selection, but a tick holds every group's union.**
   Quest selects independently per source group (groups of 4 full-attn layers;
   the 27B has 4 groups), and all chosen pages co-reside in one shared live map
   through the forward before finalize demotes them. The pool was sized
   `k+window+chunk` for a single group (339 blocks), so the first long decode
   exhausted it promoting the 4-group union. Fix: `n_groups*k + window + chunk`
   per slot (1107 blocks on the 27B). The tiny model has one full-attn layer /
   one group, so only the remote 27B run exercises the multi-group peak.

## Why CI stayed green

`tests/test_sparse_engine.py` submits only 128-token prompts into a fixed
64-block pool: they fit the device pool (defect 1 unhit) and never allocate
more pages than frames, so ids never recycle (defect 2 unhit). The new gate
builds a 13-frame pool, submits 512 tokens (32 pages), and asserts the run
finishes with more demotions than frames — both defects fail it red.

## Rule

A tier whose identity is a recycled resource id must be keyed by the caller's
stable identity, not the resource. And a capacity test has to size the request
past the device tier, or the whole point of the spill tier is untested.
