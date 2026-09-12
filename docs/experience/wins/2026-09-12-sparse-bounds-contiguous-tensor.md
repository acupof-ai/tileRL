# Sparse bounds storage: one contiguous tensor per request, not one per page — 2026-09-12

> Status: **CPU gates green; the 128k decode-tick saving is pending-remote**
> (the host-bound cost is a dispatch-count argument and an op-count gate, not
> yet a measured H20 tick delta).

## Context

`SparseTracker` kept `bounds[req][page]` as one fp16
`[n_full, Hkv, 2, D]` tensor per logical page. `SparseForward._select`
rebuilt each source plane's candidate tensor every tick with
`torch.stack([bounds[rid][p][plane] for p in cand])`. At 128k a row has ~8192
candidate pages and 4 source planes, so each decode tick dispatched ~8192
per-page slices plus a stack per plane — host-bound before any kernel ran.

## What worked

One preallocated fp16 tensor per request, `bounds_t[rid]` of shape
`[cap, n_full, Hkv, 2, D]`, grown by doubling from `INIT_CAP=64`. A logical
page addresses its row directly; `bounds_valid[rid]` marks written rows and
`bounds_count[rid]` is the high-water mark (a promoted candidate already has
bounds, so the finalize loop skips it).

`_select` gathers all candidate rows with one `index_select` and slices the
plane afterward:

```python
bounds = self.tracker.bounds_rows(r["req_id"], cand)[:, plane]
```

The old dict API is gone: `has_bounds` queries the validity mask,
`bounds_rows` gathers, `bounds_bytes` sums valid rows × the cached
bytes-per-page. The token path is unchanged — full-k sparse still matches the
dense engine token-for-token.

## Gates

- Existing sparse gates unchanged: `tests/test_sparse_engine.py` (token-equal
  at full k, demote/promote cycling) and the `page_bounds` ledger row in
  `tests/test_memory_ledger.py`.
- `test_select_tensor_op_count_is_constant_in_candidate_count` — a
  `TorchDispatchMode` counts aten ops inside `_select`; the count is equal at
  8 and 64 candidates. Positive control: the old per-page-slice + stack shape
  dispatches 9 ops at 8 and 65 at 64 (the per-page slice is the growing term;
  `stack` on a list is one op).

## Rule

A hot loop that handles N items must dispatch a constant number of ops; a
list comprehension that indexes one small tensor per item is N dispatches,
even when the final `stack` is one.
