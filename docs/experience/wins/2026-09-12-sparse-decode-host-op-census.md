# Sparse decode host-op census: what blocks graph capture — CPU engine, 2026-09-12

> Status: **CPU census (counts are device-independent).** The card work is the
> tick-time delta after #534's pin lands, not these op counts. Reproduce with
> `scripts/census_sparse_decode_host_ops.py <tag>` on each head.

## Why

A captured decode graph replays a fixed op graph; it cannot run Python that
builds tensors from lists or reads a scalar back to the host every replay. The
sparse decode path does all of that — selection and tiering run in eager
Python — so before making the tick capturable we need the map of exactly where
the host work is, and which of the two open PRs removes what.

Counts are aten ops dispatched inside one `engine.step()` with every row in
decode, via a `TorchDispatchMode`; per-plane numbers wrap
`SparseForward.attention_args` (one call per full-attn layer — tiny has one
full-attn layer, so "per plane" is that one call over the whole batch).
Buckets:

- **sync item/bool/float** — `aten._local_scalar_dense` (`.item()` /
  `bool(t)` / `float(t)`): a device→scalar stall on a card;
- **sync tolist** — `aten._tolist.data` host copy;
- **python-built tensors** — `aten.lift_fresh*`: `torch.tensor([...])` /
  `torch.tensor(n)` from a Python object, a host allocation fed into aten.

tiny engine, `sparse_k=2`, prompt six pages so selection is real (candidate
set > k). dense is the same engine with no sparse tracker.

## Table — one decode tick

| head | source | aten ops | item/bool/float | tolist | py-built tensors | H↔D copies |
|---|---|---:|---:|---:|---:|---:|
| main | dense B=1 | 516 | 3 | 0 | 7 | 44 |
| main | sparse B=1 k=2 | 600 | 4 | 0 | 10 | 72 |
| main | dense B=8 | 1227 | 24 | 0 | 35 | 107 |
| main | sparse B=8 k=2 | **1878** | 32 | 0 | 52 | **331** |
| #534 pin | dense B=1 | 516 | 3 | 0 | 7 | 44 |
| #534 pin | sparse B=1 k=2 | 528 | 4 | 0 | 10 | 46 |
| #534 pin | dense B=8 | 1227 | 24 | 0 | 35 | 107 |
| #534 pin | sparse B=8 k=2 | **1302** | 32 | 0 | 52 | **123** |
| #527 bounds | dense B=1 | 516 | 3 | 0 | 7 | 44 |
| #527 bounds | sparse B=1 k=2 | 608 | 8 | 0 | 10 | 72 |
| #527 bounds | dense B=8 | 1227 | 24 | 0 | 35 | 107 |
| #527 bounds | sparse B=8 k=2 | **1942** | 64 | 0 | 52 | 331 |

Per attention plane (`attention_args` over the batch): tiny has one plane;
8 / 50 aten ops at B=1/B=8 on every head, 0 scalar/tolist syncs, 2 / 9
py-built tensors. The plane call itself is lean; the tick-wide excess is
tiering, not scoring.

## What the numbers say

1. **#534 (cross-tick hot pin) removes the whole selection delta at B=8:
   1878 → 1302 aten ops and 331 → 123 H↔D copies, within ~6% of the dense
   1227/107.** On main the sparse-minus-dense excess is demote/promote churn
   — `_page_blob` (D2H) + `promote_keyed` (H2D) fire for every selected page
   every tick even when the selection slides by only one page. Pinning keeps a
   stable selection resident, so a steady tick moves zero pages. The residual
   +75 aten at B=8 is the selection/packing machinery, not data movement.
2. **#527 (contiguous bounds) does not change the tiny tick count (6 pages ≪
   its 128k target); it trades 8 extra `_local_scalar_dense` at B=1 for the
   constant-op guarantee.** Its win is op count staying flat as candidate
   pages grow to ~8192, invisible at this pool — the dedicated dispatch-mode
   gate proves 8 vs 64 candidates dispatch the same number. The extra scalars
   are the validity-mask indexing (`bounds_valid[page]`).
3. **Scalar syncs (`item/bool/float`) scale with batch even after pin (32 at
   B=8)** and are the remaining graph-capture blocker: a replay cannot stall
   for a `.item()`. py-built tensors (52 at B=8) likewise rebuild per replay.

## Top 5 sites (file:function), B=8 — main

Raw aten-op hot spots (include forward work dense also pays):

| ops | site |
|---:|---|
| 430 | `src/tilerl/model.py:_gdn` |
| 240 | `src/tilerl/testing.py:paged_attention` |
| 208 | `src/tilerl/testing.py:paged_attention` (other line) |
| 160 | `src/tilerl/model.py:_base_linear` |
| 112 | `src/tilerl/kv_cache.py:_page_blob` |

Sparse-minus-dense — the host work the selection adds on main, B=8:

| Δ ops | site |
|---:|---|
| +112 | `src/tilerl/kv_cache.py:_page_blob` (D2H K plane) |
| +112 | `src/tilerl/kv_cache.py:_page_blob` (D2H V plane) |
| +96 | `src/tilerl/kv_cache.py:promote_keyed` (H2D K) |
| +96 | `src/tilerl/kv_cache.py:promote_keyed` (H2D V) |
| +56 | `src/tilerl/kv_cache.py:_page_blob` (pinned alloc) |

Sparse-minus-dense after #534, B=8 — the residual to make capturable:

| Δ ops | site |
|---:|---|
| +24 | `src/tilerl/sparse_engine.py:__init__` (own-table build) |
| +24 | `src/tilerl/sparse_engine.py:__init__` (own-table build) |
| +16 | `src/tilerl/kv_cache.py:write_tokens` |
| +16 | `src/tilerl/sparse_engine.py:attention_args` (packed table) |
| +16 | `src/tilerl/sparse_engine.py:attention_args` (packed table) |

## Rule

Measure the host work by differential against dense, not by raw tick count:
the model-forward sites (`_gdn`, `_base_linear`, `paged_attention`) dominate
raw ops but are the same graph-capturable work in both engines. The sparse
blocker is the sparse-MINUS-dense delta — on main it was per-tick page
movement (fixed by pinning a stable selection); after pin it is per-tick
Python table construction and scalar reads, which is the next capturability
unit.

## Results

| date | machine | target | result |
|---|---|---|---|
| 2026-09-12 | Mac CPU (tiny, 6-page prompt, k=2) | cpu | sparse B=8 decode 1878→1302 aten ops / 331→123 H↔D on #534; residual selection delta is Python table build + 32 scalar syncs |
