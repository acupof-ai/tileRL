# The sparse decode tick is device-tensor selection with a fixed-width table — 2026-09-12

> Status: **CPU gates green; CUDA-graph replay of the sparse tick is
> pending-remote.** This PR makes selection itself graph-capturable (no host
> sync, one fixed shape); wrapping it in a `torch.cuda.CUDAGraph` and measuring
> the decode ms is the sm90 card follow-up.

## Context

#523 measured dense decode at 47.99 ms eager vs 11.68 ms captured (H20, B=1).
The cross-tick hot pin (#534) proved residency is settled (0 promotions, ~0.1
demotions/tick) yet sparse decode was still 45.9 ms — that is dense-EAGER
parity, not a sparsity regression. The residual is the missing graph, not page
scoring (Quest is ~0.07 GFLOP/tick) or fetch. But the sparse tick could not be
captured: `SparseForward.attention_args` built the packed block table with a
per-row Python loop — `.tolist()` on the selection, `torch.tensor([...])` per
row, `int()` seq_len — and the table width changed every tick. Every one of
those is a host sync or a data-dependent shape inside capture.

## What worked

Selection split into three pure device-tensor stages, batched over B rows:

- `quest_scores_batched(q[B,...], bounds[B,Cp,...]) -> [B,Cp]` — the same
  chunked Quest score, the page split commuting over rows exactly as over
  pages (bit-identical to the single-row form).
- `select_members` — top-k UNION forced-window membership as a bool
  `[B,Cp]` (topk + scatter + masked window OR; no host branch).
- `order_members` — compaction to a FIXED width k in sequence order via one
  stable `torch.sort` (members before padding, ties keep candidate order).
  `argsort` has a static `[B,Cp]` output shape; `nonzero`/boolean-index does
  not, so it was the wrong compaction under capture.

Chosen LOGICAL pages map to physical blocks through a new device twin of the
resident dict: `SparseTracker.l2p_t[rid] = [cap] long, -1 if not resident`,
mirrored one scalar write per resolve/evict. The packed table is
`[B, k + own_window]` fixed for the bucket — selected pages in the compact
leading columns, each row's own scattered at its device-computed `nsel`
offset, `seq_len = nsel*16 + own_len` a device tensor. Padding columns are 0
and seq_len already excludes them, so the kernel reads exactly the live
columns.

The eager path is unchanged and still owns PROMOTION ticks: a captured gather
cannot fetch a cold page. The engine routes a tick to the device path only in
the pin steady state; residency is checked on the host before capture
(`all_chosen_resident`), so the tensor path itself holds no `.item()` guard
(a guard there would be the same host sync it exists to remove).

## Gates

- `test_device_select_packed_table_matches_eager_at_b1_and_b8` — the device
  table's leading compact physical columns and seq_len equal the eager packed
  table at B=1 and B=8.
- `test_device_select_packs_a_fixed_width_table_with_no_host_sync_at_b1_and_b8`
  — table is exactly `[B, k+own_window]`, and a TorchDispatchMode asserts no
  `aten._local_scalar_dense` (what `.item()`/`bool(tensor)` lower to) runs in
  attention_args. Negative control: the same guard trips on the eager path.
- `test_device_select_engine_tokens_equal_eager_sparse_at_full_k` — through
  the real Engine, device-selection decode is token-for-token equal to eager
  sparse at full k.
- `test_quest_scores_batched_matches_single_row_bit_for_bit`.

## Rule

A captured CUDA tick may contain no host-readable scalar and no data-dependent
shape: topk + stable sort + a fixed output width replace `.tolist()`/`nonzero`,
and a Python-side residency check replaces the tempting in-kernel `.item()`
assert. Physical mapping has to exist as a device tensor before selection can
run without syncing.

## Results

| date | commit | machine | target | result |
|---|---|---|---|---|
| 2026-09-12 | (PR head) | Mac CPU | cpu f32 | device-selection packed table bit/token-equal to eager sparse at B=1/B=8; zero host scalar syncs; fixed k+own width; 12/12 sparse-engine gates |

Raw artifacts: `src/tilerl/sparse_engine.py`, `src/tilerl/engine.py`,
`tests/test_sparse_engine.py`. CUDAGraph capture and the sparse-vs-dense
captured decode ms on sm90 pending a free card.
