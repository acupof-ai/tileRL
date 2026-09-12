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

A captured gather cannot fetch a cold page, and the chosen set is only known
mid-forward (Quest scores need the new token's post-rope q), so a pre-forward
`.item()` residency check can neither route before attention nor stay sync
free. Instead the device path is resident **by construction**: it scores only
candidates whose `l2p >= 0` (a device eligibility mask fed to `select_members`),
so every chosen page is resident and the phys gather cannot return -1 — no
host sync, no phantom block 0. A page cold or on the SSD mmap that the Quest
top-k now wants enters the hot set at a **refresh**: every `SPARSE_REFRESH_TICKS`
(default `R=8`) decode ticks — and every prefill chunk — the engine runs the
unchanged eager path, which scores ALL candidates and promotes what it names
through the cross-tick pin. Decode selections drift slowly, so up to R ticks
of selection staleness trades for R-1 captured ticks; `R=1` is every-tick
eager (zero staleness, the token-equality gate). `R` is the quality knob to
measure against the k output-fidelity table (`R=1` vs larger).

`R` is the routing knob, `k` the width knob: `k` decides how many pages a
selection names, `R` how often a device-only tick is allowed to name a
resident page it already held instead of pulling in a newly-cold one.

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
- `test_device_select_excludes_a_cold_candidate_and_eager_promotes_it` — the
  SSD-spill hole: the top-scoring candidate is cold (l2p=-1); the device path
  excludes it (no phantom block 0 in the physical table) while eager refresh
  re-selects and resolves (promotes) it.
- `test_refresh_r1_device_selection_is_token_equal_to_eager_sparse` — R=1
  every tick is an eager full-candidate refresh: zero staleness, tokens equal.
- `test_refresh_r8_routes_device_for_seven_ticks_then_eager_promotes` — R=8:
  seven resident-only device ticks then one eager refresh that promotes a cold
  page on a 24-page context exceeding the hot set.

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
