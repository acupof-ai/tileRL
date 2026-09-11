# Sparse-KV ledger: index keys, hot pages, cold pages priced from nbytes — CPU, 2026-09-11

> Status: **pending-remote** — the plan rows, dry-run and kernel rows are
> derived on CPU and equal the real fp8 tensor storage on the 27B geometry;
> nothing has run on a card and no sparse pool exists yet.

## Context

Sparse-KV unit A ([design](../design-sparse-kv.md)): the memory ledger has to
price a sparse deployment before the selection kernels land. The device holds
the weights, a small learned index key per page, and the selected hot pages;
the rest of the KV sits on the host. Three new owners enter `memory.plan` and
two selection kernels enter `kernel_cost`, priced by the same `nbytes` ledger
as every other row — no profiled number.

## What changed

`memory.plan(..., sparse={'num_rows','context','k_pages','scorer'})` adds:

- **index_keys** (learned scorer): `pages x 4 source layers x 4 index heads`
  keys, each a `[di=128]` fp8 tensor with one f32 per-128 scale =
  `nbytes(Format(8, scales=((128,f32),)), (128,))` = 132 B. On the 27B at 256k
  (16,384 pages): 2,112 B/page = 132 B/token = **33.0 MiB**
  (34,603,008 B). `--scorer bounds` instead prices Quest page bounds
  (`2 x layers x heads x head_dim` fp16 = 64 KiB/page, 4 KiB/token).
- **kv_hot**: each row pins `k_pages` (128) selected pages plus the always
  attended 8-page window = 136 pages. A source group reuses one selection for
  its four full-attn layers, and the four groups cover the layer set, so the
  per-row hot bytes are `136 x one whole fp8 KV block` = **69.06 MiB**
  (72,417,280 B; block 532,480 B).
- **kv_cold** (host tier): every written page the hot set does not pin.
  `hot + cold per row == pages x one block` — the dense total is conserved.

Sparse mode emits these rows **instead of** the dense `kv_pool`, so the pinned
hot subset is not counted twice. The device-peak residual
(`peak = Σ static + transient`) subtracts only device-tier static rows:
`kv_cold` and the existing host-resident ISO frames are not on the card and
must not come out of the device peak.

`serve --dry-run --checkpoint DIR --sparse-k K --scorer index|bounds
--max-ctx CTX --max-batch B` prints the sparse ledger from safetensors headers
(no engine). The built-engine `--dry-run` refuses `--sparse-k`: it builds a
dense pool whose measured residency cannot reconcile with sparse rows, so
sparse is derived-only until a sparse pool exists.

`kernel_cost.sparse_indexer_rows` prices the selection itself on the decode
tick: one `sparse_indexer_score` launch reads all four sources' keys
(**33 MiB** HBM, 2·pages·src·ih·di = **0.067 GFLOP** at 256k) and
`sparse_cold_fetch` carries the 69.06 MiB worst-case refetch across **PCIe on
a separate `pcie_bytes` field**, once per source group — never mixed into the
HBM byte bound (design: ~0.01 ms score, 1.4 ms fetch worst case at 50 GB/s;
the real per-token delta is a card measurement, not a property of the design).

## Gates

- `test_sparse_rows_match_real_tensor_storage_on_27b`: the derived
  index_keys bytes equal the storage of the actual `[pages,4,4,128]` fp8
  payload plus its f32 scales (34,603,008 B); kv_hot = 136·532,480;
  hot + cold == pages·block.
- `test_sparse_bounds_scorer_uses_quest_bounds_bytes`: the bounds scorer
  prices the Quest tensor, not the index keys.
- `test_sparse_dry_run_checkpoint_prints_three_rows`: the tiny header dry-run
  emits the three owners with correct tiers and no dense kv_pool.
- `test_sparse_indexer_rows_match_design_account_on_27b`: the kernel rows
  reproduce 33 MiB / 0.067 GFLOP and the 69.06 MiB once-per-group PCIe fetch.

Full hermetic 660 passed / 19 skipped / 6 xfailed; ruff clean.

## Rule

A derived device ledger is trustworthy only to the exact byte: every sparse
row is priced by `nbytes` at a named shape and gated against the storage of
the tensor the engine will allocate. HBM and PCIe bytes are separate columns;
a host-tier owner never enters the device-peak residual.

## Results

| date | commit | machine | target | result |
|---|---|---|---|---|
| 2026-09-11 | (PR head) | Mac CPU | cpu f32 | 33.0 MiB keys / 69.06 MiB hot derived == fp8 storage; 4 sparse gates; 660 pass |

Raw: `src/tilerl/memory.py` (`sparse_rows`), `src/tilerl/kernel_cost.py`
(`sparse_indexer_rows`), `tests/test_memory_ledger.py`,
`tests/test_kernel_cost.py`; 27B card residency and the sparse attention
kernel are later units, `pending-remote`.
