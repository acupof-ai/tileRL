# Sparse KV selection

A dense checkpoint is converted into one whose full-attention layers attend to
a selected subset of the context. The device then holds the weights, a small
index key per token, and the selected pages; the rest of the KV lives in host
RAM or on the SSD. The mechanism is DeepSeek V3.2's sparse attention (lightning
indexer, top-k selection, sparse attention) placed on the paged KV, the cost
model and the tape this tree already has. Serving and training run the same
selection through the same engine.

## The account on the 27B

16 full-attention layers, 4 KV heads x 256, fp8 KV: `kv_format(256)` gives
33,280 B per token (32 planes x 4 x 256 + 512 B of f32 scales). Index keys per
token: 16 layers x (128 B fp8 + 4 B f32 scale) = 2,112 B, 15.8x smaller; one
key per token per layer, shared by the index query heads as in V3.2. Selection
is top-k **pages**, k_pages = 128 (2048 tokens) per layer per row: 128 pages x
2 planes x 4 x 16 x 260 B x 16 layers = 65 MiB. Selecting top-k tokens instead
would pin between 128 and 2048 pages (up to 1,040 MiB per row), so the unit of
selection is the page and the hot budget is fixed by k_pages.
Weights stay 22.759 GiB (served faces), so the smallest card is 32 GB.

| 256k context | dense fp8 KV on device | sparse: index + hot pages | cold KV (host / SSD) |
|---|---|---|---|
| B=1 | 8.125 GiB | 528 MiB + 65 MiB = 0.58 GiB | 8.125 GiB |
| B=8 | 65.0 GiB (does not fit an H20) | 4.63 GiB | 65.0 GiB |

Derived from `nbytes`; nothing above is measured yet. The dense column is the
P6 ledger's row (`2026-09-11-p6-long-context-budget-on-one-h20.md`).

Per decode step at 256k, B=1: the indexer reads 528 MiB of keys (0.13 ms at
4 TB/s) and 4.3 GFLOP (4 index query heads against the one shared key per
token); the tick's weight read is 22.36 GB (5.6 ms at 4 TB/s), so scoring is
2% of the tick. Fetching hot pages from the host is 65 MiB per row worst case
(1.3 ms at 50 GB/s PCIe); that the per-token delta is a few pages is a
prediction to be measured on the card, not a property of the design. The
bound is in `kernel_cost` as two more rows, priced by the same rule as every
other kernel (bytes per HBM direction crossed, PCIe bytes as their own column).

## Selection is page-granular

`BLOCK_TOKENS = 16`. The indexer scores tokens; the selector max-pools scores
over each page and returns the top-k_pages blocks. The page table gains
one field, `location in {device, host, ssd}`, and the block ids in a
`block_table` row are the selected pages. `paged_attention` does not change:
it receives a block table whose length is the selected set, not the context.
With `k >= context` the block table is the full one and the output equals dense
paged attention; this is the correctness gate.

Cold pages move through the same pinned path `DramSnapshots` uses for state
snapshots, extended from state entries to KV blocks. A page is written once
(prefill or decode append), demoted when it leaves every row's selected set,
promoted when the selector names it. Prefix sharing is unchanged: shared pages
are read-only wherever they live.

Prefill is chunked already; a chunk's queries select from pages written by
earlier chunks, and the chunk's own pages stay on the device until the chunk
ends. The union of a chunk's selected sets is fetched once, not per query.

## Cost model rows

`memory.plan` adds three owners, priced by `nbytes` like every other row:

```
index_keys  device  count = tokens_resident, fmt = Format(bits=8, scales=((128, f32),)), shape [layers, 128]
kv_hot      device  count = rows x k_pages x full-attn layers, per_kv_block_bytes / planes x 2  (k_pages = 128)
kv_cold     host|ssd count = pages_written - pages_on_device, per_kv_block_bytes
```

`kv_pool` keeps its meaning (pages on the device); `kv_hot` is the part of it
the selector pins, so `kv_hot <= kv_pool`. `serve --dry-run --sparse-k K` prints
the table for a context and batch; the gate is the same byte equality the other
rows carry: on the tiny model, derived `index_keys` and `kv_hot` equal the
storage bytes of the allocated tensors.

## Training the indexer

Converting a dense model is two stages, both through the tape, both through
`train --recipe`:

1. **Warm-up.** Every weight frozen, dense attention. The indexer's softmax over
   the context is fit with KL to the dense attention mass summed over heads and
   L1-normalised per query. Only the indexer's parameters carry gradients, so
   the tape holds one small op per full-attention layer; the target is computed
   chunk by chunk from the dense scores the frozen forward already produces.
   V3.2 reports 2.1B tokens for this stage.
2. **Sparse fine-tune.** Selection on, every weight trained, the indexer loss
   restricted to the selected set. The backward is the sparse attention
   backward plus the indexer backward, both in TileLang's
   `examples/dsa_sparse_finetune` (`indexer_bwd.py`, `sparse_mla_bwd.py`).
   V3.2 reports 944B tokens; on one card the stage is what the RL loop already
   does, with selection on.

Acceptance is pre-registered before a card run: top-k recall of dense
attention mass at k=2048 >= 0.9 on 128k held-out prompts after warm-up, and
the P1 eval delta between sparse and dense <= the run-to-run noise measured by
two dense seeds. A miss on recall is a science result (the single-card token
budget was not enough), recorded in `errors/` with the token count.

## Kernels

Copied, not written. `examples/deepseek_v32/` has `fp8_lighting_indexer.py`,
`topk_selector.py`, `sparse_mla_fwd*.py`, `sparse_mla_bwd.py`;
`examples/dsa_sparse_finetune/` has the training pair. Ours is GQA (4 KV heads,
256), not MLA, so the sparse attention kernel is the gather form of the
existing `paged_attention` cell rather than a port of `sparse_mla`: the block
table already gathers pages, and the selected set is a shorter block table.
The indexer and selector are new ops with CPU twins first, under the same
registry rules as every other kernel.

## What does not change

`submit`/`poll`, `StepLimits`, the one-forward-per-tick loop, the captured
decode tick (the selector runs inside it at a fixed k), `PagedKvPool`'s block
API, `paged_attention`'s signature, the prefix store's read-only rule, and the
`peak = static + transient` invariant, which now has three more static rows.

## Ownership

| Unit | Owner | Gate |
|------|-------|------|
| A. `plan` rows `index_keys`/`kv_hot`/`kv_cold`, `--sparse-k` dry-run, `kernel_cost` indexer rows | 52 | derived == storage bytes on tiny |
| B. indexer op + page selector, CPU twins, `k >= context` equals dense | cc | `test_sparse_equals_dense_at_full_k` |
| C. page `location`, demote/promote of KV blocks on `DramSnapshots`' path | 5f | a demoted page promoted reads back byte-equal; decode tokens equal with and without demotion |
| D. tape: indexer KL warm-up recipe, indexer backward, sparse attention backward | 65 | gradcheck on tiny; warm-up recipe runs one step on tiny |
| E. sm90 cell: indexer + selector from `deepseek_v32`, bench rows with `%bound` | after the runbook, cards 0-7 | roofline table in the wins entry |

A-D need no card. Each unit is one PR with a non-author review on goal fit,
entropy (no second KV mechanism, no field without a consumer) and the 27B path.
