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
33,280 B per token (32 planes x 4 x 256 + 512 B of f32 scales); one page of
`BLOCK_TOKENS` = 16 is 532,480 B across all layers. Selection is top-k
**pages**, k_pages = 128 (2048 tokens) plus the 8-page window, computed at 4
index source layers and shared by each layer's group of 4, so the hot set of a
row is (128 + 8) pages x 532,480 B = 69.1 MiB. Selecting top-k tokens instead
would pin between 128 and 2048 pages (up to 1,040 MiB per row), so the unit of
selection is the page and the hot budget is fixed by k_pages. Index keys (the
learned indexer, V4.1 form): per page, per source layer, 4 index heads x
`Format(bits=8, scales=((128, f32),))` = 4 x 132 B = 528 B; 4 source layers =
2,112 B per page = 132 B per token, 33.0 MiB at 256k (16,384 pages). The
training-free bounds scorer instead holds 4 KiB per token fp16 (1 GiB at
256k). Weights stay 22.759 GiB (served faces), so the smallest card is 32 GB.

| 256k context | dense fp8 KV on device | sparse, learned indexer: keys + hot pages | sparse, bounds scorer | cold KV (host / SSD) |
|---|---|---|---|---|
| B=1 | 8.125 GiB | 33 MiB + 69 MiB = 0.10 GiB | 1 GiB + 69 MiB = 1.07 GiB | 8.125 GiB |
| B=8 | 65.0 GiB (does not fit an H20) | 0.80 GiB | 8.5 GiB | 65.0 GiB |

Derived from `nbytes`; nothing above is measured yet. The dense column is the
P6 ledger's row (`2026-09-11-p6-long-context-budget-on-one-h20.md`).

Per decode step at 256k, B=1: the learned indexer reads 33 MiB of keys at the
4 source layers (0.01 ms at 4 TB/s) and 0.07 GFLOP; the bounds scorer reads
1 GiB (0.26 ms); the tick's weight read is 22.36 GB (5.6 ms at 4 TB/s), so
scoring is under 5% of the tick either way. Fetching hot pages from the host
is 69 MiB per row worst case (1.4 ms at 50 GB/s PCIe), once per group, not per
layer; that the per-token delta is a few pages is a
prediction to be measured on the card, not a property of the design. The
bound is in `kernel_cost` as two more rows, priced by the same rule as every
other kernel (bytes per HBM direction crossed, PCIe bytes as their own column).

## Two scorers, one selector

The selector consumes per-page scores `[rows, pages]` per layer and returns
the top-k_pages block table. Two scorers produce them:

1. **Page bounds, training-free (day 1).** Per page, per layer, per KV head,
   the elementwise min and max of K over its 16 tokens (Quest). The score is
   the upper bound `sum(max(q*kmin, q*kmax))`. Bytes: 2 x 4 heads x 256 x 16
   layers x 2 B = 64 KiB per page in fp16, 4 KiB per token (2 KiB in fp8),
   written once at append time from the K the pool already holds; no weights,
   no training. This is
   what runs the dense checkpoint at 256k on the V100 without changing it.
2. **Learned indexer, in DeepSeek-V4.1's form (ckl, 2026-09-11).** The unit
   of indexing is the page, not the token: an indexer-K is projected from each
   page's K (V4.1 projects it from the m-token entry; `candidate_block_size`
   8 there, `BLOCK_TOKENS` 16 here), an indexer-Q is projected from the layer
   input H with `ih` index heads of dim `di`, the score of a page is
   `sum_h ReLU(q_h . k_h)` and top-k pages follow. Selection is computed at
   index source layers and reused by the layers after them (V4.1
   `index_source_layer_ids` every 4-6 layers; here 4 sources over the 16
   full-attention layers, groups of 4), so one hot set serves a group and the
   cold-page fetch is paid once per group. The local window is always
   attended: the last `n_win` = 128 tokens (8 pages) join the selected set and
   one softmax runs over [selected pages ; window]. Bytes with `ih` 4, `di`
   128, fp8: 512 B per page per source layer, 4 source layers = 2 KiB per page
   = 128 B per token, 32 MiB at 256k; scoring reads 32 MiB per step. Deferred
   from V4.1: learned entry compression (attention over entries instead of
   tokens), cross-layer KV reuse, the hierarchical 16k candidate pool.

Both keep the same rows, the same tiering and the same `k >= context` gate;
`serve --scorer bounds|index`. The V100 (sm70, 32 GB, f32 IO, eager decode)
is the first card target: weights 22.759 GiB leave ~8 GiB, so dense fp16 KV
stops at 64k tokens for one row; with bounds selection the device holds
1 GiB of bounds plus 128 MiB of fp16 hot pages at 256k, and the cold KV sits
in host RAM (16 GiB per row in fp16) behind PCIe Gen3 (~12 GB/s, so a full
128 MiB refetch is 11 ms; the delta is what the bench must show).

## Measured on the 27B

The account above was a prediction; the numbers below are measured on the
Qwen3.8-27B held Chinese-wiki spans (256 seeded sampled query positions per
span, 2026-09-11/12). Source entries:
[wins/2026-09-11-learned-indexer-on-the-tape.md](experience/wins/2026-09-11-learned-indexer-on-the-tape.md)
(recall, warm-up),
[wins/2026-09-12-indexer-learns-concentrated-teacher-cpu.md](experience/wins/2026-09-12-indexer-learns-concentrated-teacher-cpu.md)
(learnability).

**Recall of dense attention mass vs k (window included).** random / bounds /
oracle top-k over 256 positions:

| ctx | k | random | bounds | oracle |
|---|---:|---:|---:|---:|
| 16384 | 128 | 0.136 | 0.235 | 0.370 |
| 16384 | 256 | 0.252 | 0.400 | 0.571 |
| 16384 | 512 | 0.503 | 0.668 | 0.815 |
| 32768 | 128 | 0.064 | 0.158 | 0.296 |
| 32768 | 256 | 0.133 | 0.263 | 0.455 |
| 32768 | 512 | 0.248 | 0.433 | 0.652 |
| 32768 | 1024 | 0.507 | 0.687 | 0.858 |

At 32k even the oracle top-128 holds only 0.30 of mass and the oracle does not
reach 0.9 until k ≈ 1100–1200; bounds tracks ~0.17–0.22 below the oracle. The
full-attention mass is diffuse — this is the property that sets k, not the
scorer (random/bounds/oracle differ only by a constant gap).

**The 16 full-attn layers are genuinely global.** At sampled queries ≥ 2048,
page-0 sink mass is 0.005 / 0.002 and the last-8-page window is 0.001 / 0.004
of total mass at 16k / 32k. Locality lives in the GDN layers; including the
window in recall adds ~0, so the recall numbers are not a window artifact.

**The warm-up recipe is sound; the 27B teacher is the limit.** The learned
warm-up made 27B recall worse (0.28 → 0.10) toward the diffuse teacher, but a
card-free CPU-tiny control that drives the real page-mass KL, tape backward
and AdamW against a concentrated, exactly-representable fixed teacher fits
and generalizes on held batches across 3 seeds (n=1024 points each): KL
4.02 ≈ ln(56) uniform → 0.73–0.82 at lr 0.02 and → 2.38–2.43 at lr/10,
recall@4 0.08 → 0.18–0.24 both LRs. The loss and heads-pooled target learn;
diffuse mass gives them little to recover. The indexer unit parks until k and
output fidelity settle.

**The SLO changes from mass to output fidelity.** The pre-registered
"recall ≥ 0.9 at k=2048" mass gate is replaced by output fidelity — per-token
KL(dense‖sparse), top-1 agreement and greedy-continuation agreement vs the
dense forward (`scripts/output_fidelity.py`,
`scripts/fidelity_checks.py`): diffuse mass can still yield token-equal
outputs, so mass fraction is the wrong acceptance quantity. The chosen k is
pending the sm70 full-k continuity row (k = pages−8 must be KL ≈ 0 / top1 ≈ 1)
and the multi-span dense-vs-sparse table; no k above 128 is the default yet.

**k > 128 also needs an engine/ledger change, not just a flag.** The hot pool
is sized `slots × (n_groups·k + WINDOW_PAGES + chunk) + 1` — the four source
groups each reserve their own k pages (no cross-group union). Measured on the
V100 (#539 review): k=128 → 1107 blocks / 1.08 GiB; k=1024 → 4137 pages per
slot ≈ 4.0 GiB f32 K (≈ 8 GiB K+V) at one slot and 33,097 blocks ≈ 32 GiB K at
the default 8 slots, which does not fit with the 27B weights on a 32 GB sm70
card. Any k decision above 128 requires the per-tick resident set to hold the
cross-group union (engine change), and the ledger `kv_hot` row must count
`n_groups·k` rather than `k`. See the [#539](https://github.com/acupof-ai/tileRL/pull/539)
review thread.

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

### The resident pool is a cross-group UNION, sized `union_cap`

The hot pool is currently sized for the worst case — groups choosing disjoint
page sets:

```
pages/slot = n_groups * k + WINDOW_PAGES + chunk_pages
```

The residency boundary is already the union (finalize keeps exactly the union
of the groups' picks plus the own span; #534), but the pool pays for `n_groups`
independent k's. On the 27B `n_groups = 4`; at k=1024 that is 4096 + 8 + 33 =
4137 pages/slot. One sm70 f32 block is `2 * 16 * 4 * 16 * 256 * 4` = 2,097,152
B, so this is 8.08 GiB/slot — 64.6 GiB at 8 slots; k=512 is 4.08 GiB/slot
(32.6 GiB at 8). If the 8k fidelity row says the model needs that k, the
per-group sizing makes sparse unusable at batch even though the four groups'
top-k sets overlap heavily in practice.

The pool becomes one shared per-slot pool of resident pages sized

```
union_cap = k + WINDOW_PAGES + chunk_pages + h_pages(overlap headroom)
```

where `h_pages` is fixed from measurement, not guessed: cc's 32k/8k runs log
the per-tick union size `|∪_g S_g ∪ own|` (a one-counter probe alongside the
fidelity runs), and `h_pages` is the observed high-water minus k over the run
plus a named slack. `union_cap = n_groups*k + ...` must remain a legal setting
and reproduce current outputs exactly (the continuity gate).

**Eviction when the union exceeds `union_cap`.** Today the within-tick victim
guarantees every reserved pick a frame (the "no unreserved victim" error) — that
guarantee cannot hold below `n_groups*k`. Picks then contest the cap by one
rule: order every `(group, page)` pick by how marginal it is to its OWN group —
its score gap over that group's k-th pick — and drop the smallest-gap picks
until the union fits. The forced window and the own span never compete. A group
that loses a pick attends to its remaining picks, i.e. it reads as that group
running a smaller k for the tick; the rule is max-min on within-group rank, so
no group loses two picks while another keeps a pick more marginal to it. This
is deterministic and score-only, so it runs in the device path: eligibility is
the current `l2p >= 0` mask AND "the page survives the marginality clipping",
computed from the same batched scores with no new host sync. Demotion still
goes through the one-batch D2H context.

**Ledger.** The `kv_hot` derived row changes from `n_groups*k + W + chunk` to
`union_cap` per slot (the allocated ceiling, as today), and stats gains one
measured counter, the tick union size, so the bench prints ceiling vs held
distribution. The eager refresh promotes the highest-aggregate-score missing
pages up to free union slots.

**Gates (CPU tiny):** (1) `h_pages` large enough that the union never clips —
token-identical to the current per-group sizing; (2) `union_cap = k` with
fixtures giving disjoint group preferences clips exactly the named
lowest-marginal picks, symmetric across groups, and tokens equal an oracle
that clips each group's list the same way; (3) ledger derived == measured at a
fixed `union_cap`; (4) full-k (`k >= pages`) still equals dense. Implementation
lands only after the 8k row fixes the k this serves.

## Cost model rows

`memory.plan` adds three owners, priced by `nbytes` like every other row:

```
index_keys  device  count = pages_resident x 4 source layers x 4 heads, fmt = Format(bits=8, scales=((128, f32),)), shape [128]   (learned indexer; bounds scorer: pages x 16 layers x [2, 4, 256] bf16)
kv_hot      device  count = rows x (k_pages + 8 window) x 4 groups, bytes = per_kv_block_bytes / 4  (a group is 4 of the 16 layers' planes)
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
   pages is fit with KL to the dense attention mass summed over heads and
   pooled per page, L1-normalised per query. Only the indexer's parameters
   carry gradients, so the tape holds one small op per source layer; the
   target is computed chunk by chunk from the dense scores the frozen forward
   already produces. V3.2 reports 2.1B tokens for this stage; V4.1's report
   does not state how its indexer learns, so a straight-through softmax on the
   selection (aupai's CSA2 choice) is the alternative to A/B against KL.
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

> 2026-09-12: the 27B measurements below supersede this mass gate — the SLO is
> now output fidelity vs dense. The default k is undecided; k=128 remains the
> shipped value until the fidelity table lands. See
> [Measured on the 27B](#measured-on-the-27b).

## Kernels

Copied, not written. `examples/deepseek_v32/` has `fp8_lighting_indexer.py`,
`topk_selector.py`, `sparse_mla_fwd*.py`, `sparse_mla_bwd.py`;
`examples/dsa_sparse_finetune/` has the training pair. Ours is GQA (4 KV heads,
256), not MLA, so the sparse attention kernel is the gather form of the
existing `paged_attention` cell rather than a port of `sparse_mla`: the block
table already gathers pages, and the selected set is a shorter block table.
The indexer and selector are new ops with CPU twins first, under the same
registry rules as every other kernel.

## Prefix sharing under sparsity (stopgap, then the host-tier publish)

The first sparse engine ships with prefix caching **off for sparse builds**
(`build_engine(sparse_k>0)` forces `NoPrefixStore`). The dense publish path
hands `req.blocks[:length//16]` — live device blocks — to `PrefixStore.insert`,
but sparse finalize demotes every private page to the host tier each tick, so
at a publish boundary that slice is empty and the store raises
("64 tokens need 4 blocks, got 0"). Building the default store anyway crashed
the first request whose prompt crossed one interior chunk boundary (≥ 64
tokens), which is why the refusal is enforced in `build_engine`, not left as a
documentation warning.

The upgrade that restores prefix sharing for sparse builds:

1. **Publish at the boundary after demotion, into a shared namespace.**
   Finalize demotes the chunk-boundary pages, then publishes them; a
   published entry names host-held pages, not device blocks. Private cold
   blobs are keyed `(req_id, logical_page)` via
   `demote_page(block, key=…)` (#528) — the physical frame is freed and
   recycled, so there is no stable block id to name. That key is
   request-private and cannot identify a prefix shared *across requests*,
   so publishing does not hand over the private blob: it clone-holds the
   page in a SECOND, req-independent namespace keyed by the page's content
   hash (`HostKvPages.share_hold`). The choice of namespace is the load-
   bearing one — a store-owned prefix id or a content/logical key works;
   the content hash needs no id allocation and deduplicates equal pages
   for free.
2. **Store entries carry the bounds snapshot with the pages.** The
   selector's `SparseTracker` bounds are request-private in the first cut;
   a prefix entry must adopt the published pages' Quest bounds (device-
   resident, small — the `page_bounds` cost row: 64 KiB/page on the 27B)
   exactly as a dense entry carries the GDN state snapshot. A hit then
   restores bounds and shared cold blobs with zero host copies: the page
   stays cold until the selector names it, and promotion copies the
   read-only shared blob into a private fresh block through
   `promote_keyed((rid, page))` for the request's PRIVATE copy (the
   #500 `promote_page` phys-keyed seam is not this path).
3. **Hit adoption promotes lazily, reconciling the two keys.** `_admit`'s
   retain-and-refcount path assumes live device blocks; the sparse hit
   adopts bounds + the shared content keys without promotion, records each
   page's content key for the request, and `_sparse_resolve` looks the
   selected page up by that shared key, copies it into a private block
   keyed `(rid, page)`, and unlinks the page from the shared entry —
   first use does the shared→private key reconciliation. The store's
   read-only rule is unchanged.
4. **Eviction is three-way coherent.** Dropping a prefix entry must
   release the shared blob references (a page shared with a surviving
   entry keeps its blob), the bounds, and any device promotion together;
   the shared blob is refcounted for this, and the host tier's
   `share_release`/`forget` are the two host handles.

Until this lands, sparse serving pays a full prefill per distinct prompt; the
dense engine keeps prefix caching unchanged.

## What does not change

`submit`/`poll`, `StepLimits`, the one-forward-per-tick loop, the captured
decode tick (the selector runs inside it at a fixed k), `PagedKvPool`'s block
API, `paged_attention`'s signature, the prefix store's read-only rule, and the
`peak = static + transient` invariant, which now has three more static rows.
The one first-cut exception is the bullet above: a sparse build runs with
`NoPrefixStore` until host-tier publishing lands.

## Ownership

| Unit | Owner | Gate |
|------|-------|------|
| A. `plan` rows `index_keys`/`kv_hot`/`kv_cold`, `--sparse-k` dry-run, `kernel_cost` indexer rows | 52 | derived == storage bytes on tiny |
| B. page-bounds scorer + selector, CPU twins, `k >= context` equals dense; then the sm70 cell and the V100 256k bench | cc | `test_sparse_equals_dense_at_full_k`; V100 tokens/s + device bytes table at 128k/256k |
| C. page `location`, demote/promote of KV blocks on `DramSnapshots`' path | 5f | a demoted page promoted reads back byte-equal; decode tokens equal with and without demotion |
| D. learned indexer in V4.1 form (page keys, source layers, window), KL warm-up recipe, indexer backward, sparse attention backward | 65 | gradcheck on tiny; warm-up recipe runs one step on tiny |
| E. sm90 cell: indexer + selector from `deepseek_v32`, bench rows with `%bound` | after the runbook, cards 0-7 | roofline table in the wins entry |

A-D start now in parallel and need no card; B's V100 half follows its CPU half. Each unit is one PR with a non-author review on goal fit,
entropy (no second KV mechanism, no field without a consumer) and the 27B path.
