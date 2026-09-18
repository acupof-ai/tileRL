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
dense forward (`scripts/fidelity_engine.py`): diffuse mass can still yield token-equal
outputs, so mass fraction is the wrong acceptance quantity. The sm70 full-k
continuity row landed in #546 (k=all is token-identical: KV writers had ignored
`page_base`); the remaining pending item is the multi-span dense-vs-sparse
fidelity table. No k above 128 is the default yet.

**k > 128 also needs an engine change, not just a flag.** The hot pool
is sized `slots × (n_groups·k + WINDOW_PAGES + chunk) + 1` and `memory.sparse_rows`
prices that allocated pool exactly (the ledger half landed; see "Cost model
rows"). Measured on the V100 (#539 review): k=128 → 1107 blocks / 1.08 GiB;
k=1024 → 4137 pages per slot ≈ 4.0 GiB f32 K (≈ 8 GiB K+V) at one slot and
33,097 blocks ≈ 32 GiB K at the default 8 slots, which does not fit with the
27B weights on a 32 GB sm70 card. What remains for k above 128 is only the
per-tick cross-group UNION engine change (`union_cap`, parked — see below).
See the [#539](https://github.com/acupof-ai/tileRL/pull/539) review thread.

## Hybrid dense/sparse serve (#586)

`--sparse-min-tokens N` runs prompts up to N tokens **dense** on the captured
decode graph: the whole context is pinned in the device pool and no sparse
sharing applies. Longer prompts run the sparse path (eager ticks, this
document). A prompt whose dense pin would not fit the device pool even empty is
rerouted sparse in `submit` instead of queueing.

The two regimes share one engine, so dense admit reserves the sparse hot
ceiling (`memory.sparse_rows`'s `kv_hot`): without that reservation a sparse
row admitted alongside dense ones raises "hot pool undersized" when the
selector's victim search cannot find a frame. `--sparse-prefill-tokens` bounds
one sparse prefill tick so a dense request's wait on the mixed engine stays
near one second.

## Selection is page-granular

`BLOCK_TOKENS = 16`. The indexer scores tokens; the selector max-pools scores
over each page and returns the top-k_pages blocks. The page table gains
one accessor, `page_location(...) -> {device, host, free}`, and the block ids in a
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

**Ledger.** The plan row already equals the pool ceiling: `memory.sparse_rows`
prices the allocated pool — `num_slots × (n_groups·k + WINDOW_PAGES + chunk)
+ 1` spare, matching `sparse_pool_num_blocks`. The live engine row
(`_memory_rows`) stays different on purpose: per-tick residency
(`sum(len(r.blocks))` / `_measured_peak` on current residents), the HELD set
this tick, not allocated capacity — never assert the two are equal.

Introducing `union_cap` keeps that split: the plan row prices `union_cap` per
slot (the union, or the worst case `n_groups*k + W + chunk` that reproduces
today's pool), while stats gains one measured counter, the tick union size, so
the bench prints the planned ceiling vs held-union distribution. The eager
refresh promotes the highest-aggregate-score missing pages up to free union
slots.

**Gates (CPU tiny):** (1) `h_pages` large enough that the union never clips —
token-identical to the current per-group sizing; (2) `union_cap = k` with
fixtures giving disjoint group preferences clips exactly the named
lowest-marginal picks, symmetric across groups, and tokens equal an oracle
that clips each group's list the same way; (3) plan `kv_hot` == allocated pool
bytes at a fixed `union_cap` (the CEILING), separately from the live row which
equals measured tick residency (the HELD set) — never assert those two equal;
(4) full-k (`k >= pages`) still equals dense. **Parked 2026-09-12:** the 8k/32k
fidelity rows kept the default at k=128, where k=128 is no worse than k=256 on
generated tokens, so the worst-case `n_groups*k` pool already fits and no
default needs `union_cap`. Revive when a measured k above ~128 is chosen.

## Cost model rows

`memory.plan` adds three owners, priced by `nbytes` like every other row:

```
index_keys  device  count = pages_resident x 4 source layers x 4 heads, fmt = Format(bits=8, scales=((128, f32),)), shape [128]   (learned indexer; bounds scorer: pages x 16 layers x [2, 4, 256] bf16)
kv_hot      device  count = num_slots × (n_groups·k_pages + 8 window + chunk) + 1 spare — the pool build_engine allocates (worst-case disjoint groups), bytes = one whole KV block per hot page   (the parked union_cap change replaces n_groups·k with the measured union)
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

> 2026-09-12 (updated after the same-day revert #558): sparse Quest selection
> is **opt-in, not the serving default** (`DEFAULT_SPARSE_K = 0`). It shipped as
> default in #530 on short-context sm70 continuity rows and was reverted when
> end-to-end MMLU on the 27B H20 exposed a generation defect those rows did not
> cover: greedy spec-off sparse k=128 scored **0.3475 vs dense 0.9150** on the
> same 400 thinking questions (spec-on n=1282: 0.202 vs 0.859, paired delta
> −0.624 ±0.045). At MMLU lengths k=128 selects the full page set, so the
> divergence is not expected near-tie selection loss; sm70 first-token logits
> are bit-identical dense-vs-sparse at B=1 and B=8 (max abs 0.0), localizing
> the defect to the **sm90 B=8 long-generation** path under bisection
> (`fuse_projections` / host-cold f16-narrow / B=8 kwargs). The sm70 rows below
> remain valid but only for short, B=1/forced-teacher conditions: after #546
> k=all is token-identical, k=128 prefill KL 0.0023/top-1 0.981 at 8k and
> 0.019/0.949 at 32k, and a 64-token teacher-forced NLL gap was ~+0.013
> nats/token (n=3). None of those predicts free-running sm90 B=8 generation.
> The cross-group union hot pool
> [below](#the-resident-pool-is-a-cross-group-union-sized-union_cap) stays
> **parked**. Re-enable a sparse default only after an sm90 B=8 long-generation
> continuity gate passes. See [Measured on the 27B](#measured-on-the-27b).

## Kernels

Copied, not written. `examples/deepseek_v32/` has `fp8_lighting_indexer.py`,
`topk_selector.py`, `sparse_mla_fwd*.py`, `sparse_mla_bwd.py`;
`examples/dsa_sparse_finetune/` has the training pair. Ours is GQA (4 KV heads,
256), not MLA, so the sparse attention kernel is the gather form of the
existing `paged_attention` cell rather than a port of `sparse_mla`: the block
table already gathers pages, and the selected set is a shorter block table.
The indexer and selector are new ops with CPU twins first, under the same
registry rules as every other kernel.

## Prefix sharing under sparsity (shipped in #526)

Sparse builds share prefixes through `SparsePrefixCache`, a host-blob-backed
index wired in `build_engine` when the sparse tracker has sharing enabled. It
exists because sparse finalize demotes private pages to the host each tick, so
the dense `PrefixStore` (which retains live device blocks) cannot serve sparse:

1. **Content-hash namespace.** A published page names its shared host blob by
   the page content hash (`HostKvPages.share_hold`), keyed independently of the
   request-private `(req_id, logical_page)` demotion key; no id allocation and
   equal pages dedupe for free.
2. **Entries carry bounds and a GDN snapshot.** A hit adopts the published
   pages' Quest bounds exactly as a dense hit adopts a state snapshot, zero
   recompute.
3. **Lazy promote-on-select.** Adopt adopts the shared keys without promotion;
   `resolve` copies a selected page into a private block via
   `promote_keyed((rid, page))` the first time the selector names it, reconciling
   the shared and private keys there.
4. **Three-way eviction.** Dropping an entry releases the shared blob refs
   (`share_release`/`forget`, refcounted so a page shared with a surviving entry
   stays), the bounds and any device promotion together.

Only a contiguous longest-frontier prefix is published (dropped pages buffer
until 0..m-1 have all left the resident union and a state snapshot at m exists).
The store is read-only in the same sense as the dense store. `NoPrefixStore` is
now an explicit opt-out only (`sharing_enabled=False`; the RL path), no longer
forced for sparse builds.

## What does not change

`submit`/`poll`, `StepLimits`, the one-forward-per-tick loop, the captured
dense decode tick (the sparse long-prompt path is eager; the selector runs
inside the tick at a fixed k), `PagedKvPool`'s block
API, `paged_attention`'s signature, the prefix store's read-only rule, and the
`peak = static + transient` invariant, which now has three more static rows.

## Ownership

Units A–D started in parallel with no card needed; B's V100 half followed its
CPU half. That work is complete (the historical A–E table is in
[history/ownership-tables.md](history/ownership-tables.md)).
