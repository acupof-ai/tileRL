# Sparse Quest selection and speculative decode are the serving default — default flip held back on a recall FAIL, 2026-09-12

> Status: **default flipped in code, PR held draft.** `build_engine`/`serve`
> default to `sparse_k=DEFAULT_SPARSE_K` (128) `scorer=bounds`; `--sparse-k 0`
> restores the dense engine. Speculative decode runs under sparse with the draft
> head kept dense. The flip does not ship to the 27B until the output-fidelity
> table clears it: the bounds scorer's window-included recall is far below the
> 0.9 design gate at k=128.

## Context

Units A–F shipped sparse KV as an opt-in (`build_engine(sparse_k=…)`). The order
is to make sparse Quest selection the serving default, with `--sparse-k 0` the
escape hatch, and to run the spec-decode verify tick under it. Two facts shape
the cut:

- the sparse device pool holds only `K + 8-window + one chunk` pages per slot, so
  it cannot back a dense context;
- a verify tick carries the W draft query positions, which can cross a page
  boundary the dense-only sparse path never had to cover.

## What the flip changes

**Defaults.** `sparse_index.DEFAULT_SPARSE_K` (128) is the single named constant
both `build_engine` and the serve CLI use; the serve `--scorer` default flips
`index → bounds` (the learned indexer is not wired in the engine yet). `--sparse-k
0` always forwards (the CLI previously only forwarded sparse_k when truthy, so 0
never reached `build_engine` and the new 128 default won anyway). The serve path
forces `NoPrefixStore` under sparse — there is no prefix cache by default (5f's
#526 adds a host-blob prefix cache and removes this coercion). On-policy training
builds pass `sparse_k=0`: the dense full-context tape is unchanged.

**Spec under sparse — the draft stays dense.** The draft head needs every page of
the row's context, not the hot set, so it keeps a DENSE `PagedKvPool` of its own.
A sparse request carries `req.draft_blocks` (its dense block ids in the draft
pool's id space), separate from `req.blocks` (the trunk's live hot pages, emptied
every tick by demotion). `DraftHead.step` reads `draft_blocks` under sparse and
`blocks` in dense mode. The draft pool is sized dense per slot (fitted to
post-hot-pool free memory on a card, full per-slot ceiling on the CPU cell) and
appears as its own `draft_pool` ledger row. The verify tick's page scoring is the
joint MAX over the W queries — a page hot for ANY chain position is selectable.

**Verify geometry.** The decode own-span computation extended from `q_hi=seq_len`
to `q_hi=seq_len-1+tq`, so a chain whose draft positions cross into the next page
allocates that own page; without it the write landed outside the own table.

**Graph-safe packed table.** `SparseForward` now owns ONE fixed-width table built
at construction and refilled each tick, instead of allocating a width-`max` tensor
per call. The width is constant per verify width — `K + 8` for a plain decode,
`K + 9` for a verify row whose chain crosses one page boundary (W−1 ≤ 15, so it
can cross at most one). Data changes tick to tick; the shape does not, which is
what a captured verify graph requires. Sparse capture itself stays eager in this
cut; the fixed shape is the prerequisite.

**Recycled-physical-id host collision.** The cold tier used to key a demoted blob
by its physical block id. A verify tick promotes a page (freeing its frame), a
later chunk re-allocates that same LIFO frame for a different logical page, and
re-demoting it hit `HostKvPages.hold()` as a duplicate and dropped the blob
("host tier dropped block N"). The fix keys the AUTOMATIC sparse path's blobs by
the immutable `(req_id, logical page)`: `demote_page(phys, key=…)`,
`promote_keyed(key)`, `forget(key)`. The #500 manual `sparse_retier` seam keeps
physical keys (one demote-all/promote-all cycle, no recycle) and is unchanged.

## Gates (tiny CPU cell)

1. `sparse_k=2 + draft`, greedy == `sparse_k=2` no draft — exact-acceptance:
   greedy spec emits the same tokens as greedy plain decode under the same sparse
   selection.
2. `k ≥ pages + draft` == a genuinely dense engine + draft, token for token.
3. structural: over 10- and 16-page contexts the decode packed-table width is one
   context-independent constant per query width (`K+8`, or `K+9` at a boundary).
The pre-existing F gates (full-k == dense, every-tick demote/promote, default
PrefixStore coercion) and the #500 tier seam tests stay green; the dense feature
suite (prefix cache, decode graph, fp8, SSD, training guards) now passes
`sparse_k=0` explicitly. Full hermetic run: 702 passed.

## Why the flip is NOT shipped yet — the recall FAIL

65's 27B cut recall (Chinese-wiki spans, 256 positions, 16 spans, window included,
full-attn planes) at k=128+8:

| k | oracle | bounds | random |
|---:|---:|---:|---:|
| 128 | 0.296 | **0.158** | 0.064 |
| 512 | 0.652 | 0.433 | 0.248 |
| 1024 | 0.858 | 0.687 | 0.507 |

The gate was ≥0.9 mass; even the oracle does not reach it below k≈1200 because the
full-attn mass is diffuse (sink 0.2–0.5%, last-8 window 0.1–0.4%). The learned
indexer is also below bounds after 100 warm-up steps (0.10; its KL collapsed
toward uniform). The shipped k therefore waits on the output-fidelity-vs-k table
(KL / top-1 agreement at k∈{128,256,512,1024}); `DEFAULT_SPARSE_K` moves to
whatever k that table picks. Flipping the default at recall this low would change
served outputs with no fidelity bound.

## Production-path output fidelity (V100, cc, 2026-09-12)

The first end-to-end sparse-vs-dense OUTPUT row on the real V100 path (32k
context, k=128): token-distribution KL **1.05**, top-1 agreement **0.55** — sparse
agrees with dense on only ~55% of argmax tokens at KL far above a serviceable
band. This is output fidelity, not page-mass recall, and it is the binding
gate. An 8k run at k=all / 256 / 128 is queued: k=all must be token-identical
(mechanism control), then the 256/128 degradation separates "diffuse attention,
need larger k" from an engine selection bug. The default stays draft until that
row lands, and this entry cites whichever way it lands.

## Rule

A default that changes the attention set must not ship on a selection scorer that
fails its own recall gate, however clean the mechanism: sparse can be the default
in code and stay a draft PR until output fidelity clears at a named k. Under a
tiered KV design the speculative draft is a second, independent KV pool — keep it
dense for the whole context and key the host tier by an identity a recycled frame
cannot reuse.
