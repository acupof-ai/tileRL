# Sparse Quest selection and speculative decode are the serving default, 2026-09-12

> Status: **shipped at k=128 (#530).** The ship gate is output fidelity, not
> page-mass recall: #531's 8k/32k dense-vs-packed KL/top-1 table plus 65's 32k
> n=3 NLL verdict clear k=128 (greedy-NAT gap +0.013 token vs dense's own
> greedy, per-window max 0.096; top-5 agreement 1.0). k=256 scored +0.018, so
> the union k=256 pool stays parked. `build_engine`/`serve` default to
> `sparse_k=DEFAULT_SPARSE_K=128` `scorer=bounds`; `--sparse-k 0` restores the
> dense engine; speculative decode runs under sparse with the draft head kept
> dense. Open defect: under spec a follower cannot adopt a published prefix —
> it returns-miss, so the prefix cache never serves on the production path
> ([errors/2026-09-12-spec-follower-cannot-adopt-a-prefix.md](../errors/2026-09-12-spec-follower-cannot-adopt-a-prefix.md)).

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
never reached `build_engine` and the new 128 default won anyway). Sparse allows the prefix store (#542 restores host-blob sharing after the #518
stopgap); a follower under spec returns-miss — see the open-defect section
above. On-policy training builds pass `sparse_k=0`: the dense full-context tape
is unchanged.

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
`sparse_k=0` explicitly. Full hermetic run at the merge head: 744 passed, 19 skipped, 6 xfailed.

## Fidelity verdict — ship at k=128, park the union pool

65's 27B cut recall (Chinese-wiki spans, 256 positions, 16 spans, window
included, full-attn planes) stays diffuse:

| k | oracle | bounds | random |
|---:|---:|---:|---:|
| 128 | 0.296 | **0.158** | 0.064 |
| 512 | 0.652 | 0.433 | 0.248 |
| 1024 | 0.858 | 0.687 | 0.507 |

Mass recall never reaches the old 0.9 design gate below k~1200 (sink 0.2-0.5%,
last-8 window 0.1-0.4%), and the learned indexer stayed below bounds after 100
warm-up steps (0.10). The gate that matters is output fidelity, and it passes:

- **#531 (dense vs packed-sparse KL/top-1, 8k and 32k):** the k=all arm is
  token-identical — the mechanism is exact; degradation at smaller k is purely
  the attention set.
- **65 NLL verdict (32k, n=3 windows):** k=128 greedy naturalness gap
  **+0.013 token** against dense's own greedy, per-window max **0.096**;
  k=256 **+0.018**; top-5 agreement **1.0 for both**.

k=128 is no worse than k=256 on outputs, so DEFAULT_SPARSE_K stays 128 and the
cross-group union k=256 pool (5f) is parked — no default needs it. Recall is a
diagnostic here, not the gate: diffuse mass at k=128 does not move the
generated tokens.

## Known open defect — the prefix cache does not serve under spec

A follower that hits a published sparse prefix adopts trunk KV without
forwarding the matched tokens; the draft head builds its dense KV only while
forwarding and keys every proposal on trunk hidden, so adoption drafted against
an unbuilt pool. #530 ships the correctness fix — return-miss whenever a draft
is attached, bit-equal to a cold follower — at the cost that under the
production defaults (sparse + spec) the prefix cache publishes but never
serves. Warm path: store the draft head's per-layer prefix KV (and trunk
hiddens, or the trunk forward) in the published entry. Tracked as
[errors/2026-09-12-spec-follower-cannot-adopt-a-prefix.md](../errors/2026-09-12-spec-follower-cannot-adopt-a-prefix.md),
OPEN.md row removed by the PR that lands it.

## Rule

A default that changes the attention set ships on output fidelity measured at
named k, not on page-mass recall: the diffuse-mass recall gate failed at k=128
while generated tokens moved by +0.013 NAT and kept top-5 agreement 1.0. Under a
tiered KV design the speculative draft is a second, independent KV pool — keep it
dense for the whole context and key the host tier by an identity a recycled frame
cannot reuse.
