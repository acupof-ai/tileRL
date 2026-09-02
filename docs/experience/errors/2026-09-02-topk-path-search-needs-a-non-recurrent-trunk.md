# Top-k path search over a draft block is correct, published, and dead on this trunk — 2026-09-02

## Context

Proposal: instead of DFlash's `sample(draft_logits)` — top-1 per block position,
one chain, longest accepted prefix (`dflash.py:466-471`) — keep top-k (k=16) per
position, search the block's logprob matrix, and verify the longest path the
trunk accepts. The draft block is emitted in ONE parallel forward, so the k^L
path space costs no extra draft compute.

The mechanism is real and already published: **DDTree** (Ringel & Romano,
*Accelerating Speculative Decoding with Block Diffusion Draft Trees*) proves an
optimal tree uses only top-K per depth (Lemma 1), builds it with an O(B log B)
best-first heap over log-prob rank tuples, and verifies in one pass with tree
attention.

One of my own claims here was wrong and is worth recording: I estimated k=16 at
L=7 as "112 rows = 14 rung-8 launches". Both bounds are wrong. Trees are
prefix-closed, so a B-node tree costs exactly **B rows**, not k·L (which assumes
zero prefix sharing) and not k^L.

## Why it dies here, three independent reasons

**1. The 48 gated-delta layers have no attention matrix to mask.** Tree
attention is a mask, which works for our 16 full-attention layers. The other 48
are a sequential scan — `s_h = s_h * exp_g[:, step]` then `s_h += kn ⊗ d`,
folded step by step (`reference.py:788-805`). A flattened tree in one row folds
EVERY branch into ONE state trajectory: silently wrong, not slow. Each branch
needs its own trajectory, i.e. its own state. Our chain rewind already concedes
this — `keep_steps` stores a LINE of states, one per chain position, and
`select_step` picks one (`kv_cache.py:327-333`).

Cost per node: 48 layers × 48 value heads × 128 × 128 bf16 = **72 MiB**.
DDTree's useful budgets are 256-512 nodes = 18-36 GiB, on a 31.7 GB card already
holding 20.35 GB of weights. Even B=8 is 576 MiB. **Tree verification on a
hybrid-recurrent trunk is a different problem from tree verification on a pure
attention one, and DDTree's targets (Qwen3-4B/8B/30B-MoE) are all pure
attention.**

**2. Our drafter is not block-parallel, so the free path space does not exist.**
`engine.py:1078` runs `for j in range(1, self._spec_depth)`, feeding step j-1's
SAMPLED token as step j's input — position j's distribution is conditioned on
what was drafted at j-1. There is no matrix of independent per-position
distributions. Getting k candidates per depth from our 1-layer MTP head means
running it k times per depth, which is a tree DRAFT — task #18 verbatim, ladder
problem included. The block-parallel property belongs to the DFlash/DSpark head,
which is a separate architecture we have not integrated.

**3. Rung 8 is a hard ceiling, not a soft one.** `backend.py:513-521` packs X as
f16 only for M ≤ 8; above that the dispatch falls to the M=32 kernel with no
packing (`backend.py:526-534`) at 127 µs/row against rung 8's 22 µs/row. So
width 9 costs ~14.5× width 8, not 2×. We are capped at 8 rows where DDTree wants
256.

## And the prize was measured, not assumed

`scripts/probe_draft_topk.py` ranks each drafted token inside the trunk's own
ordering, 592 verified positions at ctx 512:

| top-1 | top-2 | top-4 | top-8 | top-16 | top-64 |
|---:|---:|---:|---:|---:|---:|
| 96.8% | 96.8% | 96.8% | 97.0% | 97.1% | 99.5% |

Median rank 0, p90 rank 0. **The top-1 → top-16 gap is 0.3 points.** The draft
is either right or wrong past rank 16; the "correct token sits at rank 2-16"
regime that a path search feeds on barely exists. (The absolute 96.8% is
optimistic — positions after a divergence are scored against trunk logits
conditioned on the wrong prefix, and tok/fwd 3.26 implies a true per-position
rate near 0.87 — but top-1 and top-16 are measured the same way, so the 0.3-point
gap stands.)

First run of that probe read 8.5%, which was the instrument, not the model: the
accept test is `got[j] == chains[j+1]` (`engine.py:1128`), and I had compared
trunk position 0 against draft entry 0 — off by one position.

## Rule

Before costing a mechanism, check that its enabling PROPERTY holds on your
stack. DDTree's tree search is correct and its row accounting is better than my
estimate; it is inapplicable because 75% of our layers are recurrent and because
our drafter is autoregressive. Both facts were available before any analysis.

Second: measure the prize before pricing the work. One probe against the
existing head — no integration, no kernel — bounded the whole idea at 0.3 points
and cost one pod run. A reading that contradicts a known end-to-end number by
10× is instrument failure; check the alignment against the actual accept test,
not against what the variable names suggest.
