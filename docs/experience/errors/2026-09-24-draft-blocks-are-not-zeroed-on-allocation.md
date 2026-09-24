# Draft pool blocks carry the previous row's data, and warm_draft zeroes one position — V100 sm70 / CPU tiny, 2026-09-24

> Status: open, **docstring only** — measured NOT to change any token, so the
> fix is to correct `warm_draft`'s bit-equality claim rather than to zero the
> blocks. Separate from, and not the cause of, the boundary-hidden defect in
> [2026-09-24-draft-hit-conditions-on-the-wrong-hidden.md](2026-09-24-draft-hit-conditions-on-the-wrong-hidden.md).

## What happens

`PagedKvPool.alloc_block` is `self._free.pop()` with no zeroing
(`kv_cache.py`). The draft pool is allocated the same way
(`engine.py`, `req.draft_blocks = [self._draft.kv.alloc_block() ...]`), so a row
that has not yet written a draft position reads whatever the previous tenant of
that block left there.

Measured on CPU tiny with **two different prompts and no prefix ever adopted** —
so this is not a prefix-cache effect:

| tail draft block | P2 on a fresh engine vs P2 after P1 |
|---|---|
| 23 | 0.0 |
| 24 | 2.984375 |
| 25 | 2.546875 |

The same prompt, the same request, different tokens' worth of state depending on
what ran before.

## Why `warm_draft` does not cover it

`SparseRuntime.warm_draft` restores the publisher's draft KV for the matched
pages and then zeroes **one position**:

```
boundary_blk = r.draft_blocks[matched // BLOCK_TOKENS]
dpool.k_pool[:, boundary_blk, :, 0, :].zero_()
dpool.v_pool[:, boundary_blk, :, 0, :].zero_()
```

That models a cold follower as "zeroed position 0 in block 0". A cold follower
is only zeroed at position 0 because a **fresh** pool happens to be zero — once
the pool has been used, its blocks are recycled without clearing, so the cold
follower's tail positions are stale too and the warm one's are not. The
docstring's claim that the two are bit-equal holds only on a pristine pool.

## Answered: draft attention does not read the stale positions

94's criterion for whether this is a real defect or a wrong docstring: does the
draft attention read any **unmasked** slot that a previous row left dirty?

**It does not.** Measured on CPU tiny: poison every draft slot PAST `draft_pos`
(the first uncommitted position — 64 slots over one run) with `999.0` / `-999.0`
and the emitted tokens are **byte-identical** to the unpoisoned run. Those slots
are causally masked out of the attention and never read.

The reason is in the code: `RefBackend.paged_attention` builds an explicit causal
mask (`arange(s) <= q_pos`), and `_full_attn` passes it through, so a query at
position *q* cannot attend position `> q`. `draft_pos` is defined as "the highest
position whose draft KV belongs to a committed token", the draft step runs over
`[draft_pos+1, seq_len-1]`, and everything past that is uncommitted by
construction.

**So this is a false-docstring defect, not a correctness one.** `warm_draft`'s
claim that a warm follower is "bit-equal to a cold follower" is false once the
pool has been used — stale bytes really are present in those slots — but nothing
reads them, so no token changes.

The poison experiment is what separates the two: an argument from the mask alone
would have been reading, and the first version of this probe (poisoning ALL slots,
including committed ones) did change the tokens, which is the control showing the
probe can detect a read when one exists.

## What is NOT claimed

**This is not the cause of the draft-hit token divergence.** Zeroing every draft
block at allocation removes the difference above and does **not** change the
hit-vs-miss tokens, which still diverge at index 12. The two defects were
separated by that experiment: the block-staleness fix was applied, re-measured,
and the token divergence was unchanged, which is what sent the search to the
draft's conditioning vector instead.

So this is recorded as a real, independent, still-open defect — the experiment
that found it was performed for the other bug and it would otherwise be lost.

## What a fix needs that this does not have

Which positions are legitimately readable before the row writes them. Zeroing
whole blocks at allocation is correct-but-wasteful (the pool's blocks are large
and mostly rewritten) and was only used here as a probe. The shipped fix should
either zero the same span the draft can read before it writes, or make the read
mask exclude unwritten positions — and it needs a device measurement, since the
pool geometry that decides the cost is the CUDA one.

## Gate

None. The probe that measured this was a temporary edit to `engine.py` and is
not in the tree, so there is no regression guard for it yet; that is why this
entry has no fix line.
