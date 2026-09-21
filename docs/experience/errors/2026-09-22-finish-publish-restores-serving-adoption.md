# M3 删掉 close 发布后，32k+短解码的同头 follower 全量重算 — V100, 2026-09-22

> Status: pending-remote (#796 / PR #800)

## Context

Publish-once M3 (#782, [2026-09-21-close-zero-bytes.md](2026-09-21-close-zero-bytes.md))
deleted the forced prompt-end frontier closure at request finish: a page
reaches the shared prefix only when it leaves the k+window resident union
(`offer_drop`). That is correct for the training disjoint-span workload that
motivated it, but device follow-up on the M6 serving geometry (32028-token
prompt, 8 generated tokens, pool with no pressure) found same-head followers
recomputing the whole prompt: 189–246 s per follower, zero adoption.

Device counters over the run: 45328 demotions and 45328 `offer_drop` calls,
then `keys_nonempty_calls=0`, `keys_total=0`, `xfer_calls=0`,
`peak_shared_pages=0`.

## Root Cause

`publish_dropped` closes the grow frontier only to a length m where
`{0..m-1} ⊆ pend` (pages that LEFT the union, with bounds) AND `m ∈ snaps`
(an aligned prefill-chunk boundary GDN snapshot; the snap set is capped to
`{lowest, newest}`). Closure is contiguous from page 0. In the M6 geometry
the prompt is 2001 pages + 12 tokens and sparse_k=128 plus the forced own
window (8 pages) keep the low pages resident; with the pool at 199/2213
blocks nothing evicts them (`evict_victim=0`). After 8 decode tokens the low
pages have never been offered, so no contiguous pending prefix ever reaches
even the lowest surviving snapshot and the closure returns `{}` on every
call. The natural-leave trigger is healthy; the geometry never fires it.

A proposed remediation (b'2) synthesized a prompt-end GDN snapshot from live
recurrent state at finish. A production-path probe showed it was both
unneeded and wrong: the production prefill chunker cuts the unaligned tail
(last-chunk cut to a block boundary, `engine.py:1549-1561`) so the <=15-token
remainder ships as its own forward and a real snapshot exists at the floor
page — for the 32028 prompt, page 2001 = 32016 tokens, tail recompute 12
tokens, not the 1308 a ragged-end analysis predicted. A snapshot written at
finish would instead capture post-decode recurrent state at the prompt
boundary.

## Fix

A scoped revert of one piece of M3, with none of the machinery M3 deleted:

- An origin publisher that finishes successfully closes and publishes its
  prompt prefix synchronously at finish, while its frames, private blobs and
  aligned snapshots are still live, reusing the exact per-page transfer an
  offer uses (resident frames are D2H'd inline). No #741 batch D2H context,
  no #743 background thread/futures/queue, no SSD-lift worker.
- The origin gate is `not failed and sparse_matched == 0`: a cancel, a failed
  row, or an adopted follower never publishes (an adopted row's blobs are
  already shared under the same content keys; re-closing adds a redundant
  entry and refs for zero bytes).
- `close_prompt` takes a `publishable(p)` predicate naming only pages with a
  live source (held private blob or resident frame), so the closure can never
  name a dead content key. A source-less page in the middle does not kill the
  aligned suffix: the closure falls back to the highest snapshot boundary
  below the first gap instead of dropping the whole suffix.
- The prompt-end synthesized snapshot was deleted; the deepest closure is the
  natural chunk-aligned boundary the production chunker already records, and
  the follower re-forwards only the <16-token tail.

CPU gates (`tests/test_sparse_prompt_end_publish.py`, k=128, 2002-page
prompt): short-decode publisher is adopted by an immediate same-head follower
at the exact full closure length (2002 pages, 0-token tail for the aligned
fixture); unaligned 32044-token prompt closes to 2002 pages and the follower
adopts 32032 tokens (12-token recompute); decoding past one chunk still
publishes via the natural chain (the discriminator control); an adopted row
publishes nothing at its own finish; a forced post-adopt eviction still takes
the real-demote path. Full suite 1094 passed.

Device finish-tick cost and the follower-adoption delta at M6 geometry are
pending remote; the earlier 189–246 s recompute is the before arm.

## Rule

Delete a forced lifecycle publish when the workload has no followers, but
keep the geometry where a request ends with resident prefix that an immediate
same-head follower needs: a content cache may publish at finish while the
content is still live, gated on origin identity and on a live source per page
— it must not synthesize state the forward path did not record, and must not
name a key it cannot land.
