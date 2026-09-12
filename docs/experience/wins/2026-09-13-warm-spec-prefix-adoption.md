# A spec follower adopts a warm sparse prefix, bit-equal to cold — 2026-09-13

> Status: **tiny CPU green; sm90 card parity pending-remote.** A spec follower
> now ADOPTS a published sparse prefix instead of prefilling from zero, and its
> tokens are bit-equal to a cold spec follower (`sparse_k=2` + spec, 24-page
> published prompt, 8 generated tokens). The sm90 continuity gate must pass
> before sparse is re-defaulted; this gate runs on CPU and does not exercise
> the card kernel.

## Context

Under sparse + spec the prefix cache published but could not serve a follower
(#530 returned-miss for correctness): the draft head is a second full-attn
model with its own dense KV pool, and it conditions every proposal on the
TRUNK hidden at the previous position. An adopted follower forwards neither
model over `[0..matched)`, so its draft pool is unbuilt and it has no hidden
to condition the first tail draft on. See the error entry for the diagnosis.

## What is saved

Neither of the two routes in that entry (rerun the skipped trunk forward, or
store per-position hidden) is needed:

- **per-page draft K/V (`dk`/`dv`)** rides in each shared prefix blob
  alongside the trunk K/V and bounds. The publisher's draft is dense, so its
  per-page K/V exist as soon as `draft.step` has run for that chunk.
- **ONE boundary trunk hidden** — the vector at position `matched-1` — is
  saved in the boundary snapshot. That single vector is all the first tail
  draft conditions on; subsequent drafts use the follower's own hidden.
- the GDN state/window snapshot and bounds that trunk-only adoption already
  saved are unchanged.

## What the follower does

On a hit under a draft, the entry is eligible only when it carries the
boundary hidden AND draft K/V on every key (an older trunk-only entry stays a
miss). The follower:

1. adopts trunk pages/bounds/GDN state as before;
2. copies each page's `dk`/`dv` (field-only reads, so trunk blobs are not
   pinned) into its already-reserved dense draft blocks;
3. zeroes the one boundary slot `matched%16` — the exact analog of the
   position-0 zero `DraftHead.step` does on a cold run, so no draft attends a
   frame that never existed;
4. primes `hidden/hidden_from/draft_pos = matched-1`, so the next
   `draft.step` spans the tail and writes from the boundary forward.

The publisher's earlier draft positions are never re-forwarded, and they do
not need to be: the draft is full-attention but its own page-zero (and now the
boundary slot) is zero by construction, so the restored draft state is exactly
the state a cold build of the same prefix produces.

## Publisher details

- offers are processed AFTER `draft.step` (the chunk's draft K/V is written
  after the trunk forward in a tick); a frontier that closes over many pages
  at once resolves each new page's draft block from the reserved dense span,
  not only the one page dropped that tick.
- a page that is still DEVICE-resident when its frontier closes (it is in the
  forced own window and never dropped) is snapshotted from its live physical
  frame — transfer handles held-host, spilled-SSD, and resident states.
- a transferred page whose private blob was already past the host budget is
  re-keyed to its content key for future demotes, so a re-demotion refreshes
  the shared blob instead of dropping it.

## Gate

`test_sparse_draft_follower_adopts_a_published_prefix_and_matches_cold`:
warm follower tokens == an isolated cold spec follower bit-for-bit; every
published blob carries `dk`; the follower adopts the full 24-page prefix
(`sparse_matched == 384`). The held prefix bytes (trunk + bounds + draft)
show as a measured==derived `kv_prefix` host ledger row, split out of
`kv_cold`. 768 CPU tests green.

## Rule

A second-model prefix snapshot needs that model's per-position STATE plus the
one activation at the hand-off boundary — not every position's activation.
Storing the boundary vector + the draft pages is sufficient and bit-exact
where a full hidden replay looked necessary.
