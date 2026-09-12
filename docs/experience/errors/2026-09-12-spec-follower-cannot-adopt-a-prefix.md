# Under spec decode a sparse prefix hit cannot serve a follower — 2026-09-12

> Status: **open.** #530 fixes the correctness bug by returning-miss (prefill
> from zero) whenever a draft head is attached. The warm path is not built:
> under the production defaults (sparse + spec) the prefix cache publishes but
> never serves a follower. The PR that lands the warm path removes this line.

## Context

Sparse prefix adoption copies a publisher's trunk KV into the follower, so the
follower skips the trunk forward over `[0..matched)` and prefills only the
tail. With speculative decode on, the draft head is a second model with its
own DENSE `PagedKvPool`; every proposal conditions on trunk `hidden` at that
position (`DraftHead.forward(hidden, ids, positions, kv)` slices
`hidden[:, off:off+q]`), and the draft's own KV is written only while its
forward runs.

The adopted follower forwards neither model over the matched tokens, so:

1. the draft pool for `[0..matched)` is never populated — the draft attends
   over unbuilt pages;
2. even with the pool populated, there is no trunk `hidden` at those
   positions to condition the draft proposals.

The desk-note test (5f, defect 3) demonstrates the miss path: a draft
follower on a published prefix must match a cold follower bit-for-bit. Before
the fix it adopted with `sparse_matched=80` and drafted against garbage.

## Root cause

The prefix cache was built for one model. Adoption transfers trunk state only
and assumes the saved forward is entirely skippable. Spec decode makes the
skipped forward load-bearing for a second consumer that cannot be restored
from the snapshot.

## Fix (landed)

Return-miss, never raise: with a draft attached, `_admit` skips the prefix
lookup and forces `matched = 0`, including a dense-store hit. The follower
prefills the whole prompt, both KVs build, outputs equal the cold path
exactly. Non-spec adoption is unchanged.

## Remaining work

Warm adoption under spec needs the trunk hidden at every matched position.
Two routes, both more than a few lines:

- run the draft model's own prefill over `[0..matched)` conditioned on trunk
  hidden — which requires the trunk forward the adoption skips (no save), or
- store per-position trunk hidden with the published entry and replay it into
  the draft — snapshot storage + a second restore path.

## Rule

A prefix snapshot is only restorable for consumers whose every per-position
input is inside the snapshot. A second model keyed on an intermediate
activation is such a consumer only if that activation was saved; KV state
alone is not enough.
