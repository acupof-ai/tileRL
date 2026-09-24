# A full-page prefix hit feeds the last page to the GDN state twice — V100 sm70 / CPU tiny, 2026-09-24

> Status: FIXED 2026-09-24 by #824 (`260ee7a3`). The restored snapshot is written
> back after the re-forward has produced the logits. OPEN.md row removed.

## What happens

A sparse follower whose prompt matches a published prefix in a whole number of
pages hits a special branch in `_admit` (`engine.py`): the adopted entry's GDN
snapshot is the state AFTER the last page, but no chunk would forward (zero
residual tokens), so the row would sit in PREFILL with no first-token logits.
The escape is to re-forward the last page — `prefill_from = matched -
BLOCK_TOKENS`.

The restored snapshot has **already consumed** that page. The re-forward consumes
it again, and the row then decodes from a state that has seen those 16 tokens
twice.

## Measured (CPU tiny, `RefBackend`)

The mechanism is measured, not inferred. Three quantities:

| observation | value |
|---|---|
| restored state vs a prefix-MISS engine's, at the same point | **bit-identical** (`(a) == oracle exactly: True`) |
| the same state after the re-forward | differs by **26.875** |
| `max |state|` | 28.125 |
| relative | **0.956** |

The first row is what makes this a proof rather than a correlation: the snapshot
as restored is exactly right, so the 26.875 is produced by the re-forward and
nothing else. For scale, the same code path is held to a **1.2e-2** chunk-rounding
bound elsewhere (`test_gdn_chunk_rounding_bound`); this is ~80x that.

Branch reachability, read at the admission call rather than after it:
`len(req.tokens) == matched == 384`, `prefill_from = 368 = matched - 16`. A
post-hoc read shows `len(req.tokens) == 385` because the follower's own first
token has been appended by then — which reads as "the branch was not taken" if
you sample the state too late. That mistake cost a false negative on the first
attempt.

**The emitted first token is identical** on both arms, which is why
`test_sparse_nodraft_full_prefix_resend_re_forwards_the_last_page` was green: it
compares token lists, and the damage is in the recurrent state that subsequent
tokens read.

## Why it matters

Any page-aligned prompt sent a second time to a service with sparse prefix
sharing reaches this path — the ordinary repeated-prompt case, not an exotic one.
The state divergence is ~96% of the state's own magnitude, so every token after
the first is generated from a materially wrong recurrent state.

## Fix

Hold the restored state and write it back after the re-forward has produced the
logits, before anything samples from it or publishes from it. The logits still
come from the re-forward; the state does not.

## Gate

`tests/test_sparse_engine.py::test_full_prefix_hit_does_not_re_feed_the_snapshotted_tokens`.
Compares the hit follower's GDN state against a prefix-MISS oracle on the
identical prompt, at the 1.2e-2 bound the path already carries, and asserts the
arm really was a full hit (`sparse_matched == len(prompt)`) so it cannot pass by
not taking the branch.

**Negative control:** with the write-back removed the same assertion reads
`26.875000 against max|state| 28.125000 (0.9556 relative)` — red, same reading as
the reproduction.
