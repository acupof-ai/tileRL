# A ragged prompt publishes the deepest boundary its actual walk reached — 2026-09-14

## Context

`_finish_prefills` published the last prefill boundary only when a chunk end
landed at `_last_prefill_boundary(n)`, a position predicted from the prompt
length alone. The real walk depends on the per-tick budget
`max_num_batched_tokens - len(decodes)`, so one decode row sharing the tick
shifted the walk and `last` never fired: the prompt published nothing at its
deep boundary, silently — every counter normal, the request correct. Two
schedules in the open case table lost the boundary on a real engine
(`(33, 16)` and `(49, 8)`; the other three xfail cases were the planner-only
harness rejecting a 1-token chunk, not a lost publish).
[errors/2026-09-08-a-one-token-chunk-made-last-unreachable.md](../errors/2026-09-08-a-one-token-chunk-made-last-unreachable.md)

## What Worked

Stop predicting. At every aligned interior chunk end inside the ragged tail
window (at or past the n-only prediction, which sits within 32 tokens of the
end), hold that boundary's exact state snapshot — at most two aligned chunk
ends fit in the window, so at most two clones per ragged prompt, keeping the
newest. Insert it at completion. Schedules that match the prediction are
unchanged; shifted walks now leave an entry at the deepest boundary they
actually reached. A row still publishes at most two entries (first interior
boundary plus the deepest), so the publish-flood guard is intact.

The boundary snapshot is exact in itself: it is taken at a state-pool boundary
where the GDN slot covers exactly that prefix. The rejected "deferred publish
at DONE" design paired the boundary tokens with the *whole prompt's* state;
this holds the snapshot from the boundary tick instead.

The predictor stays (tail window start); the open entry's proposed
`spill_held` store change is unnecessary — the dense SSD tier it fed was
removed in #568.

CPU gate `test_a_ragged_prompt_publishes_the_deepest_boundary_the_walk_reached`
drives a real engine over nine `(n, budget)` schedules, takes the expected
length from a planner-only drive, and asserts the store holds that entry. Red
on main for the two shifted schedules, green after. `test_kv.py` +
`test_e2e.py` unchanged otherwise (115 passed, 1 skipped, 1 pre-existing
xfail).

## Rule

When a position depends on per-tick state, a predictor keyed on the request
alone is a schedule assumption. Hold the value at the real boundary and act
when the walk ends — the predictor may scope the work (a 32-token window), it
must not gate correctness.
