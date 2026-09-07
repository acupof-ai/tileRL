# Cut the prefill publish flood: 2 entries per prompt at any length, +28% reuse — CPU, 2026-09-08

**Date:** 2026-09-08
**Machine:** local CPU target (`TILERL_TARGET=cpu`), tiny model
**Change:** `engine.py` `_finish_prefills` publishes the FIRST interior chunk boundary and the
last, never the ones between. `_Req.interior_published` is the counter.
**Verdict:** accepted. Eviction policy is refuted as the layer — see
[errors/2026-09-08-the-eviction-policy-was-the-wrong-layer.md](../errors/2026-09-08-the-eviction-policy-was-the-wrong-layer.md).

## The defect, in one number

A miss prefills from token 0 and every interior chunk boundary published one entry, so the
publish count grew with prompt length. Counted through the engine:

| arm | 2048-token prompt | 8192-token prompt |
|---|---:|---:|
| publish every boundary (before) | 4 | 16 |
| publish every 2nd | 3 | 9 |
| **first + last (this change)** | **2** | **2** |

At a 31k prompt the old path published 62 into a budget holding 6
([the cascade entry](../errors/2026-09-07-a-miss-self-reinforces.md)): one miss evicted every
other session's shared head, so the next session missed and did the same — 11 of 12 in sequence
at 14.1 s each on the H20.

The count, not the victim, is the operand. A policy chooses which of 62 entries to drop when 6
fit; it cannot make 62 fit. Only a length-independent count gets under a fixed budget, which is
why the every-2nd arm is refuted despite a better grid score than LRU: 9 publishes at 8192 tokens
is ~32 at 31k, still a flood.

## Results

`scripts/probe_prefix_eviction_policy.py`, 6 sessions × 2048 tokens × 2 turns through the engine,
reuse in tokens read at admission. 196608 possible.

| eviction policy (publisher unchanged) | grid | | publish arm (plain LRU) | grid |
|---|---:|---|---|---:|
| pure LRU | 81408 | | every boundary (control) | 81408 |
| 2class w=4 | 83968 | | every 2nd † | 93184 |
| extensions-first + reparent | 83968 | | every 4th † | 107008 |
| extensions-first, longest | 101376 | | **first + last** | **104448** |
| capped sharers | 109568 | | | |
| middles-first | 110592 | | | |
| length × sharers | 115712 | | | |

**104448, +28% over LRU**, from deleting publishes with no policy change at all.

**† these two arms are monkeypatched, the rest are real code.** The eviction column and
`first + last` are measured on the tree; the stride arms were only ever a patched
`_publish_prefix`, and that harness had a bug (below) that moved `first + last` by 7680 tokens.
So the stride numbers rank the arms and nothing more — they are not comparable to the tree
figures at the digit, and no verdict rests on their exact values. The verdict against stride is
its publish COUNT, which is arithmetic over the boundary count and independent of the harness.

## The cost, priced

Interior boundaries exist for a **partial** sharer: a later request matching 40% of an earlier
prompt. The grid cannot see this — its turn-2 request re-sends its own whole prompt and matches the
completion publish, so it never needs an interior boundary at all. That blindness is why the
publish-nothing arm scored 180224 there (also patched) and means nothing.

A separate fixture prices it: one lead prompt, then 4 followers each sharing 25/50/75% of it plus a
private tail.

| arm | partial-sharing reuse |
|---|---|
| every boundary | 24576 / 24576 |
| **first + last** | **20480 (−17%)** |
| every 2nd † | 23552 (−4%) |
| publish nothing † | 0 (−100%) |

Same provenance mark: `every boundary` and `first + last` are the tree, the other two are the
patched harness. Under it `first + last` read 18944 (−23%), so the patched cost was overstated by
6 points.

### −17% is this fixture's number, not the fix's

That whole table is plen 2048, and the cost is **not** length-independent. The first interior
boundary sits at one chunk — `max_num_batched_tokens`, 512 — and does not move with the prompt, so
the published prefixes all live in the prompt's first ~2048 tokens whatever its length. A partial
sharer of a long prompt diverges past all of them and matches an absolute cap, not a fraction.

Followers sharing 50% of the lead prompt, reuse against ideal:

| lead prompt | reuse | ideal | of ideal | resident entry lengths |
|---:|---:|---:|---:|---|
| 1024 | 1536 | 1536 | 100.0% | 512, 1024 |
| 2048 | 2560 | 3072 | 83.3% | 512, 1024, 1536, 2048 |
| 4096 | 3072 | 6144 | 50.0% | 512, 1024, 1536, 2048, 4096 |
| 8192 | 3072 | 12288 | 25.0% | 512, 1024, 1536, 2048, 8192 |
| 16384 | 3072 | 24576 | 12.5% | 512, 1024, 1536, 2048, 16384 |
| 31767 | 3072 | 47649 | **6.4%** | (the H20 cell's own prompt length) |

Reuse is pinned at 3072 tokens from 4096 onward while the ideal grows, and the publish count is 2
(1 interior) on every row — asserted, not eyeballed. **So the fix is length-independent in its
COUNT, which is what the flood argument needs, and length-dependent in its VALUE, which a single
percentage hides.** At the H20 cell's own 31767-token prompts it is **6.4% of ideal**. This is the
cost, stated as the axis rather than as one cell of it.

### Part of that cost is a default, and moving it trades one hole for another

The first boundary sits at one chunk, and the chunk size is `max_num_batched_tokens` — config, not
a property of the rule. `cli.py`'s `_build_engine` never passes it, so serve takes `build_engine`'s
**512** default and the table above is the shipped path. Raising it deepens the first boundary
**without changing the publish count**, so it is not constrained by the flood argument.

Measured at 2048, followers sharing 50%, against the 512 rows above:

| lead prompt | 512 (shipped) | 2048 |
|---:|---:|---:|
| 2048 | 83.3% | **0.0%**, and only 1 publish |
| 8192 | 25.0% | 83.3% |
| 16384 | 12.5% | 50.0% |

It helps long prompts and **opens a hole at short ones**: a prompt at or under one chunk has no
interior boundary at all, so it publishes only at completion and a partial sharer of it matches
nothing. The knob moves where the cost falls rather than removing it — a prompt-length-to-chunk-size
ratio, not a free win.

So the first-boundary depth is **a tunable that trades reuse at one prompt scale against reuse at
another**, and its other side is unmeasured here: a bigger chunk is a bigger forward, and rows
sharing a tick pad to a common width, so there is a forward and padding cost this probe does not
price. Not a proposal to change the default.

### The length-aware alternative is refuted: it gives back the whole gain

Keeping the count at K and scaling the POSITIONS — publish at K evenly spaced points, so the
boundaries stay spread through the prompt — fixes the length axis exactly as intended, at K=4:

| lead prompt | first + last | K=4 spaced |
|---:|---:|---:|
| 2048 | 83.3% | 100.0% |
| 8192 | 25.0% | 100.0% |
| 16384 | 12.5% | 100.0% |

And it is refuted anyway, on the axis that matters more: **its multi-session grid total is 81408 —
pure LRU's baseline to the token**, against first+last's 104448. It buys back partial sharing by
spending exactly the cross-session gain the fix exists for. More publishes per row is more
pressure on the shared head, which is the original cascade in miniature.

Three further defects found while testing it, kept here because they are the reasons not to revisit
it without fixing them first:

- **The 100.0% column is an artifact.** It was measured at share fraction 0.5 only, and 0.5 lands on
  a K=4 band edge at every power-of-two length — one test repeated four times. At fractions that do
  not align, the same rule reads 78.8% (0.37), 88.8% (0.61), 87.4% (0.93). The real behaviour is
  ~79–89%, not 100%.
- **K does not respond to the budget.** Measured publish count at plen 8192: 4 at budget 3, 4 at
  budget 4, 4 at 6, 4 at 12. At a 3-snapshot budget one row publishes 4 entries, so the flood
  returns at small budgets — the defect the PR exists to remove. A count gate has to assert at the
  smallest supported budget; 31k tokens is the length where it cannot fail.
- **The count is K+1, not K.** At plen 31767 it published 5. The last-boundary condition adds one
  when the last boundary is not a band edge, and an off-by-one toward MORE publishes is wrong
  exactly where the bound is load-bearing.

Evenly spaced also assumes a **uniform divergence prior** — it maximizes expected coverage only if
sharers split at a uniformly random point. That is the right null when the split distribution is
unknown, and it is an assumption, not a measurement: a skewed workload (everyone sharing a fixed
system prefix, say) wants boundaries clustered where the split actually is.

**The bill is unpaid, not absorbed**, and it is a recompute rather than a wrong answer: a partial
sharer re-prefills the span it would have matched. Where partial sharing is heavy the DRAM tier
absorbs it, but no measurement here shows how common partial sharing is in real traffic — the
live V100 child has been up 7+ hours with `prefix_hits 0`, `prefix_published 0`,
`prefill_forwards 0`, so the share distribution is genuinely unavailable and this arm ships on the
flood argument alone.

## Gate, and why one assertion was not enough

`tests/test_e2e.py::test_a_prompt_publishes_two_entries_whatever_its_length`, two mutants:

| `_finish_prefills` | gate |
|---|---|
| publish every boundary (the defect) | RED |
| publish the last boundary only | RED |
| first + last | green |

The gate asserts the count is 2 and length-invariant; **flood-safety then follows as an argument,
not as an assertion** — 2 is under any budget we support. That inference holds only while the rule
is first-and-last. Anything band- or budget-derived breaks it, and such a rule needs the count
asserted at the SMALLEST supported budget instead, since 31k tokens is the length where a
budget-blind count cannot fail (measured: the K-spaced rule published 4 at budget 3, 4, 6 and 12
alike).

The count assertion alone went **GREEN** against last-only, which is also a constant 1 and scores
*better* on any self-hit fixture while costing every partial sharer. So the gate carries a second
arm: a row sharing 1024 tokens of an earlier prompt must reuse at least half of them, which only
the first interior boundary provides. Both lengths are 4× apart because a count that is small at
one length proves nothing — the growth is the defect.

## The harness that produced the marked numbers

The publish arms were first measured by monkeypatching `_publish_prefix` with a
`{id(request): boundaries_seen}` dict. CPython recycles ids, so a fresh row inherited a completed
row's count and skipped its own first boundary too. It reported `first + last` as **112128 /
−23%** where the tree measures **104448 / −17%** — understating the gain and overstating the cost
at the same time, which is the combination that survives review because the claim looks
conservative. The real fix keeps the counter on `_Req`, which cannot alias.

`test_an_intermediate_chunk_publish_stays_out_of_the_disk_tier` needed its bound relaxed from
`>= 3` publishes to `>= 2`, since 2 is now the count. Checked for vacuity: with `spill=True` forced
on the interior publish it still goes red, so the `ssd_offered == 1` assertion is intact.

## What is not fixed

`test_a_prompts_own_publishes_evict_the_prefix_it_shares` (#252) stays `xfail(strict)`. It calls
`store.insert` eight times directly, so it never runs the publisher and this fix cannot reach it —
it is the flood, hand-written, and it documents what LRU does when something else floods the store.

## Rule

**A cache defect whose operand is a count is not fixed by choosing a better victim.** Six eviction
policies were measured against a publisher emitting 62 entries into a budget of 6; the best honest
one gained 3%. Cutting the count to 2 gained 28% with plain LRU.

**A fixture whose requests re-send their own whole prompt cannot price a change to intermediate
prefixes.** It reported 92% of theoretical maximum for the arm that destroys all cross-session
partial sharing. Two fixtures, one per direction, or the number is one-sided.
