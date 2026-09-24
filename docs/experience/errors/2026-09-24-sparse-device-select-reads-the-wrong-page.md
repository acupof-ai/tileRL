# Sparse device-select attends the wrong page when candidates have holes — V100 sm70 / CPU tiny, 2026-09-24

> Status: open — the PR that gathers by candidate position removes this line.

## What happens

`SparseForward._select_device` (`sparse_engine.py`) maps a chosen logical page to
its physical block with `self.s_l2p.gather(1, chosen)`. `s_l2p` is indexed by
**candidate position** — `fill()` writes `s_l2p[bi, :nc] = l2p[cand]` — while
`chosen` holds **logical page numbers**. The two agree only when `cand` is
`range(n)`.

`cand` is not always `range(n)`: it is built by a filtered range
(`[p for p in range(0, own_first) if tr.has_bounds(r.req_id, p)]` in
`sparse_runtime.py`, and the `p in tr.keys[...]` variant for the index scorer), so
any page without bounds punches a hole and shifts every later page by one.

## Measured (CPU tiny, `RefBackend`)

With bounds validity cleared on logical page 0, `cand` becomes `[1, 2, 3, 4]` and
the code returns the block of the wrong page:

| chosen logical page | block returned by the code | correct block |
|---|---|---|
| 1 | 171 | 171 |
| 0 | **-1** | 172 |

**The failure is silent, not caught:** the wrong slot is usually a valid block, so
the `-1`-means-cold mask does not fire and attention reads a resident page that is
the wrong page. It is not a crash and not a miss.

## Why it matters

The `reuse` path is the captured/graph path (`sparse_device_select`), so the wrong
page is attended inside a CUDA graph, where nothing can look at it. Every
selection that names a page above a hole is off by the number of holes below it.
The shipped default (`bounds` scorer, all pages bounded) makes `cand` the identity
and the bug invisible — which is why it survived: the two index spaces coincide
for every configuration that has been measured on device.

## Fix

Gather with `safe_pos`, the candidate position of each pick, instead of `chosen`.
`positions` is computed by `order_members` in candidate-position space already, so
the correct index is available at the call site and no new state is needed.

## Gate

`tests/test_sparse_engine.py::test_device_select_maps_chosen_pages_through_their_candidate_position`.
Builds a device-select engine, captures the sparse graph, punches one hole in the
bounds validity, drives the real `fill` + `_select_device` seam, and asserts the
returned physical blocks are `l2p[chosen]` for the valid prefix.

**Negative control:** reverting to `gather(1, chosen)` reads
`phys [-1, 0] for chosen [1, 0], correct [171, 0]` — red on the same assertion.

**Where the hole goes matters, and getting it wrong makes the gate vacuous.** The
first version of this gate punched the hole at logical page 3; with `k = 2` the
selection picks the first two candidate positions, which both sit *below* the
hole, where the two index spaces still agree. The negative control then **passed**
— a gate that cannot fail. Moving the hole to position 0 is what gives it teeth.
Any future edit to this gate must keep at least one *chosen* page above the hole.

## Related

Same file, same review pass, separate defect:
[the prefix hit that feeds its last page twice](2026-09-24-prefix-hit-feeds-the-last-page-twice.md).
