# The selector walk runs once per tick, not once per row — 2026-09-03

> Status: **counted, not timed.** Launch and sync counts below are read off the
> source; the wall-clock win is `pending-remote`. CPU suite 197 passed / 7
> skipped, ruff clean.

## Context

`DFlash2Head.step` walked one row at a time, and inside each row `path` walked
one slot at a time. The slot loop carries a real dependency — `prev` feeds the
next slot's transition score — but the row loop carries none, and the old code
paid for both:

```python
for j in range(hidden.shape[1]):
    score = unary[j] + succ[cand[j]].float() @ (pred[prev].float() * proj[j])
    prev = int(cand[j, int(score.argmax())])     # two host syncs, per slot, per row
```

At B=8 and width 8 that is **56 device-to-host synchronisations in one tick**,
eight separate projections through the trunk's 1271.4 M-parameter `lm_head`, and
eight through `selector.proj`. The drafter is 68.4% of a speculative tick, and a
speculative tick is about 9.2x a base tick
([engine-tick entry](2026-09-03-dflash2-on-the-engine-tick.md)); the file carried
its own marker naming this as the upgrade.

## What Worked

`path` became `paths`, which takes a list of anchors and steps every row through
slot `j` together. `prev` stays a device tensor across the walk and the whole
batch materialises once, at the end, through a single `.tolist()`.

| per tick, B=8 width 8 | before | after |
|---|---:|---:|
| host syncs in the walk | 56 | **1** |
| `lm_head` projections | 8 | **1** |
| `selector.proj` projections | 8 | **1** |
| transition matmuls | 56 | 7 einsums |
| `zeros_like` + `cat` in `_conv` | 80 | **0** |

`path(hidden, anchor, backend)` is now one line delegating to `paths`, so there
is one implementation and the single-row callers — `draft()` and the probe
script — are unchanged.

`_conv`'s second tap built a zero pad and concatenated it every call. The padded
head contributes nothing by construction, so the tap now adds into the tail
instead: `out[:, tap:] += coef[:, tap:, tap] * blocks[:, :-tap]`. Same arithmetic,
two fewer allocations per call, 80 fewer per tick.

## What this does not fix

`block_hidden` is still called per row, and it is the larger half — 92 backend
ops per row against `path`'s 3 + 4 per slot. Batching it needs padding and
masking across rows whose context lengths differ, which `_attend` does not take
today. **This change alone does not flip the speculation verdict**; the arm was
1.67x slower and the recorded ceiling for batching the whole drafter is 3.64x,
which is a division of a measured 6.20x forward reduction by a measured 1.70x
tick cost, not a third measurement.

## The test

`test_batched_walk_keeps_each_row_on_its_own_anchor` — two rows with identical
logits and different anchors must produce different walks, plus a negative
control that swaps the anchors and expects the rows to swap. The failure this
guards is silent: one row's `prev` scoring another row's candidates costs nothing
visible and drafts tokens the trunk rejects.

`test_block_drafter_on_the_engine_tick` caught the refactor itself. It
monkeypatches the drafter's readout, and moving the call site from `path` to
`paths` left the patch pointing at a method the engine no longer calls — the
oracle arm silently stopped drafting and `spec_accepted` went to 0. A patch that
no longer takes effect is the failure mode this repo has already recorded once.

## Bench snapshot

| date | commit | host | target | model | tok/s | notes |
|---|---|---|---|---|---|---|
| 2026-09-03 | (this branch) | — | cpu | tiny | — | correctness only; CPU has no speculative tick to time |
| pending-remote | | H20 gpu7 | cuda/sm90 | Qwen3.8-27B-NVFP4 | pending | 200 GSM8K B=8 W=8, both arms one process, against base 232.3 / spec 139.4 |

## Rule

A sequential dependency along one axis is not a reason to loop over the others.
Check which axis actually carries the dependency before accepting a Python loop
on the hot path.
