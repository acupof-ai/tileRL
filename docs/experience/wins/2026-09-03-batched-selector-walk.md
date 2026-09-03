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

At B=8 and width 8 that is **112 device-to-host synchronisations in one tick (14 at B=1)**,
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
| host syncs in the walk (B=8) | 112 | **1** |
| `lm_head` projections | 8 | **1** |
| `selector.proj` projections | 8 | **1** |
| host syncs in the walk (B=1) | 14 | **1** |
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
| 2026-09-04 | 40bc83c | H20 gpu7 | cuda/sm90 | Qwen3.8-27B-NVFP4 | **135.5** | B=1 W=8, spec arm; base 78.4 — **1.728x** |
| 2026-09-04 | 40bc83c | H20 gpu7 | cuda/sm90 | Qwen3.8-27B-NVFP4 | 225.4 | B=8 W=8, spec arm; base 242.9 — 0.928x, was 139.4 / 232.3 = 0.600x |

200 GSM8K, greedy, `max_new_tokens=512`, decode graph on, both arms in one
process, `--out /work/accspec_b1` and `/work/accspec512`.

| | wall | tok/s | tok/decode-fwd | block accepted | GSM8K |
|---|---:|---:|---:|---:|---:|
| B=1 base | 823.5s | 78.4 | 1.00 | — | 168/200 |
| B=1 spec w8 | 477.4s | **135.5** | 6.12 | 6.14 of 8 | 167/200 |
| B=8 base | 266.7s | 242.9 | 7.79 | — | 165/200 |
| B=8 spec w8 | 290.2s | 225.4 | 42.13 | 6.17 of 8 | 163/200 |

**Speculation wins at B=1 and loses at B=8**, and B=1 is the shape a rollout has.
At B=1 the trunk runs 6.11x fewer forwards (64565 -> 10569) for a 1.728x wall
clock, so a width-8 spec tick costs 3.54x a width-1 tick — against the 2.41x
recorded for the verify tick alone, the difference being the drafter. At B=8 the
base tick already amortises across 8 rows, the same 6.1x forward reduction is
worth less, and spec lands at 0.928x.

The B=8 arm is the one comparable to the recorded pair: base reproduces 232.3 to
+4.6% and spec goes 139.4 -> 225.4, **1.617x** for the batched walk. Acceptance is
unchanged at 6.17 against 6.18 of 8, which is the separation this entry needed —
the drafter got faster without getting different. The same holds at B=1 (6.14).

Two measurement notes, both bought the hard way. **The recorded pair was taken at
`max_new_tokens=512`**; a first re-run at 256 put base GSM8K at 77/200 = 38.5%
against the recorded 170/200, on an arm the drafter cannot touch — completions
average 236 tokens, so most hit the cap and lose their final answer. The cap is a
parameter of the result, not a runtime knob. And **every spec number in this
repo's history before this entry is a B=8 number**: `scripts/acc_spec_arms.py`
had `num_slots`, the MMLU concurrency and the GSM8K concurrency all as the
literal `8`, so the script could not have produced anything else. `--concurrency`
now exists; the B=1 rows above are the first B=1 spec measurement taken. Its own
control is `tok/decode-fwd` reading 1.00 on the B=1 base arm against 7.79 at B=8
— a dead flag would have read 7.79.

The B=1 base arm reads 78.4 against the 92.4 recorded elsewhere for B=1 decode.
Those are different measurements — 92.4 is a decode microbenchmark at d512, this
is GSM8K with real prompts and context growing past 512 — and cards 0-3 carried
another tenant at 100% throughout. The A/B is internally consistent either way,
since both arms ran in one process on card 7.

## Rule

A sequential dependency along one axis is not a reason to loop over the others.
Check which axis actually carries the dependency before accepting a Python loop
on the hot path.
