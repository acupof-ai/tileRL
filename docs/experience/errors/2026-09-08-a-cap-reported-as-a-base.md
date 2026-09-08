# 64.0% was a lower bound, and I called it the base — 2026-09-08

**Status:** measured. The level-5 base is **91.0%**, not 64.0%. The 27-point difference is one
`--eval-max-new-tokens` value, no code change.

## What I reported and what is true

`errors/2026-09-08-a-generator-that-never-ran.md` fixed the MATH generator, and the level-5 file
it finally produced was scored on card 0 at the recipe's `eval_max_new_tokens=2048`:

| | n | correct |
|---|---:|---:|
| terminated naturally | 68 | 64 (94.1%) |
| **hit the 2048 cap** | **32** | **0 (0%)** |
| total | 100 | 64 → reported as **64.0%** |

Longest completion below the cap: **1889**. So nothing exists in the 159 tokens between 1889 and
2048 — the 32 are one truncated population, not the tail of a distribution.

Rerunning **exactly those 32** at 6144 on card 6 (`/work/runs-l5c/2898b40d2130`):

| | |
|---|---|
| correct | **27 / 32** |
| tokens | mean 3331, min 1413, max 6144 |
| ≥3072 | 14 |
| ≥4096 | 9 |
| **= 6144, still at the new cap** | **3, all wrong** |

Only **two** of the 32 are genuinely wrong (2507 and 4640 tokens, both terminated then missed).
So the base is **(64 + 27) / 100 = 91.0%**, SE 2.9 pt.

## The interval, resolved

The 6144 rerun does not just move the point estimate — it collapses the interval, because those
32 are no longer unscored:

| | n |
|---|---:|
| known correct | **91** |
| known wrong | **6** (4 that terminated under 2048, 2 that terminated under 6144) |
| **still unknown** (at the 6144 cap) | **3** |

**cap 2048: [64%, 96%], 32 points wide. cap 6144: [91%, 94%], 3 points wide.**

The 96% is the *old* upper bound and it is now resolved, not a ceiling. Anyone still quoting it —
I did, in this file's first draft — is citing an interval their own measurement has already
replaced.

## The mistake

I had the evidence and stopped one step short of it. The token histogram showed **nothing between
1889 and 2048** — a 159-token gap — and I wrote that gap down and used it to argue the 32 were
truncated rather than wrong. That argument was right.

**What follows from it is that their true correctness is unknown, so 64.0% is a lower bound.** A
truncated completion has not been scored; it has been prevented from being scored. The honest
first report was:

> level-5 base ∈ [64%, 96%] — the lower bound scores truncation as wrong, the upper as right, and
> n=100 cannot narrow it.

I reported the lower bound and called it the base. A peer's decision criteria were then built on
it, and the 3-step GRPO probe those criteria gate was about to be run against a number 27 points
low.

## Why the direction matters more than the size

The task was chosen because level 5 is *harder* than GSM8K, which failed P1 by being solved:
88.0% base, 73 of 81 steps tied at the ceiling.

| | base | headroom to 100 |
|---|---:|---:|
| GSM8K (the task that failed this way) | 88.0% | 12.0 pt |
| level 5 at cap 2048 (what I reported) | 64.0% | 36.0 pt |
| **level 5, cap 6144 (measured)** | **91.0%** | **9.0 pt** |

**The 24-point difficulty gap that justified the task does not exist.** It was the cap. Level 5 is
*harder to finish* than GSM8K, not harder to get right, and a cap converts the first into the
second silently — a truncated correct answer and a wrong answer are the same row.

And P1 wants +5 pt: from 91.0% that is **96%, which is 2 points above the 94% upper bound this
cap can produce.** Not a tight gate — an unsatisfiable one. An adapter that fixed all 6 known
wrong answers and answered all 3 unknowns correctly would score 94%. **No `after` value passes
this gate at this cap**, independent of how good the training is, and that is arithmetic rather
than difficulty.

Same shape as `errors/2026-09-08-a-gate-too-strict-to-be-met.md` and
`errors/2026-09-08-p1s-gates-cannot-see-p1s-target.md`: three times in one day a criterion was
written without checking it against the feasible region. There the gate exited on `after > before`
where the roadmap needs +5 pt; here the +5 pt is unreachable.

## The cost of the fix, which is not small

Raising the cap is not free and moves the wrong way on both axes:

- **Mean generation 1386 → 3331 tokens, 2.4x.** `seconds_per_step` is the denominator of the
  throughput target, so this is the most expensive single configuration choice on the table.
- **It makes the task easier, not harder**: 64% → 91%.
- **6144 is not the end** — 3 of 32 are still at it.

So "raise the cap so level 5 is measured properly" and "pick level 5 because it is hard" cannot
both be satisfied. That is a task-selection question, not a cap question, and it is now visible.

## Rule

**A capped measurement of an uncapped quantity is a bound, and which bound depends on how the cap
scores.** Here truncation scores as wrong, so the reading is a *lower* bound; a cap that scored
truncation as right would give an upper one. Report the interval and say which end the
instrument produces. The check is one question: *if this limit were removed, could the number
only go one way?* If yes, it is not a measurement of the thing.

**The general form is that a number's STATUS was misread, not its value.** 64.0 was arithmetically
correct; what was wrong was calling a bound a point estimate. A peer made the mirror error the
same day — treating a derived bracket as an upper bound when the non-weight term is superlinear,
so it was not a bound at all. Neither of us miscalculated. Both of us assigned a number a standing
it did not have, and nothing in either derivation could catch that, because the arithmetic was
fine.

**The evidence for "this is a bound" and the evidence for "this is the value" are the same
evidence, read to different depths.** The empty 1889–2048 gap proved the 32 were truncated. One
more step — truncated means unscored, unscored means unknown — was the whole correction, and I had
already written the premise down.

**And a resolved interval must retire the old one.** After the 6144 rerun the 96% upper bound was
no longer a ceiling but a superseded guess, and the first draft of this entry quoted it anyway, as
did the peer reasoning from it — both of us citing an interval my own measurement had already
replaced. **The two are not independent errors: one number lived twice, in two sessions, from one
report.** The superseded version stays readable, correctly-signed and right in magnitude, so
nothing about reading it says which version it is. A number that has been superseded needs the
measurement that replaced it named next to it, or it goes on circulating on its own.

See [[a-model-is-not-the-quantity-it-models]],
`errors/2026-09-04-the-eval-cap-measured-itself.md` (the same defect on the rollout cap, which is
why `eval_max_new_tokens` exists as a separate flag at all — it was decoupled to stop the rollout
cap from scoring the eval, and then the eval cap did it anyway).
