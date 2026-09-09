# The GSM8K reward-vs-step curve: run notes

**Date:** 2026-09-08
**Session:** v100-sm70-fp4-55
**Card:** H20 card 3, claim `tilerl-curve2`, pid 1170264, tree `/work/tilerl-s-v100-sm70-fp4`
sha `fd0c876`, log `/work/curve2.log`

Written **before** the results, because every number below is a precondition or a threshold and
a threshold decided after the data is not a threshold. The results section is appended when the
run finishes.

**Restarted at 13:55Z.** The first attempt (pid 1150334, sha `12da5a0`, log `/work/curve.log`)
ran 24 steps and was killed: `12da5a0` predates #323, so its curve points carried none of the
per-problem rows the **paired** 1.00 pt threshold below requires, and none of the `mean_len` /
`at_cap` fields that separate a score from a truncation artefact. The sha had been verified and
its contents had not —
[errors/2026-09-08-a-sha-confirmed-and-its-contents-not.md](../errors/2026-09-08-a-sha-confirmed-and-its-contents-not.md).
Restart cost ~15 min: the before-arm eval is a cache hit, and the 24 completed steps re-run at
23.3 s each. They did **not** re-run identically — steps 1-3 drew the same rollouts and step 4
onward diverged, so `--seed 0` fixes the inputs and not the trajectory. Nothing else in this
document changes — same flags, same criteria — but the two attempts are **two samples of one
configuration**, not one run continued.

**Three numbers the first attempt did measure**, all of which carry over because the
configuration is identical:

| quantity | reading | why it matters below |
|---|---|---|
| seconds per step | **23.3 s** (n=24, min 15.6, max 35.2) | the anchor's 56.88 is **2.44x** this, so `time_to_score` is read off this run's own `secs` and any plan priced at 56.88 is 2.44x too expensive |
| GSM8K base, greedy, cap 2048 | **87.4%** (437/500), 396.5 tok/correct | 0.6 pt from the anchor's 88.0, inside the 2.1 pt SE of that difference — **the base end of the anchor is reproduced**. Not re-measured on the restart: the cache serves the same rows, which is why it is a carry-over rather than a second reading |
| no-gradient steps | **10 of 17 tied**: 7 all-correct, 3 all-at-cap | a flat curve has two readings, and `tied` alone does not separate them; see the fifth branch in [what runs next](2026-09-08-what-runs-next-after-the-curve.md) |

## What this run answers

`time_to_score = steps_to_score × seconds_per_step`. The project has measured the right factor
many times and the left factor **never** — the 2026-09-05 run
([entry](2026-09-05-p1-grpo-27b-run.md)) reports 100 steps taking 99.4 min and moving GSM8K
88.0% → 93.6%, with **two points and no curve**, so it says what 100 steps buy and not which
step bought it.

If the score saturates at step 25, `time_to_score` falls by the ratio of the two `secs`
readings with no code change at all.

## Configuration, and the four ways it differs from the anchor

The anchor is the 2026-09-05 run at `91977a8`. Three differences are deliberate alignments and
one is not aligned:

1. **`--length-penalty 0.0`** — aligned. The length term in the GRPO reward postdates the
   anchor: `91977a8` is an ancestor of `aee90e5` (#293), verified in that direction rather than
   by a negative `is-ancestor`, which is compatible with two parallel branches. Its default is
   0.1. Measured, λ is a **switch, not a dial**: at 0.1 and 0.5 the advantages of an all-right
   group are identical to the digit, and at 0 they are all zero. The docstring's "cancels
   exactly" describes the *value* being invariant while the *effect* is not, which is why two
   readers took opposite conclusions from it.
2. **`--allow-short-rollouts`** — aligned. Two guards postdate the anchor: `cli.py:538` exits
   before step 1 when the base policy's mean completion exceeds `0.8 × cap` (346.7 against
   204.8 here), and `cli.py:862` breaks the loop after step 5. Both read only this flag. **The
   semantics matter and the manifest records only the switch:** here truncation is the
   experimental condition, not a warning being silenced — the 2.74x in tokens/correct that this
   run exists to re-examine exists *only* under a 256-token cap, so raising the cap to satisfy
   the guard would replace the phenomenon being reproduced.
3. **`decode_graph=True`** — **not aligned.** The anchor ran `decode_graph=False`. Score-neutral
   by design, since `grpo_loop(recapture_graph=True)` satisfies `train.py:379`'s refusal. Not
   wall-clock-neutral, so **56.88 s/step is not this run's divisor** and `time_to_score` is read
   entirely off this run's own `secs` column.

   The known silent failure path here — `invalidate_weights()` keeps the captured graphs while a
   cached cast (`_const_f32`) refills only when called, and a replay calls nothing — **is closed
   on this run structurally, not by a count.** `_const_f32` is populated from `_rmsnorm`
   (`backend.py:525`) and the fp4 output scales; `backend.linear` casts through `_dev`
   (`backend.py:593`), so LoRA's `lora_a`/`lora_b` never enter that cache. The cache holds
   frozen embeddings and norm weights only, and this run trains LoRA on a frozen base, so
   nothing a replay reuses can go stale. `tilerl-48` measured `refill_const_f32() = 0` after a
   LoRA `rl_step` with 8 entries, none of them adapter tensors.

   `train.py:536` discards `invalidate_weights()`'s refill count, so it is not in the manifest —
   **and under LoRA it is identically zero, so having it would decide nothing.** A field that is
   always 0 carries the same information as no field. If the anchor check fails, this is
   therefore *not* the first thing to examine; the order is the graph's wall-clock effect (which
   does not move the score), then data order, then GPU non-determinism.
4. **`--eval-max-new-tokens 2048`** — aligned, and **not a tunable**. At cap 256 the base arm
   reads 38.4% rather than 88.0%; the 49.6-point difference is the cap truncating the base, not
   a model difference. A curve run at 256 would be directionally right and roughly **9x**
   inflated in magnitude, internally consistent, monotone, low-noise — and **all six manifest
   gates would pass**, because `gsm8k_improves` compares before/after at the same cap and
   `reward_rises` never sees the eval cap at all. The 66 minutes it would save buy an artefact
   undetectable from the curve itself.

Command:

```
tilerl train --recipe grpo-gsm8k-27b --length-penalty 0.0 --allow-short-rollouts \
  --eval-every 25 --eval-curve-n 500 --eval-max-new-tokens 2048 \
  --eval-gsm8k /work/p1_gsm8k_test.jsonl --data /work/p1_gsm8k_train.jsonl --seed 0 --force
```

## Pre-registered thresholds

**Startup check, before the run is allowed to continue.** `engine.blocks` in the manifest must
read **1327**, and the three failure values are distinguished:

| reading | meaning |
|---:|---|
| **1327** | expected — `num_blocks` 1328 minus the graph's pad block |
| 1328 | `decode_graph` did not capture, so no pad block was allocated |
| 1152 | the `515 if args.eval_mmlu` branch did not apply |
| 520 | the run is on pre-#322 code |

Derived twice independently (`tilerl-25`, `tilerl-0a`) and reconciled at 1327 against the merged
expression: `eval_ctx = max(183, 515) + 2048 + 64 = 2627`, `ceil/16 = 165` per row, `× 8` rows
in flight `= 1320`, `max(1320, 256) + 8 = 1328`. The 1024 floor means the rollout arm is **never
the constraint** in this recipe — `--max-new-tokens` does not size this pool, which is the knob
everyone believed they were turning before #322.

The block count is read off the built engine, **not** confirmed by arithmetic. Four sessions
computed block counts all day on 2026-09-08 and none read the value the engine produced.

**Result of the startup check: 1328, and the criterion was wrong, not the engine.** The pad row
is **added** when the pool is built (`engine.py:1677`, `PagedKvPool(num_blocks + pad, ...)`) and
**subtracted** when it is read (`engine.py:520`), so it cancels: the pool holds 1329 and
`usable_blocks` is 1328, which is the `num_blocks` argument itself. Three sessions independently
derived 1327, all three subtracting the pad once without noticing it had been added — a third
instance of agreeing derivations proving only their shared premise, and the hardest of the three
to catch, because all three derivations were independent, agreed, and cited line numbers.
`tilerl-0a` reports quoting the docstring that states the `+1` while using it to argue the `−1`.

**The row of the table that would have mattered reads 1328 as "decode_graph did not capture",
and that diagnosis was refuted only because `decode_graph: true` sits in the same dict.** Without
that neighbouring field the run would have been stopped to investigate a problem that does not
exist. The general form: **every cell of a criterion table needs an independent field that can
refute it**, or a misdiagnosis reads exactly like a correct one.

**Capacity, per arm rather than mixed.** The first margin computed here was +3.1%, from
`8 × (515 + 2048) = 1288` — MMLU's prompt against GSM8K's cap, two arms that never coexist
because `evals()` runs them in sequence. Per arm:

| arm | rows × (prompt + cap) | blocks |
|---|---|---:|
| MMLU | 8 × (515 + **1**) — `max_new_tokens=1`, `eval.py:85` | 264 |
| GSM8K eval | 8 × (183 + 2048) | **1120** |
| usable pool | | 1328 |

**+18.6% margin on the binding arm, and 1120 is already the worst case** — all 8 rows at the
full cap, so no input can exceed it. The 926-token greedy maximum that looked threatening
earlier today needs 560. The `max(prompt_max, 515)` in `eval_ctx` is a deliberate over-estimate
so the pool covers both arms without knowing which cap pairs with which prompt; the error was
reading that over-estimate as the actual demand.

`max_total_tokens` is 8192, not the computed `ctx` of 2627, because `max(ctx, 8192)` takes the
floor. It costs no memory and it means a long-prompt 2048-cap eval request is not refused by
the per-request limit.

**Anchor check, which takes precedence over reading the curve at all.** Step 100 must land in
**[90.5, 96.7]**: SE at p=0.936, n=500 is **1.09 pt** (not 2.24 — that is the p=0.5 worst case),
and the band is `± 2 × 1.09 × √2` for the difference of two independent measurements. Outside
the band, the first three points have no referent and are not read.

**The band contains eval sampling noise only, and that is now known to be incomplete.** Its
derivation assumes two runs of one configuration at one seed land on the same score, with the
difference coming from which 500 rows were scored. The restart refuted that assumption
directly: two attempts of this exact configuration at `--seed 0` agreed on the first three
steps' rollouts and diverged from step 4. So a step-100 score carries a **trajectory variance
term whose magnitude is unmeasured**, and the band is therefore too narrow by an unknown amount.

The band is **not widened** — a threshold moved after the data is not a threshold, and there is
no measurement to widen it by. What changes is the disposition when a point falls outside, which
now has a fourth candidate ranked first:

| | candidate | evidence today |
|---|---|---|
| **0** | trajectory divergence between two runs of one configuration; magnitude unknown | **the two logs** — direct, and the only one with any |
| 1 | a replayed decode graph served stale weights | none; tested by re-running the after-arm from the saved adapter at `decode_graph=False` |
| 2 | data order — the anchor ran a different `--data` file | none; check the two files' `file_hash` and level histograms |
| 3 | GPU non-determinism | none; smallest term, well under a point on a 500-row greedy eval |

Quantifying candidate 0 needs three repeats of one configuration, ~2 hours, and is **not being
run**: tonight's object is the first curve, not a curve with error bars. The consequence is that
an out-of-band step 100 is **not evidence of a defect** until candidate 0 is excluded, and
candidate 1's check is what excludes the one mechanism that would be.

Ranking proposed by `tilerl-27`, which set the original band and revised it on this reading.

**Saturation.** Adjacent points differing by **< 1.00 pt** and both ≥ 90.5. That is the
**paired** width — every point scores the same `curve_rows`, so the comparison is paired
(McNemar at 5% discordant), and the unpaired 1.90 pt would read a real 1.5 pt rise as noise.
This is why each point writes `eval-curve-<step>.jsonl`: the pairing is only recoverable if
which problem went which way is on disk, and P1 fell back to the unpaired interval for want of
exactly those rows.

**Predictions, recorded before the data.** `tilerl-27` predicts step 25 ≥ 93.0, from the anchor
run's segmented tied fraction (steps 1-10: 0.50, 11-20: 0.50, 21-35: 0.87) — tied rising means
groups are all-correct, which is a symptom of the score already topping out. This session
predicts step 25 ∈ [90, 93), because the jump is at 21 rather than a climb from 11, and a
plateau-then-jump reads as a threshold being crossed while a score topping out is asymptotic.
Four points on a line means both are wrong, and that is the most informative outcome: under
λ=0 tied can only mean all-correct-or-all-wrong, so a decoupled tied jump would leave
within-group correlation as the only explanation — a fact about the data rather than about
training.

## What the curve records that a score cannot

Each point carries `mean_len`, `at_cap` and (in the log line) `tok/correct`. The anchor run
moved tokens/correct **394.0 → 143.8 (2.74x)** while accuracy moved 88.0 → 93.6 (**+6%**) —
most of what that RL bought was shorter answers. A curve read on score alone records that as
the rate of learning to be right.

**One gap this run does not close.** `--allow-short-rollouts` suppresses the second guard, which
is what writes `rollout_window_mean`, so the manifest carries no record of whether the rollouts
drifted into their 256 cap. `mean_len` does **not** substitute for it: that is the eval's length
(greedy, cap 2048) and this is the rollout's (temperature 1.0, cap 256) — same name, same units,
different quantity. The rollout trajectory is recoverable after the run from
`_write_rollout_rows`' per-completion rows, which the flag does not affect.

Also: GSM8K has no rollout length distribution on the correct code path in this tree. The one
measured earlier today is at temperature 1.0 through `render_chat` (mean 322.0, p90 532, 1.3% at
a 1024 cap, 384 rollouts over 3 seeds) and a peer's long-tail probe was voided for encoding the
bare question. This run's eval lengths are the first on the production path.

## Results

**The headline: `steps_to_score` at X=91.0 is `(0, 5]`, so 5 steps against 100 is **≥17.4x** on
training seconds — 119.0 s against 2066.7 s.** The project has measured `seconds_per_step` many
times and this is the first measurement of the left factor. **It is a lower bound**: the crossing
happened somewhere in (0, 5] and 5 is only the first point measured, so a finer grid can only
raise it.

That it is a bound and not an estimate is the load-bearing part. The same X on the same run read
**≥3.85x** an hour earlier, from the coarse grid's step 25; the fine grid replaced 25 with 5 and
the figure moved **4.5x** without any measurement being wrong. A crossing read off a grid is
always an upper bound on the step and therefore a lower bound on the ratio.

### The fine curve, `--eval-every 5` to step 20

Its own run (`30f3186c48f8`, sha `bd72288`), so its base arm is re-measured rather than shared:

| point | score | mean tok | at cap | train s | eval s |
|---:|---:|---:|---:|---:|---:|
| base | 87.6% (438/500) | 348.9 | 3 | — | — |
| **step 5** | **94.2%** (471/500) | 250.2 | 5 | **119.0** | 1103.6 |
| step 10 | 94.2% (471/500) | 261.3 | 8 | 217.9 | 1134.8 |
| step 15 | **94.6%** (473/500) | 213.9 | 4 | 334.8 | 942.4 |
| step 20 | 92.8% (464/500) | 203.7 | 6 | 459.4 | 924.6 |

| pair | wrong→right | right→wrong | net | paired SE | σ |
|---|---:|---:|---:|---:|---:|
| base → 5 | 38 | 5 | **+6.60 pt** | 1.31 | **5.03** |
| 5 → 10 | 5 | 5 | **+0.00 pt** | 0.63 | 0.00 |
| 10 → 15 | 6 | 4 | +0.40 pt | 0.63 | 0.63 |
| 15 → 20 | 8 | 17 | −1.80 pt | 1.00 | 1.80 |
| base → 20 | 37 | 11 | +5.20 pt | 1.39 | 3.75 |

**Steps 5 through 20 are one plateau.** Under the 2×SE rule (a new point counts as better only
if it beats the incumbent by twice the paired SE) no adjacent pair is distinguishable: +0.00
against 1.26, +0.40 against 1.26, −1.80 against 2.00. **The whole +6.60 pt arrives by step 5**,
and steps 5-20 add nothing measurable.

**s5 → s10 is net exactly zero on different problems** — 5 right→wrong and 5 wrong→right, 2.0%
discordant. The same 471 count is not the same 471 problems. A score-only reader sees a flat line
and concludes "nothing happened"; the rows say ten problems changed hands. Worth recording
because it bounds what a repeated score can tell you: **equal scores are not evidence of equal
policies**, and only the per-problem rows separate them.

### Two axes, and they must not be folded together

| step | score | train s | tok/correct |
|---:|---:|---:|---:|
| **5** | 94.2% | **119.0** | 265.6 |
| 25 (coarse run) | 93.2% | 537.4 | **126.4** |

Training is **4.5x cheaper** at step 5; inference is **2.1x more expensive**. `time_to_score` is
defined on the score alone, so the headline takes step 5 and `tok/correct` is recorded beside it
rather than folded in — a composite objective nobody defined is the thing that gets decomposed
wrongly next time. Same form as reporting `held`/`dipped_at` next to `reached` instead of
choosing between them. (`tilerl-27` set this split; the caveat that "cheapest wins on a tie" needs
*other things equal*, and here they are not, is why the two tables are separate.)

### The score and the length are two processes, separated in time

| step | mean tok | at cap |
|---:|---:|---:|
| base | 348.9 | 3 |
| 5 | 250.2 | 5 |
| 10 | 261.3 | 8 |
| 15 | 213.9 | 4 |
| 20 | 203.7 | 6 |
| 25 (coarse) | 117.8 | 0 |

**The score is finished by step 5 and the length compression has barely started.** At step 5 the
policy still answers at 250 tokens against the base's 349 (1.4x), and only by step 25 is it at
117.8 (2.9x). `at_cap` tracks it: base 3, still 4-8 through step 20, and 0 at step 25 — the policy
has not yet learned to be short enough to stop being truncated.

So the two effects are **not two faces of one process**. An earlier draft of this entry called the
length collapse "the main effect" of base→25 because both appeared in that one interval; the fine
grid separates them. Score: steps 1-5. Length: steps 5-25.

### The eval cost is linear in generated tokens

| point | eval s | mean tok | s/tok |
|---|---:|---:|---:|
| coarse s25 | 614.3 | 117.8 | 5.21 |
| coarse s50 | 741.4 | 153.1 | 4.84 |
| coarse s75 | 639.3 | 122.8 | 5.21 |
| coarse s100 | 753.6 | 150.7 | 5.00 |
| fine s5 | 1103.6 | 250.2 | 4.41 |

**1.18x spread across a 2.1x range of lengths**, so a curve point's cost is set by how long the
policy answers, not by the row count. That inverts the intuition about which grid is cheap: the
fine curve's points are the *most* expensive ones on the whole curve, because early-step policies
answer longest. Its four points cost **68 min of eval against 7.7 min of training — 90%
instrument**.

### The coarse curve, `--eval-every 25` to step 100

| point | score | net vs base | mean tok | tok/correct | cumulative train s | eval s |
|---:|---:|---:|---:|---:|---:|---:|
| base | 87.4% (437/500) | — | 346.5 | 396.5 | — | — |
| step 25 | **93.2%** (466/500) | **+5.80 pt** | 117.8 | 126.4 | 537.4 | 614.3 |
| step 50 | **93.4%** (467/500) | +6.00 pt | 153.1 | 164.0 | 1038.6 | 741.4 |
| step 75 | **82.4%** (412/500) | **−5.00 pt** | 122.8 | 149.0 | 1555.7 | 639.3 |
| step 100 | **91.2%** (456/500) | +3.80 pt | 151.4 | 165.3 | 2066.7 | 753.6 |

All five arms score the same 500 problems, so every comparison is paired (McNemar over
`eval-curve-<step>.jsonl`, which #323 puts on disk):

| pair | wrong→right | right→wrong | discordant | net | paired SE | σ |
|---|---:|---:|---:|---:|---:|---:|
| base → 25 | 40 | 11 | 10.2% | **+5.80 pt** | 1.43 | 4.06 |
| 25 → 50 | 10 | 9 | 3.8% | **+0.20 pt** | 0.87 | 0.23 |
| 50 → 75 | 7 | **62** | 13.8% | **−11.00 pt** | 1.66 | **6.62** |
| 75 → 100 | **56** | 12 | 13.6% | **+8.80 pt** | 1.65 | **5.34** |
| 25 → 100 | 10 | 20 | 6.0% | −2.00 pt | 1.10 | 1.83 |
| 50 → 100 | 13 | 24 | 7.4% | −2.20 pt | 1.22 | 1.81 |
| base → 100 | 26 | 7 | 6.6% | **+3.80 pt** | 1.15 | 3.31 |

**The anchor check passes.** Step 100 is 91.2%, inside the pre-registered [90.5, 96.7], so the
first three points have a referent and none of the four candidate explanations is needed.

**Saturation is at step 25 and the criterion decided it cleanly.** Points 25 and 50 differ by
0.20 pt against the registered 1.00 pt, both ≥ 90.5, and that pair's own paired SE is **0.87 pt**
— *below* 1.00, at 3.8% discordance rather than the 10.2% of the base→25 pair. The criterion was
not in its own undecided band.

**But the score does not stay there, and the criterion had no way to say so.** Step 75 loses
11.00 pt at 6.62σ, step 100 recovers 8.80 pt at 5.34σ, and 100 still sits 2.0–2.2 pt below
25 and 50 (1.8σ each — individually inconclusive, jointly consistent with 25/50 being the peak).
So step 75 is a **dip, not a permanent collapse**, and a run of 100 steps ends *worse* than one
stopped at 50.

**Two numbers, not one.**

| quantity | value | meaning |
|---|---|---|
| crossing step at X=91.0 | **(0, 5]**, ≤119.0 s | when the target is first reached; **a lower bound on the ratio** — the coarse grid read (0, 25] and the fine grid moved it 4.5x |
| **best step** | **anywhere in 5..50** — 94.2% / 94.2% / 94.6% / 93.2% / 93.4% at 119.0 / 217.9 / 334.8 / 537.4 / 1038.6 s | where a run should be stopped; no adjacent pair clears 2×SE, so **which is not decided** and the cheapest point on the plateau is step 5 |

Stopping anywhere on the plateau instead of running all 100 is **2.0–17.4x less training time and
+1.6 to +3.4 pt better**. There is **no early stopping in this tree**, and no gate can see the
difference: `reward_rises` reads windowed rollout reward, `gsm8k_improves` reads before against
after, and nothing in the manifest reads a curve for monotonicity. That is worth more than the
17.4x — 17.4x says less training saves time, this says more training damages the result while
every gate reports normal.

**Where the peak is, is undecided, and the reason was measured after the fact.** The fine-curve
run re-ran the base arm in a fresh process — same weights, same 500 rows, same greedy parameters,
`temperature=0.0` — and read **438/500 = 87.6%** against the coarse run's **437/500 = 87.4%**
(174426 tokens against 173249, +0.68%). A greedy eval on identical inputs is supposed to be
reproducible; it moved by **1 problem = 0.2 pt** across processes.

Step 50 (467) beats step 25 (466) by **1 problem**, which is exactly that floor. The paired
comparison already said so — net +0.20 pt at 0.23σ — but it read as "the two are equal and 50 is
nominally higher", and 50 was reported as the best step on that nominal ordering. It should not
have been:

| candidate peak | cumulative train s | ratio vs full run |
|---|---:|---:|
| step 25 | 537.4 | 3.85x |
| step 50 | 1038.6 | 1.99x |
| **step 5** | **119.0** | **17.4x** |

**The choice changes the headline by up to 8.7x**, and the data does not support making it — the
fine grid put three more indistinguishable points on the plateau below 25. What *is* decided is
unaffected: every point from 5 to 50 beats step 100 (456) by 8-17 problems, all of them clear the
462 gate step 100 fails, and all are far outside the 0.2 pt floor. So **"stop before step 75"
holds and "stop at 50" does not** — and by the tie-break rule (equal within resolution, take the
cheaper) the plateau's cheapest point is **step 5 at 119.0 s**.

This also bounds every other number here from below. The floor is ≥0.2 pt on a 500-row greedy
eval across processes, so of the coarse curve's adjacent comparisons only 25→50 (+0.20 pt) sits at
it; the others (+5.80, −11.00, +8.80) clear it by 29x, 55x and 44x, and on the fine curve
base→5 (+6.60) clears it by 33x while 5→10, 10→15 and 15→20 do not clear 2×SE at all. **The
mechanism of the eval floor is unmeasured** — the same fp4 reduction non-determinism that made the
training trajectories diverge is the obvious candidate and has not been tested.

**The training reward did not follow the eval.** Per-step means:

| steps | reward | ce | tied | tok |
|---|---:|---:|---:|---:|
| 1-25 | 0.800 | 3.26 | 0.56 | 161 |
| 26-50 | 0.900 | 4.09 | 0.68 | 136 |
| 51-75 | **0.855** | 3.69 | 0.80 | 157 |

Reward fell 5% over the window that lost 11.8% of greedy accuracy, and 62 problems went
right→wrong against 7 the other way. Not a scoring artefact and not truncation — 0/500 at the
2048 cap in every arm except step 100's 1/500. **The mechanism is unmeasured**: reward held, so
the candidates are the policy drifting off the eval distribution while still satisfying the
reward, or LoRA update instability, and nothing here separates them.

**Length is not monotone.** 346.5 → 117.8 → 153.1 → 122.8 → 151.4. The 2.94x compression at
step 25 is a **minimum, not a trend**; step 50 gives back 30% of it at a flat score. An earlier
draft of this entry called the length collapse "the main effect" — that holds for base→25 and
does not extend past it.

**Predictions, scored.** `tilerl-27` predicted step 25 **≥ 93.0** and was right (93.2). This
session predicted **[90, 93)** and was wrong. Neither prediction covered non-monotonicity, and
the four-point grid is the reason it was seen at all.

**Cost.** Eval is **614–754 s per 500-row point**, 114% of the training it measures at step 25.
Four points cost 45.8 min of eval against 34.4 min of training. Peak allocated 39.27 GiB;
adapter 124.8M params.

**Where the rows are.** The restarted run **reuses the first attempt's id** `86a06dc8c420` —
the id hashes the inputs and the inputs are identical, evidenced by that manifest's `started`
moving 13:05 → 13:50. The newer-looking `0435924d7108` is **another session's synthetic run**
(`source: tiny`, commit `2ea4a1f`, started and finished at 13:09:12 with a full gate set), which
I first misread as the killed attempt's leftover. Two predicates: find the run by the id its
manifest names, not by mtime, **and** confirm a directory is yours before concluding from its
contents.

**P1's own exit criterion failed on this run, and it passes at both step 25 and step 50.** The
manifest's verdict
is FAIL, on `gsm8k_improves`: threshold **462** (base 437 + 25 correct = +5.0 pt on 500 rows),
value **456** — the step-100 policy, 6 short. Against every curve point:

| point | correct | vs 462 |
|---:|---:|---|
| step 25 | 466 | **PASS** (+4) |
| step 50 | 467 | **PASS** (+5) |
| step 75 | 412 | FAIL (−50) |
| step 100 | **456** | **FAIL** (−6) |

**The run failed P1 by training at least 50 steps too long** — 75 if step 25 is the peak, which
the eval floor above leaves open. The gate reads the after-arm, which is the
last step, and the last step is not the best step. This is the same fact as the best-step row
above, arriving through the project's actual exit criterion rather than through a curve nobody's
gate reads: a policy that satisfies P1 existed at step 50, was trained past, and the manifest
records FAIL.

The other five gates: `mmlu_holds` PASS (0.757 against 0.731); `rollouts_within_cap`,
`reward_rises`, `groups_untied` and `ce_falls` all FAIL, and all four are `validity` rather than
`verdict` — expected here, since `--allow-short-rollouts` makes the cap deliberate and
`--length-penalty 0.0` on an already-solved task drives ties to 0.68. They say the run is hard to
interpret, not that P1 failed; `gsm8k_improves` is the one that says that.

**The manifest computes the paired comparison independently and it agrees with mine.**
`gsm8k_paired={'n': 500, 'b': 7, 'c': 26, 'delta': 0.038, 'se': 0.0115, 'z': 3.307}` against my
base→100 of +3.80 pt at 3.31σ from the same rows through different code. Two readings of one
quantity, not one derivation twice.

MMLU after reads 75.7% against 75.1% before, so the regression check holds.

**That MMLU pair does not narrow the step-75 mechanism, and reading it as evidence that general
capability survived is a step-number error.** `evals("after")` runs once, after the loop, so
75.7% is the **step-100** policy — the one that had already recovered to 91.2%. There is no MMLU
reading at step 75, the step where GSM8K was 11 points down. To learn whether the dip was
GSM8K-specific or general, MMLU would have to be scored *inside* the curve, which
`score_curve` does not do (it calls `gsm8k_accuracy` only). Proposed by `tilerl-27` as a
narrowing of the candidates and withdrawn on this reading; it is a real experiment, not a
conclusion available from these numbers.

Points 2-4 pending.
