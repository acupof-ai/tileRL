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
the band, the first three points have no referent and are not read; difference 3 above is the
first suspect.

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

Appended when the run finishes.
