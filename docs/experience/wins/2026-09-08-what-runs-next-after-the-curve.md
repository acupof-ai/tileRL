# What runs next after the curve: four commands, written before the data

**Date:** 2026-09-08
**Session:** v100-sm70-fp4-55
**For run:** `curve2`, card 3, claim `tilerl-curve2`, pid 1170264, sha `fd0c876`, log `/work/curve2.log`

Written while a first attempt was running, and revised when that attempt was **restarted**: it
was launched from `12da5a0`, which predates #323, so its curve points carried neither
`mean_len` nor `at_cap` nor the per-problem rows. Read off the running process rather than
inferred — `curve.append` at `cli.py:831` in the pod tree had five fields and no `at_cap`,
`grep at_cap` returned nothing, and `/proc/1150334/cwd` plus `tilerl.cli.__file__` both
resolved to that tree. `tilerl-27` asked the question that caught it: the sha was confirmed
and what the sha contained was not.

The restart cost ~15 min, not the ~40 first estimated: the before-arm eval is a **cache hit**
(`runs/eval-cache/68256c15….json`, and `pod_sync.sh:49`'s wipe excludes `./runs`), so the
MMLU-1000 + GSM8K-500 arm does not re-run; only the 24 completed steps replay, at 23.3 s each.

**Those steps did NOT replay identically, and that refutes what this document first said.**
Measured, both logs side by side at the same step numbers:

| steps | rollouts (`tok`, `reward`) | `ce` |
|---|---|---|
| 1-3 | **identical** (250/256/248, 0.75/0.00/0.125) | **differs** — step 1 is 1.5197 vs 1.5380, 1.2% |
| 4 onward | diverged (step 4: 116 vs 128 tokens, reward 0.875 vs 1.000) | diverged |

So `--seed 0` fixes the prompt order and the sampling seeds — the first rollouts really are the
same tokens — but the **training math is not reproducible across processes**, and once one
optimizer step differs the trajectories separate. The code is not the cause:
`git diff 12da5a0..fd0c876 -- src` touches only `cli.py`'s logging and curve fields plus
`recipes.py` comments, nothing in `train.py`, `model.py` or `backend.py`.

**The mechanism is not pinned.** One verified non-code difference between the two runs is a
candidate: the first attempt's before-arm was a cache **miss** and ran MMLU 1000 + GSM8K 500 on
the card before step 1 (`curve.log` carries the `gsm8k greedy` and `mmlu 0-shot` lines,
`curve2.log` carries neither), so the two processes entered training with different allocator
and cache histories. Whether that reaches the arithmetic is untested, and plain
non-determinism in the fp4 reductions is the competing explanation. Naming which would need a
same-process repeat, which no plan here requires.

Consequence for branches A and B below: their step-25 score is **an independent sample of the
same configuration, not a refinement of `curve2`'s first point**, and it cannot be used as a
same-trajectory check. Two points 25 steps apart from different processes differ by training
noise plus this, and neither document may treat their agreement as verification.

Each branch below names one command, complete, and the reason it is that command. The
pre-registration this reads against is
[the curve's own entry](2026-09-08-the-gsm8k-reward-vs-step-curve.md).

## The step time is 23.3 s, not 56.88 s — every estimate below uses the measured one

Read off the **first attempt's** log at steps 1-24: **23.3 s/step mean** (min 15.6, max 35.2;
step 1 carries JIT). The anchor's 56.88 s/step is **2.44x** this, and the pre-registered
non-alignment (`decode_graph=True` here, `False` in the anchor) is the leading candidate —
**a candidate, not a verdict**: the two runs differ in the graph, the length penalty and the
short-rollout flag, so a single-variable attribution needs the arm that holds the other two.
The restarted run is the same configuration on `fd0c876`, so this figure carries over; it is
re-read from `curve2.log` before any plan below is launched.

Any plan priced at 56.88 s/step overestimates its training time by 2.44x. That is the whole
reason this table exists before the data: a follow-up sized against the wrong divisor either
asks for a card it does not need or is rejected as too expensive.

The step time is **strongly length-dependent** — 15.6 s at 86 tokens, 25.0 s at 233 — so a
follow-up whose rollouts run longer costs more per step than this run's mean. Rollout is
6.2-15.4 s of the 15.6-25.0 s, and the rest (fwd 2.6, bwd 6.7, optimizer 0.08) is flat.

**Eval cost per curve point is not yet measured.** It is the quantity `eval_secs` records
(#309) and it lands with the first curve point at step 25. The before-arm was a cache MISS on
the first attempt (`eval_before_cache.cache_hit: false`, read from the manifest) and covers
MMLU 1000 + GSM8K 500, so it prices neither a curve point nor a GSM8K-500-only arm. **Every
eval figure below is therefore an estimate, marked as one**, from the 09-05 anchor's 16.4 min
for a 500-row GSM8K arm.

## The four branches

### A — step 25 ≥ 93.0: saturation is inside 25 steps

Narrow "≤25" to a specific step. `--eval-every 5`, stop at 25.

```
scripts/pod_run.sh curve5 3 -- python3 -u -m tilerl.cli train \
  --recipe grpo-gsm8k-27b --length-penalty 0.0 --allow-short-rollouts \
  --steps 25 --eval-every 5 --eval-curve-n 500 --eval-max-new-tokens 2048 \
  --eval-mmlu 0 \
  --eval-gsm8k /work/p1_gsm8k_test.jsonl --data /work/p1_gsm8k_train.jsonl \
  --seed 0 --force
```

Cost: 25 × 23.3 s = **9.7 min training** + 5 curve points + 2 arm evals. At the anchor's
16.4 min per 500-row eval that is ~115 min of eval against 10 min of training, so **this run
is ~92% instrument** and its real cost is decided by a number the `curve2` run is about to
measure. Re-price it from step 25's `eval_secs` before launching.

`--eval-mmlu 0` is the one deliberate change from `curve2`'s configuration: MMLU is a
regression check on the after-arm and this run's question is entirely about the GSM8K curve's
shape. It saves the 1000-question arm twice. It also makes the `mmlu_holds` gate vacuous
(`cli.py:1083` sets `mmlu_floor = None` when `mmlu_before is None`) — acceptable here because
the follow-up is a measurement, not a P1 gate attempt, and **it must be said in the entry
rather than discovered from the manifest**.

`--seed 0` unchanged, so steps 1-25 draw the same prompts in the same order as `curve2`:
`train.py:481` takes `prompts[step % len(prompts)]` and seeds each rollout
`seed + step * group + g`. That fixes the **inputs**, not the trajectory — measured above, the
two attempts of this same configuration agreed on the first three steps' rollouts and diverged
from step 4. So the fine curve is **an independent sample of the same configuration**, and its
step 25 agreeing with `curve2`'s step 25 is a consistency reading, not a verification: a
disagreement of a few points is expected and does not mean the fine curve measured something
else.

### B — step 25 ∈ [90, 93): saturation is between 25 and 50

```
scripts/pod_run.sh curve10 3 -- python3 -u -m tilerl.cli train \
  --recipe grpo-gsm8k-27b --length-penalty 0.0 --allow-short-rollouts \
  --steps 50 --eval-every 10 --eval-curve-n 500 --eval-max-new-tokens 2048 \
  --eval-mmlu 0 \
  --eval-gsm8k /work/p1_gsm8k_test.jsonl --data /work/p1_gsm8k_train.jsonl \
  --seed 0 --force
```

Cost: 50 × 23.3 s = **19.4 min training** + 5 curve points. Same eval-dominance as A. Its step
25 and 50 are independent samples, not reproductions, for the reason given under A.

### C — the four points are near-linear: no saturation inside 100 steps

**No fine curve.** A finer grid resolves *where* a curve bends; it adds nothing to a curve
that does not bend. The card goes to `tilerl-0a`'s two-arm `--prompts-per-step` comparison
extended to 100 steps — that is a different question (does a wider step buy anything) and it
is the one a non-saturating curve leaves open.

Release the claim and tell 0a the card is free:

```
python3 /work/aupai/scripts/card_claim.py release --name tilerl-curve2
```

Under C there is **no follow-up run from this session tonight**, which is written here so the
card does not sit idle waiting for a plan that was never going to exist.

### D — the anchor fails: step 100 ∉ [90.5, 96.7]

Not a run. A checklist, in the pre-registered order, each item with the reading that settles
it rather than the hypothesis it tests.

**Candidate 0 comes first, and it is the only one with evidence.** Two attempts of this exact
configuration at `--seed 0` diverged from step 4 (the two logs), so a step-100 score carries a
trajectory variance term the band does not contain — the band is eval sampling noise only. Its
magnitude is unmeasured and is **not being measured**: three repeats of one configuration is
~2 hours. So an out-of-band step 100 is **not evidence of a defect** until this is excluded,
and item 1 is what excludes the one mechanism that would be.

1. **`decode_graph=True`, the known non-alignment.** Its wall-clock effect is now measured
   (23.3 vs 56.88 s/step) and its score effect is meant to be nil. The check is not "is the
   graph on" — the manifest says `decode_graph: true` and that is expected. It is whether a
   replay served stale weights: run the after-arm eval a second time from the saved adapter
   (`--load-adapter`) in a fresh process with `decode_graph=False`. Same score → the graph is
   exonerated for the score. Different score → the replay is the cause and `_const_f32` is
   **not** the mechanism (it holds no LoRA tensors, `backend.py:525,593`), so the next
   suspect is another cached address, not that cache.
2. **Data order.** `train.py:481` is `prompts[step % len(prompts)]` with no shuffle, so the
   trajectory is a deterministic function of the file order. The anchor ran a different
   `--data` file (levels 3-5 mixed vs level-5-only, `recipes.py:26-33`). Check: the two
   files' `file_hash`, and the level histogram of each. A different training distribution is
   a sufficient explanation and does not need a bug.
3. **GPU non-determinism.** Last, because it is the smallest term: it moves a 500-row greedy
   eval by well under a point, and the band is ±6.2.

**Before any of the three:** confirm the base arm. This run read **87.4% (437/500)** against
the anchor's 88.0 — 0.6 pt apart, inside the 2.1 pt SE of that difference, so the base is
already reproduced and a step-100 miss is about training, not about the eval protocol. If a
future run's base arm moves instead, the protocol is the first suspect and this list does not
apply.

## One reading from the live log that changes how C is diagnosed

Counted over the first 17 steps: **10 of 17 (58.8%) report `tied 1.00`**, and under
`--length-penalty 0.0` a tie means every rollout scored identically, so those steps produced
**no gradient**. They split two ways, and the split is the point:

| shape | steps | reading |
|---|---:|---|
| `reward 1.0000  tied 1.00` — all 8 correct | **7** | the policy has solved that prompt; saturation |
| `reward 0.0000  tied 1.00  tok 256` — all 8 at the cap | **3** | the cap truncated before the answer; instrument |

**41% of the no-gradient steps are the cap, 59% are the task being solved.** Those are
opposite conclusions from the same `tied` reading, which is why `tied` alone does not settle
branch C — it needs the reward alongside it. The 3 cap-shaped steps are the expected
consequence of the pre-registered condition (greedy mean 346 tokens against a 256 rollout
cap) and are why the run carries `--allow-short-rollouts`.

So a flat curve has two readings a score column cannot separate:

- the policy has saturated (nothing left to learn), or
- the policy received no gradient on most steps (nothing was learned).

**`tied` is necessary but not sufficient**, per the split above: `tied 1.00` with reward 1.0
is saturation and `tied 1.00` with reward 0.0 at the cap is the instrument. Both fields have
to travel with the point. `tilerl-0a` is adding `tied` to the curve dict (its `1a49186`);
that plus the point's existing `at_cap` gives branch C both halves, so under C read them from
0a's change rather than re-deriving from the log.

If the *reward-1* tied fraction over steps 1-100 is high, the score has saturated and the
curve is answering its question. If the *reward-0-at-cap* fraction is high, the honest
statement is **"this configuration cannot answer where the score saturates, because those
steps had no gradient"**, and the follow-up is a bigger `--max-new-tokens`, not a finer grid.
That is a fifth branch the original three did not have. At 17 steps the split is 7 and 3, so
neither reading is yet dominant.
