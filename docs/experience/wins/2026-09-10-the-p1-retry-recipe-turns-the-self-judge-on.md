# The P1 retry recipe turns the self-judge on — 2026-09-10

**Status: pending-remote.** CPU mechanism is tested; the 27B acceptance has not
run. No GPU number in this entry is measured.

## Context

The shipped `grpo-gsm8k-27b` recipe was measured and rejected on 2026-09-10
(run `b4a4e7b23ab8`, GSM8K 91.6% → 78.2%,
[errors/2026-09-10-the-length-term-was-the-gradient.md](../errors/2026-09-10-the-length-term-was-the-gradient.md)):
at a 91% base ~60% of groups are saturated at the correctness level, and inside
a saturated group the normalized advantage reduces to a pure length objective —
all-wrong groups reward the shortest wrong answer, so rollouts collapsed to
14 tokens.

Two fixes already landed in the tree; this change flips the recipe to use them.

## What changed

`grpo-gsm8k-27b` now sets `judge=True` and raises `max_new_tokens` 256 → 512.

1. **`--judge` (stage 4(b), `src/tilerl/judge.py`)** lets the policy rank the
   rollouts the binary reward cannot separate. Tests split pass/fail first; the
   judge is asked only inside the all-pass or all-fail subgroup, in both
   positions, and only the pairwise ORDER survives (Copeland win count mapped
   to evenly spaced scores in `[0.6, 1.0]` / `[0.0, 0.4]`). So the ~60%
   saturated groups carry a within-band gradient again, while no judged order
   can lift a wrong answer above a right one.
2. **The length term never reaches the advantage under `--judge`.**
   `grpo_loop` replaces `rewards` wholesale with judge scores
   (`train.py`, the `tiebreak` branch), so `r = corr − λ·L/cap` is not what is
   normalized. Independent of that, #443 (`29cb812e`) zeroes any group whose
   non-length `signal` is constant, so without the judge the 2026-09-10
   length-only gradient is silent instead of pointing at empty outputs.
3. **`judge` is in the manifest id**: it changes what the reward means, so a
   judge run must not be handed a non-judge run's finished manifest.
4. **Cap 512.** 256 tripped the rollout-length guard (base mean 347 tokens >
   `0.8 × 256`); the rejected run used the guard's own 512 prescription. Eval
   cap stays 2048.
5. **Held-out guard.** `--eval-gsm8k` now refuses any prompt that appears in
   `--data` (checked against the full file before `--eval-n`). Until now every
   CLI plumbing test passed the SAME file to both, so the held-out gate could
   be green at 100% contamination.

## Acceptance on the pod (pre-registered)

A matched seed-0/seed-1 pair of `grpo-gsm8k-27b` on card 0/1/3/6, each passing
`_finish`'s gates on a 500-question held-out set disjoint from training:

- `gsm8k_improves`: after ≥ before + 25/500 on BOTH seeds;
- `mmlu_holds`: after ≥ before − 2 pt on both;
- `groups_untied` < 0.5 with the judge on (saturated groups now ranked, not
  silent) — and if it stays ≥ 0.5 the judge is failing to separate, which is a
  reject of the judge on this model, not a retry;
- no rollout-length collapse: `tokens_last` not below `tokens_first` by the
  2026-09-10 margin (329 → 14).

Cost to watch: the judge issues C(8,2)×2 = 56 one-token generations per step,
both prompt orders concatenated into ONE batched `generate` call. Record
`secs_per_step_median` against the 34.09 s recapture baseline; a judge overhead
above the rollout saving is its own reject signal.

## What this run is predicted to show — and what it is not

Read this before the number comes back, so the verdict is not taken off the
wrong target. At a 91.6% base the groups the judge rescues are almost entirely
ALL-CORRECT (p^8 ≈ 0.50); the all-wrong fraction is negligible. The 8% of
questions the model misses live in MIXED groups (7 right / 1 wrong), where the
binary reward already gives the wrong rollout a negative advantage and the
judge's band gap does not sharpen it. The judge restores a reasoning-STYLE
gradient inside groups that were already right; it does not add a gradient that
flips wrong answers to right ones.

So the mechanism-level prediction is **flat / no collapse** — GSM8K held flat
instead of the 91.6% → 78.2% length-term regression, and rollout length held
instead of 329 → 14 tokens. Clearing the roadmap's +25/500 gate would require
the extra, UNMEASURED hypothesis that ranking better reasoning inside correct
groups transfers onto the harder questions the model currently misses; nothing
in the mechanism guarantees that. The +25/500 gate stays the recorded verdict
(the roadmap criterion is not pre-softened), but a flat, non-collapsed result
is this recipe's success condition even if that gate does not clear.

## Rule

At a high-baseline task the reward must order rollouts WITHIN the saturated
pass/fail bands without crossing them; a shaped term that fills the advantage
in a saturated group measures the shaping, not the method. The recipe carries
the mechanism that does this, and a config that cannot is rejected, not
relaunched with a retuned coefficient.
