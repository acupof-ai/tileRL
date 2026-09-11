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

## Amendment 2026-09-11: rollout cap 512 → 768 before the two-seed run

The seed-0 launch (`1feadc1ee6ee`, tree `ae1cfc1c`, card 6) did not crash and
did not time out: the rollout-length guard stopped it at **step 19 of 100** —
the last-5 mean was **454.3 completion tokens**, over the guard threshold
`_ROLLOUT_HEADROOM = 0.8` × 512 = **409.6** (`src/tilerl/cli.py:662`; the
guard's own message prescribes a cap above 568). The rollouts were growing
into the 512 cap, not collapsing.

The relaunch raises `--max-new-tokens` **512 → 768** (guard threshold then
614.4). It does NOT pass `--allow-short-rollouts` and changes nothing else in
the recipe; eval cap stays 2048.

19-step evidence kept verbatim (per-step log / FAIL row):
`reward_first 0.8063 → reward_last 0.4281` while
`tokens_first 291.6 → tokens_last 446.2` — reward falling as completions grow
is the length-drift signature the guard exists to stop a misread of;
`ce_last 0.7848`, `tied_group_fraction 0.3684`, `tied_correctness 0.6842`,
`secs_per_step_median 97.47`, `peak_gib 45.43`, steps 15–19 each at 487–512
tokens.

Baselines (verbatim from that run): **MMLU before 0.751** (751/1000),
**GSM8K before 0.916** (458/500).

Acceptance is unchanged, required on BOTH seeds: GSM8K after−before ≥ +25/500
(+5 pt) on the paired held-out set, reported with a paired McNemar test; MMLU
after ≥ before − 2 pt (≥ 0.731); `tied_group_fraction < 0.5`; no rollout-length
collapse (`tokens_last` not below `tokens_first` by the 2026-09-10 margin).

The verdict report must split held-out accuracy **by truncated vs finished**:
a row that ends at the cap scores 0 and is reported as truncated, so a
cap-bound batch cannot print headroom it does not have.

Launch: seed 0 on card 6 and seed 1 on card 7 **in parallel**, each
`--max-new-tokens 768` (~2.7 h/seed), disjoint remote trees; pre-rollout
baselines are re-measured in each process and reported per seed.


## Verdict 2026-09-11: REJECTED — the two-seed run

Both seeds finished 100 steps under cap 768 (guard silent). The pre-registered
gate was GSM8K +25/500 on BOTH seeds; only one cleared.

- seed 0 (`0622261a06f9`): GSM8K 458 → 448 (−10; McNemar b=34/c=24, z −1.31),
  MMLU 0.751 → 0.817, tied 0.21, tokens 266.8 → 281.2.
- seed 1 (`fcf91735cf1c`): GSM8K 456 → 482 (+26; b=6/c=32, z +4.22, clears
  481 by one), MMLU 0.751 → 0.739, tied 0.41, tokens 298.0 → 242.2.

MMLU held on both, tied fraction stayed under 0.5 on both, and neither seed
collapsed in length — so the judge did its mechanism job (broke saturated
groups open, no length regression). What failed is the downstream claim: the
held-out GSM8K sign flips between matched seeds. Truncated/finished split at
the 2048 eval cap: seed 1's gain is in finished answers (after: 2 truncated,
1 correct; 498 finished at 0.966), seed 0's after arm has 0 truncations yet
loses 10, so neither arm is a cap-composition artifact. Full table and
analysis: [errors/2026-09-11-p1-judge-recipe-two-seed-rejected.md](../errors/2026-09-11-p1-judge-recipe-two-seed-rejected.md).
