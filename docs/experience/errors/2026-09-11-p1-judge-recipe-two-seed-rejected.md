# P1 judge recipe: the two-seed run is rejected — seed 0 regresses GSM8K, 2026-09-11

**Status: verdict — REJECT.** The retry recipe (judge on, cap 768) removed the
2026-09-10 length collapse but does not improve held-out GSM8K on the
pre-registered two-seed criterion. This entry records the paired two-seed
outcome, including the one seed that cleared.

## Context

`grpo-gsm8k-27b` with `--judge`, `--max-new-tokens 768` (raised from 512 in
the amendment: the 0.8 rollout guard stopped seed 0 at step 19 under 512),
two seeds in parallel on H20 cards 6 and 7, tree 5e2e5794, 100 GRPO steps,
500 held-out GSM8K + 1000 MMLU evaluated per arm at the 2048 eval cap.
Acceptance was pre-registered on BOTH seeds: GSM8K after−before ≥ +25/500
(paired McNemar), MMLU after ≥ before − 2 pt, tied group fraction < 0.5, no
rollout-length collapse.

## The two seeds

| seed | run | GSM8K before→after | paired flips b/c (right→wrong / wrong→right) | McNemar z | MMLU before→after | tied | tokens first→last |
|---:|---|---:|---|---:|---:|---:|---:|
| 0 | 0622261a06f9 | 458 → 448 (−10) | 34 / 24 | −1.31 | 0.751 → 0.817 | 0.21 | 266.8 → 281.2 |
| 1 | fcf91735cf1c | 456 → 482 (+26) | 6 / 32 | +4.22 | 0.751 → 0.739 | 0.41 | 298.0 → 242.2 |

`b`/`c` are the manifest's `_mcnemar` fields: `b` = right→wrong, `c` =
wrong→right (`delta = (c - b)/n`). Seed 0 loses because 34 answers went right
to wrong and only 24 reversed; seed 1 gains because 32 flipped wrong to right
against 6 lost.

The paired test (manifest `gsm8k_paired`) confirms what the counts say: seed 1
is a real gain (32 wrong→right vs 6 right→wrong, z ≈ 4.2), seed 0 is a
non-significant loss (34 right→wrong vs 24 wrong→right, |z| 1.3).

## The gate is "both seeds"; one pass is a reject

- `gsm8k_improves`: seed 0 FAIL (448 < 458+25=483); seed 1 PASS by one question
  (482 ≥ 481). **Both were required, so the recipe is rejected.**
- `mmlu_holds`: PASS both (0.817 and 0.739 against the 0.731 floor).
- `groups_untied`: PASS both (0.21, 0.41 < 0.5) — the judge did break the
  saturated groups open, the mechanism it was added for.
- No length collapse on either seed (token means hold or fall slightly); the
  768 cap kept the guard silent for all 100 steps (`rollouts_within_cap`:
  292.1 / 271.0 mean against a 614.4 threshold).

## Accuracy split by truncated vs finished (2048 eval cap)

Required before launch; cap-bound rows score 0.

| seed | arm | truncated @2048 | truncated correct | finished | truncated-acc | finished-acc |
|---:|---|---:|---:|---:|---:|---:|
| 0 | before | 3 | 1 | 497 | 0.333 | 0.920 |
| 0 | after | 0 | 0 | 500 | — | 0.896 |
| 1 | before | 4 | 0 | 496 | 0.000 | 0.919 |
| 1 | after | 2 | 1 | 498 | 0.500 | 0.966 |

Seed 1's +26 is in the finished majority and is not a truncation-composition
artifact (2 after-cap rows, one correct). Seed 0's after arm has zero
truncations yet loses 10 — the loss is in finished answers.

## Why this is a reject and not a mixed result

This matches the entry's own pre-run prediction: at a 91% base the judge
rescues a reasoning-STYLE gradient inside already-correct all-correct groups;
it cannot, with any guarantee, transfer onto the ~8% the model misses. The
result shows the gradient direction is seed-dependent at this run size — one
seed's style change helps the hard tail (+26, significant), the other's hurts
it (−10, not significant). A method whose sign flips between matched seeds
does not clear a two-seed acceptance. MMLU rose on the losing seed and the
judge/tied mechanisms worked, so the harness is healthy; the claim the recipe
made (held-out GSM8K gains on both seeds) is what failed.

## Rule

A training recipe accepted on "both seeds" is rejected by the losing seed even
when the other seed clears, because a sign that flips across matched runs is
not an effect. The self-judge solved the failure it was built for (saturated
groups, length collapse) and still did not deliver the downstream gain —
fixing the reward's within-group ordering is necessary for a signal, not
sufficient for a transfer.
