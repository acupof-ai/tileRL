# Stop GRPO when its rollout lengths outgrow the cap — cpu, 2026-09-06

> Status: pending-remote (27B)

## Context

[Run 2](../errors/2026-09-06-the-rollouts-grew-into-the-cap.md) passed the
before-eval length check and later reached the rollout cap. The pre-flight
measurement did not constrain the changing policy.

## What Worked

After each completed GRPO step, once five steps exist, compare the last five
steps' mean completion length with 0.8 times the cap. Equality is allowed.
The first crossing stops iteration, saves the adapter and manifest, and skips
the after-eval. The failed `rollouts_within_cap` gate records the window mean,
threshold, step and exit message. `steps_completed` records actual work;
`inputs.steps` remains the requested count.

`--allow-short-rollouts` bypasses both checks. The periodic gate is skipped
when bypassed or when fewer than five steps ran. The flag now participates in
the run ID so a bypass cannot reuse the stopped run's manifest.

The CPU tiny test uses the real GRPO loop with stubbed generation and update.
At cap 20, lengths 6,8,10,12,14,16,18,20,20,20 reach window mean 16 at step 8
and 17.6 at step 9. The guarded run stops at 9; the bypass completes all 10.
Deleting only the periodic check leaves the existing pre-flight CLI test green
and fails this test at `periodic guard stopped at the wrong step` (10 versus 9).
The length term remains open; stopping contains drift and does not fix its cause.

## What it is worth, measured against run 2's trace

**It does not "catch run 2" — a human caught run 2 at step 45, and this fires at
step 44.** Replayed over the 45 recorded steps:

| threshold | fires at | window mean vs limit |
|---:|---:|---|
| 0.60 | 6 | 1244 vs 1229 |
| 0.70 | 23 | 1461 vs 1434 |
| 0.75 | 43 | 1557 vs 1536 |
| **0.80** | **44** | 1883 vs 1638 |

So its value is the **unattended** case: run 2 was watched, and a run that is not
would have burned the remaining 56 steps at a 229 s median — **3.57 h**.

**It lags the damage by 18 steps.** The first floor tie is step 26; the guard
fires at 44, 1.17 h later. No threshold fixes that: 0.75 buys one step, and 0.70
fires at 23, before any damage, on a run that was still healthy. The window mean
is a lagging indicator of a distribution that is already broken.

**The direct signal is `tied == 1.0 AND reward == 0.0`**, available at step 26 —
exactly the five floor ties (26, 32, 41, 43, 44). It cannot be `tied` alone: 14
of the 45 steps are `tied == 1.0` with `reward == 1.0`, which is a group solving
its problem, not failing it. That signal belongs to the length-term work, not
here; this guard bounds the wasted hours and nothing else.

**The guard reads the group mean; the length bucket keys on the group max.** Two
different statistics of the same rollouts, deliberately: stopping is about the
central tendency drifting, padding is about the widest row in the batch.

**When it fires the after-arm is skipped, so `mmlu_holds` and `gsm8k_improves`
have no value.** `_finish` scores a `None` value as passed, which would print
two green gates on a run that measured neither — so a guard stop marks them
`skipped: True, passed: None`, and the FAIL rests on `rollouts_within_cap` alone.
`groups_untied` and `ce_falls` are still real; `reward_rises` is computed over
the truncated history.

## Rule

Recheck a changing policy's completion length during training and persist the
reason for stopping before returning a failed exit status.

## Results

27B execution and overhead: pending-remote. CPU check:
`TILERL_TARGET=cpu uv run pytest tests/test_ledger.py tests/test_eval_rows.py`.
