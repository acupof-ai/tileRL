"""Recipes: named ``tilerl train`` flag sets; ``status`` is the gate each passed
(``pending-remote`` until it has run on its target). Explicit flags override."""

from __future__ import annotations

RECIPES: dict[str, dict] = {
    # The gate-passing settings, measured 4/4 seeds: at max_new_tokens 4 every
    # group ties whatever the reward's shape (errors/2026-09-03-tied-groups-are-
    # the-rewards-shape.md), and 2 steps of a per-step reward compares two draws
    # rather than two policies. Same shape as tests/test_rl.py's passing loop.
    "grpo-tiny-smoke": dict(
        model="tiny", rl=True, steps=12, group=6, max_new_tokens=8, lora_rank=4,
        # 8 tokens is below any real completion; the length guard is for real runs.
        lr=0.05, allow_short_rollouts=True, status="cpu: tests/test_recipes.py"),
    # docs/roadmap.md P1. Pass --data gsm8k_train.jsonl --eval-gsm8k gsm8k_test.jsonl.
    # lr: the CLI default of 1e-3 flattens the reward from step 9 on; 1e-4 does not.
    # eval_max_new_tokens 2048 is the protocol the published before/after numbers
    # were scored under; it must not follow max_new_tokens, which caps the rollouts.
    "grpo-gsm8k-27b": dict(
        model="qwen38-27b", rl=True, steps=100, group=8, max_new_tokens=256, lora_rank=16,
        micro=1, max_think_tokens=0, eval_mmlu=1000, eval_n=500, lr=1e-4,
        eval_max_new_tokens=2048,
        status="pending-remote: roadmap P1"),
    # GSM8K is solved: 88.0% uncapped base, so 81% of groups tie at the ceiling
    # (wins/2026-09-05-p1-grpo-27b-run.md). TRAINING data is level 5 only (2304 rows,
    # verified against the parquet shards -- and re-verified 09-08 by reading the source's
    # own level histogram, which gives 2304 for train Level 5 exactly). The EVAL file used
    # by run 2 is levels 3-5 mixed (165/157/178), so its 80.0% base is not a level-5
    # number -- both arms score the same file so the delta holds, it is just measured on
    # an easier set than the one trained on. A level-5-only eval file can now be built
    # (`--level 5`), which it could not before: the generator asked for a config the
    # dataset does not have (errors/2026-09-08-a-generator-that-never-ran.md).
    # max_new_tokens 2048, not 512: the base policy's mean completion is 1029 tokens
    # (measured on that mixed file, n=500), so a 512 cap truncates every rollout before
    # the \boxed{} and 5 of the first 6 steps tied at the FLOOR with reward 0.
    # eval_max_new_tokens is explicit for the same reason gsm8k's is: scoring at the
    # 512 rollout cap would measure the cap (errors/2026-09-04-the-eval-cap-measured-itself.md).
    # 2048 was MEASURING the cap on level 5, and the correction is 27 points. Measured 09-08 on
    # the level-5 file, n=100: 64/100 at cap 2048, but 32 completions hit 2048 and 0 of those 32
    # were correct, while 64 of the 68 that terminated were. Longest natural completion 1889, so
    # 1889-2048 is empty and those 32 were truncated mid-derivation, not wrong. Rerunning exactly
    # those 32 at 6144 (runs-l5c/2898b40d2130): 27 correct, 2 wrong, 3 still at the new cap. So
    # the base is 91/100 = 91.0% and the interval is [91%, 94%], not the 64.0% first reported --
    # a cap that scores truncation as wrong yields a LOWER bound, never the value
    # (errors/2026-09-08-a-cap-reported-as-a-base.md).
    #
    # Two consequences, both open and neither a cap question:
    # * P1 wants base+5 = 96%, which is 2 pt ABOVE the 94% this cap can produce, so no `after`
    #   value passes at 6144. Raise the cap, or change the criterion.
    # * level 5 was chosen for being harder than GSM8K, which failed P1 by being solved at 88.0%.
    #   At 91.0% it is EASIER. The 24-pt difficulty gap was the cap.
    # And the cap is not free: mean generation 1386 -> 3331 tokens, 2.40x, on the denominator of
    # the throughput target. Do not read a tied-group fraction under 2048 either -- 32% constant
    # all-wrong inflates it by an unattributable amount.
    "grpo-math-27b": dict(
        model="qwen38-27b", rl=True, steps=100, group=8, max_new_tokens=2048, lora_rank=16,
        micro=1, max_think_tokens=0, reward="boxed", eval_mmlu=1000, eval_n=500, lr=1e-4,
        eval_max_new_tokens=2048,
        status="pending-remote: roadmap P1, GSM8K's successor task"),
    "opd-gsm8k-27b": dict(
        model="qwen38-27b", opd=True, steps=100, max_new_tokens=256, lora_rank=16,
        eval_mmlu=1000, eval_n=500, eval_max_new_tokens=2048,
        status="pending-remote: roadmap P1"),
    # docs/roadmap.md P3, the SFT half: full-parameter ISO vs Adafactor.
    "sft-iso-27b": dict(model="qwen38-27b", optim="iso", steps=100,
                        status="pending-remote: roadmap P3"),
}


def flags(name: str) -> dict:
    """The recipe's flags, without its status."""
    return {k: v for k, v in RECIPES[name].items() if k != "status"}
