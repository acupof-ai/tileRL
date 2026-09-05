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
        lr=0.05, status="cpu: tests/test_recipes.py"),
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
    # (wins/2026-09-05-p1-grpo-27b-run.md). Sampled on the 27B before this run:
    # levels 3-5 score 75% (6/8), level 5 alone 45.8% (11/24) -- so LEVEL 5 ONLY.
    # eval_max_new_tokens is explicit for the same reason gsm8k's is: scoring at the
    # 512 rollout cap would measure the cap (errors/2026-09-04-the-eval-cap-measured-itself.md).
    "grpo-math-27b": dict(
        model="qwen38-27b", rl=True, steps=100, group=8, max_new_tokens=512, lora_rank=16,
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
