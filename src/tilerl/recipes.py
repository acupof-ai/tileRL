"""Recipes: one name per configuration that passed a gate.

A recipe is the full flag set for ``tilerl train``; explicit flags override
it, and the manifest records which recipe a run started from. ``status``
names the gate it passed — a recipe that has not run on its target says
``pending-remote`` and the CLI prints that before training.
"""

from __future__ import annotations

RECIPES: dict[str, dict] = {
    "grpo-tiny-smoke": dict(
        model="tiny", rl=True, steps=2, group=2, max_new_tokens=4, lora_rank=4,
        status="cpu: tests/test_recipes.py"),
    # docs/roadmap.md P1. Pass --data gsm8k_train.jsonl --eval-gsm8k gsm8k_test.jsonl.
    "grpo-gsm8k-27b": dict(
        model="qwen38-27b", rl=True, steps=100, group=8, max_new_tokens=256, lora_rank=16,
        think_budget=0, eval_mmlu=1000, eval_n=500,
        status="pending-remote: roadmap P1"),
    "opd-gsm8k-27b": dict(
        model="qwen38-27b", opd=True, steps=100, max_new_tokens=256, lora_rank=16,
        eval_mmlu=1000, eval_n=500,
        status="pending-remote: roadmap P1"),
    # docs/roadmap.md P3, the SFT half: full-parameter ISO vs Adafactor.
    "sft-iso-27b": dict(model="qwen38-27b", optim="iso", steps=100,
                        status="pending-remote: roadmap P3"),
}


def flags(name: str) -> dict:
    """The recipe's flags, without its status."""
    return {k: v for k, v in RECIPES[name].items() if k != "status"}
