"""Evals through a running engine — the before/after gates of a training run.
MMLU 0-shot: one letter per question, argmax over the four letter ids (the
lm-eval convention). GSM8K: greedy under the training's own template, exact
match of the last number."""

from __future__ import annotations

import os
import random
import re
from typing import Any

LETTERS = "ABCD"
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def last_number(text: str | None) -> float | None:
    found = _NUMBER.findall(text or "")
    return float(found[-1].replace(",", "")) if found else None


def answer_match(text: str | None, answer: str) -> bool:
    got, want = last_number(text), last_number(answer)
    return got is not None and want is not None and abs(got - want) < 1e-6


def generate(engine: Any, tok: Any, prompts: list[str], sp: Any, concurrency: int) -> list[str]:
    """One decoded completion per prompt; ``concurrency`` <= the engine's state slots."""
    texts: list = [None] * len(prompts)
    pending, todo = {}, list(enumerate(prompts))
    while pending or todo:
        while todo and len(pending) < concurrency:
            i, p = todo.pop()
            pending[engine.submit(tok.encode(p), sp)] = i
        engine.step()
        for wid, ids in engine.poll().items():
            texts[pending.pop(wid)] = tok.decode(ids)
    return texts


def mmlu_questions(n: int, seed: int = 0) -> tuple[list[str], list[str], list[str]]:
    """The fixed 0-shot MMLU slice (cais/mmlu test): (prompts, gold letters, subjects)."""
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = sorted(random.Random(seed).sample(range(len(ds)), n) if n < len(ds) else range(len(ds)))

    def prompt(r):
        subj = r["subject"].replace("_", " ")
        ch = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(r["choices"]))
        return (f"The following is a multiple choice question about {subj}.\n\n"
                f"{r['question'].strip()}\n{ch}\nAnswer:")

    return ([prompt(ds[i]) for i in idx], [LETTERS[ds[i]["answer"]] for i in idx],
            [ds[i]["subject"] for i in idx])


def letter(t: str | None) -> str:
    """First standalone A-D in the completion."""
    m = re.search(r"\b([ABCD])\b", t or "")
    return m.group(1) if m else "?"


def mmlu_score(engine: Any, tok: Any, prompts: list[str], concurrency: int = 32) -> list[str]:
    """One greedy letter per prompt."""
    from .engine import SamplingParams

    allowed = tuple(sorted({tok.encode(f" {c}")[-1] for c in LETTERS}
                           | {tok.encode(c)[-1] for c in LETTERS}))
    sp = SamplingParams(temperature=0.0, max_new_tokens=1, seed=0, allowed_ids=allowed)
    return generate(engine, tok, prompts, sp, concurrency)


def mmlu_accuracy(engine: Any, tok: Any, n: int, seed: int = 0,
                  concurrency: int = 32) -> tuple[int, int]:
    """(correct, total) on the fixed slice."""
    prompts, golds, _ = mmlu_questions(n, seed)
    preds = [letter(t) for t in mmlu_score(engine, tok, prompts, concurrency)]
    return sum(p == g for p, g in zip(preds, golds)), len(preds)


def gsm8k_accuracy(engine: Any, tok: Any, rows: list[dict], sampling: Any,
                   concurrency: int = 8, thinking: bool | None = None) -> tuple[int, int]:
    """(correct, total) on ``rows`` ({prompt, answer}), greedy under the
    training's own ``sampling`` (stop ids, length, no-think template)."""
    from dataclasses import replace

    from .tokenizer import render_chat

    prompts = [render_chat([("user", r["prompt"])], thinking) for r in rows]
    texts = generate(engine, tok, prompts, replace(sampling, temperature=0.0), concurrency)
    return sum(answer_match(t, r["answer"]) for t, r in zip(texts, rows)), len(rows)
