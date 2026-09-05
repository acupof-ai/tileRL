"""Evals through a running engine — the before/after gates of a training run.
MMLU 0-shot: one letter per question, argmax over the four letter ids (the
lm-eval convention). GSM8K: greedy under the training's own template, exact
match of the last number."""

from __future__ import annotations

import os
import random
import re
from typing import Any

from .math_answer import boxed_match

LETTERS = "ABCD"
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def last_number(text: str | None) -> float | None:
    found = _NUMBER.findall(text or "")
    return float(found[-1].replace(",", "")) if found else None


def answer_match(text: str | None, answer: str) -> bool:
    got, want = last_number(text), last_number(answer)
    return got is not None and want is not None and abs(got - want) < 1e-6


def generate_ids(engine: Any, tok: Any, prompts: list[str], sp: Any,
                 concurrency: int) -> list[list[int]]:
    """One completion per prompt, as the ids the engine emitted.

    Split out from ``generate`` because the token COUNT is a result, not
    bookkeeping: output tokens are the serving bill, and length is the one signal
    ``--judge`` cannot fake. Re-encoding the decoded text would answer a slightly
    different question -- decode/encode is not always a round trip.
    """
    out: list = [None] * len(prompts)
    pending, todo = {}, list(enumerate(prompts))
    while pending or todo:
        while todo and len(pending) < concurrency:
            i, p = todo.pop()
            pending[engine.submit(tok.encode(p), sp)] = i
        engine.step()
        for wid, ids in engine.poll().items():
            out[pending.pop(wid)] = ids
    return out


def generate(engine: Any, tok: Any, prompts: list[str], sp: Any, concurrency: int) -> list[str]:
    """One decoded completion per prompt; ``concurrency`` <= the engine's state slots."""
    return [tok.decode(ids) for ids in generate_ids(engine, tok, prompts, sp, concurrency)]


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
                  concurrency: int = 32, questions: Any = None,
                  per_problem: list | None = None) -> tuple[int, int, int]:
    """(correct, total, concurrency) on the slice ``seed`` picks.

    ``concurrency`` is returned because it is part of the score, not of how the
    score was obtained: it sets the batch size, the batch size sets ``M``, and
    ``M`` picks the fp4 linear arm across the ``_MGEMV``/``_MX`` boundaries, so
    two concurrencies can run two kernels on the same question. Measured on the
    27B, two arms at concurrency 8 agree on 1000 of 1000 while a concurrency-32
    run of the same slice differs on 4. Callers disagreed (``cli.py`` 8,
    ``scripts/mmlu.py`` the default) and nothing recorded which was used."""
    prompts, golds, _ = questions if questions is not None else mmlu_questions(n, seed)
    preds = [letter(t) for t in mmlu_score(engine, tok, prompts, concurrency)]
    if per_problem is not None:
        per_problem.extend({"dataset": "mmlu", "i": i, "correct": p == g,
                            "answer": g, "prediction": p, "tokens": 1}
                           for i, (p, g) in enumerate(zip(preds, golds)))
    return sum(p == g for p, g in zip(preds, golds)), len(preds), concurrency


#: `--reward` name -> the one matcher that scores it. The training reward and the
#: eval both read this, so they cannot disagree -- they did once, the reward on
#: \boxed{} and the eval on the last number, which scores `\frac{3}{2}` correct
#: against `\frac{1}{2}`. A new reward adds a row and cannot ship half-wired.
MATCHERS = {"number": answer_match, "boxed": boxed_match}


def gsm8k_accuracy(engine: Any, tok: Any, rows: list[dict], sampling: Any,
                   concurrency: int = 8, thinking: bool | None = None,
                   match: Any = None, per_problem: list | None = None,
                   ) -> tuple[int, int, int]:
    """(correct, total, completion tokens) on ``rows`` ({prompt, answer}), greedy
    under ``sampling``.

    ``match`` defaults to ``answer_match`` (last number). On MATH it must be
    ``math_answer.boxed_match``: last_number reads `\\frac{3}{2}` and `\\frac{1}{2}`
    as the same answer, 2, so a wrong rollout scores correct -- measured. The
    training reward already switched on ``--reward boxed`` while this did not.

    ``per_problem`` receives one dict per row. Two arms over the same problems can
    only get a paired interval if which problem went which way was written down;
    P1's GSM8K comparison fell back to the wider unpaired one for want of it.

    ``sampling`` must NOT be the rollout's: at the training cap this scores the cap
    rather than the policy (38.4% with mean completion 238.7 against a 256 cap, and
    ~82.5% uncapped -- errors/2026-09-04-the-eval-cap-measured-itself.md). ``cmd_train``
    builds a separate ``eval_params`` from ``--eval-max-new-tokens`` for exactly this.

    The token total is here because a training-step length says what the rollouts did
    under the training config, and this says what the policy became. Length claims
    about a trained policy have to be scored here, not on the rollouts."""
    from dataclasses import replace

    from .prompt import render_chat

    scorer = match or answer_match
    prompts = [render_chat([("user", r["prompt"])], thinking) for r in rows]
    ids = generate_ids(engine, tok, prompts, replace(sampling, temperature=0.0), concurrency)
    texts = [tok.decode(i) for i in ids]
    hits = [bool(scorer(t, r["answer"])) for t, r in zip(texts, rows)]
    if per_problem is not None:
        per_problem.extend(
            {"i": i, "correct": h, "tokens": len(d), "answer": r["answer"]}
            for i, (h, d, r) in enumerate(zip(hits, ids, rows)))
    return sum(hits), len(rows), sum(len(i) for i in ids)
