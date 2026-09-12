"""MMLU-Pro base-policy survey: accuracy, mean tokens, bare-letter rate.

Free-form greedy generation, not 1-token scoring: the bare-letter rate is the
fraction of completions that are just the option letter with no reasoning --
the collapse starting point a length-penalty reward would exploit (27, 2026-09-10).

  scripts/mmlu_pro_survey.py --file /work/mmlu_pro_500.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "packages" / "tilerl-kernels" / "src")
)

from ab_draft_depth import _build_model  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

from tilerl.engine import build_engine  # noqa: E402
from tilerl.eval import generate_ids  # noqa: E402
from tilerl.kv_cache import BLOCK_TOKENS  # noqa: E402
from tilerl.prompt import render_chat, sampling  # noqa: E402
from tilerl.server import get_tokenizer  # noqa: E402

LET = "ABCDEFGHIJ"
_BOXED = re.compile(r"\\boxed\{([A-J])\}")
_STANDALONE = re.compile(r"(?<![A-Za-z])([A-J])(?![A-Za-z])")


def final_letter(t: str) -> str:
    m = _BOXED.findall(t or "")
    if m:
        return m[-1]
    m = _STANDALONE.findall(t or "")
    return m[-1] if m else "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--source", default="/work/Qwen3.8-27B-NVFP4")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default="/work/mmlu_pro_survey.jsonl")
    a = ap.parse_args()
    os.environ.setdefault("TILERL_QWEN38_SOURCE", a.source)

    rows = [json.loads(l) for l in open(a.file)][: a.n]
    gold = [r["answer"] for r in rows]
    prompts = [render_chat([("user", r["prompt"])], False) for r in rows]

    be = get_backend()
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    tok = get_tokenizer(a.source)
    # MMLU-Pro prompts run long (question + 10 options + chat wrapper): 2048 prompt + cap.
    ctx = 2048 + a.cap
    need = -(-ctx // BLOCK_TOKENS) * a.concurrency + 8
    e = build_engine(cfg, model, be, num_blocks=need, num_slots=a.concurrency + 1,
                     max_batch=a.concurrency, max_total_tokens=ctx + 64)

    sp = sampling(tok, False, a.cap, temperature=0.0, seed=0)
    per: list = [None] * len(rows)

    def on_row(i: int, ids: list) -> None:
        per[i] = ids

    generate_ids(e, tok, prompts, sp, a.concurrency, on_row=on_row)

    n_correct = n_bare = n_trunc = total_tok = 0
    with open(a.out, "w") as f:
        for i, ids in enumerate(per):
            text = tok.decode([int(t) for t in ids])
            letter = final_letter(text)
            bare = len(text.strip()) <= 2 and text.strip()[:1] in LET
            trunc = len(ids) >= a.cap
            n_correct += letter == gold[i]
            n_bare += bare
            n_trunc += trunc
            total_tok += len(ids)
            f.write(json.dumps({"i": i, "correct": letter == gold[i], "letter": letter,
                                "bare": bare, "tokens": len(ids), "truncated": trunc,
                                "text": text}) + "\n")
    n = len(rows)
    print(f"mmlu_pro: {n_correct}/{n} = {100*n_correct/n:.1f}%  "
          f"mean_tokens={total_tok/n:.0f}  bare_letter={100*n_bare/n:.1f}%  "
          f"truncated={100*n_trunc/n:.1f}%  (cap {a.cap})")


if __name__ == "__main__":
    main()
