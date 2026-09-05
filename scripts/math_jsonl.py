"""Dump hendrycks MATH to the JSONL `tilerl train --data` reads: {"prompt", "answer"},
answer = the contents of the solution's last \\boxed{}. On the pod: HF_ENDPOINT=https://hf-mirror.com.

  python scripts/math_jsonl.py train /work/math_train.jsonl --level 3,4,5 --seed 0
  python scripts/math_jsonl.py test /work/math_test.jsonl --n 500 --seed 0

`--level` is why this exists. GSM8K at group 8 ties 87% of GRPO groups at the
ceiling by step 35 (errors/2026-09-03-p1-ties-at-the-ceiling.md), so the task has
to be hard enough that a group still disagrees with itself. Level is the knob:
raise it if the tie fraction at step 10 is not well under 0.5.

Rows whose solution has no \\boxed{} are dropped, not given an empty answer -- a
row nothing can score is a permanent tie in every group it lands in.

The prompt carries the boxing instruction, because nothing downstream adds one:
`render_chat` (prompt.py:18) emits only ChatML with the turn left open, and the
recipe runs thinking off. Without the instruction a no-think model boxes nothing,
`boxed_match` returns False for every rollout, and every group ties at the FLOOR
-- which the --level knob cannot fix, because it is not a difficulty problem.
It lives here rather than in the renderer so the JSONL is self-contained: the
same file scores the same way through `--data`, `--eval-gsm8k` and a re-score.
"""

import argparse
import json
import random

from datasets import load_dataset

from tilerl.math_answer import extract_boxed

_INSTRUCTION = "\n\nPut your final answer in \\boxed{}."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["train", "test"])
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--level", default="", help="comma-separated 1-5; empty = all")
    ap.add_argument("--seed", type=int, help="shuffle before slicing; makes --n unbiased")
    args = ap.parse_args()

    ds = load_dataset("EleutherAI/hendrycks_math", "all", split=args.split)
    keep = {f"Level {x.strip()}" for x in args.level.split(",") if x.strip()}
    rows = []
    dropped = 0
    for r in ds:
        if keep and r["level"] not in keep:
            continue
        ans = extract_boxed(r["solution"])
        if ans is None:
            dropped += 1
            continue
        rows.append({"prompt": r["problem"].strip() + _INSTRUCTION, "answer": ans})

    order = list(range(len(rows)))
    if args.seed is not None:
        random.Random(args.seed).shuffle(order)
    if args.n:
        order = order[: args.n]
    with open(args.out, "w") as f:
        for i in order:
            f.write(json.dumps(rows[i]) + "\n")
    print(f"{args.out}: {len(order)} rows, level={args.level or 'all'}, seed={args.seed}, "
          f"{dropped} dropped for no \\boxed{{}}")


if __name__ == "__main__":
    main()
