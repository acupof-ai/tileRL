"""Dump GSM8K to the JSONL `tilerl train --data` reads: {"prompt", "answer"},
answer = the number after '####'. On the pod: HF_ENDPOINT=https://hf-mirror.com.

  python scripts/gsm8k_jsonl.py train /work/gsm8k_train.jsonl [--n 512]
  python scripts/gsm8k_jsonl.py test /work/gsm8k_test_full.jsonl --seed 0

`--n` takes a PREFIX, and the test split is ordered: measured on the 27B, rows
0-199 score 84.5% against 93.0% for rows 200-499 (z=3.05, p=0.0023). So a
prefix is a biased sample -- pass `--seed` to shuffle first, and record it.
"""

import argparse
import json
import random

from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("split", choices=["train", "test"])
ap.add_argument("out")
ap.add_argument("--n", type=int, default=0, help="0 = all")
ap.add_argument("--seed", type=int, help="shuffle before slicing; makes --n unbiased")
args = ap.parse_args()
ds = load_dataset("openai/gsm8k", "main", split=args.split)
order = list(range(len(ds)))
if args.seed is not None:
    random.Random(args.seed).shuffle(order)
rows = ds.select(order[: args.n] if args.n else order)
with open(args.out, "w") as f:
    for r in rows:
        f.write(json.dumps({"prompt": r["question"].strip(),
                            "answer": r["answer"].split("####")[-1].strip()}) + "\n")
print(f"{args.out}: {len(rows)} rows, seed={args.seed}")
