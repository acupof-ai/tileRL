"""Dump GSM8K to the JSONL `tilerl train --data` reads: {"prompt", "answer"},
answer = the number after '####'. On the pod: HF_ENDPOINT=https://hf-mirror.com.

  python scripts/gsm8k_jsonl.py train /work/gsm8k_train.jsonl [--n 512]
"""

import argparse
import json

from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("split", choices=["train", "test"])
ap.add_argument("out")
ap.add_argument("--n", type=int, default=0, help="0 = all")
args = ap.parse_args()
ds = load_dataset("openai/gsm8k", "main", split=args.split)
rows = ds.select(range(args.n)) if args.n else ds
with open(args.out, "w") as f:
    for r in rows:
        f.write(json.dumps({"prompt": r["question"].strip(),
                            "answer": r["answer"].split("####")[-1].strip()}) + "\n")
print(f"{args.out}: {len(rows)} rows")
