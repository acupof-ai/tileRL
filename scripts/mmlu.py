"""0-shot MMLU (cais/mmlu test, cached offline) through tileRL's engine or
sglang's offline Engine — same prompts, one greedy token, letter scored.
Saves per-question predictions so two engines can be diffed.

  python scripts/mmlu.py --engine tilerl --source /data00/Qwen3.8-27B-NVFP4 --gpu 7 --n 1000
  PYTHONPATH=/work/sgl-src/python python scripts/mmlu.py --engine sglang --source /work/Qwen3.8-27B-bf16 --gpu 7 --n 1000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["tilerl", "sglang"], required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    from datasets import load_dataset  # noqa: E402

    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = list(range(len(ds)))
    random.Random(args.seed).shuffle(idx)
    idx = sorted(idx[: args.n])
    LETTERS = "ABCD"


    def prompt(r):
        subj = r["subject"].replace("_", " ")
        ch = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(r["choices"]))
        return f"The following is a multiple choice question about {subj}.\n\n{r['question'].strip()}\n{ch}\nAnswer:"


    prompts = [prompt(ds[i]) for i in idx]
    golds = [LETTERS[ds[i]["answer"]] for i in idx]
    subjects = [ds[i]["subject"] for i in idx]
    t0 = time.time()

    if args.engine == "tilerl":
        os.environ.setdefault("TILERL_TARGET", "cuda")
        from tilerl.config import qwen38_27b
        from tilerl.engine import SamplingParams, build_engine
        from tilerl.model import load_hf
        from tilerl.ops.backend import get_backend
        from tilerl.server import get_tokenizer

        backend = get_backend()
        model = load_hf(qwen38_27b(), args.source, fuse_projections=True)
        engine = build_engine(model.cfg, model, backend, num_blocks=2048, num_slots=64, max_batch=8,
                              max_total_tokens=8192)
        tok = get_tokenizer(args.source)
        sp = SamplingParams(temperature=0.0, max_new_tokens=1, seed=0)
        texts = [None] * len(prompts)
        pending, todo = {}, list(enumerate(prompts))
        while pending or todo:
            while todo and len(pending) < 32:  # submit allocates a state slot eagerly
                i, p = todo.pop()
                pending[engine.submit(tok.encode(p), sp)] = i
            engine.step()
            for wid, ids in engine.poll().items():
                texts[pending.pop(wid)] = tok.decode(ids)
    else:
        import sglang

        llm = sglang.Engine(model_path=args.source, trust_remote_code=True, mem_fraction_static=0.85)
        outs = llm.generate(prompts, {"temperature": 0, "max_new_tokens": 1})
        texts = [o["text"] for o in outs]
        llm.shutdown()

    elapsed = time.time() - t0
    preds = [(t.strip()[:1].upper() if t and t.strip() else "?") for t in texts]
    correct = sum(p == g for p, g in zip(preds, golds))
    by = defaultdict(lambda: [0, 0])
    for p, g, s in zip(preds, golds, subjects):
        by[s][0] += p == g
        by[s][1] += 1
    print(f"{args.engine} {args.source}: MMLU 0-shot {correct}/{len(preds)} = {100 * correct / len(preds):.1f}%  "
          f"({elapsed:.0f}s, unparsed {sum(p not in LETTERS for p in preds)})")
    worst = sorted(by.items(), key=lambda kv: kv[1][0] / kv[1][1])[:5]
    print("  weakest:", ", ".join(f"{s} {c}/{n}" for s, (c, n) in worst))
    out = args.out or f"/work/mmlu_{args.engine}.json"
    Path(out).write_text(json.dumps({"idx": idx, "pred": preds, "gold": golds, "acc": correct / len(preds),
                                     "engine": args.engine, "source": args.source, "n": len(preds)}))


if __name__ == "__main__":  # sglang spawns its scheduler with multiprocessing: no top-level work
    main()
