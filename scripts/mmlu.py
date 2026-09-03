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
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tilerl.eval import LETTERS, letter, mmlu_questions, mmlu_score  # noqa: E402
from tilerl.tokenizer import get_tokenizer  # noqa: E402

#: concurrency is part of the score: it sets B, B sets M = B*W, and M picks the
#: fp4 linear arm, so two values can run two kernels on one question. cli.py's
#: eval used 8 and this used the default 32; they are one number now.
CONCURRENCY = 8


def score_tilerl(source: str, prompts: list[str], slots: int = 64, blocks: int = 2048) -> list[str]:
    from tilerl_kernels.backend import get_backend

    from tilerl.config import qwen38_27b
    from tilerl.engine import build_engine
    from tilerl.model import load_hf

    model = load_hf(qwen38_27b(), source, fuse_projections=True)
    engine = build_engine(model.cfg, model, get_backend(), num_blocks=blocks, num_slots=slots,
                          max_batch=8, max_total_tokens=8192)
    return mmlu_score(engine, get_tokenizer(source), prompts, CONCURRENCY)


def accuracy(source: str, n: int = 200, seed: int = 0, slots: int = 64,
             blocks: int = 2048) -> tuple[int, int]:
    """bench_harness's accuracy gate."""
    prompts, golds, _ = mmlu_questions(n, seed)
    preds = [letter(t) for t in score_tilerl(source, prompts, slots=slots, blocks=blocks)]
    return sum(p == g for p, g in zip(preds, golds)), len(preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["tilerl", "sglang"], required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slots", type=int, default=64, help="GDN state slots (reduce on <40GB GPUs)")
    ap.add_argument("--blocks", type=int, default=2048, help="KV pool blocks")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    prompts, golds, subjects = mmlu_questions(args.n, args.seed)
    t0 = time.time()

    if args.engine == "tilerl":
        os.environ.setdefault("TILERL_TARGET", "cuda")
        texts = score_tilerl(args.source, prompts, slots=args.slots, blocks=args.blocks)
    else:
        import sglang

        llm = sglang.Engine(model_path=args.source, trust_remote_code=True, mem_fraction_static=0.85)
        # argmax over letter tokens' logprobs: sglang's regex FSM splits the first token
        outs = llm.generate(prompts, {"temperature": 0, "max_new_tokens": 1},
                            return_logprob=True, top_logprobs_num=20)
        tok = get_tokenizer(args.source)
        letters = ({tok.encode(f" {c}")[-1]: c for c in LETTERS}
                   | {tok.encode(c)[-1]: c for c in LETTERS})
        texts = []
        for o in outs:
            top = o["meta_info"]["output_top_logprobs"][0]
            hits = [(lp, letters[tid]) for lp, tid, *_ in top if tid in letters]
            texts.append(max(hits)[1] if hits else o["text"])
        llm.shutdown()

    elapsed = time.time() - t0
    preds = [letter(t) for t in texts]
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
    Path(out).write_text(json.dumps({"pred": preds, "gold": golds, "acc": correct / len(preds),
                                     "engine": args.engine, "source": args.source, "n": len(preds),
                                     "seed": args.seed,
                                     "concurrency": CONCURRENCY if args.engine == "tilerl" else None,
                                     "raw": texts[:50]}))


if __name__ == "__main__":  # sglang spawns its scheduler with multiprocessing: no top-level work
    main()
