"""Attribute the torch glue kernels (copies / index / elementwise) in one decode
forward to their Python call sites: eager forward of N layers under
torch.profiler(with_stack), CUDA time grouped by the innermost tilerl frame.
  python scripts/profile_glue.py /data00/Qwen3.8-27B-NVFP4 --gpu 7 --layers 8"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TILERL_TARGET", "cuda")

    import torch

    import benchkit as bk
    from tilerl.config import qwen38_27b
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.model import load_hf
    from tilerl_kernels.backend import get_backend

    backend = get_backend()
    model = load_hf(qwen38_27b(), args.source, fuse_projections=True, num_layers=args.layers)
    cfg = model.cfg
    engine = build_engine(cfg, model, backend, num_blocks=512, num_slots=16, max_batch=8,
                          max_total_tokens=8192, decode_graph=False)
    for i in range(args.batch):
        engine.submit(bk.rand_prompt(cfg.vocab_size, 512, seed=i),
                      SamplingParams(temperature=0.0, max_new_tokens=4096, seed=i))
    assert bk.settle_decode(engine, args.batch, 8)
    for _ in range(4):
        engine.step()
    torch.cuda.synchronize()

    # profiler stacks come back empty for aten ops on this torch, so attribute in Python.
    import traceback

    from torch.overrides import TorchFunctionMode

    counts = defaultdict(int)

    class Rec(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            site = "(no tilerl frame)"
            for fr in reversed(traceback.extract_stack(limit=25)):
                if "/tilerl/" in fr.filename and "profile_glue" not in fr.filename:
                    site = f"{fr.filename.split('/tilerl/')[-1]}:{fr.lineno} {fr.name}"
                    break
            name = getattr(func, "__name__", str(func))
            if name == "to" and args and isinstance(args[0], torch.Tensor):
                dst = kwargs.get("dtype") if kwargs else None
                dst = dst or next((a for a in args[1:] if isinstance(a, torch.dtype)), None)
                if dst is not None and dst != args[0].dtype:
                    name = f"to[{str(args[0].dtype)[6:]}->{str(dst)[6:]} {tuple(args[0].shape)}]"
                else:
                    name = "to[noop]"
            counts[(name, site)] += 1
            return func(*args, **(kwargs or {}))

    with Rec():
        engine.step()
    torch.cuda.synchronize()
    print(f"\n{args.layers} layers B={args.batch}: {sum(counts.values())} torch calls in one tick\n")
    print(f"{'op':<14} {'site':<70} {'n':>4}")
    for (op, site), n in sorted(counts.items(), key=lambda kv: -kv[1]):
        if op.startswith(("to[", "mul", "add", "__getitem__", "index", "cat", "clone", "contiguous")):
            print(f"{op:<44} {site:<50} {n:>4}")


if __name__ == "__main__":
    main()
