"""Sweep (min blocks per SM, grid) pairs; the cooperative launch caps the grid.

Each config compiles with a register cap derived from min_blocks, so the grid can
hold more warps in flight. GDN's state spills instead; linear is 97% of the time.
"""

import argparse
import json
import os

import torch
from tilerl_kernels.backend import Backend
from tilerl_kernels.kernels_mega import make_megakernel
from tilerl_kernels.mega import MegaForward

from tilerl.config import qwen38_27b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--configs", default="2:160,4:320,8:640,12:800")
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()
    os.environ["TILERL_SM70_NATIVE_FP8"] = "1"
    from tilerl_kernels.mega_loader import load_sm70

    cfg, backend = qwen38_27b(), Backend("cuda")
    model = load_sm70(cfg, args.checkpoint, backend)
    mega = MegaForward(model, backend, 1, max(args.context + 1, 16))
    mega.position.fill_(args.context)
    filler = next(r for r in mega.rows if r[0] == 8)
    linear_only = torch.tensor([r if r[0] == 2 else filler for r in mega.rows],
                               dtype=torch.int64, device=backend.device)

    def measure(prog, reps):
        best = float("inf")
        for _ in range(reps):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            mega.kernel(prog, mega.pointers, mega.position)
            end.record()
            torch.cuda.synchronize()
            best = min(best, start.elapsed_time(end))
        return best

    for config in args.configs.split(","):
        min_blocks, grid = (int(v) for v in config.split(":"))
        mega.kernel = make_megakernel(grid, len(mega.rows), len(mega.buffers), min_blocks=min_blocks)
        try:
            print(json.dumps({"min_blocks": min_blocks, "grid": grid,
                              "all_ms": measure(mega.program, args.reps),
                              "linear_only_ms": measure(linear_only, 1)}), flush=True)
        except Exception as error:  # a grid beyond the resident limit is a data point
            print(json.dumps({"min_blocks": min_blocks, "grid": grid,
                              "error": str(error).splitlines()[-1]}), flush=True)


if __name__ == "__main__":
    main()
