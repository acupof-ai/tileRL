"""Sweep the megakernel grid size; linear GEMV is the phase under test.

One model load, one kernel compile per grid size. Results flush per config so a
cooperative-launch deadlock at a large grid still leaves the smaller ones.
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
    parser.add_argument("--grids", default="80,160,320,640")
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

    for grid in [int(g) for g in args.grids.split(",")]:
        mega.kernel = make_megakernel(grid, len(mega.rows), len(mega.buffers))
        print(json.dumps({"grid": grid, "all_ms": measure(mega.program, args.reps),
                          "linear_only_ms": measure(linear_only, 1)}), flush=True)


if __name__ == "__main__":
    main()
