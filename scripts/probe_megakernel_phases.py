"""Attribute megakernel time by opcode with same-shape program swaps.

The instruction count is a compile-time constant, so every variant below reuses
the compiled kernel and only swaps the Program tensor's contents. The all-filler
variant is the grid-barrier floor: it runs the same number of T.sync_grid calls
with no real work.
"""

import argparse
import json
import os

import torch
from tilerl_kernels.backend import Backend
from tilerl_kernels.mega import MegaForward

from tilerl.config import qwen38_27b

NAMES = {0: "embed", 1: "norm", 2: "linear", 3: "silu", 4: "add",
         5: "rope", 6: "attention", 7: "gdn", 8: "slice", 9: "kv_write"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()
    os.environ["TILERL_SM70_NATIVE_FP8"] = "1"
    from tilerl_kernels.mega_loader import load_sm70

    cfg, backend = qwen38_27b(), Backend("cuda")
    model = load_sm70(cfg, args.checkpoint, backend)
    mega = MegaForward(model, backend, 1, max(args.context + 1, 16))
    mega.position.fill_(args.context)

    counts = {}
    for row in mega.rows:
        counts[NAMES[row[0]]] = counts.get(NAMES[row[0]], 0) + 1
    print(json.dumps({"instructions": len(mega.rows), "counts": counts}), flush=True)

    filler = next(r for r in mega.rows if r[0] == 8)

    def program(keep):
        rows = [r if r[0] in keep else filler for r in mega.rows]
        return torch.tensor(rows, dtype=torch.int64, device=backend.device)

    def measure(prog):
        best = float("inf")
        for _ in range(args.reps):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            mega.kernel(prog, mega.pointers, mega.position)
            end.record()
            torch.cuda.synchronize()
            best = min(best, start.elapsed_time(end))
        return best

    every = set(NAMES)
    for label, keep in (
        ("all", every),
        ("all_minus_attention", every - {6}),
        ("all_minus_gdn", every - {7}),
        ("all_minus_linear", every - {2}),
        ("linear_only", {2}),
        ("attention_only", {6}),
        ("gdn_only", {7}),
        ("barrier_floor", set()),
    ):
        print(json.dumps({"variant": label, "ms": measure(program(keep))}), flush=True)


if __name__ == "__main__":
    main()
