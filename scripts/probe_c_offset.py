"""How many arguments reach Backend._c already contiguous but at a non-zero offset?

Those are the ones the metal fix clones. CUDA tolerates the offset, so the clone is
charged only to metal -- this counts what each target would owe if it were not.

Not written to /tmp: a script there puts /tmp on sys.path[0] and shadows the stdlib.

  TILERL_TARGET=cuda python scripts/probe_c_offset.py           # decode tick
  TILERL_TARGET=metal python scripts/probe_c_offset.py --draft  # + the draft path
"""

import argparse
import collections

import torch
from tilerl_kernels.backend import Backend, get_backend

STATS: collections.Counter = collections.Counter()
SHAPES: collections.Counter = collections.Counter()


def _counting(self, t: torch.Tensor) -> torch.Tensor:
    """Backend._c with a tally; clones exactly when the shipped helper would."""
    if t.is_contiguous():
        STATS["contig"] += 1
        if t.storage_offset():
            STATS["contig_offset"] += 1
            SHAPES[(tuple(t.shape), str(t.dtype), t.storage_offset())] += 1
            return t.clone() if self._zero_offset else t
        return t
    STATS["noncontig"] += 1
    return t.contiguous()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=200)
    a = ap.parse_args()

    Backend._c = _counting
    from tilerl.config import tiny
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.model import build_random

    be = get_backend()
    cfg = tiny()
    eng = build_engine(cfg, build_random(cfg, seed=3), be,
                       num_blocks=64, num_slots=4, max_batch=4, max_total_tokens=512)
    eng.submit([1, 2, 3, 4, 5, 6, 7, 8], SamplingParams(max_new_tokens=8))
    STATS.clear()
    SHAPES.clear()
    for _ in range(a.ticks):
        eng.step()
        if eng.poll():
            break

    print(f"target {be.target} device {be.device} zero_offset={be._zero_offset}")
    print(f"  contiguous, offset==0 : {STATS['contig'] - STATS['contig_offset']}")
    print(f"  contiguous, offset!=0 : {STATS['contig_offset']}   <-- what metal clones")
    print(f"  non-contiguous        : {STATS['noncontig']}   (copied on every target)")
    for (shape, dtype, off), n in SHAPES.most_common(10):
        print(f"    x{n:<4d} shape={shape} dtype={dtype} offset={off}")


if __name__ == "__main__":
    main()
