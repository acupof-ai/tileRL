"""What holds the memory during a gen-4096 training FORWARD, layer by layer?

The gen-4096 OOM is at `model.py:481 forward`, inside a checkpointed MLP body, on a
290 MiB allocation matching one MLP intermediate `[T, intermediate_size]` f32 exactly
(errors/2026-09-06-chunked-attention-was-the-wrong-term.md). `autograd.checkpoint` should
keep a segment's activations from coexisting, so per-layer intermediates ought to die on
exit -- something else accumulates.

A tensor size times a layer count is NOT a live-bytes figure: what the tape retains is a
property of the handler. `checkpoint` records `args = (layer_idx, x, kv, backend)`, so each
layer's INPUT survives by construction; whether 64 of them are simultaneously live is the
measurement, not the arithmetic.

Every number here is read, never derived:

* `torch.cuda.memory_allocated()` at each checkpoint boundary -- the quantity the OOM ran
  out of.
* a census of live cuda tensors by shape at the peak, so a rise is attributed to a shape.

The tape MUST be recording or `checkpoint` takes its early return (autograd.py:41) and
this measures an unheckpointed forward while claiming otherwise. Asserted on the tape's
recorded entries, NOT on a call count: `checkpoint` is called either way, so the count is
the same in both arms (measured on tiny: 2 calls, 2 entries with recompute, 0 without).

  scripts/pod_run.sh fwdmem 6 -- /work/tl013/bin/python -u \\
      scripts/prof_forward_memory.py --gen 4096
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "packages" / "tilerl-kernels" / "src")
)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

from tilerl import autograd as ag  # noqa: E402
from tilerl import model as model_mod  # noqa: E402
from tilerl.cli import _build_model  # noqa: E402
from tilerl.train import _training_kv  # noqa: E402


def _gib(n: float) -> float:
    return n / 2**30


def _mib(n: float) -> float:
    return n / 2**20


def _both(label: str) -> tuple[int, int]:
    """memory_allocated AND the driver's used bytes. The first read 0.070 GiB before a
    27B forward, which cannot include the weights: fp4 planes served outside torch's
    allocator do not appear in it, so only DELTAS of that number are quotable."""
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated()
    free, total = torch.cuda.mem_get_info()
    print(f"# {label}: torch-allocated {_gib(alloc):.3f} GiB, "
          f"device used {_gib(total - free):.3f} of {_gib(total):.3f} GiB")
    return alloc, total - free


def _live_by_shape() -> Counter:
    live: Counter = Counter()
    seen = set()
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                st = obj.untyped_storage()
                key = (st.data_ptr(), st.nbytes())
                if key in seen:  # views share storage; count the bytes once
                    continue
                seen.add(key)
                live[tuple(obj.shape)] += st.nbytes()
        except (ReferenceError, RuntimeError):
            continue
    return live


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=4096)
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--source", default="")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--every", type=int, default=8, help="print every Nth checkpoint")
    a = ap.parse_args()
    if a.source:
        os.environ.setdefault("TILERL_QWEN38_SOURCE", a.source)

    be = get_backend()
    if be.device.type != "cuda":
        raise SystemExit("reads torch.cuda.memory_allocated; run on cuda")
    cfg, model = _build_model(a.model, seed=0, fuse_projections=True)

    t = a.prompt_tokens + a.gen
    ids = np.random.default_rng(0).integers(3, cfg.vocab_size, size=(1, t)).astype(np.int64)
    kv = _training_kv(model, 1, t, device=be.device)

    hits = {"n": 0}
    rows: list[tuple[int, int, int]] = []
    real = ag.checkpoint

    def traced(fn, *args):
        hits["n"] += 1
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        out = real(fn, *args)
        torch.cuda.synchronize()
        rows.append((hits["n"], before, torch.cuda.memory_allocated()))
        return out

    ag.checkpoint = traced
    model_mod.autograd.checkpoint = traced

    hidden_b = t * cfg.hidden_size * 4
    inter_b = t * cfg.intermediate_size * 4
    print(f"# {a.model} gen={a.gen} T={t} layers={cfg.num_layers} "
          f"hidden={cfg.hidden_size} intermediate={cfg.intermediate_size}")
    print(f"# one [T,hidden] f32 = {_mib(hidden_b):.1f} MiB; "
          f"one [T,intermediate] f32 = {_mib(inter_b):.1f} MiB "
          f"(the 290.00 MiB the OOM asked for)")
    base, _ = _both("before the forward")

    tape = ag.Tape()
    with torch.no_grad(), tape:
        model.forward(ids, np.arange(t, dtype=np.int64), kv,
                      ag.RecordingBackend(be))
        peak_in_tape, _ = _both("at forward end, inside the tape")
        live = _live_by_shape()

    # The load-bearing assertion, on the TAPE and not on the call count: `checkpoint` is
    # still called without an active tape, it just returns early at autograd.py:41, so
    # counting invocations passes on a forward that was never checkpointed. Measured on
    # tiny: 2 calls either way, but 2 tape entries with recompute and 0 without.
    recorded = sum(1 for e in tape._entries if getattr(e, "op_name", None) == "checkpoint")
    assert recorded == cfg.num_layers, (
        f"{recorded} checkpoint entries on the tape for {cfg.num_layers} layers "
        f"({hits['n']} calls): the segments were not recorded, so these numbers describe "
        f"an unheckpointed forward"
    )

    print(f"\n# {'ckpt':>5} {'before GiB':>11} {'after GiB':>10} {'delta MiB':>10}")
    for n, b, aft in rows[:: a.every]:
        print(f"  {n:>5} {_gib(b):11.3f} {_gib(aft):10.3f} {_mib(aft - b):10.1f}")
    first, last = rows[0], rows[-1]
    grew = last[2] - first[2]
    print(f"\n# segment 1 ended at {_gib(first[2]):.3f} GiB, segment {last[0]} at "
          f"{_gib(last[2]):.3f} GiB -> accumulated {_gib(grew):.3f} GiB over "
          f"{len(rows)} segments")
    print(f"# mean rise per segment {_mib(grew / max(len(rows) - 1, 1)):.1f} MiB "
          f"against {_mib(hidden_b):.1f} MiB for one retained [T,hidden] input")

    print(f"\n# forward cost, as a DELTA (the only quotable form): "
          f"{_gib(peak_in_tape - base):.3f} GiB")
    print(f"# torch peak this process: {_gib(torch.cuda.max_memory_allocated()):.3f} GiB")
    total = sum(live.values())
    print(f"# live cuda tensors: {len(live)} distinct shapes, {_gib(total):.3f} GiB "
          f"(storage-deduped)")
    print("# per-shape TOTALS over all live tensors of that shape, not single tensors")
    print(f"# {'shape':>30} {'GiB':>8} {'share':>7} {'copies':>7}")
    for shape, nbytes in sorted(live.items(), key=lambda kv: -kv[1])[:15]:
        one = 1
        for d in shape:
            one *= d
        print(f"  {str(shape):>30} {_gib(nbytes):8.3f} {nbytes / total * 100:6.1f}% "
              f"{nbytes / (one * 4):7.1f}")


if __name__ == "__main__":
    main()
