"""Every host transfer in one tick, attributed to its call site.

Counts scalar syncs (``aten._local_scalar_dense``: ``.item()``/``int(t)``, seen by a
dispatch mode) AND bulk ones (``.tolist()``/``.numpy()``, invisible to dispatch, caught by
wrapping the Tensor methods). The CPU target takes the same torch fallbacks a GPU without
the fused kernels takes, so this enumerates that arch's transfers without one.

    uv run python scripts/probe_syncs.py --phase decode|prefill|train
"""

from __future__ import annotations

import argparse
import collections
import traceback

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def _site() -> str | None:
    for f in reversed(traceback.extract_stack()):
        if "/tilerl" in f.filename and "probe_syncs" not in f.filename:
            return f"{f.filename.split('/')[-1]}:{f.lineno} {f.name}"
    return None


class Probe(TorchDispatchMode):
    def __init__(self) -> None:
        self.scalar: collections.Counter = collections.Counter()
        self.bulk: collections.Counter = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if str(func) == "aten._local_scalar_dense.default" and (s := _site()):
            self.scalar[s] += 1
        return func(*args, **(kwargs or {}))

    def __enter__(self):
        self._orig = {n: getattr(torch.Tensor, n) for n in ("tolist", "numpy")}
        for name, fn in self._orig.items():
            def wrap(t, *a, _f=fn, **kw):
                if (s := _site()) is not None:
                    self.bulk[s] += 1
                return _f(t, *a, **kw)
            setattr(torch.Tensor, name, wrap)
        return super().__enter__()

    def __exit__(self, *exc):
        for name, fn in self._orig.items():
            setattr(torch.Tensor, name, fn)
        return super().__exit__(*exc)

    def report(self, label: str) -> None:
        print(f"\n{label}: {sum(self.scalar.values())} scalar, "
              f"{sum(self.bulk.values())} bulk")
        for kind, c in (("scalar", self.scalar), ("bulk", self.bulk)):
            for site, n in c.most_common():
                print(f"  {n:5d}  {kind:6s}  {site}")


def _engine(backend, cfg, model, **kw):
    from tilerl.engine import build_engine

    return build_engine(cfg, model, backend, num_blocks=64, num_slots=8,
                        max_batch=8, max_total_tokens=512, **kw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", default="decode", choices=("decode", "prefill", "train"))
    ap.add_argument("--chunk", type=int, default=32, help="prefill: tokens per tick")
    args = ap.parse_args()

    from tilerl_kernels.backend import Backend, resolve_target

    from tilerl.config import tiny
    from tilerl.model import build_random

    cfg = tiny()
    backend = Backend(resolve_target())
    model = build_random(cfg, seed=3)
    probe = Probe()

    if args.phase == "train":
        from tilerl.autograd import Adafactor
        from tilerl.train import train_step

        opt = Adafactor(lr=1e-3)
        ids = [[1, 2, 3, 4, 5, 6, 7, 8]]
        train_step(model, ids, backend, opt)  # warm: the first step allocates state
        with probe:
            train_step(model, ids, backend, opt)
        probe.report(f"train step ({len(model.params)} params)")
        return

    from tilerl.engine import SamplingParams

    if args.phase == "prefill":
        engine = _engine(backend, cfg, model, max_num_batched_tokens=args.chunk)
        engine.submit(list(range(3, 3 + 3 * args.chunk)),
                      SamplingParams(temperature=0.0, max_new_tokens=4, seed=0))
        engine.step()  # first chunk warms the JIT for this width
        with probe:
            engine.step()
        probe.report(f"prefill tick (chunk={args.chunk}, "
                     f"{len(cfg.full_attn_layers)} full-attn layers)")
        return

    engine = _engine(backend, cfg, model)
    engine.submit([3, 4, 5, 6], SamplingParams(temperature=0.8, max_new_tokens=32, seed=0))
    for _ in range(4):  # settle past prefill into pure decode
        engine.step()
    with probe:
        engine.step()
    probe.report("decode tick")


if __name__ == "__main__":
    main()
