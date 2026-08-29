"""TP correctness gate: a sharded forward must match the unsharded one.

Runs on CPU with gloo, so it needs no GPU:

    uv run torchrun --nproc_per_node=2 scripts/tp_parity.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from tilerl.config import tiny
from tilerl.model import build_random
from tilerl.tensor_parallel import shard_params, tp_config
from tilerl_kernels.backend import Backend, resolve_target


def _logits(cfg, params, backend, ids):
    from tilerl.model import Model
    from tilerl.train import _training_kv

    model = Model(cfg, params)
    model.params = backend.materialize(model.params)
    kv = _training_kv(model, 1, len(ids), device=backend.device)
    return model.forward([ids], torch.arange(len(ids)), kv, backend)


def main() -> None:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    cfg = tiny()
    ids = list(range(8))

    backend = Backend(resolve_target())
    full = build_random(cfg, seed=0)
    ref = _logits(cfg, dict(full.params), backend, ids)

    backend.init_tp(world, rank)
    local = shard_params(dict(full.params), cfg, rank, world)
    got = _logits(tp_config(cfg, world), local, backend, ids)

    if rank == 0:
        err = (got - ref).norm() / ref.norm()
        print(f"tp={world} norm-relative logit error {err:.3e}")
        assert err < 1e-2, f"TP forward diverged: {err}"
        print("tp_parity: OK")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
