"""TP correctness gate: a sharded forward must match the unsharded one.

Runs on CPU with gloo, so it needs no GPU:

    uv run torchrun --nproc_per_node=2 scripts/tp_parity.py
"""

from __future__ import annotations

import os
from dataclasses import replace

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
    ids = list(range(8))
    backend = Backend(resolve_target())

    # untied is required: a tied head stays replicated and never exercises the vocab-parallel path.
    ok = True
    base = tiny()
    variants = [
        ("attn-only", replace(base, num_layers=1, full_attn_layers=(0,))),
        ("gdn-only", replace(base, num_layers=1, full_attn_layers=())),
        ("tied", replace(base, tie_word_embeddings=True)),
        ("untied", replace(base, tie_word_embeddings=False)),
    ]
    for tie, cfg in variants:
        full = build_random(cfg, seed=0)
        # reference with TP off, else _add_via all-reduces the unsharded residual branch.
        backend.tp_world = 1
        ref = _logits(cfg, dict(full.params), backend, ids)
        backend.init_tp(world, rank)
        backend.tp_world = world
        local = shard_params(dict(full.params), cfg, rank, world)
        got = _logits(tp_config(cfg, world), local, backend, ids)
        if rank == 0:
            assert got.shape == ref.shape, f"tie={tie}: {got.shape} vs {ref.shape}"
            err = float((got - ref).norm() / ref.norm())
            print(f"tp={world} tie={tie} norm-relative logit error {err:.3e}")
            ok = ok and err < 1e-2
    if rank == 0:
        assert ok, "TP forward diverged"
        print("tp_parity: OK")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
