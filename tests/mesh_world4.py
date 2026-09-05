"""Mesh gate: with dp>1, a tp all-reduce must not cross the dp replicas.

Four gloo ranks as (dp=2, tp=2). Each dp replica trains on a DIFFERENT batch, so
the two replicas' gradients must differ; an ungrouped all_reduce sums across all
four ranks and makes them identical, which reads as "TP works" on every
gradient-equality check and quietly destroys data parallelism.

    TILERL_TARGET=cpu python3 tests/mesh_world4.py             # the gate
    TILERL_TARGET=cpu python3 tests/mesh_world4.py --ungrouped # the control

``--ungrouped`` drops the tp process group, which is the whole-world all_reduce
this exists to prevent. It must FAIL.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path[:0] = ["src", "packages/tilerl-kernels/src"]

DP, TP = 2, 2
WORLD = DP * TP


def _run(rank: int, ungrouped: bool, out: dict) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29521")
    os.environ["TILERL_TARGET"] = "cpu"

    from tilerl import model as model_mod
    from tilerl.autograd import RecordingBackend, Tape
    from tilerl.config import tiny
    from tilerl.tensor_parallel import Mesh, shard_params, tp_config
    from tilerl.testing import RefBackend
    from tilerl.train import _training_kv

    mesh = Mesh(dp=DP, tp=TP, rank=rank)
    cfg = tiny()
    full = model_mod.build_random(cfg, seed=0, keep_master=True)
    # BOTH backends, in one run. RefBackend carries the model; the real Backend is
    # what production uses, and dropping group= from it left this gate green when
    # only RefBackend was exercised. It joins gloo on the cpu target, so there is
    # no reason for it to go unchecked.
    backend = RefBackend()
    groups = None if ungrouped else _all_tp_groups()
    backend.init_tp(WORLD, rank, groups)
    real = _real_backend(rank, groups)

    local = model_mod.Model(tp_config(cfg, TP), shard_params(full.params, cfg, mesh.tp_rank, TP))

    # The point of dp: each replica sees different data.
    ids = np.array([[1, 2, 3, 4]] if mesh.dp_rank == 0 else [[5, 6, 7, 8]], dtype=np.int64)
    tape = Tape()
    with torch.no_grad(), tape:
        logits = local.forward(ids, np.arange(4, dtype=np.int64),
                               _training_kv(local, 1, 4, device=backend.device),
                               RecordingBackend(backend))
    grads = tape.backward(torch.ones_like(logits))
    g = grads[id(local.params["embed_tokens"])]  # replicated, so comparable across ranks

    # The collective itself, asserted directly. The end-to-end gradient is a poor
    # discriminator here: embed_tokens keeps a local scatter term that depends on
    # this replica's ids, so dp0 and dp1 differ even when the all-reduce wrongly
    # spans all four ranks -- measured, and it made the first version of this gate
    # pass its own control. A rank-valued probe has no local term to hide behind:
    # grouped, ranks 0,1 sum to 1 and ranks 2,3 to 5; ungrouped, everyone gets 6.
    probe = torch.tensor([float(rank)])
    backend.all_reduce(probe)
    real_probe = torch.tensor([float(rank)])
    if real is not None:
        real.all_reduce(real_probe)

    out[rank] = (mesh.dp_rank, mesh.tp_rank, g.float().reshape(-1).tolist(),
                 probe.item(), real_probe.item() if real is not None else None)


def _real_backend(rank: int, groups):
    """The production Backend on the cpu target, sharing the initialized gloo world.

    Returns None if tilelang cannot load here; the RefBackend arm still runs and
    the caller says the real arm was skipped rather than passing silently.
    """
    try:
        from tilerl_kernels.backend import Backend, resolve_target
    except Exception:
        return None
    b = Backend(resolve_target())
    b.init_tp(WORLD, rank, groups)
    return b


def _all_tp_groups() -> list[list[int]]:
    from tilerl.tensor_parallel import Mesh

    seen: list[list[int]] = []
    for r in range(WORLD):
        g = Mesh(dp=DP, tp=TP, rank=r).tp_group()
        if g not in seen:
            seen.append(g)
    return seen


def main() -> int:
    ungrouped = "--ungrouped" in sys.argv
    from tilerl.tensor_parallel import Mesh

    mgr = mp.Manager()
    got = mgr.dict()
    mp.spawn(_run, args=(ungrouped, got), nprocs=WORLD, join=True)

    by_dp: dict[int, list[torch.Tensor]] = {}
    probes: dict[int, float] = {}
    real_probes: dict[int, float | None] = {}
    for r in range(WORLD):
        dp_rank, _tp_rank, flat, probe, real_probe = got[r]
        by_dp.setdefault(dp_rank, []).append(torch.tensor(flat))
        probes[r] = probe
        real_probes[r] = real_probe

    ok = True
    # Within one dp replica the tp ranks share a batch, so a replicated param's
    # gradient must agree: that is the tp all-reduce doing its job.
    for d, gs in by_dp.items():
        if not torch.allclose(gs[0], gs[1], rtol=2e-3, atol=1e-3):
            print(f"dp{d}: tp ranks disagree, max|d|={(gs[0] - gs[1]).abs().max():.3e}")
            ok = False

    # The tp all-reduce must span this rank's tp group and nothing else, on EACH
    # backend: they carry separate group= plumbing and one can regress alone.
    want = {r: float(sum(Mesh(dp=DP, tp=TP, rank=r).tp_group())) for r in range(WORLD)}
    if probes != want:
        print(f"RefBackend all_reduce spans the wrong ranks: got {probes}, want {want} "
              "(all four equal means it crossed the dp replicas)")
        ok = False
    if all(v is None for v in real_probes.values()):
        print("NOTE: tilelang did not import, so the production Backend arm was SKIPPED")
    elif real_probes != want:
        print(f"Backend all_reduce spans the wrong ranks: got {real_probes}, want {want}")
        ok = False

    a, b = by_dp[0][0], by_dp[1][0]
    delta = (a - b).abs().max().item()

    print(f"tp all_reduce sums: RefBackend {probes}, Backend {real_probes} "
          f"(want {want}); dp0 vs dp1 grad max|d|={delta:.3e}")
    if ungrouped:
        print("ungrouped control:", "correctly FAILED" if not ok else "PASSED -- vacuous gate")
        return 0 if not ok else 1
    print("mesh: tp all-reduce stays inside its dp replica" if ok else "mesh: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
