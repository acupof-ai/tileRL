"""dp gate: a (dp=2, tp=2) step must equal the dp=1 tp=2 step on the whole batch.

Each replica trains on its own half of the batch, so its gradients are that
half's. Averaging them across the dp group is what makes the update the whole
batch's; without it each replica walks its own way and nothing raises -- the
losses stay finite and plausible, which is why this asserts on the WEIGHTS a real
train_step produced rather than on a loss.

    TILERL_TARGET=cpu python3 tests/dp_world4.py          # the gate
    TILERL_TARGET=cpu python3 tests/dp_world4.py --no-dp  # control: no averaging

Its own file, not an arm of tests/mesh_world4.py: a second mp.spawn in a process
that already ran gloo aborts with SIGABRT (measured).
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


def _dp_step(rank: int, no_dp: bool, out: dict) -> None:
    """A real train_step at (dp=2, tp=2), each replica on its own half of the
    batch. The weights it produces must equal a dp=1 tp=2 step on the WHOLE
    batch: that is what averaging the gradients across replicas buys.

    Built through ``cli._shard``, so this gates the group plumbing the flag
    actually uses rather than a second copy of it here.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    # Set, not setdefault: the two arms have different world sizes, and a port
    # inherited from the world=2 arm puts four ranks on a store built for two
    # (measured: gloo aborts the process with EnforceNotMet, not an error).
    os.environ["MASTER_PORT"] = "29523"
    os.environ["TILERL_TARGET"] = "cpu"
    os.environ["WORLD_SIZE"], os.environ["RANK"] = str(WORLD), str(rank)

    from tilerl import model as model_mod
    from tilerl.autograd import AdamW
    from tilerl.cli import _shard
    from tilerl.config import tiny
    from tilerl.tensor_parallel import Mesh
    from tilerl.testing import RefBackend
    from tilerl.train import train_step

    cfg = tiny()
    full = model_mod.build_random(cfg, seed=0, keep_master=True)
    backend = RefBackend()
    _, local = _shard(cfg, full, TP, backend, model_mod)
    if no_dp:  # the control: keep the layout, drop the gradient averaging
        backend.dp_world = 1

    rows = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
    mine = rows[Mesh(dp=DP, tp=TP, rank=rank).dp_rank][None, :]
    loss = train_step(local, mine, backend, AdamW(lr=1e-2))
    out[rank] = (loss, {k: (v.float().tolist(), tuple(v.shape))
                        for k, v in local.params.items()})


def _dp_ref_rank(r: int, out: dict) -> None:
    """One rank of the dp=1 tp=2 reference step (module level: spawn pickles it)."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29525"
    os.environ["TILERL_TARGET"] = "cpu"
    os.environ["WORLD_SIZE"], os.environ["RANK"] = str(TP), str(r)

    from tilerl import model as model_mod
    from tilerl.autograd import AdamW
    from tilerl.cli import _shard
    from tilerl.config import tiny
    from tilerl.testing import RefBackend
    from tilerl.train import train_step

    cfg = tiny()
    full = model_mod.build_random(cfg, seed=0, keep_master=True)
    backend = RefBackend()
    _, local = _shard(cfg, full, TP, backend, model_mod)
    ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
    loss = train_step(local, ids, backend, AdamW(lr=1e-2))
    out[r] = (loss, {k: (v.float().tolist(), tuple(v.shape))
                     for k, v in local.params.items()})


def _dp_gate(no_dp: bool) -> bool:
    mgr = mp.Manager()
    ref = mgr.dict()
    mp.spawn(_dp_ref_rank, args=(ref,), nprocs=TP, join=True)
    got = mgr.dict()
    mp.spawn(_dp_step, args=(no_dp, got), nprocs=WORLD, join=True)

    from tilerl.tensor_parallel import Mesh

    ok = True
    worst = 0.0
    for r in range(WORLD):
        m = Mesh(dp=DP, tp=TP, rank=r)
        _, want = ref[m.tp_rank]  # same tp shard, dp=1, whole batch
        _, mine = got[r]
        for k, (v, s) in mine.items():
            a = torch.tensor(v).reshape(s)
            b = torch.tensor(want[k][0]).reshape(want[k][1])
            # In ulps: these are bf16 weights, so two summation orders can only
            # land whole rounding steps apart (see tests/tp_world2.py).
            d = (a.float() - b.float()).abs()
            ulp = torch.exp2(torch.floor(torch.log2(b.float().abs().clamp_min(1e-30))) - 7)
            worst = max(worst, (d / ulp).max().item())
            if (d > ulp * 1.01).sum().item():
                if ok:
                    print(f"rank {r}: {k} differs from the dp=1 step by "
                          f"{(d / ulp).max().item():.1f} ulp")
                ok = False
    print(f"dp: worst deviation from the dp=1 tp=2 step {worst:.1f} ulp")
    return ok



def _streaming_refused() -> bool:
    """dp>1 with a streaming optimizer must raise, not hang.

    It applies each gradient in completion order, which differs per rank, so the
    all-reduce would pair different tensors and gloo would abort the process.
    No spawn needed: setting dp_world reaches the same branch.
    """
    import numpy as np

    from tilerl import model as model_mod
    from tilerl.autograd import Adafactor
    from tilerl.config import tiny
    from tilerl.testing import RefBackend
    from tilerl.train import train_step

    cfg = tiny()
    backend = RefBackend()
    backend.dp_world = 2
    try:
        train_step(model_mod.build_random(cfg, seed=0, keep_master=True),
                   np.array([[1, 2, 3, 4]], dtype=np.int64), backend, Adafactor(lr=1e-3))
    except ValueError as e:
        if "streaming optimizer" in str(e):
            print("dp: a streaming optimizer under dp>1 is refused")
            return True
        print(f"dp: streaming refused for the wrong reason: {e}")
        return False
    print("dp: a streaming optimizer under dp>1 was ACCEPTED; it would abort on a real world")
    return False


def main() -> int:
    no_dp = "--no-dp" in sys.argv
    ok = _dp_gate(no_dp)
    if no_dp:
        print("no-dp control:", "correctly FAILED" if not ok else "PASSED -- vacuous gate")
        return 0 if not ok else 1
    print("dp: a (dp=2, tp=2) step matches the dp=1 step on the whole batch"
          if ok else "dp: FAILED")
    return 0 if ok and _streaming_refused() else 1


if __name__ == "__main__":
    raise SystemExit(main())
