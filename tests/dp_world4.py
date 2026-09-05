"""dp gate: a (dp=2, tp=2) step must equal the dp=1 tp=2 step on the whole batch.

Each replica trains on its own half of the batch, so its gradients are that
half's. Averaging them across the dp group is what makes the update the whole
batch's; without it each replica walks its own way and nothing raises -- the
losses stay finite and plausible, which is why this asserts on the WEIGHTS a real
train_step produced rather than on a loss.

    TILERL_TARGET=cpu python3 tests/dp_world4.py          # the gate
    TILERL_TARGET=cpu python3 tests/dp_world4.py --no-dp     # control: no averaging
    TILERL_TARGET=cpu python3 tests/dp_world4.py --scramble  # control: order check
    TILERL_TARGET=cpu python3 tests/dp_world4.py --scramble-keys  # control: key-set check

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


def _dp_step(rank: int, no_dp: bool, out: dict, streams: bool = False,
             scramble: bool | str = False) -> None:
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

    os.environ["TILERL_CHECK_DP_ORDER"] = "1"  # gate time: one extra collective

    from tilerl import model as model_mod
    from tilerl.autograd import Adafactor, AdamW
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
    if scramble:  # the control: make the apply order per-rank, as id() order was
        import tilerl.train as tr

        real = tr._order_agrees
        # By DP rank, not rank % 2: the check compares within the dp group, which
        # for rank r is {r, r+2} -- both even or both odd, so reversing on rank%2
        # reverses BOTH members of every pair and they still agree. That version
        # of this control passed, which is how the bug in it was found.
        flip = Mesh(dp=DP, tp=TP, rank=rank).dp_rank == 1
        if scramble == "keys":
            # A rank-conditional parameter set, same order: sorted(params) agrees
            # across ranks only because the key sets do, and nothing else says so.
            tr._order_agrees = lambda order, b: real(order[:-1] if flip else order, b)
        else:
            tr._order_agrees = lambda order, b: real(order[::-1] if flip else order, b)

    rows = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
    mine = rows[Mesh(dp=DP, tp=TP, rank=rank).dp_rank][None, :]
    opt = Adafactor(lr=1e-2) if streams else AdamW(lr=1e-2)
    loss = train_step(local, mine, backend, opt)
    out[rank] = (loss, {k: (v.float().tolist(), tuple(v.shape))
                        for k, v in local.params.items()})


def _dp_ref_rank(r: int, out: dict, streams: bool = False) -> None:
    """One rank of the dp=1 tp=2 reference step (module level: spawn pickles it)."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29525"
    os.environ["TILERL_TARGET"] = "cpu"
    os.environ["WORLD_SIZE"], os.environ["RANK"] = str(TP), str(r)

    from tilerl import model as model_mod
    from tilerl.autograd import Adafactor, AdamW
    from tilerl.cli import _shard
    from tilerl.config import tiny
    from tilerl.testing import RefBackend
    from tilerl.train import train_step

    cfg = tiny()
    full = model_mod.build_random(cfg, seed=0, keep_master=True)
    backend = RefBackend()
    _, local = _shard(cfg, full, TP, backend, model_mod)
    ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
    # The same optimizer as the arm it is the reference for: Adafactor and AdamW
    # produce different weights from the same gradients, so a cross-optimizer
    # comparison would measure the optimizers, not the dp reduce.
    loss = train_step(local, ids, backend, Adafactor(lr=1e-2) if streams else AdamW(lr=1e-2))
    out[r] = (loss, {k: (v.float().tolist(), tuple(v.shape))
                     for k, v in local.params.items()})


def _dp_gate(no_dp: bool, streams: bool = False, scramble: bool | str = False) -> bool:
    mgr = mp.Manager()
    ref = mgr.dict()
    mp.spawn(_dp_ref_rank, args=(ref, streams), nprocs=TP, join=True)
    got = mgr.dict()
    mp.spawn(_dp_step, args=(no_dp, got, streams, scramble), nprocs=WORLD, join=True)

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


def main() -> int:
    no_dp = "--no-dp" in sys.argv
    scramble = "keys" if "--scramble-keys" in sys.argv else "--scramble" in sys.argv
    if scramble:
        # The control for the order check itself: reverse one rank's recorded
        # order. The reduce still pairs correctly (the tape order is unchanged),
        # so this is a test of _order_agrees, and it must report, not hang.
        try:
            _dp_gate(False, streams=True, scramble=scramble)
        except Exception as e:
            reason = "different parameters, or in different orders" in str(e)
            print("scramble control: correctly reported the order mismatch" if reason
                  else f"scramble control: failed for another reason: {str(e)[-200:]}")
            return 0 if reason else 1
        print("scramble control: PASSED -- the order check is vacuous")
        return 1

    ok = _dp_gate(no_dp)
    if no_dp:
        print("no-dp control:", "correctly FAILED" if not ok else "PASSED -- vacuous gate")
        return 0 if not ok else 1
    print("dp: a (dp=2, tp=2) step matches the dp=1 step on the whole batch"
          if ok else "dp: FAILED")

    # Adafactor streams its updates and is the 27B optimizer (Adam's state is
    # 200 GiB), so dp has to work on that path or it is unusable where it counts.
    st = _dp_gate(False, streams=True)
    print("dp+streams: a streaming optimizer under dp=2 matches its dp=1 step"
          if st else "dp+streams: FAILED")
    return 0 if ok and st else 1


if __name__ == "__main__":
    raise SystemExit(main())
