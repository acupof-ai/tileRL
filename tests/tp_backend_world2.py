"""World=2 gate on the PRODUCTION Backend: its own comm selection and collectives.

tests/tp_world2.py drives RefBackend, which hardcodes gloo, so ``Backend.init_tp``'s
choice of comm backend and its sharded cross-entropy (the ``tp_world > 1`` arm of
``cross_entropy_loss_grad``) are exercised by nothing at world=2. This runs both.

    TILERL_TARGET=cpu python3 tests/tp_backend_world2.py                  # the gate
    TILERL_TARGET=cpu python3 tests/tp_backend_world2.py --no-collective  # control
    TILERL_TARGET=cpu python3 tests/tp_backend_world2.py --rank0-shard    # control
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path[:0] = ["src", "packages/tilerl-kernels/src"]

IDS = np.array([[1, 2, 3, 4]], dtype=np.int64)  # targets 2,3 on rank 0 and 4 on rank 1
VOCAB = 8


def _logits() -> torch.Tensor:
    """The same [1, 4, VOCAB] in every rank and in the parent, global RNG untouched."""
    return torch.randn(1, IDS.shape[1], VOCAB, generator=torch.Generator().manual_seed(0))


def _run(rank: int, no_collective: bool, rank0_shard: bool, out: dict) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29527")
    # setdefault, not a hard cpu: a two-card box runs this same file under
    # TILERL_TARGET=cuda and must then come out on nccl.
    os.environ.setdefault("TILERL_TARGET", "cpu")

    import torch.distributed as dist
    from tilerl_kernels.backend import Backend, resolve_target

    from tilerl.autograd import RecordingBackend, Tape

    # mp.spawn sets no LOCAL_RANK, so give each rank its own card the way torchrun would
    os.environ.setdefault("LOCAL_RANK", str(rank))

    if no_collective:  # the control: the tp_world == 1 short-circuit at every world
        Backend.all_reduce = lambda self, x: x.view_as(x)

    backend = Backend(resolve_target())
    backend.init_tp(2, rank)
    tp_world, tp_rank = backend.tp_world, backend.tp_rank

    # every probe on backend.device: nccl has no CPU backend, so a host tensor here
    # raises "No backend type associated with device type cpu" (measured, card 0)
    dev = backend.device
    # rank+1, not rank: with rank the sum 0+1 equals rank 1's own value, and a
    # dropped collective reads as correct on that rank.
    probe = torch.tensor([float(rank + 1)], device=dev)
    backend.all_reduce(probe)

    # tp_fork is identity forward and all-reduce backward, so its collective only
    # ever happens on the tape.
    x = torch.full((2,), float(rank + 1), device=dev)
    tape = Tape()
    with tape:
        y = RecordingBackend(backend).tp_fork(x)
    fork = tape.backward(torch.full_like(y, float(rank + 1)))[id(x)]

    if rank0_shard:  # the control: every rank claims rank 0's slice of the vocabulary
        backend.tp_rank = 0
    vloc = VOCAB // 2
    shard = _logits()[..., rank * vloc:(rank + 1) * vloc].contiguous().to(dev)
    loss, grad = backend.cross_entropy_loss_grad(shard, IDS)

    out[rank] = (tp_world, tp_rank, backend.device.type, str(dist.get_backend()),
                 probe.tolist(), fork.tolist(), float(loss), grad.reshape(-1).cpu().tolist(),
                 backend.device.index)


def main() -> int:
    from tilerl_kernels import reference
    from tilerl_kernels.backend import resolve_target

    # one visible card cannot hold two ranks: NCCL refuses with "Duplicate GPU detected"
    # before any assertion here runs, so say what is missing instead of dying inside it
    if resolve_target().startswith("cuda") and torch.cuda.device_count() < 2:
        print(f"SKIP: world=2 over nccl needs 2 visible cards, saw "
              f"{torch.cuda.device_count()}")
        return 0

    no_collective, rank0_shard = "--no-collective" in sys.argv, "--rank0-shard" in sys.argv

    mgr = mp.Manager()
    got = mgr.dict()
    mp.spawn(_run, args=(no_collective, rank0_shard, got), nprocs=2, join=True)

    want_loss, want_grad = reference.cross_entropy_loss_grad(_logits(), IDS)
    vloc, ok = VOCAB // 2, True
    for r in (0, 1):
        world, tp_rank, dev, comm, probe, fork, loss, flat, idx = got[r]
        want_comm = "nccl" if dev == "cuda" else "gloo"  # the rule Backend.init_tp applies
        if (world, tp_rank) != (2, r):
            print(f"rank {r}: init_tp left tp_world={world} tp_rank={tp_rank}, want (2, {r})")
            ok = False
        if comm != want_comm:
            print(f"rank {r}: comm is {comm!r} on a {dev} device, want {want_comm!r}")
            ok = False
        # torchrun sets LOCAL_RANK, so on cuda each rank must hold its OWN card
        if dev == "cuda" and idx != r:
            print(f"rank {r}: bound cuda:{idx}, want cuda:{r} -- both ranks on one card")
            ok = False
        if probe != [3.0]:
            print(f"rank {r}: all_reduce gave {probe}, want [3.0] = 1+2 over the tp group")
            ok = False
        if fork != [3.0, 3.0]:
            print(f"rank {r}: tp_fork backward gave {fork}, want [3.0, 3.0] = 1+2")
            ok = False
        if abs(loss - want_loss) > 1e-5:
            print(f"rank {r}: sharded CE loss {loss:.6f} vs unsharded {want_loss:.6f}")
            ok = False
        g = torch.tensor(flat).reshape(1, IDS.shape[1], vloc)
        w = want_grad[..., r * vloc:(r + 1) * vloc]
        if not torch.allclose(g, w, rtol=1e-5, atol=1e-7):
            print(f"rank {r}: sharded CE grad max|d|={(g - w).abs().max().item():.3e} "
                  f"vs columns [{r * vloc}, {(r + 1) * vloc}) of the unsharded gradient")
            ok = False

    _, _, dev, comm, probe, fork, loss, _, _ = got[0]
    print(f"production Backend world=2 on {dev}/{comm}: all_reduce {probe}, tp_fork bwd {fork}, "
          f"sharded CE loss {loss:.6f} vs unsharded {want_loss:.6f}")
    if no_collective or rank0_shard:
        print("negative control:", "correctly FAILED" if not ok else "PASSED -- vacuous gate")
        return 0 if not ok else 1
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
