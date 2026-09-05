"""Sharded cross-entropy gate: loss and gradient equal the unsharded CE, and no
[B, T, V] tensor is ever formed.

Two gloo ranks on the CPU target. The memory half is the reason this op exists:
all-gathering the logits to reuse the unsharded CE materialises [B, T, 248320] in
f32 on the 27B -- the 8.5 GiB that failed on 2026-08-30 -- so the gate has to
prove the full row is never built, not merely that the numbers agree.

    TILERL_TARGET=cpu python3 tests/ce_sharded_world2.py            # the gate
    TILERL_TARGET=cpu python3 tests/ce_sharded_world2.py --gather   # the control

``--gather`` swaps in the all-gather implementation, which is numerically correct
and must FAIL the memory assertion. Without it, "no full row was formed" is a
claim about a run nobody made.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.multiprocessing as mp
from torch.overrides import TorchFunctionMode

sys.path[:0] = ["src", "packages/tilerl-kernels/src"]

B, T, V, WORLD = 2, 6, 64, 2


class WidestTensor(TorchFunctionMode):
    """Records the widest last dimension any op produces.

    torch's CPU allocator is invisible to tracemalloc (measured: an 4 MB tensor
    shows as 80 bytes), so bytes cannot be asserted here. The shape can: a
    [B, T, V] intermediate is exactly what must not appear, and its last dim is V
    against V/world for every legitimate tensor.
    """

    def __init__(self) -> None:
        self.widest = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in out if isinstance(out, (list, tuple)) else [out]:
            if isinstance(t, torch.Tensor) and t.ndim and t.shape[-1] > self.widest:
                self.widest = t.shape[-1]
        return out


def _run(rank: int, gather: bool, out: dict) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29519")
    os.environ["TILERL_TARGET"] = "cpu"

    import torch.distributed as dist
    from tilerl_kernels import reference

    dist.init_process_group("gloo", world_size=WORLD, rank=rank)

    torch.manual_seed(0)  # same draw on both ranks, then each takes its slice
    logits = torch.randn(B, T, V)
    ids = torch.randint(0, V, (B, T))
    vloc = V // WORLD
    start = rank * vloc
    mine = logits[..., start : start + vloc].contiguous()

    def all_reduce(t, op):
        dist.all_reduce(t, op=dist.ReduceOp.SUM if op == "sum" else dist.ReduceOp.MAX)
        return t

    watch = WidestTensor()
    with watch:
        if gather:
            # The control: correct, and it forms the [B, T, V] row this op exists
            # to avoid.
            parts = [torch.empty_like(mine) for _ in range(WORLD)]
            dist.all_gather(parts, mine.contiguous())
            loss, g_full = reference.cross_entropy_loss_grad(torch.cat(parts, dim=-1), ids)
            grad = g_full[..., start : start + vloc].contiguous()
        else:
            loss, grad = reference.cross_entropy_sharded(mine, ids, all_reduce, start, V)

    out[rank] = (loss, grad.reshape(-1).tolist(), tuple(grad.shape), watch.widest)


def main() -> int:
    gather = "--gather" in sys.argv
    from tilerl_kernels import reference

    torch.manual_seed(0)
    logits = torch.randn(B, T, V)
    ids = torch.randint(0, V, (B, T))
    ref_loss, ref_grad = reference.cross_entropy_loss_grad(logits.clone(), ids)

    mgr = mp.Manager()
    got = mgr.dict()
    mp.spawn(_run, args=(gather, got), nprocs=WORLD, join=True)

    ok = True
    vloc = V // WORLD
    for r in range(WORLD):
        loss, flat, shape, widest = got[r]
        grad = torch.tensor(flat).reshape(shape)
        want = ref_grad[..., r * vloc : (r + 1) * vloc]
        if abs(loss - ref_loss) > 1e-5:
            print(f"rank {r}: loss {loss:.6f} vs unsharded {ref_loss:.6f}")
            ok = False
        if not torch.allclose(grad, want, rtol=1e-5, atol=1e-6):
            print(f"rank {r}: dlogits MISMATCH max|d|={(grad - want).abs().max():.3e}")
            ok = False
        # The memory claim, as a shape: nothing wider than this rank's shard.
        if widest > vloc:
            print(f"rank {r}: formed a tensor {widest} wide, shard is {vloc}")
            ok = False

    if gather:
        print("memory control:", "correctly FAILED" if not ok else "PASSED -- vacuous gate")
        return 0 if not ok else 1
    print(f"sharded CE matches unsharded (loss {ref_loss:.6f}), widest tensor {vloc} == V/world")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
