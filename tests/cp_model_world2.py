"""CP end to end: a whole model forward on two ranks must equal the unsplit one.

``cp_world2.py`` gates the attention op and ``gdn_world2.py`` the GDN scan, both
against hand-built references. This gates the LAYER wiring — ``Model.forward``
with ``tiny()``, which has one full-attention layer and one GDN layer, so both CP
paths run in the same forward and the residual stream carries one rank's tokens
through both.

That is the constraint this exists for: a token's residual lives on one rank for
every layer, so attention and GDN must agree on which rank holds which chunk. If
they disagree the forward still produces numbers.

    TILERL_TARGET=cpu python3 tests/cp_model_world2.py             # the gate
    TILERL_TARGET=cpu python3 tests/cp_model_world2.py --gdn-only  # control: attention unsplit
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path[:0] = ["src", "packages/tilerl-kernels/src"]

CP = 2
T = 16


def _rank(r: int, world: int, gdn_only: bool, out: dict) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    os.environ["TILERL_TARGET"] = "cpu"

    from tilerl import model as model_mod
    from tilerl.config import tiny
    from tilerl.tensor_parallel import Mesh, zigzag_positions
    from tilerl.testing import RefBackend
    from tilerl.train import _training_kv

    cfg = tiny()
    m = model_mod.build_random(cfg, seed=0, keep_master=True)
    backend = RefBackend()

    ids = np.arange(1, T + 1, dtype=np.int64).reshape(1, T)
    if world == 1:
        kv = _training_kv(m, 1, T, device=backend.device)
        with torch.no_grad():
            logits = m.forward(ids, np.arange(T, dtype=np.int64), kv, backend)
        out[0] = (logits.float().tolist(), list(range(T)))
        return

    mesh = Mesh(cp=world, rank=r)
    backend.init_tp(world, r, cp_groups=[mesh.cp_group()])
    if gdn_only:
        # Control: attention sees the whole sequence while GDN stays split, so
        # the two halves disagree about who owns which token -- the exact failure
        # the shared layout exists to prevent. Flipping cp_world on the BACKEND
        # for the duration of the call, rather than wrapping the method: this
        # runs once per process, and a wrapper reinstalled per call double-wraps
        # (measured: an earlier probe read a stale value through two spies).
        backend.attn_cp_off = True

    pos = zigzag_positions(T, world, backend.cp_rank)
    kv = _training_kv(m, 1, len(pos), device=backend.device)
    with torch.no_grad():
        logits = m.forward(ids[:, pos.numpy()], pos.numpy(), kv, backend)
    out[r] = (logits.float().tolist(), pos.tolist())


def main() -> int:
    gdn_only = "--gdn-only" in sys.argv

    one: dict = {}
    _rank(0, 1, False, one)
    ref = torch.tensor(one[0][0])

    mgr = mp.Manager()
    got = mgr.dict()
    mp.spawn(_rank, args=(CP, gdn_only, got), nprocs=CP, join=True)

    ok, rows = True, []
    for r in range(CP):
        mine, pos = got[r]
        mine = torch.tensor(mine)
        want = ref[:, pos]
        rel = (mine - want).abs().max().item() / max(want.abs().max().item(), 1e-9)
        rows.append((r, rel))
        if rel > 1e-5:
            ok = False

    print(" ".join(f"rank {r}: logits rel {rel:.1e}" for r, rel in rows))
    if gdn_only:
        print("gdn-only control:", "correctly FAILED" if not ok else "PASSED -- vacuous gate")
        return 0 if not ok else 1
    print("cp model cp=2: a split forward matches the unsplit one on both layer kinds"
          if ok else "cp model cp=2: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
