"""CP-2 training gate: a split sequence must produce the unsplit gradients.

Two gloo ranks on the CPU target, so it gates here and in CI. Two failures this
exists for, both silent:

  * the backward of the K/V all-gather is a **sum across ranks**, not a slice.
    ``docs/design-parallel.md`` said slice, which is right for the vocab gather
    and wrong here -- every rank's queries read every rank's keys. Measured on a
    dense reference at cp=4: slicing is off by 58% of full scale.
  * the mask has to come from ABSOLUTE positions. The gathered chunks arrive in
    rank order, which the zigzag assignment makes differ from sequence order, so
    a mask built from the tensor's shape silently attends the wrong keys.

    TILERL_TARGET=cpu python3 tests/cp_world2.py             # the gate
    TILERL_TARGET=cpu python3 tests/cp_world2.py --slice-bwd # control: slice, not reduce_scatter
    TILERL_TARGET=cpu python3 tests/cp_world2.py --seq-mask  # control: mask by index, not position
"""

from __future__ import annotations

import os
import sys

import torch
import torch.multiprocessing as mp

sys.path[:0] = ["src", "packages/tilerl-kernels/src"]

CP = 2
B, T, HQ, HKV, D = 1, 8, 4, 2, 8
SCALE = 0.5


def _inputs():
    torch.manual_seed(7)
    return (torch.randn(B, T, HQ, D), torch.randn(B, T, HKV, D),
            torch.randn(B, T, HKV, D), torch.randn(B, T, HQ, D))


def _rank(r: int, world: int, slice_bwd: bool, seq_mask: bool, out: dict) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29523")
    os.environ["TILERL_TARGET"] = "cpu"

    from tilerl import autograd
    from tilerl.autograd import RecordingBackend, Tape
    from tilerl.tensor_parallel import Mesh, zigzag_key_positions, zigzag_positions
    from tilerl.testing import RefBackend

    if slice_bwd:  # the control: the doc's claim, keep this rank's chunk
        def _slice_only(backend, g, args, kw):
            dim = kw.get("dim", 1)
            yield 0, g.chunk(backend.cp_world, dim=dim)[backend.cp_rank].contiguous()

        autograd._BWD["cp_gather"] = _slice_only

    backend = RefBackend()
    mesh = Mesh(cp=world, rank=r)
    backend.init_tp(world, r, cp_groups=[mesh.cp_group()])

    q_all, k_all, v_all, g_all = _inputs()
    pos = zigzag_positions(T, world, backend.cp_rank)
    q, k, v, g = (x[:, pos] for x in (q_all, k_all, v_all, g_all))

    rb = RecordingBackend(backend)
    tape = Tape()
    with torch.no_grad(), tape:
        kg = rb.cp_gather(k, dim=1)
        vg = rb.cp_gather(v, dim=1)
        k_pos = None if seq_mask else zigzag_key_positions(T, world)
        out_t = rb.attention(q, kg, vg, SCALE,
                             q_pos=None if seq_mask else pos, k_pos=k_pos)
    grads = tape.backward(g)
    out[r] = (pos.tolist(),
              [(grads[id(t)].float().tolist() if id(t) in grads else None) for t in (q, k, v)],
              out_t.float().tolist())


def _truth():
    """One dense attention over the whole sequence: the thing CP must reproduce."""
    from tilerl_kernels import reference

    q, k, v, g = _inputs()
    out = reference.dense_attention(q, k, v, SCALE)
    gq, gk, gv = reference.dense_attention_bwd(g, q, k, v, SCALE)
    return out, gq, gk, gv


def main() -> int:
    slice_bwd = "--slice-bwd" in sys.argv
    seq_mask = "--seq-mask" in sys.argv

    out_t, gq_t, gk_t, gv_t = _truth()
    mgr = mp.Manager()
    got = mgr.dict()
    mp.spawn(_rank, args=(CP, slice_bwd, seq_mask, got), nprocs=CP, join=True)

    ok, worst = True, {}
    for r in range(CP):
        pos, (gq, gk, gv), o = got[r]
        pos = torch.tensor(pos)
        for name, mine, truth in (("out", o, out_t), ("dQ", gq, gq_t),
                                  ("dK", gk, gk_t), ("dV", gv, gv_t)):
            assert mine is not None, f"rank {r}: no gradient for {name}"
            # Every tensor here is per-rank: the queries this rank owns, and the
            # dK/dV for the chunk it keeps after the reduce_scatter. Both index
            # by the same absolute positions.
            d = (torch.tensor(mine) - truth[:, pos]).abs().max().item()
            worst[name] = max(worst.get(name, 0.0), d)
            if d > 1e-4:
                ok = False
    scale = gk_t.abs().max().item()
    print("cp=2  " + "  ".join(f"{k} max|d| {v:.3e}" for k, v in worst.items())
          + f"   (|dK| max {scale:.3f})")

    if slice_bwd or seq_mask:
        which = "slice-bwd" if slice_bwd else "seq-mask"
        print(f"{which} control:", "correctly FAILED" if not ok else "PASSED -- vacuous gate")
        return 0 if not ok else 1
    print("cp: a split sequence matches the unsplit forward and backward" if ok else "cp: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
