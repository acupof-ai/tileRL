"""The conv1d halo under CP: a chunk's left context lives on another rank.

Every GDN layer runs a depthwise conv1d of kernel K over q/k/v before the
recurrence, so a chunk's first K-1 tokens need the last K-1 of the chunk before
it. Under CP that chunk is on a different rank -- and under zigzag, on a
different rank per chunk.

The failure is not confined to the boundary rows. Corrupted k/v feed the
recurrence, so the whole chunk is wrong: measured 0.79 relative on the 3 rows and
**0.61 on everything after them**. That is what makes this silent -- a spot check
of the tail of a chunk still looks broken-but-plausible rather than obviously
misaligned.

This is the arm ``tests/gdn_world2.py`` cannot run: that gate drives
``_gdn_chunk_fwd``, whose inputs are already past the conv.

    TILERL_TARGET=cpu python3 tests/gdn_halo_world2.py            # the gate
    TILERL_TARGET=cpu python3 tests/gdn_halo_world2.py --no-halo  # control: drop the window
"""

from __future__ import annotations

import os
import sys

import torch
import torch.multiprocessing as mp

sys.path[:0] = ["src", "packages/tilerl-kernels/src"]

CP = 2
B, T, NKH, NVH, K, V = 1, 32, 2, 4, 8, 8
KERNEL = 4
QD, KD, VD = NKH * K, NKH * K, NVH * V


def _inputs():
    torch.manual_seed(23)
    q, k, v = torch.randn(B, T, QD), torch.randn(B, T, KD), torch.randn(B, T, VD)
    a, b = torch.randn(B, T, NVH), torch.randn(B, T, NVH)
    kw = dict(z=torch.randn(B, T, VD), conv1d_weight=torch.randn(QD + KD + VD, KERNEL) * 0.2,
              dt_bias=torch.randn(NVH), a_log=torch.randn(NVH), norm_weight=torch.randn(V))
    return q, k, v, a, b, kw


def _slice_kw(kw, sl):
    return {n: (t[:, sl] if n == "z" else t) for n, t in kw.items()}


def _rank(r: int, world: int, no_halo: bool, out: dict) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")
    os.environ["TILERL_TARGET"] = "cpu"

    from tilerl_kernels import reference

    from tilerl.tensor_parallel import Mesh, zigzag_chunk_ids
    from tilerl.testing import RefBackend

    backend = RefBackend()
    mesh = Mesh(cp=world, rank=r)
    backend.init_tp(world, r, cp_groups=[mesh.cp_group()])

    q, k, v, a, b, kw = _inputs()
    n_chunks = 2 * world
    span = T // n_chunks
    ids_by_rank = [zigzag_chunk_ids(world, rr) for rr in range(world)]
    mine = ids_by_rank[r]

    # The halo travels PRE-conv, as the raw qkv rows -- that is what the conv
    # reads. Stack this rank's chunks so the collective sees one tensor.
    qkv = torch.cat([q, k, v], dim=-1)
    stacked = torch.stack([qkv[:, c * span:(c + 1) * span] for c in mine])
    halos = backend.cp_halo(stacked, ids_by_rank, KERNEL - 1)

    res = {}
    for i, c in enumerate(mine):
        sl = slice(c * span, (c + 1) * span)
        win = torch.zeros(B, KERNEL - 1, QD + KD + VD) if (no_halo or halos[i] is None) \
            else halos[i]
        # The state is supplied from the sequential truth, so this arm measures
        # the CONV halo alone and not the state scan (gdn_world2.py covers that).
        o, _, _ = reference.gdn_forward(q[:, sl], k[:, sl], v[:, sl], a[:, sl], b[:, sl],
                                        _state_before(c, span), conv_window=win,
                                        **_slice_kw(kw, sl))
        res[c] = o.tolist()
    out[r] = res


def _state_before(chunk: int, span: int):
    """The true state entering this chunk, from the sequential run."""
    from tilerl_kernels import reference

    q, k, v, a, b, kw = _inputs()
    s = torch.zeros(B, NVH, K, V)
    win = torch.zeros(B, KERNEL - 1, QD + KD + VD)
    for c in range(chunk):
        sl = slice(c * span, (c + 1) * span)
        _, s, win = reference.gdn_forward(q[:, sl], k[:, sl], v[:, sl], a[:, sl], b[:, sl], s,
                                          conv_window=win, **_slice_kw(kw, sl))
    return s


def _truth(span: int):
    from tilerl_kernels import reference

    q, k, v, a, b, kw = _inputs()
    s = torch.zeros(B, NVH, K, V)
    win = torch.zeros(B, KERNEL - 1, QD + KD + VD)
    outs = []
    for c in range(T // span):
        sl = slice(c * span, (c + 1) * span)
        o, s, win = reference.gdn_forward(q[:, sl], k[:, sl], v[:, sl], a[:, sl], b[:, sl], s,
                                          conv_window=win, **_slice_kw(kw, sl))
        outs.append(o)
    return outs


def main() -> int:
    no_halo = "--no-halo" in sys.argv
    span = T // (2 * CP)
    truth = _truth(span)

    mgr = mp.Manager()
    got = mgr.dict()
    mp.spawn(_rank, args=(CP, no_halo, got), nprocs=CP, join=True)

    ok, rows = True, []
    for r in range(CP):
        for c, o in sorted(got[r].items()):
            mine, t = torch.tensor(o), truth[c]
            scale = max(t.abs().max().item(), 1e-9)
            # Split at the kernel width: the first K-1 rows are the ones a reader
            # expects to break, and the REST is the finding -- the recurrence
            # carries the corruption forward through the whole chunk.
            head = (mine[:, :KERNEL - 1] - t[:, :KERNEL - 1]).abs().max().item() / scale
            tail = (mine[:, KERNEL - 1:] - t[:, KERNEL - 1:]).abs().max().item() / scale
            rows.append((c, head, tail))
            if max(head, tail) > 1e-6:
                ok = False

    print(" ".join(f"[c{c}] first{KERNEL - 1} {h:.1e} rest {t:.1e}" for c, h, t in rows))
    if no_halo:
        print("no-halo control:", "correctly FAILED" if not ok else "PASSED -- vacuous gate")
        return 0 if not ok else 1
    print("gdn halo cp=2: every chunk with its left context matches the sequential run"
          if ok else "gdn halo cp=2: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
