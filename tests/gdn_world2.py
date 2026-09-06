"""GDN under CP, two gloo ranks, on attention's ZIGZAG assignment.

Each rank builds the affine ``(A, B)`` of each chunk it holds from a zero state,
``cp_prefix_scan`` composes every chunk before it in SEQUENCE order, and the rank
re-runs its chunks from those prefixes. The result must equal the sequential scan
over the whole sequence.

Zigzag, not contiguous: a token's residual stream lives on one rank for all 64
layers, so GDN cannot use a different assignment from attention without an
all-to-all between every layer pair (96.6 ms/step at T=2048 B=8, against 24.2 ms
for the whole scan). The recurrence orders the CHUNKS, not the ranks -- which
rank holds chunk c is free, as long as the scan composes in sequence order.

Two cross-rank dependencies exist. This gate covers the first:
  * the **state**, through the affine scan — below;
  * the **conv1d halo** — kernel 4, so a chunk's first 3 tokens need the last 3
    of the chunk before it. NOT exercised here: this gate drives
    ``_gdn_chunk_fwd``, whose inputs are already past the conv. Measured
    separately at the ``gdn_forward`` level — dropping the halo is wrong by 0.79
    on the 3 boundary rows and **0.61 on the whole rest of the chunk**, because
    corrupted k/v feed the recurrence; carrying it gives exactly 0.0. It lands
    with the ``_gdn`` wiring, which is where a window can actually be passed.

    TILERL_TARGET=cpu python3 tests/gdn_world2.py            # the gate
    TILERL_TARGET=cpu python3 tests/gdn_world2.py --no-scan  # control: skip the prefix
    TILERL_TARGET=cpu python3 tests/gdn_world2.py --decay-a  # control: A as a scalar
"""

from __future__ import annotations

import os
import sys

import torch
import torch.multiprocessing as mp

sys.path[:0] = ["src", "packages/tilerl-kernels/src"]

CP = 2
B, T, HV, DK, DV, CHUNK = 1, 32, 4, 8, 8, 8
NCHUNK = T // CHUNK  # 4 chunks over 2 ranks: 2 each, zigzag


def _inputs():
    torch.manual_seed(11)
    return (torch.randn(B, T, HV, DK), torch.randn(B, T, HV, DK), torch.randn(B, T, HV, DV),
            -torch.rand(B, T, HV) * 0.1, torch.rand(B, T, HV), torch.randn(B, HV, DK, DV) * 0.1)


def _chunks_of(rank: int, world: int) -> list[int]:
    """Zigzag in CHUNKS: rank r holds chunk r and chunk NCHUNK-1-r."""
    return [rank, NCHUNK - 1 - rank]


def _rank(r: int, world: int, no_scan: bool, decay_a: bool, out: dict) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29529")
    os.environ["TILERL_TARGET"] = "cpu"

    from tilerl_kernels import reference
    from tilerl_kernels.reference import _gdn_chunk_fwd

    from tilerl.tensor_parallel import Mesh
    from tilerl.testing import RefBackend

    backend = RefBackend()
    mesh = Mesh(cp=world, rank=r)
    backend.init_tp(world, r, cp_groups=[mesh.cp_group()])

    q, k, v, gt, bt, s0 = _inputs()
    mine = _chunks_of(r, world)
    sl = [slice(c * CHUNK, (c + 1) * CHUNK) for c in mine]

    # (A, B) per chunk this rank holds, stacked in the order the scan expects:
    # entry i of rank r is chunk r*entries + i. Zigzag makes that NOT sequence
    # order globally -- the scan is told the chunk index, not the tensor's place.
    a_list, b_list = [], []
    for s in sl:
        a_i, b_i = reference.gdn_span_ab(q[:, s], k[:, s], v[:, s], bt[:, s], gt[:, s],
                                         chunk=CHUNK)
        if decay_a:  # control: A as the decay scalar, dropping d's dependence on s
            eye = torch.eye(DK).expand_as(a_i)
            a_i = torch.exp(gt[:, s].sum(1)).unsqueeze(-1).unsqueeze(-1) * eye
        a_list.append(a_i)
        b_list.append(b_i)
    a_pre, b_pre = backend.cp_prefix_scan(torch.stack(a_list), torch.stack(b_list),
                                          chunk_ids=mine)

    res = {}
    for i, (c, s) in enumerate(zip(mine, sl)):
        s_in = s0 if no_scan else a_pre[i] @ s0 + b_pre[i]
        o, s_out, _ = _gdn_chunk_fwd(q[:, s], k[:, s], v[:, s], bt[:, s], gt[:, s], s_in)
        res[c] = (o.tolist(), s_out.tolist())
    out[r] = res


def _truth():
    from tilerl_kernels.reference import _gdn_chunk_fwd

    q, k, v, gt, bt, s0 = _inputs()
    outs, s = [], s0
    for c in range(NCHUNK):
        sl = slice(c * CHUNK, (c + 1) * CHUNK)
        o, s, _ = _gdn_chunk_fwd(q[:, sl], k[:, sl], v[:, sl], bt[:, sl], gt[:, sl], s)
        outs.append((o, s))
    return outs


def main() -> int:
    no_scan = "--no-scan" in sys.argv
    decay_a = "--decay-a" in sys.argv

    truth = _truth()
    mgr = mp.Manager()
    got = mgr.dict()
    mp.spawn(_rank, args=(CP, no_scan, decay_a, got), nprocs=CP, join=True)

    ok, rows = True, []
    for r in range(CP):
        for c, (o, s) in sorted(got[r].items()):
            o_t, s_t = truth[c]
            ro = (torch.tensor(o) - o_t).abs().max().item() / max(o_t.abs().max().item(), 1e-9)
            rs = (torch.tensor(s) - s_t).abs().max().item() / max(s_t.abs().max().item(), 1e-9)
            rows.append((r, c, ro, rs))
            if max(ro, rs) > 1e-6:
                ok = False

    # Per CHUNK, not per rank and not worst-only. Chunk 0's prefix is the
    # identity, so it is exact in every arm; under zigzag its holder also holds
    # the LAST chunk, which has the longest prefix -- so a per-rank number would
    # mix the two and a worst-only number would hide which chunks moved.
    print(" ".join(f"[r{r} c{c}] out {ro:.1e} st {rs:.1e}" for r, c, ro, rs in rows))
    if no_scan or decay_a:
        which = "no-scan" if no_scan else "decay-a"
        print(f"{which} control:", "correctly FAILED" if not ok else "PASSED -- vacuous gate")
        return 0 if not ok else 1
    print("gdn cp=2 zigzag: every chunk from its scanned prefix matches the sequential scan"
          if ok else "gdn cp=2: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
