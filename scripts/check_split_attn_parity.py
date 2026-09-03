"""Split-KV decode attention == generic paged attention (the correctness gate).

The split kernel re-derives the online softmax in two phases, so a sign or a
missing rescale would show up here and nowhere else.

The kernel ships in the sm70 cell only — the win is filling 80 SMs, which is a
loss where T.Kernel lowers to a serial loop. But the SOURCE is target-neutral,
so this builds it straight from kernels.py and gates the math on whatever
target is selected: CPU in CI, and sm70 on the pod.

  TILERL_TARGET=cpu python3 scripts/check_split_attn_parity.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "packages/tilerl-kernels/src")
)

import torch
from tilerl_kernels import kernels
from tilerl_kernels.backend import get_backend
from tilerl_kernels.registry import SM70_KVSPLIT, SM70_KVSPLIT_WIDE


def main() -> None:
    os.environ.setdefault("TILERL_TARGET", "cpu")
    backend = get_backend()
    tgt = backend.target
    generic = kernels.make_paged_attention(tgt)
    torch.manual_seed(0)
    B, H, Hkv, D, BLOCK = 1, 4, 2, 64, 16
    dev = backend.device
    worst_l = float("inf")

    # Both shipped split counts: backend.py picks by query width, so gating only
    # one leaves half the dispatch untested.
    for KVSPLIT in (SM70_KVSPLIT, SM70_KVSPLIT_WIDE):
        # Built from source, not the registry: the kernel is registered for sm70
        # only, but its math must hold on every target it can compile for.
        split = kernels.make_paged_attention_split(tgt, KVSPLIT=KVSPLIT)
        combine = kernels.make_paged_attention_split_combine(tgt, KVSPLIT=KVSPLIT)
        # Lengths straddling BLOCK=16 and this KVSPLIT so empty and ragged
        # slices, and slices shorter than one page, are all exercised. S>1 is a
        # speculative verify width: each query gets its own causal window, which
        # is the part a split can silently get wrong.
        for n in sorted({1, 15, 16, 17, 37, 100, 129, KVSPLIT - 1, KVSPLIT, KVSPLIT + 1}):
            for S in (1, 2, 4):
                if n < S:
                    continue
                nb = (n + BLOCK - 1) // BLOCK
                q = torch.randn(B, S, H, D, device=dev)
                kc = torch.randn(nb, Hkv, BLOCK, D, device=dev)
                vc = torch.randn(nb, Hkv, BLOCK, D, device=dev)
                bt = torch.arange(nb, dtype=torch.int32, device=dev).unsqueeze(0)
                sl = torch.tensor([n], dtype=torch.int32, device=dev)
                sql = torch.tensor([S], dtype=torch.int32, device=dev)
                scale = D**-0.5

                po, pm, pl = split(q, kc, vc, bt, sl, sql, float(scale), BLOCK, 64)
                got = combine(po, pm, pl, 64)
                ref = generic(q, kc, vc, bt, sl, sql, float(scale), block_size=BLOCK, threads=64)

                # combine divides by l = sum_s w_s PL_s, so an all-empty row would
                # give 0/0. It is unreachable by construction -- n >= 1, so
                # per = ceildiv(n, KVSPLIT) >= 1 and split 0 gets p1 = min(n, per) >= 1,
                # i.e. it always runs a tile holding key 0, which every query may
                # attend. Nothing said so, and the failure would be silent: my m[0]
                # init is finite (-1e30), so an empty row divides by exactly 0 and
                # yields inf rather than the NaN an -inf init would give.
                w = torch.exp(pm.float() - pm.float().amax(dim=-1, keepdim=True))
                lsum = (w * pl.float()).sum(dim=-1)
                assert lsum.min().item() > 0.0, (
                    f"combine would divide by {lsum.min().item()} at "
                    f"KVSPLIT={KVSPLIT} n={n} S={S}: an all-empty row is reachable"
                )
                worst_l = min(worst_l, lsum.min().item())

                d = (got.float() - ref.float()).abs().max().item()
                ok = torch.allclose(got.float(), ref.float(), rtol=1e-2, atol=1e-3)
                print(f"ks={KVSPLIT:3d} n={n:4d} S={S}  max|split-generic|={d:.3e} "
                      f"{'OK' if ok else 'FAIL'}")
                assert ok, f"split-KV diverges at KVSPLIT={KVSPLIT} n={n} S={S}: {d}"

    print(f"parity OK ({tgt}) at KVSPLIT {SM70_KVSPLIT} and {SM70_KVSPLIT_WIDE}")
    print(f"combine denominator: min l = {worst_l:.4f} over every shape above (must be > 0)")


if __name__ == "__main__":
    main()
