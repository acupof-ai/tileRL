"""Is projection fusion weight-preserving? Compare the WEIGHTS, not their products.

`fuse_projections=True` changes 53 of 1000 MMLU answers with |delta logit| up to
4.46. Before blaming a kernel, the concat itself has to be cleared.

**The first version of this check compared products and was wrong.** It ran one
dense matmul on the three separate weights and one on their concat, and read the
difference as the concat's. Two GEMMs at N=48 and one at N=96 are two cuBLAS
kernels with two reduction orders, so that test measures cuBLAS, not
`_fuse_projections`. It reported 1.431e-06 on layer 4's `ab` and would have sent
the search into the concat, which is not where the 4.46 lives.

This version dequantizes and compares the weight a kernel would see -- no matmul
anywhere -- and calls `_fuse_projections` itself rather than reconstructing what
it does, so the thing under test is the production function.

Both quant formats, because the earlier filter (`all(f"{k}.wq" in params)`)
silently dropped every fp8 group: on this checkpoint `gate_up` and `ab` are fp4
(`.wq`) while `qkv` and `qkvz` are fp8 (`.w8`), and fp8 is the path that carries
an alignment guard -- the per-128-block `wscale` grid concats losslessly only if
each N but the last is a multiple of 128 (`model.py:127`). A group the guard
declines is not fused at all, which is a third outcome that has to be printed
rather than read as agreement.

No GPU: a weight comparison needs none.

    TILERL_TARGET=cpu PYTHONPATH=src:packages/tilerl-kernels/src \
    python3 scripts/probe_fusion_weights.py --source /work/Qwen3.8-27B-NVFP4
"""
from __future__ import annotations

import argparse

import torch

from tilerl.config import qwen38_27b
from tilerl.model import _fuse_projections, _projection_groups, load_hf
from tilerl_kernels import reference


def quantized(p, key) -> bool:
    return f"{key}.wq" in p or f"{key}.w8" in p


def dequant(p, key) -> torch.Tensor:
    """The f32 weight a kernel sees for one key, with its per-row epilogue scale
    folded in. fp8's oscale multiplies the output, which for a per-row scale is
    the same as scaling the row."""
    o = p.get(f"{key}.oscale")
    if f"{key}.wq" in p:
        return reference.unpack_fp4(p[f"{key}.wq"], p[f"{key}.scale"], o).float()
    w = reference.dequant_fp8(p[f"{key}.w8"], p[f"{key}.wscale"])
    return w if o is None else w * o.float().reshape(-1, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--layers", default="3,4", help="one full-attn and one gdn layer")
    a = ap.parse_args()
    cfg = qwen38_27b()
    layers = [int(x) for x in a.layers.split(",")]

    model = load_hf(cfg, a.source)
    p = model.params
    want = [(fk, g) for i in layers for fk, g in _projection_groups(cfg, i)]

    # snapshot before fusing: _fuse_projections pops the separate keys
    sep, fmt = {}, {}
    for fk, g in want:
        if not all(quantized(p, k) for k in g):
            print(f"  {fk.split('.', 1)[1]:<12} SKIPPED: not quantized in this checkpoint")
            continue
        fmt[fk] = "fp4" if f"{g[0]}.wq" in p else "fp8"
        sep[fk] = torch.cat([dequant(p, k) for k in g], dim=0)

    _fuse_projections(cfg, p)

    print(f"\n{'group':<12} {'fmt':<4} {'N':<22} {'result':<10} max|d|")
    bad = declined = 0
    for fk, g in want:
        if fk not in sep:
            continue
        name = fk.split(".", 1)[1]
        ns = str([sep[fk].shape[0]])
        if not quantized(p, fk):
            declined += 1
            print(f"{name:<12} {fmt[fk]:<4} {ns:<22} NOT FUSED  (guard declined; "
                  f"serving runs the three)")
            continue
        f = dequant(p, fk)
        same = f.shape == sep[fk].shape and torch.equal(sep[fk], f)
        d = (sep[fk] - f).abs().max().item() if f.shape == sep[fk].shape else float("nan")
        bad += not same
        print(f"{name:<12} {fmt[fk]:<4} {ns:<22} "
              f"{'EQUAL' if same else 'DIFFER':<10} {d:.3e}"
              f"{'' if same else '   <-- the concat is the bug'}")

    print()
    if bad:
        print("the concat is NOT weight-preserving: that is the bug, and the kernels "
              "are innocent.")
        raise SystemExit(1)
    print(f"every fused group reproduces the concat of its parts bit-for-bit "
          f"({declined} declined by the alignment guard, which is not a mismatch).")
    print("fusion is weight-preserving -> a kernel disagreement is the kernel's.")


if __name__ == "__main__":
    main()
