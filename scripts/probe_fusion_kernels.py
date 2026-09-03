"""Which fusion arm is closer to exact? One layer, both kernels, one f32 reference.

`fuse_projections=True` changes 53 of 1000 MMLU answers, |delta logit| up to 4.46,
against an arm-change delta of ~0.153.

`scripts/probe_fusion_weights.py` established the precondition this probe needs:
`_fuse_projections` is weight-preserving bit-for-bit on all five groups, fp4 and
fp8. So `dequant(concat(W)) == concat(dequant(W))`, and the two arms compute the
same mathematical product -- which makes ONE f32 dense matmul the exact value
BOTH arms approximate, not a third arithmetic. That is what was missing before,
and it turns a 64-layer hidden-state trace into a single-layer comparison:

    arm 0 (unfused)  three kernel calls, outputs concatenated
    arm 1 (fused)    one kernel call on the concatenated weight
    reference        x.float() @ dequant(concat(W)).T, no kernel

`|arm - ref|` ranks the two arms directly. `x` is materialized in bf16 BEFORE
either arm sees it, because `Backend._rows` casts activations to bf16 on cuda
(`backend.py:187`) -- doing it up front makes the input bit-identical for both
arms and the reference, so the input cast cannot show up as one arm's error.

The mechanism under test is N-dependence in the dispatch. `Backend.linear_fp4`
calls `self._plan("linear_fp4", M, N, K)` and pads N to the plan's `Np`
(`backend.py:346,367`); `linear_fp8` pads `wscale` to `-(-Np // 128)` blocks
(`backend.py:437`). Fusing changes N -- 12288+1024+1024 served as one GEMM
instead of three -- so the arms can land on different tilings, different N
padding, and a different accumulation order over K.

M is swept because it selects the arm boundary too (`_MGEMV=3`, `_MX=8` on
`M = B*W`), and MMLU is prefill-only: `_PREFILL_BUCKET` is 64, so M=64 is the
value that produced the 53 flips. M=1 and M=8 are the decode ticks.

**Limitation, stated because the magnitude is the open question:** `x` is
`randn * --xrms`, not a captured activation. Which arm is closer is scale-robust;
whether the gap reaches 4.46 after 64 layers is not answered here.

    CUDA_VISIBLE_DEVICES=6 TILERL_TARGET=cuda TILELANG_CACHE_DIR=/work/tilelang_cache \
    PYTHONPATH=src:packages/tilerl-kernels/src \
    python3 scripts/probe_fusion_kernels.py --source /work/Qwen3.8-27B-NVFP4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tilerl_kernels import reference
from tilerl_kernels.backend import get_backend

from tilerl.config import qwen38_27b
from tilerl.model import _fuse_projections, _projection_groups, load_hf


def fmt_of(p, key) -> str | None:
    return "fp4" if f"{key}.wq" in p else "fp8" if f"{key}.w8" in p else None


def dequant(p, key) -> torch.Tensor:
    """The f32 weight a kernel sees, per-row epilogue scale folded in."""
    o = p.get(f"{key}.oscale")
    if f"{key}.wq" in p:
        return reference.unpack_fp4(p[f"{key}.wq"], p[f"{key}.scale"], o).float()
    w = reference.dequant_fp8(p[f"{key}.w8"], p[f"{key}.wscale"])
    return w if o is None else w * o.float().reshape(-1, 1)


def call(be, p, key, x):
    """One projection through the production dispatch, exactly as serving calls it."""
    if f"{key}.wq" in p:
        return be.linear_fp4(x, p[f"{key}.wq"], p[f"{key}.scale"],
                             oscale=p.get(f"{key}.oscale")).float()
    return be.linear_fp8(x, p[f"{key}.w8"], p[f"{key}.wscale"],
                         oscale=p.get(f"{key}.oscale")).float()


def err(a, ref) -> tuple[float, float]:
    d = (a - ref).abs()
    return d.max().item(), (d.mean() / ref.abs().mean().clamp_min(1e-30)).item()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--layers", default="3,4")
    ap.add_argument("--ms", default="1,8,64", help="M values; 64 is the prefill bucket")
    ap.add_argument("--xrms", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    be = get_backend()
    assert be.device.type == "cuda", f"the kernels under test are sm90: {be.device}"
    cfg = qwen38_27b()
    ms = [int(x) for x in a.ms.split(",")]
    torch.manual_seed(a.seed)

    model = load_hf(cfg, a.source)
    p = model.params
    groups = [(fk, g) for i in a.layers.split(",") for fk, g in _projection_groups(cfg, int(i))]
    groups = [(fk, g) for fk, g in groups if all(fmt_of(p, k) for k in g)]

    # dequantized concat per group, before _fuse_projections pops the separate keys
    refw = {fk: torch.cat([dequant(p, k) for k in g], dim=0) for fk, g in groups}
    sep_keys = {fk: list(g) for fk, g in groups}
    fmt = {fk: fmt_of(p, g[0]) for fk, g in groups}
    unfused = {k: {s: p[k + s] for s in (".wq", ".scale", ".oscale", ".w8", ".wscale")
                   if k + s in p}
               for fk, g in groups for k in g}

    _fuse_projections(cfg, p)
    for k, d in unfused.items():          # fusion popped these; both arms need them live
        p.update({k + s: v for s, v in d.items()})

    # arm-vs-arm is the load-bearing column: equal error against the reference can
    # mean the arms are identical OR that they differ below the reference's own
    # noise, and only |y0 - y1| separates those.
    print(f"{'group':<12} {'fmt':<4} {'M':>4} {'N':>6}  "
          f"{'unfused max|d|':>15} {'fused max|d|':>14}  {'rel unf':>9} {'rel fus':>9}  "
          f"{'|y0-y1|':>10}  closer")
    rows = []
    for fk, g in groups:
        if not fmt_of(p, fk):
            print(f"{fk.split('.', 1)[1]:<12} {fmt[fk]:<4} "
                  f"{'':>4} {'':>6}  NOT FUSED (alignment guard declined)")
            continue
        W = refw[fk].to(be.device)
        for M in ms:
            x = (torch.randn(M, cfg.hidden_size, generator=torch.Generator().manual_seed(
                a.seed + M), dtype=torch.float32) * a.xrms).to(be.device, torch.bfloat16)
            ref = x.float() @ W.t()
            y0 = torch.cat([call(be, p, k, x) for k in sep_keys[fk]], dim=-1)
            y1 = call(be, p, fk, x)
            (m0, r0), (m1, r1) = err(y0, ref), err(y1, ref)
            arm = (y0 - y1).abs().max().item()
            bits = torch.equal(y0, y1)
            win = ("identical" if bits else
                   "unfused" if m0 < m1 else "FUSED" if m1 < m0 else "tie")
            print(f"{fk.split('.', 1)[1]:<12} {fmt[fk]:<4} {M:>4} {W.shape[0]:>6}  "
                  f"{m0:>15.3e} {m1:>14.3e}  {r0:>9.2e} {r1:>9.2e}  {arm:>10.3e}  {win}"
                  f"{'' if m0 == m1 else '  x%.1f' % (max(m0, m1) / max(min(m0, m1), 1e-30))}")
            rows.append({"group": fk, "fmt": fmt[fk], "M": M, "N": W.shape[0],
                         "unfused_max": m0, "fused_max": m1, "unfused_rel": r0,
                         "fused_rel": r1, "arm_max": arm, "bitwise_equal": bits,
                         "closer": win})
        del W
        torch.cuda.empty_cache()

    if rows:
        pf = sum(r["closer"] == "FUSED" for r in rows)
        pu = sum(r["closer"] == "unfused" for r in rows)
        ident = sum(r["bitwise_equal"] for r in rows)
        print(f"\nbitwise identical on {ident}/{len(rows)} cells; "
              f"unfused closer on {pu}, fused on {pf}.")
        if ident == len(rows):
            print("the two arms produce the SAME BITS at every M and every group: at this\n"
                  "layer fusion changes no number, so the 53 MMLU flips do not originate\n"
                  "in these projections and the search has to move elsewhere.")
        pre = [r for r in rows if r["M"] == 64]
        if pre:
            print("at M=64, the bucket MMLU actually ran: "
                  + ", ".join(f"{r['group'].split('.', 1)[1]} {r['closer']}" for r in pre))

        # negative control: "identical" is only worth reading if this comparison can
        # come out non-identical. Perturb one row of the fused weight's scale and the
        # same call must diverge -- otherwise the two arms were never being compared.
        fk, g = next((fk, g) for fk, g in groups if fmt_of(p, fk))
        M = ms[-1]
        x = torch.randn(M, cfg.hidden_size, generator=torch.Generator().manual_seed(a.seed),
                        dtype=torch.float32).to(be.device, torch.bfloat16)
        base = call(be, p, fk, x)
        skey = f"{fk}.scale" if f"{fk}.wq" in p else f"{fk}.wscale"
        saved = p[skey].clone()
        p[skey][0, 0] = p[skey][0, 0] * 1.5
        moved = (call(be, p, fk, x) - base).abs().max().item()
        p[skey].copy_(saved)
        print(f"\nnegative control ({fk.split('.', 1)[1]}, one scale element x1.5): "
              f"max|d| {moved:.3e} -> "
              f"{'the comparison DOES detect a difference' if moved > 0 else 'BROKEN: it detects nothing, so 15/15 identical is meaningless'}")
        assert moved > 0, "the comparison cannot detect a changed weight; identical proves nothing"
    if a.out:
        o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
        (o / "kernels.json").write_text(json.dumps({"xrms": a.xrms, "rows": rows}, indent=1))
        print(f"wrote {o / 'kernels.json'}")


if __name__ == "__main__":
    main()
