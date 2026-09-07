"""Assertion 1: does our GDN backward agree with an independent implementation of the same
adjoint?

The port question is not "are the kernels fast" -- the floor probe answered that (0.986 ms/call,
20.6x, step ceiling 1.470x). It is whether a ported backward would compute the SAME gradient our
tape computes today. A numerical gradcheck cannot answer it: a gradcheck validates a backward
against its own forward, so a consistent convention difference (a gate scale, a sign, a decay
placement) passes gradcheck on both sides and trains differently.

WHY fla AND NOT THE FOUR UPSTREAM KERNELS. An earlier plan was to wire
chunk_delta_bwd + chunk_o_bwd + wy_fast_bwd(_split) + scaled_dot_kkt together by hand and diff
that against reference.gdn_backward. That arm has a fatal ambiguity: MY wiring error and a real
convention mismatch produce the same symptom, and I am the least reliable part of it. fla 0.5.2
ships `chunk_gated_delta_rule_bwd` (fla/ops/gated_delta_rule/chunk.py:126) which returns the whole
adjoint -- dq, dk, dv, dbeta, dg, dh0, dA_log, ddt_bias -- and wires those same stages itself
(recompute_w_u_fwd -> chunk_gated_delta_rule_fwd_h -> chunk_bwd_dv_local ->
chunk_gated_delta_rule_bwd_dhu -> chunk_bwd_dqkwg -> prepare_wy_repr_bwd). Same decomposition,
wiring owned by the people who designed it. A disagreement is then about the math, which is the
question.

THREE CONVENTIONS THIS PROBE PINS, each of which would otherwise show up as a "wrong gradient":

  1. CHUNK. reference.py:591 chunks the eager backward at _GDN_CHUNK=128; backend.py:28 chunks the
     sm90 WY forward at _WY_CHUNK=64; fla's fused kkt+solve path requires BT==64
     (chunk_fwd.py:378). Three values. Everything here runs at 64 and says so.
  2. GATE BASE. fla scales g by RCP_LN2 (=1/ln2) on entry and uses exp2 in its kernels
     (chunk.py:63, wy_fast.py:90); ours is natural-log with the 1/ln2 folded in at the exp2 call
     sites (kernels_gdn.py:428). Mathematically identical, but the STORED g differs by 1/ln2, so
     handing fla our g raw would be a scale error dressed as a convention mismatch. We pass g in
     fla's own convention by letting fla do its own cumsum from the same pre-cumsum gate.
  3. HEAD FOLD. Our 48 value heads over 16 key heads (rep=3). fla takes k at H and v at HV and
     folds internally, so no repeat_interleave is done here -- doing one would compare a folded
     gradient against an unfolded one.

WHAT A RESULT MEANS, decided before the numbers exist so the reading is not fitted to them:
  small (<~1e-3)      -> conventions agree; precision is the only open question.
  large + structured  -> a sign or scale error; shows as a near-constant RATIO, not noise.
  same order as 2.7e-2-> UNRESOLVED. The upstream kernels already miss their own f32 reference by
                         2.7e-2 (wins/2026-09-07 + the port note), so an error of that size cannot
                         be separated from known precision loss by this arm. Reported as such,
                         not read as agreement.

  scripts/pod_run.sh gdna1 0 -- python3 -u scripts/probe_gdn_assertion1.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE.parent / "packages" / "tilerl-kernels" / "src"))

import torch  # noqa: E402

#: our per-call GDN shape, but at fla's fused chunk. 48 value heads over 16 key heads.
B, S, NKH, NVH, DK, DV = 1, 1280, 16, 48, 128, 128
CHUNK = 64  # pinned: fla's fused kkt+solve branch, and backend.py's _WY_CHUNK


def _rel(a, b):
    """max|a-b| / max|b| -- the estimator the board's gradient rows use."""
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    den = b.abs().max().item()
    return (a - b).abs().max().item() / den if den else float("nan")


def _ratio(a, b):
    """Median a/b over the entries where b is large. A sign or scale error is a near-constant
    ratio; noise is not. This is what separates outcome 2 from outcome 3."""
    a, b = a.detach().float().cpu().flatten(), b.detach().float().cpu().flatten()
    m = b.abs() > b.abs().max() * 0.01
    if m.sum() < 8:
        return None
    r = (a[m] / b[m])
    return [float(f"{r.median():.4g}"), float(f"{r.std():.4g}")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "fla is Triton, so this needs the card"
    from fla.ops.gated_delta_rule.chunk import (
        chunk_gated_delta_rule_bwd,
        chunk_gated_delta_rule_fwd_intra,
    )
    from fla.ops.gated_delta_rule.gate import gdn_gate_chunk_cumsum
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2
    from tilerl_kernels import reference

    torch.manual_seed(a.seed)
    dev = "cuda"
    out: dict = {"shape": dict(B=B, S=S, NKH=NKH, NVH=NVH, DK=DK, DV=DV, chunk=CHUNK)}

    # ---- the shared inputs, in OUR layout -------------------------------------------------
    q = torch.randn(B, S, NKH * DK, device=dev)
    k = torch.randn(B, S, NKH * DK, device=dev)
    v = torch.randn(B, S, NVH * DV, device=dev)
    g = torch.randn(B, S, NVH, device=dev)
    beta = torch.randn(B, S, NVH, device=dev)
    state = torch.zeros(B, NVH, DK, DV, device=dev)
    go = torch.randn(B, S, NVH * DV, device=dev)
    kw = dict(z=torch.randn(B, S, NVH * DV, device=dev),
              conv1d_weight=torch.randn(NKH * DK * 2 + NVH * DV, 4, device=dev) * 0.1,
              dt_bias=torch.randn(NVH, device=dev),
              a_log=torch.randn(NVH, device=dev),
              norm_weight=torch.randn(DV, device=dev))

    # ---- ours ----------------------------------------------------------------------------
    try:
        ours = reference.gdn_backward(go, q, k, v, g, beta, state, **kw)
    except Exception as exc:
        out["ours_failed"] = repr(exc)[:300]
        print(json.dumps(out, sort_keys=True), flush=True)
        return 1
    names = ("gq", "gk", "gv", "gg", "gbeta", "gstate", "gz", "gconv1d", "gdt_bias", "ga_log",
             "gnorm_weight")
    ours_d = dict(zip(names, ours))

    # ---- fla, fed the SAME post-prep tensors --------------------------------------------
    # reference.gdn_prep is the front half (conv+silu, q/k L2 norm with 1/sqrt(DK) folded into q,
    # log gate, sigmoid beta). fla's chunk backward starts AFTER the conv/norm, so the conv and
    # output-norm adjoints have no fla counterpart and are excluded rather than approximated.
    # The GATE is not excluded: fla's use_gate_in_kernel path computes it from the raw pre-softplus
    # gate and returns dA_log/ddt_bias, and its math is the same as ours --
    # `gate = -exp(A_log) * softplus(g + bias)` (fla/ops/gated_delta_rule/gate.py:150 vs
    # reference.py:914) -- so dg/dA_log/ddt_bias ARE comparable, which matters because those are
    # the three the upstream tilelang subset never covered.
    qn, kn, v_raw, gt, bt, _ = reference.gdn_prep(
        q, k, v, g, beta, DK, conv1d_weight=kw["conv1d_weight"], dt_bias=kw["dt_bias"],
        a_log=kw["a_log"])
    # fla takes k at NKH and v at NVH and folds internally: no repeat_interleave here.
    qf, kf = qn.contiguous().bfloat16(), kn.contiguous().bfloat16()
    vf, bf = v_raw.contiguous().bfloat16(), bt.contiguous().bfloat16()
    # `g_raw` is the gate BEFORE softplus/a_log/dt_bias, which is what gdn_gate_* consumes. The
    # conv+silu applies to q/k/v only -- reference.py:777 reads `g` straight from the argument --
    # so the raw `g` here IS the same tensor both sides gate from, with no prep in between.
    g_raw = g.contiguous().float()
    out["gate_in_kernel"] = True
    try:
        gf = gdn_gate_chunk_cumsum(g=g_raw, A_log=kw["a_log"].float(), scale=RCP_LN2,
                                   dt_bias=kw["dt_bias"].float(), chunk_size=CHUNK)
    except Exception as exc:
        # fall back to the gate-outside path: dA_log/ddt_bias then come back None and are
        # reported as uncovered rather than silently missing.
        out["gate_in_kernel"] = False
        out["gate_cumsum_failed"] = repr(exc)[:200]
        gf = chunk_local_cumsum(gt.float().contiguous(), chunk_size=CHUNK, scale=RCP_LN2)
    try:
        _, _, A = chunk_gated_delta_rule_fwd_intra(
            k=kf, v=vf, g=gf, beta=bf, chunk_size=CHUNK)
        fla = chunk_gated_delta_rule_bwd(
            q=qf, k=kf, v=vf, g=gf, beta=bf, A=A, scale=1.0,
            initial_state=state.float(), do=go.view(B, S, NVH, DV).contiguous().bfloat16(),
            dht=None, chunk_size=CHUNK,
            **(dict(use_gate_in_kernel=True, g_input=g_raw, A_log=kw["a_log"].float(),
                    dt_bias=kw["dt_bias"].float()) if out["gate_in_kernel"] else {}))
    except Exception as exc:
        out["fla_failed"] = repr(exc)[:400]
        out["note"] = ("fla's chunk backward refused these inputs; the arm has no number. "
                       "That is a result about the interface, not about the gradients.")
        print(json.dumps(out, sort_keys=True), flush=True)
        return 1
    dq, dk_, dv_, db, dg_, dh0, dA_log, ddt_bias = fla

    # ---- compare, on the four the two sides both produce in the same basis ---------------
    # ours folds gq/gk back onto 16 key heads (reference.py:957); fla returns them at its own
    # head count, so the fold is applied to fla's before comparing -- stated, not silent.
    rep = NVH // NKH
    if dq.shape[2] == NVH:
        dq = dq.reshape(B, S, NKH, rep, DK).sum(3)
        dk_ = dk_.reshape(B, S, NKH, rep, DK).sum(3)
    rows = {}
    for nm, mine, theirs in (
        ("gq", ours_d["gq"].reshape(B, S, NKH, DK), dq),
        ("gk", ours_d["gk"].reshape(B, S, NKH, DK), dk_),
        ("gv", ours_d["gv"].reshape(B, S, NVH, DV), dv_),
        ("gbeta", ours_d["gbeta"].reshape(B, S, NVH), db),
        ("gg", ours_d["gg"].reshape(B, S, NVH), dg_),
        ("gstate", ours_d["gstate"], dh0),
        ("ga_log", ours_d["ga_log"], dA_log),
        ("gdt_bias", ours_d["gdt_bias"], ddt_bias),
    ):
        if theirs is None:
            rows[nm] = {"uncovered": "fla returned None (gate computed outside the kernel)"}
            continue
        if tuple(mine.shape) != tuple(theirs.shape):
            rows[nm] = {"shape_mismatch": [list(mine.shape), list(theirs.shape)]}
            continue
        rows[nm] = {"rel": float(f"{_rel(mine, theirs):.4g}"),
                    "ratio_med_std": _ratio(mine, theirs)}
    out["grads"] = rows
    finite = [r["rel"] for r in rows.values() if "rel" in r]
    out["worst_rel"] = max(finite) if finite else None
    out["compared"] = sorted(n for n, r in rows.items() if "rel" in r)
    out["not_compared"] = sorted(n for n, r in rows.items() if "rel" not in r)

    # ---- the reading, by the rule stated in the docstring, not fitted after -------------
    w = out["worst_rel"]
    if w is None:
        out["verdict"] = "no comparable rows; see shape_mismatch entries"
    elif w < 1e-3:
        out["verdict"] = ("conventions AGREE (worst rel < 1e-3). Precision is the remaining "
                          "question, and the 2.7e-2 kernel-vs-reference gap is separate.")
    elif any(r.get("ratio_med_std") and r["ratio_med_std"][1] < 0.05
             for r in rows.values() if "rel" in r):
        out["verdict"] = ("a SCALE or SIGN convention differs: at least one grad's ratio is "
                          "near-constant (std < 0.05). Fixable, and it must be fixed before a "
                          "port, but it is not a precision result.")
    elif w < 5e-2:
        out["verdict"] = ("UNRESOLVED: the disagreement is the same order as the 2.7e-2 the "
                          "upstream kernels already miss their own f32 reference by, so this arm "
                          "cannot separate a convention mismatch from known precision loss.")
    else:
        out["verdict"] = ("the two adjoints DISAGREE beyond any precision explanation "
                          "(worst rel >= 5e-2) and the ratio is not constant: a real "
                          "decomposition difference, not a scale.")
    print(json.dumps(out, indent=1, sort_keys=True), flush=True)
    print("\n# EXCLUDED from this comparison, and not inferable from it: gz, gconv1d, gdt_bias,")
    print("# ga_log, gnorm_weight. fla's chunk backward starts after the prep and ends before")
    print("# the norm/gate epilogue, so those five adjoints are OURS ALONE and unchecked here.")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
