"""What would the GDN backward cost if the upstream tilelang kernels did it? (row 49, with numbers)

The GDN row is 7.799 s of a 23.194 s backward at C=128 -- 33.6%, the largest row, and the only
one with no measured floor. Every other lever is bounded near 1.1x
(wins/2026-09-07-fp4-backward-warpgroup.md re-rank), so this number decides whether the backward
is done or has a 2-3x left in it.

FOUR kernels, not one. Our `reference.gdn_backward` computes q/k/v/beta/g/state gradients plus
the norm and gate adjoints; upstream's `chunk_delta_bwd` returns dh/dh0/dv2 ONLY. Timing that
one and calling it "the GDN floor" would floor a fraction of the row and label it the whole --
the same error as pricing a region's forward and calling it the region. So the floor is the sum
over the family upstream actually ships:

  chunk_delta_bwd       dh, dh0, dv2      (the state-gradient scan)
  chunk_o_bwd           dq, dk, dw, dg    (the output adjoint)
  wy_fast_bwd_split     dk, dv, dbeta, dg (the WY representation's adjoint)
  scaled_dot_kkt        the KK^T the WY solve needs

Even that sum is a LOWER bound on a port: it is the kernels in isolation, with no glue, no
head-group fold (our 48 value heads over 16 key heads), no norm/gate adjoint, and no dtype
conversions at the boundaries. Reported as such.

Shapes are ours, not the examples': B=1 per call (384 calls = 48 GDN layers x 8 rows at micro=1),
S=1280, H=48 value heads, DK=DV=128, chunk 64. The examples run B=1 S=32768 H=8, which is 25x
the sequence and a sixth of the heads -- a per-call number from that shape says nothing about
ours.

Arm 2 is the numeric error of `chunk_delta_bwd` against **upstream's own f32 torch reference**, at
our shapes -- NOT against our `reference.gdn_backward`. The two compute different decompositions
(our chunk cache keeps M/W/d; upstream re-derives them), so a direct tensor comparison would be
measuring two algorithms rather than two precisions. And only that one example ships a torch twin,
so a port's error on dq/dk/dw/dbeta/dg is unmeasured here. That is a real limit on what this
window can answer for the bar question
(errors/2026-09-07-what-gradient-error-is-acceptable.md), not a number to interpolate.

  scripts/pod_run.sh gdnfloor 6 -- python3 -u scripts/probe_gdn_upstream_floor.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_UPSTREAM = Path("/Users/bytedance/code/tilelang/examples/gdn")
#: the pod has no tilelang checkout, only the installed wheel, so the examples are
#: staged into the synced tree; the reference repo stays read-only.
_POD_UPSTREAM = Path(__file__).resolve().parent / "_upstream_gdn"
for p in (_POD_UPSTREAM, _UPSTREAM):
    if p.is_dir():
        sys.path.insert(0, str(p))
        _SRC = p
        break
else:
    raise SystemExit("upstream examples/gdn not found; sync tilelang to the pod first")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "packages" / "tilerl-kernels" / "src")
)

import torch  # noqa: E402

#: our per-call GDN shape: micro=1 puts one row in each call, 48 value heads, T=1280, chunk 64
B, S, H, DK, DV, CHUNK = 1, 1280, 48, 128, 128, 64
CALLS = 384  # 48 GDN layers x 8 rows in one GRPO step
#: the examples take dtypes POSITIONALLY as strings for the kernel and as torch dtypes for
#: prepare_input, in this order, and not every example takes all five: scaled_dot_kkt takes the
#: first three and no DV. Passing them by keyword breaks on three of the four signatures.
_DT5 = ("bfloat16", "bfloat16", "float32", "float32", "float32")
_TDT5 = (torch.bfloat16, torch.bfloat16, torch.float32, torch.float32, torch.float32)


def _finite(out) -> bool:
    """Every returned tensor is finite. A timing from a kernel computing NaN measures a launch,
    not the work: the first run of this probe reported 0.769 ms/call and 26.4x from a cell whose
    dh/dh0/dv2 were entirely NaN, because nothing checked. Upstream's own main() config
    (block_DV=32, threads=128, num_stages=1) is that cell, and it is NaN at upstream's OWN shape
    as well as ours -- the example's defaults (64/256/0) are clean. Copying a config from an
    example's main() is not the same as copying a validated one.
    """
    ts = out if isinstance(out, (tuple, list)) else (out,)
    return all(torch.isfinite(t).all().item() for t in ts if torch.is_tensor(t))


def _median_ms(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[0], ts[-1]


def _rel_err(a, b):
    """max |a-b| / max|b| -- the estimator the board's gradient rows use."""
    a, b = a.detach().float(), b.detach().float()
    den = b.abs().max().item()
    return (a - b).abs().max().item() / den if den else float("nan")


def _arm_precision(reps):
    """Arm 2: the kernel's gradients against the example's own torch reference.

    Only `chunk_delta_bwd` ships one (`torch_chunk_gated_delta_rule_bwd_dhu`); the other three
    examples have no torch twin, so a port's error on dq/dk/dw/dbeta/dg is NOT measurable here
    and is reported as absent rather than inferred from this one kernel. The reference is
    upstream's, so this measures the KERNEL against ITS OWN math -- not against our
    `reference.gdn_backward`, which computes a different decomposition (our chunk cache keeps
    M/W/d; upstream re-derives them), so a direct tensor comparison would be comparing two
    algorithms and not two precisions.
    """
    import example_chunk_delta_bwd as m
    args = m.prepare_input(B, S, H, DK, DV, CHUNK, *_TDT5)
    kern = m.tilelang_chunk_gated_delta_rule_bwd_dhu(
        B, S, H, DK, DV, *_DT5, CHUNK, DK**-0.5,
        use_g=True, use_initial_state=True, use_final_state_gradient=True,
        block_DV=64, threads=256, num_stages=0)
    dh, dh0, dv2 = kern(*args)
    # The example's torch reference allocates its accumulators with no `device=`, so they land on
    # the CPU while `prepare_input` puts every input on the card -- it raises "found at least two
    # devices" if called as shipped. Upstream's defect, not ours; run the reference entirely on
    # CPU copies. That also drops its `allow_tf32` path, so this is f32-vs-kernel, which is the
    # comparison wanted anyway.
    cpu = tuple(t.cpu() for t in args)
    ref = m.torch_chunk_gated_delta_rule_bwd_dhu(*cpu, DK**-0.5, True, True, True, *_TDT5)
    if not _finite((dh, dh0, dv2)):
        return {"nan_output": True, "note": "kernel produced NaN; no error figure from this cell"}
    out = {}
    for name, got, want in zip(("dh", "dh0", "dv2"), (dh, dh0, dv2), ref):
        out[name] = float(f"{_rel_err(got.cpu(), want):.4g}")
    out["worst"] = max(out.values())
    return out


def _arm_delta_bwd(reps):
    """dh/dh0/dv2: the state-gradient scan over chunks."""
    import example_chunk_delta_bwd as m
    kern = m.tilelang_chunk_gated_delta_rule_bwd_dhu(
        B, S, H, DK, DV, *_DT5, CHUNK, DK**-0.5,
        use_g=True, use_initial_state=True, use_final_state_gradient=True,
        block_DV=64, threads=256, num_stages=0)
    args = m.prepare_input(B, S, H, DK, DV, CHUNK, *_TDT5)
    return kern, args, _median_ms(lambda: kern(*args), reps)


def _arm_o_bwd(reps):
    """dq/dk/dw/dg: the output adjoint."""
    import example_chunk_o_bwd as m
    kern = m.tilelang_chunk_o_bwd_dqkwg(
        B, S, H, DK, DV, *_DT5, CHUNK, DK**-0.5,
        use_g=True, use_dw=True, block_DK=64, block_DV=64, threads=256, num_stages=0)
    args = m.prepare_input(B, S, H, DK, DV, CHUNK, *_TDT5)
    return kern, args, _median_ms(lambda: kern(*args), reps)


def _arm_wy_bwd(reps):
    """dk/dv/dbeta/dg for the WY representation -- BOTH kernels, because it is two passes.

    `tilelang_wy_fast_bwd` produces (dA, dk, dv, dbeta, dg) via out_idx; `..._bwd_split` is a
    SECOND pass that consumes that dA and writes dk/dv/dbeta_k/dg_A_positive/dg_A_negative into
    caller-allocated tensors (13 params, no out_idx -- which is why passing 7 raised "expected 13
    inputs"). Upstream's own run_test runs both and then combines
    (`dbeta + dbeta_k`, `dg + dg_A_pos.sum(-1) - dg_A_neg.sum(-1)`), so timing either alone
    prices half the adjoint.
    """
    import example_wy_fast_bwd_split as m
    args = m.prepare_input(B, S, H, DK, DV, CHUNK, *_TDT5)
    k1 = m.tilelang_wy_fast_bwd(
        B, S, H, DK, DV, *_DT5, CHUNK, block_DK=64, block_DV=64, threads=256, num_stages=0)
    k2 = m.tilelang_wy_fast_bwd_split(
        B, S, H, DK, DV, *_DT5, CHUNK, block_DK=64, block_DV=64, threads=256, num_stages=0)
    dA, dk, dv, dbeta, dg = k1(*args)
    BS = CHUNK
    dev = dk.device
    outs = (dA, dk, dv,
            torch.empty(B, S, H, dtype=torch.bfloat16, device=dev),
            torch.empty(B, S, H, BS, dtype=torch.float32, device=dev),
            torch.empty(B, S, H, BS, dtype=torch.float32, device=dev))

    def both():
        k1(*args)
        k2(*args, *outs)

    return (k1, k2), args, _median_ms(both, reps)


def _arm_kkt(reps):
    """the KK^T the WY solve consumes. Takes 3 dtypes, not 5, and no DV."""
    import example_chunk_scaled_dot_kkt as m
    kern = m.tilelang_chunk_scaled_dot_kkt_fwd(
        B, S, H, DK, CHUNK, *_DT5[:3], use_g=True, block_S=CHUNK, block_DK=64,
        threads=256, num_stages=0)
    args = m.prepare_input(B, S, H, DK, *_TDT5[:3])
    return kern, args, _median_ms(lambda: kern(*args), reps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    assert torch.cuda.is_available(), "this needs the card"
    print(json.dumps({"upstream": str(_SRC), "B": B, "S": S, "H": H, "DK": DK, "DV": DV,
                      "chunk": CHUNK, "calls_per_step": CALLS}), flush=True)

    arms = (("chunk_delta_bwd", _arm_delta_bwd), ("chunk_o_bwd", _arm_o_bwd),
            ("wy_fast_bwd_split", _arm_wy_bwd), ("scaled_dot_kkt", _arm_kkt))
    rows, total = [], 0.0
    for name, fn in arms:
        try:
            kern, args, (med, lo, hi) = fn(a.reps)
            # the gate: a timing from a kernel computing NaN measures a launch, not the work
            k1 = kern[0] if isinstance(kern, tuple) else kern
            if not _finite(k1(*args)):
                row = {"kernel": name, "nan_output": True, "median_ms_unusable": round(med, 4)}
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                continue
            row = {"kernel": name, "median_ms": round(med, 4), "min_ms": round(lo, 4),
                   "max_ms": round(hi, 4), "step_secs": round(med * CALLS / 1e3, 3)}
            total += med
        except Exception as exc:
            # a shape or signature this example refuses is a finding: the port would hit it too
            row = {"kernel": name, "error": repr(exc)[:240]}
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    ok = [r for r in rows if "median_ms" in r]
    print(f"\n# {len(ok)}/{len(arms)} kernels timed")
    if len(ok) == len(arms):
        step = total * CALLS / 1e3
        print(f"# upstream family sums to {total:.4f} ms/call -> {step:.3f} s over {CALLS} calls")
        print(f"# our GDN row is 7.799 s attributed (instrumented arm, upper bound) -> "
              f"{7.799 / step:.2f}x")
        print(f"# a port capturing all of it would take backward_secs 23.194 -> "
              f"{23.194 - (7.799 - step):.3f} s ({23.194 / (23.194 - (7.799 - step)):.3f}x)")
    else:
        print("# INCOMPLETE: the sum is not a floor for the row until every kernel is timed. "
              "The rows above are what they are, individually.")
    print("# LOWER bound on a port either way: kernels in isolation, no glue, no head-group "
          "fold (48 value heads over 16 key heads), no norm/gate adjoint, no boundary casts.")

    # Arm 2: precision, where a reference exists to measure it against
    try:
        prec = _arm_precision(a.reps)
        print(f"\n{json.dumps({'precision_chunk_delta_bwd': prec}, sort_keys=True)}")
        print(f"# worst relative error {prec['worst']:.3g} vs upstream's own f32 torch reference, "
              f"at OUR shapes")
        print("# This covers dh/dh0/dv2 only. The other three kernels ship no torch twin, so a "
              "port's error on dq/dk/dw/dbeta/dg is UNMEASURED -- not inferable from this row.")
        rows.append({"precision_chunk_delta_bwd": prec})
    except Exception as exc:
        print(f"\n# precision arm FAILED: {repr(exc)[:200]}")
        rows.append({"precision_error": repr(exc)[:200]})
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
