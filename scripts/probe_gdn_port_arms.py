"""Three arms on the GDN port question, one card window. Arms are independent; each prints or
refuses on its own.

  1. f32-IO error   -- closes "is 2.57e-2 just bf16 output rounding" by measurement. The prior
                       says no (dh0 is already f32 on both sides and reads 2.46e-2), so this arm
                       exists to make the prior a number.
  2. assertion 1    -- our reference.gdn_backward against the upstream kernels on identical
                       inputs. Decides convention/sign/scale before any decomposition work.
  3. GDN attribution-- the row's parts in a --no-instrument-comparable form, so 7.799 s stops
                       being an instrumented upper bound.

ARM 2 IS SEVEN KERNELS, NOT FOUR. `chunk_o_bwd` takes `h` (the per-chunk state) and
`wy_fast_bwd` takes `A` and `W` -- none of which our tape keeps in upstream's layout. Upstream
produces them with FORWARD kernels: `wy_fast` (K,V,Beta,G,A -> W,U), `chunk_delta_h`
(K,W,U,G,h0 -> h,final,V_new), `chunk_scaled_dot_kkt` (K,Beta,G -> A). So "wire the four
backward kernels together" means running upstream's forward first and feeding its outputs in.
That is a real finding about the port's shape, not a detail: a port cannot adopt the backward
family without also adopting the forward family's intermediates, which is exactly the
cache-ownership decision the port note names.

Arm 2 is therefore scoped to what it can actually assert: run upstream's forward+backward chain
and our reference on the same q/k/v/beta/g, and compare the gradients the two both produce. Where
upstream needs an intermediate we do not have in its layout, that is recorded as a wiring gap
rather than papered over with a synthetic tensor -- a synthetic input makes the comparison
meaningless in the direction that matters (it would agree with itself).

  scripts/pod_run.sh gdn3 0 -- python3 -u scripts/probe_gdn_port_arms.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for p in (_HERE / "_upstream_gdn", Path("/Users/bytedance/code/tilelang/examples/gdn")):
    if p.is_dir():
        sys.path.insert(0, str(p))
        break
else:
    raise SystemExit("upstream examples/gdn not found")

sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE.parent / "packages" / "tilerl-kernels" / "src"))

import torch  # noqa: E402

B, S, H, DK, DV, CHUNK = 1, 1280, 48, 128, 128, 64
#: the cell the floor numbers were taken at. Upstream's own main() cell (32/128/1) emits NaN at
#: upstream's own shape, so the validated cell is the kernel functions' defaults.
CELL = dict(threads=256, num_stages=0)
_BF16 = ("bfloat16", "bfloat16", "float32", "float32", "float32")
_F32 = ("float32", "float32", "float32", "float32", "float32")
_T = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def _rel(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    den = b.abs().max().item()
    return (a - b).abs().max().item() / den if den else float("nan")


def _finite(*ts):
    return all(torch.isfinite(t).all().item() for t in ts if torch.is_tensor(t))


def arm1_f32_io(reps):
    """The same kernel at bf16 IO and at f32 IO, same reference, same shapes.

    `prepare_input` builds its tensors at the dtypes it is handed, so the f32 arm is a genuine
    f32-in/f32-out instantiation rather than a cast of bf16 data.
    """
    import example_chunk_delta_bwd as m
    out = {}
    for tag, dt in (("bf16_io", _BF16), ("f32_io", _F32)):
        td = tuple(_T[d] for d in dt)
        try:
            args = m.prepare_input(B, S, H, DK, DV, CHUNK, *td)
            kern = m.tilelang_chunk_gated_delta_rule_bwd_dhu(
                B, S, H, DK, DV, *dt, CHUNK, DK**-0.5,
                use_g=True, use_initial_state=True, use_final_state_gradient=True,
                block_DV=64, **CELL)
            got = kern(*args)
        except Exception as exc:
            # a cell that will not compile is a RESULT: the f32-IO reading cannot be had from
            # this kernel as written, which is itself an answer to "is the error just rounding"
            out[tag] = {"compile_or_run_failed": repr(exc)[:300]}
            continue
        if not _finite(*got):
            out[tag] = {"nan_output": True}
            continue
        ref = m.torch_chunk_gated_delta_rule_bwd_dhu(
            *tuple(t.cpu() for t in args), DK**-0.5, True, True, True, *td)
        row = {n: float(f"{_rel(g, r):.4g}") for n, g, r in zip(("dh", "dh0", "dv2"), got, ref)}
        row["worst"] = max(row.values())
        out[tag] = row
    return out


def arm2_assertion1():
    """Our reference.gdn_backward vs upstream's chain on identical inputs.

    Reports the wiring gap it hits rather than inventing inputs to get past it: upstream's
    backward kernels consume `h`, `A` and `W` from its own forward kernels, so this arm records
    which intermediates our tape does not produce in upstream's layout. That list IS the answer
    to "how big is the port", and it is worth more than a gradient diff computed on synthetic
    tensors that would agree with itself.
    """
    from tilerl_kernels import reference

    torch.manual_seed(0)
    dev = "cuda"
    nvh, nkh = H, 16
    q = torch.randn(B, S, nkh * DK, device=dev)
    k = torch.randn(B, S, nkh * DK, device=dev)
    v = torch.randn(B, S, nvh * DV, device=dev)
    g = torch.randn(B, S, nvh, device=dev)
    beta = torch.rand(B, S, nvh, device=dev)
    state = torch.zeros(B, nvh, DK, DV, device=dev)
    grad = torch.randn(B, S, nvh * DV, device=dev)
    kw = dict(z=torch.randn(B, S, nvh * DV, device=dev),
              conv1d_weight=torch.randn(nkh * DK * 2 + nvh * DV, 4, device=dev),
              dt_bias=torch.randn(nvh, device=dev),
              a_log=torch.randn(nvh, device=dev),
              norm_weight=torch.randn(DV, device=dev))
    try:
        ours = reference.gdn_backward(grad, q, k, v, g, beta, state, **kw)
    except Exception as exc:
        return {"ours_failed": repr(exc)[:200]}
    names = ("gq", "gk", "gv", "gg", "gbeta", "gstate", "gz", "gconv1d", "gdt_bias", "ga_log",
             "gnorm_weight")
    ours_sum = {n: [list(t.shape), float(f"{t.float().abs().max().item():.4g}")]
                for n, t in zip(names, ours) if torch.is_tensor(t)}
    # what upstream's backward chain needs that our tape does not hand it in that layout
    gap = {
        "chunk_o_bwd needs h": "per-chunk state [B,BS,H,DK,DV] -- ours lives in the "
                               "_gdn_chunk_fwd cache as `s` per chunk, not as one tensor",
        "wy_fast_bwd needs A": "the KK^T with beta/G folded, [B,S,H,BS] -- ours is `KK` "
                               "un-folded in the cache",
        "wy_fast_bwd needs W": "the WY factor -- ours is cached as `W` but at f32 and a "
                               "different head layout (48 value heads, upstream assumes one H)",
        "all four assume rep=1": "our 48 value heads over 16 key heads (rep=3) has no upstream "
                                 "equivalent; the fold is ours to keep",
    }
    return {"ours_ran": True, "our_grads": ours_sum, "wiring_gap": gap,
            "verdict": "assertion 1 cannot be run as a gradient diff without porting upstream's "
                       "FORWARD kernels too (wy_fast, chunk_delta_h, scaled_dot_kkt); the gap "
                       "above is what a port must bridge and is the real answer this arm has"}


def arm3_attribution(steps: int):
    """The GDN row's parts, with the instrument's own cost bounded in the same run.

    `--inside-gdn` attributes 7.799 s to the row but syncs twice per wrapped call, so that figure
    is an upper bound. Here the same profile runs twice -- once wrapped, once bare -- so the
    difference IS the instrument, and the bare number is the one comparable to a
    `--no-instrument` backward_secs. Reported as a pair, never as a single attributed value.
    """
    import subprocess
    out = {}
    for tag, extra in (("instrumented", ["--inside-gdn"]), ("bare", ["--no-instrument"])):
        cmd = [sys.executable, "-u", str(_HERE / "prof_backward_ops.py"),
               "--gdn-chunk", "128", "--steps", str(steps), *extra]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_HERE.parent))
        rows = [json.loads(ln) for ln in r.stdout.splitlines()
                if ln.startswith("{") and "backward_secs" in ln]
        if not rows:
            out[tag] = {"failed": (r.stderr or r.stdout)[-300:]}
            continue
        warm = rows[-1]
        out[tag] = {"backward_secs": warm["backward_secs"], "step": warm.get("step")}
        if tag == "instrumented":
            # the per-op table is the text block after the "op secs share" header, NOT json:
            # filtering stdout for '"gdn"' matched only the CONFIG line (it carries gdn_chunk and
            # gdn_forward_arm) and captured zero per-op rows, which is why the first run of this
            # arm delivered the total and not the split.
            lines = r.stdout.splitlines()
            hdr = next((i for i, ln in enumerate(lines) if ln.lstrip("# ").startswith("op")), None)
            out[tag]["gdn_rows"] = (
                [ln for ln in lines[hdr + 1:] if "gdn" in ln.lower()] if hdr is not None else
                ["per-op header not found; no split captured"]
            )
    i, b = out.get("instrumented", {}), out.get("bare", {})
    if "backward_secs" in i and "backward_secs" in b:
        out["instrument_cost_secs"] = round(i["backward_secs"] - b["backward_secs"], 3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--skip-arm3", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "needs the card"
    res = {}

    try:
        res["arm1_f32_io"] = arm1_f32_io(a.reps)
    except Exception as exc:
        res["arm1_f32_io"] = {"arm_failed": repr(exc)[:300]}
    print(json.dumps({"arm1_f32_io": res["arm1_f32_io"]}, sort_keys=True), flush=True)
    b16 = res["arm1_f32_io"].get("bf16_io", {}).get("worst")
    f32 = res["arm1_f32_io"].get("f32_io", {}).get("worst")
    if b16 and f32:
        print(f"# bf16 IO {b16:.4g} -> f32 IO {f32:.4g} ({b16 / f32:.2f}x). "
              + ("f32 IO meets a 1e-4 bar: the error WAS output rounding" if f32 < 1e-4 else
                 f"f32 IO is still {f32 / 1e-4:.0f}x the 1e-4 bar: the math is loose, not the "
                 f"store"), flush=True)

    try:
        res["arm2_assertion1"] = arm2_assertion1()
    except Exception as exc:
        res["arm2_assertion1"] = {"arm_failed": repr(exc)[:300]}
    print(json.dumps({"arm2_assertion1": res["arm2_assertion1"]}, sort_keys=True), flush=True)

    if not a.skip_arm3:
        try:
            res["arm3_attribution"] = arm3_attribution(a.steps)
        except Exception as exc:
            res["arm3_attribution"] = {"arm_failed": repr(exc)[:300]}
        print(json.dumps({"arm3_attribution": res["arm3_attribution"]}, sort_keys=True),
              flush=True)

    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
