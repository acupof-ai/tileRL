"""Step 1 of the fp8 KV work: do the two fp8 writers compile on sm90, and does the
gather the READERS will need lower at all?

Three questions, in the order that matters. The first two are the stop conditions 27 asked
for before the reader conversion starts; the third is the parity gate for the writers.

1. Does tilelang codegen a SCALAR elementwise load of an fp8 operand? Every fp8 use in this
   tree so far is a GEMM operand going into T.gemm or a dequant macro -- never
   `X[a, b, c, d]` on one element. The reader conversion is one multiply at the gather ONLY
   if this lowers; if it does not, the readers are not a maker parameter and step 2 stops.
2. Do the two writers compile, and does the pool round-trip through them match the reference
   quantizer bit for bit?
3. Does K_shared stay bf16 -- i.e. can the dequant produce a bf16 tile, or does it force f32
   and change the shared-memory budget?

Run: scripts/pod_run.sh kvfp8-gate 0 -- python3 scripts/probe_kv_fp8_kernels.py
"""

from __future__ import annotations

import json
import sys

import torch

sys.path.insert(0, "packages/tilerl-kernels/src")


def q1_scalar_fp8_load() -> dict:
    """Can a kernel read ONE fp8 element by index and use it as a number?

    The shape the readers need: gather out of a block table, multiply by a scale, write a
    bf16 tile. Nothing here is a gemm.
    """
    import tilelang
    import tilelang.language as T

    @tilelang.jit(target="cuda", pass_configs={"tl.disable_data_race_check": True})
    def gather(Q, Scale, Out, threads):
        NB, H, BS, D = T.const("NB, H, BS, D")
        Q: T.Tensor((NB, H, BS, D), "float8_e4m3fn")
        Scale: T.Tensor((NB, H, BS), "float32")
        Out: T.Tensor((NB, H, BS, D), "bfloat16")
        with T.Kernel(NB, H, threads=threads) as (nb, h):
            for t, d in T.Parallel(BS, D):
                # the reader's whole change: one scalar fp8 load times its scale
                Out[nb, h, t, d] = T.cast(
                    T.cast(Q[nb, h, t, d], "float32") * Scale[nb, h, t], "bfloat16")

    NB, H, BS, D = 4, 2, 16, 64
    ref = torch.randn(NB, H, BS, D, device="cuda", dtype=torch.float32) * 0.1
    scale = ref.abs().amax(-1).clamp_min(1e-12) / torch.finfo(torch.float8_e4m3fn).max
    q = (ref / scale[..., None]).to(torch.float8_e4m3fn)
    out = torch.empty(NB, H, BS, D, device="cuda", dtype=torch.bfloat16)
    gather(q, scale.contiguous(), out, 128)
    want = (q.float() * scale[..., None]).to(torch.bfloat16)
    ok = torch.equal(out, want)
    return {"q": "scalar fp8 load + scale -> bf16 tile", "compiled": True, "bitexact": ok,
            "max_abs_diff": float((out.float() - want.float()).abs().max())}


def q2_writers() -> dict:
    """Both fp8 writers compile, and write_tokens_fp8 matches the reference quantizer."""
    from tilerl_kernels import kernels_mma, reference

    out: dict = {}
    wt = kernels_mma.make_write_tokens_fp8("cuda")
    kernels_mma.make_attn_prep_fp8("cuda")  # compiles on first call; existence is the check
    out["makers_built"] = True

    B, S, H, D, NB, BS = 2, 8, 4, 64, 8, 16
    k = (torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1)
    v = (torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1)
    for t in range(S):  # growing magnitude: a per-block scale would compound here
        k[:, t] *= 1.0 + t
    kp = torch.zeros(NB, H, BS, D, device="cuda", dtype=torch.float8_e4m3fn)
    vp = torch.zeros_like(kp)
    ks = torch.ones(NB, H, BS, device="cuda", dtype=torch.float32)
    vs = torch.ones_like(ks)
    bt = torch.arange(B * 2, device="cuda", dtype=torch.int32).reshape(B, 2)
    sl = torch.full((B,), S, device="cuda", dtype=torch.int32)
    wt(k, v, kp, vp, ks, vs, bt, sl, sl, BS, 64)

    # the reference quantizer over the same tokens, laid out as the pool sees them
    want_k, want_ks = reference.quant_kv_fp8(
        k.permute(0, 2, 1, 3).reshape(B, H, S, D).unsqueeze(0), torch.float8_e4m3fn)
    got = kp[bt[:, 0].long()][:, :, :S]
    gb, wb = got.view(torch.uint8).cpu(), want_k[0].view(torch.uint8).cpu()
    out["kernel_vs_reference_bitexact"] = torch.equal(gb, wb)
    # Not bit-exact is expected if the scale differs by an ulp -- report HOW it differs, so
    # "false" is a tolerance question rather than a wrong-kernel question. One e4m3 step is
    # one byte of mantissa, so a differing byte off by 1 is a rounding tie, not a bug.
    diff = (gb.int() - wb.int()).abs()
    out["bytes_differing_frac"] = float((diff > 0).float().mean())
    out["byte_diff_max"] = int(diff.max())
    # A 1-code difference is only benign if it is a TIE: x/s exactly at the midpoint between
    # two e4m3 codes, where T.cast and torch .to may round opposite ways. Measure it -- take
    # the pre-quantization value, find the two codes bracketing it, and check it sits at the
    # midpoint. Anything not a tie is a real disagreement, however small the byte delta.
    where = diff > 0
    if bool(where.any()):
        s_b = ks[bt[:, 0].long()][:, :, :S]
        pre = (k.permute(0, 2, 1, 3).float() / s_b[..., None]).cpu()[where].abs()
        lo = torch.minimum(gb[where], wb[where]).view(torch.float8_e4m3fn).float().abs()
        hi = torch.maximum(gb[where], wb[where]).view(torch.float8_e4m3fn).float().abs()
        # relative distance from the midpoint, in units of the gap between the two codes
        off = (pre - (lo + hi) / 2).abs() / (hi - lo).abs().clamp_min(1e-30)
        out["tie_offset_max"] = float(off.max())
        out["all_differences_are_ties"] = bool(off.max() < 1e-3)
    else:
        out["tie_offset_max"] = 0.0
        out["all_differences_are_ties"] = True
    out["scale_max_rel"] = float(
        ((ks[bt[:, 0].long()][:, :, :S] - want_ks[0]).abs()
         / want_ks[0].abs().clamp_min(1e-30)).max())
    # and the dequantized values against the TRUTH, which is what actually matters
    deq_k = got.float() * ks[bt[:, 0].long()][:, :, :S, None]
    ref_deq = want_k[0].float() * want_ks[0][..., None]
    out["kernel_vs_reference_dequantized_max_rel"] = float(
        ((deq_k - ref_deq).abs() / ref_deq.abs().clamp_min(1e-9)).max())
    # the append property: token 0 must not be re-rounded by the 7 that follow
    deq = kp[bt[:, 0].long()][:, :, :S].float() * ks[bt[:, 0].long()][:, :, :S, None]
    truth = k.permute(0, 2, 1, 3).float()
    rel = (deq - truth).abs() / truth.abs().clamp_min(1e-9)
    out["token0_max_rel"] = float(rel[:, :, 0].max())
    out["all_tokens_max_rel"] = float(rel.max())
    return out


def q3_readers() -> dict:
    """The fp8 attention readers against the SAME kernel on a bf16 pool.

    The oracle is the bf16 pool through the bf16 maker, not a reimplementation: same q, same
    block table, the pool's dtype the only difference. Two numbers, because they fail
    differently -- max relative logit error, and absolute error over the row's amax, which is
    what bounds the attention score (per-element relative error on an fp8 grid is unbounded
    near zero by construction).
    """
    from tilerl_kernels import reference
    from tilerl_kernels.backend import get_backend

    be = get_backend()
    out: dict = {"arch": be.arch}
    B, S, Hq, Hkv, D, NB, BS = 1, 64, 8, 2, 64, 8, 16
    q = torch.randn(B, S, Hq, D, device=be.device, dtype=torch.bfloat16) * 0.1
    kv = torch.randn(NB, Hkv, BS, D, device=be.device, dtype=torch.bfloat16) * 0.1
    vv = torch.randn(NB, Hkv, BS, D, device=be.device, dtype=torch.bfloat16) * 0.1
    bt = torch.arange(NB, device=be.device, dtype=torch.int32).reshape(B, NB)
    sl = torch.full((B,), S, device=be.device, dtype=torch.int32)
    scale = 1.0 / (D ** 0.5)

    # the fp8 arm reads the quantized pool; the oracle reads the DEQUANTIZED one, so the only
    # difference is the rounding, not a different tensor
    kq, ks = reference.quant_kv_fp8(kv.unsqueeze(0), torch.float8_e4m3fn)
    vq, vs = reference.quant_kv_fp8(vv.unsqueeze(0), torch.float8_e4m3fn)
    kq, ks, vq, vs = kq[0], ks[0], vq[0], vs[0]
    k_deq = (kq.float() * ks[..., None]).to(torch.bfloat16)
    v_deq = (vq.float() * vs[..., None]).to(torch.bfloat16)

    ref = be.paged_attention(q, k_deq, v_deq, bt, sl, scale, seq_q_lens=sl)
    got = be.paged_attention(q, kq, vq, bt, sl, scale, seq_q_lens=sl, k_scale=ks, v_scale=vs)
    r, g = ref.float(), got.float()
    out["shape"] = list(g.shape)
    # arm 1: the KERNEL's dequant against the same values dequantized by torch. 0.0 here means
    # the in-kernel multiply is exact -- it is NOT fp8's accuracy cost, because both arms read
    # the same rounded numbers.
    out["kernel_dequant_max_abs_err"] = float((g - r).abs().max())
    out["kernel_dequant_max_rel_err"] = float(((g - r).abs() / r.abs().clamp_min(1e-9)).max())
    # arm 2: what fp8 KV actually costs -- the fp8 pool against the ORIGINAL bf16 one. Absolute
    # error over the output's amax, because per-element relative error on an fp8 grid is
    # unbounded near zero by construction.
    true_ref = be.paged_attention(q, kv, vv, bt, sl, scale, seq_q_lens=sl).float()
    out["fp8_vs_bf16_pool_err_over_amax"] = float(
        (g - true_ref).abs().max() / true_ref.abs().max())
    out["fp8_vs_bf16_pool_max_abs_err"] = float((g - true_ref).abs().max())
    out["bf16_pool_out_amax"] = float(true_ref.abs().max())
    # non-vacuous: a WRONG scale must move the output, or this compares two identical paths
    bad = be.paged_attention(q, kq, vq, bt, sl, scale, seq_q_lens=sl,
                             k_scale=ks.roll(1, 0), v_scale=vs)
    out["scale_roll_moves_output"] = not torch.allclose(bad.float(), g, atol=1e-6)
    return out


def main() -> int:
    print(f"torch {torch.__version__}, cuda {torch.cuda.is_available()}, "
          f"device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}",
          flush=True)
    results: dict = {}
    torch.manual_seed(7)
    try:
        results["q1_scalar_fp8_load"] = q1_scalar_fp8_load()
        print(f"\nq1_scalar_fp8_load: {json.dumps(results['q1_scalar_fp8_load'], sort_keys=True)}",
              flush=True)
    except Exception as exc:  # noqa: BLE001 -- the failure text IS the answer here
        results["q1_scalar_fp8_load"] = {"failed": f"{type(exc).__name__}: {exc}"}
        print(f"\nq1_scalar_fp8_load FAILED: {type(exc).__name__}: {exc}", flush=True)

    # 8 seeds, because unseeded this arm read bitexact=false on one run and true on the next:
    # the difference was the DRAW. One seed would only move which answer gets reported, so
    # the claim is over seeds -- if any draw is not bit-exact, that is a tie-break question
    # and byte_diff_max says whether it is one e4m3 step or something worse.
    arms = []
    for seed in range(8):
        torch.manual_seed(seed)
        try:
            arms.append({"seed": seed, **q2_writers()})
        except Exception as exc:  # noqa: BLE001
            arms.append({"seed": seed, "failed": f"{type(exc).__name__}: {exc}"})
            print(f"\nq2_writers seed={seed} FAILED: {type(exc).__name__}: {exc}", flush=True)
            break
    ok = [a for a in arms if "failed" not in a]
    results["q2_writers"] = {
        "seeds": len(arms),
        "bitexact_all_seeds": bool(ok) and all(a["kernel_vs_reference_bitexact"] for a in ok),
        "byte_diff_max_over_seeds": max((a["byte_diff_max"] for a in ok), default=None),
        "bytes_differing_frac_max": max((a["bytes_differing_frac"] for a in ok), default=None),
        "all_differences_are_ties": bool(ok) and all(a["all_differences_are_ties"] for a in ok),
        "tie_offset_max_over_seeds": max((a["tie_offset_max"] for a in ok), default=None),
        "scale_max_rel_over_seeds": max((a["scale_max_rel"] for a in ok), default=None),
        "token0_max_rel_over_seeds": max((a["token0_max_rel"] for a in ok), default=None),
        "arms": arms,
    }
    print(f"\nq2_writers over {len(arms)} seeds: "
          f"{json.dumps({k: v for k, v in results['q2_writers'].items() if k != 'arms'}, sort_keys=True)}",
          flush=True)

    torch.manual_seed(11)
    try:
        results["q3_readers"] = q3_readers()
        print(f"\nq3_readers: {json.dumps(results['q3_readers'], sort_keys=True)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        results["q3_readers"] = {"failed": f"{type(exc).__name__}: {exc}"}
        print(f"\nq3_readers FAILED: {type(exc).__name__}: {exc}", flush=True)

    print("\n" + json.dumps(results, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
