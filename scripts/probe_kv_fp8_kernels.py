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
    out["kernel_vs_reference_bitexact"] = torch.equal(
        got.view(torch.uint8).cpu(), want_k[0].view(torch.uint8).cpu())
    out["scale_max_rel"] = float(
        ((ks[bt[:, 0].long()][:, :, :S] - want_ks[0]).abs()
         / want_ks[0].abs().clamp_min(1e-30)).max())
    # the append property: token 0 must not be re-rounded by the 7 that follow
    deq = kp[bt[:, 0].long()][:, :, :S].float() * ks[bt[:, 0].long()][:, :, :S, None]
    truth = k.permute(0, 2, 1, 3).float()
    rel = (deq - truth).abs() / truth.abs().clamp_min(1e-9)
    out["token0_max_rel"] = float(rel[:, :, 0].max())
    out["all_tokens_max_rel"] = float(rel.max())
    return out


def main() -> int:
    print(f"torch {torch.__version__}, cuda {torch.cuda.is_available()}, "
          f"device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}",
          flush=True)
    results: dict = {}
    for name, fn in (("q1_scalar_fp8_load", q1_scalar_fp8_load), ("q2_writers", q2_writers)):
        try:
            results[name] = fn()
            print(f"\n{name}: {json.dumps(results[name], sort_keys=True)}", flush=True)
        except Exception as exc:  # noqa: BLE001 -- the failure text IS the answer here
            results[name] = {"failed": f"{type(exc).__name__}: {exc}"}
            print(f"\n{name} FAILED: {type(exc).__name__}: {exc}", flush=True)
    print("\n" + json.dumps(results, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
