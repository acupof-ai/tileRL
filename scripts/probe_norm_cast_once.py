"""Does casting the q/k norm output once remove the 2.0007x, and what does it cost?

Measured (probe_attn_prep.py): the discrete prelude sits 2.0007x further from a
dense f64 norm+rope than sm90's fused `attn_prep`, on 580/580 differing K-plane
elements. `kernels.py:110`'s `Y = T.empty((M, N), "bfloat16")` is the extra
rounding; the chain is f32 -> bf16 -> f32 -> f32 -> bf16 against `attn_prep`'s
f32 -> f32 -> bf16.

**Three arms on one input, before touching serving code**, because the fix's
shape depends on which claim survives:

1. `now`   -- `backend.rmsnorm` then `backend.rope`, cast to bf16 at the pool.
   The shipped discrete path.
2. `once`  -- the same, with the norm's bf16 output replaced by the f32 kernel
   pair (`rmsnorm_partial` + `rmsnorm_apply`, which the CPU cell already uses and
   sm90 overrides). One rounding, at the pool.
3. `oracle` -- `reference.attn_prelude`, f64.

The prediction, written before the run: `once` lands at half of `now`'s mean
error, because one bf16 rounding of an already-rounded value costs one more ulp
than one rounding of an exact one. If `once` does not improve, the extra rounding
is not where the 2x comes from and the fix is wrong.

Also times all three at the prefill shape, because tilerl-45's open question is
whether an f32 q/k norm output is free: it doubles that tensor's store and the
rope load, and prefill norms the whole prompt. This is a kernel-level number, not
an engine one -- it bounds the per-call cost and says nothing about tok/s.

    CUDA_VISIBLE_DEVICES=6 TILERL_TARGET=cuda TILELANG_CACHE_DIR=/work/tilelang_cache \\
    PYTHONPATH=src:packages/tilerl-kernels/src \\
    python3 scripts/probe_norm_cast_once.py
"""
from __future__ import annotations

import argparse
import time

import torch
from tilerl_kernels import reference
from tilerl_kernels.backend import _THREADS, get_backend
from tilerl_kernels.registry import _resolve

from tilerl.config import qwen38_27b


def norm_f32(be, x, w, eps):
    """rmsnorm through the f32 kernel pair, bypassing sm90's bf16 `rmsnorm_fused`.

    Same two kernels the CPU cell runs, called directly: sm90 registers
    `rmsnorm_apply` as the bf16 variant, so this reaches the f32 body only by
    building it for this target rather than by name lookup."""
    from tilerl_kernels import kernels

    x = be._f32(x)
    w = be._const_f32(w)
    lead = x.shape[:-1]
    x2 = be._c(x.reshape(-1, x.shape[-1]))
    N = x2.shape[-1]
    block_N = min(256, N)
    chunks = (N + block_N - 1) // block_N
    p = be._kernel("rmsnorm_partial")(x2, block_N, chunks, _THREADS)
    apply_f32 = kernels.make_rmsnorm_apply(be.target)
    y = apply_f32(x2, w, p, float(eps), block_N, chunks, _THREADS)
    return y.reshape(*lead, w.shape[0])


def err(y, ref, mask=None) -> tuple[float, float]:
    """Mean and max |y - ref|, over `mask` when given.

    The mask is the whole point. The 2.0007 figure was a mean over the elements
    where the two arms DIFFER; 95.2% of elements here are identical in both arms
    and contribute the same error to each, so a mean over everything dilutes the
    ratio toward 1.0 no matter how large the real effect is. Averaging over the
    wrong population is the same error as ranking by max on a shared bf16 grid."""
    d = (y.float() - ref).abs()
    if mask is not None:
        d = d[mask]
    return d.mean().item(), d.max().item()


def timeit(fn, reps: int) -> float:
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / reps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=int, default=64, help="prefill bucket")
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()

    be = get_backend()
    assert be.arch == "sm90", f"the bf16 norm output is sm90-only: {be.arch}"
    assert "rmsnorm_fused" in _resolve(be.precision, be.arch)
    cfg = qwen38_27b()
    d, rd, hkv = cfg.head_dim, cfg.effective_rotary_dim, cfg.num_kv_heads
    torch.manual_seed(0)
    x = torch.randn(1, a.tokens, hkv, d, dtype=torch.float32, device=be.device)
    w = torch.randn(d, dtype=torch.float32, device=be.device)
    pos = torch.arange(a.tokens, dtype=torch.int32, device=be.device).unsqueeze(0)

    ref = reference.attn_prelude(x, w, pos, cfg.rope_theta, cfg.rms_eps, rotary_dim=rd)

    def arm(norm):
        y = be.rope(norm(be, x, w, cfg.rms_eps), pos, cfg.rope_theta, rotary_dim=rd)
        return y.to(torch.bfloat16).float()   # the pool's cast

    now = arm(lambda b, *aa: b.rmsnorm(*aa))
    once = arm(norm_f32)

    m_now, x_now = err(now, ref)
    m_once, x_once = err(once, ref)
    moved = now != once
    differ = int(moved.sum().item())
    d_now, dx_now = err(now, ref, moved)
    d_once, dx_once = err(once, ref, moved)
    print(f"tokens {a.tokens}, hkv {hkv}, head_dim {d}, rotary {rd}")
    print(f"  over ALL {now.numel()} elements:")
    print(f"    now  (bf16 norm out): mean {m_now:.3e}  max {x_now:.3e}")
    print(f"    once (f32 norm out):  mean {m_once:.3e}  max {x_once:.3e}")
    print(f"  elements the fix moves at all: {differ}/{now.numel()} "
          f"({100 * differ / now.numel():.1f}%)")
    print(f"  over THOSE {differ} elements -- the comparable population:")
    print(f"    now  mean {d_now:.3e}  max {dx_now:.3e}")
    print(f"    once mean {d_once:.3e}  max {dx_once:.3e}")
    ratio = d_now / d_once if d_once else float("inf")
    dilute = m_now / m_once if m_once else float("inf")
    print(f"  mean-error ratio on the differing elements = {ratio:.4f}")
    print(f"  the same ratio over all elements = {dilute:.4f} <- diluted by the "
          f"{100 * (1 - differ / now.numel()):.1f}% that never move")
    if ratio < 1.5:
        print("  PREDICTION FAILED: the bf16 norm output is not the extra rounding, "
              "so casting once is the wrong fix")
    else:
        print("  the bf16 norm output IS the extra rounding: casting once removes it")

    t_now = timeit(lambda: be.rmsnorm(x, w, cfg.rms_eps), a.reps)
    t_once = timeit(lambda: norm_f32(be, x, w, cfg.rms_eps), a.reps)
    print(f"\nnorm call only, {a.tokens} tokens x {hkv} heads, {a.reps} reps:")
    print(f"  bf16 out {t_now:.4f} ms   f32 out {t_once:.4f} ms   "
          f"x{t_once / t_now:.2f}")
    print("  kernel-level only: the fused bf16 kernel is one launch and the f32 pair "
          "is two, so this conflates the dtype with the launch count.\n"
          "  It does NOT answer the prefill tok/s question, which needs an engine.")


if __name__ == "__main__":
    main()
