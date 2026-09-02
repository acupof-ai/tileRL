"""Split-KV attention: parity against a torch reference, then cost vs context.

The end-to-end decode curve was ms/tok = 31.9 + 6.20*(ctx/1K). Roofline: the
pool is bf16 head_dim 256 (config.py:147, kv_cache.py:100), so 16 layers x 4 KV
heads x 256 x 2 B x 2 = 64 KiB/token -> 1K ctx = 67 MB = 0.07 ms at 900 GB/s.
The kernel is declared f32 and the backend casts the pool on every call
(backend.py:711-713), so what the kernel actually streams is twice that,
128 KiB/token = 0.15 ms per 1K. Either way the slope was ~40-80x off.

Profiling split and combine separately found the cause: cost RISING with thread
count (4K ctx, split only: 32t 780us, 64t 950, 128t 2165, 256t 4066). Cost
rising with threads is redundancy — every thread reran the same T.serial(D)
dot. Note 32t and 64t differ by only 1.22x, not 2x: a warp is SIMD, so the
waste is bounded by warps-per-block, NOT by threads-per-block. This measures
the tiled rewrite against that baseline and checks it still computes attention.

Two costs in the end-to-end number are deliberately NOT measured here: the
bf16->f32 pool cast (O(num_blocks), not O(ctx) — it sits in the 31.9 intercept)
and the 48 GDN layers (gdn_decode_fused takes no SeqLens and has no history
loop, so it is O(1) in context and also intercept).

  scripts/v100.sh run pa '/usr/bin/python3 -u scripts/prof_attn_ctx.py [head_dim]'
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

os.environ.setdefault("TILERL_TARGET", "cuda")
from tilerl_kernels.backend import _THREADS, get_backend  # noqa: E402

# Qwen3.8-27B full-attn shape (config.py:145-147): 24 query heads, 4 KV heads,
# head dim 256. D is the axis under investigation — the dot is O(D) — so
# measuring at 128 understates the slope by ~2x. Overridable to reproduce the
# old D=128 runs; the default must match the model.
H, HKV, BLK = 24, 4, 16
D = int(sys.argv[1]) if len(sys.argv) > 1 else 256
GQ = H // HKV
CTXS = [512, 1024, 2048, 4096, 8192]
LAYERS = 16  # full-attn layers; the slope is per-token over all of them


def ms(fn, iters=20) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


def inputs(ctx: int, S: int, dev):
    """Same argument order, dtypes and contiguity as backend.py:711-719."""
    nb = (ctx + BLK - 1) // BLK
    g = torch.Generator(device="cpu").manual_seed(ctx)
    mk = lambda *s: torch.randn(*s, generator=g).to(dev)  # noqa: E731
    return (
        mk(1, S, H, D), mk(nb, HKV, BLK, D), mk(nb, HKV, BLK, D),
        torch.arange(nb, dtype=torch.int32, device=dev).reshape(1, nb),
        torch.tensor([ctx], dtype=torch.int32, device=dev),
        torch.tensor([S], dtype=torch.int32, device=dev),
    )


def reference(q, k, v, ctx: int, scale: float) -> torch.Tensor:
    """[1,S,H,D] attention over a flattened cache, S queries ending at ctx."""
    S = q.shape[1]
    kf = k.permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :ctx]  # [Hkv, ctx, D]
    vf = v.permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :ctx]
    out = torch.empty_like(q)
    for h in range(H):
        kk, vv = kf[h // GQ], vf[h // GQ]
        for t in range(S):
            n = ctx - S + t + 1
            s = (q[0, t, h] @ kk[:n].T) * scale
            out[0, t, h] = torch.softmax(s.float(), -1) @ vv[:n]
    return out


def fit(xs, ys) -> tuple[float, float]:
    """Least-squares slope/intercept — the slope is the whole question here."""
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxx = sum(x * x for x in xs)
    m = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return m, (sy - m * sx) / n


def main() -> None:
    be = get_backend()
    dev, scale = be.device, D**-0.5
    ksplit = be._kernel("paged_attention_split")
    kcomb = be._kernel("paged_attention_split_combine")
    run = lambda a: kcomb(*ksplit(*a, scale, BLK, _THREADS), _THREADS)  # noqa: E731
    print(f"# H={H} Hkv={HKV} D={D} threads={_THREADS}")

    print("\n# parity vs torch reference (max abs rel err)")
    for ctx in (512, 2048):
        for S in (1, 4):
            a = inputs(ctx, S, dev)
            got = run(a)
            want = reference(a[0], a[1], a[2], ctx, scale)
            err = ((got - want).abs() / want.abs().clamp(min=1e-3)).max().item()
            print(f"  ctx={ctx:>5} S={S}  relerr {err:.2e}  {'OK' if err < 1e-2 else 'FAIL'}")

    # Roofline uses 4 KV heads, not 24: GQA means 6 query heads share one KV
    # slice, so the UNIQUE bytes are Hkv-sized. The kernel grids over H and
    # re-reads each slice GQ=6 times, but those repeats hit L2, not HBM.
    print(f"\n# cost vs context, S=1 (x{LAYERS} layers). f32 KV, unique = Hkv*D*4*2/token")
    print(f"{'ctx':>6} {'split us':>10} {'comb us':>10} {'tot us':>10} "
          f"{'ms/tok':>8} {'MB uniq':>8} {'roof ms':>8}")
    per_tok = []
    for ctx in CTXS:
        a = inputs(ctx, 1, dev)
        t_s = ms(lambda: ksplit(*a, scale, BLK, _THREADS))
        po, pm, pl = ksplit(*a, scale, BLK, _THREADS)
        t_c = ms(lambda: kcomb(po, pm, pl, _THREADS))
        tot = (t_s + t_c) * 1000
        mb = ctx * HKV * D * 4 * 2 * LAYERS / 1e6
        per_tok.append(tot * LAYERS / 1000)
        print(f"{ctx:>6} {t_s*1000:>10.1f} {t_c*1000:>10.1f} {tot:>10.1f} "
              f"{per_tok[-1]:>8.2f} {mb:>8.1f} {mb/900:>8.3f}")

    m, b = fit([c / 1024 for c in CTXS], per_tok)
    print(f"\n  attention: ms/tok = {b:.2f} + {m:.2f}*(ctx/1K)")
    print("  engine   : ms/tok = 31.9 + 6.20*(ctx/1K)  <- slope to kill")
    print(f"  combine is flat in ctx, so it lands in the intercept, "
          f"not the slope ({per_tok[-1]:.2f} ms/tok at 8K includes it)")

    print("\n# verify width S (4096 ctx, us) — speculation pays S rows here")
    for S in (1, 2, 4, 8):
        a = inputs(4096, S, dev)
        print(f"  S={S}: {ms(lambda: run(a))*1000:>8.1f}")

    # Flat = the redundancy is gone. Rising = threads reran the same chain.
    # Expect a step at >64t from occupancy even when redundancy is fixed.
    print("\n# threads (4096 ctx, split only, us) — rising = redundant threads")
    a = inputs(4096, 1, dev)
    for th in (32, 64, 128, 256):
        print(f"  {th:>3}t: {ms(lambda: ksplit(*a, scale, BLK, th))*1000:>8.1f}")


if __name__ == "__main__":
    main()
