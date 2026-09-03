"""Does the combine's all-empty-row guard compile on sm90, and does it fire?

PR #30 wraps two expressions in ``paged_attention_combine`` so an all-empty row
yields 0 instead of NaN. That row is unreachable through ``_paged_attention_decode``
(split 0 always spans tile 0), so a live-path run cannot tell the guard from its
absence. Three checks, and only the second one is new:

1. **Compiles and is inert.** Guarded combine vs the dense ``paged_attention``
   kernel at every (W, KVSPLIT) the verify tick reaches. This is the blast-radius
   claim: a row with any non-empty split must take the arithmetic it took before.
2. **The guard fires.** Hand-built partials with PM=-inf and PL=0 on EVERY split
   -- the state the guard exists for -- fed straight to the combine kernel. Guarded
   must give 0. Then the same partials through an unguarded rebuild of the same
   kernel, which must give NaN: without that arm, "no NaN" is unfalsifiable.
3. **The invariant the guard rests on.** For n >= 1, how many of the KVSPLIT
   splits run zero tiles, and whether it is ever all of them.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/verify_combine_guard.py
"""
from __future__ import annotations

import math

import tilelang
import tilelang.language as T
import torch
from tilerl_kernels.backend import _snap_mma_tile, get_backend
from tilerl_kernels.kernels_mma import _pass_configs


def unguarded_combine(target: str, KVSPLIT: int = 16):
    """The combine as it stood before PR #30 -- the negative control. Kept here
    and not imported, because the point is to run the arithmetic the guard replaced."""

    @tilelang.jit(target=target, pass_configs=_pass_configs())
    def combine(PO, PM, PL, G, W):
        B, Hkv, D = T.const("B, Hkv, D")
        Mt = T.const("Mt")
        PO: T.Tensor((B, Hkv, KVSPLIT, Mt, D), "float32")
        PM: T.Tensor((B, Hkv, KVSPLIT, Mt), "float32")
        PL: T.Tensor((B, Hkv, KVSPLIT, Mt), "float32")
        Out = T.empty((B, W, Hkv * G, D), "bfloat16")
        with T.Kernel(B * Hkv * G * W, threads=32) as row:
            lane = T.get_thread_binding(0)
            bb = row // (Hkv * G * W)
            hkv = (row // (G * W)) % Hkv
            m0 = row % (G * W)
            w = T.alloc_local((KVSPLIT,), "float32")
            m = T.alloc_local((1,), "float32")
            l = T.alloc_local((1,), "float32")
            acc = T.alloc_local((1,), "float32")
            m[0] = -T.infinity("float32")
            for sp in T.unroll(KVSPLIT):
                m[0] = T.max(m[0], PM[bb, hkv, sp, m0])
            l[0] = 0.0
            for sp in T.unroll(KVSPLIT):
                w[sp] = T.exp2(PM[bb, hkv, sp, m0] - m[0])
                l[0] += w[sp] * PL[bb, hkv, sp, m0]
            for i in T.unroll(T.ceildiv(D, 32)):
                if i * 32 + lane < D:
                    acc[0] = 0.0
                    for sp in T.unroll(KVSPLIT):
                        acc[0] += w[sp] * PO[bb, hkv, sp, m0, i * 32 + lane]
                    Out[bb, m0 % W, hkv * G + m0 // W, i * 32 + lane] = T.cast(
                        acc[0] / l[0], "bfloat16"
                    )
        return Out

    return combine


def relerr(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


def check_live_paths(backend):
    """A verify tick's real geometries, guarded combine vs the dense kernel."""
    B, H, HKV, D, BS = 2, 24, 4, 256, 16
    g = H // HKV
    print("=== 1. compiles and is inert on every live geometry ===")
    print(f"{'W':>3} {'n':>7} {'KVSPLIT':>8} {'block_m':>8} {'relerr':>10}  finite")
    bad = 0
    for n in (64, 100, 4096):
        nb = -(-n // BS) * B
        k = torch.randn(nb, HKV, BS, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(nb, HKV, BS, D, device="cuda", dtype=torch.bfloat16)
        bt = torch.arange(nb, device="cuda", dtype=torch.int32).reshape(B, -1)
        for w in (1, 2, 4, 8):
            if g * w > 128:
                continue
            lens = torch.full((B,), n, device="cuda", dtype=torch.int32)
            qlens = torch.full((B,), w, device="cuda", dtype=torch.int32)
            q = torch.randn(B, w, H, D, device="cuda", dtype=torch.bfloat16)
            scale = D**-0.5
            # the dense kernel pads S to block_M and masks by SeqQLens
            bm = _snap_mma_tile(w, 128) if w > 1 else 16
            qp = torch.zeros(B, bm, H, D, device="cuda", dtype=torch.bfloat16)
            qp[:, :w] = q
            ref = backend._kernel("paged_attention")(
                qp, k, v, bt, lens, qlens, scale, BS, bm, 128)[:, :w].float()
            out = backend._paged_attention_decode(q, k, v, bt, lens, qlens, scale).float()
            e = relerr(out, ref)
            fin = bool(torch.isfinite(out).all())
            ok = e < 2e-2 and fin
            bad += not ok
            mt = block_table_ks(backend, bt, k, HKV, B)
            print(f"{w:>3} {n:>7} {mt:>8} {_snap_mma_tile(g * w, 128):>8} "
                  f"{e:>10.3e}  {fin}{'' if ok else '   <-- FAIL'}")
    return bad


def block_table_ks(backend, bt, k, hkv, b):
    """Mirror of the split-count choice in _paged_attention_decode."""
    max_tokens = bt.shape[1] * k.shape[2]
    wide = 16 * hkv * b < 2 * backend._sms and max_tokens >= 64 * k.shape[2]
    return 64 if (max_tokens > 65536 or wide) else 16


def check_guard_fires(backend):
    """The all-empty row, built by hand. Guarded -> 0; unguarded -> NaN."""
    print("\n=== 2. the guard fires (negative control included) ===")
    B, HKV, G, W, D, KS = 1, 1, 2, 2, 64, 16
    mt = _snap_mma_tile(G * W, 128)
    shape = (B, HKV, KS, mt, D)
    po = torch.zeros(shape, dtype=torch.float32, device="cuda")
    pm = torch.full((B, HKV, KS, mt), -math.inf, dtype=torch.float32, device="cuda")
    pl = torch.zeros((B, HKV, KS, mt), dtype=torch.float32, device="cuda")

    guarded = backend._kernel("paged_attention_combine")(po, pm, pl, G, W)
    g_nan = int(torch.isnan(guarded.float()).sum())
    print(f"  guarded  : {g_nan} NaN of {guarded.numel()} elements, "
          f"max |out| {guarded.float().abs().max().item():.3e}")

    ctrl = unguarded_combine(backend.target, KS)(po, pm, pl, G, W)
    c_nan = int(torch.isnan(ctrl.float()).sum())
    print(f"  unguarded: {c_nan} NaN of {ctrl.numel()} elements   <-- the control")
    if g_nan == 0 and c_nan == ctrl.numel():
        print("  PASS: the guard replaces NaN with 0, and its absence produces NaN")
        return 0
    print("  FAIL: " + ("the control produced no NaN, so this proves nothing"
                        if c_nan == 0 else "the guard did not clear every NaN"))
    return 1


def check_invariant():
    """Split 0 always runs a tile, so no row is ever all-empty. n >= 1."""
    print("\n=== 3. the invariant: splits running zero tiles, KVSPLIT=16 ===")
    print(f"{'n':>7} {'tiles':>6} {'per':>5} {'empty':>6}  all-empty")
    worst = 0
    for n in (1, 8, 63, 64, 65, 100, 512, 1024, 2048, 65536):
        tiles = -(-n // 64)
        per = -(-tiles // 16)
        empty = sum(1 for sp in range(16) if min(tiles, sp * per + per) - sp * per <= 0)
        worst = max(worst, empty)
        print(f"{n:>7} {tiles:>6} {per:>5} {empty:>6}  {empty == 16}")
    print(f"  max empty splits over any n: {worst}/16 -- all-empty never occurs")
    return 1 if worst == 16 else 0


def main() -> None:
    print("tilelang", tilelang.__version__, "torch", torch.__version__)
    backend = get_backend()
    assert backend.device.type == "cuda", backend.device
    cap = torch.cuda.get_device_capability()
    print(f"arch {backend.arch} sm{cap[0]}{cap[1]} {torch.cuda.get_device_name()}\n")
    bad = check_live_paths(backend) + check_guard_fires(backend) + check_invariant()
    print(f"\n{'ALL CHECKS PASSED' if bad == 0 else f'{bad} CHECK(S) FAILED'}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
