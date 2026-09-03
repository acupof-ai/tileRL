"""`attn_prep` vs the discrete norm+rope+write it replaces: q, k, gate, KV planes.

`fuse_projections=True` changes 53 of 1000 MMLU answers, |delta logit| to 4.46.
Two probes narrowed it to here:

- `probe_fusion_weights.py`: `_fuse_projections` is weight-preserving bit-for-bit,
  fp4 and fp8, so the concat is exact.
- `probe_fusion_kernels.py`: the fused and unfused GEMMs are **bitwise identical**
  at M=1/8/64 on all five groups (negative control: perturbing one scale element
  moves the output 6.007e-02, so the comparison detects differences).

So the flips are not arithmetic in the linear layer. `fuse_projections` gates a
different BRANCH: `model.py:202` `if self._has(qkv_key)` is true only when fused,
and inside it `backend.attn_prep` does q_norm + k_norm + rope + KV-write in one
launch, then returns before the discrete epilogue exists. Unfused runs
`rmsnorm` x2, `rope` x2, `write_tokens`. `attn_prep` has **no caller anywhere
else and no test anywhere** -- `git grep attn_prep -- tests/` is empty, which is
how a divergence this size stayed in the serving path.

**Read from the code before running, so the probe knows what to look at:**

1. Rope pairing agrees. Both use rotate_half `d <-> d + half`
   (`kernels.py:432`, `kernels_mma.py:88`), not GPT-J's `(2d, 2d+1)`.
2. **Partial rotary pairing AGREES, checked by arithmetic before running.**
   Qwen3.8 is partial-rotary: head_dim 256, `effective_rotary_dim` 64.
   `Backend.rope` slices `x[..., :64]`, and its kernel derives `half = D//2`
   from that slice, so it pairs `d <-> d+32` over dims 0..63 and concatenates
   dims 64..255 untouched. `attn_prep` receives `InvFreq` of length 32, so
   `RD2 = 32`, and its rope loop runs `d in Parallel(32)` touching `d` and
   `d+32` -- also dims 0..63 only. Same pairs, same untouched tail. I expected
   a mismatch here and the arithmetic says there is none, so this is NOT the
   mechanism and the probe must find it elsewhere.
3. The q dtype differs: `attn_prep` returns bf16, the discrete path hands
   `paged_attention` f32 which casts at `backend.py:501`. Same end state on the
   sm90 path, and a bf16 round trip at the output is worth ~0.4% -- enough to
   show up in a per-element comparison without being the cause of a 4.46 logit
   move.
4. `attn_prep` normalizes q over `D` **including the gate rows' stride**: it
   reads `QKV[b, t, q0 + k]` for `k in serial(D)` where `q0 = h * 2 * D`, so
   its sum covers the first `D` of the `2*D` interleaved `[query; gate]` block.
   The discrete path reshapes to `[b,t,hq,2,d]` and normalizes `q[..., 0, :]`.
   Those are the same elements. Also matching by arithmetic.

Everything readable agrees, which is why this needs a run: the divergence is
either in a place the code does not reveal, or the two arms do agree here and the
53 flips originate somewhere the previous two probes have not reached.

Gate is included because it is sliced at different points in the two arms
(after the fact from `qkv` vs before the norm from `q`), and the KV planes
because `attn_prep` writes them itself.

    CUDA_VISIBLE_DEVICES=6 TILERL_TARGET=cuda TILELANG_CACHE_DIR=/work/tilelang_cache \
    PYTHONPATH=src:packages/tilerl-kernels/src \
    python3 scripts/probe_attn_prep.py --source /work/Qwen3.8-27B-NVFP4
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tilerl.config import qwen38_27b
from tilerl.kv_cache import PagedKvPool
from tilerl.model import load_hf
from tilerl_kernels.backend import get_backend


class Kv:
    """The minimum an `attn_prep`/`write_tokens` call reads off a batch state."""

    dense = False

    def __init__(self, pool, b, s, device):
        self.kv_pool = pool
        nb = pool.k_pool.shape[-2]
        self.block_table = torch.arange(b * 8, dtype=torch.int32,
                                        device=device).reshape(b, 8)
        assert s <= nb, f"s={s} must fit one block ({nb}) so wpos stays in block 0"
        self.seq_len = torch.full((b,), s, dtype=torch.int32, device=device)
        self.seq_q_lens = torch.full((b,), s, dtype=torch.int32, device=device)


def discrete(be, qkv, wq, wk, positions, cfg, kv, layer):
    """The unfused arm: slice, reshape, rmsnorm, rope, write_tokens."""
    hq, hkv, d = cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim
    b, t = qkv.shape[0], qkv.shape[1]
    q_rows = hq * d * (2 if cfg.full_attn_gated else 1)
    q = qkv[..., :q_rows]
    k = qkv[..., q_rows:q_rows + hkv * d]
    v = qkv[..., q_rows + hkv * d:]
    q = q.reshape(b, t, hq, 2, d) if cfg.full_attn_gated else q.reshape(b, t, hq, d)
    gate = q[..., 1, :] if cfg.full_attn_gated else None
    q = q[..., 0, :] if cfg.full_attn_gated else q
    k = k.reshape(b, t, hkv, d)
    v = v.reshape(b, t, hkv, d)
    q = be.rmsnorm(q, wq, cfg.rms_eps)
    k = be.rmsnorm(k, wk, cfg.rms_eps)
    q = be.rope(q, positions, cfg.rope_theta, rotary_dim=cfg.effective_rotary_dim)
    k = be.rope(k, positions, cfg.rope_theta, rotary_dim=cfg.effective_rotary_dim)
    be.write_tokens(k, v, kv, layer)
    return q, gate


def f32_reference(qkv, wq, wk, positions, cfg):
    """Dense f32 norm+rope in torch, no kernel: the value both arms approximate.

    Ranking the arms needs this because `attn_prep` and the discrete path are
    both approximations -- their disagreeing says nothing about which is right.
    Computed in f64 and returned f32 so the reference's own rounding is two
    orders below the difference being ranked."""
    hq, hkv, d = cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim
    b, t = qkv.shape[0], qkv.shape[1]
    rd = cfg.effective_rotary_dim
    x = qkv.double()
    q_rows = hq * d * (2 if cfg.full_attn_gated else 1)
    q = x[..., :q_rows].reshape(b, t, hq, 2, d)[..., 0, :]
    k = x[..., q_rows:q_rows + hkv * d].reshape(b, t, hkv, d)

    def norm(z, w):
        # the checkpoint's norm weights load on CPU; the same class as #42's failures
        return (z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + cfg.rms_eps)
                * w.to(z.device).double())

    def rope(z):
        half = rd // 2
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, rd, 2, dtype=torch.float64,
                                                     device=z.device) / rd))
        ang = positions.double().reshape(b, t, 1, 1) * inv.reshape(1, 1, 1, half)
        c, s = ang.cos(), ang.sin()
        z = z.clone()
        x0, x1 = z[..., :half].clone(), z[..., half:rd].clone()
        z[..., :half] = x0 * c - x1 * s
        z[..., half:rd] = x1 * c + x0 * s
        return z

    return rope(norm(q, wq)).float(), rope(norm(k, wk)).float()


def rank(name, ref, disc, prep) -> dict:
    """Which arm is closer to the f32 reference on this tensor.

    Ranked by MEAN error and by per-element wins, not by max. Max|d| came out
    identical to four figures for both arms (1.559e-02 / 1.460e-02) because both
    round to the same bf16 grid, so the largest error is the quantum at the
    largest-magnitude element and is the same element in both arms -- max
    measures the grid, not the arithmetic. Only elements where the two arms
    actually differ are counted: the 99.8% they agree on contribute a tie to
    both sides and would swamp the comparison."""
    ref = ref.float()
    dd, dp = (disc.float() - ref).abs(), (prep.float() - ref).abs()
    differ = dd != dp
    n = int(differ.sum().item())
    d_wins = int((dd < dp).sum().item())
    p_wins = int((dp < dd).sum().item())
    md, mp = dd[differ].mean().item() if n else 0.0, dp[differ].mean().item() if n else 0.0
    win = ("no difference" if n == 0 else
           "discrete" if d_wins > p_wins else "attn_prep" if p_wins > d_wins else "split")
    print(f"  {name:<10} on the {n} elements that differ: "
          f"mean |discrete-ref| {md:.3e}  |attn_prep-ref| {mp:.3e}\n"
          f"  {'':<10} closer more often: discrete {d_wins}, attn_prep {p_wins}  -> {win}")
    return {"what": name, "n_differ": n, "discrete_mean": md, "attn_prep_mean": mp,
            "discrete_wins": d_wins, "attn_prep_wins": p_wins, "closer": win}


def report(name, a, b) -> dict:
    a, b = a.float(), b.float()
    d = (a - b).abs()
    mx = d.max().item()
    rel = (d.mean() / a.abs().mean().clamp_min(1e-30)).item()
    nz = int((d > 0).sum().item())
    print(f"  {name:<10} max|d| {mx:>10.3e}  rel {rel:>9.2e}  "
          f"differing elements {nz}/{d.numel()}")
    return {"what": name, "max": mx, "rel": rel, "differing": nz, "total": d.numel()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--layer", type=int, default=3, help="a full-attn layer")
    ap.add_argument("--b", type=int, default=2)
    ap.add_argument("--s", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    be = get_backend()
    assert be.device.type == "cuda", f"attn_prep is sm90-only: {be.device}"
    cfg = qwen38_27b()
    assert cfg.is_full_attn(a.layer), f"layer {a.layer} is not full-attn"
    hq, hkv, d = cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim
    rd = cfg.effective_rotary_dim

    print(f"head_dim {d}, effective_rotary_dim {rd}, full_attn_gated "
          f"{cfg.full_attn_gated}, hq {hq}, hkv {hkv}")
    # both arms pair d <-> d + rd/2 over dims 0..rd-1: discrete's kernel derives
    # half from the [:rd] slice, attn_prep's RD2 is len(InvFreq) = rd/2
    print(f"  both arms rotate d <-> d+{rd // 2} over dims 0..{rd - 1}, "
          f"tail {rd}..{d - 1} untouched -- pairing agrees by arithmetic")

    model = load_hf(cfg, a.source)
    p = model.params
    lp = f"layers.{a.layer}"
    wq, wk = p[f"{lp}.q_norm"], p[f"{lp}.k_norm"]

    b, s = a.b, a.s
    nqkv = hq * d * (2 if cfg.full_attn_gated else 1) + 2 * hkv * d
    torch.manual_seed(0)
    # f32: attn_prep annotates QKV as float32 ("the fp4 GEMV writes f32")
    qkv = torch.randn(b, s, nqkv, dtype=torch.float32, device=be.device)
    positions = torch.arange(s, dtype=torch.int32, device=be.device).unsqueeze(0).expand(b, -1)

    print("\n=== attn_prep vs the discrete norm+rope+write ===")
    rows = []
    pool_a = PagedKvPool(b * 8, hkv, d, num_layers=1, device=be.device)
    pool_b = PagedKvPool(b * 8, hkv, d, num_layers=1, device=be.device)
    kv_a, kv_b = Kv(pool_a, b, s, be.device), Kv(pool_b, b, s, be.device)

    qf = be.attn_prep(qkv, wq, wk, positions, cfg.rope_theta, rd, kv_a, 0, hq, hkv,
                      cfg.rms_eps)
    assert qf is not None, "no attn_prep kernel in this cell; nothing to compare"
    gate_f = qkv[..., :hq * 2 * d].reshape(b, s, hq, 2, d)[..., 1, :]

    qd, gate_d = discrete(be, qkv, wq, wk, positions, cfg, kv_b, 0)

    rows.append(report("q", qd, qf))
    rows.append(report("gate", gate_d, gate_f))
    ka, va = pool_a.kv_layer(0)
    kb, vb = pool_b.kv_layer(0)
    rows.append(report("k plane", kb, ka))
    rows.append(report("v plane", vb, va))

    # what a divergence is worth downstream: the same attention on the two preludes
    out_f = be.paged_attention(qf, ka, va, kv_a.block_table, kv_a.seq_len,
                               1.0 / math.sqrt(d), gate=gate_f,
                               seq_q_lens=kv_a.seq_q_lens)
    out_d = be.paged_attention(qd, kb, vb, kv_b.block_table, kv_b.seq_len,
                               1.0 / math.sqrt(d), gate=gate_d,
                               seq_q_lens=kv_b.seq_q_lens)
    print("\n=== what it is worth after attention ===")
    rows.append(report("attn out", out_d, out_f))

    worst = max(rows, key=lambda r: r["max"])
    print(f"\nlargest divergence: {worst['what']} at {worst['max']:.3e}")

    # which arm is right. The k plane is the clean tensor: both arms write bf16 into
    # a bf16 pool, so no dtype gap, unlike q (attn_prep returns bf16, discrete f32).
    print("\n=== which arm is closer to a dense f32 norm+rope ===")
    q_ref, k_ref = f32_reference(qkv, wq, wk, positions, cfg)
    ranks = [rank("q", q_ref, qd, qf)]
    # read the two pools back at the positions write_tokens/attn_prep filled
    kd_w = torch.stack([kb[kv_b.block_table[i, 0], :, :s] for i in range(b)])
    kp_w = torch.stack([ka[kv_a.block_table[i, 0], :, :s] for i in range(b)])
    ranks.append(rank("k plane", k_ref.permute(0, 2, 1, 3), kd_w, kp_w))
    kwin = ranks[-1]["closer"]
    print(f"\non the k plane -- the tensor with no dtype gap -- {kwin}.")
    if kwin in ("split", "no difference"):
        print("  neither arm is systematically closer to exact: the two straddle the\n"
              "  reference, so this is two valid roundings and not a correctness bug.\n"
              "  Which one serving picks is then a choice, not a fix.")
    else:
        print(f"  {'attn_prep, the fused/serving default,' if kwin == 'attn_prep' else 'the discrete path, the cli.py default,'} "
              f"is closer to exact more often, so the other arm is the one moving\n"
              f"  answers away from exact.")
    if all(r["max"] == 0.0 for r in rows):
        print("every tensor matches bit-for-bit: attn_prep is NOT the mechanism, "
              "and the 53 flips come from somewhere still unexamined.")
    else:
        print("attn_prep and the discrete path disagree. Which is correct is not "
              "settled by this probe -- both are approximations, and the next step "
              "is a dense f32 reference for norm+rope on the same input.")
    if a.out:
        o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
        (o / "attn_prep.json").write_text(json.dumps(
            {"head_dim": d, "rotary_dim": rd, "rows": rows, "ranks": ranks}, indent=1))
        print(f"wrote {o / 'attn_prep.json'}")


if __name__ == "__main__":
    main()
