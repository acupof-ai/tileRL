"""Torch-eager reference for every op: the parity oracle for the TileLang
kernels and the day-1 backward. f32 compute, no autograd; backward formulas
are hand-derived (agent-infer crates/autograd/src/ops/linear_attention.rs).
# ponytail: torch-eager backward, tilelang kernel when perf demands
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch


def _f32(x: torch.Tensor) -> torch.Tensor:
    return x.float() if x.dtype != torch.float32 else x


# ---------------------------------------------------------------- rmsnorm


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """y = x * rsqrt(mean(x^2, -1) + eps) * w.  x [..., N], w [N]."""
    x = _f32(x)
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * w


def rmsnorm_bwd(
    grad: torch.Tensor, x: torch.Tensor, w: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward of :func:`rmsnorm`. Returns (gx, gw)."""
    x = _f32(x)
    grad = _f32(grad)
    rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    c = (grad * w * x).mean(-1, keepdim=True)
    gx = rstd * (grad * w - (rstd * rstd) * x * c)
    gw = (grad * x * rstd).sum(dim=tuple(range(x.ndim - 1)))
    return gx, gw


# ---------------------------------------------------------------- rope


def _inv_freq(d: int, theta: float, device, dtype) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, d, 2, device=device, dtype=dtype) / d))


def _rope_apply(
    x: torch.Tensor,
    positions: torch.Tensor,
    theta: float,
    negate: bool = False,
    rotary_dim: int | None = None,
) -> torch.Tensor:
    x = _f32(x)
    d = x.shape[-1]
    rd = d if rotary_dim is None else min(rotary_dim, d)
    x_rot, x_pass = x[..., :rd], x[..., rd:]
    inv = _inv_freq(rd, theta, x.device, x.dtype)
    pos = positions.to(x.device).float()
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)  # [T] -> [1, T]
    ang = pos.unsqueeze(-1) * inv  # [B, T, rd/2]
    cos = torch.cos(ang).unsqueeze(-2)  # [B, T, 1, rd/2]
    sin = torch.sin(ang).unsqueeze(-2)
    if negate:
        sin = -sin
    # rotate_half (Qwen/Llama): d pairs with d + rd/2, not GPT-J's (2d, 2d+1)
    half = rd // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    out = torch.empty_like(x_rot)
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:] = x2 * cos + x1 * sin
    return torch.cat([out, x_pass], dim=-1)


def rope(
    x: torch.Tensor, positions: torch.Tensor, theta: float, rotary_dim: int | None = None
) -> torch.Tensor:
    """Rotary embedding on the first ``rotary_dim`` features (default: all).
    x [B, T, H, D], positions [B, T] or [T]."""
    return _rope_apply(x, positions, theta, negate=False, rotary_dim=rotary_dim)


def rope_bwd(
    grad: torch.Tensor, positions: torch.Tensor, theta: float, rotary_dim: int | None = None
) -> torch.Tensor:
    """Backward of :func:`rope`; the rotation is orthogonal, so gx = R(-angle) grad."""
    return _rope_apply(grad, positions, theta, negate=True, rotary_dim=rotary_dim)


def attn_prelude(x, w, positions, theta, eps: float, rotary_dim: int | None = None):
    """norm+rope in f64, returned f32: the exact value both attention preludes
    approximate. x [B,T,H,D], w [D].

    The oracle for `attn_prep` against the discrete
    `rmsnorm`/`rope`/`write_tokens` chain. Neither of those is ground truth for
    the other -- both are approximations -- so ranking them needs a third value
    computed wider than the difference being ranked. f64 puts the oracle's own
    rounding ~2e-16, eleven orders below the ~1.3e-03 it separates.

    Composes :func:`rmsnorm` and :func:`rope` rather than reimplementing them: a
    second rotate-half in the tree drifts and then gets quoted as an independent
    check. `_rope_apply` opens with `_f32`, which would narrow the f64 input, so
    the rotation is inlined at f64 here from the same `_inv_freq` and the same
    `d <-> d + rd/2` pairing -- checked against `rope` in the test, so the two
    cannot drift apart silently."""
    x64 = x.double()
    y = rmsnorm(x64, w.to(x64.device).double(), eps)
    d = y.shape[-1]
    rd = d if rotary_dim is None else min(rotary_dim, d)
    half = rd // 2
    inv = _inv_freq(rd, theta, y.device, torch.float64)
    pos = positions.to(y.device).double()
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
    ang = pos.unsqueeze(-1) * inv
    cos, sin = torch.cos(ang).unsqueeze(-2), torch.sin(ang).unsqueeze(-2)
    out = y.clone()
    x1, x2 = y[..., :half], y[..., half:rd]
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:rd] = x2 * cos + x1 * sin
    return out.float()


# ---------------------------------------------------------------- linear


def linear(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """y = x @ w.T + bias.  x [..., K], w [N, K]."""
    x = _f32(x)
    w = _f32(w)
    y = x @ w.t()
    if bias is not None:
        y = y + _f32(bias)
    return y


def linear_bwd(
    grad: torch.Tensor, x: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward of :func:`linear` (no bias). Returns (gx, gw)."""
    x = _f32(x)
    w = _f32(w)
    grad = _f32(grad)
    gx = grad @ w
    gw = grad.reshape(-1, grad.shape[-1]).t() @ x.reshape(-1, x.shape[-1])
    return gx, gw


# ---------------------------------------------------------------- linear fp4


#: dequantized bytes linear_frozen_bwd materializes at once (the whole lm_head
#: peaked 14.2 GiB; every other backward op in a 27B step is under 0.12)
_BWD_SLICE_BYTES = 1 << 29


def linear_frozen_bwd(grad, wq, scale, oscale=None, fp8=False):
    """dX through a frozen quantized weight (LoRA / OPD base), a slice at a
    time. ``oscale`` scales weight row n, so it folds into the [M, N] grad.
    # ponytail: eager chunked dequant; no tilelang fp8 dequant kernel yet (fp4 has one).
    # ponytail: eager path; the tilelang dequant kernel only exists for fp4
    """
    g = _f32(grad).reshape(-1, grad.shape[-1])
    if oscale is not None:
        g = g * _f32(oscale).reshape(1, -1)
    n, k = wq.shape[0], wq.shape[1] * (1 if fp8 else 2)
    # slice on the scale's row granularity so a chunk never splits an fp8 128-row block
    rows = -(-n // scale.shape[0])
    step = max(1, _BWD_SLICE_BYTES // (k * 4) // rows) * rows
    out = None
    for i in range(0, n, step):
        end = min(i + step, n)
        sc = scale[i // rows: -(-end // rows)]
        w = dequant_fp8(wq[i:end], sc) if fp8 else dequant_fp4(wq[i:end], sc)
        part = g[:, i:end] @ w
        out = part if out is None else out.add_(part)
    return out.reshape(*grad.shape[:-1], k)


def dequant_fp4(wq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize e2m1-packed weights: wq uint8 [N, K//2] (low nibble first),
    scale f32 [N, K//B], B derived from the shapes. Returns [N, K] f32."""
    if getattr(wq, "_tl_twiddled", False):  # sm90 served bytes
        wq = untwiddle_fp4(wq)
    assert wq.dtype == torch.uint8
    n, k2 = wq.shape
    nib = torch.stack([wq & 0x0F, (wq >> 4) & 0x0F], dim=-1).reshape(n, k2 * 2).long()
    mag = _E2M1_LUT.to(wq.device)[nib & 0x7]
    block = k2 * 2 // scale.shape[1]
    return mag * (1.0 - 2.0 * (nib >> 3).float()) * _f32(scale).repeat_interleave(block, dim=1)


def linear_fp4(x, wq, scale, oscale=None) -> torch.Tensor:
    """y = oscale * (x @ dequant(wq, scale).T).  x [..., K], oscale f32 [N]."""
    y = _f32(x) @ dequant_fp4(wq, scale).t()
    return y if oscale is None else y * _f32(oscale)


# ---------------------------------------------------------------- fp4 packing

#: OCP/MX e2m1 magnitudes (low 3 bits of a nibble; bit 3 = sign): the one fp4
#: grid, so NVFP4 checkpoint nibbles are served without re-quantization
_E2M1_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


# sm90's twiddled byte layout (kernels_linear._FP4_TWIDDLE_SRC): per 32-bit
# word, slot p of half-word h has its (s, e1, e0, m) bits at _TW_POS[p]; slot j
# holds elem 2j+1 and slot j+4 elem 2j so decoded bf16x2 pairs match X words.
_TW_POS = ((15, 8, 7, 6), (12, 5, 4, 3), (9, 2, 1, 0), (14, 11, 10, 13))
_TW_SLOT_ELEM = (1, 3, 5, 7, 0, 2, 4, 6)


def _fp4_codes(wq: torch.Tensor) -> torch.Tensor:
    """Packed [N, K//2] (low nibble first) -> codes [N, K] int32."""
    return torch.stack([wq & 15, wq >> 4], dim=-1).reshape(wq.shape[0], -1).to(torch.int32)


def twiddle_fp4(wq: torch.Tensor) -> torch.Tensor:
    """Natural packed fp4 -> twiddled bytes (same shape, uint8)."""
    N = wq.shape[0]
    c = _fp4_codes(wq).reshape(N, -1, 8)[:, :, list(_TW_SLOT_ELEM)]
    out = []
    for h in (0, 1):
        half = torch.zeros(c.shape[:2], dtype=torch.int32, device=wq.device)
        for p, bits in enumerate(_TW_POS):
            n = c[:, :, 4 * h + p]
            for b, pos in enumerate(bits):  # bit 3-b of the nibble -> word bit pos
                half |= ((n >> (3 - b)) & 1) << pos
        out += [(half >> 8) & 255, half & 255]
    return torch.stack(out, dim=-1).reshape(N, -1).to(torch.uint8).contiguous()


def untwiddle_fp4(wq: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`twiddle_fp4`."""
    N = wq.shape[0]
    b = wq.reshape(N, -1, 4).to(torch.int32)
    slots = []
    for h in (0, 1):
        half = (b[:, :, 2 * h] << 8) | b[:, :, 2 * h + 1]
        for bits in _TW_POS:
            nib = torch.zeros_like(half)
            for k, pos in enumerate(bits):
                nib |= ((half >> pos) & 1) << (3 - k)
            slots.append(nib)
    slots = torch.stack(slots, dim=-1)  # [N, K/8, slot]
    codes = torch.empty_like(slots)
    for slot, elem in enumerate(_TW_SLOT_ELEM):
        codes[:, :, elem] = slots[:, :, slot]
    codes = codes.reshape(N, -1)
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8).contiguous()


def pack_fp4(w: torch.Tensor, block: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a weight [N,K] into e2m1 nibbles + per-block scales
    (block_max / 6, round-to-nearest). Serving wants renorm_fp4_scale on it."""
    assert w.dim() == 2, f"pack_fp4 expects a 2D weight, got {tuple(w.shape)}"
    n, k = w.shape
    assert k % block == 0, f"fp4 block size {block} must divide K, got K={k}"
    wf = w.detach().float()
    blocks = wf.reshape(n, k // block, block)
    block_max = blocks.abs().amax(dim=-1, keepdim=True)  # [n, k//block, 1]
    scale = (block_max / 6.0).clamp_min(1e-12)
    x = (blocks / scale).clamp(-6.0, 6.0)
    lut = _E2M1_LUT.to(wf.device)
    dist = (x.abs().unsqueeze(-1) - lut).abs()  # [n, k//block, block, 8]
    idx = dist.argmin(dim=-1).to(torch.uint8)  # 0..7
    sign = (x < 0).to(torch.uint8)
    nibbles = (idx | (sign << 3)).reshape(n, k)
    wq = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    return wq.contiguous(), scale.squeeze(-1).contiguous()


def renorm_fp4_scale(scale, oscale=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a per-row power of two from the block scale into the epilogue
    scale so every row's 6*scale lands in [6,12): the w4a8 kernel requantizes
    into e4m3, where raw checkpoint magnitudes saturate. 2^p re-rounds nothing."""
    scale = _f32(scale)
    p = torch.exp2(torch.floor(torch.log2(scale.amax(dim=1, keepdim=True).clamp_min(1e-30))))
    o = p.reshape(-1)
    return (scale / p).contiguous(), (o if oscale is None else _f32(oscale) * o).contiguous()


def unpack_fp4(wq: torch.Tensor, scale: torch.Tensor, oscale=None) -> torch.Tensor:
    """Inverse of :func:`pack_fp4`; returns a bf16 [N,K] weight."""
    w = dequant_fp4(wq, scale)
    return (w if oscale is None else w * _f32(oscale).reshape(-1, 1)).to(torch.bfloat16)


# ---------------------------------------------------------------- modelopt quantized checkpoints


def dequant_nvfp4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    *,
    global_divide: bool = False,
) -> torch.Tensor:
    """NVFP4 dequant: e2m1(packed) * f8(weight_scale) * gs -> bf16 [N,K].
    ModelOpt stores the global scale's reciprocal (``global_divide``,
    agent-infer quant_format.rs ScaleApply::Divide); official NVFP4's
    weight_scale_2 is a plain multiplier."""
    gs = weight_global_scale.float()
    if global_divide:
        gs = 1.0 / gs
    return (dequant_fp4(weight_packed, weight_scale) * gs).to(torch.bfloat16)


def dequant_fp8(w8: torch.Tensor, wscale: torch.Tensor, block: int = 128) -> torch.Tensor:
    """FP8 block dequant: w8 e4m3 [N,K] * wscale [ceil(N/block), ceil(K/block)]
    -> f32. The checkpoint's "scale_inv" is the scale itself (multiplied,
    agent-infer quant_format.rs ScaleApply::Multiply)."""
    n, k = w8.shape
    s = wscale.float().repeat_interleave(block, dim=-1)[:, :k]
    s = s.repeat_interleave(block, dim=-2)[:n, :]
    return w8.float() * s


def quant_fp8(w: torch.Tensor, block: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`dequant_fp8`: one absmax/448 scale per block x block."""
    n, k = w.shape
    pn, pk = -n % block, -k % block
    wp = torch.nn.functional.pad(_f32(w), (0, pk, 0, pn))
    blocks = wp.reshape(wp.shape[0] // block, block, wp.shape[1] // block, block)
    scale = blocks.abs().amax((1, 3)).clamp_min(1e-12) / 448.0
    q = (blocks / scale[:, None, :, None]).reshape(wp.shape)
    return q[:n, :k].to(torch.float8_e4m3fn).contiguous(), scale.contiguous()


def linear_fp8(x, w8, wscale, oscale=None) -> torch.Tensor:
    """y = oscale * (x @ dequant_fp8(w8, wscale).T).  x [..., K], oscale [N]."""
    y = _f32(x) @ dequant_fp8(w8, wscale).t()
    return y if oscale is None else y * _f32(oscale)


def dequant_awq(
    qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, group_size: int
) -> torch.Tensor:
    """AutoAWQ GEMM dequant: qweight int32 [K,N//8] (int4 j at bits (j%8)*4 of
    column j//8), qzeros [K//group,N//8], scales [K//group,N] -> bf16 [N,K]."""
    k, n8 = qweight.shape
    shifts = torch.arange(8, dtype=torch.int64, device=qweight.device) * 4
    q = ((qweight.long().unsqueeze(-1) >> shifts) & 0xF).float().reshape(k, n8 * 8)
    z = ((qzeros.long().unsqueeze(-1) >> shifts) & 0xF).float()
    z = z.reshape(k // group_size, n8 * 8).repeat_interleave(group_size, dim=0)
    return ((q - z) * scales.float()).t().to(torch.bfloat16)


# ---------------------------------------------------------------- full attention (training)


def dense_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    """Causal GQA attention (training path). q [B,T,Hq,D], k/v [B,T,Hkv,D] -> [B,T,Hq,D]."""
    q = _f32(q)
    k = _f32(k)
    v = _f32(v)
    b, t, hq, d = q.shape
    hkv = k.shape[2]
    group = hq // hkv
    if group > 1:
        k = k[:, :, :, None, :].expand(b, t, hkv, group, d).reshape(b, t, hq, d)
        v = v[:, :, :, None, :].expand(b, t, hkv, group, d).reshape(b, t, hq, d)
    att = torch.einsum("bthd,bshd->bhts", q, k) * scale
    mask = torch.triu(torch.full((t, t), float("-inf"), device=q.device), diagonal=1)
    p = torch.softmax(att + mask, dim=-1)
    return torch.einsum("bhts,bshd->bthd", p, v)


def dense_attention_bwd(
    grad: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward of :func:`dense_attention`. Returns (gq, gk, gv) in model layout."""
    q = _f32(q)
    k = _f32(k)
    v = _f32(v)
    grad = _f32(grad)
    b, t, hq, d = q.shape
    hkv = k.shape[2]
    group = hq // hkv
    if group > 1:
        ke = k[:, :, :, None, :].expand(b, t, hkv, group, d).reshape(b, t, hq, d)
        ve = v[:, :, :, None, :].expand(b, t, hkv, group, d).reshape(b, t, hq, d)
    else:
        ke, ve = k, v
    att = torch.einsum("bthd,bshd->bhts", q, ke) * scale
    mask = torch.triu(torch.full((t, t), float("-inf"), device=q.device), diagonal=1)
    p = torch.softmax(att + mask, dim=-1)
    gve = torch.einsum("bhts,bthd->bshd", p, grad)
    gatt = torch.einsum("bthd,bshd->bhts", grad, ve)
    gp = p * (gatt - (gatt * p).sum(-1, keepdim=True))
    gq = torch.einsum("bhts,bshd->bthd", gp, ke) * scale
    gke = torch.einsum("bhts,bthd->bshd", gp, q) * scale
    if group > 1:
        gk = gke.reshape(b, t, hkv, group, d).sum(3)
        gv = gve.reshape(b, t, hkv, group, d).sum(3)
    else:
        gk, gv = gke, gve
    return gq, gk, gv


def attention_gate_bwd(
    grad: torch.Tensor, attn_out: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward of ``attn_out * sigmoid(gate)``."""
    s = torch.sigmoid(_f32(gate))
    grad = _f32(grad)
    return grad * s, grad * _f32(attn_out) * s * (1.0 - s)


# ---------------------------------------------------------------- gated delta (full GDN layer)


def linear_attn_bwd(
    grad: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    **kw: Any,
) -> tuple[torch.Tensor, ...]:
    """Backward of :func:`gdn_forward` — 11 grads: q,k,v,g,beta,state,z,
    conv1d,dt_bias,a_log,norm."""
    if kw.pop("seq_q_lens", None) is not None:
        raise NotImplementedError("linear_attn_bwd does not support padded mixed-length rows")
    return gdn_backward(grad, q, k, v, g, beta, state, **kw)


#: gated-delta backward chunk: 16, not upstream's 64, for precision (worst rel
#: error vs autograd: C=16 4-12e-7, C=32 1.1-2.3e-6, C=64 1.9-4.9e-6)
_GDN_CHUNK = 16


def _gdn_chunk_fwd(qc, kc, vc, bc, gtc, s):
    """One chunk of the recurrence plus every intermediate its adjoint needs.
    qc/kc [B,n,HV,DK] (L2-normed, broadcast to value heads), vc [B,n,HV,DV],
    bc/gtc [B,n,HV], s [B,HV,DK,DV]; M = (I+L)^-1 carries the intra-chunk term."""
    n = qc.shape[1]
    gc = gtc.cumsum(1)
    e = torch.exp(gc)
    gp = gc.permute(0, 2, 1)
    # clamp before exp: the masked-out upper triangle would overflow to inf -> NaN
    D = torch.exp((gp.unsqueeze(-1) - gp.unsqueeze(-2)).clamp(max=0.0))
    dev, dt = qc.device, qc.dtype
    low = torch.tril(torch.ones(n, n, dtype=dt, device=dev), -1)
    tri = torch.tril(torch.ones(n, n, dtype=dt, device=dev))
    KK = torch.einsum("bihd,bjhd->bhij", kc, kc)
    bp = bc.permute(0, 2, 1).unsqueeze(-1)
    eye = torch.eye(n, dtype=dt, device=dev)
    M = torch.linalg.solve_triangular(eye + bp * KK * D * low, eye.expand(KK.shape),
                                      upper=False, unitriangular=True)
    bV = bp * vc.permute(0, 2, 1, 3)
    beK = (bc * e).permute(0, 2, 1).unsqueeze(-1) * kc.permute(0, 2, 1, 3)
    U, W = M @ bV, M @ beK
    d = U - W @ s
    QK = torch.einsum("bihd,bjhd->bhij", qc, kc)
    A = QK * D * tri
    P = e.permute(0, 2, 1).unsqueeze(-1) * qc.permute(0, 2, 1, 3)
    out = (P @ s + A @ d).permute(0, 2, 1, 3)
    glast = gc[:, -1]
    Rw = torch.exp((glast.unsqueeze(1) - gc).clamp(max=0.0))
    R = Rw.permute(0, 2, 1).unsqueeze(-1) * kc.permute(0, 2, 1, 3)
    s_next = torch.exp(glast).unsqueeze(-1).unsqueeze(-1) * s + R.transpose(-1, -2) @ d
    return out, s_next, dict(e=e, D=D, low=low, tri=tri, KK=KK, bp=bp, M=M, bV=bV,
                             beK=beK, W=W, d=d, QK=QK, A=A, P=P, Rw=Rw, R=R,
                             glast=glast, s=s)


def _gdn_chunk_bwd(dout, dS_next, qc, kc, vc, bc, c):
    """Adjoint of :func:`_gdn_chunk_fwd`: (dq, dk, dv, dbeta, dgt, dS_start) in [B,n,HV,*]."""
    e, D, low, tri = c["e"], c["D"], c["low"], c["tri"]
    KK, bp, M, W, d, QK, s = c["KK"], c["bp"], c["M"], c["W"], c["d"], c["QK"], c["s"]
    dOc = dout.permute(0, 2, 1, 3)

    # out = P s + A d
    dP = dOc @ s.transpose(-1, -2)
    dA = dOc @ d.transpose(-1, -2)
    dS = c["P"].transpose(-1, -2) @ dOc
    dd = c["A"].transpose(-1, -2) @ dOc
    # s_next = e_n s + R^T d
    en = torch.exp(c["glast"])
    dS = dS + en.unsqueeze(-1).unsqueeze(-1) * dS_next
    dR = d @ dS_next.transpose(-1, -2)
    dd = dd + c["R"] @ dS_next
    d_en = (s * dS_next).sum(dim=(-1, -2))
    # d = U - W s
    dW = -dd @ s.transpose(-1, -2)
    dS = dS - W.transpose(-1, -2) @ dd
    # U = M (b*V), W = M (b*e*K)
    dM = dd @ c["bV"].transpose(-1, -2) + dW @ c["beK"].transpose(-1, -2)
    dbV = M.transpose(-1, -2) @ dd
    dbeK = M.transpose(-1, -2) @ dW
    dv_ = bp * dbV
    dbeta = (vc.permute(0, 2, 1, 3) * dbV).sum(-1)
    dk_ = (bc * e).permute(0, 2, 1).unsqueeze(-1) * dbeK
    kdbeK = (kc.permute(0, 2, 1, 3) * dbeK).sum(-1)
    dbeta = dbeta + e.permute(0, 2, 1) * kdbeK
    de = bc.permute(0, 2, 1) * kdbeK
    # M = (I+L)^-1, L = tril(b_i <k_i,k_j> D_ij, -1)
    dLm = (-M.transpose(-1, -2) @ dM @ M.transpose(-1, -2)) * low
    dbeta = dbeta + (dLm * KK * D).sum(-1)
    dKK = dLm * bp * D
    dD = dLm * bp * KK
    # A = tril(QK * D)
    dAm = dA * tri
    dQK = dAm * D
    dD = dD + dAm * QK
    dq = torch.einsum("bhij,bjhd->bihd", dQK, kc)
    dk_ = dk_ + torch.einsum("bhij,bihd->bhjd", dQK, qc)
    dk_ = dk_ + (dKK + dKK.transpose(-1, -2)) @ kc.permute(0, 2, 1, 3)
    # P = e * q
    dq = dq + (e.permute(0, 2, 1).unsqueeze(-1) * dP).permute(0, 2, 1, 3)
    de = de + (qc.permute(0, 2, 1, 3) * dP).sum(-1)
    # R = exp(G_n - G) * k
    dk_ = dk_ + c["Rw"].permute(0, 2, 1).unsqueeze(-1) * dR
    dRwm = (kc.permute(0, 2, 1, 3) * dR).sum(-1) * c["Rw"].permute(0, 2, 1)
    # gates: D_ij = exp(G_i - G_j), e = exp(G), Rw = exp(G_n - G), e_n = exp(G_n)
    dDm = dD * D
    dG = dDm.sum(-1) - dDm.sum(-2) + de * e.permute(0, 2, 1) - dRwm
    dG[..., -1] = dG[..., -1] + dRwm.sum(-1) + d_en * en
    # G = cumsum(gt) over the chunk, so dgt is its reverse cumsum.
    dgt = dG.flip(-1).cumsum(-1).flip(-1).permute(0, 2, 1)
    return (dq, dk_.permute(0, 2, 1, 3), dv_.permute(0, 2, 1, 3),
            dbeta.permute(0, 2, 1), dgt, dS)


def gdn_chunk_core_fla(qn, kn, v, gt, bt, state, chunk: int = 64):
    """:func:`gdn_chunk_core` through flash-linear-attention (Triton, CUDA-only):
    a measurement path for how much of fla's 9.3x survives the layer's glue."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    rep = v.shape[2] // kn.shape[2]
    q = qn.repeat_interleave(rep, dim=2)
    k = kn.repeat_interleave(rep, dim=2)
    # qn already carries 1/sqrt(key_dim), so fla's scale is 1
    o, s = chunk_gated_delta_rule(
        q=q.bfloat16(), k=k.bfloat16(), v=v.bfloat16(), g=gt.float(),
        beta=bt.bfloat16(), scale=1.0, initial_state=state.float(),
        output_final_state=True, use_qk_l2norm_in_kernel=False,
    )
    return o.float(), s.float()


def gdn_chunk_core(qn, kn, v, gt, bt, state, chunk: int = 64):
    """The gated-delta core as a chunkwise-WY decomposition of the serial scan
    in :func:`gdn_forward`. qn/kn [B,T,HK,DK] (L2-normed), v [B,T,HV,DV],
    gt/bt [B,T,HV], state [B,HV,DK,DV] -> (core [B,T,HV,DV], final state)."""
    b, t, nvh, val_dim = v.shape
    rep = nvh // kn.shape[2]
    q = qn.repeat_interleave(rep, dim=2)
    k = kn.repeat_interleave(rep, dim=2)
    s = state.clone().float()
    core = torch.zeros(b, t, nvh, val_dim, dtype=torch.float32, device=v.device)
    for c0 in range(0, t, chunk):
        sl = slice(c0, min(c0 + chunk, t))
        core[:, sl], s, _ = _gdn_chunk_fwd(q[:, sl], k[:, sl], v[:, sl], bt[:, sl],
                                           gt[:, sl], s)
    return core, s


def gdn_prep(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    key_dim: int,
    *,
    conv1d_weight: torch.Tensor,
    dt_bias: torch.Tensor,
    a_log: torch.Tensor,
    conv_window: torch.Tensor | None = None,
    seq_q_lens: torch.Tensor | None = None,
    keep_steps: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Front half of :func:`gdn_forward`, and the oracle for the ``gdn_prep``
    kernel: conv1d + SiLU over q/k/v, the q/k L2-norm with 1/sqrt(key_dim) folded
    into q, the log gate, sigmoid beta, and the next conv window. Returns
    (qn, kn, v_raw, gt, bt, new_window) -- q/k [B,T,nkh,K], v [B,T,nvh,V]."""
    dev = q.device
    q, k, v, g, beta = _f32(q), _f32(k), _f32(v), _f32(g), _f32(beta)
    conv1d_weight = _f32(conv1d_weight).to(dev)
    dt_bias, a_log = _f32(dt_bias).to(dev), _f32(a_log).to(dev)
    b, t, _ = q.shape
    nvh = g.shape[-1]
    val_dim, nkh = v.shape[-1] // nvh, q.shape[-1] // key_dim
    kernel = conv1d_weight.shape[1]
    if seq_q_lens is None:
        seq_q_lens = torch.full((b,), t, dtype=torch.long, device=dev)
    else:
        seq_q_lens = torch.as_tensor(seq_q_lens, dtype=torch.long, device=dev).reshape(b)
    qkv = torch.cat([q, k, v], dim=-1)
    new_window = None
    if conv_window is not None:
        # new window: last K-1 of (window ++ the row's valid qkv), per row
        qkv = torch.cat([_f32(conv_window).to(dev), qkv], dim=1)
        new_window = (
            torch.stack([qkv[:, s + 1 : kernel + s] for s in range(keep_steps)], dim=1)
            if keep_steps
            else torch.stack(
                [qkv[bi, : kernel - 1 + int(seq_q_lens[bi])][-(kernel - 1) :] for bi in range(b)]
            )
        ).contiguous()
    preact = torch.zeros_like(qkv)
    for tap in range(kernel):
        pad_left = kernel - 1 - tap
        padded = torch.nn.functional.pad(qkv, (0, 0, pad_left, tap))
        preact = preact + padded[:, : qkv.shape[1], :] * conv1d_weight[:, tap]
    preact = preact[:, -t:]  # drop the carried window positions
    silu = lambda x: x * torch.sigmoid(x)
    q_raw = silu(preact[..., : nkh * key_dim]).view(b, t, nkh, key_dim)
    k_raw = silu(preact[..., nkh * key_dim : 2 * nkh * key_dim]).view(b, t, nkh, key_dim)
    v_raw = silu(preact[..., 2 * nkh * key_dim :]).view(b, t, nvh, val_dim)
    qn = q_raw / torch.sqrt(q_raw.pow(2).sum(-1, keepdim=True) + 1e-12) / math.sqrt(key_dim)
    kn = k_raw / torch.sqrt(k_raw.pow(2).sum(-1, keepdim=True) + 1e-12)
    bt = torch.sigmoid(beta).view(b, t, nvh)
    gt = -torch.exp(a_log) * torch.nn.functional.softplus(g + dt_bias)
    return qn, kn, v_raw, gt, bt, new_window


def gdn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    *,
    z: torch.Tensor,
    conv1d_weight: torch.Tensor,
    dt_bias: torch.Tensor,
    a_log: torch.Tensor,
    norm_weight: torch.Tensor,
    conv_window: torch.Tensor | None = None,
    seq_q_lens: torch.Tensor | None = None,
    keep_steps: int = 0,
    chunkwise: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Full GDN layer core, the executable spec (agent-infer's host reference
    equation by equation). q/k [B,T,nkh*K], v/z [B,T,nvh*V], g/beta [B,T,nvh],
    state [B,nvh,K,V]. ``conv_window`` [B,K-1,qkv_dim] is the previous
    segment's last raw-qkv tokens (None: zero left-padding, third return
    None). ``seq_q_lens`` [B] bounds each row's scan. ``keep_steps`` > 0
    returns the state/window after each of the first keep_steps tokens with a
    leading chain-step axis (speculative verify)."""
    # ponytail: per-call param migration, keep params on the backend device at load
    dev = q.device
    state = _f32(state).to(dev)
    z = _f32(z)
    norm_weight = _f32(norm_weight).to(dev)
    b, t, _ = q.shape
    nvh, key_dim, val_dim = state.shape[1], state.shape[2], state.shape[3]
    nkh = q.shape[-1] // key_dim
    if seq_q_lens is None:
        seq_q_lens = torch.full((b,), t, dtype=torch.long, device=dev)
    else:
        seq_q_lens = torch.as_tensor(seq_q_lens, dtype=torch.long, device=dev).reshape(b)
    qn, kn, v_raw, gt, bt, new_window = gdn_prep(
        q, k, v, g, beta, key_dim,
        conv1d_weight=conv1d_weight, dt_bias=dt_bias, a_log=a_log,
        conv_window=conv_window, seq_q_lens=seq_q_lens, keep_steps=keep_steps,
    )
    silu = lambda x: x * torch.sigmoid(x)
    exp_g = torch.exp(gt)
    if chunkwise and not keep_steps and int(seq_q_lens.min()) == t:
        # the chunked form has no per-row mask: full-length rows only
        impl = gdn_chunk_core_fla if os.environ.get("TILERL_GDN_FLA") else gdn_chunk_core
        core, s = impl(qn, kn, v_raw, gt, bt, state, chunk=chunkwise)
    else:
        s_heads = [state[:, h].clone() for h in range(nvh)]
        steps: list[torch.Tensor] = []
        core = torch.zeros(b, t, nvh, val_dim, dtype=torch.float32, device=q.device)
        for step in range(t):
            active = (seq_q_lens > step).reshape(b, 1, 1)
            active_h = (seq_q_lens > step).reshape(b, 1)
            for h in range(nvh):
                kh = h * nkh // nvh
                s_h = s_heads[h] * exp_g[:, step, h].view(b, 1, 1)
                p = torch.einsum("bk,bkv->bv", kn[:, step, kh], s_h)
                d = (v_raw[:, step, h] - p) * bt[:, step, h].unsqueeze(-1)
                s_h = s_h + kn[:, step, kh].unsqueeze(-1) * d.unsqueeze(-2)
                s_heads[h] = torch.where(active, s_h, s_heads[h])
                core[:, step, h] = torch.where(
                    active_h, torch.einsum("bk,bkv->bv", qn[:, step, kh], s_h), 0
                )
            if step < keep_steps:
                steps.append(torch.stack(s_heads, dim=1))
        s = torch.stack(steps, dim=1) if keep_steps else torch.stack(s_heads, dim=1)
    normed = core * torch.rsqrt(core.pow(2).mean(-1, keepdim=True) + 1e-6)
    normed = normed * norm_weight
    out = normed * silu(z.view(b, t, nvh, val_dim))
    return out.reshape(b, t, nvh * val_dim), s, new_window


linear_attn_chunk = gdn_forward  # the op name in the backend contract


def gdn_backward(
    grad: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    *,
    z: torch.Tensor,
    conv1d_weight: torch.Tensor,
    dt_bias: torch.Tensor,
    a_log: torch.Tensor,
    norm_weight: torch.Tensor,
    conv_window: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Backward of :func:`gdn_forward`: (gq,gk,gv,gg,gbeta,gstate,gz,gconv1d,
    gdt_bias,ga_log,gnorm_weight). Only a zero ``conv_window`` is exact
    (training forwards start from one); a non-zero window raises."""
    if conv_window is not None and bool(conv_window.any()):
        raise NotImplementedError("gdn_backward does not support a non-zero conv_window")
    dev = q.device
    q = _f32(q)
    k = _f32(k)
    v = _f32(v)
    g = _f32(g)
    beta = _f32(beta)
    state = _f32(state).to(dev)
    z = _f32(z)
    conv1d_weight = _f32(conv1d_weight).to(dev)
    dt_bias = _f32(dt_bias).to(dev)
    a_log = _f32(a_log).to(dev)
    norm_weight = _f32(norm_weight).to(dev)
    go = _f32(grad)
    b, t, _ = q.shape
    nvh, key_dim, val_dim = state.shape[1], state.shape[2], state.shape[3]
    nkh = q.shape[-1] // key_dim
    kernel = conv1d_weight.shape[1]

    # ---- forward (save intermediates) ----
    qkv = torch.cat([q, k, v], dim=-1)
    preact = torch.zeros_like(qkv)
    for tap in range(kernel):
        pad_left = kernel - 1 - tap
        padded = torch.nn.functional.pad(qkv, (0, 0, pad_left, tap))
        preact = preact + padded[:, :t, :] * conv1d_weight[:, tap]
    silu = lambda x: x * torch.sigmoid(x)
    q_raw = silu(preact[..., : nkh * key_dim]).view(b, t, nkh, key_dim)
    k_raw = silu(preact[..., nkh * key_dim : 2 * nkh * key_dim]).view(b, t, nkh, key_dim)
    v_raw = silu(preact[..., 2 * nkh * key_dim :]).view(b, t, nvh, val_dim)
    rq = torch.rsqrt(q_raw.pow(2).sum(-1, keepdim=True) + 1e-12)
    rk = torch.rsqrt(k_raw.pow(2).sum(-1, keepdim=True) + 1e-12)
    qn = q_raw * rq / math.sqrt(key_dim)
    kn = k_raw * rk
    bt = torch.sigmoid(beta).view(b, t, nvh)
    sp_in = g + dt_bias
    gt = -torch.exp(a_log) * torch.nn.functional.softplus(sp_in)
    rep = nvh // nkh
    assert nkh * rep == nvh, (nkh, nvh)
    knv = kn.repeat_interleave(rep, dim=2)
    qnv = qn.repeat_interleave(rep, dim=2)
    # chunked, not per-step: the per-step scan was 62% of a 27B train step
    # (errors/2026-08-29-train-step-is-the-gdn-per-step-loop.md); chunk
    # intermediates are kept (~40 MB a layer) rather than recomputed
    starts = list(range(0, t, _GDN_CHUNK))
    caches = []
    s = state.clone()
    core = torch.zeros(b, t, nvh, val_dim, dtype=torch.float32, device=q.device)
    for c0 in starts:
        sl = slice(c0, min(c0 + _GDN_CHUNK, t))
        core[:, sl], s, cache = _gdn_chunk_fwd(qnv[:, sl], knv[:, sl], v_raw[:, sl],
                                               bt[:, sl], gt[:, sl], s)
        caches.append(cache)
    rstd = torch.rsqrt(core.pow(2).mean(-1, keepdim=True) + 1e-6)
    normed = core * rstd * norm_weight
    z4 = z.view(b, t, nvh, val_dim)
    sz = torch.sigmoid(z4)

    # ---- backward ----
    go4 = go.reshape(b, t, nvh, val_dim)
    g_normed = go4 * (z4 * sz)
    g_z = (go4 * normed * sz * (1.0 + z4 * (1.0 - sz))).reshape(b, t, nvh * val_dim)
    xhat = core * rstd
    g_y = g_normed * norm_weight
    g_norm_weight = (g_normed * xhat).sum(dim=(0, 1, 2))
    g_core = rstd * (g_y - xhat * (g_y * xhat).mean(-1, keepdim=True))

    dS = torch.zeros_like(state)
    g_qnv = torch.zeros(b, t, nvh, key_dim, dtype=torch.float32, device=q.device)
    g_knv = torch.zeros(b, t, nvh, key_dim, dtype=torch.float32, device=q.device)
    g_v_raw = torch.zeros(b, t, nvh, val_dim, dtype=torch.float32, device=q.device)
    g_bt = torch.zeros(b, t, nvh, dtype=torch.float32, device=q.device)
    g_gt = torch.zeros(b, t, nvh, dtype=torch.float32, device=q.device)
    for i in reversed(range(len(starts))):
        sl = slice(starts[i], min(starts[i] + _GDN_CHUNK, t))
        (g_qnv[:, sl], g_knv[:, sl], g_v_raw[:, sl], g_bt[:, sl], g_gt[:, sl],
         dS) = _gdn_chunk_bwd(g_core[:, sl], dS, qnv[:, sl], knv[:, sl],
                              v_raw[:, sl], bt[:, sl], caches[i])
    # value-head grads fold back onto contiguous key-head groups
    g_qn = g_qnv.reshape(b, t, nkh, rep, key_dim).sum(3)
    g_kn = g_knv.reshape(b, t, nkh, rep, key_dim).sum(3)
    g_a_log = (g_gt * gt).sum(dim=(0, 1))
    g_sp_in = g_gt * (-torch.exp(a_log)) * torch.sigmoid(sp_in)
    g_g = g_sp_in.reshape(b, t, nvh)
    g_dt_bias = g_sp_in.sum(dim=(0, 1))
    g_beta = (g_bt * bt * (1.0 - bt)).reshape(b, t, nvh)

    def _norm_bwd(g_in, x, r, s):  # y = x * r * s, r = rsqrt(sum(x^2) + eps)
        return r * s * g_in - (r**3) * s * x * (g_in * x).sum(-1, keepdim=True)

    g_q_raw = _norm_bwd(g_qn, q_raw, rq, 1.0 / math.sqrt(key_dim))
    g_k_raw = _norm_bwd(g_kn, k_raw, rk, 1.0)
    sig = torch.sigmoid(preact)
    dsilu = sig * (1.0 + preact * (1.0 - sig))
    g_preact = torch.zeros_like(preact)
    g_preact[..., : nkh * key_dim] = (
        g_q_raw.reshape(b, t, nkh * key_dim) * dsilu[..., : nkh * key_dim]
    )
    g_preact[..., nkh * key_dim : 2 * nkh * key_dim] = (
        g_k_raw.reshape(b, t, nkh * key_dim) * dsilu[..., nkh * key_dim : 2 * nkh * key_dim]
    )
    g_preact[..., 2 * nkh * key_dim :] = (
        g_v_raw.reshape(b, t, nvh * val_dim) * dsilu[..., 2 * nkh * key_dim :]
    )
    g_qkv = torch.zeros_like(qkv)
    g_conv = torch.zeros_like(conv1d_weight)
    for tap in range(kernel):
        shift = tap - (kernel - 1)
        lo_q, hi_q = max(0, shift), shift + t
        lo_g, hi_g = max(0, -shift), t
        g_qkv[:, lo_q:hi_q, :] += conv1d_weight[:, tap] * g_preact[:, lo_g:hi_g, :]
        g_conv[:, tap] = (qkv[:, lo_q:hi_q, :] * g_preact[:, lo_g:hi_g, :]).sum(dim=(0, 1))
    g_q = g_qkv[..., : q.shape[-1]]
    g_k = g_qkv[..., q.shape[-1] : q.shape[-1] + k.shape[-1]]
    g_v = g_qkv[..., q.shape[-1] + k.shape[-1] :]
    return g_q, g_k, g_v, g_g, g_beta, dS, g_z, g_conv, g_dt_bias, g_a_log, g_norm_weight


# ---------------------------------------------------------------- silu mul


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """y = silu(gate) * up = gate * sigmoid(gate) * up."""
    gate = _f32(gate)
    up = _f32(up)
    return torch.nn.functional.silu(gate) * up


def silu_mul_bwd(
    grad: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward of :func:`silu_mul`. Returns (ggate, gup)."""
    gate = _f32(gate)
    up = _f32(up)
    grad = _f32(grad)
    s = torch.sigmoid(gate)
    dsilu = s * (1.0 + gate * (1.0 - s))
    ggate = grad * up * dsilu
    gup = grad * gate * s
    return ggate, gup


# ---------------------------------------------------------------- softmax


def softmax(x: torch.Tensor, axis: int) -> torch.Tensor:
    """Numerically stable softmax along ``axis``."""
    x = _f32(x)
    return torch.softmax(x, dim=axis)


def cross_entropy_loss_grad(logits: torch.Tensor, input_ids: object) -> tuple[float, torch.Tensor]:
    """Stable shifted causal CE and its matching logit gradient. ``softmax - onehot``
    needs no second buffer, so it is written over ``logits`` when they are already
    f32 and contiguous: the returned gradient then ALIASES the input and no caller
    may read the logits afterwards. A vocab row is 2.2 GiB for a 27B group of 8 at
    T=275, and the shape-for-shape version held four of them."""
    b, t, v = logits.shape
    if t < 2:
        raise ValueError("cross_entropy_loss_grad needs at least two tokens")
    g = logits if logits.dtype == torch.float32 and logits.is_contiguous() else _f32(
        logits).contiguous()
    ids = torch.as_tensor(input_ids, dtype=torch.long, device=g.device)
    tgt = torch.full((b, t), -1, dtype=torch.long, device=g.device)
    tgt[:, :-1] = ids[:, 1:]  # -1 marks the last column, which predicts nothing
    tgt = tgt.reshape(-1)
    flat = g.reshape(b * t, v)
    keep = tgt >= 0
    y = tgt.clamp_min(0).unsqueeze(-1)
    picked = flat.gather(-1, y).squeeze(-1)
    m = flat.amax(-1)
    flat.sub_(m.unsqueeze(-1)).exp_()
    s = flat.sum(-1)
    n = float(b * (t - 1))
    loss = torch.where(keep, m + s.log() - picked, s.new_zeros(())).sum() / n
    flat.div_(s.unsqueeze(-1) * n)
    flat.scatter_add_(-1, y, keep.to(flat.dtype).unsqueeze(-1).mul_(-1.0 / n))
    g[:, -1] = 0.0
    return float(loss), g


def state_gather(states, windows, slots, layer_idx, parity=None):
    """Gather one recurrent-state layer for a batch of slots; ``parity`` [S]
    picks the live conv-window plane of the double-buffered pool."""
    slots = torch.as_tensor(slots, dtype=torch.long, device=states.device).reshape(-1)
    if windows is None:
        return states[slots, layer_idx], None
    par = torch.zeros_like(slots) if parity is None else parity[slots].long()
    return states[slots, layer_idx], windows[slots, layer_idx, par]


def state_scatter(
    states, windows, slots, layer_idx, new_state, new_window, parity=None, steps=False
) -> None:
    """Store one recurrent-state layer for a batch of slots. ``steps``: the
    tensors carry a chain-step axis and land in the pool's leading step planes."""
    slots = torch.as_tensor(slots, dtype=torch.long, device=states.device).reshape(-1)
    # device too, not just dtype: a caller holding CPU state writes into a CUDA pool
    def _as(t, ref):
        return t.to(device=ref.device, dtype=ref.dtype)

    if steps:
        ks = new_state.shape[1]
        states[slots, layer_idx, :ks] = _as(new_state, states)
        if new_window is not None:
            windows[slots, layer_idx, :ks] = _as(new_window, windows)
        return
    states[slots, layer_idx] = _as(new_state, states)
    if new_window is not None:
        par = torch.zeros_like(slots) if parity is None else parity[slots].long()
        windows[slots, layer_idx, par] = _as(new_window, windows)


# ---------------------------------------------------------------- embedding


def embedding(idx: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Gather. idx [...] int, table [V, D] -> out [..., D]."""
    return table[idx.to(torch.long)]


def embedding_bwd(grad: torch.Tensor, idx: torch.Tensor, num_rows: int) -> torch.Tensor:
    """Scatter-add backward. Returns gtable [num_rows, D]."""
    grad = _f32(grad)
    gtable = torch.zeros(num_rows, grad.shape[-1], dtype=torch.float32, device=grad.device)
    gtable.index_add_(0, idx.to(torch.long).reshape(-1), grad.reshape(-1, grad.shape[-1]))
    return gtable


# ---------------------------------------------------------------- sampling


def greedy(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Argmax token and its softmax probability. [..., V] -> ([...], [...])."""
    prob, tok = torch.softmax(_f32(logits), dim=-1).max(-1)
    return tok.to(torch.long), prob


def top_p_probs(logits: torch.Tensor, top_p: float | torch.Tensor):
    """The nucleus distribution the sampler draws from, in descending-probability
    order: ``(probs, sorted_idx)``. One definition, so scoring a token and drawing
    one cannot disagree about which distribution produced it."""
    sorted_logits, sorted_idx = torch.sort(_f32(logits), dim=-1, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    if not isinstance(top_p, torch.Tensor):
        top_p = torch.as_tensor(top_p, dtype=probs.dtype, device=probs.device)
    keep = torch.cumsum(probs, dim=-1) - probs < top_p.reshape(-1, *([1] * (probs.dim() - 1)))
    keep[..., 0] = True
    probs = probs * keep
    return probs / probs.sum(-1, keepdim=True), sorted_idx


def sample(logits: torch.Tensor, temperature: float, top_p: float, seed: int) -> torch.Tensor:
    """Top-p sampling, deterministic per seed; temperature <= 0 is greedy."""
    logits = _f32(logits)
    if temperature <= 0:
        return logits.argmax(-1).to(torch.long)
    gen = torch.Generator(device=logits.device).manual_seed(int(seed))
    if temperature != 1.0:
        logits = logits / temperature
    probs, sorted_idx = top_p_probs(logits, top_p)
    sampled = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)
    return sorted_idx.gather(-1, sampled.unsqueeze(-1)).squeeze(-1).to(torch.long)


def sample_batch(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ps: torch.Tensor,
    seeds: torch.Tensor,
    logprobs: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Batched :func:`sample`: one sort/softmax for the batch, identical draws
    per row. temperatures/top_ps/seeds are read on the host (device reads cost
    two syncs a tick plus one per sampled row). Returns the tokens and their log
    prob under the row's own nucleus distribution -- the caller cannot recompute
    that from the logits without duplicating the truncation rule. A greedy row
    (t <= 0) is scored at t=1 over the full softmax, since its point mass would
    report 0 for every token -- which is a second full-vocabulary pass, so
    ``logprobs=False`` skips it and returns None for the scores."""
    logits = _f32(logits)
    b = logits.shape[0]
    dev = logits.device
    temps = [float(t) for t in temperatures]
    out = torch.empty(b, dtype=torch.long, device=dev)
    lp = torch.empty(b, dtype=torch.float32, device=dev) if logprobs else None
    hot = [i for i, t in enumerate(temps) if t > 0]
    cold = [i for i, t in enumerate(temps) if t <= 0]
    if cold:
        ci = torch.tensor(cold, device=dev)
        out[ci] = logits[ci].argmax(-1).to(torch.long)
        if logprobs:
            lp[ci] = torch.log_softmax(logits[ci], dim=-1).max(-1).values
    if hot:
        idx = torch.tensor(hot, device=dev)
        tt = torch.tensor([temps[i] for i in hot], dtype=torch.float32, device=dev)
        tp = torch.tensor([float(top_ps[i]) for i in hot], dtype=torch.float32, device=dev)
        sub = logits[idx] / tt[:, None]
        probs, sorted_idx = top_p_probs(sub, tp)
        for k, i in enumerate(hot):
            gen = torch.Generator(device=dev).manual_seed(int(seeds[i]))
            draw = torch.multinomial(probs[k], num_samples=1, generator=gen)
            out[i] = sorted_idx[k, draw].to(torch.long)
            if logprobs:
                lp[i] = probs[k, draw].clamp_min(1e-45).log()
    return out, lp


def _check_chunk_core() -> None:
    """chunkwise-WY == the serial scan, at a 3:1 head ratio and a non-chunk-divisible T."""
    torch.manual_seed(0)
    b, t, hk, hv, dk, dv = 2, 70, 2, 6, 16, 16
    qn = torch.randn(b, t, hk, dk)
    qn = qn / qn.norm(dim=-1, keepdim=True) / math.sqrt(dk)
    kn = torch.randn(b, t, hk, dk)
    kn = kn / kn.norm(dim=-1, keepdim=True)
    v, gt = torch.randn(b, t, hv, dv), -torch.rand(b, t, hv) * 0.3
    bt, s0 = torch.rand(b, t, hv), torch.randn(b, hv, dk, dv) * 0.1
    s, core = s0.clone(), torch.zeros(b, t, hv, dv)
    for step in range(t):
        for h in range(hv):
            kh = h * hk // hv
            sh = s[:, h] * torch.exp(gt[:, step, h]).view(b, 1, 1)
            d = (v[:, step, h] - torch.einsum("bk,bkv->bv", kn[:, step, kh], sh)) * bt[
                :, step, h
            ].unsqueeze(-1)
            s[:, h] = sh + kn[:, step, kh].unsqueeze(-1) * d.unsqueeze(-2)
            core[:, step, h] = torch.einsum("bk,bkv->bv", qn[:, step, kh], s[:, h])
    for chunk in (16, 32, 64):
        c2, s2 = gdn_chunk_core(qn, kn, v, gt, bt, s0, chunk=chunk)
        ec = ((c2 - core).abs().max() / core.abs().max()).item()
        es = ((s2 - s).abs().max() / s.abs().max()).item()
        assert ec < 1e-4 and es < 1e-4, (chunk, ec, es)
    print("reference: chunkwise-WY == serial decay-first")


if __name__ == "__main__":  # runnable check: quant_fp8 inverts dequant_fp8
    torch.manual_seed(0)
    _w = torch.randn(300, 260, dtype=torch.bfloat16) * 0.02
    _q, _s = quant_fp8(_w)
    assert _q.shape == _w.shape and _s.shape == (3, 3), (_q.shape, _s.shape)
    # e4m3 keeps 3 mantissa bits: ~6% per element near a block's absmax
    _rel = (dequant_fp8(_q, _s) - _w.float()).abs().max() / _w.float().abs().max()
    assert _rel < 0.06, _rel
    print("reference: quant_fp8 round-trip OK, rel", float(_rel))
    _check_chunk_core()
