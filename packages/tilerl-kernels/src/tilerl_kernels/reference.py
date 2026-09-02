"""Torch-eager reference implementations of every tilerl op (forward + backward).

This module is the parity oracle for the TileLang kernels in :mod:`tilerl_kernels.kernels`
and the day-1 backward fallback for ops without a TileLang backward kernel
(``# ponytail: torch-eager backward, tilelang kernel when perf demands``).

Conventions: all compute is float32 (callers cast bf16/f16 to f32;
``# ponytail: f32 compute day-1, bf16 mixed precision day-2``); deterministic
(no randomness, no in-place mutation, no autograd — backward formulas are
hand-derived and match the agent-infer Rust reference at
``crates/autograd/src/ops/linear_attention.rs``). Shapes: ``B`` batch,
``T``/``C`` sequence/chunk, ``H`` heads, ``D`` head dim, ``M``/``N``/``K``
matmul dims.
"""

from __future__ import annotations

import math
import os

import torch
from typing import Any

__all__ = [
    "rmsnorm",
    "rmsnorm_bwd",
    "rope",
    "rope_bwd",
    "linear",
    "linear_bwd",
    "dequant_fp4",
    "linear_fp4",
    "pack_fp4",
    "renorm_fp4_scale",
    "unpack_fp4",
    "dequant_nvfp4",
    "dequant_fp8",
    "quant_fp8",
    "linear_fp8",
    "dequant_awq",
    "dense_attention",
    "dense_attention_bwd",
    "attention_gate_bwd",
    "linear_attn_chunk",
    "linear_attn_bwd",
    "gdn_forward",
    "gdn_chunk_core",
    "gdn_chunk_core_fla",
    "gdn_backward",
    "silu_mul",
    "silu_mul_bwd",
    "softmax",
    "cross_entropy_loss_grad",
    "state_gather",
    "state_scatter",
    "embedding",
    "embedding_bwd",
    "sample",
    "sample_batch",
]


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
    # c = mean(grad * w * x) over the last dim
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
    # rotate_half convention (Qwen/Llama): pair dim d with d+rd/2, not the
    # adjacent (2d, 2d+1) GPT-J pairing. The checkpoint's weights expect this.
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
    """Rotary embedding. x [B, T, H, D], positions [B, T] or [T].

    ``rotary_dim`` rotates only the first ``rotary_dim`` features (Qwen3.8
    partial RoPE); the rest pass through. Defaults to the full last dim.
    """
    return _rope_apply(x, positions, theta, negate=False, rotary_dim=rotary_dim)


def rope_bwd(
    grad: torch.Tensor, positions: torch.Tensor, theta: float, rotary_dim: int | None = None
) -> torch.Tensor:
    """Backward of :func:`rope`; the rotation is orthogonal, so gx = R(-angle) grad."""
    return _rope_apply(grad, positions, theta, negate=True, rotary_dim=rotary_dim)


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


#: Bytes of dequantized weight linear_frozen_bwd will materialize at once.
#: The whole lm_head is [248320, 5120] f32 = 4.74 GiB, and materializing it
#: (twice, with the oscale fold) made this one call peak 14.2 GiB — every other
#: backward op in a 27B step peaks under 0.12.
_BWD_SLICE_BYTES = 1 << 29


def linear_frozen_bwd(grad, wq, scale, oscale=None, fp8=False):
    """dX through a frozen quantized weight (LoRA / OPD base): no weight grad,
    so the base never needs a bf16 master — the only way the 27B fits one card.

    dX contracts over N, so the weight is materialized — but a slice at a time.
    ``oscale`` scales weight ROW n, so it folds into the [M, N] gradient rather
    than the [N, K] weight, which is where the fp4 kernel path already puts it.
    # ponytail: no tilelang fp8 dequant kernel yet (fp4 has one) — this is the
    # eager path, chunked.
    """
    g = _f32(grad).reshape(-1, grad.shape[-1])
    if oscale is not None:
        g = g * _f32(oscale).reshape(1, -1)
    n, k = wq.shape[0], wq.shape[1] * (1 if fp8 else 2)
    # fp4 scales one weight row each; fp8 scales a 128-row block. Slice on the
    # scale's own row granularity so a chunk boundary never splits a block.
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
    """Dequantize OCP e2m1-packed weights. wq uint8 [N, K//2] (low nibble
    first), scale f32 [N, K//B]. Returns w [N, K] f32. The block size B is
    derived from the two shapes (16 for an NVFP4 checkpoint, 32 for pack_fp4).

    Magnitudes are ``_E2M1_LUT`` ({0,.5,1,1.5,2,3,4,6}); bit 3 is the sign.
    """
    if getattr(wq, "_tl_twiddled", False):  # sm90 served bytes (Backend._served_fp4)
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

#: OCP/MX e2m1 magnitude LUT, low 3 bits of a nibble; bit 3 = sign. The one
#: fp4 grid in tileRL — pack_fp4/unpack_fp4, dequant_fp4, dequant_nvfp4 and
#: every linear_fp4 kernel decode it, so an NVFP4 checkpoint's nibbles need no
#: re-quantization to be served.
_E2M1_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


# sm90 serves fp4 in the "twiddled" byte layout tilelang's decode_fp4_to_bf16
# intrinsic expands with 18 ops per 8 elems (kernels_linear._FP4_TWIDDLE_SRC):
# per 32-bit word, element slot p of half-word h has its (s, e1, e0, m) bits at
# _TW_POS[p]; slot j (j<4) holds elem 2j+1 and slot j+4 elem 2j so the decoded
# bf16x2 pairs line up with natural bf16x2 X words.
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
    """Pack a bf16/f32 weight [N,K] into OCP e2m1 nibbles + per-block scales.

    Returns ``(wq [N,K//2] uint8 low-nibble-first, scale [N,K//block] f32)``
    with ``scale = block_max / 6`` and round-to-nearest against ``_E2M1_LUT``.
    The max representable magnitude is 6*scale, so the block max maps exactly.
    Block 32 matches the fp8 WGMMA K-tile (sm90), so the fp8 prefill path can
    apply one scale per MMA tile in f32 (no e4m3 weight requant). Serving wants
    :func:`renorm_fp4_scale` on the result.
    """
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
    """Move a per-row power of two out of the fp4 block scale into the epilogue
    scale: ``(scale * 2**-p, oscale * 2**p)``, ``p = floor(log2(row max))``, so
    every row's ``6*scale`` lands in [6,12). The w4a8 kernel dequantizes into
    e4m3 (max 448, subnormal below 2^-9), where raw checkpoint magnitudes
    saturate and ``block_max/6`` weight units collapse; 2^p is exact, so the
    split re-rounds nothing."""
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
    """ModelOpt NVFP4 dequant (Qwen3.6 MLP linears). weight_packed uint8
    [N,K//2] (two OCP/MX e2m1 nibbles per byte, low nibble first),
    weight_scale f8_e4m3 [N,K//16] (block 1x16), weight_global_scale f32 [1].
    Returns bf16 [N,K]: ``w = e2m1(packed) * f8(weight_scale) * gs`` where
    ``gs`` is ``1/weight_global_scale`` when ``global_divide`` (ModelOpt stores
    the global scale's reciprocal — agent-infer quant_format.rs:225,
    ScaleApply::Divide) and ``weight_global_scale`` directly otherwise
    (official NVFP4's ``weight_scale_2`` is a plain multiplier). The e2m1
    grid is the one :func:`dequant_fp4` decodes, which derives the block (16)
    from the scale shape — so this is that dequant plus the global scale."""
    gs = weight_global_scale.float()
    if global_divide:
        gs = 1.0 / gs
    return (dequant_fp4(weight_packed, weight_scale) * gs).to(torch.bfloat16)


def dequant_fp8(w8: torch.Tensor, wscale: torch.Tensor, block: int = 128) -> torch.Tensor:
    """FP8 block-quant dequant (Qwen3.6 GDN linears, kept native by load_hf).
    w8 f8_e4m3 [N,K], wscale f32 [ceil(N/block), ceil(K/block)] (per-block
    scale, multiplied — the checkpoint's "scale_inv" is the scale itself,
    agent-infer quant_format.rs ScaleApply::Multiply). Returns f32 [N,K]:
    ``w = f8(w8) * wscale.repeat(block)``. The same layout serves per-tensor
    FP8 (the loader expands the scalar to a constant wscale) so one kernel
    covers both."""
    n, k = w8.shape
    s = wscale.float().repeat_interleave(block, dim=-1)[:, :k]
    s = s.repeat_interleave(block, dim=-2)[:n, :]
    return w8.float() * s


def quant_fp8(w: torch.Tensor, block: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`dequant_fp8`: bf16/f32 [N,K] -> (e4m3 [N,K], f32 scales
    [ceil(N/block), ceil(K/block)]). One scale per 128x128 block, from its
    absmax against e4m3's 448 range."""
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
    """AutoAWQ GEMM dequant. qweight int32 [K,N//8] (8 int4 per int32, for 8
    consecutive output features; int4 at bits (j%8)*4 of qweight[i,j//8]),
    qzeros int32 [K//group,N//8] (same packing), scales bf16/fp16
    [K//group,N]. Returns bf16 [N,K] (PyTorch Linear layout):
    ``w[j,i] = (q - z) * s`` per group, transposed from the GEMM [K,N]."""
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
    """Causal GQA attention in model layout. q [B,T,Hq,D], k/v [B,T,Hkv,D].

    Training-path op (the paged kernel serves decode). Hq must be a multiple
    of Hkv; kv heads are repeated to match. Returns [B,T,Hq,D].
    """
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


#: Chunk length of the gated-delta scan: the sequential dimension becomes
#: t/_GDN_CHUNK. 16, not the upstream 64, for PRECISION — the chunked form is
#: the same algebra (equal to the serial scan to 1e-15 in f64) but a different
#: f32 reduction order, and its worst relative error against autograd grows with
#: the chunk. Measured over 3 shapes x 3 seeds vs the serial backward's 3-9e-7:
#:   C=16 -> 4-12e-7 (1.3-2.2x)   C=32 -> 1.1-2.3e-6   C=64 -> 1.9-4.9e-6
_GDN_CHUNK = 16


def _gdn_chunk_fwd(qc, kc, vc, bc, gtc, s):
    """One chunk of the gated-delta recurrence, plus every intermediate its
    adjoint needs. ``qc``/``kc`` [B,n,HV,DK] (L2-normed, already broadcast to
    value heads), ``vc`` [B,n,HV,DV], ``bc``/``gtc`` [B,n,HV], ``s`` [B,HV,DK,DV].

    The chunk matrices are [B,HV,n,*]: G is the chunk-local inclusive cumsum of
    the log decay, ``M = (I+L)^-1`` is the UT transform that carries the
    intra-chunk term the serial form keeps in the state.
    """
    n = qc.shape[1]
    gc = gtc.cumsum(1)
    e = torch.exp(gc)
    gp = gc.permute(0, 2, 1)
    # clamp before exp: gt <= 0 makes G non-increasing, so every entry the masks
    # keep has a non-positive difference; the discarded upper triangle would
    # overflow to inf and the mask would turn it into NaN.
    D = torch.exp((gp.unsqueeze(-1) - gp.unsqueeze(-2)).clamp(max=0.0))
    dev, dt = qc.device, qc.dtype
    low = torch.tril(torch.ones(n, n, dtype=dt, device=dev), -1)
    tri = torch.tril(torch.ones(n, n, dtype=dt, device=dev))
    KK = torch.einsum("bihd,bjhd->bhij", kc, kc)
    bp = bc.permute(0, 2, 1).unsqueeze(-1)
    eye = torch.eye(n, dtype=dt, device=dev)
    # I+L is unit lower triangular for any input, so the solve needs no pivoting.
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
    """Adjoint of :func:`_gdn_chunk_fwd`. Returns
    (dq, dk, dv, dbeta, dgt, dS_start), all in the caller's [B,n,HV,*] layout.
    Gradchecked term by term against autograd on the chunk forward."""
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
    """:func:`gdn_chunk_core` through flash-linear-attention — a MEASUREMENT
    path, not a shipped one.

    fla is Triton and CUDA-only, so it cannot be the backend (AGENTS.md: one
    TileLang source for cpu/cuda/metal). It is here to answer one question
    with a number instead of an estimate: fla runs our GDN shapes in 6.8 ms
    where our scalar-scan kernel takes 63, and this says how much of that 9.3x
    survives the layer's own glue.

    Same signature as the reference so the two are interchangeable at the call
    site. ``chunk`` is fla's chunk_size. fla wants [B,T,H,D] with the KEY heads
    already broadcast to value heads, which is what the serial form does too.
    """
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    rep = v.shape[2] // kn.shape[2]
    q = qn.repeat_interleave(rep, dim=2)
    k = kn.repeat_interleave(rep, dim=2)
    # qn already carries the 1/sqrt(key_dim) factor the serial form folds in, so
    # fla's own scale must be 1.
    o, s = chunk_gated_delta_rule(
        q=q.bfloat16(), k=k.bfloat16(), v=v.bfloat16(), g=gt.float(),
        beta=bt.bfloat16(), scale=1.0, initial_state=state.float(),
        output_final_state=True, use_qk_l2norm_in_kernel=False,
    )
    return o.float(), s.float()


def gdn_chunk_core(qn, kn, v, gt, bt, state, chunk: int = 64):
    """The gated-delta core as a chunkwise-WY decomposition, not a serial scan.

    Same recurrence as the loop in :func:`gdn_forward` — decay, delta, read out
    — reassociated so a chunk's tokens are one matmul instead of C sequential
    steps. The intra-chunk term the serial form carries in the state lives in
    ``M = (I + L)^-1``; freezing the chunk-start state loses nothing.

    ``qn``/``kn`` [B,T,HK,DK] (already L2-normed), ``v`` [B,T,HV,DV],
    ``gt``/``bt`` [B,T,HV] (log-decay <= 0, and the delta rate), ``state``
    [B,HV,DK,DV]. HV is a multiple of HK: value head h reads key head
    ``h * HK // HV``. Returns (core [B,T,HV,DV], final state).

    This is the executable spec for the five-kernel prefill pipeline
    (cumsum / scaled-dot-kkt / solve-tril / wy / delta-h / chunk-o).
    """
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
    conv_window: "torch.Tensor | None" = None,
    seq_q_lens: "torch.Tensor | None" = None,
    keep_steps: int = 0,
    chunkwise: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, "torch.Tensor | None"]:
    """Full gated-delta layer core: the executable spec for the model's GDN
    layer, mirroring agent-infer's host reference equation by equation (the
    op contract is documented in ``model.py``). ``q``/``k`` [B,T,nkh*K],
    ``v``/``z`` [B,T,nvh*V], ``g``/``beta`` [B,T,nvh], ``state`` [B,nvh,K,V].
    ``conv_window`` [B,K-1,qkv_dim] is the previous segment's last raw-qkv
    tokens, prepended so segmented decode (T=1 per forward) is exact; None =
    one-shot prefill (zero left-padding) and the third return is None.
    ``seq_q_lens`` [B] bounds the per-row scan: mixed batches pad rows to a
    shared T and only the first ``seq_q_lens[b]`` positions are real (decode
    rows: 1); None means every row is valid for all T.
    ``keep_steps`` > 0 (speculative verify): the returned state and window
    carry a leading chain-step axis — the state after EACH of the first
    ``keep_steps`` tokens ([B,KS,nvh,K,V] / [B,KS,K-1,qkv_dim]) — so the engine
    selects the accepted prefix's state without a second forward."""
    # Activations arrive on the backend device; params/state may be CPU-resident
    # (day-1: params live on CPU, the backend boundary migrates activations).
    # Gather every input on the activation device.
    # ponytail: per-call param migration, keep params on the backend device at load
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
    b, t, _ = q.shape
    nvh, key_dim, val_dim = state.shape[1], state.shape[2], state.shape[3]
    nkh = q.shape[-1] // key_dim
    kernel = conv1d_weight.shape[1]
    if seq_q_lens is None:
        seq_q_lens = torch.full((b,), t, dtype=torch.long, device=dev)
    else:
        seq_q_lens = torch.as_tensor(seq_q_lens, dtype=torch.long, device=dev).reshape(b)
    qkv = torch.cat([q, k, v], dim=-1)
    new_window = None
    if conv_window is not None:
        # Segmented decode: carry the last K-1 raw-qkv tokens. Per row, the new
        # window is the last K-1 of (window + that row's valid qkv) — mixed
        # rows have different valid lengths, so build per row.
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
    exp_g = torch.exp(gt)
    if chunkwise and not keep_steps and int(seq_q_lens.min()) == t:
        # Every row full-length: the chunked form has no per-row valid mask, and
        # a ragged batch would silently absorb padding into the recurrence.
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


#: The op name the model/backend contract uses for :func:`gdn_forward`.
linear_attn_chunk = gdn_forward


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
    conv_window: "torch.Tensor | None" = None,
) -> tuple[torch.Tensor, ...]:
    """Backward of :func:`gdn_forward`: (gq,gk,gv,gg,gbeta,gstate,gz,gconv1d,
    gdt_bias,ga_log,gnorm_weight). Gradchecked against torch.autograd (worst
    rel ~3e-7); the recurrence derivation is documented inline.
    ``conv_window`` is accepted for tape-signature parity and ignored: training
    forwards start from a zeroed window, so zero-padded recompute is exact. A
    non-zero window raises — ignoring one silently costs every grad (measured
    0.69-2.27 rel vs autograd), and only the zero case is exact.
    # ponytail: torch-eager backward, tilelang kernel when perf demands."""
    if conv_window is not None and bool(conv_window.any()):
        raise NotImplementedError("gdn_backward does not support a non-zero conv_window")
    # Same device gathering as gdn_forward: the tape replays raw saved args,
    # so params/state may be CPU-resident while grad/activations are on device.
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
    assert nkh * rep == nvh, (nkh, nvh)  # h -> h // rep, contiguous groups
    knv = kn.repeat_interleave(rep, dim=2)  # [b,t,nvh,key_dim]
    qnv = qn.repeat_interleave(rep, dim=2)
    # Chunked, not a per-step scan: the sequential dimension is t/CHUNK, and
    # only chunk-START states are kept (the interior is recomputed from one in
    # the reverse pass). The per-step form cost ~28 launches per step per layer
    # — 434K micro-ops in one 27B step, 62% of it
    # (errors/2026-08-29-train-step-is-the-gdn-per-step-loop.md).
    starts = list(range(0, t, _GDN_CHUNK))
    caches = []
    s = state.clone()
    core = torch.zeros(b, t, nvh, val_dim, dtype=torch.float32, device=q.device)
    for c0 in starts:
        sl = slice(c0, min(c0 + _GDN_CHUNK, t))
        # Keep each chunk's intermediates instead of recomputing them in the
        # reverse pass: one layer's worth is ~40 MB at CHUNK=16 and it is freed
        # when this backward returns, against a second forward per chunk.
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

    # Recurrence reverse scan, chunked: walk the chunks backwards over the
    # intermediates the forward pass kept, threading dS through.
    # _gdn_chunk_bwd is the adjoint of _gdn_chunk_fwd and gives dgt directly,
    # so there is no d/d(exp_g) intermediate.
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
    # The value-head grads fold back onto the key heads (h -> h // rep is
    # contiguous, so a reshape and a sum is the scatter-add).
    g_qn = g_qnv.reshape(b, t, nkh, rep, key_dim).sum(3)
    g_kn = g_knv.reshape(b, t, nkh, rep, key_dim).sum(3)
    g_a_log = (g_gt * gt).sum(dim=(0, 1))
    g_sp_in = g_gt * (-torch.exp(a_log)) * torch.sigmoid(sp_in)
    g_g = g_sp_in.reshape(b, t, nvh)
    g_dt_bias = g_sp_in.sum(dim=(0, 1))
    g_beta = (g_bt * bt * (1.0 - bt)).reshape(b, t, nvh)

    def _norm_bwd(g_in, x, r, s):
        # y = x * r * s, r = rsqrt(sum(x^2)+eps)
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
    """Stable shifted causal CE and its matching logit gradient."""
    b, t, v = logits.shape
    if t < 2:
        raise ValueError("cross_entropy_loss_grad needs at least two tokens")
    flat = _f32(logits[:, :-1]).reshape(-1, v)
    labels = torch.as_tensor(input_ids, dtype=torch.long, device=flat.device)[:, 1:].reshape(-1)
    loss = (torch.logsumexp(flat, dim=-1) - flat.gather(-1, labels[:, None]).squeeze(-1)).mean()
    grad = torch.softmax(flat, dim=-1)
    grad.scatter_add_(-1, labels[:, None], -torch.ones_like(labels[:, None], dtype=grad.dtype))
    grad /= flat.shape[0]
    out = torch.zeros(b, t, v, dtype=torch.float32, device=logits.device)
    out[:, :-1] = grad.reshape(b, t - 1, v)
    return float(loss), out


def state_gather(states, windows, slots, layer_idx, parity=None):
    """Gather one recurrent-state layer for a batch of slots. ``windows`` is
    the double-buffered pool plane set [S, L, 2, W, D]; ``parity`` [S] picks
    the live plane (all zeros off the sm90 decode path)."""
    slots = torch.as_tensor(slots, dtype=torch.long, device=states.device).reshape(-1)
    if windows is None:
        return states[slots, layer_idx], None
    par = torch.zeros_like(slots) if parity is None else parity[slots].long()
    return states[slots, layer_idx], windows[slots, layer_idx, par]


def state_scatter(
    states, windows, slots, layer_idx, new_state, new_window, parity=None, steps=False
) -> None:
    """Store one recurrent-state layer for a batch of slots (same plane it was
    read from). ``steps``: the tensors carry a chain-step axis (speculative
    verify) and land in the leading planes of the pool's step buffers — the
    tick's chain width, which is <= the pool's spec_steps."""
    slots = torch.as_tensor(slots, dtype=torch.long, device=states.device).reshape(-1)
    if steps:
        ks = new_state.shape[1]
        states[slots, layer_idx, :ks] = new_state.to(states.dtype)
        if new_window is not None:
            windows[slots, layer_idx, :ks] = new_window.to(windows.dtype)
        return
    states[slots, layer_idx] = new_state.to(states.dtype)
    if new_window is not None:
        par = torch.zeros_like(slots) if parity is None else parity[slots].long()
        windows[slots, layer_idx, par] = new_window.to(windows.dtype)


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


def sample(logits: torch.Tensor, temperature: float, top_p: float, seed: int) -> torch.Tensor:
    """Top-p nucleus sampling. logits [B, V] -> long [B]. Deterministic per seed.

    temperature <= 0 is greedy (argmax).
    """
    logits = _f32(logits)
    if temperature <= 0:
        return logits.argmax(-1).to(torch.long)
    gen = torch.Generator(device=logits.device).manual_seed(int(seed))
    if temperature != 1.0:
        logits = logits / temperature
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    # keep the minimal prefix whose cumulative mass >= top_p
    keep = cum - probs < top_p
    keep[..., 0] = True  # top_p <= 0 degenerates to greedy: always keep the argmax
    probs = probs * keep
    probs = probs / probs.sum(-1, keepdim=True)
    sampled = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)
    return sorted_idx.gather(-1, sampled.unsqueeze(-1)).squeeze(-1).to(torch.long)


def sample_batch(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ps: torch.Tensor,
    seeds: torch.Tensor,
) -> torch.Tensor:
    """Batched top-p: one sort/softmax over the whole batch, per-row
    multinomial with a fresh per-row generator — identical draws to
    :func:`sample` for the same (logits, temperature, top_p, seed) row.

    The win is the sort: B argmax/sort-over-V calls become one batched op
    (8.2% of the B=8 slice tick was 8 separate sorts + 8 D2H syncs).

    ``temperatures`` / ``top_ps`` / ``seeds`` are read on the HOST — the caller
    has them as Python scalars. Taking them as device tensors and reading them
    back to pick the greedy/sampled split cost two syncs a tick plus one per
    sampled row, on every target.
    """
    logits = _f32(logits)
    b = logits.shape[0]
    dev = logits.device
    temps = [float(t) for t in temperatures]
    out = torch.empty(b, dtype=torch.long, device=dev)
    hot = [i for i, t in enumerate(temps) if t > 0]
    cold = [i for i, t in enumerate(temps) if t <= 0]
    if cold:
        ci = torch.tensor(cold, device=dev)
        out[ci] = logits[ci].argmax(-1).to(torch.long)
    if hot:
        idx = torch.tensor(hot, device=dev)
        tt = torch.tensor([temps[i] for i in hot], dtype=torch.float32, device=dev)
        tp = torch.tensor([float(top_ps[i]) for i in hot], dtype=torch.float32, device=dev)
        sub = logits[idx] / tt[:, None]
        sorted_logits, sorted_idx = torch.sort(sub, dim=-1, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        keep = cum - probs < tp[:, None]
        keep[:, 0] = True
        probs = probs * keep
        probs = probs / probs.sum(-1, keepdim=True)
        for k, i in enumerate(hot):
            gen = torch.Generator(device=dev).manual_seed(int(seeds[i]))
            draw = torch.multinomial(probs[k], num_samples=1, generator=gen)
            out[i] = sorted_idx[k, draw].to(torch.long)
    return out


def _check_chunk_core() -> None:
    """chunkwise-WY must equal the serial decay-first scan it replaces.

    A 3:1 key/value head ratio and a T that is not chunk-divisible, because
    those are the two shapes the upstream kernels cannot express as-is.
    """
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
    # e4m3 keeps 3 mantissa bits, so ~6% per element near a block's absmax; the
    # gate that matters is the served model's accuracy, not this bound.
    _rel = (dequant_fp8(_q, _s) - _w.float()).abs().max() / _w.float().abs().max()
    assert _rel < 0.06, _rel
    print("reference: quant_fp8 round-trip OK, rel", float(_rel))
    _check_chunk_core()
