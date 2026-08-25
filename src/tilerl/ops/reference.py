"""Torch-eager reference implementations of every tilerl op (forward + backward).

This module is the parity oracle for the TileLang kernels in :mod:`tilerl.ops.kernels`
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

import torch

__all__ = [
    "rmsnorm",
    "rmsnorm_bwd",
    "rope",
    "rope_bwd",
    "linear",
    "linear_bwd",
    "dequant_fp4",
    "linear_fp4",
    "linear_fp4_bwd",
    "pack_fp4",
    "unpack_fp4",
    "dequant_nvfp4",
    "dequant_fp8",
    "linear_fp8",
    "linear_fp8_bwd",
    "dequant_awq",
    "dense_attention",
    "dense_attention_bwd",
    "linear_attn_chunk",
    "linear_attn_step",
    "linear_attn_bwd",
    "gdn_forward",
    "gdn_backward",
    "silu_mul",
    "silu_mul_bwd",
    "softmax",
    "embedding",
    "embedding_bwd",
    "sample",
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
    x1 = x_rot[..., 0::2]
    x2 = x_rot[..., 1::2]
    out = torch.empty_like(x_rot)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
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


def dequant_fp4(wq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize e2m1fn-packed weights. wq uint8 [N, K//2] (low nibble first),
    scale uint8 [N, K//32] (e4m3fn bytes, tileRL's packed scale format).
    Returns w [N, K] f32.

    e2m1fn magnitudes: e=0 -> {0.5, 0.75}, e=1 -> {1, 1.5}, e=2 -> {2, 3},
    e=3 -> {4, 6}; sign bit is bit 3.
    """
    assert wq.dtype == torch.uint8
    n, k2 = wq.shape
    scale = _e4m3_f32(scale)
    lo = wq & 0x0F
    hi = (wq >> 4) & 0x0F

    def decode(nib: torch.Tensor) -> torch.Tensor:
        sign = torch.where(nib & 0x08 == 0, 1.0, -1.0)
        e = ((nib >> 1) & 0x03).float()
        m = (nib & 0x01).float()
        return sign * (0.5 * torch.pow(2.0, e)) * (1.0 + 0.5 * m)

    s = scale.repeat_interleave(16, dim=1)  # [N, K//2]: one scale per byte (32 elems / 2)
    w = torch.stack([decode(lo) * s, decode(hi) * s], dim=-1).reshape(n, k2 * 2)
    return w


def linear_fp4(x: torch.Tensor, wq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """y = x @ dequant(wq, scale).T.  x [..., K]."""
    x = _f32(x)
    w = dequant_fp4(wq, scale)
    return x @ w.t()


def linear_fp4_bwd(
    grad: torch.Tensor, x: torch.Tensor, wq: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward of :func:`linear_fp4`.

    Returns (gx, g_master): gx w.r.t. x, and g_master w.r.t. the dequantized
    (bf16-master) weight — a straight-through estimator through the e2m1
    quantization, matching the model's master-weight convention.
    """
    x = _f32(x)
    grad = _f32(grad)
    w = dequant_fp4(wq, scale)
    gx = grad @ w
    g_master = grad.reshape(-1, grad.shape[-1]).t() @ x.reshape(-1, x.shape[-1])
    return gx, g_master


# ---------------------------------------------------------------- fp4 packing

#: e2m1fn (finite-number, no zero) magnitude LUT, low 3 bits of a nibble;
#: bit 3 = sign. tileRL's internal fp4 format: pack_fp4/unpack_fp4,
#: dequant_fp4, and the linear_fp4 kernel all decode this grid. It matches
#: the Hopper dequant+gemm SOTA kernel's decode (e=0 -> {0.5, 0.75}), so the
#: MMA port in kernels_mma.py is a clean copy with no grid adaptation.
_E2M1FN_LUT = torch.tensor([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

#: OCP/MX e2m1 magnitude LUT (with zero): the NVFP4 checkpoint wire format,
#: used only by dequant_nvfp4. A different grid from _E2M1FN_LUT above —
#: the checkpoint is OCP, tileRL's internal pack is e2m1fn.
_E2M1_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _e4m3_f32(scale: torch.Tensor) -> torch.Tensor:
    """Decode tileRL's packed fp4 block scale: uint8 bytes holding e4m3fn
    bits -> f32. The kernels decode the same bit pattern in-register
    (kernels.py _e4m3_fp32); this is the torch-side mirror."""
    return scale.view(torch.float8_e4m3fn).float()


def pack_fp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a bf16/f32 weight [N,K] into e2m1fn nibbles + per-32-block scales.

    Returns ``(wq [N,K//2] uint8 low-nibble-first, scale [N,K//32] uint8
    e4m3fn bytes)`` with ``scale = block_max / 6`` (rounded to e4m3) and
    round-to-nearest against the e2m1fn LUT. The max representable magnitude
    is 6*scale, so the block max maps exactly. Block 32 matches the fp8 WGMMA
    K-tile (sm90), so the fp8 prefill path can apply one scale per MMA tile
    (no e4m3 weight requant). e4m3 scales are the checkpoint's native scale
    dtype: 4x less scale traffic than f32 (~15% of decode weight traffic).
    """
    assert w.dim() == 2, f"pack_fp4 expects a 2D weight, got {tuple(w.shape)}"
    n, k = w.shape
    assert k % 32 == 0, f"fp4 block size 32 must divide K, got K={k}"
    wf = w.detach().float()
    blocks = wf.reshape(n, k // 32, 32)
    block_max = blocks.abs().amax(dim=-1, keepdim=True)  # [n, k//32, 1]
    scale = (block_max / 6.0).clamp_min(1e-12)
    x = (blocks / scale).clamp(-6.0, 6.0)
    lut = _E2M1FN_LUT.to(wf.device)
    dist = (x.abs().unsqueeze(-1) - lut).abs()  # [n, k//32, 32, 8]
    idx = dist.argmin(dim=-1).to(torch.uint8)  # 0..7
    sign = (x < 0).to(torch.uint8)
    nibbles = (idx | (sign << 3)).reshape(n, k)
    wq = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    scale_e4m3 = scale.squeeze(-1).to(torch.float8_e4m3fn).view(torch.uint8)
    return wq.contiguous(), scale_e4m3.contiguous()


def unpack_fp4(wq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_fp4`; returns a bf16 [N,K] weight."""
    n, k2 = wq.shape
    k = k2 * 2
    lo = (wq & 0xF).to(torch.uint8)
    hi = (wq >> 4).to(torch.uint8)
    nibbles = torch.stack([lo, hi], dim=-1).reshape(n, k).long()
    lut = _E2M1FN_LUT.to(wq.device)
    mag = lut[nibbles & 0x7]
    sign = 1.0 - 2.0 * (nibbles >> 3).to(torch.float32)
    vals = mag * sign  # [n, k]
    out = vals.reshape(n, k // 32, 32) * _e4m3_f32(scale).unsqueeze(-1)
    return out.reshape(n, k).to(torch.bfloat16)


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
    grid is ``_E2M1_LUT`` ({0,.5,1,1.5,2,3,4,6} + sign) — the OCP/MX grid,
    not the e2m1fn grid of :func:`dequant_fp4`."""
    n, k2 = weight_packed.shape
    lo = (weight_packed & 0xF).long()
    hi = ((weight_packed >> 4) & 0xF).long()
    lut = _E2M1_LUT.to(weight_packed.device)
    mag = torch.stack([lut[lo & 0x7], lut[hi & 0x7]], dim=-1).reshape(n, k2 * 2)
    sign = torch.stack(
        [1.0 - 2.0 * (lo >> 3).float(), 1.0 - 2.0 * (hi >> 3).float()], dim=-1
    ).reshape(n, k2 * 2)
    scale = weight_scale.float().repeat_interleave(16, dim=-1)
    gs = weight_global_scale.float()
    if global_divide:
        gs = 1.0 / gs
    return (mag * sign * scale * gs).to(torch.bfloat16)


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


def linear_fp8(x: torch.Tensor, w8: torch.Tensor, wscale: torch.Tensor) -> torch.Tensor:
    """y = x @ dequant_fp8(w8, wscale).T.  x [..., K]."""
    return _f32(x) @ dequant_fp8(w8, wscale).t()


def linear_fp8_bwd(
    grad: torch.Tensor, x: torch.Tensor, w8: torch.Tensor, wscale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward of :func:`linear_fp8` (STE, same convention as linear_fp4_bwd:
    gx w.r.t. x, g_master w.r.t. the bf16 master weight)."""
    x = _f32(x)
    grad = _f32(grad)
    w = dequant_fp8(w8, wscale)
    gx = grad @ w
    g_master = grad.reshape(-1, grad.shape[-1]).t() @ x.reshape(-1, x.shape[-1])
    return gx, g_master


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


# ---------------------------------------------------------------- gated delta (linear attention)


def _gated_delta_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain gated-delta recurrence (the 6-arg op the TileLang kernel mirrors).

    q, k, v [B, C, H, D] f32; g [B, C, H] f32 (per-head decay, already
    exponentiated); beta [B, C, H] f32 (per-head gate, already sigmoided);
    state [B, H, D, D] f32 with layout S[key, value].

    Per (b, h), serial over t (A_t = g_t * S):
        p_t = A_t^T k_t;  d_t = beta_t * (v_t - p_t)
        S = A_t + outer(k_t, d_t);  out_t = S^T q_t
    """
    q = _f32(q)
    k = _f32(k)
    v = _f32(v)
    g = _f32(g)
    beta = _f32(beta)
    state = _f32(state)
    b, c, h, d = q.shape
    S = state.clone()
    out = torch.empty(b, c, h, d, dtype=torch.float32, device=q.device)
    for t in range(c):
        S = S * g[:, t].view(b, h, 1, 1)
        p = torch.einsum("bhij,bhi->bhj", S, k[:, t])  # [B,H,D]
        dlt = beta[:, t].unsqueeze(-1) * (v[:, t] - p)
        S = S + torch.einsum("bhi,bhj->bhij", k[:, t], dlt)
        out[:, t] = torch.einsum("bhij,bhi->bhj", S, q[:, t])
    return out, S


def _gated_delta_bwd(
    grad: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Backward of :func:`_gated_delta_scan`. Returns (gq,gk,gv,gg,gbeta,gstate).

    Per (b, h), S_t the state after step t, S_{-1} = state input,
    A_t = g_t S_{t-1} (the decayed state, with which the forward computes p):
        p_t = A_t^T k_t;  d_t = beta_t (v_t - p_t)
        S_t = A_t + outer(k_t, d_t);  o_t = S_t^T q_t
    Reverse scan with dS = dL/dS_t, gd = dS^T k_t:
        dS_  = dS - beta_t outer(k_t, gd)   (= dL/dA_t)
        gg_t    = sum_ij dS_ij S_{t-1,ij}
        gbeta_t = sum_j gd_j (v_t - p_t)_j
        gv_t    = beta_t gd
        gk_t    = dS d_t - beta_t A_t gd
        dS      = g_t dS_
    """
    q = _f32(q)
    k = _f32(k)
    v = _f32(v)
    g = _f32(g)
    beta = _f32(beta)
    state = _f32(state)
    grad = _f32(grad)
    b, c, h, d = q.shape

    # Forward pass, saving p/d/states. Order must match _gated_delta_scan:
    # decay first, then p = A^T k, then the delta and the outer-product update.
    S = state.clone()
    states = torch.empty(b, c + 1, h, d, d, dtype=torch.float32, device=q.device)
    ps = torch.empty(b, c, h, d, dtype=torch.float32, device=q.device)
    deltas = torch.empty(b, c, h, d, dtype=torch.float32, device=q.device)
    states[:, 0] = S
    for t in range(c):
        S = S * g[:, t].view(b, h, 1, 1)
        p = torch.einsum("bhij,bhi->bhj", S, k[:, t])
        dlt = beta[:, t].unsqueeze(-1) * (v[:, t] - p)
        S = S + torch.einsum("bhi,bhj->bhij", k[:, t], dlt)
        states[:, t + 1] = S
        ps[:, t] = p
        deltas[:, t] = dlt

    gq = torch.empty_like(q)
    gk = torch.empty_like(k)
    gv = torch.empty_like(v)
    gg = torch.empty_like(g)
    gbeta = torch.empty_like(beta)
    dS = torch.zeros_like(state)
    for t in reversed(range(c)):
        S_prev = states[:, t]
        dS = dS + torch.einsum("bhi,bhj->bhij", q[:, t], grad[:, t])
        gd = torch.einsum("bhij,bhi->bhj", dS, k[:, t])  # [B,H,D]
        gq[:, t] = torch.einsum("bhij,bhj->bhi", states[:, t + 1], grad[:, t])
        A = g[:, t].view(b, h, 1, 1) * S_prev
        dS_ = dS - beta[:, t].view(b, h, 1, 1) * torch.einsum("bhi,bhj->bhij", k[:, t], gd)
        gg[:, t] = torch.einsum("bhij,bhij->bh", dS_, S_prev)
        gbeta[:, t] = (gd * (v[:, t] - ps[:, t])).sum(-1)
        gv[:, t] = beta[:, t].unsqueeze(-1) * gd
        gk[:, t] = torch.einsum("bhij,bhj->bhi", dS, deltas[:, t]) - beta[:, t].unsqueeze(
            -1
        ) * torch.einsum("bhij,bhj->bhi", A, gd)
        dS = g[:, t].view(b, h, 1, 1) * dS_
    return gq, gk, gv, gg, gbeta, dS


def linear_attn_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    *,
    z: "torch.Tensor | None" = None,
    conv1d_weight: "torch.Tensor | None" = None,
    dt_bias: "torch.Tensor | None" = None,
    a_log: "torch.Tensor | None" = None,
    norm_weight: "torch.Tensor | None" = None,
    conv_window: "torch.Tensor | None" = None,
) -> tuple[torch.Tensor, ...]:
    """Gated-delta linear attention. 6-arg form (``z``/``conv1d_weight``/...
    all None): the plain :func:`_gated_delta_scan` — the op the TileLang
    kernel mirrors and the parity/gradcheck tests exercise. Full-GDN form
    (kwargs present): the complete layer core :func:`gdn_forward` — the form
    :class:`tilerl.model.Model` calls; its backward is :func:`gdn_backward`.
    Returns ``(out, new_state, new_window)`` (window is None in the 6-arg
    form and for one-shot prefill).
    # ponytail: full-GDN forward is torch-eager (the TileLang chunk kernel
    # covers the plain scan); a fused GDN kernel is the perf upgrade path."""
    if z is None and conv1d_weight is None:
        out, new_state = _gated_delta_scan(q, k, v, g, beta, state)
        return out, new_state, None
    return gdn_forward(
        q,
        k,
        v,
        g,
        beta,
        state,
        z=z,
        conv1d_weight=conv1d_weight,
        dt_bias=dt_bias,
        a_log=a_log,
        norm_weight=norm_weight,
        conv_window=conv_window,
    )


def linear_attn_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    **kw: Any,
) -> tuple[torch.Tensor, ...]:
    """Single-token :func:`linear_attn_chunk`. q, k, v [B, H, D], g/beta [B, H]."""
    out, S, window = linear_attn_chunk(
        q.unsqueeze(1),
        k.unsqueeze(1),
        v.unsqueeze(1),
        g.unsqueeze(1),
        beta.unsqueeze(1),
        state,
        **kw,
    )
    return out.squeeze(1), S, window


def linear_attn_bwd(
    grad: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    *,
    z: "torch.Tensor | None" = None,
    conv1d_weight: "torch.Tensor | None" = None,
    dt_bias: "torch.Tensor | None" = None,
    a_log: "torch.Tensor | None" = None,
    norm_weight: "torch.Tensor | None" = None,
    conv_window: "torch.Tensor | None" = None,
) -> tuple[torch.Tensor, ...]:
    """Backward of :func:`linear_attn_chunk`. 6-arg form:
    :func:`_gated_delta_bwd` (6 grads). Full-GDN form: :func:`gdn_backward`
    (11 grads: q,k,v,g,beta,state,z,conv1d,dt_bias,a_log,norm)."""
    if z is None and conv1d_weight is None:
        return _gated_delta_bwd(grad, q, k, v, g, beta, state)
    return gdn_backward(
        grad,
        q,
        k,
        v,
        g,
        beta,
        state,
        z=z,
        conv1d_weight=conv1d_weight,
        dt_bias=dt_bias,
        a_log=a_log,
        norm_weight=norm_weight,
        conv_window=conv_window,
    )


# ---------------------------------------------------------------- full GDN layer core


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
) -> tuple[torch.Tensor, torch.Tensor, "torch.Tensor | None"]:
    """Full gated-delta layer core: the executable spec for the model's GDN
    layer, mirroring agent-infer's host reference equation by equation (the
    op contract is documented in ``model.py``). ``q``/``k`` [B,T,nkh*K],
    ``v``/``z`` [B,T,nvh*V], ``g``/``beta`` [B,T,nvh], ``state`` [B,nvh,K,V].
    ``conv_window`` [B,K-1,qkv_dim] is the previous segment's last raw-qkv
    tokens, prepended so segmented decode (T=1 per forward) is exact; None =
    one-shot prefill (zero left-padding) and the third return is None."""
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
    qkv = torch.cat([q, k, v], dim=-1)
    new_window = None
    if conv_window is not None:
        # Segmented decode: carry the last K-1 raw-qkv tokens. The new window
        # is the last K-1 of (window + qkv); outputs are the last T positions.
        qkv = torch.cat([_f32(conv_window).to(dev), qkv], dim=1)
        new_window = qkv[:, -(kernel - 1) :, :].contiguous()
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
    s_heads = [state[:, h].clone() for h in range(nvh)]
    core = torch.zeros(b, t, nvh, val_dim, dtype=torch.float32, device=q.device)
    for step in range(t):
        for h in range(nvh):
            kh = h * nkh // nvh
            s_h = s_heads[h] * exp_g[:, step, h].view(b, 1, 1)
            p = torch.einsum("bk,bkv->bv", kn[:, step, kh], s_h)
            d = (v_raw[:, step, h] - p) * bt[:, step, h].unsqueeze(-1)
            s_h = s_h + kn[:, step, kh].unsqueeze(-1) * d.unsqueeze(-2)
            s_heads[h] = s_h
            core[:, step, h] = torch.einsum("bk,bkv->bv", qn[:, step, kh], s_h)
    s = torch.stack(s_heads, dim=1)
    normed = core * torch.rsqrt(core.pow(2).mean(-1, keepdim=True) + 1e-6)
    normed = normed * norm_weight
    out = normed * silu(z.view(b, t, nvh, val_dim))
    return out.reshape(b, t, nvh * val_dim), s, new_window


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
    forwards start from a zeroed window, so zero-padded recompute is exact.
    # ponytail: torch-eager backward, tilelang kernel when perf demands."""
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
    exp_g = torch.exp(gt)
    s_heads = [state[:, h].clone() for h in range(nvh)]
    states = torch.empty(b, t + 1, nvh, key_dim, val_dim, dtype=torch.float32, device=q.device)
    ps = torch.empty(b, t, nvh, val_dim, dtype=torch.float32, device=q.device)
    deltas = torch.empty(b, t, nvh, val_dim, dtype=torch.float32, device=q.device)
    states[:, 0] = state
    core = torch.zeros(b, t, nvh, val_dim, dtype=torch.float32, device=q.device)
    for step in range(t):
        for h in range(nvh):
            kh = h * nkh // nvh
            s_h = s_heads[h] * exp_g[:, step, h].view(b, 1, 1)
            p = torch.einsum("bk,bkv->bv", kn[:, step, kh], s_h)
            d = (v_raw[:, step, h] - p) * bt[:, step, h].unsqueeze(-1)
            s_h = s_h + kn[:, step, kh].unsqueeze(-1) * d.unsqueeze(-2)
            s_heads[h] = s_h
            core[:, step, h] = torch.einsum("bk,bkv->bv", qn[:, step, kh], s_h)
            ps[:, step, h] = p
            deltas[:, step, h] = d
            states[:, step + 1, h] = s_h
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

    # Recurrence reverse scan. Per (b, h), with S_ = g*S_prev, p = S_^T k,
    # d = beta*(v-p), S_t = S_ + outer(k,d), o = S_t^T q:
    #   dS_t = dS + outer(q, go);  gd = dS_t^T k
    #   dS_  = dS_t - beta*outer(k, gd)
    #   gg   = sum dS_ * S_prev ;  gbeta = gd.(v-p) ;  gv = beta*gd
    #   gk   = dS_t d - beta*S_ gd ;  dS_prev = g * dS_
    dS = torch.zeros_like(state)
    g_qn = torch.zeros(b, t, nkh, key_dim, dtype=torch.float32, device=q.device)
    g_kn = torch.zeros(b, t, nkh, key_dim, dtype=torch.float32, device=q.device)
    g_v_raw = torch.zeros(b, t, nvh, val_dim, dtype=torch.float32, device=q.device)
    g_bt = torch.zeros(b, t, nvh, dtype=torch.float32, device=q.device)
    g_exp_g = torch.zeros(b, t, nvh, dtype=torch.float32, device=q.device)
    for step in reversed(range(t)):
        for h in range(nvh):
            kh = h * nkh // nvh
            dS_t = dS[:, h] + torch.einsum("bk,bv->bkv", qn[:, step, kh], g_core[:, step, h])
            gd = torch.einsum("bkv,bk->bv", dS_t, kn[:, step, kh])
            g_qn[:, step, kh] += torch.einsum(
                "bkv,bv->bk", states[:, step + 1, h], g_core[:, step, h]
            )
            dS_ = dS_t - bt[:, step, h].view(b, 1, 1) * torch.einsum(
                "bk,bv->bkv", kn[:, step, kh], gd
            )
            g_exp_g[:, step, h] = (dS_ * states[:, step, h]).sum(dim=(1, 2))
            g_bt[:, step, h] = (gd * (v_raw[:, step, h] - ps[:, step, h])).sum(dim=-1)
            g_v_raw[:, step, h] = bt[:, step, h].unsqueeze(-1) * gd
            g_kn[:, step, kh] += torch.einsum("bkv,bv->bk", dS_t, deltas[:, step, h]) - bt[
                :, step, h
            ].unsqueeze(-1) * torch.einsum(
                "bkv,bv->bk", exp_g[:, step, h].view(b, 1, 1) * states[:, step, h], gd
            )
            dS[:, h] = exp_g[:, step, h].view(b, 1, 1) * dS_
    g_gt = g_exp_g * exp_g
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
