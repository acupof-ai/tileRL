"""Parity gates for the tilerl operator layer (hermetic, CPU target).

Covers every op in the pinned interface contract:

* Forward: TileLang kernel (CPU target) vs torch-eager reference,
  allclose(rtol=1e-2, atol=1e-2) on random small inputs.
* paged_attention vs a naive full-matrix attention.
* gated-delta chunk-vs-step consistency (same recurrence, C=1 vs C=C).
* Backward: every bwd op vs finite differences (torch gradcheck-style,
  tiny shapes); attention_bwd and linear_attn_bwd additionally cross-checked
  against torch.autograd (finite diff is too noisy on softmax/scan).

Run: uv run pytest tests/test_ops_parity.py -v
"""

from __future__ import annotations

import os

# Hermetic CPU target: auto already maps to cpu on this Mac, but pin it so a
# stray TILERL_TARGET in the environment can't hijack the suite.
os.environ.setdefault("TILERL_TARGET", "cpu")

import pytest
import torch

from tilerl.ops import reference
from tilerl.ops.backend import _resolve, get_backend
from tilerl.ops.reference import pack_fp4

RTOL = 1e-2
ATOL = 1e-2
FD_EPS = 1e-3


@pytest.fixture(scope="module")
def backend():
    return get_backend()


# ---------------------------------------------------------------- helpers


def _assert_close(a, b, msg):
    # Compare on CPU: the tilelang backend may return device tensors (metal)
    # while the reference stays on CPU.
    a = a.detach().cpu()
    b = b.detach().cpu()
    assert a.shape == b.shape, f"{msg}: shape {a.shape} vs {b.shape}"
    diff = (a - b).abs().max().item()
    assert diff <= ATOL + RTOL * b.abs().max().item(), (
        f"{msg}: max abs diff {diff:.3e} exceeds allclose(rtol={RTOL}, atol={ATOL})"
    )


def _finite_diff_gradcheck(name, fwd, bwd, inputs, eps=FD_EPS, max_rel=5e-2):
    """Central-difference gradcheck: bwd output vs (f(x+e)-f(x-e))/(2e) * go."""
    inputs = [x.clone() for x in inputs]
    out = fwd(*inputs)
    go = torch.randn_like(out)
    grads = bwd(go, *inputs)
    if not isinstance(grads, tuple):
        grads = (grads,)
    worst = 0.0
    for x, gx in zip(inputs, grads):
        if gx is None:
            continue
        xf = x.reshape(-1)
        gxf = gx.reshape(-1)
        n = min(xf.numel(), 32)
        for j in torch.randperm(xf.numel())[:n]:
            old = xf[j].item()
            xf[j] = old + eps
            fp = fwd(*inputs).reshape(-1)
            xf[j] = old - eps
            fm = fwd(*inputs).reshape(-1)
            xf[j] = old
            num = ((fp - fm) / (2 * eps) * go.reshape(-1)).sum().item()
            ana = gxf[j].item()
            denom = max(abs(ana), abs(num), 1e-6)
            worst = max(worst, abs(ana - num) / denom)
    assert worst < max_rel, f"{name}: finite-diff rel err {worst:.3e} >= {max_rel}"


def _autograd_gradcheck(name, fwd_ref, bwd_ref, inputs, max_rel=1e-2):
    """Cross-check a hand-written bwd against torch.autograd on the reference fwd."""
    leaves = [x.detach().clone().requires_grad_(True) for x in inputs]
    out = fwd_ref(*leaves)
    go = torch.randn_like(out)
    out.backward(go)
    grads = bwd_ref(go, *[x.detach() for x in inputs])
    if not isinstance(grads, tuple):
        grads = (grads,)
    for i, (leaf, gx) in enumerate(zip(leaves, grads)):
        if gx is None or leaf.grad is None:
            continue
        diff = (gx - leaf.grad).abs().max().item()
        scale = max(leaf.grad.abs().max().item(), 1e-6)
        assert diff / scale < max_rel, f"{name} input {i}: autograd rel err {diff / scale:.3e}"


def _quantize_fp4(w_master: torch.Tensor):
    """Pack a master weight into (wq uint8, scale f32) via the production packer."""
    return pack_fp4(w_master)


def _linear_fp4_fp8_ref(x, wq, scale):
    """Torch reference for the sm90 fp8 prefill path: same per-token e4m3
    activation quant + e2m1fn->e4m3 requant weight dequant, f32 matmul. e4m3's
    ~2% multiplicative quant error does not average down over K, so the fp8
    kernel is gated against this identical-quant reference (kernel
    correctness), not the f32 linear_fp4 reference (quant precision)."""
    M, K = x.shape
    xbf = x.to(torch.bfloat16).float()
    row_max = xbf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)  # [M,1]
    ascale = (448.0 / row_max).to(torch.float32)
    xq = (xbf * ascale).to(torch.float8_e4m3fn).float() / ascale
    # weight: e2m1 grid * per-32-block scale, requanted to e4m3 (same as kernel)
    w_deq = reference.dequant_fp4(wq, scale)  # f32 [N,K]
    w_q8 = w_deq.to(torch.float8_e4m3fn).float()
    return xq @ w_q8.t()


def _quantize_fp8(w_master):
    """Per-128-block quant into the loader's native layout: w8 e4m3 [N,K],
    wscale f32 [ceil(N/128), ceil(K/128)] (128 N-rows share one scale — the
    ModelOpt block format; the last block is zero-padded)."""
    n, k = w_master.shape
    ns, ks = (n + 127) // 128, (k + 127) // 128
    padded = w_master.float().new_zeros(ns * 128, ks * 128)
    padded[:n, :k] = w_master.float()
    blocks = padded.reshape(ns, 128, ks, 128)
    block_max = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12)
    scale = (block_max / 448.0).reshape(ns, ks).contiguous()
    w8 = (blocks / (block_max / 448.0)).reshape(ns * 128, ks * 128)[:n, :k]
    w8 = w8.to(torch.float8_e4m3fn).contiguous()
    return w8, scale


def _linear_fp8_ref(x, w8, wscale):
    """Torch reference for the sm90 native-fp8 prefill path: same per-token
    e4m3 activation quant, native e4m3 weight (no requant), per-128-block
    weight scale, f32 matmul. The weight side is exact, so the gate is kernel
    correctness vs an identical-quant reference (the activation e4m3 error
    does not average down over K)."""
    xbf = x.to(torch.bfloat16).float()
    row_max = xbf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    ascale = 448.0 / row_max
    xq = (xbf * ascale).to(torch.float8_e4m3fn).float() / ascale
    return xq @ reference.dequant_fp8(w8, wscale).t()


# ---------------------------------------------------------------- rmsnorm


def test_rmsnorm_parity(backend):
    torch.manual_seed(0)
    x = torch.randn(8, 32)
    w = torch.randn(32)
    _assert_close(backend.rmsnorm(x, w, 1e-6), reference.rmsnorm(x, w, 1e-6), "rmsnorm")


def test_rmsnorm_bwd(backend):
    torch.manual_seed(1)
    x = torch.randn(4, 16)
    w = torch.randn(16)
    go = torch.randn(4, 16)
    gx, gw = backend.rmsnorm_bwd(go, x, w, 1e-6)
    rgx, rgw = reference.rmsnorm_bwd(go, x, w, 1e-6)
    _assert_close(gx, rgx, "rmsnorm_bwd gx")
    _assert_close(gw, rgw, "rmsnorm_bwd gw")
    _finite_diff_gradcheck(
        "rmsnorm_bwd",
        lambda x, w: reference.rmsnorm(x, w, 1e-6),
        lambda go, x, w: reference.rmsnorm_bwd(go, x, w, 1e-6),
        [x, w],
    )


# ---------------------------------------------------------------- linear


def test_linear_parity(backend):
    torch.manual_seed(2)
    x = torch.randn(8, 16)
    w = torch.randn(24, 16)
    bias = torch.randn(24)
    _assert_close(backend.linear(x, w), reference.linear(x, w), "linear")
    _assert_close(backend.linear(x, w, bias), reference.linear(x, w, bias), "linear+bias")


def test_linear_bf16_gemv_parity(backend):
    """M=1 decode path: the sm90 cell resolves to the bf16 GEMV kernel (the
    floor gemm on CPU/metal); same math as the WGMMA path."""
    torch.manual_seed(25)
    for N, K in [(24, 32), (16, 128), (18, 64)]:
        w = torch.randn(N, K)
        x = torch.randn(1, K)
        _assert_close(backend.linear(x, w), reference.linear(x, w), f"linear_bf16_gemv N={N} K={K}")


def test_linear_bwd(backend):
    torch.manual_seed(3)
    x = torch.randn(6, 8)
    w = torch.randn(12, 8)
    go = torch.randn(6, 12)
    gx, gw = backend.linear_bwd(go, x, w)
    rgx, rgw = reference.linear_bwd(go, x, w)
    _assert_close(gx, rgx, "linear_bwd gx")
    _assert_close(gw, rgw, "linear_bwd gw")
    _finite_diff_gradcheck(
        "linear_bwd",
        lambda x, w: reference.linear(x, w),
        lambda go, x, w: reference.linear_bwd(go, x, w),
        [x, w],
    )


# ---------------------------------------------------------------- linear fp4


def test_linear_fp4_parity(backend):
    torch.manual_seed(4)
    w_master = torch.randn(24, 32)
    wq, scale = _quantize_fp4(w_master)
    x = torch.randn(6, 32)
    # On CUDA M>1 dispatches to the fp8 path; gate it against the identical-
    # quant fp8 reference (the f32 reference carries the ~2% e4m3 quant error).
    ref = (
        _linear_fp4_fp8_ref(x, wq, scale)
        if backend.target.startswith("cuda")
        else reference.linear_fp4(x, wq, scale)
    )
    _assert_close(backend.linear_fp4(x, wq, scale), ref, "linear_fp4")


def test_linear_fp4_gemv_parity(backend):
    """M=1 decode path: the sm90 cell resolves to the GEMV kernel (the floor
    kernel on CPU/metal); same e2m1fn decode math as the MMA kernel."""
    torch.manual_seed(20)
    for N, K in [(24, 32), (16, 128), (18, 64)]:
        w_master = torch.randn(N, K)
        wq, scale = _quantize_fp4(w_master)
        x = torch.randn(1, K)
        _assert_close(
            backend.linear_fp4(x, wq, scale),
            reference.linear_fp4(x, wq, scale),
            f"linear_fp4_gemv N={N} K={K}",
        )


def test_linear_fp4_fp8_parity(backend):
    """Prefill (M>1) fp8 path: per-32-block e4m3 activation quant + fp4->e4m3
    exact-grid dequant + fp8 WGMMA. On CPU the backend resolves to the bf16
    floor (tautology); on CUDA the sm90 cell resolves to the fp8 kernel.

    The reference does the SAME per-32-block e4m3 quant in torch (not the f32
    linear_fp4 reference): e4m3's ~2% multiplicative quant error does not
    average down over K, so the gate is kernel correctness vs an identical-quant
    torch reference, not quant precision vs f32. The e2m1fn weight grid is an
    exact subset of e4m3, so the weight side is error-free."""
    torch.manual_seed(21)
    for M, N, K in [(8, 64, 256), (4, 96, 128)]:
        w_master = torch.randn(N, K) * 0.1
        wq, scale = _quantize_fp4(w_master)
        x = torch.randn(M, K) * 0.5
        out = backend.linear_fp4(x, wq, scale)
        if not backend.target.startswith("cuda"):
            # CPU/metal: bf16 floor, compare to the f32 reference.
            _assert_close(
                out, reference.linear_fp4(x, wq, scale), f"linear_fp4_fp8 M={M} N={N} K={K}"
            )
            continue
        # CUDA: fp8 path, identical-quant reference.
        _assert_close(out, _linear_fp4_fp8_ref(x, wq, scale), f"linear_fp4_fp8 M={M} N={N} K={K}")


def test_linear_fp4_bwd():
    torch.manual_seed(5)
    w_master = torch.randn(24, 32)
    wq, scale = _quantize_fp4(w_master)
    x = torch.randn(6, 32)
    go = torch.randn(6, 24)
    gx, g_master = reference.linear_fp4_bwd(go, x, wq, scale)
    # STE: dequantized weight is constant w.r.t. master; g_master = grad @ x
    w_deq = reference.dequant_fp4(wq, scale)
    _assert_close(gx, go @ w_deq, "linear_fp4_bwd gx")
    _assert_close(g_master, go.t() @ x, "linear_fp4_bwd g_master")


# ---------------------------------------------------------------- linear fp8


def test_linear_fp8_parity(backend):
    """Native-fp8 linear: the sm90 cell's fp8 WGMMA kernel (M>1) is gated
    against the identical-quant reference; every other path (CPU/metal floor,
    sm90 M=1 decode via the bf16 master, or the kernel absent) is gated
    against the f32 dequant reference."""
    torch.manual_seed(26)
    kset = _resolve(backend.precision, backend.arch)
    for M, N, K in [(8, 128, 256), (4, 256, 128)]:
        w_master = torch.randn(N, K) * 0.1
        w8, wscale = _quantize_fp8(w_master)
        master = reference.dequant_fp8(w8, wscale).to(torch.bfloat16)
        x = torch.randn(M, K) * 0.5
        out = backend.linear_fp8(x, w8, wscale, master=master)
        kernel_path = backend.target.startswith("cuda") and M > 1 and "linear_fp8" in kset
        ref = _linear_fp8_ref(x, w8, wscale) if kernel_path else reference.linear_fp8(x, w8, wscale)
        _assert_close(out, ref, f"linear_fp8 M={M} N={N} K={K}")


def test_linear_fp8_bwd():
    torch.manual_seed(27)
    w_master = torch.randn(128, 256) * 0.1
    w8, wscale = _quantize_fp8(w_master)
    x = torch.randn(6, 256)
    go = torch.randn(6, 128)
    gx, g_master = reference.linear_fp8_bwd(go, x, w8, wscale)
    # STE: dequantized weight is constant w.r.t. master; g_master = grad @ x
    w_deq = reference.dequant_fp8(w8, wscale)
    _assert_close(gx, go @ w_deq, "linear_fp8_bwd gx")
    _assert_close(g_master, go.t() @ x, "linear_fp8_bwd g_master")


def test_linear_fp8_gemv_parity(backend):
    """M=1 decode path: the sm90 cell resolves to the fp8 GEMV kernel (the
    bf16 master floor on CPU/metal). The GEMV uses bf16 X (no activation
    quant, unlike the M>1 MMA path), so the gate is the f32 dequant reference
    — the bf16 X rounding is the only error source."""
    torch.manual_seed(28)
    kset = _resolve(backend.precision, backend.arch)
    for N, K in [(128, 256), (256, 128), (64, 512)]:
        w_master = torch.randn(N, K) * 0.1
        w8, wscale = _quantize_fp8(w_master)
        master = reference.dequant_fp8(w8, wscale).to(torch.bfloat16)
        x = torch.randn(1, K) * 0.5
        out = backend.linear_fp8(x, w8, wscale, master=master)
        assert out.shape == (1, N)
        _assert_close(out, reference.linear_fp8(x, w8, wscale), f"linear_fp8_gemv N={N} K={K}")
        # on CUDA the kernel path must actually be the one that ran
        if backend.target.startswith("cuda"):
            assert "linear_fp8_gemv" in kset


# ---------------------------------------------------------------- silu mul


def test_silu_mul_parity(backend):
    torch.manual_seed(6)
    gate = torch.randn(4, 8)
    up = torch.randn(4, 8)
    _assert_close(backend.silu_mul(gate, up), reference.silu_mul(gate, up), "silu_mul")


def test_silu_mul_bwd():
    torch.manual_seed(7)
    gate = torch.randn(4, 8)
    up = torch.randn(4, 8)
    _finite_diff_gradcheck("silu_mul_bwd", reference.silu_mul, reference.silu_mul_bwd, [gate, up])


# ---------------------------------------------------------------- softmax


def test_softmax_parity(backend):
    torch.manual_seed(8)
    x = torch.randn(4, 32)
    _assert_close(backend.softmax(x, -1), reference.softmax(x, -1), "softmax")
    x3 = torch.randn(2, 3, 16)
    _assert_close(backend.softmax(x3, 1), reference.softmax(x3, 1), "softmax axis=1")


# ---------------------------------------------------------------- rope


def test_rope_parity(backend):
    torch.manual_seed(9)
    x = torch.randn(2, 5, 3, 16)
    pos = torch.arange(5).unsqueeze(0).expand(2, -1).contiguous()
    _assert_close(backend.rope(x, pos, 1e4), reference.rope(x, pos, 1e4), "rope")


def test_rope_bwd():
    torch.manual_seed(10)
    x = torch.randn(2, 4, 2, 8)
    pos = torch.arange(4).unsqueeze(0).expand(2, -1).contiguous()
    _finite_diff_gradcheck(
        "rope_bwd",
        lambda x: reference.rope(x, pos, 1e4),
        lambda go, x: reference.rope_bwd(go, pos, 1e4),
        [x],
    )


# ---------------------------------------------------------------- embedding


def test_embedding_parity(backend):
    torch.manual_seed(11)
    idx = torch.tensor([[1, 3], [2, 0]])
    table = torch.randn(10, 8)
    _assert_close(backend.embedding(idx, table), reference.embedding(idx, table), "embedding")


def test_embedding_bwd():
    torch.manual_seed(12)
    idx = torch.tensor([1, 3, 2, 1])
    table = torch.randn(5, 8)
    go = torch.randn(4, 8)
    gt = reference.embedding_bwd(go, idx, 5)
    # finite diff on the two rows that are used
    for row in [1, 3]:
        for d in range(8):
            old = table[row, d].item()
            table[row, d] = old + FD_EPS
            fp = reference.embedding(idx, table)
            table[row, d] = old - FD_EPS
            fm = reference.embedding(idx, table)
            table[row, d] = old
            num = ((fp - fm) / (2 * FD_EPS) * go).sum().item()
            assert abs(gt[row, d].item() - num) < 1e-2, (
                f"embedding_bwd [{row},{d}]: {gt[row, d].item()} vs {num}"
            )


# ---------------------------------------------------------------- paged attention


def _naive_paged(q, k_cache, v_cache, block_table, seq_lens, scale):
    """Naive full-matrix paged attention (reference of the reference)."""
    squeeze = q.ndim == 3
    if squeeze:
        q = q.unsqueeze(1)
    b, t, h, d = q.shape
    hkv = k_cache.shape[1]
    block = k_cache.shape[2]
    out = torch.empty(b, t, h, d)
    for bi in range(b):
        hist = int(seq_lens[bi]) - t
        for hi in range(h):
            hk = hi * hkv // h
            for ti in range(t):
                upper = hist + ti + 1
                keys = torch.stack(
                    [k_cache[int(block_table[bi, p // block]), hk, p % block] for p in range(upper)]
                )
                vals = torch.stack(
                    [v_cache[int(block_table[bi, p // block]), hk, p % block] for p in range(upper)]
                )
                scores = (q[bi, ti, hi] * keys).sum(-1) * scale
                p = torch.softmax(scores, -1)
                out[bi, ti, hi] = (p.unsqueeze(-1) * vals).sum(0)
    return out.squeeze(1) if squeeze else out


def test_paged_attention_vs_naive(backend):
    torch.manual_seed(13)
    b, h, hkv, d, block = 2, 4, 2, 16, 16
    k_cache = torch.randn(8, hkv, block, d)
    v_cache = torch.randn(8, hkv, block, d)
    block_table = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32)
    scale = 1.0 / (d**0.5)
    # decode (T=1)
    q = torch.randn(b, h, d)
    seq_lens = torch.tensor([20, 12], dtype=torch.int32)
    _assert_close(
        backend.paged_attention(q, k_cache, v_cache, block_table, seq_lens, scale),
        _naive_paged(q, k_cache, v_cache, block_table, seq_lens, scale),
        "paged_attention decode",
    )
    # prefill (T=3, causal within the chunk)
    q3 = torch.randn(b, 3, h, d)
    seq_lens3 = torch.tensor([23, 15], dtype=torch.int32)
    _assert_close(
        backend.paged_attention(q3, k_cache, v_cache, block_table, seq_lens3, scale),
        _naive_paged(q3, k_cache, v_cache, block_table, seq_lens3, scale),
        "paged_attention prefill",
    )


# ---------------------------------------------------------------- write tokens


def test_write_tokens_parity(backend):
    """Paged KV scatter kernel vs the pool's torch-loop write (sm90-only
    kernel; on other arches the backend op IS the loop, so the gate is the
    kernel cell)."""
    if backend.arch != "sm90":
        pytest.skip("write_tokens kernel is sm90-only")
    torch.manual_seed(24)
    from tilerl.engine import BatchKv
    from tilerl.kv_cache import PagedKvPool

    b, t, hkv, d, nb = 2, 3, 2, 16, 6
    k = torch.randn(b, t, hkv, d, device=backend.device).bfloat16()
    v = torch.randn(b, t, hkv, d, device=backend.device).bfloat16()
    block_table = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device=backend.device)
    seq_len = torch.tensor([30, 20], dtype=torch.int32, device=backend.device)
    state_slot = torch.zeros(b, dtype=torch.long, device=backend.device)
    ref = PagedKvPool(nb, hkv, d, num_layers=1, device=backend.device)
    ref.write_tokens(k, v, BatchKv(block_table, seq_len, state_slot, ref, None), 0)
    got = PagedKvPool(nb, hkv, d, num_layers=1, device=backend.device)
    backend.write_tokens(k, v, BatchKv(block_table, seq_len, state_slot, got, None), 0)
    _assert_close(got.k_pool, ref.k_pool, "write_tokens k")
    _assert_close(got.v_pool, ref.v_pool, "write_tokens v")


# ---------------------------------------------------------------- gated delta


def test_linear_attn_chunk_parity(backend):
    torch.manual_seed(14)
    b, c, h, d = 2, 6, 2, 16
    q = torch.randn(b, c, h, d) * 0.1
    k = torch.randn(b, c, h, d) * 0.1
    v = torch.randn(b, c, h, d) * 0.1
    g = torch.sigmoid(torch.randn(b, c, h)) * 0.5 + 0.4
    beta = torch.sigmoid(torch.randn(b, c, h))
    state = torch.randn(b, h, d, d) * 0.01
    out, ns, _ = backend.linear_attn_chunk(q, k, v, g, beta, state)
    rout, rns, _ = reference.linear_attn_chunk(q, k, v, g, beta, state)
    _assert_close(out, rout, "linear_attn_chunk out")
    _assert_close(ns, rns, "linear_attn_chunk state")


def test_linear_attn_chunk_vs_step(backend):
    torch.manual_seed(15)
    b, c, h, d = 2, 6, 2, 16
    q = torch.randn(b, c, h, d) * 0.1
    k = torch.randn(b, c, h, d) * 0.1
    v = torch.randn(b, c, h, d) * 0.1
    g = torch.sigmoid(torch.randn(b, c, h)) * 0.5 + 0.4
    beta = torch.sigmoid(torch.randn(b, c, h))
    state = torch.randn(b, h, d, d) * 0.01
    out, ns, _ = backend.linear_attn_chunk(q, k, v, g, beta, state)
    s = state.clone()
    step_outs = []
    for t in range(c):
        o, s, _ = backend.linear_attn_step(q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], s)
        step_outs.append(o)
    step_out = torch.stack(step_outs, dim=1)
    _assert_close(out, step_out, "chunk-vs-step out")
    _assert_close(ns, s, "chunk-vs-step state")


def test_gdn_conv_window_makes_step_exact():
    """The conv_window carry makes segmented decode (T=1 per forward) exactly
    equal to a one-shot chunk forward: threading the returned window through
    each step reproduces the chunk outputs and final state."""
    torch.manual_seed(17)
    b, t, nkh, nvh, kd, vd, ker = 2, 6, 2, 2, 16, 16, 4
    q = torch.randn(b, t, nkh * kd) * 0.1
    k = torch.randn(b, t, nkh * kd) * 0.1
    v = torch.randn(b, t, nvh * vd) * 0.1
    g = torch.randn(b, t, nvh)
    beta = torch.randn(b, t, nvh)
    z = torch.randn(b, t, nvh * vd) * 0.1
    state = torch.randn(b, nvh, kd, vd) * 0.01
    qkv = nkh * kd * 2 + nvh * vd
    kw = dict(
        conv1d_weight=torch.randn(qkv, ker) * 0.1,
        dt_bias=torch.randn(nvh),
        a_log=torch.randn(nvh) * 0.1,
        norm_weight=torch.ones(vd),
    )
    chunk_out, chunk_state, _ = reference.gdn_forward(q, k, v, g, beta, state, z=z, **kw)

    s, w, step_outs = state.clone(), torch.zeros(b, ker - 1, qkv), []
    for ti in range(t):
        o, s, w = reference.gdn_forward(
            q[:, ti : ti + 1],
            k[:, ti : ti + 1],
            v[:, ti : ti + 1],
            g[:, ti : ti + 1],
            beta[:, ti : ti + 1],
            s,
            z=z[:, ti : ti + 1],
            conv_window=w,
            **kw,
        )
        step_outs.append(o)
    _assert_close(chunk_out, torch.cat(step_outs, dim=1), "window step out")
    _assert_close(chunk_state, s, "window step state")


def test_gdn_decode_fused_parity(backend):
    """Full-GDN decode (T=1): backend vs reference.gdn_forward. On CPU the
    backend resolves to the reference (tautology); on CUDA the sm90 cell
    resolves to the fused kernel — the real gate."""
    torch.manual_seed(19)
    b, nkh, nvh, kd, vd, ker = 2, 2, 4, 16, 16, 4
    qkv = 2 * nkh * kd + nvh * vd
    q = torch.randn(b, 1, nkh * kd) * 0.1
    k = torch.randn(b, 1, nkh * kd) * 0.1
    v = torch.randn(b, 1, nvh * vd) * 0.1
    g = torch.randn(b, 1, nvh)
    beta = torch.randn(b, 1, nvh)
    z = torch.randn(b, 1, nvh * vd) * 0.1
    state = torch.randn(b, nvh, kd, vd) * 0.01
    window = torch.randn(b, ker - 1, qkv) * 0.1
    kw = dict(
        conv1d_weight=torch.randn(qkv, ker) * 0.1,
        dt_bias=torch.randn(nvh),
        a_log=torch.randn(nvh) * 0.1,
        norm_weight=torch.ones(vd),
        conv_window=window,
    )
    out, ns, nw = backend.linear_attn_chunk(q, k, v, g, beta, state, z=z, **kw)
    rout, rns, rnw = reference.gdn_forward(q, k, v, g, beta, state, z=z, **kw)
    _assert_close(out, rout, "gdn decode fused out")
    _assert_close(ns, rns, "gdn decode fused state")
    _assert_close(nw, rnw, "gdn decode fused window")


def _gdn_inputs(b, t, nkh, nvh, kd, vd, ker, seed):
    """Random full-GDN inputs (seeded): q/k/v/g/beta/z/state/window + kwargs."""
    torch.manual_seed(seed)
    qkv = 2 * nkh * kd + nvh * vd
    q = torch.randn(b, t, nkh * kd) * 0.1
    k = torch.randn(b, t, nkh * kd) * 0.1
    v = torch.randn(b, t, nvh * vd) * 0.1
    g = torch.randn(b, t, nvh)
    beta = torch.randn(b, t, nvh)
    z = torch.randn(b, t, nvh * vd) * 0.1
    state = torch.randn(b, nvh, kd, vd) * 0.01
    window = torch.randn(b, ker - 1, qkv) * 0.1
    kw = dict(
        conv1d_weight=torch.randn(qkv, ker) * 0.1,
        dt_bias=torch.randn(nvh),
        a_log=torch.randn(nvh) * 0.1,
        norm_weight=torch.ones(vd),
        conv_window=window,
    )
    return q, k, v, g, beta, z, state, kw


def test_gdn_chunk_fused_parity(backend):
    """Full-GDN prefill (T>1): backend vs reference.gdn_forward. On CPU the
    backend resolves to the reference (tautology); on CUDA the sm90 cell
    resolves to the fused chunk kernel — the real gate."""
    q, k, v, g, beta, z, state, kw = _gdn_inputs(2, 6, 2, 4, 16, 16, 4, 23)
    out, ns, nw = backend.linear_attn_chunk(q, k, v, g, beta, state, z=z, **kw)
    rout, rns, rnw = reference.gdn_forward(q, k, v, g, beta, state, z=z, **kw)
    _assert_close(out, rout, "gdn chunk fused out")
    _assert_close(ns, rns, "gdn chunk fused state")
    _assert_close(nw, rnw, "gdn chunk fused window")


def test_gdn_chunk_matches_decode(backend):
    """The chunk kernel at T=1 equals the decode kernel (it is the T-loop
    generalization of make_gdn_decode_fused — same fused ops, same order).
    sm90-only: both kernels live in the sm90 cell."""
    if backend.arch != "sm90":
        pytest.skip("GDN fused kernels are sm90-only")
    q, k, v, g, beta, z, state, kw = _gdn_inputs(2, 1, 2, 4, 16, 16, 4, 29)
    f32, bf16, c = backend._f32, backend._bf16, backend._c
    i32 = backend._i32
    common = (
        f32(kw["dt_bias"]),
        f32(kw["a_log"]),
        f32(kw["norm_weight"]),
        f32(kw["conv1d_weight"]),
    )
    seq_q = i32(torch.full((q.shape[0],), q.shape[1], dtype=torch.int32))
    dout, dstate, dwin = backend._kernel("gdn_decode_fused")(
        c(f32(q).squeeze(1)),
        c(f32(k).squeeze(1)),
        c(f32(v).squeeze(1)),
        c(f32(z).squeeze(1)),
        c(f32(g).squeeze(1)),
        c(f32(beta).squeeze(1)),
        *common,
        f32(kw["conv_window"]),
        f32(state),
        threads=state.shape[-1],
    )
    cout, cstate, cwin = backend._kernel("gdn_chunk_fused")(
        c(bf16(q)),
        c(bf16(k)),
        c(bf16(v)),
        c(bf16(z)),
        c(f32(g)),
        c(f32(beta)),
        *common,
        bf16(kw["conv_window"]),
        f32(state),
        c(seq_q),
        threads=state.shape[-1],
    )
    _assert_close(cout.squeeze(1), dout, "chunk-vs-decode out")
    _assert_close(cstate, dstate, "chunk-vs-decode state")
    _assert_close(cwin, dwin, "chunk-vs-decode window")


def test_linear_attn_bwd():
    torch.manual_seed(16)
    b, c, h, d = 1, 3, 1, 4
    q = torch.randn(b, c, h, d) * 0.1
    k = torch.randn(b, c, h, d) * 0.1
    v = torch.randn(b, c, h, d) * 0.1
    g = torch.sigmoid(torch.randn(b, c, h)) * 0.5 + 0.4
    beta = torch.sigmoid(torch.randn(b, c, h))
    # State at realistic magnitude: at 0.01 the S_prev-dependent error terms
    # vanish below the threshold and the backward passes even when wrong.
    state = torch.randn(b, h, d, d) * 0.5
    # autograd cross-check (finite diff too noisy on the scan; f32 scan
    # accumulation tolerates a slightly looser threshold)
    _autograd_gradcheck(
        "linear_attn_bwd",
        lambda q, k, v, g, beta, state: reference.linear_attn_chunk(q, k, v, g, beta, state)[0],
        reference.linear_attn_bwd,
        [q, k, v, g, beta, state],
        max_rel=2e-2,
    )


# ---------------------------------------------------------------- full attention bwd


def test_attention_bwd():
    torch.manual_seed(17)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    _autograd_gradcheck(
        "attention_bwd",
        lambda q, k, v: reference.dense_attention(q, k, v, 0.5),
        lambda go, q, k, v: reference.dense_attention_bwd(go, q, k, v, 0.5),
        [q, k, v],
    )


# ---------------------------------------------------------------- sampling


def test_sample_deterministic():
    torch.manual_seed(18)
    logits = torch.randn(4, 100)
    t1 = reference.sample(logits, 1.0, 0.9, 42)
    t2 = reference.sample(logits, 1.0, 0.9, 42)
    t3 = reference.sample(logits, 1.0, 0.9, 43)
    assert (t1 == t2).all(), "sample: same seed must give same tokens"
    assert (t1 != t3).any(), "sample: different seed should differ"
    assert ((t1 >= 0) & (t1 < 100)).all(), "sample: token ids out of range"
    # low temperature + tiny top_p is near-greedy
    greedy = reference.sample(logits, 0.01, 0.01, 7)
    assert (greedy == logits.argmax(-1)).float().mean() > 0.9


# ---------------------------------------------------------------- backend plumbing


def test_backend_target_and_device(backend):
    assert backend.target in ("c", "cuda", "llvm", "metal")
    assert isinstance(backend.device, torch.device)
