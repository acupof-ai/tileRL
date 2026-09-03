"""Parity gates: every backend op vs the torch-eager reference at
allclose(rtol=1e-2, atol=1e-2); every backward vs finite differences or
torch.autograd."""

from __future__ import annotations

import os

os.environ.setdefault("TILERL_TARGET", "cpu")

import pytest
import torch

from tilerl_kernels import reference
from tilerl_kernels.backend import _MX, _resolve, get_backend
from tilerl_kernels.reference import pack_fp4, renorm_fp4_scale, twiddle_fp4, untwiddle_fp4

RTOL = 1e-2
ATOL = 1e-2
FD_EPS = 1e-3


@pytest.fixture(scope="module")
def backend():
    return get_backend()


def _assert_close(a, b, msg):
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


_quantize_fp4 = pack_fp4  # scripts/probe_linear_err.py imports it


def _linear_fp4_fp8_ref(x, wq, scale):
    """The w4a8 path's identical-quant reference: e4m3's ~2% quant error does
    not average down over K, so the gate is kernel correctness, not precision."""
    xbf = x.to(torch.bfloat16).float()
    ascale = 448.0 / xbf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    xq = (xbf * ascale).to(torch.float8_e4m3fn).float() / ascale
    w_q8 = reference.dequant_fp4(wq, scale).to(torch.float8_e4m3fn).float()
    return xq @ w_q8.t()


def _quantize_fp8(w_master):
    """Per-128-block e4m3 quant in the loader's layout: w8 [N,K], wscale [ceil(N/128), ceil(K/128)]."""
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
    """The w8a8 path's identical-quant reference."""
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
    """M=1: the sm90 cell resolves to the bf16 GEMV."""
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


@pytest.mark.parametrize("block", [32, 16])
def test_linear_fp4_grid(backend, block):
    """The nibble decode against a literal e2m1 table: pack/dequant/kernel share
    one grid constant, so every other fp4 test passes with a wrong grid."""
    ocp = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    nib = torch.arange(16, dtype=torch.uint8).repeat(block // 16)  # K = block
    idx = (nib & 7).long()
    want = torch.where(nib >= 8, -ocp[idx], ocp[idx]) * 2.0
    wq = (nib[0::2] | (nib[1::2] << 4)).reshape(1, block // 2)
    got = backend.linear_fp4(torch.eye(block), wq, torch.ones(1, 1), oscale=torch.full((1,), 2.0))[
        :, 0
    ]
    _assert_close(got, want, f"fp4 grid block={block}")


def test_fp4_w4a8_e4m3_range():
    """renorm_fp4_scale keeps 6 * scale inside e4m3's range for the w4a8
    requant (raw checkpoint scales saturate at 50% error); gated in torch since
    the CPU dequant target is f32."""
    torch.manual_seed(9)
    wq, scale = pack_fp4(torch.randn(64, 256) * 0.02, 16)  # real 27B magnitudes
    gs = scale.max() / 448.0  # NVFP4 stores block scales as e4m3
    ckpt = (scale / gs).to(torch.float8_e4m3fn).float()
    scale2, oscale = renorm_fp4_scale(ckpt, gs.reshape(1).expand(64))
    assert 6 * scale2.max() <= 448
    x = torch.randn(8, 256)
    ref = reference.linear_fp4(x, wq, ckpt, gs.expand(64))  # f32 weight dequant
    w8 = reference.dequant_fp4(wq, scale2).to(torch.float8_e4m3fn).float()
    rel = (((x @ w8.t()) * oscale - ref).abs().max() / ref.abs().max()).item()
    assert rel <= 0.03, f"w4a8 e4m3 weight dequant error {rel:.3f}"


def test_linear_fp4_parity(backend):
    torch.manual_seed(4)
    wq, scale = pack_fp4(torch.randn(24, 32))
    # the reference follows the dispatch: fp4 through M=_MX, w4a8 above
    for m in (6, _MX + 8):
        x = torch.randn(m, 32)
        fp8_path = backend.target.startswith("cuda") and m > _MX
        ref = _linear_fp4_fp8_ref(x, wq, scale) if fp8_path else reference.linear_fp4(x, wq, scale)
        _assert_close(backend.linear_fp4(x, wq, scale), ref, f"linear_fp4 M={m}")


def test_linear_fp4_gemv_parity(backend):
    """M=1 GEMV and the M-row GEMV (a row-indexing bug there is invisible at M=1)."""
    torch.manual_seed(20)
    for N, K in [(24, 32), (16, 128), (18, 64)]:
        wq, scale = pack_fp4(torch.randn(N, K))
        for M in (1, 2, 3, 4):
            x = torch.randn(M, K)
            _assert_close(
                backend.linear_fp4(x, wq, scale),
                reference.linear_fp4(x, wq, scale),
                f"linear_fp4_gemv M={M} N={N} K={K}",
            )


def test_linear_fp4_fp8_parity(backend):
    """Prefill w4a8 path (M > _MX) vs the identical-quant reference on CUDA;
    the f32 reference on CPU/metal."""
    torch.manual_seed(21)
    for M, N, K in [(_MX + 8, 64, 256), (_MX + 4, 96, 128)]:
        wq, scale = pack_fp4(torch.randn(N, K) * 0.1)
        x = torch.randn(M, K) * 0.5
        out = backend.linear_fp4(x, wq, scale)
        ref = _linear_fp4_fp8_ref if backend.target.startswith("cuda") else reference.linear_fp4
        _assert_close(out, ref(x, wq, scale), f"linear_fp4_fp8 M={M} N={N} K={K}")


# ---------------------------------------------------------------- linear fp8


def test_linear_fp8_parity(backend):
    """Native fp8 linear on sm90; a cell without the kernel raises (materialize
    converts the weight to bf16 at load instead)."""
    torch.manual_seed(26)
    kset = _resolve(backend.precision, backend.arch)
    for M, N, K in [(8, 128, 256), (4, 256, 128), (_MX + 8, 128, 256)]:
        w8, wscale = _quantize_fp8(torch.randn(N, K) * 0.1)
        x = torch.randn(M, K) * 0.5
        if "linear_fp8" not in kset:
            with pytest.raises(NotImplementedError, match="linear_fp8"):
                backend.linear_fp8(x, w8, wscale)
            continue
        out = backend.linear_fp8(x, w8, wscale)
        # the activation stays bf16 through M=_MX (exact to 3e-4); e4m3 above (~2.6%)
        w8a8 = backend.target.startswith("cuda") and M > _MX
        ref = _linear_fp8_ref(x, w8, wscale) if w8a8 else reference.linear_fp8(x, w8, wscale)
        _assert_close(out, ref, f"linear_fp8 M={M} N={N} K={K}")


def test_ref_backend_fp8_surface():
    from tilerl.testing import RefBackend

    x = torch.randn(2, 128)
    w8, wscale = _quantize_fp8(torch.randn(64, 128))
    backend = RefBackend()
    _assert_close(backend.linear_fp8(x, w8, wscale), reference.linear_fp8(x, w8, wscale), "ref fp8")


def test_linear_fp8_gemv_parity(backend):
    """fp8 GEMV (bf16 X, no activation quant) vs the f32 dequant reference."""
    torch.manual_seed(28)
    kset = _resolve(backend.precision, backend.arch)
    for N, K in [(128, 256), (256, 128), (64, 512)]:
        w8, wscale = _quantize_fp8(torch.randn(N, K) * 0.1)
        for M in (1, 2, 3, 4):
            x = torch.randn(M, K) * 0.5
            if "linear_fp8_gemv" not in kset:
                with pytest.raises(NotImplementedError, match="linear_fp8"):
                    backend.linear_fp8(x, w8, wscale)
                continue
            out = backend.linear_fp8(x, w8, wscale)
            assert out.shape == (M, N)
            _assert_close(out, reference.linear_fp8(x, w8, wscale),
                          f"linear_fp8_gemv M={M} N={N} K={K}")


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
    kv_dtype = torch.bfloat16  # the pool's dtype; an f32 cache tested nothing on CUDA
    k_cache = torch.randn(8, hkv, block, d).to(kv_dtype)
    v_cache = torch.randn(8, hkv, block, d).to(kv_dtype)
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
    # verify widths: one KV read serves the GQA group at every chain position.
    # 8 query heads over one KV head put the M tile at 32 and 64 rows — both
    # wide branches, and the tile is what the warp count has to divide.
    # n=65 is the other axis: n % 64 in [1, W-1] leaves the last KV tile past the
    # low chain positions' causal bound while its split is still non-empty.
    k1 = torch.randn(10, 1, block, d).to(kv_dtype)
    v1 = torch.randn(10, 1, block, d).to(kv_dtype)
    bt1 = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=torch.int32)
    for w, n in ((4, 24), (8, 28), (8, 65)):
        qw = torch.randn(b, w, 8, d)
        lens_w = torch.tensor([n, n - 8], dtype=torch.int32)
        _assert_close(
            backend.paged_attention(qw, k1, v1, bt1, lens_w, scale),
            _naive_paged(qw, k1, v1, bt1, lens_w, scale),
            f"paged_attention verify width {w} len {n}",
        )


# ---------------------------------------------------------------- write tokens


def test_write_tokens_parity(backend):
    """Paged KV scatter kernel vs the pool's torch-loop write (sm90 only)."""
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


def test_gdn_conv_window_makes_step_exact():
    """Segmented decode (T=1 per forward, window carried) equals a one-shot chunk forward."""
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
    """GDN decode (T=1): backend vs reference.gdn_forward (a tautology off sm90)."""
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


def _gdn_inputs(b, t, nkh, nvh, kd, vd, ker, seed, scale=1.0):
    torch.manual_seed(seed)
    sc = 0.1 * scale
    qkv = 2 * nkh * kd + nvh * vd
    q = torch.randn(b, t, nkh * kd) * sc
    k = torch.randn(b, t, nkh * kd) * sc
    v = torch.randn(b, t, nvh * vd) * sc
    g = torch.randn(b, t, nvh)
    beta = torch.randn(b, t, nvh)
    z = torch.randn(b, t, nvh * vd) * sc
    state = torch.randn(b, nvh, kd, vd) * 0.1 * sc
    window = torch.randn(b, ker - 1, qkv) * sc
    kw = dict(
        conv1d_weight=torch.randn(qkv, ker) * 0.1,
        dt_bias=torch.randn(nvh),
        a_log=torch.randn(nvh) * 0.1,
        norm_weight=torch.ones(vd),
        conv_window=window,
    )
    return q, k, v, g, beta, z, state, kw


def test_gdn_chunk_fused_parity_full_scale(backend):
    """The chunk gate at the model's real input magnitude: a rejected pipeline
    passed at scale 0.1 and was 26% wrong at 1.0. Tolerance 5%, not 1%: the
    bf16-IO kernel lands at 1.2% here, which is the format's own noise."""
    q, k, v, g, beta, z, state, kw = _gdn_inputs(2, 96, 2, 6, 16, 16, 4, 37, scale=10.0)
    got = backend.linear_attn_chunk(q, k, v, g, beta, state, z=z, **kw)
    ref = reference.gdn_forward(q, k, v, g, beta, state, z=z, **kw)
    bad = []
    for a, b_, name in zip(got, ref, ("out", "state", "window")):
        if a is None or b_ is None:
            continue
        b_ = b_.to(a.device)
        rel = (a - b_).abs().max().item() / max(b_.abs().max().item(), 1e-9)
        if rel >= 0.05:
            bad.append(f"{name} {100 * rel:.1f}%")
    assert not bad, "gdn full-scale relative error: " + ", ".join(bad)


def test_gdn_chunk_fused_parity(backend):
    """GDN prefill (T>1): backend vs reference.gdn_forward (a tautology off sm90)."""
    q, k, v, g, beta, z, state, kw = _gdn_inputs(2, 6, 2, 4, 16, 16, 4, 23)
    out, ns, nw = backend.linear_attn_chunk(q, k, v, g, beta, state, z=z, **kw)
    rout, rns, rnw = reference.gdn_forward(q, k, v, g, beta, state, z=z, **kw)
    _assert_close(out, rout, "gdn chunk fused out")
    _assert_close(ns, rns, "gdn chunk fused state")
    _assert_close(nw, rnw, "gdn chunk fused window")


def test_gdn_chunkwise_matches_serial():
    """chunkwise-WY equals the serial scan over a full layer (conv, norms, gates included)."""
    q, k, v, g, beta, z, state, kw = _gdn_inputs(2, 96, 2, 6, 16, 16, 4, 31)
    ref = reference.gdn_forward(q, k, v, g, beta, state, z=z, **kw)
    for chunk in (16, 32, 64):
        got = reference.gdn_forward(q, k, v, g, beta, state, z=z, chunkwise=chunk, **kw)
        for a, b_, name in zip(got, ref, ("out", "state", "window")):
            _assert_close(a, b_, f"gdn chunkwise({chunk}) {name}")


@pytest.mark.parametrize("t", [1, 4])
def test_gdn_chunk_matches_decode(backend, t):
    """The chunk kernel equals the in-place decode kernel, per-chain-step planes
    included, at both a plain decode width and a verify width (sm90 only)."""
    if backend.arch != "sm90":
        pytest.skip("GDN fused kernels are sm90-only")
    q, k, v, g, beta, z, state, kw = _gdn_inputs(2, t, 2, 4, 16, 16, 4, 29)
    f32, bf16, c = backend._f32, backend._bf16, backend._c
    i32 = backend._i32
    b, nvh, vd = state.shape[0], state.shape[1], state.shape[-1]
    common = (
        f32(kw["dt_bias"]),
        f32(kw["a_log"]),
        f32(kw["norm_weight"]),
        f32(kw["conv1d_weight"]),
    )
    seq_q = i32(torch.full((b,), t, dtype=torch.int32))
    states = f32(state).unsqueeze(1).contiguous()  # pool [B, 1 layer, ...], updated in place
    win = f32(kw["conv_window"]).unsqueeze(1)  # [B, W, D] -> planes [B, 1 layer, 2, W, D]
    windows = torch.stack([win, torch.zeros_like(win)], dim=2).contiguous()
    par = i32(torch.zeros(b, dtype=torch.int32))
    dstep = torch.empty((b, 1, t, *state.shape[1:]), dtype=torch.float32, device=backend.device)
    dstepw = torch.empty((b, 1, t, *kw["conv_window"].shape[1:]), dtype=torch.float32,
                         device=backend.device)
    dout = backend._kernel("gdn_decode_fused")(
        c(f32(q)),
        c(f32(k)),
        c(f32(v)),
        c(f32(z)),
        c(f32(g)),
        c(f32(beta)),
        *common,
        windows,
        par,
        states,
        i32(torch.arange(b, dtype=torch.int32)),
        dstep,
        dstepw,
        0,
        t,
        threads=vd,
    )
    cstep = torch.empty((b, t, *state.shape[1:]), dtype=torch.float32, device=backend.device)
    cstepw = torch.empty((b, t, *kw["conv_window"].shape[1:]), dtype=torch.bfloat16,
                         device=backend.device)
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
        cstep,
        cstepw,
        threads=vd,
    )
    # the chunk kernel returns the raw recurrence output; apply the caller's norm + z-gate
    cout = backend.silu_mul(
        backend._dev(z, torch.float32).reshape(b, t, nvh, vd),
        backend.rmsnorm(cout.reshape(b, t, nvh, vd),
                        backend._const_f32(kw["norm_weight"]), 1e-6),
    ).reshape(cout.shape)
    _assert_close(cout, dout, "chunk-vs-decode out")
    _assert_close(cstate, states[:, 0], "chunk-vs-decode state")
    _assert_close(cstep, dstep[:, 0], "chunk-vs-decode step states")
    _assert_close(cwin, windows[:, 0, 1], "chunk-vs-decode window")  # written to the other plane
    _assert_close(cstepw.float(), dstepw[:, 0], "chunk-vs-decode step windows")


def test_gdn_chunk_adjoint_is_exact():
    """The chunked backward is the exact adjoint of the chunked forward (f64, ~1e-15 vs autograd)."""
    torch.manual_seed(3)
    b, n, h, dk, dv = 2, 6, 3, 5, 4
    f64 = dict(dtype=torch.float64)
    q = torch.randn(b, n, h, dk, **f64)
    k = torch.nn.functional.normalize(torch.randn(b, n, h, dk, **f64), dim=-1)
    v = torch.randn(b, n, h, dv, **f64)
    gt = -torch.rand(b, n, h, **f64) * 0.5
    bt = torch.rand(b, n, h, **f64)
    st = torch.randn(b, h, dk, dv, **f64)
    leaves = [x.clone().requires_grad_(True) for x in (q, k, v, bt, gt, st)]
    out, nxt, _ = reference._gdn_chunk_fwd(*leaves)
    dout, dnxt = torch.randn_like(out), torch.randn_like(nxt)
    ((out * dout).sum() + (nxt * dnxt).sum()).backward()
    _, _, cache = reference._gdn_chunk_fwd(q, k, v, bt, gt, st)
    got = reference._gdn_chunk_bwd(dout, dnxt, q, k, v, bt, cache)
    for name, g, leaf in zip(["q", "k", "v", "beta", "gt", "state"], got, leaves):
        rel = (g - leaf.grad).abs().max() / leaf.grad.abs().max().clamp_min(1e-30)
        assert rel < 1e-12, f"chunk adjoint d{name}: rel {rel:.2e}"


def test_gdn_bwd_spans_chunks():
    """gdn_backward across several chunks with a partial tail (test_gdn_bwd's T=3 never leaves one)."""
    for t in (reference._GDN_CHUNK, 2 * reference._GDN_CHUNK + 5):
        q, k, v, g, beta, z, state, kw = _gdn_inputs(1, t, 2, 2, 8, 8, 8, 24)
        window = torch.zeros_like(kw["conv_window"])
        order = ("conv1d_weight", "dt_bias", "a_log", "norm_weight")

        def fwd(*leaves):
            qq, kk, vv, gg, bb, ss, zz = leaves[:7]
            return reference.gdn_forward(qq, kk, vv, gg, bb, ss, z=zz,
                                         conv_window=window,
                                         **dict(zip(order, leaves[7:])))[0]

        def bwd(go, *args):
            qq, kk, vv, gg, bb, ss, zz = args[:7]
            return reference.gdn_backward(go, qq, kk, vv, gg, bb, ss, z=zz,
                                          conv_window=window,
                                          **dict(zip(order, args[7:])))

        _autograd_gradcheck(f"gdn_bwd t={t}", fwd, bwd,
                            [q, k, v, g, beta, state, z] + [kw[n] for n in order],
                            max_rel=5e-5)


def test_gdn_bwd():
    """gdn_backward vs torch.autograd (finite differences are too noisy across the scan)."""
    q, k, v, g, beta, z, state, kw = _gdn_inputs(1, 3, 1, 1, 4, 4, 4, 24)
    window = torch.zeros_like(kw["conv_window"])
    order = ("conv1d_weight", "dt_bias", "a_log", "norm_weight")

    def fwd(*leaves):
        qq, kk, vv, gg, bb, ss, zz = leaves[:7]
        rest = dict(zip(order, leaves[7:]))
        return reference.gdn_forward(qq, kk, vv, gg, bb, ss, z=zz, conv_window=window, **rest)[0]

    def bwd(go, *args):
        qq, kk, vv, gg, bb, ss, zz = args[:7]
        rest = dict(zip(order, args[7:]))
        return reference.gdn_backward(go, qq, kk, vv, gg, bb, ss, z=zz, conv_window=window, **rest)

    _autograd_gradcheck("gdn_bwd", fwd, bwd, [q, k, v, g, beta, state, z] + [kw[n] for n in order])
    with pytest.raises(NotImplementedError):  # only the zero window is exact
        reference.gdn_backward(torch.zeros(1, 3, 4), q, k, v, g, beta, state, z=z, **kw)


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
    greedy = reference.sample(logits, 0.01, 0.01, 7)
    assert (greedy == logits.argmax(-1)).float().mean() > 0.9


def test_sample_batch_matches_per_row():
    torch.manual_seed(18)
    logits = torch.randn(5, 100)
    temps = torch.tensor([1.0, 0.0, 0.8, 0.0, 1.0])  # rows 1,3 greedy
    top_ps = torch.tensor([0.9, 1.0, 0.5, 1.0, 0.95])
    seeds = torch.tensor([42, 7, 99, 3, 42])
    batched = reference.sample_batch(logits, temps, top_ps, seeds)
    for i in range(5):
        one = reference.sample(logits[i : i + 1], float(temps[i]), float(top_ps[i]), int(seeds[i]))
        assert batched[i] == one[0], f"row {i}: batch {batched[i]} vs per-row {one[0]}"


# ---------------------------------------------------------------- misc


def test_cross_entropy_is_stable_and_matches_gradient(backend):
    logits = torch.tensor([[[0.0, -100.0], [0.0, 0.0]]])
    loss, grad = backend.cross_entropy_loss_grad(logits, [[0, 1]])
    assert loss == pytest.approx(100.0)
    assert torch.equal(grad[0, 0], torch.tensor([1.0, -1.0]))


def test_fp4_twiddle_round_trip():
    """sm90 serves the twiddled byte layout; save_hf must undo it exactly."""
    wq = torch.randint(0, 256, (6, 32), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    tw = twiddle_fp4(wq)
    assert tw.shape == wq.shape and not torch.equal(tw, wq)
    assert torch.equal(untwiddle_fp4(tw), wq)


def test_frozen_bwd_chunking_matches_whole():
    """Chunked dX equals the one-shot result: a chunk boundary that splits an
    fp8 128-row scale block reads the wrong scale."""
    ref = reference
    torch.manual_seed(0)
    n, k = 1024, 256
    osc, g = torch.rand(n) + 0.5, torch.randn(3, 5, n)
    cases = [
        ((torch.randn(n, k) * 0.3).to(torch.float8_e4m3fn),
         torch.rand(-(-n // 128), -(-k // 128)) + 0.5, True),
        (torch.randint(0, 255, (n, k // 2), dtype=torch.uint8),
         torch.rand(n, k // 16) + 0.5, False),
    ]
    for wq, scale, fp8 in cases:
        big = ref._BWD_SLICE_BYTES
        try:
            ref._BWD_SLICE_BYTES = 1 << 30
            whole = ref.linear_frozen_bwd(g, wq, scale, oscale=osc, fp8=fp8)
            ref._BWD_SLICE_BYTES = 4096  # forces several chunks
            part = ref.linear_frozen_bwd(g, wq, scale, oscale=osc, fp8=fp8)
        finally:
            ref._BWD_SLICE_BYTES = big
        assert torch.allclose(whole, part, rtol=1e-4, atol=1e-4), f"fp8={fp8}"
