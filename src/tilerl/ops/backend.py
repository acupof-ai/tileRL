"""TileLang backend: kernel dispatch and the op interface.

This is the ONLY module in tilerl that talks to tilelang directly. Everything
above ``tilerl.ops`` calls the :class:`Backend` methods, never tilelang or
torch internals (torch is the tensor container only). The precision×arch
dispatch matrix and target resolution live in :mod:`tilerl.ops.registry`.

Forward ops compile and run the TileLang kernels in :mod:`tilerl.ops.kernels`
(cached per shape/dtype by tilelang's eager JIT). Backward ops without a
TileLang kernel run the torch-eager reference in :mod:`tilerl.ops.reference`
— the parity oracle and day-1 training fallback.
``# ponytail: torch-eager backward, tilelang kernel when perf demands``

CPU target facts (tilelang 0.1.13, macOS arm64, 2026-08-23): ``target="c"``
is the working CPU target (``"llvm"`` has no LLVM codegen in the wheel).
tilelang's own ``"auto"`` resolves to metal on Apple Silicon; tileRL's
default is CPU, so :func:`resolve_target` maps ``auto`` to ``"cuda"`` when a
CUDA device is visible and ``"c"`` otherwise. Kernels are f32-only: eager JIT
does not specialize on dtype, so bf16/f16 inputs are cast to f32 at the
boundary and outputs are f32.
Metal target facts (tilelang 0.1.13, Apple Silicon, 2026-08-24): the same
kernel source compiles for ``target="metal"`` with no per-target forks; the
Metal cell of the dispatch matrix reuses ``_CPU_KERNELS``. Tensors must live
on torch's ``"mps"`` device (``Backend.device``), and kernel I/O goes through
the torch-MPS adapter in tilelang's tvm_ffi runtime.
CUDA target facts (tilelang 0.1.13, H20/sm90, 2026-08-24): the same source
compiles for ``target="cuda"``; the sm90 cell uses the MMA (WGMMA) schedules
in kernels_linear.py — shared-memory tiled T.gemm with pipelining, the SOTA
pattern from examples/gemm/example_gemm.py. The sm90 MMA kernels are bf16-IO
(bf16 WGMMA, f32 accumulate); the CUDA path casts to bf16 once at the
boundary, while CPU/metal keep the f32 kernels. The MMA kernels require
block M/N divisible by 16 and the reduction dim divisible by 32; the CUDA
path of linear/linear_bwd/linear_fp4 zero-pads tails so the kernel always
sees exact tiles (decode M=1 pads to 16; all model N/K dims are already
multiples of 32). ``Backend.device`` pins ``cuda:<current>`` — ``torch.device("cuda")``
(index None) is not the device kernel outputs land on. Eager JIT invokes
NVCC per (shape, dtype): first call per shape costs 30-120s+, but tilelang
caches the compiled artifact on disk (``~/.tilelang/cache``, override with
``TILELANG_CACHE_DIR``) — a warm second-process call is ~0.2s (verified
2026-08-25). On the pod, point the cache at a persistent path (/work) or
every container restart re-pays the NVCC builds.
"""

from __future__ import annotations

import os
import weakref
from typing import Any

import torch

from . import kernels
from . import kernels_attn
from . import kernels_gdn
from . import kernels_linear
from . import kernels_mma
from . import reference
from .registry import _arch_for, _resolve, resolve_target

__all__ = ["Backend", "get_backend", "resolve_target"]

_THREADS = 64


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _snap_mma_tile(m: int, cap: int) -> int:
    """Snap an MMA tile M to a warp-partition-valid size.

    WGMMA Square policy needs each warp's rows to land on a valid partition;
    empirically 16/32/64/128 compile and 48/80/96/112 do not (tilelang
    0.1.13). Mixed batches land on arbitrary M (rows x chunk), so snap up.
    """
    return min(cap, next((s for s in (16, 32, 64, 128) if s >= m), 128))


def _pad2d(t: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Zero-pad a 2D tensor to [rows, cols] (bottom/right)."""
    pr, pc = rows - t.shape[0], cols - t.shape[1]
    if pr == 0 and pc == 0:
        return t
    return torch.nn.functional.pad(t, (0, pc, 0, pr))


def _pad1d(t: torch.Tensor, n: int) -> torch.Tensor:
    p = n - t.shape[0]
    return t if p == 0 else torch.nn.functional.pad(t, (0, p))


#: CUDA linear family as data: (op, M-regime) -> (kernel, K pad, N cap, N tile).
#: The regimes are measured crossovers, not guesses — GEMV at M=1, 8-way
#: K-split decode at M<=16, 2-way prefill above
#: (docs/experience/wins/2026-08-26-batch-decode-h2.md). The fp4->e4m3 arms
#: tile N at 64 because the kernel overrides block_N to _FP4_BLOCK_N=64 — a
#: 32-tile pad lets its grid read past the padded WQ.
_CUDA_PLAN = {
    ("linear", "gemv"): ("linear_bf16_gemv", 256, 4, 4),
    ("linear_fp4", "gemv"): ("linear_fp4_gemv", 256, 4, 4),
    ("linear_fp8", "gemv"): ("linear_fp8_gemv", 512, 4, 4),
    ("linear_fp4", "decode"): ("linear_fp4_fp8_decode", 512, 128, 64),
    ("linear_fp4", "prefill"): ("linear_fp4_fp8", 128, 128, 64),
    ("linear_fp8", "decode"): ("linear_fp8", 128, 128, 16),
    ("linear_fp8", "prefill"): ("linear_fp8", 128, 128, 16),
}


class Backend:
    """Resolved tilelang target plus lazily-compiled kernels.

    Kernel factories are called once per process; tilelang's eager JIT caches
    compiled code per (shape, dtype, kwargs) signature.
    """

    #: Backend identity (mirrors testing.RefBackend.name; used for logging).
    name = "tilelang"

    def __init__(self, target: str):
        self.target = target
        if target == "metal":
            self.device = torch.device("mps")
        elif target.startswith("cuda"):
            # torch.device("cuda") (index None) is not the device kernel
            # outputs land on (cuda:0) — pin the current device so the
            # boundary migration targets the right one.
            self.device = torch.device("cuda", torch.cuda.current_device())
        else:
            self.device = torch.device("cpu")
        self.precision = "bf16"
        self.arch = _arch_for(target)
        self._kernels: dict[str, object] = {}
        self._inv_freq_cache: dict[tuple[int, float], torch.Tensor] = {}
        self._const_f32_cache: dict[tuple[int, int | None], tuple[Any, int, torch.Tensor]] = {}
        self._ones_cache: dict[int, torch.Tensor] = {}

    def _kernel(self, name: str):
        k = self._kernels.get(name)
        if k is None:
            k = _resolve(self.precision, self.arch)[name](self.target)
            self._kernels[name] = k
        return k

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _c(t: torch.Tensor) -> torch.Tensor:
        """TileLang kernels check static strides — views must be contiguous."""
        return t if t.is_contiguous() else t.contiguous()

    def _dev(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Cast dtype and move to the backend device at the tilelang boundary.

        Kernels require inputs on the target device (mps for metal); CPU-side
        callers (parity tests, CPU-resident params) pass CPU tensors, so the
        boundary migrates them. No-op when dtype/device already match.
        """
        if t.dtype != dtype or t.device != self.device:
            return t.to(device=self.device, dtype=dtype)
        return t

    def _f32(self, t: torch.Tensor) -> torch.Tensor:
        return self._dev(t, torch.float32)

    def _bf16(self, t: torch.Tensor) -> torch.Tensor:
        return self._dev(t, torch.bfloat16)

    def _i32(self, t: torch.Tensor) -> torch.Tensor:
        return self._dev(t, torch.int32)

    def _inv_freq(self, d: int, theta: float) -> torch.Tensor:
        key = (d, theta)
        inv = self._inv_freq_cache.get(key)
        if inv is None:
            # On the backend device: a CPU-cached tensor would H2D-copy on
            # every rope call, and the copy is illegal inside a captured
            # decode tick (CUDA graph capture rejects unpinned CPU->CUDA).
            inv = 1.0 / (
                theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=self.device) / d)
            )
            self._inv_freq_cache[key] = inv
        return inv

    def _rows(self, x: torch.Tensor):
        # sm90 kernels are bf16-IO, CPU/metal f32; cast once at the boundary.
        io = torch.bfloat16 if self.target.startswith("cuda") else torch.float32
        return x.shape[:-1], self._c(self._dev(x, io).reshape(-1, x.shape[-1]))

    def _epilogue(self, y2, oscale, lead, n: int):
        # ponytail: torch epilogue for the per-row scale, fold into the kernel
        # accumulator if a sweep says it matters.
        return (y2 if oscale is None else y2 * self._const_f32(oscale)).reshape(*lead, n)

    def _plan(self, op: str, m: int, n: int, k: int):
        """(kernel, Mp, Np, Kp, block_M, block_N), or None when this cell has no
        specialized kernel — the caller falls through to its generic path."""
        if not self.target.startswith("cuda"):
            return None
        hit = _CUDA_PLAN.get((op, "gemv" if m == 1 else "decode" if m <= 16 else "prefill"))
        if hit is None:
            return None
        kernel, kpad, cap, tile = hit
        if kernel not in _resolve(self.precision, self.arch):
            return None
        bM = 1 if m == 1 else _snap_mma_tile(m, 128)
        bN = _round_up(min(cap, n), tile)
        return kernel, _round_up(m, bM), _round_up(n, bN), _round_up(k, kpad), bM, bN

    # ------------------------------------------------------------ add

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Elementwise add (residual stream). Recorded by the tape."""
        return a + b

    # ------------------------------------------------------------ rmsnorm

    def rmsnorm(self, x, w, eps):
        x = self._f32(x)
        w = self._const_f32(w)
        lead = x.shape[:-1]
        x2 = self._c(x.reshape(-1, x.shape[-1]))
        # Split-K: chunk the reduction dim across blocks (phase 1 partial
        # sums, phase 2 reduce+normalize) so decode (M=1) is not one serial
        # block. block_N=256 -> 20 blocks per row at the 5120 hidden size.
        N = x2.shape[-1]
        block_N = min(256, N)
        num_chunks = (N + block_N - 1) // block_N
        p = self._kernel("rmsnorm_partial")(x2, block_N, num_chunks, _THREADS)
        y = self._kernel("rmsnorm_apply")(x2, w, p, float(eps), block_N, num_chunks, _THREADS)
        return y.reshape(*lead, w.shape[0])

    def rmsnorm_bwd(self, grad, x, w, eps):
        grad = self._f32(grad)
        x = self._f32(x)
        w = self._f32(w)
        lead = x.shape[:-1]
        x2 = self._c(x.reshape(-1, x.shape[-1]))
        g2 = self._c(grad.reshape(-1, grad.shape[-1]))
        rstd = self._kernel("rmsnorm_rstd")(x2, eps=float(eps), threads=_THREADS)
        gx = self._kernel("rmsnorm_bwd_x")(g2, x2, w, rstd, threads=_THREADS)
        # gw: column-parallel reduction; reference is one einsum and faster
        # than a second kernel launch on CPU day-1.
        # ponytail: torch-eager gw, tilelang bwd-w kernel when perf demands
        gw = (g2 * x2 * rstd.unsqueeze(-1)).sum(dim=0)
        return gx.reshape(*lead, w.shape[0]), gw

    # ------------------------------------------------------------ rope

    def rope(self, x, positions, theta, rotary_dim=None):
        x = self._f32(x)
        rd = int(rotary_dim) if rotary_dim is not None else x.shape[-1]
        if rd < x.shape[-1]:
            x_rot = self._c(x[..., :rd])
            x_pass = x[..., rd:]
        else:
            x_rot, x_pass = self._c(x), None
        pos = self._i32(positions)
        if pos.ndim == 1:
            pos = pos.unsqueeze(0).expand(x_rot.shape[0], -1)
        inv = self._inv_freq(rd, float(theta)).to(x.device)
        k = self._kernel("rope")
        y = k(x_rot, pos.contiguous(), inv, threads=_THREADS)
        if x_pass is not None:
            y = torch.cat([y, x_pass], dim=-1)
        return y

    def rope_bwd(self, grad, positions, theta, rotary_dim=None):
        # Orthogonal rotation: gx = R(-angle) grad. Reference (one line).
        # ponytail: torch-eager backward, tilelang kernel when perf demands
        return reference.rope_bwd(grad, positions, theta, rotary_dim=rotary_dim)

    # ------------------------------------------------------------ linear

    def linear(self, x, w, bias=None):
        lead, x2 = self._rows(x)
        w = self._dev(w, x2.dtype)
        M, K, N = x2.shape[0], x2.shape[1], w.shape[0]
        plan = self._plan("linear", M, N, K)
        if plan is not None:
            # Decode GEMV: stream W once (2 bytes/elem) instead of padding M
            # to 16 WGMMA rows; the K-tail is zero-padded like the WGMMA path.
            kernel, _, Np, Kp, _, bN = plan
            y = self._kernel(kernel)(_pad2d(x2, 1, Kp), _pad2d(w, Np, Kp), 32, bN)[:1, :N]
            return (y if bias is None else y + self._f32(bias)).reshape(*lead, N)
        bM, bN = min(64, M), min(64, N)
        # _f32 before the cuda branch: every target needs the bias on-device.
        bias = (
            torch.zeros(N, dtype=torch.float32, device=self.device)
            if bias is None
            else self._f32(bias)
        )
        if self.target.startswith("cuda"):
            # WGMMA tiles: block M/N %16, reduction K %32; pad tails so the
            # MMA kernel sees exact tiles (no OOB loads).
            bM, bN = _round_up(bM, 16), _round_up(bN, 16)
            x2 = _pad2d(x2, _round_up(M, bM), _round_up(K, 32))
            w = _pad2d(w, _round_up(N, bN), _round_up(K, 32))
            bias = _pad1d(bias, w.shape[0])
        y = self._kernel("gemm_nt")(x2, w, bias, bM, bN, _THREADS)
        return y[:M, :N].reshape(*lead, N)

    def linear_bwd(self, grad, x, w):
        io = torch.bfloat16 if self.target.startswith("cuda") else torch.float32
        grad = self._dev(grad, io)
        x = self._dev(x, io)
        w = self._dev(w, io)
        g2 = self._c(grad.reshape(-1, grad.shape[-1]))
        x2 = self._c(x.reshape(-1, x.shape[-1]))
        M, N, K = g2.shape[0], g2.shape[1], w.shape[1]
        if self.target.startswith("cuda"):
            # gx = g2 @ w (gemm_nn): reduction N, output [M, K]
            bM, bK = _round_up(min(64, M), 16), _round_up(min(64, K), 16)
            gx = self._kernel("gemm_nn")(
                _pad2d(g2, _round_up(M, bM), _round_up(N, 32)),
                _pad2d(w, _round_up(N, 32), _round_up(K, bK)),
                bM,
                bK,
                _THREADS,
            )[:M, :K]
            # gw = g2.T @ x2 (gemm_tn): reduction M, output [N, K]
            bN = _round_up(min(64, N), 16)
            gw = self._kernel("gemm_tn")(
                _pad2d(g2, _round_up(M, 32), _round_up(N, bN)),
                _pad2d(x2, _round_up(M, 32), _round_up(K, bK)),
                bN,
                bK,
                _THREADS,
            )[:N, :K]
        else:
            bM = min(64, M)
            bN = min(64, K)
            gx = self._kernel("gemm_nn")(g2, w, bM, bN, _THREADS)
            gw = self._kernel("gemm_tn")(g2, x2, min(64, N), min(64, K), _THREADS)
        return gx.reshape(*grad.shape[:-1], K), gw

    # ------------------------------------------------------------ linear fp4

    def _served_fp4(self, wq):
        """The fp4 bytes this cell's kernels read. sm90 kernels decode the
        twiddled layout; the served tensor is rewritten in place ONCE (flagged)
        so graph capture, save_hf (which untwiddles by the flag) and CPU-resident
        callers all see one truth. Other cells read the natural layout."""
        wq = self._dev(wq, wq.dtype)  # uint8: device migration only
        if "linear_fp4_gemv" in _resolve(self.precision, self.arch) and not getattr(
            wq, "_tl_twiddled", False
        ):
            wq.copy_(reference.twiddle_fp4(wq))
            wq._tl_twiddled = True
        return wq

    def linear_fp4(self, x, wq, scale, master=None, oscale=None):
        # ``master`` is recording-only (the STE grad lands on it); the kernel
        # uses wq/scale.
        wq = self._served_fp4(wq)
        scale = self._f32(scale)
        lead, x2 = self._rows(x)
        M, K, N = x2.shape[0], x2.shape[1], wq.shape[0]
        blk = K // scale.shape[1]  # scale block from the loaded weight (16 or 32)
        plan = self._plan("linear_fp4", M, N, K)
        if plan is not None:
            kernel, Mp, Np, Kp, bM, bN = plan
            # K-tail: zero-padded X, and padded nibbles (0x00) decode to 0.0.
            wq, scale = _pad2d(wq, Np, Kp // 2), _pad2d(scale, Np, Kp // blk)
            if M == 1:
                # Decode GEMV: one activation row, stream+dequant WQ once; the
                # per-row oscale is folded into the kernel epilogue.
                osc = self._ones(Np) if oscale is None else self._const_f32(oscale, Np)
                y2 = self._kernel(kernel)(_pad2d(x2, 1, Kp), wq, scale, osc, 32, bN, blk)[:1, :N]
                return self._epilogue(y2, None, lead, N)
            else:
                # w4a8: per-token e4m3 activation quant + fp4->e4m3 dequant +
                # fp8 WGMMA, K-split into f32 atomic adds on a zeroed output
                # (the AScale divide distributes over it).
                x2 = _pad2d(x2, Mp, Kp)
                xq = torch.empty((Mp, Kp), dtype=torch.float8_e4m3fn, device=self.device)
                ascale = torch.empty((Mp,), dtype=torch.float32, device=self.device)
                self._kernel("quant_fp8")(x2, xq, ascale, 256)
                y2 = torch.zeros((Mp, Np), dtype=torch.float32, device=self.device)
                self._kernel(kernel)(xq, wq, scale, ascale, y2, bM, bN, blk, _THREADS)
                y2 = y2[:M, :N]
        else:
            bM, bN = min(64, M), min(64, N)
            if self.target.startswith("cuda"):
                # WGMMA tiles %16, reduction K %64 (the fp4 dequant K-tile);
                # bM snaps to a warp-partition-valid size (48/80/96/112 fail
                # under Square policy).
                bM, bN = _snap_mma_tile(M, 64), _round_up(bN, 16)
                Mp, Np, Kp = _round_up(M, bM), _round_up(N, bN), _round_up(K, 64)
                x2 = _pad2d(x2, Mp, Kp)
                wq, scale = _pad2d(wq, Np, Kp // 2), _pad2d(scale, Np, Kp // blk)
            y2 = self._kernel("linear_fp4")(x2, wq, scale, bM, bN, blk, _THREADS)[:M, :N]
        return self._epilogue(y2, oscale, lead, N)

    # ------------------------------------------------------------ linear fp8

    def linear_fp8(self, x, w8, wscale, master=None, oscale=None):
        # ``master`` is recording-only (the STE grad lands on it); the kernel
        # uses w8/wscale. Only sm90 has an fp8 kernel — elsewhere the weight is
        # bf16 by now, see :meth:`materialize`.
        lead, K, N = x.shape[:-1], x.shape[-1], w8.shape[0]
        M = x.numel() // K
        plan = self._plan("linear_fp8", M, N, K)
        if plan is None:
            raise NotImplementedError(
                f"linear_fp8 has no kernel in the ({self.precision!r}, {self.arch!r}) cell — "
                "run Backend.materialize on the params at load, it converts fp8 to bf16"
            )
        kernel, Mp, Np, Kp, bM, bN = plan
        # Zero-padding the per-128-block wscale kills the K-tail.
        x2 = _pad2d(self._c(self._dev(x, torch.bfloat16).reshape(M, K)), Mp, Kp)
        w8 = _pad2d(self._dev(w8, w8.dtype), Np, Kp)
        wscale = _pad2d(self._const_f32(wscale), -(-Np // 128), Kp // 128)
        if M == 1:
            # Decode GEMV: stream e4m3 W once (1 byte/elem), bf16 X; oscale folded.
            osc = self._ones(Np) if oscale is None else self._const_f32(oscale, Np)
            y2 = self._kernel(kernel)(x2, w8, wscale, osc, 32, bN)[:1, :N]
            return self._epilogue(y2, None, lead, N)
        else:
            xq = torch.empty((Mp, Kp), dtype=torch.float8_e4m3fn, device=self.device)
            ascale = torch.empty((Mp,), dtype=torch.float32, device=self.device)
            self._kernel("quant_fp8")(x2, xq, ascale, 256)
            y2 = self._kernel(kernel)(xq, w8, wscale, ascale, bM, bN, _THREADS)[:M, :N]
        return self._epilogue(y2, oscale, lead, N)

    def materialize(self, params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert quantized weights to what this cell serves, then migrate.

        fp4 is native on every registered cell; fp8 only on sm90, so elsewhere
        ``.w8/.wscale/.oscale`` collapse into a bf16 weight here — once, at
        wiring time, never as a per-call fallback. It never rewrites a tensor
        that exists: a train step calls this every tick and the optimizer's
        moments are keyed by ``id(param)``.
        """
        out = dict(params)
        if "linear_fp8" not in _resolve(self.precision, self.arch):
            for base in sorted(k[:-3] for k in params if k.endswith(".w8")):
                if base not in out:  # rebuild the weight from the served bytes
                    w = reference.dequant_fp8(params[f"{base}.w8"], params[f"{base}.wscale"])
                    osc = params.get(f"{base}.oscale", torch.ones(1))
                    osc = osc.to(w.device, torch.float32).reshape(-1, 1)
                    out[base] = (w * osc).to(torch.bfloat16)
                for suffix in (".w8", ".wscale", ".oscale"):
                    out.pop(base + suffix, None)
        return {k: v.to(self.device) for k, v in out.items()}

    # ------------------------------------------------------------ attention

    def paged_attention(
        self, q, k_cache, v_cache, block_table, seq_lens, scale, gate=None, seq_q_lens=None
    ):
        squeeze = q.ndim == 3
        if squeeze:
            q = q.unsqueeze(1)  # [B, H, D] -> [B, 1, H, D]
        b, s = q.shape[0], q.shape[1]
        if seq_q_lens is None:
            seq_q_lens = torch.full((b,), s, dtype=torch.int32)
        if self.arch == "sm90":
            # MMA kernel is bf16-IO and tiles queries at block_M: pad S to a
            # multiple (the kernel's history/mask use the true per-row lengths
            # in seq_q_lens, so padding rows do not shift the causal window).
            block_m = 64 if s >= 64 else 16
            q = self._dev(self._c(q), torch.bfloat16)
            pad = -s % block_m
            if pad:
                q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad))
            out = self._kernel("paged_attention")(
                q,
                self._dev(k_cache, torch.bfloat16),
                self._dev(v_cache, torch.bfloat16),
                self._i32(block_table),
                self._i32(seq_lens),
                self._i32(seq_q_lens),
                float(scale),
                int(k_cache.shape[2]),
                block_m,
                128,
            )[:, :s]
        else:
            q = self._f32(q)
            k_cache = self._f32(k_cache)
            v_cache = self._f32(v_cache)
            bt = self._i32(block_table).contiguous()
            sl = self._i32(seq_lens).contiguous()
            sql = self._i32(seq_q_lens).contiguous()
            k = self._kernel("paged_attention")
            out = k(
                q,
                k_cache,
                v_cache,
                bt,
                sl,
                sql,
                float(scale),
                block_size=int(k_cache.shape[2]),
                threads=_THREADS,
            )
        out = out.squeeze(1) if squeeze else out
        if gate is not None:
            out = out * torch.sigmoid(self._dev(gate, out.dtype))
        return out

    def attention(self, q, k, v, scale, gate=None):
        """Dense causal GQA attention (training path). q/k/v [B,T,H,D]."""
        # ponytail: torch-eager forward, tilelang kernel when perf demands
        out = reference.dense_attention(q, k, v, float(scale))
        if gate is not None:
            out = out * torch.sigmoid(self._f32(gate))
        return out

    # ------------------------------------------------------------ write tokens

    def write_tokens(self, k, v, kv, layer_idx):
        """Scatter k/v [B,T,Hkv,D] into the paged pool at [seq_len-T, seq_len).

        sm90: one capturable kernel — the host loop it replaces syncs GPU->CPU
        per token (block table / seq_len are device tensors) and cannot sit
        inside a captured decode tick. Other arches: the pool's torch-loop
        fallback (the dev/parity path).
        """
        if "write_tokens" not in _resolve(self.precision, self.arch):
            kv.kv_pool.write_tokens(k, v, kv, layer_idx)
            return
        pool = kv.kv_pool
        k_plane, v_plane = pool.kv_layer(layer_idx)
        b, s = k.shape[0], k.shape[1]
        sql = getattr(kv, "seq_q_lens", None)
        if sql is None:
            sql = torch.full((b,), s, dtype=torch.int32)
        # .contiguous(): the ABI is packed. A bf16 view (e.g. v sliced from
        # the fused-qkv GEMV output) survives _dev's no-op cast and violates
        # it at B>=2; the f32 WGMMA path's cast already copied.
        self._kernel("write_tokens")(
            self._dev(k, torch.bfloat16).contiguous(),
            self._dev(v, torch.bfloat16).contiguous(),
            k_plane,
            v_plane,
            self._i32(kv.block_table).contiguous(),
            self._i32(kv.seq_len).contiguous(),
            self._i32(sql).contiguous(),
            int(pool.k_pool.shape[-2]),
            _THREADS,
        )

    def attn_prep(self, qkv, wq, wk, positions, theta, rotary_dim, kv, layer_idx, hq, hkv, eps):
        """Fused q/k norm + RoPE + K/V pool write off the fused-qkv output.
        Returns normalized+rotated q [B,S,hq,D] (bf16). None if the arch has
        no fused kernel — caller takes the unfused path."""
        if "attn_prep" not in _resolve(self.precision, self.arch):
            return None
        pool = kv.kv_pool
        k_plane, v_plane = pool.kv_layer(layer_idx)
        b, s = qkv.shape[0], qkv.shape[1]
        sql = getattr(kv, "seq_q_lens", None)
        if sql is None:
            sql = torch.full((b,), s, dtype=torch.int32)
        pos = self._i32(positions)
        if pos.ndim == 1:
            pos = pos.unsqueeze(0).expand(b, -1)
        return self._kernel("attn_prep")(
            self._dev(qkv, torch.bfloat16).contiguous(),
            self._f32(wq).contiguous(),
            self._f32(wk).contiguous(),
            pos.contiguous(),
            self._inv_freq(int(rotary_dim), float(theta)).to(self.device),
            k_plane,
            v_plane,
            self._i32(kv.block_table).contiguous(),
            self._i32(kv.seq_len).contiguous(),
            self._i32(sql).contiguous(),
            float(eps),
            int(hq),
            int(hkv),
            int(pool.k_pool.shape[-2]),
            _THREADS,
        )

    def attention_bwd(self, grad, q, k, v, scale):
        # ponytail: torch-eager backward, tilelang kernel when perf demands
        return reference.dense_attention_bwd(grad, q, k, v, float(scale))

    def attention_gate_bwd(self, grad, attn_out, gate):
        return reference.attention_gate_bwd(grad, attn_out, gate)

    # ------------------------------------------------------------ gated delta

    def linear_attn_chunk(self, q, k, v, g, beta, state, **kw):
        """Full-GDN layer core. sm90: T=1 uses the fused decode kernel, T>1 the
        fused chunk kernel; other arches use the torch-eager reference."""
        _kset = _resolve(self.precision, self.arch)
        if q.shape[1] > 1 and "gdn_chunk_fused" in _kset:
            return self._gdn_chunk_fused(q, k, v, g, beta, state, **kw)
        return reference.gdn_forward(q, k, v, g, beta, state, **kw)

    def gdn_decode(self, q, k, v, g, beta, pool, slots, layer, **kw):
        """Serving decode (T=1) GDN core, one launch, state updated IN PLACE in
        ``pool.states[slots, layer]`` (sm90 only — returns None elsewhere and the
        caller takes the tape-recorded gather -> linear_attn_chunk -> scatter
        path). The conv window is gathered/scattered (its q/k columns are
        shared across blocks, so it cannot be shifted in place)."""
        if "gdn_decode_fused" not in _resolve(self.precision, self.arch):
            return None
        slots_i = self._i32(slots).contiguous()
        window = self._c(self._f32(pool.conv_windows[slots.long(), layer]))
        out, new_window = self._kernel("gdn_decode_fused")(
            self._c(self._f32(q).squeeze(1)),
            self._c(self._f32(k).squeeze(1)),
            self._c(self._f32(v).squeeze(1)),
            self._c(self._f32(kw["z"]).squeeze(1)),
            self._c(self._bf16(g).squeeze(1)),
            self._c(self._bf16(beta).squeeze(1)),
            self._c(self._const_f32(kw["dt_bias"])),
            self._c(self._const_f32(kw["a_log"])),
            self._c(self._const_f32(kw["norm_weight"])),
            self._c(self._const_f32(kw["conv1d_weight"])),
            window,
            pool.states,
            slots_i,
            int(layer),
            threads=pool.states.shape[-1],
        )
        pool.conv_windows[slots.long(), layer] = new_window.to(pool.conv_windows.dtype)
        return out.unsqueeze(1)

    def _gdn_chunk_fused(self, q, k, v, g, beta, state, **kw):
        """Fused GDN chunk prefill (T>1): one launch for the whole layer core.

        q/k/v/g/beta [B, T, ...] keep their T dim (the kernel scans it
        serially per (value head, batch)). conv_window is always a tensor
        (the model carries it; None means zero left-padding). ``seq_q_lens``
        [B] (when present) bounds the per-row scan: mixed batches pad decode
        rows to the chunk's T with a per-row bound of 1.
        """
        window = kw.get("conv_window")
        has_window = window is not None
        if not has_window:
            window = torch.zeros(
                q.shape[0],
                kw["conv1d_weight"].shape[1] - 1,
                q.shape[-1] + k.shape[-1] + v.shape[-1],
                dtype=torch.float32,
                device=self.device,
            )
        seq_q = kw.get("seq_q_lens")
        if seq_q is None:
            seq_q = torch.full((q.shape[0],), q.shape[1], dtype=torch.int32)
        out, new_state, new_window = self._kernel("gdn_chunk_fused")(
            self._c(self._bf16(q)),
            self._c(self._bf16(k)),
            self._c(self._bf16(v)),
            self._c(self._bf16(kw["z"])),
            self._c(self._f32(g)),
            self._c(self._f32(beta)),
            self._c(self._const_f32(kw["dt_bias"])),
            self._c(self._const_f32(kw["a_log"])),
            self._c(self._const_f32(kw["norm_weight"])),
            self._c(self._const_f32(kw["conv1d_weight"])),
            self._c(self._bf16(window)),
            self._c(self._f32(state)),
            self._c(self._i32(seq_q)),
            threads=state.shape[-1],
        )
        return out, new_state, (self._f32(new_window) if has_window else None)

    def linear_attn_bwd(self, grad, q, k, v, g, beta, state, **kw):
        # ponytail: torch-eager backward (gdn example_chunk_delta_bwd is the
        # CUDA-scheduled tilelang upgrade path), tilelang kernel when perf demands
        kw.pop("seq_q_lens", None)  # serving-only; training batches are uniform T
        return reference.linear_attn_bwd(grad, q, k, v, g, beta, state, **kw)

    def state_gather(self, states, windows, slots, layer_idx):
        return reference.state_gather(states, windows, slots, layer_idx)

    def state_scatter(self, states, windows, slots, layer_idx, new_state, new_window):
        reference.state_scatter(states, windows, slots, layer_idx, new_state, new_window)

    # ------------------------------------------------------------ silu mul

    def silu_mul(self, gate, up):
        shape = gate.shape
        io = torch.bfloat16 if self.arch == "sm90" else torch.float32  # sm90 kernel is bf16-IO
        gate = self._c(self._dev(gate, io).reshape(-1))
        up = self._c(self._dev(up, io).reshape(-1))
        y = self._kernel("silu_mul")(gate, up, 1024, _THREADS)
        return y.reshape(shape)

    def silu_mul_bwd(self, grad, gate, up):
        # ponytail: torch-eager backward, tilelang kernel when perf demands
        return reference.silu_mul_bwd(grad, gate, up)

    # ------------------------------------------------------------ softmax

    def softmax(self, x, axis):
        x = self._f32(x)
        if axis != -1 and axis != x.ndim - 1:
            x = x.movedim(axis, -1)
            y = self._kernel("softmax")(x.reshape(-1, x.shape[-1]), threads=_THREADS).reshape(
                x.shape
            )
            return y.movedim(-1, axis)
        k = self._kernel("softmax")
        return k(x.reshape(-1, x.shape[-1]), threads=_THREADS).reshape(x.shape)

    def cross_entropy_loss_grad(self, logits, input_ids):
        # ponytail: torch-eager training loss, tilelang kernel when perf demands
        return reference.cross_entropy_loss_grad(logits, input_ids)

    # ------------------------------------------------------------ embedding

    def _const_f32(self, t, pad_to: int | None = None):
        """f32 cast of a PARAMETER (norm weight, scale, GDN vector, embedding
        table), cached across ticks; optionally zero-padded to ``pad_to`` rows.
        Per-call casts were ~110 launches per 8 decode layers (the 27B embedding
        alone ~6ms/tick). The optimizer's in-place copy_ bumps _version and
        invalidates the entry. Identity is checked via weakref: a fresh model
        can reuse a freed tensor's address, and data_ptr alone would hand back
        the stale cast. Never call this on an activation."""
        if t.dtype == torch.float32 and t.device == self.device and pad_to is None:
            return t
        key = (t.data_ptr(), pad_to)
        hit = self._const_f32_cache.get(key)
        if hit is not None:
            ref, ver, c = hit
            if ref() is not t:
                del self._const_f32_cache[key]  # address reused by a different tensor
            elif ver == t._version:
                return c
        c = self._f32(t)
        if pad_to is not None and pad_to != c.shape[0]:
            c = torch.nn.functional.pad(c, (0, pad_to - c.shape[0]))
        self._const_f32_cache[key] = (weakref.ref(t), t._version, c)
        return c

    def _ones(self, n: int):
        t = self._ones_cache.get(n)
        if t is None:
            t = self._ones_cache[n] = torch.ones(n, dtype=torch.float32, device=self.device)
        return t

    def embedding(self, idx, table):
        table = self._const_f32(table)
        idx_flat = self._i32(idx).reshape(-1).contiguous()
        k = self._kernel("embedding")
        y = k(idx_flat, table, threads=_THREADS)
        return y.reshape(*idx.shape, table.shape[-1])

    def embedding_bwd(self, grad, idx, num_rows):
        # ponytail: torch-eager backward, tilelang kernel when perf demands
        return reference.embedding_bwd(grad, idx, num_rows)

    # ------------------------------------------------------------ sampling

    def sample(self, logits, temperature, top_p, seed):
        # ponytail: torch-eager sampling, tilelang sample kernel when perf demands
        return reference.sample(logits, temperature, top_p, seed)

    def sample_batch(self, logits, temperatures, top_ps, seeds):
        return reference.sample_batch(logits, temperatures, top_ps, seeds)


_BACKEND: Backend | None = None


def get_backend() -> Backend:
    """Process-wide backend singleton (target resolved once)."""
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = Backend(resolve_target())
    return _BACKEND
