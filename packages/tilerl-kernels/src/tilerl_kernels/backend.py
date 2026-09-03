"""TileLang backend: the op interface every layer above calls. Forward ops
run the kernels the registry cell resolves; backward ops without a kernel run
the torch-eager reference.
# ponytail: torch-eager backward, tilelang kernel when perf demands

sm90 kernels are bf16-IO WGMMA tiles (M/N multiples of 16, K of 32); this
module pads tails so a kernel always sees exact tiles. CPU/metal run the f32
kernels. Eager JIT calls NVCC per shape (30-120s cold; ~0.2s from
``TILELANG_CACHE_DIR``, point it at persistent storage on the pod).
"""

from __future__ import annotations

import os
import weakref
from typing import Any

import torch

from . import kernels_linear
from . import reference
from .registry import _arch_for, _resolve, resolve_target

__all__ = ["Backend", "get_backend", "resolve_target"]

_THREADS = 64
#: chunkwise-WY chunk length. gdn_state_scan and gdn_chunk_o size h by S // chunk, so a T
#: that is not a whole multiple writes past it. A verify width that reached one would also
#: reach ``_full_rows``'s host sync, illegal under graph capture: keep _MAX_VERIFY_W below.
_WY_CHUNK = 64


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _snap_mma_tile(m: int, cap: int) -> int:
    # WGMMA Square policy: 16/32/64/128 compile, 48/80/96/112 do not
    return min(cap, next((s for s in (16, 32, 64, 128) if s >= m), 128))


def _pad2d(t: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    pr, pc = rows - t.shape[0], cols - t.shape[1]
    if pr == 0 and pc == 0:
        return t
    return torch.nn.functional.pad(t, (0, pc, 0, pr))


def _pad1d(t: torch.Tensor, n: int) -> torch.Tensor:
    p = n - t.shape[0]
    return t if p == 0 else torch.nn.functional.pad(t, (0, p))


_MX = 8  # mma8 row cap: decode rows on the tensor cores
#: rows up to which the M-row GEMV beats mma8 (27B decode replay, H20, ms:
#: gemv 11.2/17.5/27.1/30.1 at M=1..4, mma8 27 flat); TILERL_MGEMV=0 disables
_MGEMV = int(os.environ.get("TILERL_MGEMV", "3"))
#: every kernel that loops `X // _RED_TILE` floor-divides, so a reduction
#: dim padded to anything else drops its tail without a word: at 32 rows
#: under TILERL_RED_TILE=64 the weight gradient came out exactly zero.
_MMA_RED = kernels_linear._RED_TILE

#: CUDA linear family: (op, M-regime) -> (kernel, K pad, N cap, N tile).
#: Crossovers measured in wins/2026-08-26-batch-decode-h2.md. The fp4->e4m3
#: arms tile N at 64 because the kernel overrides block_N to _FP4_BLOCK_N.
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
    """Resolved tilelang target plus lazily-compiled kernels."""

    name = "tilelang"
    tp_world = 1
    tp_rank = 0

    def init_tp(self, world: int, rank: int) -> None:
        """Join the TP group; framework layers never import torch.distributed."""
        if world == 1:
            return
        import torch.distributed as dist

        if not dist.is_initialized():
            comm = "nccl" if self.device.type == "cuda" else "gloo"
            dist.init_process_group(comm, world_size=world, rank=rank)
        self.tp_world, self.tp_rank = world, rank

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Sum across the TP group in place. 21.5 us floor per call on 6 H20s,
        flat from 20 KB to 1.3 MB: the cost is per layer, not per byte."""
        if self.tp_world == 1:
            return x
        import torch.distributed as dist

        dist.all_reduce(x)
        return x

    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Concatenate from every rank along ``dim`` (vocab-parallel lm_head)."""
        if self.tp_world == 1:
            return x
        import torch.distributed as dist

        x = x.contiguous()
        parts = [torch.empty_like(x) for _ in range(self.tp_world)]
        dist.all_gather(parts, x)
        return torch.cat(parts, dim=dim)

    def __init__(self, target: str):
        self.target = target
        if target == "metal":
            self.device = torch.device("mps")
        elif target.startswith("cuda"):
            # index None is not the device kernel outputs land on
            self.device = torch.device("cuda", torch.cuda.current_device())
            # fp32 matmuls in the eager backward were 23% of a train step on SIMT cores
            tf32 = os.environ.get("TILERL_TF32", "1") == "1"
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32
        else:
            self.device = torch.device("cpu")
        self.precision = "bf16"
        self.arch = _arch_for(target)
        self._kernels: dict[str, object] = {}
        self._inv_freq_cache: dict[tuple[int, float], torch.Tensor] = {}
        self._const_f32_cache: dict[tuple[int, int | None], tuple[Any, int, torch.Tensor]] = {}
        self._ones_cache: dict[int, torch.Tensor] = {}
        self._step_scratch: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._full_rows_memo: tuple[Any, int, bool] | None = None
        self._sms = (
            torch.cuda.get_device_properties(self.device).multi_processor_count
            if self.device.type == "cuda"
            else 1
        )

    def _kernel(self, name: str, *args):
        """``args`` are factory (compile-time variant) arguments; they key the cache."""
        k = self._kernels.get((name, args))
        if k is None:
            k = _resolve(self.precision, self.arch)[name](self.target, *args)
            self._kernels[(name, args)] = k
        return k

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _c(t: torch.Tensor) -> torch.Tensor:
        return t if t.is_contiguous() else t.contiguous()

    def _dev(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Cast and migrate at the tilelang boundary (callers may pass CPU tensors)."""
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
            # on device: a CPU->CUDA copy is illegal inside a captured decode tick
            inv = 1.0 / (
                theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=self.device) / d)
            )
            self._inv_freq_cache[key] = inv
        return inv

    def _rows(self, x: torch.Tensor):
        io = torch.bfloat16 if self.target.startswith("cuda") else torch.float32
        return x.shape[:-1], self._c(self._dev(x, io).reshape(-1, x.shape[-1]))

    def _epilogue(self, y2, oscale, lead, n: int):
        # ponytail: torch epilogue for the per-row scale, fold into the kernel if a sweep says so
        return (y2 if oscale is None else y2 * self._const_f32(oscale)).reshape(*lead, n)

    def _plan(self, op: str, m: int, n: int, k: int):
        """(kernel, Mp, Np, Kp, block_M, block_N), or None when this cell has no
        specialized kernel."""
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
        return a + b

    # ------------------------------------------------------------ rmsnorm

    def rmsnorm(self, x, w, eps):
        x = self._f32(x)
        w = self._const_f32(w)
        lead = x.shape[:-1]
        x2 = self._c(x.reshape(-1, x.shape[-1]))
        N = x2.shape[-1]
        if "rmsnorm_fused" in _resolve(self.precision, self.arch):
            y = self._kernel("rmsnorm_fused")(x2, w, float(eps), 256)
            return y.reshape(*lead, w.shape[0])
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
        # ponytail: torch-eager backward, tilelang kernel when perf demands
        return reference.rope_bwd(grad, positions, theta, rotary_dim=rotary_dim)

    # ------------------------------------------------------------ linear

    #: ``residual=`` is serving-only: the tape records backend.add instead
    fuses_residual = True

    def linear(self, x, w, bias=None, residual=None):
        lead, x2 = self._rows(x)
        w = self._dev(w, x2.dtype)
        M, K, N = x2.shape[0], x2.shape[1], w.shape[0]
        plan = self._plan("linear", M, N, K)
        if plan is not None:  # decode GEMV
            kernel, _, Np, Kp, _, bN = plan
            y = self._kernel(kernel)(_pad2d(x2, 1, Kp), _pad2d(w, Np, Kp), 32, bN)[:1, :N]
            y = (y if bias is None else y + self._f32(bias)).reshape(*lead, N)
            return y if residual is None else y + residual
        bM, bN = min(64, M), min(64, N)
        bias = (
            torch.zeros(N, dtype=torch.float32, device=self.device)
            if bias is None
            else self._f32(bias)
        )
        if self.target.startswith("cuda"):
            bM, bN = _snap_mma_tile(bM, 64), _snap_mma_tile(bN, 64)
            x2 = _pad2d(x2, _round_up(M, bM), _round_up(K, _MMA_RED))
            w = _pad2d(w, _round_up(N, bN), _round_up(K, _MMA_RED))
            bias = _pad1d(bias, w.shape[0])
        y = self._kernel("gemm_nt")(x2, w, bias, bM, bN, _THREADS)
        y = y[:M, :N].reshape(*lead, N)
        return y if residual is None else y + residual

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
            bM, bK = _snap_mma_tile(min(64, M), 64), _snap_mma_tile(min(64, K), 64)
            gx = self._kernel("gemm_nn")(
                _pad2d(g2, _round_up(M, bM), _round_up(N, _MMA_RED)),
                _pad2d(w, _round_up(N, _MMA_RED), _round_up(K, bK)),
                bM,
                bK,
                _THREADS,
            )[:M, :K]
            # gw = g2.T @ x2 (gemm_tn): reduction M, output [N, K]
            bN = _snap_mma_tile(min(64, N), 64)
            gw = self._kernel("gemm_tn")(
                _pad2d(g2, _round_up(M, _MMA_RED), _round_up(N, bN)),
                _pad2d(x2, _round_up(M, _MMA_RED), _round_up(K, bK)),
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
        """sm90 reads the twiddled layout: rewritten in place once and flagged,
        so graph capture and save_hf (which untwiddles by the flag) see one truth."""
        wq = self._dev(wq, wq.dtype)
        if "linear_fp4_gemv" in _resolve(self.precision, self.arch) and not getattr(
            wq, "_tl_twiddled", False
        ):
            wq.copy_(reference.twiddle_fp4(wq))
            wq._tl_twiddled = True
        return wq

    def linear_fp4(self, x, wq, scale, master=None, oscale=None, residual=None):
        # ``master`` is recording-only (the STE grad lands on it)
        wq = self._served_fp4(wq)
        scale = self._f32(scale)
        lead, x2 = self._rows(x)
        M, K, N = x2.shape[0], x2.shape[1], wq.shape[0]
        blk = K // scale.shape[1]  # the checkpoint's scale block (16 or 32)
        # M-row GEMV on the M=1 plan (the decode plan's n_partition is a 4096-thread block)
        if 2 <= M <= _MGEMV and (gp := self._plan("linear_fp4", 1, N, K)) is not None:
            gk, _, gNp, gKp, _, gbN = gp
            gwq, gsc = _pad2d(wq, gNp, gKp // 2), _pad2d(scale, gNp, gKp // blk)
            osc = self._ones(gNp) if oscale is None else self._const_f32(oscale, gNp)
            res = self._residual(residual, N, gNp, rows=M)
            y2 = self._kernel(gk, M)(
                _pad2d(x2, M, gKp), gwq, gsc, osc,
                res if res is not None else self._residual(None, N, gNp, rows=M),
                32, gbN, blk,
            )[:M, :N]
            y = y2.reshape(*lead, N)
            return y if res is not None else y + residual
        plan = self._plan("linear_fp4", M, N, K)
        if plan is not None:
            kernel, Mp, Np, Kp, bM, bN = plan
            wq, scale = _pad2d(wq, Np, Kp // 2), _pad2d(scale, Np, Kp // blk)
            # M=1 stays on the GEMV: mma8 measured 2.2x slower there (39.9 vs 87 tok/s)
            if 2 <= M <= _MX and "linear_fp4_mma8" in _resolve(self.precision, self.arch):
                Np32 = _round_up(N, 32)
                wq, scale = _pad2d(wq, Np32, Kp // 2), _pad2d(scale, Np32, Kp // blk)
                osc = self._ones(Np32) if oscale is None else self._const_f32(oscale, Np32)
                xm = _pad2d(x2, _MX, Kp)
                res = None if residual is None else self._f32(residual).reshape(M, N)
                r2 = self._zeros2(_MX, Np32) if res is None or Np32 != N else _pad2d(res, _MX, N)
                y2 = self._kernel("linear_fp4_mma8")(xm, wq, scale, osc, r2, blk)[:M, :N]
                y = y2.reshape(*lead, N)
                return y if res is None or r2.shape[1] == N else y + residual
            if M == 1:
                osc = self._ones(Np) if oscale is None else self._const_f32(oscale, Np)
                res = self._residual(residual, N, Np)
                if res is not None:
                    y2 = self._kernel(kernel)(_pad2d(x2, 1, Kp), wq, scale, osc, res, 32, bN, blk)
                    return y2[:1, :N].reshape(*lead, N)
                y2 = self._kernel(kernel)(
                    _pad2d(x2, 1, Kp), wq, scale, osc, self._residual(None, N, Np), 32, bN, blk
                )[:1, :N]
                return self._epilogue(y2, None, lead, N) + residual
            else:
                # w4a8: per-token e4m3 activation quant, K-split atomic adds on a zeroed Y
                x2 = _pad2d(x2, Mp, Kp)
                xq = torch.empty((Mp, Kp), dtype=torch.float8_e4m3fn, device=self.device)
                ascale = torch.empty((Mp,), dtype=torch.float32, device=self.device)
                self._kernel("quant_fp8")(x2, xq, ascale, 256)
                y2 = torch.zeros((Mp, Np), dtype=torch.float32, device=self.device)
                self._kernel(kernel)(xq, wq, scale, ascale, y2, bM, bN, blk, _THREADS)
                y2 = y2[:M, :N]
        else:
            bM, bN = min(64, M), min(64, N)
            if self.target.startswith("cuda"):  # K pads to the fp4 dequant K-tile
                bM, bN = _snap_mma_tile(M, 64), _round_up(bN, 16)
                Mp, Np, Kp = _round_up(M, bM), _round_up(N, bN), _round_up(K, 64)
                x2 = _pad2d(x2, Mp, Kp)
                wq, scale = _pad2d(wq, Np, Kp // 2), _pad2d(scale, Np, Kp // blk)
            y2 = self._kernel("linear_fp4")(x2, wq, scale, bM, bN, blk, _THREADS)[:M, :N]
        y = self._epilogue(y2, oscale, lead, N)
        return y if residual is None else y + residual

    # ------------------------------------------------------------ linear fp8

    def linear_fp8(self, x, w8, wscale, master=None, oscale=None, residual=None):
        # only sm90 has an fp8 kernel; elsewhere materialize() made the weight bf16
        lead, K, N = x.shape[:-1], x.shape[-1], w8.shape[0]
        M = x.numel() // K
        plan = self._plan("linear_fp8", M, N, K)
        if plan is None:
            raise NotImplementedError(
                f"linear_fp8 has no kernel in the ({self.precision!r}, {self.arch!r}) cell — "
                "run Backend.materialize on the params at load, it converts fp8 to bf16"
            )
        if 2 <= M <= _MGEMV and (gp := self._plan("linear_fp8", 1, N, K)) is not None:
            gk, _, gNp, gKp, _, gbN = gp
            osc = self._ones(gNp) if oscale is None else self._const_f32(oscale, gNp)
            res = self._residual(residual, N, gNp, rows=M)
            y2 = self._kernel(gk, M)(
                _pad2d(self._c(self._dev(x, torch.bfloat16).reshape(M, K)), M, gKp),
                _pad2d(self._dev(w8, w8.dtype), gNp, gKp),
                _pad2d(self._const_f32(wscale), -(-gNp // 128), gKp // 128), osc,
                res if res is not None else self._residual(None, N, gNp, rows=M), 32, gbN,
            )[:M, :N]
            y = y2.reshape(*lead, N)
            return y if res is not None else y + residual
        kernel, Mp, Np, Kp, bM, bN = plan
        x2 = _pad2d(self._c(self._dev(x, torch.bfloat16).reshape(M, K)), Mp, Kp)
        w8 = _pad2d(self._dev(w8, w8.dtype), Np, Kp)
        wscale = _pad2d(self._const_f32(wscale), -(-Np // 128), Kp // 128)
        if 2 <= M <= _MX and "linear_fp8_mma8" in _resolve(self.precision, self.arch):
            Np32 = _round_up(N, 32)
            w8 = _pad2d(w8, Np32, Kp)
            wscale = _pad2d(wscale, -(-Np32 // 128), Kp // 128)
            osc = self._ones(Np32) if oscale is None else self._const_f32(oscale, Np32)
            xm = _pad2d(x2, _MX, Kp)
            res = None if residual is None else self._f32(residual).reshape(M, N)
            r2 = self._zeros2(_MX, Np32) if res is None or Np32 != N else _pad2d(res, _MX, N)
            y2 = self._kernel("linear_fp8_mma8")(xm, w8, wscale, osc, r2)[:M, :N]
            y = y2.reshape(*lead, N)
            return y if res is None or r2.shape[1] == N else y + residual
        if M == 1:
            osc = self._ones(Np) if oscale is None else self._const_f32(oscale, Np)
            res = self._residual(residual, N, Np)
            if res is not None:
                return self._kernel(kernel)(x2, w8, wscale, osc, res, 32, bN)[:1, :N].reshape(*lead, N)
            y2 = self._kernel(kernel)(x2, w8, wscale, osc, self._residual(None, N, Np), 32, bN)[:1, :N]
            return self._epilogue(y2, None, lead, N) + residual
        else:
            xq = torch.empty((Mp, Kp), dtype=torch.float8_e4m3fn, device=self.device)
            ascale = torch.empty((Mp,), dtype=torch.float32, device=self.device)
            self._kernel("quant_fp8")(x2, xq, ascale, 256)
            y2 = self._kernel(kernel)(xq, w8, wscale, ascale, bM, bN, _THREADS)[:M, :N]
        y = self._epilogue(y2, oscale, lead, N)
        return y if residual is None else y + residual

    def materialize(self, params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert quantized weights to what this cell serves, then migrate:
        off sm90, .w8/.wscale/.oscale collapse into a bf16 weight once at wiring.
        Never rewrites an existing tensor: optimizer moments are keyed by id(param)."""
        out = dict(params)
        if "linear_fp8" not in _resolve(self.precision, self.arch):
            for base in sorted(k[:-3] for k in params if k.endswith(".w8")):
                if base not in out:
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
        if self.arch == "sm90" and s == 1 and "paged_attention_decode" in _resolve(self.precision, self.arch):
            out = self._paged_attention_decode(q, k_cache, v_cache, block_table, seq_lens, scale)
        elif self.arch == "sm90":
            # pad S to block_M; seq_q_lens keeps the causal window on the true lengths
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

    def _paged_attention_decode(self, q, k_cache, v_cache, block_table, seq_lens, scale):
        b, h, d = q.shape[0], q.shape[2], q.shape[3]
        hkv = k_cache.shape[1]
        # split count from the pool's reach (host-static, graph-safe): 64 splits
        # past 64K tokens or when the 16-grid under-fills the SMs
        max_tokens = block_table.shape[1] * k_cache.shape[2]
        wide = 16 * hkv * b < 2 * self._sms and max_tokens >= 64 * k_cache.shape[2]
        ks, sfx = (64, "_64") if (max_tokens > 65536 or wide) else (16, "")
        key = ("attn_ws", b, hkv, d, ks)
        ws = self._ones_cache.get(key)
        if ws is None:  # static workspace: graph-capturable
            ws = self._ones_cache[key] = (
                torch.empty(b, hkv, ks, 16, d, dtype=torch.float32, device=self.device),
                torch.empty(b, hkv, ks, 16, dtype=torch.float32, device=self.device),
                torch.empty(b, hkv, ks, 16, dtype=torch.float32, device=self.device),
            )
        po, pm, pl = ws
        self._kernel("paged_attention_decode" + sfx)(
            self._dev(self._c(q.reshape(b, h, d)), torch.bfloat16),
            self._dev(k_cache, torch.bfloat16), self._dev(v_cache, torch.bfloat16),
            self._i32(block_table), self._i32(seq_lens), po, pm, pl, float(scale),
            int(k_cache.shape[2]),
        )
        return self._kernel("paged_attention_combine" + sfx)(po, pm, pl, h // hkv).reshape(b, 1, h, d)

    def attention(self, q, k, v, scale, gate=None):
        """Dense causal GQA attention (training path). q/k/v [B,T,H,D]."""
        # ponytail: torch-eager forward, tilelang kernel when perf demands
        out = reference.dense_attention(q, k, v, float(scale))
        if gate is not None:
            out = out * torch.sigmoid(self._f32(gate))
        return out

    # ------------------------------------------------------------ write tokens

    def write_tokens(self, k, v, kv, layer_idx):
        """Scatter k/v [B,T,Hkv,D] into the paged pool: one capturable kernel on
        sm90, the pool's torch loop elsewhere."""
        if "write_tokens" not in _resolve(self.precision, self.arch):
            kv.kv_pool.write_tokens(k, v, kv, layer_idx)
            return
        pool = kv.kv_pool
        k_plane, v_plane = pool.kv_layer(layer_idx)
        b, s = k.shape[0], k.shape[1]
        sql = getattr(kv, "seq_q_lens", None)
        if sql is None:
            sql = torch.full((b,), s, dtype=torch.int32)
        # .contiguous(): a bf16 view sliced from the fused-qkv output survives _dev's no-op cast
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
        """Fused q/k norm + RoPE + K/V pool write; returns q [B,S,hq,D] bf16,
        or None when the arch has no fused kernel."""
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
            self._f32(qkv).contiguous(),
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

    def linear_frozen_bwd(self, grad, wq, scale, oscale=None, fp8=False):
        """dX through a frozen quantized weight, no weight grad. The kernel
        dequantizes in one pass; the reference costs ~1.5 GB of temporaries
        per call, 448 calls a step."""
        kset = _resolve(self.precision, self.arch)
        # ponytail: the kernel bakes scale block 16 (every shipped checkpoint);
        # register a second kernel if a 32 ever ships
        blk = wq.shape[1] * 2 // self._f32(scale).shape[1]
        if not fp8 and blk == 16 and "linear_fp4_bwd" in kset:
            wq = self._served_fp4(wq)
            n, k = wq.shape[0], wq.shape[1] * 2
            g = self._bf16(grad).reshape(-1, grad.shape[-1])
            if oscale is not None:  # scales weight row n: fold into [M,N], not [N,K]
                g = g * self._bf16(oscale).reshape(1, -1)
            m = g.shape[0]
            bM, bN = _snap_mma_tile(min(64, m), 64), 64  # bN=128 measured 0.962x
            gx = self._kernel("linear_fp4_bwd")(
                _pad2d(self._c(g), _round_up(m, bM), _round_up(n, _MMA_RED)),
                _pad2d(wq, _round_up(n, _MMA_RED), _round_up(k, bN) // 2),
                _pad2d(self._const_f32(scale), _round_up(n, _MMA_RED),
                       _round_up(k, bN) // blk),
                bM, bN, _THREADS,
            )[:m, :k]
            return gx.reshape(*grad.shape[:-1], k)
        # ponytail: torch-eager backward, tilelang dequant only exists for fp4
        return reference.linear_frozen_bwd(grad, wq, scale, oscale=oscale, fp8=fp8)

    def attention_bwd(self, grad, q, k, v, scale):
        # ponytail: torch-eager backward, tilelang kernel when perf demands
        return reference.dense_attention_bwd(grad, q, k, v, float(scale))

    def attention_gate_bwd(self, grad, attn_out, gate):
        return reference.attention_gate_bwd(grad, attn_out, gate)

    # ------------------------------------------------------------ gated delta

    def linear_attn_chunk(self, q, k, v, g, beta, state, **kw):
        """Full-GDN layer core: chunkwise-WY for full-length rows, else the
        fused serial kernel on sm90, else the per-step reference."""
        kset = _resolve(self.precision, self.arch)
        t, chunkable = q.shape[1], q.shape[1] > 1 and not kw.get("keep_steps")
        # A/B lever: the chunkwise reference, all bmm + a triangular solve (or fla)
        ref_chunk = int(os.environ.get("TILERL_GDN_CHUNKWISE", "0"))
        if ref_chunk and chunkable:
            return reference.gdn_forward(q, k, v, g, beta, state, chunkwise=ref_chunk, **kw)
        # the WY kernels scan whole chunks; a ragged length keeps the serial kernel
        wy = "gdn_state_scan" not in kset or t % _WY_CHUNK == 0
        if wy and chunkable and self._full_rows(kw.get("seq_q_lens"), t):
            return self._gdn_chunk_wy(q, k, v, g, beta, state, **kw)
        if t > 1 and "gdn_chunk_fused" in kset:
            return self._gdn_chunk_fused(q, k, v, g, beta, state, **kw)
        return reference.gdn_forward(q, k, v, g, beta, state, **kw)

    def _full_rows(self, seq_q_lens, t: int) -> bool:
        """Every row's query span is the whole T -- the chunkwise form has no
        per-row mask. ``.min()`` is a host sync and one tensor object reaches all
        64 layers, so without the memo a prefill tick pays 64 pipeline drains."""
        if seq_q_lens is None:
            return True
        hit = self._full_rows_memo
        if hit is not None and hit[0]() is seq_q_lens and hit[1] == t:
            return hit[2]
        full = int(seq_q_lens.min()) == t
        self._full_rows_memo = (weakref.ref(seq_q_lens), t, full)
        return full

    def _gdn_prep(self, q, k, v, g, beta, state, **kw):
        """``gdn_prep``'s six operands, marshalled -- the parity oracle is
        :func:`reference.gdn_prep`. The sm90 cell binds one thread to a head
        column of q, k and v alike, so it needs key_dim == value_dim; the f32
        cell loops each extent separately and cannot fail the same way."""
        b, t = q.shape[0], q.shape[1]
        nvh, kd, vd = state.shape[1], state.shape[2], state.shape[3]
        assert kd == vd, f"gdn_prep binds one thread per head column: {kd} != {vd}"
        hk = q.shape[-1] // kd
        ker = kw["conv1d_weight"].shape[1]
        io = torch.bfloat16 if self.target.startswith("cuda") else torch.float32
        window = kw.get("conv_window")
        win = (
            self._f32(window)
            if window is not None
            else torch.zeros(b, ker - 1, 2 * hk * kd + nvh * vd, device=self.device)
        )
        return self._kernel("gdn_prep")(
            self._c(self._dev(q, io)).view(b, t, hk, kd),
            self._c(self._dev(k, io)).view(b, t, hk, kd),
            self._c(self._dev(v, io)).view(b, t, nvh, vd),
            self._c(self._f32(g)),
            self._c(self._f32(beta)),
            self._const_f32(kw["dt_bias"]),
            self._const_f32(kw["a_log"]),
            self._const_f32(kw["conv1d_weight"]),
            self._c(win),
            threads=vd,
        )

    def _gdn_chunk_wy(self, q, k, v, g, beta, state, **kw):
        """Chunkwise-WY prefill: gdn_prep, the WY core, gdn_post. The layer's
        conv / norm / gate glue is two launches here, sixty as torch ops."""
        b, t = q.shape[0], q.shape[1]
        nvh, vd = state.shape[1], state.shape[3]
        io = torch.bfloat16 if self.target.startswith("cuda") else torch.float32
        window = kw.get("conv_window")
        qn, kn, vn, gt, bt, new_window = self._gdn_prep(q, k, v, g, beta, state, **kw)
        if "gdn_state_scan" in _resolve(self.precision, self.arch):
            core, new_state = self._gdn_wy_core(qn, kn, vn, gt, bt, state)
        else:  # no WY schedule in this cell: the chunkwise reference is the core
            core, new_state = reference.gdn_chunk_core(
                qn, kn, vn, gt, bt, self._f32(state), chunk=_WY_CHUNK
            )
        out = self._kernel("gdn_post")(
            self._c(core).view(-1, vd),
            self._c(self._dev(kw["z"], io)).view(-1, vd),
            self._const_f32(kw["norm_weight"]),
            1e-6,
            vd,
        )
        return out.view(b, t, nvh * vd), new_state, (new_window if window is not None else None)

    def _gdn_wy_core(self, q, k, v, g, beta, state, chunk: int = _WY_CHUNK):
        """fla's chunk_gated_delta_rule_fwd stage for stage: cumsum, kkt,
        solve_tril, w/u, the inter-chunk state scan, o. gdn_prep already put
        1/sqrt(key_dim) in q, so the o scale is 1."""
        # a tail chunk writes past h, which gdn_state_scan sizes S // chunk
        assert q.shape[1] % chunk == 0, f"WY core needs whole chunks: {q.shape[1]} % {chunk}"
        kern = self._kernel
        gc = kern("gdn_chunk_cumsum")(g, chunk)
        a = kern("gdn_solve_tril")(kern("gdn_chunk_kkt")(k, beta, gc, chunk), chunk)
        w, u = kern("gdn_chunk_wu")(k, v, beta, gc, a, chunk)
        # the scan's gemm operand is bf16, so the state rounds on entry either way
        h, new_state, v_new = kern("gdn_state_scan")(
            k, w, u, gc, self._c(self._bf16(state)), chunk
        )
        return kern("gdn_chunk_o")(q, k, v_new, h, gc, chunk, 1.0), new_state

    def gdn_decode(self, q, k, v, g, beta, pool, slots, layer, **kw):
        """Decode (T=1) GDN core in one launch, state updated in place in
        ``pool.states[slots, layer]``; None off sm90 (caller takes the
        gather -> linear_attn_chunk -> scatter path)."""
        if "gdn_decode_fused" not in _resolve(self.precision, self.arch):
            return None
        slots_i = self._i32(slots).contiguous()
        out = self._kernel("gdn_decode_fused")(
            self._c(self._f32(q).squeeze(1)),
            self._c(self._f32(k).squeeze(1)),
            self._c(self._f32(v).squeeze(1)),
            self._c(self._f32(kw["z"]).squeeze(1)),
            self._c(self._f32(g).squeeze(1)),
            self._c(self._f32(beta).squeeze(1)),
            self._c(self._const_f32(kw["dt_bias"])),
            self._c(self._const_f32(kw["a_log"])),
            self._c(self._const_f32(kw["norm_weight"])),
            self._c(self._const_f32(kw["conv1d_weight"])),
            pool.conv_windows,
            pool.win_parity,
            pool.states,
            slots_i,
            int(layer),
            threads=pool.states.shape[-1],
        )
        return out.unsqueeze(1)

    def flip_window_parity(self, pool, slots) -> None:
        """After a decode tick's last GDN layer: the written planes become live."""
        idx = slots.long()
        pool.win_parity[idx] = 1 - pool.win_parity[idx]

    def _gdn_chunk_fused(self, q, k, v, g, beta, state, **kw):
        """Fused GDN chunk prefill (T>1). ``keep_steps`` > 0 (speculative verify)
        returns the per-chain-step state/window planes instead of the chunk-end pair."""
        ks = int(kw.get("keep_steps") or 0)
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
        b, nvh, kd, vd = state.shape
        wshape = tuple(window.shape[1:])
        if ks:
            step_states = torch.empty((b, ks, nvh, kd, vd), dtype=torch.float32,
                                      device=self.device)
            step_windows = torch.empty((b, ks) + wshape, dtype=torch.bfloat16,
                                       device=self.device)
        else:  # unread KS=1 operands, reused
            key = (b, nvh, kd, vd) + wshape
            if key not in self._step_scratch:
                self._step_scratch[key] = (
                    torch.empty((b, 1, nvh, kd, vd), dtype=torch.float32, device=self.device),
                    torch.empty((b, 1) + wshape, dtype=torch.bfloat16, device=self.device),
                )
            step_states, step_windows = self._step_scratch[key]
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
            step_states,
            step_windows,
            threads=state.shape[-1],
        )
        # gated RMSNorm + z-gate here, off the kernel's per-token critical path
        core = out.reshape(b, out.shape[1], nvh, vd)
        out = self.silu_mul(
            self._dev(kw["z"], kw["z"].dtype).reshape(core.shape),
            self.rmsnorm(core, self._const_f32(kw["norm_weight"]), 1e-6),
        ).reshape(out.shape)
        if ks:
            return out, step_states, (self._f32(step_windows) if has_window else None)
        return out, new_state, (self._f32(new_window) if has_window else None)

    def linear_attn_bwd(self, grad, q, k, v, g, beta, state, **kw):
        # ponytail: torch-eager backward, gdn example_chunk_delta_bwd when perf demands
        kw.pop("seq_q_lens", None)  # serving-only
        return reference.linear_attn_bwd(grad, q, k, v, g, beta, state, **kw)

    def state_gather(self, states, windows, slots, layer_idx, parity=None):
        return reference.state_gather(states, windows, slots, layer_idx, parity)

    def state_scatter(self, states, windows, slots, layer_idx, new_state, new_window,
                      parity=None, steps=False):
        reference.state_scatter(
            states, windows, slots, layer_idx, new_state, new_window, parity, steps
        )

    # ------------------------------------------------------------ silu mul

    def silu_mul(self, gate, up):
        shape = gate.shape
        gate = self._c(self._f32(gate).reshape(-1))
        up = self._c(self._f32(up).reshape(-1))
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

    def _const_f32(self, t, pad_to: int | None = None, dtype=torch.float32):
        """Cached cast of a PARAMETER (never an activation), invalidated by
        _version (optimizer copy_) and by weakref identity (a freed address
        can be reused by a fresh model)."""
        if t.dtype == dtype and t.device == self.device and pad_to is None:
            return t
        key = (t.data_ptr(), pad_to, dtype)
        hit = self._const_f32_cache.get(key)
        if hit is not None:
            ref, ver, c = hit
            if ref() is not t:
                del self._const_f32_cache[key]
            elif ver == t._version:
                return c
        c = self._dev(t, dtype)
        if pad_to is not None and pad_to != c.shape[0]:
            c = torch.nn.functional.pad(c, (0, pad_to - c.shape[0]))
        self._const_f32_cache[key] = (weakref.ref(t), t._version, c)
        return c

    def _ones(self, n: int):
        t = self._ones_cache.get(n)
        if t is None:
            t = self._ones_cache[n] = torch.ones(n, dtype=torch.float32, device=self.device)
        return t

    def _zeros2(self, m: int, n: int):
        t = self._ones_cache.get(("zeros2", m, n))
        if t is None:
            t = self._ones_cache[("zeros2", m, n)] = torch.zeros(m, n, dtype=torch.float32, device=self.device)
        return t

    def _residual(self, residual, n: int, np_: int, rows: int = 1):
        """GEMV epilogue Res rows: the residual (f32) or a cached zero block;
        None when the padded width differs, the caller adds in torch."""
        if residual is None:
            t = self._ones_cache.get(("zeros", rows, np_))
            if t is None:
                t = self._ones_cache[("zeros", rows, np_)] = torch.zeros(
                    rows, np_, dtype=torch.float32, device=self.device
                )
            return t
        if residual.numel() != n * rows or np_ != n:
            return None
        return self._f32(residual).reshape(rows, n).contiguous()

    def embedding(self, idx, table):
        # CUDA reads the bf16 table as-is (2.4 GiB vs a 4.7 GiB f32 copy); the C target cannot codegen bf16
        if table.dtype == torch.bfloat16 and self.target.startswith("cuda"):
            table, dt = self._c(table.to(self.device)), "bfloat16"
        else:
            table, dt = self._const_f32(table), "float32"
        idx_flat = self._i32(idx).reshape(-1).contiguous()
        k = self._kernel("embedding", dt)
        y = k(idx_flat, table, threads=_THREADS)
        return y.reshape(*idx.shape, table.shape[-1])

    def embedding_bwd(self, grad, idx, num_rows):
        # ponytail: torch-eager backward, tilelang kernel when perf demands
        return reference.embedding_bwd(grad, idx, num_rows)

    # ------------------------------------------------------------ sampling

    def greedy(self, logits):
        # ponytail: torch-eager, tilelang kernel when perf demands
        return reference.greedy(logits)

    def sample_batch(self, logits, temperatures, top_ps, seeds):
        return reference.sample_batch(logits, temperatures, top_ps, seeds)


_BACKEND: Backend | None = None


def get_backend() -> Backend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = Backend(resolve_target())
    return _BACKEND
