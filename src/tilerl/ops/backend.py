"""TileLang backend: target resolution, kernel dispatch, and the op interface.

This is the ONLY module in tilerl that talks to tilelang directly. Everything
above ``tilerl.ops`` calls the :class:`Backend` methods, never tilelang or
torch internals (torch is the tensor container only).

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
in kernels_mma.py — shared-memory tiled T.gemm with pipelining, the SOTA
pattern from examples/gemm/example_gemm.py. The sm90 MMA kernels are bf16-IO
(bf16 WGMMA, f32 accumulate); the CUDA path casts to bf16 once at the
boundary, while CPU/metal keep the f32 kernels. The MMA kernels require
block M/N divisible by 16 and the reduction dim divisible by 32; the CUDA
path of linear/linear_bwd/linear_fp4 zero-pads tails so the kernel always
sees exact tiles (decode M=1 pads to 16; all model N/K dims are already
multiples of 32). ``Backend.device`` pins ``cuda:<current>`` — ``torch.device("cuda")``
(index None) is not the device kernel outputs land on. Eager JIT invokes
NVCC per (shape, dtype): first call per shape costs 30-120s+.
# ponytail: shape cache / AOT before 27B serving
"""

from __future__ import annotations

import os

import torch

from . import kernels
from . import kernels_mma
from . import reference

__all__ = ["Backend", "get_backend", "resolve_target"]

_THREADS = 64


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _pad2d(t: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Zero-pad a 2D tensor to [rows, cols] (bottom/right)."""
    pr, pc = rows - t.shape[0], cols - t.shape[1]
    if pr == 0 and pc == 0:
        return t
    return torch.nn.functional.pad(t, (0, pc, 0, pr))


def _pad1d(t: torch.Tensor, n: int) -> torch.Tensor:
    p = n - t.shape[0]
    return t if p == 0 else torch.nn.functional.pad(t, (0, p))


# ---------------------------------------------------------------------------
# Precision x arch dispatch matrix.
#
# Kernel sets are keyed by (precision, arch); _resolve walks the fallback
# chain exact -> (precision, "any") -> ("any", "any"). Adding fp8 or a new SM
# arch is ONE _register() call. Day-1: bf16/fp4 on CPU (f32-compute kernels,
# bf16 cast at the boundary); GPU arches are pending-remote slots — registered
# so the matrix is honest, NotImplementedError on use. Full matrix:
# docs/support-matrix.md.
# ---------------------------------------------------------------------------

_REGISTRY: dict[tuple[str, str], dict[str, object]] = {}


def _register(precision: str, arch: str, kernels: dict[str, object]) -> None:
    _REGISTRY[(precision, arch)] = kernels


def _resolve(precision: str, arch: str) -> dict[str, object]:
    for key in ((precision, arch), (precision, "any"), ("any", "any")):
        if key in _REGISTRY:
            if not _REGISTRY[key]:
                raise NotImplementedError(
                    f"({precision!r}, {arch!r}) is pending-remote bring-up "
                    "(see docs/support-matrix.md)"
                )
            return _REGISTRY[key]
    raise KeyError(f"no kernel set for ({precision!r}, {arch!r})")


_CPU_KERNELS = {  # bf16 on CPU: the f32 TileLang kernels (bf16 cast at the boundary)
    "rmsnorm_partial": kernels.make_rmsnorm_partial,
    "rmsnorm_apply": kernels.make_rmsnorm_apply,
    "rmsnorm_rstd": kernels.make_rmsnorm_rstd,
    "rmsnorm_bwd_x": kernels.make_rmsnorm_bwd_x,
    "gemm_nt": kernels.make_gemm_nt,
    "gemm_nn": kernels.make_gemm_nn,
    "gemm_tn": kernels.make_gemm_tn,
    "silu_mul": kernels.make_silu_mul,
    "softmax": kernels.make_softmax,
    "rope": kernels.make_rope,
    "embedding": kernels.make_embedding,
    "linear_fp4": kernels.make_linear_fp4,
    "paged_attention": kernels.make_paged_attention,
    "linear_attn_chunk": kernels.make_linear_attn_chunk,
}
_register("bf16", "cpu", _CPU_KERNELS)
# fp4 is a weight format, not a compute dtype: its cell reuses the bf16 set
# (linear_fp4 is in it; the rest of the layer is the bf16 path).
_register("fp4", "cpu", _CPU_KERNELS)
# metal: same target-neutral kernel source, except the three gemms — Metal's
# T.gemm lowering rejects global operands, so the metal cell swaps in the
# naive FMA schedules from kernels.py (see "gemm (naive FMA schedule)").
_METAL_KERNELS = {
    **_CPU_KERNELS,
    "gemm_nt": kernels.make_gemm_nt_naive,
    "gemm_nn": kernels.make_gemm_nn_naive,
    "gemm_tn": kernels.make_gemm_tn_naive,
}
_register("bf16", "metal", _METAL_KERNELS)
_register("fp4", "metal", _METAL_KERNELS)
# sm90: the MMA (WGMMA) schedules from kernels_mma.py — shared-memory tiled
# T.gemm with pipelining, the SOTA pattern from examples/gemm/example_gemm.py.
# The naive FMA gemms stay in kernels.py as the metal/other-arch fallback.
# The MMA kernels require block M/N divisible by 16 and the reduction dim
# divisible by _RED_TILE (32); the CUDA path of linear/linear_bwd/linear_fp4
# zero-pads tails so the kernel always sees exact tiles.
_SM90_KERNELS = {
    **_CPU_KERNELS,
    "gemm_nt": kernels_mma.make_gemm_nt_mma,
    "gemm_nn": kernels_mma.make_gemm_nn_mma,
    "gemm_tn": kernels_mma.make_gemm_tn_mma,
    "linear_fp4": kernels_mma.make_linear_fp4_mma,
    "linear_fp4_gemv": kernels_mma.make_linear_fp4_gemv,
    "gdn_decode_fused": kernels_mma.make_gdn_decode_fused,
    "gdn_chunk_fused": kernels_mma.make_gdn_chunk_fused,
}
_register("bf16", "sm90", _SM90_KERNELS)
_register("fp4", "sm90", _SM90_KERNELS)
for _arch in ("sm100", "sm120", "rocm"):
    _register("bf16", _arch, {})  # pending-remote slot


def _arch_for(target: str) -> str:
    """Arch tag for the matrix: cpu | sm90 | sm100 | ... | rocm | metal."""
    if target == "c":
        return "cpu"
    if target.startswith("cuda") and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f"sm{major}{minor}"
    return {"hip": "rocm", "metal": "metal"}.get(target, target)


def resolve_target() -> str:
    """Resolve the tilelang target string for this process.

    ``TILERL_TARGET`` overrides; accepts the friendly names ``cpu|cuda|rocm|
    metal|auto`` (cpu -> ``"c"``, the working CPU target; rocm -> ``"hip"``).
    ``auto`` (the default) maps to ``"cuda"`` when a CUDA device is visible and
    ``"c"`` otherwise — tilelang's own ``auto`` would pick metal on this Mac,
    which is not the dev/CI path.
    """
    target = os.environ.get("TILERL_TARGET", "auto").strip().lower()
    aliases = {"cpu": "c", "rocm": "hip", "": "auto"}
    target = aliases.get(target, target)
    if target == "auto":
        return "cuda" if torch.cuda.is_available() else "c"
    return target


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
        #: f32 cast of the embedding table, keyed by (data_ptr, _version).
        #: The 27B table is 248320x5120 bf16 (~2.5GB); re-casting it every
        #: tick costs ~6ms on H20. The optimizer's in-place copy_ bumps
        #: _version, so a train step invalidates the entry.
        self._embed_f32: dict[int, tuple[int, torch.Tensor]] = {}

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

    def _i32(self, t: torch.Tensor) -> torch.Tensor:
        return self._dev(t, torch.int32)

    def _inv_freq(self, d: int, theta: float) -> torch.Tensor:
        key = (d, theta)
        inv = self._inv_freq_cache.get(key)
        if inv is None:
            inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
            self._inv_freq_cache[key] = inv
        return inv

    # ------------------------------------------------------------ add

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Elementwise add (residual stream). Recorded by the tape."""
        return a + b

    # ------------------------------------------------------------ rmsnorm

    def rmsnorm(self, x, w, eps):
        x = self._f32(x)
        w = self._f32(w)
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
        # sm90 kernels are bf16-IO; CPU/metal kernels are f32. Cast once at
        # the boundary (the model is bf16-master — no bf16->f32->bf16 trip).
        io = torch.bfloat16 if self.target.startswith("cuda") else torch.float32
        x = self._dev(x, io)
        w = self._dev(w, io)
        lead = x.shape[:-1]
        x2 = self._c(x.reshape(-1, x.shape[-1]))
        M, K, N = x2.shape[0], x2.shape[1], w.shape[0]
        bM, bN = min(64, M), min(64, N)
        if self.target.startswith("cuda"):
            # WGMMA tiles: block M/N %16, reduction K %32; pad tails so the
            # MMA kernel sees exact tiles (no OOB loads).
            bM, bN = _round_up(bM, 16), _round_up(bN, 16)
            x2 = _pad2d(x2, _round_up(M, bM), _round_up(K, 32))
            w = _pad2d(w, _round_up(N, bN), _round_up(K, 32))
            if bias is not None:
                bias = _pad1d(self._f32(bias), w.shape[0])
        if bias is None:
            bias = torch.zeros(w.shape[0], dtype=torch.float32, device=self.device)
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

    def linear_fp4(self, x, wq, scale, master=None):
        # ``master`` is recording-only (the STE grad lands on it); the kernel
        # uses wq/scale. sm90 kernels are bf16-IO; CPU/metal kernels are f32.
        wq = self._dev(wq, wq.dtype)  # uint8: device migration only
        scale = self._f32(scale)
        io = torch.bfloat16 if self.target.startswith("cuda") else torch.float32
        x = self._dev(x, io)
        lead = x.shape[:-1]
        x2 = self._c(x.reshape(-1, x.shape[-1]))
        M, K, N = x2.shape[0], x2.shape[1], wq.shape[0]
        if (
            self.target.startswith("cuda")
            and M == 1
            and "linear_fp4_gemv" in _resolve(self.precision, self.arch)
        ):
            # Decode GEMV: one activation row, stream+dequant WQ once. Block K
            # is reduce_thread(32) * micro_size_k(8 bf16) = 256; e2m1fn has no
            # zero, so the K-tail is killed by the zero-padded Scale (same
            # trick as the MMA path).
            Kp = _round_up(K, 256)
            Np = _round_up(N, 4)
            y = self._kernel("linear_fp4_gemv")(
                _pad2d(x2, 1, Kp),
                _pad2d(wq, Np, Kp // 2),
                _pad2d(scale, Np, Kp // 16),
                32,
                4,
            )
            return y[:1, :N].reshape(*lead, N)
        bM, bN = min(64, M), min(64, N)
        if self.target.startswith("cuda"):
            # WGMMA tiles %16, reduction K %32. e2m1fn has no zero, so padded
            # WQ bytes (0x00 -> 0.5) are killed by the zero-padded Scale.
            bM, bN = _round_up(bM, 16), _round_up(bN, 16)
            Mp, Np, Kp = _round_up(M, bM), _round_up(N, bN), _round_up(K, 32)
            x2 = _pad2d(x2, Mp, Kp)
            wq = _pad2d(wq, Np, Kp // 2)
            scale = _pad2d(scale, Np, Kp // 16)
        y = self._kernel("linear_fp4")(x2, wq, scale, bM, bN, _THREADS)
        return y[:M, :N].reshape(*lead, N)

    def linear_fp4_bwd(self, grad, x, wq, scale, master=None):
        # ponytail: torch-eager backward, tilelang kernel when perf demands
        return reference.linear_fp4_bwd(grad, x, wq, scale)

    # ------------------------------------------------------------ attention

    def paged_attention(self, q, k_cache, v_cache, block_table, seq_lens, scale, gate=None):
        q = self._f32(q)
        k_cache = self._f32(k_cache)
        v_cache = self._f32(v_cache)
        if q.ndim == 3:
            q = self._c(q.unsqueeze(1))  # [B, H, D] -> [B, 1, H, D]
            squeeze = True
        else:
            squeeze = False
        bt = self._i32(block_table).contiguous()
        sl = self._i32(seq_lens).contiguous()
        k = self._kernel("paged_attention")
        out = k(
            q,
            k_cache,
            v_cache,
            bt,
            sl,
            float(scale),
            block_size=int(k_cache.shape[2]),
            threads=_THREADS,
        )
        out = out.squeeze(1) if squeeze else out
        if gate is not None:
            out = out * torch.sigmoid(self._f32(gate))
        return out

    def attention(self, q, k, v, scale, gate=None):
        """Dense causal GQA attention (training path). q/k/v [B,T,H,D]."""
        # ponytail: torch-eager forward, tilelang kernel when perf demands
        out = reference.dense_attention(q, k, v, float(scale))
        if gate is not None:
            out = out * torch.sigmoid(self._f32(gate))
        return out

    def attention_bwd(self, grad, q, k, v, scale):
        # ponytail: torch-eager backward, tilelang kernel when perf demands
        return reference.dense_attention_bwd(grad, q, k, v, float(scale))

    # ------------------------------------------------------------ gated delta

    def linear_attn_chunk(self, q, k, v, g, beta, state, **kw):
        if kw.get("z") is not None or kw.get("conv1d_weight") is not None:
            # Full-GDN layer core. sm90: T=1 uses the fused decode kernel,
            # T>1 the fused chunk kernel; other arches use the torch-eager
            # reference.
            _kset = _resolve(self.precision, self.arch)
            if q.shape[1] == 1 and "gdn_decode_fused" in _kset:
                return self._gdn_decode_fused(q, k, v, g, beta, state, **kw)
            if q.shape[1] > 1 and "gdn_chunk_fused" in _kset:
                return self._gdn_chunk_fused(q, k, v, g, beta, state, **kw)
            return reference.gdn_forward(q, k, v, g, beta, state, **kw)
        ker = self._kernel("linear_attn_chunk")
        out, new_state = ker(
            self._c(self._f32(q)),
            self._c(self._f32(k)),
            self._c(self._f32(v)),
            self._c(self._f32(g)),
            self._c(self._f32(beta)),
            self._c(self._f32(state)),
            threads=_THREADS,
        )
        return out, new_state, None

    def _gdn_decode_fused(self, q, k, v, g, beta, state, **kw):
        """Fused GDN decode (T=1): one launch for the whole layer core.

        q/k/v/g/beta [B, 1, ...] are squeezed to [B, ...]. conv_window is
        always a tensor (the model carries it; None means zero left-padding).
        """
        window = kw.get("conv_window")
        if window is None:
            window = torch.zeros(
                q.shape[0],
                kw["conv1d_weight"].shape[1] - 1,
                q.shape[-1] + k.shape[-1] + v.shape[-1],
                dtype=torch.float32,
                device=self.device,
            )
        out, new_state, new_window = self._kernel("gdn_decode_fused")(
            self._c(self._f32(q).squeeze(1)),
            self._c(self._f32(k).squeeze(1)),
            self._c(self._f32(v).squeeze(1)),
            self._c(self._f32(kw["z"]).squeeze(1)),
            self._c(self._f32(g).squeeze(1)),
            self._c(self._f32(beta).squeeze(1)),
            self._c(self._f32(kw["dt_bias"])),
            self._c(self._f32(kw["a_log"])),
            self._c(self._f32(kw["norm_weight"])),
            self._c(self._f32(kw["conv1d_weight"])),
            self._c(self._f32(window)),
            self._c(self._f32(state)),
            threads=state.shape[-1],
        )
        return out.unsqueeze(1), new_state, new_window

    def _gdn_chunk_fused(self, q, k, v, g, beta, state, **kw):
        """Fused GDN chunk prefill (T>1): one launch for the whole layer core.

        q/k/v/g/beta [B, T, ...] keep their T dim (the kernel scans it
        serially per (value head, batch)). conv_window is always a tensor
        (the model carries it; None means zero left-padding).
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
        out, new_state, new_window = self._kernel("gdn_chunk_fused")(
            self._c(self._f32(q)),
            self._c(self._f32(k)),
            self._c(self._f32(v)),
            self._c(self._f32(kw["z"])),
            self._c(self._f32(g)),
            self._c(self._f32(beta)),
            self._c(self._f32(kw["dt_bias"])),
            self._c(self._f32(kw["a_log"])),
            self._c(self._f32(kw["norm_weight"])),
            self._c(self._f32(kw["conv1d_weight"])),
            self._c(self._f32(window)),
            self._c(self._f32(state)),
            threads=state.shape[-1],
        )
        return out, new_state, (new_window if has_window else None)

    def linear_attn_step(self, q, k, v, g, beta, state, **kw):
        out, new_state, new_window = self.linear_attn_chunk(
            self._c(q.unsqueeze(1)),
            self._c(k.unsqueeze(1)),
            self._c(v.unsqueeze(1)),
            self._c(g.unsqueeze(1)),
            self._c(beta.unsqueeze(1)),
            state,
            **kw,
        )
        return out.squeeze(1), new_state, new_window

    def linear_attn_bwd(self, grad, q, k, v, g, beta, state, **kw):
        # ponytail: torch-eager backward (gdn example_chunk_delta_bwd is the
        # CUDA-scheduled tilelang upgrade path), tilelang kernel when perf demands
        return reference.linear_attn_bwd(grad, q, k, v, g, beta, state, **kw)

    # ------------------------------------------------------------ silu mul

    def silu_mul(self, gate, up):
        shape = gate.shape
        gate = self._f32(gate).reshape(-1)
        up = self._f32(up).reshape(-1)
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

    # ------------------------------------------------------------ embedding

    def _embed_table_f32(self, table):
        """f32 cast of the embedding table, cached across ticks. The 27B
        table is ~2.5GB bf16; casting it per tick costs ~6ms on H20. The
        optimizer's in-place copy_ bumps _version, invalidating the entry.
        # ponytail: the tied lm_head path (linear) re-casts the same table;
        # route it through this cache when a tied model is served."""
        key = table.data_ptr()
        ver = table._version
        hit = self._embed_f32.get(key)
        if hit is not None and hit[0] == ver:
            return hit[1]
        t = self._f32(table)
        self._embed_f32[key] = (ver, t)
        return t

    def embedding(self, idx, table):
        table = self._embed_table_f32(table)
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


_BACKEND: Backend | None = None


def get_backend() -> Backend:
    """Process-wide backend singleton (target resolved once)."""
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = Backend(resolve_target())
    return _BACKEND
