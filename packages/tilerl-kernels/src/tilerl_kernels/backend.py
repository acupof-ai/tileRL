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

from . import kernels_linear, reference
from .registry import _arch_for, _resolve, resolve_target, sm70_kvsplit

__all__ = ["Backend", "get_backend", "resolve_target"]

_THREADS = 64
#: chunkwise-WY chunk length. gdn_state_scan and gdn_chunk_o size h by S // chunk, so a T
#: that is not a whole multiple writes past it.
_WY_CHUNK = 64
# ponytail: the DFlash2 verify block; wider runs take the M-tiled kernel until
# the crossover between the two is measured
_MAX_VERIFY_W = 8
# a whole-chunk verify width would reach _full_rows' host sync, illegal under graph capture
assert _MAX_VERIFY_W < _WY_CHUNK


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _snap_mma_tile(m: int, cap: int) -> int:
    # WGMMA Square policy: 16/32/64/128 compile, 48/80/96/112 do not
    return min(cap, next((s for s in (16, 32, 64, 128) if s >= m), 128))


def _sm70_chunks(rows: int, top: int = 32) -> list[tuple[int, int, int]]:
    """(offset, real rows, compiled rung) per launch for the sm70 fp4 GEMV.

    The ladder is 1/2/4/8/``top``: a chunk pays its rung's full row count, so the
    LAST chunk of a non-multiple M drops to a smaller rung instead of padding to
    ``top``. M=40 is 32 + 8, not two 32-row launches. Pure integer arithmetic,
    split out because the interesting failure is invisible where it is exercised:
    a slicing bug shows up only for M that does not divide ``top``, and the sm70
    branch never runs on the CPU target where the parity tests live.
    """
    out, m = [], 0
    while m < rows:
        r = min(top, rows - m)
        out.append((m, r, 1 if r == 1 else 2 if r <= 2 else 4 if r <= 4 else 8 if r <= 8 else top))
        m += r
    return out


def _pad2d(t: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    pr, pc = rows - t.shape[0], cols - t.shape[1]
    if pr == 0 and pc == 0:
        return t
    # F.pad CROPS on a negative pad rather than raising, so an oversized tensor
    # here becomes well-formed garbage: a [5120, 12800] weight silently returns
    # [5120, 5120]. One shipped call site was RELYING on that crop -- linear_fp8's
    # mma8 branch asked for 8 rows from an x2 already padded to 16 -- and it read
    # as intentional nowhere, so this guard turned it into a hard error on every
    # M in 2..8 (measured: 1191 calls at M=8 in one sm90 B=8 tick). That call site
    # now slices explicitly. Any negative pad reaching here is a caller bug.
    if pr < 0 or pc < 0:
        raise ValueError(f"_pad2d: {tuple(t.shape)} exceeds the target [{rows}, {cols}]")
    return torch.nn.functional.pad(t, (0, pc, 0, pr))


def _fit_rows(t: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Pad to [rows, cols], or drop rows off a plane the caller padded to a WIDER rung.

    The mma8 branches re-target tensors the plan line above already padded, and the
    two rungs do not order: Np = _round_up(N, bN) overshoots Np32 = _round_up(N, 32)
    whenever bN's rounding jumps past 32 (N=24 on the fp4 decode plan: bN=64 gives
    Np=64 against Np32=32). Every row between the real N and either rung is zero pad,
    so dropping rows here is exact -- verified in test_weights.py against padding the
    unpadded plane straight to the narrow rung. `main` got this from F.pad's crop
    silently; _pad2d now refuses a negative pad, so the three re-target sites say it.
    """
    return t[:rows] if t.shape[0] > rows and t.shape[1] == cols else _pad2d(t, rows, cols)


def _pad1d(t: torch.Tensor, n: int) -> torch.Tensor:
    p = n - t.shape[0]
    return t if p == 0 else torch.nn.functional.pad(t, (0, p))


_MX = 8  # mma8 row cap: decode rows on the tensor cores
#: rows up to which the M-row GEMV beats mma8 (27B decode replay, H20, ms:
#: gemv 11.2/17.5/27.1/30.1 at M=1..4, mma8 27 flat); TILERL_MGEMV=0 disables
_MGEMV = int(os.environ.get("TILERL_MGEMV", "3"))
#: Output columns per thread in the sm70 fp4 GEMV; 2 shares one X load across two
#: columns (HFMA2 per LDG 3.53 -> 6.06, 1.82x at M=32), 1 is the A/B arm
#: (wins/2026-09-03-ncols2-raises-loads-per-fma.md).
_NCOLS = int(os.environ.get("TILERL_NCOLS", "2"))
#: Lowest rung ncols=2 is used on. Below it the GEMV is bandwidth-bound, so halving
#: the grid only starves it: dense decode measured 39.1 -> 37.2 tok/s at 4096 with
#: ncols on at M=1 (errors/2026-09-03-ncols2-cost-5-percent-of-decode.md). The sm70
#: ladder is 1/2/4/8/32, so this covers prefill AND a verify tick of 9..32 rows --
#: in SERVING, B*W=16 at max_batch=4 and depth 3 rounds UP to the 32 rung and keeps
#: ncols=2. Note the row count is B*W, not W: a bench that submits one request runs
#: 4 rows on the 4 rung with ncols OFF, which is how the first spec A/B compared this
#: kernel with itself (errors/2026-09-03-the-spec-ncols-ab-ran-at-b1.md).
_NCOLS_MIN_M = 32
#: the MMA kernels floor-divide `X // _RED_TILE`, so a reduction dim padded to
#: anything else silently drops its tail (errors/2026-09-03-red-tile-zeroed-the-weight-gradient.md)
_MMA_RED = kernels_linear._RED_TILE

#: CUDA linear family: (op, M-regime) -> (kernel, K pad, N cap, N tile).
#: Crossovers measured in wins/2026-08-26-batch-decode-h2.md. The fp4->e4m3
#: arms tile N at 64 because the kernel overrides block_N to _FP4_BLOCK_N.
_CUDA_PLAN = {
    ("linear", "gemv"): ("linear_bf16_gemv", 256, 4, 4),
    # kpad=512: the fp4 GEMV strides block_K=reduce_thread(32)*micro(16)=512 with
    # no k<K guard, so K pads to 512 (was 256 — OOB when Kp%512!=0, e.g. a
    # TP-sharded down_proj K=4352; 27B single-card dims all divide 512).
    ("linear_fp4", "gemv"): ("linear_fp4_gemv", 512, 4, 4),
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
    #: this rank's index in the TP group. #106 deleted it as unread; the sharded
    #: cross-entropy needs it to locate its slice of the vocabulary.
    tp_rank = 0
    #: this rank's tp process group; None means "the whole world is one tp group"
    _tp_pg = None
    #: ranks holding the same shard in the other dp replicas, and their group.
    #: 1 means no data parallelism, so the gradient reduce below is a no-op.
    dp_world = 1
    _dp_pg = None
    #: ranks holding the other chunks of this rank's sequence, and their group.
    cp_world = 1
    cp_rank = 0
    _cp_pg = None

    def init_tp(self, world: int, rank: int, tp_groups: list[list[int]] | None = None,
                dp_groups: list[list[int]] | None = None,
                cp_groups: list[list[int]] | None = None) -> None:
        """Join the TP group; framework layers never import torch.distributed.

        ``tp_groups`` is EVERY tp group in the mesh, in mesh order
        (``[m.tp_group() for m in ...]``, deduped by the caller). Omit it and the
        whole world is one tp group, which is right for a pure-TP run and WRONG
        the moment dp > 1: an ungrouped ``all_reduce`` sums across the dp
        replicas too, averaging away the independence dp exists for.

        All of them, not just this rank's, because ``new_group`` is itself
        collective -- every rank must call it for every group, in the same
        order, or the ranks that skipped one deadlock the first time it is used.
        The same applies to ``dp_groups`` and ``cp_groups``, and to the ORDER of
        the three loops: every rank builds all tp groups, then all dp, then all cp.
        """
        if world == 1:
            return
        import torch.distributed as dist

        if not dist.is_initialized():
            comm = "nccl" if self.device.type == "cuda" else "gloo"
            dist.init_process_group(comm, world_size=world, rank=rank)

        def join(groups: list[list[int]], axis: str) -> list[int]:
            mine, pg_mine = None, None
            for g in groups:
                pg = dist.new_group(list(g))
                if rank in g:
                    mine, pg_mine = g, pg
            if mine is None:
                raise ValueError(f"rank {rank} is in none of the {axis} groups {groups}")
            return mine, pg_mine

        if tp_groups:
            mine, self._tp_pg = join(tp_groups, "tp")
            self.tp_world, self.tp_rank = len(mine), mine.index(rank)
        else:
            self.tp_world, self.tp_rank = world, rank
        if dp_groups:
            mine, self._dp_pg = join(dp_groups, "dp")
            self.dp_world = len(mine)
        if cp_groups:
            mine, self._cp_pg = join(cp_groups, "cp")
            self.cp_world, self.cp_rank = len(mine), mine.index(rank)

    def dp_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Average one gradient across the dp replicas, in place.

        Mean, not sum: each replica's loss already averages over its own rows, so
        summing would scale the update by ``dp_world`` -- a learning-rate change
        that hides as a convergence difference rather than an error."""
        if self.dp_world == 1:
            return x
        import torch.distributed as dist

        dist.all_reduce(x, group=self._dp_pg)
        return x.div_(self.dp_world)

    def tp_fork(self, x: torch.Tensor) -> torch.Tensor:
        """Identity forward, all-reduce backward: the dual of ``all_reduce``.

        A replicated activation feeding column-parallel linears needs no forward
        collective, but each rank's backward produces only its own shard's share
        of dX. Without this the norm below reads dX/world and the loss still
        looks right, which is the silent factor-of-world the TP gates check."""
        if self.tp_world == 1:
            return x
        return x.view_as(x)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Sum across the TP group in place. 21.5 us floor per call on 6 H20s,
        flat from 20 KB to 1.3 MB: the cost is per layer, not per byte.
        Returns a distinct view: the tape addresses tensors by id(), and an op
        that returns its own input pops the gradient it just wrote for it."""
        if self.tp_world == 1:
            return x
        import torch.distributed as dist

        dist.all_reduce(x, group=self._tp_pg)
        return x.view_as(x)

    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Concatenate from every rank along ``dim`` (vocab-parallel lm_head)."""
        if self.tp_world == 1:
            return x
        import torch.distributed as dist

        x = x.contiguous()
        parts = [torch.empty_like(x) for _ in range(self.tp_world)]
        dist.all_gather(parts, x, group=self._tp_pg)
        return torch.cat(parts, dim=dim)

    def cp_gather(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """Concatenate every cp rank's sequence chunk, in cp_rank order.

        Chunks come back in RANK order, which is not sequence order under the
        zigzag assignment -- the caller pairs this with the matching positions
        and never infers them from the index."""
        if self.cp_world == 1:
            return x
        import torch.distributed as dist

        x = x.contiguous()
        parts = [torch.empty_like(x) for _ in range(self.cp_world)]
        dist.all_gather(parts, x, group=self._cp_pg)
        return torch.cat(parts, dim=dim)

    def cp_reduce_scatter(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """Sum across the cp group and keep this rank's chunk: the backward of
        :meth:`cp_gather`.

        A slice alone is wrong. Every rank's queries read every rank's K/V, so
        each rank's dK is a partial sum over the whole sequence -- measured on a
        dense reference at cp=4, keeping only the local slice is off by 58% of
        full scale, while the summed version matches to 2.4e-07."""
        if self.cp_world == 1:
            return x
        import torch.distributed as dist

        parts = [p.contiguous() for p in x.chunk(self.cp_world, dim=dim)]
        out = torch.empty_like(parts[0])
        dist.reduce_scatter(out, parts, group=self._cp_pg)
        return out

    def cp_prefix_scan(self, a: torch.Tensor, b: torch.Tensor, chunk_ids=None):
        """Exclusive scan of the affine pairs ``(A, B)`` over the cp group, so a
        rank starts each of its chunks from ``A_pre @ s + B_pre`` rather than
        waiting for its predecessor. See :func:`reference.affine_prefix_scan`."""
        if self.cp_world == 1:
            return None, None
        return reference.affine_prefix_scan(a, b, self._cp_pg, self.cp_rank,
                                            self.cp_world, chunk_ids)

    def cp_halo(self, x: torch.Tensor, ids_by_rank: list[list[int]], width: int):
        """The last ``width`` rows of the chunk BEFORE each of this rank's chunks.

        A depthwise conv of kernel K needs K-1 rows of left context, and under CP
        those live on whichever rank holds chunk c-1 — a different rank per chunk
        once the layout is zigzag. Dropping them does not just dirty the boundary
        rows: the corrupted k/v feed the recurrence, so the whole chunk is wrong
        (measured 0.61 relative after the boundary against 0.79 on the 3 rows).

        ``ids_by_rank[r]`` is rank r's chunk indices in the order it stacks them,
        so the layout rule lives at the caller and is not restated here. One
        ``all_gather`` of the tails rather than point-to-point: the payload is
        0.21 MiB per layer at B=1 and 1.69 at B=8 on the 27B, against a 24 MiB
        state, and the zigzag source pattern is irregular enough that hand-rolled
        sends would be the larger risk. ``None`` for chunk 0, which has no
        predecessor.
        """
        if self.cp_world == 1:
            return [None] * len(ids_by_rank[self.cp_rank])
        import torch.distributed as dist

        tails = x[:, :, -width:].contiguous()  # [chunks, B, width, D] per rank
        parts = [torch.empty_like(tails) for _ in range(self.cp_world)]
        dist.all_gather(parts, tails, group=self._cp_pg)
        owner = {c: (r, i) for r, ids in enumerate(ids_by_rank) for i, c in enumerate(ids)}
        return [None if c == 0 else parts[owner[c - 1][0]][owner[c - 1][1]]
                for c in ids_by_rank[self.cp_rank]]

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
        # Kernel I/O dtype: only sm90 has bf16 tensor cores; sm70's MMA is
        # fp16-only and its cell is the CPU f32 source, so it takes f32 IO.
        # Invariant: f32-kernel call sites (e.g. gemm_nt) assume io is NOT
        # bf16/fp16 and skip the _f32 wrap — flipping sm70 to bf16/fp16 here
        # silently feeds those kernels the wrong dtype. sm70's fp16 GEMV does
        # its f32->fp16 cvt inside the kernel, not via io.
        # main had this inline at four call sites as `cuda ? bf16 : f32` -- a TARGET
        # test. Keep that for cpu and metal (f32, unchanged) and carve out only sm70,
        # which needs f32 on a cuda target. Writing it as `sm90 ? bf16 : f32` flipped
        # cpu and metal too once build_engine started passing this as the KV pool
        # dtype, and writing it as `sm70 ? f32 : bf16` flips them the other way:
        # _arch_for returns cpu/metal/sm70/sm90, so neither binary split isolates
        # sm70. The CPU cell is the parity target for every kernel here, and its
        # suite stayed green through the first version because a parity check
        # compares TileLang against a torch reference in the SAME process, moving
        # both sides together.
        self.io = torch.float32 if self.arch in ("cpu", "metal", "sm70") else torch.bfloat16
        # Declared beside io for the same reason: the store (materialize) and the
        # kernel annotation must not drift (wins/2026-09-02-kv-pool-dtype-is-the-kernel-abi).
        self.scale_io = torch.float16 if self.arch == "sm70" else torch.float32
        #: Metal's packed ABI rejects a kernel argument whose byte_offset is not 0.
        #: Arch-gated, not unconditional: sm90 hands _c 6 contiguous-but-offset views
        #: per decode tick at 6.68 us a clone, 40 us it does not owe (cpu and metal
        #: are 0). Re-derive with scripts/probe_c_offset.py before deleting this.
        self._zero_offset = self.arch == "metal"
        # Narrow dtype for the embedding gather only — NOT io, which the f32
        # kernels depend on. Half the table's bytes on the card, and the gather
        # widens to f32 on read.
        self.embed_io = {"sm90": torch.bfloat16, "sm70": torch.float16}.get(
            self.arch, torch.float32
        )
        # The dtype the fp4 GEMV reads X in. sm70's twiddled ladder is f16, so an
        # elementwise op feeding a linear can write f16 and skip the dispatch's
        # cast; everything else keeps io.
        self.gemv_io = torch.float16 if self.arch == "sm70" else self.io
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

    def has_kernel(self, name: str) -> bool:
        """Is ``name`` served by a real kernel in this (precision, arch) cell?

        The public form of the registry probe, for callers above this package
        that must not guess: a weight format is only worth producing when the
        kernel that consumes it exists, or the op silently takes the torch
        fallback instead.
        """
        return name in _resolve(self.precision, self.arch)

    def _kernel(self, name: str, *args, **kw):
        """``args``/``kw`` are factory (compile-time variant) arguments; they key the cache.

        Keywords are accepted so a caller can name a late factory parameter instead
        of counting past the ones before it -- passing ncols positionally landed on
        `abl` and silently ran an ablation kernel that returns wrong numbers.
        """
        key = (name, args, tuple(sorted(kw.items())))
        k = self._kernels.get(key)
        if k is None:
            k = _resolve(self.precision, self.arch)[name](self.target, *args, **kw)
            self._kernels[key] = k
        return k

    # ------------------------------------------------------------ helpers

    def _c(self, t: torch.Tensor) -> torch.Tensor:
        # A row slice is contiguous AND starts partway into its storage. The Metal ABI
        # rejects that non-zero byte_offset; CUDA tolerates it. `.contiguous()` cannot
        # fix it -- the view already satisfies its predicate -- so the reset is a copy,
        # and the copy is charged only to the target that needs it: measured 6 offset
        # views per sm90 decode tick at 6.68 us each, 40 us a tick that CUDA does not
        # owe (wins/2026-09-04-metal-is-green-and-28x-off-torch-eager.md).
        if t.is_contiguous():
            return t.clone() if self._zero_offset and t.storage_offset() else t
        return t.contiguous()

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

    def _rows(self, x: torch.Tensor, keep_f16: bool = False):
        """(leading shape, 2-D contiguous rows at this cell's IO dtype).

        ``keep_f16`` leaves an f16 input alone: the sm70 GEMV wants X in f16, so
        widening it to io (f32) here only to narrow it again at the launch is two
        passes over the same bytes — 305 of them per token on the 27B. The
        elementwise kernels that produce X now emit f16 directly.
        """
        if keep_f16 and x.dtype == torch.float16 and x.device == self.device:
            return x.shape[:-1], self._c(x.reshape(-1, x.shape[-1]))
        return x.shape[:-1], self._c(self._dev(x, self.io).reshape(-1, x.shape[-1]))

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
        Kp = _round_up(k, kpad)
        # The fp4/fp8 GEMVs stride block_K = 32*16 = 512 with no k<K guard, so the
        # padded K must be a 512 multiple — kpad guarantees it (guard the contract
        # here rather than let a mismatched plan read OOB on the device).
        if op in ("linear_fp4", "linear_fp8") and m == 1:
            assert Kp % 512 == 0, f"gemv kpad={kpad} leaves Kp={Kp} not a 512 multiple"
        return kernel, _round_up(m, bM), _round_up(n, bN), Kp, bM, bN

    # ------------------------------------------------------------ add

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    # ------------------------------------------------------------ rmsnorm

    def rmsnorm(self, x, w, eps, narrow: bool = False):
        """``narrow``: the caller feeds the result straight to a linear, so let
        the kernel write the GEMV's own IO dtype instead of f32 that the dispatch
        would narrow anyway. Off by default — q_norm/k_norm feed rope and
        attention, which are f32, and round-tripping through f16 there would drop
        13 mantissa bits for nothing."""
        return self._rmsnorm(x, w, eps, out_f32=False, narrow=narrow)

    def rmsnorm_f32(self, x, w, eps):
        """rmsnorm whose output stays f32. For q/k norm, whose result reaches the
        bf16 KV pool through rope: a bf16 store here would round twice.

        A separate method rather than a keyword on :meth:`rmsnorm` because
        `_TapeBackend.__getattr__` forwards **kwargs into the recorded entry and
        `_default("rmsnorm")` passes them to `rmsnorm_bwd(grad, x, w, eps)`, which
        would raise. It is registered in `autograd._BWD` against the same
        `rmsnorm_bwd` -- the backward of a norm does not depend on the dtype the
        forward stored -- and an unregistered op records NO tape entry at all,
        silently (measured: 1 entry for rmsnorm, 0 for an unregistered twin)."""
        return self._rmsnorm(x, w, eps, out_f32=True, narrow=False)

    def _rmsnorm(self, x, w, eps, *, out_f32: bool, narrow: bool = False):
        x = self._f32(x)
        w = self._const_f32(w)
        lead = x.shape[:-1]
        x2 = self._c(x.reshape(-1, x.shape[-1]))
        N = x2.shape[-1]
        kset = _resolve(self.precision, self.arch)
        key = "rmsnorm_fused_f32" if out_f32 else "rmsnorm_fused"
        if key in kset:
            y = self._kernel(key)(x2, w, float(eps), 256)
            return y.reshape(*lead, w.shape[0])
        block_N = min(256, N)
        num_chunks = (N + block_N - 1) // block_N
        p = self._kernel("rmsnorm_partial")(x2, block_N, num_chunks, _THREADS)
        # the cell's rmsnorm_apply is f32 everywhere but sm90, which overrides it
        # to bf16; out_f32 there needs the fused f32 kernel, handled above. narrow
        # is the opposite request -- write the GEMV's IO dtype for a consumer that
        # is a linear -- so it cannot combine with out_f32.
        key = ("rmsnorm_apply_narrow"
               if narrow and not out_f32 and self.gemv_io != torch.float32
               else "rmsnorm_apply")
        y = self._kernel(key)(x2, w, p, float(eps), block_N, num_chunks, _THREADS)
        return y.reshape(*lead, w.shape[0])

    def rmsnorm_bwd(self, grad, x, w, eps, narrow: bool = False):
        # narrow is a forward-only output-dtype choice; the tape replays the
        # forward's kwargs verbatim, so it has to be accepted and ignored here.
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
        io = self.io
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
        """The fp4 bytes this cell's kernels read: rewritten in place once and
        tagged in ``_tl_layout``, so graph capture and save_hf (which untwiddles
        by the tag) see one truth. sm90 decodes the bf16-twiddled layout, sm70
        the fp16-twiddled twin (its GEMV has no bf16x2 math); other cells read
        the natural layout."""
        wq = self._dev(wq, wq.dtype)
        if getattr(wq, "_tl_layout", "natural") != "natural":
            return wq
        if "linear_fp4_gemv" not in _resolve(self.precision, self.arch):
            return wq
        if self.arch == "sm90":
            wq.copy_(reference.twiddle_fp4(wq))
            wq._tl_layout = "tw-bf16"
        elif self.arch == "sm70":
            wq.copy_(reference.twiddle_fp4_f16(wq))
            wq._tl_layout = "tw-f16"
        return wq

    def linear_fp4(self, x, wq, scale, master=None, oscale=None, residual=None):
        # ``master`` is recording-only (the STE grad lands on it)
        wq = self._served_fp4(wq)
        # An f16 plane belongs to the twiddled ladder below — the only consumer
        # compiled for it. materialize applies both rewrites together, so the two
        # always arrive together on the shipped path.
        sh = scale.dtype == self.scale_io == torch.float16
        scale = scale if sh else self._f32(scale)
        # Keep an f16 X: the twiddled sm70 ladder below is the only consumer and
        # it wants f16 anyway. Every other branch takes io (f32) as before.
        lead, x2 = self._rows(x, keep_f16=self.arch == "sm70")
        M, K, N = x2.shape[0], x2.shape[1], wq.shape[0]
        blk = K // scale.shape[1]  # the checkpoint's scale block (16 or 32)
        # M-row GEMV on the M=1 plan (the decode plan's n_partition is a 4096-thread block).
        # sm90 only: its maker takes a compile-time M and packs M activation rows into one
        # weight stream; sm70's GEMV is M=1-only (no packed FMA), so M>1 goes to the ladder.
        if (
            self.arch == "sm90"
            and 2 <= M <= _MGEMV
            and (gp := self._plan("linear_fp4", 1, N, K)) is not None
        ):
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
        # sm70 serves fp16-twiddled bytes, which the generic kernels (natural
        # nibbles) cannot read, so every M goes through the twiddle ladder.
        # M<=8 (decode/verify batch): one launch for M rows, X pre-packed f16.
        # M>8 (prefill) chunks with the M=32 twin: 32 rows share one W stream,
        # so M=512 is 16 launches/layer instead of 512. An untwiddle copy OOMs
        # here (the forward's GPU is full — the scratch that motivated eager
        # materialize twiddle).
        if self.arch == "sm70" and getattr(wq, "_tl_layout", "natural") != "natural":
            _, _, Np, Kp, _, bN = self._plan("linear_fp4", 1, N, K)
            wq1, sc1 = _pad2d(wq, Np, Kp // 2), _pad2d(scale, Np, Kp // blk)
            osc1 = self._ones(Np) if oscale is None else self._const_f32(oscale, Np)
            # Round M up the compiled ladder, and hand the kernel X pre-packed as
            # f16: otherwise it re-reads X per block and converts inside the tile
            # loop (78% of the M=8 bytes, 32 of ~49 per-row instructions). Packing
            # is worth 4.2x at M=32 and 1.1-1.45x at M=1, bit-exact — both paths
            # round to nearest f16. It used to be passed ONLY below M=8, which is
            # what made M>8 look like a hardware cliff at 122-128 us/row: the
            # extern is templated on M with no upper bound. 32 is the top rung, so
            # prefill chunks (M=512 is 16 launches/layer, not 512).
            chunks = _sm70_chunks(M)
            # Each chunk writes its own slice. `y2 = cat([y2, y])` rebuilt every
            # row accumulated so far on each iteration, so a 512-row prefill
            # copied 4352 rows where 512 would do — 269 ms of a 4269 ms tick
            # (6.3%), and quadratic in the chunk count. Decode is one chunk and
            # keeps the kernel's own buffer, so it still allocates nothing.
            # NOT _zeros2: that hands back a shared cached block, and other
            # callers read it as their residual while this one writes into it.
            # kernel's Y is f32 (kernels_linear.py: Y = T.empty((M, N), "float32")).
            y2 = torch.empty(M, N, dtype=torch.float32, device=self.device) if len(chunks) > 1 else None
            # ncols=2 only for the top rung. It pays where the GEMV is compute-bound
            # (prefill, M=32: 1.82x) and COSTS 4.9% of dense decode, because at M=1
            # the kernel is bandwidth-bound so there is no arithmetic to win, and
            # halving the grid starves shapes already at 5-33% of peak
            # (errors/2026-09-03-ncols2-cost-5-percent-of-decode.md).
            # Also requires Np == N: a padded plane pairs a real column with a PAD
            # column (the kernel derives half from its own N), and that garbage lands
            # inside the [:Mr, :N] slice below.
            nc2 = _NCOLS if Np == N and N % 2 == 0 else 1
            for m, Mr, Mk in chunks:
                nc = nc2 if Mk >= _NCOLS_MIN_M else 1
                # ncols by KEYWORD: positionally the 6th factory arg is `abl`, and
                # passing nc there ran the X_REUSE / NO_SCALE ablations instead --
                # both return wrong numbers, and X_REUSE's deleted loads read as a
                # 3.8x prefill "win" (errors/2026-09-03-the-ab-measured-abl-not-ncols.md).
                y = self._kernel("linear_fp4_gemv_sm70_m", Mk, 4, True, sh, ncols=nc)(
                    _pad2d(x2[m : m + Mr], Mk, Kp).to(torch.float16),
                    wq1, sc1, osc1, self._zeros2(Mk, Np), 32, bN, blk,
                )[:Mr, :N]
                if y2 is None:
                    y2 = y
                else:
                    y2[m : m + Mr] = y
            y = self._epilogue(y2, None, lead, N)
            return y if residual is None else y + residual
        plan = self._plan("linear_fp4", M, N, K)
        # Past the twiddled ladder every kernel is f32-IO. keep_f16 above may have
        # handed us f16 rows (sm70 with a narrow producer), so widen once here
        # rather than at each of the four branches below.
        if x2.dtype != self.io:
            x2 = self._dev(x2, self.io)
        if plan is not None:
            kernel, Mp, Np, Kp, bM, bN = plan
            wq, scale = _pad2d(wq, Np, Kp // 2), _pad2d(scale, Np, Kp // blk)
            # M=1 stays on the GEMV: mma8 measured 2.2x slower there (39.9 vs 87 tok/s)
            if 2 <= M <= _MX and "linear_fp4_mma8" in _resolve(self.precision, self.arch):
                Np32 = _round_up(N, 32)
                # _fit_rows, not _pad2d: Np32 can be NARROWER than the Np the line
                # above already padded these to. bN comes from the decode plan's N
                # tile (64), so N=24 gives Np=64 against Np32=32 -- the shape the
                # sm90 suite reported as `_pad2d: (64, 256) exceeds the target
                # [32, 256]`. Rows 24..64 are zero pad either way.
                wq = _fit_rows(wq, Np32, Kp // 2)
                scale = _fit_rows(scale, Np32, Kp // blk)
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
            # generic linear_fp4 is f32-IO; restore f32 for sm70's M>1 fallback
            # (sm90 never lands here — it has an MMA kernel). The scale plane
            # must be f32 too: sm70 serves f16 but reaches this only if its
            # twiddled-layout branch above did not claim the call.
            assert scale.dtype == torch.float32, "generic linear_fp4 wants an f32 scale plane"
            y2 = self._kernel("linear_fp4")(self._f32(x2), wq, scale, bM, bN, blk, _THREADS)[:M, :N]
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
            # Same non-ordering as the fp4 twin: Np32 can be narrower than the Np
            # these were padded to on the three lines above.
            w8 = _fit_rows(w8, Np32, Kp)
            wscale = _fit_rows(wscale, -(-Np32 // 128), Kp // 128)
            osc = self._ones(Np32) if oscale is None else self._const_f32(oscale, Np32)
            # x2 was already padded to Mp (=_snap_mma_tile(M,128), so 16 for every
            # M in 2..8) at the plan line above, and _pad2d REFUSES to shrink -- so
            # this asked for [8, K] from a 16-row tensor and raised
            # `_pad2d: (16, 5120) exceeds the target [8, 5120]`, killing every M in
            # 2..8: all speculation and dense B=8. M<=_MX<=Mp always, so the drop
            # is exact.
            xm = _fit_rows(x2, _MX, Kp)
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
        # Both serving rewrites need a kernel that reads the bytes: narrow the
        # scale plane (3.20 -> 1.60 GB on sm70, riding the device move so the f32
        # plane never lands on the card) and twiddle the nibbles. The twiddle is
        # here, not lazily in _served_fp4, because it allocates a same-size
        # scratch and by the first forward the KV cache + activations have left no
        # room on a 32GB card. Tagged so a train step's re-materialize skips it.
        _twiddle = {"sm90": reference.twiddle_fp4, "sm70": reference.twiddle_fp4_f16}.get(self.arch)
        served = _twiddle is not None and "linear_fp4_gemv" in _resolve(self.precision, self.arch)
        narrow = served and self.scale_io != torch.float32
        # The embedding table rides the same trick as the scale plane, and for a
        # sharper reason: materialize puts the bf16 table on the card, then the
        # gather's f32 cast made a SECOND copy — 2.37 + 4.74 = 7.11 GiB for the
        # 27B's 248320x5120, which is what OOMed `serve` on a 32 GB card. Cast
        # during the move and only the narrow one ever exists. Untied only: a
        # tied table is ALSO the lm_head weight, and that linear wants f32 IO.
        untied = any(k == "lm_head" or k.startswith("lm_head.") for k in out)
        emb = "embed_tokens" if untied and self.embed_io != torch.float32 else None
        moved = {
            k: v.to(self.device, self.scale_io)
            if narrow and k.endswith(".scale")
            else v.to(self.device, self.embed_io)
            if k == emb
            else v.to(self.device)
            for k, v in out.items()
        }
        if served:
            for k in moved:
                if k.endswith(".wq") and getattr(moved[k], "_tl_layout", "natural") == "natural":
                    moved[k].copy_(_twiddle(moved[k]))
                    moved[k]._tl_layout = "tw-bf16" if self.arch == "sm90" else "tw-f16"
        return moved

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
        # the M tile is the GQA group at every chain position: a verify width
        # rides the decode path while g*s still fits it
        chain = s <= _MAX_VERIFY_W and s * (q.shape[2] // k_cache.shape[1]) <= 128
        if self.arch == "sm90" and chain and "paged_attention_decode" in _resolve(self.precision, self.arch):
            out = self._paged_attention_decode(
                q, k_cache, v_cache, block_table, seq_lens, seq_q_lens, scale
            )
        elif self.arch == "sm70" and "paged_attention_split" in _resolve(
            self.precision, self.arch
        ):
            # Same idea without T.gemm/bf16 (sm70 has neither): the history is
            # split across the grid, so B=1 fills the card instead of running H
            # blocks at one thread each. S is in the grid too, which is what
            # makes a speculative verify affordable — the dense kernel is serial
            # in S as well as in history (39 ms at S=1 -> 1018 ms at S=4).
            #
            # Split count by query width (host-static, graph-safe; same idiom as
            # _paged_attention_decode). PO scales with S, and the two constraints
            # sit at different widths: at S=1 32 splits beat 16 by 1.20x while PO
            # is 3 MiB, and at prefill width they are 1.005x apart while PO is
            # 1.5 GiB and OOMs a 32 GB card. So spend splits where they are free.
            ks = sm70_kvsplit(s)
            po, pm, pl = self._kernel("paged_attention_split", KVSPLIT=ks)(
                self._f32(q),
                self._f32(k_cache),
                self._f32(v_cache),
                self._i32(block_table).contiguous(),
                self._i32(seq_lens).contiguous(),
                self._i32(seq_q_lens).contiguous(),
                float(scale),
                int(k_cache.shape[2]),
                _THREADS,
            )
            out = self._kernel("paged_attention_split_combine", KVSPLIT=ks)(
                po, pm, pl, _THREADS
            )
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

    def _paged_attention_decode(self, q, k_cache, v_cache, block_table, seq_lens, seq_q_lens,
                                scale):
        b, w, h, d = q.shape
        hkv = k_cache.shape[1]
        g = h // hkv
        block_m = _snap_mma_tile(g * w, 128)
        assert g * w <= block_m, f"GQA group x width {g * w} past the M tile cap {block_m}"
        # split count from the pool's reach (host-static, graph-safe): 64 splits
        # past 64K tokens or when the 16-grid under-fills the SMs
        max_tokens = block_table.shape[1] * k_cache.shape[2]
        wide = 16 * hkv * b < 2 * self._sms and max_tokens >= 64 * k_cache.shape[2]
        ks, sfx = (64, "_64") if (max_tokens > 65536 or wide) else (16, "")
        key = ("attn_ws", b, hkv, d, ks, block_m)
        ws = self._ones_cache.get(key)
        if ws is None:  # static workspace: graph-capturable
            ws = self._ones_cache[key] = (
                torch.empty(b, hkv, ks, block_m, d, dtype=torch.float32, device=self.device),
                torch.empty(b, hkv, ks, block_m, dtype=torch.float32, device=self.device),
                torch.empty(b, hkv, ks, block_m, dtype=torch.float32, device=self.device),
            )
        po, pm, pl = ws
        self._kernel("paged_attention_decode" + sfx)(
            self._dev(self._c(q), torch.bfloat16),
            self._dev(k_cache, torch.bfloat16), self._dev(v_cache, torch.bfloat16),
            self._i32(block_table), self._i32(seq_lens), self._i32(seq_q_lens),
            po, pm, pl, float(scale), int(k_cache.shape[2]), block_m,
        )
        return self._kernel("paged_attention_combine" + sfx)(po, pm, pl, g, w)

    def attention(self, q, k, v, scale, gate=None, q_pos=None, k_pos=None):
        """Dense causal GQA attention (training path). q [B,Tq,H,D], k/v [B,Tk,H,D].

        ``q_pos``/``k_pos`` are absolute sequence positions, needed once CP gives
        this rank a subset of the queries against every rank's keys."""
        # ponytail: torch-eager forward, tilelang kernel when perf demands
        out = reference.dense_attention(q, k, v, float(scale), q_pos, k_pos)
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
        # The pool's dtype is the kernel's: sm70 allocates f32 so attention
        # does not cast the whole plane per call.
        io = k_plane.dtype
        self._kernel("write_tokens")(
            self._dev(k, io).contiguous(),
            self._dev(v, io).contiguous(),
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
        blk = wq.shape[1] * 2 // scale.shape[1]
        # sm70 serves an f16 scale plane, and the sm90-only guard below excludes it:
        # whoever registers linear_fp4_bwd there widens at load, not via the
        # _const_f32 below (it would cache a permanent f32 copy of the whole plane).
        if not fp8 and blk == 16 and "linear_fp4_bwd" in kset:
            assert scale.dtype == torch.float32, "linear_fp4_bwd wants an f32 scale plane"
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

    def attention_bwd(self, grad, q, k, v, scale, q_pos=None, k_pos=None):
        # ponytail: torch-eager backward, tilelang kernel when perf demands
        return reference.dense_attention_bwd(grad, q, k, v, float(scale), q_pos, k_pos)

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
        io = self.io
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
        io = self.io
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

    def gdn_decode(self, q, k, v, g, beta, pool, slots, layer, keep_steps=0, **kw):
        """GDN core for a T-token decode tick in one launch, state updated in
        place in ``pool.states[slots, layer]``; None off sm90, or when T>1 is
        not a verify tick whose rows are all T wide (the caller then takes the
        gather -> linear_attn_chunk -> scatter path). ``keep_steps`` sends the
        per-chain-step state straight to the pool's step planes, so the verify
        pays no scatter."""
        t = q.shape[1]
        if "gdn_decode_fused" not in _resolve(self.precision, self.arch) or (
            t > 1 and keep_steps != t
        ):
            return None
        return self._kernel("gdn_decode_fused")(
            self._c(self._f32(q)),
            self._c(self._f32(k)),
            self._c(self._f32(v)),
            self._c(self._f32(kw["z"])),
            self._c(self._f32(g)),
            self._c(self._f32(beta)),
            self._c(self._const_f32(kw["dt_bias"])),
            self._c(self._const_f32(kw["a_log"])),
            self._c(self._const_f32(kw["norm_weight"])),
            self._c(self._const_f32(kw["conv1d_weight"])),
            pool.conv_windows,
            pool.win_parity,
            pool.states,
            self._i32(slots).contiguous(),
            # aliased onto the live planes when ks=0 leaves them unwritten
            pool.step_states if keep_steps else pool.states.unsqueeze(2),
            pool.step_windows if keep_steps else pool.conv_windows,
            int(layer),
            int(keep_steps),
            threads=pool.states.shape[-1],
        )

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
        if self.tp_world == 1:
            return reference.cross_entropy_loss_grad(logits, input_ids)
        # Vocab-parallel head: reduce to scalars per row rather than gathering the
        # [B, T, vocab] f32 row (0.947 MiB per row, 1.89 GiB at B=8 T=256 on the 27B).
        import torch.distributed as dist

        def all_reduce(t, op):
            # group= or the CE statistics are reduced across the dp replicas too,
            # and the loss stays finite and plausible while measuring the wrong batch.
            dist.all_reduce(t, op=dist.ReduceOp.SUM if op == "sum" else dist.ReduceOp.MAX,
                            group=self._tp_pg)
            return t

        # Every shard is exactly vloc wide: pad_vocab rounds the vocabulary to
        # to*world before sharding, so there is no ragged last shard for
        # tp_rank * vloc to skew.
        vloc = logits.shape[-1]
        return reference.cross_entropy_sharded(
            logits, input_ids, all_reduce, self.tp_rank * vloc, vloc * self.tp_world
        )

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
        # Read the table in whatever narrow dtype materialize put on the card and
        # widen on the gather; cast here only if it is still f32-or-wider. The
        # 27B's table is 248320x5120 — bf16 in the checkpoint (2.37 GiB), f32
        # 4.74 — and this used to cast unconditionally and cache the result, so
        # BOTH lived on the card: 7.11 GiB for one tensor, which OOMed `serve` on
        # its first token. Never narrow here: that would just be the same second
        # copy one dtype smaller. materialize is the only place that decides.
        dt = {torch.bfloat16: "bfloat16", torch.float16: "float16"}.get(table.dtype)
        if dt is None or table.dtype != self.embed_io:
            table, dt = self._const_f32(table), "float32"
        else:
            table = self._c(table.to(self.device))
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

    def sample_batch(self, logits, temperatures, top_ps, seeds, logprobs=True):
        return reference.sample_batch(logits, temperatures, top_ps, seeds, logprobs)


_BACKEND: Backend | None = None


def get_backend() -> Backend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = Backend(resolve_target())
    return _BACKEND
