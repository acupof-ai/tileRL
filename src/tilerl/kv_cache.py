"""Paged KV cache, linear-attention state pool, and prefix store.

Host-side bookkeeping (plain ints/lists) over torch tensors on the target
device, after agent-infer's ``host_paged_kv_pool.rs`` / ``prefix_store.rs``.
# ponytail: one refcount per block counts every owner (slots + prefix store);
# no preempt/swap, no cpu-offload — ``alloc_block`` raises on exhaustion.
"""

from __future__ import annotations

import contextlib
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .kv_tiers import DramSnapshots, HostKvPages

from .precision import Format, kv_format, nbytes

#: Tokens per physical KV block (paged-attention page size).
BLOCK_TOKENS = 16

_MASK64 = (1 << 64) - 1


def _rolling_hash(prev: int, token: int) -> int:
    # +1 so token 0 still perturbs the state; collisions are verified by PrefixStore.
    return ((prev * 1000003) ^ (token + 1)) & _MASK64


def _default_device() -> torch.device:
    # Lazy: this module must not import tilelang; a broken backend degrades to CPU.
    try:
        from tilerl_kernels.backend import get_backend

        return get_backend().device
    except Exception:  # noqa: BLE001
        return torch.device("cpu")


def _kv_fp8_ref():
    # Lazy for the same reason as the backend above, and so a tree without the two
    # reference functions still imports this module.
    from tilerl_kernels.reference import dequant_kv_fp8, quant_kv_fp8

    return quant_kv_fp8, dequant_kv_fp8


class PagedKvPool:
    """Paged K/V storage with a free-list allocator and per-block refcount.

    ``k_pool``/``v_pool`` are ``[num_planes, num_blocks, num_kv_heads,
    BLOCK_TOKENS, head_dim]``; block id N names the same page in every plane.
    Only full-attn layers own a plane: ``layer_map`` gives each plane's GLOBAL
    layer index, and every layer-indexed method takes a global index.

    A block with refcount > 1 is shared (a live slot plus the prefix store).
    Only whole blocks are ever published, so a shared block is never appended to
    and no copy-on-write is needed; :meth:`PrefixStore.insert` enforces that.

    ``kv_fp8`` stores the planes in that fp8 dtype with an f32 ``k_scale``/``v_scale``
    per ``(plane, block, kv_head, token)`` — one scale over head_dim
    (docs/experience/wins/2026-09-07-fp8-kv-pool-per-token-scales.md). The write paths
    then QUANTIZE; a plain ``.to(fp8)``
    would drop the scale and store plausible garbage.
    """

    def __init__(
        self,
        num_blocks: int,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        layer_map: tuple[int, ...] | None = None,
        kv_fp8: torch.dtype | None = None,
        cold_dtype: torch.dtype | None = None,
    ) -> None:
        self._layer_map = tuple(range(num_layers)) if layer_map is None else tuple(layer_map)
        self._plane = {g: d for d, g in enumerate(self._layer_map)}
        self.num_blocks = num_blocks
        self.num_layers = len(self._layer_map)
        self.num_kv_heads = num_kv_heads
        #: dtype a demoted page's K/V are stored in on the host tier; None keeps the
        #: native pool dtype. sm70's pool is f32 (no f16 attention path on Volta) and
        #: the host has half the room, so its cold pages narrow to f16 on the D2H copy
        #: and widen back on promote. The fp8 scale planes always stay native.
        self.cold_dtype = cold_dtype
        self.head_dim = head_dim
        self.device = _default_device() if device is None else torch.device(device)
        #: the IO dtype the attention kernel reads, which is the store dtype only off fp8
        self.dtype = dtype
        self.kv_fp8 = kv_fp8
        shape = (self.num_layers, num_blocks, num_kv_heads, BLOCK_TOKENS, head_dim)
        self.k_pool = torch.zeros(shape, dtype=kv_fp8 or dtype, device=self.device)
        self.v_pool = torch.zeros(shape, dtype=kv_fp8 or dtype, device=self.device)
        sshape = (self.num_layers, num_blocks, num_kv_heads, BLOCK_TOKENS)
        self.k_scale = None if kv_fp8 is None else torch.ones(sshape, device=self.device)
        self.v_scale = None if kv_fp8 is None else torch.ones(sshape, device=self.device)
        self._free: list[int] = list(range(num_blocks))
        self.refcount: list[int] = [0] * num_blocks
        #: HostKvPages tier when sparse KV demotion is enabled; None = dense pool.
        self.cold: HostKvPages | None = None

    def attach_cold(self, tier: HostKvPages) -> None:
        self.cold = tier

    def cold_page_nbytes(self) -> int:
        """Host bytes one demoted page occupies — same sum ``_page_blob`` builds:
        K+V in the cold dtype across every plane, plus the native fp8 scales."""
        store = self.cold_dtype or self.kv_fp8 or self.dtype
        es = torch.empty((), dtype=store).element_size()
        per = 2 * self.num_kv_heads * BLOCK_TOKENS * self.head_dim * es
        if self.kv_fp8 is not None:  # f32 k_scale/v_scale over head_dim
            per += 2 * self.num_kv_heads * BLOCK_TOKENS * 4
        return self.num_layers * per

    def cold_capacity_blocks(self) -> int:
        """Whole demoted pages the cold tier can hold: pinned host budget plus the
        countable SSD spill budget (0 without a host tier / spill file)."""
        if self.cold is None:
            return 0
        per = self.cold_page_nbytes()
        return (self.cold.budget_bytes + self.cold.ssd_capacity_bytes) // per

    def plane_of(self, layer_idx: int) -> int:
        """Pool plane for a model layer. The fp8 writers need the raw plane, not kv_layer()."""
        return self._plane[layer_idx]

    def kv_layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The attention operands for a layer. Off fp8 these are the pool planes themselves.

        Under fp8 this DEQUANTIZES the whole plane, every block, allocating two f32 copies of
        the entire pool -- 0.1 ms to 87.5 ms per tick at the 27B's shape, measured, because
        the cost is proportional to num_blocks and not to the sequence. Use
        :meth:`kv_operands` on any path that has a kernel able to read fp8; this stays for
        readers that cannot (the CPU cell, whose C backend has no sub-f32 type) and for tests.
        """
        p = self._plane[layer_idx]
        if self.kv_fp8 is None:
            return self.k_pool[p], self.v_pool[p]
        _, dequant = _kv_fp8_ref()
        return (dequant(self.k_pool[p : p + 1], self.k_scale[p : p + 1])[0].to(self.dtype),
                dequant(self.v_pool[p : p + 1], self.v_scale[p : p + 1])[0].to(self.dtype))

    def kv_operands(self, layer_idx: int) -> tuple[torch.Tensor, ...]:
        """``(k, v, k_scale, v_scale)`` -- the raw planes, no copy, scales None off fp8."""
        p = self._plane[layer_idx]
        if self.kv_fp8 is None:
            return self.k_pool[p], self.v_pool[p], None, None
        return self.k_pool[p], self.v_pool[p], self.k_scale[p], self.v_scale[p]

    @property
    def bytes_per_token(self) -> int:
        """K+V bytes one token costs across every plane, the fp8 scale plane included."""
        shape = (2 * self.num_layers, self.num_kv_heads, BLOCK_TOKENS, self.head_dim)
        plain = Format(self.k_pool.dtype.itemsize * 8)
        fmt = kv_format(self.head_dim) if self.kv_fp8 is not None else plain
        return nbytes(fmt, shape) // BLOCK_TOKENS

    def alloc_block(self) -> int:
        if not self._free:
            raise RuntimeError(f"PagedKvPool exhausted: all {self.num_blocks} blocks in use")
        block = self._free.pop()
        self.refcount[block] = 1
        return block

    def retain(self, block: int) -> None:
        if self.refcount[block] == 0:
            raise RuntimeError(f"retain: block {block} is free (refcount 0)")
        self.refcount[block] += 1

    def free_block(self, block: int) -> None:
        if self.refcount[block] <= 0:
            raise RuntimeError(f"free_block: block {block} already free (double free)")
        self.refcount[block] -= 1
        if self.refcount[block] == 0:
            self._free.append(block)

    def is_shared(self, block: int) -> bool:
        return self.refcount[block] > 1

    # ----------------------------------------------------- sparse-KV page tiering
    def _page_blob(self, block: int, *, non_blocking: bool = False) -> tuple[dict, int]:
        """A host copy of one page across every plane: K, V, and (under fp8) both
        per-token scale planes. Pinned when the pool is on a card so the promote
        H2D is async-capable; on the CPU cell it is a plain clone.

        ``non_blocking`` launches the D2H into the pinned host buffers without a
        per-page sync; a ``demotions()`` batch must synchronize before the device
        frame or these host buffers are reused/freed."""
        cuda = self.k_pool.is_cuda
        #: K/V narrow on the host copy when a cold dtype is set; the f32 fp8 scale
        #: planes stay their native dtype.
        cold_dtype = self.cold_dtype
        #: frame_snapshots() batches finish-publish copies: non-blocking D2H with
        #: one sync at context exit.
        non_blocking = non_blocking or getattr(self, "_snapshot_batching", False)
        planes = [
            ("k", self.k_pool[:, block], cold_dtype),
            ("v", self.v_pool[:, block], cold_dtype),
        ]
        if self.k_scale is not None:
            planes += [("ks", self.k_scale[:, block], None),
                       ("vs", self.v_scale[:, block], None)]
        blob = {}
        n = 0
        for key, t, cast_dtype in planes:
            # A pinned cross-dtype copy_ narrows on the D2H path directly; no extra
            # device cast tensor is allocated.
            host = torch.empty(t.shape, dtype=(cast_dtype or t.dtype),
                               device="cpu", pin_memory=cuda)
            host.copy_(t, non_blocking=(non_blocking and cuda))
            blob[key] = host
            n += host.numel() * host.element_size()
        return blob, n

    def _sync_cold(self) -> None:
        """One device sync before reused frames/host buffers are touched. This is
        the spy point the batched-demote gate counts (mock torch.cuda.synchronize)."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def demote_page(self, block: int, key=None) -> int:
        """Move one page (all planes of one block id, fp8 scales included) to the
        pinned host tier and release its device block back to THIS pool — no
        second pool. Returns the held byte count. A prefix-shared page is
        read-only wherever it lives and must not be demoted; a pool without a
        cold tier cannot demote; the block has to be live. The caller removes the
        freed id from its block tables (promotion allocates a different block).

        ``key`` names the host blob independently of the physical id: a frame is
        recycled (LIFO) while an older page's blob is still cold, so keying on
        the block id collides. The sparse engine keys on (req, logical page);
        the #500 demote-all/promote-all seam leaves it the physical id.

        Inside a ``with pool.demotions()`` batch the D2H copies launch
        non-blocking into pinned staging and the frame is NOT freed here — it is
        retained until the batch's single end sync, then held + freed. Reusing a
        frame before that sync would overwrite data an in-flight D2H reads."""
        if self.cold is None:
            raise RuntimeError("demote_page: no host page tier attached")
        if self.refcount[block] <= 0:
            raise RuntimeError(f"demote_page: block {block} is not live")
        if self.is_shared(block):
            raise RuntimeError(f"demote_page: block {block} is prefix-shared (refcount>1)")
        store_key = block if key is None else key
        if getattr(self, "_demote_batching", False):
            # Launch the D2H non-blocking; the batch owns the frame until sync.
            blob, n = self._page_blob(block, non_blocking=True)
            self._pending_demotes.append((block, store_key, blob, n))
            return n
        blob, n = self._page_blob(block)
        if not self.cold.hold(store_key, blob, n):
            raise RuntimeError(f"demote_page: host tier dropped block {block}")
        self.free_block(block)  # sole owner -> back to the same pool
        return n

    @contextlib.contextmanager
    def demotions(self):
        """Batch the D2H copies of a tick's departing pages into ONE device sync
        before any frame is reused. Each demote_page launches its copies
        non-blocking into its own pinned blob and the frame stays live (so a
        recycled allocation cannot overwrite it); at exit we synchronize once,
        then hold every blob in the cold tier and return all frames to the pool.
        Off cuda the copies are plain synchronous clones, so this only changes
        bookkeeping there. The demote half of ``promotions``."""
        pending = getattr(self, "_pending_demotes", [])
        held_before = len(pending)
        self._pending_demotes = pending
        self._demote_batching = True
        try:
            yield self
        finally:
            self._demote_batching = False
            batch = pending[held_before:]
            if batch:
                # Wait once for every in-flight non-blocking D2H; only now are the
                # pinned blobs valid and the frames safe to reuse.
                self._sync_cold()
                # On a hold failure the frames were already pulled from the caller's
                # block tables (demote_page runs before this exit). Return EVERY
                # batch frame to the pool before the error propagates, so the row
                # the engine then finishes does not leak or double-free the blocks:
                # _release frees only what is still in req.blocks.
                i = 0
                try:
                    for block, store_key, blob, n in batch:
                        if not self.cold.hold(store_key, blob, n):
                            raise RuntimeError(
                                f"demotions: host tier dropped block {block}")
                        self.free_block(block)
                        i += 1
                except BaseException:
                    # Frames [0,i) already held+freed above. Free only the rest:
                    # their callers pulled them from req.blocks before this exit,
                    # so _release does not see them; freeing them again here keeps
                    # the pool balanced (no double free, no leak).
                    for block, _k, _blob, _n in batch[i:]:
                        if self.refcount[block] > 0:
                            self.free_block(block)
                    raise
            del pending[held_before:]

    @contextlib.contextmanager
    def frame_snapshots(self):
        """Batch the D2H copies of several :meth:`_page_blob` snapshots into ONE
        device sync. Finish-publish snapshots a whole still-resident prefix off
        live frames; without this each snapshot's copy synchronised per page
        (measured 1 s of a 2.6 s request-finish tick). The frames are NOT freed
        or demoted here — the caller owns their lifecycle — so unlike
        ``demotions()`` this only batches the copies. Pinned blobs returned
        inside the context must not be read (or spilled to disk) until its exit
        sync — the caller defers every cold-tier commit until then."""
        self._snapshot_batching = True
        try:
            yield self
        finally:
            self._snapshot_batching = False
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

    def promote_keyed(self, key) -> int:
        """Reload a blob held under ``demote_page(key=...)`` into a FRESH block
        and return its new id. Raises if it was never held or byte-LRU evicted
        (the selector must not name it). Inside a ``with pool.promotions()`` the
        H2D is non-blocking with one sync at batch end; otherwise it syncs here."""
        if self.cold is None:
            raise RuntimeError("promote_keyed: no host page tier attached")
        blob = self.cold.take(key)
        if blob is None:
            raise RuntimeError(f"promote_keyed: {key!r} is not held on the host")
        new = self.alloc_block()
        nb = blob["k"].is_pinned()
        batched = getattr(self, "_promote_batching", False)
        self.k_pool[:, new].copy_(blob["k"], non_blocking=nb)
        self.v_pool[:, new].copy_(blob["v"], non_blocking=nb)
        if self.k_scale is not None:
            self.k_scale[:, new].copy_(blob["ks"], non_blocking=nb)
            self.v_scale[:, new].copy_(blob["vs"], non_blocking=nb)
        if batched:
            # Retained by the promotions() context until its single end-of-batch
            # sync; no per-page wait here.
            self._pending_promote_blobs.append(blob)
        elif self.device.type == "cuda":
            # The pinned blob is released when this call returns; a non_blocking
            # H2D still in flight would then read a buffer the host allocator may
            # reuse.
            torch.cuda.synchronize(self.device)
        return new

    @contextlib.contextmanager
    def promotions(self):
        """Batch the H2D copies of several promote_keyed calls into ONE device
        sync at exit. Each promote still allocates its frame and launches its
        copies non-blocking; the source blobs are retained here until the single
        end-of-batch synchronization, so a tick that fetches N pages pays one
        sync, not N. Off cuda the copies are plain synchronous clones, so this
        only changes bookkeeping there."""
        self._pending_promote_blobs = getattr(self, "_pending_promote_blobs", [])
        held_before = len(self._pending_promote_blobs)
        self._promote_batching = True
        try:
            yield self
        finally:
            self._promote_batching = False
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            del self._pending_promote_blobs[held_before:]

    def promote_page(self, old_block: int) -> int:
        """Promote a blob keyed by its old physical block (#500 seam). The device
        block id changes across a round trip; the caller splices the new id in."""
        return self.promote_keyed(old_block)

    def page_location(self, block: int) -> str:
        """Where the logical page named by an id currently lives. A demoted id is
        no longer a live device block, so this must be asked before treating an id
        as a physical frame: 'device' (live refcount), 'host' (in the cold tier),
        or 'free'."""
        if self.cold is not None and block in self.cold:
            return "host"
        if 0 <= block < self.num_blocks and self.refcount[block] > 0:
            return "device"
        return "free"

    def shared_promote(self, blob: dict) -> int:
        """Copy a SHARED prefix-page blob (K/V + fp8 scales) into a fresh PRIVATE
        block. The shared blob stays read-only with the store; the new block is the
        adopting request's own. The promoted K/V use the pool dtype (cold narrowing
        widened here, same as promote_page).

        Batched like :meth:`promote_keyed`: inside ``promotions()`` every copy is
        non-blocking and there is ONE sync at batch exit, not one per page. A
        prefix hit selects ~200 shared pages on a refresh tick; an unconditional
        per-page sync here turned that into ~200 full-stream stalls (measured
        1.5-1.9 s refresh ticks, model envelope only — ssd_mmap=0)."""
        new = self.alloc_block()
        nb = blob["k"].is_pinned()
        self.k_pool[:, new].copy_(blob["k"], non_blocking=nb)
        self.v_pool[:, new].copy_(blob["v"], non_blocking=nb)
        if self.k_scale is not None and "ks" in blob:
            self.k_scale[:, new].copy_(blob["ks"], non_blocking=nb)
            self.v_scale[:, new].copy_(blob["vs"], non_blocking=nb)
        batched = getattr(self, "_promote_batching", False)
        if not batched and self.device.type == "cuda":
            # The pinned blob could be reused/freed on return; an in-flight
            # non-blocking H2D would then read stale bytes.
            torch.cuda.synchronize(self.device)
        return new

    def _store_fp8(self, plane: int, blk: torch.Tensor, off: torch.Tensor,
                   k: torch.Tensor, v: torch.Tensor) -> None:
        """Quantize ``k``/``v`` ([n, num_kv_heads, head_dim]) into ``blk``/``off``.

        A write touches only the tokens it writes: the scale is per (block, head, token), so a
        later token's larger absmax cannot saturate an earlier one and there is nothing to
        re-round. Quantizing from the stored fp8 instead compounds -- 0.338 vs 0.059 max rel
        error over 16 appends, worst on the FIRST token written.
        """
        quant, _ = _kv_fp8_ref()
        for pool, scale, x in ((self.k_pool, self.k_scale, k), (self.v_pool, self.v_scale, v)):
            # [n,H,D] -> [n,H,1,D]: quant reduces over the last axis, one scale per token-head
            q, s = quant(x.to(self.device, torch.float32).unsqueeze(2), self.kv_fp8)
            pool[plane, blk, :, off] = q[:, :, 0]
            scale[plane, blk, :, off] = s[:, :, 0]

    def write_block(
        self,
        block: int,
        offset: int,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: int = 0,
    ) -> None:
        """Write ``k``/``v`` ([num_kv_heads, n, head_dim]) at ``layer``/token ``offset``."""
        if k.shape != v.shape:
            raise ValueError(f"write_block: k/v shape mismatch {k.shape} vs {v.shape}")
        if k.ndim != 3 or k.shape[0] != self.num_kv_heads or k.shape[2] != self.head_dim:
            raise ValueError(
                f"write_block: expected [num_kv_heads={self.num_kv_heads}, n, "
                f"head_dim={self.head_dim}], got {tuple(k.shape)}"
            )
        n = k.shape[1]
        if offset < 0 or offset + n > BLOCK_TOKENS:
            raise ValueError(
                f"write_block: span [{offset}, {offset + n}) outside block of {BLOCK_TOKENS}"
            )
        layer = self._plane[layer]
        if self.kv_fp8 is not None:
            self._store_fp8(
                layer,
                torch.full((n,), block, dtype=torch.long, device=self.device),
                torch.arange(offset, offset + n, device=self.device),
                k.transpose(0, 1),
                v.transpose(0, 1),
            )
            return
        self.k_pool[layer, block, :, offset : offset + n].copy_(
            k.to(self.device, self.k_pool.dtype)
        )
        self.v_pool[layer, block, :, offset : offset + n].copy_(
            v.to(self.device, self.v_pool.dtype)
        )

    def write_tokens(self, k: torch.Tensor, v: torch.Tensor, kv, layer_idx: int) -> None:
        """Write k/v [B,T,Hkv,D] at each row's tail ``[seq_len-seq_q, seq_len)``
        through the block table. Torch fallback for cells without the scatter
        kernel; the engine guarantees those positions are exclusively owned.
        # ponytail: 2 syncs/layer (the two tolist), not 0 — a mask instead of a
        # per-row length is the kernel's job on a cell that has one.
        """
        b, t, _, _ = k.shape
        sql = kv.seq_q_lens
        plane = self._plane[layer_idx]
        dev = self.k_pool.device
        lens = [t] * b if sql is None else sql.tolist()
        ends = kv.seq_len.tolist()
        for bi in range(b):
            sq = int(lens[bi])
            # pos starts on the block table's device; the pool may live on
            # another (CPU table + mps pool is the metal parity path).
            pos = torch.arange(int(ends[bi]) - sq, int(ends[bi]),
                               device=kv.block_table.device)
            # Sparse path: the table holds only this row's OWN span, whose first
            # column is logical page ``page_base[bi]`` (default 0 = dense table).
            base = int(kv.page_base[bi]) if getattr(kv, "page_base", None) is not None else 0
            blk = kv.block_table[bi, (pos // BLOCK_TOKENS - base).clamp_min(0)].to(dev)
            off = (pos % BLOCK_TOKENS).to(dev)
            if self.kv_fp8 is not None:
                # per row, because rows own disjoint blocks but share none of their spans
                self._store_fp8(plane, blk.long(), off, k[bi, :sq], v[bi, :sq])
                continue
            self.k_pool[plane, blk, :, off, :] = k[bi, :sq].to(self.k_pool.dtype)
            self.v_pool[plane, blk, :, off, :] = v[bi, :sq].to(self.v_pool.dtype)

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - len(self._free)

    @staticmethod
    def blocks_for_tokens(tokens: int) -> int:
        return (tokens + BLOCK_TOKENS - 1) // BLOCK_TOKENS



class LinearStatePool:
    """Per-slot state for the gated-delta layers.

    ``states``: [num_slots, num_linear_layers, num_heads, head_dim, head_dim].
    ``conv_windows``: [num_slots, num_linear_layers, 2, kernel-1, qkv_dim] — two
    planes selected by ``win_parity[slot]``: the sm90 fused decode kernel reads
    plane p and writes 1-p (an in-place shift would race across blocks), then
    the tick flips the parity. ``step_states``/``step_windows`` hold the state
    after each chain step of a speculative verify (``select_step`` adopts one).
    """

    def __init__(
        self,
        num_slots: int,
        num_linear_layers: int,
        num_heads: int,
        head_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        conv_window: int = 0,
        conv_dim: int = 0,
        spec_steps: int = 0,
    ) -> None:
        self.device = _default_device() if device is None else torch.device(device)
        self.num_slots = num_slots
        self.states = torch.zeros(
            num_slots,
            num_linear_layers,
            num_heads,
            head_dim,
            head_dim,
            dtype=dtype,
            device=self.device,
        )
        self.conv_windows = (
            torch.zeros(
                num_slots,
                num_linear_layers,
                2,
                conv_window,
                conv_dim,
                dtype=dtype,
                device=self.device,
            )
            if num_linear_layers > 0 and conv_window > 0
            else None
        )
        self.step_states = (
            torch.zeros(num_slots, num_linear_layers, spec_steps, num_heads, head_dim, head_dim,
                        dtype=dtype, device=self.device)
            if spec_steps and num_linear_layers > 0
            else None
        )
        self.step_windows = (
            torch.zeros(num_slots, num_linear_layers, spec_steps, conv_window, conv_dim,
                        dtype=dtype, device=self.device)
            if self.step_states is not None and self.conv_windows is not None
            else None
        )
        self.win_parity = torch.zeros(num_slots, dtype=torch.int32, device=self.device)
        self._free: list[int] = list(range(num_slots))

    def select_step(self, slot: int, step: int) -> None:
        if self.step_states is None:
            return
        self.states[slot].copy_(self.step_states[slot, :, step])
        if self.step_windows is not None:
            self.window_restore(slot, self.step_windows[slot, :, step])

    def window_snapshot(self, slot: int) -> torch.Tensor | None:
        if self.conv_windows is None:
            return None
        return self.conv_windows[slot, :, int(self.win_parity[slot])].clone()

    def window_restore(self, slot: int, snap: torch.Tensor) -> None:
        self.conv_windows[slot, :, 0].copy_(snap)
        self.win_parity[slot] = 0

    @property
    def free_slots(self) -> int:
        return len(self._free)

    def alloc_slot(self) -> int:
        if not self._free:
            raise RuntimeError(f"LinearStatePool exhausted: all {self.num_slots} slots in use")
        slot = self._free.pop()
        # The zeroing can raise (an OOM on a 27B is not hypothetical) after the pop has
        # already taken the slot out of _free, and the caller never receives it -- so
        # nothing can free_slot it and the pool loses a slot for the process's life.
        try:
            self.states[slot].zero_()
            if self.conv_windows is not None:
                self.conv_windows[slot].zero_()
        except Exception:
            self._free.append(slot)
            raise
        return slot

    def free_slot(self, slot: int) -> None:
        if slot in self._free:
            raise RuntimeError(f"free_slot: slot {slot} already free (double free)")
        self._free.append(slot)


def _nbytes(state: Any) -> int:
    if isinstance(state, torch.Tensor):
        return state.nbytes
    return sum(_nbytes(s) for s in state) if isinstance(state, tuple) else 0


@dataclass(frozen=True)
class PrefixHit:
    """Matched token count, the store-retained blocks covering [0, length) and the
    recurrent-state snapshot taken at that boundary."""

    length: int
    blocks: tuple[int, ...]
    state: Any = None


@dataclass(slots=True)
class _Entry:
    eid: int
    tokens: tuple[int, ...]
    blocks: tuple[int, ...]
    h: int
    state: Any
    nbytes: int
    #: the snapshot lives in the DRAM tier, not in ``state``; ``nbytes`` is kept so a
    #: promotion can re-charge exactly what the demotion credited
    demoted: bool = False


class NoPrefixStore:
    """Never matches, never retains: a training rollout must not serve KV
    computed under an earlier policy. Also the miss-path double for tests."""

    def lookup(self, tokens: Sequence[int]) -> PrefixHit | None:
        return None

    def insert(self, tokens: Sequence[int], blocks: Sequence[int], state: Any = None) -> bool:
        return False

    def retire(self, tokens: Sequence[int]) -> bool:
        return False

    def evict_until_free(self, blocks: int) -> None:
        return None

    def reclaimable_blocks(self) -> int:
        return 0

    def clear(self) -> None:
        return None

    def stats(self) -> dict[str, int]:
        return {"entries": 0, "capacity": 0, "entries_capacity": 0, "state_bytes": 0,
                "lookups_matched": 0, "lookups_missed": 0,
                "evictions": 0, "blocks_freed": 0, "superseded": 0}


class PrefixStore:
    """Rolling-hash prefix cache over a :class:`PagedKvPool`.

    An entry maps a prefix's full hash to (token tuple, physical blocks, state
    snapshot); every hash hit is verified against the stored tokens. Insert
    retains every block. Under ``state_bytes`` pressure a ``dram`` tier, when one is
    passed, takes the LRU snapshot and the entry stays matchable; without a tier, and
    always at ``capacity`` or under block pressure, LRU eviction releases blocks and
    snapshot together.

    LRU, not FIFO: both real workloads re-read a growing prefix -- a chat client
    resends the whole conversation each turn, and a group of rollouts shares one
    prompt -- so the hottest entry is the oldest one, and FIFO evicted exactly it.
    """

    def __init__(
        self,
        pool: PagedKvPool,
        capacity: int = 4096,
        state_bytes: int = 8 << 30,
        dram: DramSnapshots | None = None,
    ) -> None:
        self._pool = pool
        self.capacity = capacity
        self.state_bytes = state_bytes
        self._dram = dram
        self._state_used = 0
        self._roll = _rolling_hash
        self._entries: dict[int, list[_Entry]] = {}
        # Recency IS the iteration order, so there is no second structure to keep in
        # step with it: the eviction victim is the first key, and a hit moves its key
        # to the end. A parallel deque plus dict is what this replaces.
        self._by_id: OrderedDict[int, _Entry] = OrderedDict()
        self._next_id = 0
        self.lookups_matched = 0
        self.lookups_missed = 0
        self.evictions = 0
        # A publisher retiring its own superseded entry. Separate from `evictions` because
        # it is the absence of pressure, not pressure.
        self.superseded = 0
        self.blocks_freed = 0
        #: byte size of one GDN snapshot, learned from the first insert
        self._snapshot_bytes = 0

    def _hash_all(self, tokens: Sequence[int]) -> int:
        h = 0
        for t in tokens:
            h = self._roll(h, int(t))
        return h

    def insert(self, tokens: Sequence[int], blocks: Sequence[int], state: Any = None) -> bool:
        """Cache ``tokens`` (covered by ``blocks``) with its ``state`` snapshot and retain
        the blocks; True when a new entry was retained, False for a duplicate.
        """
        tokens = tuple(int(t) for t in tokens)
        blocks = tuple(blocks)
        if len(blocks) * BLOCK_TOKENS > len(tokens):
            raise ValueError(
                f"insert: {len(blocks)} blocks cover {len(blocks) * BLOCK_TOKENS} tokens but only "
                f"{len(tokens)} were given; publishing a partial block shares a page a slot is "
                "still appending to"
            )
        if not tokens:
            return False
        expected = PagedKvPool.blocks_for_tokens(len(tokens))
        if len(blocks) != expected:
            raise ValueError(
                f"insert: {len(tokens)} tokens need {expected} blocks, got {len(blocks)}"
            )
        h = self._hash_all(tokens)
        for e in self._entries.get(h, ()):
            if e.tokens == tokens:
                return False
        entry = _Entry(self._next_id, tokens, blocks, h, state, _nbytes(state))
        self._state_used += entry.nbytes
        self._next_id += 1
        self._entries.setdefault(h, []).append(entry)
        self._by_id[entry.eid] = entry
        for b in blocks:
            self._pool.retain(b)
        if state is not None and self._snapshot_bytes == 0:
            self._snapshot_bytes = sum(t.numel() * t.element_size()
                                       for t in state if t is not None)
        while len(self._by_id) > self.capacity or self._state_used > self.state_bytes:
            # State-byte pressure with a tier is not a reason to lose a prefix: demote the
            # LRU snapshot instead and keep the entry matchable. Only when nothing is left
            # to demote (or there is no tier) does the entry go.
            # The count test is not a conflated guard: `_demote_one` keeps the entry in
            # `_by_id`, so a demote cannot make the count term shrink and only eviction can
            # satisfy it. This reads "the count is not the binding term" — twice mistaken for
            # code that makes the demote path unreachable (2026-09-08).
            if (
                self._dram is not None
                and len(self._by_id) <= self.capacity
                and self._demote_one()
            ):
                continue
            self._evict_one()
        return True

    def lookup(self, tokens: Sequence[int]) -> PrefixHit | None:
        """Longest stored prefix of ``tokens``, or ``None``."""
        tokens = tuple(int(t) for t in tokens)
        h = 0
        prefix_hashes: list[int] = []
        for t in tokens:
            h = self._roll(h, t)
            prefix_hashes.append(h)
        for i in range(len(tokens), 0, -1):
            for e in self._entries.get(prefix_hashes[i - 1], ()):
                if e.tokens == tokens[:i]:
                    if e.demoted:
                        # 12.7 ms measured for a 27B snapshot at 11.52 GiB/s pinned,
                        # against 163 s to re-prefill the 11019-token prompt it serves.
                        e.state = self._dram.promote(e.eid, self._pool.device)
                        e.demoted = False
                        if e.state is None:
                            # The tier's byte LRU dropped it. Adopting the blocks without
                            # the snapshot would run the GDN layers from a zero state over
                            # KV that is not zero -- wrong, and silent. Drop the entry and
                            # keep looking at shorter prefixes.
                            e.nbytes = 0
                            self._drop(e)
                            break
                        self._state_used += e.nbytes
                    self.lookups_matched += 1  # per LOOKUP; /health's prefix_hits is per admission
                    self._by_id.move_to_end(e.eid)  # this is the whole of "recently used"
                    return PrefixHit(i, e.blocks, e.state)
        self.lookups_missed += 1
        return None

    def evict_until_free(self, blocks: int) -> None:
        while self._pool.free_blocks < blocks and self._by_id:
            self._evict_one()

    def reclaimable_blocks(self) -> int:
        """Blocks eviction would actually free: refcount 0 once the store's own holds go."""
        held: dict[int, int] = {}
        for entry in self._by_id.values():
            for b in entry.blocks:
                held[b] = held.get(b, 0) + 1
        # Per entry, not per block: a growing prefix republishes, so shared blocks sit above 1.
        return sum(1 for b, n in held.items() if self._pool.refcount[b] == n)

    def _drop(self, entry: _Entry, *, evicted: bool = True) -> None:
        """Remove one entry and release everything it holds. The single teardown path:
        eviction, a promotion that came back empty, and ``clear`` all go through it, so
        the block frees and the byte accounting cannot drift between them.

        ``evicted=False`` for a publisher retiring its OWN superseded entry: the bytes and
        blocks go back the same way, but it is not eviction pressure, and counting it as
        one would hide the pressure `/health` exists to show."""
        del self._by_id[entry.eid]
        chain = self._entries[entry.h]
        chain.remove(entry)
        if not chain:
            del self._entries[entry.h]
        before = self._pool.free_blocks
        for b in entry.blocks:
            self._pool.free_block(b)
        # Blocks, not entries: `free_block` is a refcount decrement, so an entry whose blocks
        # a live request retains frees nothing while still counting an eviction.
        self.blocks_freed += self._pool.free_blocks - before
        if not entry.demoted:
            self._state_used -= entry.nbytes
        if self._dram is not None:
            self._dram.forget(entry.eid)
        if evicted:
            self.evictions += 1
        else:
            self.superseded += 1

    def _evict_one(self) -> None:
        eid = next(iter(self._by_id))  # least recently used
        self._drop(self._by_id[eid])

    def retire(self, tokens: Sequence[int]) -> bool:
        """Drop the publisher's own earlier entry for exactly ``tokens``. True if one went.

        A conversation publishes a nested family, each key a prefix of the next, and only
        the longest can ever serve it again -- so the previous one is dead the moment the
        next lands. Retiring it here bounds one conversation's live contribution instead of
        leaving `budget` of them for LRU to churn through.
        """
        tokens = tuple(int(t) for t in tokens)
        for e in self._entries.get(self._hash_all(tokens), ()):
            if e.tokens == tokens:
                self._drop(e, evicted=False)
                return True
        return False

    def _demote_one(self) -> bool:
        """Move the LRU resident snapshot to the DRAM tier, keeping the entry. True when
        one moved -- False means every entry is already demoted and only eviction is left.

        This is what relieves `state_bytes` pressure without giving up a prefix: the
        entry keeps its tokens and its blocks and stays in the index, so `lookup` still
        matches it and `promote` brings the snapshot back on the hit. Measured live: 43
        of 43 evictions happened with 64% of the block pool free, i.e. every one was
        state bytes and every one was avoidable this way.
        """
        for eid, entry in self._by_id.items():
            if entry.state is None or entry.demoted:
                continue
            if not self._dram.demote(eid, entry.state):
                return False
            self._state_used -= entry.nbytes
            entry.state, entry.demoted = None, True
            return True
        return False

    def clear(self) -> None:
        """Drop every entry: the KV behind them was computed under older weights.

        An optimizer step invalidates the whole store at once, so this is
        ``_evict_one`` to exhaustion rather than a second teardown path -- the
        block frees and the state-bytes accounting have to match it exactly. A demoted
        snapshot goes with its entry, because ``_drop`` calls the tier's ``forget``: a
        stale snapshot surviving in DRAM is exactly the off-policy state this exists to
        refuse.
        """
        while self._by_id:
            self._evict_one()

    def _entries_capacity(self) -> int:
        """How many entries can be resident at once, from whichever budget binds.

        A snapshot is a constant size at any prefix length, so bytes cap the count. The
        host tier counts too: `_demote_one` moves a snapshot there and leaves the entry
        matchable, so a store with `state_bytes=0` and a tier holds entries, not none.
        """
        if self._snapshot_bytes <= 0:
            return self.capacity
        avail = self.state_bytes + (0 if self._dram is None else self._dram.budget_bytes)
        return min(self.capacity, avail // self._snapshot_bytes)

    def stats(self) -> dict[str, int]:
        st = {
            "entries": len(self._by_id),
            "capacity": self.capacity,
            "entries_capacity": self._entries_capacity(),
            "state_bytes": self._state_used,
            # The budget beside the fill, for the reason `dram_budget` exists: `state_bytes`
            # alone cannot say whether the store is at its ceiling, so a reader cannot tell
            # state pressure from block pressure. Set from mem_get_info at build time and
            # otherwise unknowable from outside.
            "state_bytes_budget": self.state_bytes,
            "lookups_matched": self.lookups_matched,
            "lookups_missed": self.lookups_missed,
            "evictions": self.evictions,
            "superseded": self.superseded,
            "blocks_freed": self.blocks_freed,
            "demoted": sum(1 for e in self._by_id.values() if e.demoted),
        }
        if self._dram is not None:
            st.update(self._dram.stats())
        return st


@dataclass
class BatchKv:
    """Batch-level KV descriptor for one model forward (the model reads it duck-typed).

    ``seq_len`` is each row's logical length AFTER this forward. ``seq_q_lens`` is
    the per-row valid query count; rows are left-aligned and padded to a shared T,
    and None means every row is valid for the full T.
    """

    block_table: torch.Tensor  # [B, num_blocks] long, padded with 0
    seq_len: torch.Tensor  # [B] long
    state_slot: torch.Tensor  # [B] long
    kv_pool: Any
    state_pool: Any
    seq_q_lens: torch.Tensor | None = None  # [B] valid query tokens per row
    keep_steps: int = 0  # verify: keep the recurrent state after each of the first N chain tokens
    #: sparse engine only: [B] logical page index of block_table column 0 (dense: 0/None)
    page_base: torch.Tensor | None = None
    #: sparse engine only: the per-tick SparseForward selection descriptor
    sparse: Any = None
    #: Draft sliding-window probe only: when set, paged_attention READS this
    #: descriptor (a trailing-window view) while write_tokens still writes through
    #: the full table/seq_len above, so the read window never moves a write or
    #: drops retained draft KV. None = read the full descriptor (default).
    read_kv: BatchKv | None = None

    def inputs_for(self, ids, pos, row: int) -> dict:
        """Clone of every tensor the forward reads for row ``row``.

        Tools call this instead of guessing what counts as an input: the token
        ids, positions, block table, the K/V (and fp8 scales) it names, the
        recurrent state, conv window and window parity for its slot. The
        recurrent state does not route through the block table, so every
        ad-hoc dump kept missing it
        (errors/2026-09-09-decode-graph-run-to-run-nondeterminism.md).
        """
        ids_t = ids if isinstance(ids, torch.Tensor) else torch.as_tensor(ids)
        pos_t = pos if isinstance(pos, torch.Tensor) else torch.as_tensor(pos)
        n = (int(self.seq_len[row]) + BLOCK_TOKENS - 1) // BLOCK_TOKENS
        blocks = self.block_table[row, :n]
        slot = int(self.state_slot[row])
        pool, state = self.kv_pool, self.state_pool
        d = {
            "ids": ids_t[row].clone(),
            "pos": pos_t[row].clone(),
            "block_table": blocks.clone(),
            "seq_len": self.seq_len[row].clone(),
            "state_slot": self.state_slot[row].clone(),
            "k": pool.k_pool[:, blocks, ...].clone(),
            "v": pool.v_pool[:, blocks, ...].clone(),
            "states": state.states[slot].clone(),
            "win_parity": state.win_parity[slot].clone(),
        }
        if self.seq_q_lens is not None:
            d["seq_q_lens"] = self.seq_q_lens[row].clone()
        if pool.k_scale is not None:
            d["k_scale"] = pool.k_scale[:, blocks, ...].clone()
            d["v_scale"] = pool.v_scale[:, blocks, ...].clone()
        if state.conv_windows is not None:
            d["conv_windows"] = state.conv_windows[slot].clone()
        return d


