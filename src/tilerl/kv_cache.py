"""Paged KV cache, linear-attention state pool, and prefix store.

Host-side bookkeeping (plain ints/lists) over torch tensors on the target
device, after agent-infer's ``host_paged_kv_pool.rs`` / ``prefix_store.rs``.
# ponytail: one refcount per block counts every owner (slots + prefix store);
# no preempt/swap, no cpu-offload — ``alloc_block`` raises on exhaustion.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .precision import Format, kv_format, nbytes

#: Tokens per physical KV block (paged-attention page size).
BLOCK_TOKENS = 16

_MASK64 = (1 << 64) - 1


def _blob_bytes(st: dict) -> int:
    """Bytes of a spilled blob, for the fetch-rate accounting: state snapshot or KV+scale."""
    return sum(t.numel() * t.element_size()
               for t in (st.get("states"), st.get("windows"), st.get("k"), st.get("v"),
                         st.get("ks"), st.get("vs")) if t is not None)


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
    (docs/design-fp8-kv.md). The write paths then QUANTIZE; a plain ``.to(fp8)``
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
    def _page_blob(self, block: int) -> tuple[dict, int]:
        """A host copy of one page across every plane: K, V, and (under fp8) both
        per-token scale planes. Pinned when the pool is on a card so the promote
        H2D is async-capable; on the CPU cell it is a plain clone."""
        cuda = self.k_pool.is_cuda
        #: K/V narrow on the host copy when a cold dtype is set; the f32 fp8 scale
        #: planes stay their native dtype.
        cold_dtype = self.cold_dtype
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
            host.copy_(t)
            blob[key] = host
            n += host.numel() * host.element_size()
        return blob, n

    def demote_page(self, block: int) -> int:
        """Move one page (all planes of one block id, fp8 scales included) to the
        pinned host tier and release its device block back to THIS pool — no
        second pool. Returns the held byte count. A prefix-shared page is
        read-only wherever it lives and must not be demoted; a pool without a
        cold tier cannot demote; the block has to be live. The caller removes the
        freed id from its block tables (promotion allocates a different block)."""
        if self.cold is None:
            raise RuntimeError("demote_page: no host page tier attached")
        if self.refcount[block] <= 0:
            raise RuntimeError(f"demote_page: block {block} is not live")
        if self.is_shared(block):
            raise RuntimeError(f"demote_page: block {block} is prefix-shared (refcount>1)")
        blob, n = self._page_blob(block)
        if not self.cold.hold(block, blob, n):
            raise RuntimeError(f"demote_page: host tier dropped block {block}")
        self.free_block(block)  # sole owner -> back to the same pool
        return n

    def promote_page(self, old_block: int) -> int:
        """Reload a demoted page into a FRESH block allocated from this pool and
        return its new id. The device block id changes across a round trip; the
        caller splices the new id into the block table. Raises if the host tier
        never held it or byte-LRU evicted it (the selector must not name it)."""
        if self.cold is None:
            raise RuntimeError("promote_page: no host page tier attached")
        blob = self.cold.take(old_block)
        if blob is None:
            raise RuntimeError(f"promote_page: block {old_block} is not held on the host")
        new = self.alloc_block()
        nb = blob["k"].is_pinned()
        self.k_pool[:, new].copy_(blob["k"], non_blocking=nb)
        self.v_pool[:, new].copy_(blob["v"], non_blocking=nb)
        if self.k_scale is not None:
            self.k_scale[:, new].copy_(blob["ks"], non_blocking=nb)
            self.v_scale[:, new].copy_(blob["vs"], non_blocking=nb)
        # The pinned blob is released when this call returns; a non_blocking H2D
        # still in flight would then read a buffer the host allocator may reuse.
        # Synchronous contract: a batched/prefetched promote is the later perf job.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return new

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


class HostKvPages:
    """Pinned-host tier for DEMOTED KV pages (the sparse-KV cold set), the block
    counterpart of :class:`DramSnapshots` for state. Holds one blob per demoted
    block id: every plane's K/V and the fp8 ``k_scale``/``v_scale`` planes.

    Lifecycle mirrors the state tier but the pool frees the device frame on
    :meth:`PagedKvPool.demote_page` and promotion ALLOCATES A NEW block from the
    same pool — a demoted id is a tier key, not a live device block. Byte-LRU
    evicts when the pinned budget binds. Prefix-shared pages are never demoted
    (they stay read-only wherever they live); :meth:`demote_page` refuses them.
    """

    def __init__(self, budget_bytes: int = 4 << 30) -> None:
        self.budget_bytes = budget_bytes
        self._held: OrderedDict[int, int] = OrderedDict()
        self._blobs: dict[int, dict] = {}
        self._used = 0
        self.demotions = 0
        self.promotions = 0
        self.drops = 0

    def __contains__(self, block_id: int) -> bool:
        return block_id in self._held

    def hold(self, block_id: int, blob: dict, nbytes: int) -> bool:
        """Pin one page blob; True when held. A page larger than the whole budget is
        dropped, and LRU pages are evicted to fit (a demotion that evicts itself is a
        drop, not a hold)."""
        if not nbytes or nbytes > self.budget_bytes or block_id in self._held:
            self.drops += 1
            return False
        self._blobs[block_id] = blob
        self._held[block_id] = nbytes
        self._used += nbytes
        self.demotions += 1
        while self._used > self.budget_bytes:
            victim, dropped = self._held.popitem(last=False)
            self._blobs.pop(victim, None)
            self._used -= dropped
            self.drops += 1
            if victim == block_id:
                self._blobs.pop(block_id, None)
                return False
        return True

    def take(self, block_id: int) -> dict | None:
        """Remove and return the page blob, or None if it was never held or evicted."""
        n = self._held.pop(block_id, None)
        blob = self._blobs.pop(block_id, None)
        if n is None or blob is None:
            return None
        self._used -= n
        self.promotions += 1
        return blob

    def forget(self, block_id: int) -> None:
        n = self._held.pop(block_id, None)
        self._blobs.pop(block_id, None)
        if n is not None:
            self._used -= n

    @property
    def bytes_held(self) -> int:
        return self._used

    def stats(self) -> dict[str, int]:
        return {
            "kv_cold_pages": len(self._held),
            "kv_cold_bytes": self._used,
            "kv_cold_demotions": self.demotions,
            "kv_cold_promotions": self.promotions,
            "kv_cold_drops": self.drops,
        }


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


class DramSnapshots:
    """Pinned-host tier for GDN snapshots, byte-LRU. Holds snapshots, never KV.

    The snapshot is what binds. At 27B it is 144 MiB against 2.125 MiB per KV block, and
    `build_engine` sets `state_bytes` to a quarter of free memory, so a V100 holds 9
    entries. Measured on the live server: 54 published, **43 evicted with 64% of the
    block pool still free** — every eviction was state bytes, not pool pressure.

    So this tier does not receive evicted entries; it lets them stop being evicted.
    Under `state_bytes` pressure `PrefixStore` DEMOTES the least-recently-used entry's
    snapshot to the host and leaves the entry otherwise intact — same tokens, same
    blocks, still in the index. A later hit promotes it back. Entries only leave the
    store when the block pool needs their blocks, which is the one case a snapshot tier
    cannot help with.

    Pinned because unpinned H2D measured 6.12 GiB/s against 11.52 pinned on this link
    (1.88x). Default budget 4 GiB, not the 25 GiB free: pinned pages cannot be swapped,
    and this pod has 31 GiB of RAM against a 32 GiB card, so pinning most of it would
    destabilise the host rather than the process. 4 GiB is 28 snapshots, 3.1x HBM's 9.
    """

    def __init__(self, budget_bytes: int = 4 << 30) -> None:
        self.budget_bytes = budget_bytes
        self._held: OrderedDict[int, tuple[Any, int]] = OrderedDict()
        self._used = 0
        self.demotions = 0
        self.promotions = 0
        self.drops = 0
        #: wall time inside demote/promote. An idle-card probe put a 144 MiB pinned copy
        #: at 11.55 ms, but a demotion happens mid-prefill where it contends for the link
        #: and forces a sync; the measured cost has to come from the live path.
        self.demote_ms = 0.0
        self.promote_ms = 0.0

    def _to_host(self, state: Any) -> Any:
        if isinstance(state, torch.Tensor):
            host = torch.empty_like(state, device="cpu", pin_memory=state.is_cuda)
            host.copy_(state)
            return host
        return tuple(None if s is None else self._to_host(s) for s in state)

    def demote(self, eid: int, state: Any) -> bool:
        """Copy a snapshot to the host; True when it is held. Evicts by bytes to fit."""
        n = _nbytes(state)
        if not n or n > self.budget_bytes:
            self.drops += 1
            return False
        t0 = time.perf_counter()
        self._held[eid] = (self._to_host(state), n)
        self.demote_ms += (time.perf_counter() - t0) * 1000
        self._used += n
        self.demotions += 1
        while self._used > self.budget_bytes:
            dropped_eid, (_, dropped) = self._held.popitem(last=False)
            self._used -= dropped
            self.drops += 1
            if dropped_eid == eid:
                return False
        return True

    def promote(self, eid: int, device: torch.device) -> Any | None:
        """Move a snapshot back to ``device``, or None if it is gone. Releases the host
        copy: the store owns the snapshot again and may demote it a second time."""
        got = self._held.pop(eid, None)
        if got is None:
            return None
        state, n = got
        self._used -= n
        self.promotions += 1
        t0 = time.perf_counter()
        out = _to_device(state, device)
        self.promote_ms += (time.perf_counter() - t0) * 1000
        return out

    def forget(self, eid: int) -> None:
        got = self._held.pop(eid, None)
        if got is not None:
            self._used -= got[1]

    def stats(self) -> dict[str, int]:
        return {
            # The budget, not just the fill: `dram_bytes` is what is held, so it is 0 both
            # when the tier is off and when it is on and nothing has been demoted yet --
            # an operator who set --dram-bytes cannot tell the two apart from it.
            "dram_budget": self.budget_bytes,
            "dram_entries": len(self._held),
            "dram_bytes": self._used,
            "dram_demotions": self.demotions,
            "dram_promotions": self.promotions,
            "dram_drops": self.drops,
            "dram_demote_ms": int(self.demote_ms),
            "dram_promote_ms": int(self.promote_ms),
        }


def _to_device(state: Any, device: torch.device) -> Any:
    if isinstance(state, torch.Tensor):
        return state.to(device, non_blocking=True)
    return tuple(None if s is None else _to_device(s, device) for s in state)


class KvTier:
    """SSD byte-store below the HBM pool: spilled prefix KV + GDN snapshots.

    A prefix evicted from the pool spills here instead of being dropped; a later
    lookup reloads it into fresh blocks, skipping the prefill recompute. On a
    32 GB V100 with a full host there is no DRAM residency tier, so it is
    HBM→SSD.

    # ponytail: raw bf16 spill, fp8 tier-quant is 2x capacity if SSD fills
    """

    def __init__(self, path: str, fingerprint: str, min_tokens: int = 4 * BLOCK_TOKENS,
                 max_pending: int = 32, max_bytes: int = 20 * 2**30) -> None:
        import queue

        # One chunk (4 blocks = 64 tokens), not the 2048 the eviction-driven version used:
        # write-through spills at chunk boundaries, so a 2048 floor refuses every publish.
        self.min_tokens = min_tokens
        # Bounds in-flight writes; measured not to bind (peak `_pending` 4 against 32, 0
        # refusals) and never set to a non-default anywhere in the tree, so `max_bytes` is
        # what protects the host: errors/2026-09-06-the-max-pending-cap-is-not-the-queue-that-binds.md.
        self._max_pending = max_pending
        self.offered = 0
        self.refusals = 0
        # Stage timers: three guesses at the per-publish cost were wrong in a row, so the
        # spill reports where its time goes instead of being guessed at a fourth time.
        self.copy_ms = 0.0
        self.gather_ms = 0.0
        # The save was the one stage the timers above skipped, and it is the one the
        # "~100 ms" in five comments described. This timer wraps `torch.save` with no
        # fsync, so it is PAGE-CACHE time: 273 ms for a 320.6 MiB entry on the pod's
        # /work, where the durable cost of the same entry is 1337 (5.75x). Do not size a
        # cap on it (errors/2026-09-06-ssd-save-ms-is-page-cache-time.md). Under a real
        # 12-session workload it reads 164 ms per save on a 292.5 MiB half-entry, and
        # summed it is 30% of wall clock -- on the writer thread, so not 30% of any tick.
        self.save_ms = 0.0
        self.saves = 0
        self.over_budget = 0  # byte-budget evictions
        self._healthy = True  # daemon failure (disk full/perm) flips this to refuse
        # Size-based LRU: total on-disk bytes capped at max_bytes; the daemon
        # evicts the least-recently-accessed entry's files after each write.
        # resident()/load_kv()/load_state() touch an entry to MRU.
        self._max_bytes = max_bytes
        self._lru: OrderedDict[int, int] = OrderedDict()
        self._total = 0
        # Never rmtree the caller's path -- it may be a shared dir. Own a fixed subdir
        # marked by a sentinel that carries the fingerprint.
        self._dir = os.path.join(os.fspath(path), "tilerl_kvtier")
        marker = os.path.join(self._dir, ".kvtier")
        self._marker, self._fingerprint, self._generation = marker, fingerprint, 0
        if os.path.exists(self._dir) and not os.path.exists(marker):
            raise RuntimeError(f"{self._dir} exists but is not a KvTier dir (no .kvtier marker)")
        os.makedirs(self._dir, exist_ok=True)
        # Cold start KEEPS what is on disk when the fingerprint matches: a runtime tier
        # needs lookups to reach past the newest entry, but after a restart HBM is empty so
        # EVERY lookup reaches back. Wiping here made a cold hit impossible by construction.
        self.recovered = self._recover(marker, fingerprint)
        # Deferred write: spill_kv runs inside a decode tick, so it does only the
        # GPU->CPU copy + enqueue; a daemon flushes the save off-tick. On the pod's /work
        # one 320.6 MiB entry is 273 ms as `ssd_save_ms` counts it and **1337 ms durable**
        # (240 MiB/s) -- the timer has no fsync, so it stops before the device has the
        # bytes. The 641.8 this comment used to quote was measured on a Mac whose volume
        # writes 26x faster (errors/2026-09-06-ssd-save-ms-is-page-cache-time.md).
        # The durable figure is the DEVICE's rate, not this queue's: the daemon pops an
        # entry once `torch.save` returns, which is a page-cache accept, so `_pending`
        # empties at ~1784 MiB/s and a 585.0 MiB-per-offer workload never fills it
        # (errors/2026-09-06-the-max-pending-cap-is-not-the-queue-that-binds.md).
        # _pending/_pending_st serve blobs not yet on disk, so resident()/load see them.
        self._pending: dict[int, dict] = {}
        self._pending_st: dict[int, dict] = {}
        self._lock = threading.Lock()
        self._q: queue.Queue = queue.Queue()
        # A daemon thread here is starved by Engine.step()'s GIL hold unless step()
        # yields; see errors/2026-09-10-prefetch-deadline-gil-contention.md
        self._writer = threading.Thread(target=self._flush_loop, daemon=True)
        # read side: the torch.load runs off-tick, not inside step() under the lock
        self._fetches: dict[int, dict] = {}   # key -> {"blob", "st", "tokens"}, collected by take()
        self._fetching: set[int] = set()      # queued or mid-read
        # A fetched read ALWAYS parks: a fetch that lost the deadline race still serves
        # the next same-prefix request, and the read it did is already paid for. The cap
        # bounds parked buffers that nothing follows (27B: ~157 MiB each): without it a
        # flood of prefetch-and-recompute rows pins host RAM forever.
        self._park_keys: deque[int] = deque()
        self._max_parked = 2
        self.prefetches = 0
        self.fetches_ready = 0
        # Loads that failed (unreadable / truncated / raced-eviction spill). A fetch that
        # finishes is never dropped: it parks even if its deadline expired, and eviction
        # from the parked-fetch FIFO below only sheds the warm memory copy -- the entry is
        # still on disk, so a later lookup faults it in like any resident entry.
        self.fetch_drops = 0
        self.fetch_ms = 0.0
        self.fetch_bytes = 0
        self.snapshot_bytes = 0
        self.tick_loads = 0
        self._rq: queue.Queue = queue.Queue()
        # Test seam for the fetch-hold gate: ``_fetch_started`` fires once a dequeued read
        # has registered its key in ``_fetching`` (fetch_in_flight is provably true), and
        # the reader blocks on ``_fetch_gate`` until the test clears it. Both are
        # always-set/no-op in production -- a sleep cannot make the in-flight precondition
        # deterministic.
        self._fetch_started = threading.Event()
        self._fetch_gate = threading.Event()
        self._fetch_gate.set()
        # Same starvation as _writer above; the GIL yield in step() is what lets
        # torch.load finish inside the prefetch deadline
        self._reader = threading.Thread(target=self._fetch_loop, daemon=True)
        self._reader.start()
        self._writer.start()

    def hold_fetches_for_test(self) -> tuple[threading.Event, threading.Event]:
        """Park the reader mid-fetch and return (started, gate). A dequeued read signals
        ``started`` with the key already in ``_fetching`` and waits on ``gate``; ``set()``
        releases it. Test-only: makes the hold's in-flight precondition a controlled state
        instead of a race the runner speed decides."""
        self._fetch_started.clear()
        self._fetch_gate.clear()
        return self._fetch_started, self._fetch_gate

    def _recover(self, marker: str, fingerprint: str) -> int:
        """Adopt the spill files already on disk, or wipe them. Returns entries adopted.

        The marker holds the fingerprint the files were written under. On a match the
        index is rebuilt from the directory listing and every entry is servable; on a
        mismatch -- new weights, a different tokenizer, a changed BLOCK_TOKENS -- the
        files describe a model that no longer exists and are removed. A mismatch is the
        normal case after training, so `clear()` invalidating the tier is a fingerprint
        bump and not a directory walk.

        Only sizes are read here, not tensors: a 20 GiB directory would otherwise be
        loaded to answer a question the filename already answers, and every load path
        re-verifies the stored tokens anyway.
        """
        prev = None
        with contextlib.suppress(OSError), open(marker) as f:
            prev = f.read().strip()
        if prev is not None and prev != fingerprint:
            for name in os.listdir(self._dir):
                if name.endswith((".kv", ".st")):
                    with contextlib.suppress(OSError):
                        os.remove(os.path.join(self._dir, name))
            prev = None
        with open(marker, "w") as f:
            f.write(fingerprint)
        if prev is None:
            return 0
        # A key is servable only with BOTH halves present -- a fault-in loads the state
        # first and drops the key when it is missing -- and a half-written pair should not
        # occupy the byte budget meanwhile.
        sizes: dict[int, list[int]] = {}
        for name in os.listdir(self._dir):
            stem, _, ext = name.rpartition(".")
            if ext not in ("kv", "st"):
                continue
            try:
                key = int(stem, 16)
                sz = os.path.getsize(os.path.join(self._dir, name))
            except (ValueError, OSError):
                continue
            sizes.setdefault(key, [0, 0])[0 if ext == "kv" else 1] = sz
        for key, (kv, st) in sizes.items():
            if kv and st:
                self._lru[key] = kv + st
                self._total += kv + st
                # constant at every length, so any recovered entry's .st size is S
                self.snapshot_bytes = st
            else:
                for ext in (".kv", ".st"):
                    with contextlib.suppress(OSError):
                        os.remove(os.path.join(self._dir, f"{key & _MASK64:016x}{ext}"))
        return len(self._lru)

    def _flush_loop(self) -> None:
        while True:
            tag, blob, dst, events = self._q.get()
            # The copies were launched non_blocking, so the host buffers are not valid
            # until their events fire. Waiting HERE is the point: the prefill path pays a
            # launch and this thread pays the transfer.
            for ev in events:
                if ev is not None:
                    ev.synchronize()
            # Write only while the entry is still pending: a drop() that raced us
            # already removed it, and writing now would resurrect an evicted
            # prefix on disk (write-back invalidation → wrong tokens).
            k = tag[1] if isinstance(tag, tuple) else tag
            table = self._pending_st if isinstance(tag, tuple) else self._pending
            with self._lock:
                if table.get(k) is not blob:
                    continue
            try:
                ts = time.perf_counter()
                torch.save(blob, dst)
                self.save_ms += (time.perf_counter() - ts) * 1000
                self.saves += 1
            except Exception:  # noqa: BLE001 - disk full / perm: stop trusting the tier
                self._healthy = False
                continue
            with self._lock:
                still_pending = table.get(k) is blob
                if still_pending:
                    table.pop(k, None)
            if still_pending:
                self._track_written(k, dst)
                continue
            # a drop() landed mid-save: undo the write so the evicted prefix stays gone
            with contextlib.suppress(FileNotFoundError):
                os.remove(dst)

    def _track_written(self, key: int, path: str) -> None:
        """Register a spilled file's size and evict LRU entries while over
        ``max_bytes``. Called by the flush daemon after a successful save."""
        try:
            sz = os.path.getsize(path)
        except OSError:
            return
        with self._lock:
            self._lru[key] = self._lru.get(key, 0) + sz
            self._lru.move_to_end(key)
            self._total += sz
            # Enforced, not just counted: a budget that only reports is unbounded disk
            # growth, and this repo has filled a disk once. An entry still being written is
            # never the victim -- dropping it mid-save would resurrect it when the daemon
            # finishes, and `_flush_loop` reads the pending table to decide that.
            while self._total > self._max_bytes and len(self._lru) > 1:
                victim = next(
                    (k for k in self._lru
                     if k not in self._pending and k not in self._pending_st),
                    None,
                )
                if victim is None:
                    break  # every entry is still in flight
                self._total -= self._lru.pop(victim)
                self.over_budget += 1
                for path in (self._kv(victim), self._st(victim)):
                    with contextlib.suppress(FileNotFoundError):
                        os.remove(path)

    def _touch_lru(self, key: int) -> None:
        with self._lock:
            if key in self._lru:
                self._lru.move_to_end(key)

    def _kv(self, key: int) -> str:
        return os.path.join(self._dir, f"{key & _MASK64:016x}.kv")

    def _st(self, key: int) -> str:
        return os.path.join(self._dir, f"{key & _MASK64:016x}.st")

    def spill_kv(self, key: int, tokens: tuple[int, ...], blocks: Sequence[int],
                 pool: PagedKvPool) -> bool:
        # True = accepted. The tier owns the length floor and the capacity refusal, so a
        # composite tier can vary them per level. A refusal is counted, not just returned:
        # refusals/offered is what says whether the device keeps up with write-through.
        if len(blocks) * BLOCK_TOKENS < self.min_tokens:
            return False
        self.offered += 1
        with self._lock:
            if not self._healthy or len(self._pending) >= self._max_pending:
                self.refusals += 1
                return False
        # The gather is timed separately: `torch.stack` over N block slices is N slice
        # kernels plus a device-to-device copy of the whole entry, all on the prefill
        # stream, and it is a candidate for the residual cost.
        tg = time.perf_counter()
        kk = torch.stack([pool.k_pool[:, b] for b in blocks])
        vv = torch.stack([pool.v_pool[:, b] for b in blocks])
        self.gather_ms += (time.perf_counter() - tg) * 1000
        k, ev_k = self._to_host(kk)
        v, ev_v = self._to_host(vv)
        # Store tokens too: files are keyed by a 64-bit hash, so a collision would
        # otherwise load a different prefix's KV. load_kv verifies before copying.
        blob = {"k": k, "v": v, "tokens": tuple(tokens)}
        if pool.k_scale is not None:
            # fp8 bytes without their scale reload as a different tensor, silently
            blob["ks"] = pool.k_scale[:, blocks].cpu()
            blob["vs"] = pool.v_scale[:, blocks].cpu()
        with self._lock:
            self._pending[key] = blob
        self._q.put((key, blob, self._kv(key), [ev_k, ev_v]))
        return True

    def _to_host(self, t: torch.Tensor):
        """``(host tensor, event)`` -- a pinned async D2H, or a plain copy off CUDA.

        `.cpu()` on a pageable destination is SYNCHRONOUS and lands mid-prefill: measured
        on H20 card 6, write-through cost 0.925 s of a 2.041 s request, and stage timers put
        660 ms of it in this copy against 4 ms in the block gather. So the destination is
        pinned and the copy is `non_blocking`; the event is what the daemon waits on.

        `torch.empty(pin_memory=True)` per spill, deliberately -- torch's
        CachingHostAllocator already reuses pinned blocks, and two hand-written pools on top
        of it both measured WORSE: keyed per (numel, dtype), 0.421 s against 0.383 s; as a
        two-slot arena, 1.640 s, because a request's 6 publishes outrun a depth the flush
        daemon only frees after `torch.save` and 16 of 18 spills fell back to pageable.
        """
        if t.device.type != "cuda":
            return t.contiguous().cpu(), None
        tg = time.perf_counter()
        t = t.contiguous()
        host = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
        host.copy_(t, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self.copy_ms += (time.perf_counter() - tg) * 1000
        return host, ev

    def prefetch(self, key: int, tokens: tuple[int, ...]) -> bool:
        """True = queued or already in flight. Holds no blocks, so a dropped one costs
        a host buffer and unwinds nothing."""
        with self._lock:
            if key in self._fetches or key in self._fetching:
                return True
            if not (key in self._lru or (key in self._pending and key in self._pending_st)):
                return False
            self._fetching.add(key)
        self.prefetches += 1
        self._rq.put((key, tokens))
        return True

    def fetch_pending(self, key: int) -> bool:
        with self._lock:
            return key in self._fetching

    def fetching_keys(self) -> frozenset[int]:
        """Snapshot of every fetch mid-read; cheap to intersect inside a spin."""
        with self._lock:
            return frozenset(self._fetching)

    def any_fetching(self) -> bool:
        with self._lock:
            return bool(self._fetching)

    def take(self, key: int):
        """The prefetched pair, or None. `st` may be absent on an older parked entry."""
        with self._lock:
            done = self._fetches.pop(key, None)
            self._park_keys = deque(k for k in self._park_keys if k != key)
        if done is None or "blob" not in done:
            return None
        return done

    def _fetch_loop(self) -> None:
        while True:
            key, tokens = self._rq.get()
            # The key is already in _fetching (prefetch registered it before enqueue); the
            # gate lets a test hold the read here so the in-flight window is deterministic.
            self._fetch_started.set()
            self._fetch_gate.wait()
            with self._lock:
                blob = self._pending.get(key)
                st = self._pending_st.get(key)
            if blob is None or st is None:
                try:
                    ts = time.perf_counter()
                    if blob is None:
                        blob = torch.load(self._kv(key), map_location="cpu")
                        self.fetch_bytes += _blob_bytes(blob)
                    # the .st too: reading it in `load_state` put 157 MiB on the tick
                    if st is None:
                        st = torch.load(self._st(key), map_location="cpu")
                        self.fetch_bytes += _blob_bytes(st)
                    self.fetch_ms += (time.perf_counter() - ts) * 1000
                except Exception:  # noqa: BLE001 - truncated / corrupt / raced eviction
                    self.drop(key)
                    self.fetch_drops += 1
                    with self._lock:
                        self._fetching.discard(key)
                    continue
            with self._lock:
                self._fetching.discard(key)
                # Always park: the deadline gates whether a row WAITS, not whether a
                # finished read is kept -- the load is paid for either way, and the next
                # request with this prefix faults it in from memory instead of disk.
                self._fetches[key] = {"blob": blob, "st": st, "tokens": tokens}
                if key not in self._park_keys:
                    self._park_keys.append(key)
                while len(self._park_keys) > self._max_parked:
                    victim = self._park_keys.popleft()
                    self._fetches.pop(victim, None)
            self.fetches_ready += 1

    def load_kv(self, key: int, tokens: tuple[int, ...], blocks: Sequence[int],
                pool: PagedKvPool, blob: dict | None = None) -> bool:
        # False = data gone (a raced eviction dropped it) OR a hash collision
        # stored a different prefix — caller treats either as a miss. Serves a
        # still-pending blob from memory, closing the resident()/load TOCTOU.
        if blob is None:
            with self._lock:
                blob = self._pending.get(key)
        if blob is None:
            if not os.path.exists(self._kv(key)):
                return False
            # A truncated file is the crash case, not a theoretical one: the daemon writes
            # off-tick, so a kill between `torch.save` starting and finishing leaves a
            # partial blob that `_recover` then adopts by size. Treat an unreadable file as
            # a miss and drop it, rather than raising inside a lookup.
            try:
                # counted: this is the number that can refute "the fetch became async"
                self.tick_loads += 1
                blob = torch.load(self._kv(key), map_location="cpu")
            except Exception:  # noqa: BLE001 - truncated / corrupt spill
                self.drop(key)
                return False
        if blob.get("tokens") != tuple(tokens):
            return False  # hash collision: these bytes belong to a different prefix
        self._touch_lru(key)
        # one index_copy_ per plane: the per-block loop was 3,750 launches at 30k tokens
        idx = torch.as_tensor(list(blocks), device=pool.device)
        k = self._planes_on_device(blob["k"], pool)
        v = self._planes_on_device(blob["v"], pool)
        if (pool.k_scale is None) != ("ks" not in blob):
            # Belt to _weight_fingerprint's brace: it now carries kv_fp8, so a flag flip
            # against the same --ssd-path yields a different fingerprint -- but an explicit
            # ssd_fingerprint routes around that. Dropping the entry costs a re-prefill; the
            # dtype-mismatched index_copy_ below raises out of _admit, failing every request.
            return False
        if pool.k_scale is None:
            pool.k_pool.index_copy_(1, idx, k)
            pool.v_pool.index_copy_(1, idx, v)
            return True
        # index assignment, not index_copy_: torch has no index_copy_ for fp8 on CPU
        pool.k_pool[:, idx], pool.v_pool[:, idx] = k, v
        pool.k_scale.index_copy_(1, idx, blob["ks"].to(pool.device))
        pool.v_scale.index_copy_(1, idx, blob["vs"].to(pool.device))
        return True

    def _planes_on_device(self, t: torch.Tensor, pool: PagedKvPool) -> torch.Tensor:
        """Permute AFTER the transfer: `.to(cuda)` on a non-contiguous view materialises
        a host temp of the whole blob first. Same-device is a no-op, so `"c"` cannot see
        this."""
        return t.to(pool.device, non_blocking=t.is_pinned()).permute(1, 0, 2, 3, 4)

    def spill_state(self, key: int, tokens: tuple[int, ...], states, windows) -> None:
        st, ev_s = self._to_host(states)
        win, ev_w = (None, None) if windows is None else self._to_host(windows)
        blob = {"states": st, "windows": win, "tokens": tuple(tokens)}
        with self._lock:
            self._pending_st[key] = blob
        self._q.put((("st", key), blob, self._st(key), [ev_s, ev_w]))

    def load_state(self, key: int, tokens: tuple[int, ...], blob: dict | None = None):
        # None = gone or a hash-collision mismatch — caller degrades to a miss.
        if blob is None:
            with self._lock:
                blob = self._pending_st.get(key)
        if blob is None:
            if not os.path.exists(self._st(key)):
                return None
            try:
                # reached only with no prefetch: below the break-even, or no reader
                blob = torch.load(self._st(key), map_location="cpu")
            except Exception:  # noqa: BLE001 - truncated / corrupt spill, same as load_kv
                self.drop(key)
                return None
        if blob.get("tokens") != tuple(tokens):
            return None
        self._touch_lru(key)
        return blob["states"], blob["windows"]

    def read_bytes_per_s(self) -> float:
        """B in the break-even, over both planes a hit reads. 0 before the first read."""
        return 0.0 if self.fetch_ms <= 0 else self.fetch_bytes / (self.fetch_ms / 1000.0)

    def resident(self, key: int) -> bool:
        """Whether this key is in the in-memory index, without touching the disk.

        The candidate filter for a lookup: a query walks every prefix length, and
        a `torch.load` on each would be one or two file reads per length. This is a dict
        probe, so the disk is read only for the one candidate that survives.
        """
        with self._lock:
            return key in self._lru or (key in self._pending and key in self._pending_st)

    def drop(self, key: int) -> None:
        with self._lock:
            self._pending.pop(key, None)
            self._pending_st.pop(key, None)
            self._total -= self._lru.pop(key, 0)
        for p in (self._kv(key), self._st(key)):
            with contextlib.suppress(FileNotFoundError):
                os.remove(p)

    def invalidate(self) -> None:
        """Make every file on disk unreadable without walking the directory.

        An optimizer step calls this. Rewriting the marker means the next `_recover`
        fingerprint-mismatches and removes the files then; until that restart the
        in-memory index is what gates reads, and it is cleared here. One write instead of
        a 20 GiB unlink walk inside a training step on a 229 MB/s device.
        """
        self._generation += 1
        with contextlib.suppress(OSError), open(self._marker, "w") as f:
            f.write(f"{self._fingerprint}#{self._generation}")
        with self._lock:
            self._pending.clear()
            self._pending_st.clear()
            self._lru.clear()
            self._total = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            pending = len(self._pending) + len(self._pending_st)
            entries, total = len(self._lru), self._total
        return {
            "ssd_entries": entries,
            "ssd_bytes": total,
            "ssd_recovered": self.recovered,
            "ssd_offered": self.offered,
            "ssd_refusals": self.refusals,
            "ssd_gather_ms": int(self.gather_ms),
            "ssd_copy_ms": int(self.copy_ms),
            "ssd_save_ms": int(self.save_ms),
            "ssd_saves": self.saves,
            "ssd_evictions": self.over_budget,
            "ssd_pending": pending,
            "ssd_healthy": int(self._healthy),
            "ssd_prefetches": self.prefetches,
            "ssd_fetches_ready": self.fetches_ready,
            "ssd_fetch_drops": self.fetch_drops,
            "ssd_tick_loads": self.tick_loads,
            "ssd_fetch_ms": int(self.fetch_ms),
            # with fetch_ms this gives B for the run
            "ssd_fetch_bytes": self.fetch_bytes,
            # retained at 0 for readers: the lookup path no longer reads the .st
            "ssd_state_load_ms": 0,
            "ssd_state_loads": 0,
        }


@dataclass(frozen=True)
class PrefixHit:
    """Matched token count, the store-retained blocks covering ``[0, length)``
    and the recurrent-state snapshot taken at that boundary."""

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

    def insert(self, tokens: Sequence[int], blocks: Sequence[int], state: Any = None,
               spill: bool = True) -> bool:
        return False

    def retire(self, tokens: Sequence[int]) -> bool:
        return False

    def evict_until_free(self, blocks: int) -> None:
        return None

    def reclaimable_blocks(self) -> int:
        return 0

    def prefetch_if_worth_it(self, tokens: Sequence[int], prefill_rate: float) -> bool:
        return False

    def break_even_tokens(self, prefill_rate: float) -> int:
        return 1 << 31

    def fetch_in_flight(self, tokens: Sequence[int]) -> bool:
        return False

    def fetching_keys(self) -> frozenset[int]:
        return frozenset()

    def boundary_keys(self, tokens: Sequence[int]) -> frozenset[int]:
        return frozenset()

    @property
    def has_ssd(self) -> bool:
        return False

    def any_fetching(self) -> bool:
        return False

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
        ssd: KvTier | None = None,
    ) -> None:
        self._pool = pool
        self.capacity = capacity
        self.state_bytes = state_bytes
        self._dram = dram
        self._ssd = ssd
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
        self.ssd_hits = 0
        self.ssd_faults = 0
        # S: constant at every prefix length, and what puts the break-even above zero
        self._snapshot_bytes = 0
        self.fetch_waits = 0

    @property
    def has_ssd(self) -> bool:
        return self._ssd is not None

    def any_fetching(self) -> bool:
        return self._ssd is not None and self._ssd.any_fetching()

    def fetching_keys(self) -> frozenset[int]:
        """Snapshot of tier keys mid-read, for a spin loop's cheap membership test."""
        return frozenset() if self._ssd is None else self._ssd.fetching_keys()

    def boundary_keys(self, tokens: Sequence[int]) -> frozenset[int]:
        """The hashes at every whole-block boundary of ``tokens`` -- the keys the
        prefetch ladder can have queued for this prompt."""
        if self._ssd is None:
            return frozenset()
        h, keys = 0, set()
        for i, t in enumerate(tokens, 1):
            h = self._roll(h, int(t))
            if i % BLOCK_TOKENS == 0:
                keys.add(h)
        return frozenset(keys)

    def _hash_all(self, tokens: Sequence[int]) -> int:
        h = 0
        for t in tokens:
            h = self._roll(h, int(t))
        return h

    def fetch_in_flight(self, tokens: Sequence[int]) -> bool:
        """True while a prefetch for some prefix of ``tokens`` is still reading.

        Same length ladder `prefetch_if_worth_it` queues on, so the engine asks about
        exactly the fetches it started.
        """
        if self._ssd is None:
            return False
        h = 0
        for i, t in enumerate(tokens, 1):
            h = self._roll(h, int(t))
            if i % BLOCK_TOKENS == 0 and self._ssd.fetch_pending(h):
                return True
        return False

    def break_even_tokens(self, prefill_rate: float) -> int:
        """Prefix length above which fetching beats recomputing, at this prefill rate.

        `(S + n*k)/B < n/R`, so `n* = (S/B) / (1/R - k/B)`: S the recurrent snapshot
        (constant at any length), k the KV bytes per token, B the tier's read rate, R
        tokens/s of prefill. Every operand is read here rather than fixed --
        `k` is 64 KiB on a bf16 pool, 128 on sm70's f32 one and 32 KiB + scale on fp8,
        and the two archs' `R` differ by more than an order of magnitude, so a constant
        would be wrong on one of them (docs/design-ssd-read-path.md).

        Returns 2**31 when `k/B >= 1/R`: the device cannot stream KV as fast as the card
        recomputes it, and no length pays.
        """
        if self._ssd is None or prefill_rate <= 0:
            return 1 << 31
        k = self._pool.bytes_per_token
        b = self._ssd.read_bytes_per_s()
        s = self._snapshot_bytes or self._ssd.snapshot_bytes
        if b <= 0:
            # unmeasured tier fetches once and calibrates; a restart is the case it exists for
            return 0
        if s <= 0:
            return 1 << 31
        denom = 1.0 / prefill_rate - k / b
        return (1 << 31) if denom <= 0 else int(s / b / denom)

    def prefetch_if_worth_it(self, tokens: Sequence[int], prefill_rate: float) -> bool:
        """Start reading the longest resident prefix, if fetching beats recomputing.
        Allocation-free, which is why it can live where the prefix MATCH cannot."""
        if self._ssd is None:
            return False
        if len(tokens) < self.break_even_tokens(prefill_rate):
            return False
        h, hashes = 0, []
        for t in tokens:
            h = self._roll(h, int(t))
            hashes.append(h)
        for i in range(len(tokens) - len(tokens) % BLOCK_TOKENS, 0, -BLOCK_TOKENS):
            if self._ssd.prefetch(hashes[i - 1], tuple(tokens[:i])):
                return True
        return False

    def insert(self, tokens: Sequence[int], blocks: Sequence[int], state: Any = None,
               spill: bool = True) -> bool:
        """Cache ``tokens`` (covered by ``blocks``) with its ``state`` snapshot and retain
        the blocks; True when a new entry was retained, False for a duplicate.

        ``spill=False`` keeps the entry in HBM but does not offer it to the disk tier. The
        caller uses it for a publish a later one supersedes: measured on H20 card 6, one
        2729-token prompt publishes 6 times and spills 1624 MB, of which only the longest
        entry (325 MB) is ever read -- 5.0x the bytes for nothing, because a GDN snapshot
        is a CONSTANT ~157 MB at every prefix length.
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
        # Write-through: a GPU->CPU copy plus an enqueue here, with the save off-tick on
        # a daemon (1337 ms durable for a 320.6 MiB entry on the pod's /work; `ssd_save_ms`
        # reports 273 because it has no fsync), so a full queue refuses rather than
        # blocking prefill. Both
        # halves go or neither -- a fault-in needs the pair. `resident` skips what is already
        # on disk, without which every fault-in writes back the bytes it just read.
        if state is not None and self._snapshot_bytes == 0:
            self._snapshot_bytes = sum(t.numel() * t.element_size()
                                       for t in state if t is not None)
        if (spill and self._ssd is not None and state is not None and not self._ssd.resident(h)
                and self._ssd.spill_kv(h, tokens, blocks, self._pool)):
            self._ssd.spill_state(h, tokens, state[0], state[1])
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
            # Nothing resident at this length. Before trying a shorter prefix, ask the disk:
            # after a restart HBM is empty, so the LONGEST prefix on disk is what this loop
            # would otherwise walk straight past on its way to a miss.
            if self._ssd is not None and self._ssd.resident(prefix_hashes[i - 1]):
                key = prefix_hashes[i - 1]
                if self._ssd.fetch_pending(key):
                    # a miss for now: reading it here too puts the 1.7 s back on the tick
                    self.fetch_waits += 1
                    break
                hit = self._fault_in(key, tokens[:i], fetched=self._ssd.take(key))
                if hit is not None:
                    return hit
        self.lookups_missed += 1
        return None

    def _fault_in(self, h: int, tokens: tuple[int, ...],
                  fetched: dict | None = None) -> PrefixHit | None:
        """Reload one prefix from the SSD tier into fresh blocks, or None.

        The reload allocates from the pool and hands the entry to `insert`, so the faulted
        prefix is an ordinary resident entry afterwards -- one code path owns retain,
        eviction and the byte accounting. `resident` gated the call, so at most one
        candidate length pays a `torch.load`.

        `fetched` is the reader thread's `{"blob", "st"}`; without it both `torch.load`s
        happen here, on the tick.
        """
        blob = None if fetched is None else fetched.get("blob")
        st = None if fetched is None else fetched.get("st")
        need = PagedKvPool.blocks_for_tokens(len(tokens))
        # Only a whole-block prefix can be adopted: `insert` refuses a partial block,
        # because publishing one shares a page a slot is still appending to. Every publish
        # point is block-aligned, so this is a guard, not a path.
        if len(tokens) % BLOCK_TOKENS:
            return None
        # Both halves or neither: adopting KV without the snapshot would run the GDN
        # layers from a zero state over KV that is not zero -- wrong, and silent.
        loaded = self._ssd.load_state(h, tokens, blob=st)
        if loaded is None:
            self._ssd.drop(h)
            return None
        self.evict_until_free(need)
        if self._pool.free_blocks < need:
            return None
        blocks = [self._pool.alloc_block() for _ in range(need)]
        try:
            if not self._ssd.load_kv(h, tokens, blocks, self._pool, blob=blob):
                self.ssd_faults += 1
                self._ssd.drop(h)
                return None
            state = (loaded[0].to(self._pool.device),
                     None if loaded[1] is None else loaded[1].to(self._pool.device))
            # insert() takes its own retain on every block, so the alloc refcount dropped
            # below is not the last one. Freeing before the insert would put a block on the
            # free list while this hit is still handing it out.
            if not self.insert(tokens, blocks, state):
                return None
        finally:
            for b in blocks:
                self._pool.free_block(b)
        self.ssd_hits += 1
        self.lookups_matched += 1
        return PrefixHit(len(tokens), tuple(blocks), state)

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

        The SSD tier is invalidated by bumping its fingerprint, not by deleting files.
        Deleting is a 20 GiB directory walk on a 229 MB/s device inside an optimizer
        step; a fingerprint bump makes every file unreadable at the next `_recover` and
        costs one write. `_drop` deliberately does NOT touch the SSD -- an entry evicted
        from HBM is exactly what a cold hit should still find on disk.
        """
        while self._by_id:
            self._evict_one()
        if self._ssd is not None:
            self._ssd.invalidate()

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
        if self._ssd is not None:
            st.update(self._ssd.stats())
            st["ssd_hits"] = self.ssd_hits
            st["ssd_faults"] = self.ssd_faults
            # 0 on an engine that holds the row; nonzero means a lookup bypassed the hold.
            st["ssd_fetch_waits"] = self.fetch_waits
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


