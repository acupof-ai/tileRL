"""Paged KV cache, linear-attention state pool, and prefix store.

Day-1 flagship: the host-side KV subsystem. Storage is torch tensors on the
backend's target device; all bookkeeping is plain Python ints/lists (the
allocator runs on host, like agent-infer's ``HostPagedKvPool`` — device-side
allocation is a day-2 optimization). Design mirrors the *ideas* of
agent-infer/crates/infer-seam (read-only ref): ``host_paged_kv_pool.rs`` (free-
list page allocator, per-page ownership counts, copy-on-write before writing a
shared page) and ``prefix_store.rs`` (prefix retention over the same pool).

Day-1 simplifications (ponytail): ONE refcount per block counts every owner
(live slots + the prefix store); the reference splits retain vs attach counts
to drive eviction policy, but FIFO eviction here only needs the total. No CP
sequence sharding, no evict-drop/reinstate, no fixed-band pages. Reject-under-
pressure admission: the engine fails loudly on pool exhaustion
(``alloc_block`` raises). ``# ponytail: no preempt/swap, cpu-offload day-2``
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

__all__ = [
    "BLOCK_TOKENS",
    "PagedKvPool",
    "LinearStatePool",
    "PrefixStore",
    "PrefixHit",
]

#: Tokens per physical KV block (paged-attention page size).
BLOCK_TOKENS = 16

_MASK64 = (1 << 64) - 1


def _rolling_hash(prev: int, token: int) -> int:
    """One step of the prefix rolling hash: ``h' = (h * P) xor (token + 1)``.

    ``+ 1`` so token 0 still perturbs the state. 64-bit masked; collisions are
    expected at scale, which is why :class:`PrefixStore` verifies token
    contents on every hit.
    """
    return ((prev * 1000003) ^ (token + 1)) & _MASK64


def _default_device() -> torch.device:
    """Target device from the backend singleton, falling back to CPU.

    Lazy import: this module must not import tilelang, and the backend is the
    only place that resolves ``TILERL_TARGET``. Anything less than a fully
    scaffolded backend degrades to CPU — the portable default and the CI/dev
    target on this Mac.
    """
    try:
        from tilerl_kernels.backend import get_backend

        return get_backend().device
    except Exception:  # noqa: BLE001 - any scaffold gap -> CPU default
        return torch.device("cpu")


class PagedKvPool:
    """Paged K/V storage with a free-list allocator and per-block refcount.

    Storage layout, exposed as-is for ``paged_attention`` to gather by block
    id and for the model to index per layer: ``k_pool``/``v_pool`` are
    ``[num_planes, num_blocks, num_kv_heads, BLOCK_TOKENS, head_dim]`` bf16.
    One allocator + refcount set is shared across layers: block id N names the
    same page in every layer, which is exactly what one block table indexes.

    Only full-attn layers own a plane: ``layer_map`` gives the GLOBAL layer
    index of each plane (dense storage — the 27B has 16 full-attn layers, not
    64). All layer-indexed methods take a GLOBAL index and map it; a
    non-full-attn layer raises KeyError — GDN layers must never touch the pool.

    Ownership: every block held by anyone (a live slot's block table, or the
    prefix store) carries a refcount. :meth:`alloc_block` hands out a block
    with refcount 1; :meth:`retain` adds an owner; :meth:`free_block` removes
    one and recycles the block only when the last owner is gone. A block with
    refcount > 1 is *shared* — :meth:`cow_for_append` duplicates it before a
    slot writes, so cached prefixes stay byte-stable.
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
    ) -> None:
        # plane d -> global layer index; identity when no map is given.
        self._layer_map = tuple(range(num_layers)) if layer_map is None else tuple(layer_map)
        self._plane = {g: d for d, g in enumerate(self._layer_map)}
        self.num_blocks = num_blocks
        self.num_layers = len(self._layer_map)
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = _default_device() if device is None else torch.device(device)
        shape = (self.num_layers, num_blocks, num_kv_heads, BLOCK_TOKENS, head_dim)
        self.k_pool = torch.zeros(shape, dtype=dtype, device=self.device)
        self.v_pool = torch.zeros(shape, dtype=dtype, device=self.device)
        self._free: list[int] = list(range(num_blocks))
        self.refcount: list[int] = [0] * num_blocks

    def kv_layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(k, v)`` plane for a GLOBAL layer index (dense-mapped)."""
        p = self._plane[layer_idx]
        return self.k_pool[p], self.v_pool[p]

    # ------------------------------------------------------------------ alloc

    def alloc_block(self) -> int:
        """Pop a free block and return it with refcount 1."""
        if not self._free:
            raise RuntimeError(f"PagedKvPool exhausted: all {self.num_blocks} blocks in use")
        block = self._free.pop()
        self.refcount[block] = 1
        return block

    def retain(self, block: int) -> None:
        """Add one owner (prefix-store pin or slot adoption)."""
        if self.refcount[block] == 0:
            raise RuntimeError(f"retain: block {block} is free (refcount 0)")
        self.refcount[block] += 1

    def free_block(self, block: int) -> None:
        """Release one ownership of ``block``; recycle when the last owner goes.

        Raises on double free (refcount already 0), which would corrupt the
        free list.
        """
        if self.refcount[block] <= 0:
            raise RuntimeError(f"free_block: block {block} already free (double free)")
        self.refcount[block] -= 1
        if self.refcount[block] == 0:
            self._free.append(block)

    def is_shared(self, block: int) -> bool:
        """True if more than one owner would observe a write to ``block``."""
        return self.refcount[block] > 1

    def cow_for_append(self, block: int) -> int:
        """Return a writable block for a slot that owns one ref of ``block``.

        If the block is shared, allocate a fresh block, copy the full K/V
        contents of every layer, and transfer the slot's ownership to it
        (every other owner keeps the old block). If exclusive, return
        ``block`` unchanged.
        """
        if not self.is_shared(block):
            return block
        new = self.alloc_block()
        self.k_pool[:, new].copy_(self.k_pool[:, block])
        self.v_pool[:, new].copy_(self.v_pool[:, block])
        self.free_block(block)
        return new

    # --------------------------------------------------------------- token IO

    def write_block(
        self,
        block: int,
        offset: int,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: int = 0,
    ) -> None:
        """Write ``k``/``v`` ([num_kv_heads, n, head_dim]) at ``layer``/token
        ``offset``. Writing a shared block is a hard error (would silently
        mutate a cached prefix) — call :meth:`cow_for_append` first.
        """
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
        if self.is_shared(block):
            raise RuntimeError(
                f"write_block: block {block} is shared (refcount {self.refcount[block]}); "
                "call cow_for_append first"
            )
        layer = self._plane[layer]
        self.k_pool[layer, block, :, offset : offset + n].copy_(
            k.to(self.device, self.k_pool.dtype)
        )
        self.v_pool[layer, block, :, offset : offset + n].copy_(
            v.to(self.device, self.v_pool.dtype)
        )

    def write_tokens(self, k: torch.Tensor, v: torch.Tensor, kv, layer_idx: int) -> None:
        """Write k/v [B,T,Hkv,D] at the per-row tail ``[seq_len-seq_q,
        seq_len)`` (``seq_q`` = ``kv.seq_q_lens``, default T), scattered
        through the batch's block table.

        The engine guarantees those positions land on blocks owned exclusively
        by the request (tail blocks on prefill, the fresh append block on
        decode), so shared prefix blocks are never written.
        Torch-loop fallback for arches without the ``write_tokens`` scatter
        kernel (the sm90 backend op replaces it — its per-token ``int()``
        syncs make this loop uncapturable).
        # ponytail: per-token python loop, vectorized scatter day-2
        """
        b, t, _, _ = k.shape
        sql = getattr(kv, "seq_q_lens", None)
        plane = self._plane[layer_idx]
        for bi in range(b):
            sq = t if sql is None else int(sql[bi])
            base = int(kv.seq_len[bi]) - sq
            for ti in range(sq):
                pos = base + ti
                blk = int(kv.block_table[bi, pos // BLOCK_TOKENS])
                self.k_pool[plane, blk, :, pos % BLOCK_TOKENS, :] = k[bi, ti]
                self.v_pool[plane, blk, :, pos % BLOCK_TOKENS, :] = v[bi, ti]

    # ------------------------------------------------------ queries/admission

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - len(self._free)

    @staticmethod
    def blocks_for_tokens(tokens: int) -> int:
        """Physical blocks needed to hold ``tokens`` tokens."""
        return (tokens + BLOCK_TOKENS - 1) // BLOCK_TOKENS


class LinearStatePool:
    """Per-slot state for the gated-delta linear-attention layers.

    ``states``: [num_slots, num_linear_layers, num_heads, head_dim, head_dim]
    bf16 on the target device. ``conv_windows``: [num_slots, num_linear_layers,
    kernel-1, qkv_dim] — the raw-qkv carry that makes segmented decode exact
    (absent when the model has no GDN layers). Both are zeroed on alloc so a
    fresh sequence starts from the zero recurrent state / empty carry.
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
        self.states = torch.zeros(
            num_slots,
            num_linear_layers,
            num_heads,
            head_dim,
            head_dim,
            dtype=dtype,
            device=self.device,
        )
        # Two planes per slot, selected by win_parity[slot]: the sm90 fused
        # decode kernel reads plane p and writes 1-p (its q/k columns are
        # shared across blocks, so an in-place shift would race), then the tick
        # flips the parity. Every other path indexes through the parity too.
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
        # Speculative verify: the chunk op writes the state/window after every
        # chain step here, and the accepted length picks one (select_step).
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
        """Adopt the state the last verify forward left after chain token ``step``."""
        if self.step_states is None:  # no recurrent layers: nothing to rewind
            return
        self.states[slot].copy_(self.step_states[slot, :, step])
        if self.step_windows is not None:
            self.window_restore(slot, self.step_windows[slot, :, step])

    def window_snapshot(self, slot: int) -> torch.Tensor | None:
        """[L, W, D] clone of the slot's live window plane (host sync on parity)."""
        if self.conv_windows is None:
            return None
        return self.conv_windows[slot, :, int(self.win_parity[slot])].clone()

    def window_restore(self, slot: int, snap: torch.Tensor) -> None:
        self.conv_windows[slot, :, 0].copy_(snap)
        self.win_parity[slot] = 0

    def alloc_slot(self) -> int:
        """Pop a free slot, zero its state, and return it."""
        if not self._free:
            raise RuntimeError(
                f"LinearStatePool exhausted: all {self.states.shape[0]} slots in use"
            )
        slot = self._free.pop()
        self.states[slot].zero_()
        if self.conv_windows is not None:
            self.conv_windows[slot].zero_()
        return slot

    def free_slot(self, slot: int) -> None:
        """Return ``slot`` to the free list. Raises on double free."""
        if slot in self._free:
            raise RuntimeError(f"free_slot: slot {slot} already free (double free)")
        self._free.append(slot)


@dataclass(frozen=True)
class PrefixHit:
    """Result of a prefix lookup: matched token count and the blocks covering
    ``[0, length)`` (already retained by the store — the caller adopts them
    with :meth:`PagedKvPool.retain`)."""

    length: int
    blocks: tuple[int, ...]


class _Entry:
    __slots__ = ("eid", "tokens", "blocks", "h")

    def __init__(self, eid: int, tokens: tuple[int, ...], blocks: tuple[int, ...], h: int) -> None:
        self.eid = eid
        self.tokens = tokens
        self.blocks = blocks
        self.h = h


class PrefixStore:
    """Rolling-hash prefix cache over a :class:`PagedKvPool`.

    Keys are per-token rolling hashes ``h_i = roll(h_{i-1}, token_i)``; an
    entry maps a prefix's full hash to (token tuple, physical blocks). The
    engine inserts each completed block span (full blocks plus the current
    partial); lookup returns the LONGEST stored prefix of the query.
    Collision-safe: hash hits are verified against the stored token contents,
    so a colliding entry with different tokens is a miss. Insert retains
    every block in the pool; FIFO eviction at ``capacity`` releases them.
    Blocks a live slot still holds stay allocated. ``on_evict`` (set by the
    engine) is called with an evicted entry's tokens so side tables keyed by
    the same tuple cannot outlive it.
    """

    def __init__(
        self,
        pool: PagedKvPool,
        capacity: int = 4096,
    ) -> None:
        self._pool = pool
        self.capacity = capacity
        self.on_evict: Callable[[tuple[int, ...]], None] | None = None
        self._roll = _rolling_hash
        self._entries: dict[int, list[_Entry]] = {}
        self._by_id: dict[int, _Entry] = {}
        self._fifo: deque[int] = deque()
        self._next_id = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _hash_all(self, tokens: Sequence[int]) -> int:
        h = 0
        for t in tokens:
            h = self._roll(h, int(t))
        return h

    def insert(self, tokens: Sequence[int], blocks: Sequence[int]) -> None:
        """Cache ``tokens`` (covered by ``blocks``) and retain the blocks.

        ``len(blocks)`` must equal ``ceil(len(tokens) / BLOCK_TOKENS)``. A
        duplicate insert (same tokens) is a no-op.
        """
        tokens = tuple(int(t) for t in tokens)
        blocks = tuple(blocks)
        if not tokens:
            return
        expected = PagedKvPool.blocks_for_tokens(len(tokens))
        if len(blocks) != expected:
            raise ValueError(
                f"insert: {len(tokens)} tokens need {expected} blocks, got {len(blocks)}"
            )
        h = self._hash_all(tokens)
        for e in self._entries.get(h, ()):
            if e.tokens == tokens:
                return  # already cached
        entry = _Entry(self._next_id, tokens, blocks, h)
        self._next_id += 1
        self._entries.setdefault(h, []).append(entry)
        self._by_id[entry.eid] = entry
        self._fifo.append(entry.eid)
        for b in blocks:
            self._pool.retain(b)
        self._evict_if_needed()

    def lookup(self, tokens: Sequence[int]) -> PrefixHit | None:
        """Longest stored prefix of ``tokens``, or ``None``.

        Every hash hit is verified against the stored token contents, so a
        collision on different tokens is a miss.
        """
        tokens = tuple(int(t) for t in tokens)
        h = 0
        prefix_hashes: list[int] = []
        for t in tokens:
            h = self._roll(h, t)
            prefix_hashes.append(h)
        for i in range(len(tokens), 0, -1):
            for e in self._entries.get(prefix_hashes[i - 1], ()):
                if e.tokens == tokens[:i]:
                    self.hits += 1
                    return PrefixHit(i, e.blocks)
        self.misses += 1
        return None

    def _evict_if_needed(self) -> None:
        while len(self._by_id) > self.capacity:
            self._evict_one()

    def evict_until_free(self, blocks: int) -> None:
        """Evict oldest entries until the pool can satisfy an allocation."""
        while self._pool.free_blocks < blocks and self._fifo:
            self._evict_one()

    def _evict_one(self) -> None:
        eid = self._fifo.popleft()
        entry = self._by_id.pop(eid)
        chain = self._entries[entry.h]
        chain.remove(entry)
        if not chain:
            del self._entries[entry.h]
        for b in entry.blocks:
            self._pool.free_block(b)
        if self.on_evict is not None:
            self.on_evict(entry.tokens)
        self.evictions += 1

    def stats(self) -> dict[str, int]:
        """Cache counters: entries, capacity, hits, misses, evictions."""
        return {
            "entries": len(self._by_id),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }
