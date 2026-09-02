"""Paged KV cache, linear-attention state pool, and prefix store.

Host-side bookkeeping (plain ints/lists) over torch tensors on the target
device, after agent-infer's ``host_paged_kv_pool.rs`` / ``prefix_store.rs``.
# ponytail: one refcount per block counts every owner (slots + prefix store);
# no preempt/swap, no cpu-offload — ``alloc_block`` raises on exhaustion.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

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


class PagedKvPool:
    """Paged K/V storage with a free-list allocator and per-block refcount.

    ``k_pool``/``v_pool`` are ``[num_planes, num_blocks, num_kv_heads,
    BLOCK_TOKENS, head_dim]``; block id N names the same page in every plane.
    Only full-attn layers own a plane: ``layer_map`` gives each plane's GLOBAL
    layer index, and every layer-indexed method takes a global index.

    A block with refcount > 1 is shared (a live slot plus the prefix store);
    :meth:`cow_for_append` duplicates it before a slot writes.
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
        p = self._plane[layer_idx]
        return self.k_pool[p], self.v_pool[p]

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

    def cow_for_append(self, block: int) -> int:
        """Writable block for a slot owning one ref of ``block``: a copy if shared."""
        if not self.is_shared(block):
            return block
        new = self.alloc_block()
        self.k_pool[:, new].copy_(self.k_pool[:, block])
        self.v_pool[:, new].copy_(self.v_pool[:, block])
        self.free_block(block)
        return new

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
            blk = kv.block_table[bi, pos // BLOCK_TOKENS].to(dev)
            off = (pos % BLOCK_TOKENS).to(dev)
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

    def alloc_slot(self) -> int:
        if not self._free:
            raise RuntimeError(f"LinearStatePool exhausted: all {self.num_slots} slots in use")
        slot = self._free.pop()
        self.states[slot].zero_()
        if self.conv_windows is not None:
            self.conv_windows[slot].zero_()
        return slot

    def free_slot(self, slot: int) -> None:
        if slot in self._free:
            raise RuntimeError(f"free_slot: slot {slot} already free (double free)")
        self._free.append(slot)


@dataclass(frozen=True)
class PrefixHit:
    """Matched token count and the store-retained blocks covering ``[0, length)``."""

    length: int
    blocks: tuple[int, ...]


@dataclass(slots=True)
class _Entry:
    eid: int
    tokens: tuple[int, ...]
    blocks: tuple[int, ...]
    h: int


class NoPrefixStore:
    """Never matches, never retains: a training rollout must not serve KV
    computed under an earlier policy. Also the miss-path double for tests."""

    on_evict: Callable[[tuple[int, ...]], None] | None = None

    def lookup(self, tokens: Sequence[int]) -> PrefixHit | None:
        return None

    def insert(self, tokens: Sequence[int], blocks: Sequence[int]) -> None:
        return None

    def evict_until_free(self, blocks: int) -> None:
        return None


class PrefixStore:
    """Rolling-hash prefix cache over a :class:`PagedKvPool`.

    An entry maps a prefix's full hash to (token tuple, physical blocks); every
    hash hit is verified against the stored tokens. Insert retains every block;
    FIFO eviction at ``capacity`` (or under block pressure) releases them and
    calls ``on_evict(tokens)`` so side tables keyed by the same tuple follow.
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
        """Cache ``tokens`` (covered by ``blocks``) and retain the blocks; a duplicate is a no-op."""
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
                return
        entry = _Entry(self._next_id, tokens, blocks, h)
        self._next_id += 1
        self._entries.setdefault(h, []).append(entry)
        self._by_id[entry.eid] = entry
        self._fifo.append(entry.eid)
        for b in blocks:
            self._pool.retain(b)
        while len(self._by_id) > self.capacity:
            self._evict_one()

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
                    self.hits += 1
                    return PrefixHit(i, e.blocks)
        self.misses += 1
        return None

    def evict_until_free(self, blocks: int) -> None:
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
        return {
            "entries": len(self._by_id),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }
