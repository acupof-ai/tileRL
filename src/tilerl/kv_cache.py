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

import contextlib
import os
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

__all__ = [
    "BLOCK_TOKENS",
    "PagedKvPool",
    "LinearStatePool",
    "PrefixStore",
    "NoPrefixStore",
    "PrefixHit",
    "KvTier",
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
        Torch fallback for arches without the ``write_tokens`` scatter kernel —
        the CPU target, and any cell that has not registered it yet.

        Indexed, not looped: the per-token ``int(block_table[...])`` this
        replaced was ``b * seq_q`` host syncs PER LAYER, so a 512-token prefill
        chunk of the 27B cost 8192 of them a tick. The two remaining are one
        ``tolist()`` each for the whole batch.
        # ponytail: 2 syncs/layer, not 0 — dropping them needs a mask instead
        # of a per-row length, which is the kernel's job on a cell that has one.
        """
        b, t, _, _ = k.shape
        sql = getattr(kv, "seq_q_lens", None)
        plane = self._plane[layer_idx]
        dev = self.k_pool.device
        lens = [t] * b if sql is None else sql.tolist()
        ends = kv.seq_len.tolist()
        for bi in range(b):
            sq = int(lens[bi])
            # pos indexes the block table first, so it starts on ITS device;
            # the pool may live on another (a CPU-resident table with an mps
            # pool is the metal parity path).
            pos = torch.arange(int(ends[bi]) - sq, int(ends[bi]),
                               device=kv.block_table.device)
            blk = kv.block_table[bi, pos // BLOCK_TOKENS].to(dev)
            off = (pos % BLOCK_TOKENS).to(dev)
            self.k_pool[plane, blk, :, off, :] = k[bi, :sq].to(self.k_pool.dtype)
            self.v_pool[plane, blk, :, off, :] = v[bi, :sq].to(self.v_pool.dtype)

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
    ``[0, length)``. A resident hit's blocks are already retained (the caller
    adopts them with :meth:`PagedKvPool.retain`); a cold hit has empty
    ``blocks`` and a ``reload_key`` — the caller allocs fresh blocks and asks
    the tier to fill them (see :class:`KvTier`)."""

    length: int
    blocks: tuple[int, ...]
    reload_key: int | None = None


class KvTier:
    """SSD byte-store below the HBM pool: spilled prefix KV + GDN snapshots.

    A prefix evicted from the pool spills here instead of being dropped; a later
    lookup reloads it into fresh blocks, skipping the prefill recompute. On a
    32 GB V100 with a full host there is no DRAM residency tier, so it is
    HBM→SSD.

    # ponytail: sync reload (torch.load), pinned-ring async prefetch when hit
    #   latency bites; raw bf16 spill, fp8 tier-quant is 2x capacity if SSD fills
    """

    def __init__(self, path: str, min_tokens: int = 2048, max_pending: int = 32,
                 max_bytes: int = 100 * 2**30) -> None:
        import queue
        import shutil
        import threading

        self.min_tokens = min_tokens  # per-tier floor (composite tier sets one per level)
        # bound in-flight writes: bursty eviction can enqueue faster than the
        # disk drains, and an unbounded queue OOMs a 31GB host. Over the cap,
        # spill_kv refuses — the graceful drop the store already handles.
        self._max_pending = max_pending
        self._healthy = True  # daemon failure (disk full/perm) flips this to refuse
        # Size-based LRU: total on-disk bytes capped at max_bytes; the daemon
        # evicts the least-recently-accessed entry's files after each write.
        # has()/load_kv()/load_state() touch an entry to MRU.
        self._max_bytes = max_bytes
        self._lru: "OrderedDict[int, int]" = OrderedDict()
        self._total = 0
        # Never rmtree the caller's path — it may be a shared dir. Own a fixed
        # subdir marked by a sentinel file; only wipe a dir that carries the
        # marker (a dead process's spill files are orphans, safe to clear).
        self._dir = os.path.join(os.fspath(path), "tilerl_kvtier")
        marker = os.path.join(self._dir, ".kvtier")
        if os.path.isdir(self._dir) and os.path.exists(marker):
            shutil.rmtree(self._dir, ignore_errors=True)
        elif os.path.exists(self._dir):
            raise RuntimeError(f"{self._dir} exists but is not a KvTier dir (no .kvtier marker)")
        os.makedirs(self._dir, exist_ok=True)
        open(marker, "w").close()
        # Deferred write: spill_kv runs inside a decode tick, so it does only the
        # GPU->CPU copy + enqueue; a daemon flushes the ~100ms torch.save off-tick.
        # _pending/_pending_st serve blobs not yet on disk, so has()/load see them.
        self._pending: dict[int, dict] = {}
        self._pending_st: dict[int, dict] = {}
        self._lock = threading.Lock()
        self._q: "queue.Queue" = queue.Queue()
        self._writer = threading.Thread(target=self._flush_loop, daemon=True)
        self._writer.start()

    def _flush_loop(self) -> None:
        while True:
            tag, blob, dst = self._q.get()
            # Write only while the entry is still pending: a drop() that raced us
            # already removed it, and writing now would resurrect an evicted
            # prefix on disk (write-back invalidation → wrong tokens).
            k = tag[1] if isinstance(tag, tuple) else tag
            table = self._pending_st if isinstance(tag, tuple) else self._pending
            with self._lock:
                if table.get(k) is not blob:
                    continue
            try:
                torch.save(blob, dst)
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
            while self._total > self._max_bytes and len(self._lru) > 1:
                victim = next(
                    (k for k in self._lru if k not in self._pending and k not in self._pending_st),
                    None,
                )
                if victim is None:
                    break  # every entry is still being written
                vs = self._lru.pop(victim)
                self._total -= vs
                for p in (self._kv(victim), self._st(victim)):
                    with contextlib.suppress(FileNotFoundError):
                        os.remove(p)

    def _touch_lru(self, key: int) -> None:
        with self._lock:
            if key in self._lru:
                self._lru.move_to_end(key)

    def _kv(self, key: int) -> str:
        return os.path.join(self._dir, f"{key & _MASK64:016x}.kv")

    def _st(self, key: int) -> str:
        return os.path.join(self._dir, f"{key & _MASK64:016x}.st")

    def spill_kv(self, key: int, tokens: tuple[int, ...], blocks: Sequence[int],
                 pool: "PagedKvPool") -> bool:
        # True = accepted. The tier owns both the length floor and the capacity
        # refusal (the store never pre-gates), so a composite tier can vary them
        # per level. Refuse below min_tokens or when the writer is behind/dead.
        if len(blocks) * BLOCK_TOKENS < self.min_tokens:
            return False
        with self._lock:
            if not self._healthy or len(self._pending) >= self._max_pending:
                return False
        k = torch.stack([pool.k_pool[:, b] for b in blocks]).contiguous().cpu()
        v = torch.stack([pool.v_pool[:, b] for b in blocks]).contiguous().cpu()
        # Store tokens too: files are keyed by a 64-bit hash, so a collision would
        # otherwise load a different prefix's KV. load_kv verifies before copying.
        blob = {"k": k, "v": v, "tokens": tuple(tokens)}
        with self._lock:
            self._pending[key] = blob
        self._q.put((key, blob, self._kv(key)))
        return True

    def load_kv(self, key: int, tokens: tuple[int, ...], blocks: Sequence[int],
                pool: "PagedKvPool") -> bool:
        # False = data gone (a raced eviction dropped it) OR a hash collision
        # stored a different prefix — caller treats either as a miss. Serves a
        # still-pending blob from memory, closing the has()/load TOCTOU.
        with self._lock:
            blob = self._pending.get(key)
        if blob is None:
            if not os.path.exists(self._kv(key)):
                return False
            blob = torch.load(self._kv(key), map_location="cpu")
        if blob.get("tokens") != tuple(tokens):
            return False  # hash collision: these bytes belong to a different prefix
        self._touch_lru(key)
        for i, b in enumerate(blocks):
            pool.k_pool[:, b].copy_(blob["k"][i].to(pool.device))
            pool.v_pool[:, b].copy_(blob["v"][i].to(pool.device))
        return True

    def spill_state(self, key: int, tokens: tuple[int, ...], states, windows) -> None:
        blob = {"states": states.cpu(), "windows": None if windows is None else windows.cpu(),
                "tokens": tuple(tokens)}
        with self._lock:
            self._pending_st[key] = blob
        self._q.put((("st", key), blob, self._st(key)))

    def load_state(self, key: int, tokens: tuple[int, ...]):
        # None = gone or a hash-collision mismatch — caller degrades to a miss.
        with self._lock:
            blob = self._pending_st.get(key)
        if blob is None:
            if not os.path.exists(self._st(key)):
                return None
            blob = torch.load(self._st(key), map_location="cpu")
        if blob.get("tokens") != tuple(tokens):
            return None
        self._touch_lru(key)
        return blob["states"], blob["windows"]

    def has(self, key: int, tokens: tuple[int, ...]) -> bool:
        # A cold hit is valid only if BOTH the KV and the state are present AND
        # their stored tokens match (a 64-bit hash collision stores a different
        # prefix). Checking here means submit's loads cannot then fail-mismatch.
        with self._lock:
            kv = self._pending.get(key)
            st = self._pending_st.get(key)
        if kv is None:
            if not os.path.exists(self._kv(key)):
                return False
            kv = torch.load(self._kv(key), map_location="cpu")
        if st is None:
            if not os.path.exists(self._st(key)):
                return False
            st = torch.load(self._st(key), map_location="cpu")
        t = tuple(tokens)
        if kv.get("tokens") == t and st.get("tokens") == t:
            self._touch_lru(key)
            return True
        return False

    def drop(self, key: int) -> None:
        with self._lock:
            self._pending.pop(key, None)
            self._pending_st.pop(key, None)
            self._total -= self._lru.pop(key, 0)
        for p in (self._kv(key), self._st(key)):
            with contextlib.suppress(FileNotFoundError):
                os.remove(p)


class _Entry:
    __slots__ = ("eid", "tokens", "blocks", "h")

    def __init__(self, eid: int, tokens: tuple[int, ...], blocks: tuple[int, ...], h: int) -> None:
        self.eid = eid
        self.tokens = tokens
        self.blocks = blocks
        self.h = h


class NoPrefixStore:
    """Prefix store that never matches and never retains.

    For a training rollout: a cached prefix serves KV computed under an EARLIER
    policy, which silently makes an on-policy method off-policy. Also the
    miss-path double for tests.
    """

    on_evict: "Callable[[tuple[int, ...], int], None] | None" = None
    on_demote: "Callable[[tuple[int, ...], int], None] | None" = None

    def lookup(self, tokens: Sequence[int]) -> "PrefixHit | None":
        return None

    def insert(self, tokens: Sequence[int], blocks: Sequence[int]) -> None:
        return None


class PrefixStore:
    """Rolling-hash prefix cache over a :class:`PagedKvPool`.

    Keys are per-token rolling hashes ``h_i = roll(h_{i-1}, token_i)``; an
    entry maps a prefix's full hash to (token tuple, physical blocks). The
    engine inserts each completed block span (full blocks plus the current
    partial); lookup returns the LONGEST stored prefix of the query.
    Collision-safe: hash hits are verified against the stored token contents,
    so a colliding entry with different tokens is a miss. Insert retains
    every block in the pool; FIFO eviction at ``capacity`` releases them.
    Blocks a live slot still holds stay allocated. With a ``tier``, an eviction
    spills to it (cold index) instead of dropping; ``on_demote(tokens, key)``
    moves the side snapshot with it and ``on_evict(tokens, key)`` fires only
    when an entry leaves the store entirely.
    """

    def __init__(
        self,
        pool: PagedKvPool,
        capacity: int = 4096,
        tier: "KvTier | None" = None,
        tier_capacity: int = 256,
    ) -> None:
        self._pool = pool
        self.capacity = capacity
        # on_evict: entry left the store entirely; on_demote: spilled to the tier
        self.on_evict: Callable[[tuple[int, ...], int], None] | None = None
        self.on_demote: Callable[[tuple[int, ...], int], None] | None = None
        self._roll = _rolling_hash
        self._entries: dict[int, list[_Entry]] = {}
        self._by_id: dict[int, _Entry] = {}
        self._fifo: deque[int] = deque()
        # SSD tier + its cold index (blocks freed, bytes on disk keyed by e.h)
        self._tier = tier
        self.tier_capacity = tier_capacity
        self._cold: dict[int, list[_Entry]] = {}
        self._cold_fifo: deque[int] = deque()
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
        # Retire a cold twin of the same prefix (reloaded, now re-published):
        # it shares this hash, so leaving it lets its later eviction drop(h) and
        # delete THIS resident entry's spilled file. Same-hash different-tokens
        # cold entries stay — only the exact-token twin is stale.
        for e in list(self._cold.get(h, ())):
            if e.tokens == tokens:
                self._drop_cold(e)
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
        collision on different tokens is a miss. A resident hit returns the
        pool blocks; a cold hit (spilled to the tier) returns empty blocks and
        ``reload_key`` — the caller allocs fresh blocks and reloads. Both hot
        and cold hits touch their FIFO to MRU (LRU, so a startup system prompt
        is not the first victim)."""
        tokens = tuple(int(t) for t in tokens)
        h = 0
        prefix_hashes: list[int] = []
        for t in tokens:
            h = self._roll(h, t)
            prefix_hashes.append(h)
        for i in range(len(tokens), 0, -1):
            key = prefix_hashes[i - 1]
            for e in self._entries.get(key, ()):
                if e.tokens == tokens[:i]:
                    self.hits += 1
                    self._touch(self._fifo, e.eid)
                    return PrefixHit(i, e.blocks)
            for e in self._cold.get(key, ()):
                if e.tokens == tokens[:i]:
                    self.hits += 1
                    self._touch(self._cold_fifo, e.eid)
                    return PrefixHit(i, (), reload_key=e.h)
        self.misses += 1
        return None

    @staticmethod
    def _touch(fifo: deque, eid: int) -> None:
        try:
            fifo.remove(eid)
        except ValueError:
            return
        fifo.append(eid)

    def _evict_if_needed(self) -> None:
        # capacity bounds RESIDENT entries; cold (spilled) entries are bounded
        # separately by tier_capacity. _fifo holds exactly the resident ids.
        while len(self._fifo) > self.capacity:
            self._evict_one()

    def evict_until_free(self, blocks: int) -> None:
        """Evict oldest entries until the pool can satisfy an allocation."""
        while self._pool.free_blocks < blocks and self._fifo:
            self._evict_one()

    def _evict_one(self) -> None:
        eid = self._fifo.popleft()
        entry = self._by_id[eid]
        chain = self._entries[entry.h]
        chain.remove(entry)
        if not chain:
            del self._entries[entry.h]
        # spill_kv reads the blocks (so it runs before free_block) and owns the
        # accept/reject; a refusal drops as before.
        spilled = self._tier is not None and self._tier.spill_kv(
            entry.h, entry.tokens, entry.blocks, self._pool
        )
        for b in entry.blocks:
            self._pool.free_block(b)
        if spilled:
            if self.on_demote is not None:
                self.on_demote(entry.tokens, entry.h)
            entry.blocks = ()
            self._cold.setdefault(entry.h, []).append(entry)
            self._cold_fifo.append(eid)
            self._evict_cold_if_needed()
        else:
            self._by_id.pop(eid)
            if self.on_evict is not None:
                self.on_evict(entry.tokens, entry.h)
        self.evictions += 1

    def _drop_cold(self, entry: "_Entry") -> None:
        """Remove one cold entry from the index; drop its tier file only when no
        other cold entry shares the hash (files are hash-named, so a same-hash
        twin still needs it)."""
        self._by_id.pop(entry.eid, None)
        with contextlib.suppress(ValueError):
            self._cold_fifo.remove(entry.eid)
        chain = self._cold.get(entry.h)
        if chain and entry in chain:
            chain.remove(entry)
            if not chain:
                del self._cold[entry.h]
        if self._tier is not None and entry.h not in self._cold:
            self._tier.drop(entry.h)
        if self.on_evict is not None:
            self.on_evict(entry.tokens, entry.h)

    def _evict_cold_if_needed(self) -> None:
        while len(self._cold_fifo) > self.tier_capacity:
            eid = self._cold_fifo[0]
            self._drop_cold(self._by_id[eid])

    def stats(self) -> dict[str, int]:
        """Cache counters: entries, capacity, hits, misses, evictions."""
        return {
            "entries": len(self._by_id),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }
