"""Paged KV cache, linear-attention state pool, and prefix store.

Host-side bookkeeping (plain ints/lists) over torch tensors on the target
device, after agent-infer's ``host_paged_kv_pool.rs`` / ``prefix_store.rs``.
# ponytail: one refcount per block counts every owner (slots + prefix store);
# no preempt/swap, no cpu-offload — ``alloc_block`` raises on exhaustion.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

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

    A block with refcount > 1 is shared (a live slot plus the prefix store).
    Only whole blocks are ever published, so a shared block is never appended to
    and no copy-on-write is needed; :meth:`PrefixStore.insert` enforces that.
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

    def clear(self) -> None:
        self._held.clear()
        self._used = 0

    def stats(self) -> dict[str, int]:
        return {
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

    #: Nothing is retained, so nothing is ever evicted. Declared rather than left to a
    #: getattr default at the read site: a duck-type miss there reports 0 evictions for a
    #: real store too, which is exactly the "no pressure" reading a tier below would act on.
    evictions = 0

    def lookup(self, tokens: Sequence[int]) -> PrefixHit | None:
        return None

    def insert(self, tokens: Sequence[int], blocks: Sequence[int], state: Any = None) -> bool:
        return False

    def evict_until_free(self, blocks: int) -> None:
        return None

    def clear(self) -> None:
        return None

    def stats(self) -> dict[str, int]:
        return {"entries": 0, "capacity": 0, "state_bytes": 0, "hits": 0, "misses": 0,
                "evictions": 0}


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
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _hash_all(self, tokens: Sequence[int]) -> int:
        h = 0
        for t in tokens:
            h = self._roll(h, int(t))
        return h

    def insert(self, tokens: Sequence[int], blocks: Sequence[int], state: Any = None) -> bool:
        """Cache ``tokens`` (covered by ``blocks``) with its ``state`` snapshot and retain
        the blocks; True when a new entry was retained, False for a duplicate."""
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
        while len(self._by_id) > self.capacity or self._state_used > self.state_bytes:
            # State-byte pressure with a tier is not a reason to lose a prefix: demote the
            # LRU snapshot instead and keep the entry matchable. Only when nothing is left
            # to demote (or there is no tier) does the entry go.
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
                    self.hits += 1
                    self._by_id.move_to_end(e.eid)  # this is the whole of "recently used"
                    return PrefixHit(i, e.blocks, e.state)
        self.misses += 1
        return None

    def evict_until_free(self, blocks: int) -> None:
        while self._pool.free_blocks < blocks and self._by_id:
            self._evict_one()

    def _drop(self, entry: _Entry) -> None:
        """Remove one entry and release everything it holds. The single teardown path:
        eviction, a promotion that came back empty, and ``clear`` all go through it, so
        the block frees and the byte accounting cannot drift between them."""
        del self._by_id[entry.eid]
        chain = self._entries[entry.h]
        chain.remove(entry)
        if not chain:
            del self._entries[entry.h]
        for b in entry.blocks:
            self._pool.free_block(b)
        if not entry.demoted:
            self._state_used -= entry.nbytes
        if self._dram is not None:
            self._dram.forget(entry.eid)
        self.evictions += 1

    def _evict_one(self) -> None:
        eid = next(iter(self._by_id))  # least recently used
        self._drop(self._by_id[eid])

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

    def stats(self) -> dict[str, int]:
        st = {
            "entries": len(self._by_id),
            "capacity": self.capacity,
            "state_bytes": self._state_used,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
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


