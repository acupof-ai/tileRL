"""Host and SSD KV tiers and the cold-start boot store.

Split out of kv_cache.py (docs/design-architecture.md step 9): device pools
stay there; host RAM demotion (HostKvPages), its mmap'd SSD spill
(ColdSsdFile), recurrent-state snapshots (DramSnapshots) and the
--kv-store boot directory (KvBootStore) live here. Imports point into
kv_cache one-way; pools never name a tier class.
"""

from __future__ import annotations

import contextlib
import os
import pickle
import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from .kv_cache import _MASK64, BLOCK_TOKENS, _nbytes, _rolling_hash

if TYPE_CHECKING:
    from .kv_cache import PagedKvPool


class SpillWriteError(OSError):
    """A cold page the caller must persist could not be written to the spill
    file (an unwritable directory, ENOSPC, an I/O error). Distinct from the
    shared-prefix case, which is a cache the engine may simply forget."""


def _shared_ssd_path(ssd_path: str) -> str:
    """The shared-prefix spill file is a sibling of the private spill path."""
    return (ssd_path[:-4] if ssd_path.endswith(".bin") else ssd_path) + ".prefix.bin"


#: Canonical shared-spill blob layouts, keyed by a stable field-set signature.
#: A normal engine produces exactly two: a cold page {k,v,bounds} and a warm
#: spec page {k,v,bounds,dk,dv}. dtype is intentionally NOT part of the signature
#: (narrow-f16 cold pages vs f32 frame blobs share a layout family); within one
#: bucket every blob carries the same fields, and the file freezes their
#: shapes/dtypes at first write. An unexpected field set never opens its own
#: file — it spills to RAM instead, so the number of sibling files is bounded.
_SHARED_SIG_COLD = ("bounds", "k", "v")
_SHARED_SIG_WARM = ("bounds", "dk", "dv", "k", "v")
_SHARED_SIGS = frozenset({_SHARED_SIG_COLD, _SHARED_SIG_WARM})
#: Hard cap on sibling spill files regardless of future whitelist additions.
_MAX_SHARED_BUCKETS = 4


def _blob_sig(blob: dict):
    """Canonical field-set signature for a shared blob, or None when the field
    set is not a recognized layout (caller keeps such a blob in RAM)."""
    sig = tuple(sorted(blob.keys()))
    return sig if sig in _SHARED_SIGS else None


def _shared_bucket_path(ssd_path: str, sig) -> str:
    """Sibling spill path per layout: cold uses the plain .prefix.bin, any other
    recognized layout a suffixed sibling so the two fixed strides never mix."""
    base = _shared_ssd_path(ssd_path)
    if sig == _SHARED_SIG_COLD:
        return base
    return (
        base[:-4] + ".w" + str(len(sig)) + ".bin"
        if base.endswith(".bin")
        else base + ".w" + str(len(sig))
    )


def _prefix_spill_bounded() -> bool:
    """Env gate for the shared-prefix (.prefix.bin) spill cap + disk reclaim.
    Default OFF: the shared spill stays append-grown and admission-unbounded
    (a publish-only cache), preserving prior behavior and tests. Set
    TILERL_COLD_PREFIX_SSD_CAP=1 to bound it by --cold-ssd-bytes and return
    freed trailing extents to the filesystem."""
    return os.environ.get("TILERL_COLD_PREFIX_SSD_CAP", "").strip() not in (
        "",
        "0",
        "false",
        "False",
    )


def _close_bg_publish() -> bool:
    """Env gate for moving the request-close private->shared page transfer
    (host pop + private-SSD lift + .prefix.bin write) off the close critical
    path onto one bounded single-consumer thread. Default OFF: the transfer
    stays inline on the step thread, byte-identical. Set
    TILERL_CLOSE_BG_PUBLISH=1 to enable."""
    return os.environ.get("TILERL_CLOSE_BG_PUBLISH", "").strip() not in ("", "0", "false", "False")


def assert_spill_writable(path: str) -> None:
    """Open/creates ``path`` for writing at build time, so a serve pointed at an
    unwritable spill location refuses to START instead of wedging the first tick
    whose demotion exceeds the host budget. Leaves the file behind: the spill
    constructor creates it lazily on first spill and treats an existing file as
    normal. Names the path and errno in the error."""
    if not path:
        return
    try:
        with open(path, "ab"):
            pass
    except OSError as e:
        raise SpillWriteError(
            f"cold spill path {path!r} is not writable "
            f"(errno {e.errno}: {e.strerror}); pass a writable --cold-ssd-path"
        ) from e


class _SlotToken:
    """One-shot settlement handle for a ColdSsdFile reserved slot. Exactly one of
    commit()/release() takes effect; dispose() in a finally releases if neither
    ran. Every path balances the file's mapping _borrowed exactly once. Methods
    acquire the file's non-reentrant _mlock, so never call them while already
    holding it (use the file's *_locked internals there)."""

    __slots__ = ("file", "slot", "mapping", "_settled")

    def __init__(self, file, slot, mapping):
        self.file = file
        self.slot = slot
        self.mapping = mapping
        self._settled = False

    def commit(self, key) -> bool:
        if self._settled:
            return False
        self._settled = True
        with self.file._mlock:
            self.file._commit_slot_locked(key, self.slot)
        return True

    def release(self) -> bool:
        if self._settled:
            return False
        self._settled = True
        with self.file._mlock:
            self.file._release_slot_locked(self.slot)
        return True

    def dispose(self) -> None:
        """finally hook: release unless already committed/released."""
        if not self._settled:
            self.release()


class ColdSsdFile:
    """One mmap'd spill file for cold pages past the host-RAM budget.

    Fixed-stride slots; the caller keys a page by an OPAQUE stable key (the
    sparse engine passes ``(req_id, logical page)``) and this file maps it to a
    monotonic slot whose offset is ``header + slot*stride``. Keys cannot be
    slots: a physical frame is recycled while an older page stays cold, so
    numbering slots by the key would alias two pages. Freed slots return to a
    LIFO free list. Presence is the in-memory key->slot map — this is the
    *serving* spill for one process, not boot (KvBootStore keys by prefix and
    has its own manifest). The file grows on a new slot; the mapping is
    remapped on growth.
    """

    HEADER = 4096
    #: The mapping and file grow one extent at a time, not one slot: every growth
    #: remaps, and remapping per high-water slot rebuilt the whole mmap thousands
    #: of times during a cold fill. Unused extent slots are at most one chunk of
    #: tail space.
    GROWTH_SLOTS = 64

    def __init__(
        self,
        path: str,
        spec: list[tuple[str, tuple, str, int]],
        step_timing=None,
        reclaim: bool = False,
    ) -> None:
        import json
        import mmap

        import numpy as np

        self._np = np
        self._mmap = mmap
        #: When True, forgetting the last live slot of a trailing extent truncates
        #: the file back, so a release wave returns physical disk instead of
        #: leaving the spill grown to its high-water mark forever. LIFO free-slot
        #: reuse already bounds the high slot under a steady working set; this only
        #: reclaims the TAIL. Off (default) the file is append-grown as before.
        # ponytail: tail-collapse only; interior free extents are reused via the
        # LIFO free list. Add FALLOC_FL_PUNCH_HOLE (linux) / F_PUNCHHOLE (apfs)
        # if mid-file fragmentation ever leaves physical bytes above the cap.
        self._reclaim = reclaim
        #: live slot count per extent index (extent = slot // GROWTH_SLOTS).
        self._extent_live: list[int] = []
        #: Optional env-gated step timer. Every path that touches the mmap adds
        #: its wall time to `ssd_ms`, which the owning HostKvPages drains into
        #: the step timer. Bracket here rather than at each call site so a new
        #: reader cannot silently go uncounted.
        self.step_timing = step_timing
        self.ssd_ms = 0.0
        self._path = path
        self._spec = spec
        self.stride = sum(n for *_k, n in spec)
        self._slot_of: dict = {}
        self._free_slots: list = []
        self._next_slot = 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        if new:
            with open(path, "wb") as create:
                create.close()  # create so the seekable r+b handle can open it
        self._f = open(path, "r+b")  # noqa: SIM115 — long-lived mmap-backed handle, closed in close()
        if new:
            self._f.write(json.dumps(spec).encode().ljust(self.HEADER, b"\0"))
            self._f.flush()
        self._map = None
        self._cap = (os.path.getsize(path) - self.HEADER) // self.stride
        #: Metadata lock, SEPARATE from the owning tier's _tlock. Slot allocation,
        #: the key->slot map, free list, extent live counts and the mmap rotation
        #: all run under it. Lock order is tier _tlock -> this lock, always; the
        #: background publish worker takes it for a short reserve/commit only and
        #: does the slow disk bytes with NEITHER lock held.
        self._mlock = threading.Lock()
        #: Keys a background lift is reading / about to consume. While pinned the
        #: slot is NOT returned to the free list by forget and is invisible to
        #: allocation, so the step thread cannot recycle (or a reclaim truncate
        #: the extent of) the source bytes a lock-outside lift still needs. The
        #: lift unpins in its phase-3 commit or in its failure rollback.
        self._pinned_keys: set = set()
        self._reserved: set = set()
        #: Count of lock-outside readers/writers currently using a mapping that a
        #: concurrent grow must not munmap. Only the single publish worker borrows
        #: (once per 3-phase lift); _remap leaves the borrowed mapping alive and a
        #: later prune closes the stale generations.
        self._borrowed = 0
        self._maps: list = []
        # A reopened file's slots are all FREE (the slot map is in-memory; a
        # reopened serving spill resolves nothing), so it starts with zero live
        # extents regardless of its on-disk high-water size.
        self._extent_live = []
        self._remap(self._cap)

    def _remap(self, cap: int) -> None:
        """Map exactly ``cap`` fixed-stride slots (the file already holds them).
        Called under _mlock. Old mappings are kept in _maps rather than closed so
        a lock-outside IO borrowing one is never munmapped mid-copy;
        _prune_maps_locked reaps stale generations once no borrow is active."""
        self._cap = cap
        n_ext = (cap + self.GROWTH_SLOTS - 1) // self.GROWTH_SLOTS
        if n_ext > len(self._extent_live):
            self._extent_live.extend([0] * (n_ext - len(self._extent_live)))
        if cap:
            new_map = self._mmap.mmap(self._f.fileno(), self.HEADER + cap * self.stride)
            self._maps.append(new_map)
            self._map = new_map
            self._prune_maps_locked()

    def _prune_maps_locked(self) -> None:
        """Close every mapping generation except the newest, unless a lock-outside
        IO is borrowing one (then keep them all; the commit path prunes again once
        the borrow returns). Held mappings cost address space, not RSS."""
        if self._borrowed or len(self._maps) <= 1:
            return
        for old in self._maps[:-1]:
            with contextlib.suppress(ValueError):
                old.close()
        self._maps = self._maps[-1:]

    def _alloc_slot_locked(self) -> int:
        """Allocate a slot that is neither free-nor-reusable nor reserved."""
        free = [s for s in self._free_slots if s not in self._reserved]
        if free:
            slot = free[-1]
            del self._free_slots[self._free_slots.index(slot)]
            return slot
        slot = self._next_slot
        while slot in self._reserved:  # reserved slots past the cursor
            self._next_slot += 1
            slot = self._next_slot
        self._next_slot = slot + 1
        if slot >= self._cap:
            cap = self._cap + self.GROWTH_SLOTS
            os.ftruncate(self._f.fileno(), self.HEADER + cap * self.stride)
            self._remap(cap)
        return slot

    def reserve_slot(self):
        """Allocate + grow under the file lock, mark reserved, and return a
        one-shot _SlotToken. Nobody else can read/allocate the slot until
        commit()/release(); its mapping is borrow-protected for a lock-outside
        write. The token is the ONLY way to settle a reservation, so a caller's
        try/finally cannot leak it, and commit/release each balance the borrow
        exactly once regardless of which path ran."""
        with self._mlock:
            slot = self._alloc_slot_locked()
            self._reserved.add(slot)
            self._borrowed += 1
            return _SlotToken(self, slot, self._map)

    # ---- locked internals (caller already holds self._mlock) ----
    def _release_slot_locked(self, slot: int) -> None:
        """Roll a reserved slot back (free list + drop borrow). Idempotent for a
        slot already committed/released. Non-reentrant-lock safe: no acquire."""
        if slot not in self._reserved:
            return
        self._reserved.discard(slot)
        self._free_slots.append(slot)
        self._borrowed = max(0, self._borrowed - 1)
        self._prune_maps_locked()

    def _commit_slot_locked(self, key, slot: int) -> None:
        """Publish a reserved slot (install key, count extent, drop borrow).
        Idempotent/no-op if the slot was already settled."""
        if slot not in self._reserved:
            return
        self._reserved.discard(slot)
        self._slot_of[key] = slot
        self._extent_live[slot // self.GROWTH_SLOTS] += 1
        self._borrowed = max(0, self._borrowed - 1)
        self._prune_maps_locked()

    def release_borrow(self) -> None:
        """End one source-read mapping borrow (distinct from a destination
        reserve token). Non-reentrant-lock safe wrapper."""
        with self._mlock:
            self._borrowed = max(0, self._borrowed - 1)
            self._prune_maps_locked()

    def _drop_read_borrow_locked(self) -> None:
        self._borrowed = max(0, self._borrowed - 1)
        self._prune_maps_locked()

    def _copy_into(self, mapping, slot: int, blob: dict) -> None:
        """Byte copy into a reserved slot through its borrow-protected mapping.
        NO lock: the slot is reserved (no other writer) and the mapping cannot be
        munmapped while borrowed."""
        off = self.HEADER + slot * self.stride
        for k, _shape, _dt, n in self._spec:
            t = blob[k].detach()
            if t.is_cuda:
                t = t.cpu()  # numpy/mmap needs a host tensor; bounds may arrive on device
            dst = torch.from_numpy(
                self._np.frombuffer(mapping, dtype=self._np.uint8, count=n, offset=off)
            )
            dst.copy_(t.contiguous().view(torch.uint8).reshape(-1))
            off += n

    def write_reserved(self, slot: int, blob: dict, mapping) -> None:
        """The slow disk bytes of one reserved slot, copied with NO lock held."""
        self._copy_into(mapping, slot, blob)

    def commit_slot(self, key, slot: int) -> None:
        """Lock-acquiring publish of a reserved slot (token.commit is preferred
        in the worker; this stays for any non-worker/standalone use)."""
        with self._mlock:
            self._commit_slot_locked(key, slot)

    def release_slot(self, slot: int) -> None:
        """Lock-acquiring rollback of a reserved slot (token.release preferred)."""
        with self._mlock:
            self._release_slot_locked(slot)

    def _shrink_trailing_extents(self) -> bool:
        """Release physical disk of every fully-free extent at the high-water end:
        lower _next_slot into the last extent that still holds a live slot and
        truncate the file to it, one remap. Does nothing mid-file (those slots
        cycle through the LIFO free list). An extent with a slot a lock-outside IO
        has RESERVED (live count not yet incremented) is treated as occupied, so
        truncation can never SIGBUS a borrowed mapping's bytes. Returns False when
        the best-effort truncate itself failed (the slot/extent bookkeeping still
        stands; the file keeps the over-allocated size and reuses the slots)."""
        e = len(self._extent_live) - 1
        while (
            e >= 0
            and self._extent_live[e] == 0
            and not any(s // self.GROWTH_SLOTS == e for s in self._reserved)
        ):
            self._extent_live.pop()
            e -= 1
        cap = 0 if e < 0 else (e + 1) * self.GROWTH_SLOTS
        if cap >= self._cap:
            return True
        # The freed trailing slots are no longer reachable: drop them from the LIFO
        # reuse list and pull the monotonic cursor back so a later write grows fresh.
        self._free_slots = [s for s in self._free_slots if s < cap]
        self._next_slot = min(self._next_slot, cap)
        # Best-effort physical reclaim: the slot/extent BOOKKEEPING above already
        # released the space; a truncate failure (ENOSPC race, FS quirk) must not
        # propagate and abort the caller's already-settled state — the file keeps
        # its (now over-allocated) size and reuses the slots via the free list.
        try:
            os.ftruncate(self._f.fileno(), self.HEADER + cap * self.stride)
            self._remap(cap)
        except OSError:
            return False
        return True

    def _charge(self, t: float) -> None:
        self.ssd_ms += (time.perf_counter() - t) * 1000.0

    def write(self, key, blob: dict) -> None:
        if self.step_timing is None:
            return self._write(key, blob)
        t = time.perf_counter()
        try:
            return self._write(key, blob)
        finally:
            self._charge(t)

    def _write(self, key, blob: dict) -> None:
        with self._mlock:
            slot = self._alloc_slot_locked()
            self._copy_into(self._map, slot, blob)
            self._slot_of[key] = slot
            self._extent_live[slot // self.GROWTH_SLOTS] += 1
        # No per-page flush: the page cache writes this back; the serving spill
        # is an in-process capacity tier, not a durability log (KvBootStore is).

    def read(self, key, pin: bool) -> dict:
        if self.step_timing is None:
            return self._read(key, pin)
        t = time.perf_counter()
        try:
            return self._read(key, pin)
        finally:
            self._charge(t)

    def _read(self, key, pin: bool) -> dict:
        with self._mlock:
            slot = self._slot_of[key]
            off = self.HEADER + slot * self.stride
            mapping = self._map
            blob = {}
            for k, shape, dt, n in self._spec:
                t = torch.empty(shape, dtype=getattr(torch, dt), device="cpu", pin_memory=pin)
                # flat byte views: view(uint8) changes the trailing dim, never numel,
                # so both sides flatten first. A numpy view of the WHOLE writable mmap
                # (offset, not a read-only bytes slice) feeds copy_ with zero copies.
                src = torch.from_numpy(
                    self._np.frombuffer(mapping, dtype=self._np.uint8, count=n, offset=off)
                )
                t.view(torch.uint8).reshape(-1).copy_(src)
                blob[k] = t
                off += n
            return blob

    def borrow_read(self, key, pin: bool):
        """Resolve key -> (slot, mapping) under the file lock, borrow the mapping
        AND pin the source key, so the slow bytes can be read with NO lock held
        (read_borrowed). The mapping borrow ends with release_mapping_borrow once
        the bytes are in the caller's blob, but the KEY STAYS PINNED until
        consume_pinned (phase 3) or rollback_pinned (failure): forget skips a
        pinned key, so the step thread cannot recycle the source slot mid-lift.
        None when the key is gone."""
        with self._mlock:
            slot = self._slot_of.get(key)
            if slot is None or key in self._pinned_keys:
                return None
            self._borrowed += 1
            self._pinned_keys.add(key)
            shapes = [(k, shape, dt, n) for k, shape, dt, n in self._spec]
            return slot, self._map, pin, shapes

    def release_mapping_borrow(self) -> None:
        """End the mapping borrow after read_borrowed finished copying; the
        source key remains pinned until consume/rollback."""
        with self._mlock:
            self._borrowed = max(0, self._borrowed - 1)
            self._prune_maps_locked()

    def consume_pinned(self, key) -> tuple[bool, bool]:
        """Forget a pinned key's slot and unpin it. Returns (freed, reclaim_ok):
        freed is True when it actually popped a live slot (caller decrements
        bytes once); reclaim_ok is False only when a live slot was freed but the
        env-gated trailing-extent truncate failed (the caller then treats the
        lift as failed and releases its destination token)."""
        with self._mlock:
            return self._consume_pinned_locked(key)

    def _consume_pinned_locked(self, key) -> tuple[bool, bool]:
        """Non-reentrant variant: consume under an already-held _mlock. Whether
        the key was pinned by a lift OR is an ordinary spill slot owned by the
        caller's in-flight publish, this is the single release: pop, free the
        slot, count the extent down (best-effort shrink), unpin."""
        slot = self._slot_of.pop(key, None)
        self._pinned_keys.discard(key)
        if slot is None:
            return False, True
        self._free_slots.append(slot)
        self._extent_live[slot // self.GROWTH_SLOTS] -= 1
        reclaim_ok = True
        if self._reclaim:
            reclaim_ok = self._shrink_trailing_extents()
        return True, reclaim_ok

    def rollback_pinned(self, key) -> None:
        """Unpin a key WITHOUT freeing its slot (the source is NOT consumed):
        used only when the borrow itself never yielded ownership. The normal
        failed-lift path consumes the source, so this is rarely needed."""
        with self._mlock:
            self._pinned_keys.discard(key)

    def read_borrowed(self, borrowed) -> dict:
        """Copy one slot's bytes via a borrow from borrow_read; NO file lock."""
        slot, mapping, pin, spec = borrowed
        off = self.HEADER + slot * self.stride
        blob = {}
        for k, shape, dt, n in spec:
            t = torch.empty(shape, dtype=getattr(torch, dt), device="cpu", pin_memory=pin)
            src = torch.from_numpy(
                self._np.frombuffer(mapping, dtype=self._np.uint8, count=n, offset=off)
            )
            t.view(torch.uint8).reshape(-1).copy_(src)
            blob[k] = t
            off += n
        return blob

    def read_field(self, key, field: str, pin: bool = False):
        if self.step_timing is None:
            return self._read_field(key, field, pin)
        t = time.perf_counter()
        try:
            return self._read_field(key, field, pin)
        finally:
            self._charge(t)

    def _read_field(self, key, field: str, pin: bool = False):
        """Read ONE tensor of a slot by its spec name (partial read). Returns None
        when the key is absent OR this slot's layout has no such field (a cold
        page asked for a warm-only draft plane): a missing owned field is a clean
        cache miss for the caller, never a KeyError on the step thread."""
        with self._mlock:
            slot = self._slot_of.get(key)
            if slot is None:
                return None
            off = self.HEADER + slot * self.stride
            for k, shape, dt, n in self._spec:
                if k == field:
                    t = torch.empty(shape, dtype=getattr(torch, dt), device="cpu", pin_memory=pin)
                    src = torch.from_numpy(
                        self._np.frombuffer(self._map, dtype=self._np.uint8, count=n, offset=off)
                    )
                    t.view(torch.uint8).reshape(-1).copy_(src)
                    return t
                off += n
            return None

    def forget(self, key) -> None:
        with self._mlock:
            if key in self._pinned_keys:
                # A background lift is reading/consuming this slot: do not
                # recycle it or shrink its extent out from under the lock-outside
                # IO. consume_pinned frees it when the lift finishes.
                return
            slot = self._slot_of.pop(key, None)
            if slot is not None:
                self._free_slots.append(slot)
                self._extent_live[slot // self.GROWTH_SLOTS] -= 1
                if self._reclaim:
                    self._shrink_trailing_extents()

    def __contains__(self, key) -> bool:
        with self._mlock:
            return key in self._slot_of

    def __len__(self) -> int:
        with self._mlock:
            return len(self._slot_of)

    def close(self) -> None:
        with self._mlock:
            for mapping in self._maps:
                with contextlib.suppress(ValueError):
                    mapping.close()
            self._maps = []
            self._map = None
            self._f.close()


def _blob_spec(blob: dict) -> list[tuple[str, tuple, str, int]]:
    """The deterministic per-slot layout: key, shape, dtype name, bytes."""
    return [
        (k, tuple(t.shape), str(t.dtype).replace("torch.", ""), t.numel() * t.element_size())
        for k, t in blob.items()
    ]


class HostKvPages:
    """Pinned-host tier for DEMOTED KV pages (the sparse-KV cold set), the block
    counterpart of :class:`DramSnapshots` for state. Holds one blob per demoted
    key (an int physical block on the #500 seam, a (req, page) tuple on the
    sparse path): every plane's K/V and the fp8 ``k_scale``/``v_scale`` planes.

    Lifecycle mirrors the state tier but the pool frees the device frame on
    :meth:`PagedKvPool.demote_page` and promotion ALLOCATES A NEW block from the
    same pool — a cold key is a tier key, not a live device block. Byte-LRU
    evicts when the pinned budget binds. Prefix-shared pages are never demoted
    (they stay read-only wherever they live); :meth:`demote_page` refuses them.
    """

    def __init__(
        self,
        budget_bytes: int = 4 << 30,
        ssd_path: str = "",
        ssd_capacity_bytes: int = 0,
        step_timing=None,
        bg_publish: bool | None = None,
        bg_depth: int | None = None,
        bg_wait_s: float | None = None,
        bg_max_payload_bytes: int | None = None,
    ) -> None:
        # Fail fast: both the private spill and its shared-prefix sibling must be
        # writable now, because the failure used to surface only after the host
        # budget bound mid-decode (V100 /data00 root-owned, 2026-09-14).
        assert_spill_writable(ssd_path)
        if ssd_path:  # "" means no spill; the derived sibling would be ".prefix.bin"
            assert_spill_writable(_shared_ssd_path(ssd_path))
        #: One re-entrant lock over every RAM/SSD/shared structure. The step
        #: thread and the optional background publish thread take it for short
        #: sections; RLock so a locked public method may call another. The one
        #: place that must NOT hold it is a follower waiting on a publish event.
        self._tlock = threading.RLock()
        self.budget_bytes = budget_bytes
        #: opaque cold key -> held bytes / blob (int block on #500, (req,page) tuple on sparse)
        self._held: OrderedDict[Any, int] = OrderedDict()
        self._blobs: dict[Any, dict] = {}
        self._used = 0
        self.demotions = 0
        self.promotions = 0
        self.drops = 0
        #: pages evicted past the host budget live here, under the opaque cold
        #: key (int phys for #500, (req, page) tuple for sparse); "" = SSD off.
        self._ssd_path = ssd_path
        self._ssd = None
        #: Env-gated step timer, handed to each ColdSsdFile so mmap time lands
        #: in its own segment rather than inside a RAM/LRU bucket.
        self.step_timing = step_timing
        self._ssd_bytes = 0
        self._ssd_page_bytes: dict = {}
        #: Countable SSD spill capacity for admission. 0 with a path means "auto":
        #: the free space of the filesystem holding the spill file (read lazily so
        #: constructing a cold tier never touches a not-yet-created file).
        self._ssd_capacity_bytes = ssd_capacity_bytes
        self._staging: dict | None = None  # one reused pinned promote buffer
        #: SHARED prefix pages, content-addressed by the page-token rolling hash,
        #: held on behalf of a SparsePrefixCache entry. key -> [n, refs, blob|None].
        #: Every store entry covering the page adds a ref; the blob lives under the
        #: SAME pinned budget as private pages and LRU-spills to a prefix spill
        #: file (a page is TRANSFERRED from private to shared, never cloned — the
        #: uncapped clone was the 256k host OOM: a second full KV copy). None blob
        #: = spilled; share_take reads it back.
        self._shared: OrderedDict[int, list] = OrderedDict()
        #: running sum of RAM-resident shared bytes; updated at each RAM/spill
        #: transition so the budget loop never sums all shared records per eviction
        #: (the 256k profile: the O(n) sum was ~9% of post-budget prefill).
        self._shared_ram = 0
        #: Shared SPILL buckets keyed by canonical blob layout signature
        #: (_SHARED_SIG_COLD / _SHARED_SIG_WARM). A record's immutable tag in
        #: _shared[K][2] selects its bucket; None means RAM. _shared_ssd_bytes is
        #: the SUM across every bucket (one global cap, never per-bucket).
        self._shared_ssds: dict = {}
        self._shared_ssd_bytes = 0
        #: Blobs currently held in pinned RAM, keyed by content key. A tag=None
        #: record lives only here; a tag=sig spilled record may ALSO cache its blob
        #: here after a share_take read-through (the tag still says spilled).
        self._shared_blobs: dict = {}
        #: Once a shared-prefix spill raises, shared SSD spill is OFF for the
        #: process: a shared entry is a cache the engine can forget, so a write
        #: failure must not wedge the tick (the 2026-09-14 V100 hang). Private
        #: spill failure is different — a live row needs that page — and raises.
        self.shared_spill_disabled = False
        self.shared_spill_error = ""
        #: Env-gated (TILERL_COLD_PREFIX_SSD_CAP): the publish-only prefix spill
        #: is bounded by the same --cold-ssd-bytes admission cap and reclaims
        #: freed trailing extents to disk. Off by default (append-grown, unbounded).
        self.prefix_spill_bounded = bool(ssd_path) and _prefix_spill_bounded()
        #: one RAM LRU across private and shared pages: ("p",key)/("s",key) -> n.
        self._ram_order: OrderedDict[tuple[str, Any], int] = OrderedDict()
        # ----- optional background private->shared publish (TILERL_CLOSE_BG_PUBLISH) -----
        #: content key -> event fired once the key is committed (or abandoned).
        #: A follower whose lookup entry names a not-yet-committed key waits on
        #: the event instead of raising, and treats a timeout as a cache miss.
        self._pub_pending: dict[int, threading.Event] = {}
        #: share_ref/share_release calls that land before the worker commits the
        #: key; folded into the record's refcount at commit.
        self._pub_pending_refs: dict[int, int] = {}
        #: private keys an in-flight job is about to transfer; forget() must not
        #: remove them before the worker consumes the job (bounded by queue depth).
        self._pub_private: set = set()
        self.bg_enabled = _close_bg_publish() if bg_publish is None else bool(bg_publish)
        # Default queue depth when neither the caller nor TILERL_CLOSE_BG_DEPTH
        # gives one. 512 is enough for a small CPU test; build_engine replaces it
        # with a context-derived depth for serving. The env var always wins so an
        # operator can bound object count without a code change.
        self.bg_depth = (
            int(os.environ["TILERL_CLOSE_BG_DEPTH"])
            if "TILERL_CLOSE_BG_DEPTH" in os.environ and bg_depth is None
            else (512 if bg_depth is None else bg_depth)
        )
        self.bg_wait_s = (
            float(os.environ.get("TILERL_CLOSE_BG_WAIT_S", "30"))
            if bg_wait_s is None
            else bg_wait_s
        )
        #: Hard cap on host bytes held by queued job payloads ABOVE the pinned
        #: cold budget (0 = off). Two payloads are allocated before commit but
        #: enter the shared budget (and its LRU spill) only then: the "hold"
        #: frame blob the 1PR batch made, and a warm-spec "kv" job's attached
        #: draft dk/dv host snapshots (a cold page and its draft block overlap, so
        #: a close burst can attach them to nearly every page). The kv base blob
        #: is already budgeted/on-disk and is not charged. The cap keeps
        #: budget + queued <= 2x budget; an over-cap offer commits inline, which
        #: spills immediately, so a worker lag cannot pin unbounded bytes.
        self.bg_max_bytes = (
            int(os.environ["TILERL_CLOSE_BG_MAX_BYTES"])
            if "TILERL_CLOSE_BG_MAX_BYTES" in os.environ and bg_max_payload_bytes is None
            else (0 if bg_max_payload_bytes is None else bg_max_payload_bytes)
        )
        self._pub_payload_bytes = 0
        self.bg_queued = 0
        self.bg_degraded = 0  # queue full -> caller transferred inline
        self.bg_timeouts = 0
        self.bg_failed = 0
        self._pub_q: queue.Queue | None = None
        self._pub_thread: threading.Thread | None = None
        #: Test seam: invoked by the worker with the dequeued job and NO tier lock
        #: held; production is a no-op. Lets a gate park the worker before commit
        #: while the test drives ref/forget races on the step-thread side.
        self._pub_before_job = lambda job: None
        if self.bg_enabled:
            self._start_publisher()

    def _start_publisher(self) -> None:
        self._pub_q = queue.Queue(maxsize=self.bg_depth)
        self._pub_thread = threading.Thread(
            target=self._publish_worker, name="tilerl-cold-publish", daemon=True
        )
        self._pub_thread.start()

    def _publish_worker(self) -> None:
        """Single consumer: run each page's private->shared transfer off the
        step thread. It is the ONLY other thread touching this tier. Never
        touches CUDA: every source byte is already host-resident or on disk.

        A kv job's private source is OWNED by the job from enqueue
        (_pub_private): once the worker borrows the private slot every terminal
        path consumes it exactly once — a read failure or a non-OSError write
        failure is a prompt miss, not a rollback that leaves the source
        reachable. Only the never-borrowed paths (source already gone, or a
        duplicate content key committed inline) do its zero/ordinary cleanup.
        The destination reservation is a _SlotToken settled exactly once on
        EVERY exception. Lock order is _tlock -> ColdSsdFile._mlock, one
        direction; the slow disk bytes hold neither."""
        while True:
            job = self._pub_q.get()
            try:
                if job is None:
                    return
                self._pub_before_job(job)  # test seam: runs with NO tier lock held
                kind, private_key, shared_key, payload = job
                with self._tlock:
                    self._pub_payload_bytes -= self._job_payload_n(job)
                try:
                    if kind == "kv":
                        event = self._run_kv_publish(private_key, shared_key, payload)
                    else:  # "hold": a host frame blob already folded with bounds
                        blob, n = payload
                        with self._tlock:
                            self.share_hold(shared_key, blob, n)
                            event = self._fold_and_event(shared_key)
                except BaseException:
                    # Safety net only: the kv paths settle their own source and
                    # destination on every failure they can name. This clears the
                    # bookkeeping a totally unexpected exception left behind, so a
                    # follower still gets a fired event instead of a hang.
                    if kind == "kv" and self._ssd is not None:
                        self._ssd.rollback_pinned(private_key)
                    with self._tlock:
                        self.bg_failed += 1
                        self._pub_private.discard(private_key)
                        self._pub_pending_refs.pop(shared_key, None)
                        event = self._pub_pending.pop(shared_key, None)
                if event is not None:
                    event.set()
            finally:
                self._pub_q.task_done()

    def _run_kv_publish(self, private_key, shared_key, extra):
        """Phase 1 (short _tlock): pop a RAM source, or classify an SSD source
        and snapshot its byte size. Then run the RAM commit, dup consume, SSD
        lift, or source-gone miss. Returns the future event."""
        with self._tlock:
            n = self._held.pop(private_key, None)
            blob = self._blobs.pop(private_key, None)
            self._ram_order.pop(("p", private_key), None)
            if n is not None:
                self._used -= n
            ram = blob is not None
            existing = self._shared.get(shared_key)
            if ram:
                src_n = 0
            else:
                ssd_present = self._ssd is not None and private_key in self._ssd
                src_n = (
                    self._ssd_page_bytes.get(private_key, self._ssd.stride)
                    if self._ssd is not None
                    else 0
                )
        if ram:
            if extra:
                blob.update(extra)
                n += sum(t.numel() * t.element_size() for t in extra.values() if torch.is_tensor(t))
            with self._tlock:
                # share_hold also folds the phase-2 inline racer (ref++), so the
                # RAM blob is dropped rather than double-charged in that case.
                self._pub_private.discard(private_key)
                self.share_hold(shared_key, blob, n)
                return self._fold_and_event(shared_key)
        if not ssd_present:
            # Source externally evicted before the worker ran; never borrowed,
            # so ZERO source frees/decrements here.
            with self._tlock:
                self.bg_failed += 1
                self._pub_private.discard(private_key)
                return self._fold_and_event(shared_key)
        if existing is not None:
            return self._commit_dup_ssd_publish(private_key, shared_key, src_n)
        return self._lift_ssd_publish(private_key, shared_key, extra, src_n)

    def _commit_dup_ssd_publish(self, private_key, shared_key, src_n):
        """An SSD-source job whose content key was committed inline while the job
        sat queued. Never borrowed the source (no read), so the key is NOT
        pinned: free it through the spill file's ORDINARY forget under _tlock,
        decrement private bytes once, and ref++ the existing record."""
        with self._tlock:
            self._ssd_page_bytes.pop(private_key, None)
            if private_key in self._ssd:
                self._ssd.forget(private_key)
                self._ssd_bytes -= src_n
            self._pub_private.discard(private_key)
            existing = self._shared.get(shared_key)
            if existing is not None:
                existing[1] += 1
            return self._fold_and_event(shared_key)

    def _lift_ssd_publish(self, private_key, shared_key, extra, src_n):
        """The private source is on SSD and the content key is unpublished.
        Borrow the private slot (mapping + KEY pin), read its bytes with NO lock,
        reserve and write a destination bucket slot with NO lock, then publish
        under _tlock.

        Terminal accounting:
        - read raises (ANY exception): release the mapping borrow, then consume
          the owned private source once (free, extent--, bytes--, unpin) -> miss.
        - destination write raises OSError: release the token, disable shared
          spill, keep the blob in RAM, commit succeeds.
        - destination write raises any OTHER exception: release the token,
          consume the owned source once -> miss. A copy/RuntimeError is not a
          disk-capacity signal; silently pinning corrupted bytes as shared is
          worse than a prompt miss.
        - unrecognized layout / over cap / bucket open failed: no token, the
          blob stays in RAM, commit succeeds.
        """
        borrowed = self._ssd.borrow_read(private_key, False)
        if borrowed is None:
            # take() bypassed the tier marker and consumed the slot between
            # phase 1 and the borrow: that external owner already did the
            # decrements, so do nothing to the source here.
            with self._tlock:
                self.bg_failed += 1
                self._pub_private.discard(private_key)
                return self._fold_and_event(shared_key)
        try:
            blob = self._ssd.read_borrowed(borrowed)
        except BaseException:
            self._ssd.release_mapping_borrow()
            with self._tlock:
                self._consume_owned_source(private_key, src_n)
                self.bg_failed += 1
                return self._fold_and_event(shared_key)
        self._ssd.release_mapping_borrow()
        if extra:
            blob.update(extra)
        n = src_n + sum(t.numel() * t.element_size() for t in extra.values() if torch.is_tensor(t))
        token = None
        sig = _blob_sig(blob)
        over_cap = (
            not self._ssd_path
            or self.shared_spill_disabled
            or (self.prefix_spill_bounded and self._shared_ssd_bytes + n > self.ssd_capacity_bytes)
        )
        if sig is not None and not over_cap:
            try:
                bucket = (
                    self._ensure_shared_ssd(blob)
                    if sig == _SHARED_SIG_COLD
                    else self._open_shared_bucket(sig, blob)
                )
                if bucket is not None:
                    token = bucket.reserve_slot()
                    try:
                        bucket.write_reserved(token.slot, blob, token.mapping)
                    except OSError:
                        token.dispose()  # settled release; blob falls back to RAM
                        token = None
                        self._disable_shared_spill()
            except OSError:
                # Bucket open/grow failed: keep the blob in RAM, same as a write
                # OSError. dispose() is a no-op here (reserve never returned).
                if token is not None:
                    token.dispose()
                token = None
                self._disable_shared_spill()
            except BaseException:
                # Read already SUCCEEDED, so this job owns the private source:
                # consume it exactly once and resolve as a miss on ANY other
                # exception from open/reserve/write.
                if token is not None:
                    token.dispose()
                with self._tlock:
                    self._consume_owned_source(private_key, src_n)
                    self.bg_failed += 1
                    return self._fold_and_event(shared_key)
        with self._tlock:
            return self._finalize_ssd_lift(private_key, shared_key, blob, n, src_n, sig, token)

    def _consume_owned_source(self, private_key, src_n) -> bool:
        """Consume the in-flight private source exactly once (caller holds
        _tlock): pop its byte record, free+unpin the pinned spill slot, decrement
        the private byte total, clear the in-flight marker. Returns the spill
        file's reclaim result (False = trailing-extent truncate failed)."""
        self._ssd_page_bytes.pop(private_key, None)
        reclaim_ok = True
        if self._ssd is not None:
            freed, reclaim_ok = self._ssd.consume_pinned(private_key)
            if freed:
                self._ssd_bytes -= src_n
        self._pub_private.discard(private_key)
        return reclaim_ok

    def _finalize_ssd_lift(self, private_key, shared_key, blob, n, src_n, sig, token):
        """Phase 3 under _tlock. Consume the owned private source once. Re-check
        the content key: an inline publisher that committed during the lock-free
        IO wins, so this job releases ITS destination token and only folds a ref
        (never overwrites the record, double-charges bytes, or orphans the inline
        slot). A failed trailing-extent reclaim on the source is likewise a miss
        that releases the token rather than publishing against unsettled
        accounting. Otherwise publish spilled (token commit) or RAM (token None)."""
        reclaim_ok = self._consume_owned_source(private_key, src_n)
        existing = self._shared.get(shared_key)
        if existing is not None or not reclaim_ok:
            if token is not None:
                token.dispose()  # inline record owns a slot, or source reclaim failed
            if not reclaim_ok:
                self.bg_failed += 1
            if existing is not None:
                existing[1] += 1
            return self._fold_and_event(shared_key)
        if token is not None:
            token.commit(("s", shared_key))
            self._shared_ssd_bytes += n
            self._shared[shared_key] = [n, 1, sig]
        else:
            self._shared[shared_key] = [n, 1, None]
            self._shared_blobs[shared_key] = blob
            self._shared_ram += n
            self._ram_order[("s", shared_key)] = n
            self._enforce_budget()
        return self._fold_and_event(shared_key)

    def _fold_and_event(self, shared_key):
        """Fold the pre-commit ref delta into a committed record and return the
        future event (the worker fires it after releasing _tlock). No committed
        record means the delta is dropped (the queued publish is the only owner
        and it missed)."""
        event = self._pub_pending.pop(shared_key, None)
        delta = self._pub_pending_refs.pop(shared_key, 0)
        rec = self._shared.get(shared_key)
        if rec is not None:
            rec[1] += delta
            if rec[1] <= 0:
                self.share_release(shared_key)
        return event

    def _ensure_shared_ssd(self, blob: dict):
        """Open (once) and return the cold-layout {k,v,bounds} shared bucket.
        Seam kept separate from _open_shared_bucket so gates can inject around
        the cold lift; identical semantics."""
        return self._open_shared_bucket(_SHARED_SIG_COLD, blob)

    def _open_shared_bucket(self, sig, blob):
        """Lazily open the shared spill bucket for a recognized blob layout.
        Returns the file, or None when spilling is off/disabled, the field set
        is unrecognized (it stays in RAM rather than opening an unbounded number
        of sibling files), or the sibling-file cap is reached / open failed."""
        with self._tlock:
            bucket = self._shared_ssds.get(sig)
            if bucket is not None:
                return bucket
            if (
                not self._ssd_path
                or self.shared_spill_disabled
                or len(self._shared_ssds) >= _MAX_SHARED_BUCKETS
            ):
                return None
            try:
                bucket = ColdSsdFile(
                    _shared_bucket_path(self._ssd_path, sig),
                    _blob_spec(blob),
                    step_timing=self.step_timing,
                    reclaim=self.prefix_spill_bounded,
                )
            except OSError:
                self._disable_shared_spill()
                return None
            self._shared_ssds[sig] = bucket
            return bucket

    @property
    def _shared_ssd(self):
        """The cold-layout ({k,v,bounds}) shared bucket; None until first cold
        spill. Compatibility seam for callers/gates that predate warm buckets."""
        return self._shared_ssds.get(_SHARED_SIG_COLD)

    def _disable_shared_spill(self) -> None:
        with self._tlock:
            if self.shared_spill_disabled:
                return
            self.shared_spill_disabled = True
            self.shared_spill_error = "background lift write failure"
            print(
                f"[cold] shared-prefix spill to {_shared_ssd_path(self._ssd_path)!r} "
                "failed once in a background lift; shared SSD spill disabled "
                "for this process, shared pages stay in RAM",
                flush=True,
            )

    @property
    def bytes_held(self) -> int:
        """Pinned host RAM held right now, private blobs + shared prefix blobs.
        Both share the one budget; SSD is separate and not counted here."""
        with self._tlock:
            return self._used + self._shared_ram

    @property
    def ssd_bytes(self) -> int:
        with self._tlock:
            return self._ssd_bytes

    @property
    def ssd_capacity_bytes(self) -> int:
        """Countable SSD spill budget used by admission. 0 when spilling is off.
        Auto (no explicit cap) = free bytes on the filesystem holding the file."""
        if not self._ssd_path:
            return 0
        if self._ssd_capacity_bytes:
            return self._ssd_capacity_bytes
        try:
            stat = os.statvfs(os.path.dirname(self._ssd_path) or ".")
            return stat.f_bavail * stat.f_frsize
        except OSError:
            return 0

    def __contains__(self, key) -> bool:
        with self._tlock:
            return key in self._held or (self._ssd is not None and key in self._ssd)

    def hold(self, key, blob: dict, nbytes: int) -> bool:
        """Pin one page blob; True when held (in host RAM or spilled to SSD).

        A page larger than the whole host budget demotes straight to the spill
        file when one is attached. Over-budget pages evict LRU to SSD; without a
        spill file an LRU/self eviction is a drop, as before. ``key`` is opaque:
        an int physical id (#500 retier seam) or a (req, page) tuple (sparse)."""
        with self._tlock:
            if not nbytes or key in self:
                self.drops += 1
                return False
            if nbytes > self.budget_bytes:
                if not self._ssd_path:
                    self.drops += 1
                    return False
                self._write_ssd(key, blob, nbytes)
                return True
            self._blobs[key] = blob
            self._held[key] = nbytes
            self._used += nbytes
            self._ram_order[("p", key)] = nbytes
            self.demotions += 1
            self._enforce_budget()
            return True

    def _enforce_budget(self) -> None:
        """Evict oldest RAM-resident pages (private AND shared prefix, one LRU)
        until pinned bytes fit the budget. A private page spills to the private
        file or is dropped; a shared (refcounted) page spills to the prefix file.
        A shared page with no spill file stays — a store entry references it.

        Pops from the OrderedDict front: each RAM hold is one O(1) eviction
        attempt, never an O(RAM entries) snapshot per hold. A record that no
        longer matches RAM state (promoted/forgotten/shared-spilled between
        enqueue and sweep) is popped here and gone, not skipped and re-scanned."""
        while self._used + self._shared_ram > self.budget_bytes:
            # One circuit over the entries present when it began: a parked
            # unspillable page is re-appended past the circuit counter, so it is
            # examined at most once per pass, exactly the old snapshot-for-loop
            # semantics. A circuit that evicts nothing returns.
            remaining = len(self._ram_order)
            while remaining > 0:
                (ns, k), n = self._ram_order.popitem(last=False)
                remaining -= 1
                if ns == "p":
                    blob = self._blobs.pop(k, None)
                    if blob is None or self._held.get(k) != n:
                        continue  # promoted/forgotten between enqueue and sweep
                    if k in self._pub_private:
                        # A background publish is about to transfer this blob;
                        # re-park it rather than spill/drop the worker's source.
                        self._blobs[k] = blob
                        self._ram_order[("p", k)] = n
                        continue
                    self._held.pop(k)
                    self._used -= n
                    if self._ssd_path:
                        self._write_ssd(k, blob, n)
                    else:
                        self.drops += 1
                    break
                rec = self._shared.get(k)
                blob = None if rec is None else self._shared_blobs.get(k)
                if rec is None or blob is None or rec[0] != n:
                    continue  # stale shared LRU entry (spilled/released/changed)
                if not self._ssd_path:
                    self._ram_order[(ns, k)] = n  # re-park; cannot drop a refcounted page
                    continue
                if self._shared_evict_ram(k):
                    break
                # shared spill is disabled: the page stays in RAM; park it at the
                # back and keep looking from the front for a private page.
                self._ram_order[(ns, k)] = n
            else:
                return  # nothing evict-able remains in one LRU circuit

    def _write_ssd(self, key, blob: dict, nbytes: int) -> None:
        """Move a page already removed from host accounting onto the spill file.
        A live row can still name a private cold page, so a write failure is a
        SpillWriteError the engine turns into a request failure (not a wedge)."""
        try:
            if self._ssd is None:
                self._ssd = ColdSsdFile(
                    self._ssd_path, _blob_spec(blob), step_timing=self.step_timing
                )
            self._ssd.write(key, blob)
        except OSError as e:
            raise SpillWriteError(
                f"private cold spill write to {self._ssd_path!r} failed for key "
                f"{key!r} (errno {e.errno}: {e.strerror})"
            ) from e
        self._ssd_bytes += nbytes
        self._ssd_page_bytes[key] = nbytes

    def take(self, key) -> dict | None:
        """Remove and return the page blob — from host RAM, or read back through
        the SSD spill path — or None if neither tier has it. A key an in-flight
        background publish owns is not taken: its private bytes are committed to
        the shared namespace or freed by that job, so taking them here would race
        the job's own consume and double-decrement _ssd_bytes / recycle a pinned
        slot. Caller promotes via the shared record after the job commits."""
        with self._tlock:
            if key in self._pub_private or (
                self._ssd is not None and key in self._ssd._pinned_keys
            ):
                return None
            n = self._held.pop(key, None)
            blob = self._blobs.pop(key, None)
            self._ram_order.pop(("p", key), None)
            if blob is not None:
                self._used -= n
                self.promotions += 1
                return blob
            if self._ssd is not None and key in self._ssd:
                pin = torch.cuda.is_available()  # promote H2D wants a pinned source
                blob = self._ssd.read(key, pin)
                self._ssd.forget(key)
                self._ssd_bytes -= self._ssd_page_bytes.pop(key, self._ssd.stride)
                self.promotions += 1
                return blob
            return None

    def peek(self, key) -> dict | None:
        """A held RAM blob WITHOUT removing it (clone source for the shared index).
        Does not reach SSD: a page cloned for sharing must still be host-resident."""
        with self._tlock:
            return self._blobs.get(key)

    def forget(self, key) -> None:
        with self._tlock:
            if key in self._pub_private:
                # A background publish job will consume this private blob; its
                # publisher release is the signed ref delta, not a removal here.
                return
            n = self._held.pop(key, None)
            self._blobs.pop(key, None)
            self._ram_order.pop(("p", key), None)
            if n is not None:
                self._used -= n
            elif self._ssd is not None and key in self._ssd:
                self._ssd.forget(key)
                self._ssd_bytes -= self._ssd_page_bytes.pop(key, self._ssd.stride)

    def stats(self) -> dict[str, int]:
        with self._tlock:
            return {
                "kv_cold_pages": len(self._held),
                "kv_cold_bytes": self._used,
                "kv_cold_ssd_pages": 0 if self._ssd is None else len(self._ssd),
                "kv_cold_ssd_bytes": self._ssd_bytes,
                "kv_cold_shared_pages": len(self._shared),
                "kv_cold_shared_bytes": self._shared_ram,
                "kv_cold_shared_ssd_bytes": self._shared_ssd_bytes,
                "kv_cold_shared_ssd_bounded": int(self.prefix_spill_bounded),
                "kv_cold_demotions": self.demotions,
                "kv_cold_promotions": self.promotions,
                "kv_cold_drops": self.drops,
                "kv_cold_bg_queued": self.bg_queued,
                "kv_cold_bg_degraded": self.bg_degraded,
                "kv_cold_bg_timeouts": self.bg_timeouts,
                "kv_cold_bg_failed": self.bg_failed,
            }

    def close(self) -> None:
        """Drain queued publishes, stop the worker, then release the spill files'
        mmap and handles (host RAM is GC'd with the blobs)."""
        self.stop_publisher()
        with self._tlock:
            if self._ssd is not None:
                self._ssd.close()
                self._ssd = None
            for bucket in self._shared_ssds.values():
                bucket.close()
            self._shared_ssds.clear()

    # ----- shared, content-addressed prefix pages (sparse PrefixStore seam) -----
    def share_hold(self, key: int, blob: dict, nbytes: int) -> None:
        """Hold (or refcount) one published prefix page's blob under ``key``. The
        blob is TRANSFERRED from private storage (same allocation, no second
        copy): the caller has already popped it from ``_blobs``. It enters the
        one pinned budget and LRU-spills to the prefix file when the budget
        binds; bounds ride in ``blob['bounds']`` and spill with it.
        Idempotent on key — just adds a reference. With no ssd_path a shared page
        cannot spill or drop, so it pins in RAM until its entry ages out and all
        refs release — the host bound is then the published-prefix working set."""
        with self._tlock:
            rec = self._shared.get(key)
            if rec is not None:
                rec[1] += 1
                return
            self._shared[key] = [nbytes, 1, None]
            self._shared_blobs[key] = blob
            self._shared_ram += nbytes
            self._ram_order[("s", key)] = nbytes
            self._enforce_budget()

    def offer_publish(self, private_key, shared_key: int, extra: dict | None) -> bool:
        """Enqueue one private->shared page transfer for the background worker
        (TILERL_CLOSE_BG_PUBLISH). The private blob stays exactly where it is —
        RAM or the private spill file — until the worker consumes it; the page is
        reserved under ``shared_key`` immediately, so a follower blocks on its
        event instead of taking a miss, and share_ref/share_release calls landing
        before the commit accumulate as a signed delta folded into refcount 1.

        Returns False when the key is already committed (a duplicate content key)
        or the bounded queue is full: the caller then transfers inline, which
        bounds queue memory and never drops a publish. Reservation and enqueue are
        one locked step, so a key can never be reserved without its job queued."""
        return self._enqueue(("kv", private_key, shared_key, extra), private_key, shared_key)

    def offer_hold(self, shared_key: int, blob: dict, nbytes: int) -> bool:
        """Enqueue one already-host frame blob's shared hold (the device-resident
        close page whose D2H the 1PR batch synced before its frame freed). Same
        reservation/future contract as :meth:`offer_publish`."""
        return self._enqueue(("hold", None, shared_key, (blob, nbytes)), None, shared_key)

    def _job_payload_n(self, job) -> int:
        """Host bytes a queued job pins ABOVE the pinned cold budget until the
        worker commits it:

        - "hold": the whole frame blob (1PR batch made it; it enters the shared
          budget and its LRU spill only at commit).
        - "kv": the base blob is already budgeted (RAM) or on disk (SSD), but a
          warm-spec job's ``extra`` carries freshly-snapshotted host draft K/V
          (dk/dv, hundreds of KiB per warm page) plus bounds. Those tensors are
          NOT in the cold budget until commit, so they ARE charged; a cold page
          and its draft block overlap almost entirely, so a close burst can
          attach dk/dv to nearly every page.
        Over-cap offers degrade inline, which spills at once — the bound is
        payload, not object count."""
        if job[0] == "hold":
            return int(job[3][1])
        extra = job[3]
        if not extra:
            return 0
        return sum(t.numel() * t.element_size() for t in extra.values() if torch.is_tensor(t))

    def _enqueue(self, job, private_key, shared_key: int) -> bool:
        with self._tlock:
            if not self.bg_enabled or self._pub_q is None:
                return False
            if shared_key in self._shared or shared_key in self._pub_pending:
                return False
            payload_n = self._job_payload_n(job)
            if self.bg_max_bytes and self._pub_payload_bytes + payload_n > self.bg_max_bytes:
                self.bg_degraded += 1
                return False
            try:
                self._pub_q.put_nowait(job)
            except queue.Full:
                self.bg_degraded += 1
                return False
            self._pub_payload_bytes += payload_n
            self._pub_pending[shared_key] = threading.Event()
            self._pub_pending_refs[shared_key] = 0
            if private_key is not None:
                self._pub_private.add(private_key)
            self.bg_queued += 1
            return True

    def drain_publishes(self, timeout: float | None = None) -> bool:
        """Block until every queued publish has committed (shutdown join). False
        on timeout. Pending reservations whose jobs never ran are abandoned:
        their events fire so a waiter takes a miss rather than hanging."""
        if self._pub_q is None:
            return True
        if timeout is None:
            self._pub_q.join()
            return True
        deadline = time.monotonic() + timeout
        with self._pub_q.all_tasks_done:
            while self._pub_q.unfinished_tasks and time.monotonic() < deadline:
                self._pub_q.all_tasks_done.wait(deadline - time.monotonic())
        ok = self._pub_q.unfinished_tasks == 0
        if not ok:
            with self._tlock:
                for key, ev in list(self._pub_pending.items()):
                    self._pub_pending.pop(key, None)
                    self._pub_pending_refs.pop(key, None)
                    ev.set()
        return ok

    def stop_publisher(self, timeout: float | None = None) -> None:
        """Drain every queued publish, then send the sentinel and join. Timeout
        bounds only the final join (a worker that ignores sentinels); the drain
        itself waits for all queued jobs because a clean shutdown is not a crash.
        Idempotent."""
        q, t = self._pub_q, self._pub_thread
        if q is None:
            return
        self.drain_publishes(None)
        q.put(None)
        if t is not None and timeout is not None:
            t.join(timeout)
        elif t is not None:
            t.join()
        self._pub_thread = None
        self._pub_q = None  # offers after stop fall back to inline transfers

    def has_pending(self) -> bool:
        """True while at least one background publish is queued/uncommitted."""
        with self._tlock:
            return bool(self._pub_pending)

    def has_all_keys(self, keys) -> bool:
        """Non-blocking commit check: every key is a live shared record. An
        adopt runs this under the engine lock after wait_committed returned; a
        False result adopts nothing (timeout, failed publish, or a release that
        landed in between) instead of letting a later share_take raise."""
        with self._tlock:
            return all(k in self._shared for k in keys)

    def wait_committed(self, keys, timeout_s: float | None = None) -> bool:
        """OFF-LOCK wait for background publishes of ``keys`` to commit, then
        confirm every key is a live shared record. False on timeout OR a queued
        publish that failed/abandoned: the caller adopts nothing (a cache miss).

        The only call that blocks on the publish worker. It must run with the
        engine lock released: share_take/share_take_field stay non-blocking so no
        forward/admit path can stall a tick on the worker."""
        if timeout_s is None:
            timeout_s = self.bg_wait_s
        deadline = time.monotonic() + timeout_s
        ok = True
        for key in keys:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                ok = False
                break
            ev = self._pub_pending.get(key)
            if ev is not None and not ev.wait(remaining):
                self.bg_timeouts += 1
                ok = False
                break
        if not ok:
            return False
        with self._tlock:
            return all(k in self._shared for k in keys)

    def share_hold_kv(self, private_key, shared_key: int, extra: dict | None = None) -> int:
        """Transfer one private blob to a shared content key (no clone): pop it
        from the private namespace (RAM or private spill file), fold in ``extra``
        (the small host bounds), and hand it to the shared namespace. The blob
        starts spilled (written straight to the prefix file) when it was already
        past the host budget - no copy is pulled into RAM. Returns bytes held."""
        with self._tlock:
            n = self._held.pop(private_key, None)
            blob = self._blobs.pop(private_key, None)
            self._ram_order.pop(("p", private_key), None)
            self._pub_private.discard(private_key)
            if n is not None:
                self._used -= n
            if blob is None:
                # already past the host budget: lift the blob off the private spill
                # file once (disk, not RSS), fold in bounds, and place it straight in
                # the shared spill namespace - nothing added to host RAM.
                if self._ssd is None or private_key not in self._ssd:
                    return 0
                # Dedupe BEFORE the lift: a second publisher of the same content key
                # (identical prompts both past the host budget) consumes its private
                # copy and ref++s, instead of resetting refs / leaking the first
                # prefix-file slot under live refs. Bounds are deterministic per
                # content, so the existing record already carries them.
                existing = self._shared.get(shared_key)
                if existing is not None:
                    n = self._ssd_page_bytes.pop(private_key, self._ssd.stride)
                    self._ssd.forget(private_key)
                    self._ssd_bytes -= n
                    existing[1] += 1
                    return existing[0]
                n = self._ssd_page_bytes.pop(private_key, self._ssd.stride)
                blob = self._ssd.read(private_key, False)
                self._ssd.forget(private_key)
                self._ssd_bytes -= n
                if extra:
                    blob.update(extra)
                    n += sum(
                        t.numel() * t.element_size() for t in extra.values() if torch.is_tensor(t)
                    )
                sig = _blob_sig(blob)
                if sig is not None and self._write_shared_ssd(shared_key, blob, n, sig):
                    self._shared[shared_key] = [n, 1, sig]
                else:
                    # Unwritable/over-cap sibling spill: keep the in-hand blob in
                    # RAM rather than a record that says "spilled" but is
                    # unreadable (an unrecognized field set is RAM-only too).
                    self._shared[shared_key] = [n, 1, None]
                    self._shared_blobs[shared_key] = blob
                    self._shared_ram += n
                    self._ram_order[("s", shared_key)] = n
                return n
            if extra:
                blob.update(extra)
                n += sum(t.numel() * t.element_size() for t in extra.values() if torch.is_tensor(t))
            self.share_hold(shared_key, blob, n)
            return n

    def _shared_evict_ram(self, key: int) -> bool:
        """Try to spill one RAM-resident shared page to the prefix file.

        Returns True when RAM was freed. On an OSError the page STAYS in RAM and
        shared SSD spill is disabled for the process: the shared blob is a cache
        a prefix entry references, so dropping it would make a later follower
        raise, while keeping it is just the no-spill configuration — memory
        bounded, tokens unchanged. The budget loop moves on to a private page."""
        n, refs, tag = self._shared[key]
        blob = self._shared_blobs.get(key)
        sig = _blob_sig(blob) if blob is not None else None
        if sig is None or not self._write_shared_ssd(key, blob, n, sig):
            return False
        self._shared[key] = [n, refs, sig]
        self._shared_blobs.pop(key, None)
        self._shared_ram -= n
        self._ram_order.pop(("s", key), None)
        return True

    def _write_shared_ssd(self, key: int, blob: dict, nbytes: int, sig) -> bool:
        """Spill one RAM-resident shared page into its signature bucket; True
        when written. An OSError disables shared spill for the process and
        leaves the page in RAM (log once). Never raises.

        With TILERL_COLD_PREFIX_SSD_CAP the shared buckets are bounded TOGETHER
        by the same --cold-ssd-bytes admission the PRIVATE spill reports: a page
        that would exceed it is NOT written and returns False, so the caller
        keeps it in host RAM (the published cache is allowed to forget, never to
        fill the disk). Bounded mode also reclaims freed trailing extents on
        forget. The gate is off by default."""
        if not self._ssd_path or self.shared_spill_disabled:
            return False
        if self.prefix_spill_bounded and self._shared_ssd_bytes + nbytes > self.ssd_capacity_bytes:
            return False
        bucket = self._shared_ssds.get(sig)
        if bucket is None:
            if len(self._shared_ssds) >= _MAX_SHARED_BUCKETS:
                return False
            try:
                bucket = ColdSsdFile(
                    _shared_bucket_path(self._ssd_path, sig),
                    _blob_spec(blob),
                    step_timing=self.step_timing,
                    reclaim=self.prefix_spill_bounded,
                )
            except OSError as e:
                self.shared_spill_disabled = True
                self.shared_spill_error = f"errno {e.errno}: {e.strerror}"
                print(
                    f"[cold] shared-prefix spill to {_shared_bucket_path(self._ssd_path, sig)!r} "
                    f"failed once ({self.shared_spill_error}); shared SSD spill disabled "
                    f"for this process, shared pages stay in RAM",
                    flush=True,
                )
                return False
            self._shared_ssds[sig] = bucket
        try:
            bucket.write(("s", key), blob)
        except OSError as e:
            self.shared_spill_disabled = True
            self.shared_spill_error = f"errno {e.errno}: {e.strerror}"
            print(
                f"[cold] shared-prefix spill to {_shared_bucket_path(self._ssd_path, sig)!r} "
                f"failed once ({self.shared_spill_error}); shared SSD spill disabled "
                f"for this process, shared pages stay in RAM",
                flush=True,
            )
            return False
        self._shared_ssd_bytes += nbytes
        return True

    def share_take(self, key: int) -> dict | None:
        """A read-only REFERENCE to a shared page blob. Read-through: a spilled
        page is loaded from its signature bucket (without removing it — the
        store entry still owns it; promotion copies it into a private fresh
        block). None when the key is not a shared page OR its background publish
        has not committed yet — this never blocks on the worker; a follower
        waits once, off the engine lock, via wait_committed before adopting."""
        with self._tlock:
            rec = self._shared.get(key)
            if rec is None:
                return None
            blob = self._shared_blobs.get(key)
            tag = rec[2]
            if blob is None and tag is not None:
                bucket = self._shared_ssds.get(tag)
                if bucket is not None and ("s", key) in bucket:
                    blob = bucket.read(("s", key), torch.cuda.is_available())
                    self._shared_blobs[key] = blob
                    self._shared_ram += rec[0]
                    bucket.forget(("s", key))
                    self._shared_ssd_bytes -= rec[0]
                    self._ram_order[("s", key)] = rec[0]
                    self._shared.move_to_end(key)
                    self._enforce_budget()
            return blob

    def share_take_field(self, key: int, field: str):
        """One named tensor of a shared page blob (``bounds``/``dk``),
        read-through from the record's signature bucket when the blob is
        spilled, without loading its K/V. A field this record's layout does not
        own is a clean None (cache miss), never a KeyError. Non-blocking: None
        while a background publish is still queued."""
        with self._tlock:
            rec = self._shared.get(key)
            if rec is None:
                return None
            blob = self._shared_blobs.get(key)
            if blob is not None:
                return blob.get(field)
            tag = rec[2]
            if tag is not None:
                bucket = self._shared_ssds.get(tag)
                if bucket is not None:
                    return bucket.read_field(("s", key), field)
            return None

    def share_release(self, key: int) -> None:
        """Drop one store reference; the blob is deleted/spilled-slot freed at
        the last reference. A release landing while the key's publish is still
        queued accumulates as a signed delta the worker folds into refcount 1."""
        with self._tlock:
            if key in self._pub_pending:
                self._pub_pending_refs[key] -= 1
                return
            rec = self._shared.get(key)
            if rec is None:
                return
            rec[1] -= 1
            if rec[1] > 0:
                return
            n, _refs, tag = self._shared.pop(key)
            self._ram_order.pop(("s", key), None)
            if self._shared_blobs.pop(key, None) is not None:
                self._shared_ram -= n
            if tag is not None:
                bucket = self._shared_ssds.get(tag)
                if bucket is not None and ("s", key) in bucket:
                    bucket.forget(("s", key))
                    self._shared_ssd_bytes -= n

    def share_ref(self, key: int) -> None:
        """Add one store reference to an already-shared key (a frozen prefix
        copy shares the blob of the entry it was snapshotted from). No-op when
        the key has already been released (evicted between freeze and ref).
        Pending (background publish queued, not yet committed): accumulate."""
        with self._tlock:
            if key in self._pub_pending:
                self._pub_pending_refs[key] += 1
                return
            rec = self._shared.get(key)
            if rec is not None:
                rec[1] += 1

    def share_keys(self) -> frozenset[int]:
        with self._tlock:
            return frozenset(self._shared) | frozenset(self._pub_pending)

    def drain_ssd_ms(self) -> float:
        """Milliseconds spent touching the mmap'd spill files since the last
        drain, across the private and shared files. The engine drains this into
        the step timer at the end of the tick that paid it."""
        with self._tlock:
            ms = 0.0
            files = [self._ssd, *self._shared_ssds.values()]
            for f in files:
                if f is not None:
                    ms += f.ssd_ms
                    f.ssd_ms = 0.0
            return ms

    def shared_bytes(self) -> int:
        """Pinned RAM held for shared prefix blobs (test/ledger diagnostic)."""
        with self._tlock:
            return self._shared_ram


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


def _crc32(data: bytes) -> int:
    import zlib

    return zlib.crc32(data) & 0xFFFFFFFF


class KvBootStore:
    """Cold-start KV store on a local filesystem: a fully prefilled context saved
    once is reloaded into fresh blocks on a later `serve --kv-store DIR` whose
    request prefix matches, skipping the (2.25 h at 256k on a V100) prefill.

    It is the third source a promoted page can come from — host RAM
    (:class:`HostKvPages`) and the sparse cold tier's SSD spill
    (:class:`ColdSsdFile`) are the other two — but unlike the cold spill a boot
    entry's pages are written for the
    WHOLE context deliberately, in one fixed-stride file per tensor so a page is
    one ``pread``. Layout under ``<dir>/<hash>/`` (≤5 files):

      * ``manifest.json`` — tokens, plane/block/token/head shape, dtypes, fp8 flag,
        and a per-page CRC32 over K+V+scales. A hash collision is rejected by the
        token list; a flipped page is rejected by its CRC.
      * ``k.bin`` / ``v.bin`` — all planes' pages, fixed ``page_bytes`` stride,
        stored in the cold dtype (f16 on sm70).
      * ``scale.bin`` — fp8 k/v per-token scales (f32); absent off fp8.
      * ``aux.pt`` — the recurrent GDN snapshot (states + conv window + parity)
        at the prefix boundary; the attention pages without it run the GDN layers
        from a zero state over nonzero KV, silently wrong.

    This rung bulk-loads the whole context into HBM at admit, which already skips
    the prefill — the point. Per-page lazy SSD promote and the eager Quest-bounds
    load (the selector needs them without recomputing over resident K) are the
    sparse follow-up.
    """

    AUX = "aux.pt"
    MANIFEST = "manifest.json"

    def __init__(self, path: str, fingerprint: str) -> None:
        import json

        self._json = json
        self._fingerprint = fingerprint
        # Own a subdir: never write directly into a caller's directory.
        self._root = os.path.join(os.fspath(path), "tilerl_kvboot")
        self._marker = os.path.join(self._root, ".kvboot")
        if os.path.exists(self._root) and not os.path.exists(self._marker):
            raise RuntimeError(f"{self._root} exists but is not a KvBootStore dir")
        os.makedirs(self._root, exist_ok=True)
        with open(self._marker, "w") as f:
            f.write(fingerprint)

    # ----------------------------------------------------------------- key / layout
    def _entry_dir(self, h: int) -> str:
        return os.path.join(self._root, f"{h & _MASK64:016x}")

    @staticmethod
    def hash_tokens(tokens: Sequence[int]) -> int:
        h = 0
        for t in tokens:
            h = _rolling_hash(h, int(t))
        return h

    def exists(self, tokens: Sequence[int]) -> bool:
        h = self.hash_tokens(tokens)
        mf = os.path.join(self._entry_dir(h), self.MANIFEST)
        if not os.path.exists(mf):
            return False
        try:
            m = self._read_manifest(h)
            return (
                m["tokens"] == [int(t) for t in tokens]
                and m.get("fingerprint") == self._fingerprint
            )
        except (OSError, ValueError, KeyError):
            return False

    def _read_manifest(self, h: int) -> dict:
        with open(os.path.join(self._entry_dir(h), self.MANIFEST)) as f:
            return self._json.load(f)

    def bytes_total(self) -> int:
        """On-disk bytes across every saved entry's K/V/scales/aux (manifest excluded).
        The ledger's kv_cold(ssd) row is priced against this."""
        total = 0
        if not os.path.isdir(self._root):
            return 0
        for name in os.listdir(self._root):
            p = os.path.join(self._root, name)
            if os.path.isdir(p):
                total += sum(
                    os.path.getsize(os.path.join(p, f)) for f in os.listdir(p) if f != self.MANIFEST
                )
        return total

    def entries(self) -> int:
        return (
            sum(1 for n in os.listdir(self._root) if os.path.isdir(os.path.join(self._root, n)))
            if os.path.isdir(self._root)
            else 0
        )

    @staticmethod
    def _dt(name: str):
        return getattr(torch, name)

    # ------------------------------------------------------------------ save
    def save(
        self, tokens: Sequence[int], pool: PagedKvPool, blocks: Sequence[int], state: Any
    ) -> int:
        """Write one full context (the pages named by ``blocks`` in sequence order) and
        its recurrent snapshot. Pages are gathered to the host in the pool's cold dtype.
        Returns bytes written. Atomic: the manifest is written last, so a crash leaves
        no entry ``exists`` returns True for."""
        import tempfile

        tokens = [int(t) for t in tokens]
        h = self.hash_tokens(tokens)
        d = self._entry_dir(h)
        os.makedirs(d, exist_ok=True)
        cold = pool.cold_dtype
        # Boot save is a deliberate one-time offline write, never on a decode tick.
        k = torch.stack([pool.k_pool[:, b] for b in blocks]).cpu()
        v = torch.stack([pool.v_pool[:, b] for b in blocks]).cpu()
        if cold is not None:
            k, v = k.to(cold), v.to(cold)
        k, v = k.contiguous(), v.contiguous()
        nblk, nplanes = k.shape[0], k.shape[1]

        def raw(t: torch.Tensor) -> bytes:
            return t.view(torch.uint8).numpy().tobytes()

        kb, vb = raw(k), raw(v)
        kstep, vstep = len(kb) // nblk, len(vb) // nblk
        scale_bytes = b""
        sstep = 0
        if pool.k_scale is not None:
            ks = torch.stack([pool.k_scale[:, b] for b in blocks]).cpu().contiguous()
            vs = torch.stack([pool.v_scale[:, b] for b in blocks]).cpu().contiguous()
            ksb, vsb = raw(ks), raw(vs)
            ssk, ssv = len(ksb) // nblk, len(vsb) // nblk
            # interleave per page so one page's CRC/checksum covers its K, V, k_scale
            # AND v_scale contiguously.
            scale_bytes = b"".join(
                ksb[i * ssk : (i + 1) * ssk] + vsb[i * ssv : (i + 1) * ssv] for i in range(nblk)
            )
            sstep = ssk + ssv
        written = 0
        tmp = tempfile.mkdtemp(prefix=".kvboot-", dir=d)
        try:
            with open(os.path.join(tmp, "k.bin"), "wb") as f:
                f.write(kb)
                written += len(kb)
            with open(os.path.join(tmp, "v.bin"), "wb") as f:
                f.write(vb)
                written += len(vb)
            if scale_bytes:
                with open(os.path.join(tmp, "scale.bin"), "wb") as f:
                    f.write(scale_bytes)
                    written += len(scale_bytes)
            crcs = [
                _crc32(
                    kb[i * kstep : (i + 1) * kstep]
                    + vb[i * vstep : (i + 1) * vstep]
                    + (scale_bytes[i * sstep : (i + 1) * sstep] if sstep else b"")
                )
                for i in range(nblk)
            ]
            if state is not None:
                torch.save(state, os.path.join(tmp, self.AUX))
                written += os.path.getsize(os.path.join(tmp, self.AUX))
            manifest = {
                "tokens": tokens,
                "fingerprint": self._fingerprint,
                "n_blocks": nblk,
                "n_planes": nplanes,
                "n_kv_heads": pool.num_kv_heads,
                "head_dim": pool.head_dim,
                "block_tokens": BLOCK_TOKENS,
                "k_dtype": str(k.dtype).replace("torch.", ""),
                "v_dtype": str(v.dtype).replace("torch.", ""),
                "kv_fp8": str(pool.kv_fp8).replace("torch.", "") if pool.kv_fp8 else None,
                "shape": list(k.shape),  # [blocks, planes, heads, tokens, head_dim]
                "page_kv_bytes": kstep,  # K bytes of one page (V equal)
                "page_crc32": crcs,
                "has_state": state is not None,
            }
            with open(os.path.join(tmp, self.MANIFEST), "w") as f:
                self._json.dump(manifest, f)
            # Move data files first, the manifest last: a crash between leaves a
            # half-written dir that no `exists` (which reads the manifest) adopts.
            for fn in sorted(os.listdir(tmp)):
                if fn != self.MANIFEST:
                    os.replace(os.path.join(tmp, fn), os.path.join(d, fn))
            os.replace(os.path.join(tmp, self.MANIFEST), os.path.join(d, self.MANIFEST))
        finally:
            with contextlib.suppress(OSError):
                os.rmdir(tmp)
        return written

    # ------------------------------------------------------------------ load
    def boot(self, tokens: Sequence[int], pool: PagedKvPool) -> dict | None:
        """Load a matching full context into FRESH device blocks, restoring pages through
        the same promote widening (cold dtype -> pool dtype). Returns
        ``{blocks, state, length}`` or None on no-match/hash-mismatch/CRC failure.
        ``state`` is None when the entry was saved without one; ``length`` is the
        block-aligned token count. The caller splices ``blocks`` into the request,
        copies ``state`` into its slot, and sets its materialized length."""
        tokens = [int(t) for t in tokens]
        h = self.hash_tokens(tokens)
        d = self._entry_dir(h)
        try:
            mf = self._read_manifest(h)
        except (OSError, ValueError, KeyError):
            return None
        if mf["tokens"] != tokens:
            return None  # hash collision: different prefix
        try:
            nblk = mf["n_blocks"]
            shape = tuple(mf["shape"])

            def read_raw(name):
                with open(os.path.join(d, name), "rb") as f:
                    return f.read()

            kb, vb = read_raw("k.bin"), read_raw("v.bin")
            kstep = mf["page_kv_bytes"]
            vstep = len(vb) // nblk
            sb = read_raw("scale.bin") if mf.get("kv_fp8") else b""
            sstep = len(sb) // nblk if nblk else 0
            for i, want in enumerate(mf["page_crc32"]):
                seg = (
                    kb[i * kstep : (i + 1) * kstep]
                    + vb[i * vstep : (i + 1) * vstep]
                    + (sb[i * sstep : (i + 1) * sstep] if sstep else b"")
                )
                if _crc32(seg) != want:
                    raise ValueError(f"page {i} checksum mismatch")

            def as_tensor(rawb, shape_, dtype):
                u = torch.frombuffer(bytearray(rawb), dtype=torch.uint8)
                return u.view(dtype).reshape(shape_)

            k = as_tensor(kb, shape, self._dt(mf["k_dtype"]))
            v = as_tensor(vb, shape, self._dt(mf["v_dtype"]))
            scales = None
            if sb:
                nh = mf["n_kv_heads"]
                npl = mf["n_planes"]
                # one scale page = [plane, head, token] for k then v, f32
                per = npl * nh * BLOCK_TOKENS * 4
                ksf = b"".join(sb[i * sstep : i * sstep + per] for i in range(nblk))
                vsf = b"".join(sb[i * sstep + per : (i + 1) * sstep] for i in range(nblk))
                sshape = (nblk, npl, nh, BLOCK_TOKENS)
                ks = as_tensor(ksf, sshape, torch.float32)
                vs = as_tensor(vsf, sshape, torch.float32)
                scales = (ks, vs)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            # The store stays loud about corrupt on-disk state; the ENGINE's
            # admit path narrows these into a cache miss. Raising here keeps a
            # direct store user (and tests) able to tell corruption from a miss.
            raise RuntimeError(f"corrupt or unreadable boot entry {h:016x}: {exc}") from exc

        out_blocks: list[int] = []
        try:
            for _ in range(nblk):
                out_blocks.append(pool.alloc_block())
            idx = torch.as_tensor(out_blocks, device=pool.device)
            dev = pool.device
            nb = pool.k_pool.is_cuda

            # block-major [B,plane,H,T,D] -> plane-major [plane,B,H,T,D], widening the
            # narrow cold dtype to the pool dtype via copy_ (same as promote_page).
            def copy_pages(host, dst, fp8: bool):
                x = host.permute(1, 0, 2, 3, 4).to(dev, non_blocking=nb)
                if fp8:
                    dst[:, idx] = x  # no index_copy_ for fp8 on CPU
                else:
                    dst.index_copy_(1, idx, x.to(dst.dtype))

            copy_pages(k, pool.k_pool, pool.kv_fp8 is not None)
            copy_pages(v, pool.v_pool, pool.kv_fp8 is not None)
            if scales is not None:
                ks, vs = scales
                pool.k_scale.index_copy_(1, idx, ks.permute(1, 0, 2, 3).to(dev, non_blocking=nb))
                pool.v_scale.index_copy_(1, idx, vs.permute(1, 0, 2, 3).to(dev, non_blocking=nb))
            if pool.k_pool.is_cuda:
                torch.cuda.synchronize(pool.device)
            state = None
            if mf.get("has_state"):
                # Inside the cleanup block: aux.pt loads AFTER the page copies, so
                # a torch.load failure must free the freshly allocated blocks
                # (F19), then re-raise for the engine's admit path to miss.
                state = torch.load(os.path.join(d, self.AUX), map_location="cpu")
        except (OSError, RuntimeError, pickle.UnpicklingError, EOFError):
            for b in out_blocks:
                pool.free_block(b)
            raise
        return {"blocks": out_blocks, "state": state, "length": len(tokens)}
