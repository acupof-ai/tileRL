"""PROBE-ONLY (#805, branch probe/805-serve-sm70; never merge to main as-is).

v2 real 1-tick-delay sparse refresh — controller.

Cadence (SPARSE_REFRESH_TICKS=8): seven plain captured graph ticks, then one
CARRY tick. At the END of the 7th plain tick its post-rope q was cloned into
sf._q_prev during replay and its candidate staging still sits in sf.s_bounds /
cand_idx. prepare() runs the all-candidate refresh SELECTION with that q
(sf.select_refresh — the same quest_scores_batched/select_members/
order_members primitives as the captured path, no residency mask) and remembers
the LOGICAL picks. The carry tick's pre_replay hook then PROMOTES those pages
(via the real runtime.resolve, AFTER fill() resolved the own window so own
allocation cannot evict a just-picked page) and ARMS the physical blocks into
the captured sf, so the carry tick is a graph replay — never an eager trunk.

sf.refresh_tick makes finalize run the pin reconcile the ordinary captured
path skips.

Milestone 1 (this file): INLINE preparation on the step thread (no worker, no
side stream). It validates select+promote+override installation deterministically
on CPU; the async path must be token-identical to it. The async worker/side
stream is layered onto the SAME sf.select_refresh entry, so the installation
gate compares two drivers of one selection implementation, never a test-local
copy.

Env: TILERL_SPARSE_V2=inline (this milestone) | async (later) | unset/off.
"""

from __future__ import annotations

import os

import torch


def lag_mode() -> str | None:
    m = os.environ.get("TILERL_SPARSE_V2", "off")
    if m in ("", "off", "0"):
        return None
    if m not in ("inline", "async"):
        raise ValueError(f"TILERL_SPARSE_V2={m!r}; want off|inline|async")
    return m


class LagController:
    """Drives the per-cycle prepare/arm sequence for one SparseRuntime."""

    def __init__(self, runtime, mode: str):
        self.rt = runtime
        self.mode = mode
        # Logical picks from prepare() at the 7th plain tick:
        # {group: (chosen [B,k], nsel [B])}; consumed at the carry pre_replay.
        self._sel: dict | None = None
        self.fallback_cycles = 0
        self.carry_cycles = 0

    def is_carry(self) -> bool:
        from .sparse_engine import SPARSE_REFRESH_TICKS

        return self.rt.ticks_since_refresh + 1 >= SPARSE_REFRESH_TICKS

    def is_ready(self) -> bool:
        return self._sel is not None

    # -- 7th plain tick: selection only -----------------------------------
    def prepare(self, sf, srows) -> None:
        q_prev = sf._q_prev
        if not q_prev:
            self._sel = None  # no captured q yet: carry will fall back
            return
        if self.mode == "async":
            # The carry tick's fill() overwrites s_bounds/cand_idx before this
            # selection would otherwise read them. A real side stream must run
            # against a SNAPSHOT taken now (the post-replay fence point), so the
            # async driver selects on clones; inline reads live staging (still
            # valid because nothing fills between prepare and its select). The
            # two must agree — the snapshot is the only difference and the
            # installation gate proves it. q is already a per-replay clone.
            q_prev = {g: q.clone() for g, q in q_prev.items()}
            cand = sf.cand_idx.clone()
            n_cand = sf.n_cand.clone()
            win = sf.win.clone()
            bounds = [b.clone() for b in sf.s_bounds] if sf.s_bounds else None
            staging = (cand, n_cand, win, bounds)
        else:
            staging = None
        self._sel = sf.select_refresh(q_prev, staging=staging)

    # -- carry tick, pre_replay (after fill): promote + arm ----------------
    def arm_callback(self, sf):
        if self._sel is None:
            self.fallback_cycles += 1
            return
        per_group, use_rows = self._promote(sf, self._sel)
        sf.arm_override(per_group, use_rows)
        self._sel = None
        self.carry_cycles += 1

    def _promote(self, sf, sel):
        """Resolve each group's chosen LOGICAL pages to physical blocks through
        the real runtime.resolve, building fixed [B,k] phys/chosen/nsel
        staging. Runs after sf.fill(): add the picks to THIS tick's reserved
        set first, so a full-pool victim eviction cannot drop a page the carry
        replay is about to attend (mirrors eager _select's reserved.update)."""
        rt = self.rt
        B, k = sf.b, sf.tracker.k_pages
        dev = sf.device
        per_group = {}
        use = torch.zeros(B, dtype=torch.bool, device=dev)
        for bi in range(min(len(sf.rows), B)):
            use[bi] = True
        for g, (chosen_t, nsel_t) in sel.items():
            phys_t = torch.zeros(B, k, dtype=torch.long, device=dev)
            chosen_cpu = chosen_t.tolist()
            nsel_cpu = [int(x) for x in nsel_t.tolist()]
            for bi, rw in enumerate(sf.rows):
                r = rw["req"]
                reserved = rw["reserved"]
                pages = [p for p in chosen_cpu[bi][: nsel_cpu[bi]] if p != 0]
                reserved.update(pages)  # protect picks before any eviction
                blocks = [rt.resolve(r, p, reserved) for p in pages]
                for j, blk in enumerate(blocks):
                    phys_t[bi, j] = blk
            per_group[g] = (phys_t, chosen_t, nsel_t)
        return per_group, use
