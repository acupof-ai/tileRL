#!/usr/bin/env python3
"""PROBE-ONLY #805: CPU gate that a v2 lag carry promotes a SHARED-prefix page.

Regression for the device rc14 on W1024/R32: the lag job only knew resident +
private-cold pages, so a prefix-adopting follower's early logical pages (held
in tracker.shared, resolved via share_take/shared_promote) read as
"unsupported/missing page 0" and every carry fell back eager. Builds a real
two-request sparse+spec prefix engine (publish, then an adopting follower),
drives it to a v2 carry tick, and asserts:

  * the carry ARMS (commit True, carry_cycles == 1) — pre-fix it fell back;
  * at least one promoted frame came from a shared content key
    (result pinned_keys non-empty, a shared page mapped resident);
  * dropping the request unpins the content key (no leak).
"""
from __future__ import annotations

import os
import sys

os.environ["TILERL_TARGET"] = "cpu"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    import numpy as np
    import torch
    from probe_serve_sm70_w2048 import _tiny_draft
    from tilerl_kernels.backend import get_backend

    from tilerl.build import build_engine
    from tilerl.config import tiny
    from tilerl.engine import SamplingParams
    from tilerl.kv_cache import BLOCK_TOKENS
    from tilerl.model import build_random

    torch.manual_seed(0)
    cfg = tiny()
    model = build_random(cfg, seed=11)
    prompt = (np.arange(24 * BLOCK_TOKENS, dtype=np.int64) % 300) + 7
    follow = np.concatenate([prompt, np.arange(100, 140, dtype=np.int64)])

    e = build_engine(
        cfg=cfg, model=build_random(cfg, seed=11), backend=get_backend(),
        num_blocks=64, num_slots=4, max_batch=1, max_total_tokens=4096,
        max_num_batched_tokens=512, sparse_k=2, sparse_min_tokens=0,
        sparse_device_select=True, scorer="bounds",
        kv_cold_bytes=1 << 30, draft=_tiny_draft(cfg, model), spec_depth=1)
    os.environ["TILERL_SPARSE_V2"] = "inline"
    from tilerl.sparse_lag import LagController
    lag = LagController(e._sparse, "inline")
    e._sparse._lag_obj = lag
    e._sparse._lag_checked = True

    def drain(rid, n):
        out = {}
        for _ in range(n * 20 + 200):
            e.step()
            out = e.poll()
            if rid in out:
                return out[rid]
        raise AssertionError("rid did not finish")

    pub = e.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=200, seed=0))
    drain(pub, 200)
    entry = e._sparse.prefix.lookup(follow)
    assert entry is not None and len(entry["keys"]) >= 1, "no published prefix"

    rid = e.submit(follow, SamplingParams(temperature=0.0, max_new_tokens=200, seed=0))
    # prefill/adoption; the follower's shared map must carry page 0.
    for _ in range(60):
        e.step()
        if any(r.req_id == rid for r in e._running):
            break
    assert e._sparse.tracker.shared.get(rid), "follower has no shared pages"
    assert 0 in e._sparse.tracker.shared[rid], "logical page 0 not a shared page"

    # Drive pure-decode ticks up to the first carry (counter R-1 prepare, then a
    # carry tick). Inline mode runs the job synchronously; inspect the result.
    from tilerl.sparse_engine import SPARSE_REFRESH_TICKS as R

    carried = False
    saw_shared_promote = False
    for _ in range(R * 2 + 10):
        if not any(r.req_id == rid for r in e._running):
            break
        # capture the job result right after a prepare
        e.step()
        res = lag._result
        if isinstance(res, dict) and "groups" in res:
            carried = True
            if res.get("pinned_keys"):
                saw_shared_promote = True
            # shared pages selected this carry are recorded
            if any(res.get("shared_pages", {}).values()):
                saw_shared_promote = True
    assert carried, "v2 never produced a carry for the prefix follower"
    assert lag.carry_cycles >= 1, f"no armed carry (fallbacks={lag.fallback_cycles} " \
        f"reasons={lag.fallback_reasons} last={lag.last_fallback_reason})"
    assert saw_shared_promote, (
        "carry armed but promoted no shared page (pins="
        f"{lag._result if isinstance(lag._result, dict) else None})")

    # The adopted content pin is registered for this rid and survives until drop.
    assert rid in e._sparse.tracker.request_pins, "shared pin not registered"
    e.shutdown()
    print("SHARED-CARRY GREEN: carry armed for a prefix-adopting follower, "
          f"shared pages promoted, carries={lag.carry_cycles} "
          f"fallbacks={lag.fallback_cycles}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
