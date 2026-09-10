"""Probe: the spin-until-ready loop's cost on the full 27B with SSD on.

OPEN.md row 17's open remainder: the slice bench (wins/2026-09-10-spin-until-ready)
measured -0.1%/-0.2%, but the slice's .st is 9.8 MB and no fetch was in flight
during decode, so the spin never ran -- those numbers prove "spin costs nothing
when not running", not "spin is cheap on 27B". The 27B snapshot is 155.2 MiB
(states 144.0 + conv_window 11.25), so a fetch spans many GIL windows and the
spin actually runs, approaching the 50 ms bound per tick.

This probe runs the warm/cold prefetch scenario on the real 27B and times each
step() bucketed by any_fetching() state, so the spin's cost is the tick-time
difference between the two buckets. Prints: arch + seed_rate (the deadline's
inputs), tick-time distribution by bucket, whether the 50 ms bound fired, the
fetch outcome (ready/drops/fetch_ms), and decode tok/s.

Usage (pod): uv run python scripts/probe_spin_cost_27b.py
Defaults to /work/Qwen3.8-27B-NVFP4; override with TILERL_27B_CKPT.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time

import numpy as np
from tilerl_kernels.backend import get_backend

from tilerl import kv_cache as kvmod
from tilerl.config import qwen38_27b
from tilerl.engine import BLOCK_TOKENS, SamplingParams, build_engine
from tilerl.model import load_hf

CKPT = os.environ.get("TILERL_27B_CKPT", "/work/Qwen3.8-27B-NVFP4")
SPIN_BOUND_MS = 50.0


def drain(eng, secs=120.0):
    end = time.time() + secs
    while time.time() < end:
        eng.step()
        if not (list(eng._running) + list(eng._waiting)):
            break
    eng.poll()


def main():
    be = get_backend()
    assert be.device.type == "cuda", be.device
    arch = getattr(be, "arch", "")
    print(f"device={be.device} arch={arch}", flush=True)

    cfg = qwen38_27b()
    model = load_hf(cfg, CKPT, keep_master=False)
    model.params = be.materialize(model.params)
    print(f"27B loaded from {CKPT}", flush=True)

    params = SamplingParams(temperature=0.0, max_new_tokens=32, seed=3)
    rng = np.random.default_rng(11)
    conv = rng.integers(3, 320, size=256).astype(np.int64)
    warm, other = conv[:128], rng.integers(3, 320, size=64).astype(np.int64)

    with tempfile.TemporaryDirectory() as tmp:
        def engine_at():
            return build_engine(
                cfg, model, be, num_blocks=64, num_slots=4, max_batch=4,
                max_total_tokens=2048, ssd_path=tmp, ssd_min_tokens=BLOCK_TOKENS,
            )

        # --- warm arm: spill to SSD ---
        warm_eng = engine_at()
        warm_eng.submit(warm, params)
        drain(warm_eng)
        for _ in range(400):
            if warm_eng.stats()["ssd_entries"] >= 1:
                break
            time.sleep(0.01)
        wst = warm_eng.stats()
        print(f"warm: ssd_entries={wst['ssd_entries']} ssd_offered={wst['ssd_offered']} "
              f"ssd_bytes={wst['ssd_bytes']}", flush=True)

        # --- cold arm: recover, submit extended prompt, time each tick ---
        cold = engine_at()
        cst0 = cold.stats()
        rate = cold.prefill_rate
        deadline_ms = len(list(warm) + list(other)) / rate * 1000
        snap_mib = cold._prefix._ssd.snapshot_bytes / (1024 * 1024)
        print(f"cold(recovered): ssd_recovered={cst0['ssd_recovered']} "
              f"entries={cst0['ssd_entries']} seed_rate={rate:.1f} "
              f"deadline={deadline_ms:.1f}ms snapshot={snap_mib:.1f}MiB",
              flush=True)

        cold.submit(list(warm) + list(other), params)

        # Time each step, bucketed by any_fetching() sampled right before it.
        spin_ticks, quiet_ticks = [], []
        bound_fires = 0
        for _ in range(2000):
            fetching = cold._prefix.any_fetching()
            t0 = time.perf_counter()
            cold.step()
            dt = (time.perf_counter() - t0) * 1000
            if fetching:
                spin_ticks.append(dt)
                if dt >= SPIN_BOUND_MS * 0.98:
                    bound_fires += 1
            else:
                quiet_ticks.append(dt)
            if not (list(cold._running) + list(cold._waiting)):
                break
        cold.poll()

        def dist(name, xs):
            if not xs:
                print(f"  {name}: (none)", flush=True)
                return
            xs = sorted(xs)
            n = len(xs)
            print(f"  {name}: n={n} min={xs[0]:.2f} med={xs[n // 2]:.2f} "
                  f"p90={xs[int(n * 0.9)]:.2f} max={xs[-1]:.2f} ms", flush=True)

        print("tick wall time by bucket:", flush=True)
        dist("fetching (spin runs)", spin_ticks)
        dist("quiet (no spin)     ", quiet_ticks)
        print(f"50ms bound fired on {bound_fires}/{len(spin_ticks)} spin ticks", flush=True)

        cst1 = cold.stats()
        print(f"cold(after request 1): ssd_prefetches={cst1['ssd_prefetches']} "
              f"fetches_ready={cst1['ssd_fetches_ready']} "
              f"fetch_drops={cst1['ssd_fetch_drops']} "
              f"fetch_ms={cst1['ssd_fetch_ms']} fetch_bytes={cst1['ssd_fetch_bytes']} "
              f"tick_loads={cst1['ssd_tick_loads']} hits={cst1['ssd_hits']}", flush=True)

        # --- request 2: the same prefix, with its HBM entry evicted so ONLY the parked
        # SSD pair can serve it. This is the fix's payoff: a fetch whose requester missed
        # its deadline finished in the background; the next same-prefix request faults it
        # in from the parked host copy (zero tick-side disk read), instead of paying the
        # 117 ms read again -- which the pre-fix code discarded, so there was nothing to take.
        key = cold._prefix._hash_all(list(warm))
        n_dropped_hbm = 0
        for e in list(cold._prefix._entries.get(key, ())):
            cold._prefix._drop(e)
            n_dropped_hbm += 1
        print(f"dropped {n_dropped_hbm} resident HBM entry at the matched boundary",
              flush=True)

        tick_loads = {"n": 0}
        real_load = kvmod.torch.load

        def counting(*a, **k):
            if threading.current_thread() is threading.main_thread():
                tick_loads["n"] += 1
            return real_load(*a, **k)

        hits_before = cold.stats()["ssd_hits"]
        kvmod.torch.load = counting
        t2 = time.perf_counter()
        cold.submit(list(warm), params)
        drain(cold)
        req2_wall = time.perf_counter() - t2
        kvmod.torch.load = real_load

        cst2 = cold.stats()
        print(f"cold(request 2 same-prefix): ssd_hits={cst2['ssd_hits']} "
              f"(+{cst2['ssd_hits'] - hits_before}) "
              f"tick_loads={tick_loads['n']} wall={req2_wall:.2f}s "
              f"prefill_from reuse expected: matched via parked pair", flush=True)
        print("VERDICT:", end=" ", flush=True)
        if cst2["ssd_hits"] > hits_before and tick_loads["n"] == 0:
            print("PARKED PAIR SERVED (fetches_ready>0, 0 tick reads)", flush=True)
        else:
            print("NOT SERVED FROM PARK -- re-read or recomputed", flush=True)


if __name__ == "__main__":
    main()
