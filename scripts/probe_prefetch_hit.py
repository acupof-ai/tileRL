"""Probe: why does test_a_prefetched_hit_reads_nothing_on_the_calling_thread get
ssd_recovered=1, ssd_hits=0 on CUDA?

Replicates the test scenario and prints the three numbers 27 asked for:
  1. prefetch trigger count (ssd_prefetches)
  2. prefetch completion count (ssd_fetches_ready)
  3. hit count (ssd_hits)
plus the supporting values that decide them.
"""

from __future__ import annotations

import tempfile
import time

import numpy as np
from tilerl_kernels.backend import get_backend

from tilerl.config import tiny
from tilerl.engine import BLOCK_TOKENS, SamplingParams, build_engine
from tilerl.model import build_random


def drain(eng, secs=10.0):
    end = time.time() + secs
    while time.time() < end:
        eng.step()
        if not (list(eng._running) + list(eng._waiting)):
            break
    eng.poll()


def main():
    cfg = tiny()
    params = SamplingParams(temperature=0.0, max_new_tokens=2, seed=3)
    rng = np.random.default_rng(11)
    conv = rng.integers(3, 320, size=256).astype(np.int64)
    warm, other = conv[:128], rng.integers(3, 320, size=64).astype(np.int64)

    with tempfile.TemporaryDirectory() as tmp:
        def engine_at():
            return build_engine(
                cfg, build_random(cfg, seed=13), get_backend(), num_blocks=64,
                num_slots=4, max_batch=4, max_total_tokens=2048,
                ssd_path=tmp, ssd_min_tokens=BLOCK_TOKENS,
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
        print(f"warm: ssd_entries={wst['ssd_entries']} ssd_offered={wst['ssd_offered']}")

        # --- cold arm: recover, then submit the extended prompt ---
        cold = engine_at()
        cst0 = cold.stats()
        print(f"cold(recovered): ssd_recovered={cst0['ssd_recovered']} "
              f"ssd_entries={cst0['ssd_entries']}")

        rate = cold.prefill_rate
        be = cold._prefix.break_even_tokens(rate)
        b = cold._prefix._ssd.read_bytes_per_s()
        snap = cold._prefix._ssd.snapshot_bytes
        k = cold._prefix._pool.bytes_per_token
        print(f"cold(before submit): prefill_rate={rate:.1f} break_even={be} "
              f"read_bps={b:.1f} snapshot_bytes={snap} bytes_per_token={k}")

        t0 = time.perf_counter()
        cold.submit(list(warm) + list(other), params)
        t_submit = time.perf_counter() - t0

        # right after submit: did prefetch queue?
        cst1 = cold.stats()
        inflight = cold._prefix.fetch_in_flight(list(warm) + list(other))
        print(f"cold(after submit): ssd_prefetches={cst1['ssd_prefetches']} "
              f"fetch_in_flight={inflight} submit_latency={t_submit*1000:.1f}ms")

        # check the deadline
        req = cold._waiting[0] if cold._waiting else None
        if req is not None:
            dl = req.fetch_deadline
            now = time.perf_counter()
            print(f"  req.fetch_deadline={dl:.4f} now={now:.4f} "
                  f"remaining={(dl - now)*1000:.1f}ms" if dl else "  req.fetch_deadline=0")

        drain(cold)

        cst2 = cold.stats()
        print(f"cold(after drain): ssd_prefetches={cst2['ssd_prefetches']} "
              f"ssd_fetches_ready={cst2['ssd_fetches_ready']} "
              f"ssd_hits={cst2['ssd_hits']} "
              f"ssd_fetch_waits={cst2['ssd_fetch_waits']} "
              f"ssd_fetch_drops={cst2['ssd_fetch_drops']} "
              f"ssd_recovered={cst2['ssd_recovered']}")
        print(f"  fetch_ms={cst2['ssd_fetch_ms']} fetch_bytes={cst2['ssd_fetch_bytes']} "
              f"tick_loads={cst2['ssd_tick_loads']}")


if __name__ == "__main__":
    main()
