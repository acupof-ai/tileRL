"""Sweep: does the prefetch deadline margin grow with prompt length on CPU?

The flake in test_a_prefetched_hit_reads_nothing_on_the_calling_thread:
fetch ~1.2s (GIL-window-bound, ~14 windows) vs deadline = tokens/seed_rate.
27's tick-count model: fetch ticks ~constant, deadline ticks ~ tokens, so
margin should grow with length. Measure 192/384/768 to check.

Prints per point: total tokens, deadline_ms, fetch_ms, ssd_hits.
"""

from __future__ import annotations

import tempfile
import time

import numpy as np
from tilerl_kernels.backend import get_backend

from tilerl.config import tiny
from tilerl.engine import BLOCK_TOKENS, SamplingParams, build_engine
from tilerl.model import build_random


def run_point(warm_len: int, other_len: int) -> None:
    cfg = tiny()
    params = SamplingParams(temperature=0.0, max_new_tokens=2, seed=3)
    rng = np.random.default_rng(11)
    conv = rng.integers(3, 320, size=warm_len + other_len).astype(np.int64)
    warm, other = conv[:warm_len], rng.integers(3, 320, size=other_len).astype(np.int64)

    with tempfile.TemporaryDirectory() as tmp:
        def engine_at():
            return build_engine(
                cfg, build_random(cfg, seed=13), get_backend(), num_blocks=256,
                num_slots=4, max_batch=4, max_total_tokens=8192,
                ssd_path=tmp, ssd_min_tokens=BLOCK_TOKENS,
            )

        warm_eng = engine_at()
        warm_eng.submit(warm, params)
        end = time.time() + 30
        while time.time() < end:
            warm_eng.step()
            if not (list(warm_eng._running) + list(warm_eng._waiting)):
                break
        for _ in range(400):
            if warm_eng.stats()["ssd_entries"] >= 1:
                break
            time.sleep(0.01)

        cold = engine_at()
        rate = cold.prefill_rate
        total = len(warm) + len(other)
        deadline_ms = total / rate * 1000

        cold.submit(list(warm) + list(other), params)
        # Step until the fetch completes (or 30s), then read the counters.
        end = time.time() + 30
        while time.time() < end:
            cold.step()
            st = cold.stats()
            if st["ssd_fetches_ready"] >= 1 or st["ssd_fetch_drops"] >= 1:
                break
        # keep stepping a bit so the hit can land
        for _ in range(200):
            cold.step()
            if not (list(cold._running) + list(cold._waiting)):
                break
        st = cold.stats()
        print(
            f"warm={warm_len} other={other_len} total={total} "
            f"deadline_ms={deadline_ms:.0f} fetch_ms={st['ssd_fetch_ms']} "
            f"hits={st['ssd_hits']} drops={st['ssd_fetch_drops']} "
            f"ready={st['ssd_fetches_ready']}",
            flush=True,
        )


if __name__ == "__main__":
    for warm_len, other_len in [(128, 64), (256, 128), (512, 256)]:
        run_point(warm_len, other_len)
