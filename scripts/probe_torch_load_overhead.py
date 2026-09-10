"""Probe: torch.load takes 0.4ms in the main thread but 70ms in the fetch loop.
Hypothesis: GIL contention — the reader thread can't get the GIL while the
main thread is in a tight step() loop.

Measurements:
1. torch.load in main thread (baseline)
2. torch.load in background thread, main thread idle
3. torch.load in background thread, main thread in busy loop (reproduces the test)
4. Zipfile entry names for torch 2.11
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import zipfile

import torch


def time_fn(fn, n=5):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[0], times[len(times) // 2], times


def main():
    print(f"torch={torch.__version__}")

    with tempfile.TemporaryDirectory() as tmp:
        # Create files matching the SSD tier's shapes
        kv_blob = {
            "states": torch.randn(2, 4, 8, 8, 8),
            "windows": torch.randn(2, 4, 8, 8),
        }
        st_tensor = torch.randn(4, 8, 8, 8)
        kv_path = os.path.join(tmp, "test.kv")
        st_path = os.path.join(tmp, "test.st")
        torch.save(kv_blob, kv_path)
        torch.save(st_tensor, st_path)
        print(f"kv={os.path.getsize(kv_path)}B st={os.path.getsize(st_path)}B")

        # List zipfile entries (torch 2.11 may not use "data.pkl")
        with zipfile.ZipFile(kv_path) as zf:
            print(f"zip entries: {zf.namelist()}")

        # Warmup
        _ = torch.load(kv_path, map_location="cpu")

        # 1. Main thread baseline
        def load_both():
            torch.load(kv_path, map_location="cpu")
            torch.load(st_path, map_location="cpu")

        mn, md, all_t = time_fn(load_both, 5)
        print(f"\n1. main thread:        min={mn:6.1f}ms  median={md:6.1f}ms  "
              f"all=[{', '.join(f'{t:.1f}' for t in all_t)}]")

        # 2. Background thread, main thread idle
        results = []
        def bg_load():
            t0 = time.perf_counter()
            load_both()
            results.append((time.perf_counter() - t0) * 1000)

        times_bg = []
        for _ in range(5):
            results.clear()
            t = threading.Thread(target=bg_load)
            t.start()
            t.join()
            times_bg.append(results[0])
        times_bg.sort()
        print(f"2. bg thread (idle):   min={times_bg[0]:6.1f}ms  median={times_bg[2]:6.1f}ms  "
              f"all=[{', '.join(f'{t:.1f}' for t in times_bg)}]")

        # 3. Background thread, main thread in busy loop (reproduces the test)
        times_busy = []
        for _ in range(5):
            results.clear()
            t = threading.Thread(target=bg_load)
            t.start()
            # Busy loop: the main thread holds the GIL, only releasing on the
            # sys.setswitchinterval timeout (default 5ms in CPython 3.11+)
            end = time.perf_counter() + 2.0  # 2s cap
            x = 0
            while time.perf_counter() < end and t.is_alive():
                x += 1  # pure Python, holds GIL
            t.join()
            times_busy.append(results[0])
        times_busy.sort()
        print(f"3. bg thread (busy):   min={times_busy[0]:6.1f}ms  median={times_busy[2]:6.1f}ms  "
              f"all=[{', '.join(f'{t:.1f}' for t in times_busy)}]")

        # 4. Same as 3 but with a sleep(0) in the busy loop (yields GIL)
        times_yield = []
        for _ in range(5):
            results.clear()
            t = threading.Thread(target=bg_load)
            t.start()
            end = time.perf_counter() + 2.0
            while time.perf_counter() < end and t.is_alive():
                time.sleep(0)  # explicitly yield GIL
            t.join()
            times_yield.append(results[0])
        times_yield.sort()
        print(f"4. bg thread (yield):  min={times_yield[0]:6.1f}ms  median={times_yield[2]:6.1f}ms  "
              f"all=[{', '.join(f'{t:.1f}' for t in times_yield)}]")


if __name__ == "__main__":
    main()
