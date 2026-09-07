"""Sample the engine's pool state on a fixed cadence, one JSONL row per sample.

Two reasons this is not a curl loop.

`free_blocks` is not a /health field: the endpoint exposes `pool_used_blocks` and
`blocks_total`, so the free count is derived here and the derivation is recorded rather
than done in someone's head later.

And /health takes the engine lock (`engine.py:731`), which `step()` holds across each
forward -- measured 6-18 s under load on the V100, and past 30 s with a full pool. So a
sample is a MEASUREMENT of that contention as well as a read of the pool: `secs` is kept
per row, and a sampler that blocks is reporting the defect, not failing.

Rows are appended as they are taken, so a kill cannot lose the series -- the same lesson
as the timing proxy, which lost a whole trial's rows by writing only at exit.

Usage:
    uv run python scripts/sample_pool.py --url http://10.37.2.27:8000 --every 20 \
        --out /tmp/pool.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import json
import signal
import time
import urllib.error
import urllib.request


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--every", type=float, default=20.0)
    ap.add_argument("--out", default="/tmp/pool.jsonl")
    #: Above the 6-18 s contention measured on the V100, so a slow sample is recorded as
    #: slow rather than dropped; a timeout row is still a data point about the lock.
    ap.add_argument("--timeout", type=float, default=120.0)
    a = ap.parse_args()

    open(a.out, "w").close()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    print(f"sampling {a.url}/health every {a.every}s -> {a.out}", flush=True)

    n = 0
    with contextlib.suppress(KeyboardInterrupt):
        while True:
            t0 = time.monotonic()
            # Absolute epoch, not just `n`: a row has to be alignable against another
            # instrument's rows, and a cadence index cannot say which turn a sample fell in.
            row: dict = {"n": n, "t": round(time.time(), 3)}
            try:
                with urllib.request.urlopen(f"{a.url}/health", timeout=a.timeout) as r:
                    s = json.load(r)["stats"]
                row.update({k: s[k] for k in (
                    "waiting", "running", "finished", "blocks_used", "blocks_total",
                    "pool_used_blocks", "slots_used", "prefix_hits", "prefix_misses",
                    "prefix_published", "prefix_evictions", "prefill_forwards",
                    "decode_forwards", "tokens_generated")})
                # .get, not the strict index above: this samples a RUNNING server, which
                # may predate the key. Absent reads 0, which is also its pre-row-62 value.
                row["prefix_superseded"] = s.get("prefix_superseded", 0)
                # Derived, because /health does not expose it: the allocator's own test at
                # engine.py:557 is `free_blocks < needed`, so this is the quantity that
                # decides a 503.
                row["free_blocks"] = s["blocks_total"] - s["pool_used_blocks"]
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["secs"] = round(time.monotonic() - t0, 3)
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            n += 1
            time.sleep(max(0.0, a.every - (time.monotonic() - t0)))

    with open(a.out, encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f]
    ok = [r for r in rows if "error" not in r]
    print(f"\nsamples {len(rows)}  ok {len(ok)}  failed {len(rows) - len(ok)}")
    if ok:
        free = [r["free_blocks"] for r in ok]
        secs = sorted(r["secs"] for r in ok)
        print(f"free_blocks: min {min(free)}  max {max(free)}  last {free[-1]}"
              f" of {ok[-1]['blocks_total']}")
        print(f"prefix_evictions: {ok[0]['prefix_evictions']} -> {ok[-1]['prefix_evictions']}"
              f"   superseded: {ok[0]['prefix_superseded']} -> {ok[-1]['prefix_superseded']}")
        print(f"/health seconds: min {secs[0]:.2f}  median {secs[len(secs) // 2]:.2f}"
              f"  max {secs[-1]:.2f}")
    print(f"rows: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
