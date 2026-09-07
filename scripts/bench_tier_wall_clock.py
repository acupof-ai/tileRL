"""Per-turn wall clock with the SSD tier off vs on, across session counts.

The verdict criterion (#152's): the tier ships on the serve path only if wall clock per
turn is not worse at the session count where HBM overflows. That needs three things this
measures and one it checks rather than assumes.

**The session count where HBM overflows is a measured quantity, not 12.** Two different
ceilings exist and they are not the same threshold: the block pool filling (`pool_used_blocks`
against `blocks_total`, which forces `evict_until_free`) and the SSD tier's own byte budget
filling (`ssd_evictions` climbing). At `--max-ctx 49152 --slots 3` the arrival-rate probe saw
37 evictions in 72 offers at 12 sessions, so the byte budget was already thrashing there
while nothing said whether the pool was. This prints both operands and both ceilings for
every cell, so the write-up can say where each threshold falls instead of forcing a verdict
at a session count that may be below either one.

**One process per arm, not per cell.** A fresh process pays the first TileLang compile of
every shape it meets, and a first-position JIT has produced a clean, false table in this
repo before. Each arm serves every session count in one server, and `compiles` is asserted
0 for the measured cells after a warm-up arm has populated the cache.

**Rows stream.** Each cell prints as it completes, so a crash in cell 5 does not cost
cells 1-4 -- and the summary is derived from rows that are already in the log, so a
teardown failure costs the derivation and not the measurement.

    scripts/pod_run.sh tierwall 6 -- /work/tl013/bin/python -u \\
        scripts/bench_tier_wall_clock.py --sessions 2,8,12
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, "scripts")
from bench_chat_interleaved import _fillers  # noqa: E402

PORT = 8129
SPILL = "/work/tier_wall_spill"


def _stats(port: int) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10.0) as r:
        return json.loads(r.read())["stats"]


def _compiles(log: str) -> int:
    try:
        with open(log, encoding="utf-8", errors="replace") as f:
            return sum("begins to compile" in line for line in f)
    except OSError:
        return -1


def _wait_up(port: int, proc, boot_s: float) -> dict:
    end = time.monotonic() + boot_s
    while time.monotonic() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited {proc.returncode}")
        try:
            return _stats(port)
        except (urllib.error.URLError, OSError, KeyError):
            time.sleep(1.0)
    raise TimeoutError(f"server not up within {boot_s}s")


def _turn(port: int, model: str, convs: list, c: int, req_s: float) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"model": model, "messages": convs[c],
                         "max_tokens": 32, "temperature": 0.0}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=req_s) as r:
        out = json.loads(r.read())
    return {"wall_s": time.monotonic() - t0, "out": out}


def run_arm(args, arm: str, spill: str, log: str) -> list[dict]:
    """One server, every session count. Returns one row per (sessions, turn) cell."""
    cmd = [args.python, "-u", "-m", "tilerl.cli", "serve", "--model", args.model,
           "--host", "127.0.0.1", "--port", str(PORT), "--max-batch", "1",
           "--max-ctx", str(args.max_ctx), "--slots", str(args.slots)]
    if spill:
        cmd += ["--ssd-path", spill]
    # setdefault, not override: a hardcoded /work made the child recompile on any other box.
    env = dict(os.environ)
    env.setdefault("TILELANG_CACHE_DIR", "/work/tilelang_cache")
    rows: list[dict] = []
    with open(log, "wb") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                                cwd=args.repo)
    try:
        st0 = _wait_up(PORT, proc, args.boot_s)
        # The tier being off must be visible in the data, not inferred from the flag:
        # a tier-off /health emits no ssd_* keys at all.
        has_tier = any(k.startswith("ssd_") for k in st0)
        assert has_tier == bool(spill), (
            f"arm {arm}: spill={spill!r} but ssd_* keys present={has_tier}")
        blocks_total = int(st0["blocks_total"])
        print(f"# arm {arm} up: blocks_total={blocks_total} "
              f"tier={'on' if has_tier else 'off'}", flush=True)
        for n in args.sessions:
            fillers = _fillers(n)
            convs: list[list] = [[] for _ in fillers]
            for turn in range(args.turns):
                for c, filler in enumerate(fillers):
                    convs[c].append({"role": "user",
                                     "content": filler * args.grow * (turn + 1)})
                    before = _stats(PORT)
                    t = _turn(PORT, args.model, convs, c, args.req_s)
                    after = _stats(PORT)
                    convs[c].append({"role": "assistant",
                                     "content": t["out"]["choices"][0]["message"]["content"]})
                    row = {
                        "arm": arm, "sessions": n, "turn": turn, "conv": c,
                        "wall_s": round(t["wall_s"], 3),
                        "prompt_tokens": int(t["out"]["usage"]["prompt_tokens"]),
                        # Both operands and both ceilings, every cell: the HBM threshold
                        # and the SSD-byte threshold are different and only one was ever
                        # observed. A cell that reports one cannot say which bound.
                        "pool_used_blocks": after.get("pool_used_blocks"),
                        "blocks_total": blocks_total,
                        "prefix_evictions": after.get("prefix_evictions", 0),
                        "prefix_published": after.get("prefix_published", 0),
                        "prefix_hits": after.get("prefix_hits", 0),
                        "d_prefix_evictions": (after.get("prefix_evictions", 0)
                                               - before.get("prefix_evictions", 0)),
                        # Read together: a publisher retiring its own entry counts as superseded, not eviction,
                        # so the eviction delta alone falls whether pressure eased or moved.
                        "d_prefix_superseded": (after.get("prefix_superseded", 0)
                                                - before.get("prefix_superseded", 0)),
                        "ssd_evictions": after.get("ssd_evictions", 0),
                        "ssd_offered": after.get("ssd_offered", 0),
                        "ssd_refusals": after.get("ssd_refusals", 0),
                        "ssd_hits": after.get("ssd_hits", 0),
                        "ssd_bytes": after.get("ssd_bytes", 0),
                        "compiles": _compiles(log),
                    }
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--python", default="/work/tl013/bin/python")
    ap.add_argument("--repo", default=os.environ.get("REMOTE_DIR"),
                    required="REMOTE_DIR" not in os.environ,
                    help="server cwd, one tree per session. No fallback: a wrong tree produces a number, not an error")
    ap.add_argument("--sessions", default="2,8,12",
                    help="comma-separated session counts, served in one process per arm")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--grow", type=int, default=40)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--max-ctx", type=int, default=49152)
    ap.add_argument("--boot-s", type=float, default=900.0)
    ap.add_argument("--req-s", type=float, default=1800.0)
    ap.add_argument("--spill", default=SPILL)
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--out", default="/work/tier_wall.json")
    a = ap.parse_args()
    a.sessions = [int(x) for x in a.sessions.split(",") if x.strip()]

    # A warm-up arm so the measured arms meet a populated TileLang cache. Without it the
    # FIRST arm pays every compile and reads slower for a reason that is not the tier.
    if not a.skip_warmup:
        warm = a.spill + "_warmup"
        shutil.rmtree(warm, ignore_errors=True)
        os.makedirs(warm, exist_ok=True)
        w = argparse.Namespace(**{**vars(a), "sessions": [2], "turns": 1})
        run_arm(w, "jitwarm", warm, "/work/tier_wall_jitwarm.log")
        shutil.rmtree(warm, ignore_errors=True)

    rows: list[dict] = []
    # Tier off first, then on, into a wiped spill dir: a tier that adopts a previous run's
    # files starts with hits it did not earn (measured: ssd_recovered=39 once).
    shutil.rmtree(a.spill, ignore_errors=True)
    rows += run_arm(a, "off", "", "/work/tier_wall_off.log")
    os.makedirs(a.spill, exist_ok=True)
    rows += run_arm(a, "on", a.spill, "/work/tier_wall_on.log")

    print("\n=== summary (derived from the rows above, which are already in the log)",
          flush=True)
    summary = {}
    for n in a.sessions:
        cell = {}
        for arm in ("off", "on"):
            w = [r["wall_s"] for r in rows if r["arm"] == arm and r["sessions"] == n]
            if not w:
                continue
            cell[arm] = {"turns": len(w), "mean_s": round(float(np.mean(w)), 3),
                         "median_s": round(float(np.median(w)), 3),
                         "sd_s": round(float(np.std(w, ddof=1)), 3) if len(w) > 1 else None}
        if "off" in cell and "on" in cell and cell["off"]["mean_s"]:
            cell["on_over_off"] = round(cell["on"]["mean_s"] / cell["off"]["mean_s"], 4)
        # Which ceiling, if either, this session count actually reached.
        for arm in ("off", "on"):
            rs = [r for r in rows if r["arm"] == arm and r["sessions"] == n]
            if not rs:
                continue
            peak = max((r["pool_used_blocks"] or 0) for r in rs)
            cell[f"{arm}_pool_peak_pct"] = round(100.0 * peak / rs[0]["blocks_total"], 1)
            cell[f"{arm}_prefix_evictions"] = max(r["prefix_evictions"] for r in rs)
            cell[f"{arm}_prefix_superseded"] = max(r.get("d_prefix_superseded", 0) for r in rs)
            if arm == "on":
                cell["ssd_evictions"] = max(r["ssd_evictions"] for r in rs)
                cell["ssd_offered"] = max(r["ssd_offered"] for r in rs)
                cell["ssd_hits"] = max(r["ssd_hits"] for r in rs)
        summary[n] = cell
    bad = [(r["arm"], r["sessions"], r["compiles"]) for r in rows if r["compiles"] > 0]
    print(json.dumps({"per_sessions": summary,
                      "cells_with_compiles": bad,
                      "compiles_clean": not bad}, indent=2, sort_keys=True), flush=True)
    if bad:
        print(f"WARNING: {len(bad)} cells compiled during measurement; those wall clocks "
              f"include a JIT and are not comparable", flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "summary": {str(k): v for k, v in summary.items()}},
                  f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
