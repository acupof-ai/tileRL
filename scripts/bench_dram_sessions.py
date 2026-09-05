"""Does the host snapshot tier pay in WALL CLOCK at N concurrent sessions?

The tier's condition is `sessions > HBM snapshot budget`, and the numbers on record for
it are HIT COUNTS: 2 sessions -> 0 promotions, 9 -> 17, 12 -> 24, and 0/63 evictions
without the tier against 24/0 with it. A promotion is not a saved second. The one number
that decides whether `--dram-bytes` is worth setting is wall clock per turn, and it has
never been measured -- the only wall-clock figure for the tier is the SINGLE-session arm,
where it is 1.51x WORSE.

So: same interleaved conversations, two arms differing only in `--dram-bytes`, one server
process per arm, wall clock per turn as the headline and per-session promotion counts as a
column beside it.

Four things this borrows from `bench_write_through`, each because leaving it out produced
a false table on this pod before:

* One server process per arm. A flag read at construction cannot be flipped in place, and
  a restart is also what empties HBM between arms.
* A JIT warm-up arm before the measured ones. 6 compiles inside a measured window is worth
  more than the effect.
* `compiles` counted per arm from the server log, and any non-zero count marks the whole
  run INVALID rather than being noted in prose.
* Alternating arm order per rep, so start-order drift cancels instead of loading one arm.

And one it borrows from `bench_chat_interleaved`: the distinct-prefix guard. Prefixes hash
from token 0, so N copies of one filler collapse into a single entry and the hit rate
measures the fixture. That function is imported rather than copied.

  scripts/pod_run.sh dramwall 6 -- /work/tl013/bin/python -u \
      scripts/bench_dram_sessions.py --sessions 2 --reps 2
  scripts/pod_run.sh dramwall 6 -- /work/tl013/bin/python -u \
      scripts/bench_dram_sessions.py --sessions 8 --reps 2
  scripts/pod_run.sh dramwall 6 -- /work/tl013/bin/python -u \
      scripts/bench_dram_sessions.py --sessions 12 --reps 2
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, "scripts")
from bench_chat_interleaved import _fillers, _label  # noqa: E402 — the prefix fixture

#: One snapshot is 144 MiB f32 on sm70 and HBM holds 9 (free/4). The tier has to hold more
#: than that to change anything, and 4 GiB = 28 is the figure the design page settled on:
#: pinned pages cannot swap and the pod has 31 GiB of RAM against a 32 GiB card.
DEFAULT_BUDGET = 4 << 30


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _stats(port: int) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10.0) as r:
        return json.loads(r.read())["stats"]


def _wait_up(port: int, proc: subprocess.Popen, deadline_s: float) -> None:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode} before serving")
        try:
            _stats(port)
            return
        except (urllib.error.URLError, OSError, KeyError):
            time.sleep(1.0)
    raise TimeoutError(f"server not up within {deadline_s}s")


def _compiles(log: str) -> int:
    try:
        with open(log, encoding="utf-8", errors="replace") as f:
            return sum("begins to compile" in line for line in f)
    except OSError:
        return -1


def _arm(args, name: str, budget: int) -> dict:
    """One server, one interleaved conversation set, wall clock per turn."""
    log = os.path.join(args.log_dir, f"dram_{name}.log")
    cmd = [
        args.python, "-u", "-m", "tilerl.cli", "serve",
        "--model", args.model, "--host", "127.0.0.1", "--port", str(args.port),
        "--max-batch", str(args.max_batch), "--max-ctx", str(args.max_ctx),
        "--slots", str(args.slots),
    ]
    if budget:
        cmd += ["--dram-bytes", str(budget)]
    # One shared cache across arms, so the JIT warm-up arm actually warms the measured
    # ones; `compiles` is the check that it did.
    env = dict(os.environ, TILELANG_CACHE_DIR=os.path.join(args.log_dir, "tilelang_cache"))
    with open(log, "wb") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                                cwd=args.repo)
    url = f"http://127.0.0.1:{args.port}"
    fillers = _fillers(args.sessions)
    turns: list[dict] = []
    try:
        _wait_up(args.port, proc, args.boot_s)
        # The flag must be visible before any turn is timed: an arm that silently ran
        # without the tier would report the `off` wall clock under the `on` label.
        st0 = _stats(args.port)
        got = st0.get("dram_budget")
        if bool(budget) != (got is not None):
            raise RuntimeError(
                f"arm {name} asked for dram_bytes={budget} and /health reports "
                f"dram_budget={got!r}: the arms are not what they are labelled"
            )
        convs: list[list[dict]] = [[] for _ in fillers]
        for turn in range(args.turns):
            for c, filler in enumerate(fillers):
                convs[c].append({"role": "user", "content": filler * args.grow * (turn + 1)})
                before = _stats(args.port)
                t0 = time.perf_counter()
                out = _post(f"{url}/v1/chat/completions",
                            {"model": args.model, "messages": convs[c],
                             "max_tokens": args.max_tokens, "temperature": 0.0},
                            args.req_s)
                wall = time.perf_counter() - t0
                after = _stats(args.port)
                convs[c].append({"role": "assistant",
                                 "content": out["choices"][0]["message"]["content"]})
                d = {k: after.get(k, 0) - before.get(k, 0)
                     for k in ("prefix_hits", "prefix_evictions",
                               "dram_demotions", "dram_promotions")}
                turns.append({"turn": turn, "conv": _label(c), "wall_s": round(wall, 3),
                              "prompt_tokens": out.get("usage", {}).get("prompt_tokens", 0),
                              **d})
                print(f"  [{name}] turn {turn} {_label(c)} wall={wall:7.2f}s "
                      f"hits={d['prefix_hits']} promote={d['dram_promotions']} "
                      f"evict={d['prefix_evictions']}", flush=True)
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    return {
        "arm": name.rstrip("0123456789_"),
        "compiles": _compiles(log),
        # Per turn, not per run: a run total is dominated by the growing prompt, so two
        # arms differ by the turn schedule as much as by the tier.
        "median_turn_s": round(statistics.median(t["wall_s"] for t in turns), 3),
        "total_s": round(sum(t["wall_s"] for t in turns), 2),
        "promotions": sum(t["dram_promotions"] for t in turns),
        "demotions": sum(t["dram_demotions"] for t in turns),
        "hits": sum(t["prefix_hits"] for t in turns),
        "evictions": sum(t["prefix_evictions"] for t in turns),
        # Per session, not a mean: a mean hides the case where the tier pays off for one
        # conversation and costs every other one.
        "per_session": {
            _label(c): {
                "hits": sum(t["prefix_hits"] for t in turns if t["conv"] == _label(c)),
                "promotions": sum(t["dram_promotions"] for t in turns
                                  if t["conv"] == _label(c)),
            }
            for c in range(args.sessions)
        },
        "turns": turns,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default="/work/tl013/bin/python")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--sessions", type=int, default=12,
                    help="interleaved conversations; the tier's condition is "
                         "sessions > 9, so 2 and 8 are the arms that must NOT win")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--grow", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--max-batch", type=int, default=1)
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--log-dir", default="/work",
                    help="where the per-arm server logs go; /work survives a container "
                         "restart on the pod, and `compiles` is counted from these files")
    ap.add_argument("--boot-s", type=float, default=900.0)
    ap.add_argument("--req-s", type=float, default=1800.0)
    args = ap.parse_args()
    if args.sessions < 1:
        ap.error("--sessions must be >= 1")

    fillers = _fillers(args.sessions)
    if len({s[:20] for s in fillers}) != args.sessions:
        raise SystemExit(
            f"{args.sessions} sessions produced {len({s[:20] for s in fillers})} distinct "
            "prefixes: the conversations would share cache entries and the hit rate would "
            "measure the fixture, not the tier"
        )

    print(f"jit warmup ({args.sessions} sessions, 1 turn)", flush=True)
    warm = argparse.Namespace(**{**vars(args), "turns": 1})
    _arm(warm, "jitwarm", 0)

    rows = []
    for i in range(args.reps):
        pair = [("on", args.budget), ("off", 0)] if i % 2 == 0 else [("off", 0),
                                                                    ("on", args.budget)]
        for label, budget in pair:
            row = _arm(args, f"{label}_{i}", budget)
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "turns"}), flush=True)

    on = [r["median_turn_s"] for r in rows if r["arm"] == "on"]
    off = [r["median_turn_s"] for r in rows if r["arm"] == "off"]
    med_on, med_off = statistics.median(on), statistics.median(off)
    verdict = {
        "sessions": args.sessions,
        "budget_bytes": args.budget,
        "n_per_arm": len(on),
        "on_median_turn_s": on,
        "off_median_turn_s": off,
        "median_on_s": round(med_on, 3),
        "median_off_s": round(med_off, 3),
        "delta_s": round(med_on - med_off, 3),
        "speedup": round(med_off / med_on, 3) if med_on else None,
        # Spread within one arm bounds what a single pair could have claimed.
        "spread_on_s": round(max(on) - min(on), 3),
        "spread_off_s": round(max(off) - min(off), 3),
        "promotions_on": sum(r["promotions"] for r in rows if r["arm"] == "on"),
        "evictions_on": sum(r["evictions"] for r in rows if r["arm"] == "on"),
        "evictions_off": sum(r["evictions"] for r in rows if r["arm"] == "off"),
    }
    if any(r["compiles"] for r in rows):
        verdict["INVALID"] = ("TileLang compiled inside a measured window ("
                              + ", ".join(f"{r['arm']}={r['compiles']}" for r in rows)
                              + "), so the arms differ by JIT")
    elif not verdict["promotions_on"]:
        verdict["INVALID"] = (
            f"the on arm took 0 promotions at {args.sessions} sessions, so both arms ran "
            "the same path and any delta is drift, not the tier"
        )
    elif abs(verdict["delta_s"]) < max(verdict["spread_on_s"], verdict["spread_off_s"]):
        verdict["INCONCLUSIVE"] = (
            f"|delta| {abs(verdict['delta_s'])}s is inside the within-arm spread "
            f"{max(verdict['spread_on_s'], verdict['spread_off_s'])}s"
        )
    print(json.dumps({"verdict": verdict, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
