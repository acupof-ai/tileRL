"""What does write-through cost the prefill it runs inside?

`insert` offers every publish to the tier: a GPU->CPU copy of the KV and the GDN snapshot
plus an enqueue, with the ~100 ms torch.save handed to a daemon. The copy is synchronous
and lands mid-prefill, so it is charged to the request that published it. This measures
that charge with the ONLY variable being `--ssd-path`.

Both arms serve the same prompt against an empty store, so neither ever reads the tier --
`ssd_hits` must be 0 in both, and the assert says so. The `on` arm additionally reports
`ssd_offered` and `ssd_refusals`: a refusal means the bounded queue filled, which is the
failure mode that would make write-through cost more than the copy.

Alternating arms, n runs each, because a single pair cannot separate a 5% effect from
start-order drift -- measured on this pod, the same arm varies 2.986-3.044 s run to run.

What this cost was, and what it is, on H20 card 6 with one 2729-token prompt:

    every publish spilled, pageable copy   +0.925 s   45.3%   6 offers
    every publish, pinned + non_blocking   +0.383 s   19.1%   6 offers
    last publish only, pageable            +0.207 s   10.0%   1 offer
    SHIPPED: last publish, pinned          +0.180 s    8.96%  1 offer

Three guesses at the per-publish cost were wrong before the stage timers were added, so
they stay on /health: `ssd_gather_ms` is 1 ms and `ssd_copy_ms` is 170 ms, which says the
whole cost is the device-to-host copy and none of it is the 171-slice block gather.

  scripts/pod_run.sh wt 6 -- /work/tl013/bin/python -u \
      scripts/bench_write_through.py --reps 4 --tokens 3000
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.request

_FILLER = (
    "Explain in detail how a paged key-value cache serves a transformer decode step, "
    "including how block tables map logical positions to physical pages. "
)


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


def _prompt(target_tokens: int) -> str:
    words = max(1, int(target_tokens / 1.3))
    text = (_FILLER * (words // len(_FILLER.split()) + 2)).split()
    return " ".join(text[:words]) + " Summarize the mechanism in one sentence."


def _compiles(log: str) -> int:
    try:
        with open(log, encoding="utf-8", errors="replace") as f:
            return sum("begins to compile" in line for line in f)
    except OSError:
        return -1


def _arm(args, name: str, spill: str, prompt: str) -> dict:
    log = f"/work/wt_{name}.log"
    cmd = [
        args.python, "-u", "-m", "tilerl.cli", "serve",
        "--model", args.model, "--host", "127.0.0.1", "--port", str(args.port),
        "--max-batch", "1", "--max-ctx", str(args.max_ctx), "--slots", str(args.slots),
    ]
    if spill:
        cmd += ["--ssd-path", spill]
        if args.min_tokens:
            cmd += ["--ssd-min-tokens", str(args.min_tokens)]
    env = dict(os.environ, TILELANG_CACHE_DIR="/work/tilelang_cache")
    with open(log, "wb") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                                cwd=args.repo)
    try:
        _wait_up(args.port, proc, args.boot_s)
        before = _stats(args.port)
        t0 = time.monotonic()
        r = _post(f"http://127.0.0.1:{args.port}/v1/messages",
                  {"model": args.model, "max_tokens": args.gen,
                   "messages": [{"role": "user", "content": prompt}]}, args.req_s)
        wall = time.monotonic() - t0
        after = _stats(args.port)
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    d = lambda k: int(after.get(k, 0)) - int(before.get(k, 0))  # noqa: E731
    return {
        "arm": name.rstrip("0123456789_"),
        "wall_s": round(wall, 3),
        "compiles": _compiles(log),
        "prompt_tokens": int(r["usage"]["input_tokens"]),
        "prefix_published": d("prefix_published"),
        "ssd_offered": d("ssd_offered"),
        "ssd_refusals": d("ssd_refusals"),
        "ssd_hits": d("ssd_hits"),
        # Stage attribution, so the residual cost is read rather than guessed again.
        "gather_ms": d("ssd_gather_ms"),
        "contig_ms": d("ssd_contig_ms"),
        "copy_ms": d("ssd_copy_ms"),
        "pin_misses": d("ssd_pin_misses"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--python", default="/work/tl013/bin/python")
    ap.add_argument("--repo", default="/work/tilerl")
    ap.add_argument("--spill", default="/work/wt_bench")
    ap.add_argument("--port", type=int, default=8124)
    ap.add_argument("--reps", type=int, default=4, help="runs per arm, alternating")
    ap.add_argument("--tokens", type=int, default=3000)
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--boot-s", type=float, default=900.0)
    ap.add_argument("--req-s", type=float, default=1800.0)
    ap.add_argument("--min-tokens", type=int, default=0,
                    help="raise the tier's spill floor (0 = its default 64). Every publish "
                         "carries a CONSTANT ~157 MB GDN snapshot regardless of prefix "
                         "length, so 6 publishes copy 941 MB of state to serve one entry; "
                         "this is the lever on that, and the projection says it is a weak "
                         "one -- verify rather than assume")
    args = ap.parse_args()

    prompt = _prompt(args.tokens)
    # JIT warm start, same reason as bench_ssd_restart: 6 compiles in one arm's window is
    # worth more than the effect being measured.
    warm = args.spill + "_warmup"
    shutil.rmtree(warm, ignore_errors=True)
    os.makedirs(warm, exist_ok=True)
    _arm(args, "jitwarm", warm, prompt)
    shutil.rmtree(warm, ignore_errors=True)

    rows = []
    for i in range(args.reps):
        # A FRESH spill dir per `on` run: a reused one would let the second run find its
        # own earlier entries resident and skip the write, which is the cost under test.
        d = f"{args.spill}_{i}"
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        # Alternate the order so start-order drift cancels rather than loading one arm.
        pair = [("on", d), ("off", "")] if i % 2 == 0 else [("off", ""), ("on", d)]
        for label, spill in pair:
            row = _arm(args, f"{label}_{i}", spill, prompt)
            rows.append(row)
            print(json.dumps(row), flush=True)
        shutil.rmtree(d, ignore_errors=True)

    on = [r["wall_s"] for r in rows if r["arm"] == "on"]
    off = [r["wall_s"] for r in rows if r["arm"] == "off"]
    med_on, med_off = statistics.median(on), statistics.median(off)
    verdict = {
        "n_per_arm": len(on),
        "on_s": on,
        "off_s": off,
        "median_on_s": round(med_on, 3),
        "median_off_s": round(med_off, 3),
        "cost_s": round(med_on - med_off, 3),
        "cost_pct": round(100 * (med_on - med_off) / med_off, 2),
        # Spread within one arm bounds what a single pair could have claimed.
        "spread_on_s": round(max(on) - min(on), 3),
        "spread_off_s": round(max(off) - min(off), 3),
        "offered_total": sum(r["ssd_offered"] for r in rows),
        "refusals_total": sum(r["ssd_refusals"] for r in rows),
    }
    if any(r["compiles"] for r in rows):
        verdict["INVALID"] = ("TileLang compiled inside a measured window ("
                              + ", ".join(f"{r['arm']}={r['compiles']}" for r in rows)
                              + "), so the arms differ by JIT")
    elif any(r["ssd_hits"] for r in rows):
        verdict["INVALID"] = (
            "an arm took an SSD hit; both arms must run against an empty store or this "
            "measures the read path instead of the write path"
        )
    elif not verdict["offered_total"]:
        verdict["INVALID"] = (
            "the `on` arm offered nothing to the tier, so `on` and `off` ran the same code"
        )
    elif abs(verdict["cost_s"]) < max(verdict["spread_on_s"], verdict["spread_off_s"]):
        verdict["WITHIN_NOISE"] = (
            f"the {verdict['cost_s']:+.3f} s difference is smaller than the "
            f"{max(verdict['spread_on_s'], verdict['spread_off_s']):.3f} s spread inside one "
            f"arm, so write-through costs nothing this bench can resolve at n={len(on)}"
        )
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
