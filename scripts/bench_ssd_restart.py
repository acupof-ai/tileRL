"""Does a restart fault the prefix off disk, or is the second run just warmer?

The SSD tier's whole claim is that after a restart HBM is empty so every returning
conversation's first turn reaches back to disk. Measuring it as "start, serve, restart,
serve" does not test that claim: the tilelang JIT cache is shared across starts
(`TILELANG_CACHE_DIR=/work/tilelang_cache`), the page cache holds the weights, and both
make the SECOND start faster whatever the tier does.

So three arms, each its own server start, each with a throwaway warm-up request before
the measured one so no JIT lands inside the timed window:

    cold     empty spill dir            -> the number to beat
    faulted  the dir cold just filled   -> the tier's number
    control  a DIFFERENT empty dir      -> must land back at `cold`

`control` is the arm that makes this a measurement. If it comes out as fast as
`faulted`, the speedup was start order and not the tier, and the run says nothing.

  python scripts/bench_ssd_restart.py --card 6 --model qwen38-27b --tokens 3000
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
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
    # ~1.3 tokens per word for this filler; deliberately NOT block-aligned, since a
    # ragged length is the case the publish fix exists for.
    words = max(1, int(target_tokens / 1.3))
    text = (_FILLER * (words // len(_FILLER.split()) + 2)).split()
    return " ".join(text[:words]) + " Summarize the mechanism in one sentence."


def _serve(args, spill: str, log: str):
    cmd = [
        args.python, "-u", "-m", "tilerl.cli", "serve",
        "--model", args.model, "--host", "127.0.0.1", "--port", str(args.port),
        "--max-batch", "1", "--max-ctx", str(args.max_ctx), "--slots", str(args.slots),
    ]
    if spill:
        cmd += ["--ssd-path", spill]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.card),
               TILELANG_CACHE_DIR="/work/tilelang_cache")
    with open(log, "wb") as f:
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                                cwd=args.repo)


def _arm(args, name: str, spill: str, prompt: str) -> dict:
    log = f"/work/ssd_restart_{name}.log"
    proc = _serve(args, spill, log)
    try:
        _wait_up(args.port, proc, args.boot_s)
        # Warm-up: JIT and any first-call allocation land here, not in the measured window.
        # A different short prompt, so it publishes nothing the measured one can match.
        _post(f"http://127.0.0.1:{args.port}/v1/messages",
              {"model": args.model, "max_tokens": 4,
               "messages": [{"role": "user", "content": "hi"}]}, args.req_s)
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
        "arm": name,
        "wall_s": round(wall, 3),
        "prompt_tokens": int(r["usage"]["input_tokens"]),
        "output_tokens": int(r["usage"]["output_tokens"]),
        "ms_per_prompt_token": round(1000 * wall / max(1, r["usage"]["input_tokens"]), 3),
        "ssd_hits": d("ssd_hits"),
        "ssd_faults": d("ssd_faults"),
        "ssd_entries": int(after.get("ssd_entries", 0)),
        "ssd_recovered": int(after.get("ssd_recovered", 0)),
        "ssd_offered": d("ssd_offered"),
        "ssd_refusals": d("ssd_refusals"),
        "prefix_hits": d("prefix_hits"),
        "prefix_published": d("prefix_published"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, default=6)
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--python", default="/work/tl013/bin/python")
    ap.add_argument("--repo", default="/work/tilerl")
    ap.add_argument("--spill", default="/work/ssd_tier_bench")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--tokens", type=int, default=3000, help="target prompt length")
    ap.add_argument("--gen", type=int, default=8, help="tokens to generate; keep small so "
                    "the wall clock is prefill")
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--boot-s", type=float, default=900.0)
    ap.add_argument("--req-s", type=float, default=1800.0)
    args = ap.parse_args()

    prompt = _prompt(args.tokens)
    main_dir, ctrl_dir = args.spill, args.spill + "_control"
    for d in (main_dir, ctrl_dir):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    rows = [_arm(args, "cold", main_dir, prompt)]
    print(json.dumps(rows[-1]), flush=True)
    rows.append(_arm(args, "faulted", main_dir, prompt))
    print(json.dumps(rows[-1]), flush=True)
    rows.append(_arm(args, "control", ctrl_dir, prompt))
    print(json.dumps(rows[-1]), flush=True)

    cold, faulted, control = rows
    verdict = {
        "speedup_faulted_over_cold": round(cold["wall_s"] / faulted["wall_s"], 3),
        "speedup_control_over_cold": round(cold["wall_s"] / control["wall_s"], 3),
        "faulted_recovered_entries": faulted["ssd_recovered"],
        "faulted_ssd_hits": faulted["ssd_hits"],
        "control_ssd_hits": control["ssd_hits"],
    }
    # The two assertions that decide whether the number means anything.
    if faulted["ssd_hits"] < 1:
        verdict["INVALID"] = (
            f"the faulted arm took {faulted['ssd_hits']} SSD hits with "
            f"{faulted['ssd_recovered']} entries recovered, so whatever it measured was "
            "not the tier"
        )
    elif control["ssd_hits"] != 0:
        verdict["INVALID"] = (
            f"the control arm took {control['ssd_hits']} SSD hits from a directory that "
            "was created empty"
        )
    elif verdict["speedup_control_over_cold"] > verdict["speedup_faulted_over_cold"] * 0.9:
        verdict["INVALID"] = (
            f"the control is {verdict['speedup_control_over_cold']}x faster than cold with "
            f"an empty tier, against the faulted arm's "
            f"{verdict['speedup_faulted_over_cold']}x -- the speedup is start order, not "
            "the tier"
        )
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
