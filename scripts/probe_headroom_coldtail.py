#!/usr/bin/env python3
"""One-command per-arm load for the --device-headroom-mib V100 cold-tail test
(#702; runbook: docs/experience/errors/2026-09-17-cold-tier-full-finalize-
relocation-long-tail.md).

The operator boots ONE arm's serve (TILERL_DEVICE_HEADROOM_MIB=0|384|768,
TILERL_STEP_TIMING=1, TILERL_STEP_TIMING_SLOW_MS=0) and then runs this against
it. It does NOT restart the server — restarting belongs to ops, one boot per
arm, so two flows never race. It drives the full load against the already-up
arm and ends with a single machine-readable line:

    ARM_DONE <headroom_mib> <decode_tok_s>   (exit 0 on success)

Sequence per arm:
  1. fill  — N independent (non-prefix-sharing) >=32k sparse prompts force the
     shared cold pool to the full-tier state (the W-sweep filler shape); assert
     cold occupancy actually crossed --min-cold-gb before measuring;
  2. warm  — one primed-warm 32k prompt streamed; time first-token -> last-token
     (decode-only tok/s, prefill excluded), the baseline's quantity;
  3. read  — /health build fields + nvidia-smi physical free, then parse THIS
     arm's server log and split slow ticks into type-1 finalize batches
     (offers_pages>0) and type-2 hollow forwards (no offers, 1.1-1.4 s, inner
     model well under total).

Writes one JSON summary per arm and prints ARM_DONE. SLOW_MS must be 0 so every
tick is logged — p50/p90 and the >300 ms fraction need the full distribution.
Pure stdlib, no torch; runs from the laptop against the pod URL. control0 must
match the historical baseline口径 (2.69/4.25/6.56/7.86 tok/s), so the warm
quantity is first-token→last-token decode tok/s over a 32k context.

Run: python3 scripts/probe_headroom_coldtail.py --self-check
     python3 scripts/probe_headroom_coldtail.py arm --url http://127.0.0.1:8000 \
         --headroom 768 --log /root/servehybridsse.log --out /root/arm768.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
import uuid

MODEL = "qwen38-27b"
# One 32k slot's hot ceiling (n_groups*k + WINDOW + chunk = 4*128+8+33); blocks.
HOT_BLOCKS_PER_SLOT = 553

_WORDS = [
    "paged", "cache", "scheduler", "latency", "block", "allocation", "prefix",
    "sharing", "cold", "tier", "migration", "attention", "kernel", "quantized",
    "weight", "draft", "head", "graph", "capture", "ledger", "capacity",
    "planning", "eviction", "promote", "demote", "resident", "window",
    "selection", "decode",
]


def prompt_32k(n_tokens: int, seed: int) -> str:
    """Independent filler: a uuid lead per seed so the N prompts share no prefix
    (shared pages would not fill the pool), then a deterministic word stream."""
    out = [f"Reference {uuid.UUID(int=(seed or 1) % (1 << 128)).hex}. "]
    i = 0
    while True:
        for w in _WORDS:
            out.append(w.upper() if (i + seed) % 11 == 0 else w)
            i += 1
            if i >= n_tokens:
                return " ".join(out) + "."


def _post(url: str, body: dict, timeout: float, stream: bool) -> dict:
    body = {**body, "stream": stream}
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    if not stream:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    n = 0
    t0 = t_first = t_last = time.perf_counter()
    saw = False
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            s = raw.decode(errors="replace").strip()
            if not s.startswith("data:"):
                continue
            s = s[5:].strip()
            if s == "[DONE]":
                break
            d = json.loads(s)["choices"][0]["delta"].get("content", "")
            if d:
                now = time.perf_counter()
                if not saw:
                    t_first = now
                    saw = True
                t_last = now
                n += 1
    return {"chunks": n, "wall_s": t_last - t0,
            "decode_s": (t_last - t_first) if saw else 0.0}


def _body(prompt: str, gen: int) -> dict:
    return {"model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": gen, "enable_thinking": False}


def health(url: str) -> dict:
    with urllib.request.urlopen(url + "/health", timeout=10) as r:
        return json.loads(r.read())["stats"]


def wait_ready(url: str, trials: int) -> None:
    for _ in range(trials):
        try:
            health(url)
            return
        except OSError:
            time.sleep(2)
    raise SystemExit("server not reachable on /health within ready wait")


def nvidia_free_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


# --- log parsing -------------------------------------------------------------

_TICK = re.compile(r"\[step-timing\] tick (\d+) total=(-?\d+)ms (.*)")
_KV = re.compile(r"(\w+)=(-?\d+)ms")
_OFFERS = re.compile(r"offers_pages=(\d+)")
_FREE = re.compile(r" free=(-?\d+)MiB")


def parse_tick(line: str) -> dict | None:
    m = _TICK.search(line)
    if not m:
        return None
    n, total, rest = int(m.group(1)), int(m.group(2)), m.group(3)
    segs = {k: int(v) for k, v in _KV.findall(rest)}
    mo = _OFFERS.search(rest)
    offers = int(mo.group(1)) if mo else 0
    mf = _FREE.search(rest)
    # type-2 hollow: no finalize batch, 1.1-1.4 s total, model inner <= 60% total.
    hollow = offers == 0 and total >= 1100 and segs.get("model", 0) <= int(0.6 * total)
    return {"n": n, "total_ms": total, "offers_pages": offers,
            "finalize_ms": segs.get("sparse_finalize", 0),
            "model_ms": segs.get("model", 0),
            "free_mib": int(mf.group(1)) if mf else None,
            "type1": offers > 0, "type2": hollow}


def pct(xs: list[int], q: float) -> int:
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))]


def parse_log(log_path: str) -> dict:
    with open(log_path, errors="replace") as fh:
        ticks = [t for line in fh if (t := parse_tick(line))]
    totals = [t["total_ms"] for t in ticks]
    t1 = [t for t in ticks if t["type1"]]
    t2 = [t for t in ticks if t["type2"]]
    fin = [t["finalize_ms"] for t in t1 if t["finalize_ms"] > 0]
    pages = [t["offers_pages"] for t in t1]
    return {
        "ticks": len(ticks),
        "p50_ms": pct(totals, 0.50), "p90_ms": pct(totals, 0.90),
        "max_ms": max(totals, default=0),
        "frac_over_300": round(len([x for x in totals if x > 300]) / len(ticks), 3)
        if ticks else 0.0,
        "type1_finalize_ticks": len(t1),
        "type1_finalize_ms_median": pct(fin, 0.5),
        "type1_finalize_ms_max": max(fin, default=0),
        "type1_offers_pages_median": pct(pages, 0.5),
        "type1_offers_pages_max": max(pages, default=0),
        "type2_hollow_ticks": len(t2),
        "steady_free_mib": ticks[-1]["free_mib"] if ticks else None,
    }


# --- the one-command arm -----------------------------------------------------

def cmd_arm(a) -> int:
    wait_ready(a.url, a.ready_trials)
    h0 = health(a.url)
    pt_rows = []
    for i in range(a.fill_n):
        out = _post(a.url + "/v1/chat/completions",
                    _body(prompt_32k(a.prompt_tokens, i + 1), a.fill_gen),
                    a.timeout, stream=False)
        pt = out.get("usage", {}).get("prompt_tokens", 0)
        pt_rows.append(pt)
        if pt < a.prompt_tokens * 0.9:
            print(f"WARN fill {i}: only {pt} prompt tokens (want ~{a.prompt_tokens})",
                  flush=True)
    hfill = health(a.url)
    cold_gb = hfill.get("kv_cold_bytes", 0) / 2**30
    if cold_gb < a.min_cold_gb:
        print(f"FILL-INSUFFICIENT cold={cold_gb:.2f}GiB < {a.min_cold_gb}GiB; "
              "refusing to measure a non-full tier", flush=True)
        return 3

    s = _post(a.url + "/v1/chat/completions",
              _body(prompt_32k(a.prompt_tokens, 0), a.warm_gen),
              a.timeout, stream=True)
    hw = health(a.url)
    decode_tok = max(1, s["chunks"] - 1)
    tok_s = round(decode_tok / s["decode_s"], 3) if s["decode_s"] else 0.0
    blocks_total = hw.get("blocks_total", 0)
    summary = {
        "arm_headroom_mib": a.headroom,
        "fill": {"n": a.fill_n, "prompt_tokens": pt_rows,
                 "cold_gb": round(cold_gb, 3),
                 "cold_bytes_before": h0.get("kv_cold_bytes", 0)},
        "warm32k": {"decode_tok_s": tok_s, "content_chunks": s["chunks"],
                    "first_to_last_s": round(s["decode_s"], 3),
                    "wall_s": round(s["wall_s"], 3)},
        "health": {
            "sparse_headroom_bytes": hw.get("sparse_headroom_bytes", 0),
            "sparse_headroom_dropped_blocks":
                hw.get("sparse_headroom_dropped_blocks", 0),
            "blocks_total": blocks_total,
            "resident_slots_floor": blocks_total // HOT_BLOCKS_PER_SLOT,
            "slots_total_state_pool": hw.get("slots_total", 0),
            "process_free_mib": hw.get("device_free_bytes", 0) >> 20,
            "physical_free_mib_nvidia": nvidia_free_mib(),
        },
        "ticks": parse_log(a.log),
    }
    with open(a.out, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    print(f"ARM_DONE {a.headroom} {tok_s}", flush=True)
    return 0


def cmd_compare(a) -> int:
    """Lay the three arm summaries next to the historical full/empty bands.
    Run after all three ARM_DONE; it decides nothing, it places the numbers."""
    arms = {}
    for spec in a.arms:
        name, path = spec.split("=", 1)
        with open(path) as fh:
            arms[name] = json.load(fh)
    base = {"full_tier_tok_s": [2.69, 4.25, 6.56], "empty_tier_tok_s": 7.86,
            "full_finalize_ms": [629, 667], "empty_finalize_ms": [81, 186],
            "full_frac_over_300": [0.40, 0.58], "empty_frac_over_300": 0.10}
    rows = {}
    for name, s in sorted(arms.items()):
        h, t = s["health"], s["ticks"]
        rows[name] = {
            "headroom_mib": s["arm_headroom_mib"],
            "warm32k_decode_tok_s": s["warm32k"]["decode_tok_s"],
            "p50/p90/max_ms": [t["p50_ms"], t["p90_ms"], t["max_ms"]],
            "frac_over_300": t["frac_over_300"],
            "finalize_med/max_ms": [t["type1_finalize_ms_median"],
                                    t["type1_finalize_ms_max"]],
            "offers_med/max_pages": [t["type1_offers_pages_median"],
                                     t["type1_offers_pages_max"]],
            "type1_ticks": t["type1_finalize_ticks"],
            "type2_hollow_ticks": t["type2_hollow_ticks"],
            "blocks_total": h["blocks_total"],
            "resident_slots_floor": h["resident_slots_floor"],
            "dropped_blocks": h["sparse_headroom_dropped_blocks"],
            "process_free_mib": h["process_free_mib"],
            "physical_free_mib": h["physical_free_mib_nvidia"],
            "cold_gb": s["fill"]["cold_gb"],
        }
    print(json.dumps({"baseline": base, "arms": rows}, indent=1))
    return 0


def _self_check() -> int:
    l1 = ("[step-timing] tick 7 total=1259ms plan=1ms stats=2ms forward=1250ms "
          "sparse_select=3ms model=330ms sparse_finalize=629ms sample=1ms "
          "[offers_pages=132] free=210MiB reserved=30000MiB path=eager sparse=1 "
          "fwd_host=1250ms fwd_gpu=400ms why=sync_wait")
    t = parse_tick(l1)
    assert t["type1"] and not t["type2"]
    assert t["finalize_ms"] == 629 and t["offers_pages"] == 132 and t["free_mib"] == 210
    l2 = ("[step-timing] tick 8 total=1303ms plan=1ms forward=1300ms model=350ms "
          "sample=1ms  free=206MiB reserved=30000MiB path=eager sparse=1 "
          "fwd_host=1300ms fwd_gpu=420ms why=sync_wait")
    t = parse_tick(l2)
    assert t["type2"] and not t["type1"] and t["model_ms"] == 350
    t = parse_tick("[step-timing] tick 9 total=180ms forward=176ms model=170ms why=cpu")
    assert t and not t["type1"] and not t["type2"]
    assert pct([100, 200, 300, 400], 0.5) == 200
    assert pct(list(range(100, 1100, 100)), 0.9) == 900  # 10 pts, 9th nearest-rank
    assert len(prompt_32k(32000, 1).split()) > 30000
    assert prompt_32k(100, 1) != prompt_32k(100, 2)  # seeds -> independent
    print("probe_headroom_coldtail self-check ok")
    return 0


def main() -> int:
    if "--self-check" in sys.argv[1:]:
        return _self_check()
    ap = argparse.ArgumentParser()
    s = ap.add_subparsers(dest="cmd")
    # Bare invocation runs the self-check: the repo-wide
    # test_main_selfchecks gate runs every hermetic script with NO args and
    # requires rc 0, so a missing subcommand must default to the self-check
    # rather than argparse-rc-2.
    a = s.add_parser("selfcheck")
    a.set_defaults(fn=_self_check)
    a = s.add_parser("arm")
    a.add_argument("--url", default="http://127.0.0.1:8000")
    a.add_argument("--headroom", type=int, required=True,
                   help="this arm's TILERL_DEVICE_HEADROOM_MIB (0/384/768)")
    a.add_argument("--log", required=True, help="this arm's server stderr/stdout log")
    a.add_argument("--out", required=True, help="arm summary JSON path")
    a.add_argument("--fill-n", type=int, default=5)
    a.add_argument("--prompt-tokens", type=int, default=32000)
    a.add_argument("--fill-gen", type=int, default=8)
    a.add_argument("--warm-gen", type=int, default=32)
    a.add_argument("--min-cold-gb", type=float, default=7.0)
    a.add_argument("--timeout", type=float, default=7200.0)
    a.add_argument("--ready-trials", type=int, default=600)
    a.set_defaults(fn=cmd_arm)
    c = s.add_parser("compare")
    c.add_argument("--arms", nargs="+", required=True,
                   help="name=path per arm, e.g. control=/r/0.json target=/r/768.json")
    c.set_defaults(fn=cmd_compare)
    ns = ap.parse_args()
    return getattr(ns, "fn", _self_check)()


if __name__ == "__main__":
    # __main__ guard carries an assert so the scripts closure audit's set7
    # recognizes this hermetic self-check; bare invocation must exit 0.
    if {"--self-check", "selfcheck"} & set(sys.argv[1:]) or len(sys.argv) == 1:
        rc = _self_check()
        assert rc == 0
        raise SystemExit(rc)
    raise SystemExit(main())
