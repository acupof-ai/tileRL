#!/usr/bin/env python3
"""PROBE-ONLY #063455: client-side streaming smoothness measurement.

Hits /v1/chat/completions with stream=true and records, per SSE frame that
carries text (content OR reasoning_content), the wall arrival time and the
delta. Stdlib only. Bracket /health around the request for the engine's own
decode_forwards / tokens_generated / spec counters, which settle whether a
frame carries one or two committed tokens (the client cannot tokenize a delta
exactly). Reports inter-arrival p50/p90/p99/max, the >150 ms stalls with
their frame indices (period ~ every 32 forwards = sparse eager refresh), and
frames vs forwards. Run on the serving host (localhost) to drop network jitter.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def health(base: str) -> dict:
    with urllib.request.urlopen(base + "/health", timeout=30) as r:
        return json.load(r)["stats"]


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else 0.0


def run(base: str, prompt: str, max_tokens: int, thinking: bool, label: str) -> None:
    body = {
        "model": "qwen38-27b",
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
    }
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    s0 = health(base)
    t0 = time.perf_counter()
    ttft = None
    frames: list[tuple[float, int, str]] = []  # (arrival, chars, kind)
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            pl = line[5:].strip()
            if pl == "[DONE]":
                break
            try:
                ch = json.loads(pl)
            except json.JSONDecodeError:
                continue
            for c in ch.get("choices", []):
                d = c.get("delta") or {}
                piece = d.get("content") or d.get("reasoning_content")
                if piece:
                    now = time.perf_counter() - t0
                    if ttft is None:
                        ttft = now
                    kind = "r" if d.get("reasoning_content") else "c"
                    frames.append((now, len(piece), kind))
    wall = time.perf_counter() - t0
    s1 = health(base)
    gap_keys = (
        "decode_forwards",
        "prefill_forwards",
        "spec_drafted",
        "spec_accepted",
        "tokens_generated",
    )
    dd = {k: s1[k] - s0[k] for k in gap_keys}

    arr = [f[0] for f in frames]
    gaps = [arr[i] - arr[i - 1] for i in range(1, len(arr))]
    gaps_ms = [g * 1000 for g in gaps]
    stalls = [(i, round(gaps_ms[i - 1], 1)) for i in range(1, len(arr)) if gaps_ms[i - 1] > 150]
    stall_idx = [i for i, _ in stalls]
    stall_gap_diffs = [stall_idx[j] - stall_idx[j - 1] for j in range(1, len(stall_idx))]
    char_hist: dict[int, int] = {}
    for _, nch, _ in frames:
        char_hist[nch] = char_hist.get(nch, 0) + 1
    fwd = max(dd["decode_forwards"], 1)
    print(f"===== {label} =====")
    print(
        f"frames={len(frames)} ttft_ms={ttft * 1000:.0f} wall_s={wall:.1f} "
        f"decode_fwd={fwd} tokens_gen={dd['tokens_generated']} "
        f"prefill_fwd={dd['prefill_forwards']}"
    )
    print(
        f"wall_tok_s={dd['tokens_generated'] / wall:.2f} "
        f"tok/fwd={dd['tokens_generated'] / fwd:.3f} "
        f"accept={dd['spec_accepted'] / max(dd['spec_drafted'], 1):.3f} "
        f"frames/fwd={len(frames) / fwd:.3f}"
    )
    if gaps_ms:
        print(
            f"gap_ms p50={pct(gaps_ms, 0.5):.1f} p90={pct(gaps_ms, 0.9):.1f} "
            f"p99={pct(gaps_ms, 0.99):.1f} max={max(gaps_ms):.1f} mean="
            f"{statistics.mean(gaps_ms):.1f}"
        )
    print(f"stalls>150ms: n={len(stalls)} idx={stall_idx[:24]}")
    print(f"stall spacings(frames): {stall_gap_diffs[:24]}")
    print(f"per-frame char len hist(top): {sorted(char_hist.items(), key=lambda x: -x[1])[:8]}")
    kinds = {}
    for _, _, k in frames:
        kinds[k] = kinds.get(k, 0) + 1
    print(f"frame kinds={kinds}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    short = "用大约200字解释为什么天空是蓝色的，并再给一个相关的生活例子。"
    run(args.base, short, 1024, True, "short thinking ON 1024")
    run(args.base, short, 512, False, "short thinking OFF 512")
    with open("/home/chenkailun.c/serve805_prompt0.txt") as f:
        long_prompt = f.read()
    run(args.base, long_prompt, 512, True, "long 37.6k 512")


if __name__ == "__main__":
    main()
