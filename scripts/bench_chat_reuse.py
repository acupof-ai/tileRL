"""Multi-turn chat prefix reuse against a running server: per-turn wall time and the
store's own hit counters.

A chat client resends the whole conversation each turn, so turn N's prompt is turn
N-1's prompt plus the assistant reply plus the new user text. That shared span is
what the prefix store exists for. Run it against a live `tilerl serve`:

  python scripts/bench_chat_reuse.py --url http://localhost:8000 --turns 6 \
      --build fused+graph --card 6

Emits one chat_turn_wall_s record per turn plus prefix_hits rows to
docs/experience/bench/measurements.jsonl (schema: docs/bench-schema.md). The
reuse speedup (turn 1 / turn 2 wall) is a view, not a record -- a measurement
is one turn.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchrec  # noqa: E402

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


def _get(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--grow", type=int, default=3, help="filler sentences added per turn")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--build", required=True, choices=list(benchrec.BUILDS))
    ap.add_argument("--target", default="sm90", choices=list(benchrec.TARGETS))
    ap.add_argument("--device-name", default="H20")
    ap.add_argument("--card", type=int, required=True)
    ap.add_argument("--model-name", default="27B-nvfp4")
    args = ap.parse_args()

    common = {"target": args.target, "build": args.build, "model": args.model_name,
              "device": {"name": args.device_name, "card": args.card},
              "commit": benchrec.git_commit(), "dirty": benchrec.git_dirty(), "cmd": " ".join(sys.argv)}
    msgs: list[dict] = []
    for turn in range(args.turns):
        msgs.append({"role": "user", "content": _FILLER * args.grow * (turn + 1)})
        before = _get(f"{args.url}/health")["stats"]
        t0 = time.perf_counter()
        out = _post(
            f"{args.url}/v1/chat/completions",
            {"model": "qwen38-27b", "messages": msgs,
             "max_tokens": args.max_tokens, "temperature": 0.0},
            args.timeout,
        )
        wall = time.perf_counter() - t0
        after = _get(f"{args.url}/health")["stats"]
        reply = out["choices"][0]["message"]["content"]
        msgs.append({"role": "assistant", "content": reply})
        usage = out.get("usage", {})
        pt = usage.get("prompt_tokens")
        hits = after["prefix_hits"] - before["prefix_hits"]
        print(f"turn {turn}  prompt={pt:6}  wall={wall:8.2f}s  hits={hits}  "
              f"prefills={after['prefill_forwards'] - before['prefill_forwards']}", flush=True)

        shape = {"turn": turn, "prompt_tokens": pt}
        rec = {
            "metric": "chat_turn_wall_s", "value": round(wall, 3), "unit": "s",
            "shape": shape, "warm": {"state": "warm", "compiles": None},
            "n": 1, "spread": 0.0, **common,
        }
        rec["floor"] = benchrec.measured_best_floor(rec, lower_is_better=True)
        rid = benchrec.append(rec)
        if hits > 0:
            benchrec.append({
                "metric": "prefix_hits", "value": hits, "unit": "hits",
                "shape": {"turn": turn}, "warm": {"state": "warm", "compiles": None},
                "n": 1, "spread": 0.0,
                "floor": {"value": 1.0, "unit": "hits", "kind": "baseline",
                          "derivation": "1.0 = one block-boundary hit is the smallest nonzero reuse; "
                                        "0 hits means no reuse"},
                **common,
            })
        print(f"  record {rid} appended", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
