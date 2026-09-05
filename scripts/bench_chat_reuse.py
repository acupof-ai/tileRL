"""Multi-turn chat prefix reuse against a running server: per-turn wall time and the
store's own hit counters.

A chat client resends the whole conversation each turn, so turn N's prompt is turn
N-1's prompt plus the assistant reply plus the new user text. That shared span is
what the prefix store exists for. Run it against a live `tilerl serve`:

  python scripts/bench_chat_reuse.py --url http://localhost:8000 --turns 6
"""

from __future__ import annotations

import argparse
import json
import time
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
    args = ap.parse_args()

    msgs: list[dict] = []
    rows = []
    for turn in range(args.turns):
        msgs.append({"role": "user", "content": _FILLER * args.grow * (turn + 1)})
        before = _get(f"{args.url}/health")["stats"]
        t0 = time.perf_counter()
        out = _post(
            f"{args.url}/v1/chat/completions",
            {
                "model": "qwen38-27b",
                "messages": msgs,
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
            },
            args.timeout,
        )
        wall = time.perf_counter() - t0
        after = _get(f"{args.url}/health")["stats"]
        reply = out["choices"][0]["message"]["content"]
        msgs.append({"role": "assistant", "content": reply})
        usage = out.get("usage", {})
        rows.append(
            {
                "turn": turn,
                "prompt_tokens": usage.get("prompt_tokens"),
                "wall_s": round(wall, 2),
                "hits": after["prefix_hits"] - before["prefix_hits"],
                "published": after["prefix_published"] - before["prefix_published"],
                "prefill_forwards": after["prefill_forwards"] - before["prefill_forwards"],
            }
        )
        r = rows[-1]
        print(
            f"turn {turn}  prompt={r['prompt_tokens']:6}  wall={r['wall_s']:8.2f}s  "
            f"hits={r['hits']}  published={r['published']}  "
            f"prefills={r['prefill_forwards']}",
            flush=True,
        )

    st = _get(f"{args.url}/health")["stats"]
    print(json.dumps({"rows": rows, "final_stats": st}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
