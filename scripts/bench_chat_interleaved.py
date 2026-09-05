"""Two conversations interleaved: the workload a snapshot tier is supposed to win.

A single conversation re-reads only its newest prefix, so the LRU entry a demotion picks
is never asked for again — measured on the V100, 43 demotions and 0 promotions, with the
wall clock 1.51x worse than no tier at all. That makes the tier pure overhead there.

Interleaving A1 B1 A2 B2 ... changes it: while B is being served, A's entries age to the
LRU end and get demoted, and A's next turn asks for one back. If the tier does not win
here it does not win anywhere, so this is the arm that decides whether it ships.

  python scripts/bench_chat_interleaved.py --turns 4 --grow 40
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

_TOPICS = (
    "Explain in detail how a paged key-value cache serves a transformer decode step, "
    "including how block tables map logical positions to physical pages. ",
    "Describe how a gated delta network keeps recurrent state across a chunked prefill, "
    "and what the conv window carries between chunks. ",
)


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10.0) as r:
        return json.loads(r.read())["stats"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--grow", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    convs: list[list[dict]] = [[] for _ in _TOPICS]
    rows = []
    for turn in range(args.turns):
        for c, filler in enumerate(_TOPICS):
            convs[c].append({"role": "user", "content": filler * args.grow * (turn + 1)})
            before = _get(f"{args.url}/health")
            t0 = time.perf_counter()
            out = _post(
                f"{args.url}/v1/chat/completions",
                {"model": "qwen38-27b", "messages": convs[c],
                 "max_tokens": args.max_tokens, "temperature": 0.0},
                args.timeout,
            )
            wall = time.perf_counter() - t0
            after = _get(f"{args.url}/health")
            convs[c].append(
                {"role": "assistant", "content": out["choices"][0]["message"]["content"]}
            )
            d = {
                k: after.get(k, 0) - before.get(k, 0)
                for k in ("prefix_hits", "prefix_published", "prefix_evictions",
                          "dram_demotions", "dram_promotions")
            }
            n = out.get("usage", {}).get("prompt_tokens", 0)
            rows.append({"turn": turn, "conv": "AB"[c], "prompt_tokens": n,
                         "wall_s": round(wall, 2), **d})
            print(
                f"turn {turn} conv {'AB'[c]}  prompt={n:6d}  wall={wall:8.2f}s  "
                f"hits={d['prefix_hits']}  demote={d['dram_demotions']}  "
                f"promote={d['dram_promotions']}  evict={d['prefix_evictions']}",
                flush=True,
            )

    st = _get(f"{args.url}/health")
    total = round(sum(r["wall_s"] for r in rows), 2)
    print(json.dumps({"rows": rows, "total_wall_s": total, "final_stats": st}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
