"""N conversations interleaved: the workload a snapshot tier is supposed to win.

A single conversation re-reads only its newest prefix, so the LRU entry a demotion picks
is never asked for again — measured on the V100, 43 demotions and 0 promotions, with the
wall clock 1.51x worse than no tier at all. That makes the tier pure overhead there.

Interleaving A1 B1 A2 B2 ... changes it: while B is being served, A's entries age to the
LRU end and get demoted, and A's next turn asks for one back. If the tier does not win
here it does not win anywhere, so this is the arm that decides whether it ships.

`--sessions` exists because the tier's condition is `sessions > snapshot budget` and the
published verdict swept only the budget, holding sessions at 2. A one-axis sweep finds a
threshold and a threshold reads like a law; the other axis reversed it, 0/63 -> 24/0 hits.
Default stays 2 so the published arm reproduces.

  python scripts/bench_chat_interleaved.py --turns 4 --grow 40                # published arm
  python scripts/bench_chat_interleaved.py --turns 4 --grow 40 --sessions 12  # the other axis
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


def _fillers(n: int) -> list[str]:
    """`n` fillers whose FIRST tokens differ, so each conversation gets its own prefix hash.

    Prefixes are hashed from token 0, so beyond the two distinct topics below, `n` copies of
    one topic would collide into a single entry and the hit rate would measure the fixture
    rather than the tier. Past that point each filler is index-prefixed.

    `n <= 2` returns the topics UNCHANGED, byte for byte, so the published two-session arm
    reproduces -- an index prefix there would move every prompt length and every prefix hash,
    and the arm on record could not be re-run. The asymmetry costs little across arms: the
    prefix is 16 characters and appears once, while the filler after it is repeated
    `grow * (turn + 1)` times.
    """
    if n <= len(_TOPICS):
        return list(_TOPICS[:n])
    return [f"Session {i} of {n}. {_TOPICS[i % len(_TOPICS)]}" for i in range(n)]


def _label(i: int) -> str:
    return chr(ord("A") + i) if i < 26 else f"S{i}"


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
    ap.add_argument("--sessions", type=int, default=2,
                    help="interleaved conversations; 2 reproduces the published arm")
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--grow", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()
    if args.sessions < 1:
        ap.error("--sessions must be >= 1")

    fillers = _fillers(args.sessions)
    assert len(set(fillers)) == args.sessions, "fillers collide, so sessions share a prefix"
    convs: list[list[dict]] = [[] for _ in fillers]
    rows = []
    for turn in range(args.turns):
        for c, filler in enumerate(fillers):
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
            rows.append({"turn": turn, "conv": _label(c), "prompt_tokens": n,
                         "wall_s": round(wall, 2), **d})
            print(
                f"turn {turn} conv {_label(c)}  prompt={n:6d}  wall={wall:8.2f}s  "
                f"hits={d['prefix_hits']}  demote={d['dram_demotions']}  "
                f"promote={d['dram_promotions']}  evict={d['prefix_evictions']}",
                flush=True,
            )

    st = _get(f"{args.url}/health")
    total = round(sum(r["wall_s"] for r in rows), 2)
    # Per session, not a mean: a mean hides both a fixture collision (one session takes
    # every hit) and the case where the tier pays off for one conversation only.
    per_session = {
        _label(c): {
            "prefix_hits": sum(r["prefix_hits"] for r in rows if r["conv"] == _label(c)),
            "dram_promotions": sum(r["dram_promotions"] for r in rows if r["conv"] == _label(c)),
            "dram_demotions": sum(r["dram_demotions"] for r in rows if r["conv"] == _label(c)),
        }
        for c in range(args.sessions)
    }
    for label, v in per_session.items():
        print(f"session {label}: hits={v['prefix_hits']} promote={v['dram_promotions']} "
              f"demote={v['dram_demotions']}", flush=True)
    print(json.dumps({"sessions": args.sessions, "turns": args.turns, "rows": rows,
                      "per_session": per_session, "total_wall_s": total,
                      "final_stats": st}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
