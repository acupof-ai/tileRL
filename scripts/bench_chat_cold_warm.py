"""One prompt, one arm per server restart: the only variable is what the store holds.

`bench_chat_reuse.py` measures turn N against turn N-1, which changes the prompt as
well as the store. This serves ONE target prompt, either with an empty store (`--arm
cold`) or after its conversation head alone has been served (`--arm warm`).

Both arms MUST run against a freshly restarted server. Running them in one process
does not work: the cold arm publishes entries covering its whole prompt, so a warm arm
after it matches its own earlier self and reports the speedup of re-sending an
identical prompt (measured: 10.3x) rather than of multi-turn reuse.

  python scripts/bench_chat_cold_warm.py --arm cold --grow 40   # restart, then
  python scripts/bench_chat_cold_warm.py --arm warm --grow 40
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


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10.0) as r:
        return json.loads(r.read())["stats"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--arm", choices=("cold", "warm"), required=True)
    ap.add_argument("--grow", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    head = {"role": "user", "content": _FILLER * args.grow}
    tail = {"role": "user", "content": _FILLER * args.grow * 2}
    target = [head, {"role": "assistant", "content": "ok"}, tail]

    def serve(msgs: list[dict], label: str) -> tuple[float, int, dict]:
        before = _get(f"{args.url}/health")
        t0 = time.perf_counter()
        out = _post(
            f"{args.url}/v1/chat/completions",
            {"model": "qwen38-27b", "messages": msgs,
             "max_tokens": args.max_tokens, "temperature": 0.0},
            args.timeout,
        )
        wall = time.perf_counter() - t0
        after = _get(f"{args.url}/health")
        n = out.get("usage", {}).get("prompt_tokens", 0)
        d = {k: after[k] - before[k] for k in ("prefix_hits", "prefix_published")}
        print(f"{label:14s} prompt={n:6d}  wall={wall:8.2f}s  hits={d['prefix_hits']}  "
              f"published={d['prefix_published']}", flush=True)
        return wall, n, d

    st = _get(f"{args.url}/health")
    assert st["prefix_published"] == 0, (
        f"the store already holds {st['prefix_published']} publishes; restart the server "
        f"before this arm or it measures a contaminated store"
    )
    if args.arm == "warm":
        serve([head], "warm-up(head)")
    wall, n, d = serve(target, args.arm)
    print(json.dumps({
        "arm": args.arm, "prompt_tokens": n, "wall_s": round(wall, 2),
        "ms_per_token": round(wall * 1000 / n, 2), "hits": d["prefix_hits"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
