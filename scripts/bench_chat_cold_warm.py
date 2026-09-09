"""One prompt, one arm per server restart: the only variable is what the store holds.

`bench_chat_reuse.py` measures turn N against turn N-1, which changes the prompt as
well as the store. This serves ONE target prompt, either with an empty store (`--arm
cold`) or after its conversation head alone has been served (`--arm warm`).

Both arms MUST run against a freshly restarted server. Running them in one process
does not work: the cold arm publishes entries covering its whole prompt, so a warm arm
after it matches its own earlier self and reports the speedup of re-sending an
identical prompt (measured: 10.3x) rather than of multi-turn reuse.

  python scripts/bench_chat_cold_warm.py --arm cold --grow 40 \
      --build fused+graph --target sm90 --card 6   # restart, then
  python scripts/bench_chat_cold_warm.py --arm warm --grow 40 \
      --build fused+graph --target sm90 --card 6

Emits one chat_turn_wall_s record per arm (plus a prefix_hits row when the
warm arm hit) to docs/experience/bench/measurements.jsonl (schema:
docs/bench-schema.md). --build is required: a client cannot see the server's
build, and the arm only means something against a stated one.
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
    benchrec.add_record_args(ap, client_side=True)
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
    common = benchrec.record_common(args)
    rec = {
        "metric": "chat_turn_wall_s", "value": round(wall, 3), "unit": "s",
        "shape": {"turn": 0, "prompt_tokens": n, "arm": args.arm},
        "warm": {"state": args.arm, "compiles": 0},
        "n": 1, "spread": 0.0, **common,
    }
    rec["floor"] = benchrec.measured_best_floor(rec, lower_is_better=True)
    print(f"record {benchrec.append(rec)} appended", flush=True)
    if d["prefix_hits"] > 0:
        hrec = {
            "metric": "prefix_hits", "value": d["prefix_hits"], "unit": "hits",
            "shape": {"turn": 0, "arm": args.arm},
            "warm": {"state": "warm", "compiles": 0},
            "n": 1, "spread": 0.0, **common,
        }
        hrec["floor"] = {
            "value": 1.0, "unit": "hits", "kind": "baseline",
            "derivation": "1.0 = one block-boundary hit is the smallest nonzero "
                          "reuse; 0 hits means no reuse",
        }
        print(f"record {benchrec.append(hrec)} appended", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
