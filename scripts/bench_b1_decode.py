"""B=1 decode rate through the server, two-point with a large token delta.

Streaming is not incremental (server._stream emits one delta at the end), and a
small delta (1 vs 65) is swamped by prefill variance. So: run the same prompt at
two max_tokens values far apart and take the slope — the prefill term cancels.

  python3 scripts/bench_b1_decode.py --build fused+graph --card 6 [--ctx 1024] [--lo 32] [--hi 288]

Emits one decode_tok_s record to docs/experience/bench/measurements.jsonl
(schema: docs/bench-schema.md). --build is required: a client cannot see the
server's build, and eager vs fused+graph is 6.4x on this metric.
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchrec  # noqa: E402

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "qwen38-27b"
FILLER = "The quick brown fox jumps over the lazy dog. "
# Must not hit EOS before --hi, or the slope is measured over a short run.
TASK = "Count from 1 to 600, separated by commas. Output only the numbers."


def one(prompt: str, max_tokens: int) -> tuple[int, int, float]:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    u = d["usage"]
    return u["prompt_tokens"], u["completion_tokens"], time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=0, help="approx prompt tokens (0 = short)")
    ap.add_argument("--lo", type=int, default=32)
    ap.add_argument("--hi", type=int, default=288)
    ap.add_argument("--repeat", type=int, default=3, help="slope repetitions for spread")
    ap.add_argument("--build", required=True, choices=list(benchrec.BUILDS))
    ap.add_argument("--target", default="sm90", choices=list(benchrec.TARGETS))
    ap.add_argument("--device-name", default="H20")
    ap.add_argument("--card", type=int, required=True, help="GPU card the server runs on")
    ap.add_argument("--model-name", default="27B-nvfp4")
    args = ap.parse_args()

    prompt = (FILLER * max(1, args.ctx // 10) + "\n" + TASK) if args.ctx else TASK
    # Warmup to --hi: with a draft, each accepted-chain WIDTH captures its own
    # graph, and a short warmup leaves the rest to compile inside the timed lo
    # point (the slope inverted once: 52.7 tok/s read as 1.3).
    one(prompt, args.hi)
    rates = []
    pt = glo = ghi = 0
    for _ in range(args.repeat):
        pt, glo, tlo = one(prompt, args.lo)
        _, ghi, thi = one(prompt, args.hi)
        if ghi <= glo:
            raise SystemExit(f"EOS too early: {glo} -> {ghi} tokens; use a longer-output task")
        if thi <= tlo:
            raise SystemExit(f"hi ran faster than lo ({thi:.2f}s <= {tlo:.2f}s): unwarmed or noisy")
        rates.append((ghi - glo) / (thi - tlo))
    rates.sort()
    rate = rates[len(rates) // 2]
    spread = (max(rates) - min(rates)) / rate if rate else 0.0
    print(f"prompt_tok={pt}  {glo}tok vs {ghi}tok  x{args.repeat}")
    print(f"decode={rate:.1f} tok/s  ({1000 / rate:.0f} ms/tok)  spread {100 * spread:.1f}%")

    rid = benchrec.append({
        "metric": "decode_tok_s", "value": round(rate, 2), "unit": "tok/s",
        "target": args.target, "build": args.build, "model": args.model_name,
        "shape": {"batch": 1, "ctx": pt},
        "warm": {"state": "warm", "compiles": 0},
        "n": args.repeat, "spread": round(spread, 4),
        "device": {"name": args.device_name, "card": args.card},
        "sha": benchrec.git_sha(), "cmd": " ".join(sys.argv),
        "floor": {"value": 129.0, "unit": "tok/s", "kind": "roofline",
                  "derivation": "129 tok/s = 30.9 GB weights / 4 TB/s H20 HBM "
                                "(wins/2026-08-24-sota-all-levers.md)"},
    })
    print(f"record {rid} appended")


if __name__ == "__main__":
    main()
