"""B=1 decode rate through the server, two-point with a large token delta.

Streaming is not incremental (server._stream emits one delta at the end), and a
small delta (1 vs 65) is swamped by prefill variance. So: run the same prompt at
two max_tokens values far apart and take the slope — the prefill term cancels.

  python3 scripts/bench_b1_decode.py [--ctx 1024] [--lo 32] [--hi 288]
"""

import argparse
import json
import time
import urllib.request

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
    args = ap.parse_args()

    prompt = (FILLER * max(1, args.ctx // 10) + "\n" + TASK) if args.ctx else TASK
    pt, glo, tlo = one(prompt, args.lo)
    _, ghi, thi = one(prompt, args.hi)
    if ghi <= glo:
        raise SystemExit(f"EOS too early: {glo} -> {ghi} tokens; use a longer-output task")
    rate = (ghi - glo) / max(thi - tlo, 1e-3)
    print(f"prompt_tok={pt}  {glo}tok={tlo:.2f}s  {ghi}tok={thi:.2f}s")
    print(f"decode={rate:.1f} tok/s  ({1000 / rate:.0f} ms/tok)  over {ghi - glo} tokens")


if __name__ == "__main__":
    main()
