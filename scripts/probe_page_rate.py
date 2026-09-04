"""Two ways to divide the same request, on the same host, in the same run.

The page's number and the earlier 39-40 tok/s differ by 1.257x while acceptance moved
1.010x, so the gap is a measurement口径, not the engine. This prints both windows per
request so the difference is attributed rather than guessed:

  decode  = tokens / (last_frame - first_frame)   what the page shows
  wall    = tokens / (last_frame - request_sent)  prefill and queueing folded in

Run on the pod: from the Mac, RTT lands inside both windows and neither number is the
engine's.
"""

import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
BODY = {
    "messages": [{"role": "user", "content": "Explain speculative decoding in three sentences."}],
    "max_tokens": 200,
    "stream": True,
    "stream_options": {"include_usage": True},
    "chat_template_kwargs": {"enable_thinking": True},
    "temperature": 0.0,
}


def one(max_tokens=200):
    body = dict(BODY, max_tokens=max_tokens)
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    sent = time.perf_counter()
    first = None
    usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
                continue
            if (obj.get("choices") or [{}])[0].get("delta", {}).get("content") and first is None:
                first = time.perf_counter()
    end = time.perf_counter()
    assert first is not None and usage, "no content frames arrived"
    n = usage["completion_tokens"]
    return (first - sent) * 1000, n, n / (end - first), n / (end - sent)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "3"
    if arg == "--sweep":
        # Decode rate is not one number: the tick grows with context (#47/#57), so a
        # rate quoted without its length is not comparable to another one.
        for m in (200, 400, 800, 1600):
            _, n, decode, _ = one(m)
            print(f"max_tokens {m:5d}  got {n:5d}  decode {decode:.1f} tok/s")
        raise SystemExit
    rows = [one() for _ in range(int(arg))]
    for i, (ttft, n, decode, wall) in enumerate(rows):
        print(f"run{i}: ttft {ttft:.0f}ms  tokens {n}  decode {decode:.1f}  wall {wall:.1f} tok/s")
    dec = sorted(r[2] for r in rows)[len(rows) // 2]
    wal = sorted(r[3] for r in rows)[len(rows) // 2]
    print(f"median decode {dec:.1f}  wall {wal:.1f} tok/s  ratio {dec / wal:.3f}x")
