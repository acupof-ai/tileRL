"""The served decode rate over HTTP, from the engine's own forward counter.

Every other serve probe here drives the engine in-process (`prof_serve_gap.py`,
`prof_serve_ramp.py`, `probe_sse_deltas.py`). This one goes through the socket,
because that is the path a user hits and it is where three separate instruments
gave a wrong number in one sitting:

  * **wall_ms / tokens charges prefill to decode.** #24 measured prefill at
    31 ms/prompt token, ~310 ms of a 1650 ms request, against a bench figure
    (35.56 ms/tick) that is a decode-only window by construction. That
    comparison read as a 15% serve regression that does not exist.
  * **`curl -N` buffers the SSE body**, so time-to-first-chunk comes back 0.
  * **Counting SSE frames measures the poll period, not ticks.** `server.py`'s
    stream loop sleeps 0.02 s between peeks, and 76 tokens arriving in 39 frames
    is 1.95 tok/frame — close enough to the real 1.88 tok/forward to look like a
    tick count while being an artifact of the sleep.

What is attributable: `/health` carries cumulative `decode_forwards`,
`prefill_forwards` and `mixed_forwards`. Bracketing ONE request gives that
request's forwards, so tok/forward and acceptance are the engine's own numbers
rather than derived from the clock, and `prefill_forwards == 1` proves the
prefill sits outside the decode window instead of being assumed to.

    python3 scripts/probe_served_rate.py 127.0.0.1 8000 "数到二十" 128
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request


def main() -> None:
    host, port, prompt, maxtok = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    base = f"http://{host}:{port}"

    def health() -> dict:
        with urllib.request.urlopen(f"{base}/health", timeout=30) as r:
            return json.load(r)["stats"]

    body = json.dumps({"model": "qwen38-27b", "temperature": 0, "max_tokens": maxtok,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=body,
                                headers={"Content-Type": "application/json"})

    s0 = health()
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        # strict=False: the reply carries raw newlines inside the JSON string, which
        # json.loads rejects by default -- and the failure looks like a server bug.
        out = json.loads(r.read().decode(), strict=False)
    wall = (time.perf_counter() - t0) * 1000
    s1 = health()

    d = {k: s1[k] - s0[k] for k in
         ("decode_forwards", "prefill_forwards", "mixed_forwards", "spec_drafted",
          "spec_accepted", "tokens_generated")}
    fwd = max(d["decode_forwards"], 1)
    n = out["usage"]["completion_tokens"]

    print(f"prompt_tokens    {out['usage']['prompt_tokens']}")
    print(f"completion       {n}")
    print(f"decode_forwards  {d['decode_forwards']}")
    print(f"prefill_forwards {d['prefill_forwards']}")
    print(f"mixed_forwards   {d['mixed_forwards']}")
    print(f"tok_per_forward  {n / fwd:.3f}")
    print(f"drafts/forward   {d['spec_drafted'] / fwd:.3f}")
    print(f"acceptance       {d['spec_accepted'] / max(d['spec_drafted'], 1):.3f}")
    print(f"wall_ms          {wall:.0f}   (prefill + decode + HTTP + tokenize)")
    print(f"end_to_end_tok_s {1000 * n / wall:.1f}")
    # Everything outside a decode forward is lumped as overhead rather than split:
    # this instrument cannot see inside it, and naming a split it cannot measure is
    # how the three errors above happened.
    tick = 35.56  # wins/2026-09-04-depth-4-stalls-...: depth 1, B=1, ctx=1024, eager
    print(f"\nagainst bench's {tick} ms/tick: decode {tick * fwd:.0f} ms, "
          f"other {wall - tick * fwd:.0f} ms ({(wall - tick * fwd) / wall * 100:.0f}%)")


if __name__ == "__main__":
    main()
