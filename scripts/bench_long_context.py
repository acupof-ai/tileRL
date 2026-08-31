"""Long-context benchmark through the running server: TTFT + decode tok/s at
1K/2K/4K prompt tokens, B=1. Two requests per length: max_tokens=1 gives
TTFT (prefill + 1 decode), max_tokens=33 gives decode rate from the delta.

  python3 scripts/bench_long_context.py
"""

import json
import time
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "qwen38-27b"

# One sentence ~10 tokens; repeat to hit the target. The server reports the
# real prompt_tokens, so the target is approximate.
FILLER = "The quick brown fox jumps over the lazy dog. "


def make_prompt(target_tokens: int) -> str:
    reps = max(1, target_tokens // 10)
    return FILLER * reps + "\nSummarize the above in one sentence."


def one(prompt: str, max_tokens: int) -> tuple[int, float]:
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
    return d["usage"]["prompt_tokens"], time.perf_counter() - t0


def main() -> None:
    print(f"{'prompt_tok':>10} {'ttft_s':>8} {'decode_t/s':>11} {'wall_33_s':>10}")
    for target in (1024, 2048, 4096):
        prompt = make_prompt(target)
        pt, t1 = one(prompt, 1)  # TTFT ≈ prefill + 1 decode
        _, t33 = one(prompt, 33)  # prefill + 33 decode
        dec = 32 / max(t33 - t1, 0.01)
        print(f"{pt:>10} {t1:>8.1f} {dec:>11.1f} {t33:>10.1f}")


if __name__ == "__main__":
    main()
