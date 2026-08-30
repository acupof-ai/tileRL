"""B=8 eval through the running server: 8 concurrent requests, aggregate t/s.

Hits the OpenAI-compatible /v1/chat/completions on localhost:8000 with 8
parallel requests and reports aggregate tok/s (total output tokens / wall
time from first send to last response). The server already has the model
loaded, so this needs no second model instance.

  python3 scripts/eval_b8_server.py
"""

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "qwen38-27b"

PROMPTS = [
    "The capital of France is",
    "The largest planet in the solar system is",
    "The speed of light is approximately",
    "The author of Romeo and Juliet is",
    "The chemical symbol for gold is",
    "The tallest mountain on Earth is",
    "The currency of Japan is",
    "The primary language spoken in Brazil is",
]


def one(prompt: str) -> tuple[int, str]:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    dt = time.perf_counter() - t0
    n = d["usage"]["completion_tokens"]
    return n, d["choices"][0]["message"]["content"], dt


t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(one, PROMPTS))
wall = time.perf_counter() - t0

total = sum(n for n, _, _ in results)
print(f"B=8  wall={wall:.1f}s  total_tokens={total}  aggregate={total / wall:.1f} tok/s")
for i, (n, text, dt) in enumerate(results):
    print(f"  req{i}: {n} tok in {dt:.1f}s -> {text[:60]!r}")
