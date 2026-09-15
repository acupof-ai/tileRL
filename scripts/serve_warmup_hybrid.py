#!/usr/bin/env python3
"""Both-mode warmup for scripts/serve_hybrid_v100.sh.

JIT and graph capture happen on first use: one dense request, one sparse
(longer than --sparse-min-tokens), and a four-way batch so the B=4 decode path
is compiled before traffic arrives. Pure HTTP; base/model come from the
environment so nothing here names a host.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import time
import urllib.request
import uuid

BASE = os.environ.get("LIVENESS_BASE", "http://127.0.0.1:8000")
MODEL = os.environ.get("LIVENESS_MODEL", "qwen38-27b")
# Sparse threshold must stay above the launcher's --sparse-min-tokens (8192).
DENSE_TOKENS = int(os.environ.get("WARMUP_DENSE_TOKENS", "7000"))
SPARSE_TOKENS = int(os.environ.get("WARMUP_SPARSE_TOKENS", "9000"))

_BODY = (
    "The quarterly systems review covered scheduler latency, block allocation, prefix "
    "sharing, cold tier migration, attention kernels, quantized weights, draft heads, "
    "graph capture, memory ledgers and capacity planning. "
)


def prompt(n):
    return f"Reference code {uuid.uuid4().hex}. " + _BODY * (n // 90 + 1)


def send(i, p):
    d = {
        "model": MODEL,
        "messages": [{"role": "user", "content": p + f" Session {i}: ok?"}],
        "temperature": 0,
        "max_tokens": 8,
        "enable_thinking": False,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(d).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        json.load(r)


def main():
    t0 = time.time()
    send(0, prompt(DENSE_TOKENS))
    print(f"warmup: dense-{DENSE_TOKENS} {time.time() - t0:.1f}s", flush=True)
    t1 = time.time()
    send(1, prompt(SPARSE_TOKENS))
    print(f"warmup: sparse-{SPARSE_TOKENS} {time.time() - t1:.1f}s", flush=True)
    t2 = time.time()
    with cf.ThreadPoolExecutor(4) as ex:
        list(ex.map(send, range(4), [prompt(2000)] * 4))
    print(f"warmup: 4x2k {time.time() - t2:.1f}s total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
