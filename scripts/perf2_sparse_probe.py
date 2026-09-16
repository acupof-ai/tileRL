"""Sparse decode probe: TTFT + steady-state decode tok/s for one streaming request.

Builds the prompt to an EXACT token count with the checkpoint tokenizer (filler
words compress under chat templates, so a char count is not a token count), so
the context actually crosses --sparse-min-tokens into the sparse path.
Times the first SSE choice chunk (content or reasoning) as TTFT.

Usage: python scripts/perf2_sparse_probe.py --base-url http://127.0.0.1:8011 \
       --prompt-tokens 9000 --max-tokens 128 --runs 3
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

CKPT = os.environ.get("TILERL_QWEN38_SOURCE", "/work/tilerl-ckpt/Qwen3.8-27B-NVFP4")


def build_prompt(n_tok: int) -> str:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(CKPT, "tokenizer.json"))
    words = " quantum topology manifold gradient tensor attention entropy synthesis"
    ids = tok.encode(words * (n_tok // 12 + 4)).ids[:n_tok]
    return tok.decode(ids)


def run(base: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": "qwen38-27b",
        "messages": [{"role": "user", "content": prompt + "\n\nContinue with a long factual paragraph."}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if not chunk.get("choices"):
                usage = chunk.get("usage")
                continue
            ch = chunk["choices"][0]
            delta = ch.get("delta", {})
            text = delta.get("content") or delta.get("reasoning_content")
            if text and ttft is None:
                ttft = time.perf_counter() - t0
    wall = time.perf_counter() - t0
    ntok = (usage or {}).get("completion_tokens")
    ptok = (usage or {}).get("prompt_tokens")
    decode_s = wall - ttft if ttft else None
    return {"prompt_tokens": ptok, "completion_tokens": ntok,
            "ttft_s": round(ttft, 3) if ttft else None,
            "decode_tok_s": round((ntok - 1) / decode_s, 2) if decode_s and ntok else None,
            "wall_s": round(wall, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--prompt-tokens", type=int, default=9000)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()
    prompt = build_prompt(args.prompt_tokens)
    for i in range(args.runs):
        print(json.dumps({"run": i, "prompt_arg": args.prompt_tokens,
                          **run(args.base_url, prompt, args.max_tokens)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
