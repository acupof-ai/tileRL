"""Decode tok/s and tokens-per-trunk-forward across realistic workloads.

bench_b1_decode.py only measures "count to 600" — a degenerate task whose next
token any draft head predicts, so its numbers say nothing about serving. Same
two-point slope here (prefill cancels), run over coding / dialogue / thinking /
long-context prompts, with counting kept as the control.

The speculation headline is TOKENS PER TRUNK DECODE FORWARD, not
spec_accepted/spec_drafted. verify_lens truncates every chain on the draft's own
confidence before the counters see it (engine.py:1085 -> :1091), and a tick where
nothing survived skips _verify altogether (engine.py:795), so that ratio scores
the truncation policy's calibration, not the head — it reads ~1.00 even where the
head is in fact being rejected. tok/fwd cannot be gamed that way: truncating a
chain lowers it. Baseline is exactly 1.00 (speculation off); ceiling is
1 + spec_depth. Read it beside tok/s — a wider verify tick also costs more.

  python3 scripts/bench_workloads.py [--lo 32] [--hi 288] [--only coding]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

MODEL = "qwen38-27b"
FILLER = "The quick brown fox jumps over the lazy dog. "

DIALOGUE = [
    {"role": "user", "content": "I'm designing a paged KV cache. Block size 16 or 32?"},
    {
        "role": "assistant",
        "content": "16 wastes less padding on short sequences; 32 halves the block-table "
        "lookups and the page-table memory. For mixed traffic 16 is the safer default.",
    },
    {"role": "user", "content": "Traffic is mostly 2K-token chats behind a long system prompt."},
    {
        "role": "assistant",
        "content": "Then a shared prefix dominates and larger blocks pay off: the system "
        "prompt is reused block-aligned, and fewer blocks make the prefix hash cheaper.",
    },
    {
        "role": "user",
        "content": "Walk me through everything I'd change to move from 16 to 32, and which "
        "parts could break silently. Be specific and thorough, and cover each subsystem "
        "in turn: allocator, block table, prefix hashing, attention indexing, eviction.",
    },
]

WORKLOADS = [
    {
        "name": "coding",
        # Every workload must still be generating at --hi or the slope is
        # measured over a short run; the finish_reason gate below enforces it,
        # so these prompts deliberately ask for more than 288 tokens of output.
        "messages": [
            {
                "role": "user",
                "content": "Write a Python LRUCache class with O(1) get/put using a dict and a "
                "doubly linked list, then pytest tests covering eviction order, updating an "
                "existing key, and capacity 1. Include docstrings and type hints. Then add a "
                "TTLCache subclass with per-key expiry and its own tests, and finish with a "
                "paragraph on the thread-safety implications of each.",
            }
        ],
    },
    {"name": "dialogue", "messages": DIALOGUE},
    {"name": "thinking", "messages": DIALOGUE, "think": True},
    {
        "name": "long-ctx",
        "messages": [
            {
                "role": "user",
                # ~10 tokens per sentence, same rule as bench_long_context.py.
                "content": FILLER * 400 + "\nDescribe the document above in detail: what it "
                "says, how it is structured, what is unusual about it, and how you would "
                "compress it without losing information. Be thorough.",
            }
        ],
    },
    {  # CONTROL: degenerate, perfectly draftable — the contrast is the point.
        "name": "counting",
        "messages": [
            {"role": "user", "content": "Count from 1 to 600, separated by commas. Numbers only."}
        ],
    },
]

KEYS = (
    "decode_forwards",
    "mixed_forwards",
    "tokens_generated",
    "finished",
    "spec_drafted",
    "spec_accepted",
)


def chat(
    url: str, messages: list[dict], max_tokens: int, think: bool
) -> tuple[int, int, str, float]:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": think},
        }
    ).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    u = d["usage"]
    return (
        u["prompt_tokens"],
        u["completion_tokens"],
        d["choices"][0]["finish_reason"],
        time.perf_counter() - t0,
    )


def health(url: str) -> dict:
    with urllib.request.urlopen(url + "/health", timeout=30) as r:
        return json.loads(r.read())["stats"] or {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=32)
    ap.add_argument("--hi", type=int, default=288)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--only", help="run one workload by name")
    args = ap.parse_args()

    rows = []
    for w in WORKLOADS:
        if args.only and w["name"] != args.only:
            continue
        msgs, think = w["messages"], w.get("think", False)
        # Per-workload warmup, at --hi, never --lo: each workload has its own
        # chain-width profile, and with a draft head every accepted-chain WIDTH
        # captures its own CUDA graph. A --lo warmup only reaches the widths of
        # the first --lo tokens and the rest are captured inside the timed lo
        # point — that inverts the slope and made 52.7 tok/s read as 1.3.
        # Greedy (temperature 0) makes the warmup retrace the hi run exactly,
        # and the lo trajectory is its prefix, so this covers both timed points.
        chat(args.url, msgs, args.hi, think)
        h0 = health(args.url)
        pt, glo, flo, tlo = chat(args.url, msgs, args.lo, think)
        _, ghi, fhi, thi = chat(args.url, msgs, args.hi, think)
        h1 = health(args.url)
        d = {k: h1.get(k, 0) - h0.get(k, 0) for k in KEYS}
        # EOS before the cap silently shortens the slope: ghi > glo still holds
        # when a coding prompt stops at 40 of 288 tokens, and an 8-token delta
        # is noise. Demand both points actually hit their cap.
        if flo != "length" or fhi != "length":
            raise SystemExit(
                f"{w['name']}: EOS before the cap ({glo}/{args.lo} {flo}, {ghi}/{args.hi} {fhi}); "
                "the slope would be measured over a short run — lengthen the prompt"
            )
        if thi <= tlo:
            raise SystemExit(f"{w['name']}: hi faster than lo ({thi:.2f}s <= {tlo:.2f}s), noisy")
        # /health is engine-global: any other client in the window corrupts
        # every delta. Our two requests and nothing else. A mixed tick bumps
        # decode_forwards but never speculates (engine.py:790), so it would
        # dilute tok/fwd — with strictly sequential requests there are none.
        if d["finished"] != 2 or d["mixed_forwards"]:
            raise SystemExit(
                f"{w['name']}: window not isolated (finished={d['finished']} != 2, "
                f"mixed={d['mixed_forwards']}); other traffic hit the server"
            )
        rate = (ghi - glo) / (thi - tlo)
        # The honest speedup. Engine-side counters, not the client's token
        # counts, so numerator and denominator come from the same ledger. Each
        # request's FIRST output token is sampled by its prefill forward
        # (engine.py:881), not a decode forward, so drop one per finished
        # request or tok/fwd reads high.
        per_fwd = (d["tokens_generated"] - d["finished"]) / max(d["decode_forwards"], 1)
        print(
            f"{w['name']}: prompt_tok={pt} {glo}tok={tlo:.2f}s {ghi}tok={thi:.2f}s "
            f"forwards={d['decode_forwards']} "
            # NOT an acceptance rate — post-verify_lens, see the module docstring.
            f"submitted_match={d['spec_accepted']}/{d['spec_drafted']}"
        )
        rows.append((w["name"], pt, rate, per_fwd))

    if not rows:
        raise SystemExit(f"no workload named {args.only!r}")
    print(f"\n{'workload':<10} {'prompt_tok':>10} {'tok/s':>8} {'tok/fwd':>8}")
    for name, pt, rate, per_fwd in rows:
        print(f"{name:<10} {pt:>10} {rate:>8.1f} {per_fwd:>8.2f}")
    print("tok/fwd: 1.00 = no speculation, ceiling 1 + spec_depth.")


if __name__ == "__main__":
    main()
