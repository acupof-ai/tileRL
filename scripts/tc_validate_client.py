#!/usr/bin/env python3
"""PROBE-ONLY: ThinkingCap vs old-model acceptance client for the V100 window.

Runs against a live server started from the production launcher (only the
checkpoint/draft lines changed). For each serve805 prompt it streams one
temperature-0 completion, timing the wall the same way the sweep does, and
reads /health before/after to get Δspec_accepted/Δspec_drafted (acceptance is
the main model-swap variable) and Δdecode_forwards for effective tok/s.

Per prompt it also emits cheap degeneracy signals (max single-token fraction,
longest repeated n-gram span) so the preset "degenerate/repetition -> do not
switch" is a number, not a vibe.

Usage:
  tc_validate_client.py --url http://127.0.0.1:8000 --prompts ~/serve805_prompts.jsonl \
      --tag old --out-prefix /tmp/tc/old
  # one prompt only (per-prompt fresh-service mode when #821 is absent):
  ... --only-prompt 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request


def http_json(url: str, payload: dict | None = None, timeout: float = 30.0):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def health_stats(url: str) -> dict:
    d = http_json(f"{url}/health", timeout=10)
    if d.get("status") != "ok":
        raise SystemExit(f"health not ok: {str(d)[:300]}")
    return d["stats"]


def load_prompts(path: str, want: int) -> list[list[int] | str]:
    rows = []
    with open(path) as f:
        for ln in f:
            if ln.strip():
                rows.append(json.loads(ln))
    out = [r.get("input_ids") if "input_ids" in r else r["text"] for r in rows]
    if len(out) < want:
        raise SystemExit(f"only {len(out)} prompts, need {want}")
    return out[:want]


def render_prompt(tokenizer, item) -> str:
    """32k serve805 prompts ship as input_ids; the chat route takes text. Decode
    (BPE round-trips for normal text) and let the server re-apply the chat
    template. Plain text passes through."""
    if isinstance(item, str):
        return item
    return tokenizer.decode([int(x) for x in item], skip_special_tokens=True)


def stream_chat(url: str, text: str, max_new: int, thinking_off: bool):
    """POST a streamed chat; return (assistant_text, wall_s, prompt_tokens,
    completion_tokens). SSE lines parsed without a third-party client."""
    payload = {
        "model": "qwen38-27b",
        "messages": [{"role": "user", "content": text}],
        "temperature": 0.0, "max_tokens": max_new, "stream": True,
        "stream_options": {"include_usage": True},
    }
    if thinking_off:
        payload["enable_thinking"] = False
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    pieces, usage = [], None
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            ch = json.loads(body)
            if ch.get("usage"):
                usage = ch["usage"]
            delta = (ch.get("choices") or [{}])[0].get("delta", {})
            if isinstance(delta, dict) and delta.get("content"):
                pieces.append(delta["content"])
    wall = time.perf_counter() - t0
    text_out = "".join(pieces)
    pt = usage.get("prompt_tokens") if usage else None
    ct = usage.get("completion_tokens") if usage else None
    return text_out, wall, pt, ct


def degeneracy(text: str) -> dict:
    toks = text.split()
    n = len(toks)
    max_frac = 0.0
    top_tok = None
    if n:
        counts = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        top_tok, c = max(counts.items(), key=lambda kv: kv[1])
        max_frac = round(c / n, 4)
    # longest token run repeated 3+ times consecutively
    run_tok, run_n, best, best_tok = None, 0, 0, None
    for t in toks:
        if t == run_tok:
            run_n += 1
        else:
            run_tok, run_n = t, 1
        if run_n > best:
            best, best_tok = run_n, t
    return {"words": n, "top_token_frac": max_frac, "top_token": top_tok,
            "longest_same_token_run": best, "run_token": best_tok}


def compare(old_json: str, tc_json: str) -> int:
    """Relative acceptance gate: ThinkingCap accept >= 0.8 x old model, on the
    same prompts measured the same way. Prints both absolutes and the in/post
    split; rc0 pass, rc1 fail."""
    with open(old_json) as f:
        old = json.load(f)
    with open(tc_json) as f:
        tc = json.load(f)
    a_old, a_tc = old["aggregate_accept_rate"], tc["aggregate_accept_rate"]
    ratio = (a_tc / a_old) if a_old else None

    def split(rows):
        di = sum(r["drafted_in"] for r in rows)
        ai = sum(r["accept_in"] for r in rows)
        dp = sum(r["drafted_post"] for r in rows)
        ap = sum(r["accept_post"] for r in rows)
        return {"in": round(ai / di, 4) if di else None,
                "post": round(ap / dp, 4) if dp else None,
                "drafted_in": di, "drafted_post": dp}

    deg = [{"i": r["i"], "tag": r["tag"], **r["degenerate"]}
           for r in tc["prompts"] if r["degenerate"]["top_token_frac"] >= 0.3
           or r["degenerate"]["longest_same_token_run"] >= 20]
    rep = {"old_accept": a_old, "thinkingcap_accept": a_tc,
           "tc_over_old": round(ratio, 4) if ratio is not None else None,
           "gate_threshold": 0.8, "pass": bool(ratio is not None and ratio >= 0.8),
           "old_eff_tok_s": old.get("aggregate_eff_tok_s"),
           "tc_eff_tok_s": tc.get("aggregate_eff_tok_s"),
           "split_old": split(old["prompts"]),
           "split_tc": split(tc["prompts"]),
           "degenerate_flags": deg}
    # Degeneracy is reported for human judgment, not auto-failed: the cutover
    # criterion "degenerate/repetition" is a reading of the outputs, and the
    # thresholds here are signals, not contracts.
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0 if rep["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2, metavar=("OLD_JSON", "TC_JSON"),
                    help="report mode: relative acceptance gate, no server")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--prompts", default=os.path.expanduser(
        "~/serve805_prompts.jsonl"))
    ap.add_argument("--n-prompts", type=int, default=6)
    ap.add_argument("--only-prompt", type=int, default=-1,
                    help="run a single prompt index (per-prompt restart mode)")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--thinking-off", action="store_true",
                    help="send enable_thinking:false explicitly")
    ap.add_argument("--short-chat", action="store_true",
                    help="bypass the prompt file: one fixed short question")
    ap.add_argument("--tag", default="", help="old | thinkingcap")
    ap.add_argument("--out-prefix", default="")
    args = ap.parse_args()
    if args.compare:
        return compare(args.compare[0], args.compare[1])

    tokenizer = None
    if not args.short_chat:
        from tilerl.cli import _qwen38_tokenizer

        tokenizer = _qwen38_tokenizer()
        prompts = load_prompts(args.prompts, args.n_prompts)
        if args.only_prompt >= 0:
            prompts = [(args.only_prompt, prompts[args.only_prompt])]
        else:
            prompts = list(enumerate(prompts))
    else:
        prompts = [(0, "用一句话介绍你自己，然后只给出 17*23 的数字答案。")]

    rows = []
    for idx, item in prompts:
        text = item if args.short_chat else render_prompt(tokenizer, item)
        before = health_stats(args.url)
        out, wall, pt, ct = stream_chat(
            args.url, text, args.max_new, args.thinking_off)
        after = health_stats(args.url)

        d_acc = int(after["spec_drafted"]) - int(before["spec_drafted"])
        a_acc = int(after["spec_accepted"]) - int(before["spec_accepted"])
        d_fwd = int(after["decode_forwards"]) - int(before["decode_forwards"])
        accept = (a_acc / d_acc) if d_acc else None
        eff = ((d_fwd + a_acc) / wall) if wall else None
        out_s = (ct / wall) if ct else None
        row = {"i": idx, "tag": args.tag,
               "thinking_off": args.thinking_off,
               "prompt_tokens": pt, "completion_tokens": ct,
               "wall_s": round(wall, 3),
               "spec_drafted": d_acc, "spec_accepted": a_acc,
               "decode_forwards": d_fwd,
               "accept_rate": round(accept, 4) if accept is not None else None,
               "accept_in": int(after.get("spec_accept_in", 0))
                           - int(before.get("spec_accept_in", 0)),
               "drafted_in": int(after.get("spec_drafted_in", 0))
                            - int(before.get("spec_drafted_in", 0)),
               "accept_post": int(after.get("spec_accept_post", 0))
                              - int(before.get("spec_accept_post", 0)),
               "drafted_post": int(after.get("spec_drafted_post", 0))
                              - int(before.get("spec_drafted_post", 0)),
               "eff_tok_s": round(eff, 3) if eff is not None else None,
               "output_tok_s": round(out_s, 3) if out_s is not None else None,
               "degenerate": degeneracy(out),
               "head": out[:200]}
        rows.append(row)
        print(f"[{args.tag}] prompt {idx}: {ct} tok in {wall:.1f}s "
              f"accept={row['accept_rate']} eff={row['eff_tok_s']} "
              f"topfrac={row['degenerate']['top_token_frac']}", flush=True)

    drafted = sum(r["spec_drafted"] for r in rows)
    accepted = sum(r["spec_accepted"] for r in rows)
    forwards = sum(r.get("decode_forwards", 0) for r in rows)
    walls = sum(r["wall_s"] for r in rows)
    summary = {"tag": args.tag, "prompts": rows,
               "aggregate_accept_rate": round(accepted / drafted, 4)
                   if drafted else None,
               "aggregate_spec_accepted": accepted,
               "aggregate_spec_drafted": drafted,
               "aggregate_eff_tok_s": round((forwards + accepted) / walls, 3)
                   if walls else None}
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    suffix = f"_{args.only_prompt}" if args.only_prompt >= 0 else ""
    with open(f"{args.out_prefix}{suffix}.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"SUMMARY {args.tag}{suffix} accept={summary['aggregate_accept_rate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
