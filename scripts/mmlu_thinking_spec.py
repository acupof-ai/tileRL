#!/usr/bin/env python3
"""MMLU thinking + speculative decode, dense vs sparse paired arms.

Same prompts, same seed, same engine settings in both arms; only sparse_k
differs (0 = dense, --k = bounds selection). Both arms load the MTP draft head
(``model_mtp.safetensors`` beside the checkpoint, spec_depth=1) and run with
thinking ON: the prompt opens <think>, the engine forces the close after the
think budget, and the answer letter is extracted from the continuation.

Per arm: post-think accuracy, mean output tokens, wall-clock tok/s, and spec
drafted/accepted (stats counters). A wall-clock guard (--deadline-min) starts
no new question past the deadline, so the run is bounded even if one long
thinker is in flight. Same prompt/seed subset as scripts/mmlu.py.

Usage (card):
  scripts/mmlu_thinking_spec.py SOURCE --n 100 --k 128 --gpu 7 \
      --draft model_mtp.safetensors --deadline-min 90
CPU gate (tiny; real MTP draft is card-only):
  scripts/mmlu_thinking_spec.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

from tilerl.engine import SamplingParams
from tilerl.eval import letter, mmlu_questions

CONCURRENCY = 8
MAX_NEW = 2048
MAX_THINK = 512
ANS_RE = re.compile(r"\b([ABCD])\b")


def answer_letter(text: str) -> str:
    """Letter from the post-</think> continuation; last standalone letter as a
    fallback when the close marker is missing (a truncated thinker)."""
    tail = text.split("</think>", 1)[1] if "</think>" in text else text[-200:]
    m = ANS_RE.search(tail)
    return m.group(1) if m else (letter(text) or "?")


def thinking_prompts(raw_prompts: list[str]) -> list[str]:
    """Wrap each 0-shot MMLU question in ChatML with the think block open."""
    from tilerl.prompt import render_chat

    out = []
    for p in raw_prompts:
        body = p.split("Answer:")[0].rstrip()
        out.append(render_chat([("user", body + "\nAnswer with one letter.")], True))
    return out


def chat_control_prompts(n: int) -> list[str]:
    """Plain ChatML prompts, think block NOT forced open — the 93%-acceptance
    chat regime. Same renderer, only the template differs."""
    from tilerl.prompt import render_chat

    msgs = [
        "Explain why the sky is blue in two sentences.",
        "Write a short greeting for a new team member.",
        "Summarize what a hash table is in one sentence.",
        "Name three primary colors.",
        "What is 17 times 5?",
        "Give one benefit of regular exercise.",
        "Define the word entropy briefly.",
        "Suggest a healthy breakfast in a few words.",
    ]
    return [render_chat([("user", msgs[i % len(msgs)])], False) for i in range(n)]


def _drain(engine, tok, prompts, sp, deadline_s, on_done=None):
    """Submit at most CONCURRENCY; stop starting questions past the deadline.
    Returns completions in prompt order. on_done(idx, text, n_done, elapsed_s)
    fires per completed question for live progress / partial flushes."""
    t0 = time.time()
    out: list = [None] * len(prompts)
    pending, todo = {}, list(enumerate(prompts))
    n_done = 0
    while pending or todo:
        while todo and len(pending) < CONCURRENCY:
            if deadline_s is not None and time.time() - t0 > deadline_s:
                todo.clear()
                break
            i, p = todo.pop()
            pending[engine.submit(tok.encode(p), sp)] = i
        engine.step()
        for wid, ids in engine.poll().items():
            idx = pending.pop(wid)
            text = tok.decode(ids)
            out[idx] = text
            n_done += 1
            if on_done is not None:
                on_done(idx, text, n_done, time.time() - t0)
    return out, time.time() - t0


def run_arm(source: str, prompts: list[str], k: int, draft_path: str | None,
            tok, backend, max_ctx: int, deadline_s: float | None,
            on_done=None, force_think: bool = True) -> dict:
    from tilerl.cli import _build_model
    from tilerl.engine import build_engine

    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True,
                              backend=backend)
    draft = None
    if draft_path:
        from tilerl.spec import load_draft

        draft = load_draft(model, draft_path)
    # Sparse drops pages to a host cold tier, so a sparse build needs a cold
    # budget (guarded in build_engine). 16 GiB; dense passes 0.
    engine_kw = dict(num_blocks=0, num_slots=CONCURRENCY + 2,
                     max_batch=CONCURRENCY, max_total_tokens=max_ctx,
                     max_num_batched_tokens=512, sparse_k=k, scorer="bounds",
                     draft=draft, spec_depth=1)
    if k:
        engine_kw["kv_cold_bytes"] = 1 << 34
    engine = build_engine(cfg, model, backend, **engine_kw)
    sp = SamplingParams(
        temperature=0.0, seed=0, max_new_tokens=MAX_NEW,
        **({"max_think_tokens": MAX_THINK,
            "end_think_ids": tuple(tok.encode("</think>\n\n"))} if force_think else {}),
        stop_token_ids=tuple(getattr(tok, "stop_token_ids", ())))
    try:
        texts, elapsed = _drain(engine, tok, prompts, sp, deadline_s, on_done)
        stats = engine.stats()
    finally:
        engine.shutdown()
    n_out = sum(len(tok.encode(t or "")) for t in texts)
    # Deadline cutoff leaves interspersed None slots (todo is LIFO), so score by
    # absolute index — gold[:n_done] would align the first n_done golds against
    # whatever slots happened to finish.
    done_idx = [i for i, t in enumerate(texts) if t is not None]
    preds = [answer_letter(texts[i] or "") for i in done_idx]
    return {
        "n_done": len(done_idx),
        "done_idx": done_idx,
        "predictions": preds,
        "raw": [t for t in texts[:20]],
        "mean_output_tokens": n_out / max(1, len(done_idx)),
        "elapsed_s": round(elapsed, 1),
        "tok_s": round(n_out / max(1e-9, elapsed), 2),
        "spec_drafted": stats.get("spec_drafted", 0),
        "spec_accepted": stats.get("spec_accepted", 0),
        "spec_accept_in": stats.get("spec_accept_in", 0),
        "spec_drafted_in": stats.get("spec_drafted_in", 0),
        "spec_accept_post": stats.get("spec_accept_post", 0),
        "spec_drafted_post": stats.get("spec_drafted_post", 0),
        "spec_accept_capcross": stats.get("spec_accept_capcross", 0),
        "spec_drafted_capcross": stats.get("spec_drafted_capcross", 0),
        "sparse_k": k,
    }


def selftest():
    """Tiny CPU: dense k=0 and full-k sparse k=64 both drain two prompts and
    answer; the harness's prompt/letter plumbing runs end to end."""
    from tilerl_kernels.backend import get_backend

    from tilerl import config as config_mod
    from tilerl.eval import generate_ids
    from tilerl.kv_cache import BLOCK_TOKENS
    from tilerl.model import build_random

    cfg = config_mod.tiny(2048)
    backend = get_backend()

    class _Tok:
        stop_token_ids = ()

        def encode(self, s):
            return list(s.encode())[:200]

        def decode(self, ids):
            return bytes(ids).decode("utf8", "replace")

    prompts = ["A. 1 B. 2 Answer: "] * 2

    def arm(k):
        from tilerl.engine import build_engine

        e = build_engine(cfg, build_random(cfg, 0), backend, num_blocks=64,
                         num_slots=4, max_batch=2, max_total_tokens=1024,
                         sparse_k=k, scorer="bounds")
        sp = SamplingParams(temperature=0.0, seed=0, max_new_tokens=4)
        ids = generate_ids(e, _Tok(), prompts, sp, 2)
        e.shutdown()
        return ids

    dense, sparse = arm(0), arm(64 // BLOCK_TOKENS * BLOCK_TOKENS)
    assert len(dense) == 2 and all(len(x) == 4 for x in dense + sparse)
    assert answer_letter("...</think>\n\nThe answer is B.") == "B"
    print("selftest OK: 2 prompts drained on dense k=0 and full-k sparse; "
          "letter extraction B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--draft", default="model_mtp.safetensors",
                    help="draft path relative to the checkpoint; pass '' for no spec")
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--deadline-min", type=float, default=90.0)
    ap.add_argument("--out", default="/work/mmlu_thinking_spec.json")
    ap.add_argument("--arm", choices=["both", "dense", "sparse"], default="both",
                    help="one arm per process for a two-card run; pair afterward")
    ap.add_argument("--first-n", type=int, default=0,
                    help="use only the first N of the --n seeded slice (subset of a larger run)")
    ap.add_argument("--chat-control", type=int, default=0,
                    help="N plain-chat prompts (no forced think), spec on: acceptance control")
    ap.add_argument("--pair", nargs=2, metavar=("DENSE_JSON", "SPARSE_JSON"),
                    help="merge two single-arm JSONs into one paired result")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if args.pair:
        pair_arms(*args.pair, out=args.out)
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TILERL_TARGET", "cuda")
    from tilerl_kernels.backend import get_backend

    from tilerl.tokenizer import get_tokenizer

    tok = get_tokenizer(args.source)
    backend = get_backend()
    draft_path = (None if not args.draft else
                  args.draft if os.path.isabs(args.draft)
                  else os.path.join(args.source, args.draft))

    if args.chat_control:
        # Same engine build (dense, draft, spec_depth=1, B=8, greedy), plain
        # chat template, no forced think. Acceptance here vs the MMLU thinking
        # arm isolates content/template from an engine defect.
        prompts = chat_control_prompts(args.chat_control)
        a = run_arm(args.source, prompts, 0, draft_path, tok, backend,
                    args.max_ctx, args.deadline_min * 60, force_think=False)
        acc = a["spec_accepted"] / max(1, a["spec_drafted"])
        print(f"CHAT_CONTROL n={a['n_done']} spec_accept={acc:.3f} "
              f"({a['spec_accepted']}/{a['spec_drafted']}) "
              f"tok/s={a['tok_s']:.1f} mean_tok={a['mean_output_tokens']:.0f}",
              flush=True)
        with open(args.out, "w") as fh:
            json.dump({"chat_control": args.chat_control,
                       "spec_accept": round(acc, 5), **a}, fh)
        return

    raw, golds, subjects = mmlu_questions(args.n, args.seed)
    # --first-n takes the leading questions of the SAME seeded --n slice, so a
    # 400-run is a true subset of a 2000-run and pairs on identical absolute ids.
    if args.first_n:
        raw, golds, subjects = raw[:args.first_n], golds[:args.first_n], subjects[:args.first_n]
    prompts = thinking_prompts(raw)
    tok = get_tokenizer(args.source)
    backend = get_backend()
    draft_path = (None if not args.draft else
                  args.draft if os.path.isabs(args.draft)
                  else os.path.join(args.source, args.draft))
    result = {"n": args.n, "seed": args.seed, "k": args.k,
              "draft": bool(draft_path), "max_think": MAX_THINK,
              "max_new": MAX_NEW, "concurrency": CONCURRENCY,
              "gold": golds, "subjects": subjects, "arms": {}}
    wanted = {"both": ["dense", "sparse"], "dense": ["dense"],
              "sparse": ["sparse"]}[args.arm]
    arms = [("dense", 0), ("sparse", args.k)]

    def make_progress(label):
        # Live per-question completion line + partial JSON flush every 50, so a
        # deadline cutoff or kill leaves a readable n and accuracy.
        preds: dict[int, str] = {}
        texts: dict[int, str] = {}

        def on_done(idx, text, n_done, elapsed_s):
            preds[idx] = answer_letter(text or "")
            texts[idx] = text or ""
            correct = sum(preds[i] == golds[i] for i in preds)
            print(f"progress {label} {n_done}/{args.n} acc={correct/max(1,n_done):.4f} "
                  f"{elapsed_s:.0f}s", flush=True)
            if n_done % 50 == 0:
                done = sorted(preds)
                ntok = sum(len(tok.encode(texts[i])) for i in done)
                with open(args.out, "w") as fh:
                    json.dump({"n": args.n, "seed": args.seed, "k": args.k,
                               "arms": {label: {"n_done": len(done), "done_idx": done,
                                                "predictions": [preds[i] for i in done],
                                                "correct": correct,
                                                "accuracy": correct / max(1, n_done),
                                                "tok_s": round(ntok / max(1e-9, elapsed_s), 2),
                                                "partial": True}}}, fh)
        return on_done

    for label, k in arms:
        if label not in wanted:
            continue
        a = run_arm(args.source, prompts, k, draft_path, tok, backend,
                    args.max_ctx, args.deadline_min * 60, on_done=make_progress(label))
        gold = [golds[i] for i in a["done_idx"]]
        a["correct"] = sum(p == g for p, g in zip(a["predictions"], gold))
        a["accuracy"] = a["correct"] / max(1, a["n_done"])
        result["arms"][label] = a
        print(f"{label}: n={a['n_done']} acc={a['accuracy']:.3f} "
              f"mean_tok={a['mean_output_tokens']:.0f} tok/s={a['tok_s']:.1f} "
              f"spec_acc={a['spec_accepted']}/{a['spec_drafted']} ({a['elapsed_s']}s)",
              flush=True)
        if a["spec_drafted_in"] or a["spec_drafted_post"]:
            print(f"SEGMENT {label} inside={a['spec_accept_in']}/{a['spec_drafted_in']} "
                  f"({a['spec_accept_in']/max(1,a['spec_drafted_in']):.3f}) "
                  f"post_think={a['spec_accept_post']}/{a['spec_drafted_post']} "
                  f"({a['spec_accept_post']/max(1,a['spec_drafted_post']):.3f}) "
                  f"capcross={a['spec_accept_capcross']}/{a['spec_drafted_capcross']} "
                  f"({a['spec_accept_capcross']/max(1,a['spec_drafted_capcross']):.3f})",
                  flush=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh)
    if set(wanted) != {"dense", "sparse"}:
        print(f"single-arm {args.arm} -> {args.out}; pair with --pair", flush=True)
        return
    d, s = result["arms"]["dense"], result["arms"]["sparse"]
    print(f"DELTA sparse-dense: acc {s['accuracy'] - d['accuracy']:+.3f} "
          f"tok/s {s['tok_s'] / max(d['tok_s'], 1e-9) - 1:+.2f}x "
          f"mean_out_tok {s['mean_output_tokens'] - d['mean_output_tokens']:+.0f}")
    report_paired(result, d, s, golds, args.out)


def report_paired(result, d, s, golds, out):
    # Paired test on the questions BOTH arms finished: discordant cells
    # b = sparse-right/dense-wrong, c = dense-right/sparse-wrong. A 2-point gap
    # needs ~2000 paired questions at 10% discordance for 80% power; n=100's
    # paired half-width is ~6 points, so print the CI, not just the point delta.
    common = sorted(set(d["done_idx"]) & set(s["done_idx"]))
    if not common:
        return
    import math

    dr = {i: p for i, p in zip(d["done_idx"], d["predictions"])}
    sr = {i: p for i, p in zip(s["done_idx"], s["predictions"])}
    b = sum(sr[i] == golds[i] and dr[i] != golds[i] for i in common)
    c = sum(dr[i] == golds[i] and sr[i] != golds[i] for i in common)
    delta = (b - c) / len(common)
    hw = 1.96 * math.sqrt((b + c) / len(common) ** 2)
    result["paired"] = {"n_common": len(common), "sparse_only_correct": b,
                        "dense_only_correct": c, "delta": round(delta, 4),
                        "ci95_halfwidth": round(hw, 4)}
    print(f"PAIRED n={len(common)} discordants b={b} c={c} "
          f"delta={delta:+.3f} ±{hw:.3f} (95%)")
    with open(out, "w") as fh:
        json.dump(result, fh)


def pair_arms(dense_json, sparse_json, out):
    """Merge two one-arm runs (separate cards) by question index and report."""
    with open(dense_json) as fh:
        d_doc = json.load(fh)
    with open(sparse_json) as fh:
        s_doc = json.load(fh)
    d, s = d_doc["arms"]["dense"], s_doc["arms"]["sparse"]
    assert d_doc["seed"] == s_doc["seed"] and d_doc["n"] == s_doc["n"], "slice mismatch"
    result = dict(d_doc)
    result["arms"] = {"dense": d, "sparse": s}
    result["paired_two_cards"] = {
        "dense_json": dense_json, "sparse_json": sparse_json,
        "note": "one arm per process/card; pairing is by question id (done_idx), not process"}
    print(f"dense:  n={d['n_done']} acc={d['accuracy']:.4f} tok/s={d['tok_s']:.1f} "
          f"spec {d['spec_accepted']}/{d['spec_drafted']}")
    print(f"sparse: n={s['n_done']} acc={s['accuracy']:.4f} tok/s={s['tok_s']:.1f} "
          f"spec {s['spec_accepted']}/{s['spec_drafted']}")
    print(f"DELTA sparse-dense acc {s['accuracy'] - d['accuracy']:+.4f} "
          f"tok/s {s['tok_s'] / max(d['tok_s'], 1e-9) - 1:+.2f}x")
    report_paired(result, d, s, d_doc["gold"], out)


if __name__ == "__main__":
    main()
