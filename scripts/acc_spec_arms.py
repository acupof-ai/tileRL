"""MMLU + GSM8K in two arms — no speculation, then a width-W verify tick — in
one process on one card, through the same ``eval.py`` the training path uses.

Speculation's claim is equality, not similarity: greedy output must be the
unspeculated output. Two arms can both score 742/1000 and still disagree per
question, so every completion is kept and diffed, and the accuracy row and the
throughput row come from the same session.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/acc_spec_arms.py \
        --source /work/Qwen3.8-27B-NVFP4 --draft /work/Qwen3.8-27B-DFlash2 \
        --gsm8k /work/gsm8k_test.jsonl --width 8 --out /work/accspec
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch

from tilerl.config import qwen38_27b
from tilerl.engine import build_engine
from tilerl.eval import answer_match, generate, letter, mmlu_questions, mmlu_score
from tilerl.kv_cache import NoPrefixStore
from tilerl.model import load_hf
from tilerl.prompt import render_chat, sampling
from tilerl.tokenizer import get_tokenizer
from tilerl_kernels.backend import get_backend


def phase(engine, fn):
    """Run one eval phase; return its texts plus the engine deltas over it."""
    torch.cuda.synchronize()
    s0, t0 = engine.stats(), time.perf_counter()
    texts = fn()
    torch.cuda.synchronize()
    secs, s1 = time.perf_counter() - t0, engine.stats()
    d = {k: s1[k] - s0[k] for k in ("tokens_generated", "decode_forwards", "prefill_forwards",
                                    "mixed_forwards", "spec_drafted", "spec_accepted")}
    d["secs"] = secs
    d["tok_per_s"] = d["tokens_generated"] / secs
    ticks = d["decode_forwards"]
    d["tok_per_decode_forward"] = d["tokens_generated"] / ticks if ticks else 0.0
    return texts, d


def trace_kernel_lengths(engine) -> dict[str, Counter]:
    """Residue mod 64 of the KV length a wide decode tick hands the decode
    kernel (``SeqLens = seq_len - 1 + W``), the width histogram beside it, and
    the same residue for every tick whose trunk logits came back NaN. A split
    that is non-empty but wholly masked lives at ``n % 64`` in ``[1, W-1]``, so
    these two histograms together say whether that geometry is what breaks."""
    t = {"res": Counter(), "widths": Counter(), "nan": Counter(), "nan_first": Counter()}
    inner, verify = engine._run_forward, engine._verify

    def traced(decodes, prefills, chunks):
        if decodes and not prefills and engine._draft is not None:
            w = max((1 + len(r.drafts) for r in decodes), default=1)
            t["widths"][w] += 1
            if w > 1:
                for r in decodes:
                    t["res"][(r.seq_len - 1 + w) % 64] += 1
        return inner(decodes, prefills, chunks)

    seen_nan: set[int] = set()

    def traced_verify(rows, chains, logits, hidden):
        bad = torch.isnan(logits).flatten(1).any(dim=1)
        for i, r in enumerate(rows):
            if bool(bad[i]):
                n = r.seq_len - 1 + len(chains[i])
                t["nan"][n % 64] += 1
                if r.req_id not in seen_nan:
                    seen_nan.add(r.req_id)
                    t["nan_first"][n % 64] += 1
        return verify(rows, chains, logits, hidden)

    engine._run_forward = traced
    engine._verify = traced_verify
    return t


def arm(name, cfg, model, backend, tok, draft_path, width, mmlu_n, rows, params, graph):
    from tilerl.spec import load_draft

    draft = load_draft(model, draft_path) if draft_path else None
    engine = build_engine(cfg, model, backend, num_blocks=512, num_slots=8, draft=draft,
                          spec_depth=max(1, width - 1), decode_graph=graph,
                          prefix_store=NoPrefixStore())
    tr = trace_kernel_lengths(engine)
    out = {"arm": name, "width": width if draft else 1, "decode_graph": graph}

    if mmlu_n:
        prompts, golds, subjects = mmlu_questions(mmlu_n, seed=0)
        texts, d = phase(engine, lambda: mmlu_score(engine, tok, prompts, concurrency=8))
        preds = [letter(t) for t in texts]
        out["mmlu"] = dict(correct=sum(p == g for p, g in zip(preds, golds)), total=len(preds),
                           timing=d, subjects=subjects, gold=golds, pred=preds, text=texts,
                           prompt_tokens=[len(tok.encode(p)) for p in prompts])
        print(f"[{name}] mmlu 0-shot {out['mmlu']['correct']}/{len(preds)} = "
              f"{100 * out['mmlu']['correct'] / len(preds):.1f}%  {d['secs']:.1f}s  "
              f"decode fwd {d['decode_forwards']}  prefill fwd {d['prefill_forwards']}  "
              f"drafted {d['spec_drafted']}", flush=True)

    if rows:
        prompts = [render_chat([("user", r["prompt"])], False) for r in rows]
        sp = replace(params, temperature=0.0)
        texts, d = phase(engine, lambda: generate(engine, tok, prompts, sp, 8))
        ok = [answer_match(t, r["answer"]) for t, r in zip(texts, rows)]
        out["gsm8k"] = dict(correct=sum(ok), total=len(rows), timing=d, text=texts,
                            answer=[r["answer"] for r in rows],
                            prompt_tokens=[len(tok.encode(p)) for p in prompts])
        print(f"[{name}] gsm8k greedy {sum(ok)}/{len(rows)} = {100 * sum(ok) / len(rows):.1f}%  "
              f"{d['secs']:.1f}s  {d['tok_per_s']:.1f} tok/s  "
              f"{d['tok_per_decode_forward']:.2f} tok/decode-forward", flush=True)

    out.update({k: dict(v) for k, v in tr.items()})
    res, widths, nan = tr["res"], tr["widths"], tr["nan"]
    if res:
        risky = sum(v for k, v in res.items() if 1 <= k <= width - 1)
        print(f"[{name}] wide ticks {sum(widths[w] for w in widths if w > 1)}, widths {dict(widths)}"
              f"\n[{name}] kernel KV length mod 64: {len(res)}/64 residues reached, "
              f"{risky} row-ticks at residues [1,{width - 1}] of {sum(res.values())}"
              f"\n[{name}] NaN trunk logits: {sum(nan.values())} row-ticks, residues "
              f"{sorted(nan)}; first NaN per request at residues {dict(tr['nan_first'])}", flush=True)

    # The training engine's snapshot side table (PR #6's leak) on the workload that OOMed P1.
    out["prefix_snapshots"] = len(getattr(engine, "_prefix_state", {}))
    out["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
    print(f"[{name}] peak cuda {out['peak_gib']:.2f} GiB, "
          f"{out['prefix_snapshots']} retained state snapshots", flush=True)
    engine = draft = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--draft", required=True, help="draft head safetensors (the spec arm)")
    p.add_argument("--gsm8k")
    p.add_argument("--mmlu-n", type=int, default=1000)
    p.add_argument("--gsm8k-n", type=int, default=500)
    p.add_argument("--width", type=int, default=8, help="verify tick width: 1 committed + W-1 drafts")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--decode-graph", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    backend = get_backend()
    assert backend.device.type == "cuda", "needs TILERL_TARGET=cuda"
    cfg = qwen38_27b()
    tok = get_tokenizer(args.source)
    model = load_hf(cfg, args.source)
    rows = ([json.loads(ln) for ln in Path(args.gsm8k).read_text().splitlines() if ln.strip()]
            [: args.gsm8k_n] if args.gsm8k else [])
    # The recipe's own params: thinking off (max_think_tokens=0), 256 new tokens, seed 0.
    params = sampling(tok, False, args.max_new_tokens, temperature=0.0, max_think_tokens=0, seed=0)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = [arm("base", cfg, model, backend, tok, None, 1, args.mmlu_n, rows, params,
                args.decode_graph),
            arm(f"spec-w{args.width}", cfg, model, backend, tok, args.draft, args.width,
                args.mmlu_n, rows, params, args.decode_graph)]
    for a in arms:
        (out / f"{a['arm']}.json").write_text(json.dumps(a))

    print("\n=== equality: the spec arm must reproduce the base arm's strings ===")
    for suite in ("mmlu", "gsm8k"):
        if suite not in arms[0]:
            continue
        a, b = arms[0][suite], arms[1][suite]
        diff = [i for i, (x, y) in enumerate(zip(a["text"], b["text"])) if x != y]
        print(f"{suite}: base {a['correct']}/{a['total']}  spec {b['correct']}/{b['total']}  "
              f"differing completions {len(diff)}/{a['total']}")
        for i in diff[:5]:
            k = next((j for j in range(min(len(a['text'][i]), len(b['text'][i])))
                      if a["text"][i][j] != b["text"][i][j]), None)
            print(f"  q{i}: diverges at char {k}\n    base {a['text'][i][:400]!r}"
                  f"\n    spec {b['text'][i][:400]!r}")
        (out / f"{suite}-diff.json").write_text(json.dumps(
            {"differing": diff, "base": [a["text"][i] for i in diff],
             "spec": [b["text"][i] for i in diff]}))

    print("\n=== throughput, same session ===")
    for a in arms:
        for suite in ("mmlu", "gsm8k"):
            if suite in a:
                t = a[suite]["timing"]
                print(f"{a['arm']:>10} {suite:>6}  {t['secs']:8.1f}s  {t['tok_per_s']:8.1f} tok/s  "
                      f"{t['tok_per_decode_forward']:5.2f} tok/decode-fwd  "
                      f"drafted {t['spec_drafted']} accepted {t['spec_accepted']}")


if __name__ == "__main__":
    main()
