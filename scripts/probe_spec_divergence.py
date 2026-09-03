"""Why a width-W verify tick's greedy output differs from the unspeculated one.

Two arms in one process on one card. Every sampled entry records the token and
the top-2 logits the sampler saw, keyed by (prompt, generated index). Two
candidate causes, one discriminator each:

* a near-tie decided differently by two tile geometries -- the committed token
  IS the verify tile's own argmax, and the base arm's top-2 gap at the
  divergence sits below the arm-to-arm logit difference;
* an acceptance or indexing error -- the committed token is NOT the tile's
  argmax, or the divergence sits at a wide gap.

    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/probe_spec_divergence.py \
        --source /work/Qwen3.8-27B-NVFP4 \
        --draft /work/Qwen3.8-27B-NVFP4/model_mtp.safetensors \
        --gsm8k /work/gsm8k_test.jsonl --gsm8k-n 8 --width 8 --out /work/specgap
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
from tilerl_kernels.backend import get_backend

from tilerl.config import qwen38_27b
from tilerl.engine import build_engine
from tilerl.eval import answer_match, generate
from tilerl.kv_cache import NoPrefixStore
from tilerl.model import load_hf
from tilerl.prompt import render_chat, sampling
from tilerl.tokenizer import get_tokenizer


def instrument(engine):
    """rec[prompt][gen_idx] = (token, top1, top2, argmax, chain_width, is_verify)."""
    rec: dict[int, dict[int, tuple]] = {}
    ids: dict[int, list[int]] = {}
    pidx: dict[int, int] = {}
    order = {"n": 0}
    width: dict[int, int] = {}
    sample, verify, submit, poll = (
        engine._sample_batch, engine._verify, engine.submit, engine.poll)

    def w_submit(input_ids, params=None):
        rid = submit(input_ids, params)
        pidx[rid] = order["n"]
        order["n"] += 1
        return rid

    def w_sample(rows):
        toks = sample(rows)
        if rows:
            lg = torch.stack([l for _, l, _ in rows]).float()
            v, i = lg.topk(2, dim=-1)
            v, i = v.tolist(), i.tolist()
            for n, ((r, _, g), t) in enumerate(zip(rows, toks)):
                rec.setdefault(pidx[r.req_id], {})[g] = (
                    int(t), v[n][0], v[n][1], i[n][0], width.get(r.req_id, 1),
                    bool(width))
        return toks

    def w_verify(rows, chains, logits, hidden):
        width.clear()
        for i, r in enumerate(rows):
            width[r.req_id] = len(chains[i])
        out = verify(rows, chains, logits, hidden)
        width.clear()
        return out

    def w_poll():
        done = poll()
        for rid, out in done.items():
            ids[pidx[rid]] = list(out)
        return done

    engine._sample_batch, engine._verify = w_sample, w_verify
    engine.submit, engine.poll = w_submit, w_poll
    return rec, ids


def arm(name, cfg, model, backend, tok, draft_path, width, rows, params):
    from tilerl.spec import load_draft

    draft = load_draft(model, draft_path) if draft_path else None
    engine = build_engine(cfg, model, backend, num_blocks=512, num_slots=8, draft=draft,
                          spec_depth=max(1, width - 1), decode_graph=False,
                          prefix_store=NoPrefixStore())
    rec, ids = instrument(engine)
    prompts = [render_chat([("user", r["prompt"])], False) for r in rows]
    torch.cuda.synchronize(); t0 = time.perf_counter()
    texts = generate(engine, tok, prompts, params, 8)
    torch.cuda.synchronize(); secs = time.perf_counter() - t0
    s = engine.stats()
    ok = sum(answer_match(t, r["answer"]) for t, r in zip(texts, rows))
    print(f"[{name}] gsm8k {ok}/{len(rows)}  {secs:.1f}s  drafted {s['spec_drafted']} "
          f"accepted {s['spec_accepted']}  decode fwd {s['decode_forwards']}", flush=True)
    engine = draft = None
    torch.cuda.empty_cache()
    return {"name": name, "texts": texts, "rec": rec, "ids": ids}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--gsm8k", required=True)
    p.add_argument("--gsm8k-n", type=int, default=8)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    import tilelang
    print("tilelang", tilelang.__version__, flush=True)
    backend = get_backend()
    assert backend.device.type == "cuda"
    cfg = qwen38_27b()
    tok = get_tokenizer(a.source)
    model = load_hf(cfg, a.source)
    rows = [json.loads(ln) for ln in Path(a.gsm8k).read_text().splitlines() if ln.strip()][: a.gsm8k_n]
    params = replace(sampling(tok, False, a.max_new_tokens, temperature=0.0,
                              max_think_tokens=0, seed=0), temperature=0.0)

    base = arm("base", cfg, model, backend, tok, None, 1, rows, params)
    spec = arm(f"spec-w{a.width}", cfg, model, backend, tok, a.draft, a.width, rows, params)

    print("\n=== first divergence per prompt: the top-2 gap is the discriminator ===")
    summary = []
    for i in range(len(rows)):
        b, s = base["ids"].get(i, []), spec["ids"].get(i, [])
        if b == s:
            print(f"q{i}: identical ({len(b)} tokens)")
            continue
        k = next((j for j in range(min(len(b), len(s))) if b[j] != s[j]), min(len(b), len(s)))
        br, sr = base["rec"].get(i, {}).get(k), spec["rec"].get(i, {}).get(k)
        row = {"q": i, "idx": k, "base_tok": b[k] if k < len(b) else None,
               "spec_tok": s[k] if k < len(s) else None, "base": br, "spec": sr}
        summary.append(row)
        f = lambda r: ("-" if r is None else
                       f"tok {r[0]} argmax {r[3]} top1 {r[1]:.6f} top2 {r[2]:.6f} "
                       f"gap {r[1] - r[2]:.3e} W={r[4]} verify={r[5]}")
        print(f"q{i}: diverges at generated index {k} of ({len(b)},{len(s)})")
        print(f"     base  {f(br)}")
        print(f"     spec  {f(sr)}")
        if sr is not None and sr[0] != sr[3]:
            print("     !! committed token is NOT the verify tile's argmax")
        # the same position's gap in both arms, plus the logit distance
        if br and sr:
            print(f"     top1 |base-spec| = {abs(br[1] - sr[1]):.3e}, "
                  f"argmax agree = {br[3] == sr[3]}")

    # (a) how far apart are the two tile geometries on IDENTICAL history?
    #     every position strictly before this prompt's first divergence.
    # (b) how often is the base arm's own top-2 gap below that scale?
    import statistics as st
    first = {r["q"]: r["idx"] for r in summary}
    delta, basegap, notargmax, nspec = [], [], 0, 0
    for i in range(len(rows)):
        cut = first.get(i, len(base["ids"].get(i, [])))
        br, sr = base["rec"].get(i, {}), spec["rec"].get(i, {})
        for k in range(cut):
            if k in br:
                basegap.append(br[k][1] - br[k][2])
                if k in sr:
                    delta.append(abs(br[k][1] - sr[k][1]))
        for k, v in sr.items():
            nspec += 1
            notargmax += v[0] != v[3]
    q = lambda xs, f: sorted(xs)[min(len(xs) - 1, int(f * len(xs)))]
    print(f"\n=== arm-to-arm |delta top1| on identical history, n={len(delta)} ===")
    print(f"  median {st.median(delta):.3e}  p90 {q(delta, .9):.3e}  max {max(delta):.3e}")
    print(f"=== base-arm top-2 gap over the same {len(basegap)} positions ===")
    print(f"  median {st.median(basegap):.3e}  p10 {q(basegap, .1):.3e}  min {min(basegap):.3e}")
    for thr in (q(delta, .5), q(delta, .9), max(delta)):
        n = sum(g < thr for g in basegap)
        print(f"  positions with gap < {thr:.3e}: {n}/{len(basegap)} = {100*n/len(basegap):.2f}%")
    print(f"=== spec arm: committed token != tile argmax on "
          f"{notargmax}/{nspec} sampled entries ===")
    gaps = [r["spec"][1] - r["spec"][2] for r in summary if r["spec"]]
    print(f"\nmismatching prompts {len(summary)}/{len(rows)}")
    if gaps:
        bg = sorted(r["base"][1] - r["base"][2] for r in summary if r["base"])
        print(f"base-arm top-2 gap AT the divergences: {[f'{g:.3e}' for g in bg]}")
        print("committed != tile argmax count:",
              sum(1 for r in summary if r["spec"] and r["spec"][0] != r["spec"][3]))
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "divergence.json").write_text(json.dumps(
        {"summary": summary, "base_text": base["texts"], "spec_text": spec["texts"]}))


if __name__ == "__main__":
    main()
