"""Which tile geometry moves a verify tick's logits: the linears' or attention's.

Five arms (W=1,2,3,4,8) in one process on one card, same questions, same seeds,
greedy. Every sampled position records the top-2 logits the sampler saw, keyed by
(prompt, generated index), so any two arms can be differenced on the positions
where their histories still agree.

Attention's M tile is ``_snap_mma_tile(G*W, 128)``, G=6, a function of W alone
(B is a grid dimension, not a tile dimension): 16, 16, 32, 32, 64 at W=1,2,3,4,8.

The fp4 linear arm is picked by **M = B*W** against ``_MGEMV`` (3) and ``_MX``
(8) -- NOT by W. ``concurrency`` sets B, and B falls from the concurrency down
to 1 as rows retire, so one arm sweeps several linear kernels:

    B  W=1        W=2         W=3          W=4          W=8
    1  gemv(M=1)  gemv(M=2)   gemv(M=3)    mma8         mma8
    2  gemv(M=2)  mma8        mma8         mma8         w4a8(Mp=16)
    4  mma8       mma8        w4a8(Mp=16)  w4a8(Mp=16)  w4a8(Mp=32)
    8  mma8       w4a8(Mp=16) w4a8(Mp=32)  w4a8(Mp=32)  w4a8(Mp=64)

So at ``--concurrency 1`` (M = W) each rung moves one factor and W=1<->W=2 is a
null control: one gemv kernel on the M=1 plan, one attention tile. Above B=1 no
rung isolates attention, but W=3<->W=4 still nearly holds the linears -- same
arm at B=2,3,4,6,7,8, differing only at B=1 and B=5 -- while attention's tile is
32 for both. Measured at concurrency 8: that rung is the ONLY one at 0.000e+00
median with 7 of 8 completions bit-identical, and every rung that moves the
linears sits at ~1.5e-01. The projections own the divergence; attention's tile
does not.

    an earlier version of this docstring read the table as if M were W, which is
    true only at B=1, and called W=1<->W=2 a null control at any B.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/probe_width_ladder.py \
        --source /work/Qwen3.8-27B-NVFP4 \
        --draft /work/Qwen3.8-27B-NVFP4/model_mtp.safetensors \
        --gsm8k /work/gsm8k_test.jsonl --gsm8k-n 8 --out /work/wladder
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time
from dataclasses import replace
from pathlib import Path

import torch
from tilerl_kernels.backend import _MGEMV, _MX, _snap_mma_tile, get_backend

from tilerl.config import qwen38_27b
from tilerl.engine import build_engine
from tilerl.eval import answer_match, generate
from tilerl.kv_cache import NoPrefixStore
from tilerl.model import load_hf
from tilerl.prompt import render_chat, sampling
from tilerl.tokenizer import get_tokenizer


def instrument(engine):
    """rec[prompt][gen_idx] = (token, top1, top2, argmax, width, is_verify)."""
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


def arm(width, cfg, model, backend, tok, draft_path, rows, params, conc):
    from tilerl.spec import load_draft

    name = f"w{width}"
    draft = load_draft(model, draft_path) if width > 1 else None
    engine = build_engine(cfg, model, backend, num_blocks=512, num_slots=8, draft=draft,
                          spec_depth=max(1, width - 1), decode_graph=False,
                          prefix_store=NoPrefixStore())
    rec, ids = instrument(engine)
    prompts = [render_chat([("user", r["prompt"])], False) for r in rows]
    torch.cuda.synchronize(); t0 = time.perf_counter()
    texts = generate(engine, tok, prompts, params, conc)
    torch.cuda.synchronize(); secs = time.perf_counter() - t0
    s = engine.stats()
    ok = sum(answer_match(t, r["answer"]) for t, r in zip(texts, rows))
    print(f"[{name}] gsm8k {ok}/{len(rows)}  {secs:.1f}s  drafted {s['spec_drafted']} "
          f"accepted {s['spec_accepted']}  decode fwd {s['decode_forwards']}", flush=True)
    engine = draft = None
    torch.cuda.empty_cache()
    return {"w": width, "texts": texts, "rec": rec, "ids": ids,
            "gsm8k": ok, "secs": secs, "stats": dict(s)}


def q(xs, f):
    return sorted(xs)[min(len(xs) - 1, int(f * len(xs)))]


def compare(a, b, nq):
    """|delta top-1| between two arms over positions before their first divergence,
    and how much of the reference arm's top-2 margin sits under that scale."""
    delta, gap, first = [], [], {}
    for i in range(nq):
        ia, ib = a["ids"].get(i, []), b["ids"].get(i, [])
        k = next((j for j in range(min(len(ia), len(ib))) if ia[j] != ib[j]),
                 min(len(ia), len(ib)))
        first[i] = (k, ia == ib, len(ia), len(ib))
        ra, rb = a["rec"].get(i, {}), b["rec"].get(i, {})
        for j in range(k):
            if j in ra and j in rb:
                delta.append(abs(ra[j][1] - rb[j][1]))
                gap.append(ra[j][1] - ra[j][2])
    label = f"W={a['w']} <-> W={b['w']}"
    if not delta:
        print(f"\n=== {label}: no shared positions ===")
        return {"pair": label, "n": 0}
    diff = sum(1 for v in first.values() if not v[1])
    med = st.median(delta)
    under = sum(g < med for g in gap)
    print(f"\n=== {label} ===")
    print(f"  completions differing: {diff}/{nq}")
    print(f"  |delta top-1| on identical history, n={len(delta)}: "
          f"median {med:.3e}  p90 {q(delta, .9):.3e}  max {max(delta):.3e}")
    print(f"  W={a['w']} top-2 gap over the same positions: "
          f"median {st.median(gap):.3e}  p10 {q(gap, .1):.3e}  min {min(gap):.3e}")
    for f, thr in (("median", med), ("p90", q(delta, .9)), ("max", max(delta))):
        n = sum(g < thr for g in gap)
        print(f"  positions with top-2 gap < {f} delta ({thr:.3e}): "
              f"{n}/{len(gap)} = {100 * n / len(gap):.2f}%")
    return {"pair": label, "n": len(delta), "differing": diff,
            "delta_median": med, "delta_p90": q(delta, .9), "delta_max": max(delta),
            "gap_median": st.median(gap), "gap_min": min(gap),
            "under_median": under,
            "first": {str(k): v for k, v in first.items()}}


def fp4_arm(m):
    """Which linear_fp4 arm M lands in (backend.py:353,370,380,392)."""
    if 2 <= m <= _MGEMV:
        return f"gemv(M={m})"
    if 2 <= m <= _MX:
        return "mma8(8 rows)"
    return "gemv(M=1)" if m == 1 else f"w4a8(bM={_snap_mma_tile(m, 128)})"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--gsm8k", required=True)
    p.add_argument("--gsm8k-n", type=int, default=8)
    p.add_argument("--widths", default="1,2,3,4,8")
    p.add_argument("--max-new-tokens", type=int, default=256)
    # B enters the linear arm through M = B*W; 1 is what makes a rung one-factor
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    import tilelang
    print("tilelang", tilelang.__version__, flush=True)
    backend = get_backend()
    assert backend.device.type == "cuda"
    cfg = qwen38_27b()
    g = cfg.num_attention_heads // cfg.num_kv_heads
    widths = [int(w) for w in a.widths.split(",")]
    print(f"geometry (_MGEMV={_MGEMV} _MX={_MX} G={g}, concurrency {a.concurrency}):")
    print("  attention block_m is a function of W alone:")
    for w in widths:
        bm = _snap_mma_tile(g * w, 128)
        print(f"    W={w}: block_m {bm:3d}  policy "
              f"{'FullRow' if bm >= 32 else 'Square ':7s} P cast "
              f"{'direct' if bm >= 32 else 'via shared'}")
    print("  fp4 linear arm is a function of M = B*W, and B falls to 1 as rows retire:")
    print("    B  " + "  ".join(f"W={w:<14}" for w in widths))
    for b in range(1, a.concurrency + 1):
        print(f"    {b}  " + "  ".join(f"{fp4_arm(b * w):<16}" for w in widths))
    if a.concurrency > 1:
        print("  NOTE: no rung isolates attention above B=1 -- see the docstring.")

    tok = get_tokenizer(a.source)
    model = load_hf(cfg, a.source)
    rows = [json.loads(ln) for ln in Path(a.gsm8k).read_text().splitlines()
            if ln.strip()][: a.gsm8k_n]
    params = replace(sampling(tok, False, a.max_new_tokens, temperature=0.0,
                              max_think_tokens=0, seed=0), temperature=0.0)

    arms = {w: arm(w, cfg, model, backend, tok, a.draft, rows, params, a.concurrency)
            for w in widths}

    # consecutive rungs isolate one factor each; the endpoints reproduce the
    # recorded W=1<->W=8 number so the ladder is anchored to it
    pairs = [(widths[i], widths[i + 1]) for i in range(len(widths) - 1)]
    if len(widths) > 2:
        pairs.append((widths[0], widths[-1]))
    cmps = [compare(arms[x], arms[y], len(rows)) for x, y in pairs]

    print("\n=== committed token != verify tile's own argmax ===")
    for w in widths:
        entries = [v for r in arms[w]["rec"].values() for v in r.values() if v[5]]
        print(f"  W={w}: {sum(v[0] != v[3] for v in entries)}/{len(entries)}")

    print("\n=== ladder: median |delta top-1| per rung ===")
    for c in cmps:
        if c["n"]:
            print(f"  {c['pair']:16s} {c['delta_median']:.3e}  "
                  f"(n={c['n']}, {c['differing']}/{len(rows)} completions differ)")

    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "ladder.json").write_text(json.dumps({
        "widths": widths, "cmps": cmps,
        "arms": {str(w): {k: arms[w][k] for k in ("texts", "gsm8k", "secs", "stats")}
                 for w in widths}}, indent=1))
    print(f"\nwrote {o / 'ladder.json'}")


if __name__ == "__main__":
    main()
