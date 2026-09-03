"""Do the 4 MMLU answers that moved between concurrencies sit at near-ties?

Recorded on clean main: 746/1000 at concurrency 32 and 742/1000 at concurrency 8,
same slice (``seed`` pins the draw). The mechanism is known -- concurrency sets B,
B sets ``M = B*W``, and ``M`` picks the fp4 linear arm across ``_MGEMV``/``_MX``,
so two concurrencies can run two reduction orders on one question. What is NOT
known is whether the questions that actually flipped are the near-ties that
mechanism predicts.

**Prediction, written before the run:** every flipped question's top-2 logit gap
(over the four letter tokens) sits below the ~0.153 median |delta top-1| the
width ladder measured for an arm change. A flip at a wide gap is not arithmetic;
it is a defect, and it matters far more than 0.4%.

One process, one card, one slice, but a **fresh engine with no prefix store per
arm**: sharing an engine would let the first arm's published prefixes shorten the
second arm's prefill, which changes the padded width and so changes ``M`` -- the
variable under test. Prefix hits are asserted zero in both arms.

Every question's four letter logits are recorded, not only the ones that flip, so
the flip rate has a denominator: the fraction of ALL questions sitting inside the
arm-change delta is what says whether 4 is the expected count.

    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/probe_mmlu_concurrency.py \
        --source /work/Qwen3.8-27B-NVFP4 --n 1000 --out /work/mmlucc
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path

import torch

from tilerl.config import qwen38_27b
from tilerl.engine import SamplingParams, build_engine
from tilerl.eval import LETTERS, letter, mmlu_questions
from tilerl.kv_cache import NoPrefixStore
from tilerl.model import load_hf
from tilerl.tokenizer import get_tokenizer
from tilerl_kernels.backend import get_backend


def arm(engine, tok, prompts, conc, allowed):
    """One greedy letter per prompt, plus the four letter logits at that position."""
    from tilerl.eval import generate

    margins: dict[int, list[float]] = {}
    sample = engine._sample_batch
    order: dict[int, int] = {}
    submit = engine.submit
    n = {"i": 0}

    def w_submit(input_ids, params=None):
        rid = submit(input_ids, params)
        order[rid] = n["i"]
        n["i"] += 1
        return rid

    def w_sample(rows):
        # every MMLU answer is the single token off the prefill forward
        if rows:
            lg = torch.stack([l for _, l, _ in rows]).float()
            sel = lg[:, list(allowed)]
            for k, (r, _, _) in enumerate(rows):
                margins.setdefault(order[r.req_id], sel[k].tolist())
        return sample(rows)

    engine.submit, engine._sample_batch = w_submit, w_sample
    try:
        sp = SamplingParams(temperature=0.0, max_new_tokens=1, seed=0, allowed_ids=allowed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        texts = generate(engine, tok, prompts, sp, conc)
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
    finally:
        engine.submit, engine._sample_batch = submit, sample
    return texts, margins, secs


def top2_gap(logits: list[float]) -> float:
    s = sorted(logits, reverse=True)
    return s[0] - s[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrencies", default="8,32")
    ap.add_argument("--ladder-delta", type=float, default=0.153,
                    help="median |delta top-1| for an arm change, from the width ladder")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    backend = get_backend()
    assert backend.device.type == "cuda", backend.device
    cfg = qwen38_27b()
    tok = get_tokenizer(a.source)
    model = load_hf(cfg, a.source)
    prompts, golds, subjects = mmlu_questions(a.n, a.seed)
    allowed = tuple(sorted({tok.encode(f" {c}")[-1] for c in LETTERS}
                           | {tok.encode(c)[-1] for c in LETTERS}))
    concs = [int(x) for x in a.concurrencies.split(",")]

    arms = {}
    for c in concs:
        # A fresh engine with no prefix store per arm. Sharing one engine would let
        # the first arm's published prefixes shorten the second arm's prefill, which
        # changes the padded width and so changes M -- the variable under test.
        engine = build_engine(cfg, model, backend, num_blocks=2048, num_slots=c + 2,
                              max_batch=c, max_total_tokens=8192,
                              prefix_store=NoPrefixStore())
        texts, margins, secs = arm(engine, tok, prompts, c, allowed)
        preds = [letter(t) for t in texts]
        ok = sum(p == g for p, g in zip(preds, golds))
        print(f"[conc {c:>3}] {ok}/{len(preds)} = {100 * ok / len(preds):.1f}%  {secs:.1f}s",
              flush=True)
        arms[c] = {"preds": preds, "margins": margins, "ok": ok, "secs": secs,
                   "hits": engine.stats()["prefix_hits"]}
        assert arms[c]["hits"] == 0, f"prefix hits at conc {c}: not a clean arm"
        engine = None
        torch.cuda.empty_cache()

    lo, hi = concs[0], concs[-1]
    A, B = arms[lo], arms[hi]
    flips = [i for i in range(len(prompts)) if A["preds"][i] != B["preds"][i]]
    gaps = [top2_gap(A["margins"][i]) for i in range(len(prompts)) if i in A["margins"]]
    print(f"\n=== concurrency {lo} vs {hi}: {A['ok']} vs {B['ok']} correct, "
          f"{len(flips)} answers differ of {len(prompts)} ===")
    if gaps:
        q = lambda f: sorted(gaps)[min(len(gaps) - 1, int(f * len(gaps)))]  # noqa: E731
        print(f"top-2 gap over ALL questions (conc {lo}): median {st.median(gaps):.3e}  "
              f"p10 {q(.1):.3e}  min {min(gaps):.3e}")
        under = sum(g < a.ladder_delta for g in gaps)
        print(f"  questions with gap < {a.ladder_delta} (ladder's arm-change delta): "
              f"{under}/{len(gaps)} = {100 * under / len(gaps):.2f}%")

    rows = []
    for i in flips:
        ga = top2_gap(A["margins"][i]) if i in A["margins"] else float("nan")
        gb = top2_gap(B["margins"][i]) if i in B["margins"] else float("nan")
        d = (max(abs(x - y) for x, y in zip(A["margins"][i], B["margins"][i]))
             if i in A["margins"] and i in B["margins"] else float("nan"))
        rows.append({"q": i, "subject": subjects[i], "gold": golds[i],
                     "pred_lo": A["preds"][i], "pred_hi": B["preds"][i],
                     "gap_lo": ga, "gap_hi": gb, "arm_delta": d,
                     "near_tie": bool(ga < a.ladder_delta)})
        print(f"  q{i:<4} {subjects[i][:22]:<22} gold {golds[i]} "
              f"{A['preds'][i]}->{B['preds'][i]}  gap {ga:.3e}/{gb:.3e}  "
              f"|delta| {d:.3e}  {'NEAR-TIE' if ga < a.ladder_delta else 'WIDE GAP <-- defect'}")

    if rows:
        nt = sum(r["near_tie"] for r in rows)
        print(f"\n{nt}/{len(rows)} flips are near-ties (gap < {a.ladder_delta}).")
        print("PREDICTION HELD" if nt == len(rows) else
              "PREDICTION FAILED: a flip at a wide gap is a defect, not arithmetic")
    else:
        print("\nno flips: the two concurrencies agree on every answer")

    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "concurrency.json").write_text(json.dumps({
        "n": a.n, "seed": a.seed, "concurrencies": concs, "ladder_delta": a.ladder_delta,
        "correct": {str(c): arms[c]["ok"] for c in concs},
        "secs": {str(c): arms[c]["secs"] for c in concs},
        "flips": rows, "gap_median": st.median(gaps) if gaps else None,
    }, indent=1))
    print(f"wrote {o / 'concurrency.json'}")


if __name__ == "__main__":
    main()
