"""Do the 4 MMLU answers that moved between concurrencies sit at near-ties?

Recorded on clean main: 746/1000 at concurrency 32 and 742/1000 at concurrency 8,
same slice (``seed`` pins the draw). The mechanism is known -- concurrency sets B,
B sets ``M = B*W``, and ``M`` picks the fp4 linear arm across ``_MGEMV``/``_MX``,
so two concurrencies can run two reduction orders on one question. What is NOT
known is whether the questions that actually flipped are the near-ties that
mechanism predicts.

**Prediction, written before the first run:** every flipped question's top-2 gap
sits below the ~0.153 median |delta top-1| the width ladder measured for an arm
change. A flip at a wide gap is a defect, not arithmetic.

**That first run REFUTED the premise, not the prediction.** Concurrency 8 and 32
returned 742/1000 each and **0 of 1000 answers differ**, with 41 questions (4.1%)
sitting under 0.153 -- so near-ties existed and none moved. Concurrency does not
change an MMLU answer, and the reason is that ``M`` on a prefill tick is set by
``_PREFILL_BUCKET`` (64) from the prompt's own padded length, not by the batch:
concurrency batches independent rows, each keeping its own width. The
``M = B*W`` reasoning is a DECODE-tick mechanism, and MMLU at
``max_new_tokens=1`` runs no decode ticks.

The variable that does differ between the two callers is **fusion**:
``scripts/mmlu.py`` passes ``fuse_projections=True``, ``cli.py`` defaults to
False. Fusion concatenates q/k/v (and gate/up) into one weight, so the fused GEMM
has a different N and a different reduction structure than three separate ones.
The probe now crosses both knobs and labels which one moved each flip.

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
    ap.add_argument("--fuse", default="0,1", help="fuse_projections arms: 0, 1, or 0,1")
    ap.add_argument("--ladder-delta", type=float, default=0.153,
                    help="median |delta top-1| for an arm change, from the width ladder")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    backend = get_backend()
    assert backend.device.type == "cuda", backend.device
    cfg = qwen38_27b()
    tok = get_tokenizer(a.source)
    prompts, golds, subjects = mmlu_questions(a.n, a.seed)
    fuses = [bool(int(x)) for x in a.fuse.split(",")]
    allowed = tuple(sorted({tok.encode(f" {c}")[-1] for c in LETTERS}
                           | {tok.encode(c)[-1] for c in LETTERS}))
    concs = [int(x) for x in a.concurrencies.split(",")]

    arms = {}
    for fuse in fuses:
        # fuse_projections concatenates q/k/v (and gate/up) into one weight, so the
        # fused GEMM has a different N and a different reduction structure than three
        # separate ones. scripts/mmlu.py passes True, cli.py defaults to False.
        model = load_hf(cfg, a.source, fuse_projections=fuse)
        for c in concs:
            # A fresh engine with no prefix store per arm: shared prefixes would
            # shorten the next arm's prefill and change its padded width.
            engine = build_engine(cfg, model, backend, num_blocks=2048, num_slots=c + 2,
                                  max_batch=c, max_total_tokens=8192,
                                  prefix_store=NoPrefixStore())
            texts, margins, secs = arm(engine, tok, prompts, c, allowed)
            preds = [letter(t) for t in texts]
            ok = sum(p == g for p, g in zip(preds, golds))
            key = (c, fuse)
            print(f"[conc {c:>3} fuse {int(fuse)}] {ok}/{len(preds)} = "
                  f"{100 * ok / len(preds):.1f}%  {secs:.1f}s", flush=True)
            arms[key] = {"preds": preds, "margins": margins, "ok": ok, "secs": secs,
                         "hits": engine.stats()["prefix_hits"]}
            assert arms[key]["hits"] == 0, f"prefix hits at {key}: not a clean arm"
            engine = None
            torch.cuda.empty_cache()
        model = None
        torch.cuda.empty_cache()

    # every pair of arms, so which knob moves an answer is attributable
    keys = sorted(arms)
    pairs = [(x, y) for i, x in enumerate(keys) for y in keys[i + 1:]]
    allrows = []
    for x, y in pairs:
        A, B = arms[x], arms[y]
        flips = [i for i in range(len(prompts)) if A["preds"][i] != B["preds"][i]]
        gaps = [top2_gap(A["margins"][i]) for i in range(len(prompts)) if i in A["margins"]]
        lbl = f"conc{x[0]}/fuse{int(x[1])} vs conc{y[0]}/fuse{int(y[1])}"
        knob = ("concurrency" if x[1] == y[1] else
                "fusion" if x[0] == y[0] else "both")
        print(f"\n=== {lbl}  [{knob}]: {A['ok']} vs {B['ok']} correct, "
              f"{len(flips)} of {len(prompts)} answers differ ===")
        if gaps and not allrows:
            q = lambda f: sorted(gaps)[min(len(gaps) - 1, int(f * len(gaps)))]  # noqa: E731
            under = sum(g < a.ladder_delta for g in gaps)
            print(f"  top-2 gap over ALL questions: median {st.median(gaps):.3e}  "
                  f"p10 {q(.1):.3e}  min {min(gaps):.3e}")
            print(f"  questions with gap < {a.ladder_delta}: {under}/{len(gaps)} = "
                  f"{100 * under / len(gaps):.2f}%  <- the flip count's denominator")
        for i in flips:
            ga, gb = top2_gap(A["margins"][i]), top2_gap(B["margins"][i])
            d = max(abs(u - v) for u, v in zip(A["margins"][i], B["margins"][i]))
            near = ga < a.ladder_delta
            allrows.append({"pair": lbl, "knob": knob, "q": i, "subject": subjects[i],
                            "gold": golds[i], "a": A["preds"][i], "b": B["preds"][i],
                            "gap_a": ga, "gap_b": gb, "arm_delta": d, "near_tie": near})
            print(f"  q{i:<4} {subjects[i][:20]:<20} gold {golds[i]} "
                  f"{A['preds'][i]}->{B['preds'][i]}  gap {ga:.3e}  |delta| {d:.3e}  "
                  f"{'NEAR-TIE' if near else 'WIDE GAP <-- defect'}")
        if not flips:
            print("  no flips: this knob moves no answer")
    if allrows:
        nt = sum(r["near_tie"] for r in allrows)
        print(f"\n{nt}/{len(allrows)} flips are near-ties (gap < {a.ladder_delta}).")
        print("PREDICTION HELD" if nt == len(allrows) else
              "PREDICTION FAILED: a flip at a wide gap is a defect, not arithmetic")
    else:
        print("\nno flips on any pair: neither knob moves an MMLU answer")
    # the denominator, off the first arm: how many questions COULD flip under an
    # arm-sized perturbation, which is what makes a flip count readable
    ref = arms[keys[0]]["margins"]
    allgaps = [top2_gap(v) for v in ref.values()]

    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "concurrency.json").write_text(json.dumps({
        "n": a.n, "seed": a.seed, "concurrencies": concs, "ladder_delta": a.ladder_delta,
        "fuse": [int(f) for f in fuses],
        "correct": {f"conc{k[0]}_fuse{int(k[1])}": arms[k]["ok"] for k in arms},
        "secs": {f"conc{k[0]}_fuse{int(k[1])}": arms[k]["secs"] for k in arms},
        "flips": allrows,
        "gap_median": st.median(allgaps) if allgaps else None,
        "under_delta": sum(g < a.ladder_delta for g in allgaps),
        "questions_scored": len(allgaps),
    }, indent=1))
    print(f"wrote {o / 'concurrency.json'}")


if __name__ == "__main__":
    main()
