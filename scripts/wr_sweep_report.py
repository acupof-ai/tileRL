#!/usr/bin/env python3
"""Aggregate the W×R sweep. No engine; reads the worker's artifacts.

Speed comes from each arm's FREE run (<tag>.json prompts[*].eff_tok_s / wall /
graph-eager tick split). Quality comes from the TEACHER-FORCED runs: every R
arm and its W-group's R=1 force the SAME anchor (R=1's free-run stream), so
their recorded per-position trunk logits are comparable 1:1.

  binding gate  top1 agreement of the R arm's TF logits vs R=1's, all
                positions AND the first 16 separately, >= 0.99
  report-only   symmetric KL (KL(a||b), KL(b||a)) median/p90/mean, and the
                top1/top2 margin distribution at disagreeing positions
  self gate     the R=1 arm's TF top1 must equal its own anchor at every
                position (self_anchor_gate_ok in its json)

Interruptible: a missing arm is reported, not fatal.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

WINDOWS = (128, 1024)
RS = (1, 16, 32, 8)
GATE = 0.99
FIRST_N = 16


def load_tf(d: str, idx: int) -> dict | None:
    pt = f"{d}/tf_{idx:03d}.pt"
    if not os.path.exists(pt):
        return None
    obj = torch.load(pt, map_location="cpu", weights_only=True)
    return {"top1": obj["top1"], "logits": obj["logits"].float()}


def kl_pair(la: torch.Tensor, lb: torch.Tensor) -> tuple[float, float]:
    pa = torch.softmax(la, -1)
    lga = torch.log_softmax(la, -1)
    lgb = torch.log_softmax(lb, -1)
    return float((pa * (lga - lgb)).sum().item()), float(
        (torch.softmax(lb, -1) * (lgb - lga)).sum().item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = []
    for w in WINDOWS:
        for r in RS:
            tag = f"arm_W{w}_R{r}"
            meta_path = f"{args.dir}/{tag}.json"
            if not os.path.exists(meta_path):
                rows.append({"window": w, "R": r, "missing": True})
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            if not meta.get("structural_gate_ok"):
                print(f"FATAL {tag} structural gate not ok", file=sys.stderr)
                return 1
            speed = meta["prompts"]
            row = {
                "window": w, "R": r,
                "mean_eff_tok_s": round(statistics.mean(p["eff_tok_s"] for p in speed), 3),
                "mean_wall_s": round(statistics.mean(p["wall_s"] for p in speed), 2),
                "graph_tick_ms_mean": round(
                    statistics.mean(p["graph_tick_ms_mean"] for p in speed
                                     if p["graph_ticks"]), 3),
                "eager_tick_ms_mean": round(
                    statistics.mean(p["eager_tick_ms_mean"] for p in speed
                                     if p["eager_refresh_ticks"]), 3),
                "self_anchor_gate_ok": meta.get("self_anchor_gate_ok"),
            }
            if r == 1:
                if not meta.get("self_anchor_gate_ok"):
                    print(f"FATAL {tag} R=1 self-anchor gate not ok", file=sys.stderr)
                    return 1
            else:
                ref_dir = f"{args.dir}/arm_W{w}_R1_tf"
                arm_dir = f"{args.dir}/{tag}_tf"
                agree_all, agree_first, kl_ab, kl_ba, margins = [], [], [], [], []
                for idx in range(len(speed)):
                    a = load_tf(arm_dir, idx)
                    b = load_tf(ref_dir, idx)
                    if a is None or b is None:
                        continue
                    n = min(len(a["top1"]), len(b["top1"]))
                    ta, tb = a["top1"][:n], b["top1"][:n]
                    eq = ta == tb
                    agree_all.append(float(eq.float().mean()))
                    agree_first.append(float(eq[:FIRST_N].float().mean()))
                    la, lb = a["logits"][:n], b["logits"][:n]
                    ka, kb = kl_pair(la, lb)
                    kl_ab.append(ka)
                    kl_ba.append(kb)
                    for i in (~eq).nonzero().flatten().tolist():
                        ta2 = la[i].topk(2)
                        tb2 = lb[i].topk(2)
                        margins.append({
                            "pos": i,
                            "a_top1": int(ta2.indices[0]),
                            "a_margin": round(float(ta2.values[0] - ta2.values[1]), 4),
                            "b_top1": int(tb2.indices[0]),
                            "b_margin": round(float(tb2.values[0] - tb2.values[1]), 4),
                        })

                def stats(xs):
                    if not xs:
                        return None
                    xs = sorted(xs)
                    return {"mean": round(statistics.mean(xs), 6),
                            "median": round(statistics.median(xs), 6),
                            "p90": round(xs[min(len(xs) - 1,
                                                int(round(0.9 * (len(xs) - 1))))], 6)}

                row.update({
                    "n_prompts_compared": len(agree_all),
                    "top1_agreement": round(statistics.mean(agree_all), 6) if agree_all else None,
                    "top1_agreement_first16": round(statistics.mean(agree_first), 6)
                        if agree_first else None,
                    "kl_a_b": stats(kl_ab), "kl_b_a": stats(kl_ba),
                    "n_disagreements": len(margins),
                    "margins_sample": margins[:20],
                    "gate_ok": bool(agree_all and statistics.mean(agree_all) >= GATE
                                    and statistics.mean(agree_first) >= GATE),
                })
            rows.append(row)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"gate": GATE, "arms": rows}, f, indent=2)
    print(f"{'W':>5} {'R':>3} {'effTok/s':>8} {'graph ms':>8} {'eager ms':>8} "
          f"{'agree':>6} {'first16':>7} {'gate':>5} {'nDiv':>5} {'KLmed+/-':>10}")
    for r in rows:
        if r.get("missing"):
            print(f"{r['window']:>5} {r['R']:>3}  (missing)")
            continue
        klm = r.get("kl_a_b")
        kls = f"{klm['median']:.4f}/{r['kl_b_a']['median']:.4f}" if klm else "-"
        print(f"{r['window']:>5} {r['R']:>3} {r['mean_eff_tok_s']:>8} "
              f"{str(r['graph_tick_ms_mean']):>8} {str(r['eager_tick_ms_mean']):>8} "
              f"{str(r.get('top1_agreement', '-')):>6} "
              f"{str(r.get('top1_agreement_first16', '-')):>7} "
              f"{str(r.get('gate_ok', 'ref')):>5} {str(r.get('n_disagreements', '-')):>5} "
              f"{kls:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
