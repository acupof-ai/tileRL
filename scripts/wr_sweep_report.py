#!/usr/bin/env python3
"""Aggregate the W×R sweep: quality vs the R=1 reference of the SAME window,
plus speed. Reads probe_wr_sweep_worker outputs only — no engine, no CUDA.

Quality: each prompt's output compared token-by-token against that window's
R=1 run: agreement, first divergence position (null = identical over the
shared length), first_diverge mod R (refresh-boundary clustering).

Speed: mean wall and eff tok/s across prompts per arm, read from each arm's
<out>.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys


def load_pp(d: str) -> dict[int, list[int]]:
    out = {}
    for p in sorted(glob.glob(f"{d}/ref_*.json")):
        with open(p) as f:
            obj = json.load(f)
        idx = int(p.rsplit("_", 1)[-1].split(".")[0])
        out[idx] = obj["output"]
    return out


def first_diverge(a: list[int], b: list[int]) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir holding arm_Wwww_Rrr[.json,_pp/]")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    windows = (128, 1024)
    rs = (1, 8, 16, 32)
    rows = []
    for w in windows:
        refs = None
        for r in rs:
            tag = f"arm_W{w}_R{r}"
            path = f"{args.dir}/{tag}.json"
            if not os.path.exists(path):
                # An interrupted window: completed arms on disk still count.
                rows.append({"window": w, "R": r, "missing": True})
                continue
            with open(path) as f:
                meta = json.load(f)
            if not meta.get("structural_gate_ok"):
                print(f"FATAL {tag} structural gate not ok", file=sys.stderr)
                return 1
            if r == 1 and os.path.isdir(f"{args.dir}/{tag}_pp"):
                refs = load_pp(f"{args.dir}/{tag}_pp")
            arm = load_pp(f"{args.dir}/{tag}_pp")
            q = []
            if refs is not None and set(arm) == set(refs):
                for idx in sorted(refs):
                    fd = first_diverge(refs[idx], arm[idx])
                    q.append({
                        "idx": idx,
                        "agreement": round(
                            sum(a == b for a, b in zip(refs[idx], arm[idx]))
                            / min(len(refs[idx]), len(arm[idx])), 6),
                        "first_diverge": fd,
                        "mod_R": (fd % r) if fd is not None else None,
                    })
            fds = sorted(x["first_diverge"] for x in q if x["first_diverge"] is not None)
            tok = [p["eff_tok_s"] for p in meta["prompts"]]
            walls = [p["wall_s"] for p in meta["prompts"]]
            rows.append({
                "window": w, "R": r,
                "quality_available": refs is not None and set(arm) == set(refs),
                "n_prompts": len(q),
                "median_first_diverge": statistics.median(fds) if fds else None,
                "min_first_diverge": fds[0] if fds else None,
                "n_identical": sum(1 for x in q if x["first_diverge"] is None),
                "mean_agreement": round(statistics.mean(x["agreement"] for x in q), 6) if q else None,
                "divergence_mod_R": [x["mod_R"] for x in q],
                "mean_wall_s": round(statistics.mean(walls), 2),
                "mean_eff_tok_s": round(statistics.mean(tok), 3),
                "per_prompt": q,
                "speed_prompts": meta["prompts"],
            })
    payload = {"arms": rows}
    text = json.dumps(payload, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(f"{'W':>5} {'R':>3} {'ident':>5} {'med1stDiv':>9} "
          f"{'meanAgree':>9} {'wall_s':>7} {'eff_tok/s':>9}")
    for r in rows:
        if r.get("missing"):
            print(f"{r['window']:>5} {r['R']:>3}   (missing — interrupted before this arm)")
            continue
        print(f"{r['window']:>5} {r['R']:>3} {r['n_identical']:>5} "
              f"{str(r['median_first_diverge']):>9} {r['mean_agreement']:>9} "
              f"{r['mean_wall_s']:>7} {r['mean_eff_tok_s']:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
