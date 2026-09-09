"""Per-decode-forward wall time for the W=1 and W=8 ticks, plus a spec-vs-spec
rerun control for the completion diff — one process, warm cache, card 6.

The 2026-09-09 reproduction measured the W=8 spec arm at 126.5 tok/s against the
recorded 135.5, with acceptance and tok/decode-fwd reproduced. Reverse-derived
from tok/s and tok/decode-fwd, the W=8 verify tick looked ~8% slower (48.9 vs
45.2 ms) while the W=1 tick matched — but a reverse-derived millisecond carries
both quantities' populations, so this script times each decode tick directly.
Runs the spec arm twice and diffs completions: ~38/200 divergences between base
and spec were recorded with n=1; two identical spec runs say whether that is
run-to-run nondeterminism or a spec-path bug.

    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/acc_spec_tick_timing.py \
        --source /work/Qwen3.8-27B-NVFP4 --draft /work/Qwen3.8-27B-DFlash2 \
        --gsm8k /work/gsm8k_test.jsonl --out /work/accspec_tick
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from tilerl import engine as engine_mod
from tilerl.config import qwen38_27b
from tilerl.model import load_hf
from tilerl.prompt import sampling
from tilerl.tokenizer import get_tokenizer

# One timing list per arm, swapped between arm() calls. Class-level patch so it
# survives arm()'s own instance-level trace wrapping (it wraps this wrapper).
_cur: list[tuple[int, float]] = []
_orig_run_forward = engine_mod.Engine._run_forward


def _timed_run_forward(self, decodes, prefills, chunks):
    if decodes and not prefills:
        w = 1 + max((len(r.drafts) for r in decodes), default=0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = _orig_run_forward(self, decodes, prefills, chunks)
        torch.cuda.synchronize()
        _cur.append((w, (time.perf_counter() - t0) * 1000))
        return r
    return _orig_run_forward(self, decodes, prefills, chunks)


engine_mod.Engine._run_forward = _timed_run_forward


def _stats(ms: list[float]) -> dict:
    s = sorted(ms)
    n = len(s)
    return {"n": n, "mean": sum(s) / n, "p50": s[n // 2], "p90": s[int(n * 0.9)],
            "max": s[-1]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--gsm8k", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from acc_spec_arms import arm
    from tilerl_kernels.backend import get_backend

    backend = get_backend()
    assert backend.device.type == "cuda", "needs TILERL_TARGET=cuda"
    cfg = qwen38_27b()
    tok = get_tokenizer(args.source)
    model = load_hf(cfg, args.source)
    rows = [json.loads(ln) for ln in Path(args.gsm8k).read_text().splitlines() if ln.strip()][:200]
    params = sampling(tok, False, 512, temperature=0.0, max_think_tokens=0, seed=0)

    global _cur
    outs = {}
    for name, draft_path, width in (("base", None, 1), ("spec-a", args.draft, 8),
                                   ("spec-b", args.draft, 8)):
        ticks: list[tuple[int, float]] = []
        _cur = ticks
        sp = replace(params, temperature=0.0)
        out = arm(name, cfg, model, backend, tok, draft_path, width, 0, rows, sp, True, 1)
        by_width: dict[int, list[float]] = {}
        for w, ms in ticks:
            by_width.setdefault(w, []).append(ms)
        out["tick_ms"] = {str(w): _stats(ms) for w, ms in sorted(by_width.items())}
        outs[name] = out
        print(f"[{name}] tick ms by width: " +
              "  ".join(f"W={w} n={d['n']} mean={d['mean']:.2f} p50={d['p50']:.2f} "
                        f"p90={d['p90']:.2f} max={d['max']:.2f}"
                        for w, d in out["tick_ms"].items()), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, o in outs.items():
        (out / f"{name}.json").write_text(json.dumps(o))

    a, b = outs["spec-a"]["gsm8k"]["text"], outs["spec-b"]["gsm8k"]["text"]
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    print(f"\nspec-vs-spec: {len(diff)}/200 completions differ across two identical runs "
          f"(base-vs-spec was 38/200)", flush=True)
    acc = {n: (outs[n]["gsm8k"]["correct"], outs[n]["gsm8k"]["timing"]["tok_per_s"])
           for n in outs}
    print("accuracy/tok-s: " + "  ".join(f"{n} {v[0]}/200 {v[1]:.1f}" for n, v in acc.items()),
          flush=True)


if __name__ == "__main__":
    main()
