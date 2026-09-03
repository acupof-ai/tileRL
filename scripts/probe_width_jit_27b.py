"""How much of the P1 step time is TileLang compiling a new rectangle?

`wins/2026-09-03-grpo-step-width-is-a-jit-key.md` measured the mechanism on the
tiny model: `grpo_loop` sized its training batch from the rollout's own output,
so a step whose completions ended at a new length compiled a fresh kernel set --
**239.2 s across 8 distinct widths against 0.6 s fixed**, and 37.7 s for a novel
width against 71 ms for a repeat.

The recorded P1 number that this re-reads is `secs_per_step_median = 60.45` at
`max_new_tokens=256` on the 27B (`errors/2026-09-03-p1-ties-at-the-ceiling.md`),
where real GSM8K completions make the width vary freely. The tiny constant does
not transfer -- more kernels, bigger shapes, a different compiler path -- so this
measures the 27B constant directly and nothing else.

Method: call `rl_step` on the real checkpoint with LoRA, at widths that have
never been seen (`novel`) and then at a width already compiled (`repeat`). Same
process, same card, same batch shape apart from T. The difference between the
two is compile time; the `repeat` figure is the step's real compute.

**A cap on what this licenses**: it measures `rl_step`, not a whole GRPO step. A
P1 step is rollout + `rl_step`, and the rollout is the larger half at
`max_new_tokens=256`. So the number here bounds how much of 60.45 s was compile,
it does not by itself say what the fixed-width step costs end to end.

TILELANG_CACHE_DIR must point at a directory that does not exist yet -- the
probe asserts that. A shared cache makes an already-compiled width a hit wearing
the ``novel`` label, which is how two earlier runs of this probe reported 1.3x
and 1.0x for the same quantity
(errors/2026-09-03-probe-rebuilt-the-setup-in-the-wrong-order.md).

    CUDA_VISIBLE_DEVICES=6 TILERL_TARGET=cuda TILELANG_CACHE_DIR=/work/tlc_fresh_$$ \\
    PYTHONPATH=src:packages/tilerl-kernels/src python3 scripts/probe_width_jit_27b.py \\
        --source /work/Qwen3.8-27B-NVFP4
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from tilerl_kernels.backend import get_backend

from tilerl.autograd import AdamW
from tilerl.config import qwen38_27b
from tilerl.model import add_lora, load_hf
from tilerl.train import rl_step

_MAX_WIDTHS = 8  # iteration cap: every loop here is bounded


def one(model, backend, opt, trainable, group, plen, gen, seed) -> float:
    """One rl_step at total width plen+gen. Returns wall seconds."""
    t = plen + gen
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, model.cfg.vocab_size, (group, t)).astype(np.int64)
    adv = np.asarray(rng.normal(size=group), dtype=np.float32)
    plens = np.full(group, plen, dtype=np.int64)
    slens = np.full(group, t, dtype=np.int64)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    rl_step(model, ids, adv, plens, backend, opt, trainable=trainable,
            seq_lens=slens, micro=1)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--gen", type=int, default=256, help="the P1 recipe's max_new_tokens")
    ap.add_argument("--widths", type=int, default=3, help=f"novel widths to try (<= {_MAX_WIDTHS})")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    n_widths = min(a.widths, _MAX_WIDTHS)

    # A shared cache turns a HIT into a "novel" measurement and manufactures the
    # conclusion that compiling is free: two earlier runs of this probe reported
    # 1.3x and 1.0x for the same quantity because they shared one cache dir.
    # errors/2026-09-03-probe-rebuilt-the-setup-in-the-wrong-order.md
    cache = os.environ.get("TILELANG_CACHE_DIR", "")
    assert cache and not os.path.exists(cache), (
        f"TILELANG_CACHE_DIR={cache!r} must be set and NOT yet exist: a novel "
        "width that is already compiled is a cache hit wearing the label"
    )

    be = get_backend()
    assert be.device.type == "cuda", be.device
    cfg = qwen38_27b()
    # keep_master=False: LoRA on a frozen base needs no bf16 masters (~27 GB on
    # the 27B, and with them the backward OOMs at 95.01 of 95.22 GiB), which is
    # what _train_adapters does for the P1 run this probe is pricing.
    model = load_hf(cfg, a.source, keep_master=False)
    # materialize BEFORE add_lora: materialize() rebuilds any param whose
    # device/dtype differs and the new object has a new id(), so adapters
    # attached first point at tensors the forward never reads -- the tape then
    # records no parameter gradient and _step's `assert acc, _NO_GRAD` fires.
    # _train_adapters gets this ordering from build_engine; here it is explicit.
    model.params = be.materialize(model.params)
    trainable = add_lora(model, rank=16)
    opt = AdamW(lr=1e-4)
    print(f"27B loaded, {len(trainable)} LoRA tensors, group {a.group}, "
          f"prompt {a.prompt_len}, gen {a.gen}", flush=True)

    # Warm: the first call compiles everything shape-independent as well, so it
    # is reported separately and excluded from the novel-width figures.
    warm = one(model, be, opt, trainable, a.group, a.prompt_len, a.gen, 0)
    print(f"[warm ] T={a.prompt_len + a.gen} {warm:8.1f}s   (first call: all kernels)",
          flush=True)

    novel, repeat = [], []
    for i in range(1, n_widths + 1):
        s = one(model, be, opt, trainable, a.group, a.prompt_len, a.gen - i, 100 + i)
        novel.append(s)
        print(f"[novel] T={a.prompt_len + a.gen - i} {s:8.1f}s", flush=True)
    for i in range(1, n_widths + 1):
        s = one(model, be, opt, trainable, a.group, a.prompt_len, a.gen - i, 200 + i)
        repeat.append(s)
        print(f"[rpt  ] T={a.prompt_len + a.gen - i} {s:8.1f}s   (already compiled)",
              flush=True)

    mn, mr = float(np.median(novel)), float(np.median(repeat))
    print(f"\nnovel  median {mn:8.2f}s  (n={len(novel)})")
    print(f"repeat median {mr:8.2f}s  (n={len(repeat)})")
    print(f"compile cost per new width: {mn - mr:8.2f}s   ratio {mn / mr:.1f}x")
    print("\nP1 recorded secs_per_step_median = 60.45 (rollout + rl_step, gen 256).")
    print(f"  rl_step alone at a repeated width: {mr:.2f}s")
    print(f"  a novel width adds {mn - mr:.2f}s to whichever step hits it.")
    print("  This bounds how much of 60.45 was compile; it does not price the "
          "whole fixed-width step, because the rollout is the other half.")

    if a.out:
        o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
        (o / "width_jit_27b.json").write_text(json.dumps(
            {"group": a.group, "prompt_len": a.prompt_len, "gen": a.gen,
             "warm_s": warm, "novel_s": novel, "repeat_s": repeat,
             "novel_median_s": mn, "repeat_median_s": mr,
             "compile_per_width_s": mn - mr}, indent=1))
        print(f"wrote {o / 'width_jit_27b.json'}")


if __name__ == "__main__":
    main()
