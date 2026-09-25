"""Mixed-batch device gate for the prefill draft skip (awb impl#D3).

Solo prefill cannot show the 6.5-10 s cost perf1 measured: it appears when a
prefilling row shares the draft forward with decoding rows (the w=512 prefill
widens the shared attention batch). This starts short requests first so they
are decoding, then admits the long prompt, and reports per PREFILL tick the
decode/prefill row counts, rows handed to draft.step, and its CUDA wall.

    ARM=old|new ... python3 scripts/probe_draft_prefill_mixed.py \
        --source ... --draft ... --long 5200 --short 300
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from tilerl import engine as engine_mod
from tilerl.build import build_engine
from tilerl.config import qwen38_27b
from tilerl.kv_cache import NoPrefixStore
from tilerl.model import load_hf
from tilerl.spec import load_draft

ARM = os.environ.get("ARM", "new")
_ticks: list = []
_cur = {"dec": 0, "pf": 0}


def _install():
    orig_run_forward = engine_mod.Engine._run_forward
    orig_draft_step = engine_mod.Engine._draft_step

    def run_forward(self, decodes, prefills, chunks):
        _cur["dec"], _cur["pf"] = len(decodes), len(prefills)
        return orig_run_forward(self, decodes, prefills, chunks)

    def draft_step(self, rows):
        pf = _cur["pf"]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = orig_draft_step(self, rows)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        if pf:
            _ticks.append(
                {"dec": _cur["dec"], "pf": pf, "draft_rows": len(rows), "ms": round(ms, 1)}
            )
        return r

    engine_mod.Engine._run_forward = run_forward
    engine_mod.Engine._draft_step = draft_step
    if ARM == "old":
        engine_mod.Engine._draft_prefill_skip_ids = lambda self, p, c: set()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--long", type=int, default=5200)
    p.add_argument("--short", type=int, default=300)
    p.add_argument("--new", type=int, default=64)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--chunk", type=int, default=512)
    args = p.parse_args()

    from tilerl_kernels.backend import get_backend

    from tilerl.engine import SamplingParams

    _install()
    backend = get_backend()
    assert backend.device.type == "cuda"
    cfg = qwen38_27b()
    model = load_hf(cfg, args.source)
    draft = load_draft(model, args.draft, attn_window_tokens=args.window)
    engine = build_engine(
        cfg,
        model,
        backend,
        num_blocks=512,
        num_slots=4,
        max_batch=4,
        max_total_tokens=16384,
        max_num_batched_tokens=args.chunk,
        decode_graph=True,
        prefix_store=NoPrefixStore(),
        draft=draft,
        spec_depth=1,
        sparse_k=128,
        sparse_min_tokens=0,
    )

    sp = SamplingParams(temperature=0.0, max_new_tokens=256, seed=0)
    for _ in range(3):
        engine.submit([20 + i % 200 for i in range(args.short)], sp)
    for _ in range(6):  # move the short rows into decode before the long prompt
        engine.poll()
        engine.step()
    rid = engine.submit(
        [10 + i % 200 for i in range(args.long)],
        SamplingParams(temperature=0.0, max_new_tokens=args.new, seed=0),
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = []
    for _ in range(100000):
        d = engine.poll()
        if rid in d:
            out = d[rid]
            break
        engine.step()
    torch.cuda.synchronize()
    st = engine.stats()
    res = {
        "arm": ARM,
        "ttft_ms": round((time.perf_counter() - t0) * 1000, 1),
        "pf_ticks": _ticks,
        "pf_draft_ms_total": round(sum(t["ms"] for t in _ticks if t["draft_rows"]), 1),
        "mixed_pf_ticks": sum(1 for t in _ticks if t["dec"] and t["draft_rows"]),
        "spec_accepted": st["spec_accepted"],
        "spec_drafted": st["spec_drafted"],
        "long_out_head": out[:16],
    }
    print(json.dumps(res))
    from pathlib import Path

    Path.home().joinpath(f"draft_mixed_{ARM}.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
