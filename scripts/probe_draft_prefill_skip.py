"""Device gate for skipping the draft forward on prefill chunks (awb impl#D3).

Two process arms on the SAME fixed prompt, differing only in whether
Engine._draft_prefill_skip_ids is patched to the OLD behavior (never skip):

    ARM=old|new CUDA_VISIBLE_DEVICES=... PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/probe_draft_prefill_skip.py \
        --source /work/Qwen3.8-27B-NVFP4 --draft <head> --tokens 5200 --new 128

Reports: prefill-tick draft forwards, their summed CUDA-event-free wall ms,
decode spec accepted/drafted, and the emitted token list (compare across arms).
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

_pf_tick = {"v": False}
_pf_draft = {"calls": 0, "ms": 0.0}
_pf_ticks: list = []


def _install_probes() -> None:
    orig_build_plan = engine_mod.Engine._build_plan
    orig_draft_step = engine_mod.Engine._draft_step

    def build_plan(self):
        dec, pf, ch = orig_build_plan(self)
        _pf_tick["v"] = bool(pf)
        return dec, pf, ch

    def draft_step(self, rows):
        if _pf_tick["v"]:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = orig_draft_step(self, rows)
            torch.cuda.synchronize()
            if rows:
                _pf_draft["calls"] += 1
                _pf_draft["ms"] += (time.perf_counter() - t0) * 1000
            return r
        return orig_draft_step(self, rows)

    engine_mod.Engine._build_plan = build_plan
    engine_mod.Engine._draft_step = draft_step
    if ARM == "old":
        engine_mod.Engine._draft_prefill_skip_ids = lambda self, p, c: set()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--tokens", type=int, default=5200)
    p.add_argument("--new", type=int, default=128)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--sparse-k", type=int, default=128)
    args = p.parse_args()

    from tilerl_kernels.backend import get_backend

    _install_probes()
    backend = get_backend()
    assert backend.device.type == "cuda", "needs TILERL_TARGET=cuda"
    cfg = qwen38_27b()
    model = load_hf(cfg, args.source)
    draft = load_draft(model, args.draft, attn_window_tokens=args.window)
    engine = build_engine(
        cfg,
        model,
        backend,
        num_blocks=512,
        num_slots=1,
        max_batch=1,
        max_total_tokens=args.tokens + args.new + 64,
        max_num_batched_tokens=args.chunk,
        decode_graph=True,
        prefix_store=NoPrefixStore(),
        draft=draft,
        spec_depth=1,
        sparse_k=args.sparse_k,
        sparse_min_tokens=0,
    )

    ids = [10 + (i % 200) for i in range(args.tokens)]
    from tilerl.engine import SamplingParams

    torch.cuda.synchronize()
    t_start = time.perf_counter()
    ttft = None
    rid = engine.submit(ids, SamplingParams(temperature=0.0, max_new_tokens=args.new, seed=0))
    out = []
    for _ in range(100000):
        d = engine.poll()
        if rid in d:
            out = d[rid]
            break
        engine.step()
        if ttft is None:
            for r in engine._running:
                if r.req_id == rid and r.output:
                    torch.cuda.synchronize()
                    ttft = (time.perf_counter() - t_start) * 1000
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t_start) * 1000
    st = engine.stats()
    res = {
        "arm": ARM,
        "tokens": args.tokens,
        "chunk": args.chunk,
        "window": args.window,
        "ttft_ms": round(ttft, 1) if ttft is not None else None,
        "wall_ms": round(wall, 1),
        "pf_draft_calls": _pf_draft["calls"],
        "pf_draft_ms": round(_pf_draft["ms"], 1),
        "spec_drafted": st["spec_drafted"],
        "spec_accepted": st["spec_accepted"],
        "out_head": out[:32],
        "out_len": len(out),
    }
    print(json.dumps(res))
    from pathlib import Path

    out_path = Path(os.environ.get("D3_OUT", Path.home() / f"draft_skip_{ARM}.json"))
    out_path.write_text(json.dumps({"res": res, "out": out}))


if __name__ == "__main__":
    main()
