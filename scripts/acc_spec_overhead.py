"""Where the non-decode wall goes — prefill / encode / detokenize / scheduling —
per arm, 50 GSM8K questions, to attribute the 31.6s -> 71.9s overhead growth
between 09657c0 and the current sha.

Decode and prefill come from a class-level patch on ``Engine._run_forward``
(sync on both sides); encode/detokenize from tokenizer wrappers; scheduling is
the remainder of the arm wall. Run at both shas and diff the columns.

    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/acc_spec_overhead.py \
        --source /work/Qwen3.8-27B-NVFP4 --draft /work/Qwen3.8-27B-DFlash2 \
        --gsm8k /work/gsm8k_test.jsonl --n 50 --out /work/accspec_oh
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import torch

from tilerl import engine as engine_mod
from tilerl.config import qwen38_27b
from tilerl.engine import build_engine
from tilerl.eval import generate
from tilerl.kv_cache import NoPrefixStore
from tilerl.model import load_hf
from tilerl.prompt import render_chat, sampling
from tilerl.tokenizer import get_tokenizer

_cur: dict = {}
_orig_run_forward = engine_mod.Engine._run_forward


def _timed_run_forward(self, decodes, prefills, chunks):
    if decodes or prefills:
        kind = "prefill" if prefills else "decode"
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = _orig_run_forward(self, decodes, prefills, chunks)
        torch.cuda.synchronize()
        _cur[kind] = _cur.get(kind, 0.0) + (time.perf_counter() - t0)
        return r
    return _orig_run_forward(self, decodes, prefills, chunks)


engine_mod.Engine._run_forward = _timed_run_forward


def _run_arm(cfg, model, backend, tok, rows, sp, draft_path, width):
    from tilerl.spec import load_draft

    for name in ("encode", "detok"):
        _cur[name] = 0.0
    _cur["encode_n"] = _cur["detok_n"] = 0
    enc, dec_t = tok.encode, tok.decode
    tok.encode = lambda s: _wrap(enc, s, "encode")
    tok.decode = lambda ids: _wrap(dec_t, ids, "detok")
    draft = load_draft(model, draft_path) if draft_path else None
    engine = build_engine(cfg, model, backend, num_blocks=512, num_slots=1, max_batch=1,
                          draft=draft, spec_depth=max(1, width - 1), decode_graph=True,
                          prefix_store=NoPrefixStore())
    prompts = [render_chat([("user", r["prompt"])], False) for r in rows]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    generate(engine, tok, prompts, sp, 1)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    stats = engine.stats()
    tok.encode, tok.decode = enc, dec_t
    engine = draft = None
    torch.cuda.empty_cache()
    return wall, dict(_cur), stats


def _wrap(fn, arg, name):
    t0 = time.perf_counter()
    r = fn(arg)
    _cur[name] = _cur.get(name, 0.0) + (time.perf_counter() - t0)
    _cur[f"{name}_n"] = _cur.get(f"{name}_n", 0) + 1
    return r


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--gsm8k", required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from tilerl_kernels.backend import get_backend

    backend = get_backend()
    assert backend.device.type == "cuda", "needs TILERL_TARGET=cuda"
    cfg = qwen38_27b()
    tok = get_tokenizer(args.source)
    model = load_hf(cfg, args.source)
    rows = [json.loads(ln) for ln in Path(args.gsm8k).read_text().splitlines() if ln.strip()][: args.n]
    sp = replace(sampling(tok, False, 512, temperature=0.0, max_think_tokens=0, seed=0),
                 temperature=0.0)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    order = (("base", None, 1), ("spec-w8", args.draft, 8))
    if os.environ.get("OH_REVERSE"):
        order = tuple(reversed(order))
    for name, draft_path, width in order:
        _cur.clear()
        wall, parts, stats = _run_arm(cfg, model, backend, tok, rows, sp, draft_path, width)
        dec, pre = parts.get("decode", 0.0), parts.get("prefill", 0.0)
        enc, det = parts.get("encode", 0.0), parts.get("detok", 0.0)
        sched = wall - dec - pre - enc - det
        per_q = {k: v / args.n for k, v in
                 (("wall", wall), ("decode", dec), ("prefill", pre), ("encode", enc),
                  ("detok", det), ("scheduling", sched))}
        report[name] = {"wall": wall, "decode": dec, "prefill": pre, "encode": enc,
                        "detok": det, "scheduling": sched, "per_question": per_q,
                        "encode_n": parts.get("encode_n", 0), "detok_n": parts.get("detok_n", 0),
                        "prefix_hits": stats.get("prefix_hits"),
                        "prefix_misses": stats.get("prefix_misses")}
        print(f"[{name}] wall {wall:.1f}s  decode {dec:.1f}  prefill {pre:.1f}  "
              f"encode {enc:.4f}s x{parts.get('encode_n', 0)}  "
              f"detok {det:.4f}s x{parts.get('detok_n', 0)}  "
              f"scheduling {sched:.1f}  prefix_hits {stats.get('prefix_hits')}  "
              f"(per-q: sched {per_q['scheduling']*1000:.0f}ms)", flush=True)

    (out / "overhead.json").write_text(json.dumps(report))


if __name__ == "__main__":
    main()
