"""Profile where the B=1 prefill wall goes, in five buckets per the 2026-09-09
review: (1) tokenize + prompt render, (2) block allocation / prefix match (host
side; the KV write is in-kernel and lands in bucket 3), (3) kernel time
(Model.forward, sync on both sides), (4) chunk-loop scheduling (plan minus
admit, forward host, step remainder), (5) the prefix store path
(_match_prefix + store calls; ~0 under NoPrefixStore).

The first prefill step's kernel time is recorded separately: a one-time first-
use cost lives there, not in the steady-state per-question numbers.

    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/acc_spec_prefill_profile.py \
        --source /work/Qwen3.8-27B-NVFP4 --gsm8k /work/gsm8k_test.jsonl \
        --n 50 --out /work/accpf
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
from tilerl_kernels.backend import Backend

from tilerl import engine as engine_mod
from tilerl.config import qwen38_27b
from tilerl.engine import build_engine
from tilerl.eval import generate
from tilerl.kv_cache import NoPrefixStore
from tilerl.model import Model, load_hf
from tilerl.prompt import render_chat, sampling
from tilerl.tokenizer import get_tokenizer

_state = {"pf_step": False}
_pf: dict = {}
_kernel_steps: list = []
_first_kernel: list = []
_counts = {"pf_steps": 0, "dec_steps": 0}
_plan_sel: Counter = Counter()


def _add(bucket: str, dt: float) -> None:
    _pf[bucket] = _pf.get(bucket, 0.0) + dt


_orig_step = engine_mod.Engine.step
_orig_build_plan = engine_mod.Engine._build_plan
_orig_admit = getattr(engine_mod.Engine, "_admit", None)
_orig_match_prefix = engine_mod.Engine._match_prefix
_orig_run_forward = engine_mod.Engine._run_forward
_orig_model_forward = Model.forward


def _timed_step(self):
    t0 = time.perf_counter()
    r = _orig_step(self)
    dt = time.perf_counter() - t0
    if _state["pf_step"]:
        _add("step_total", dt)
        _counts["pf_steps"] += 1
    else:
        _counts["dec_steps"] += 1
    _state["pf_step"] = False
    return r


def _timed_build_plan(self):
    t0 = time.perf_counter()
    r = _orig_build_plan(self)
    if r[1]:  # prefills nonempty -> this is a prefill step
        _state["pf_step"] = True
        _add("plan", time.perf_counter() - t0)
    return r


def _timed_admit(self, req):
    t0 = time.perf_counter()
    r = _orig_admit(self, req)
    _add("admit", time.perf_counter() - t0)  # admit only runs on prefill requests
    return r


def _timed_match_prefix(self, tokens):
    t0 = time.perf_counter()
    r = _orig_match_prefix(self, tokens)
    _add("match_prefix", time.perf_counter() - t0)
    return r


def _timed_run_forward(self, decodes, prefills, chunks):
    t0 = time.perf_counter()
    r = _orig_run_forward(self, decodes, prefills, chunks)
    if _state["pf_step"]:
        _add("forward", time.perf_counter() - t0)
    return r


def _timed_model_forward(self, *a, **kw):
    # Sync only on prefill steps: a synchronize during decode-graph capture is
    # illegal and silently flips the engine to eager decode for every tick.
    if not _state["pf_step"]:
        return _orig_model_forward(self, *a, **kw)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = _orig_model_forward(self, *a, **kw)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    _add("kernel", dt)
    _kernel_steps.append(dt)
    if not _first_kernel:
        _first_kernel.append(dt)
    return r


engine_mod.Engine.step = _timed_step
engine_mod.Engine._build_plan = _timed_build_plan
if hasattr(engine_mod.Engine, "_admit"):  # 09657c0 admits inline in _build_plan
    engine_mod.Engine._admit = _timed_admit
engine_mod.Engine._match_prefix = _timed_match_prefix
engine_mod.Engine._run_forward = _timed_run_forward
Model.forward = _timed_model_forward

_orig_plan = Backend._plan


def _rec_plan(self, op, m, n, k):
    r = _orig_plan(self, op, m, n, k)
    if _state["pf_step"]:
        _plan_sel[(op, m, str(r))] += 1
    return r


Backend._plan = _rec_plan


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
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

    render_t = 0.0
    prompts = []
    for r in rows:
        t0 = time.perf_counter()
        prompts.append(render_chat([("user", r["prompt"])], False))
        render_t += time.perf_counter() - t0

    encode_t = 0.0
    enc_orig = tok.encode

    def _enc(s):
        nonlocal encode_t
        t0 = time.perf_counter()
        r = enc_orig(s)
        encode_t += time.perf_counter() - t0
        return r

    tok.encode = _enc
    engine = build_engine(cfg, model, backend, num_blocks=512, num_slots=1, max_batch=1,
                          decode_graph=True, prefix_store=NoPrefixStore())

    store_t = 0.0
    store = engine._prefix
    for name in ("lookup", "prefetch_if_worth_it", "evict_until_free", "reclaimable_blocks"):
        fn = getattr(store, name)

        def _wrap_store(*a, _fn=fn, **kw):
            nonlocal store_t
            t0 = time.perf_counter()
            r = _fn(*a, **kw)
            store_t += time.perf_counter() - t0
            return r

        setattr(store, name, _wrap_store)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    generate(engine, tok, prompts, sp, 1)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    n = args.n
    plan = _pf.get("plan", 0.0)
    admit = _pf.get("admit", 0.0)
    forward = _pf.get("forward", 0.0)
    kernel = _pf.get("kernel", 0.0)
    step_total = _pf.get("step_total", 0.0)
    rest = _kernel_steps[1:] if len(_kernel_steps) > 1 else _kernel_steps
    report = {
        "wall": wall,
        "per_question": {
            "1. tokenize+render": (encode_t + render_t) / n,
            "2. admit (block alloc)": admit / n,
            "   match_prefix": _pf.get("match_prefix", 0.0) / n,
            "   store calls": store_t / n,
            "3. kernel (Model.forward)": kernel / n,
            "4a. plan excl admit": (plan - admit) / n,
            "4b. forward host": (forward - kernel) / n,
            "4c. step remainder": (step_total - plan - forward) / n,
        },
        "totals_50q": {
            "1. tokenize+render": encode_t + render_t,
            "2. admit": admit,
            "3. kernel": kernel,
            "4. scheduling": (step_total - kernel - admit),
            "step_total": step_total,
            "wall": wall,
        },
        "kernel_first_step": _first_kernel[0] if _first_kernel else None,
        "kernel_rest_median": statistics.median(rest) if rest else None,
        "kernel_rest_max": max(rest) if rest else None,
        "pf_steps": _counts["pf_steps"],
        "dec_steps": _counts["dec_steps"],
    }
    print(f"wall {wall:.1f}s  pf_steps {_counts['pf_steps']}  dec_steps {_counts['dec_steps']}")
    for k, v in report["per_question"].items():
        print(f"  {k:30s} {v*1000:8.1f} ms/q")
    print(f"  kernel first step {(_first_kernel[0] if _first_kernel else 0)*1000:.1f} ms, "
          f"rest median {statistics.median(rest)*1000 if rest else 0:.1f} ms, "
          f"rest max {max(rest)*1000 if rest else 0:.1f} ms")
    print("linear dispatch during prefill (op, M -> plan): count")
    for (op, m, plan), c in sorted(_plan_sel.items()):
        print(f"  {op} M={m}: {plan} x{c}")
    report["linear_dispatch"] = {
        f"{op} M={m} -> {plan}": c for (op, m, plan), c in sorted(_plan_sel.items())
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prefill_profile.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
