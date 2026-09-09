"""Spec-arm prefill profile using the engine's OWN timing hooks — no added device
syncs. The previous double-sync bracket instrument (Model.forward + draft.step,
synced on both sides) perturbed the spec arm +44% wall (178.9 vs 128.4 s at
HEAD) and disagreed with the overhead harness >5% at both shas, so per the
2026-09-09 review gate it produced no reportable numbers. This version reads
fields the engine already maintains:

  _prefill_secs   host span of every prefill tick (engine.py:1046), sync-free —
                  the sample step's logits drain drains the GPU implicitly
  _draft_ms       CUDA events around each draft step (engine.py:1295), the
                  engine's own sanctioned draft instrument (one event sync per
                  tick, priced in the tick-timing bench)

Cross-validation, two independent readings that must agree within 5% at BOTH
shas or the instrument is fixed before any number is reported:

  1. wall (one sync pair around generate)
  2. prefill_secs + draft_ms + decode_remainder (+ encode/detok, ~0.02 s)
  3. decode_remainder / decode_forwards vs the directly-timed W=8 tick
     (43.52 ms at 09657c0, 41.96 at HEAD) — a remainder-derived number only
     stands with a direct-measurement control.

    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/acc_spec_prefill_profile.py \
        --source /work/Qwen3.8-27B-NVFP4 --gsm8k /work/gsm8k_test.jsonl \
        --n 50 --out /work/accspf --draft /work/Qwen3.8-27B-NVFP4/model_mtp.safetensors
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from tilerl.config import qwen38_27b
from tilerl.engine import build_engine
from tilerl.eval import generate
from tilerl.kv_cache import NoPrefixStore
from tilerl.model import load_hf
from tilerl.prompt import render_chat, sampling
from tilerl.spec import load_draft
from tilerl.tokenizer import get_tokenizer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--gsm8k", required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--out", required=True)
    p.add_argument("--draft", default=None,
                   help="draft head path: run the spec arm (W=8) and time draft.step")
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

    prompts = [render_chat([("user", r["prompt"])], False) for r in rows]

    encode_t = 0.0
    enc_orig = tok.encode

    def _enc(s):
        nonlocal encode_t
        t0 = time.perf_counter()
        r = enc_orig(s)
        encode_t += time.perf_counter() - t0
        return r

    tok.encode = _enc
    draft = load_draft(model, args.draft) if args.draft else None
    engine = build_engine(cfg, model, backend, num_blocks=512, num_slots=1, max_batch=1,
                          draft=draft, spec_depth=max(1, 8 - 1) if draft else 0,
                          decode_graph=True, prefix_store=NoPrefixStore())
    if draft is not None:
        engine._draft_ms = []  # enable the engine's own event-sync draft timing

    detok_t = 0.0
    dec_orig = tok.decode

    def _dec(ids):
        nonlocal detok_t
        t0 = time.perf_counter()
        r = dec_orig(ids)
        detok_t += time.perf_counter() - t0
        return r

    tok.decode = _dec

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    generate(engine, tok, prompts, sp, 1)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    n = args.n
    prefill_s = engine._prefill_secs
    prefill_tokens = engine._prefill_tokens
    prefill_fwds = engine._prefill_forwards
    draft_ev = list(engine._draft_ms or [])
    draft_s = sum(ms for _, ms in draft_ev) / 1000.0
    draft_fwds = sum(f for f, _ in draft_ev)
    stats = engine.stats()
    dec_fwds = stats["decode_forwards"]
    tok_gen = stats["tokens_generated"]
    spec_acc = stats.get("spec_accepted")
    spec_drafted = stats.get("spec_drafted")

    decode_s = wall - prefill_s - draft_s - encode_t - detok_t
    dec_per_fwd = decode_s / dec_fwds if dec_fwds else 0.0
    accept = (tok_gen / dec_fwds) if dec_fwds else 0.0
    spec_rate = (spec_acc / spec_drafted) if spec_drafted else None

    # Cross-validation reading 2 vs 1: the buckets must close to wall.
    closure = (prefill_s + draft_s + decode_s + encode_t + detok_t) / wall
    # Reading 3: decode per forward vs the directly-timed W=8 tick.
    tick_096 = 43.52
    tick_head = 41.96

    report = {
        "wall": wall,
        "buckets": {
            "prefill_secs": prefill_s,
            "draft_ms": draft_s,
            "decode_remainder": decode_s,
            "encode": encode_t,
            "detok": detok_t,
        },
        "closure_buckets_over_wall": closure,
        "prefill_tokens": prefill_tokens,
        "prefill_forwards": prefill_fwds,
        "decode_forwards": dec_fwds,
        "draft_forwards": draft_fwds,
        "decode_s_per_decode_fwd_ms": dec_per_fwd * 1000,
        "tick_direct_ms": {"09657c0": tick_096, "HEAD": tick_head},
        "tokens_generated": tok_gen,
        "tok_per_decode_fwd": accept,
        "spec_accepted": spec_acc,
        "spec_drafted": spec_drafted,
        "spec_accept_rate": spec_rate,
        "per_question": {
            "prefill": prefill_s / n,
            "draft": draft_s / n,
            "decode": decode_s / n,
        },
    }
    print(f"wall {wall:.1f}s  prefill {prefill_s:.1f}s  draft {draft_s:.1f}s  "
          f"decode_rem {decode_s:.1f}s  enc {encode_t:.3f}s  detok {detok_t:.3f}s")
    print(f"closure {closure:.4f} (reading 2 vs 1; must be ~1.000)")
    print(f"prefill_fwds {prefill_fwds}  decode_fwds {dec_fwds}  draft_fwds {draft_fwds}")
    print(f"decode {dec_per_fwd*1000:.2f} ms/fwd  (direct tick: {tick_096} at 09657c0, "
          f"{tick_head} at HEAD)")
    print(f"tokens_generated {tok_gen}  tok/decode_fwd {accept:.2f}  "
          f"spec_accepted {spec_acc}  spec_drafted {spec_drafted}  "
          f"accept_rate {spec_rate}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prefill_profile.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
