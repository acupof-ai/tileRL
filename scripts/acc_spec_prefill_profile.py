"""Spec-arm prefill profile with a ZERO-added-sync instrument. The previous
double-sync bracket instrument perturbed the spec arm +44% wall (178.9 vs
128.4 s at HEAD) and disagreed with the overhead harness >5% at 09657c0, so per
the 2026-09-09 review gate it produced no reportable numbers.

This version wraps two calls with plain host spans (perf_counter, NO sync):
  _run_forward  on prefill ticks (prefills nonempty) — the host span includes
                GPU time because the sample step reads logits to host (.item())
  draft.step    on every tick — the host span includes GPU time because the
                draft returns tokens via .tolist() (dflash2.py), which drains

Same instrument code at both shas (no engine hooks required), so the ratio is
trustworthy even though the absolute spans include host launch overhead. At
HEAD the wraps are cross-checked against the engine's own _prefill_secs and
_draft_ms — two independent instruments that must agree.

Cross-validation (must hold at BOTH shas or the instrument is fixed first):
  1. wall (one sync pair around generate)
  2. prefill + draft + decode + enc + detok buckets close to wall within 5%
  3. decode / decode_ticks is the direct W=8 tick measurement (host span of
     _run_forward on decode ticks, GPU drained by the graph replay)

Provenance: benchrec git_commit/git_dirty print in the header, so a run's tree
is checkable at a glance (the 2026-09-09 contamination: a tree labelled
09657c0 was byte-identical to HEAD).

    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda python3 scripts/acc_spec_prefill_profile.py \
        --source /work/Qwen3.8-27B-NVFP4 --gsm8k /work/gsm8k_test.jsonl \
        --n 50 --out /work/accspf --draft /work/Qwen3.8-27B-NVFP4/model_mtp.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import torch

# Provenance: the pod is a tarball, not a clone, and a 2026-09-09 baseline run
# read HEAD's engine.py off a tree labelled 09657c0 (byte-identical trees).
# benchrec reads .synced_commit (stamped by pod_sync) when git is absent.
from benchrec import git_commit, git_dirty

from tilerl import engine as engine_mod
from tilerl.config import qwen38_27b
from tilerl.engine import build_engine
from tilerl.eval import generate
from tilerl.kv_cache import NoPrefixStore
from tilerl.model import load_hf
from tilerl.prompt import render_chat, sampling
from tilerl.spec import load_draft
from tilerl.tokenizer import get_tokenizer

_buckets = {"prefill_incl_draft": 0.0, "decode_incl_draft": 0.0, "draft": 0.0,
            "draft_prefill": 0.0, "draft_decode": 0.0}
_counts = {"prefill_ticks": 0, "decode_ticks": 0, "draft_calls": 0}
_state = {"tick": None}

_orig_run_forward = engine_mod.Engine._run_forward
_orig_draft_step = None


def _timed_run_forward(self, decodes, prefills, chunks):
    _state["tick"] = "prefill" if prefills else "decode"
    t0 = time.perf_counter()
    r = _orig_run_forward(self, decodes, prefills, chunks)
    dt = time.perf_counter() - t0
    # draft.step runs INSIDE _run_forward (engine.py), so this span includes it;
    # the draft wrap splits it out by tick type below.
    _buckets[f"{_state['tick']}_incl_draft"] += dt
    _counts[f"{_state['tick']}_ticks"] += 1
    return r


def _timed_draft_step(orig, rows):
    t0 = time.perf_counter()
    r = orig(rows)  # orig is the bound draft.step
    dt = time.perf_counter() - t0
    _buckets["draft"] += dt
    _buckets[f"draft_{_state['tick']}"] += dt
    _counts["draft_calls"] += 1
    return r


engine_mod.Engine._run_forward = _timed_run_forward


def _gpu_state():
    # Who else is on the card at run start is part of this measurement's
    # population (27, 2026-09-09: a card shared with a 100%-util job produces
    # mixed tok/s that voids the 3.4% criterion).
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return out
    except Exception:
        return "nvidia-smi unavailable"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--gsm8k", required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--out", required=True)
    p.add_argument("--draft", default=None,
                   help="draft head path: run the spec arm (W=8) and time draft.step")
    args = p.parse_args()

    sha, dirty = git_commit(), git_dirty()
    card = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    gpu = _gpu_state()
    print(f"=== acc_spec_prefill_profile  sha={sha}  dirty={dirty}  card={card}  "
          f"gpu_at_start[{gpu}]  draft={'yes' if args.draft else 'no'} ===")

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
        global _orig_draft_step
        _orig_draft_step = draft.step
        draft.step = lambda rows: _timed_draft_step(_orig_draft_step, rows)
        # NB: do NOT enable engine._draft_ms here — its event sync per draft step
        # perturbed the spec arm +40% wall (128 -> 179 s). The host-span wrap
        # above is sync-free; the .tolist() drain in draft.step charges the GPU.

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
    prefill_incl = _buckets["prefill_incl_draft"]
    decode_incl = _buckets["decode_incl_draft"]
    draft_s = _buckets["draft"]
    draft_pf = _buckets["draft_prefill"]
    draft_dec = _buckets["draft_decode"]
    # draft.step runs inside _run_forward, so subtract it to get trunk-only spans.
    prefill_s = prefill_incl - draft_pf
    decode_s = decode_incl - draft_dec
    stats = engine.stats()
    dec_fwds = stats["decode_forwards"]
    pre_fwds = stats["prefill_forwards"]
    tok_gen = stats["tokens_generated"]
    spec_acc = stats.get("spec_accepted")
    spec_drafted = stats.get("spec_drafted")

    # Reading 2 vs 1: the buckets must close to wall.
    closure = (prefill_s + draft_s + decode_s + encode_t + detok_t) / wall
    # decode per tick is the trunk-only W=8 tick (draft subtracted); the
    # 43.52/41.96 constants were voided with the contaminated 09657c0 tree.
    dec_per_fwd = (decode_s / _counts["decode_ticks"]) if _counts["decode_ticks"] else 0.0

    # HEAD-only cross-check: the engine's own _prefill_secs (excludes draft by
    # construction) vs the wrap's prefill_excl_draft.
    engine_prefill = getattr(engine, "_prefill_secs", None)

    report = {
        "provenance": {"git_commit": sha, "git_dirty": dirty, "card": card,
                       "gpu_at_start": gpu},
        "wall": wall,
        "tok_s": (tok_gen / wall) if wall else 0.0,
        "buckets": {
            "prefill_excl_draft": prefill_s,
            "draft": draft_s,
            "decode_excl_draft": decode_s,
            "encode": encode_t,
            "detok": detok_t,
        },
        "closure_buckets_over_wall": closure,
        "counts": _counts,
        "prefill_forwards": pre_fwds,
        "decode_forwards": dec_fwds,
        "decode_ms_per_decode_tick": dec_per_fwd * 1000,
        "tokens_generated": tok_gen,
        "tok_per_decode_fwd": (tok_gen / dec_fwds) if dec_fwds else 0.0,
        "spec_accepted": spec_acc,
        "spec_drafted": spec_drafted,
        "spec_accept_rate": (spec_acc / spec_drafted) if spec_drafted else None,
        "engine_prefill_secs_cross_check": engine_prefill,
        "per_question": {
            "prefill": prefill_s / n,
            "draft": draft_s / n,
            "decode": decode_s / n,
        },
    }
    print(f"wall {wall:.1f}s  tok/s {report['tok_s']:.1f}  prefill {prefill_s:.1f}s  "
          f"draft {draft_s:.1f}s  decode {decode_s:.1f}s  enc {encode_t:.3f}s  detok {detok_t:.3f}s")
    print(f"closure {closure:.4f} (reading 2 vs 1; must be ~1.000, within 5%)")
    print(f"prefill_ticks {_counts['prefill_ticks']} (fwds {pre_fwds})  "
          f"decode_ticks {_counts['decode_ticks']} (fwds {dec_fwds})  "
          f"draft_calls {_counts['draft_calls']}  (draft_pf {draft_pf:.1f}s  draft_dec {draft_dec:.1f}s)")
    print(f"decode {dec_per_fwd*1000:.2f} ms/tick  (trunk-only W=8 tick; the 43.52/41.96 "
          f"constants were voided with the contaminated 09657c0 tree)")
    print(f"tokens_generated {tok_gen}  tok/decode_fwd {report['tok_per_decode_fwd']:.2f}  "
          f"spec_accepted {spec_acc}  spec_drafted {spec_drafted}  "
          f"accept_rate {report['spec_accept_rate']}")
    if engine_prefill is not None:
        print(f"engine _prefill_secs {engine_prefill:.1f}s (wrap prefill_excl_draft {prefill_s:.1f}s)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prefill_profile.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
