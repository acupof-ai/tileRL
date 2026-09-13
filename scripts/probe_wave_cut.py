#!/usr/bin/env python3
"""#557 follow-up / the open B=8 spec-wave defect
(docs/experience/errors/2026-09-13-sm90-b8-spec-wave-not-reproducible.md),
card-5 H20, served shape.

Two IDENTICAL waves of the same 8 rows, each wave a freshly built engine:

  wave 1: spec OFF  (sparse eager; main forces decode_graph off under sparse)
  wave 2: spec ON   (real MTP draft, W=1; same prompts, same per-row seeds)

The earlier symptom was cold-vs-cold across two processes agreeing only 3/8.
This cut instead asks whether the W=1 verify tick perturbs output at all:
spec-on sparse-vs-sparse must reproduce spec-off row for row under greedy.
Per row: token-equal over STEPS, else the first differing committed step and
the full-logit max_abs of BOTH arms' trunk logits AT that committed position.

A logit stored at a global position is OVERWRITTEN each tick: a rejected draft
tail scores a position one tick early, and the tick that actually commits the
position scores it again — the final write is the committed trunk logit.

One process, two sequential 27B loads (first engine shut down before the
second). Card 5 by default. CPU counterpart:
test_b8_spec_wave_reproduces_a_b8_plain_wave_row_for_row in
tests/test_sparse_engine.py.
"""

import os
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np
from tilerl_kernels.backend import get_backend

from tilerl import cli
from tilerl.cli import _build_model
from tilerl.engine import Engine, SamplingParams, build_engine
from tilerl.tokenizer import get_tokenizer

SRC = os.environ.get("SRC", "/work/tilerl-ckpt/Qwen3.8-27B-NVFP4")
DRAFT = os.environ.get("DRAFT", SRC + "/model_mtp.safetensors")
N = 8
PREFILL = int(os.environ.get("PREFILL", "400"))
STEPS = int(os.environ.get("STEPS", "40"))
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "5")
os.environ.setdefault("TILERL_TARGET", "cuda")
cli._QWEN38_SOURCE = SRC
tok = get_tokenizer(SRC)
backend = get_backend()
rng = np.random.default_rng(0)
_prompts = [
    np.asarray(
        [7] + rng.integers(10, tok.vocab_size - 100, size=PREFILL - 1).tolist(), dtype=np.int64
    )
    for _ in range(N)
]


def run(spec_on: bool):
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True, backend=backend)
    kw = dict(
        num_blocks=0,
        num_slots=N + 2,
        max_batch=N,
        max_total_tokens=PREFILL + 1024,
        max_num_batched_tokens=512,
        sparse_k=128,
        scorer="bounds",
        kv_cold_bytes=1 << 34,
        decode_graph=False,
    )
    if spec_on:
        from tilerl.spec import load_draft

        kw["draft"] = load_draft(model, DRAFT)
        kw["spec_depth"] = 1
    e = build_engine(cfg, model, backend, **kw)

    # (rid, global committed position) -> (argmax, full logits). Plain
    # assignment: the tick that commits g is the last tick that scores g.
    logits = {}
    orig_sample = Engine._sample_batch

    def hook(self2, rows):
        for r, l, g in rows:
            logits[r.req_id, int(g)] = (int(l.argmax()), l.detach().float().clone().cpu())
        return orig_sample(self2, rows)

    Engine._sample_batch = hook

    sp = SamplingParams(temperature=0.0, seed=0, max_new_tokens=STEPS)
    rids = [e.submit(p, sp) for p in _prompts]
    toks = {r: [] for r in rids}
    deadline = time.time() + 1800
    try:
        while True:
            e.step()
            for r, tt in e.poll().items():
                toks[r].extend(int(t) for t in tt)
            if all(len(toks[r]) >= STEPS for r in rids):
                break
            if time.time() > deadline:
                print("TIMEOUT", {r: len(toks[r]) for r in rids}, flush=True)
                break
    finally:
        Engine._sample_batch = orig_sample
        e.shutdown()
    return {
        "toks": [toks[r][:STEPS] for r in rids],
        "lg": {i: {g: v for (rid, g), v in logits.items() if rid == r} for i, r in enumerate(rids)},
    }


print("WAVE spec-off then spec-on, same 8 rows, fresh engines", flush=True)
off = run(False)
on = run(True)
eq = sum(off["toks"][i] == on["toks"][i] for i in range(N))
print(f"WAVE token-equal rows: {eq}/{N}", flush=True)
for i in range(N):
    a, b = off["toks"][i], on["toks"][i]
    for j in range(min(len(a), len(b))):
        if a[j] != b[j]:
            la = off["lg"][i].get(j)
            lb = on["lg"][i].get(j)
            gap = float((la[1] - lb[1]).abs().max()) if la and lb else float("nan")
            print(
                f"row{i} first_diff step{j} off_tok={a[j]} on_tok={b[j]} logit_maxabs={gap:.4g}",
                flush=True,
            )
            break
    else:
        print(f"row{i} equal ({len(a)} tok)", flush=True)
print("WAVECUT_DONE", flush=True)
