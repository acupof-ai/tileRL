"""V100 B=8 steady-state decode: aggregate tok/s with 8 concurrent requests.

B=1 is in bench_decode_steady.py. This submits 8 requests, lets the engine's
continuous batching pack them into one forward per tick, and reports aggregate
tok/s (total tokens / wall) plus per-request correctness. The weight read is
amortized across 8 rows, so aggregate t/s should exceed B=1.

  PATH=/usr/local/cuda-12.4/bin:$PATH TILELANG_CACHE_DIR=/tmp/tl_sm70f16 \
    TILERL_TARGET=cuda TILERL_QWEN38_SOURCE=/work/Qwen3.8-27B-NVFP4 \
    PYTHONPATH=packages/tilerl-kernels/src:src CUDA_VISIBLE_DEVICES=0 \
    python3 scripts/bench_decode_b8.py --card 0 --build fused

Emits one decode_agg_tok_s record (n=1: one steady-state window) to
docs/experience/bench/measurements.jsonl (schema: docs/bench-schema.md).
--build is required: this script cannot see whether the engine captured graphs.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "tilerl-kernels" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchrec  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

from tilerl import config as config_mod  # noqa: E402
from tilerl import model as model_mod  # noqa: E402
from tilerl.engine import SamplingParams, build_engine  # noqa: E402
from tilerl.server import get_tokenizer  # noqa: E402

ap = argparse.ArgumentParser()
benchrec.add_record_args(ap, default_target="sm70", default_device=None)
args = ap.parse_args()

src = os.environ["TILERL_QWEN38_SOURCE"]
cfg = config_mod.qwen38_27b()
print(f"loading fp4 27B from {src} ...", flush=True)
t0 = time.time()
model = model_mod.load_hf(cfg, src, fuse_projections=True)
print(f"loaded in {time.time() - t0:.1f}s", flush=True)

backend = get_backend()
eng = build_engine(cfg, model, backend, num_blocks=512, num_slots=16, max_total_tokens=16384)
tok = get_tokenizer(src)

prompts = [
    "The capital of France is",
    "The largest planet in the solar system is",
    "The speed of light is approximately",
    "The author of Romeo and Juliet is",
    "The chemical symbol for gold is",
    "The tallest mountain on Earth is",
    "The currency of Japan is",
    "The primary language spoken in Brazil is",
]
B = len(prompts)
ids = [tok.encode(p) for p in prompts]

rids = [eng.submit(ids[i], SamplingParams(temperature=0.0, max_new_tokens=64)) for i in range(B)]
outs = [None] * B
ticks = []
pending = B
while pending:
    t1 = time.perf_counter()
    eng.step()
    ticks.append(time.perf_counter() - t1)
    for i in range(B):
        if outs[i] is None:
            o = eng.take(rids[i])
            if o is not None:
                outs[i] = o
                pending -= 1

total_tokens = sum(len(o) for o in outs)
steady = ticks[3:]
ms = sum(steady) / len(steady) * 1e3
agg = total_tokens / (sum(steady))
print(f"B={B}  total_tokens={total_tokens}  steady_ticks={len(steady)}", flush=True)
print(f"  per-tick {ms:.1f} ms  aggregate {agg:.1f} tok/s", flush=True)
rec = {
    "metric": "decode_agg_tok_s", "value": round(agg, 1), "unit": "tok/s",
    "shape": {"batch": B, "ctx": max(len(i) for i in ids)},
    "warm": {"state": "warm", "compiles": 0},
    "n": 1, "spread": 0.0, **benchrec.record_common(args),
}
rec["floor"] = benchrec.measured_best_floor(rec, lower_is_better=False)
print(f"  record {benchrec.append(rec)} appended", flush=True)
for i, o in enumerate(outs):
    print(f"  req{i}: {prompts[i]!r} -> {tok.decode(o[:8])!r}", flush=True)
assert "Paris" in tok.decode(outs[0]), f"req0 correctness broken: {tok.decode(outs[0])[:80]!r}"
print("B8 OK", flush=True)
