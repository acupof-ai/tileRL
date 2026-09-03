"""V100 steady-state decode: per-step timing, not total/count.

The e2e's len(out)/dt divides by the ~540s first-tick JIT+capture — the
"548s-in-the-window trap" (docs/experience/wins/2026-08-29-sm70-volta-fp4-cell.md).
This measures per-step wall time, skips the first 3 ticks (prefill + capture),
and reports steady-state ms/tick + tok/s. Also asserts the decode is correct
("Paris") and that graph capture did not fall back to eager.

  PATH=/usr/local/cuda-12.4/bin:$PATH TILELANG_CACHE_DIR=/tmp/tl_sm70f16 \
    TILERL_TARGET=cuda TILERL_QWEN38_SOURCE=/data00/.../Qwen3.8-27B-NVFP4 \
    PYTHONPATH=packages/tilerl-kernels/src:src CUDA_VISIBLE_DEVICES=0 \
    python3 scripts/bench_decode_steady.py
"""

import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "tilerl-kernels" / "src"))


from tilerl import config as config_mod  # noqa: E402
from tilerl import model as model_mod  # noqa: E402
from tilerl.engine import SamplingParams, build_engine  # noqa: E402
from tilerl.server import get_tokenizer  # noqa: E402

src = os.environ["TILERL_QWEN38_SOURCE"]
cfg = config_mod.qwen38_27b()
print(f"loading fp4 27B from {src} ...", flush=True)
t0 = time.time()
model = model_mod.load_hf(cfg, src, fuse_projections=True)
print(f"loaded in {time.time() - t0:.1f}s", flush=True)

backend = __import__("tilerl_kernels.backend", fromlist=["get_backend"]).get_backend()
eng = build_engine(cfg, model, backend, num_blocks=256, num_slots=16, max_total_tokens=8192)
tok = get_tokenizer(src)
prompt = "The capital of France is"
ids = tok.encode(prompt)

# Capture graph-capture-fallback warnings (the engine warns + flips to eager).
captured_warnings = []
with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")
    rid = eng.submit(ids, SamplingParams(temperature=0.0, max_new_tokens=64))
    out, ticks = None, []
    while out is None:
        out = eng.take(rid)
        if out is not None:
            break
        t1 = time.perf_counter()
        eng.step()
        ticks.append(time.perf_counter() - t1)
    captured_warnings = [str(w.message) for w in wlist if "graph capture failed" in str(w.message)]

text = tok.decode(out)
print("OUTPUT:", repr(text[:200]), flush=True)
assert "Paris" in text, f"correctness broken: {text[:100]!r}"
assert not captured_warnings, f"graph capture fell back to eager: {captured_warnings}"

# Steady-state: skip the first 3 ticks (prefill eager + first decode capture).
steady = ticks[3:]
ms = sum(steady) / len(steady) * 1e3
print(f"ticks={len(ticks)} steady={len(steady)}  {ms:.1f} ms/tick  {1e3 / ms:.1f} tok/s", flush=True)
print(f"first 3 ticks: {[f'{t * 1e3:.0f}ms' for t in ticks[:3]]}", flush=True)
print("STEADY OK", flush=True)
