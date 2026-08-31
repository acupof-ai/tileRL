"""Profile the decode tick: GPU forward vs sampling vs Python overhead.

Also reports tick 1 (prefill) time to measure the M=32 chunking speedup.
Run on the V100 pod. First run includes JIT; re-run for warm-cache numbers.

  PATH=/usr/local/cuda-12.4/bin:$PATH TILELANG_CACHE_DIR=/tmp/tl_sm70f16 \
    TILERL_TARGET=cuda TILERL_QWEN38_SOURCE=/data00/.../Qwen3.8-27B-NVFP4 \
    PYTHONPATH=packages/tilerl-kernels/src:src CUDA_VISIBLE_DEVICES=0 \
    python3 scripts/prof_decode_tick.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "tilerl-kernels" / "src"))

import torch  # noqa: E402

from tilerl import config as config_mod  # noqa: E402
from tilerl import model as model_mod  # noqa: E402
from tilerl.engine import Engine, SamplingParams, build_engine  # noqa: E402
from tilerl.server import get_tokenizer  # noqa: E402

src = os.environ["TILERL_QWEN38_SOURCE"]
cfg = config_mod.qwen38_27b()
print("loading...", flush=True)
t0 = time.time()
model = model_mod.load_hf(cfg, src, fuse_projections=True)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

from tilerl_kernels.backend import get_backend  # noqa: E402

backend = get_backend()
eng = build_engine(cfg, model, backend, num_blocks=64, num_slots=8, max_total_tokens=8192)
tok = get_tokenizer(src)

prompts = ["The capital of France is", "The largest planet is", "Speed of light is",
           "Romeo and Juliet author", "Gold symbol is", "Tallest mountain is",
           "Japan currency is", "Brazil language is"]
B = len(prompts)
ids = [tok.encode(p) for p in prompts]
rids = [eng.submit(ids[i], SamplingParams(temperature=0.0, max_new_tokens=16)) for i in range(B)]

# Per-tick timing: accumulate sub-phase ms into `cur`, snapshot per tick.
cur = {}
rows = []

# Count + time ALL backend method calls (the prefill's real bottleneck).
be_timings = {}  # name -> [count, total_ms]
for _name in dir(backend):
    if _name.startswith("_"):
        continue
    _attr = getattr(backend, _name)
    if not callable(_attr):
        continue
    _orig = _attr

    def _make_wrapper(n, o):
        def w(*a, **kw):
            t0 = time.perf_counter()
            r = o(*a, **kw)
            dt = (time.perf_counter() - t0) * 1e3
            e = be_timings.setdefault(n, [0, 0.0])
            e[0] += 1
            e[1] += dt
            return r
        return w

    setattr(backend, _name, _make_wrapper(_name, _orig))

orig_plan = Engine._build_plan
orig_fwd = Engine._run_forward
orig_sample = Engine._sample_batch


def _acc(key, ms):
    cur[key] = cur.get(key, 0) + ms


def patched_plan(self):
    t0 = time.perf_counter()
    r = orig_plan(self)
    _acc("plan", (time.perf_counter() - t0) * 1e3)
    return r


def patched_fwd(self, d, p, c):
    t0 = time.perf_counter()
    r = orig_fwd(self, d, p, c)
    torch.cuda.synchronize()
    _acc("fwd", (time.perf_counter() - t0) * 1e3)
    return r


def patched_sample(self, rows_):
    t0 = time.perf_counter()
    r = orig_sample(self, rows_)
    _acc("sample", (time.perf_counter() - t0) * 1e3)
    return r


Engine._build_plan = patched_plan
Engine._run_forward = patched_fwd
Engine._sample_batch = patched_sample

outs = [None] * B
pending = B
while pending:
    cur.clear()
    t1 = time.perf_counter()
    eng.step()
    dt = (time.perf_counter() - t1) * 1e3
    cur["step"] = dt
    rows.append(dict(cur))
    nwait = len(getattr(eng, "_waiting", []))
    nrun = len(getattr(eng, "_running", []))
    nfin = len(getattr(eng, "_finished", {}))
    ngraphs = len(getattr(eng, "_decode_graphs", {}))
    print(f"tick {len(rows)}: {dt:.1f} ms  graphs={ngraphs}  "
          f"wait={nwait} run={nrun} fin={nfin}", flush=True)
    for i in range(B):
        if outs[i] is None:
            o = eng.take(rids[i])
            if o is not None:
                outs[i] = o
                pending -= 1

print(f"\nfirst req0: {tok.decode(outs[0][:8])!r}", flush=True)

# Prefill (tick 1)
r0 = rows[0]
print(f"\n=== Prefill (tick 1, includes JIT on cold run) ===", flush=True)
print(f"  step:       {r0.get('step', 0):.1f} ms", flush=True)
print(f"  build_plan: {r0.get('plan', 0):.1f} ms", flush=True)
print(f"  forward:    {r0.get('fwd', 0):.1f} ms", flush=True)
print(f"  sampling:   {r0.get('sample', 0):.1f} ms", flush=True)
print(f"  linear_fp4 calls: {be_timings.get('linear_fp4', [0,0])[0]}  "
      f"total: {be_timings.get('linear_fp4', [0,0])[1]:.1f} ms", flush=True)
print(f"\n  Backend method breakdown (prefill tick 1):", flush=True)
for name, (cnt, ms) in sorted(be_timings.items(), key=lambda x: -x[1][1]):
    if ms > 100:  # only show methods taking >100ms
        print(f"    {name:25s} {cnt:>6d} calls  {ms:>10.1f} ms  avg {ms/cnt:.2f} ms", flush=True)

# Decode breakdown (ticks 4+, skip JIT + graph capture)
dec = rows[3:]
if dec:
    n = len(dec)
    avg = {k: sum(r.get(k, 0) for r in dec) / n for k in ("step", "plan", "fwd", "sample")}
    # _sample_batch's .tolist() is a GPU→CPU sync that waits for ALL prior GPU
    # work (graph replay + sampling kernels), so its wall time ≈ GPU total.
    gpu = avg["sample"]
    cpu = avg["step"] - gpu
    print(f"\n=== Decode tick breakdown (ticks 4+, avg of {n}) ===", flush=True)
    print(f"  GPU (fwd+sample, .tolist sync): {gpu:.2f} ms ({gpu/avg['step']*100:.1f}%)", flush=True)
    print(f"  Python (plan+setup+commit):     {cpu:.2f} ms ({cpu/avg['step']*100:.1f}%)", flush=True)
    print(f"  ─────────────────────────────────────", flush=True)
    print(f"  Total step:                     {avg['step']:.2f} ms", flush=True)
