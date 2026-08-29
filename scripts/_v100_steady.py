"""Clean steady-state tok/s: warm fully (compile + graph capture), then time
ONLY the steady decode steps, printing first-step vs steady to expose any
compilation leaking into the timed window."""
import os
import sys
import time

import torch

from tilerl import config as config_mod
from tilerl import model as model_mod
from tilerl.engine import SamplingParams, build_engine
from tilerl.server import get_tokenizer
from tilerl_kernels.backend import get_backend


def main():
    src = os.environ["TILERL_QWEN38_SOURCE"]
    cfg = config_mod.qwen38_27b()
    model = model_mod.load_hf(cfg, src, fuse_projections=True)
    backend = get_backend()
    print("device", torch.cuda.get_device_name(0), backend.arch, flush=True)
    eng = build_engine(cfg, model, backend, num_blocks=256, num_slots=16, max_total_tokens=8192)
    tok = get_tokenizer(src)
    ids = tok.encode("The capital of France is")

    # WARM: one full request so JIT + graph capture happen here, off the clock.
    rid = eng.submit(ids, SamplingParams(temperature=0.0, max_new_tokens=16))
    out = None
    while out is None:
        out = eng.take(rid)
        if out is None:
            eng.step()
    print("warm done, out:", repr(tok.decode(out)[:80]), flush=True)

    # TIMED: second request, per-step timing to separate first (may still JIT a
    # new shape) from steady replay.
    rid = eng.submit(ids, SamplingParams(temperature=0.0, max_new_tokens=64))
    step_ms = []
    out = None
    while out is None:
        out = eng.take(rid)
        if out is not None:
            break
        torch.cuda.synchronize()
        s = time.time()
        eng.step()
        torch.cuda.synchronize()
        step_ms.append((time.time() - s) * 1000)
    # first step is prefill (+ maybe capture); steady = decode ticks after it
    steady = step_ms[2:] if len(step_ms) > 3 else step_ms
    import statistics
    med = statistics.median(steady)
    print(f"steps: {len(step_ms)}, first={step_ms[0]:.1f}ms, "
          f"steady median={med:.1f}ms => {1000/med:.1f} tok/s", flush=True)
    print(f"first 6 steps ms: {[round(x,1) for x in step_ms[:6]]}", flush=True)
    print("STEADY OK", flush=True)


if __name__ == "__main__":
    main()
