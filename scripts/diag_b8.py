"""Quick B=8 diagnostic: per-tick time + graph capture status."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "tilerl-kernels" / "src"))

import torch  # noqa: E402

from tilerl import config as config_mod  # noqa: E402
from tilerl import model as model_mod  # noqa: E402
from tilerl.engine import SamplingParams, build_engine  # noqa: E402
from tilerl.server import get_tokenizer  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

src = os.environ["TILERL_QWEN38_SOURCE"]
cfg = config_mod.qwen38_27b()
print("loading...", flush=True)
t0 = time.time()
model = model_mod.load_hf(cfg, src, fuse_projections=True)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

backend = get_backend()
eng = build_engine(cfg, model, backend, num_blocks=64, num_slots=8, max_total_tokens=8192)
print(f"limits: max_batch={eng.limits.max_batch} max_num_batched_tokens={eng.limits.max_num_batched_tokens}", flush=True)
tok = get_tokenizer(src)

prompts = ["The capital of France is", "The largest planet is", "Speed of light is",
           "Romeo and Juliet author", "Gold symbol is", "Tallest mountain is",
           "Japan currency is", "Brazil language is"]
B = len(prompts)
ids = [tok.encode(p) for p in prompts]
rids = [eng.submit(ids[i], SamplingParams(temperature=0.0, max_new_tokens=16)) for i in range(B)]

outs = [None] * B
pending = B
tick = 0
while pending:
    t1 = time.perf_counter()
    eng.step()
    dt = time.perf_counter() - t1
    tick += 1
    graph_on = getattr(eng, "_decode_graph_on", "?")
    ngraphs = len(getattr(eng, "_decode_graphs", {}))
    df = getattr(eng, "_decode_forwards", "?")
    pf = getattr(eng, "_prefill_forwards", "?")
    mf = getattr(eng, "_mixed_forwards", "?")
    nwait = len(getattr(eng, "_waiting", []))
    nrun = len(getattr(eng, "_running", []))
    nfin = len(getattr(eng, "_finished", {}))
    nfail = len(getattr(eng, "_failed", {}))
    rinfo = [(r.req_id, r.phase, r.seq_len, len(r.output), r.prefill_from)
             for r in getattr(eng, "_running", [])]
    print(f"tick {tick}: {dt*1e3:.1f} ms  graph_on={graph_on}  graphs={ngraphs}  "
          f"decode_fwd={df} prefill_fwd={pf} mixed_fwd={mf}  "
          f"wait={nwait} run={nrun} fin={nfin} fail={nfail}  "
          f"running={rinfo}", flush=True)
    for i in range(B):
        if outs[i] is None:
            o = eng.take(rids[i])
            if o is not None:
                outs[i] = o
                pending -= 1
                print(f"  req{i} done: {tok.decode(o[:8])!r}", flush=True)

print(f"\nfirst req0: {tok.decode(outs[0][:8])!r}", flush=True)
