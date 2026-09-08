"""What consumes the 19.75 GiB between build_engine and step_one?

`probe_rl_budget.py` measured 72.09 GiB allocated / 22.51 free after build_engine, and the
f32 embedding copy `step_one` asks for (4.74 GiB) FITS there. The failing run had 3.38 free.
So something between the two spends ~19.75 GiB, and the traceback's location (`step_one`)
is where the allocation failed, not where the memory went.

Phase-attributed high-water marks, one arm, `micro` swept -- because if the consumer is tape
activations then `micro` is a lever that does not change the measurement (`_step`'s docstring:
grad_fn normalizes by the whole batch, so the update is identical however the rows split).

Stops before the optimizer step on purpose: this prices the phases, it does not train.
"""
import argparse
import os

import numpy as np
import torch

os.environ.setdefault("TILERL_TARGET", "cuda")

from tilerl_kernels.backend import get_backend  # noqa: E402

from tilerl.autograd import Adafactor  # noqa: E402
from tilerl.cli import _build_model  # noqa: E402
from tilerl.engine import SamplingParams, build_engine  # noqa: E402
from tilerl.kv_cache import NoPrefixStore  # noqa: E402
from tilerl.train import _drain, group_advantages, rl_step  # noqa: E402

GiB = 2**30

ap = argparse.ArgumentParser()
ap.add_argument("--group", type=int, default=4)
ap.add_argument("--max-new-tokens", type=int, default=128)
ap.add_argument("--prompt-tokens", type=int, default=64)
ap.add_argument("--micro", default="0,1,2", help="comma list; 0 = whole group at once")
ap.add_argument("--blocks", type=int, default=256)
ap.add_argument("--drop-quantized", action="store_true",
                help="free the served bytes the way cli.py:275 does at the training entry "
                     "points; 16.88 GiB on the 27B, and full fine-tuning never reads them")
a = ap.parse_args()


def mark(tag):
    al = torch.cuda.memory_allocated() / GiB
    pk = torch.cuda.max_memory_allocated() / GiB
    free = torch.cuda.mem_get_info()[0] / GiB
    print(f"  {tag:34} allocated {al:7.2f}  peak {pk:7.2f}  free {free:6.2f}", flush=True)
    return pk


cfg, model = _build_model("qwen38-27b", seed=0, keep_master=True)
if a.drop_quantized:
    from tilerl.model import drop_quantized

    n_before = len(model.params)
    drop_quantized(model)
    print(f"drop_quantized: {n_before} -> {len(model.params)} tensors", flush=True)
backend = get_backend()
engine = build_engine(cfg, model, backend, num_blocks=a.blocks, num_slots=8,
                      decode_graph=False, prefix_store=NoPrefixStore())
base = mark("after build_engine")
params_gib = sum(v.numel() * v.element_size() for v in model.params.values()) / GiB
print(f"  params on card {params_gib:.2f} GiB\n", flush=True)

rng = np.random.default_rng(0)
prompt = rng.integers(3, cfg.vocab_size, size=a.prompt_tokens).astype(np.int64)
sampling = SamplingParams(max_new_tokens=a.max_new_tokens, temperature=1.0)

for micro in [int(x) for x in a.micro.split(",")]:
    print(f"micro={micro} (0 = whole group of {a.group} at once)", flush=True)
    torch.cuda.reset_peak_memory_stats()
    mark("start of arm")

    ids = [engine.submit(prompt.tolist(), SamplingParams(
        max_new_tokens=a.max_new_tokens, temperature=1.0, seed=g)) for g in range(a.group)]
    done = _drain(engine, ids, "budget probe rollout")
    comps = [done[i] for i in ids]
    roll = mark("after rollout (drain)")

    # A reward that varies, so advantages are not all zero and the backward is real.
    rewards = [float(len(c) % 7) / 7.0 for c in comps]
    adv = group_advantages(rewards, a.group)
    gen = max(len(c) for c in comps)
    batch = np.stack([np.concatenate([prompt, np.asarray(c, dtype=np.int64),
                                      np.zeros(gen - len(c), dtype=np.int64)]) for c in comps])
    plens = np.full(a.group, len(prompt), dtype=np.int64)
    slens = np.array([len(prompt) + len(c) for c in comps], dtype=np.int64)

    try:
        rl_step(model, batch, adv, plens, backend, Adafactor(lr=0.0), seq_lens=slens,
                micro=micro)
        step = mark("after rl_step (fwd+bwd+update)")
        print(f"  rollout adds {roll - base:6.2f} GiB, the step adds {step - roll:6.2f} GiB, "
              f"total over params {step - params_gib:6.2f}\n", flush=True)
    except torch.OutOfMemoryError as e:
        need = str(e).split("Tried to allocate ")[-1].split(" ")[0] if "Tried" in str(e) else "?"
        print(f"  OOM in rl_step, asked for {need} -- this micro does not fit\n", flush=True)
        torch.cuda.empty_cache()
