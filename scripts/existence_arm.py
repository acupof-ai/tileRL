#!/usr/bin/env python3
"""Does invalidate_weights() refill the casts a kept graph would replay stale?

The LoRA arms cannot answer this: `rl_step` passes `trainable`, so only
lora_a/lora_b move, and no adapter reaches a kernel through `_const_f32`. Every
cached parameter's `_version` is unchanged, so the walk correctly refills 0 --
structurally, not by accident. A full-parameter update is the only configuration
where a _const_f32-routed parameter (oscale, wscale, rmsnorm weight, the gdn
constants) actually moves.

tiny, not the 27B: full-parameter AdamW on the 27B is 200.4 GiB of moments
(autograd.py, Adafactor's docstring). The question here is a mechanism, not a
throughput, so the smallest model that captures a graph answers it.

Three assertions, and the vacuity guard first because the other two pass
trivially on a config that captured nothing or refilled nothing.
"""
import json
import os
import sys

sys.path[:0] = [f"{os.environ['REMOTE_DIR']}/src",
                f"{os.environ['REMOTE_DIR']}/packages/tilerl-kernels/src"]

import torch  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

from tilerl.autograd import AdamW  # noqa: E402
from tilerl.config import tiny  # noqa: E402
from tilerl.engine import SamplingParams, build_engine  # noqa: E402
from tilerl.kv_cache import NoPrefixStore  # noqa: E402
from tilerl.model import build_random  # noqa: E402

out = {}


def rollout(eng, prompt, gen=6):
    rid = eng.submit(list(prompt), SamplingParams(max_new_tokens=gen, temperature=0.0,
                                                  top_p=1.0, top_k=0, seed=0))
    for _ in range(gen * 8):
        eng.step()
        done = eng.poll()
        if rid in done:
            return done[rid]
    raise RuntimeError("rollout did not finish")


backend, cfg = get_backend(), tiny()
model = build_random(cfg, seed=7, keep_master=True)
eng = build_engine(cfg, model, backend, num_blocks=64, num_slots=4, max_batch=4,
                   max_total_tokens=512, decode_graph=True,
                   prefix_store=NoPrefixStore())
prompt = list(range(1, 9))

base = rollout(eng, prompt)
held = len(eng._decode_graphs)
out["graphs_held"] = held
out["graph_arm_vacuous"] = held == 0

# A full-parameter step: no `trainable`, so the optimizer touches model.params.
opt = AdamW(lr=1.0)
# `step()` calls begin() then step_one per param; driving step_one directly leaves
# _step at 0 and the bias correction divides by zero. begin() is that increment.
opt.begin()
grads = {id(p): torch.randn_like(p.float()) * 5.0
         for p in model.params.values() if p.is_floating_point()}
addrs = {k: p.data_ptr() for k, p in model.params.items()}
for k, p in model.params.items():
    if p.is_floating_point():
        opt.step_one(p, grads[id(p)], key=k)
# Address changes, not value changes: 0 IS the result -- an in-place update is
# what makes a baked address still valid. `rollout_changed` shows the values moved.
out["params_reallocated"] = sum(1 for k, p in model.params.items() if p.data_ptr() != addrs[k])

# Which parameters CAN be refilled: _const_f32 caches nothing for a tensor that is
# already f32/on-device/unpadded (backend.py:1416), so an f32-only model would report
# 0 for a reason that has nothing to do with the refill. tiny's norm weights are bf16,
# so they cache; count them first and assert the walk found them.
cached_before = len(backend._const_f32_cache)
out["cache_entries"] = cached_before

refilled = eng.invalidate_weights()
out["casts_refilled"] = refilled
out["refill_arm_vacuous"] = refilled == 0
# Not just >0: every cached entry whose parameter moved must be refilled, or the walk
# is skipping some and a kept graph replays those stale.
out["refilled_all_cached"] = refilled == cached_before
out["graphs_after"] = len(eng._decode_graphs)

after = rollout(eng, prompt)
out["rollout_changed"] = after != base

# The graph arm must agree with eager on the SAME weights: a kept graph that
# replays a stale cast differs from an engine that has no graph to replay.
eager = build_engine(cfg, model, backend, num_blocks=64, num_slots=4, max_batch=4,
                     max_total_tokens=512, decode_graph=False,
                     prefix_store=NoPrefixStore())
out["graph_matches_eager"] = rollout(eager, prompt) == after

print(json.dumps(out, indent=2, sort_keys=True), flush=True)
bad = [k for k in ("graph_arm_vacuous", "refill_arm_vacuous") if out[k]]
if bad:
    print(f"VACUOUS: {bad} -- this arm proved nothing", flush=True)
    sys.exit(1)
if not out["refilled_all_cached"]:
    print(f"FAIL: refilled {out['casts_refilled']} of {out['cache_entries']} cached casts",
          flush=True)
    sys.exit(1)
if not out["rollout_changed"]:
    print("FAIL: a full-parameter update did not change the rollout", flush=True)
    sys.exit(1)
if not out["graph_matches_eager"]:
    print("FAIL: the kept graph disagrees with eager after the update", flush=True)
    sys.exit(1)
print("OK: casts refilled, kept graph agrees with eager after a full-parameter update")
