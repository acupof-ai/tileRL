"""One warm GRPO step at 27B with the phase split, so `forward_secs` gets a card number.

The denominator exists: 131.579 s, backward 71.529 (54.4%), rollout 59.965 (45.6%) at
a16ff9c, gen 1024, card 6 (wins/2026-09-06-one-grpo-step-is-54-percent-backward.md).
What that entry cannot say is how much of its "backward" is the FORWARD -- `rl_step`
timed forward + loss + backward as one bucket. On tiny the forward is 34% of it, and if
that holds here then ~24 s of the 71.5 s that priced every backward-kernel lever is
forward.

gen 1024 and group 8 to match a16ff9c's shape, so the split slots into its numbers
rather than needing its own baseline.

Two steps, not one: step 0 pays the JIT and the pool fit, and only the warm step is the
measurement -- a16ff9c's own cold step was 1.396x its warm one.

    TILERL_TARGET=cpu python3 scripts/step_phase_split.py --tiny   # the flow, no GPU
    scripts/pod_run.sh phasesplit 0 -- python3 -u scripts/step_phase_split.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "packages/tilerl-kernels/src")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true", help="the tiny model, for the CPU dry run")
    ap.add_argument("--steps", type=int, default=2, help="step 0 pays the JIT; the last is warm")
    ap.add_argument("--gen", type=int, default=1024, help="a16ff9c's shape, so the split slots in")
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import numpy as np
    from tilerl_kernels.backend import Backend, resolve_target

    from tilerl.cli import _build_model
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import NoPrefixStore
    from tilerl.model import add_lora
    from tilerl.train import AdamW, grpo_loop

    if a.tiny:
        a.gen, a.group, a.steps = 8, 2, 2

    backend = Backend(resolve_target())
    cfg, model = _build_model("tiny" if a.tiny else "qwen38-27b", seed=0, keep_master=False)
    prompts = [np.arange(8 if a.tiny else 256, dtype=np.int64)]

    # Sized off the ask, the way cli.py:538 does it, so this measures the configuration
    # `tilerl train` would run rather than a pool of a different size.
    from tilerl.kv_cache import BLOCK_TOKENS

    ctx = max(max(map(len, prompts)) + a.gen + 64, 1024)
    engine = build_engine(cfg, model, backend, num_slots=a.group, max_batch=a.group,
                          num_blocks=-(-ctx // BLOCK_TOKENS) * a.group + a.group,
                          max_total_tokens=max(ctx, 8192),
                          decode_graph=True, prefix_store=NoPrefixStore())

    # BEFORE the step, not after: two of 48's bugs on 2026-09-07 were a pool of the wrong
    # size whose downstream numbers were all consistent with each other. room_for is the
    # engine's own answer and is net of the captured tick's pad block, which `num_blocks`
    # is not.
    room = engine.room_for(len(prompts[0]))
    print(f"# pool: usable_blocks {engine.usable_blocks}, room_for(prompt {len(prompts[0])}) "
          f"= {room} tokens, asking for {a.gen}")
    if room < a.gen:
        print(f"REFUSED: the pool holds {room} new tokens and the run asks {a.gen}. Every "
              f"phase number would describe a shape nobody meant to measure.")
        return 1

    # A reward that is not constant: every group tying makes every advantage zero and
    # the backward a no-op, which would time an empty step.
    trainable = add_lora(model, rank=16)
    rows = []
    t_all = time.perf_counter()
    for i, (r, ce, secs, tied, ntok, timings, width) in enumerate(grpo_loop(
            engine, model, prompts, lambda p, c: float(len(c) % 2), a.steps,
            backend, AdamW(lr=1e-4),
            group=a.group, sampling=SamplingParams(max_new_tokens=a.gen, temperature=1.0),
            trainable=trainable, micro=1, recapture_graph=True)):
        # Rounded for the log, FULL precision kept for the identity check below: summing
        # values rounded to 4 places and then asserting 1e-6 fails on the rounding, not
        # on a missing phase -- measured, 1e-4 of disagreement on the tiny model.
        row = {"step": i + 1, "secs": secs, **dict(sorted(timings.items()))}
        rows.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                          for k, v in row.items()}), flush=True)

    warm = rows[-1]
    secs = warm["secs"]
    print(f"\n# {'tiny' if a.tiny else 'qwen38-27b'} gen={a.gen} group={a.group} micro=1 "
          f"steps={a.steps}, warm step is #{warm['step']}")
    print(f"# {'phase':<20} {'secs':>9} {'% of step':>10}")
    for k in ("rollout_secs", "forward_secs", "backward_only_secs", "optimizer_secs",
              "invalidate_secs", "other_secs"):
        print(f"  {k:<20} {warm[k]:9.3f} {warm[k] / secs * 100:9.2f}%")
    print(f"  {'step':<20} {secs:9.3f} {100.0:9.2f}%")

    # The number this run exists for, stated as the correction it is: a16ff9c published
    # backward_secs as "backward" and it contains the forward.
    bwd = warm["backward_secs"]
    print(f"\n# backward_secs {bwd:.3f} s = forward {warm['forward_secs']:.3f} + "
          f"backward_only {warm['backward_only_secs']:.3f}")
    print(f"# the published 'backward' bucket is {warm['forward_secs'] / bwd * 100:.1f}% forward")

    # The identity the CPU gate asserts, re-checked here: a phase outside every timer
    # would show up as a gap, and a card run is exactly where a new one could appear.
    parts = warm["rollout_secs"] + bwd + warm["optimizer_secs"] + warm["other_secs"]
    if abs(parts - secs) > 1e-6:
        print(f"\nREFUSED: the phases sum to {parts:.6f} against a step of {secs:.6f}")
        return 1
    print(f"# phases reconstruct the step to {abs(parts - secs):.2e} s")
    print(f"# wall clock for {a.steps} steps incl. load: {time.perf_counter() - t_all:.1f} s")

    if a.out:
        Path(a.out).write_text(json.dumps({"rows": rows, "gen": a.gen, "group": a.group},
                                          indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
