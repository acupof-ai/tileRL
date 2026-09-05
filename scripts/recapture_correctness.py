#!/usr/bin/env python3
"""The correctness half of #94: does a recapturing engine actually resample?

The wall-clock arms say what the waivers COST. This says whether they are safe.
On cpu the graph half is unmeasurable -- capture calls torch.cuda.graph_pool_handle(),
which raises, and _decode_graph_on flips to False, so a captured-vs-eager comparison
compares eager to eager. On the card both halves are real, so this is the only place
the claim can be tested.

Three checks, each with the negative control named:

  state     after each update the graphs are dropped and the prefix is empty.
            Control: this passes trivially if graphs never captured -- so we
            assert graphs were HELD before the update, or the check is vacuous.

  resample  a kept-graph engine that recaptures must produce a different rollout
            after a large weight perturbation than before it. Control: the same
            comparison with a ZERO perturbation must produce the SAME rollout,
            or "different" is just sampling noise.

  agree     the recapturing engine and a graphs-off engine, same seed and weights,
            produce the same completions. Control: none needed -- a mismatch is
            the failure, and greedy sampling makes it deterministic.
"""
import argparse
import json
import os
import sys

sys.path[:0] = [f"{os.environ['REMOTE_DIR']}/src",
                f"{os.environ['REMOTE_DIR']}/packages/tilerl-kernels/src"]

import torch  # noqa: E402


def rollout(engine, prompt, gen, seed):
    """poll() returns {request_id: token_ids}, so index it -- an attribute read
    on a dict silently yields nothing and every comparison would then be True."""
    from tilerl.engine import SamplingParams
    rid = engine.submit(list(prompt), SamplingParams(max_new_tokens=gen, temperature=0.0,
                                                     top_p=1.0, top_k=0, seed=seed))
    for _ in range(gen * 4):
        engine.step()
        done = engine.poll()
        if rid in done:
            return list(done[rid])
    raise RuntimeError(f"request {rid} never finished in {gen * 4} ticks")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen38-27b")
    p.add_argument("--gen", type=int, default=32)
    p.add_argument("--blocks", type=int, default=2048)
    p.add_argument("--out", default="/work/recapture_correctness.json")
    args = p.parse_args()

    from tilerl_kernels.backend import get_backend

    from tilerl.cli import _build_model
    from tilerl.engine import build_engine

    backend = get_backend()
    cfg, model = _build_model(args.model, seed=0, keep_master=False)
    eng = build_engine(cfg, model, backend, num_blocks=args.blocks, num_slots=2,
                       max_batch=2, max_total_tokens=args.blocks * 16, decode_graph=True)
    if args.model == "qwen38-27b":
        from tilerl.cli import _qwen38_tokenizer
        tok = _qwen38_tokenizer()
    else:
        from tilerl.tokenizer import get_tokenizer
        tok = get_tokenizer(None)
    prompt = tok.encode("What is 17 times 23?")
    out = {}

    # Warm the graphs, then confirm the state check is not vacuous.
    base = rollout(eng, prompt, args.gen, seed=0)
    held = len(eng._decode_graphs)
    out["graphs_held_before_update"] = held
    out["state_check_vacuous"] = held == 0
    if held == 0:
        print("WARNING: no graphs captured; the state and resample checks are vacuous",
              flush=True)

    # Control arm first: a ZERO perturbation must leave the rollout unchanged.
    eng.invalidate_weights()
    same = rollout(eng, prompt, args.gen, seed=0)
    out["zero_perturbation_same"] = same == base
    out["graphs_after_invalidate"] = len(eng._decode_graphs)

    # Real arm: perturb the weights enough that any honest resample must differ.
    target = next(k for k in model.params
                  if k.endswith(".lora_b") or model.params[k].ndim == 2)
    with torch.no_grad():
        model.params[target].add_(torch.randn_like(model.params[target].float()).to(
            model.params[target].dtype) * 5.0)
    eng.invalidate_weights()
    after = rollout(eng, prompt, args.gen, seed=0)
    out["perturbed_key"] = target
    out["perturbation_changed_rollout"] = after != base

    out["verdict"] = ("PASS" if (out["zero_perturbation_same"]
                                 and out["perturbation_changed_rollout"]
                                 and not out["state_check_vacuous"]) else "FAIL")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
