#!/usr/bin/env python3
"""P2.0 recapture-after-update correctness on a real card, LoRA path (what P1 trains).

Full-parameter SFT cannot coexist with the captured serving engine on one 96 GiB
H20 (weights 23.3 + bf16 masters 54.6 + graph + tape OOM at the backward; AdamW's
f32 moments alone are 200 GiB). LoRA trains a 124.8M-param adapter beside the
frozen fp4 base, so it fits. The recapture premise is unchanged and is what P1
needs: AdamW.step_one ends in an in-place p.copy_() on every adapter tensor, so
a decode graph captured before the step replays the UPDATED adapter values, and
greedy tokens after must equal an eager engine carrying identical post-step
adapters.

Two engines built SEQUENTIALLY (a 27B engine is most of the card), both
seed-0 served models with a seed-0 rank-16 adapter and the same fixed-seed step:

  captured graphs on; add_lora; warm -> base -> graphs held; no-step invalidate
           control (must be identical); one LoRA SFT step; invalidate -> after
  eager     graphs off; add_lora; warm -> the same LoRA step -> after_eager

after == after_eager. Controls: graphs HELD before and after; no-step rollout
bit-identical; the update changed tokens. Also prints each phase wall time.
"""

import argparse
import json
import os
import sys
import time

sys.path[:0] = [f"{os.environ.get('REMOTE_DIR', '.')}/src",
                f"{os.environ.get('REMOTE_DIR', '.')}/packages/tilerl-kernels/src"]

import torch  # noqa: E402


def rollout(engine, prompt_ids, gen, seed=0):
    from tilerl.engine import SamplingParams

    rid = engine.submit(list(prompt_ids),
                        SamplingParams(temperature=0.0, top_p=1.0, top_k=0,
                                      max_new_tokens=gen, seed=seed))
    for _ in range(gen * 8):
        engine.step()
        done = engine.poll()
        if rid in done:
            return [int(t) for t in done[rid]]
    raise RuntimeError("rollout never finished")


def one_lora_step(model, trainable, backend, seq_len, lr, seed=1234):
    """One in-place LoRA SFT step over fixed-seed tokens; AdamW ends in p.copy_()."""
    from tilerl.autograd import AdamW
    from tilerl.train import train_step

    ids = torch.randint(0, model.cfg.vocab_size, (1, seq_len),
                        generator=torch.Generator().manual_seed(seed))
    return train_step(model, ids, backend, AdamW(lr=lr, betas=(0.9, 0.95),
                                                eps=1e-8, weight_decay=0.1),
                      trainable=trainable)


def build(cfg, model, backend, blocks, graph):
    from tilerl.engine import build_engine
    from tilerl.kv_cache import NoPrefixStore

    return build_engine(cfg, model, backend, num_blocks=blocks, num_slots=2,
                        max_batch=2, max_total_tokens=blocks * 16,
                        decode_graph=graph, prefix_store=NoPrefixStore())


def steady_tick_ms(engine, prompt_ids, warm_gen=16, measure_gen=48):
    """Steady-state post-step DECODE tick cost. A first short request warms/captures so
    its JIT and graph capture are excluded; the measured request times engine.step()
    only while the row is in the decode phase (prefill steps are skipped). Returns the
    median ms/tick over the decode window — the P2.0 wall number, NOT end-to-end time."""
    w = engine.submit(list(prompt_ids),
                      __import__("tilerl.engine", fromlist=["SamplingParams"])
                      .SamplingParams(temperature=0.0, max_new_tokens=warm_gen, seed=0))
    for _ in range(warm_gen * 8):
        engine.step()
        if w in engine.poll():
            break
    rid = engine.submit(list(prompt_ids),
                        __import__("tilerl.engine", fromlist=["SamplingParams"])
                        .SamplingParams(temperature=0.0, max_new_tokens=measure_gen, seed=0))
    samples = []
    for _ in range(measure_gen * 8):
        t = time.perf_counter()
        engine.step()
        dt = (time.perf_counter() - t) * 1000
        if any(r.req_id == rid and r.phase == 2 for r in engine._running):
            samples.append(dt)
        if rid in engine.poll():
            break
    samples.sort()
    return round(samples[len(samples) // 2], 2), round(min(samples), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--source", default="/work/tilerl-ckpt/Qwen3.8-27B-NVFP4")
    ap.add_argument("--out", default="/work/recapture_lora.json")
    args = args = ap.parse_args()

    os.environ["TILERL_QWEN38_SOURCE"] = args.source
    from tilerl_kernels.backend import get_backend

    from tilerl.cli import _build_model, _qwen38_tokenizer
    from tilerl.model import add_lora

    assert torch.cuda.is_available(), "card-only gate"
    backend = get_backend()
    tok = _qwen38_tokenizer()
    prompt = tok.encode("What is 17 times 23?")
    times = {}

    # phase 1: captured. add_lora MUST come after build_engine: building materializes
    # the params onto the device, and an adapter attached before points at a CPU tensor
    # the forward never reads (train_step then sees zero grads). Production cli.py
    # attaches in the same order.
    t0 = time.perf_counter()
    cfg, m_cap = _build_model("qwen38-27b", seed=0, fuse_projections=False)
    cap = build(cfg, m_cap, backend, args.blocks, True)
    lora_cap = add_lora(m_cap, rank=args.rank)
    rollout(cap, prompt, args.gen)  # warm / capture
    base = rollout(cap, prompt, args.gen)
    held = len(cap._decode_graphs)
    assert held > 0, "no graphs captured — the card half is vacuous"
    cap.invalidate_weights()
    control = rollout(cap, prompt, args.gen)
    no_step_unchanged = control == base

    one_lora_step(m_cap, lora_cap, backend, args.seq, args.lr)
    cap.invalidate_weights()
    held_after = len(cap._decode_graphs)
    after_cap = rollout(cap, prompt, args.gen)
    cap_med, cap_min = steady_tick_ms(cap, prompt)
    cap.shutdown()
    del cap, m_cap
    torch.cuda.empty_cache()
    times["captured_secs"] = round(time.perf_counter() - t0, 1)
    times["captured_decode_ms_median"] = cap_med
    times["captured_decode_ms_min"] = cap_min

    # phase 2: eager, same seed adapter + same data step
    t0 = time.perf_counter()
    cfg, m_eag = _build_model("qwen38-27b", seed=0, fuse_projections=False)
    eager = build(cfg, m_eag, backend, args.blocks, False)
    lora_eag = add_lora(m_eag, rank=args.rank)
    rollout(eager, prompt, args.gen)
    one_lora_step(m_eag, lora_eag, backend, args.seq, args.lr)
    after_eager = rollout(eager, prompt, args.gen)
    eag_med, eag_min = steady_tick_ms(eager, prompt)
    eager.shutdown()
    times["eager_secs"] = round(time.perf_counter() - t0, 1)
    times["eager_decode_ms_median"] = eag_med
    times["eager_decode_ms_min"] = eag_min

    out = dict(graphs_held_before=held, graphs_held_after=held_after,
               no_step_unchanged=no_step_unchanged,
               update_changed_tokens=after_cap != base,
               recaptured_equals_eager=after_cap == after_eager,
               tokens_captured=after_cap, tokens_eager=after_eager, **times)
    out["verdict"] = ("PASS" if (held > 0 and held_after == held and no_step_unchanged
                                 and out["update_changed_tokens"]
                                 and out["recaptured_equals_eager"]) else "FAIL")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("tokens_")}, indent=2),
          flush=True)
    if out["verdict"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
