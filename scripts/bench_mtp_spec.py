"""MTP spec decode bench: B=1/B=8, alpha + tok/s, losslessness check.

Loads the full 27B + MTP head, builds two engines (spec-OFF, spec-ON) sharing
the model, and runs the same greedy prompt through both. Verifies token-id
identity, then measures steady-state decode tok/s and the acceptance rate.

Usage:
    TILERL_TARGET=cuda CUDA_VISIBLE_DEVICES=7 \\
        PYTHONPATH=src python3 scripts/bench_mtp_spec.py /data00/Qwen3.8-27B-NVFP4
"""

from __future__ import annotations

import argparse
import time

import torch

from tilerl.config import qwen38_27b
from tilerl.engine import SamplingParams, build_engine
from tilerl.model import load_hf
from tilerl.mtp import load_mtp_head
from tilerl.ops.backend import get_backend


def _rand_prompt(vocab: int, n: int, seed: int) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n,), generator=gen).tolist()


def _drive(engine, wids, max_steps=8192):
    """Drive until all wids finish. Returns {wid: output}."""
    done: dict[int, list[int]] = {}
    for _ in range(max_steps):
        engine.step()
        done.update(engine.poll())
        if all(w in done for w in wids):
            return done
    raise RuntimeError("requests did not finish")


def _drive_timed(engine, wids, warmup_ticks, max_steps=8192):
    """Drive with untimed warmup, then time until all finish.

    Returns (done, wall_s). The warmup covers prefill + bootstrap + the first
    spec tick (JIT compile) so the timed region is steady-state.
    """
    done: dict[int, list[int]] = {}
    for _ in range(warmup_ticks):
        engine.step()
        done.update(engine.poll())
        if all(w in done for w in wids):
            return done, 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(max_steps):
        engine.step()
        done.update(engine.poll())
        if all(w in done for w in wids):
            break
    else:
        raise RuntimeError("requests did not finish")
    torch.cuda.synchronize()
    return done, time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="HF checkpoint directory (with model_mtp.safetensors)")
    p.add_argument("--n-decode", type=int, default=64)
    p.add_argument("--prompt-len", type=int, default=512)
    args = p.parse_args()

    backend = get_backend()
    if backend.device.type != "cuda":
        raise SystemExit("needs CUDA target (TILERL_TARGET=cuda)")
    cfg = qwen38_27b()

    t0 = time.perf_counter()
    model = load_hf(cfg, args.source, fuse_projections=True)
    mtp = load_mtp_head(args.source)
    print(f"load: {time.perf_counter() - t0:.1f}s", flush=True)

    common = dict(
        num_blocks=1024,
        num_slots=16,
        max_batch=8,
        max_total_tokens=16384,
        decode_graph=False,  # eager: fair spec-on vs spec-off comparison
    )
    eng_off = build_engine(cfg, model, backend, **common)
    eng_on = build_engine(cfg, model, backend, **common, mtp_head=mtp, spec_decode=True)
    print(f"engines built (spec_decode={eng_on._spec_decode})", flush=True)

    for B in (1, 8):
        prompt = _rand_prompt(cfg.vocab_size, args.prompt_len, seed=11)
        sp = SamplingParams(temperature=0.0, max_new_tokens=args.n_decode, seed=42)
        # Warmup: B requests stagger into decode over ~2B ticks (each prefill
        # spans a large chunk + a small remainder, one prefill per tick), then
        # a few spec ticks cover the T=2 + MTP + rollback JIT.
        warmup = 2 * B + 8

        # Warmup + losslessness.
        wids_off = [eng_off.submit(prompt, sp) for _ in range(B)]
        out_off = _drive(eng_off, wids_off)
        wids_on = [eng_on.submit(prompt, sp) for _ in range(B)]
        out_on = _drive(eng_on, wids_on)
        for i in range(B):
            a, b = out_off[wids_off[i]], out_on[wids_on[i]]
            if a != b:
                print(f"MISMATCH B={B} req {i}: len off={len(a)} on={len(b)}")
                for k, (x, y) in enumerate(zip(a, b)):
                    if x != y:
                        print(f"  first diff at token {k}: off={x} on={y}")
                        break
                raise SystemExit(1)
        print(f"B={B}: lossless OK ({len(out_off[wids_off[0]])} tokens/req)", flush=True)

        # Timed spec-OFF.
        wids_off = [eng_off.submit(prompt, sp) for _ in range(B)]
        out_off, wall_off = _drive_timed(eng_off, wids_off, warmup)
        toks_off = sum(len(out_off[w]) for w in wids_off)

        # Timed spec-ON.
        wids_on = [eng_on.submit(prompt, sp) for _ in range(B)]
        out_on, wall_on = _drive_timed(eng_on, wids_on, warmup)
        toks_on = sum(len(out_on[w]) for w in wids_on)

        stats = eng_on.stats()
        acc, rej = stats.get("spec_accepts", 0), stats.get("spec_rejects", 0)
        alpha = acc / max(1, acc + rej)
        # Per-request tok/s (aggregate for B=8).
        tps_off = toks_off / wall_off
        tps_on = toks_on / wall_on
        print(
            f"B={B}: OFF {tps_off:.1f} tok/s | ON {tps_on:.1f} tok/s | "
            f"speedup {tps_on / tps_off:.3f} | alpha {alpha:.4f} ({acc}/{acc + rej})",
            flush=True,
        )

    free, total = torch.cuda.mem_get_info()
    print(f"\npeak mem: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    print(f"bench_commit: {__import__('os').environ.get('BENCH_COMMIT', 'unknown')}", flush=True)


if __name__ == "__main__":
    main()
