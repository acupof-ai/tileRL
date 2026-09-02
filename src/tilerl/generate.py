"""Offline batch generation across N devices, one PROCESS per device.

A process per device, not :class:`~tilerl.parallel.DataParallelEngine`: that
wrapper runs N CUDA contexts in one interpreter and the Python half of every
tick serialises on the GIL. Eight processes measured 7.54x on 8 H20s
(wins/2026-08-29-data-parallel-scales.md). Each worker owns a disjoint slice
of the prompts and writes its own file: no queue, no shared state.

# ponytail: prompts are assigned by stride (rank i takes i::N); a skewed corpus
# leaves cards idle at the tail, and the upgrade is a shared work queue.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _read_prompts(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_worker(
    rank: int,
    world: int,
    device: int,
    source: str | None,
    prompts_path: str,
    out_path: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    max_batch: int,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    # Imported after CUDA_VISIBLE_DEVICES: the Backend binds the current device on construction.
    import torch

    from .engine import SamplingParams, build_engine
    from tilerl_kernels.backend import Backend, resolve_target

    rows = _read_prompts(prompts_path)[rank::world]
    if not rows:
        Path(out_path).write_text("")
        return

    backend = Backend(resolve_target())
    if source:
        from .config import qwen36_27b
        from .model import load_hf

        cfg = qwen36_27b()
        model = load_hf(cfg, source, fuse_projections=True)
    else:
        from .config import tiny
        from .model import build_random

        cfg = tiny()
        model = build_random(cfg, seed=0, fuse_projections=True)

    longest = max(len(r["token_ids"]) for r in rows)
    engine = build_engine(
        cfg, model, backend,
        num_blocks=max(256, 2 * max_batch * -(-(longest + max_new_tokens) // 16)),
        num_slots=max_batch, max_batch=max_batch,
        max_total_tokens=max(8192, longest + max_new_tokens + 64),
    )

    # Sliding window: submit allocates the state slot, so the pool bounds in-flight requests.
    ids: dict[int, int] = {}
    nxt, wrote = 0, 0
    with open(out_path, "w") as f:
        while wrote < len(rows):
            while nxt < len(rows) and len(ids) < max_batch:
                rid = engine.submit(
                    rows[nxt]["token_ids"],
                    SamplingParams(temperature=temperature, top_p=top_p,
                                   seed=seed + nxt, max_new_tokens=max_new_tokens),
                )
                ids[rid] = nxt
                nxt += 1
            engine.step()
            for rid, out in engine.poll().items():
                i = ids.pop(rid)
                f.write(json.dumps({
                    "index": i, "rank": rank,
                    "token_ids": rows[i]["token_ids"],
                    "output_ids": out,
                    "finished": True,
                }) + "\n")
                wrote += 1
    del engine
    torch.cuda.empty_cache()


def generate(
    prompts: str,
    out: str,
    devices: list[int],
    source: str | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 0,
    max_batch: int = 32,
) -> dict[str, Any]:
    """Fan the corpus across ``devices``, one process each, then merge.
    Returns counters: prompts in, rows written, seconds, tokens/s."""
    import multiprocessing as mp
    import time

    world = len(devices)
    n = len(_read_prompts(prompts))
    parts = [f"{out}.part{r}" for r in range(world)]
    ctx = mp.get_context("spawn")  # CUDA is not fork-safe
    t0 = time.perf_counter()
    procs = [
        ctx.Process(target=run_worker, args=(
            r, world, d, source, prompts, parts[r],
            max_new_tokens, temperature, top_p, seed, max_batch,
        ))
        for r, d in enumerate(devices)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    bad = [r for r, p in enumerate(procs) if p.exitcode != 0]
    if bad:
        raise RuntimeError(f"generate: worker(s) {bad} failed; see their output above")

    written = tokens = 0
    with open(out, "w") as f:
        for part in parts:
            with open(part) as g:
                for line in g:
                    tokens += len(json.loads(line)["output_ids"] or ())
                    written += 1
                    f.write(line)
            os.unlink(part)
    dt = time.perf_counter() - t0
    return {"prompts": n, "rows": written, "seconds": round(dt, 2),
            "tokens": tokens, "tok_s": round(tokens / dt, 1) if dt else 0.0}


if __name__ == "__main__":  # runnable check: stride covers the corpus exactly
    rows = list(range(37))
    world = 8
    seen = [r for w in range(world) for r in rows[w::world]]
    assert sorted(seen) == rows, "stride assignment must partition the corpus"
    assert len({len(rows[w::world]) for w in range(world)}) <= 2, "shares differ by >1"
    print("generate: stride partition OK")
