"""Pod A/B plumbing: timing, relerr, markdown table, bench-entry draft.
Family scripts build inputs at the canonical shapes and call ab()
(docs/design-kernels.md, "SOTA iteration loop").
"""

from __future__ import annotations

import os

import torch

# slice4 = 4 layers of the 27B
GDN_PREFILL = dict(B=1, T=512, QD=2048, nvh=48, K=128, V=128, KER=4)


def timeit(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def relerr(actual, ref):
    rmax = ref.abs().max().item()
    return (actual.float() - ref.float()).abs().max().item() / rmax if rmax > 0 else 0.0


def ab(name, arms, ref, iters=20, tol=1e-2):
    """arms: [(name, fn)], fn() -> tuple of tensors matching ref. Returns rows (name, ms, errs, ok)."""
    rows = []
    for aname, fn in arms:
        outs = fn()
        torch.cuda.synchronize()
        errs = [relerr(o, r) for o, r in zip(outs, ref)]
        ms = timeit(fn, iters)
        rows.append((aname, ms, errs, max(errs) < tol))
    print(f"\n## {name}\n")
    print("| arm | ms | rel-err | verdict |")
    print("|---|---:|---:|---|")
    for aname, ms, errs, ok in rows:
        es = ", ".join(f"{e:.2e}" for e in errs)
        print(f"| {aname} | {ms:.4f} | {es} | {'OK' if ok else 'FAIL'} |")
    commit = os.environ.get("BENCH_COMMIT", "?")
    print(
        f"\nentry draft: H20 pod, cuda/sm90, commit {commit}, "
        f"mean of {iters} iters per arm, same process (contention-independent ratio)."
    )
    return rows


# engine timing helpers, CPU-safe (verify_h20_fp4's module-level GPU census hard-exits on CPU)


def rand_prompt(vocab: int, n: int, seed: int) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n,), generator=gen).tolist()


def sync(backend) -> None:
    if backend.device.type == "cuda":
        torch.cuda.synchronize()


def drive(engine, wid: int, max_steps: int) -> list[int]:
    for _ in range(max_steps):
        engine.step()
        done = engine.poll()
        if wid in done:
            return done[wid]
    raise RuntimeError(f"request {wid} did not finish in {max_steps} steps")


def SETTLE_BUDGET(b: int, extra: int | None = None) -> int:
    """Size max_new_tokens from this: a budget that binds mid-window reports empty ticks as slow."""
    return 4 * b + (64 + 8 * b if extra is None else extra) + 40


def settle_decode(engine, b: int, extra: int) -> bool:
    """Step until `b` rows are all in pure decode; False if the budget runs out."""
    from tilerl.engine import _PHASE_DECODE

    for _ in range(SETTLE_BUDGET(b, extra)):
        engine.step()
        run = engine._running
        if len(run) == b and all(r.phase == _PHASE_DECODE for r in run):
            return True
    return False


def time_prefill(engine, backend, cfg, length: int, decode_ms: float) -> tuple[float, float]:
    """One request to completion minus one decode tick, as verify_h20_fp4.check_perf does."""
    from tilerl.engine import SamplingParams

    wid = engine.submit(
        rand_prompt(cfg.vocab_size, length, seed=length),
        SamplingParams(temperature=0.0, max_new_tokens=1, seed=0),
    )
    sync(backend)
    t0 = __import__("time").perf_counter()
    drive(engine, wid, max(1024, length // 64))
    sync(backend)
    prefill_ms = (__import__("time").perf_counter() - t0) * 1e3 - decode_ms
    return prefill_ms, 1000.0 * length / prefill_ms
